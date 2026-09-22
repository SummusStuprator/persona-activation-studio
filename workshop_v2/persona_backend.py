"""Offline, read-only HF/PEFT backend implementing the chat runtime contract.

Separate process and separate numerical identity from the GGUF runtime.
Original safetensors stay on disk; no merged checkpoint, download or optimizer.
"""
from pathlib import Path
import gc, hashlib, inspect, json, threading, types, os
import numpy as np
import torch
import torch.nn.functional as F
from .persona_store import attest, unchanged, safe_header


def promoted_linear(module,x):
    if module.weight.dtype==torch.bfloat16 and x.device.type=='cpu':
        return F.linear(x.float(),module.weight.float(),module.bias.float() if module.bias is not None else None).to(x.dtype)
    return F.linear(x,module.weight,module.bias)


def byte_decoder():
    values=list(range(ord('!'),ord('~')+1))+list(range(161,173))+list(range(174,256))
    chars=values[:]; n=0
    for b in range(256):
        if b not in values: values.append(b);chars.append(256+n);n+=1
    return dict(zip(map(chr,chars),values))


class Engine:
    def __init__(self):
        self.lock=threading.RLock();self.handle=False;self.network=None
        self.model=None;self.hooks=[];self.original_forwards=[]
        self.cuda_available=False;self.bank_diagnostics=[]

    def open(self, model, context=2048, mode="Auto", precision="auto", **kwargs):
        try:
            return self._open(model, context, mode, precision)
        except BaseException:
            self.close()
            raise

    def _open(self, model, context, mode, precision):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        import transformers,peft
        self.close();model=attest(model)
        if not 128<=context<=min(model['max_context'],8192): raise ValueError('Persona chat context must be 128 to 8192 and within the base limit.')
        import psutil
        from importlib.util import find_spec
        from .persona_device import select_device
        from .resource_policy import settings
        self.cuda_available = torch.cuda.is_available()
        free, total = torch.cuda.mem_get_info() if self.cuda_available and mode.upper() != 'CPU' else (0, 0)
        self.device_plan = select_device(model, context, mode, precision,
            cuda_available=self.cuda_available, vram_free=free, vram_total=total,
            nf4_available=find_spec('bitsandbytes') is not None)
        self.device = self.device_plan['device']
        self.dtype = torch.bfloat16 if self.device == 'cpu' or torch.cuda.is_bf16_supported() else torch.float16
        if precision == 'bf16' and self.dtype != torch.bfloat16:
            raise RuntimeError('This GPU does not support BF16. Use Auto precision.')
        vm = psutil.virtual_memory(); policy = settings()
        reserve = max(policy['reserve_ram_gib'] * 2**30, vm.total * policy['reserve_ram_fraction'])
        required = self.device_plan['required_ram_bytes'] + reserve
        if vm.available < required:
            raise MemoryError(f'Persona load requires {required/2**30:.2f} GiB free RAM; {vm.available/2**30:.2f} GiB available.')
        self.device_plan['ram_reserve_bytes'] = int(reserve)
        self.device_plan['compute_dtype'] = str(self.dtype).split('.')[-1]
        if self.device_plan['precision'] != 'nf4':
            self.device_plan['precision'] = 'bf16' if self.dtype == torch.bfloat16 else 'fp16'
        for path in model['files']:
            if path.endswith('.safetensors'): safe_header(path)
        torch.set_num_threads(max(1,int(os.environ.get("WORKSHOP_CPU_THREADS","2"))))
        load_options = {'device_map': {'': self.device}, 'torch_dtype': self.dtype}
        if self.device_plan['precision'] == 'nf4':
            from transformers import BitsAndBytesConfig
            load_options['quantization_config'] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type='nf4',
                bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=self.dtype)
        base=AutoModelForCausalLM.from_pretrained(model['base_path'],local_files_only=True,
            trust_remote_code=False,use_safetensors=True,
            low_cpu_mem_usage=True,attn_implementation='sdpa', **load_options)
        self.network=PeftModel.from_pretrained(base,model['adapter_path'],local_files_only=True,
            is_trainable=False,autocast_adapter_dtype=True,
            device_map={'': self.device}, torch_device=self.device)
        from peft.utils import get_peft_model_state_dict
        saved=safe_header(Path(model['adapter_path'])/'adapter_model.safetensors')
        actual=get_peft_model_state_dict(self.network,adapter_name='default')
        if set(saved)!=set(actual) or any(list(actual[k].shape)!=saved[k]['shape'] for k in saved):
            self.close();raise ValueError('Adapter tensor names or shapes did not match the loaded model exactly.')
        self.network.eval();self.network.requires_grad_(False)
        if any(p.requires_grad for p in self.network.parameters()): raise RuntimeError('Inference model was not frozen.')
        self.tokenizer=AutoTokenizer.from_pretrained(model['tokenizer_path'],local_files_only=True,
            trust_remote_code=False,use_fast=True)
        base=self.network.get_base_model()
        self.blocks=list(base.model.layers);self.layers=len(self.blocks);self.dim=int(base.config.hidden_size)
        self.vocab_size=int(base.get_output_embeddings().weight.shape[0])
        if self.layers!=model['layers'] or self.dim!=model['dim'] or len(self.tokenizer)>self.vocab_size:
            raise ValueError('Model architecture or tokenizer dimensions disagree with discovery metadata.')
        for module in self.network.modules():
            if self.device == 'cpu' and isinstance(module,torch.nn.Linear):
                self.original_forwards.append((module,module.forward))
                module.forward=types.MethodType(promoted_linear,module)
        self.hooks=[m.register_forward_hook(self._hook(k)) for k,m in enumerate(self.blocks)]
        self.special=set(self.tokenizer.all_special_ids);self.bytes=byte_decoder()
        self.bytelevel=model['architecture'] in ('qwen3','qwen2')
        eos=self.network.generation_config.eos_token_id
        self.eog=set(eos if isinstance(eos,list) else [eos]);self.eog.add(self.tokenizer.eos_token_id)
        self.eog.discard(None);self._pieces={}
        files=[Path(__file__),Path(__file__).with_name('persona_store.py'),Path(__file__).with_name('persona_device.py')]
        code=hashlib.sha256(b''.join(p.read_bytes() for p in files)).hexdigest()[:12]
        from importlib.metadata import version
        quant = self.device_plan['precision']
        numerical = 'cpu-bf16-fp32linear' if self.device == 'cpu' else f'cuda-sm{torch.cuda.get_device_capability()}-{quant}-{self.dtype}'
        if quant == 'nf4': numerical += '-bnb' + version('bitsandbytes')
        self.abi=f'peft-chat-v2-torch{torch.__version__}-tf{transformers.__version__}-peft{peft.__version__}-{numerical}-{code}'
        model['source_digest'] = model['digest']
        model['digest'] = 'peft-' + hashlib.sha256((model['source_digest'] + '\0' + self.abi).encode()).hexdigest()
        model.update(runtime_profile='persona_peft_'+self.device.split(':')[0],backend='persona_peft',activation_dtype=str(self.dtype).split('.')[-1],
            chat_template=self.tokenizer.chat_template or '',bos_id=self.tokenizer.bos_token_id,
            eos_id=self.tokenizer.eos_token_id,default_system='',
            numerical_backend=self.abi,capabilities=['chat','live_steering','activations','response_training'])
        self.model=model;self.context=context;self.handle=True
        self.table=np.zeros((self.layers,self.dim),np.float32);self.operation='add';self.strength=0.
        self._keep_logits='logits_to_keep' in inspect.signature(base.forward).parameters
        self.reset()
        try:
            unchanged(model);self.evaluate(self.tokenize('Local adapter load check.'))
            if not np.isfinite(self.capture()).all():raise ValueError('Non-finite decoder activations.')
            self.reset()
        except Exception:self.close();raise
        return {'weights_frozen':True,'adapter_tensors':len(saved),'hooks':self.layers,'backend':self.abi,'plan':self.device_plan}

    def _hook(self,k):
        def apply(module,args,output):
            h=output[0] if isinstance(output,tuple) else output
            if h.ndim!=3 or h.shape[0]!=1 or h.shape[-1]!=self.dim: raise RuntimeError('Unexpected decoder residual shape.')
            before=h.detach().float()
            self.pre[k]=before[0,-1].clone()
            self.mean[k]=before[0].mean(0)
            changed=h
            if self.strength and np.any(self.table[k]):
                v=torch.as_tensor(self.table[k],device=h.device,dtype=torch.float32)
                delta=(self.strength*v if self.operation=='add' else
                       -self.strength*(before@v).unsqueeze(-1)*v)
                changed=(before+delta).to(h.dtype)
            self.post[k]=changed[0,-1].detach().float().clone()
            return (changed,)+output[1:] if isinstance(output,tuple) else changed
        return apply

    def reset(self):
        self.past=None;self.position=0;self._logits=None
        self.pre={};self.post={};self.mean={}

    def clear_steering(self):
        if hasattr(self,'table'):self.table.fill(0)
        self.strength=0.;self.operation='add'

    def steer(self,directions,strength,mode='add'):
        x=np.asarray(directions,np.float32)
        if x.shape!=(self.layers,self.dim) or not np.isfinite(x).all() or not np.isfinite(strength): raise ValueError('Invalid steering array.')
        if mode not in ('add','erase'):raise ValueError('Unknown steering mode.')
        if mode=='erase':
            if not 0<=strength<=1:raise ValueError('Invalid erasure fraction.')
            x=x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)
        self.table=x.copy();self.strength=float(strength);self.operation=mode

    def tokenize(self,text):
        ids=np.asarray(self.tokenizer.encode(text,add_special_tokens=False),np.int32)
        if not len(ids):raise ValueError('Prompt is empty.')
        return ids

    def evaluate(self,tokens,all_logits=False):
        ids=np.asarray(tokens,np.int64)
        if ids.ndim!=1 or not len(ids) or ids.min()<0 or ids.max()>=self.vocab_size:raise ValueError('Invalid token sequence.')
        if self.position+len(ids)>self.context:raise ValueError('Persona context exceeded.')
        if self.position==0:unchanged(self.model)
        self.pre={};self.post={};self.mean={}
        kw={'logits_to_keep':1} if self._keep_logits else {}
        with torch.inference_mode():
            result=self.network(torch.as_tensor(ids.copy(),device=self.device).unsqueeze(0),past_key_values=self.past,
                                use_cache=True,**kw)
        self.past=result.past_key_values;self.position+=len(ids)
        self._logits=result.logits[0,-1].detach().float().cpu().numpy().copy()
        if len(self.pre)!=self.layers or not np.isfinite(self._logits).all():raise RuntimeError('Incomplete hooks or non-finite logits.')

    def capture(self,which=0):
        table={0:self.pre,1:self.post,2:self.mean}.get(which)
        if table is None or len(table)!=self.layers:raise ValueError('No complete activation capture is available.')
        return torch.stack([table[k] for k in range(self.layers)]).cpu().numpy().copy()

    def logits(self):
        if self._logits is None:raise ValueError('Evaluate tokens before reading logits.')
        return self._logits.copy()

    def is_eog(self,token):return int(token) in self.eog

    def piece(self,token):
        token=int(token)
        if token not in self._pieces:
            text=self.tokenizer.convert_ids_to_tokens(token)
            if text is None:piece=f'<unused:{token}>'.encode()
            elif token in self.special:piece=text.encode('utf-8')
            elif self.bytelevel:piece=bytes(self.bytes[c] for c in text)
            elif text.startswith('<0x') and text.endswith('>'):piece=bytes([int(text[3:-1],16)])
            else:piece=text.replace('▁',' ').encode('utf-8')
            self._pieces[token]=piece
        return self._pieces[token]

    def chat(self,messages):
        if not messages or any(m.get('role') not in ('system','user','assistant') for m in messages):raise ValueError('Invalid chat messages.')
        return self.tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)

    def extract(self,text,mean=False):
        try:
            self.clear_steering();self.reset();self.evaluate(self.tokenize(text),all_logits=mean)
            return self.capture(2 if mean else 0)
        finally:self.clear_steering();self.reset()

    def close(self):
        self.handle=False
        for hook in self.hooks:hook.remove()
        self.hooks=[]
        for m,fn in self.original_forwards:m.forward=fn
        self.original_forwards=[]
        self.network=None;self.past=None;self.blocks=[]
        self.pre={};self.post={};self.mean={};self._logits=None
        gc.collect()
        if getattr(self,'device','cpu').startswith('cuda') and torch.cuda.is_initialized():
            torch.cuda.empty_cache()

from .chat_native import extract_response,score_response
Engine.extract_response=extract_response
Engine.score_response=score_response
