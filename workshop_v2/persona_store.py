"""Local-only discovery of immutable PEFT adapter snapshots. No ML imports or downloads."""
from pathlib import Path
import hashlib, json, os, struct, time, sys
ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / 'persona-sources.json'
SUPPORTED = {'qwen3', 'qwen2', 'llama', 'mistral'}

def settings():
    defaults = {
        "project_roots": [str(ROOT/"workspace"/"persona-project")],
        "model_roots": [str(ROOT/"workspace"/"models"/"persona")],
        "cache_roots": [str(Path(os.environ.get("HF_HOME", Path.home()/".cache/huggingface"))/"hub")],
        "python": sys.executable,
        "base_overrides": {},
        "settle_seconds": 5,
        "max_depth": 5,
    }
    if CONFIG.exists():
        defaults.update(read_json(CONFIG))
    def expand(value):
        p=Path(value).expanduser()
        return str((ROOT/p).resolve() if not p.is_absolute() else p.resolve())
    for key in ("project_roots","model_roots","cache_roots"):
        defaults[key]=[expand(x) for x in defaults.get(key,[]) if x]
    py=defaults.get("python")
    defaults["python"]=expand(py) if py and ("/" in str(py) or "\\" in str(py)) else (py or sys.executable)
    defaults["base_overrides"]={k:expand(v) for k,v in defaults.get("base_overrides",{}).items()}
    return defaults

def read_json(path):
    with open(path, encoding='utf-8-sig') as f: return json.load(f)

def file_hash(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()

def signature(paths):
    return {str(Path(p).absolute()): [Path(p).stat().st_size, Path(p).stat().st_mtime_ns]
            for p in sorted(set(paths),key=str)}

def safe_header(path):
    """Validate bounds without loading a tensor or deserializing pickle."""
    size=Path(path).stat().st_size
    with open(path,'rb') as f:
        raw=f.read(8)
        if len(raw)!=8: raise ValueError('Incomplete safetensors header.')
        n=struct.unpack('<Q',raw)[0]
        if not 2<=n<=64*1024*1024 or n+8>size: raise ValueError('Invalid safetensors header length.')
        h=json.loads(f.read(n))
    tensors={k:v for k,v in h.items() if k!='__metadata__'}
    if not tensors: raise ValueError('No tensors in safetensors file.')
    widths={'F64':8,'F32':4,'F16':2,'BF16':2,'I64':8,'I32':4,'I16':2,'I8':1,'U8':1,'BOOL':1}
    intervals=[]
    for name,v in tensors.items():
        a,b=v['data_offsets'];shape=v['shape'];dtype=v['dtype']
        if dtype not in widths or any(not isinstance(x,int) or x<0 for x in shape): raise ValueError('Unsupported tensor format.')
        count=1
        for x in shape: count*=x
        if not 0<=a<=b<=size-8-n or b-a!=count*widths[dtype]: raise ValueError('Incomplete or inconsistent tensor payload: '+name)
        intervals.append((a,b))
    intervals.sort()
    if intervals[0][0]!=0 or any(x[1]!=y[0] for x,y in zip(intervals,intervals[1:])) or intervals[-1][1]!=size-8-n:
        raise ValueError('Tensor payload offsets do not exactly cover file.')
    return tensors

def base_files(folder):
    p=Path(folder)
    if not (p/'config.json').is_file(): raise ValueError('Local base config is missing.')
    index=p/'model.safetensors.index.json'
    if index.exists():
        names=sorted(set(read_json(index)['weight_map'].values()))
        if any(Path(n).name!=n or not n.endswith('.safetensors') for n in names): raise ValueError('Invalid base shard paths.')
        weights=[p/n for n in names]
    elif (p/'model.safetensors').is_file(): weights=[p/'model.safetensors']
    else: raise ValueError('No complete safetensors base weights; pickle checkpoints are not supported.')
    if any(not f.is_file() or f.stat().st_size<16 for f in weights): raise ValueError('Base weight shard is missing or incomplete.')
    if not (p/'tokenizer_config.json').is_file() or not (p/'tokenizer.json').is_file(): raise ValueError('Local fast tokenizer files are missing.')
    files=weights+[p/'config.json',p/'tokenizer_config.json',p/'tokenizer.json']
    files += [p/n for n in ('generation_config.json','special_tokens_map.json','chat_template.jinja','model.safetensors.index.json') if (p/n).is_file()]
    return files,weights

def resolve_base(reference, revision, adapter, config):
    candidates=[]
    override=config.get('base_overrides',{}).get(reference)
    if override: candidates=[Path(override)]
    elif Path(reference).is_absolute(): candidates=[Path(reference)]
    elif (Path(adapter)/reference).is_dir(): candidates=[Path(adapter)/reference]
    else:
        for cache in config['cache_roots']:
            repository=Path(cache)/('models--'+reference.replace('/','--'))
            ref=revision or 'main'; ref_file=repository/'refs'/ref
            if ref_file.is_file():
                commit=ref_file.read_text().strip()
                if len(commit)!=40 or any(c not in '0123456789abcdef' for c in commit.lower()): raise ValueError('Invalid cached revision.')
                candidates.append(repository/'snapshots'/commit)
            elif revision: candidates.append(repository/'snapshots'/revision)
            else: candidates.extend((repository/'snapshots').glob('*'))
    complete=[]
    for path in candidates:
        try: base_files(path);complete.append(path.absolute())
        except (OSError,ValueError,KeyError): pass
    complete=list(dict.fromkeys(complete))
    if len(complete)!=1:
        raise ValueError('Base model is missing locally or ambiguous; set base_overrides for '+reference+'. No download attempted.')
    return complete[0]

def inspect_adapter(adapter, config=None, now=None):
    config=config or settings();adapter=Path(adapter).absolute(); parent=adapter.parent if adapter.name=='adapter' else adapter
    ac=read_json(adapter/'adapter_config.json')
    if ac.get('peft_type')!='LORA' or ac.get('task_type')!='CAUSAL_LM': raise ValueError('Only causal-LM LoRA/rsLoRA adapters are enabled.')
    if ac.get('auto_mapping') or ac.get('alora_invocation_tokens') or ac.get('layer_replication'):
        raise ValueError('Custom mapping, invocation-gated adapters and replicated layers require separate support.')
    weights=adapter/'adapter_model.safetensors'
    safe_header(weights)
    reference=ac.get('base_model_name_or_path')
    if not isinstance(reference,str) or not reference: raise ValueError('Adapter does not declare its base model.')
    base=resolve_base(reference,ac.get('revision'),adapter,config)
    files,shards=base_files(base);bc=read_json(base/'config.json')
    if bc.get('auto_map') or bc.get('model_type') not in SUPPORTED:
        raise ValueError('Base architecture is not supported by this persona backend: '+str(bc.get('model_type')))
    if bc.get('quantization_config'): raise ValueError('Prequantized HF bases require an explicit backend; not silently reinterpreted.')
    manifest=read_json(parent/'model_manifest.json') if (parent/'model_manifest.json').exists() else {}
    report=read_json(parent/'training_report.json') if (parent/'training_report.json').exists() else {}
    gate=report.get('install_gate_pass')
    if gate is None and manifest.get('source_staging'):
        rp=Path(manifest['source_staging'])/'training_report.json'
        if rp.is_file(): gate=read_json(rp).get('install_gate_pass')
    if gate is False: raise ValueError('Training report says installation gate failed.')
    if any(part.startswith('checkpoint-') or 'staging' in part.lower() or part.startswith('_') for part in adapter.parts[len(parent.parent.parts):]):
        raise ValueError('Staging/intermediate checkpoints are not chat releases.')
    # Adapters may carry their own updated tokenizer. Do not silently ignore it.
    token_folder=adapter if (adapter/'tokenizer.json').is_file() else (parent if (parent/'tokenizer.json').is_file() else base)
    token_files=[p for p in token_folder.iterdir() if p.name in ('tokenizer.json','tokenizer_config.json','special_tokens_map.json','chat_template.jinja','added_tokens.json')]
    all_files=list(dict.fromkeys(files+[adapter/'adapter_config.json',weights]+token_files))
    newest=max(p.stat().st_mtime for p in all_files)
    if (time.time() if now is None else now)-newest < config.get('settle_seconds',60):
        raise ValueError('Files were modified recently; wait until training/export finishes, then refresh.')
    label=parent.name
    model={'name':label,'aliases':[label], 'path':str(adapter),'adapter_path':str(adapter),
           'base_path':str(base),'base_reference':reference,'tokenizer_path':str(token_folder),
           'backend':'persona_peft','architecture':bc['model_type'],'family':bc['model_type']+' + LoRA',
           'layers':int(bc['num_hidden_layers']),'dim':int(bc['hidden_size']),
           'max_context':int(bc.get('max_position_embeddings',2048)),
           'base_bytes':sum(p.stat().st_size for p in shards),'adapter_bytes':weights.stat().st_size,
           'size_gb':(sum(p.stat().st_size for p in shards)+weights.stat().st_size)/1e9,
           'files':[str(p) for p in all_files],'file_signature':signature(all_files),
           'gate_status':'recorded pass' if gate is True else 'not recorded',
           'profile':manifest.get('profile',label),'reason':None}
    model['selection_id']=hashlib.sha256(str(adapter).casefold().encode()).hexdigest()[:16]
    return model

def discover(config=None):
    config=config or settings();roots=list(config['model_roots'])+list(config['project_roots'])
    # Treat profiles.json as data, never execute launchers or training scripts.
    for root in config['project_roots']:
        p=Path(root)/'profiles.json'
        if p.is_file():
            doc=read_json(p)
            for key in ('model_root','output_root'):
                if isinstance(doc.get(key),str): roots.append(doc[key])
    found={}; excluded=[]
    for root in dict.fromkeys(roots):
        root=Path(root)
        if not root.is_dir(): excluded.append({'path':str(root),'reason':'Directory not present.'});continue
        for current,dirs,files in os.walk(root):
            depth=len(Path(current).relative_to(root).parts)
            dirs[:]=[d for d in dirs if not d.startswith('.') and d!='__pycache__' and depth<config.get('max_depth',4)]
            if 'adapter_config.json' not in files: continue
            path=Path(current)
            if any('staging' in p.lower() or p.startswith('checkpoint-') for p in path.relative_to(root).parts):
                excluded.append({'path':str(path),'reason':'Staging/intermediate checkpoint; install a completed release first.'});continue
            try:
                item=inspect_adapter(path,config);found[str(path).casefold()]=item
            except (OSError,ValueError,KeyError,TypeError) as exc: excluded.append({'path':str(path),'reason':str(exc)})
    return sorted(found.values(),key=lambda x:(x['gate_status']!='recorded pass',x['name'].casefold())),excluded

def attest(model):
    """Hash bytes at load, bind vectors to base+adapter+tokenizer, reject stale selections."""
    paths=[Path(p) for p in model['files']]
    before=signature(paths)
    if before!=model['file_signature']: raise ValueError('Model changed since discovery. Refresh the inventory before loading.')
    hashes={str(p):file_hash(p) for p in paths}
    if signature(paths)!=before: raise ValueError('Model files changed while hashing; training may still be writing them.')
    base_hashes=[(Path(p).name,h) for p,h in hashes.items() if Path(p).parent==Path(model['base_path'])]
    base_digest=hashlib.sha256(json.dumps(sorted(base_hashes)).encode()).hexdigest()
    adapter_digest=hashes[str(Path(model['adapter_path'])/'adapter_model.safetensors')]
    canonical=sorted((Path(p).name,h) for p,h in hashes.items())
    out=dict(model,digest='peft-'+hashlib.sha256(json.dumps(canonical).encode()).hexdigest(),
             base_digest=base_digest,adapter_sha256=adapter_digest,source_hashes=hashes)
    return out

def unchanged(model):
    if signature(model['files'])!=model['file_signature']:
        raise ValueError('Loaded model files changed on disk. Unload and refresh; old directions cannot be reused silently.')
