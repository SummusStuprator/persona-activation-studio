"""Actual offline CUDA/NF4 training mechanics with a tiny temporary checkpoint."""
from pathlib import Path
import json
import tempfile
import torch
import bitsandbytes as bnb
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import Qwen2Config, Qwen2ForCausalLM, PreTrainedTokenizerFast, TrainingArguments
from persona.trainer_v4 import load_base_model, CapabilityPreservingTrainer, PersonaDataset, DynamicCollator
from workshop_v2.resource_policy import ModelLaunchLease, ModelLease, checkpoint

assert torch.cuda.is_available(), 'CUDA is required for this opt-in test.'
with ModelLaunchLease(),ModelLease(),tempfile.TemporaryDirectory(prefix='studio-qlora-gpu-') as temp:
    checkpoint()
    root=Path(temp);base_path=root/'base'
    cfg=Qwen2Config(vocab_size=64,hidden_size=64,intermediate_size=128,
        num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=2,
        max_position_embeddings=64,bos_token_id=1,eos_token_id=2)
    torch.manual_seed(123)
    Qwen2ForCausalLM(cfg).save_pretrained(base_path)
    vocab={f't{i}':i for i in range(64)}
    tokenizer=PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel(vocab,unk_token='t0')),
        unk_token='t0',bos_token='t1',eos_token='t2',pad_token='t0')
    tokenizer.save_pretrained(base_path)
    base,tokenizer,bf16=load_base_model(str(base_path),training=True)
    quantized=sum(isinstance(m,bnb.nn.Linear4bit) for m in base.modules())
    assert quantized>0,'NF4 layers were not loaded.'
    base=prepare_model_for_kbit_training(base,use_gradient_checkpointing=False)
    model=get_peft_model(base,LoraConfig(r=2,lora_alpha=4,target_modules=['q_proj','v_proj'],
        lora_dropout=0.,task_type='CAUSAL_LM'))
    initial={n:p.detach().clone() for n,p in model.named_parameters() if p.requires_grad}
    ids=[1,3,4,5,6,7,2]
    features=[dict(input_ids=ids,attention_mask=[1]*len(ids),labels=[-100]+ids[1:],replay_mask=0)]*4
    args=TrainingArguments(output_dir=str(root/'run'),max_steps=2,
        per_device_train_batch_size=1,gradient_accumulation_steps=1,learning_rate=.001,
        report_to=[],save_strategy='no',logging_strategy='no',remove_unused_columns=False,
        bf16=bf16,fp16=not bf16,disable_tqdm=True)
    trainer=CapabilityPreservingTrainer(model=model,args=args,
        train_dataset=PersonaDataset(features),data_collator=DynamicCollator(tokenizer))
    result=trainer.train()
    changed=any(not torch.equal(initial[n],p.detach()) for n,p in model.named_parameters() if n in initial)
    assert changed,'Training did not update the LoRA parameters.'
    assert trainer.state.global_step==2
    assert torch.isfinite(torch.tensor(result.training_loss))
    print(json.dumps(dict(status='passed',cuda_device=torch.cuda.get_device_name(0),
        nf4_layers=quantized,optimizer_steps=2,adapter_changed=changed,
        scope='tiny temporary production-loader/production-trainer CUDA mechanics; not a persona-quality result')))
