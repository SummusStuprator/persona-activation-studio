"""Real, offline optimizer/save/resume check using Studio's trainer and tiny LoRA."""
from pathlib import Path
import sys
from studio_paths import CODE_ROOT as ROOT
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from types import SimpleNamespace
import json
import os
import tempfile
try:
    import torch
    from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
    from transformers import Qwen2Config, Qwen2ForCausalLM, TrainerCallback, TrainingArguments, set_seed
except ModuleNotFoundError as exc:
    print({'status':'skipped','reason':'optimizer resume smoke requires the optional training dependencies; install Studio [train]','missing':exc.name})
    raise SystemExit(0)
from persona.trainer_v4 import (CapabilityPreservingTrainer, DynamicCollator,
    PersonaDataset, find_latest_valid_checkpoint, load_anchor_rows)

os.environ['HF_HUB_OFFLINE'] = '1'
torch.set_num_threads(2)

class StopAtTwo(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == 2:
            control.should_training_stop = True
        return control


def network():
    set_seed(123)
    cfg = Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
        max_position_embeddings=64, bos_token_id=1, eos_token_id=2)
    return get_peft_model(Qwen2ForCausalLM(cfg), LoraConfig(r=2,lora_alpha=4,
        target_modules=['q_proj','v_proj'],lora_dropout=0.,task_type='CAUSAL_LM'))

def trainer(folder, stop=False):
    features=[]
    for index in range(12):
        ids=[1,3+index,18+index,35,36,37,38,2]
        replay = int(index % 3 == 0)
        features.append(dict(input_ids=ids,attention_mask=[1]*8,
            labels=[-100]*8 if replay else [-100]*3+ids[3:],replay_mask=replay))
    args=TrainingArguments(output_dir=str(folder),max_steps=4,
        per_device_train_batch_size=1,gradient_accumulation_steps=1,
        learning_rate=.001,save_steps=2,save_strategy='steps',save_total_limit=2,
        report_to=[],use_cpu=True,bf16=False,fp16=False,disable_tqdm=True,
        remove_unused_columns=False,seed=123,data_seed=123,logging_strategy='no')
    return CapabilityPreservingTrainer(model=network(),args=args,
        train_dataset=PersonaDataset(features),
        data_collator=DynamicCollator(SimpleNamespace(pad_token_id=0,eos_token_id=2)),
        callbacks=[StopAtTwo()] if stop else [])


with tempfile.TemporaryDirectory(prefix='studio-optimizer-smoke-') as temporary:
    root=Path(temporary)
    uninterrupted=trainer(root/'full');uninterrupted.train()
    expected={k:v.detach().clone() for k,v in get_peft_model_state_dict(uninterrupted.model).items()}
    partial=trainer(root/'resume',stop=True);partial.train()
    assert partial.state.global_step == 2
    checkpoint=find_latest_valid_checkpoint(root/'resume')
    assert checkpoint and checkpoint.name == 'checkpoint-2'
    resumed=trainer(root/'resume');resumed.train(resume_from_checkpoint=str(checkpoint))
    assert resumed.state.global_step == 4
    actual=get_peft_model_state_dict(resumed.model)
    difference=max((expected[k]-actual[k]).abs().max().item() for k in expected)
    assert difference < 1e-6, difference
    anchors=root/'anchors.json'
    anchors.write_text(json.dumps([{'messages':[
        {'role':'user','content':'A normal greeting'},
        {'role':'assistant','content':'Hello, good to see you.'}]}]),encoding='utf-8')
    assert len(load_anchor_rows(anchors)) == 1
    print(json.dumps(dict(status='passed',trainer='CapabilityPreservingTrainer',
        checkpoint='checkpoint-2',resumed_step=4,max_adapter_difference=difference,
        anchor_message_format='passed',scope='tiny offline LoRA; supervised and KL replay; no persona-quality claim')))
