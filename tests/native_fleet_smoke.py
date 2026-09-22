"""Opt-in real GGUF generation, numeric injection and reset checks."""
from pathlib import Path
import argparse
import json
import time
import numpy as np
from isolated_engine import Engine
from model_store import grouped_inventory
from workshop_v2.core import run
from workshop_v2.reasoning import format_chat
from workshop_v2.steering_controls import check_injection

parser=argparse.ArgumentParser()
parser.add_argument('--model',action='append',required=True)
parser.add_argument('--output',required=True)
parser.add_argument('--require-gpu',action='store_true')
parser.add_argument('--max-tokens',type=int,default=192)
parser.add_argument('--gpu-layers',type=int)
args=parser.parse_args()
models,_=grouped_inventory()
report={'scope':'Bounded generation and native mechanics; not behavioral specificity','models':[]}
output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
for name in args.model:
    model=next((m for m in models if m['name']==name or name in m.get('aliases',[])),None)
    if model is None:raise RuntimeError('GGUF not discovered: '+name)
    engine=Engine();item={'model':name};started=time.monotonic()
    try:
        engine.open(model,context=512,mode='Manual' if args.gpu_layers is not None else 'Auto',manual=args.gpu_layers or 0)
        if args.require_gpu:assert engine.load_plan['gpu_layers']>0,engine.load_plan
        prompt=format_chat(engine,[{'role':'user','content':'Describe an ordinary rainy afternoon in two sentences.'}],thinking='disabled',preserve=False)
        ids=engine.tokenize(prompt);engine.reset();engine.evaluate(ids)
        baseline_logits=engine.logits();before=engine.capture(0)
        table=np.zeros_like(before);layer=engine.layers//2
        vector=np.random.default_rng(123).normal(size=engine.dim).astype(np.float32)
        table[layer]=vector/np.linalg.norm(vector)*np.linalg.norm(before[layer])*.05
        engine.reset();engine.steer(table,1.,'add');engine.evaluate(ids)
        pre,post=engine.capture(0),engine.capture(1)
        error=check_injection(table,[{'mode':'add'}],pre,post,True)
        assert np.linalg.norm(post[layer]-pre[layer])>0
        engine.clear_steering();engine.reset();engine.evaluate(ids)
        reset_error=float(np.max(np.abs(baseline_logits-engine.logits())))
        assert reset_error<1e-4,reset_error
        result=run(engine,prompt,{},max_tokens=args.max_tokens,temperature=0.,seed=123)
        assert result['token_ids'] and result['answer_text'].strip()
        item.update(status='passed',tokens=len(result['token_ids']),answer=result['answer_text'],
            stop_reason=result['stop_reason'],plan=engine.load_plan,abi=engine.abi,
            injection_error=error,reset_logit_error=reset_error,audit=result['saved_to'])
    except Exception as exc:
        import traceback
        item.update(status='failed',error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc())
    finally:
        engine.close()
    item['seconds']=round(time.monotonic()-started,2);report['models'].append(item)
    output.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    print(name,item['status'],item['seconds'],flush=True)
if any(m['status']!='passed' for m in report['models']):raise SystemExit(1)
print('PASS: all requested GGUF mechanics checks; workers closed.')
