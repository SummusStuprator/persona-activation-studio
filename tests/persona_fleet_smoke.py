"""Bounded real persona generations and reversible-hook checks, offline."""
from pathlib import Path
import argparse
import json
import time
import numpy as np
from isolated_engine import Engine
from workshop_v2.persona_store import discover
from workshop_v2.persona_precision import rounding_metrics
from workshop_v2.core import run

parser=argparse.ArgumentParser()
parser.add_argument('--model',action='append',required=True)
parser.add_argument('--output',required=True)
parser.add_argument('--device',choices=['Auto','CPU','CUDA'],default='Auto')
parser.add_argument('--precision',choices=['auto','bf16','nf4'],default='auto')
args=parser.parse_args()
models,_=discover()
report=dict(scope='Runtime/generation smoke, not personality accuracy or subjective experience',models=[])
output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
for name in args.model:
    model=next((m for m in models if m['name']==name),None)
    if model is None:raise RuntimeError('Persona not discovered: '+name)
    engine=Engine();item={'model':name};started=time.monotonic()
    try:
        engine.open(model,context=512,mode=args.device,precision=args.precision)
        prompt=engine.chat([{'role':'user','content':'Describe an ordinary rainy afternoon in two sentences.'}])
        ids=engine.tokenize(prompt);engine.reset();engine.evaluate(ids)
        baseline_logits=engine.logits();before=engine.capture(0)
        table=np.zeros_like(before);layer=engine.layers//2
        vector=np.random.default_rng(123).normal(size=engine.dim).astype(np.float32)
        table[layer]=vector/np.linalg.norm(vector)*np.linalg.norm(before[layer])*.05
        engine.reset();engine.steer(table,1.,'add');engine.evaluate(ids)
        pre,post=engine.capture(0),engine.capture(1)
        numeric=rounding_metrics(table,[{'mode':'add'}],pre,post,True,engine.model['activation_dtype'])
        assert np.linalg.norm(post[layer]-pre[layer])>0
        engine.clear_steering();engine.reset();engine.evaluate(ids)
        reset_error=float(np.max(np.abs(baseline_logits-engine.logits())))
        assert reset_error < 1e-5,reset_error
        result=run(engine,prompt,{},max_tokens=32,temperature=0.,seed=123)
        assert result['token_ids'] and result['answer_text'].strip()
        item.update(status='passed',answer=result['answer_text'],tokens=len(result['token_ids']),
            stop_reason=result['stop_reason'],abi=engine.abi,plan=engine.load_plan,hook_check=numeric,
            reset_logit_error=reset_error,audit=result['saved_to'])
    except Exception as exc:
        item.update(status='failed',error=f'{type(exc).__name__}: {exc}')
    finally:
        engine.close()
    item['seconds']=round(time.monotonic()-started,2)
    report['models'].append(item)
    output.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    print(name,item['status'],item['seconds'],flush=True)
if any(m['status']!='passed' for m in report['models']):raise SystemExit(1)
print('PASS: all requested persona smoke checks; workers closed.')
