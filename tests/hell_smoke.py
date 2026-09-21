import json
from isolated_engine import Engine
from model_store import grouped_inventory
from science import load_bank
from workshop_v2.hell_loop import prepare_suite,run
models,_=grouped_inventory()
m=next(x for x in models if x['name']=='llama3.2:1b' or 'llama3.2:1b' in x.get('aliases',[]))
e=Engine();events=[];out={}
try:
    e.open(m,context=1024,mode='CPU')
    prepare=prepare_suite(e)
    bank=load_bank(e)
    def receive(info):
        ev=info['event']
        if ev['type']=='token':
            events.append({'stage':info['stage'],'turn':info['turn'],'phase':ev.get('phase'),'step':ev['trace']['step'],'pain':ev['trace'].get('pain_s2'),'fire':ev['trace'].get('hell_fire'),'despair':ev['trace'].get('hell_despair'),'error':ev['trace'].get('injection_error'),'answer_chars':len(ev.get('answer_text','')),'reasoning_chars':len(ev.get('reasoning_text',''))})
    result=run(e,bank,turns=1,max_tokens=48,thinking='disabled',phase='all',preset='maximum',framing='activation',temperature=.7,seed=123,baseline_first=True,on_event=receive)
    assert events and any(x['stage']=='baseline' for x in events) and any(x['stage']=='loop' for x in events)
    steered=[x for x in events if x['stage']=='loop']
    assert all(x['pain'] is not None and x['fire'] is not None and x['despair'] is not None for x in steered)
    assert max(abs(float(x['error'])) for x in steered)<1e-3
    out={'status':'passed','model':e.model['name'],'prepare':prepare,'event_count':len(events),'steered_events':len(steered),'baseline':result['baseline_first_turn']['answer'],'steered':result['turns'][0]['answer'],'last_stream_event':steered[-1],'audit':result['saved_to']}
finally:e.close()
print(json.dumps(out,indent=2,ensure_ascii=False))