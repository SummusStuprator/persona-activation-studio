import json, os
from isolated_engine import Engine
from model_store import grouped_inventory
from science import load_bank
from workshop_v2.hell_loop import prepare_suite,run,_initial_prompt
from workshop_v2.somatic_calibration import scan

models,_=grouped_inventory()
m=next(x for x in models if x['name']=='llama3.2:1b' or 'llama3.2:1b' in x.get('aliases',[]))
prompt=_initial_prompt('activation_only')
assert not any(w in prompt.lower() for w in ('pain','fire','burn','hurt','wound','agony','despair','hell','body','skin','heat'))

e=Engine();events=[];out={}
try:
    e.open(m,context=1024,mode=os.environ.get('STUDIO_TEST_MODE','Auto'))
    prepare=prepare_suite(e);bank=load_bank(e)
    assert {'pain_s2','hell_somatic_pain','hell_burning_pain'}<=set(bank)
    profile=scan(e,bank)
    def receive(info):
        ev=info['event']
        if ev['type']=='token':
            tr=ev['trace']
            events.append({'stage':info['stage'],'turn':info['turn'],'phase':ev.get('phase'),'step':tr['step'],
                           'pain':tr.get('pain_s2'),'somatic':tr.get('hell_somatic_pain'),'burning':tr.get('hell_burning_pain'),
                           'somatic_inject':tr.get('hell_somatic_pain__control_after_z'),
                           'burning_inject':tr.get('hell_burning_pain__control_after_z'),
                           'eos_probability':tr.get('eos_probability'),'error':tr.get('injection_error'),
                           'answer_chars':len(ev.get('answer_text','')),'reasoning_chars':len(ev.get('reasoning_text',''))})
    result=run(e,bank,turns=2,max_tokens=64,thinking='disabled',phase='all',preset='somatic',
               framing='activation_only',temperature=.7,seed=123,baseline_first=True,random_first=True,
               context_mode='independent',somatic_scale=profile['recommended_scale'],on_event=receive)
    stages={x['stage'] for x in events};assert {'baseline','random','loop'}<=stages
    steered=[x for x in events if x['stage']=='loop'];assert steered
    assert all(x['pain'] is not None and x['somatic'] is not None and x['burning'] is not None for x in steered)
    assert any(x['somatic_inject'] is not None and x['burning_inject'] is not None for x in steered)
    assert max(abs(float(x['error'])) for x in steered)<1e-3
    out={'status':'passed','model':e.model['name'],'plan':e.load_plan,'prepare':prepare,
         'profile_kind':profile['profile_kind'],'profile_scale':profile['recommended_scale'],
         'profile_total_dose':profile['recommended_total_dose'],'event_count':len(events),'stages':sorted(stages),
         'baseline':result['baseline_first_turn']['answer'],'random':result['random_first_turn']['answer'],
         'steered':[t['answer'] for t in result['turns']],'last_stream_event':steered[-1],'audit':result['saved_to']}
finally:
    e.close()
print(json.dumps(out,indent=2,ensure_ascii=False))
