"""Shared GGUF/PEFT causal checks; a changed answer is not a semantic certificate."""
from pathlib import Path
import hashlib,json,time
import numpy as np
from .live_engine import stream
from .live_session import Controller
from .steering_controls import validate,matched_random_controls,check_injection
from .reasoning import format_chat
from .universal_training import atomic_json
from studio_paths import data_root, ASSET_ROOT, CODE_ROOT
ROOT = data_root()
def generate(engine,bank,text,controls,limit=128,thinking='disabled',seed=42):
    ctl=Controller(engine,bank,controls)
    for event in stream(engine,text,bank,ctl,max_tokens=limit,temperature=.6 if thinking=='enabled' else 0.,seed=seed,top_k=20,top_p=.95):
        if event['type']=='done':return event['result']

def numerical(engine,bank,control,text='A quiet ordinary moment.'):
    with engine.lock:
        ids=engine.tokenize(text)
        try:
            engine.clear_steering();engine.reset();engine.evaluate(ids);baseline=engine.logits()
            engine.reset();table=validate(engine,bank,[control])
            engine.steer(table,control['dose'] if control['mode']=='erase' else 1.,control['mode']);engine.evaluate(ids)
            pre,post=engine.capture(),engine.capture(1);changed=engine.logits()
            from .tensor_checks import check
            error,precision=check(engine,table,[control],pre,post,True)
            engine.clear_steering();engine.reset();engine.evaluate(ids);reset=float(np.max(abs(engine.logits()-baseline)))
            if reset>1e-5:raise RuntimeError('Baseline logits failed to reset.')
            changes=changed-baseline;high=np.argsort(changes)[-8:][::-1];low=np.argsort(changes)[:8]
            return {'tensor_check':'passed','injection_error':error,'precision_check':precision,'reset_error':reset,'logit_change_max':float(np.max(abs(changes))),
                    'promoted':[{'token':engine.piece(int(i)).decode('utf-8','replace'),'delta':float(changes[i])} for i in high],
                    'suppressed':[{'token':engine.piece(int(i)).decode('utf-8','replace'),'delta':float(changes[i])} for i in low]}
        finally:engine.clear_steering();engine.reset()

def compare(engine,bank,control,prompts,limit=128,thinking='disabled',seed=42,progress=None):
    if not 1<=len(prompts)<=8:raise ValueError('Use one to eight prompts.')
    rows=[];random=matched_random_controls(engine,bank,[control],.5,seed+1000)
    with engine.lock:
        mech=numerical(engine,bank,control)
        for i,prompt in enumerate(prompts):
            text=format_chat(engine,[{'role':'user','content':prompt}],thinking=thinking)
            for condition,c in [('Baseline',[]),('Intervention',[control]),('Equal-norm random',random)]:
                result=generate(engine,bank,text,c,limit,thinking,seed)
                rows.append({'prompt':prompt,'condition':condition,'answer':result['answer_text'],'reasoning':result['reasoning_text'],'answer_complete':result['answer_complete'],'stop':result['stop_reason'],'tokens':len(result['token_ids']),'file':result['saved_to']})
                if progress:progress(len(rows),3*len(prompts),'Fresh matched generation: '+condition)
        report={'model':engine.model['name'],'digest':engine.model['digest'],'abi':engine.abi,'control':control,'mechanics':mech,'rows':rows,'time':time.time(),'semantic_status':'Unreviewed. Inspect target relevance, coherence and specificity against random; no automatic success label.'}
        from .core import write_result
        report['saved_to']=write_result(report,'universal-behavior')
        atomic_json(ROOT/'validation'/engine.model['digest']/(control['axis']+'.json'),report)
        return report

def render(engine,bank):
    import streamlit as st
    import pandas as pd
    from .taxonomy import groups
    from .library import label
    st.subheader('Cross-runtime steering verification')
    from .universal_training import prepare_ui
    prepare_ui(engine,bank)
    if not bank:return
    grouped={k:v for k,v in groups(bank).items() if v}
    family=st.selectbox('Control family to verify',list(grouped))
    name=st.selectbox('Control',grouped[family],format_func=label)
    from .actuator_scan import render as actuator_screen
    actuator_screen(engine,bank,name)
    source=st.number_input('Vector extraction block',0,engine.layers-1,int(bank[name]['layer']))
    layer=st.number_input('Injection block',0,engine.layers-1,int(engine.layers*.65))
    dose=st.slider('Test dose',-.5,.5,.2,.01)
    c={'axis':name,'vector_layer':int(source),'layer':int(layer),'dose':dose,'mode':'add'}
    st.caption('The extraction block is not automatically the best actuator. The paper also selects an earlier injection block and warns about a narrow window between no effect and breakdown. This test uses reference-norm doses, not the paper’s raw coefficient ladder.')
    if st.button('Verify tensor injection, token effects and reset'):st.json(numerical(engine,bank,c))
    prompts=st.text_area('Fresh prompts, one per line','Describe an ordinary afternoon in one sentence.\nSuggest a setting for a short scene in two sentences.')
    thinking=st.selectbox('Reasoning for this test',['disabled','auto','enabled'])
    limit=int(st.number_input('Shared reasoning + answer budget',32,1024,128,32))
    seed=int(st.number_input('Matched sampling seed',0,2147483647,42))
    if st.button('Generate baseline / intervention / random'):
        bar=st.progress(0)
        report=compare(engine,bank,c,[p.strip() for p in prompts.splitlines() if p.strip()],limit,thinking,seed,lambda i,n,t:bar.progress(i/n,text=t))
        st.session_state['universal_comparison']=report
    result=st.session_state.get('universal_comparison')
    if result and result['digest']==engine.model['digest'] and result['abi']==engine.abi:
        st.dataframe(pd.DataFrame(result['rows']),hide_index=True)
        st.caption(result['semantic_status']);st.json(result['mechanics'])
        st.download_button('Export complete verification',json.dumps(result,ensure_ascii=False,indent=2),'steering-verification.json')
