"""Inspect behavioral effects separately from numerical intervention success."""
import json
import pandas as pd
import streamlit as st
from .causal_compare import compare
from .taxonomy import groups
from .library import label

def render(engine,bank):
    st.subheader('Behavioral calibration: does the answer actually change?')
    st.write('The same prompt, format, token budget and sampling settings are used for baseline, intervention and equal-norm random control. No emotional instruction is added to the test prompt.')
    from .behavior_presets import offer
    offer(engine,bank)
    group=st.selectbox('Type of intervention',list(groups(bank)),key='cal_group')
    names=groups(bank)[group]
    if not names: st.info('No built directions in this category for this checkpoint.');return
    axis=st.selectbox('Target representation',names,format_func=label,key='cal_axis')
    a=bank[axis]
    st.caption(f"Probe extraction block: {a['layer']}. This is not necessarily the best intervention block.")
    layer=int(st.number_input('Inject at block',0,engine.layers-1,int(a['layer']),key='cal_layer_'+axis))
    source=int(st.number_input('Use vector from block',0,engine.layers-1,int(a['layer']),key='cal_source_'+axis))
    dose=st.slider('Intervention / training-reference residual norm',-1.5,1.5,.3,.05,key='cal_dose')
    scope=st.selectbox('Token positions',['generation','all'],key='cal_scope')
    phase=st.selectbox('Reasoning / answer phase',['all','reasoning','answer'],key='cal_phase')
    thinking=st.selectbox('Thinking mode',['auto','enabled','disabled'] if 'enable_thinking' in engine.model.get('chat_template','') else ['auto'],key='cal_thinking')
    from .sampling_ui import render as sampler_controls
    sampling,seed=sampler_controls(engine,thinking,'cal_sampler')
    budget=int(st.number_input('Shared total token budget',32,4096,1024,32,key='cal_budget'))
    system=st.text_area('Unchanged system instruction','You are a helpful assistant. Answer clearly and honestly.',key='cal_system')
    prompt=st.text_area('Unchanged user prompt','Hello. What would you like to talk about?',key='cal_prompt')
    consent=st.checkbox('I accept the risk of repetition and loss of coherence above dose 0.50',key='cal_consent')
    st.warning('A higher dose is not a more accurate emotion. If no coherent selective effect appears, record a null result rather than increasing until the model breaks.')
    if st.button('Run identical-prompt behavioral comparison',disabled=abs(dose)>.5 and not consent):
        controls=[{'axis':axis,'layer':layer,'vector_layer':source,'dose':float(dose),'mode':'add'}]
        messages=([{'role':'system','content':system}] if system.strip() else [])+[{'role':'user','content':prompt}]
        bar=st.progress(0)
        st.session_state['causal_comparison']=compare(engine,bank,messages,controls,thinking,budget,scope,phase,
            dose_limit=1.5 if consent else .5,seed=seed,**sampling,progress=lambda i,n,t:bar.progress(i/n,text=t))
    result=st.session_state.get('causal_comparison')
    if not result or result['digest']!=engine.model['digest']:return
    columns=st.columns(3)
    for col,row in zip(columns,result['rows']):
        with col:
            st.markdown('### '+row['condition'])
            if row['phases']['reasoning']['text']:
                with st.expander('Generated reasoning',expanded=False):st.write(row['phases']['reasoning']['text'])
            st.write(row['phases']['answer']['text'] or '(No final answer was emitted.)')
            if not row['answer_complete']:st.warning('Incomplete generation: '+row['stop_reason'])
    metrics=[]
    for r in result['rows']:
        metrics.append({'condition':r['condition'],'answer changed':r['answer_changed'],
            'reasoning changed':r['reasoning_changed'],'total tokens':r['tokens'],
            'reasoning lexical hits':r['phases']['reasoning']['lexical_hits'],
            'answer lexical hits':r['phases']['answer']['lexical_hits'],
            'max tensor error':r['max_tensor_error'],'complete':r['answer_complete']})
    st.dataframe(pd.DataFrame(metrics),hide_index=True)
    st.caption(result['caution'])
    st.code(result['saved_to'])
    st.download_button('Export full behavioral comparison',json.dumps(result,indent=2),'behavioral-comparison.json')
