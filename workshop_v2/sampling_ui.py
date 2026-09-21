"""Visible, reproducible sampler controls shared by chat and comparisons."""
import streamlit as st
from .sampling import qwen_defaults

def render(engine,thinking,key='sampler'):
    qwen=engine.model.get('architecture','') in ('qwen35','qwen35moe')
    choices=['Greedy research','Custom']
    if qwen:choices.insert(1,'Qwen general-task profile')
    profile=st.selectbox('Sampling profile',choices,index=1 if qwen else 0,key=key+'_profile')
    if profile=='Qwen general-task profile':
        settings=qwen_defaults(thinking!='disabled')
        st.caption('Qwen model-card general-task parameter profile. The local GGUF may be a modified checkpoint; this does not guarantee the published model behavior.')
    elif profile=='Greedy research':
        settings=dict(temperature=0.,top_k=40,top_p=1.,presence_penalty=0.)
        if qwen and thinking!='disabled':
            st.warning('Greedy decoding can be a poor chat default for this reasoning checkpoint. In the local audit it sometimes exhausted the budget without producing a final answer.')
    else:
        settings=dict(temperature=st.slider('Temperature',0.,2.,.7,.05,key=key+'_temperature'),
            top_k=int(st.number_input('Top-k (0 means unlimited)',0,min(engine.vocab_size,1000),20,key=key+'_top_k')),
            top_p=st.slider('Top-p',.05,1.,.95,.05,key=key+'_top_p'),
            presence_penalty=st.slider('Presence penalty',0.,2.,0.,.1,key=key+'_presence'))
    seed=int(st.number_input('Sampling seed',0,2147483647,42,key=key+'_seed'))
    st.caption('Sampler: '+', '.join(f'{k}={v}' for k,v in settings.items())+f', seed={seed}. Presence penalty uses generated reply tokens; plotted model probabilities are before sampling adjustments.')
    return settings,seed
