"""Transfer existing tested numeric settings to live chat, never a prompt or answer."""
import json
from pathlib import Path
import streamlit as st
ROOT=Path(__file__).resolve().parent.parent

def apply(preset):
    s=preset['ui_settings'];name=preset['axis']
    for key in list(st.session_state):
        if key.startswith('live_dose_'):st.session_state[key]=0.
    st.session_state.update(live_emotions=[],live_behaviors=[],live_concepts=[],live_research=[name],
        live_high=True,live_op='add',live_previous_op='add',live_phase=s['cal_phase'],live_thinking=s['cal_thinking'],
        live_budget=s['cal_budget'],live_sampler_profile=s['cal_sampler_profile'],live_sampler_seed=s['cal_sampler_seed'])
    st.session_state['live_dose_'+name]=s['cal_dose']
    st.session_state['live_source_'+name]=s['cal_source_'+name]
    st.session_state['live_block_'+name]=s['cal_layer_'+name]

def render(engine,bank,busy):
    path=ROOT/'calibration/behavior-presets.json'
    if not path.exists():return
    eligible=[p for p in json.loads(path.read_text(encoding='utf-8')) if p['digest']==engine.model['digest'] and p['abi']==engine.abi and p['axis'] in bank and bank[p['axis']]['meta']['arrays_sha256']==p['arrays_sha256']]
    if not eligible:return
    with st.expander('Existing tested research settings, now usable in this chat'):
        for p in eligible:
            st.write(p['name']);st.warning(p['caution'])
            st.button('Apply numeric settings to chat',on_click=apply,args=(p,),disabled=busy,key='live_preset_'+p['axis'])
        st.caption('Only controls and sampling change. Your prompt, system instruction and conversation are untouched. This does not make the single-prompt result a general guarantee.')
