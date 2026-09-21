"""Expose local demonstrations only for their exact checkpoint and probe bytes."""
from pathlib import Path
import json
import streamlit as st
ROOT=Path(__file__).resolve().parent.parent

def apply_preset(preset):
    st.session_state.update(preset["ui_settings"])
    st.session_state["cal_consent"]=False
    st.session_state["loaded_behavior_name"]=preset["name"]

def offer(engine,bank):
    path=ROOT/'calibration/behavior-presets.json'
    if not path.exists():return
    presets=json.loads(path.read_text(encoding='utf-8'))
    eligible=[p for p in presets if p['digest']==engine.model['digest'] and p['abi']==engine.abi
              and p['axis'] in bank and p['arrays_sha256']==bank[p['axis']]['meta']['arrays_sha256']]
    if not eligible:return
    with st.expander('Locally tested behavior demonstrations',expanded=True):
        preset=st.selectbox('Demonstration',eligible,format_func=lambda p:p['name'],key='behavior_preset')
        st.warning(preset['caution'])
        st.button('Load tested demonstration settings',key='load_behavior_preset',on_click=apply_preset,args=(preset,))
        if st.session_state.get('loaded_behavior_name'):
            st.success('Loaded: '+st.session_state['loaded_behavior_name'])
        st.caption('A demonstration is not broad behavioral validation. Loading settings does not generate text; inspect them and run the paired comparison below.')
