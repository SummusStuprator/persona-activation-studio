"""Visible counterfactuals for the last completed chat turn."""
import json
import streamlit as st
from .causal_compare import replay_comparison

def render(engine,bank,recorded):
    if not recorded or not recorded.get('controls'):return
    if not any(float(c['dose']) for c in recorded['controls']):return
    with st.expander('Did steering change this reply? Compare the same history'):
        st.caption('The original steered reply is retained. Baseline and random controls use the exact same prompt, history, budget, seed and sampling settings. These runs execute no tools.')
        if st.button('Compare this reply with baseline and random control',key='compare_last_reply'):
            bar=st.progress(0)
            st.session_state['reply_comparison']=replay_comparison(engine,bank,recorded,
                lambda i,n,t:bar.progress(i/n,text=t))
        report=st.session_state.get('reply_comparison')
        if not report or report['original_run']!=recorded['saved_to']:return
        for column,row in zip(st.columns(3),report['rows']):
            with column:
                st.markdown('**'+row['condition']+'**')
                if row['phases']['reasoning']['text']:
                    with st.expander('Emitted reasoning'):st.write(row['phases']['reasoning']['text'])
                st.write(row['phases']['answer']['text'] or '(No final answer.)')
                st.caption(row['stop_reason'])
        st.caption(report['caution'])
        st.download_button('Save exact-reply comparison',json.dumps(report,indent=2),'reply-comparison.json')
