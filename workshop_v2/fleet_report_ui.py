"""Display only reports produced by this workshop's fleet test runner."""
from pathlib import Path
import json
import streamlit as st
import pandas as pd
ROOT=Path(__file__).resolve().parent.parent

def render():
    st.subheader('Fleet verification reports')
    st.caption('Tested, failed, and resource-deferred are separate outcomes. Generation and tensor checks do not certify emotional or conceptual specificity.')
    paths=sorted((ROOT/'reports').glob('fleet-r8*.json'))
    if not paths:st.info('No fleet report has been produced yet.');return
    selected=st.selectbox('Completed or in-progress report',paths,format_func=lambda p:p.name)
    report=json.loads(selected.read_text(encoding='utf-8'))
    st.dataframe(pd.DataFrame([{'model':m['name'],'status':m['status'],'directions':len(m['directions']) if 'directions' in m else 'not inspected','tensor_check':m.get('mechanics',{}).get('tensor_check','not run'),'simple_math_correct':m.get('mechanics',{}).get('arithmetic_correct'),'reason':m.get('reason','')} for m in report.get('models',[])]),hide_index=True)
    st.write('Models in this inventory snapshot:',len(report.get('inventory_names',[])))
    with st.expander('Complete evidence and per-model outputs'):st.json(report)
    st.download_button('Download fleet evidence',json.dumps(report,indent=2),'fleet-evidence.json')
