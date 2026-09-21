"""Replay recorded token-step control changes with baseline and matched random controls."""
import json
from .live_engine import stream
from .steering_controls import matched_random_controls
from .core import write_result

class Replay:
    def __init__(self,events):self.events=events
    def before_decode(self,step,phase):
        eligible=[e for e in self.events if e['effective_before_token']<=step]
        e=eligible[-1] if eligible else {'revision':0,'controls':[],'phase':'all','dose_limit':.5}
        return dict(e)
    def cancelled(self):return False

def compare(engine,bank,result,progress=None):
    events=result.get('control_events') or [{'revision':0,'effective_before_token':0,'controls':result.get('controls',[]),'phase':result.get('control_phase','all'),'dose_limit':result.get('dose_limit',.5)}]
    random=[dict(e,controls=matched_random_controls(engine,bank,e['controls'],e['dose_limit'],65013)) for e in events]
    configs=[('Baseline',[{'revision':0,'effective_before_token':0,'controls':[],'phase':'all','dose_limit':.5}]),('Intervention replay',events),('Equal-norm random replay',random)]
    settings={k:result[k] for k in ('max_tokens','temperature','seed','top_k','top_p','presence_penalty','scope')}
    out=[]
    for i,(name,schedule) in enumerate(configs):
        final=None
        for event in stream(engine,result['prompt'],bank,Replay(schedule),result['watch'],**settings):
            if event['type']=='done':final=event['result']
        final['label']=name;out.append(final)
        if progress:progress(i+1,3,name)
    record={'original_run':result['saved_to'],'rows':out,'prompt_unchanged':len({r['prompt_sha256'] for r in out})==1,
            'scope':'Three fresh native generations; replay uses original effective token steps, not wall-clock timing. Different outputs can have different phases at the same step. No reranking or output substitution.'}
    record['saved_to']=write_result(record,'live-counterfactual');return record

def replay_ui(engine,bank,result):
    import streamlit as st
    with st.expander('Compare the actual live steering schedule'):
        st.caption('Runs three fresh generations from the identical prompt and seed. The intervention schedule is replayed at its original token indices, including controls switched off mid-reply.')
        if st.button('Replay baseline / intervention / random',key='live_replay'):
            bar=st.progress(0)
            st.session_state['live_replay_result']=compare(engine,bank,result,lambda i,n,t:bar.progress(i/n,text=t))
        record=st.session_state.get('live_replay_result')
        if record and record['original_run']==result['saved_to']:
            cols=st.columns(3)
            for col,row in zip(cols,record['rows']):
                with col:
                    st.markdown('**'+row['label']+'**')
                    with st.expander('Emitted reasoning'):st.write(row['reasoning_text'])
                    st.write(row['answer_text']);st.caption(row['stop_reason'])
            st.download_button('Export live counterfactuals',json.dumps(record,indent=2),file_name='live-counterfactuals.json')
