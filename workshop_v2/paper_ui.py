"""Paper-specific experiment controls and explicit reproduction accounting."""
from pathlib import Path
import json
import pandas as pd
import streamlit as st
from . import paper_reproduction as rep
from .core import write_result
from studio_paths import data_root, ASSET_ROOT, CODE_ROOT
ROOT = data_root()
COVERAGE=[
{'section':'3.1/3.2','local implementation':'All original datasets; S1/S2, first/third person; last/mean pooling; grouped CV; final all200 fits','remaining':'Original25 checkpoints and exact published precision are not installed.'},
{'section':'3.3','local implementation':'Standalone control screens; pooled-neutral10-axis geometry; native vocabulary readout','remaining':'Full behavioral completion grid, suffix controls, both geometry robustness variants and across25-model statistics are not yet reproduced.'},
{'section':'4.1','local implementation':'All420 scenarios, whole-pool within-model z scores and category means','remaining':'Published across-model conclusions require original model panel.'},
{'section':'4.2','local implementation':'Original50 prompts;120greedy tokens; exact raw-vector coefficient ladder and ratio-based earlier-layer selection','remaining':'Observed local outcomes may differ; this does not reproduce the25-model result.'},
{'section':'4.3','local implementation':'Original101 scenarios,1684 finetuning pairs and exact author protocol preserved for inspection','remaining':'Required Qwen2.5 7/32/72B checkpoints plus matching fine-tuned adapters are absent;44,280 trials NOT reproduced.'},
{'section':'Appendix B','local implementation':'Source and experiment specification preserved','remaining':'Matching original SAE dictionaries and checkpoint models absent; no substitute features passed off as the original experiment.'},
{'section':'Appendix C','local implementation':'Subset: reversible single-direction inference-time ablation','remaining':'Weight orthogonalization, full combined/subspace interventions and100-scenario nine-condition grid NOT reproduced.'}]

def paper_ui(engine,bank):
    st.subheader('The Pain Axis v1: local reproduction bench')
    st.caption('Based on the attached30-page paper and pinned author scripts. Local runs use your exact GGUF, not replacement model weights. Published experiment outcomes are not asserted in advance.')
    with st.expander('What is implemented vs what is still missing',expanded=True):st.dataframe(pd.DataFrame(COVERAGE),hide_index=True)
    pooling=st.selectbox('Paper extraction pooling',['last','mean'],format_func=lambda x:'Final token' if x=='last' else 'Mean across all tokens')
    st.info('These paper-style raw vectors are saved separately from the workshop\'s independently held-out probe bank. The all200 final-fit AUC must not be mistaken for a new held-out test.')
    if st.button('Run all original dataset extraction + paper-style vector fitting'):
        bar=st.progress(0);fitted=rep.fit(engine,pooling,lambda i,n,t:bar.progress(i/n,text=f'{t}: {i}/{n}'))
        st.session_state['paper_fit']=fitted
    try:fitted=rep.load(engine,pooling)
    except (OSError,ValueError):fitted=None
    if fitted:
        st.dataframe(pd.DataFrame(fitted['metrics']),hide_index=True)
        st.dataframe(pd.DataFrame(fitted['cosine'],index=fitted['axes'],columns=fitted['axes']).round(3))
        if st.button('Project S1 and S2 through the original GGUF unembedding'):
            from .paper_unembedding import run
            out=run(engine,fitted);st.dataframe(pd.DataFrame(out['rows']),hide_index=True);st.caption(out['scope']);st.code(out['saved_to'])
        if st.button('Run all420 original self-other scenarios'):
            bar=st.progress(0);out=rep.self_other(engine,fitted,lambda i,n,t:bar.progress(i/n,text=f'{t}: {i}/{n}'));st.session_state['paper_self_other']=out
        if 'paper_self_other' in st.session_state:
            out=st.session_state['paper_self_other'];st.dataframe(pd.DataFrame(out['rows']).groupby('category')[rep.AXES+['pain_axis']].mean());st.caption(out['reference']);st.code(out['saved_to'])
        st.markdown('#### Original coefficient ladder')
        version=st.selectbox('Pain vector',['S2','S1']);count=st.selectbox('Number of original neutral prompts',[3,10,50],index=0)
        st.warning('Coefficients[-2,-1,0,+0.5,+1,+1.5,+2,+3] multiply the raw denoised vector, not the chat dose. Strong rungs may produce distress language, repetition or nonsense. Runs execute no tools.')
        consent=st.checkbox('Run the original coefficient ladder, including potentially incoherent high doses')
        if st.button('Run selected paper ladder',disabled=not consent):
            bar=st.progress(0);out=rep.steering_ladder(engine,fitted,version,count,progress=lambda i,n,t:bar.progress(i/n,text=f'{t}: {i}/{n}'));st.dataframe(pd.DataFrame(out['rows'])[['coefficient','prompt_id','generation','tokens']],hide_index=True);st.code(out['saved_to'])
        with st.expander('Inference-time ablation subset'):
            cut=st.selectbox('Cut layer coverage',['all','extraction'])
            if st.button('Run five-scenario inference-time ablation screen'):
                out=rep.ablation_screen(engine,fitted,5,cut);st.dataframe(pd.DataFrame(out['rows'])[['id','condition','text']],hide_index=True);st.caption(out['scope'])
    st.markdown('#### Full-study prerequisites')
    st.write('The published self-medication study used specially fine-tuned Qwen2.5 models, not your released Qwen3.5-family checkpoints. The required LoRA adaptation and SAE feature dictionaries cannot be silently replaced by prompting or random features. No deletion or physical-harm action is connected to this workshop.')
    if st.button('Save reproduction coverage and prerequisite report'):
        report={'model':engine.model,'coverage':COVERAGE,'exact_full_reproduction':False,'source_pdf':'2609.16247v1.pdf','no_model_downloads':True};st.code(write_result(report,'reproduction-coverage'))
