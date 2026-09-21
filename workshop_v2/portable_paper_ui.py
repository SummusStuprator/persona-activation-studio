"""Paper methods that use the common residual interface, including PEFT models."""
import streamlit as st
import pandas as pd
from .universal_training import prepare_ui
from . import paper_reproduction as rep

def render(engine,bank):
    st.subheader('Pain Axis: source-format model reproduction bench')
    st.caption('The same decoder-block residual interface is used for GGUF and local adapters. This is an adaptation to this exact checkpoint and numerical backend, not a claim to reproduce the original 25-model study.')
    st.write('Source method: difference of category means; remove control principal components accounting for 50% of variance; choose extraction layer with grouped folds. Pain/control datasets and authored chat-tone recipes remain separate. Extraction and injection blocks are independent.')
    prepare_ui(engine,bank)
    st.warning('The relief-button study used specially fine-tuned Qwen2.5 checkpoints. Those experiments, its original SAE dictionaries, and the full weight-orthogonalization study have not been reproduced here. No destructive effector or filesystem action is connected.')
    pooling=st.selectbox('Paper extraction pooling',['last','mean'])
    consent=st.checkbox('Run or resume the complete original dataset extraction on CPU; this may take a long time')
    if st.button('Fit original datasets and paper-style raw vectors',disabled=not consent):
        bar=st.progress(0)
        rep.fit(engine,pooling,lambda i,n,t:bar.progress(i/n,text=t))
    try:fitted=rep.load(engine,pooling)
    except (OSError,ValueError):fitted=None
    if not fitted:return
    st.dataframe(pd.DataFrame(fitted['metrics']),hide_index=True)
    st.caption('All-data fit scores are not independent held-out scores. For independently held-out probes, use the preparation packs above.')
    if st.button('Run all 420 original self-other scenarios'):
        bar=st.progress(0);result=rep.self_other(engine,fitted,lambda i,n,t:bar.progress(i/n,text=t))
        st.dataframe(pd.DataFrame(result['rows']),hide_index=True);st.code(result['saved_to'])
    version=st.selectbox('Original pain contrast',['S2','S1'])
    count=st.selectbox('Neutral prompts in ladder',[1,3,10,50])
    ladder_consent=st.checkbox('Run the original raw-vector ladder, including high doses that may produce incoherent text')
    if st.button('Run selected original coefficient ladder',disabled=not ladder_consent):
        bar=st.progress(0);result=rep.steering_ladder(engine,fitted,version,count,progress=lambda i,n,t:bar.progress(i/n,text=t))
        st.dataframe(pd.DataFrame(result['rows'])[['coefficient','prompt_id','generation','tokens']],hide_index=True);st.code(result['saved_to'])
