"""Full-vocabulary, optionally full-dimensional native Jacobian lens controls."""
import json
from pathlib import Path
import pandas as pd
import streamlit as st
from . import jspace

def jacobian_ui(engine,bank):
    st.subheader('Jacobian lens: residual transport and full vocabulary')
    st.write('Read intermediate residuals through an averaged residual-to-residual Jacobian, then the actual checkpoint normalization and unembedding. The vocabulary is no longer restricted to a supplied word list.')
    st.warning('Full-dimensional does not mean a full view of the model\'s mind. This is a finite-difference implementation. The published estimator averages 1000 pretraining-like prompts; a small local calibration is an exploratory lens, not a replicated global-workspace finding.')
    layer=int(st.number_input('Source block',0,engine.layers-1,engine.layers//2,key='j_source'))
    choices=[8,32,128,'Full residual dimension']
    choice=st.selectbox('Input dimension coverage',choices,index=0)
    rank=engine.dim if isinstance(choice,str) else min(choice,engine.dim)
    contexts=st.text_area('Calibration texts (one paragraph per line)','\n'.join(jspace.DEFAULT_CORPUS),height=140)
    max_tokens=st.selectbox('Calibration token cap',[32,64,128],index=1)
    skip=int(st.number_input('Skip leading positions',0,32,16))
    step=st.selectbox('Finite-difference step / source norm',[.002,.01,.03,.1,.2],index=3)
    st.caption('Large steps estimate finite-scale transport, not an infinitesimal derivative. Low-precision numerical stability must be checked.')
    rows=[r.strip() for r in contexts.splitlines() if r.strip()]
    st.caption(f'{len(rows)} texts x {rank} input dimensions x 2 signed passes, plus step-size checks. Full mode covers all {engine.dim} input and output residual dimensions. Checkpoints resume after interruption.')
    consent=st.checkbox('Run this calibration locally; larger ranks and models can take a long time')
    if st.button('Fit / resume native Jacobian lens',disabled=not consent):
        bar=st.progress(0)
        lens=jspace.fit(engine,layer,rows,rank,max_tokens,skip,epsilon_fraction=step,progress=lambda i,n,t:bar.progress(i/n,text=f'{t}: {i}/{n}'))
        st.session_state['full_lens']=lens
    files=jspace.available(engine)
    if files:
        path=st.selectbox('Saved compatible lens',files,format_func=lambda p:p.name)
        if st.button('Load saved lens'):st.session_state['full_lens']=jspace.load(path,engine)
    lens=st.session_state.get('full_lens')
    if not lens:return
    st.info(f"Loaded numerical lens: block {lens['layer']}, {lens['rank']}/{engine.dim} input dimensions, all {engine.vocab_size:,} vocabulary tokens")
    with st.expander('Calibration provenance and finite-difference checks'):
        st.json({k:v for k,v in lens.items() if k not in ('transport','basis')})
    try:jspace.check_quality(lens)
    except ValueError as exc:st.error(str(exc));return
    st.caption(f"Maximum step-size sensitivity: {max(lens.get('fd_relative_error',[1])):.1%}. This is a numerical screen, not semantic validation.")
    text=st.text_area('Text to read','A gentle breeze moves across the water.',key='j_read_text')
    positions=st.text_input('Token positions, comma-separated; -1 is last','-1',key='j_positions')
    if st.button('Read full-vocabulary J-lens'):
        out=jspace.read(engine,text,lens,[int(p.strip()) for p in positions.split(',')])
        st.session_state['j_readout']=out
    if 'j_readout' in st.session_state:
        out=st.session_state['j_readout'];st.dataframe(pd.DataFrame(out['rows']),hide_index=True)
        st.caption(out['scope']);st.code(out['all_vocabulary_arrays'])
        st.download_button('Export ranked readout',json.dumps(out,indent=2),'j-lens-readout.json')
    if st.button('Inspect complete fitted singular spectrum'):
        s=jspace.spectrum(lens)
        st.line_chart(pd.DataFrame({'singular value':s['singular_values'],'cumulative energy':s['energy']}))
    with st.expander('Transport singular-subspace ablation (not J-space)'):
        k=int(st.number_input('Leading singular directions to remove',1,min(64,lens['rank']),min(4,lens['rank'])))
        fraction=st.slider('Projection removal fraction',0.,1.,.5,.05)
        if st.button('Test single-position subspace removal'):st.json(jspace.erase_subspace(engine,text,lens,k,fraction))
        st.caption('Tests causal token changes and reset. This is not the full paper\'s all-layer global-workspace experiment.')
    st.markdown('### Sparse vocabulary-frame J-space')
    st.caption('The paper defines J-space by sparse nonnegative combinations of vocabulary-indexed lens vectors. It is not the leading singular-vector subspace shown above.')
    from . import jframe
    if st.button('Build / load full-vocabulary sparse frame'):
        bar=st.progress(0);st.session_state['j_frame']=jframe.build(engine,lens,lambda i,n,t:bar.progress(i/n,text=f'{t}: {i}/{n}'))
    frame=st.session_state.get('j_frame')
    if frame and frame['lens_sha256']==lens['arrays_sha256']:
        k=int(st.number_input('Sparse vocabulary components',1,25,12))
        if st.button('Decompose activation into sparse J-frame coordinates'):
            out=jframe.decompose(engine,text,lens,frame,k);st.dataframe(pd.DataFrame(out['rows']),hide_index=True);st.metric('Explained residual variance fraction',f"{out['reconstruction_fraction']:.4f}");st.caption(out['scope'])
        with st.expander('Swap two vocabulary-frame coordinates'):
            source=int(st.number_input('Source vocabulary token ID',0,engine.vocab_size-1,0));target=int(st.number_input('Target vocabulary token ID',0,engine.vocab_size-1,1))
            st.write(repr(engine.piece(source).decode('utf-8','replace')),'↔',repr(engine.piece(target).decode('utf-8','replace')))
            if st.button('Test coordinate swap at the final prompt position'):st.json(jframe.swap(engine,text,lens,frame,source,target))
