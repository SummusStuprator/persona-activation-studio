"""Same live Chat workspace, explicit source-format persona backend."""
import json
import pandas as pd
import streamlit as st
from .persona_store import discover,settings

@st.cache_data(ttl=30)
def inventory():return discover()

def render(engine):
    st.header('Persona adapter runtime')
    if st.button('Refresh persona inventory'):
        inventory.clear();st.rerun()
    config=settings()
    all_models,excluded=inventory()
    unverified=st.checkbox('Include installed adapters without a recorded gate pass',value=False)
    models=[m for m in all_models if m['gate_status']=='recorded pass' or unverified]
    st.caption(f'{len(models)} selectable adapters / {len(all_models)} complete candidates. Future compatible installed outputs are found on refresh; no filename suffix is required.')
    with st.expander('Discovery paths and excluded models'):
        st.json({k:config[k] for k in ('project_roots','model_roots','cache_roots')})
        st.dataframe(pd.DataFrame(excluded),hide_index=True)
        st.caption('Training scripts are not executed. Staging, failed gates, partial files, unknown architectures and missing local bases are excluded.')
    if not models:
        st.info('No compatible completed adapter found. Finish its export/installation, wait 60 seconds, and refresh. Existing GGUF chat is available under the other Model source.')
        return bool(engine.handle)
    labels={m['selection_id']:m for m in models}
    if st.session_state.get('persona_choice') not in labels:st.session_state.pop('persona_choice',None)
    chosen=st.selectbox('Persona model',list(labels),format_func=lambda x:labels[x]['name'],key='persona_choice')
    model=labels[chosen]
    if model['architecture']!='qwen3':st.warning('Architecture loader is experimental here; this release was tested on Qwen3 persona adapters.')
    st.caption(f"{model['base_reference']} / {model['layers']} blocks x {model['dim']} dimensions / {model['adapter_bytes']/1e6:.1f} MB adapter")
    st.caption('Source-format safetensors + LoRA, not GGUF. Uses the existing shared cached base. No download, conversion, merging or weight update.')
    mode=st.selectbox('Persona device',['Auto','CUDA','CPU'],key='persona_device')
    precision=st.selectbox('Weight precision',['auto','bf16','nf4'],format_func=lambda p:{'auto':'Auto','bf16':'BF16','nf4':'4-bit NF4'}[p],key='persona_precision',disabled=mode=='CPU')
    if mode=='CPU':precision='bf16'
    contexts=[n for n in (512,1024,2048,4096,8192) if n<=model['max_context']]
    if st.session_state.get('persona_context') not in contexts:st.session_state.pop('persona_context',None)
    context=st.selectbox('Persona context tokens',contexts,index=min(2,len(contexts)-1),key='persona_context')
    st.caption('Auto selects CUDA BF16, then CUDA NF4, then CPU according to available memory. CUDA requires a GPU-capable persona Python; NF4 also requires bitsandbytes.')
    if st.button('Load persona model',type='primary'):
        try:
            with st.spinner('Checking source hashes, loading the existing base + adapter and verifying decoder hooks...'):
                engine.open(model,context=context,mode=mode,precision=precision)
            st.success('Persona adapter and activation hooks ready.')
        except Exception as exc:st.error(str(exc))
    if st.button('Unload model / free memory'):
        engine.close();st.rerun()
    with st.expander('Selected source and loaded identity'):
        st.write('Adapter:',model['adapter_path']);st.write('Base:',model['base_path'])
        st.write('Gate:',model['gate_status'])
        if engine.handle:st.json(engine.load_plan)
    if engine.handle and engine.model.get('backend')!='persona_peft':
        st.info('A GGUF is still loaded. The dropdown is a selection, not an automatic model switch; click Load persona model.')
    return bool(engine.handle)
