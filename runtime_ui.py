"""Model discovery, honest compatibility states and memory controls."""
import json
import pandas as pd
import streamlit as st
from model_store import ROOT, grouped_inventory, plan_load

@st.cache_data(ttl=120)
def model_list(): return grouped_inventory()

def render(engine):
    from workshop_v2.live_ui import manager
    active=manager().active()
    if active:
        with st.sidebar:
            st.header('Persona adapter runtime' if engine.model.get('backend')=='persona_peft' else 'GGUF runtime')
            st.info('Live generation owns the model: '+engine.model['name'])
            st.caption('Model switching, unloading and other experiments are paused until generation finishes or is stopped. Live chat controls remain available.')
        return bool(engine.handle)
    with st.sidebar:
        st.caption('Cooperative mode: one workshop model at a time, adaptive Auto GPU offload from currently free VRAM, and a 10% host-RAM reserve. Persona adapters remain CPU-only.')
        source=st.selectbox('Model source',['Choose a source','Ollama GGUF','Persona adapters'],key='model_source')
        if source=='Choose a source':
            st.info('Choose the existing GGUF store or the persona adapter folder to begin. No model is loaded automatically.')
            return bool(engine.handle)
        if source=='Persona adapters':
            from workshop_v2.persona_runtime_ui import render as persona_render
            return persona_render(engine)
        st.header('GGUF runtime')
        if st.button('Refresh model inventory'):
            model_list.clear(); st.rerun()
        with st.spinner('Inspecting local GGUF metadata; the first scan may take longer while the computer is busy...'):
            models, rows = model_list()
        st.caption(f'{len(models)} unique GGUF checkpoints · {len(rows)} Ollama names · zero weight copies')
        if not models:
            st.error('No local decoder GGUF models found in the configured Ollama model stores.')
            return False
        names = [m['name'] for m in models]
        selected = st.selectbox('Local model', names, key='model_choice')
        model = models[names.index(selected)]
        report_path = ROOT/'compatibility-report.json'
        if report_path.exists():
            try:
                records = json.loads(report_path.read_text(encoding='utf-8')).get('models', [])
                record = next((r for r in records if r['digest'] == model['digest']), None)
                if record and record['status'] == 'passed':
                    st.caption('Prior compatibility scan: generation + activation injection + reset passed. See the emotion audit for this build for current tests.')
                elif record:
                    st.warning('Last compatibility test failed; native details are available below. A runtime update may change this result.')
                    with st.expander('Last compatibility test details'):
                        st.write(record.get('native_errors') or record.get('error'))
                else: st.caption('Not yet tested on this installation.')
            except (ValueError, OSError):
                st.caption('Compatibility report is being refreshed.')
        mode = st.selectbox('Memory mode', ['Auto', 'CPU', 'Manual'], key='memory_mode')
        context = st.selectbox('Context tokens', [512, 1024, 2048, 4096, 8192], index=2)
        manual = 0
        if mode == 'Manual':
            manual = st.slider('GPU layers (0 = CPU)', 0, model.get('logical_layers',model['layers'])+1, min(model.get('logical_layers',model['layers']), 12))
        from workshop_v2.resource_policy import settings as resource_settings
        cfg=resource_settings()
        effective_mode=mode  # Explicit manual offload still obeys RAM admission and the shared lease.
        plan = plan_load(model, context, effective_mode, manual)
        st.caption(f"{model['family']} · {model['layers']} blocks · {model['size_gb']:.2f} GB on disk")
        st.caption(f"Proposed offload: {plan['gpu_layers']} layers · {plan['hardware']['gpu_free_gib']:.1f} GiB GPU free · {plan['hardware']['ram_available_gib']:.1f} GiB RAM free")
        for message in plan['warnings']: st.warning(message)
        if st.button('Load model', type='primary'):
            try:
                with st.spinner('Loading the original GGUF in an isolated worker and testing hooks...'):
                    engine.open(model, context=context, mode=effective_mode, manual=(0 if effective_mode=='CPU' else manual))
                for key in ('comparison','probe','scenarios','controls','block','direction','baseline','dose_add','dose_erase'):
                    st.session_state.pop(key, None)
                st.success('Model and layer activations ready.')
            except Exception as exc: st.error(str(exc))
        if st.button('Unload model / free memory'):
            engine.close(); st.rerun()
        with st.expander('Same weights, other Ollama names'):
            st.write(model['aliases']); st.code(model['path'], language=None)
            st.caption('These aliases share the same base GGUF. Their Ollama system prompts and tools are not applied here.')
        with st.expander('All models and exclusions'):
            st.dataframe(pd.DataFrame(rows)[['name','family','size_gb','reason']], hide_index=True)
        with st.expander('Memory plan and runtime details'):
            st.json(engine.load_plan if engine.handle else plan)
            st.caption('Large models can use CPU/RAM plus partial GPU offload. Auto retries lower offload after load failures. No model weights are copied.')
        st.caption('Text decoder only. Image/audio encoders, cloud models and separate adapters are not silently substituted.')
        st.code(str(ROOT), language=None)
    return bool(engine.handle)
