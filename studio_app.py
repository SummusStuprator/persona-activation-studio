from __future__ import annotations
import json, os, subprocess, sys, time
from pathlib import Path
import pandas as pd
import streamlit as st
from studio_config import REPO_ROOT, ensure_workspace, load_config
from studio_paths import ASSET_ROOT, data_root

st.set_page_config(page_title='Persona Activation Studio',page_icon='🧠',layout='wide')
P=ensure_workspace()

def jobs_dir():
    p=P.workspace/'jobs';p.mkdir(parents=True,exist_ok=True);return p

def launch_job(kind,args):
    from studio_jobs import launch
    try:
        return launch(REPO_ROOT,jobs_dir(),kind,args)
    except (OSError,ValueError) as exc:
        st.error(str(exc));st.stop()

@st.cache_data(ttl=30)
def quick_ollama_count():
    from model_store import grouped_inventory
    return len(grouped_inventory()[0])


@st.cache_data(ttl=120)
def quick_persona_count():
    from workshop_v2.persona_store import discover
    return len(discover()[0])

def render_jobs():
    rows=[]; active_jobs=[]
    for p in sorted(jobs_dir().glob('*.json'),reverse=True)[:20]:
        try:
            from studio_jobs import read_state
            d=read_state(p)
            if d['status'] == 'running': active_jobs.append(p)
            rows.append({'kind':d['kind'],'pid':d.get('pid'),'status':d['status'],'exit_code':d.get('return_code'),'started':time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(d['started'])),'log':d['log']})
        except Exception:pass
    if rows:st.dataframe(pd.DataFrame(rows),hide_index=True,width='stretch')
    else:st.info('No jobs yet. Use the collection, dataset, or training page to start one.')
    if active_jobs:
        selected=st.selectbox('Running job',active_jobs,format_func=lambda p:p.stem,key='cancel_job')
        if st.button('Stop selected job'):
            from studio_jobs import cancel
            try:cancel(selected);st.rerun()
            except Exception as exc:st.error(str(exc))
    logs=sorted(jobs_dir().glob('*.log'),key=lambda x:x.stat().st_mtime,reverse=True)
    if logs:
        pick=st.selectbox('Job log',[str(x.name) for x in logs],key='joblog')
        from studio_jobs import log_tail
        text=log_tail(jobs_dir()/pick)
        st.code(text[-12000:],language=None)

page=st.sidebar.radio('Studio',['Overview','Collect X data','Build dataset','Train persona','Chat','Hell lab','Activations','Emotion library','Concept builder','Experiments','J-space','Paper reproduction','Jobs','Guide'])
st.sidebar.caption(str(P.workspace))

if page=='Overview':
    st.title('Persona Activation Studio')
    st.write('One local project for X collection → context-aware persona datasets → LoRA training → white-box activation analysis and steering.')
    st.info('Start with the offline demo: install the core app, run studio demo, then explore Guide. Real persona training also needs the training extra and an NVIDIA CUDA GPU.')
    st.caption('Personas approximate writing patterns. Activation and pain-language scores do not establish subjective experience.')
    if st.button('Refresh model counts'):
        with st.spinner('Scanning model metadata...'):
            st.session_state['overview_counts'] = (quick_persona_count(), quick_ollama_count())
    counts = st.session_state.get('overview_counts', ('—', '—'))
    cols=st.columns(4)
    external=load_config().get('runtime',{}).get('default_dataset')
    revisions={str(x.resolve()) for x in P.datasets.glob('*') if x.is_dir()}
    if external and Path(external).is_dir():revisions.add(str(Path(external).resolve()))
    cols[0].metric('Dataset revisions',len(revisions))
    cols[1].metric('Persona adapters',counts[0])
    cols[2].metric('GGUF models',counts[1])
    from studio_paths import data_root
    vector_root=data_root()/'vectors'
    cols[3].metric('Direction banks',sum(x.is_dir() for x in vector_root.glob('*')))
    st.code('studio scrape setup\nstudio scrape scrape --handle @example\nstudio scrape context --handle @example\nstudio dataset build\nstudio profile add example\nstudio train example\nstudio app',language='text')
    if external:st.caption('Existing dataset used in place: '+external)
    if st.button('Run software checks'):
        pid,log=launch_job('software-check',['check']);st.success('Checks started; see Jobs for the outcome.')
    st.subheader('Workspace paths');st.json({k:str(v) for k,v in P.__dict__.items()})

elif page=='Collect X data':
    st.title('Collect X data')
    st.write('Session credentials stay in the local twscrape database under the workspace and are ignored by Git.')
    st.warning('Collection access is not training permission. X’s Developer Agreement restricts using X Content to train or fine-tune foundation models. Review Guide → docs/DATA_AND_PERMISSIONS.md before collecting or training.')
    handles=st.text_area('Handles, one per line','',height=140)
    hs=[x.strip().lstrip('@') for x in handles.splitlines() if x.strip()]
    c1,c2,c3=st.columns(3)
    if c1.button('Scrape / resume',disabled=not hs):
        args=['scrape','scrape'];[args.extend(['--handle',h]) for h in hs];pid,log=launch_job('scrape',args);st.success(f'Started PID {pid}: {log.name}')
    if c2.button('Fetch reply context',disabled=not hs):
        args=['scrape','context'];[args.extend(['--handle',h]) for h in hs];pid,log=launch_job('context',args);st.success(f'Started PID {pid}: {log.name}')
    if c3.button('Rebuild exports',disabled=not hs):
        args=['scrape','export'];[args.extend(['--handle',h]) for h in hs];pid,log=launch_job('export',args);st.success(f'Started PID {pid}: {log.name}')
    st.info('Initial session setup is interactive in a terminal: `studio scrape setup`.')
    status=P.exports/'enriched_replies_by_profile.csv'
    if status.exists():st.dataframe(pd.read_csv(status),hide_index=True,width='stretch')
    render_jobs()

elif page=='Build dataset':
    st.title('Build a frozen dataset revision')
    revision=st.text_input('Revision name',time.strftime('revision-%Y%m%d'))
    overwrite=st.checkbox('Overwrite that revision if it exists',False)
    if st.button('Build dataset',type='primary'):
        args=['dataset','build','--revision',revision]+(['--overwrite'] if overwrite else []);pid,log=launch_job('dataset',args);st.success(f'Started PID {pid}: {log.name}')
    dirs=sorted([x for x in P.datasets.glob('*') if x.is_dir()],key=lambda x:x.stat().st_mtime,reverse=True)
    if dirs:st.dataframe(pd.DataFrame([{'revision':x.name,'path':str(x),'modified':time.ctime(x.stat().st_mtime)} for x in dirs]),hide_index=True,width='stretch')
    render_jobs()

elif page=='Train persona':
    st.title('Train persona adapter')
    from studio_cli import _read_profiles
    profiles=_read_profiles().get('profiles',[])
    selected=st.selectbox('Existing profile',['New profile']+[x['profile'] for x in profiles])
    name=st.text_input('Profile name','' if selected=='New profile' else selected,key='profile-name-'+selected)
    tier=st.selectbox('Dataset tier',['core','extended'])
    resume=st.checkbox('Resume an unfinished training run',False)
    revisions=sorted([x for x in P.datasets.glob('*') if x.is_dir()],key=lambda x:x.stat().st_mtime,reverse=True)
    default_dataset=str(revisions[0]) if revisions else load_config().get('runtime',{}).get('default_dataset','')
    dataset=st.text_input('Dataset revision directory',default_dataset,disabled=resume)
    base=st.text_input('Base model','Qwen/Qwen3-4B',disabled=resume)
    steps=st.number_input('Max optimizer steps (0 = automatic)',0,5000,0,10,disabled=resume)
    anchors=st.text_input('Optional anchors JSON',str(P.anchors/(name+'.json')) if name else '',disabled=resume)
    run_dir=st.text_input('Exact unfinished run directory (optional)',disabled=not resume)
    if resume:st.info('Resume retains the saved model, dataset contents, profile, anchors and step budget. It never silently starts over.')
    if st.button('Save profile config',disabled=not name or resume):
        cmd=[sys.executable,'-m','studio_cli','profile','add',name,'--tier',tier]+(['--max-steps',str(int(steps))] if steps else [])
        result=subprocess.run(cmd,cwd=REPO_ROOT,capture_output=True,text=True)
        if result.returncode:st.error(result.stderr)
        else:st.success('Profile saved.')
    if st.button('Resume training' if resume else 'Start training',type='primary',disabled=not name):
        args=['train',name]
        if resume:args+=['--resume']+(['--run-dir',run_dir] if run_dir else [])
        else:
            args+=['--model',base]+(['--dataset',dataset] if dataset else [])+(['--max-steps',str(int(steps))] if steps else [])
            if anchors and Path(anchors).expanduser().is_file():args+=['--anchors',anchors]
        pid,log=launch_job('train-'+name,args);st.success(f'Job started or already active: PID {pid}; {log.name}')
    render_jobs()

elif page in ('Chat','Hell lab','Activations','Emotion library','Concept builder','Experiments','J-space','Paper reproduction'):
    from workshop_v2.runtime import get_engine
    from runtime_ui import render as model_loader
    from science import load_bank
    engine=get_engine()
    st.title(page)
    from workshop_v2.live_ui import manager
    if manager().active() and page != 'Chat':
        st.warning('A live generation owns the model. Return to Chat and stop it before using another workspace.')
        st.stop()
    if not model_loader(engine):
        st.write('Choose a GGUF or persona adapter in the sidebar and load it.')
    else:
        identity=(engine.model['digest'],engine.abi,engine.process.pid)
        if st.session_state.get('studio_model_identity')!=identity:
            for key in list(st.session_state):
                if key.startswith(('live_','hell_','mixdose_','mixsource_','mixlayer_','cal_','chat_sampler_')) or key in ('dialogue','latest_run','workshop_probe','watch_axes','control_axes','system_instruction','local_lens','full_lens','j_frame','j_readout'):
                    st.session_state.pop(key,None)
            st.session_state['studio_model_identity']=identity
            if engine.model.get('backend')=='persona_peft':st.session_state['live_thinking']='disabled' if 'enable_thinking' in engine.model.get('chat_template','') else 'auto'
        bank=load_bank(engine)
        st.caption(f"Loaded: {engine.model['name']} · {engine.layers} blocks · {len(bank)} directions")
        if page=='Hell lab':
            from workshop_v2.hell_loop import render as hell;hell(engine,bank)
        elif page=='Chat':
            from workshop_v2.live_ui import chat;chat(engine,bank)
        else:
            from workshop_v2.ui import explore, library_ui
            from workshop_v2.custom_ui import concepts
            from workshop_v2.universal_experiments import render as experiments
            from workshop_v2.jspace_ui import jacobian_ui
            from workshop_v2.paper_ui import paper_ui
            from workshop_v2.portable_paper_ui import render as portable_paper
            routes={'Activations':explore,'Emotion library':library_ui,
                    'Concept builder':concepts,'Experiments':experiments,
                    'J-space':jacobian_ui,
                    'Paper reproduction':portable_paper if engine.model.get('backend')=='persona_peft' else paper_ui}
            if page=='J-space' and engine.model.get('backend')=='persona_peft':
                st.info('J-space currently requires a native GGUF model. Other activation workspaces support persona adapters.')
            else:
                routes[page](engine,bank)

elif page=='Jobs':
    st.title('Background jobs');render_jobs()

elif page=='Guide':
    st.title('Guide')
    for name in ('README.md','docs/INSTALL.md','docs/QUICKSTART.md','docs/USER_GUIDE.md','docs/DATA_AND_PERMISSIONS.md','docs/HELL_MODE.md','docs/TROUBLESHOOTING.md','docs/VALIDATION.md'):
        p=ASSET_ROOT/name
        if p.exists():
            with st.expander(name,expanded=name=='README.md'):st.markdown(p.read_text(encoding='utf-8'))
