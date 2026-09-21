from __future__ import annotations
import json, os, subprocess, sys, time
from pathlib import Path
import pandas as pd
import streamlit as st
from studio_config import REPO_ROOT, ensure_workspace

st.set_page_config(page_title='Persona Activation Studio',page_icon='🧠',layout='wide')
P=ensure_workspace()

def jobs_dir():
    p=P.workspace/'jobs';p.mkdir(parents=True,exist_ok=True);return p

def launch_job(kind,args):
    stamp=time.strftime('%Y%m%d-%H%M%S');folder=jobs_dir();log=folder/f'{stamp}-{kind}.log';meta=folder/f'{stamp}-{kind}.json'
    command=[sys.executable,'-m','studio_cli']+list(args)
    out=open(log,'w',encoding='utf-8',buffering=1)
    proc=subprocess.Popen(command,cwd=REPO_ROOT,stdout=out,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    meta.write_text(json.dumps({'pid':proc.pid,'kind':kind,'command':command,'log':str(log),'started':time.time()},indent=2),encoding='utf-8')
    return proc.pid,log

@st.cache_data(ttl=30)
def quick_ollama_count():
    try:
        import urllib.request
        host=os.environ.get('OLLAMA_HOST','http://127.0.0.1:11434').rstrip('/')
        with urllib.request.urlopen(host+'/api/tags',timeout=2) as r:
            return len(json.load(r).get('models',[]))
    except Exception:
        return 0

def render_jobs():
    rows=[]
    for p in sorted(jobs_dir().glob('*.json'),reverse=True)[:20]:
        try:
            d=json.loads(p.read_text(encoding='utf-8'));pid=int(d['pid'])
            import psutil;running=psutil.pid_exists(pid)
            rows.append({'kind':d['kind'],'pid':pid,'running':running,'started':time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(d['started'])),'log':d['log']})
        except Exception:pass
    if rows:st.dataframe(pd.DataFrame(rows),hide_index=True,width='stretch')
    logs=sorted(jobs_dir().glob('*.log'),key=lambda x:x.stat().st_mtime,reverse=True)
    if logs:
        pick=st.selectbox('Job log',[str(x.name) for x in logs],key='joblog')
        text=(jobs_dir()/pick).read_text(encoding='utf-8',errors='replace')
        st.code(text[-12000:],language=None)

page=st.sidebar.radio('Studio',['Overview','Collect X data','Build dataset','Train persona','Chat','Hell lab','Activations','Emotion library','Concept builder','Experiments','J-space','Paper reproduction','Jobs','Guide'])
st.sidebar.caption(str(P.workspace))

if page=='Overview':
    st.title('Persona Activation Studio')
    st.write('One local project for X collection → context-aware persona datasets → LoRA training → white-box activation analysis and steering.')
    cols=st.columns(4)
    cols[0].metric('Dataset revisions',len([x for x in P.datasets.glob('*') if x.is_dir()]))
    cols[1].metric('Installed personas',len([x for x in P.models.glob('*') if x.is_dir()]))
    cols[2].metric('Local Ollama models',quick_ollama_count())
    cols[3].metric('Built direction banks',len([x for x in (REPO_ROOT/'vectors').glob('*') if x.is_dir()]) if (REPO_ROOT/'vectors').exists() else 0)
    st.code('studio scrape setup\nstudio scrape scrape --handle @example\nstudio scrape context --handle @example\nstudio dataset build\nstudio profile add example\nstudio train example\nstudio app',language='text')
    st.subheader('Workspace paths');st.json({k:str(v) for k,v in P.__dict__.items()})

elif page=='Collect X data':
    st.title('Collect X data')
    st.write('Session credentials stay in the local twscrape database under the workspace and are ignored by Git.')
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
    name=st.text_input('Profile name')
    tier=st.selectbox('Dataset tier',['core','extended'])
    base=st.text_input('Base model','Qwen/Qwen3-4B')
    steps=st.number_input('Max optimizer steps (0 = automatic)',0,5000,0,10)
    anchors=st.text_input('Optional anchors JSON',str(P.anchors/(name+'.json')) if name else '')
    if st.button('Save profile config',disabled=not name):
        cmd=[sys.executable,'-m','studio_cli','profile','add',name,'--tier',tier]+(['--max-steps',str(int(steps))] if steps else [])
        subprocess.check_call(cmd,cwd=REPO_ROOT);st.success('Profile saved.')
    if st.button('Start training',type='primary',disabled=not name):
        args=['train',name,'--model',base]+(['--max-steps',str(int(steps))] if steps else [])
        if anchors and Path(anchors).expanduser().is_file():args+=['--anchors',anchors]
        pid,log=launch_job('train-'+name,args);st.success(f'Started PID {pid}: {log.name}')
    render_jobs()

elif page in ('Chat','Hell lab'):
    from workshop_v2.runtime import get_engine
    from runtime_ui import render as model_loader
    from science import load_bank
    engine=get_engine()
    st.title(page)
    if not model_loader(engine):
        st.write('Choose a GGUF or persona adapter in the sidebar and load it.')
    else:
        bank=load_bank(engine)
        st.caption(f"Loaded: {engine.model['name']} · {engine.layers} blocks · {len(bank)} directions")
        if page=='Hell lab':
            from workshop_v2.hell_loop import render as hell;hell(engine,bank)
        else:
            from workshop_v2.live_ui import chat;chat(engine,bank)

elif page=='Jobs':
    st.title('Background jobs');render_jobs()

elif page=='Guide':
    st.title('Guide')
    for name in ('README.md','docs/USER_GUIDE.md','docs/HELL_MODE.md'):
        p=REPO_ROOT/name
        if p.exists():
            with st.expander(name,expanded=name=='README.md'):st.markdown(p.read_text(encoding='utf-8'))