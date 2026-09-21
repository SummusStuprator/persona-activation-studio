from __future__ import annotations
import argparse, hashlib, json, os, shutil, subprocess, sys, time, urllib.request
from pathlib import Path
from studio_config import REPO_ROOT, ensure_workspace, paths, load_config

def run_module(module,args):
    cmd=[sys.executable,'-m',module]+[str(x) for x in args]
    print('+',' '.join(cmd),flush=True)
    return subprocess.call(cmd,cwd=REPO_ROOT)

def init_cmd(args):
    p=ensure_workspace()
    cfg=REPO_ROOT/'studio.toml'
    if not cfg.exists():shutil.copy2(REPO_ROOT/'studio.toml.example',cfg)
    sources=REPO_ROOT/'persona-sources.json'
    if not sources.exists():
        sources.write_text(json.dumps({
            'project_roots':[str(p.project)],'model_roots':[str(p.models)],
            'cache_roots':[str(p.hf_cache/'hub')],'python':sys.executable,
            'base_overrides':{},'settle_seconds':5,'max_depth':5
        },indent=2),encoding='utf-8')
    profiles=p.workspace/'profiles.json'
    if not profiles.exists():profiles.write_text(json.dumps({'profiles':[]},indent=2),encoding='utf-8')
    print('Workspace:',p.workspace)
    print('Persona project:',p.project)
    print('Models:',p.models)
    return 0

def doctor_cmd(args):
    p=ensure_workspace();checks={}
    checks['python']=sys.version.split()[0]
    checks['git']=shutil.which('git')
    checks['cmake']=shutil.which('cmake')
    runtime=REPO_ROOT/'native'/'runtime'
    checks['native_runtime']=str(runtime) if runtime.is_dir() and any(runtime.iterdir()) else None
    checks['ollama_store']=str(Path(os.environ.get('OLLAMA_MODELS',Path.home()/'.ollama/models')).expanduser())
    try:
        with urllib.request.urlopen(os.environ.get('OLLAMA_HOST','http://127.0.0.1:11434').rstrip('/')+'/api/tags',timeout=2) as r:checks['ollama_api']=r.status==200
    except Exception:checks['ollama_api']=False
    for mod in ('streamlit','numpy','sklearn','twscrape','torch','transformers','peft'):
        try:__import__(mod);checks[mod]=True
        except Exception:checks[mod]=False
    print(json.dumps(checks,indent=2));return 0

def scrape_cmd(args):
    p=ensure_workspace();base=['--db',str(p.account_db)]
    sub=args.scrape_command
    tail=[]
    if sub in ('scrape','status','context','export'):
        tail+=['--output-dir',str(p.exports)]
    for h in getattr(args,'handle',[]) or []:tail+=['--handle',h]
    if sub=='setup':
        if args.alias:tail+=['--alias',args.alias]
        if args.replace:tail+=['--replace']
        if args.from_clipboard:tail+=['--from-clipboard']
    if sub=='scrape':
        if args.recent_only:tail+=['--recent-only']
        if args.start:tail+=['--start',args.start]
        if args.end:tail+=['--end',args.end]
    if sub=='context' and args.export_only:tail+=['--export-only']
    return run_module('persona.x_scraper',base+[sub]+tail)

def dataset_cmd(args):
    p=ensure_workspace()
    revision=args.revision or time.strftime('revision-%Y%m%d-%H%M%S')
    out=Path(args.output).expanduser().resolve() if args.output else p.datasets/revision
    cmd=['--input',str(p.project),'--output',str(out)]
    if args.overwrite:cmd+=['--overwrite']
    code=run_module('persona.dataset_builder',cmd)
    if code==0:print('Dataset revision:',out)
    return code

def _profiles_path():return ensure_workspace().workspace/'profiles.json'
def _read_profiles():
    p=_profiles_path();return json.loads(p.read_text(encoding='utf-8')) if p.exists() else {'profiles':[]}

def profile_cmd(args):
    doc=_read_profiles()
    if args.profile_command=='list':
        print(json.dumps(doc,indent=2));return 0
    item={'profile':args.name,'dataset_profile':args.dataset_profile or args.name,'dataset_tier':args.tier}
    if args.max_steps:item['max_steps']=args.max_steps
    if args.max_rows:item['max_authentic_rows']=args.max_rows
    if args.max_length:item['max_length']=args.max_length
    items=[x for x in doc.get('profiles',[]) if str(x.get('profile','')).casefold()!=args.name.casefold()]
    items.append(item);doc['profiles']=items
    _profiles_path().write_text(json.dumps(doc,indent=2),encoding='utf-8')
    print('Saved profile config:',item);return 0

def _latest_dataset(p):
    dirs=[d for d in p.datasets.iterdir() if d.is_dir()] if p.datasets.exists() else []
    if not dirs:raise FileNotFoundError('No dataset revisions found. Run: studio dataset build')
    return max(dirs,key=lambda d:d.stat().st_mtime)

def train_cmd(args):
    p=ensure_workspace();dataset=Path(args.dataset).expanduser().resolve() if args.dataset else _latest_dataset(p)
    config=Path(args.config).expanduser().resolve() if args.config else _profiles_path()
    stamp=time.strftime('%Y%m%d-%H%M%S')
    staging=p.staging/(args.profile+'-'+stamp)
    cmd=['train','--dataset-root',str(dataset),'--config',str(config),'--profile',args.profile,'--output',str(staging),'--model',args.model]
    anchor=Path(args.anchors).expanduser().resolve() if args.anchors else p.anchors/(args.profile+'.json')
    if anchor.is_file():cmd+=['--anchors',str(anchor)]
    if args.resume:cmd+=['--resume']
    if args.max_steps:cmd+=['--max-steps',str(args.max_steps)]
    code=run_module('persona.trainer_v4',cmd)
    if code:return code
    report_path=staging/'training_report.json'
    if not report_path.is_file():raise RuntimeError('Training finished without training_report.json')
    report=json.loads(report_path.read_text(encoding='utf-8'))
    if report.get('install_gate_pass') and not args.no_install:
        slug=args.profile+'-'+args.model.split('/')[-1].replace(':','-')+'-'+stamp
        dest=p.models/slug
        dest.mkdir(parents=True,exist_ok=False)
        shutil.copytree(staging/'adapter',dest/'adapter')
        for name in ('training_report.json','run_state.json','test_generations.jsonl','prepared_test.jsonl','prepared_validation.jsonl'):
            src=staging/name
            if src.is_file():shutil.copy2(src,dest/name)
        manifest={'profile':args.profile,'base_model':args.model,'source_staging':str(staging),'installed_at':time.strftime('%Y-%m-%dT%H:%M:%S'),'install_gate_pass':True}
        (dest/'model_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
        print('Installed gate-passing adapter without intermediate checkpoints:',dest)
    else:print('Candidate kept in staging:',staging)
    return 0

def benchmark_cmd(args):
    p=ensure_workspace();out=p.workspace/'benchmark';out.mkdir(parents=True,exist_ok=True)
    cmd=['--model-root',str(p.models),'--trainer',str(REPO_ROOT/'persona'/'trainer_v4.py'),'--output-root',str(out),'--base-model',args.base_model]
    if args.quick:cmd+=['--quick']
    if args.only:cmd+=['--only']+args.only
    return run_module('persona.benchmark',cmd)

def research_data_cmd(args):
    base='https://raw.githubusercontent.com/google-research/google-research/master/goemotions/'
    files={
      'data/emotions.txt':'45c3ef86782d2a4d7fedcd6d8c111aa0d0e94720689bd164fac94fefb4495a89',
      'data/train.tsv':'1c254a142be5c00e80d819b9ae1bbd36d94b2eeb8f4b1271846508d57e57d9c5',
      'data/dev.tsv':'575489c079c9de1097062a01738f998590d6b7ead66dd1c9fd1d2ba01fd8bc62',
      'data/test.tsv':'0587b2dd8b27b97352adbfc3fb083d46005c8946657fdc2b1ca8b1cc7f1f8be4',
    }
    root=REPO_ROOT/'datasets'/'goemotions'
    for name,expected in files.items():
        target=root/name;target.parent.mkdir(parents=True,exist_ok=True)
        if target.is_file() and hashlib.sha256(target.read_bytes()).hexdigest()==expected:
            print('verified',target);continue
        print('download',base+name,flush=True)
        request=urllib.request.Request(base+name,headers={'User-Agent':'PersonaActivationStudio/0.1'})
        with urllib.request.urlopen(request,timeout=60) as response:data=response.read()
        actual=hashlib.sha256(data).hexdigest()
        if actual!=expected:raise RuntimeError(f'GoEmotions checksum mismatch for {name}: {actual}')
        temporary=target.with_suffix(target.suffix+'.tmp');temporary.write_bytes(data);temporary.replace(target)
        print('saved',target,len(data))
    print('GoEmotions ready:',root)
    return 0

def app_cmd(args):
    cfg=load_config();port=args.port or int(cfg.get('runtime',{}).get('port',8899))
    cmd=[sys.executable,'-m','streamlit','run',str(REPO_ROOT/'studio_app.py'),'--server.address=127.0.0.1',f'--server.port={port}','--server.headless=true','--server.fileWatcherType=none','--browser.gatherUsageStats=false']
    return subprocess.call(cmd,cwd=REPO_ROOT)

def native_cmd(args):
    script=REPO_ROOT/'scripts'/('build-native.ps1' if os.name=='nt' else 'build-native.sh')
    if os.name=='nt':return subprocess.call(['powershell.exe','-NoProfile','-ExecutionPolicy','Bypass','-File',str(script)]+(['-Cuda'] if args.cuda else []),cwd=REPO_ROOT)
    env=dict(os.environ,CUDA='ON' if args.cuda else 'OFF');return subprocess.call(['bash',str(script)],cwd=REPO_ROOT,env=env)

def main(argv=None):
    ap=argparse.ArgumentParser(prog='studio',description='Persona Activation Studio')
    sub=ap.add_subparsers(dest='command',required=True)
    sub.add_parser('init').set_defaults(func=init_cmd)
    sub.add_parser('doctor').set_defaults(func=doctor_cmd)
    a=sub.add_parser('app');a.add_argument('--port',type=int);a.set_defaults(func=app_cmd)
    n=sub.add_parser('native');n.add_argument('--cuda',action='store_true');n.set_defaults(func=native_cmd)
    sub.add_parser('research-data').set_defaults(func=research_data_cmd)
    s=sub.add_parser('scrape');ss=s.add_subparsers(dest='scrape_command',required=True)
    q=ss.add_parser('setup');q.add_argument('--alias',default='studio_x');q.add_argument('--replace',action='store_true');q.add_argument('--from-clipboard',action='store_true')
    ss.add_parser('accounts')
    for name in ('scrape','status','context','export'):
        q=ss.add_parser(name);q.add_argument('--handle',action='append',required=True)
        if name=='scrape':q.add_argument('--recent-only',action='store_true');q.add_argument('--start');q.add_argument('--end')
        if name=='context':q.add_argument('--export-only',action='store_true')
    s.set_defaults(func=scrape_cmd)
    d=sub.add_parser('dataset');ds=d.add_subparsers(dest='dataset_command',required=True);q=ds.add_parser('build');q.add_argument('--revision');q.add_argument('--output');q.add_argument('--overwrite',action='store_true');q.set_defaults(func=dataset_cmd)
    pr=sub.add_parser('profile');ps=pr.add_subparsers(dest='profile_command',required=True);ps.add_parser('list');q=ps.add_parser('add');q.add_argument('name');q.add_argument('--dataset-profile');q.add_argument('--tier',default='core',choices=['core','extended']);q.add_argument('--max-steps',type=int,default=0);q.add_argument('--max-rows',type=int,default=0);q.add_argument('--max-length',type=int,default=0);pr.set_defaults(func=profile_cmd)
    t=sub.add_parser('train');t.add_argument('profile');t.add_argument('--dataset');t.add_argument('--config');t.add_argument('--model',default='Qwen/Qwen3-4B');t.add_argument('--anchors');t.add_argument('--max-steps',type=int,default=0);t.add_argument('--resume',action='store_true');t.add_argument('--no-install',action='store_true');t.set_defaults(func=train_cmd)
    b=sub.add_parser('benchmark');b.add_argument('--base-model',default='Qwen/Qwen3-4B');b.add_argument('--quick',action='store_true');b.add_argument('--only',nargs='*');b.set_defaults(func=benchmark_cmd)
    sub.add_parser('seal').set_defaults(func=lambda a:(print('SEALED',len(__import__('integrity').seal_release()['files'])) or 0))
    sub.add_parser('verify').set_defaults(func=lambda a:(print(__import__('integrity').verify_release(native=True)) or 0))
    args=ap.parse_args(argv);return int(args.func(args) or 0)

if __name__=='__main__':raise SystemExit(main())
