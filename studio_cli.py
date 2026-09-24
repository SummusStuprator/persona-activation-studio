from __future__ import annotations
import argparse, hashlib, json, os, shutil, subprocess, sys, time, urllib.request
from pathlib import Path
from studio_config import REPO_ROOT, ensure_workspace, paths, load_config
from studio_paths import ASSET_ROOT, data_root, config_path

def run_module(module,args):
    python=load_config()['python'] if module in ('persona.trainer_v4','persona.benchmark') else sys.executable
    cmd=[str(python),'-m',module]+[str(x) for x in args]
    print('+',' '.join(cmd),flush=True)
    return subprocess.call(cmd,cwd=REPO_ROOT)

def init_cmd(args):
    p=ensure_workspace()
    cfg=config_path(); cfg.parent.mkdir(parents=True,exist_ok=True)
    if not cfg.exists():shutil.copy2(ASSET_ROOT/'studio.toml.example',cfg)
    sources=data_root()/'persona-sources.json'
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
    from studio_doctor import command
    return command(args)


def scrape_cmd(args):
    from persona.handles import normalized_handle
    handles = [normalized_handle(h) for h in getattr(args, 'handle', []) or []]
    p=ensure_workspace();base=['--db',str(p.account_db)]
    sub=args.scrape_command
    tail=[]
    if sub in ('scrape','status','context','export'):
        tail+=['--output-dir',str(p.exports)]
    for h in handles:tail+=['--handle',h]
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
    import re
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',revision):raise ValueError('Use a plain dataset revision name, not a path. Use --output for an explicit path.')
    out=Path(args.output).expanduser().resolve() if args.output else p.datasets/revision
    cmd=['--input',str(p.project),'--output',str(out),'--exports',str(p.exports)]
    if args.overwrite:cmd+=['--overwrite']
    code=run_module('persona.dataset_builder',cmd)
    if code==0:print('Dataset revision:',out)
    return code

def _profiles_path():return ensure_workspace().workspace/'profiles.json'
def _read_profiles():
    p=_profiles_path();return json.loads(p.read_text(encoding='utf-8')) if p.exists() else {'profiles':[]}

def profile_cmd(args):
    import re
    if args.profile_command!='list' and not re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_.-]*',args.name):raise ValueError('Use a plain profile name.')
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
    from training_runs import resolve_training_run
    p=ensure_workspace()
    if not args.resume and not args.dataset and not any(p.datasets.iterdir()):
        args.dataset=load_config().get('runtime',{}).get('default_dataset')
    staging,spec=resolve_training_run(args,p)
    stamp=staging.name[len(args.profile)+1:]
    cmd=['train','--dataset-root',spec['dataset'],'--config',spec['config'],
         '--profile',args.profile,'--output',str(staging),'--model',spec['model']]
    if spec.get('anchors'):cmd+=['--anchors',spec['anchors']]
    if args.resume:cmd+=['--resume']
    if spec['max_steps']:cmd+=['--max-steps',str(spec['max_steps'])]
    print('Resuming:' if args.resume else 'Training run:',staging,flush=True)
    from workshop_v2.resource_policy import ModelLaunchLease,ModelLease,checkpoint
    with ModelLaunchLease(),ModelLease():
        checkpoint()
        code=run_module('persona.trainer_v4',cmd)
    if code:return code
    report_path=staging/'training_report.json'
    if not report_path.is_file():raise RuntimeError('Training finished without training_report.json')
    report=json.loads(report_path.read_text(encoding='utf-8'))
    if report.get('install_gate_pass') and not args.no_install:
        slug=args.profile+'-'+spec['model'].split('/')[-1].replace(':','-')+'-'+stamp
        dest=p.models/slug
        dest.mkdir(parents=True,exist_ok=False)
        shutil.copytree(staging/'adapter',dest/'adapter')
        for name in ('training_report.json','run_state.json','test_generations.jsonl','prepared_test.jsonl','prepared_validation.jsonl'):
            src=staging/name
            if src.is_file():shutil.copy2(src,dest/name)
        manifest={'profile':args.profile,'base_model':spec['model'],'source_staging':str(staging),'installed_at':time.strftime('%Y-%m-%dT%H:%M:%S'),'install_gate_pass':True}
        (dest/'model_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
        print('Installed gate-passing adapter without intermediate checkpoints:',dest)
    else:print('Candidate kept in staging:',staging)
    return 0

def plan_cmd(args):
    p=ensure_workspace()
    dataset=args.dataset or load_config().get('runtime',{}).get('default_dataset')
    if not dataset:dataset=str(_latest_dataset(p))
    config=args.config or str(_profiles_path())
    output=p.workspace/'training-plan.json'
    code=run_module('persona.trainer_v4',['plan','--dataset-root',dataset,'--config',config,'--output',output])
    if code:return code
    report=json.loads(output.read_text(encoding='utf-8'))
    rows=report.get('profiles',[])
    return 0 if rows and all(row.get('status')=='ready' for row in rows) else 1

def benchmark_cmd(args):
    p=ensure_workspace();out=Path(args.output_root).expanduser().resolve() if args.output_root else p.workspace/'benchmark';out.mkdir(parents=True,exist_ok=True)
    cmd=['--model-root',str(Path(args.model_root).expanduser().resolve() if args.model_root else p.models),'--trainer',str(REPO_ROOT/'persona'/'trainer_v4.py'),'--output-root',str(out),'--base-model',args.base_model]
    if args.quick:cmd+=['--quick']
    if args.only:cmd+=['--only']+args.only
    cmd += ['--temperature',str(args.temperature),'--max-new-tokens',str(args.max_new_tokens),'--heldout-count',str(args.heldout_count)]
    from workshop_v2.resource_policy import ModelLaunchLease, ModelLease
    with ModelLaunchLease(), ModelLease():
        return run_module('persona.benchmark',cmd)

def research_data_cmd(args):
    base='https://raw.githubusercontent.com/google-research/google-research/master/goemotions/'
    files={
      'data/emotions.txt':'45c3ef86782d2a4d7fedcd6d8c111aa0d0e94720689bd164fac94fefb4495a89',
      'data/train.tsv':'1c254a142be5c00e80d819b9ae1bbd36d94b2eeb8f4b1271846508d57e57d9c5',
      'data/dev.tsv':'575489c079c9de1097062a01738f998590d6b7ead66dd1c9fd1d2ba01fd8bc62',
      'data/test.tsv':'0587b2dd8b27b97352adbfc3fb083d46005c8946657fdc2b1ca8b1cc7f1f8be4',
    }
    root=data_root()/'datasets'/'goemotions'
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

def demo_cmd(args):
    python = load_config()['python'] if args.train else sys.executable
    command = [str(python), '-m', 'studio_demo', '--device', args.device]
    if args.train: command.append('--train')
    return subprocess.call(command, cwd=REPO_ROOT)

def app_cmd(args):
    cfg=load_config();port=args.port or int(cfg.get('runtime',{}).get('port',8899))
    cmd=[sys.executable,'-m','streamlit','run',str(REPO_ROOT/'studio_app.py'),'--server.address=127.0.0.1',f'--server.port={port}','--server.headless=true','--server.fileWatcherType=none','--browser.gatherUsageStats=false']
    return subprocess.call(cmd,cwd=REPO_ROOT)

def native_cmd(args):
    from studio_native import build
    build(args.cuda, args.jobs, args.architectures, args.cuda_root)
    return 0

def check_cmd(args):
    from tempfile import TemporaryDirectory
    checks=[['-m','unittest','discover','-s',str(ASSET_ROOT/'tests'),'-p','test_*.py','-v']]
    checks += [[str(ASSET_ROOT/'tests'/name)] for name in (
        'physical_profile_smoke.py','pair_cache_smoke.py',
        'physical_recipe_quality.py','ui_smoke.py')]
    with TemporaryDirectory(prefix='studio-check-') as directory:
        environment = dict(os.environ, STUDIO_HOME=directory,
            STUDIO_CONFIG=str(Path(directory)/'studio.toml'),
            STUDIO_LOCK_DIR=str(Path(directory)/'locks'),
            OLLAMA_MODELS=str(Path(directory)/'ollama'), HF_HOME=str(Path(directory)/'hf'),
            PYTHONPATH=str(REPO_ROOT), PYTHONUTF8='1', PYTHONIOENCODING='utf-8')
        for tail in checks:
            print('CHECK:', ' '.join(tail), flush=True)
            code=subprocess.call([sys.executable,'-X','utf8']+tail,cwd=directory,env=environment)
            if code: return code
    print('PASS: software checks',flush=True)
    return 0

def main(argv=None):
    ap=argparse.ArgumentParser(prog='studio',description='Persona Activation Studio')
    from studio_version import VERSION
    ap.add_argument('--version',action='version',version=VERSION)
    sub=ap.add_subparsers(dest='command',required=True)
    sub.add_parser('init').set_defaults(func=init_cmd)
    d=sub.add_parser('doctor');d.add_argument('--require',action='append',choices=['core','native','train','scrape']);d.set_defaults(func=doctor_cmd)
    sub.add_parser('check',help='Run isolated, model-free software checks').set_defaults(func=check_cmd)
    a=sub.add_parser('app');a.add_argument('--port',type=int);a.set_defaults(func=app_cmd)
    n=sub.add_parser('native');n.add_argument('--cuda',action='store_true');n.add_argument('--jobs',type=int,default=2);n.add_argument('--architectures',default='native');n.add_argument('--cuda-root');n.set_defaults(func=native_cmd)
    sub.add_parser('research-data').set_defaults(func=research_data_cmd)
    demo=sub.add_parser('demo');demo.add_argument('--train',action='store_true');demo.add_argument('--device',choices=['CPU','CUDA'],default='CPU');demo.set_defaults(func=demo_cmd)
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
    t=sub.add_parser('train');t.add_argument('profile');t.add_argument('--dataset');t.add_argument('--config');t.add_argument('--model',help='Base model; defaults to Qwen/Qwen3-4B for new runs, or the pinned model on resume');t.add_argument('--anchors');t.add_argument('--max-steps',type=int,default=0);t.add_argument('--resume',action='store_true');t.add_argument('--run-dir',help='Exact unfinished Studio run to resume');t.add_argument('--no-install',action='store_true');t.set_defaults(func=train_cmd)
    q=sub.add_parser('plan',help='Validate configured persona datasets without training');q.add_argument('--dataset');q.add_argument('--config');q.set_defaults(func=plan_cmd)
    b=sub.add_parser('benchmark');b.add_argument('--model-root');b.add_argument('--output-root');b.add_argument('--max-new-tokens',type=int,default=160);b.add_argument('--heldout-count',type=int,default=4);b.add_argument('--temperature',type=float,default=0.0);b.add_argument('--base-model',default='Qwen/Qwen3-4B');b.add_argument('--quick',action='store_true');b.add_argument('--only',nargs='*');b.set_defaults(func=benchmark_cmd)
    sub.add_parser('seal').set_defaults(func=lambda a:(print('SEALED',len(__import__('integrity').seal_release()['files'])) or 0))
    sub.add_parser('verify').set_defaults(func=lambda a:(print(__import__('integrity').verify_release(native=True)) or 0))
    args=ap.parse_args(argv)
    try:return int(args.func(args) or 0)
    except (ValueError,FileNotFoundError,RuntimeError,MemoryError) as exc:ap.exit(1,str(exc)+'\n')

if __name__=='__main__':raise SystemExit(main())
