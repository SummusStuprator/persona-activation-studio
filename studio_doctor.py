"""Readiness checks distinguish optional pipelines from the core application."""
from pathlib import Path
from importlib.util import find_spec
import json
import os
import subprocess
import sys
from studio_config import REPO_ROOT, load_config, paths


def report():
    cfg=load_config();p=paths()
    core={name:bool(find_spec(name)) for name in ('streamlit','numpy','sklearn','psutil','jinja2')}
    data={'core':all(core.values()),'core_modules':core,'scrape':bool(find_spec('twscrape')),
          'scrape_session_database_exists':p.account_db.is_file(),
          'training_python':str(cfg['python']),'train':False,
          'default_dataset_exists':Path(cfg.get('runtime',{}).get('default_dataset','__none__')).is_dir()}
    probe="import json,torch,transformers,peft,bitsandbytes; print(json.dumps({'cuda':torch.cuda.is_available(),'torch':torch.__version__,'transformers':transformers.__version__,'peft':peft.__version__}))"
    try:
        result=subprocess.run([str(cfg['python']),'-X','utf8','-c',probe],cwd=REPO_ROOT,
            capture_output=True,text=True,encoding='utf-8',timeout=45)
        if result.returncode==0:
            training=json.loads(result.stdout.strip().splitlines()[-1])
            data['training']=training;data['train']=bool(training['cuda'])
        else:data['training_error']=result.stderr[-1200:]
    except (OSError,ValueError,subprocess.TimeoutExpired) as exc:
        data['training_error']=str(exc)
    native_probe="import json; from backend import Engine; e=Engine(); print(json.dumps({'abi':e.abi,'cuda':e.cuda_available,'cuda_error':getattr(e,'cuda_error',None)}))"
    data['native']=False
    try:
        result=subprocess.run([sys.executable,'-X','utf8','-c',native_probe],cwd=REPO_ROOT,
            capture_output=True,text=True,encoding='utf-8',timeout=45)
        if result.returncode==0:
            data['native_runtime']=json.loads(result.stdout.strip().splitlines()[-1])
            data['native']=True
        else:data['native_error']=result.stderr[-1200:]
    except (OSError,ValueError,subprocess.TimeoutExpired) as exc:
        data['native_error']=str(exc)
    data['note']='Native checks load libraries, not model weights. Session database presence does not verify X authentication.'
    return data


def command(args):
    data=report()
    print(json.dumps(data,indent=2))
    required=getattr(args,'require',None) or ['core']
    failed=[name for name in required if not data.get(name)]
    if failed:
        print('Not ready: '+', '.join(failed),file=sys.stderr)
        return 1
    return 0
