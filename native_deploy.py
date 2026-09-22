"""Validate a complete native runtime, then swap it with a rollback backup."""
from pathlib import Path
import argparse
import hashlib
import shutil
import subprocess
import sys
import time
import uuid

from studio_paths import data_root, CODE_ROOT
ROOT=data_root()


def deploy(build, cuda=False, build_info=None):
    from workshop_v2.resource_policy import ModelLaunchLease, ModelLease
    build=Path(build).resolve()
    runtime=ROOT/'native/runtime'
    stage=ROOT/'native'/('runtime-stage-'+uuid.uuid4().hex[:10])
    stage.mkdir(parents=True)
    files=[p for p in build.rglob('*') if p.is_file() and
        (p.suffix.lower() in ('.dll','.dylib','.so') or '.so.' in p.name)]
    try:
        for source in files:
            target=stage/source.name
            if target.exists():
                if hashlib.sha256(target.read_bytes()).digest()!=hashlib.sha256(source.read_bytes()).digest():
                    raise RuntimeError('Conflicting built libraries: '+source.name)
            else:
                shutil.copy2(source,target)
        if not files:raise RuntimeError('No compiled libraries found.')
        import json
        metadata=dict(build_info or {}, built_at=time.time(), libraries={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in stage.iterdir() if p.is_file()})
        (stage/'build-info.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
        code=("import sys; from backend import Engine; e=Engine(runtime=sys.argv[1]); "
              "assert sys.argv[2]!='1' or e.cuda_available, getattr(e,'cuda_error','CUDA backend registration failed'); "
              "print('Runtime load passed; CUDA=',e.cuda_available)")
        result=subprocess.run([sys.executable,'-c',code,str(stage),'1' if cuda else '0'],
            cwd=CODE_ROOT,capture_output=True,text=True,timeout=90)
        if result.returncode:
            raise RuntimeError('New runtime failed validation; existing runtime unchanged.\n'+result.stderr)
        print(result.stdout.strip())
        with ModelLaunchLease(),ModelLease():
            backup=ROOT/'backups'/('native-runtime-'+time.strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:6])
            backup.parent.mkdir(parents=True,exist_ok=True)
            if runtime.exists():runtime.replace(backup)
            try:
                stage.replace(runtime)
            except BaseException:
                if backup.exists():backup.replace(runtime)
                raise
        print('Runtime ready:',runtime)
        print('Previous runtime retained:',backup)
    finally:
        if stage.exists():shutil.rmtree(stage)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('build')
    parser.add_argument('--cuda',action='store_true')
    args=parser.parse_args()
    deploy(args.build,args.cuda)
