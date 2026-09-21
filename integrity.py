"""Optional source integrity manifest for reproducible local releases.

A clean checkout is allowed to run unsealed. Running studio seal creates a manifest;
once it exists, launches verify the tracked source snapshot before workers start.
"""
from pathlib import Path
from functools import lru_cache
import hashlib, json
ROOT=Path(__file__).resolve().parent
MANIFEST=ROOT/'trusted-files.json'
BUILD_ID='persona-activation-studio-0.1.0'

@lru_cache(maxsize=512)
def fingerprint(path,size,modified):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def _files():
    files=list(ROOT.glob('*.py'))
    files+=list((ROOT/'workshop_v2').glob('*.py'))
    files+=list((ROOT/'persona').glob('*.py'))
    files+=list((ROOT/'native').glob('*.cpp'))
    files+=list((ROOT/'docs').glob('*.md'))
    files+=[p for p in (ROOT/'scripts').glob('*') if p.is_file()]
    files+=list((ROOT/'tests').glob('*.py'))
    files+=[ROOT/'pyproject.toml',ROOT/'studio.toml.example',ROOT/'README.md',ROOT/'THIRD_PARTY.md',ROOT/'native'/'CMakeLists.txt']
    return [p for p in files if p.exists()]

def verify_release(native=False):
    if not MANIFEST.exists():return BUILD_ID+'-unsealed'
    manifest=json.loads(MANIFEST.read_text(encoding='utf-8'))
    changed=[]
    for name,expected in manifest.get('files',{}).items():
        p=ROOT/name
        try:
            s=p.stat();actual=fingerprint(str(p),s.st_size,s.st_mtime_ns)
            if actual!=expected:changed.append(name)
        except OSError:changed.append(name)
    if changed:raise RuntimeError('Source differs from sealed manifest: '+', '.join(changed))
    return manifest.get('build_id',BUILD_ID)

def seal_release():
    manifest={'build_id':BUILD_ID,'files':{}}
    for p in _files():
        s=p.stat()
        manifest['files'][p.relative_to(ROOT).as_posix()]=fingerprint(str(p),s.st_size,s.st_mtime_ns)
    MANIFEST.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    return manifest

if __name__=='__main__':
    import sys
    if sys.argv[1:]==['--seal']:print('SEALED',len(seal_release()['files']))
    else:print('VERIFIED',verify_release(native=True))