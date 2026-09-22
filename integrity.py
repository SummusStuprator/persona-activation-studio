"""Hash application code and packaged resources for a local release snapshot."""
from pathlib import Path
from functools import lru_cache
import hashlib
import json
import os
import time
from studio_paths import CODE_ROOT, ASSET_ROOT, data_root
from studio_version import VERSION

ROOT = CODE_ROOT
MANIFEST = data_root() / 'trusted-files.json'
BUILD_ID = 'persona-activation-studio-' + VERSION


@lru_cache(maxsize=1024)
def fingerprint(path, size, modified):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _hash(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _files():
    import tomllib
    specification=tomllib.loads((ASSET_ROOT/'pyproject.toml').read_text(encoding='utf-8'))
    modules=specification.get('tool',{}).get('setuptools',{}).get('py-modules',[])
    code=[CODE_ROOT/(name+'.py') for name in modules if (CODE_ROOT/(name+'.py')).is_file()]
    if ASSET_ROOT==CODE_ROOT:
        code.extend(p for p in (CODE_ROOT/'setup.py',) if p.is_file())
    for package in ('workshop_v2', 'persona'):
        code.extend((CODE_ROOT / package).glob('*.py'))
    assets = []
    for pattern in ('docs/*.md', 'scripts/*', 'tests/*.py', 'native/*.cpp',
                    'native/CMakeLists.txt', 'paper/datasets/*.json',
                    'studio.toml.example', 'README.md', 'pyproject.toml'):
        assets.extend(p for p in ASSET_ROOT.glob(pattern) if p.is_file())
    return {'code/' + p.relative_to(CODE_ROOT).as_posix(): p for p in code} | {
        'assets/' + p.relative_to(ASSET_ROOT).as_posix(): p for p in assets}


def verify_release(native=False):
    if not MANIFEST.exists():
        return BUILD_ID + '-unsealed'
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    if manifest.get('format_version') != 2 or not manifest.get('files'):
        raise RuntimeError('Release manifest format changed. Review the source and run: studio seal')
    current = _files()
    expected = manifest['files']
    changed = sorted(name for name in current.keys() | expected.keys()
                     if name not in current or name not in expected or _hash(current[name]) != expected[name])
    if changed:
        raise RuntimeError('Source differs from the release snapshot: ' + ', '.join(changed) +
                           '. Review these changes, then run: studio seal')
    return manifest['build_id']


def seal_release():
    manifest = {'format_version': 2, 'build_id': BUILD_ID,
                'created_at': time.time(), 'files': {name: _hash(p) for name, p in sorted(_files().items())}}
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    temporary = MANIFEST.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    os.replace(temporary, MANIFEST)
    return manifest


if __name__ == '__main__':
    import sys
    if sys.argv[1:] == ['--seal']:
        print('SEALED', len(seal_release()['files']))
    else:
        print('VERIFIED', verify_release(native=True))
