"""Validate release contents and write commit-bound checksums. Run from a checkout."""
from pathlib import Path, PurePosixPath
import argparse
import ast
import hashlib
import json
import re
import subprocess
import tarfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
BLOCKED_DIRS = {'backups', 'workspace', 'vectors', 'calibration', 'runs', 'sessions',
                'logs', 'uploads', '.venv', '.vendor', 'paper_reproduction'}
BLOCKED_SUFFIXES = {'.gguf', '.safetensors', '.bin', '.pt', '.pth', '.npz', '.db', '.log'}
PRIVATE_NAMES = {'studio.toml', 'persona-sources.json', 'resource-policy.json',
                 'trusted-files.json', '.env'}
SECRET = re.compile(rb'(?:gh[pousr]_|github_pat_)[A-Za-z0-9_]{30,}|'
                    rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----')
USER_PATH = re.compile(rb'(?:[A-Za-z]:[\\/]+Users[\\/]+|/home/)[A-Za-z0-9_.-]+')


def validate_entry(name, data):
    parts = PurePosixPath(name).parts
    if name.startswith('/') or '..' in parts or '\\' in name:
        raise ValueError('Invalid archive path: ' + name)
    if set(parts) & BLOCKED_DIRS and name != 'workspace/.gitkeep':
        raise ValueError('Generated directory in release: ' + name)
    if PurePosixPath(name).suffix.lower() in BLOCKED_SUFFIXES or parts[-1] in PRIVATE_NAMES:
        raise ValueError('Private/runtime file in release: ' + name)
    if SECRET.search(data) or USER_PATH.search(data):
        raise ValueError('Credential or workstation-path pattern in: ' + name)


def source_matches(packaged, source):
    if b'\0' not in packaged and b'\0' not in source:
        return packaged.replace(b'\r\n', b'\n') == source.replace(b'\r\n', b'\n')
    return packaged == source


def version():
    tree = ast.parse((ROOT / 'studio_version.py').read_text(encoding='utf-8'))
    return next(ast.literal_eval(n.value) for n in tree.body
                if isinstance(n, ast.Assign) and any(getattr(t, 'id', '') == 'VERSION' for t in n.targets))


def git(*args):
    return subprocess.check_output(['git', '-C', str(ROOT), *args], text=True).strip()


def archive_entries(path):
    if path.suffix == '.whl':
        with zipfile.ZipFile(path) as archive:
            return {item.filename: archive.read(item) for item in archive.infolist() if not item.is_dir()}
    with tarfile.open(path, 'r:gz') as archive:
        result = {}
        for item in archive.getmembers():
            if item.isdir():
                continue
            if not item.isfile():
                raise ValueError('Nonregular archive entry: ' + item.name)
            result[item.name.split('/', 1)[1]] = archive.extractfile(item).read()
        return result


def verify(folder, write=False):
    current = version()
    citation = (ROOT / 'CITATION.cff').read_text(encoding='utf-8')
    if not re.search(r'^version: [\"\']?' + re.escape(current) + r'[\"\']?$', citation, re.M):
        raise ValueError('CITATION.cff version does not match studio_version.py')
    if ('## ' + current) not in (ROOT / 'CHANGELOG.md').read_text(encoding='utf-8'):
        raise ValueError('Current version is missing from CHANGELOG.md')
    names = git('ls-files', '-z').split('\0')
    for name in filter(None, names):
        validate_entry(name, (ROOT / name).read_bytes())
    artifacts = [folder / f'persona_activation_studio-{current}-py3-none-any.whl',
                 folder / f'persona_activation_studio-{current}.tar.gz']
    required = ('studio.toml.example', 'LICENSE', 'THIRD_PARTY.md', 'docs/INSTALL.md',
                'docs/PUBLISHING.md', 'paper/LICENSE', 'paper/datasets/4.2_neutral_50.json',
                'native/bridge.cpp', 'native/CMakeLists.txt', 'tests/ui_smoke.py')
    hashes = {}
    for path in artifacts:
        entries = archive_entries(path)
        for name, data in entries.items():
            validate_entry(name, data)
        prefix = 'studio_assets/' if path.suffix == '.whl' else ''
        for name in required:
            if prefix + name not in entries:
                raise ValueError(f'{path.name}: missing {name}')
        for name in names:
            target = name if name in entries else prefix + name
            if target in entries and (ROOT / name).is_file():
                if not source_matches(entries[target], (ROOT / name).read_bytes()):
                    raise ValueError(f'{path.name}: stale packaged file {name}')
        if not source_matches(entries.get('studio_version.py', b''), (ROOT / 'studio_version.py').read_bytes()):
            raise ValueError(path.name + ': incorrect packaged version')
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    report = {'version': current, 'commit': git('rev-parse', 'HEAD'),
              'tree': git('rev-parse', 'HEAD^{tree}'), 'artifacts': hashes}
    checksum_text = ''.join(f'{digest}  {name}\n' for name, digest in hashes.items())
    if write:
        if git('status', '--porcelain'):
            raise ValueError('Commit source changes before writing release metadata.')
        (folder / 'SHA256SUMS.txt').write_text(checksum_text, encoding='utf-8', newline='\n')
        (folder / 'release.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8', newline='\n')
    else:
        saved = json.loads((folder / 'release.json').read_text(encoding='utf-8'))
        if saved != report or (folder / 'SHA256SUMS.txt').read_text(encoding='utf-8') != checksum_text:
            raise ValueError('Release checksums or commit metadata do not match.')
    print(json.dumps(report, indent=2))
    print('PASS: source, packaged resources, version, and archive checksums')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dist', type=Path, default=ROOT / 'dist')
    parser.add_argument('--write', action='store_true', help='Write checksums after a clean build')
    args = parser.parse_args()
    verify(args.dist.resolve(), args.write)
