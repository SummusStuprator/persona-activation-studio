"""Validate release contents and write commit-bound checksums. Run from a checkout."""
from pathlib import Path, PurePosixPath
import argparse
import ast
import hashlib
import io
import json
import re
import subprocess
import tarfile
import tomllib
import zipfile

ROOT = Path(__file__).resolve().parents[1]
BLOCKED_DIRS = {'backups', 'workspace', 'vectors', 'calibration', 'runs', 'sessions',
                'logs', 'uploads', '.venv', '.vendor', 'paper_reproduction',
                'cache', 'preparation', 'validation', 'reports', 'x_exports', 'research'}
BLOCKED_SUFFIXES = {'.gguf', '.safetensors', '.bin', '.pt', '.pth', '.npz', '.db', '.log',
                    '.sqlite', '.sqlite3', '.key', '.token', '.pem', '.p12', '.pfx'}
PRIVATE_NAMES = {'studio.toml', 'persona-sources.json', 'resource-policy.json',
                 'trusted-files.json', '.env'}
SECRET = re.compile(rb'(?:gh[pousr]_|github_pat_)[A-Za-z0-9_]{30,}|'
                    rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|'
                    rb'(?:auth_token|ct0)\s*=\s*[A-Fa-f0-9]{32,}')
USER_PATH = re.compile(rb'(?:[A-Za-z]:[\\/]+Users[\\/]+|/home/|/Users/)[A-Za-z0-9_.-]+')


def validate_entry(name, data):
    parts = PurePosixPath(name).parts
    if (not parts or name != PurePosixPath(name).as_posix() or name.startswith('/')
            or '..' in parts or '\\' in name or ':' in name):
        raise ValueError('Invalid archive path: ' + name)
    if set(parts) & BLOCKED_DIRS and name != 'workspace/.gitkeep':
        raise ValueError('Generated directory in release: ' + name)
    if (PurePosixPath(name).suffix.lower() in BLOCKED_SUFFIXES or parts[-1] in PRIVATE_NAMES
            or parts[-1].startswith('.env.') or re.search(r'\.(?:db|sqlite3?)-(?:wal|shm|journal)$', name)):
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
    def add(result, name, data):
        validate_entry(name, data)
        if name in result:
            raise ValueError('Duplicate archive entry: ' + name)
        result[name] = data

    if path.suffix == '.whl':
        with zipfile.ZipFile(path) as archive:
            result = {}
            for item in archive.infolist():
                if item.is_dir():
                    continue
                mode = (item.external_attr >> 16) & 0o170000
                if mode not in (0, 0o100000):
                    raise ValueError('Nonregular archive entry: ' + item.filename)
                add(result, item.filename, archive.read(item))
            return result
    with tarfile.open(path, 'r:gz') as archive:
        result = {}
        for item in archive.getmembers():
            if item.isdir():
                continue
            if not item.isfile():
                raise ValueError('Nonregular archive entry: ' + item.name)
            # Check the full name before dropping the sdist's top directory.
            validate_entry(item.name, b'')
            top, separator, name = item.name.partition('/')
            if not separator or top != path.name.removesuffix('.tar.gz'):
                raise ValueError('Unexpected source archive root: ' + item.name)
            add(result, name, archive.extractfile(item).read())
        return result


def wheel_target(name, specification):
    """Map each shipped source file to its wheel location; metadata is generated."""
    config = specification['tool']['setuptools']
    if name in {module + '.py' for module in config['py-modules']}:
        return name
    parts = PurePosixPath(name).parts
    if parts[0] in config['packages'] and PurePosixPath(name).suffix in {'.py', '.md'}:
        return name
    assets = {'README.md', 'LICENSE', 'THIRD_PARTY.md', 'studio.toml.example',
              'pyproject.toml', 'CONTRIBUTING.md', 'CHANGELOG.md', 'SECURITY.md', 'CITATION.cff'}
    if name in assets or parts[0] in {'docs', 'scripts', 'native', 'paper', 'tests', 'examples', 'constraints'}:
        return 'studio_assets/' + name
    return None


def verify_source_payload(entries, names, wheel=False):
    specification = tomllib.loads((ROOT / 'pyproject.toml').read_text(encoding='utf-8'))
    for name in names:
        if name == 'workspace/.gitkeep':
            continue
        target = wheel_target(name, specification) if wheel else name
        if target is None:
            continue
        if target not in entries:
            raise ValueError('Missing packaged source/resource: ' + name)
        if not source_matches(entries[target], (ROOT / name).read_bytes()):
            raise ValueError('Stale packaged file: ' + name)


def verify_history():
    """Scan all locally reachable Git blobs; report paths, never matching content."""
    # rev-list --objects names each blob only once; inspect every tree so a reused
    # blob with a private historical filename is not hidden by another pathname.
    objects = {}
    failures = []
    seen_paths = set()
    for commit in git('rev-list', '--all').splitlines():
        tree = subprocess.check_output(['git', '-C', str(ROOT), 'ls-tree', '-r', '-z', commit])
        for entry in filter(None, tree.split(b'\0')):
            metadata, raw_name = entry.split(b'\t', 1)
            mode, kind, raw_oid = metadata.decode('ascii').split()
            name, oid = raw_name.decode('utf-8'), raw_oid
            if name not in seen_paths:
                seen_paths.add(name)
                try:
                    validate_entry(name, b'')
                except ValueError as exc:
                    failures.append(str(exc))
            if kind != 'blob' or mode not in ('100644', '100755'):
                failures.append('Nonregular Git entry: ' + name)
            else:
                objects.setdefault(oid, name)
    if objects:
        payload = ('\n'.join(objects) + '\n').encode('ascii')
        data = subprocess.check_output(['git', '-C', str(ROOT), 'cat-file', '--batch'], input=payload)
        stream = io.BytesIO(data)
        for oid, name in objects.items():
            header = stream.readline().decode('ascii').split()
            if len(header) != 3 or header[:2] != [oid, 'blob']:
                raise ValueError('Unexpected Git object response')
            content = stream.read(int(header[2]))
            if stream.read(1) != b'\n':
                raise ValueError('Incomplete Git object response')
            try:
                validate_entry(name, content)
            except ValueError as exc:
                failures.append(f'{oid[:12]}: {exc}')
    if failures:
        raise ValueError('History audit failed:\n' + '\n'.join(failures))
    print(f'PASS: pattern scan of {len(objects)} reachable Git blobs and {len(seen_paths)} paths (local refs only)')


def verify(folder, write=False):
    if git('status', '--porcelain'):
        raise ValueError('Commit source changes before auditing release artifacts.')
    current = version()
    citation = (ROOT / 'CITATION.cff').read_text(encoding='utf-8')
    if not re.search(r'^version: [\"\']?' + re.escape(current) + r'[\"\']?$', citation, re.M):
        raise ValueError('CITATION.cff version does not match studio_version.py')
    if ('## ' + current) not in (ROOT / 'CHANGELOG.md').read_text(encoding='utf-8'):
        raise ValueError('Current version is missing from CHANGELOG.md')
    names = list(filter(None, git('ls-files', '-z').split('\0')))
    for name in names:
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
        verify_source_payload(entries, names, wheel=path.suffix == '.whl')
        if not source_matches(entries.get('studio_version.py', b''), (ROOT / 'studio_version.py').read_bytes()):
            raise ValueError(path.name + ': incorrect packaged version')
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    report = {'version': current, 'commit': git('rev-parse', 'HEAD'),
              'tree': git('rev-parse', 'HEAD^{tree}'), 'artifacts': hashes}
    checksum_text = ''.join(f'{digest}  {name}\n' for name, digest in hashes.items())
    if write:
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
    parser.add_argument('--history', action='store_true', help='Scan locally reachable history instead of distributions')
    args = parser.parse_args()
    if args.history:
        verify_history()
    else:
        verify(args.dist.resolve(), args.write)
