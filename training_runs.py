"""Resolve training output directories without loading any models."""
from pathlib import Path
import json
import hashlib
import shutil
import time
import uuid

MANIFEST = 'studio-run.json'

def content_hash(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()

def dataset_hash(folder):
    digest=hashlib.sha256()
    for path in sorted(p for p in Path(folder).rglob('*') if p.is_file()):
        digest.update(path.relative_to(folder).as_posix().encode('utf-8')+b'\0')
        digest.update(content_hash(path).encode('ascii')+b'\n')
    return digest.hexdigest()


def has_checkpoint(folder):
    required = ('trainer_state.json', 'optimizer.pt', 'scheduler.pt')
    for item in folder.glob('checkpoint-*'):
        if not item.name.removeprefix('checkpoint-').isdigit():
            continue
        state = all((item / name).is_file() for name in required)
        weights = any((item / name).is_file() for name in (
            'adapter_model.safetensors', 'adapter_model.bin'))
        if state and weights:
            return True
    return False

def resolve_training_run(args, paths):
    name = args.profile
    if not name or name in ('.', '..') or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-' for c in name):
        raise ValueError('Use a plain profile name, not a path.')
    requested = getattr(args, 'run_dir', None)
    if requested and not args.resume:
        raise ValueError('--run-dir requires --resume.')
    if args.resume:
        folders = [Path(requested).expanduser().resolve()] if requested else list(paths.staging.glob('*'))
        candidates = []
        for folder in folders:
            try:
                spec = json.loads((folder / MANIFEST).read_text(encoding='utf-8'))
            except (OSError, ValueError):
                continue
            if spec.get('profile') != name or (args.model and spec.get('model') != args.model):
                continue
            if not has_checkpoint(folder) or (folder / 'training_report.json').exists():
                continue
            candidates.append((folder, spec))
        if len(candidates) != 1:
            raise ValueError(f'Found {len(candidates)} resumable runs. Select an unfinished run with --resume --run-dir PATH. A saved Studio manifest and complete checkpoint are required; nothing was restarted.')
        folder, spec = candidates[0]
        if args.dataset and Path(args.dataset).expanduser().resolve() != Path(spec['dataset']):
            raise ValueError('Resume dataset differs from the pinned dataset revision.')
        if args.max_steps and args.max_steps != spec['max_steps']:
            raise ValueError('Resume must retain the original max-steps setting.')
        for key in ('config', 'anchors'):
            supplied = getattr(args, key, None)
            saved = spec.get(key)
            if supplied and (not saved or Path(supplied).expanduser().read_bytes() != Path(saved).read_bytes()):
                raise ValueError(f'Resume {key} differs from the pinned copy.')
            if saved and not Path(saved).is_file():
                raise ValueError(f'Missing pinned {key}: {saved}')
        if not Path(spec['dataset']).is_dir():
            raise ValueError('The pinned dataset directory no longer exists.')
        if spec.get('dataset_sha256') != dataset_hash(Path(spec['dataset'])):
            raise ValueError('Dataset content changed since this run was created; resume refused.')
        for key in ('config','anchors'):
            if spec.get(key) and spec.get(key+'_sha256') != content_hash(spec[key]):
                raise ValueError(f'Pinned {key} content changed; resume refused.')
        return folder, spec
    datasets = [p for p in paths.datasets.glob('*') if p.is_dir()]
    dataset = Path(args.dataset).expanduser().resolve() if args.dataset else max(datasets, key=lambda p:p.stat().st_mtime, default=None)
    if dataset is None or not dataset.is_dir():
        raise ValueError('No dataset revision found. Run studio dataset build first.')
    config = Path(args.config).expanduser().resolve() if args.config else paths.workspace/'profiles.json'
    if not config.is_file():
        raise ValueError('No profile configuration found. Run studio profile add first.')
    anchor = Path(args.anchors).expanduser().resolve() if args.anchors else paths.anchors/(name+'.json')
    if args.anchors and not anchor.is_file():
        raise ValueError('The requested anchors file does not exist.')
    folder = paths.staging/(name+'-'+time.strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:6])
    folder.mkdir(parents=True, exist_ok=False)
    shutil.copy2(config, folder/'profile-config.json')
    if anchor.is_file():
        shutil.copy2(anchor, folder/'anchors.json')
    spec = dict(profile=name, model=args.model or 'Qwen/Qwen3-4B', dataset=str(dataset),
                config=str(folder/'profile-config.json'), anchors=str(folder/'anchors.json') if anchor.is_file() else None,
                max_steps=args.max_steps, created_at=time.time())
    spec['dataset_sha256']=dataset_hash(dataset)
    for key in ('config','anchors'):
        if spec.get(key):spec[key+'_sha256']=content_hash(spec[key])
    (folder/MANIFEST).write_text(json.dumps(spec, indent=2)+'\n', encoding='utf-8')
    return folder, spec
