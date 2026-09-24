"""Content identity for resumable benchmark outputs."""
from pathlib import Path
import hashlib
import json
import os
import importlib.metadata

def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def base_snapshot(reference):
    path = Path(reference).expanduser()
    if path.is_dir():
        return path.resolve()
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(reference, local_files_only=True)).resolve()

def tree_hash(folder):
    files = sorted(p for p in Path(folder).rglob('*') if p.is_file() and p.suffix in ('.json', '.jsonl', '.safetensors', '.bin', '.jinja'))
    return hashlib.sha256(json.dumps([(p.relative_to(folder).as_posix(), file_hash(p)) for p in files], separators=(',', ':')).encode()).hexdigest()

def identity(models, probes, args):
    snapshots = {m['base_model']: base_snapshot(m['base_model']) for m in models}
    spec = {'schema': 2, 'probes': probes,
            'sampler': {k: getattr(args, k) for k in ('temperature', 'top_p', 'top_k', 'max_new_tokens', 'heldout_count', 'quick', 'skip_base_nll')},
            'trainer_sha256': file_hash(args.trainer),
            'benchmark_sha256': file_hash(Path(__file__).with_name('benchmark.py')),
            'identity_sha256': file_hash(__file__),
            'bases': {k: tree_hash(v) for k, v in snapshots.items()},
            'models': [{'id': m['model_id'], 'adapter': tree_hash(Path(m['adapter'])),
                        'heldout': m['heldout'], 'training_report': m['training_report']} for m in models],
            'versions': {k: importlib.metadata.version(k) for k in ('torch', 'transformers', 'peft', 'bitsandbytes')}}
    digest = hashlib.sha256(json.dumps(spec, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    return digest, spec, snapshots

def require_compatible(output, fingerprint):
    output = Path(output)
    files = ('benchmark_results.jsonl', 'nll_scores.json')
    if any((output / f).exists() for f in files):
        try:
            saved = json.loads((output / 'benchmark_manifest.json').read_text(encoding='utf-8'))
        except (OSError, ValueError):
            saved = {}
        if saved.get('fingerprint') != fingerprint:
            raise ValueError('Benchmark inputs changed. Select a new --output-root; existing results were retained.')
