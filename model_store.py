"""Read existing Ollama manifests and GGUF metadata without copying weights."""
from __future__ import annotations
import functools
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import psutil
from gguf_header import read_header
from studio_paths import data_root, ASSET_ROOT, CODE_ROOT
ROOT = data_root()
GIB = 1024 ** 3

def store_roots() -> list[Path]:
    candidates = [os.environ.get('OLLAMA_MODELS'), str(Path.home()/'.ollama/models')]
    settings = ROOT / 'lab-settings.json'
    if settings.exists():
        candidates += json.loads(settings.read_text(encoding='utf-8')).get('model_directories', [])
    return list(dict.fromkeys(Path(p).expanduser().resolve() for p in candidates if p))

@functools.lru_cache(maxsize=128)
def _metadata(path: str, size: int, modified: int) -> dict:
    fields, tensors = read_header(path)
    def field(name, default=None):
        return fields.get(name, default)
    arch = field('general.architecture', '')
    blocks = int(field(arch + '.block_count', 0))
    layer_bytes = [0] * blocks
    other_bytes = output_bytes = 0
    for tensor_name, tensor_bytes in tensors:
        match = re.match(r'blk\.(\d+)\.', tensor_name)
        n = int(tensor_bytes)
        if match and int(match[1]) < blocks:
            layer_bytes[int(match[1])] += n
        elif tensor_name.startswith('output'):
            output_bytes += n
        else:
            other_bytes += n
    return {'architecture': arch, 'family': arch, 'layers': blocks,
            'logical_layers': blocks*int(field(arch+'.num_loops',1)),
            'dim': int(field(arch + '.embedding_length', 0)),
            'train_context': int(field(arch + '.context_length', 0)),
            'file_type': field('general.file_type'), 'layer_bytes': layer_bytes,
            'other_bytes': other_bytes, 'output_bytes': output_bytes,
            'chat_template': field('tokenizer.chat_template', ''),
            'bos_id': field('tokenizer.ggml.bos_token_id'),
            'eos_id': field('tokenizer.ggml.eos_token_id'),
            'split_index': int(field('split.no', 0)),
            'split_count': int(field('split.count', 1))}

def metadata(path: str) -> dict:
    p = Path(path); stat = p.stat()
    with p.open('rb') as source:
        if source.read(4) != b'GGUF': raise ValueError('The weight file is not GGUF.')
    cache_dir=ROOT/'cache'/'gguf-headers';cache_dir.mkdir(parents=True,exist_ok=True)
    key=hashlib.sha256(str(p).encode('utf-8')).hexdigest()
    cache=cache_dir/(key+'.json')
    if cache.exists():
        try:
            doc=json.loads(cache.read_text(encoding='utf-8'))
            if doc.get('path')==str(p) and doc.get('size')==stat.st_size and doc.get('mtime_ns')==stat.st_mtime_ns:
                return doc['metadata']
        except (OSError,ValueError,KeyError):pass
    result=dict(_metadata(str(p),stat.st_size,stat.st_mtime_ns))
    tmp=cache.with_suffix('.tmp')
    tmp.write_text(json.dumps({'path':str(p),'size':stat.st_size,'mtime_ns':stat.st_mtime_ns,'metadata':result}),encoding='utf-8')
    tmp.replace(cache)
    return result

def _blob(store: Path, digest: str) -> Path:
    if not re.fullmatch(r'sha256:[0-9a-f]{64}', digest):
        raise ValueError('Invalid content-addressed blob digest.')
    return store / 'blobs' / digest.replace(':', '-')

def inventory() -> list[dict]:
    rows = []
    for store in store_roots():
        for manifest in sorted((store/'manifests').rglob('*')):
            if not manifest.is_file(): continue
            parts = list(manifest.relative_to(store/'manifests').parts)
            if len(parts) < 3: continue
            tag = parts.pop()
            if parts[:2] == ['registry.ollama.ai', 'library']: parts = parts[2:]
            elif parts[0] == 'registry.ollama.ai': parts = parts[1:]
            row = {'name': '/'.join(parts) + ':' + tag, 'reason': '',
                   'manifest': str(manifest), 'store': str(store), 'size_gb': 0,
                   'family': '', 'path': '', 'digest': '', 'quant': ''}
            try:
                raw = manifest.read_bytes(); doc = json.loads(raw)
                row['ollama_digest'] = hashlib.sha256(raw).hexdigest()
                layers = doc.get('layers') or []
                paths = [_blob(store, x['digest']) for x in layers
                         if x.get('mediaType') == 'application/vnd.ollama.image.model']
                row['adapter_paths'] = [str(_blob(store, x['digest'])) for x in layers
                    if x.get('mediaType') == 'application/vnd.ollama.image.adapter']
                row['projector_paths'] = [str(_blob(store, x['digest'])) for x in layers
                    if x.get('mediaType') == 'application/vnd.ollama.image.projector']
                if not paths: raise ValueError('Cloud-only or no local GGUF weight layer.')
                infos = [(p, metadata(str(p))) for p in paths]
                infos.sort(key=lambda item: item[1]['split_index'])
                paths = [p for p, _ in infos]
                if infos[0][1]['split_count'] != len(paths):
                    raise ValueError('GGUF shards are incomplete in this manifest.')
                if len(paths) > 1: raise ValueError('Split GGUF is detected but not yet supported by this bridge; no files were merged or copied.')
                row.update(infos[0][1])
                row['paths'] = [str(p) for p in paths]
                row['path'] = str(paths[0])
                row['size_bytes'] = sum(p.stat().st_size for p in paths)
                row['size_gb'] = round(row['size_bytes']/1e9, 2)
                row['digest'] = paths[0].name if len(paths) == 1 else 'sha256-' + hashlib.sha256('|'.join(p.name for p in paths).encode()).hexdigest()
                if len(infos) > 1:
                    row['layer_bytes'] = [sum(info['layer_bytes'][i] for _, info in infos) for i in range(row['layers'])]
                    for key in ('output_bytes', 'other_bytes'):
                        row[key] = sum(info[key] for _, info in infos)
                if row['adapter_paths']:
                    raise ValueError('Separate adapter present; base-only loading would change this model.')
                if not row['layers']: raise ValueError('Not a supported decoder-block GGUF model.')
            except Exception as exc:
                row['reason'] = str(exc)
            rows.append(row)
    names = set()
    return [r for r in rows if not (r['name'] in names or names.add(r['name']))]

def grouped_inventory() -> tuple[list[dict], list[dict]]:
    rows = inventory(); groups = {}
    for row in rows:
        if row['reason']: continue
        key = row['digest']
        if key not in groups: groups[key] = dict(row, aliases=[])
        groups[key]['aliases'].append(row['name'])
    preferred = ['llama3.2:1b', 'smollm2:360m', 'gemma3:latest', 'qwen3:4b']
    for row in groups.values():
        row['name'] = next((n for n in preferred if n in row['aliases']),
                           min(row['aliases'], key=len))
    return sorted(groups.values(), key=lambda r: (r['name'] != 'llama3.2:1b', r['size_gb'])), rows

def hardware() -> dict:
    mem = psutil.virtual_memory()
    result = {'ram_available_gib': mem.available/GIB, 'ram_total_gib': mem.total/GIB,
              'gpu_free_gib': 0.0, 'gpu_total_gib': 0.0, 'gpu_name': 'CPU only'}
    try:
        out = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.free,memory.total',
            '--format=csv,noheader,nounits'], timeout=5, creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0)).decode()
        name, free, total = out.splitlines()[0].rsplit(',', 2)
        result.update(gpu_name=name.strip(), gpu_free_gib=float(free)/1024,
                      gpu_total_gib=float(total)/1024)
    except (OSError, subprocess.SubprocessError, ValueError): pass
    return result

def plan_load(model: dict, context: int = 2048, mode: str = 'Auto', manual: int = 0) -> dict:
    hw = hardware(); layers = model['layers']
    budget = max(0, (hw['gpu_free_gib'] * 0.9 - 1.0 - context/8192) * GIB)
    sizes = model['layer_bytes']; offloaded = 0
    count = 0
    if mode != 'CPU':
        for size in reversed(sizes):
            if offloaded + size > budget: break
            offloaded += size; count += 1
        if count == layers and offloaded + model['output_bytes'] <= budget:
            offloaded += model['output_bytes']; count += 1
    if mode == 'Manual':
        count = min(max(int(manual), 0), layers+1)
        offloaded = sum(sizes[max(0,layers-count):])
        if count > layers: offloaded += model['output_bytes']
    logical=model.get('logical_layers',layers)
    if mode=='Manual' and logical>layers:
        count=min(max(int(manual),0),logical+1)
        physical_count=min(layers,max(0,count-(logical-layers)))
        offloaded=sum(sizes[layers-physical_count:]) if physical_count else 0
        if count>logical:offloaded+=model['output_bytes']
    cpu_gib = max(0, model['size_bytes'] - offloaded)/GIB
    warnings = []
    if cpu_gib + 2 > hw['ram_available_gib']:
        warnings.append('Estimated host weight footprint is near available RAM; the host may page and run slowly. Close other model sessions or use a smaller context.')
    logical=model.get('logical_layers',layers)
    if logical>layers and count>0 and mode!='Manual':count+=logical-layers
    if mode=='Manual' and logical>layers:count=min(max(int(manual),0),logical+1)
    return {'mode': mode, 'gpu_layers': count, 'total_layers': logical,
            'context': context, 'estimated_gpu_weights_gib': offloaded/GIB,
            'estimated_cpu_weights_gib': cpu_gib, 'hardware': hw, 'warnings': warnings,
            'note': 'Planning estimate, not a guarantee. KV cache, recurrent state, graph and tensor placement depend on architecture.'}
