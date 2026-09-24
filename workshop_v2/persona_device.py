"""Memory estimates and device selection for single-device PEFT inference."""
from pathlib import Path
import json

GIB = 2**30


def estimates(model, context):
    config_path = Path(model['base_path']) / 'config.json'
    config = json.loads(config_path.read_text(encoding='utf-8'))
    width = int(config['hidden_size'])
    layers = int(config['num_hidden_layers'])
    heads = int(config['num_attention_heads'])
    kv_heads = int(config.get('num_key_value_heads', heads))
    head_dim = int(config.get('head_dim', width // heads))
    kv_bytes = 2 * 2 * layers * kv_heads * head_dim * int(context)
    base = int(model['base_bytes'])
    adapter = 2 * int(model['adapter_bytes'])
    embeddings = int(config['vocab_size']) * width * 2
    if not config.get('tie_word_embeddings', False):
        embeddings *= 2
    embeddings = min(base, embeddings)
    nf4 = int((base - embeddings) * .30) + embeddings + adapter
    bf16 = base + adapter
    workspace = int(.75 * GIB) + kv_bytes
    shards = [Path(p).stat().st_size for p in model.get('files', [])
              if Path(p).parent == Path(model['base_path']) and p.endswith('.safetensors')]
    return {'bf16': bf16 + workspace, 'nf4': nf4 + workspace,
            'cpu': bf16 + workspace, 'cuda_host': max(shards or [base]) + GIB,
            'kv_cache_bytes': kv_bytes}


def select_device(model, context, mode, precision, *, cuda_available,
                  vram_free=0, vram_total=0, nf4_available=False):
    mode = mode.upper()
    precision = precision.lower()
    if mode not in ('AUTO', 'CPU', 'CUDA'):
        raise ValueError('Persona device must be Auto, CPU, or CUDA.')
    if precision not in ('auto', 'bf16', 'nf4'):
        raise ValueError('Persona precision must be auto, bf16, or nf4.')
    if mode == 'CPU' and precision == 'nf4':
        raise ValueError('NF4 persona inference requires CUDA.')
    need = estimates(model, context)
    reserve = max(int(.5 * GIB), int(vram_total * .1))
    choices = ('bf16', 'nf4') if precision == 'auto' else (precision,)
    if mode != 'CPU' and cuda_available:
        for choice in choices:
            if choice == 'nf4' and not nf4_available:
                continue
            if need[choice] + reserve <= vram_free:
                return {'device': 'cuda:0', 'precision': choice,
                        'required_ram_bytes': need['cuda_host'],
                        'required_vram_bytes': need[choice] + reserve,
                        'vram_free_bytes': vram_free,
                        'kv_cache_bytes': need['kv_cache_bytes'],
                        'requested_mode': mode, 'fallback_reason': None}
    if mode == 'CUDA' or precision == 'nf4':
        if not cuda_available:
            raise RuntimeError('CUDA is unavailable in the configured persona Python environment.')
        if precision == 'nf4' and not nf4_available:
            raise RuntimeError('NF4 requires bitsandbytes in the persona Python environment.')
        available = [need[c] for c in choices if c != 'nf4' or nf4_available]
        required = min(available or [need['nf4']]) + reserve
        raise MemoryError(f'CUDA needs approximately {required / GIB:.2f} GiB free VRAM; '
                          f'{vram_free / GIB:.2f} GiB is available. '
                          'Unload another GPU model or reduce the context.')
    reason = None if mode == 'CPU' else 'CUDA unavailable or insufficient free VRAM'
    return {'device': 'cpu', 'precision': 'bf16',
            'required_ram_bytes': need['cpu'], 'required_vram_bytes': 0,
            'kv_cache_bytes': need['kv_cache_bytes'],
            'requested_mode': mode, 'fallback_reason': reason}
