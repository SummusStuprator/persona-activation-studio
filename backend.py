"""Exact local GGUF weights, real residual hooks, no model-executed tools."""
from __future__ import annotations
import ctypes as C
import hashlib
import json
import os
import shutil
from pathlib import Path
import re
import threading
import time
import urllib.request
import numpy as np
from studio_paths import data_root, ASSET_ROOT, CODE_ROOT
ROOT = data_root()
RUNTIME = Path(os.environ.get('STUDIO_NATIVE_RUNTIME', ROOT / 'native' / 'runtime')).expanduser().resolve()
OLLAMA = os.environ.get('OLLAMA_HOST', 'http://127.0.0.1:11434').rstrip('/')
ABI = 'studio-llama-c0bc8591e'
os.environ['GGML_CUDA_DISABLE_GRAPHS'] = '1'
os.environ['PATH'] = str(RUNTIME) + os.pathsep + os.environ['PATH']
os.environ.setdefault('OMP_NUM_THREADS', '6')
import sys

def _library(runtime: Path, stem: str) -> Path:
    names = ([stem + '.dll'] if os.name == 'nt' else
             ['lib' + stem + '.dylib', stem + '.dylib'] if sys.platform == 'darwin' else
             ['lib' + stem + '.so', stem + '.so'])
    for name in names:
        p = runtime / name
        if p.is_file(): return p
    raise FileNotFoundError(f"Missing native library {stem!r} in {runtime}. Run the native build step.")

F32 = np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS')
I32 = np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS')

def ollama_json(path: str, data: dict | None = None) -> dict:
    body = None if data is None else json.dumps(data).encode()
    req = urllib.request.Request(OLLAMA + path, body, {'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=20) as response:
        return json.load(response)

from model_store import inventory
from contracts import validate_axis
from lifecycle import clean_generation

class Engine:
    def __init__(self, runtime=None) -> None:
        self.runtime = Path(runtime).expanduser().resolve() if runtime else RUNTIME
        bridge = _library(self.runtime, 'activation_bridge')
        llama_lib = _library(self.runtime, 'llama')
        from integrity import fingerprint
        libraries=sorted(p for p in self.runtime.iterdir() if p.is_file() and (p.suffix in ('.dll','.so','.dylib') or '.so.' in p.name))
        self.runtime_files_sha256={p.name:fingerprint(str(p),p.stat().st_size,p.stat().st_mtime_ns) for p in libraries}
        digest=hashlib.sha256(json.dumps(self.runtime_files_sha256,sort_keys=True).encode()).hexdigest()[:16]
        self.abi = ABI + '-v2-' + digest
        os.environ['PATH'] = str(self.runtime) + os.pathsep + os.environ.get('PATH','')
        self.lock = threading.RLock()
        self.dll_directory = os.add_dll_directory(str(self.runtime)) if hasattr(os,'add_dll_directory') else None
        self.cuda_directories=[]
        if hasattr(os,'add_dll_directory'):
            candidates=[]
            if os.environ.get('CUDA_PATH'):candidates.append(Path(os.environ['CUDA_PATH'])/'bin')
            nvcc=shutil.which('nvcc')
            if nvcc:candidates.append(Path(nvcc).resolve().parent)
            for directory in dict.fromkeys(p for base in candidates for p in (base,base/'x64')):
                if directory.is_dir():self.cuda_directories.append(os.add_dll_directory(str(directory)))
        self.dll = C.CDLL(str(bridge))
        self.ggml = C.CDLL(str(_library(self.runtime,'ggml')))
        self.ggml.ggml_backend_load.argtypes = [C.c_char_p]
        self.ggml.ggml_backend_load.restype = C.c_void_p
        self.cuda_available = False
        try:
            cuda_path=_library(self.runtime,'ggml-cuda')
            self.cuda = C.CDLL(str(cuda_path))
            self.cuda_available = bool(self.ggml.ggml_backend_load(os.fsencode(cuda_path)))
        except (OSError,FileNotFoundError) as exc:
            self.cuda_error=str(exc)
        self.handle = None
        self.model = None
        P, I, S, B = C.c_void_p, C.c_int, C.c_char_p, C.POINTER(C.c_char)
        self._bind('lab_error', S, [])
        self._bind('lab_open', P, [S, S, I, I, I])
        self._bind('lab_close', None, [P])
        for name in ('lab_dim', 'lab_layers', 'lab_vocab_size'):
            self._bind(name, I, [P])
        self._bind('lab_is_eog', I, [P, I])
        self._bind('lab_meta', I, [P, S, B, I])
        self._bind('lab_reset', None, [P])
        self._bind('lab_tokenize', I, [P, S, I, I32, I])
        self._bind('lab_piece', I, [P, I, B, I])
        self._bind('lab_template', I, [P, C.POINTER(S), C.POINTER(S), I, B, I])
        self._bind('lab_set_direction', None, [P, F32, I, C.c_float])
        self._bind('lab_eval', I, [P, I32, I, I])
        self._bind('lab_capture', None, [P, F32, I])
        self._bind('lab_logits', None, [P, F32])

    def _bind(self, name, result, arguments):
        fn = getattr(self.dll, name)
        fn.restype, fn.argtypes = result, arguments

    def _error(self) -> RuntimeError:
        return RuntimeError((self.dll.lab_error() or b'Native bridge error').decode('utf-8', 'replace'))

    def close(self) -> None:
        with self.lock:
            if self.handle:
                self.dll.lab_close(self.handle)
                self.handle = None
    def open(self, model: dict, gpu_layers: int = 99, context: int = 2048) -> None:
        with self.lock:
            if model.get('reason'): raise ValueError(model['reason'])
            if gpu_layers > 0 and not self.cuda_available:
                raise RuntimeError('CUDA runtime unavailable. Select explicit CPU mode; no silent fallback.')
            self.close()
            self.context = context
            self.model = dict(model, runtime_files_sha256=self.runtime_files_sha256)
            self.handle = self.dll.lab_open(os.fsencode(model['path']), os.fsencode(self.runtime), gpu_layers, context, int(os.environ.get('WORKSHOP_CPU_THREADS','6')))
            if not self.handle: raise self._error()
            self.dim = self.dll.lab_dim(self.handle)
            self.layers = self.dll.lab_layers(self.handle)
            self.vocab_size = self.dll.lab_vocab_size(self.handle)
            self.clear_steering()
            self.reset()
            try:
                self.evaluate(self.tokenize('A small local activation test.'))
                if not np.isfinite(self.capture()).all(): raise RuntimeError('Non-finite residual activations.')
            except Exception:
                self.close()
                raise
            self.reset()

    def reset(self) -> None:
        if not self.handle: raise RuntimeError('Load a model first.')
        self.dll.lab_reset(self.handle)

    def clear_steering(self) -> None:
        self.dll.lab_set_direction(self.handle, np.zeros((self.layers, self.dim), np.float32), 0, 0)

    def steer(self, directions: np.ndarray, strength: float, mode: str = 'add') -> None:
        vector = np.ascontiguousarray(directions, dtype=np.float32)
        if vector.shape != (self.layers, self.dim) or not np.isfinite(vector).all():
            raise ValueError('Direction must be finite and match every model dimension.')
        if mode not in ('add', 'erase'): raise ValueError('Unknown intervention mode.')
        if not np.isfinite(strength): raise ValueError('Strength must be finite.')
        if mode == 'erase':
            norms = np.linalg.norm(vector, axis=1, keepdims=True)
            vector = np.ascontiguousarray(vector / np.maximum(norms, 1e-12))
            if not 0 <= strength <= 1: raise ValueError('Erasure fraction must be between 0 and 1.')
        self.dll.lab_set_direction(self.handle, vector, 1 if mode == 'add' else 2, strength)

    def tokenize(self, text: str) -> np.ndarray:
        raw = text.encode('utf-8')
        tokens = np.empty(len(raw) + 16, np.int32)
        n = self.dll.lab_tokenize(self.handle, raw, len(raw), tokens, len(tokens))
        if n < 0: raise RuntimeError('Tokenizer buffer was too small.')
        if n == 0: raise ValueError('Prompt is empty.')
        return np.ascontiguousarray(tokens[:n])

    def evaluate(self, tokens: np.ndarray, all_logits: bool = False) -> None:
        tokens = np.ascontiguousarray(tokens, dtype=np.int32)
        if self.dll.lab_eval(self.handle, tokens, len(tokens), int(all_logits)) != 0:
            raise self._error()

    def capture(self, which: int = 0) -> np.ndarray:
        result = np.empty((self.layers, self.dim), np.float32)
        self.dll.lab_capture(self.handle, result, which)
        return result
    def chat(self, messages: list[dict]) -> str:
        if not messages or any(m.get('role') not in ('system', 'user', 'assistant') for m in messages):
            raise ValueError('Use system, user, or assistant chat messages.')
        roles = (C.c_char_p * len(messages))(*[m['role'].encode() for m in messages])
        texts = (C.c_char_p * len(messages))(*[m['content'].encode('utf-8') for m in messages])
        out = C.create_string_buffer(131072)
        n = self.dll.lab_template(self.handle, roles, texts, len(messages), out, len(out))
        if n < 0:
            from template_support import render_chat
            return render_chat(self, messages)
        if n >= len(out): raise ValueError('Chat template exceeds 128 KiB.')
        return out.raw[:n].decode('utf-8')

    def logits(self) -> np.ndarray:
        result = np.empty(self.vocab_size, np.float32)
        self.dll.lab_logits(self.handle, result)
        if not np.isfinite(result).all(): raise RuntimeError('Non-finite model logits; reduce steering.')
        return result

    def is_eog(self, token: int) -> bool:
        return bool(self.dll.lab_is_eog(self.handle,int(token)))

    def piece(self, token: int) -> bytes:
        out = C.create_string_buffer(1024)
        n = self.dll.lab_piece(self.handle, token, out, len(out))
        if n < 0: raise RuntimeError('Token representation is too long.')
        return out.raw[:n]

    def extract(self, text: str, mean: bool = False) -> np.ndarray:
        with self.lock:
            self.clear_steering()
            self.reset()
            self.evaluate(self.tokenize(text), all_logits=mean)
            return self.capture(2 if mean else 0)
    @clean_generation
    def generate(self, text: str, bank: dict, axis: str | None = None, layer: int | None = None,
                 strength: float = 0, mode: str = 'add', max_tokens: int = 96,
                 temperature: float = 0, seed: int = 42) -> dict:
        with self.lock:
            started = time.monotonic()
            if mode not in ('add','erase','random'): raise ValueError('Invalid intervention mode.')
            if not np.isfinite(strength) or not np.isfinite(temperature) or temperature < 0: raise ValueError('Invalid strength or temperature.')
            for name, item in bank.items():
                validate_axis(self, item['meta']['name'] if item.get('control_only') else name, item)
            if axis is not None and axis not in bank: raise ValueError('Unknown direction.')
            if not 1 <= max_tokens <= 256: raise ValueError('Generation limit: 1 to 256 tokens.')
            if abs(strength) > 0.5 and mode != 'erase': raise ValueError('Start within a 0.5 residual-norm fraction.')
            self.clear_steering()
            self.reset()
            table = np.zeros((self.layers, self.dim), np.float32)
            rng = np.random.default_rng(seed)
            if axis is not None:
                a = bank[axis]
                layer = a['layer'] if layer is None else layer
                if not 0 <= layer < self.layers: raise ValueError('Invalid block index.')
                vector = a['directions'][layer].copy()
                if mode == 'random':
                    vector = np.random.default_rng(seed + 1000).normal(size=self.dim).astype(np.float32)
                    vector /= np.linalg.norm(vector)
                table[layer] = vector if mode == 'erase' else vector * a['norm'][layer]
                self.steer(table, strength, 'erase' if mode == 'erase' else 'add')
            tokens = self.tokenize(text)
            if len(tokens) + max_tokens > self.context:
                self.clear_steering()
                raise ValueError(f'Shorten the prompt or generation to fit the {self.context:,}-token context.')
            pieces, traces, generated = [], [], []
            try:
                self.evaluate(tokens)
                for step in range(max_tokens):
                    pre, post = self.capture(0), self.capture(1)
                    logits = self.logits().astype(np.float64)
                    probabilities = np.exp(logits - logits.max())
                    probabilities /= probabilities.sum()
                    if temperature <= 0:
                        token = int(logits.argmax())
                    else:
                        indices = np.argpartition(logits, -40)[-40:]
                        weights = np.exp((logits[indices] - logits[indices].max()) / temperature)
                        token = int(rng.choice(indices, p=weights / weights.sum()))
                    if self.dll.lab_is_eog(self.handle, token): break
                    piece = self.piece(token)
                    pieces.append(piece)
                    generated.append(token)
                    trace = {'step': step, 'token': piece.decode('utf-8', 'replace'),
                             'entropy_nats': float(-np.sum(probabilities * np.log(probabilities + 1e-30))),
                             'chosen_token_probability': float(probabilities[token])}
                    for name, a in bank.items():
                        if a.get('control_only', False): continue
                        k = a['layer']
                        trace[name] = float((pre[k] @ a['directions'][k] - a['center'][k]) / a['scale'][k])
                        trace[name+'__after'] = float((post[k] @ a['directions'][k] - a['center'][k]) / a['scale'][k])
                    if axis is not None:
                        trace['injected_norm_fraction'] = float(np.linalg.norm(post[layer] - pre[layer]) / max(np.linalg.norm(pre[layer]), 1e-12))
                        trace['target_projection_before'] = float(pre[layer] @ bank[axis]['directions'][layer])
                        trace['target_projection_after'] = float(post[layer] @ bank[axis]['directions'][layer])
                        a = bank[axis]
                        trace['target_z_before'] = float((pre[layer] @ a['directions'][layer]-a['center'][layer])/a['scale'][layer])
                        trace['target_z_after'] = float((post[layer] @ a['directions'][layer]-a['center'][layer])/a['scale'][layer])
                    traces.append(trace)
                    if step + 1 < max_tokens: self.evaluate(np.array([token], np.int32))
            finally:
                self.clear_steering()
                self.reset()
            return {'text': b''.join(pieces).decode('utf-8', 'replace'), 'trace': traces,
                    'token_ids': generated, 'prompt_tokens': len(tokens), 'prompt': text,
                    'axis': axis, 'layer': layer, 'strength': strength, 'mode': mode,
                    'temperature': temperature, 'seed': seed, 'abi': self.abi,
                    'model': self.model, 'seconds': round(time.monotonic() - started, 3),
                    'intervention_vector_before_strength': table[layer].tolist() if axis is not None else None,
                    'probe_provenance': {name: a['meta'] for name, a in bank.items() if not a.get('control_only', False)},
                    'scope': 'Intervention on all prompt and generated tokens; state reset between runs.'}

def save_run(result: dict, kind: str = 'generation') -> Path:
    directory = ROOT / 'runs'
    directory.mkdir(exist_ok=True)
    filename = directory / (time.strftime('%Y%m%d-%H%M%S') + f'-{time.time_ns() % 1000000:06d}-{kind}.json')
    filename.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    return filename

from workshop_v2.native_research import mean_transport, unembed_residual, read_positions
Engine.mean_transport = mean_transport
Engine.unembed_residual = unembed_residual
Engine.read_positions = read_positions

from workshop_v2.chat_native import extract_response, score_response
Engine.extract_response = extract_response
Engine.score_response = score_response
