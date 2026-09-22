"""Keep native model faults and GPU allocations out of the web process."""
from __future__ import annotations
import atexit
import functools
import json
import subprocess
import sys
import secrets
import uuid
from multiprocessing.connection import Client
import os
import psutil
from pathlib import Path
import threading
import time
import tempfile
from model_store import ROOT, plan_load
from studio_paths import CODE_ROOT
METHODS = {'open', 'close', 'reset', 'clear_steering', 'steer', 'tokenize',
           'evaluate', 'capture', 'chat', 'logits', 'piece','is_eog', 'extract', 'generate'}
METHODS.update({'mean_transport','unembed_residual','read_positions','extract_response','score_response'})

class Engine:
    def __init__(self):
        self.lock = threading.RLock()
        self.process = self.connection = None
        self._loaded = False; self.model = None
        self.load_plan = {}; self.timeout = 300
        self.log_path = ROOT / 'logs' / 'model-worker.log'
        self.log_path.parent.mkdir(parents=True,exist_ok=True)
        atexit.register(self.close)

    @property
    def handle(self):
        return self._loaded and self.process is not None and self.process.poll() is None

    def _start(self):
        if os.name == 'nt':
            family='AF_PIPE';address=r'\\.\pipe\PersonaActivationStudio-' + uuid.uuid4().hex
        else:
            family='AF_UNIX';address=str(Path(tempfile.gettempdir())/('persona-activation-studio-'+uuid.uuid4().hex+'.sock'))
            try:Path(address).unlink()
            except FileNotFoundError:pass
        self.worker_address=address;self.worker_family=family
        auth = secrets.token_bytes(32)
        env = os.environ.copy()
        env['ACTIVATION_LAB_WORKER_KEY'] = auth.hex()
        env['ACTIVATION_LAB_RUNTIME'] = str(self.runtime_path)
        env['STUDIO_WORKER_FAMILY'] = family
        env['PYTHONUTF8'] = '1'
        env['PYTHONIOENCODING'] = 'utf-8'
        with self.log_path.open('ab') as log:
            self.process = subprocess.Popen([getattr(self,'worker_python',sys.executable), str(getattr(self,'worker_script',CODE_ROOT/'model_worker.py')), address],
                cwd=CODE_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        for _ in range(300):
            if self.process.poll() is not None:
                raise RuntimeError('Model worker startup failed. See '+str(self.log_path))
            try:
                self.connection = Client(address, family=family, authkey=auth)
                return
            except (FileNotFoundError, ConnectionRefusedError, OSError):
                time.sleep(0.1)
        self.close()
        raise RuntimeError('Model worker did not connect. See '+str(self.log_path))

    def _call(self, method, *args, **kwargs):
        with self.lock:
            if self.process is None: raise RuntimeError('Load a model first.')
            try:
                self.connection.send((method, args, kwargs))
                if not self.connection.poll(self.timeout):
                    self.close()
                    raise TimeoutError('Model operation exceeded its limit; only its worker was stopped.')
                reply = self.connection.recv()
            except (EOFError, BrokenPipeError, OSError) as exc:
                code = self.process.returncode if self.process else None
                self.close()
                raise RuntimeError(f'Native model worker exited ({code}). The lab is still running. See {self.log_path}') from exc
            if not reply['ok']: raise RuntimeError(reply['error'])
            state = reply['state']; self._loaded = state.pop('loaded')
            self.__dict__.update(state)
            return reply['value']

    def __getattr__(self, name):
        if name in METHODS: return functools.partial(self._call, name)
        raise AttributeError(name)

    def close(self):
        with self.lock:
            self._loaded = False
            process, connection = self.process, self.connection
            self.process = self.connection = None
            if process is None: return
            try:
                descendants = psutil.Process(process.pid).children(recursive=True)
            except psutil.Error:
                descendants = []
            try:
                if process.poll() is None:
                    if connection is not None: connection.send(('shutdown', (), {}))
                    try: process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        process.terminate()
                        process.wait(timeout=3)
            except (OSError, EOFError, BrokenPipeError, subprocess.TimeoutExpired):
                if process.poll() is None: process.kill()
            finally:
                if connection is not None: connection.close()
                for child in reversed(descendants):
                    try:
                        if child.is_running(): child.terminate()
                    except psutil.Error: pass
                _, alive = psutil.wait_procs(descendants, timeout=2)
                for child in alive:
                    try: child.kill()
                    except psutil.Error: pass
                if getattr(self,'worker_family',None)=='AF_UNIX':
                    try:Path(self.worker_address).unlink()
                    except (FileNotFoundError,OSError):pass

    def open(self, model, gpu_layers=None, context=2048, mode='Auto', manual=0, precision='auto'):
        with self.lock:
            if model.get('reason'): raise ValueError(model['reason'])
            from integrity import verify_release
            verify_release()
            self.close()
            from workshop_v2.resource_policy import guard_load,settings as resource_settings,ModelLaunchLease
            launch=None
            try:
                if resource_settings()['cooperative']:launch=ModelLaunchLease()
                resource_plan=guard_load(model,context=context,mode=mode,manual=manual)
            except RuntimeError as exc:
                if launch is not None:launch.close()
                raise MemoryError('Deferred: another Studio model launch is already in progress. No process was stopped.') from exc
            except Exception:
                if launch is not None:launch.close()
                raise
            if model.get('backend') == 'persona_peft':
                from workshop_v2.persona_store import settings
                self.worker_python = settings()['python']
                if not Path(self.worker_python).is_file():
                    import shutil
                    resolved=shutil.which(str(self.worker_python))
                    if not resolved:raise RuntimeError('Configured persona Python environment does not exist.')
                    self.worker_python=resolved
                self.worker_script = CODE_ROOT/'persona_worker.py'
                self.runtime_path = ROOT
                self.log_path = ROOT/'logs'/('persona-'+model['selection_id']+'.log')
                self.timeout = 300
                try:
                    self._start()
                    details=self._call('open', model, context=context, mode=mode, precision=precision)
                    self.load_plan = {'backend':'persona_peft', **details['plan'], 'base':self.model['base_path'],
                        'adapter':self.model['adapter_path'], 'digest':self.model['digest'], 'abi':self.abi,
                        'context':context, 'weight_files_copied':False, 'weights_frozen':True}
                    self._record('loaded')
                    if launch is not None:launch.close();launch=None
                    return
                except Exception:
                    self.close()
                    if launch is not None:launch.close();launch=None
                    raise
            self.worker_python=sys.executable
            self.worker_script=CODE_ROOT/'model_worker.py'
            if gpu_layers is not None:
                mode, manual = ('CPU', 0) if gpu_layers == 0 else ('Manual', gpu_layers)
            profile = 'runtime'
            candidate = ROOT/'native'/profile
            self.runtime_path = Path(os.environ['STUDIO_NATIVE_RUNTIME']).expanduser().resolve() if os.environ.get('STUDIO_NATIVE_RUNTIME') else (candidate if candidate.is_dir() and any(candidate.iterdir()) else ROOT/'native'/'runtime')
            model = dict(model, runtime_profile=profile)
            self.log_path = ROOT/'logs'/('model-'+model['digest'][:23]+'.log')
            self.load_plan = plan_load(model, context, mode, manual)
            first = self.load_plan['gpu_layers']
            attempts = [first]
            if mode == 'Auto' and first: attempts += list(dict.fromkeys([first//2, 0]))
            errors = []
            for count in attempts:
                try:
                    actual_mode='CPU' if count==0 else 'Manual'
                    # Recheck host RAM when fewer GPU layers (or CPU fallback) are used.
                    guard_load(model,context=context,mode=actual_mode,manual=count)
                    actual_plan=plan_load(model,context,actual_mode,count)
                    self.load_plan.update(actual_plan,requested_mode=mode)
                    self._start()
                    self._call('open', model, gpu_layers=count, context=context)
                    self.load_plan.update(gpu_layers=count, fallback_errors=errors)
                    self._record('loaded')
                    if launch is not None:launch.close();launch=None
                    return
                except Exception as exc:
                    errors.append(f'{count} GPU layers: {exc}')
                    self.close()
                    text = str(exc).lower()
                    if not any(word in text for word in ('load failed', 'context creation', 'worker exited', 'memory', 'cuda')):
                        break
            self.model = model
            self._record('failed', '\n'.join(errors))
            if launch is not None:launch.close();launch=None
            raise RuntimeError('\n'.join(errors) + '\nNative details: ' + str(self.log_path))

    def _record(self, status, error=''):
        folder = ROOT/'compatibility'; folder.mkdir(exist_ok=True)
        record = {'name': self.model['name'], 'digest': self.model['digest'],
                  'architecture': self.model.get('architecture'), 'status': status,
                  'error': error, 'time': time.time(), 'plan': self.load_plan,
                  'scope': 'Load and finite activation check; not a trained-direction validation.'}
        path = folder/(self.model['digest']+'.json')
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(record, indent=2), encoding='utf-8')
        temporary.replace(path)
