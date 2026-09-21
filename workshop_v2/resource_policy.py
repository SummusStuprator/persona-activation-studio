"""Resource cooperation for this workshop only; never pauses unrelated applications."""
from contextlib import contextmanager
from pathlib import Path
import json, os, time
import psutil
ROOT = Path(__file__).resolve().parent.parent
DEFAULTS = dict(cooperative=True, threads=2, reserve_ram_gib=.75,
                reserve_ram_fraction=.02, checkpoint_floor_gib=.5,
                checkpoint_floor_fraction=.015, delay_per_example=0.0,
                allow_gpu_if_available=True)


def settings():
    result = DEFAULTS.copy()
    path = ROOT / 'resource-policy.json'
    if path.exists():
        result.update(json.loads(path.read_text(encoding='utf-8')))
    try:
        from studio_config import load_config
        configured=load_config().get('runtime',{}).get('cpu_threads')
        if configured is not None:result['threads']=configured
    except Exception:
        pass
    result['threads'] = max(1, min(32, int(result['threads'])))
    return result


def configure_process():
    """Only called within a workshop-owned inference/test process."""
    cfg = settings()
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = str(cfg['threads'])
    os.environ['WORKSHOP_CPU_THREADS'] = str(cfg['threads'])
    if cfg['cooperative']:
        if not cfg.get('allow_gpu_if_available', True):
            os.environ['CUDA_VISIBLE_DEVICES']=''
        try:
            psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if os.name == 'nt' else 10)
        except (psutil.Error, AttributeError):
            pass
    claim_worker_slot()
    return cfg


def load_budget(model, context=2048, mode='Auto', manual=0):
    cfg = settings()
    free = psutil.virtual_memory().available / 2**30
    if model.get('backend') == 'persona_peft':
        weight = (model['base_bytes'] + model['adapter_bytes']) / 2**30
        working = weight + 1.25
    else:
        weight = Path(model['path']).stat().st_size / 2**30
        try:
            from model_store import plan_load
            effective_mode = 'Auto' if cfg['cooperative'] and mode == 'Manual' else mode
            split = plan_load(model, context, effective_mode, 0 if effective_mode != 'Manual' else manual)
            host_weight = float(split['estimated_cpu_weights_gib'])
            gpu_weight = float(split['estimated_gpu_weights_gib'])
        except Exception:
            host_weight, gpu_weight, split = weight, 0.0, None
        working = host_weight + .75
    vm = psutil.virtual_memory()
    raw_total = getattr(vm,'total',0)
    total = (float(raw_total) / 2**30) if isinstance(raw_total,(int,float)) else 0.0
    reserve = max(float(cfg['reserve_ram_gib']), total * float(cfg.get('reserve_ram_fraction',0.0)))
    required = working + reserve
    return dict(available_gib=free, weight_gib=weight, estimated_working_gib=working,
                estimated_cpu_weights_gib=(host_weight if model.get('backend') != 'persona_peft' else weight),
                estimated_gpu_weights_gib=(gpu_weight if model.get('backend') != 'persona_peft' else 0.0),
                reserve_gib=reserve, required_gib=required,
                fits=free >= required, cooperative=cfg['cooperative'], threads=cfg['threads'])


def guard_load(model, context=2048, mode='Auto', manual=0):
    plan = load_budget(model, context, mode, manual)
    if plan['cooperative'] and not plan['fits']:
        raise MemoryError('Deferred to protect other workloads: approximately '
                          f"{plan['required_gib']:.1f} GiB free RAM required including reserve; "
                          f"{plan['available_gib']:.1f} GiB available. No other process was stopped.")
    if plan['cooperative']:
        try:
            with ModelLease():pass
        except RuntimeError as exc:
            raise MemoryError('Deferred: another Studio model worker owns the cooperative inference slot. No process was stopped.') from exc
    return plan


def checkpoint():
    cfg = settings()
    vm = psutil.virtual_memory()
    free = vm.available / 2**30
    raw_total = getattr(vm,'total',0)
    total = (float(raw_total)/2**30) if isinstance(raw_total,(int,float)) else 0.0
    floor = max(float(cfg['checkpoint_floor_gib']), total*float(cfg.get('checkpoint_floor_fraction',0.0)))
    if cfg['cooperative'] and free < floor:
        raise MemoryError('Workshop yielded at an example/token boundary because free RAM fell below '
                          f"{floor:.1f} GiB. Resume later; completed preparation chunks are retained.")
    if cfg['cooperative'] and cfg['delay_per_example']:
        time.sleep(min(.5, float(cfg['delay_per_example'])))

class ModelLease:
    """OS-released lock shared by workshop workers; unrelated applications are untouched."""
    def __init__(self):
        self.file=None
        path=ROOT/'logs'/'one-model.lock';path.parent.mkdir(exist_ok=True)
        handle=open(path,'a+b')
        if handle.tell()==0:handle.write(b'0');handle.flush()
        handle.seek(0)
        try:
            if os.name=='nt':
                import msvcrt
                msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except (OSError,BlockingIOError):
            handle.close();raise RuntimeError('Another Studio model worker owns the cooperative inference slot. Wait for its test or chat to finish and unload before loading another model.')
        self.file=handle
    def close(self):
        if self.file is not None:self.file.close();self.file=None
    def __enter__(self):return self
    def __exit__(self,*args):self.close()

def claim_worker_slot():
    import sys,atexit
    global _WORKER_LEASE
    if settings()['cooperative'] and Path(sys.argv[0]).name in ('model_worker.py','persona_worker.py'):
        _WORKER_LEASE=ModelLease();atexit.register(_WORKER_LEASE.close)
