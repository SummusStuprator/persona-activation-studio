"""Local jobs with durable outcomes and PID-reuse-safe liveness checks."""
from pathlib import Path
from contextlib import contextmanager
import json
import os
import re
import subprocess
import sys
import time
import uuid
import psutil


def write_state(path, data):
    path = Path(path)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    temporary.write_text(json.dumps(data, indent=2), encoding='utf-8')
    os.replace(temporary, path)


@contextmanager
def launch_lock(folder):
    with (Path(folder) / 'launch.lock').open('a+b') as handle:
        if handle.tell() == 0:
            handle.write(b'0'); handle.flush()
        handle.seek(0)
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def read_state(path):
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    state = data.get('status', 'unknown')
    if state in ('succeeded', 'failed', 'interrupted'):
        return data
    alive = False
    try:
        proc = psutil.Process(int(data['pid']))
        alive = abs(proc.create_time() - float(data['process_started'])) < .1
    except (KeyError, TypeError, ValueError, psutil.Error):
        pass
    grace = state == 'launching' and time.time() - data.get('started', 0) < 30
    data['status'] = state if alive or grace else 'interrupted'
    return data


def launch(root, folder, kind, args):
    root, folder = Path(root), Path(folder)
    if not re.fullmatch(r'[A-Za-z0-9_.-]+', kind):
        raise ValueError('Job kind must be a plain name.')
    folder.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, '-m', 'studio_cli', *map(str, args)]
    with launch_lock(folder):
        for path in folder.glob('*.json'):
            try:
                previous = read_state(path)
                if previous.get('command') == command and previous['status'] in ('running', 'launching'):
                    return previous.get('pid'), Path(previous['log'])
            except (OSError, ValueError, KeyError):
                continue
        stamp = time.strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:8]
        meta = folder / f'{stamp}-{kind}.json'
        log = meta.with_suffix('.log')
        data = dict(kind=kind, command=command, cwd=str(root), log=str(log),
                    started=time.time(), status='launching', pid=None)
        write_state(meta, data)
        try:
            with log.open('w', encoding='utf-8') as output:
                proc = subprocess.Popen([sys.executable, '-m', 'studio_jobs', str(meta)],
                    cwd=root, stdout=output, stderr=subprocess.STDOUT,
                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        except Exception as exc:
            write_state(meta, dict(data, status='failed', error=str(exc), finished=time.time()))
            raise
        return proc.pid, log


def run(path):
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    data.update(status='running', pid=os.getpid(), process_started=psutil.Process().create_time())
    write_state(path, data)
    code = 1
    try:
        code = subprocess.call(data['command'], cwd=data['cwd'])
    except BaseException as exc:
        data['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        data.update(status='succeeded' if code == 0 else 'failed',
                    return_code=code, finished=time.time())
        write_state(path, data)
    return code


def log_tail(path, limit=16000):
    with Path(path).open('rb') as handle:
        handle.seek(0, 2)
        handle.seek(max(0, handle.tell() - limit))
        return handle.read(limit).decode('utf-8', errors='replace')


if __name__ == '__main__':
    raise SystemExit(run(sys.argv[1]))
