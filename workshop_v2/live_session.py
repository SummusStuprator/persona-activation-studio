"""Thread-safe generation jobs. Worker threads never call Streamlit APIs."""
from __future__ import annotations
import copy
import threading
import time
import uuid
from .steering_controls import validate
from .live_engine import stream

class Controller:
    def __init__(self,engine,bank,controls=(),phase='all',dose_limit=.5,lease_seconds=0):
        self.engine=engine;self.bank=bank;self.cv=threading.Condition()
        self._stop=False;self._paused=False;self.lease=lease_seconds;self.last_seen=time.monotonic()
        self.value={'revision':-1,'controls':[],'phase':phase,'dose_limit':dose_limit}
        self.requests=[];self.pause_at=None
        self.update(controls,phase,dose_limit)
    def update(self,controls,phase='all',dose_limit=.5):
        # Validation is pure array work: no native calls and no decode lock.
        validate(self.engine,self.bank,controls,dose_limit)
        if phase not in ('all','reasoning','answer'):raise ValueError('Invalid phase.')
        with self.cv:
            if self._stop:raise ValueError('This generation is stopped.')
            value={'controls':copy.deepcopy(list(controls)),'phase':phase,'dose_limit':float(dose_limit)}
            if self.value['revision']>=0 and all(self.value.get(k)==v for k,v in value.items()):return self.value['revision']
            value.update(revision=self.value['revision']+1,requested_at=time.time())
            self.value=value;self.requests.append(copy.deepcopy(value));self.cv.notify_all()
            return value['revision']
    def heartbeat(self):
        with self.cv:self.last_seen=time.monotonic()
    def before_decode(self,step,phase):
        with self.cv:
            if self.pause_at is not None and step>=self.pause_at:
                self._paused=True;self.pause_at=None
            while self._paused and not self._stop:
                if self.lease and time.monotonic()-self.last_seen>self.lease:self._stop=True;break
                self.cv.wait(.2)
            if self.lease and time.monotonic()-self.last_seen>self.lease:self._stop=True
            return None if self._stop else copy.deepcopy(self.value)
    def pause(self):
        with self.cv:self._paused=True
    def resume(self):
        with self.cv:self._paused=False;self.cv.notify_all()
    def stop(self):
        with self.cv:self._stop=True;self.cv.notify_all()
    def cancelled(self):
        with self.cv:return self._stop
    @property
    def paused(self):
        with self.cv:return self._paused

class LiveJob:
    def __init__(self,engine,bank,owner,prompt,controls,phase,dose_limit,settings,watch,lease=120):
        self.id=uuid.uuid4().hex;self.owner=owner;self.engine=engine;self.bank=bank
        self.controller=Controller(engine,bank,controls,phase,dose_limit,lease)
        self.model_identity=(engine.model["digest"],engine.abi,engine.process.pid)
        self.prompt=prompt;self.settings=dict(settings);self.watch=list(watch)
        self.lock=threading.Lock();self.state='queued';self.result=None;self.error=None
        self.output={'text':'','answer_text':'','reasoning_text':'','phase':'answer'}
        self.traces=[];self.profiles=[];self.started=time.time();self.thread=None
    @property
    def busy(self):return self.state in ('queued','running')
    def start(self):
        self.thread=threading.Thread(target=self._run,name='NativeChat-'+self.id[:8],daemon=True)
        self.thread.start();return self
    def _run(self):
        with self.lock:self.state='running'
        generator=None
        try:
            generator=stream(self.engine,self.prompt,self.bank,self.controller,self.watch,**self.settings)
            for event in generator:
                with self.lock:
                    if event['type']=='token':
                        self.output={k:event[k] for k in ('text','answer_text','reasoning_text','phase')}
                        self.traces.append(event['trace']);self.profiles.append(event['profiles'])
                    else:self.result=event['result']
            with self.lock:self.state='finished'
        except Exception as exc:
            from .core import write_result
            with self.lock:
                self.error=type(exc).__name__+': '+str(exc);self.state='failed'
                failure={'job_id':self.id,'error':self.error,'prompt':self.prompt,
                         'text':self.output['text'],'trace':self.traces,'control_requests':self.controller.requests,
                         'model':self.engine.model,'settings':self.settings}
            try:write_result(failure,'live-error')
            except OSError:pass
        finally:
            if generator is not None:generator.close()
    def snapshot(self):
        with self.lock:
            return {'id':self.id,'state':self.state,'error':self.error,'result':self.result,
                    'output':dict(self.output),'trace':list(self.traces),'layer_profiles':list(self.profiles),
                    'requested_revision':self.controller.value['revision'],'paused':self.controller.paused}

class Manager:
    def __init__(self):self.lock=threading.Lock();self.job=None
    def start(self,engine,bank,owner,prompt,controls,phase,dose_limit,settings,watch):
        with self.lock:
            if self.job and self.job.busy:raise RuntimeError('A generation is already active. Stop it before starting another.')
            # Another native experiment must not be in progress either.
            if not engine.lock.acquire(blocking=False):raise RuntimeError('The model is busy with another experiment.')
            engine.lock.release()
            self.job=LiveJob(engine,bank,owner,prompt,controls,phase,dose_limit,settings,watch)
            return self.job.start()
    def active(self):
        with self.lock:return self.job if self.job and self.job.busy else None
