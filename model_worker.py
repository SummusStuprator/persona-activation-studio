"""Standalone worker: authenticated authenticated local IPC, no network listener."""
import os
import sys
from multiprocessing.connection import Listener
from workshop_v2.resource_policy import configure_process
configure_process()
from integrity import verify_release
verify_release(native=True)
from backend import Engine
METHODS = {'open','close','reset','clear_steering','steer','tokenize','evaluate',
           'capture','chat','logits','piece','is_eog','extract','generate'}
METHODS.update({'mean_transport','unembed_residual','read_positions','extract_response','score_response'})

def main():
    key = bytes.fromhex(os.environ.pop('ACTIVATION_LAB_WORKER_KEY'))
    runtime = os.environ.get('ACTIVATION_LAB_RUNTIME')
    with Listener(sys.argv[1], family=os.environ.get('STUDIO_WORKER_FAMILY','AF_PIPE'), authkey=key) as server:
        connection = server.accept()
        engine = Engine(runtime=runtime)
        try:
            while True:
                method, args, kwargs = connection.recv()
                if method == 'shutdown': break
                try:
                    if method not in METHODS: raise ValueError('Unknown worker method.')
                    result = getattr(engine,method)(*args,**kwargs)
                    state = {k:getattr(engine,k,None) for k in
                             ('model','dim','layers','vocab_size','context','cuda_available','abi')}
                    state['loaded'] = bool(engine.handle)
                    connection.send({'ok':True,'value':result,'state':state})
                except Exception as exc:
                    connection.send({'ok':False,'error':type(exc).__name__+': '+str(exc)})
        except (EOFError, BrokenPipeError):
            pass
        finally:
            engine.close()
            connection.close()

if __name__ == '__main__':
    main()
