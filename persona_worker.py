"""Persona backend in the existing ML environment; authenticated local IPC, local files only."""
import os,sys
from multiprocessing.connection import Listener
for key in ('HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','HF_DATASETS_OFFLINE','HF_HUB_DISABLE_TELEMETRY'):
    os.environ[key]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['TOKENIZERS_PARALLELISM']='false'
from workshop_v2.resource_policy import configure_process
configure_process()
from integrity import verify_release
verify_release()
# Bind before heavyweight imports so the client can distinguish slow import from failure.
METHODS={'open','close','reset','clear_steering','steer','tokenize','evaluate','capture','chat',
         'logits','piece','is_eog','extract','extract_response','score_response'}
def main():
    auth=bytes.fromhex(os.environ.pop('ACTIVATION_LAB_WORKER_KEY'))
    with Listener(sys.argv[1],family=os.environ.get('STUDIO_WORKER_FAMILY','AF_PIPE'),authkey=auth) as server:
        conn=server.accept()
        from workshop_v2.persona_backend import Engine
        engine=Engine()
        try:
            while True:
                method,args,kwargs=conn.recv()
                if method=='shutdown':break
                try:
                    if method not in METHODS:raise ValueError('Persona backend does not implement '+method)
                    value=getattr(engine,method)(*args,**kwargs)
                    if method=='open':
                        import torch
                        torch.set_num_threads(int(os.environ.get('WORKSHOP_CPU_THREADS','2')))
                    state={k:getattr(engine,k,None) for k in ('model','dim','layers','vocab_size','context','cuda_available','abi')}
                    state['loaded']=bool(engine.handle)
                    conn.send({'ok':True,'value':value,'state':state})
                except Exception as exc:
                    import traceback;traceback.print_exc()
                    conn.send({'ok':False,'error':type(exc).__name__+': '+str(exc)})
        except (EOFError,BrokenPipeError):pass
        finally:engine.close();conn.close()
if __name__=='__main__':main()
