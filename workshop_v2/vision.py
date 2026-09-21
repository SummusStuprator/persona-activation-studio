"""Optional local-Ollama image output. Never pretends to capture vision activations."""
import base64, json, urllib.request
from .core import identity, write_result
ENDPOINT='http://127.0.0.1:11434'
def request(path,body):
    req=urllib.request.Request(ENDPOINT+path,json.dumps(body).encode(),{'Content-Type':'application/json'})
    return urllib.request.urlopen(req,timeout=180)
def generate(model,prompt,image,mime):
    if model.get('reason') or not model.get('path'): raise ValueError('Select a local GGUF model.')
    if mime not in ('image/png','image/jpeg','image/webp') or len(image)>8*1024*1024: raise ValueError('Use a PNG/JPEG/WebP image under 8 MiB.')
    with request('/api/show',{'model':model['name']}) as reply: details=json.load(reply)
    if 'vision' not in details.get('capabilities',[]): raise ValueError('Ollama does not advertise vision for this checkpoint.')
    if details.get('remote_host') or details.get('remote_model'): raise ValueError('Cloud inference is not allowed here.')
    with request('/api/chat',{'model':model['name'],'stream':False,'keep_alive':0,
          'messages':[{'role':'user','content':prompt,'images':[base64.b64encode(image).decode()]}],
          'options':{'num_ctx':2048,'num_predict':256,'temperature':0}}) as reply:
        result=json.load(reply)
    out={'model':model['name'],'text':result.get('message',{}).get('content',''),
         'mode':'Ollama image baseline','internal_activations':None,'steering':None,
         'warning':'This is a separate multimodal output path, not the instrumented native decoder.'}
    out['saved_to']=write_result(out,'vision-baseline'); return out
