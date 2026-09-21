"""Explicit local Ollama multimodal transport. No fabricated activation telemetry."""
from pathlib import Path
import base64,hashlib,io,json,time
import requests
ROOT=Path(__file__).resolve().parent.parent
HOST='http://127.0.0.1:11434'

def request(path,body=None,timeout=15):
    session=requests.Session();session.trust_env=False
    try:
        r=session.get(HOST+path,timeout=timeout) if body is None else session.post(HOST+path,json=body,timeout=timeout)
        r.raise_for_status();return r.json()
    finally:session.close()

def inventory():return request('/api/tags').get('models',[])

def describe(name):
    value=request('/api/show',{'model':name})
    if value.get('remote_host') or value.get('remote_model'):raise ValueError('Cloud models are not enabled. Use a locally installed model.')
    return value

def save_image(data):
    from PIL import Image,ImageOps
    if len(data)>8*1024*1024:raise ValueError('Each image must be smaller than 8 MiB.')
    with Image.open(io.BytesIO(data)) as image:
        if image.width*image.height>16000000:raise ValueError('Image exceeds 16 megapixels.')
        image=ImageOps.exif_transpose(image).convert('RGB');image.thumbnail((1536,1536))
        buff=io.BytesIO();image.save(buff,format='PNG');raw=buff.getvalue()
        name=hashlib.sha256(raw).hexdigest()+'.png';folder=ROOT/'uploads';folder.mkdir(exist_ok=True)
        path=folder/name
        if not path.exists():path.write_bytes(raw)
        return {'file':name,'sha256':hashlib.sha256(raw).hexdigest(),'width':image.width,'height':image.height,'format':'PNG','preprocessing':'EXIF orientation applied, metadata removed, longest edge capped at 1536 pixels'}

def image_bytes(image):
    name=image.get('file','')
    if Path(name).name!=name or not name.endswith('.png'):raise ValueError('Invalid stored image reference.')
    data=(ROOT/'uploads'/name).read_bytes()
    if hashlib.sha256(data).hexdigest()!=image['sha256']:raise ValueError('Stored image changed.')
    return data

def wire_messages(messages):
    clean=[]
    for m in messages:
        x={k:m[k] for k in ('role','content','tool_calls','tool_name') if k in m}
        if m.get('reasoning'):x['thinking']=m['reasoning']
        if m.get('images'):x['images']=[base64.b64encode(image_bytes(i)).decode('ascii') for i in m['images']]
        clean.append(x)
    return clean

def stream_chat(name,messages,max_tokens=256,thinking=False,tools=False):
    from .web_search_tool import TOOL
    from .resource_policy import guard_load
    details=describe(name);caps=details.get('capabilities',[])
    if any(m.get('images') for m in messages) and 'vision' not in caps:raise ValueError('This model does not advertise image support. Images were not discarded or turned into a text transcript.')
    if tools and 'tools' not in caps:raise ValueError('This model does not advertise native tool calling.')
    loaded=request('/api/ps').get('models',[])
    if loaded:raise RuntimeError('Ollama already has a model resident. Cooperative mode will not reload or evict it; wait for that workload to release its model.')
    size=next((m['size'] for m in inventory() if m['name']==name),None)
    if size is None:raise ValueError('Local model tag disappeared.')
    import psutil
    from .resource_policy import settings
    policy=settings();need=size/2**30+policy['reserve_ram_gib']+.75
    if psutil.virtual_memory().available/2**30<need:raise MemoryError(f'Image/tool chat deferred: approximately {need:.1f} GiB RAM required including reserve.')
    if not 16<=int(max_tokens)<=1024:raise ValueError('Use 16 to 1024 output tokens.')
    body={'model':name,'messages':wire_messages(messages),'stream':True,'keep_alive':0,
          'options':{'num_gpu':0,'num_thread':policy['threads'],'num_ctx':4096,'num_predict':int(max_tokens),'temperature':0.}}
    if 'thinking' in caps:body['think']=bool(thinking)
    if tools:body['tools']=[TOOL]
    session=requests.Session();session.trust_env=False
    output='';reasoning='';calls=[];started=time.time();done=None
    try:
        with session.post(HOST+'/api/chat',json=body,stream=True,timeout=(10,90)) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if time.time()-started>300:raise TimeoutError('Local multimodal generation exceeded its five-minute bound.')
                if not line:continue
                item=json.loads(line)
                if item.get('error'):raise RuntimeError(item['error'])
                message=item.get('message',{});output+=message.get('content','');reasoning+=message.get('thinking','');calls+=message.get('tool_calls',[])
                yield {'type':'token','answer_text':output,'reasoning_text':reasoning}
                if item.get('done'):done=item;break
    finally:session.close()
    if done is None:raise RuntimeError('Ollama stream ended without a completion record.')
    result={'model':name,'messages':messages,'text':output,'answer_text':output,'reasoning_text':reasoning,'tool_calls':calls,'stop_reason':done.get('done_reason','unknown'),'answer_complete':bool(output.strip()) and done.get('done_reason')=='stop','seconds':round(time.time()-started,3),'backend':'local_ollama_multimodal_tools','activations':None,'steering_applied':False,'options':body['options'],'token_count':done.get('eval_count'),'capabilities':caps}
    from .core import write_result
    result['saved_to']=write_result(result,'ollama-chat')
    yield {'type':'done','result':result}

_raw_stream_chat=stream_chat

def stream_chat(*args,**kwargs):
    """Share the native worker's cooperative slot during multimodal inference too."""
    from .resource_policy import ModelLease
    with ModelLease():
        yield from _raw_stream_chat(*args,**kwargs)

_raw_describe=describe

def describe(name):
    value=_raw_describe(name)
    from model_store import grouped_inventory
    rows=grouped_inventory()[1]
    matches=[r for r in rows if r['name']==name and r.get('path') and Path(r['path']).is_file()]
    if not matches:raise ValueError('No existing local GGUF file was found for this Ollama tag. Cloud aliases are not used for image/tool chat.')
    with open(matches[0]['path'],'rb') as f:
        if f.read(4)!=b'GGUF':raise ValueError('Selected local model is not a GGUF checkpoint.')
    return value
