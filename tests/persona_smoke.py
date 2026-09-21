import json, os
from isolated_engine import Engine
from workshop_v2.persona_store import discover

target=os.environ.get('STUDIO_TEST_PERSONA','').strip()
if not target:
    print(json.dumps({'status':'skipped','reason':'Set STUDIO_TEST_PERSONA to a discovered persona name.'}))
    raise SystemExit(0)
models,_=discover()
m=next((x for x in models if x['name']==target or target in x.get('aliases',[])),None)
if m is None:raise SystemExit('Requested STUDIO_TEST_PERSONA was not discovered: '+target)
e=Engine();out={}
try:
    e.open(m,context=512,mode='CPU')
    prompt=e.chat([{'role':'user','content':'Reply with exactly: studio persona works'}])
    ids=e.tokenize(prompt);e.reset();e.evaluate(ids)
    import numpy as np
    out={'name':e.model['name'],'backend':e.model['backend'],'layers':e.layers,'dim':e.dim,'abi':e.abi,'finite_logits':bool(np.isfinite(e.logits()).all()),'status':'passed'}
finally:e.close()
print(json.dumps(out,indent=2))