import json, numpy as np
from isolated_engine import Engine
from model_store import grouped_inventory
models,_=grouped_inventory()
m=next(x for x in models if x['name']=='llama3.2:1b' or 'llama3.2:1b' in x.get('aliases',[]))
e=Engine();out={}
try:
    e.open(m,context=512,mode='Auto')
    ids=e.tokenize(e.chat([{'role':'user','content':'Reply with only 56.'}]))
    e.reset();e.evaluate(ids);logits=e.logits();assert np.isfinite(logits).all(), 'Non-finite native logits';out={'model':e.model['name'],'layers':e.layers,'dim':e.dim,'abi':e.abi,'finite':bool(np.isfinite(logits).all()),'plan':e.load_plan,'status':'passed'}
finally:e.close()
print(json.dumps(out,indent=2))