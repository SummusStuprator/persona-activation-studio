"""Section3.3 vocabulary projection using original GGUF tiles, no copied head."""
from pathlib import Path
import numpy as np
from .gguf_head import project
from .core import write_result,identity

def run(engine,fitted,top=60,progress=None):
    if fitted['digest']!=engine.model['digest'] or fitted['abi']!=engine.abi:raise ValueError('Paper-vector identity mismatch.')
    layer=fitted['layer'];vectors=np.stack([fitted['vectors'][n][layer] for n in ('S1_1P','S2_1P')],axis=1)
    vectors/=np.maximum(np.linalg.norm(vectors,axis=0,keepdims=True),1e-12)
    scores,metadata=project(engine,vectors,progress);rows=[]
    for col,axis in enumerate(('S1','S2')):
        for end,order in [('promoted',np.argsort(scores[:,col])[-top:][::-1]),('suppressed',np.argsort(scores[:,col])[:top])]:
            for rank,token in enumerate(order):
                rows.append({'axis':axis,'end':end,'rank':rank+1,'token_id':int(token),'token':engine.piece(int(token)).decode('utf-8','replace'),'score':float(scores[token,col])})
    result={**identity(engine),'layer':layer,'rows':rows,'head':metadata,
            'scope':'Dot product of unit S1/S2 directions with all rows of the stored GGUF unembedding; no normalization, softmax or prompt-conditioned logits.'}
    result['saved_to']=write_result(result,'paper-unembedding')
    path=Path(result['saved_to']).with_suffix('.npz');np.savez_compressed(path,scores=scores)
    result['full_scores']=str(path);return result
