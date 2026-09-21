"""Small paired behavioral tests and direct next-token causal measurements."""
import numpy as np
from .core import run, write_result, identity
from .jacobian import effect

def dose_sweep(engine,bank,axis,prompts,layer,doses=(-.15,0.,.15),progress=None):
    if len(prompts)>6 or len(doses)>7: raise ValueError('Limit sweep to six prompts and seven doses.')
    rows=[]; control=bank[axis]
    for pi,text in enumerate(prompts):
        for dose in doses:
            result=run(engine,text,bank,controls=[{'axis':axis,'layer':layer,'dose':float(dose),'mode':'add'}],watch=[axis],max_tokens=48)
            rows.append({'prompt':text,'dose':float(dose),'axis':axis,'layer':layer,'text':result['text'],
                         'stop_reason':result['stop_reason'],'saved_to':result['saved_to']})
        # Match per-layer norm but use a predeclared random direction, not a new prompt.
        rng=np.random.default_rng(1024); random_axis=dict(control)
        vectors=rng.normal(size=control['directions'].shape).astype(np.float32)
        vectors/=np.linalg.norm(vectors,axis=1,keepdims=True)
        random_axis['directions']=vectors; random_axis['meta']=dict(control['meta'],name='random_sweep')
        random_bank=dict(bank,random_sweep=random_axis)
        result=run(engine,text,random_bank,controls=[{'axis':'random_sweep','layer':layer,'dose':.15,'mode':'add'}],watch=[axis],max_tokens=48)
        rows.append({'prompt':text,'dose':.15,'axis':'random norm-matched control','layer':layer,'text':result['text'],'saved_to':result['saved_to'],'stop_reason':result['stop_reason']})
        if progress: progress(pi+1,len(prompts),'Paired dose sweep')
    out={**identity(engine),'rows':rows,'caution':'Exploratory sweep. Do not tune on these prompts and then report them as independent validation.'}
    out['saved_to']=write_result(out,'dose-sweep'); return out

def causal_readout(engine,bank,axis,text,layer,dose=.03):
    from .core import validate_controls
    c={'axis':axis,'layer':layer,'mode':'add','dose':dose}
    table=validate_controls(engine,bank,[c]); tokens=engine.tokenize(text)
    with engine.lock:
        base,h=effect(engine,tokens); changed,post=effect(engine,tokens,table,1.); restored,_=effect(engine,tokens)
    difference=changed-base
    def rows(ids): return [{'token':engine.piece(int(i)).decode('utf-8','replace'),'logit_change':float(difference[i])} for i in ids]
    p=np.exp(base-base.max());p/=p.sum();q=np.exp(changed-changed.max());q/=q.sum()
    out={**identity(engine),'axis':axis,'layer':layer,'dose':dose,
         'up':rows(np.argsort(difference)[-12:][::-1]),'down':rows(np.argsort(difference)[:12]),
         'kl_baseline_to_intervention':float(np.sum(p*np.log((p+1e-300)/(q+1e-300)))),
         'reset_max_logit_error':float(np.max(abs(restored-base))),
         'caution':'Actual same-prompt causal next-token changes, not raw unembedding or a J-lens.'}
    out['saved_to']=write_result(out,'causal-readout');return out
