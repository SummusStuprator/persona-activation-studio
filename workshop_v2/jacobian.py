"""Restricted same-position finite-difference sensitivity; NOT a full J-lens."""
import hashlib, json, time
from pathlib import Path
import numpy as np
from scipy.optimize import nnls
from .core import identity, write_result
from studio_paths import data_root, ASSET_ROOT, CODE_ROOT
ROOT = data_root()
DEFAULT_CONTEXTS=['The guide explains a new topic to the class.','A traveler considers the next step of the journey.','The group discusses what happened yesterday.','Someone describes an ordinary afternoon at home.']

def vocabulary_ids(engine,words):
    prefix=engine.tokenize('Answer:'); kept=[]; rejected=[]
    for word in words:
        full=engine.tokenize('Answer: '+word.strip())
        if len(full)==len(prefix)+1 and np.array_equal(full[:-1],prefix):
            kept.append((word.strip(),int(full[-1])))
        else: rejected.append(word)
    # Different spellings may tokenize identically; do not duplicate a dictionary atom.
    unique={token:word for word,token in kept}
    return [(word,token) for token,word in unique.items()],rejected

def effect(engine,tokens,table=None,dose=0.):
    try:
        engine.clear_steering(); engine.reset()
        if len(tokens)>1: engine.evaluate(tokens[:-1])
        if table is not None: engine.steer(table,float(dose),'add')
        engine.evaluate(tokens[-1:])
        logits=engine.logits().astype(np.float64)
        return logits-logits.mean(),engine.capture(1)
    finally:
        engine.clear_steering(); engine.reset()

def calibrate(engine,bank,words,layer,rank=8,contexts=None,progress=None):
    if not 0<=layer<engine.layers or not 2<=rank<=32: raise ValueError('Choose an available layer and rank 2 to 32.')
    contexts=contexts or DEFAULT_CONTEXTS
    if not 3<=len(contexts)<=12: raise ValueError('Use 3 to 12 separate calibration prompts.')
    kept,rejected=vocabulary_ids(engine,words)
    if len(kept)<3: raise ValueError('At least three chosen words must be single tokens in this tokenizer.')
    candidates=np.array([t for _,t in kept]); rng=np.random.default_rng(701)
    columns=[a['directions'][layer] for a in bank.values() if not a.get('control_only')]
    columns=columns[:rank//2]+[rng.normal(size=engine.dim) for _ in range(rank)]
    basis=np.linalg.qr(np.stack(columns,axis=1))[0][:,:rank].astype(np.float32)
    derivatives=[]; anchors=[]; consistency=[]
    with engine.lock:
        for ci,text in enumerate(contexts):
            tokens=engine.tokenize(text)
            _,h=effect(engine,tokens); anchors.append(h[layer]); eps=max(float(np.linalg.norm(h[layer]))*.002,1e-3)
            partial=[]
            for j in range(rank):
                table=np.zeros((engine.layers,engine.dim),np.float32); table[layer]=basis[:,j]
                plus,_=effect(engine,tokens,table,eps); minus,_=effect(engine,tokens,table,-eps)
                d=(plus[candidates]-minus[candidates])/(2*eps); partial.append(d)
                if j==0:
                    p2,_=effect(engine,tokens,table,eps/2); m2,_=effect(engine,tokens,table,-eps/2)
                    d2=(p2[candidates]-m2[candidates])/eps
                    consistency.append(float(np.linalg.norm(d-d2)/max(np.linalg.norm(d2),1e-8)))
                if progress: progress(ci*rank+j+1,len(contexts)*rank,'Finite-difference output sensitivity')
            derivatives.append(partial)
    average=np.mean(derivatives,axis=0); directions=basis@average
    norms=np.linalg.norm(directions,axis=0); valid=norms>1e-7
    if valid.sum()<3: raise ValueError('Too few nonzero sensitivity directions.')
    directions=(directions[:,valid]/norms[valid]).astype(np.float32)
    words=[w for (w,t),v in zip(kept,valid) if v]; candidates=candidates[valid]
    anchor=np.array(anchors,np.float32).mean(0)
    result={**identity(engine),'layer':layer,'rank':rank,'words':words,'token_ids':candidates.tolist(),
            'rejected_multitoken':rejected,'contexts':contexts,'fd_relative_error':consistency,
            'method':'Same-position, small-corpus, low-rank finite differences of centered output logits. Includes context-dependent output normalization.',
            'limitations':'Not Anthropic J-space: no full residual Jacobian, no averaging over future positions, no thousand-prompt corpus, no vocabulary-wide frame. No consciousness inference.'}
    folder=ROOT/'jacobian'/engine.model['digest']; folder.mkdir(parents=True,exist_ok=True)
    target=folder/f'layer-{layer}-rank-{rank}.npz'
    np.savez_compressed(target,directions=directions,anchor=anchor,basis=basis,derivatives=average)
    result['arrays_sha256']=hashlib.sha256(target.read_bytes()).hexdigest()
    target.with_suffix('.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return dict(result,directions=directions,anchor=anchor,saved_to=str(target))

def read(engine,text,lens,k=4):
    if lens['digest']!=engine.model['digest'] or lens['abi']!=engine.abi: raise ValueError('Lens belongs to a different checkpoint/runtime.')
    h=engine.extract(text)[lens['layer']]-lens['anchor']; d=lens['directions']
    scores=h@d
    selection=np.argsort(scores)[-min(k,len(scores)):][::-1]
    weights,_=nnls(d[:,selection].astype(np.float64),h.astype(np.float64))
    residual=h-d[:,selection]@weights
    rows=[{'word':lens['words'][int(i)],'local_projection':float(scores[i]),'nonnegative_coefficient':float(w)} for i,w in zip(selection,weights)]
    return {'rows':rows,'reconstruction_fraction':float(1-(residual@residual)/max(h@h,1e-12)),
            'warning':'Sparse fit in a restricted local sensitivity dictionary; not a report of hidden thoughts.'}

def intervene(engine,text,lens,word,dose=.01):
    if lens['digest']!=engine.model['digest'] or lens['abi']!=engine.abi: raise ValueError('Lens identity mismatch.')
    if not -.1<=dose<=.1: raise ValueError('Sensitivity experiment dose must be within 0.10.')
    tokens=engine.tokenize(text); layer=lens['layer']
    with engine.lock:
        base,h=effect(engine,tokens); table=np.zeros_like(h)
        table[layer]=lens['directions'][:,lens['words'].index(word)]*np.linalg.norm(h[layer])
        changed,_=effect(engine,tokens,table,dose); restored,_=effect(engine,tokens)
    delta=changed-base; ids=np.argsort(delta)[-10:][::-1]
    result={**identity(engine),'word':word,'dose':dose,'layer':layer,'reset_error':float(np.max(abs(restored-base))),
            'logit_changes':[{'token':engine.piece(int(i)).decode('utf-8','replace'),'change':float(delta[i])} for i in ids]}
    result['saved_to']=write_result(result,'sensitivity-intervention'); return result
