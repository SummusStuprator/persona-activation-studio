"""Token sampling with recorded, reproducible settings."""
import numpy as np

def draw(logits,rng,history=(),temperature=0.,top_k=40,top_p=1.,presence_penalty=0.):
    raw=np.asarray(logits,dtype=np.float64)
    if not np.isfinite(raw).all():raise ValueError('Non-finite logits.')
    if not 0<=temperature<=2 or not 0<top_p<=1 or not 0<=presence_penalty<=2:
        raise ValueError('Invalid sampling settings.')
    if not isinstance(top_k,(int,np.integer)) or top_k<0:
        raise ValueError('Invalid top-k count.')
    p=np.exp(raw-raw.max());p/=p.sum()
    adjusted=raw.copy()
    if history and presence_penalty:adjusted[np.unique(history)]-=presence_penalty
    if temperature==0:
        token=int(np.argmax(adjusted));return p,token,1.
    k=min(top_k or len(raw),len(raw))
    ids=np.argpartition(adjusted,-k)[-k:]
    ids=ids[np.argsort(adjusted[ids])[::-1]]
    q=np.exp((adjusted[ids]-adjusted[ids].max())/temperature);q/=q.sum()
    keep=np.cumsum(q)-q<top_p
    ids=ids[keep];q=q[keep];q/=q.sum()
    selected=int(rng.choice(len(ids),p=q))
    return p,int(ids[selected]),float(q[selected])

def qwen_defaults(thinking=True):
    return dict(temperature=1. if thinking else .7,top_p=.95 if thinking else .8,
                top_k=20,presence_penalty=1.5)

VERSION='numpy-topk-topp-presence-v1'
SOURCE='https://huggingface.co/Qwen/Qwen3.5-35B-A3B'
