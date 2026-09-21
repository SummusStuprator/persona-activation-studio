"""Sparse nonnegative vocabulary frame, distinct from a singular-vector subspace.

Read only the small F32 normalization tensor from the existing GGUF. Obtain
W_U times transported columns through the native runtime, cancelling RMSNorm's
learned gain. No model head or model-weight copy is saved. The cached vocabulary
frame is a derived analysis artifact, and may be large for full-width models.
"""
from pathlib import Path
import struct,json,hashlib
import numpy as np
from scipy.optimize import nnls
from threadpoolctl import threadpool_limits
from .core import write_result,identity


def norm_gain(engine):
    if engine.model.get('architecture',engine.model.get('family')) not in ('llama','qwen2','qwen3','qwen35','qwen35moe','nanbeige'):
        raise ValueError('Sparse-frame head recovery is validated for standard RMSNorm decoders only; vocabulary readout remains available.')
    path=Path(engine.model['path']);size=path.stat().st_size
    with path.open('rb',buffering=1024*1024) as f:
        def num(fmt):
            n=struct.calcsize('<'+fmt);b=f.read(n)
            if len(b)!=n:raise ValueError('Truncated GGUF header.')
            return struct.unpack('<'+fmt,b)[0]
        def string(keep=True):
            n=num('Q')
            if n>size-f.tell():raise ValueError('Invalid GGUF string.')
            if keep:return f.read(n).decode('utf-8','replace')
            f.seek(n,1)
        fmts={0:'B',1:'b',2:'H',3:'h',4:'I',5:'i',6:'f',7:'?',10:'Q',11:'q',12:'d'}
        def value(kind):
            if kind in fmts:return num(fmts[kind])
            if kind==8:string(False);return None
            if kind!=9:raise ValueError('Unsupported GGUF metadata.')
            sub,count=num('I'),num('Q')
            if count>10000000:raise ValueError('Invalid GGUF array.')
            if sub in fmts:f.seek(struct.calcsize('<'+fmts[sub])*count,1)
            else:
                for _ in range(count):value(sub)
        if f.read(4)!=b'GGUF' or num('I') not in (2,3):raise ValueError('Expected little-endian GGUF.')
        nt,nk=num('Q'),num('Q');align=32
        for _ in range(nk):
            key=string();v=value(num('I'))
            if key=='general.alignment':align=int(v)
        found=None
        for _ in range(nt):
            name=string();rank=num('I');shape=[num('Q') for _ in range(rank)];kind=num('I');offset=num('Q')
            if name=='output_norm.weight':found=(shape,kind,offset)
        if found is None or found[1]!=0 or np.prod(found[0])!=engine.dim:raise ValueError('Expected a small float32 output RMSNorm gain tensor.')
        start=(f.tell()+align-1)//align*align;f.seek(start+found[2]);gain=np.frombuffer(f.read(4*engine.dim),dtype='<f4').copy()
    if not np.isfinite(gain).all() or np.any(abs(gain)<1e-8):raise ValueError('Cannot invert output normalization gain.')
    return gain


def head_projector(engine):
    gain=norm_gain(engine);zero=engine.unembed_residual(np.zeros(engine.dim,np.float32))['logits'].astype(np.float64)
    def project(v):
        x=np.asarray(v,np.float32)/gain;n=float(np.linalg.norm(x))
        if n<1e-12:return np.zeros(engine.vocab_size,np.float32)
        scores=engine.unembed_residual(np.ascontiguousarray(x/n,np.float32))['logits'].astype(np.float64)
        # Common sqrt(d)-like scale is harmless for normalized frame vectors/rank.
        return ((scores-zero)*n).astype(np.float32)
    rng=np.random.default_rng(864);a=rng.normal(size=engine.dim).astype(np.float32);b=rng.normal(size=engine.dim).astype(np.float32)
    lhs=project(a+b);rhs=project(a)+project(b)
    error=float(np.linalg.norm(lhs-rhs)/max(np.linalg.norm(rhs),1e-10))
    if error>.002:raise ValueError(f'Native head linearity check failed ({error:.4g}); refusing a misleading W_U frame.')
    return project,error


def build(engine,lens,progress=None):
    from .jspace import check_quality
    check_quality(lens)
    if lens['digest']!=engine.model['digest'] or lens['abi']!=engine.abi:raise ValueError('Lens identity mismatch.')
    path=Path(lens['saved_to']).with_name(Path(lens['saved_to']).stem+'-frame.npz')
    if path.exists():
        meta=json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
        if meta.get('head_source') and meta.get('lens_sha256')==lens['arrays_sha256'] and hashlib.sha256(path.read_bytes()).hexdigest()==meta.get('arrays_sha256'):
            with np.load(path,allow_pickle=False) as d:return dict(meta,frame=d['frame'],basis=lens['basis'])
    estimate=engine.vocab_size*lens['rank']*4
    if estimate>2*1024**3:raise ValueError('Derived frame exceeds 2 GiB. Use a lower rank or a smaller checkpoint; no weights were duplicated.')
    from .gguf_head import project as tiled_project
    from .jspace import check_quality
    check_quality(lens)
    frame,head_meta=tiled_project(engine,lens['transport'],progress)
    # Tiled W_U multiplication avoids input-quantization nonlinearity in native GEMV.
    error=None
    norms=np.linalg.norm(frame,axis=1,keepdims=True);frame/=np.maximum(norms,1e-12)
    np.savez_compressed(path,frame=frame)
    meta={**identity(engine),'layer':lens['layer'],'rank':lens['rank'],'vocabulary_size':engine.vocab_size,
          'lens_sha256':lens['arrays_sha256'],'linearity_error':error,'head_source':head_meta,'saved_to':str(path),
          'arrays_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
          'method':'Unit rows of W_U J in the fitted input basis. Read-only GGUF head tiles; no native GEMV linearity assumption. Sparse nonnegative orthogonal matching pursuit with NNLS refits.',
          'limits':'Finite-difference corpus lens and approximate greedy sparse optimization; not an exact global nearest-point solver.'}
    path.with_suffix('.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    return dict(meta,frame=frame,basis=lens['basis'])


def decompose(engine,text,lens,frame,k=12):
    from .jspace import check_quality
    check_quality(lens)
    if frame['digest']!=engine.model['digest'] or frame['abi']!=engine.abi or frame['lens_sha256']!=lens['arrays_sha256']:raise ValueError('Frame identity mismatch.')
    if not 1<=k<=25:raise ValueError('Use sparsity 1 to 25.')
    h=engine.extract(text)[lens['layer']].astype(np.float64);basis=frame['basis'];atoms=frame['frame'];target=basis.T@h
    selected=[];weights=np.zeros(0);residual=target.copy()
    with threadpool_limits(limits=4):
        for _ in range(k):
            correlations=atoms@residual
            if selected:correlations[selected]=-np.inf
            idx=int(np.argmax(correlations))
            if correlations[idx]<=1e-8:break
            selected.append(idx);weights=nnls(atoms[selected].T.astype(np.float64),target,maxiter=300)[0]
            residual=target-atoms[selected].T@weights
    reconstruction=basis@(atoms[selected].T@weights) if selected else np.zeros_like(h)
    fraction=float(1-np.sum((h-reconstruction)**2)/max(float(h@h),1e-12))
    rows=[{'token_id':idx,'token':engine.piece(idx).decode('utf-8','replace'),'coefficient':float(w)} for idx,w in zip(selected,weights) if w>1e-9]
    result={**identity(engine),'layer':lens['layer'],'k':k,'rows':rows,'reconstruction_fraction':fraction,'frame':frame['saved_to'],
            'scope':'Sparse nonnegative vocabulary-frame fit. This is not an SVD subspace and not an exhaustive account of reasoning.'}
    result['saved_to']=write_result(result,'jspace-sparse');return result


def swap(engine,text,lens,frame,source_id,target_id):
    from .jspace import check_quality
    check_quality(lens)
    if frame['digest']!=engine.model['digest'] or frame['abi']!=engine.abi:raise ValueError('Frame mismatch.')
    if not 0<=source_id<engine.vocab_size or not 0<=target_id<engine.vocab_size or source_id==target_id:raise ValueError('Select two distinct vocabulary IDs.')
    vectors=frame['basis']@frame['frame'][[source_id,target_id]].T
    if np.linalg.matrix_rank(vectors)<2:raise ValueError('Selected frame vectors are degenerate.')
    tokens=engine.tokenize(text);layer=lens['layer']
    with engine.lock:
        try:
            engine.clear_steering();engine.reset();engine.evaluate(tokens);h=engine.capture()[layer];base=engine.logits()
            coordinates=np.linalg.pinv(vectors)@h;delta=vectors@(coordinates[::-1]-coordinates)
            if np.linalg.norm(delta)>.5*np.linalg.norm(h):raise ValueError('Swap would exceed a 50% residual perturbation. Choose different tokens.')
            engine.reset()
            if len(tokens)>1:engine.evaluate(tokens[:-1])
            table=np.zeros((engine.layers,engine.dim),np.float32);table[layer]=delta
            engine.steer(table,1.,'add');engine.evaluate(tokens[-1:]);after=engine.capture(1)[layer];changed=engine.logits()
            error=float(np.max(abs(np.linalg.pinv(vectors)@after-coordinates[::-1])))
            engine.clear_steering();engine.reset();engine.evaluate(tokens);reset=float(np.max(abs(engine.logits()-base)))
            diff=changed-base;ids=np.argsort(diff)[-10:][::-1]
            result={**identity(engine),'layer':layer,'source_id':source_id,'target_id':target_id,'coordinate_error':error,'reset_error':reset,
                    'changes':[{'token':engine.piece(int(i)).decode('utf-8','replace'),'delta':float(diff[i])} for i in ids],
                    'scope':'Single-position two-frame-coordinate swap; orthogonal complement left unchanged. Not all-layer intervention.'}
            result['saved_to']=write_result(result,'jspace-coordinate-swap');return result
        finally:engine.clear_steering();engine.reset()
