"""Vocabulary-wide Jacobian lens from native GGUF residual transport.

The estimator matches the author code's source-mean/target-sum reduction.
Central finite differences are an explicit numerical approximation to gradients.
No candidate-word restriction; full-dimensional fitting is resumable.
"""
from pathlib import Path
import hashlib,json,time
import numpy as np
from .core import identity,write_result
from studio_paths import data_root, ASSET_ROOT, CODE_ROOT
ROOT = data_root()
DEFAULT_CORPUS=[
'The museum guide described how the objects in the room had been collected over many years. Visitors asked questions about the materials, the dates, and the people who made them. A notebook on the desk recorded each item in a careful list.',
'The travelers compared the routes on a map before leaving the station. One road followed the river, while another crossed the hills and passed several villages. They checked the timetable and discussed where to stop for lunch during the journey.']

def _atomic_arrays(path,**arrays):
    temp=path.with_suffix('.tmp.npz');np.savez_compressed(temp,**arrays);temp.replace(path)

def fit(engine,layer,contexts=None,rank=None,max_tokens=64,skip_first=16,epsilon_fraction=.1,progress=None,resume=True):
    layer=int(layer);dim=engine.dim;rank=dim if rank is None else int(rank)
    contexts=list(DEFAULT_CORPUS if contexts is None else contexts)
    if not 0<=layer<engine.layers or not 1<=rank<=dim:raise ValueError('Invalid layer/rank.')
    if not contexts or len(contexts)>1000:raise ValueError('Use 1 to 1000 calibration texts.')
    if not 0<epsilon_fraction<=.2:raise ValueError('Finite-difference step must be in (0, 0.2].')
    spec={**identity(engine),'layer':layer,'dim':dim,'rank':rank,'contexts':contexts,
          'max_tokens':int(max_tokens),'skip_first':int(skip_first),'epsilon_fraction':epsilon_fraction,
          'method_version':'source-mean-target-sum-central-fd-v2'}
    key=hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest()[:20]
    folder=ROOT/'jacobian'/engine.model['digest'];folder.mkdir(parents=True,exist_ok=True)
    checkpoint=folder/(key+'-checkpoint.npz');target=folder/(key+'-lens.npz')
    rng=np.random.default_rng(672)
    basis=np.eye(dim,dtype=np.float32) if rank==dim else np.linalg.qr(rng.normal(size=(dim,rank)))[0].astype(np.float32)
    sums=np.zeros((dim,rank),np.float64);done=0;checks=[];valid_counts=[]
    if resume and checkpoint.exists():
        with np.load(checkpoint,allow_pickle=False) as data:
            sums=data['sums'];done=int(data['done']);checks=data['checks'].tolist();valid_counts=data['valid_counts'].tolist()
    total=len(contexts)*rank;started=time.monotonic()
    with engine.lock:
        for ci,text in enumerate(contexts):
            if (ci+1)*rank<=done:continue
            tokens=engine.tokenize(text)[:max_tokens]
            base=engine.mean_transport(tokens,layer,skip_first=skip_first)
            eps=max(np.linalg.norm(base['source_mean'])*epsilon_fraction,.0001)
            if len(valid_counts)<=ci:valid_counts.append(base['n_valid'])
            for j in range(rank):
                index=ci*rank+j
                if index<done:continue
                a=engine.mean_transport(tokens,layer,basis[:,j],float(eps),skip_first)['final_mean'].astype(np.float64)
                b=engine.mean_transport(tokens,layer,basis[:,j],-float(eps),skip_first)['final_mean'].astype(np.float64)
                derivative=(a-b)/(2*eps)
                if j in (0,rank//2,rank-1):
                    aa=engine.mean_transport(tokens,layer,basis[:,j],float(eps/2),skip_first)['final_mean'].astype(np.float64)
                    bb=engine.mean_transport(tokens,layer,basis[:,j],-float(eps/2),skip_first)['final_mean'].astype(np.float64)
                    small=(aa-bb)/eps
                    checks.append(float(np.linalg.norm(derivative-small)/max(np.linalg.norm(small),1e-9)))
                if not np.isfinite(derivative).all():raise ValueError('Non-finite Jacobian column.')
                sums[:,j]+=derivative;done=index+1
                if done%16==0 or done==total:
                    _atomic_arrays(checkpoint,sums=sums,done=done,checks=np.array(checks),valid_counts=np.array(valid_counts))
                if progress:progress(done,total,'Native GGUF Jacobian columns; saved every 16 columns')
    transport=(sums/len(contexts)).astype(np.float32)
    _atomic_arrays(target,transport=transport,basis=basis)
    meta={**spec,'n_prompts':len(contexts),'valid_positions':valid_counts,'full_input_dimension':rank==dim,
          'vocabulary_size':engine.vocab_size,'output_residual_dimension':dim,
          'fd_relative_error':checks,'quality':'unstable' if max(checks,default=1)>.25 else 'finite-step approximation','seconds':round(time.monotonic()-started,3),
          'arrays_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
          'estimator':'mean prompts; mean valid source positions; sum current/future valid target derivatives; skip leading positions and final token',
          'limitations':'Finite differences, not autodiff. Corpus-dependent linearization, not a complete mind readout. Small default corpus does not reproduce the 1000-prompt study. Reduced-rank fits omit the orthogonal input subspace.',
          'saved_to':str(target)}
    target.with_suffix('.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    return {**meta,'transport':transport,'basis':basis}

def load(path,engine):
    path=Path(path);meta=json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
    if meta['digest']!=engine.model['digest'] or meta['abi']!=engine.abi:raise ValueError('Lens checkpoint/runtime mismatch.')
    if hashlib.sha256(path.read_bytes()).hexdigest()!=meta['arrays_sha256']:raise ValueError('Lens arrays changed.')
    with np.load(path,allow_pickle=False) as data:
        transport=data['transport'];basis=data['basis']
    if transport.shape!=(engine.dim,meta['rank']) or basis.shape!=transport.shape:raise ValueError('Invalid lens dimensions.')
    if not np.isfinite(transport).all() or not np.isfinite(basis).all():raise ValueError('Non-finite lens.')
    return {**meta,'transport':transport,'basis':basis}

def available(engine):
    paths=[]
    for path in (ROOT/'jacobian'/engine.model['digest']).glob('*-lens.npz'):
        try:
            meta=json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
            check_quality(meta)
            if meta['abi']==engine.abi:paths.append(path)
        except (OSError,ValueError,KeyError):pass
    return sorted(paths)

def check_quality(lens):
    if max(lens.get('fd_relative_error',[1]),default=1)>.25:
        raise ValueError('This numerical lens fails the step-size stability screen; use a larger step and refit. It is not a validated Jacobian.')

def read(engine,text,lens,positions=(-1,),top_k=15):
    check_quality(lens)
    if lens['digest']!=engine.model['digest'] or lens['abi']!=engine.abi:raise ValueError('Lens identity mismatch.')
    if not 1<=len(positions)<=32:raise ValueError('Read 1 to 32 selected positions.')
    raw=engine.read_positions(text,list(positions));rows=[];scores=[];errors=[];layer=int(lens['layer'])
    with engine.lock:
        for pos,h in raw['residuals'].items():
            transported=lens['transport']@(lens['basis'].T@h[layer])
            result=engine.unembed_residual(np.ascontiguousarray(transported,np.float32))
            logits=result['logits'].astype(np.float64);scores.append(logits.astype(np.float32));errors.append(result['replacement_error'])
            p=np.exp(logits-logits.max());p/=p.sum()
            for i in np.argsort(logits)[-top_k:][::-1]:
                rows.append({'position':int(pos),'input_token':engine.piece(raw['token_ids'][int(pos)]).decode('utf-8','replace'),
                             'layer':layer,'token_id':int(i),'token':engine.piece(int(i)).decode('utf-8','replace'),
                             'lens_logit':float(logits[i]),'lens_softmax_weight':float(p[i])})
    result={**identity(engine),'text':text,'lens':lens['saved_to'],'positions':list(raw['residuals']),
            'rows':rows,'vocabulary_size':engine.vocab_size,'replacement_error':max(errors),
            'scope':'All vocabulary scores from actual native normalization and unembedding. These lens weights are not the model next-token probabilities.'}
    path=Path(write_result(result,'jspace-vocabulary'))
    np.savez_compressed(path.with_suffix('.npz'),logits=np.stack(scores),token_ids=np.arange(engine.vocab_size),positions=np.array(list(raw['residuals'])))
    result['saved_to']=str(path);result['all_vocabulary_arrays']=str(path.with_suffix('.npz'));return result

def spectrum(lens):
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=4):
        _,s,vt=np.linalg.svd(lens['transport'],full_matrices=False)
    energy=np.cumsum(s*s)/max(float(s@s),1e-20)
    return {'singular_values':s,'energy':energy,'source_axes':lens['basis']@vt.T}

def erase_subspace(engine,text,lens,rank=4,fraction=.5):
    if lens['digest']!=engine.model['digest'] or lens['abi']!=engine.abi:raise ValueError('Lens mismatch.')
    if not 1<=rank<=lens['rank'] or not 0<=fraction<=1:raise ValueError('Invalid subspace cut.')
    axes=spectrum(lens)['source_axes'][:,:rank];tokens=engine.tokenize(text);layer=lens['layer']
    with engine.lock:
        try:
            engine.clear_steering();engine.reset();engine.evaluate(tokens)
            baseline=engine.logits();before=engine.capture()[layer]
            delta=-fraction*(axes@(axes.T@before))
            engine.reset()
            if len(tokens)>1:engine.evaluate(tokens[:-1])
            table=np.zeros((engine.layers,engine.dim),np.float32);table[layer]=delta
            engine.steer(table,1.,'add');engine.evaluate(tokens[-1:])
            after=engine.capture(1)[layer];changed=engine.logits()
            error=float(np.max(np.abs(axes.T@after-(1-fraction)*(axes.T@before))))
            engine.clear_steering();engine.reset();engine.evaluate(tokens);reset=float(np.max(abs(engine.logits()-baseline)))
            d=changed-baseline;ids=np.argsort(d)[-10:][::-1]
            result={**identity(engine),'layer':layer,'rank':rank,'fraction':fraction,'coordinate_error':error,'reset_error':reset,
                    'changes':[{'token':engine.piece(int(i)).decode('utf-8','replace'),'delta':float(d[i])} for i in ids],
                    'scope':'Single-position removal of top right-singular subspace of this fitted transport. Not all-layer J-space ablation.'}
            result['saved_to']=write_result(result,'jspace-subspace');return result
        finally:engine.clear_steering();engine.reset()
