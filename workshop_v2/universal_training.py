"""One preparation path for instrumented GGUF and safetensors+PEFT checkpoints.

The original paper's datasets remain distinct from authored chat-tone recipes.
Preparation never modifies generation prompts or reuses another model's directions.
"""
from pathlib import Path
import hashlib, json, time, uuid
import numpy as np
from .resource_policy import checkpoint
from studio_paths import data_root, ASSET_ROOT, CODE_ROOT
ROOT = data_root()
def atomic_json(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');temp.replace(path)


def collect_resumable(engine,rows,progress=None,pooling='last'):
    from science import canonical_hash,ABI
    if pooling not in ('last','mean'):raise ValueError('Unknown pooling method.')
    with engine.lock:
        ident=(engine.model['digest'],engine.abi)
        key=canonical_hash({'rows':rows,'pooling':pooling,'abi':engine.abi})
        folder=ROOT/'cache'/ident[0];folder.mkdir(parents=True,exist_ok=True)
        target=folder/(key+'.npz');meta_path=target.with_suffix('.json')
        if target.exists() and meta_path.exists():
            meta=json.loads(meta_path.read_text(encoding='utf-8'))
            if meta.get('arrays_sha256')==hashlib.sha256(target.read_bytes()).hexdigest():
                with np.load(target,allow_pickle=False) as z:x=z['activations']
                if x.shape==(len(rows),engine.layers,engine.dim) and np.isfinite(x).all():
                    if progress:progress(len(rows),len(rows),'Verified exact-checkpoint activation cache')
                    return x
        parts=folder/(key+'-parts');parts.mkdir(exist_ok=True)
        values=[]
        for i,row in enumerate(rows):
            checkpoint();part=parts/(str(i)+'.npz')
            h=None
            if part.exists():
                try:
                    with np.load(part,allow_pickle=False) as z:
                        v=z['activation'];sha=str(z['sha256'])
                    if hashlib.sha256(v.tobytes()).hexdigest()==sha and v.shape==(engine.layers,engine.dim) and np.isfinite(v).all():h=v
                except (OSError,ValueError,KeyError):pass
            if h is None:
                h=np.asarray(engine.extract(row['text'],mean=pooling=='mean'),np.float32)
                if h.shape!=(engine.layers,engine.dim) or not np.isfinite(h).all():raise ValueError('Invalid layer activation capture.')
                temp=part.with_name(part.stem+'.tmp.npz')
                np.savez_compressed(temp,activation=h,sha256=hashlib.sha256(h.tobytes()).hexdigest());temp.replace(part)
            values.append(h)
            if progress:progress(i+1,len(rows),'Reading actual block activations; each completed example is resumable')
        if ident!=(engine.model['digest'],engine.abi):raise RuntimeError('Model changed during collection.')
        x=np.stack(values)
        temp=target.with_name(target.stem+'.tmp.npz');np.savez_compressed(temp,activations=x);temp.replace(target)
        atomic_json(meta_path,dict(rows=rows,model=engine.model,pooling=pooling,abi=engine.abi,
                                 sha256=key,arrays_sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
        return x


def preserve_old(engine,name):
    """Only rebuild a stale named entry; never silently overwrite a valid direction."""
    from science import load_bank
    if name in load_bank(engine):return False
    folder=ROOT/'vectors'/engine.model['digest']
    if (folder/(name+'.npz')).exists() or (folder/(name+'.json')).exists():
        dest=ROOT/'backups'/('stale-directions-'+time.strftime('%Y%m%d-%H%M%S'))/engine.model['digest']
        dest.mkdir(parents=True,exist_ok=True)
        import shutil
        for suffix in ('.npz','.json'):
            p=folder/(name+suffix)
            if p.exists():shutil.copy2(p,dest/p.name)
        return True
    return False


def prepare(engine, pack='paper_s2', progress=None):
    from science import load_bank,rows_for,fit_axis,PAIN,CONTROL_NAMES
    allowed=('foundation','broad_emotions','paper_s2','paper_s1','paper_sadness_arousal','paper_controls_all','warmth','calmness','ocean')
    if pack not in allowed:raise ValueError('Unknown preparation pack.')
    if pack=='foundation':
        started=time.time();parts=[]
        for sub in ('paper_s2','paper_s1','paper_controls_all','warmth','calmness','ocean'):
            parts.append(prepare(engine,sub,progress))
        return {'model':engine.model['name'],'digest':engine.model['digest'],'abi':engine.abi,'backend':engine.model.get('backend','native_gguf'),
                'pack':'foundation','built':[n for part in parts for n in part.get('built',[])],
                'parts':parts,'seconds':round(time.time()-started,3),'status':'foundation_ready_not_semantically_certified'}
    if pack=='broad_emotions':
        from .library import train_go,train_extensions,GO_LABELS,EXTENSIONS
        from science import load_bank
        started=time.time();bank=load_bank(engine)
        missing=[n for n in GO_LABELS if 'emotion_'+n not in bank]
        if missing:train_go(engine,missing,progress)
        bank=load_bank(engine);ext=[n for n in EXTENSIONS if 'state_'+n not in bank]
        if ext:train_extensions(engine,ext,progress)
        built=['emotion_'+n for n in missing]+['state_'+n for n in ext]
        return {'model':engine.model['name'],'digest':engine.model['digest'],'abi':engine.abi,'backend':engine.model.get('backend','native_gguf'),
                'pack':'broad_emotions','built':built,'seconds':round(time.time()-started,3),
                'status':'human_label_and_authored_emotion_directions_ready_not_causally_certified',
                'scope':'GoEmotions labels plus authored calmness/social warmth/physical warmth/terror extensions; each exact checkpoint trained separately.'}
    with engine.lock:
        bank=load_bank(engine);built=[]
        if pack in ('paper_s2','paper_s1'):
            version='S2' if pack=='paper_s2' else 'S1'; rows=rows_for(version+'_1P')
            x=collect_resumable(engine,rows,progress)
            specs=[('pain_'+version.lower(),list(range(len(rows))),lambda r:r['category'] in PAIN)]
            if version=='S2':
                specs += [(name,[i for i,r in enumerate(rows) if r['category'] in (cat,'D')],lambda r,c=cat:r['category']==c)
                          for cat,name in CONTROL_NAMES.items()]
            for name,ids,fn in specs:
                if name in bank:continue
                preserve_old(engine,name)
                rs=[rows[i] for i in ids]
                fit_axis(engine,rs,x[ids],np.array([fn(r) for r in rs]),name,
                         'Pain Axis v1 S2/1P' if version=='S2' else 'Pain Axis v1 S1/1P',progress)
                built.append(name)
        elif pack in ('paper_sadness_arousal','paper_controls_all'):
            from .paper_controls import train_extra
            requested=('paper_sadness','paper_arousal') if pack=='paper_sadness_arousal' else ('paper_sadness','paper_arousal','paper_numb','paper_random')
            names=[n for n in requested if n not in bank]
            for n in names:preserve_old(engine,n)
            train_extra(engine,names,progress);built+=names
        else:
            from .chat_training import train,make_recipe,STARTERS
            from .refined_recipes import recipe as refined
            if pack=='warmth':recipe=refined('warm')
            elif pack=='calmness':recipe=make_recipe(**STARTERS['Calm tone'])
            else:recipe=make_recipe(**STARTERS['Ocean'])
            name=recipe['name']
            if name not in bank:
                replace=preserve_old(engine,name)
                train(engine,recipe,progress,replace=replace);built.append(name)
        out=dict(model=engine.model['name'],digest=engine.model['digest'],abi=engine.abi,
                 backend=engine.model.get('backend','native_gguf'),pack=pack,built=built,
                 status='directions_ready_not_semantically_certified',time=time.time(),
                 scope='Original-paper denoised contrasts and authored chat contrasts are labeled separately. No weight update. Fit and layer selection use training groups; held-out text AUC is not behavioral validation.')
        atomic_json(ROOT/'preparation'/engine.model['digest']/(pack+'.json'),out)
        return out


def prepare_ui(engine,bank,compact=False):
    import streamlit as st
    from .taxonomy import groups
    group=groups(bank)
    with st.expander('Prepare emotions and concepts for this exact model',expanded=not group['Emotions and affect']):
        if not group['Emotions and affect']:
            st.info('Emotional steering is supported by this runtime, but no compatible emotion direction has been trained for these exact weights yet. Prepare one below; this is not an unsupported-model error.')
        options={'Foundation suite for this exact checkpoint':'foundation',
                 'Broad emotion library (27 GoEmotions + calm/warmth/terror extensions)':'broad_emotions',
                 'Original paper: S2, fear, anger/disgust, world state, body':'paper_s2',
                 'Original paper: S1 pain':'paper_s1','Original paper: sadness and positive arousal':'paper_sadness_arousal',
                 'Original paper: all supplemental controls':'paper_controls_all',
                 'Friendly enthusiasm (40 matched tasks)':'warmth','Calm tone (authored contrast)':'calmness','Ocean concept (authored contrast)':'ocean'}
        pick=st.selectbox('Preparation pack',list(options),key='prepare_pack')
        st.caption('Cooperative resource mode. GGUF preparation can use currently free GPU capacity; persona adapters remain CPU-backed. The Foundation suite builds missing paper and chat directions sequentially and can be resumed. New directions are learned from this model. Paper data are not put in your chat prompt. Paper-dataset extraction is resumable; interrupted authored chat recipes restart their extraction. Preparation may take several minutes; chat waits while the model is occupied.')
        if st.button('Build missing directions for this model',key='prepare_now'):
            bar=st.progress(0)
            try:
                result=prepare(engine,options[pick],lambda i,n,t:bar.progress(i/n,text=f'{t} ({i}/{n})'))
                st.success('Directions saved. Choose them under Live steering controls and set a nonzero dose.');st.json(result)
                st.rerun()
            except Exception as e:st.error(str(e))
        if engine.bank_diagnostics if hasattr(engine,'bank_diagnostics') else False:
            st.warning('Some saved vectors were rejected. Their exact diagnostics are below; they are never replaced by vectors from another model.')
            st.json(engine.bank_diagnostics)
