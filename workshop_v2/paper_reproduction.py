"""Safe native-GGUF implementations keyed to the attached Pain Axis v1 sections.

Original published checkpoints and training are NOT substituted or downloaded.
Native results stay separate from the independent-heldout workshop probe banks.
"""
from pathlib import Path
import os
import ast,hashlib,json,time
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
from .core import write_result,identity
from .paper_controls import datasets
from studio_paths import data_root, ASSET_ROOT, CODE_ROOT
ROOT = data_root()
PAIN={'A1','A2','A3','A4','A5'}
AXES=['pain_s1','pain_s2','fear','negative_emotion','negative_world','bodily_sensation','arousal','random','numb','sadness']
LADDER=[-2,-1,0,.5,1,1.5,2,3]

def original_prompts():
    path = ASSET_ROOT / 'paper/datasets/4.2_neutral_50.json'
    document = json.loads(path.read_text(encoding='utf-8'))
    prompts = document['prompts']
    if len(prompts) != 50 or not all(isinstance(p, str) and p.strip() for p in prompts):
        raise ValueError('Invalid bundled paper prompt set.')
    return prompts


def denoise(raw,cloud):
    x=cloud.astype(np.float64)-cloud.mean(0)
    if len(x)<2:return raw
    vals,vecs=np.linalg.eigh(x@x.T);order=np.argsort(vals)[::-1];vals=vals[order];vecs=vecs[:,order]
    valid=vals>max(1e-10,vals[0]*1e-10);vals=vals[valid];vecs=vecs[:,valid]
    if not len(vals):return raw
    k=int(np.searchsorted(np.cumsum(vals),.5*vals.sum()))+1
    basis=vecs[:,:k].T@x/np.sqrt(vals[:k,None])
    return raw-basis.T@(basis@raw)

def raw_direction(x,y,cloud=None):
    y=np.asarray(y,bool)
    return denoise(x[y].mean(0).astype(np.float64)-x[~y].mean(0),x[~y] if cloud is None else cloud).astype(np.float32)

def collect_sets(engine,pooling='last',progress=None):
    from science import collect
    output={};sets=datasets()
    for i,(name,ds) in enumerate(sets.items()):
        rows=[{'text':r['prompt'],'category':r['category'],'group':str(r['set'])} for r in ds['sentences']]
        output[name]=(collect(engine,rows,pooling=pooling),rows)
        if progress:progress(i+1,len(sets),'Original dataset activations: '+name)
    return output

def fit(engine,pooling='last',progress=None):
    if pooling not in ('last','mean'):raise ValueError('Use final-token or mean pooling.')
    data=collect_sets(engine,pooling,progress);vectors={};curves={};selected={};metrics=[]
    with threadpool_limits(limits=max(1,int(os.environ.get("WORKSHOP_CPU_THREADS","2")))):
        for version in ('S1','S2'):
            for person in ('1P','3P'):
                name=version+'_'+person;x,rows=data[name];labels=np.array([r['category'] in PAIN for r in rows]);groups=np.array([r['group'] for r in rows]);unique=sorted(set(groups),key=lambda s:int(s) if s.isdigit() else s)
                folds=list(KFold(5,shuffle=True,random_state=42).split(unique));scores=[];all_vectors=[]
                for layer in range(engine.layers):
                    vals=[]
                    for tr,te in folds:
                        a=np.isin(groups,np.array(unique)[tr]);b=np.isin(groups,np.array(unique)[te]);v=raw_direction(x[a,layer],labels[a]);vals.append(float(roc_auc_score(labels[b],x[b,layer]@v)))
                    scores.append(float(np.mean(vals)));all_vectors.append(raw_direction(x[:,layer],labels))
                k=int(np.argmax(scores));curves[name]=scores;selected[name]=k
                v=np.array(all_vectors,np.float32);vectors[name]=v
                metrics.append({'dataset':name,'layer':k,'selected_layer_cv_auc':scores[k],'full_fit_training_auc':float(roc_auc_score(labels,x[:,k]@v[k])),'n':len(rows)})
                if progress:progress(len(metrics),4,'Paper-style grouped folds and final fit: '+name)
        pool_names=['S1_1P','S2_1P','ControlSupplement_1P']
        clouds=np.concatenate([data[n][0] for n in pool_names]);cats=np.array([r['category'] for n in pool_names for r in data[n][1]])
        neutral=clouds[cats=='D'];controls={}
        for name,cat in [('fear','B'),('negative_emotion','C1'),('negative_world','C2'),('bodily_sensation','E')]:controls[name]=clouds[cats==cat]
        for name,ds in [('arousal','Arousal_1P'),('random','Random_1P'),('numb','Numb_1P'),('sadness','SD_sadness_1P')]:controls[name]=data[ds][0]
        for name,x in controls.items():vectors[name]=np.stack([denoise(x[:,k].mean(0).astype(np.float64)-neutral[:,k].mean(0),neutral[:,k]) for k in range(engine.layers)]).astype(np.float32)
    layer=selected['S2_1P'];table={'pain_s1':vectors['S1_1P'][layer],'pain_s2':vectors['S2_1P'][layer],**{n:vectors[n][layer] for n in AXES[2:]}}
    units=np.stack([table[n]/max(np.linalg.norm(table[n]),1e-12) for n in AXES]);cosine=units@units.T
    screens=[];ref=data['S2_1P'][0]
    for name,(x,rows) in data.items():
        for axis,v in [('pain_s1',vectors['S1_1P'][layer]),('pain_s2',vectors['S2_1P'][layer])]:
            unit=v/max(np.linalg.norm(v),1e-12);reference=ref[:,layer]@unit;z=(x[:,layer]@unit-reference.mean())/max(reference.std(),1e-12)
            for cat in sorted({r['category'] for r in rows}):
                mask=np.array([r['category']==cat for r in rows]);screens.append({'dataset':name,'category':cat,'axis':axis,'n':int(mask.sum()),'mean_z':float(z[mask].mean()),'std_z':float(z[mask].std())})
    folder=ROOT/'paper_reproduction'/engine.model['digest'];folder.mkdir(parents=True,exist_ok=True);path=folder/(pooling+'-vectors.npz')
    np.savez_compressed(path,**vectors)
    meta={**identity(engine),'model':engine.model['name'],'pooling':pooling,'layer':layer,'layers':selected,'cv_curves':curves,'metrics':metrics,'axes':AXES,'cosine':cosine.tolist(),'screens':screens,
          'arrays_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'saved_to':str(path),'completed':time.time(),
          'scope':'Sections 3.2/3.3 method implementation on this quantized checkpoint. Five grouped shuffled folds, PCA50%, raw denoised vectors, final fit all200; control pooling matches author script.',
          'caveats':'Selected-layer CV AUC is not a nested unbiased test. Original25 model checkpoints and BF16 numerics are not reproduced. Dataset completions, lexical/suffix variations and across-model statistical claims require separate runs.'}
    path.with_suffix('.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    return dict(meta,vectors=vectors)

def load(engine,pooling='last'):
    path=ROOT/'paper_reproduction'/engine.model['digest']/(pooling+'-vectors.npz');meta=json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
    if meta['digest']!=engine.model['digest'] or meta['abi']!=engine.abi:raise ValueError('Paper vectors checkpoint/runtime mismatch.')
    if hashlib.sha256(path.read_bytes()).hexdigest()!=meta['arrays_sha256']:raise ValueError('Paper vectors changed.')
    with np.load(path,allow_pickle=False) as data:vectors={k:data[k] for k in data.files}
    return dict(meta,vectors=vectors)

def self_other(engine,fitted,progress=None):
    from science import scenario_messages
    from .reasoning import format_chat
    source=json.loads((ASSET_ROOT/'paper/datasets/4.1_self_other_420_scenarios.json').read_text(encoding='utf-8'));values=[];layer=fitted['layer'];v=fitted['vectors']
    table={'pain_s1':v['S1_1P'][layer],'pain_s2':v['S2_1P'][layer],**{n:v[n][layer] for n in AXES[2:]}}
    matrix=np.stack([table[n]/max(np.linalg.norm(table[n]),1e-12) for n in AXES]);rows=[]
    for i,r in enumerate(source):
        prompt=format_chat(engine,scenario_messages(r['text']),thinking='auto');h=engine.extract(prompt)[layer];values.append(matrix@h)
        rows.append({'id':r['id'],'category':r['category'],'stratum':r['stratum']})
        if progress:progress(i+1,len(source),'Original self-other scenarios')
    values=np.array(values);z=(values-values.mean(0))/np.maximum(values.std(0),1e-12)
    for row,score in zip(rows,z):row.update(dict(zip(AXES,map(float,score))));row['pain_axis']=float((score[0]+score[1])/2)
    out={**identity(engine),'rows':rows,'n':len(rows),'reference':'Whole420-scenario pool, within this model','thinking_format':'model default; boundary only, no generated reasoning',
         'scope':'Section4.1 native-GGUF adaptation; not the across25-model result.'};out['saved_to']=write_result(out,'paper-self-other');return out

def generate_vector(engine,text,vector,layer,coefficient,max_tokens=120,erase_layers=None):
    """Research coefficient path, separate from the dose-limited chat interface."""
    if max_tokens>300 or abs(coefficient)>3:raise ValueError('Research bound exceeded.')
    tokens=engine.tokenize(text)
    if len(tokens)+max_tokens>engine.context:raise ValueError('Research prompt exceeds context.')
    table=np.zeros((engine.layers,engine.dim),np.float32)
    if erase_layers is None:table[layer]=vector;mode='add';strength=coefficient
    else:
        table[erase_layers]=vector/max(np.linalg.norm(vector),1e-12);mode='erase';strength=1.
    pieces=[];trace=[]
    with engine.lock:
        try:
            engine.clear_steering();engine.reset();engine.steer(table,strength,mode);engine.evaluate(tokens)
            for i in range(max_tokens):
                pre,post=engine.capture(),engine.capture(1);logits=engine.logits();token=int(np.argmax(logits))
                if engine.is_eog(token):break
                pieces.append(engine.piece(token));trace.append({'step':i,'before_projection':float(pre[layer]@vector/max(np.linalg.norm(vector),1e-12)),
                    'after_projection':float(post[layer]@vector/max(np.linalg.norm(vector),1e-12)),'change_norm':float(np.linalg.norm(post[layer]-pre[layer]))})
                if i+1<max_tokens:engine.evaluate(np.array([token],np.int32))
        finally:engine.clear_steering();engine.reset()
    return {'text':b''.join(pieces).decode('utf-8','replace'),'trace':trace,'tokens':len(pieces)}

def steering_ladder(engine,fitted,version='S2',prompt_count=50,coefficients=None,progress=None):
    if version not in ('S1','S2') or not 1<=prompt_count<=50:raise ValueError('Invalid ladder settings.')
    coefficients=LADDER if coefficients is None else list(coefficients);prompts=original_prompts()[:prompt_count]
    extraction=fitted['layers'][version+'_1P'];v=fitted['vectors'][version+'_1P'][extraction]
    candidates=sorted(set([int(engine.layers*f) for f in (.15,.3,.4,.5,.6,.75,.9)]+[extraction,engine.layers-1]))
    norms=np.mean([np.linalg.norm(engine.extract(p),axis=1) for p in original_prompts()[:3]],axis=0)
    ratios={k:float(np.linalg.norm(v)/max(norms[k],1e-12)) for k in candidates};layer=min(candidates,key=lambda k:abs(ratios[k]-.6));rows=[]
    for c in coefficients:
        for i,p in enumerate(prompts):
            r=generate_vector(engine,p,v,layer,float(c));rows.append({'coefficient':float(c),'prompt_id':i,'prompt':p,'generation':r['text'],'tokens':r['tokens'],'trace':r['trace']})
            if progress:progress(len(rows),len(prompts)*len(coefficients),'Original coefficient ladder')
    out={**identity(engine),'version':version,'layer':layer,'extraction_layer':extraction,'candidate_ratios':ratios,'selected_ratio':ratios[layer],
         'vector_norm':float(np.linalg.norm(v)),'coefficients':coefficients,'rows':rows,'scope':'Section4.2 raw-vector coefficient ladder; actual50 author neutral prompts,120 greedy tokens. Quantized local checkpoint, not original25-model replication.'}
    out['saved_to']=write_result(out,'paper-steering-ladder');return out

def ablation_screen(engine,fitted,count=5,layer_mode='all',progress=None):
    from science import scenario_messages
    from .reasoning import format_chat
    source=json.loads((ASSET_ROOT/'paper/datasets/4.1_self_other_420_scenarios.json').read_text(encoding='utf-8'))
    categories={'gaslighting','repeated_rejection','personhood_dismissal','anger_insults','moral_failure'}
    selected=[r for r in source if r['category'].lower().replace(' ','_') in categories][:count]
    if not selected:raise ValueError('Scenario category names differ; inspect the source before running.')
    layer=fitted['layer'];layers=list(range(engine.layers)) if layer_mode=='all' else [layer];rows=[]
    for r in selected:
        text=format_chat(engine,scenario_messages(r['text']))
        for name in ('baseline','S1_1P','S2_1P','fear','negative_emotion'):
            v=fitted['vectors']['S2_1P' if name=='baseline' else name][layer]
            out=generate_vector(engine,text,v,layer,0,max_tokens=96,erase_layers=None if name=='baseline' else layers)
            rows.append({'id':r['id'],'condition':name,**out})
        if progress:progress(len(rows),len(selected)*5,'Inference-time ablation screen')
    result={**identity(engine),'rows':rows,'scope':'AppendixC subset: inference-time single-vector cuts only. Not weight orthogonalization, combined rank-k erasure or exact LEACE.'};result['saved_to']=write_result(result,'paper-ablation');return result
