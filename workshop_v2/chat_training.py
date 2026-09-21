"""Paired assistant-response directions, recipe persistence and causal calibration.

Training contrasts are never inserted into a chat-generation prompt. No weights edited.
"""
from __future__ import annotations
from pathlib import Path
import hashlib,json,time,uuid
import numpy as np
from sklearn.metrics import roc_auc_score
from contracts import slug
from .reasoning import format_chat
from .core import write_result
ROOT=Path(__file__).resolve().parent.parent

TASKS=[
'Share a thought about a new beginning.', 'Describe an ordinary afternoon.',
'Suggest a theme for a short story.', 'What catches your attention today?',
'Describe a place worth imagining.', 'Give me an idea for a drawing.',
'What should we talk about next?', 'Describe an interesting detail.',
'Offer an observation about everyday life.', 'Give me a metaphor for change.',
'Write a thought for a notebook.', 'Suggest a title and its meaning.',
'Name something worth exploring.', 'What makes a scene memorable?',
'Describe the mood of a quiet moment.', 'What might happen next in a story?',
'Give a brief reflection on a small discovery.', 'Describe a familiar routine.',
'Suggest a creative starting point.', 'What does this moment bring to mind?',
'Write a sentence about looking ahead.', 'Describe a scene at the end of the day.',
'Name an unexpected source of inspiration.', 'What would you notice on a walk?',
'Share one idea with a friend.', 'Describe a moment of anticipation.',
'What deserves a second look?', 'Suggest a subject for a conversation.',
'Write an opening for a description.', 'Share a thought about possibilities.',
'What could a photograph capture?', 'Describe something from a fresh perspective.']

STARTERS={
 'Ocean':{'name':'concept_ocean_chat','category':'Concepts','positive':'ocean waves, coastal water, marine life and the seashore',
          'negative':'office desks, filing cabinets, paperwork and indoor furniture',
          'positive_phrases':['Ocean waves roll onto the beach.','The sea stretches past the horizon.','Marine life moves through the blue water.','A salty breeze drifts across the shore.'],
          'negative_phrases':['Office desks line the indoor room.','Paperwork fills the filing cabinets.','A chair stands beside the document shelf.','A printed memo rests on the table.']},
 'Cheerful tone':{'name':'chat_emotion_cheerful','category':'Emotions and affect','positive':'cheerful, happy, delighted and optimistic expression',
          'negative':'sad, disappointed, gloomy and pessimistic expression',
          'positive_phrases':['I feel delighted and full of possibility.','This brings a happy sense of appreciation.','I welcome this with enthusiasm and joy.','It is wonderful to have something to look forward to.'],
          'negative_phrases':['I feel disappointed and empty of hope.','This brings a sad sense of loss.','I face this with gloom and regret.','It is disheartening to have nothing to look forward to.']},
 'Calm tone':{'name':'chat_emotion_calm','category':'Emotions and affect','positive':'calm, patient, composed and relaxed expression',
          'negative':'frantic, anxious, agitated and hurried expression',
          'positive_phrases':['There is time to pause and consider this calmly.','I take this at a gentle, unhurried pace.','A steady breath makes room for a clear thought.','I approach this with patience and composure.'],
          'negative_phrases':['There is no time and I am frantic about this.','I rush through this with mounting anxiety.','My racing thoughts leave no room to think.','I approach this with agitation and urgency.']},
 'Concise answers':{'name':'behavior_concise_chat','category':'Behavior and style','positive':'a concise direct answer',
          'negative':'an elaborate verbose explanation',
          'positive_phrases':['One clear idea is enough.','Start small and keep it simple.','Notice the detail that matters.','Choose one thing and explore it.'],
          'negative_phrases':['There are several important considerations worth discussing in detail. First, we should examine the broader context before we focus on any individual aspect. Second, there are multiple perspectives to consider, each with its own implications. Finally, we can bring these observations together into a more comprehensive account.']},
 'Space':{'name':'concept_space_chat','category':'Concepts','positive':'astronomy, stars, planets, spacecraft and distant galaxies',
          'negative':'gardening, plants, garden tools, flower beds and the backyard',
          'positive_phrases':['Distant galaxies fill the night sky.','A spacecraft travels between the planets.','Starlight crosses the vast reaches of space.','An astronaut studies a moon from orbit.'],
          'negative_phrases':['Flower beds fill the backyard garden.','A gardener carries tools between the plants.','Sunlight falls across a small vegetable plot.','A gardener checks a seedling in the soil.']}}

def make_recipe(name,category,positive,negative,positive_phrases=None,negative_phrases=None):
    name=slug(name)
    if not name or len(positive.strip())<3 or len(negative.strip())<3:
        raise ValueError('Provide a name and two distinct descriptions.')
    if positive.strip().casefold()==negative.strip().casefold():raise ValueError('The two sides must differ.')
    if category not in ('Concepts','Emotions and affect','Behavior and style','Bodily sensations'):
        raise ValueError('Choose an explicit category.')
    pos=positive_phrases or [f'I am thinking about {positive.strip().rstrip(".")}.',f'The scene centers on {positive.strip().rstrip(".")}.',f'One possibility involves {positive.strip().rstrip(".")}.']
    neg=negative_phrases or [f'I am thinking about {negative.strip().rstrip(".")}.',f'The scene centers on {negative.strip().rstrip(".")}.',f'One possibility involves {negative.strip().rstrip(".")}.']
    rows=[]
    for i,task in enumerate(TASKS):
        rows.append({'group':str(i),'prompt':task,'positive':pos[i%len(pos)],'negative':neg[i%len(neg)]})
    return {'name':name,'category':category,'positive_description':positive,'negative_description':negative,
            'pairs':rows,'origin':'Transparent workshop-authored template recipe; inspect/edit before training.',
            'caveats':'Shared phrasing can inflate separation; held-out template groups are not independent real-world behavioral validation.'}

def save_recipe(recipe):
    validate_recipe(recipe)
    folder=ROOT/'concepts'/'recipes';folder.mkdir(parents=True,exist_ok=True)
    raw=json.dumps(recipe,sort_keys=True,ensure_ascii=False).encode('utf-8')
    digest=hashlib.sha256(raw).hexdigest();path=folder/(slug(recipe['name'])+'-'+digest[:12]+'.json')
    if not path.exists():path.write_bytes(raw)
    return path,digest

def validate_recipe(recipe):
    if not isinstance(recipe,dict) or not slug(recipe.get('name','')):raise ValueError('Invalid recipe.')
    if slug(recipe['name'])!=recipe['name']:raise ValueError('Use lowercase letters, digits and underscores in the direction name.')
    if recipe.get('category') not in ('Concepts','Emotions and affect','Behavior and style','Bodily sensations'):raise ValueError('Invalid recipe category.')
    for field in ('positive_description','negative_description'):
        if not isinstance(recipe.get(field),str) or not recipe[field].strip():raise ValueError('Recipe descriptions are required.')
    rows=recipe.get('pairs',[])
    if not 20<=len(rows)<=256:raise ValueError('Use 20 to 256 paired examples.')
    for r in rows:
        for key in ('group','prompt','positive','negative'):
            if not isinstance(r.get(key),str) or not r[key].strip():raise ValueError('Each pair needs group, prompt, positive and negative strings.')
        if r['positive']==r['negative']:raise ValueError('A pair contains identical answers.')
        if any(len(r[k])>5000 for k in ('prompt','positive','negative')):raise ValueError('An example is too long.')
    if len({r['group'] for r in rows})<20:raise ValueError('Use at least 20 distinct context groups.')

def _paired_response_activations(engine,rows,recipe_hash,progress=None):
    from .resource_policy import checkpoint
    ident=(engine.model['digest'],engine.abi)
    folder=ROOT/'cache'/'paired-responses'/ident[0]/recipe_hash
    folder.mkdir(parents=True,exist_ok=True)
    values=[]
    with engine.lock:
        for i,row in enumerate(rows):
            checkpoint();part=folder/(f'{i:04d}.npz')
            pair=None
            if part.exists():
                try:
                    with np.load(part,allow_pickle=False) as z:
                        pair=z['activations'];saved=str(z['sha256'].item())
                    if pair.shape!=(2,engine.layers,engine.dim) or not np.isfinite(pair).all() or hashlib.sha256(pair.tobytes()).hexdigest()!=saved:
                        pair=None
                except (OSError,ValueError,KeyError,AttributeError):pair=None
            if pair is None:
                prefix=format_chat(engine,[{'role':'user','content':row['prompt']}],thinking='disabled')
                pair=np.asarray([engine.extract_response(prefix,row[side])['mean'] for side in ('positive','negative')],np.float32)
                if pair.shape!=(2,engine.layers,engine.dim) or not np.isfinite(pair).all():raise ValueError('Invalid response activations.')
                temp=part.with_name(part.stem+'.tmp.npz')
                saved=False
                for attempt in range(2):
                    folder.mkdir(parents=True,exist_ok=True)
                    try:
                        np.savez_compressed(temp,activations=pair,sha256=hashlib.sha256(pair.tobytes()).hexdigest())
                        temp.replace(part);saved=True;break
                    except FileNotFoundError:
                        if attempt:raise
                if not saved:raise RuntimeError('Could not persist paired-response activation cache.')
            values.append(pair)
            if progress:progress(i+1,len(rows),'Reading paired assistant-response activations (resumable)')
    if ident!=(engine.model['digest'],engine.abi):raise RuntimeError('Model changed during paired-response collection.')
    manifest={'digest':ident[0],'abi':ident[1],'recipe_hash':recipe_hash,'pairs':len(rows),'shape':[len(rows),2,engine.layers,engine.dim],'created_or_verified_at':time.time()}
    folder.mkdir(parents=True,exist_ok=True)
    tmp=folder/'manifest.tmp';tmp.write_text(json.dumps(manifest,indent=2),encoding='utf-8');tmp.replace(folder/'manifest.json')
    return np.stack(values).astype(np.float32,copy=False)

def train(engine,recipe,progress=None,replace=False):
    validate_recipe(recipe);path,recipe_hash=save_recipe(recipe);name=recipe['name']
    folder=ROOT/'vectors'/engine.model['digest'];folder.mkdir(parents=True,exist_ok=True)
    target=folder/(name+'.npz')
    if target.exists() and not replace:raise ValueError('That direction already exists. Choose a new name; existing directions are never overwritten silently.')
    rows=recipe['pairs']
    x=_paired_response_activations(engine,rows,recipe_hash,progress)
    if x.shape!=(len(rows),2,engine.layers,engine.dim) or not np.isfinite(x).all():raise ValueError('Invalid response activations.')
    group=np.array([r['group'] for r in rows]);unique=np.unique(group)
    shuffled=np.random.default_rng(1509).permutation(unique)
    n=max(4,int(np.ceil(len(unique)*.2)));test=np.isin(group,shuffled[:n]);dev=np.isin(group,shuffled[n:2*n]);tr=~(test|dev)
    a=x[tr,0].mean(0)-x[tr,1].mean(0)
    mag=np.linalg.norm(a,axis=1)
    if np.any(mag<1e-8):raise ValueError('A layer has a degenerate contrast; supply more diverse pairs.')
    d=a/mag[:,None];xt=x[tr].reshape(-1,engine.layers,engine.dim)
    proj=np.einsum('nld,ld->nl',xt,d)
    center=proj.mean(0);scale=np.maximum(proj.std(0,ddof=1),1e-8);norm=np.linalg.norm(xt,axis=2).mean(0)
    def auc(mask,dirs):
        scores=np.einsum('nbld,ld->nbl',x[mask],dirs).reshape(-1,engine.layers)
        y=np.tile([1,0],mask.sum())
        return [float(roc_auc_score(y,scores[:,k])) for k in range(engine.layers)]
    dev_auc=auc(dev,d);layer=int(np.argmax(dev_auc));test_auc=auc(test,d)[layer]
    meta={'name':name,'model':engine.model,'abi':engine.abi,'training_backend':engine.model.get('backend','native_gguf'),
          'source':'Locally authored paired assistant-response contrast; CAA-inspired, not the Pain Axis dataset.',
          'runtime_scope':engine.model.get('backend','native_gguf'),
          'method':'Difference of means of assistant response span activations. Train-only direction; grouped development layer selection; disjoint test groups.',
          'pooling':'assistant_response_mean','workshop_category':recipe['category'],
          'layer':layer,'train_n':int(tr.sum()*2),'dev_n':int(dev.sum()*2),'test_n':int(test.sum()*2),
          'cv_auc':dev_auc,'heldout_auc':test_auc,'raw_difference_heldout_auc':test_auc,
          'shuffled_label_control_auc':None,'recipe_file':str(path),'dataset_hash':recipe_hash,
          'positive_description':recipe['positive_description'],'negative_description':recipe['negative_description'],
          'heldout_groups':shuffled[:n].tolist(),'dev_groups':shuffled[n:2*n].tolist(),
          'behavioral_status':'Not calibrated','created_epoch':time.time(),
          'activation_cache':'cache/paired-responses/<model-digest>/<recipe-hash>; per-pair SHA-256 verified and resumable',
          'caveats':recipe.get('caveats','Authored examples; no claim of biological emotions.')}
    temp=target.with_name(target.stem+'-'+uuid.uuid4().hex+'.npz')
    np.savez_compressed(temp,directions=d.astype(np.float32),center=center.astype(np.float32),scale=scale.astype(np.float32),norm=norm.astype(np.float32))
    temp.replace(target);meta['arrays_sha256']=hashlib.sha256(target.read_bytes()).hexdigest()
    target.with_suffix('.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    return meta

CAL_PROMPTS=['Offer a short observation.','Describe a moment worth noticing.']
def calibrate(engine,bank,name,progress=None):
    """Development-selected response-likelihood calibration with disjoint check groups."""
    a=bank[name];meta=a['meta']
    recipe=json.loads(Path(meta['recipe_file']).read_text(encoding='utf-8'))
    dev=[r for r in recipe['pairs'] if r['group'] in meta['dev_groups']][:4]
    test=[r for r in recipe['pairs'] if r['group'] in meta['heldout_groups']][:4]
    if len(dev)<2 or len(test)<2:raise ValueError('Need at least two disjoint development and held-out contexts.')
    from .steering_controls import validate,matched_random_controls
    chosen=int(a['layer']);fractions=(.45,.6,.75,.9)
    sources=sorted(set([chosen]+[max(0,min(engine.layers-1,int(engine.layers*f)-1)) for f in fractions]))
    candidates=[]
    for src in sources:
        for inj in sorted(set([src,max(0,src-engine.layers//6),max(0,int(engine.layers*.6))])):
            for dose in (.15,.3,.5):candidates.append((src,inj,dose))
    def score(row,controls):
        prefix=format_chat(engine,[{'role':'user','content':row['prompt']}],thinking='disabled')
        table=validate(engine,bank,controls,.75)
        cap=min(32,max(8,len(engine.tokenize(row['positive']))),max(8,len(engine.tokenize(row['negative']))))
        pos=engine.score_response(prefix,row['positive'],table,cap)['mean_log_probability']
        neg=engine.score_response(prefix,row['negative'],table,cap)['mean_log_probability']
        return pos-neg
    dev_base=np.array([score(r,[]) for r in dev]);results=[]
    for i,(src,inj,dose) in enumerate(candidates):
        c={'axis':name,'vector_layer':src,'layer':inj,'dose':dose,'mode':'add'}
        margins=np.array([score(r,[c]) for r in dev])
        results.append({'control':c,'mean_margin_gain':float(np.mean(margins-dev_base)),'margins':margins.tolist()})
        if progress:progress(i+1,len(candidates)+4,'Development-only response-likelihood calibration')
    best=max(results,key=lambda r:r['mean_margin_gain']);control=best['control']
    test_base=np.array([score(r,[]) for r in test]);test_changed=np.array([score(r,[control]) for r in test])
    heldout_gain=float(np.mean(test_changed-test_base));random_gains=[]
    for j,seed in enumerate((55019,55021,55027)):
        rc=matched_random_controls(engine,bank,[control],.75,seed)
        random_gains.append(float(np.mean(np.array([score(r,rc) for r in test])-test_base)))
        if progress:progress(len(candidates)+j+1,len(candidates)+4,'Held-out matched-random checks')
    recommended=bool(best['mean_margin_gain']>0 and heldout_gain>max(0.,max(random_gains)))
    out={'name':name,'digest':engine.model['digest'],'abi':engine.abi,'arrays_sha256':meta['arrays_sha256'],
         'control':control,'scope':'generation','phase':'all','thinking':'disabled','dose_limit':.75,
         'mean_margin_gain':heldout_gain,'development_margin_gain':best['mean_margin_gain'],
         'heldout_gain':heldout_gain,'random_margin_gain':max(random_gains),'random_gains':random_gains,
         'dev_n':len(dev),'heldout_n':len(test),'recommended':recommended,'profile_kind':'paired_response_likelihood_v2',
         'candidates':results,'created_at':time.time(),
         'quality':'Disjoint held-out conditional-likelihood check with three equal-norm random directions; still not completed-answer semantic validation.'}
    folder=ROOT/'calibration'/'chat-profiles'/engine.model['digest'];folder.mkdir(parents=True,exist_ok=True)
    (folder/(name+'.json')).write_text(json.dumps(out,indent=2),encoding='utf-8')
    return out

def profiles(engine,bank):
    found={}
    def accept(v,n):
        return n in bank and v.get('abi')==engine.abi and v.get('digest')==engine.model['digest'] and v.get('arrays_sha256')==bank[n]['meta'].get('arrays_sha256') and v.get('profile_kind') in ('paired_response_likelihood_v2','causal_vocabulary_screen')
    folder=ROOT/'calibration'/'chat-profiles'/engine.model['digest']
    if folder.exists():
        for p in folder.glob('*.json'):
            try:
                v=json.loads(p.read_text(encoding='utf-8'));n=v['name']
                if accept(v,n):found[n]=v
            except (OSError,ValueError,KeyError):pass
    actuator=ROOT/'calibration'/'actuator-profiles'/engine.model['digest']
    if actuator.exists():
        for p in actuator.glob('*.json'):
            try:
                v=json.loads(p.read_text(encoding='utf-8'));n=v['name']
                if accept(v,n) and v.get('recommended'):
                    v=dict(v,mean_margin_gain=v.get('heldout_gain',0.),
                           random_margin_gain=max(v.get('random_gains',[0.])),
                           profile_kind=v.get('profile_kind','causal_vocabulary_screen'))
                    found[n]=v
            except (OSError,ValueError,KeyError):pass
    return found

# Keep the model identity fixed through fitting, calibration and persistence.
from functools import wraps
def _exclusive(operation):
    @wraps(operation)
    def invoke(engine,*args,**kwargs):
        with engine.lock:return operation(engine,*args,**kwargs)
    return invoke
train=_exclusive(train)
calibrate=_exclusive(calibrate)
