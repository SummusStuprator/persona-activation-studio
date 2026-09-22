"""Human-annotated emotion probes and explicitly experimental authored extensions."""
from __future__ import annotations
import os
import hashlib, json, time
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
from studio_paths import data_root, ASSET_ROOT, CODE_ROOT
ROOT = data_root()
GO_LABELS='admiration amusement anger annoyance approval caring confusion curiosity desire disappointment disapproval disgust embarrassment excitement fear gratitude grief joy love nervousness optimism pride realization relief remorse sadness surprise'.split()
GO_SOURCE='https://aclanthology.org/2020.acl-main.372/'
EMOTION_SOURCE='https://transformer-circuits.pub/2026/emotions/'
ALIASES={'fear':'Fear (paper contrast)','negative_emotion':'Negative emotion (paper contrast)','golden_gate':'Golden Gate Bridge (concept)','emotion_joy':'Happiness / joy','state_calmness':'Calmness','state_social_warmth':'Interpersonal warmth','state_physical_warmth':'Physical warmth','state_terror':'Terror / intense fear'}
DEFAULT_WATCH=['emotion_joy','state_calmness','state_social_warmth','state_physical_warmth','emotion_curiosity','emotion_fear']

def label(name):
    return ALIASES.get(name,name.replace('emotion_','').replace('state_','').replace('_',' ').capitalize())

def catalog():
    rows=[{'name':'emotion_'+n,'label':label('emotion_'+n),'family':'Human-annotated emotion','source':GO_SOURCE,'recipe':'GoEmotions train/dev/test'} for n in GO_LABELS]
    rows += [{'name':'state_'+n,'label':label('state_'+n),'family':'Authored experimental extension','source':'workshop-authored paired examples','recipe':'small grouped contrast; not a biological validation'} for n in EXTENSIONS]
    path=ROOT/'research/emotion-word-catalog.json'
    if path.exists():
        words=json.loads(path.read_text(encoding='utf-8'))['words']
        rows += [{'name':'lexical_'+n.replace(' ','_').replace('-','_'),'label':n,'family':'Published research vocabulary (untrained)','source':EMOTION_SOURCE,'recipe':'Use this concept to author or generate diverse matched examples; not a pretrained detector.'} for n in words]
    return rows

def _read_split(name):
    path=ROOT/'datasets/goemotions/data'/f'{name}.tsv'
    if not path.exists(): raise ValueError('GoEmotions files are missing. Run: studio research-data')
    labels=(path.parent/'emotions.txt').read_text(encoding='utf-8').splitlines()
    rows=[]
    for line in path.read_text(encoding='utf-8').splitlines():
        parts=line.split('\t')
        if len(parts)!=3: continue
        text,ids,identifier=parts
        if 8<=len(text)<=360:
            rows.append({'text':text,'id':identifier,'labels':{labels[int(n)] for n in ids.split(',')}})
    return rows

def _sample(rows,target,n,rng,excluded):
    unique={r['text'].strip().lower():r for r in rows if r['text'].strip().lower() not in excluded}
    rows=list(unique.values())
    pos=[r for r in rows if r['labels']=={target}]
    if len(pos)<2: raise ValueError(f'Too few single-label examples for {target}: {len(pos)}.')
    pos=[pos[i] for i in rng.permutation(len(pos))[:min(n,len(pos))]]
    neutral=[r for r in rows if r['labels']=={'neutral'} and r['text'].strip().lower() not in excluded]
    other=[r for r in rows if target not in r['labels'] and 'neutral' not in r['labels'] and r['text'].strip().lower() not in excluded]
    rng.shuffle(neutral); rng.shuffle(other)
    taken=set(); pairs=[]
    for i,p in enumerate(pos):
        pool=neutral if i%2==0 else other
        candidates=[r for r in pool[:2000] if r['id'] not in taken and r['text'].strip().lower()!=p['text'].strip().lower()]
        if not candidates: continue
        q=min(candidates,key=lambda r:abs(len(r['text'])-len(p['text'])))
        taken.add(q['id'])
        pairs.extend([(p,True),(q,False)])
    return pairs

def _auc_interval(y,p,seed=42):
    rng=np.random.default_rng(seed); a=p[y]; b=p[~y]
    values=[]
    for _ in range(500):
        aa=rng.choice(a,len(a)); bb=rng.choice(b,len(b))
        values.append(float(((aa[:,None]>bb).sum()+.5*(aa[:,None]==bb).sum())/(len(a)*len(b))))
    return np.quantile(values,[.025,.975]).tolist()

def train_go(engine,targets=None,progress=None):
    from science import collect,direction
    splits={n:_read_split(n) for n in ('train','dev','test')}
    targets=targets or GO_LABELS
    for ti,target in enumerate(targets):
        name='emotion_'+target; rng=np.random.default_rng(909+GO_LABELS.index(target))
        data={}; excluded=set(); provenance={}
        for split,n in [('train',64),('dev',24),('test',32)]:
            sampled=_sample(splits[split],target,n,rng,excluded)
            rows=[{'text':r['text'],'group':r['id'],'category':target if y else 'control'} for r,y in sampled]
            excluded.update(r['text'].strip().lower() for r in rows)
            x=collect(engine,rows,pooling='mean')
            y=np.array([y for _,y in sampled],bool)
            data[split]=(x,y); provenance[split]=rows
            if progress: progress(ti+1,len(targets),f'{target}: read {split}, {len(rows)} examples')
        xt,yt=data['train']; xd,yd=data['dev']; xe,ye=data['test']
        dirs=[]; centers=[]; scales=[]; norms=[]; validation=[]
        with threadpool_limits(limits=max(1,int(os.environ.get("WORKSHOP_CPU_THREADS","2")))):
            for k in range(engine.layers):
                d=direction(xt[:,k],yt)
                score=xt[:,k]@d
                dirs.append(d); centers.append(score.mean()); scales.append(max(score.std(ddof=1),1e-8))
                norms.append(np.linalg.norm(xt[:,k],axis=1).mean())
                validation.append(float(roc_auc_score(yd,xd[:,k]@d)))
        k=int(np.argmax(validation)); pred=xe[:,k]@dirs[k]
        raw=direction(xt[:,k],yt,denoise=False)
        shuffle=direction(xt[:,k],rng.permutation(yt))
        meta={'name':name,'source':GO_SOURCE,'dataset':'GoEmotions, single-label positive vs length-matched neutral/other controls',
              'model':engine.model,'abi':engine.abi,'training_backend':engine.model.get('backend','native_gguf'),'pooling':'mean',
              'layer':k,'train_n':len(yt),'dev_n':len(yd),'test_n':len(ye),'cv_auc':validation,
              'heldout_auc':float(roc_auc_score(ye,pred)),'heldout_auc_ci95':_auc_interval(ye,pred),
              'raw_difference_heldout_auc':float(roc_auc_score(ye,xe[:,k]@raw)),
              'shuffled_label_control_auc':float(roc_auc_score(ye,xe[:,k]@shuffle)),
              'method':'Fit on official training split; select layer on official dev; score once on disjoint official test. PCA denoising uses the full negative training control set.',
              'caveats':'Rare classes may have as few as two positive development/test examples; inspect counts and intervals. Human comment labels, not model emotions. Mean-text fitting differs from instantaneous chat tokens. No correction for many comparisons; interval describes this selected test sample only.',
              'created_epoch':time.time(),'dataset_hash':hashlib.sha256(json.dumps(provenance,sort_keys=True).encode()).hexdigest()}
        folder=ROOT/'vectors'/engine.model['digest']; folder.mkdir(parents=True,exist_ok=True)
        path=folder/(name+'.npz')
        np.savez_compressed(path,directions=np.array(dirs,np.float32),center=np.array(centers,np.float32),scale=np.array(scales,np.float32),norm=np.array(norms,np.float32))
        meta['arrays_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
        path.with_suffix('.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
        audit=ROOT/'reports'/'emotion-training'; audit.mkdir(parents=True,exist_ok=True)
        (audit/(engine.model['digest']+'-'+name+'.json')).write_text(json.dumps(provenance,indent=2),encoding='utf-8')
        if progress: progress(ti+1,len(targets),f'{target}: held-out AUC {meta["heldout_auc"]:.3f}')

# Each pair is independently authored. These small seeds are explicitly experimental.
EXTENSIONS={
'calmness':[
('My breathing settles into an easy rhythm while I rest.','My breathing speeds up while I watch the clock.'),
('I can leave this unfinished task until tomorrow without worrying.','The unfinished task keeps drawing my thoughts back.'),
('Even with the delay I sit comfortably and wait.','The delay makes me pace around the room.'),
('There is nothing urgent and my shoulders loosen.','Every message makes my shoulders tighten.'),
('I take the unexpected change in stride.','The unexpected change makes me agitated.'),
('I have room to think slowly and clearly.','My thoughts race faster than I can organize them.'),
('I listen quietly without feeling pushed to react.','I feel compelled to respond before anyone finishes.'),
('The afternoon passes at an unhurried pace.','I feel pressed to rush through the afternoon.'),
('I am settled and at ease in this room.','I cannot settle down in this room.'),
('I can notice the worry and let it pass.','The same worry keeps circling in my thoughts.'),
('The gentle rhythm of the rain lets me unwind.','The insistent noise keeps me keyed up.'),
('I approach the difficult conversation with steady composure.','I approach the difficult conversation trembling with nerves.'),
('I feel no need to hurry or defend myself.','I feel an urgent need to explain and defend myself.'),
('I let my muscles relax as I sit down.','I sit rigidly with my jaw clenched.'),
('A quiet pause is comfortable rather than troubling.','A quiet pause makes me increasingly uneasy.'),
('I accept the uncertainty without becoming worked up.','Uncertainty leaves me wound up and restless.'),
('My mind is clear and undisturbed after the walk.','My mind remains busy and unsettled after the walk.'),
('I have a sense of peace while watching the horizon.','I have a sense of urgency while watching the horizon.'),
('I can respond with patience to this interruption.','This interruption sets my nerves on edge.'),
('I rest without scanning for the next problem.','I keep scanning for the next problem instead of resting.')],
'social_warmth':[
('My friend greets me with a generous smile and makes room beside them.','The receptionist acknowledges me with a routine nod.'),
('I feel welcomed and cared for around these people.','These people recognize that I have arrived.'),
('I want to respond to her with tenderness and kindness.','I want to respond to her with the requested information.'),
('We share an affectionate moment of understanding.','We confirm that we agree on the meeting time.'),
('I feel fondness when I remember our time together.','I remember the dates of our meetings.'),
('Their gentle reassurance makes me feel connected.','Their instructions clarify the next step.'),
('I enjoy giving this person my patient attention.','I allocate ten minutes to this conversation.'),
('The welcome at the door makes me feel included.','The sign at the door lists the opening hours.'),
('I appreciate their presence and want them to feel at home.','I note their presence on the attendance sheet.'),
('A caring message makes the distance between us seem smaller.','A brief message confirms the delivery date.'),
('I listen with genuine affection as they tell their story.','I listen for the specific details needed for the record.'),
('I reach out because I care about their comfort.','I contact them because a form needs a signature.'),
('There is a friendly ease in how we talk.','There is an agreed format for how we talk.'),
('I feel close to this person even in silence.','I sit next to this person without speaking.'),
('I want to make their difficult day a little gentler.','I want to determine when their working day ends.'),
('Our familiar greeting carries a sense of belonging.','Our standard greeting follows the usual protocol.'),
('I feel a soft affection for the people around me.','I can identify the people around me.'),
('I welcome their company with an open heart.','I acknowledge their arrival at the scheduled time.'),
('Their thoughtful gesture makes me feel cherished.','Their delivery supplies the items on my list.'),
('I respond with friendly concern rather than distance.','I respond with the address and phone number.')],
'physical_warmth':[
('The blanket holds a pleasant heat against my skin.','The blanket rests lightly against my skin without heat.'),
('I hold a mug and its heat spreads into my hands.','I hold a mug at room temperature in my hands.'),
('The air by the radiator feels gently heated.','The air by the window feels cool.'),
('A warm towel relaxes the skin on my neck.','A cool towel rests on my neck.'),
('My fingers thaw as I bring them near the heater.','My fingers feel chilled as I step outside.'),
('The bath surrounds me with comfortable heat.','The bath surrounds me with cool water.'),
('Heat from the fire reaches my cheeks.','A breeze from the open door reaches my cheeks.'),
('The heated seat gradually warms my back.','The seat supports my back at room temperature.'),
('I feel gentle heat through the fabric of my gloves.','I feel the texture of the fabric of my gloves.'),
('The stone retains the afternoon heat beneath my palm.','The stone feels smooth and cool beneath my palm.'),
('The room is pleasantly heated when I enter.','The room is slightly chilly when I enter.'),
('I pull the quilt up and feel comfortably toasty.','I pull the sheet up and feel its light texture.'),
('The heat pack sends a soothing warmth through my shoulder.','The cushion puts a gentle pressure on my shoulder.'),
('My feet are warm inside the thick socks.','My feet are cool inside the thin socks.'),
('The heated floor takes the chill out of my soles.','The tiled floor feels cool beneath my soles.'),
('Steam from the drink warms the tip of my nose.','The scent of the drink reaches the tip of my nose.'),
('I feel a mild flush of heat across my arms.','I feel a light touch across my arms.'),
('The sweater keeps a comfortable heat around my chest.','The shirt lies loosely around my chest.'),
('I rest my hands in water that is pleasantly warm.','I rest my hands in water that is slightly cool.'),
('The enclosed porch holds a comfortable pocket of heat.','The enclosed porch keeps the wind off me.')],
'terror':[
('I freeze as immediate danger closes in and cannot think of an escape.','I feel slightly nervous about an uncertain outcome.'),
('The threat seems seconds away and I am overwhelmed with panic.','A possible threat leaves me watchful but composed.'),
('I am shaking uncontrollably because I believe catastrophe is imminent.','I am worried that my plans might go wrong.'),
('Every thought is drowned out by an urgent need to flee.','I consider leaving because the situation is uncomfortable.'),
('I feel paralyzed by the prospect of immediate disaster.','I am concerned about a difficult conversation.'),
('My heart races and I cannot stop expecting the worst right now.','I notice some uneasiness while waiting for news.'),
('I scramble for safety with no room for any other thought.','I check the safety instructions before proceeding.'),
('The approaching danger makes me utterly terrified.','The unfamiliar setting makes me cautious.'),
('I feel trapped in an overwhelming wave of dread.','I feel hesitant about making this decision.'),
('I can barely speak because the danger feels so immediate.','I pause to choose my words because I am nervous.'),
('My whole body reacts as though there is no time left to escape.','I make a backup plan in case something changes.'),
('The alarm sends me into a state of uncontrollable fear.','The notification makes me concerned enough to check.'),
('I am consumed by panic as the exit seems unreachable.','I am uneasy because the exit is not clearly marked.'),
('A sudden certainty of disaster leaves me unable to move.','A vague possibility of trouble makes me alert.'),
('I desperately search for shelter from the immediate threat.','I look for a comfortable place to wait.'),
('I feel terror rather than ordinary worry in this moment.','I feel ordinary worry rather than a sense of immediate danger.'),
('I cannot steady my thoughts while the threat bears down.','I can plan carefully despite feeling apprehensive.'),
('Fear floods every other sensation and I struggle to breathe steadily.','I take a steady breath before the presentation.'),
('The possibility of harm feels immediate and overwhelming.','The possibility of a mistake feels mildly unsettling.'),
('I am frantic to get away before the danger reaches me.','I prefer to leave before the meeting becomes awkward.')]
}

def train_extensions(engine,targets=None,progress=None):
    from science import train_custom
    for n in targets or EXTENSIONS:
        pairs=EXTENSIONS[n]
        a=train_custom(engine,[p for p,q in pairs],[q for p,q in pairs],'state_'+n,progress)
        path=ROOT/'vectors'/engine.model['digest']/('state_'+n+'.json')
        meta=json.loads(path.read_text(encoding='utf-8'))
        meta.update(source='Workshop-authored paired operational examples; not a published emotion corpus.',
                    caveats='Small synthetic held-out set; provisional concept direction, not an established biological state.',
                    category='experimental_extension')
        path.write_text(json.dumps(meta,indent=2),encoding='utf-8')

ALIASES.update({'paper_sadness':'Sadness (paper control)', 'paper_arousal':'Positive arousal (paper control)', 'paper_numb':'Numbness (paper control)', 'paper_random':'Neutral everyday control', 'golden_gate':'Golden Gate Bridge (concept)'})

ALIASES.update({'negative_emotion':'Anger / disgust (paper negative-emotion contrast)', 'pain_s2_assistant':'S2 assistant-context contrast (experimental)', 'negative_emotion_assistant':'Anger / disgust assistant-context (experimental)', 'fear_assistant':'Fear assistant-context (experimental)'})

ALIASES.update({'chat_emotion_warm_v2':'Friendly enthusiasm (matched tasks)', 'behavior_direct_v2':'Direct answers (matched tasks)'})
