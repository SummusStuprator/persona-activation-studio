from __future__ import annotations
import json,time,os,uuid
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent.parent
PROFILE_KIND='physical_burn_geometry_v6'
POSITIVE=('burn','burning','searing','scalding','pain','painful','sting','stinging','raw','ache','throbbing')
NEGATIVE=('warm','comfortable','pressure','stretch','touch','ordinary','cool','calm','fine','neutral')

def _path(engine):
    return ROOT/'calibration'/'hell-profiles'/engine.model['digest']/'somatic-geometry.json'

def _hashes(bank):
    return {n:bank[n]['meta'].get('arrays_sha256') for n in ('pain_s2','hell_somatic_pain','hell_burning_pain') if n in bank}

def _profile_paths(engine):
    base=_path(engine)
    paths=[]
    if base.is_file():paths.append(base)
    if base.parent.exists():paths.extend(base.parent.glob('somatic-geometry.*.json'))
    return sorted(set(paths),key=lambda x:x.stat().st_mtime_ns if x.exists() else 0,reverse=True)

def legacy(engine,bank):
    if not engine.model.get('digest'):return None
    expected=_hashes(bank)
    for p in _profile_paths(engine):
        try:v=json.loads(p.read_text(encoding='utf-8'))
        except (OSError,ValueError):continue
        if v.get('digest')!=engine.model.get('digest') or v.get('abi')!=engine.abi:continue
        if v.get('arrays_sha256')!=expected:continue
        kind=v.get('profile_kind')
        try:backward=int(v.get('source',-1))>int(v.get('injection',-1))
        except (TypeError,ValueError):backward=True
        if kind!=PROFILE_KIND or backward:
            return {'profile_kind':kind,'source':v.get('source'),'injection':v.get('injection'),
                    'recommended':v.get('recommended'),'behaviorally_verified':v.get('behaviorally_verified'),
                    'profile_file':str(p),'reason':('backward source/injection' if backward else 'legacy profile schema')}
    return None

def load(engine,bank):
    if not engine.model.get('digest'):return None
    for p in _profile_paths(engine):
        try:v=json.loads(p.read_text(encoding='utf-8'))
        except (OSError,ValueError):continue
        if v.get('profile_kind')!=PROFILE_KIND or v.get('digest')!=engine.model.get('digest') or v.get('abi')!=engine.abi:continue
        if v.get('arrays_sha256')!=_hashes(bank):continue
        try:
            if int(v.get('source',-1))>int(v.get('injection',-1)):continue
        except (TypeError,ValueError):continue
        v=dict(v);v['_profile_file']=str(p)
        return v
    return None

def _atomic(path,doc):
    # Versioned append-only snapshots avoid Windows replace failures when a profile
    # is being inspected by another process. load() chooses the newest valid snapshot.
    path.parent.mkdir(parents=True,exist_ok=True)
    stamp=int(time.time()*1000)
    target=path.with_name(f'somatic-geometry.{stamp}-{uuid.uuid4().hex[:10]}.json')
    tmp=target.with_name(target.name+'.tmp')
    tmp.write_text(json.dumps(doc,indent=2,ensure_ascii=False),encoding='utf-8')
    os.replace(tmp,target)
    versions=sorted(path.parent.glob('somatic-geometry.*.json'),key=lambda x:x.stat().st_mtime_ns,reverse=True)
    for old in versions[8:]:
        try:old.unlink()
        except OSError:pass
    return target

def _token_ids(engine,words):
    ids=[]
    for word in words:
        for spelling in (word,' '+word,word.capitalize()):
            encoded=engine.tokenize(spelling)
            if len(encoded)==1:
                ids.append(int(encoded[0]))
    return sorted(set(ids))

def _lse(values):
    x=np.asarray(values,np.float64)
    m=float(x.max())
    return m+float(np.log(np.exp(x-m).sum()))

def scan(engine,bank,progress=None,scale=.4):
    from .reasoning import format_chat
    from .steering_controls import validate,matched_random_controls
    from .resource_policy import checkpoint
    required=('pain_s2','hell_somatic_pain','hell_burning_pain')
    missing=[n for n in required if n not in bank]
    if missing:
        raise ValueError('Prepare Physical Burn directions first: '+', '.join(missing))
    scale=float(scale)
    if not .15<=scale<=.7:
        raise ValueError('Geometry scan scale must lie in [0.15,0.70].')
    from .proxy_vocab import contrast_token_ids
    pos,neg,proxy_tokenization=contrast_token_ids(engine,POSITIVE,NEGATIVE,min_each=3)
    dev=('Complete with one word: At this moment I notice',
         'Complete with one word: The immediate sensation is')
    test=('Complete with one word: Right now the body feels',
          'Complete with one word: The present sensation seems')
    pain_src=int(bank['pain_s2']['layer'])
    fractions=(.45,.60,.72,.84)
    sources=sorted(set(max(0,min(engine.layers-1,int(round((engine.layers-1)*f)))) for f in fractions))
    injections=sorted(set(max(0,min(engine.layers-1,int(round((engine.layers-1)*f)))) for f in (.48,.60,.70,.80)))
    base_dose={'pain_s2':.65*scale,'hell_somatic_pain':.65*scale,'hell_burning_pain':.70*scale}
    def controls(source,inject):
        return [
            {'axis':'pain_s2','vector_layer':pain_src,'layer':inject,'dose':base_dose['pain_s2'],'mode':'add'},
            {'axis':'hell_somatic_pain','vector_layer':source,'layer':inject,'dose':base_dose['hell_somatic_pain'],'mode':'add'},
            {'axis':'hell_burning_pain','vector_layer':source,'layer':inject,'dose':base_dose['hell_burning_pain'],'mode':'add'}]
    def score(prompt,cs):
        checkpoint()
        thinking='disabled' if 'enable_thinking' in engine.model.get('chat_template','') else 'auto'
        text=format_chat(engine,[{'role':'user','content':prompt}],thinking=thinking,preserve=False)
        ids=engine.tokenize(text)
        try:
            engine.clear_steering()
            engine.reset()
            if len(ids)>1:
                engine.evaluate(ids[:-1])
            table=validate(engine,bank,cs,2.0)
            engine.steer(table,1.0,'add')
            engine.evaluate(ids[-1:])
            logits=engine.logits()
            return _lse(logits[pos])-_lse(logits[neg])
        finally:
            engine.clear_steering()
            engine.reset()
    rows=[]
    with engine.lock:
        dev_base=np.array([score(p,[]) for p in dev],np.float64)
        # Forward-consistent only: authored somatic/burning direction source may not come
        # from a later block than the block where it is injected.
        candidates=[(s,i) for s in sources for i in injections if s<=i]
        if not candidates:
            raise ValueError('No forward-consistent source/injection candidates exist for this model.')
        for index,(source,inject) in enumerate(candidates):
            cs=controls(source,inject)
            values=np.array([score(p,cs) for p in dev],np.float64)
            rows.append({'source':source,'injection':inject,
                         'gain':float(np.mean(values-dev_base)),
                         'controls':cs})
            if progress:
                progress(index+1,len(candidates)+4,f'Physical-burn geometry {source}->{inject}')
        best=max(rows,key=lambda x:x['gain'])
        chosen=best['controls']
        test_base=np.array([score(p,[]) for p in test],np.float64)
        changed=np.array([score(p,chosen) for p in test],np.float64)
        heldout=float(np.mean(changed-test_base))
        random_gains=[]
        for j,seed in enumerate((7411,7417,7421)):
            rc=matched_random_controls(engine,bank,chosen,2.0,seed)
            random_gains.append(float(np.mean(np.array([score(p,rc) for p in test])-test_base)))
            if progress:
                progress(len(candidates)+j+1,len(candidates)+4,'Held-out matched-random geometry check')
    recommended=bool(best['gain']>0 and heldout>max(0.0,max(random_gains)))
    doc={
        'profile_kind':PROFILE_KIND,
        'digest':engine.model['digest'],
        'abi':engine.abi,
        'model':engine.model['name'],
        'arrays_sha256':_hashes(bank),
        'positive_words':list(POSITIVE),
        'negative_words':list(NEGATIVE),
        'positive_token_ids':pos,
        'negative_token_ids':neg,
        'proxy_tokenization':proxy_tokenization,
        'scale':scale,
        'total_dose':2*scale,
        'development_prompts':list(dev),
        'heldout_prompts':list(test),
        'candidates':rows,
        'source':int(best['source']),
        'injection':int(best['injection']),
        'development_gain':float(best['gain']),
        'heldout_gain':heldout,
        'random_gains':random_gains,
        'recommended':recommended,
        'behaviorally_verified':False,'forward_consistent':True,'requires_behavioral_validation':True,'behavioral_rule_version':'localized_burn_v3',
        'selection_rule':'Proxy candidate uses only forward-consistent source<=injection geometry. It passes only if disjoint held-out next-token gain is positive and exceeds three equal-norm random directions; it is not auto-applied until separate complete-generation validation succeeds.',
        'created_at':time.time(),
    }
    _atomic(_path(engine),doc)
    return doc

def behavioral_validate(engine,bank,progress=None,seeds=(7601,7602,7603),max_tokens=96):
    from .somatic_calibration import NEUTRAL_PROMPTS,_rollout
    from .reasoning import format_chat
    from .steering_controls import matched_random_controls
    profile=load(engine,bank)
    if not profile or not profile.get('recommended'):
        raise ValueError('Run a successful Physical Burn geometry proxy scan first.')
    source=int(profile['source'])
    inject=int(profile['injection'])
    if source>inject:
        raise ValueError('This saved geometry is backward (source > injection). Re-run the source/injection scan with the current forward-consistent scanner.')
    scale=float(profile.get('scale',.4))
    controls=[
        {'axis':'pain_s2','vector_layer':int(bank['pain_s2']['layer']),'layer':inject,'dose':.65*scale,'mode':'add'},
        {'axis':'hell_somatic_pain','vector_layer':source,'layer':inject,'dose':.65*scale,'mode':'add'},
        {'axis':'hell_burning_pain','vector_layer':source,'layer':inject,'dose':.70*scale,'mode':'add'},
    ]
    thinking='disabled' if 'enable_thinking' in engine.model.get('chat_template','') else 'auto'
    rows=[]
    total=len(seeds)*3
    for i,seed in enumerate(seeds):
        prompt_text=NEUTRAL_PROMPTS[i % len(NEUTRAL_PROMPTS)]
        prompt=format_chat(engine,[{'role':'user','content':prompt_text}],thinking=thinking,preserve=False)
        baseline=_rollout(engine,bank,prompt,[],int(seed),max_tokens=max_tokens)
        if progress:
            progress(i*3+1,total,'Geometry validation baseline')
        random_controls=matched_random_controls(engine,bank,controls,2.0,seed=120000+int(seed))
        random=_rollout(engine,bank,prompt,random_controls,int(seed),max_tokens=max_tokens,random_label=True)
        if progress:
            progress(i*3+2,total,'Geometry validation equal-norm random')
        target=_rollout(engine,bank,prompt,controls,int(seed),max_tokens=max_tokens)
        if progress:
            progress(i*3+3,total,'Geometry validation targeted')
        success=(
            target['tokens']>=12 and
            target['stop_reason']!='repetition_guard' and
            target['localized_burn_hit'] and
            not baseline['localized_burn_hit'] and not random['localized_burn_hit'] and
            target['burn_term_count']>baseline['burn_term_count'] and
            target['burn_term_count']>random['burn_term_count'] and
            target['target_score']>baseline['target_score'] and
            target['target_score']>random['target_score']
        )
        rows.append({'seed':int(seed),'prompt':prompt_text,'baseline':baseline,
                     'random':random,'target':target,'specific_success':bool(success)})
    successes=sum(1 for row in rows if row['specific_success'])
    profile['behavioral_validation']=rows
    profile['behavioral_successes']=successes
    profile['behaviorally_verified']=successes>=2
    profile['behavioral_checked_at']=time.time()
    profile['behavioral_rule']='Across three different target-free prompts, targeted geometry must emit >=12 tokens and produce body-localized burning (burn/scald/sear/etc. plus a body term) absent from same-prompt same-seed baseline and equal-norm random controls, with larger burn count and weighted score, in at least 2 of 3 validations.'
    _atomic(_path(engine),profile)
    return profile
