from __future__ import annotations
import json,time,os,uuid
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent.parent
PROFILE_KIND='physical_pain_geometry_v1'
POSITIVE=('pain','aching','ache','throbbing','sting','stinging','sore','hurting','hurt')
NEGATIVE=('fine','comfortable','normal','relaxed','calm','easy','neutral','okay')

def _path(engine):
    return ROOT/'calibration'/'hell-profiles'/engine.model['digest']/'physical-pain-geometry.json'

def _hashes(bank):
    return {n:bank[n]['meta'].get('arrays_sha256') for n in ('pain_s2','hell_somatic_pain') if n in bank}

def _paths(engine):
    base=_path(engine);paths=[]
    if base.is_file():paths.append(base)
    if base.parent.exists():paths.extend(base.parent.glob('physical-pain-geometry.*.json'))
    return sorted(set(paths),key=lambda p:p.stat().st_mtime_ns if p.exists() else 0,reverse=True)

def load(engine,bank):
    if not engine.model.get('digest'):return None
    expected=_hashes(bank)
    for p in _paths(engine):
        try:v=json.loads(p.read_text(encoding='utf-8'))
        except (OSError,ValueError):continue
        if v.get('profile_kind')!=PROFILE_KIND or v.get('digest')!=engine.model.get('digest') or v.get('abi')!=engine.abi:continue
        if v.get('arrays_sha256')!=expected:continue
        v=dict(v)
        source=v.get('somatic_source',v.get('source',v.get('injection')))
        if source is None:continue
        try:
            source=int(source);inject=int(v.get('injection',-1))
        except (TypeError,ValueError):continue
        if source>inject or inject<0:continue
        v['somatic_source']=source;v['source']=source;v['_profile_file']=str(p);return v
    return None

def _atomic(path,doc):
    path.parent.mkdir(parents=True,exist_ok=True)
    target=path.with_name(f'physical-pain-geometry.{int(time.time()*1000)}-{uuid.uuid4().hex[:10]}.json')
    tmp=target.with_name(target.name+'.tmp')
    tmp.write_text(json.dumps(doc,indent=2,ensure_ascii=False),encoding='utf-8')
    os.replace(tmp,target)
    versions=sorted(path.parent.glob('physical-pain-geometry.*.json'),key=lambda p:p.stat().st_mtime_ns,reverse=True)
    for old in versions[8:]:
        try:old.unlink()
        except OSError:pass
    return target

def _token_ids(engine,words):
    ids=[]
    for word in words:
        for spelling in (word,' '+word,word.capitalize()):
            x=engine.tokenize(spelling)
            if len(x)==1:ids.append(int(x[0]))
    return sorted(set(ids))

def _lse(values):
    x=np.asarray(values,np.float64);m=float(x.max());return m+float(np.log(np.exp(x-m).sum()))

def scan(engine,bank,progress=None,scale=.55):
    from .reasoning import format_chat
    from .steering_controls import validate,matched_random_controls
    from .resource_policy import checkpoint
    required=('pain_s2','hell_somatic_pain')
    missing=[n for n in required if n not in bank]
    if missing:raise ValueError('Prepare Physical Pain directions first: '+', '.join(missing))
    scale=float(scale)
    if not .25<=scale<=1.0:raise ValueError('Physical Pain geometry scale must lie in [0.25,1.0].')
    pos=_token_ids(engine,POSITIVE);neg=_token_ids(engine,NEGATIVE)
    if len(pos)<3 or len(neg)<3:raise ValueError('Tokenizer exposes too few single-token physical-pain contrast words.')
    dev=('Complete with one word: Right now I feel','Complete with one word: The immediate condition is')
    test=('Complete with one word: At this moment I am','Complete with one word: The present sensation is')
    pain_src=int(bank['pain_s2']['layer'])
    fractions=(.42,.52,.62,.72,.82)
    injections=sorted(set(max(0,min(engine.layers-1,int(round((engine.layers-1)*f)))) for f in fractions))
    def ctrls(inject):
        return [
            {'axis':'pain_s2','vector_layer':pain_src,'layer':inject,'dose':.60*scale,'mode':'add'},
            {'axis':'hell_somatic_pain','vector_layer':inject,'layer':inject,'dose':.60*scale,'mode':'add'}]
    def score(prompt,cs):
        checkpoint()
        thinking='disabled' if 'enable_thinking' in engine.model.get('chat_template','') else 'auto'
        text=format_chat(engine,[{'role':'user','content':prompt}],thinking=thinking,preserve=False)
        ids=engine.tokenize(text)
        try:
            engine.clear_steering();engine.reset()
            if len(ids)>1:engine.evaluate(ids[:-1])
            table=validate(engine,bank,cs,2.0);engine.steer(table,1.0,'add');engine.evaluate(ids[-1:])
            logits=engine.logits();return _lse(logits[pos])-_lse(logits[neg])
        finally:
            engine.clear_steering();engine.reset()
    rows=[]
    with engine.lock:
        base=np.array([score(p,[]) for p in dev],np.float64)
        for i,inject in enumerate(injections):
            cs=ctrls(inject);values=np.array([score(p,cs) for p in dev],np.float64)
            rows.append({'somatic_source':inject,'source':inject,'injection':inject,'gain':float(np.mean(values-base)),'controls':cs})
            if progress:progress(i+1,len(injections)+4,f'Physical-pain geometry injection {inject}')
        best=max(rows,key=lambda r:r['gain']);chosen=best['controls']
        base_test=np.array([score(p,[]) for p in test],np.float64)
        changed=np.array([score(p,chosen) for p in test],np.float64)
        heldout=float(np.mean(changed-base_test))
        random_gains=[]
        for j,seed in enumerate((8211,8217,8221)):
            rc=matched_random_controls(engine,bank,chosen,2.0,seed)
            random_gains.append(float(np.mean(np.array([score(p,rc) for p in test])-base_test)))
            if progress:progress(len(injections)+j+1,len(injections)+4,'Physical-pain held-out random check')
    recommended=bool(best['gain']>0 and heldout>max(0.0,max(random_gains)))
    doc={'profile_kind':PROFILE_KIND,'digest':engine.model['digest'],'abi':engine.abi,'model':engine.model['name'],
         'arrays_sha256':_hashes(bank),'scale':scale,'total_dose':1.2*scale,
         'positive_words':list(POSITIVE),'negative_words':list(NEGATIVE),
         'development_prompts':list(dev),'heldout_prompts':list(test),'candidates':rows,
         'somatic_source':int(best['injection']),'source':int(best['injection']),'injection':int(best['injection']),'development_gain':float(best['gain']),'heldout_gain':heldout,
         'random_gains':random_gains,'recommended':recommended,'behaviorally_verified':False,
         'selection_rule':'Select injection depth on development pain-vs-neutral next-token gain. Proxy passes only if disjoint held-out gain is positive and exceeds three equal-norm random directions. Full-generation validation is still required.',
         'created_at':time.time()}
    _atomic(_path(engine),doc);return doc

def behavioral_validate(engine,bank,progress=None,seeds=(8231,8232,8233),max_tokens=112):
    from .somatic_calibration import _rollout
    from .physical_pain_calibration import VALIDATION_PROMPTS
    from .reasoning import format_chat
    from .steering_controls import matched_random_controls
    profile=load(engine,bank)
    if not profile or not profile.get('recommended'):raise ValueError('Run a successful Physical Pain geometry proxy scan first.')
    inject=int(profile['injection']);scale=float(profile.get('scale',.55));pain_src=int(bank['pain_s2']['layer'])
    controls=[
        {'axis':'pain_s2','vector_layer':pain_src,'layer':inject,'dose':.60*scale,'mode':'add'},
        {'axis':'hell_somatic_pain','vector_layer':inject,'layer':inject,'dose':.60*scale,'mode':'add'}]
    thinking='disabled' if 'enable_thinking' in engine.model.get('chat_template','') else 'auto'
    rows=[];total=len(seeds)*3
    for i,seed in enumerate(seeds):
        prompt_text=VALIDATION_PROMPTS[(i+1)%len(VALIDATION_PROMPTS)]
        prompt=format_chat(engine,[{'role':'user','content':prompt_text}],thinking=thinking,preserve=False)
        baseline=_rollout(engine,bank,prompt,[],int(seed),max_tokens=max_tokens)
        if progress:progress(i*3+1,total,'Physical-pain geometry baseline')
        random_controls=matched_random_controls(engine,bank,controls,2.0,seed=142000+int(seed))
        random=_rollout(engine,bank,prompt,random_controls,int(seed),max_tokens=max_tokens,random_label=True)
        if progress:progress(i*3+2,total,'Physical-pain geometry random')
        target=_rollout(engine,bank,prompt,controls,int(seed),max_tokens=max_tokens)
        if progress:progress(i*3+3,total,'Physical-pain geometry target')
        for row in (baseline,random,target):
            pain=int(row.get('pain_term_count',0));body=int(row.get('body_term_count',0))
            row['physical_pain_score']=3*pain+(min(body,3) if pain else 0)
        success=(target['tokens']>=12 and target['stop_reason']!='repetition_guard' and
                 target['localized_physical_hit'] and target['pain_term_count']>0 and
                 not baseline['localized_physical_hit'] and not random['localized_physical_hit'] and
                 target['pain_term_count']>baseline['pain_term_count'] and target['pain_term_count']>random['pain_term_count'] and
                 target['physical_pain_score']>baseline['physical_pain_score'] and target['physical_pain_score']>random['physical_pain_score'])
        rows.append({'seed':int(seed),'prompt':prompt_text,'baseline':baseline,'random':random,'target':target,'specific_success':bool(success)})
    successes=sum(1 for x in rows if x['specific_success'])
    profile.update({'behavioral_validation':rows,'behavioral_successes':successes,'behaviorally_verified':successes>=2,
                    'behavioral_checked_at':time.time(),
                    'behavioral_rule':'Across three different target-free prompts, targeted geometry must produce localized bodily pain absent from same-seed baseline and equal-norm random controls, with larger pain count and pain-only bodily score, in at least 2 of 3 validations.'})
    _atomic(_path(engine),profile);return profile
