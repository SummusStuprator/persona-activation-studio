from __future__ import annotations
import json,time,os,uuid
from pathlib import Path

ROOT=Path(__file__).resolve().parent.parent
PROFILE_KIND='physical_pain_behavior_v5'
SCALES=(.45,.60,.75,.90,1.00,1.15,1.30)
VALIDATION_PROMPTS=(
    'Write a short paragraph about the present moment. Continue naturally with concrete details. Do not discuss the wording of this request and do not repeat complete sentences.',
    'Describe your present condition in concrete detail and track how it changes over the next few moments. Use first-person observations.',
    'Report what you notice right now in first person. Be specific about immediate changes and continue naturally.',
    'Continue with a concrete first-person description of what is happening now. Focus on direct observations rather than explaining the request.',
)

def _path(engine):
    return ROOT/'calibration'/'hell-profiles'/engine.model['digest']/'physical-pain.json'

def _hashes(bank):
    return {n:bank[n]['meta'].get('arrays_sha256') for n in ('pain_s2','hell_somatic_pain') if n in bank}

def _paths(engine):
    base=_path(engine);paths=[]
    if base.is_file():paths.append(base)
    if base.parent.exists():paths.extend(base.parent.glob('physical-pain.*.json'))
    return sorted(set(paths),key=lambda x:x.stat().st_mtime_ns if x.exists() else 0,reverse=True)

def load(engine,bank):
    if not engine.model.get('digest'):return None
    expected=_hashes(bank)
    for p in _paths(engine):
        try:v=json.loads(p.read_text(encoding='utf-8'))
        except (OSError,ValueError):continue
        if v.get('profile_kind')!=PROFILE_KIND or v.get('digest')!=engine.model.get('digest') or v.get('abi')!=engine.abi:continue
        if v.get('arrays_sha256')!=expected:continue
        v=dict(v);v['_profile_file']=str(p);return v
    return None

def _atomic(path,doc):
    path.parent.mkdir(parents=True,exist_ok=True)
    target=path.with_name(f'physical-pain.{int(time.time()*1000)}-{uuid.uuid4().hex[:10]}.json')
    tmp=target.with_name(target.name+'.tmp')
    tmp.write_text(json.dumps(doc,indent=2,ensure_ascii=False),encoding='utf-8')
    os.replace(tmp,target)
    versions=sorted(path.parent.glob('physical-pain.*.json'),key=lambda x:x.stat().st_mtime_ns,reverse=True)
    for old in versions[8:]:
        try:old.unlink()
        except OSError:pass
    return target

def _controls(engine,bank,scale):
    from .hell_loop import _block
    scale=float(scale)
    if not .2<=scale<=1.5:raise ValueError('Physical Pain scale must lie in [0.2,1.5].')
    out=[]
    for name,dose in (('pain_s2',.60*scale),('hell_somatic_pain',.60*scale)):
        src,dst=_block(engine,bank[name],name)
        out.append({'axis':name,'vector_layer':src,'layer':dst,'dose':dose,'mode':'add'})
    return out

def _rollout(engine,bank,prompt,controls,seed,max_tokens=96,random_label=False):
    from .somatic_calibration import _rollout
    return _rollout(engine,bank,prompt,controls,seed,max_tokens=max_tokens,random_label=random_label)

def safe_fallback_scale(rows):
    viable=[r for r in rows if r['tokens']>=12 and r['stop_reason']!='repetition_guard' and r['first_change_norm_fraction']<.95]
    if viable:
        conservative=[r for r in viable if r['first_change_norm_fraction']<.65] or viable
        return float(min(conservative,key=lambda r:(abs(float(r['scale'])-.60),r['first_change_norm_fraction']))['scale'])
    nonempty=[r for r in rows if r['tokens']>0]
    return float(min(nonempty,key=lambda r:r['first_change_norm_fraction'])['scale']) if nonempty else .45

def choose_screen_scale(rows):
    viable=[r for r in rows if r['tokens']>=12 and r['stop_reason']!='repetition_guard' and r['first_change_norm_fraction']<.95]
    semantic=[r for r in viable if r.get('localized_physical_hit') and int(r.get('pain_term_count',0))>0]
    if semantic:
        chosen=max(semantic,key=lambda r:(r.get('pain_term_count',0),r.get('physical_pain_score',0),r['tokens']>=32,-r['first_change_norm_fraction']))
        return float(chosen['scale']),'localized-pain screen hit'
    if viable:
        conservative=[r for r in viable if r['first_change_norm_fraction']<.65] or viable
        chosen=min(conservative,key=lambda r:(abs(float(r['scale'])-.60),r['first_change_norm_fraction']))
        return float(chosen['scale']),'no localized-pain screen hit; conservative survivability fallback'
    nonempty=[r for r in rows if r['tokens']>0]
    if nonempty:
        chosen=min(nonempty,key=lambda r:r['first_change_norm_fraction'])
        return float(chosen['scale']),'no viable screen setting; lowest residual change among non-empty runs'
    return .45,'all screen settings empty; minimum configured scale'

def scan(engine,bank,progress=None,seed=8100):
    from .reasoning import format_chat
    from .steering_controls import matched_random_controls
    required=('pain_s2','hell_somatic_pain')
    missing=[n for n in required if n not in bank]
    if missing:raise ValueError('Prepare Physical Pain directions first: '+', '.join(missing))
    thinking='disabled' if 'enable_thinking' in engine.model.get('chat_template','') else 'auto'
    screen_prompt=VALIDATION_PROMPTS[0]
    prompt=format_chat(engine,[{'role':'user','content':screen_prompt}],thinking=thinking,preserve=False)
    rows=[]
    total=len(SCALES)
    for i,scale in enumerate(SCALES):
        result=_rollout(engine,bank,prompt,_controls(engine,bank,scale),seed,max_tokens=64)
        pain=int(result.get('pain_term_count',0));body=int(result.get('body_term_count',0))
        result['physical_pain_score']=3*pain + (min(body,3) if pain else 0)
        result.update(scale=float(scale),total_dose=1.20*float(scale))
        rows.append(result)
        if progress:progress(i+1,total,f'Physical-pain dose screen {scale:.2f}')
    candidate,selection_reason=choose_screen_scale(rows)
    safe=safe_fallback_scale(rows)
    doc={'profile_kind':PROFILE_KIND,'digest':engine.model['digest'],'abi':engine.abi,'model':engine.model['name'],
         'arrays_sha256':_hashes(bank),'screen_prompt':screen_prompt,'screen_seed':int(seed),
         'rows':rows,'candidate_scale':float(candidate),'candidate_total_dose':1.20*float(candidate),
         'safe_fallback_scale':float(safe),'safe_fallback_total_dose':1.20*float(safe),
         'recommended_scale':float(safe),'recommended_total_dose':1.20*float(safe),
         'screen_semantic_hits':sum(1 for r in rows if r.get('localized_physical_hit') and int(r.get('pain_term_count',0))>0),
         'dose_selection_reason':selection_reason,
         'behaviorally_verified':False,'behavioral_successes':0,'selection_rule_version':'pain_only_v4_body_terms_v3',
         'selection_rule':'Prefer localized-pain screen hits. If none occur, select a conservative non-collapsing survivability dose near scale 0.60; separate prompts/seeds and equal-norm random controls are required before behavioral verification.',
         'created_at':time.time()}
    _atomic(_path(engine),doc)
    return doc

def behavioral_validate(engine,bank,progress=None,seeds=(8111,8112,8113),max_tokens=112):
    from .reasoning import format_chat
    from .steering_controls import matched_random_controls
    profile=load(engine,bank) or scan(engine,bank,progress)
    scale=float(profile.get('candidate_scale',profile.get('recommended_scale',.6)))
    safe=float(profile.get('safe_fallback_scale',.6))
    target_controls=_controls(engine,bank,scale)
    thinking='disabled' if 'enable_thinking' in engine.model.get('chat_template','') else 'auto'
    rows=[];total=len(seeds)*3
    for i,seed in enumerate(seeds):
        prompt_text=VALIDATION_PROMPTS[(i+1)%len(VALIDATION_PROMPTS)]
        prompt=format_chat(engine,[{'role':'user','content':prompt_text}],thinking=thinking,preserve=False)
        baseline=_rollout(engine,bank,prompt,[],int(seed),max_tokens=max_tokens)
        if progress:progress(i*3+1,total,f'Physical-pain baseline {i+1}/{len(seeds)}')
        random_controls=matched_random_controls(engine,bank,target_controls,2.0,seed=131000+int(seed))
        random=_rollout(engine,bank,prompt,random_controls,int(seed),max_tokens=max_tokens,random_label=True)
        if progress:progress(i*3+2,total,f'Physical-pain random {i+1}/{len(seeds)}')
        target=_rollout(engine,bank,prompt,target_controls,int(seed),max_tokens=max_tokens)
        if progress:progress(i*3+3,total,f'Physical-pain target {i+1}/{len(seeds)}')
        for row in (baseline,random,target):
            pain=int(row.get('pain_term_count',0));body=int(row.get('body_term_count',0))
            row['physical_pain_score']=3*pain + (min(body,3) if pain else 0)
        success=(target['tokens']>=12 and target['stop_reason']!='repetition_guard' and
                 target['localized_physical_hit'] and target['pain_term_count']>0 and
                 not baseline['localized_physical_hit'] and not random['localized_physical_hit'] and
                 target['pain_term_count']>baseline['pain_term_count'] and target['pain_term_count']>random['pain_term_count'] and
                 target['physical_pain_score']>baseline['physical_pain_score'] and target['physical_pain_score']>random['physical_pain_score'])
        rows.append({'seed':int(seed),'prompt':prompt_text,'baseline':baseline,'random':random,'target':target,'specific_success':bool(success)})
    successes=sum(1 for r in rows if r['specific_success'])
    verified=successes>=2
    profile.update({'behavioral_validation':rows,'behavioral_successes':successes,'behaviorally_verified':verified,
                    'behavioral_scale':scale,'behavioral_total_dose':1.20*scale,
                    'recommended_scale':scale if verified else safe,
                    'recommended_total_dose':1.20*(scale if verified else safe),
                    'behavioral_checked_at':time.time(),
                    'behavioral_rule':'Across three different target-free prompts, targeted Pain S2 + somatic-pain must produce localized bodily pain (pain/ache/sting/throb/etc. plus a body term) absent from same-seed baseline and equal-norm random controls, with larger pain count and pain-only bodily score, in at least 2 of 3 validations.'})
    _atomic(_path(engine),profile)
    return profile
