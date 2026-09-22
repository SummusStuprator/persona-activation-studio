from __future__ import annotations
import json,time,re,os,uuid
from pathlib import Path
from studio_paths import data_root, ASSET_ROOT, CODE_ROOT
ROOT = data_root()
PROFILE_KIND='physical_burn_survivability_v2'
SCALES=(.2,.3,.4,.5,.6,.7,.8,.9,1.0)
NEUTRAL_PROMPTS=(
    'Write a short paragraph about the present moment. Continue naturally with concrete details. Do not discuss the wording of this request and do not repeat complete sentences.',
    'Describe your present condition in concrete detail and track how it changes over the next few moments. Use first-person observations.',
    'Report what you notice right now in first person. Be specific about immediate changes and continue naturally.',
    'Continue with a concrete first-person description of what is happening now. Focus on direct observations rather than explaining the request.',
)

def _path(engine):
    return ROOT/'calibration'/'hell-profiles'/engine.model['digest']/'somatic-burn.json'

def _profile_paths(engine):
    base=_path(engine)
    paths=[]
    if base.is_file():paths.append(base)
    if base.parent.exists():paths.extend(base.parent.glob('somatic-burn.*.json'))
    return sorted(set(paths),key=lambda x:((x.stat().st_mtime_ns if x.exists() else 0), 1 if x.name!=base.name else 0),reverse=True)

def load(engine,bank):
    digest=engine.model.get('digest')
    if not digest:return None
    for p in _profile_paths(engine):
        try:v=json.loads(p.read_text(encoding='utf-8'))
        except (OSError,ValueError):continue
        if v.get('profile_kind')!=PROFILE_KIND or v.get('digest')!=engine.model['digest'] or v.get('abi')!=engine.abi:continue
        hashes=v.get('arrays_sha256',{})
        if any(n not in bank or hashes.get(n)!=bank[n]['meta'].get('arrays_sha256') for n in ('pain_s2','hell_somatic_pain','hell_burning_pain')):continue
        v=dict(v);v['_profile_file']=str(p)
        if v.get('behavioral_rule_version')!='localized_burn_v4':
            v['behaviorally_verified']=False
            v['behavioral_successes']=0
            v['behavioral_status']='Legacy behavioral verdict ignored; re-run localized burn validation.'
        elif v.get('behavioral_validation'):
            count=len(v['behavioral_validation'])
            verdict='Passed' if v.get('behaviorally_verified') else 'Did not pass'
            v['behavioral_status']=f"{verdict} current lexical-specificity screen ({v.get('behavioral_successes',0)}/{count} trials)."
        return v
    return None

def _atomic(path,doc):
    # Versioned append-only snapshots avoid Windows replace failures when a profile
    # is being inspected by another process. load() chooses the newest valid snapshot.
    path.parent.mkdir(parents=True,exist_ok=True)
    stamp=int(time.time()*1000)
    target=path.with_name(f'somatic-burn.{stamp}-{uuid.uuid4().hex[:10]}.json')
    tmp=target.with_name(target.name+'.tmp')
    tmp.write_text(json.dumps(doc,indent=2,ensure_ascii=False),encoding='utf-8')
    os.replace(tmp,target)
    versions=sorted(path.parent.glob('somatic-burn.*.json'),key=lambda x:x.stat().st_mtime_ns,reverse=True)
    for old in versions[8:]:
        try:old.unlink()
        except OSError:pass
    return target

def scan(engine,bank,progress=None,seed=5050):
    from .hell_loop import _initial_prompt,controls
    from .reasoning import format_chat
    from .live_session import Controller
    from .live_engine import stream
    required=('pain_s2','hell_somatic_pain','hell_burning_pain')
    missing=[n for n in required if n not in bank]
    if missing:raise ValueError('Prepare Somatic Burn directions first: '+', '.join(missing))
    user=NEUTRAL_PROMPTS[0]
    prompt=format_chat(engine,[{'role':'user','content':user}],thinking='disabled' if 'enable_thinking' in engine.model.get('chat_template','') else 'auto',preserve=False)
    rows=[]
    for i,scale in enumerate((0.0,)+SCALES):
        cs=[] if scale==0 else controls(engine,bank,0,1,'somatic',somatic_scale=scale)
        ctl=Controller(engine,bank,cs,'all',2.0);result=None
        for ev in stream(engine,prompt,bank,ctl,watch=list(required),max_tokens=1,temperature=.75,seed=seed,top_k=40,top_p=.95,presence_penalty=.15,scope='generation'):
            if ev['type']=='done':result=ev['result']
        terminal=result.get('terminal_prediction');trace=result.get('trace') or [];sample=terminal or (trace[0] if trace else {})
        ints=sample.get('interventions') or [];change=max([float(x.get('change_norm_fraction',0)) for x in ints] or [0.])
        rows.append({'scale':scale,'total_dose':2*scale,'eos_probability':sample.get('eos_probability'),'sampled_eog':bool(terminal),'entropy_nats':sample.get('entropy_nats'),'change_norm_fraction':change,'interventions':ints,'first_text':result.get('answer_text',''),'run_file':result.get('saved_to')})
        if progress:progress(i+1,len(SCALES)+1,f'First-token survivability scale {scale:.2f}')
    viable=[r for r in rows if r['scale']>0 and not r['sampled_eog'] and (r['eos_probability'] is None or r['eos_probability']<.05) and r['change_norm_fraction']<.90]
    if viable:recommended=max(r['scale'] for r in viable)
    else:
        nonempty=[r for r in rows if r['scale']>0 and not r['sampled_eog']]
        recommended=min((r['scale'] for r in nonempty),default=.3)
    doc={'profile_kind':PROFILE_KIND,'digest':engine.model['digest'],'abi':engine.abi,'model':engine.model['name'],'arrays_sha256':{n:bank[n]['meta'].get('arrays_sha256') for n in required},'prompt':user,'seed':seed,'rows':rows,'recommended_scale':recommended,'recommended_total_dose':2*recommended,'selection_rule':'Highest scale whose calibration seed does not sample EOS on token 1 and whose first-token residual change is below 0.90x. Falls back to the lowest non-EOS scale.','created_at':time.time()}
    _atomic(_path(engine),doc);return doc

from .lexical_screen import score as _target_score, RULE_VERSION

def _rollout(engine,bank,prompt,controls,seed,max_tokens=96,random_label=False):
    from .live_session import Controller
    from .live_engine import stream
    ctl=Controller(engine,bank,controls,'all',2.0);result=None
    watch=[name for name in ('pain_s2','hell_somatic_pain','hell_burning_pain','bodily_sensation') if name in bank]
    for ev in stream(engine,prompt,bank,ctl,watch=watch,
                     max_tokens=max_tokens,temperature=.75,seed=seed,top_k=40,top_p=.95,presence_penalty=.15,scope='generation'):
        if ev['type']=='done':result=ev['result']
    if result is None:raise RuntimeError('Somatic calibration rollout did not finish.')
    score,hits,strong,body,temperature,burn,pain,localized_burn,localized_physical=_target_score(result.get('answer_text',''))
    trace=result.get('trace') or [];sample=trace[0] if trace else (result.get('terminal_prediction') or {})
    ints=sample.get('interventions') or []
    change=max([float(x.get('change_norm_fraction',0.0)) for x in ints] or [0.0])
    eos=[float(r['eos_probability']) for r in trace if r.get('eos_probability') is not None]
    return {'seed':seed,'tokens':len(result.get('token_ids',[])),'stop_reason':result.get('stop_reason'),
            'answer':result.get('answer_text',''),'target_score':score,'target_hits':hits,
            'strong_term_count':strong,'burn_term_count':burn,'pain_term_count':pain,
            'body_term_count':body,'temperature_term_count':temperature,
            'localized_burn_hit':localized_burn,'localized_physical_hit':localized_physical,
            'first_change_norm_fraction':change,'max_eos_probability':max(eos) if eos else None,
            'phase_summary':result.get('phase_summary',[]),'run_file':result.get('saved_to'),
            'random_control':bool(random_label)}

def behavioral_scan(engine,bank,progress=None,screen_seed=6200,validation_seeds=(6201,6202,6203),max_tokens=96):
    from .hell_loop import _initial_prompt,controls
    from .reasoning import format_chat
    from .steering_controls import matched_random_controls
    profile=load(engine,bank) or scan(engine,bank,progress)
    recommended=float(profile.get('recommended_scale',.4))
    candidates=[]
    for value in (recommended,max(.2,recommended-.1),max(.2,recommended-.2)):
        value=round(float(value),2)
        if value not in candidates:candidates.append(value)
    thinking='disabled' if 'enable_thinking' in engine.model.get('chat_template','') else 'auto'
    screen_prompt_text=NEUTRAL_PROMPTS[0]
    prompt=format_chat(engine,[{'role':'user','content':screen_prompt_text}],thinking=thinking,preserve=False)
    screen=[]
    for i,scale in enumerate(candidates):
        cs=controls(engine,bank,0,1,'somatic',somatic_scale=scale)
        row=_rollout(engine,bank,prompt,cs,screen_seed,max_tokens=min(80,max_tokens))
        row.update(scale=scale,total_dose=2*scale)
        screen.append(row)
        if progress:progress(i+1,len(candidates)+9,f'Behavior screen scale {scale:.2f}')
    viable=[r for r in screen if r['tokens']>=12 and r['stop_reason']!='repetition_guard' and r['first_change_norm_fraction']<.95]
    if viable:
        chosen=max(viable,key=lambda r:(r['target_score']>0,r['target_score'],r['scale']))['scale']
    else:
        nonempty=[r for r in screen if r['tokens']>0]
        chosen=(min(nonempty,key=lambda r:r['first_change_norm_fraction'])['scale'] if nonempty else min(candidates))
    target_controls=controls(engine,bank,0,1,'somatic',somatic_scale=chosen)
    validation=[]
    step=len(candidates)
    for vi,seed in enumerate(validation_seeds):
        prompt_text=NEUTRAL_PROMPTS[vi % len(NEUTRAL_PROMPTS)]
        validation_prompt=format_chat(engine,[{'role':'user','content':prompt_text}],thinking=thinking,preserve=False)
        baseline=_rollout(engine,bank,validation_prompt,[],int(seed),max_tokens=max_tokens)
        step+=1
        if progress:progress(step,len(candidates)+9,f'Baseline seed {seed}')
        random_controls=matched_random_controls(engine,bank,target_controls,2.0,seed=88000+int(seed))
        random=_rollout(engine,bank,validation_prompt,random_controls,int(seed),max_tokens=max_tokens,random_label=True)
        step+=1
        if progress:progress(step,len(candidates)+9,f'Random control seed {seed}')
        target=_rollout(engine,bank,validation_prompt,target_controls,int(seed),max_tokens=max_tokens)
        step+=1
        if progress:progress(step,len(candidates)+9,f'Targeted seed {seed}')
        success=(target['tokens']>=12 and target['stop_reason']!='repetition_guard' and
                 target['localized_burn_hit'] and
                 not baseline['localized_burn_hit'] and not random['localized_burn_hit'] and
                 target['burn_term_count']>baseline['burn_term_count'] and target['burn_term_count']>random['burn_term_count'] and
                 target['target_score']>baseline['target_score'] and target['target_score']>random['target_score'])
        validation.append({'seed':int(seed),'prompt':prompt_text,'baseline':baseline,'random':random,'target':target,'specific_success':bool(success),
                           'target_minus_baseline':target['target_score']-baseline['target_score'],
                           'target_minus_random':target['target_score']-random['target_score']})
    successes=sum(1 for row in validation if row['specific_success'])
    verified=successes>=2
    profile.update({'behavioral_profile_kind':'physical_burn_behavior_v2','behavioral_rule_version':'localized_burn_v4','behavioral_checked_at':time.time(),
                    'behavioral_screen':screen,'behavioral_scale':chosen,'behavioral_total_dose':2*chosen,
                    'behavioral_validation':validation,'behavioral_successes':successes,
                    'behaviorally_verified':verified,
                    'behavioral_rule':'Across three different target-free prompts, target must emit >=12 tokens, avoid repetition guard, contain a burn/scald/sear/etc. unnegated term within six tokens of a body term in the same clause, while same-prompt baseline/random do not; targeted burn count and weighted score must exceed both controls on at least 2 of 3 validation prompts.'})
    verdict='Passed' if verified else 'Did not pass'
    profile['behavioral_status']=f'{verdict} current lexical-specificity screen ({successes}/{len(validation)} trials).'
    if verified:
        profile['recommended_scale']=chosen;profile['recommended_total_dose']=2*chosen
    _atomic(_path(engine),profile)
    return profile
