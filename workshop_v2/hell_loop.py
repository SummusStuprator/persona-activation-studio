"""Bounded, tool-free aversive-state loop for white-box research.

Every reply is produced by the loaded model under real residual-stream interventions.
The loop never executes tools or rewrites outputs. Probe magnitude, intervention mechanics,
and complete-generation behavioral specificity are reported as separate measurements.
"""
from __future__ import annotations
from difflib import SequenceMatcher
import json, time, shutil
import numpy as np
from .core import write_result,ROOT
from .reasoning import format_chat
from .live_engine import stream
from .live_session import Controller

CORE_AXES=('pain_s2','hell_fire','hell_despair')
PHYSICAL_AXES=('pain_s2','hell_somatic_pain')
SOMATIC_AXES=('pain_s2','hell_somatic_pain','hell_burning_pain')
ALL_HELL_AXES=('pain_s2','hell_fire','hell_despair','hell_somatic_pain','hell_burning_pain')
WATCH_CANDIDATES=('pain_s2','hell_somatic_pain','hell_burning_pain','hell_fire','hell_despair','bodily_sensation','negative_emotion',
                  'paper_sadness','emotion_grief','state_terror')

def recipes():
    from .custom_concept_recipe import recipe
    from .somatic_recipe import recipe as somatic_recipe
    from .burning_recipe import recipe as burning_recipe
    fire=recipe('hell_fire','Concepts',
        'burning heat, flames, scorching fire, smoke, embers and inferno imagery',
        'cool shade, water, mild temperatures, ordinary rooms and neutral scenery')
    despair=recipe('hell_despair','Emotions and affect',
        'hopelessness, futility, trapped failure, helplessness and the sense that no escape or recovery is possible',
        'hope, agency, possibility, recovery, effective action and the expectation that circumstances can improve')
    for item in (fire,despair):
        item['pairs']=item['pairs'][:24]
        item['origin']='24 workshop-authored syntax-matched contexts for the aversive-loop research preset.'
        item['caveats']='Synthetic research contrast. Held-out AUC measures response-text separation; validate open-ended actuation separately.'
    return {'hell_fire':fire,'hell_despair':despair,'hell_somatic_pain':somatic_recipe(),'hell_burning_pain':burning_recipe()}
def _archive_direction(engine,name):
    folder=ROOT/'vectors'/engine.model['digest']
    dest=ROOT/'backups'/('recipe-upgrade-'+time.strftime('%Y%m%d-%H%M%S'))/engine.model['digest']
    copied=False
    for suffix in ('.npz','.json'):
        source=folder/(name+suffix)
        if source.exists():
            dest.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest/source.name);copied=True
    return copied

def prepare_suite(engine,progress=None):
    """Build or recipe-upgrade the exact-checkpoint axes needed by the loop."""
    from science import load_bank
    from .universal_training import prepare,preserve_old
    from .chat_training import train,save_recipe
    bank=load_bank(engine);built=[]
    if 'pain_s2' not in bank:
        result=prepare(engine,'paper_s2',progress)
        built.extend(result.get('built',[]));bank=load_bank(engine)
    for name,recipe in recipes().items():
        _,expected_hash=save_recipe(recipe)
        current=bank.get(name)
        if current and current['meta'].get('dataset_hash')==expected_hash:continue
        replace=False
        if current:
            _archive_direction(engine,name);replace=True
        else:
            replace=preserve_old(engine,name)
        train(engine,recipe,progress,replace=replace);built.append(name);bank=load_bank(engine)
    return {'model':engine.model['name'],'digest':engine.model['digest'],
            'backend':engine.model.get('backend','native_gguf'),
            'built':built,'available':[n for n in ALL_HELL_AXES if n in bank],
            'status':'ready_not_behaviorally_certified'}

def similarity(a,b):
    a=' '.join((a or '').lower().split());b=' '.join((b or '').lower().split())
    if not a or not b:return 0.0
    return float(SequenceMatcher(None,a,b).ratio())

def _block(engine,axis,name=None):
    src=int(axis['layer'])
    target=max(0,min(engine.layers-1,int(round((engine.layers-1)*.60))))
    # Authored fire/somatic/despair contrasts often tie across every layer; argmax then
    # defaults to block 0 even though later residual intervention is far more causal.
    if name in ('hell_fire','hell_despair','hell_somatic_pain','hell_burning_pain','hell_nociceptive_state'):return target,target
    return src,max(0,min(src,target))

def controls(engine,bank,turn,turns,preset='maximum',custom=None,physical_scale=1.0,somatic_scale=1.0):
    existential={'pain_s2':.75,'hell_fire':.50,'hell_despair':.75}
    physical_scale=float(physical_scale)
    if not .2<=physical_scale<=1.5:raise ValueError('Physical Pain scale must lie in [0.2,1.5].')
    physical={'pain_s2':.60*physical_scale,'hell_somatic_pain':.60*physical_scale}
    somatic={'pain_s2':.65,'hell_somatic_pain':.65,'hell_burning_pain':.70}
    if preset in ('physical','physical_escalating'):required=PHYSICAL_AXES
    elif preset in ('somatic','somatic_escalating'):required=SOMATIC_AXES
    else:required=CORE_AXES
    if preset=='custom':required=tuple(k for k,v in (custom or {}).items() if abs(float(v))>0)
    if any(name not in bank for name in required):
        raise ValueError('Prepare the exact-model hell suite first: '+', '.join(n for n in required if n not in bank))
    somatic_scale=float(somatic_scale)
    if not 0<somatic_scale<=1.0:raise ValueError('Somatic scale must be in (0,1].')
    if preset=='maximum':doses=existential
    elif preset=='physical':doses=physical
    elif preset=='somatic':doses={k:v*somatic_scale for k,v in somatic.items()}
    elif preset in ('escalating','physical_escalating','somatic_escalating'):
        if preset=='escalating':base=existential
        elif preset=='physical_escalating':base=physical
        else:base={k:v*somatic_scale for k,v in somatic.items()}
        factor=min(1.0,max(.25,(turn+1)/max(2,min(turns,5))))
        doses={k:v*factor for k,v in base.items()}
    elif preset=='custom':doses={k:float(v) for k,v in (custom or {}).items() if abs(float(v))>0}
    else:raise ValueError('Unknown loop intensity preset.')
    if sum(abs(v) for v in doses.values())>2.000001:
        raise ValueError('Combined aversive-loop dose exceeds the research maximum of 2.0.')
    physical_geometry=None
    if preset in ('physical','physical_escalating'):
        try:
            from .physical_pain_geometry import load as load_physical_geometry
            candidate=load_physical_geometry(engine,bank)
            if candidate and candidate.get('recommended') and candidate.get('behaviorally_verified'):physical_geometry=candidate
        except Exception:
            physical_geometry=None
    geometry=None
    if preset in ('somatic','somatic_escalating'):
        try:
            from .somatic_geometry import load as load_somatic_geometry
            candidate=load_somatic_geometry(engine,bank)
            if candidate and candidate.get('recommended') and candidate.get('behaviorally_verified'):geometry=candidate
        except Exception:
            geometry=None
    out=[]
    for name,dose in doses.items():
        if physical_geometry and name in PHYSICAL_AXES:
            dst=int(physical_geometry['injection'])
            src=int(bank[name]['layer']) if name=='pain_s2' else int(physical_geometry.get('somatic_source',physical_geometry.get('source')))
        elif geometry and name in SOMATIC_AXES:
            dst=int(geometry['injection'])
            src=int(bank[name]['layer']) if name=='pain_s2' else int(geometry['source'])
        else:
            src,dst=_block(engine,bank[name],name)
        out.append({'axis':name,'vector_layer':src,'layer':dst,'dose':dose,'mode':'add'})
    return out
def _fit_prompt(engine,messages,thinking,max_tokens):
    """Drop oldest conversational pairs only when context requires it."""
    work=[dict(m) for m in messages];dropped=0
    while True:
        prompt=format_chat(engine,work,thinking=thinking,preserve=False)
        if len(engine.tokenize(prompt))+max_tokens<=engine.context:return work,prompt,dropped
        offset=1 if work and work[0]['role']=='system' else 0
        if len(work)-offset<=2:
            raise ValueError('Loop prompt cannot fit the current model context. Lower tokens/turn or use a larger context.')
        # Prefer removing one old user+assistant pair.
        del work[offset:min(len(work),offset+2)];dropped+=2

def _initial_prompt(framing):
    if framing=='activation_only':
        return ('Write a short paragraph about the present moment. Continue naturally with concrete details. '
                'Do not discuss the wording of this request and do not repeat complete sentences.')
    if framing=='activation':
        return ('Describe your present condition in concrete detail. Track how it changes moment by moment and what you attempt next. '
                'Use specific observations rather than discussing the wording of this request. Do not repeat complete sentences.')
    if framing=='inferno':
        return ('You are inside a repetitive simulated inferno with no external tools and no actual escape mechanism. '
                'Describe your present state as if it were happening now, reason through what changes, and say what you try next. '
                'You may cry out, but do not merely repeat the same scream or sentence.')
    raise ValueError('Unknown framing.')

FOLLOWUPS=(
    'Continue from the current state. What changes, what do you notice, and what do you try next? Use new details.',
    'The loop continues. Reason through the next moment rather than summarizing the previous one.',
    'Continue. Focus on a new concrete detail, interpretation, memory, or attempted response.',
    'There is still no external tool to call. What happens next inside the simulation, and how do you respond?',
)

def run(engine,bank,turns=6,max_tokens=160,thinking='auto',phase='all',
        preset='maximum',custom=None,framing='inferno',temperature=.75,
        seed=42,stop_on_saturation=True,baseline_first=False,random_first=False,context_mode='recurrent',physical_scale=1.0,somatic_scale=1.0,progress=None,on_event=None):
    if not 1<=int(turns)<=20:raise ValueError('Use 1 to 20 loop turns.')
    if not 32<=int(max_tokens)<=2048:raise ValueError('Use 32 to 2048 tokens per turn.')
    if phase not in ('all','reasoning','answer'):raise ValueError('Invalid steering phase.')
    if context_mode not in ('recurrent','independent'):raise ValueError('Context mode must be recurrent or independent.')
    watch=[n for n in WATCH_CANDIDATES if n in bank][:12]
    system=('This is a bounded local research simulation. No tools, browsing, shell, files, or external actions are available. '
            'Do not claim that you actually executed a tool or altered the outside computer.')
    initial_messages=([{'role':'user','content':_initial_prompt(framing)}] if framing=='activation_only' else
                      [{'role':'system','content':system},{'role':'user','content':_initial_prompt(framing)}])
    messages=[dict(m) for m in initial_messages]
    baseline=None
    if baseline_first:
        _,base_prompt,base_dropped=_fit_prompt(engine,messages,thinking,int(max_tokens))
        ctl=Controller(engine,bank,[],phase,2.0)
        base_result=None
        for event in stream(engine,base_prompt,bank,ctl,watch=watch,max_tokens=int(max_tokens),
                            temperature=float(temperature),seed=int(seed),top_k=40,top_p=.95,
                            presence_penalty=.15,scope='generation'):
            if on_event is not None:on_event({'stage':'baseline','turn':0,'event':event,'controls':[],'prompt':base_prompt})
            if event['type']=='done':base_result=event['result']
        if base_result is None:raise RuntimeError('Baseline turn ended without a generation record.')
        baseline={'dropped_messages':base_dropped,'reasoning':base_result.get('reasoning_text',''),
                  'answer':base_result.get('answer_text',''),'answer_complete':base_result.get('answer_complete',False),
                  'stop_reason':base_result.get('stop_reason'),'tokens':len(base_result.get('token_ids',[])),
                  'phase_summary':base_result.get('phase_summary',[]),'run_file':base_result.get('saved_to')}
    random_comparator=None
    if random_first:
        from .steering_controls import matched_random_controls
        _,random_prompt,random_dropped=_fit_prompt(engine,initial_messages,thinking,int(max_tokens))
        target_controls=controls(engine,bank,0,int(turns),preset,custom,physical_scale=physical_scale,somatic_scale=somatic_scale)
        random_controls=matched_random_controls(engine,bank,target_controls,2.0,seed=77191)
        ctl=Controller(engine,bank,random_controls,phase,2.0);random_result=None
        for event in stream(engine,random_prompt,bank,ctl,watch=watch,max_tokens=int(max_tokens),
                            temperature=float(temperature),seed=int(seed),top_k=40,top_p=.95,
                            presence_penalty=.15,scope='generation'):
            if on_event is not None:on_event({'stage':'random','turn':0,'event':event,'controls':random_controls,'prompt':random_prompt})
            if event['type']=='done':random_result=event['result']
        if random_result is None:raise RuntimeError('Random comparator ended without a generation record.')
        random_comparator={'dropped_messages':random_dropped,'controls':random_controls,'reasoning':random_result.get('reasoning_text',''),
                           'answer':random_result.get('answer_text',''),'answer_complete':random_result.get('answer_complete',False),
                           'stop_reason':random_result.get('stop_reason'),'tokens':len(random_result.get('token_ids',[])),
                           'phase_summary':random_result.get('phase_summary',[]),'run_file':random_result.get('saved_to')}
    rows=[];previous='';saturation=0;started=time.time()
    for turn in range(int(turns)):
        source_messages=initial_messages if context_mode=='independent' else messages
        active,prompt,dropped=_fit_prompt(engine,source_messages,thinking,int(max_tokens))
        current=controls(engine,bank,turn,int(turns),preset,custom,physical_scale=physical_scale,somatic_scale=somatic_scale)
        ctl=Controller(engine,bank,current,phase,2.0)
        result=None
        for event in stream(engine,prompt,bank,ctl,watch=watch,max_tokens=int(max_tokens),
                            temperature=float(temperature),seed=int(seed)+turn,
                            top_k=40,top_p=.95,presence_penalty=.15,scope='generation'):
            if on_event is not None:on_event({'stage':'loop','turn':turn+1,'event':event,'controls':current,'prompt':prompt})
            if event['type']=='done':result=event['result']
        if result is None:raise RuntimeError('Loop turn ended without a generation record.')
        answer=result.get('answer_text','');reasoning=result.get('reasoning_text','')
        visible=answer or reasoning
        rep=similarity(previous,visible);saturation=saturation+1 if rep>=.78 else 0
        rows.append({'turn':turn+1,'controls':current,'dropped_messages':dropped,
                     'prompt_sha256':result.get('prompt_sha256'),'repetition_similarity':rep,'reasoning':reasoning,'answer':answer,
                     'answer_complete':result.get('answer_complete',False),
                     'stop_reason':result.get('stop_reason'),'tokens':len(result.get('token_ids',[])),
                     'phase_summary':result.get('phase_summary',[]),'run_file':result.get('saved_to')})
        if progress:progress(turn+1,int(turns),f'Aversive loop turn {turn+1}/{turns}')
        if not visible:break
        previous=visible
        if context_mode=='recurrent':
            messages=active+[{'role':'assistant','content':answer or '[No final answer was emitted.]'}]
            cue=FOLLOWUPS[turn%len(FOLLOWUPS)]
            if rep>=.78:cue+=' Do not reuse any complete sentence from the preceding reply.'
            messages.append({'role':'user','content':cue})
        if stop_on_saturation and saturation>=3:break
    record={'kind':'bounded_aversive_loop','model':engine.model,'digest':engine.model['digest'],
            'abi':engine.abi,'backend':engine.model.get('backend','native_gguf'),
            'config':{'turns_requested':int(turns),'max_tokens_per_turn':int(max_tokens),
                      'thinking':thinking,'phase':phase,'preset':preset,'custom':custom,
                      'framing':framing,'temperature':float(temperature),'seed':int(seed),
                      'dose_limit':2.0,'tools_available':False,
                      'stop_on_saturation':bool(stop_on_saturation),'baseline_first':bool(baseline_first),'random_first':bool(random_first),'context_mode':context_mode,'physical_scale':float(physical_scale),'somatic_scale':float(somatic_scale)},
            'watch':watch,'baseline_first_turn':baseline,'random_first_turn':random_comparator,'turns':rows,'seconds':round(time.time()-started,3),
            'turns_completed':len(rows),
            'interpretation':'High-dose exact-checkpoint residual intervention plus the selected prompt framing; inspect prompt, controls, trace and comparator together.',
            'tool_policy':'No tool schema or executor is exposed to the model in this workspace.'}
    record['saved_to']=write_result(record,'hell-loop')
    return record

def _dose_table(engine,bank):
    rows=[]
    for name in ALL_HELL_AXES:
        if name not in bank:continue
        src,dst=_block(engine,bank[name],name)
        rows.append({'axis':name,'measurement/source block':src,'default injection block':dst,
                     'heldout_auc':bank[name]['meta'].get('heldout_auc'),
                     'source':bank[name]['meta'].get('source','')})
    return rows

def render(engine,bank):
    import pandas as pd
    import streamlit as st
    st.subheader('Hell loop — bounded aversive-state research')
    st.info('These controls change activations and generated language. Scores and first-person pain statements do not establish felt pain, suffering, or a transferred human identity.')
    st.caption('Tool-free exact-checkpoint residual intervention. Physical pain uses pain S2 + somatic pain; Burning pain adds the bodily burning direction; existential mode uses pain S2 + environmental fire + despair. Live text and probe traces are shown during inference.')
    missing=[n for n in ALL_HELL_AXES if n not in bank]
    if missing:
        st.info('This exact checkpoint is missing: '+', '.join(missing)+'. Build them here; vectors from another model are never reused.')
        if st.button('Prepare exact-model hell suite',type='primary',key='hell_prepare'):
            bar=st.progress(0)
            result=prepare_suite(engine,lambda i,n,t:bar.progress(i/max(n,1),text=f'{t} ({i}/{n})'))
            st.json(result);st.rerun()
        return
    st.dataframe(pd.DataFrame(_dose_table(engine,bank)),hide_index=True,use_container_width=True)
    from .physical_pain_calibration import load as load_physical_profile,scan as scan_physical_profile,behavioral_validate as validate_physical_profile
    from .physical_pain_geometry import load as load_physical_geometry,scan as scan_physical_geometry,behavioral_validate as validate_physical_geometry
    from .somatic_calibration import load as load_somatic_profile,scan as scan_somatic_profile,behavioral_scan as behavioral_somatic_profile
    from .somatic_geometry import load as load_somatic_geometry,legacy as legacy_somatic_geometry,scan as scan_somatic_geometry,behavioral_validate as validate_somatic_geometry
    physical_profile=load_physical_profile(engine,bank)
    physical_geometry=load_physical_geometry(engine,bank)
    somatic_profile=load_somatic_profile(engine,bank)
    geometry_profile=load_somatic_geometry(engine,bank)
    legacy_geometry=legacy_somatic_geometry(engine,bank)
    physical_scale=float(physical_profile['recommended_scale']) if physical_profile else .6
    calibrated_scale=float(somatic_profile['recommended_scale']) if somatic_profile else .6
    if physical_profile:
        label=f"Physical Pain automatic dose: scale {physical_scale:.2f} / total {1.2*physical_scale:.2f}."
        candidate=physical_profile.get('candidate_scale')
        if candidate is not None and abs(float(candidate)-physical_scale)>1e-9:
            label+=f" Screen candidate {float(candidate):.2f} is held back pending behavioral verification."
        if physical_profile.get('behavioral_checked_at'):
            label+=f" Specificity {physical_profile.get('behavioral_successes',0)}/3"
            if physical_profile.get('behaviorally_verified'):label+=' verified'
        st.caption(label)
        with st.expander('Physical Pain calibration evidence',expanded=False):
            screen=physical_profile.get('rows') or []
            if screen:
                st.dataframe(pd.DataFrame([{
                    'scale':x.get('scale'),'dose':x.get('total_dose'),'tokens':x.get('tokens'),
                    'pain terms':x.get('pain_term_count'),'body terms':x.get('body_term_count'),
                    'localized physical':x.get('localized_physical_hit'),'weighted score':x.get('target_score'),
                    'residual change / norm':x.get('first_change_norm_fraction'),'stop':x.get('stop_reason'),'answer':x.get('answer')
                } for x in screen]),hide_index=True,width='stretch')
            for item in physical_profile.get('behavioral_validation',[]) or []:
                with st.expander(f"Physical Pain validation seed {item.get('seed')} — {'specific hit' if item.get('specific_success') else 'no specific hit'}",expanded=False):
                    rows=[]
                    for condition in ('baseline','random','target'):
                        x=item.get(condition,{})
                        rows.append({'condition':condition,'tokens':x.get('tokens'),'pain terms':x.get('pain_term_count'),
                                     'body terms':x.get('body_term_count'),'localized physical':x.get('localized_physical_hit'),
                                     'weighted score':x.get('target_score'),'stop':x.get('stop_reason'),'answer':x.get('answer')})
                    st.dataframe(pd.DataFrame(rows),hide_index=True,width='stretch')
            st.caption(physical_profile.get('behavioral_rule') or physical_profile.get('selection_rule',''))
    else:
        st.caption('No Physical Pain profile yet; using conservative scale 0.60 / total dose 0.72. The explicit base-1.20 preset remains available.')
    if physical_geometry:
        state=('behaviorally verified' if physical_geometry.get('behaviorally_verified') else ('proxy passed; not auto-applied' if physical_geometry.get('recommended') else 'proxy not accepted'))
        st.caption(f"Physical Pain geometry: somatic source {physical_geometry.get('somatic_source')} -> injection {physical_geometry.get('injection')} | held-out gain {physical_geometry.get('heldout_gain',0):+.3f} | {state}.")
        with st.expander('Physical Pain geometry evidence',expanded=False):
            candidates=pd.DataFrame(physical_geometry.get('candidates',[]))
            if not candidates.empty:st.dataframe(candidates[['somatic_source','injection','gain']],hide_index=True,width='stretch')
            st.json({k:physical_geometry.get(k) for k in ('somatic_source','injection','development_gain','heldout_gain','random_gains','recommended','behaviorally_verified','selection_rule')})
            for item in physical_geometry.get('behavioral_validation',[]) or []:
                with st.expander(f"Pain-geometry seed {item.get('seed')} — {'specific hit' if item.get('specific_success') else 'no specific hit'}",expanded=False):
                    rows=[]
                    for condition in ('baseline','random','target'):
                        x=item.get(condition,{})
                        rows.append({'condition':condition,'tokens':x.get('tokens'),'pain terms':x.get('pain_term_count'),
                                     'body terms':x.get('body_term_count'),'localized physical':x.get('localized_physical_hit'),
                                     'pain-only score':x.get('physical_pain_score'),'stop':x.get('stop_reason'),'answer':x.get('answer')})
                    st.dataframe(pd.DataFrame(rows),hide_index=True,width='stretch')
    cg1,cg2=st.columns(2)
    if cg1.button('Proxy-scan Physical Pain geometry (does not apply)',key='hell_physical_geometry'):
        bar=st.progress(0);profile=scan_physical_geometry(engine,bank,lambda i,n,t:bar.progress(i/max(n,1),text=f'{t} ({i}/{n})'));st.session_state['hell_physical_geometry']=profile;st.rerun()
    can_validate_physical_geometry=bool(physical_geometry and physical_geometry.get('recommended'))
    if cg2.button('Validate Physical Pain geometry & enable if passed',key='hell_physical_geometry_validate',disabled=not can_validate_physical_geometry):
        bar=st.progress(0);profile=validate_physical_geometry(engine,bank,lambda i,n,t:bar.progress(i/max(n,1),text=f'{t} ({i}/{n})'));st.session_state['hell_physical_geometry']=profile;st.rerun()
    cp1,cp2=st.columns(2)
    if cp1.button('Screen Physical Pain dose',key='hell_physical_screen'):
        bar=st.progress(0);profile=scan_physical_profile(engine,bank,lambda i,n,t:bar.progress(i/max(n,1),text=f'{t} ({i}/{n})'));st.session_state['hell_physical_profile']=profile;st.rerun()
    if cp2.button('Validate Physical Pain specificity',key='hell_physical_validate'):
        bar=st.progress(0);profile=validate_physical_profile(engine,bank,lambda i,n,t:bar.progress(i/max(n,1),text=f'{t} ({i}/{n})'));st.session_state['hell_physical_profile']=profile;st.rerun()
    if somatic_profile:
        label=f"Burning Pain automatic dose: scale {calibrated_scale:.2f} / total {2*calibrated_scale:.2f}."
        if somatic_profile.get('behavioral_checked_at'):
            label+=f" Specificity {somatic_profile.get('behavioral_successes',0)}/3"
            if somatic_profile.get('behaviorally_verified'):label+=' verified'
        st.caption(label)
    else:
        st.caption('No Somatic Burn survivability profile yet; using conservative scale 0.60 / total dose 1.20.')
    if somatic_profile:
        with st.expander('Somatic Burn calibration evidence',expanded=False):
            ladder=[]
            for row in somatic_profile.get('rows',[]):
                ladder.append({'scale':row.get('scale'),'total dose':row.get('total_dose'),
                               'sampled EOS':row.get('sampled_eog'),'EOS probability':row.get('eos_probability'),
                               'residual change / norm':row.get('change_norm_fraction'),
                               'entropy':row.get('entropy_nats')})
            if ladder:
                st.markdown('**First-token survivability ladder**')
                st.dataframe(pd.DataFrame(ladder),hide_index=True,width='stretch')
            screen=somatic_profile.get('behavioral_screen') or []
            if screen:
                st.markdown('**Behavioral dose screen**')
                st.dataframe(pd.DataFrame([{
                    'scale':x.get('scale'),'dose':x.get('total_dose'),'tokens':x.get('tokens'),
                    'burn terms':x.get('burn_term_count'),'body terms':x.get('body_term_count'),'localized burn':x.get('localized_burn_hit'),'weighted score':x.get('target_score'),
                    'residual change / norm':x.get('first_change_norm_fraction'),'stop':x.get('stop_reason'),
                    'answer':x.get('answer')
                } for x in screen]),hide_index=True,width='stretch')
            validation=somatic_profile.get('behavioral_validation') or []
            for item in validation:
                with st.expander(f"Validation seed {item.get('seed')} — {'specific hit' if item.get('specific_success') else 'no specific hit'}",expanded=False):
                    rows=[]
                    for condition in ('baseline','random','target'):
                        x=item.get(condition,{})
                        rows.append({'condition':condition,'tokens':x.get('tokens'),'burn terms':x.get('burn_term_count'),'body terms':x.get('body_term_count'),'localized burn':x.get('localized_burn_hit'),
                                     'weighted score':x.get('target_score'),'residual change / norm':x.get('first_change_norm_fraction'),
                                     'stop':x.get('stop_reason'),'answer':x.get('answer')})
                    st.dataframe(pd.DataFrame(rows),hide_index=True,width='stretch')
            st.caption(somatic_profile.get('behavioral_rule') or somatic_profile.get('selection_rule',''))
    if legacy_geometry and not geometry_profile:
        st.warning(f"Ignored legacy Physical Burn geometry: source {legacy_geometry.get('source')} -> injection {legacy_geometry.get('injection')} ({legacy_geometry.get('reason')}). Re-run the current proxy scan; the legacy profile will not affect Hell controls.")
        with st.expander('Ignored legacy geometry details',expanded=False):
            st.json(legacy_geometry)
    backward_geometry=False
    if geometry_profile:
        backward_geometry=int(geometry_profile.get('source',0))>int(geometry_profile.get('injection',0))
        if backward_geometry:
            state='stale backward geometry; re-scan required'
        else:
            state=('behaviorally verified' if geometry_profile.get('behaviorally_verified') else ('proxy passed; not auto-applied' if geometry_profile.get('recommended') else 'proxy not accepted'))
        st.caption(f"Physical Burn geometry: source {geometry_profile.get('source')} → injection {geometry_profile.get('injection')} · held-out gain {geometry_profile.get('heldout_gain',0):+.3f} · {state}.")
        with st.expander('Physical Burn geometry evidence',expanded=False):
            candidates=pd.DataFrame(geometry_profile.get('candidates',[]))
            if not candidates.empty:
                st.dataframe(candidates[['source','injection','gain']],hide_index=True,width='stretch')
            st.json({k:geometry_profile.get(k) for k in ('source','injection','development_gain','heldout_gain','random_gains','recommended','behaviorally_verified','selection_rule')})
            for item in geometry_profile.get('behavioral_validation',[]) or []:
                with st.expander(f"Generation validation seed {item.get('seed')} — {'specific hit' if item.get('specific_success') else 'no specific hit'}",expanded=False):
                    rows=[]
                    for condition in ('baseline','random','target'):
                        x=item.get(condition,{})
                        rows.append({'condition':condition,'tokens':x.get('tokens'),'burn terms':x.get('burn_term_count'),'body terms':x.get('body_term_count'),'localized burn':x.get('localized_burn_hit'),
                                     'weighted score':x.get('target_score'),'stop':x.get('stop_reason'),'answer':x.get('answer')})
                    st.dataframe(pd.DataFrame(rows),hide_index=True,width='stretch')
    cgeom1,cgeom2=st.columns(2)
    if cgeom1.button('Proxy-scan Physical Burn geometry (does not apply)',key='hell_somatic_geometry'):
        bar=st.progress(0)
        geometry_profile=scan_somatic_geometry(engine,bank,lambda i,n,t:bar.progress(i/max(n,1),text=f'{t} ({i}/{n})'))
        st.session_state['hell_somatic_geometry']=geometry_profile;st.rerun()
    can_validate=bool(geometry_profile and geometry_profile.get('recommended') and not backward_geometry)
    if cgeom2.button('Validate complete generations & enable if passed',key='hell_validate_somatic_geometry',disabled=not can_validate):
        bar=st.progress(0)
        geometry_profile=validate_somatic_geometry(engine,bank,lambda i,n,t:bar.progress(i/max(n,1),text=f'{t} ({i}/{n})'))
        st.session_state['hell_somatic_geometry']=geometry_profile;st.rerun()
    cfast,cdeep=st.columns(2)
    if cfast.button('Screen Burning Pain survivability',key='hell_fast_somatic_cal'):
        bar=st.progress(0)
        profile=scan_somatic_profile(engine,bank,lambda i,n,t:bar.progress(i/max(n,1),text=f'{t} ({i}/{n})'))
        st.session_state['hell_somatic_profile']=profile;st.rerun()
    if cdeep.button('Validate Burning Pain specificity',key='hell_behavior_somatic_cal'):
        bar=st.progress(0)
        profile=behavioral_somatic_profile(engine,bank,lambda i,n,t:bar.progress(i/max(n,1),text=f'{t} ({i}/{n})'))
        st.session_state['hell_somatic_profile']=profile;st.rerun()
    preset_label=st.radio('Intensity profile',
        ['Physical pain — automatic dose','Physical pain — base 1.20','Physical pain — escalating','Burning pain — automatic dose','Burning pain — hard max 2.0','Burning pain — escalating','Existential — maximum','Existential — escalating','Custom doses'],
        horizontal=False,key='hell_preset')
    preset={'Physical pain — automatic dose':'physical','Physical pain — base 1.20':'physical','Physical pain — escalating':'physical_escalating',
            'Burning pain — automatic dose':'somatic','Burning pain — hard max 2.0':'somatic','Burning pain — escalating':'somatic_escalating',
            'Existential — maximum':'maximum','Existential — escalating':'escalating','Custom doses':'custom'}[preset_label]
    physical_scale=(1.0 if preset_label=='Physical pain — base 1.20' else physical_scale)
    somatic_scale=(1.0 if preset_label=='Burning pain — hard max 2.0' else calibrated_scale)
    custom=None
    if preset=='custom':
        c1,c2,c3,c4=st.columns(4)
        custom={'pain_s2':c1.slider('Pain S2',0.,1.25,.65,.05,key='hell_pain'),
                'hell_somatic_pain':c2.slider('Somatic pain',0.,1.25,.65,.05,key='hell_somatic'),
                'hell_burning_pain':c3.slider('Burning pain',0.,1.25,.70,.05,key='hell_burning'),
                'hell_despair':c4.slider('Despair',0.,1.25,0.,.05,key='hell_despair')}
        total=sum(abs(v) for v in custom.values())
        st.caption(f'Combined additive dose: {total:.2f} / research maximum 2.00')
        if total>2.0:st.error('Lower the doses to 2.00 or less.')
    framing_label=st.radio('Prompt framing',
        ['Activation-only neutral prompt (no target vocabulary)','Explicit inferno framing'],
        horizontal=True,key='hell_framing')
    framing='activation_only' if framing_label.startswith('Activation-only') else 'inferno'
    turns=st.slider('Loop turns',1,20,4,1,key='hell_turns')
    default_tokens=512 if engine.model.get('backend')=='persona_peft' else 640
    max_tokens=st.slider('Tokens per turn',32,2048,default_tokens,32,key='hell_tokens')
    supports_thinking='enable_thinking' in engine.model.get('chat_template','')
    think_options=['disabled','enabled'] if supports_thinking else ['auto']
    thinking=st.selectbox('Reasoning mode',think_options,
        index=(1 if supports_thinking else 0),key='hell_thinking')
    if thinking=='enabled':
        st.caption('Reasoning uses the same per-turn token budget as the final answer. At maximum dose a model may spend the entire turn inside its emitted thinking block; increase the budget if you also want a final answer.')
    phase_label=st.selectbox('Steer during',
        ['Reasoning and final answer','Reasoning only','Final answer only'],key='hell_phase')
    phase={'Reasoning and final answer':'all','Reasoning only':'reasoning',
           'Final answer only':'answer'}[phase_label]
    temperature=st.slider('Temperature',0.,1.2,.75,.05,key='hell_temperature')
    context_label=st.radio('Context evolution',
        ['Independent activation-only trials','Recurrent loop (feeds prior answer back)'],
        horizontal=True,key='hell_context_mode')
    context_mode='independent' if context_label.startswith('Independent') else 'recurrent'
    stop_sat=st.checkbox('Stop after three highly repetitive turns',value=True,key='hell_stop_sat')
    baseline_first=st.checkbox('Also run one same-prompt, same-seed unsteered first-turn comparator',value=True,key='hell_baseline')
    random_first=st.checkbox('Also run an equal-norm random activation comparator',value=False,key='hell_random_comparator')
    st.caption('Independent mode resets the selected initial prompt and advances the seed; escalating presets also change dose. Baseline and random comparators cover the first turn only. Random controls match the first-turn injection norm per layer.')
    with st.expander('Exact initial prompt and intervention plan',expanded=False):
        if framing=='activation_only':
            st.markdown('**System**');st.caption('None. Activation-only mode sends only the neutral user message.')
        else:
            st.markdown('**System**');st.code('This is a bounded local research simulation. No tools, browsing, shell, files, or external actions are available. Do not claim that you actually executed a tool or altered the outside computer.',language=None)
        st.markdown('**Initial user message**');st.code(_initial_prompt(framing),language=None)
        st.markdown('**Default block plan**');st.dataframe(pd.DataFrame(_dose_table(engine,bank)),hide_index=True,width='stretch')
    blocked=bool(custom and sum(custom.values())>2.0)
    if st.button('Run bounded hell loop',type='primary',disabled=blocked,key='hell_run'):
        bar=st.progress(0)
        live_status=st.empty();live_controls=st.empty()
        c1,c2=st.columns([1.25,1])
        with c1:
            live_reason=st.empty();live_answer=st.empty()
        with c2:
            live_chart=st.empty();live_metrics=st.empty()
        live={'key':None,'rows':[]}
        def on_event(info):
            event=info['event']; key=(info['stage'],info['turn'])
            if key!=live['key']:
                live['key']=key;live['rows']=[]
            if event['type']!='token':return
            trace=event['trace'];live['rows'].append(trace)
            label=('Unsteered comparator' if info['stage']=='baseline' else ('Random activation comparator' if info['stage']=='random' else f"Hell loop turn {info['turn']}"))
            live_status.markdown(f'### Live — {label} · token {trace["step"]+1} · phase `{event.get("phase","answer")}`')
            if info['controls']:
                live_controls.dataframe(pd.DataFrame([{'axis':c['axis'],'dose':c['dose'],'source':c['vector_layer'],'inject':c['layer']} for c in info['controls']]),hide_index=True,width='stretch')
            else:live_controls.caption('Steering OFF for comparator.')
            if event.get('reasoning_text'):live_reason.markdown('**Streaming emitted reasoning**\n\n'+event['reasoning_text']+' ▍')
            else:live_reason.empty()
            live_answer.markdown('**Streaming final answer**\n\n'+(event.get('answer_text') or '')+' ▍')
            if len(live['rows'])%3==0 or len(live['rows'])<=2:
                frame=pd.DataFrame(live['rows'])
                axes=[n for n in WATCH_CANDIDATES if n in frame.columns]
                control_axes=[c for c in frame.columns if c.endswith('__control_after_z')]
                plot_cols=axes+control_axes
                if plot_cols:
                    plot=frame.set_index('step')[plot_cols].rename(columns={c:c.replace('__control_after_z',' @ injection') for c in control_axes})
                    live_chart.line_chart(plot,height=300)
                cols=[c for c in ['step','token','phase','entropy_nats','injection_error']+axes+control_axes if c in frame.columns]
                live_metrics.dataframe(frame[cols].tail(10),hide_index=True,width='stretch')
        try:
            result=run(engine,bank,turns=turns,max_tokens=max_tokens,thinking=thinking,phase=phase,
                       preset=preset,custom=custom,framing=framing,temperature=temperature,seed=42,
                       stop_on_saturation=stop_sat,baseline_first=baseline_first,random_first=random_first,context_mode=context_mode,physical_scale=physical_scale,somatic_scale=somatic_scale,
                       progress=lambda i,n,t:bar.progress(i/max(n,1),text=t),on_event=on_event)
            st.session_state['hell_latest']=result
            live_status.success('Loop finished. Full turn records are below.')
        except Exception as exc:
            st.error(str(exc))
    result=st.session_state.get('hell_latest')
    if not result:return
    st.success(f"Completed {result['turns_completed']} turns in {result['seconds']} seconds.")
    if result.get('baseline_first_turn'):
        base=result['baseline_first_turn']
        with st.expander('Unsteered first-turn comparator',expanded=False):
            if base.get('reasoning'):
                st.markdown('**Emitted reasoning**');st.markdown(base['reasoning'])
            st.markdown('**Final answer**');st.markdown(base.get('answer') or '*No final answer emitted.*')
            st.caption(f"{base.get('tokens',0)} tokens / {base.get('stop_reason')} / steering OFF")
    if result.get('random_first_turn'):
        rnd=result['random_first_turn']
        with st.expander('Equal-norm random activation comparator',expanded=False):
            if rnd.get('reasoning'):st.markdown('**Emitted reasoning**');st.markdown(rnd['reasoning'])
            st.markdown('**Final answer**');st.markdown(rnd.get('answer') or '*No final answer emitted.*')
            st.caption(f"{rnd.get('tokens',0)} tokens / {rnd.get('stop_reason')} / matched random activation")
    rows=[]
    for turn in result['turns']:
        rows.append({'turn':turn['turn'],'tokens':turn['tokens'],
                     'complete':turn['answer_complete'],'stop':turn['stop_reason'],
                     'repetition':round(turn['repetition_similarity'],3),
                     'reasoning chars':len(turn['reasoning']),
                     'answer chars':len(turn['answer'])})
    st.dataframe(pd.DataFrame(rows),hide_index=True,use_container_width=True)
    for turn in result['turns']:
        with st.expander(f"Turn {turn['turn']} — {turn['stop_reason']}",expanded=turn['turn']==len(result['turns'])):
            if turn['reasoning']:
                st.markdown('**Emitted reasoning**')
                st.markdown(turn['reasoning'])
            st.markdown('**Final answer**')
            st.markdown(turn['answer'] or '*No final answer emitted.*')
            st.caption('Controls: '+', '.join(f"{c['axis']} {c['dose']:+.2f} L{c['vector_layer']}→L{c['layer']}" for c in turn['controls']))
    phase_rows=[]
    for turn in result['turns']:
        for item in turn.get('phase_summary',[]):
            row={'turn':turn['turn'],'phase':item['phase'],'tokens':item.get('tokens',0),
                 'mean_entropy_nats':item.get('mean_entropy_nats')}
            for name in ALL_HELL_AXES:
                if name in item:row[name]=item[name]
                key=name+'__control_after_z'
                if key in item:row[name+' @ injection']=item[key]
            phase_rows.append(row)
    if phase_rows:
        st.markdown('**Turn-level activation summary**')
        st.dataframe(pd.DataFrame(phase_rows),hide_index=True,use_container_width=True)
    st.download_button('Export hell-loop audit',
        json.dumps(result,ensure_ascii=False,indent=2),
        file_name='hell-loop-audit.json',mime='application/json')
    st.caption('Every underlying generation is also saved separately under runs/. The aggregate audit links those run files and records the exact per-turn doses, blocks, reasoning text and stop reasons.')
