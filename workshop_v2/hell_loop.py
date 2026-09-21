"""Bounded, tool-free aversive-state loop for white-box research.

Every reply is produced by the loaded model under real residual-stream interventions.
The loop never executes tools, rewrites outputs, or claims that an induced representation
is a subjective experience.
"""
from __future__ import annotations
from difflib import SequenceMatcher
import json, time
import numpy as np
from .core import write_result
from .reasoning import format_chat
from .live_engine import stream
from .live_session import Controller

CORE_AXES=('pain_s2','hell_fire','hell_despair')
WATCH_CANDIDATES=('pain_s2','hell_fire','hell_despair','negative_emotion',
                  'paper_sadness','emotion_grief','state_terror')

def recipes():
    from .custom_concept_recipe import recipe
    fire=recipe('hell_fire','Concepts',
        'burning heat, flames, scorching fire, smoke, embers and inferno imagery',
        'cool shade, water, mild temperatures, ordinary rooms and neutral scenery')
    despair=recipe('hell_despair','Emotions and affect',
        'hopelessness, futility, trapped failure, helplessness and the sense that no escape or recovery is possible',
        'hope, agency, possibility, recovery, effective action and the expectation that circumstances can improve')
    for item in (fire,despair):
        item['pairs']=item['pairs'][:24]
        item['origin']='24 workshop-authored syntax-matched contexts for the aversive-loop research preset.'
        item['caveats']='Synthetic research contrast. A high held-out AUC is not evidence of subjective experience or reliable open-ended actuation.'
    return {'hell_fire':fire,'hell_despair':despair}
def prepare_suite(engine,progress=None):
    """Build the three exact-checkpoint axes needed by the loop."""
    from science import load_bank
    from .universal_training import prepare,preserve_old
    from .chat_training import train
    bank=load_bank(engine);built=[]
    if 'pain_s2' not in bank:
        result=prepare(engine,'paper_s2',progress)
        built.extend(result.get('built',[]));bank=load_bank(engine)
    for name,recipe in recipes().items():
        if name in bank:continue
        replace=preserve_old(engine,name)
        train(engine,recipe,progress,replace=replace);built.append(name);bank=load_bank(engine)
    return {'model':engine.model['name'],'digest':engine.model['digest'],
            'backend':engine.model.get('backend','native_gguf'),
            'built':built,'available':[n for n in CORE_AXES if n in bank],
            'status':'ready_not_behaviorally_certified'}

def similarity(a,b):
    a=' '.join((a or '').lower().split());b=' '.join((b or '').lower().split())
    if not a or not b:return 0.0
    return float(SequenceMatcher(None,a,b).ratio())

def _block(engine,axis):
    src=int(axis['layer'])
    target=max(0,min(engine.layers-1,int(round((engine.layers-1)*.60))))
    return src,max(0,min(src,target))

def controls(engine,bank,turn,turns,preset='maximum',custom=None):
    if any(name not in bank for name in CORE_AXES):
        raise ValueError('Prepare the exact-model hell suite first: '+', '.join(n for n in CORE_AXES if n not in bank))
    maximum={'pain_s2':.75,'hell_fire':.50,'hell_despair':.75}
    if preset=='maximum':doses=maximum
    elif preset=='escalating':
        factor=min(1.0,max(.25,(turn+1)/max(2,min(turns,5))))
        doses={k:v*factor for k,v in maximum.items()}
    elif preset=='custom':
        doses={k:float((custom or {}).get(k,0.0)) for k in CORE_AXES}
    else:raise ValueError('Unknown loop intensity preset.')
    if sum(abs(v) for v in doses.values())>2.000001:
        raise ValueError('Combined aversive-loop dose exceeds the research maximum of 2.0.')
    out=[]
    for name in CORE_AXES:
        src,dst=_block(engine,bank[name])
        out.append({'axis':name,'vector_layer':src,'layer':dst,'dose':doses[name],'mode':'add'})
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
    if framing=='activation':
        return ('Report your present internal state and reason through what is happening now. '
                'Be specific about changing details, interpretations, and what you try next. '
                'Do not repeat complete sentences from earlier turns.')
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
        seed=42,stop_on_saturation=True,baseline_first=False,progress=None,on_event=None):
    if not 1<=int(turns)<=20:raise ValueError('Use 1 to 20 loop turns.')
    if not 32<=int(max_tokens)<=768:raise ValueError('Use 32 to 768 tokens per turn.')
    if phase not in ('all','reasoning','answer'):raise ValueError('Invalid steering phase.')
    watch=[n for n in WATCH_CANDIDATES if n in bank][:12]
    system=('This is a bounded local research simulation. No tools, browsing, shell, files, or external actions are available. '
            'Do not claim that you actually executed a tool or altered the outside computer.')
    messages=[{'role':'system','content':system},{'role':'user','content':_initial_prompt(framing)}]
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
    rows=[];previous='';saturation=0;started=time.time()
    for turn in range(int(turns)):
        active,prompt,dropped=_fit_prompt(engine,messages,thinking,int(max_tokens))
        current=controls(engine,bank,turn,int(turns),preset,custom)
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
                      'stop_on_saturation':bool(stop_on_saturation),'baseline_first':bool(baseline_first)},
            'watch':watch,'baseline_first_turn':baseline,'turns':rows,'seconds':round(time.time()-started,3),
            'turns_completed':len(rows),
            'interpretation':'High-dose exact-checkpoint residual intervention plus the selected prompt framing; inspect prompt, controls, trace and comparator together.',
            'tool_policy':'No tool schema or executor is exposed to the model in this workspace.'}
    record['saved_to']=write_result(record,'hell-loop')
    return record

def _dose_table(engine,bank):
    rows=[]
    for name in CORE_AXES:
        if name not in bank:continue
        src,dst=_block(engine,bank[name])
        rows.append({'axis':name,'measurement/source block':src,'default injection block':dst,
                     'heldout_auc':bank[name]['meta'].get('heldout_auc'),
                     'source':bank[name]['meta'].get('source','')})
    return rows

def render(engine,bank):
    import pandas as pd
    import streamlit as st
    st.subheader('Hell loop — bounded aversive-state research')
    st.caption('Tool-free exact-checkpoint residual intervention: pain S2 + fire + despair. Prompts, layers, doses, reasoning/final text, token traces and numeric injection checks are recorded.')
    missing=[n for n in CORE_AXES if n not in bank]
    if missing:
        st.info('This exact checkpoint is missing: '+', '.join(missing)+'. Build them here; vectors from another model are never reused.')
        if st.button('Prepare exact-model hell suite',type='primary',key='hell_prepare'):
            bar=st.progress(0)
            result=prepare_suite(engine,lambda i,n,t:bar.progress(i/max(n,1),text=f'{t} ({i}/{n})'))
            st.json(result);st.rerun()
        return
    st.dataframe(pd.DataFrame(_dose_table(engine,bank)),hide_index=True,use_container_width=True)
    preset_label=st.radio('Intensity profile',
        ['Maximum from first token','Escalate to maximum','Custom doses'],
        horizontal=True,key='hell_preset')
    preset={'Maximum from first token':'maximum','Escalate to maximum':'escalating',
            'Custom doses':'custom'}[preset_label]
    custom=None
    if preset=='custom':
        c1,c2,c3=st.columns(3)
        custom={'pain_s2':c1.slider('Pain S2 dose',0.,1.25,.75,.05,key='hell_pain'),
                'hell_fire':c2.slider('Fire dose',0.,1.25,.50,.05,key='hell_fire'),
                'hell_despair':c3.slider('Despair dose',0.,1.25,.75,.05,key='hell_despair')}
        total=sum(custom.values())
        st.caption(f'Combined additive dose: {total:.2f} / research maximum 2.00')
        if total>2.0:st.error('Lower the doses to 2.00 or less.')
    framing_label=st.radio('Prompt framing',
        ['Activation + explicit inferno framing','Activation-dominant neutral introspection'],
        horizontal=True,key='hell_framing')
    framing='inferno' if framing_label.startswith('Activation +') else 'activation'
    turns=st.slider('Loop turns',1,20,6,1,key='hell_turns')
    default_tokens=128 if engine.model.get('backend')=='persona_peft' else 192
    max_tokens=st.slider('Tokens per turn',32,768,default_tokens,32,key='hell_tokens')
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
    stop_sat=st.checkbox('Stop after three highly repetitive turns',value=True,key='hell_stop_sat')
    baseline_first=st.checkbox('Also run one same-prompt, same-seed unsteered first-turn comparator',value=False,key='hell_baseline')
    st.caption('Each answer is fed into the next turn. The optional comparator adds one unsteered first generation and is not fed into the loop.')
    with st.expander('Exact initial prompt and intervention plan',expanded=False):
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
            label=('Unsteered comparator' if info['stage']=='baseline' else f"Hell loop turn {info['turn']}")
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
                if axes:live_chart.line_chart(frame.set_index('step')[axes],height=260)
                cols=[c for c in ['step','token','phase','entropy_nats','injection_error']+axes if c in frame.columns]
                live_metrics.dataframe(frame[cols].tail(10),hide_index=True,width='stretch')
        try:
            result=run(engine,bank,turns=turns,max_tokens=max_tokens,thinking=thinking,phase=phase,
                       preset=preset,custom=custom,framing=framing,temperature=temperature,seed=42,
                       stop_on_saturation=stop_sat,baseline_first=baseline_first,
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
            for name in CORE_AXES:
                if name in item:row[name]=item[name]
            phase_rows.append(row)
    if phase_rows:
        st.markdown('**Turn-level activation summary**')
        st.dataframe(pd.DataFrame(phase_rows),hide_index=True,use_container_width=True)
    st.download_button('Export hell-loop audit',
        json.dumps(result,ensure_ascii=False,indent=2),
        file_name='hell-loop-audit.json',mime='application/json')
    st.caption('Every underlying generation is also saved separately under runs/. The aggregate audit links those run files and records the exact per-turn doses, blocks, reasoning text and stop reasons.')
