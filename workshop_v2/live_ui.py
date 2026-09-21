"""Responsive chat; all displayed response text comes from the native generation job."""
import json,uuid,time
import pandas as pd
import streamlit as st
from .live_session import Manager
from .chat_training import profiles
from .library import label,DEFAULT_WATCH
from .taxonomy import groups
from .reasoning import format_chat

@st.cache_resource
def manager():return Manager()

def clear_doses(job=None):
    for key in list(st.session_state):
        if key.startswith('live_dose_'):st.session_state[key]=0.
    if job and job.busy:
        value=job.controller.value
        job.controller.update([],value['phase'],value['dose_limit'])

def controls(engine,bank,job):
    available=groups(bank);known=profiles(engine,bank)
    with st.expander('Live steering controls',expanded=True):
        st.caption('Native activation changes only. Controls never add instructions to your chat prompt. Change a control while the answer is streaming to affect the next not-yet-computed token.')
        active=[]
        for title,key in [('Emotions and affect','live_emotions'),('Behavior and style','live_behaviors'),('Concepts','live_concepts')]:
            active+=st.multiselect(title,available.get(title,[]),format_func=label,key=key)
        with st.expander('Advanced: pain, body and research controls'):
            extra=available.get('Pain research',[])+available.get('Bodily sensations',[])+available.get('Research controls',[])
            active+=st.multiselect('Research directions',extra,format_func=label,key='live_research')
        high=st.checkbox('Enable experimental high doses',value=False,key='live_high')
        limit=1.5 if high else .5
        if high:st.warning('Above 0.50, effects may include off-topic text, hostile language, incomplete reasoning or nonsense. Higher is not more accurate.')
        op=st.selectbox('Live operation',['add','erase'],format_func=lambda x:'Add / subtract' if x=='add' else 'Erase one direction',key='live_op')
        if st.session_state.get('live_previous_op')!=op:
            clear_doses(job);st.session_state['live_previous_op']=op
        requested=[]
        for name in active:
            a=bank[name];profile=known.get(name);default=profile['control'] if profile else {'layer':a['layer'],'vector_layer':a['layer'],'dose':0.}
            suggested=float(default.get('dose',0.))
            behaviorally_verified=bool(profile and profile.get('behaviorally_verified'))
            if profile and not behaviorally_verified:
                default=dict(default,dose=0.)
            if profile and profile['mean_margin_gain']<=max(0.,profile['random_margin_gain']):
                st.warning('This calibration did not beat its random control. It starts at zero; manual use remains experimental.')
            st.markdown('**'+label(name)+'**')
            if profile:
                kind=profile.get('profile_kind','legacy_likelihood')
                if behaviorally_verified:
                    st.caption('Behaviorally verified saved setting; complete-answer effects can still vary by context.')
                else:
                    st.caption(f'Saved candidate: {kind.replace("_"," ")}; suggested dose {suggested:+.2f}. It starts at zero because proxy validation is not completed-answer behavioral validation.')
            else:st.caption('No causally screened chat setting saved. A trained probe is not automatically a useful steering control.')
            dosekey='live_dose_'+name;src='live_source_'+name;dst='live_block_'+name
            if dosekey not in st.session_state:st.session_state[dosekey]=min(.5,float(default['dose'])) if op=='add' else 0.
            lo,hi=(-limit,limit) if op=='add' else (0.,1.)
            if not lo<=float(st.session_state[dosekey])<=hi:st.session_state[dosekey]=0.
            dose=st.slider('Strength: '+label(name),lo,hi,step=.01,key=dosekey)
            with st.expander('Layer settings: '+label(name)):
                source=st.number_input('Vector source: '+label(name),0,engine.layers-1,int(default.get('vector_layer',a['layer'])),key=src)
                block=st.number_input('Injection block: '+label(name),0,engine.layers-1,int(default['layer']),key=dst)
                st.caption(f"Held-out text AUC {a['meta'].get('heldout_auc',0):.3f}; test n={a['meta'].get('test_n',0)}. Source and injection layers need not coincide.")
            requested.append({'axis':name,'layer':int(block),'vector_layer':int(source),'dose':dose,'mode':op})
        phase=st.selectbox('Apply during',['all','reasoning','answer'],format_func=lambda x:{'all':'Reasoning and final answer','reasoning':'Reasoning only','answer':'Final answer only'}[x],key='live_phase')
        st.button('Clear steering now',on_click=clear_doses,args=(job,),key='live_clear')
        st.caption('Clearing prevents future injection. It does not undo words or cached effects already produced. Start a new chat for a clean context.')
    from .steering_controls import validate
    good=True
    try:
        validate(engine,bank,requested,limit)
        if job and job.busy:job.controller.update(requested,phase,limit)
    except Exception as exc:
        good=False
        if job and job.busy:
            job.controller.pause()
            if not job.controller.cancelled():job.controller.update([],phase,limit)
        st.error(str(exc)+' Steering OFF has been requested and active generation paused at the next decode boundary. Correct the settings, then resume.')
    if good and not any(c['dose'] for c in requested):st.info('Steering requested OFF: all doses are zero. Watching a probe does not steer it.')
    with st.expander('Watch activations (independent of steering)',expanded=False):
        watch=st.multiselect('Visible probes',list(bank),default=list(dict.fromkeys(n for n in ['chat_emotion_warm_v2','persona_friendly_enthusiasm','chat_emotion_calm']+DEFAULT_WATCH if n in bank)),format_func=label,key='live_watch',disabled=bool(job and job.busy))
        st.caption('Select probes before starting the reply. Emotion, concept and behavioral projections are separate measurements, not inferred feelings.')
    return requested,phase,limit,watch,good

def _save_finished(job,snapshot):
    if snapshot['state'] not in ('finished','failed') or st.session_state.get('live_committed')==job.id:return
    st.session_state['live_committed']=job.id
    result=snapshot['result']
    if result:
        st.session_state.setdefault('dialogue',[]).append({'role':'assistant','content':result['answer_text'],
            'reasoning':result['reasoning_text'],'run_file':result['saved_to'],'stop_reason':result['stop_reason']})
        st.session_state['latest_run']=result
        from .ui import save_chat
        save_chat(job.engine)
    elif snapshot.get('error'):
        partial=snapshot['output']
        if partial.get('text'):
            st.session_state.setdefault('dialogue',[]).append({'role':'assistant','content':partial.get('answer_text',''),'reasoning':partial.get('reasoning_text',''),'stop_reason':'error'})
        elif st.session_state.get('dialogue') and st.session_state['dialogue'][-1]['role']=='user':
            st.session_state['dialogue'].pop()
        st.session_state.pop('latest_run',None)

@st.fragment(run_every=.5)
def running_panel(job,bank):
    job.controller.heartbeat();snap=job.snapshot()
    if snap['state'] in ('finished','failed'):
        _save_finished(job,snap)
        if st.session_state.get('live_completed_rerun')!=job.id:
            st.session_state['live_completed_rerun']=job.id;st.rerun()
    with st.container(border=True):
        st.markdown('**Live generation**')
        phase=snap['output'].get('phase','answer');n=len(snap['trace'])
        applied=snap['trace'][-1]['control_revision'] if n else 'pending'
        st.caption(f"State: {'paused' if snap['paused'] and job.busy else snap['state']} | Generated tokens: {n} | Phase: {phase} | Requested revision: {snap['requested_revision']} | Applied revision: {applied}")
        a,b,c=st.columns(3)
        a.button('Pause generation',disabled=not job.busy or snap['paused'],on_click=job.controller.pause,key='pause_'+job.id)
        b.button('Resume generation',disabled=not job.busy or not snap['paused'],on_click=job.controller.resume,key='resume_'+job.id)
        c.button('Stop generation',disabled=not job.busy,on_click=job.controller.stop,key='stop_'+job.id)
        if snap['paused']:st.caption('Pause takes effect at the next decode boundary. You can change controls now, then resume the same reply.')
    if job.busy:
        with st.chat_message('assistant'):
            if snap['output']['reasoning_text']:
                with st.expander('Model reasoning (emitted text)',expanded=False):st.markdown(snap['output']['reasoning_text'])
            st.markdown(snap['output']['answer_text']+(' ▍' if phase=='answer' else ''))
    if snap['error']:st.error(snap['error'])
    if snap['trace']:
        with st.expander('Live control timeline and projections',expanded=False):
            frame=pd.DataFrame(snap['trace'])
            axes=[n for n in job.watch if n in frame]
            if axes:st.line_chart(frame.set_index('step')[axes])
            st.dataframe(frame[['step','token','phase','control_revision','control_enabled','injection_error']].tail(24),hide_index=True)
            st.caption('Injection error tests the requested tensor change, not emotion. Persona BF16 storage rounding is recorded separately in precision_check; zero excess error does not mean zero rounding.')

def chat(engine,bank):
    mgr=manager();owner=st.session_state.setdefault('live_owner',uuid.uuid4().hex)
    active=mgr.active()
    if active and active.owner!=owner:
        st.warning('Another browser tab owns the current generation. Finish or stop it there before starting here.');return
    job=mgr.job if mgr.job and mgr.job.owner==owner else None
    if job and job.model_identity!=(engine.model['digest'],engine.abi,engine.process.pid):job=None
    if job and not job.busy:_save_finished(job,job.snapshot())
    from .live_presets import render as research_presets
    research_presets(engine,bank,bool(job and job.busy))
    if not (job and job.busy):
        from .universal_training import prepare_ui
        prepare_ui(engine,bank,compact=True)
    requested,phase,limit,watch,valid=controls(engine,bank,job)
    busy=bool(job and job.busy)
    with st.expander('Reasoning and generation settings',expanded=False):
        options=['auto','disabled','enabled'] if 'enable_thinking' in engine.model.get('chat_template','') else ['auto']
        thinking=st.selectbox('Thinking mode',options,key='live_thinking',disabled=busy)
        preserve=st.checkbox('Replay previous emitted reasoning',value=False,key='live_preserve',disabled=busy)
        system=st.text_area('Visible system instruction',engine.model.get('default_system','You are a helpful assistant. Answer clearly and honestly.'),key='live_system',disabled=busy)
        from .sampling_ui import render
        if not busy:sampling,seed=render(engine,thinking,'live_sampler')
        else:sampling={k:job.settings[k] for k in ('temperature','top_k','top_p','presence_penalty')};seed=job.settings['seed'];st.caption('Sampling is fixed for this reply. Steering controls above remain live.')
        budget=int(st.number_input('Reasoning + answer token budget',16,min(4096,engine.context-16),min(1024,engine.context-128),step=16,key='live_budget',disabled=busy))
        delay=st.slider('Inspection delay per token (seconds)',0.,.5,0.,.01,key='live_delay',disabled=busy)
        allow_search=st.checkbox('Allow web-search proposals (approve each query)',value=False,key='live_web',disabled=busy)
        if allow_search:
            from .web_search_tool import INSTRUCTION
            st.code(INSTRUCTION)
        st.caption('Sampling is separate from activation steering. Web search adds the displayed tool protocol only when enabled; no shell or arbitrary file tool is exposed.')
    if st.button('New chat',disabled=busy):
        st.session_state['dialogue']=[];st.session_state['session_id']=uuid.uuid4().hex
        st.session_state.pop('latest_run',None);mgr.job=None;st.rerun()
    conversation=st.session_state.setdefault('dialogue',[]);st.session_state.setdefault('session_id',uuid.uuid4().hex)
    from .chat_ui import render_message
    from .conversation_editor import render_history
    render_history(engine,mgr,busy)
    if allow_search:
        from .web_search_tool import render_pending
        if render_pending(conversation,busy):mgr.job=None;job=None
    regenerate=st.session_state.pop('regenerate_after_edit',False)
    prompt=st.chat_input('Message your local model; steering applies to this actual reply',key='live_prompt',disabled=busy or not valid)
    if (prompt or regenerate) and not busy and valid:
        effective_system=system
        if allow_search:
            from .web_search_tool import INSTRUCTION
            effective_system+='\n'+INSTRUCTION
        messages=([{'role':'system','content':effective_system}] if effective_system.strip() else [])+conversation+([{'role':'user','content':prompt}] if prompt else [])
        try:
            if any(m.get('images') for m in messages):raise ValueError('This context contains images. Use the Local Ollama image/tool transport or explicitly remove the attachments; images are not silently ignored.')
            formatted=format_chat(engine,messages,thinking,preserve)
            count=len(engine.tokenize(formatted))
            if count+budget>engine.context:raise ValueError(f'Prompt {count} plus reply budget {budget} exceeds context {engine.context}. Nothing was truncated.')
            settings={**sampling,'seed':seed,'max_tokens':budget,'scope':'generation','token_delay':delay}
            job=mgr.start(engine,bank,owner,formatted,requested,phase,limit,settings,watch)
            if prompt:conversation.append({'role':'user','content':prompt});render_message(conversation[-1])
            busy=job.busy
        except Exception as exc:st.error(str(exc))
    if job:running_panel(job,bank)
    result=st.session_state.get('latest_run')
    if result and not busy:
        st.caption(f"Last reply: {len(result['token_ids'])} tokens | {result['stop_reason']} | {result['seconds']:.2f}s")
        if not result.get('answer_complete',False):
            st.warning('This is an incomplete or stopped reply, not a validated completed answer. Inspect its reasoning and stop reason; no final text was fabricated.')
        with st.expander('Exact prompt and intervention audit'):
            st.code(result['prompt']);st.json(result.get('control_events',[]))
            st.caption('No control label, target definition or example is appended to this prompt.')
            st.download_button('Export raw generation audit',json.dumps(result,indent=2),file_name='live-generation.json')
        from .ui import charts
        with st.expander('Detailed activation inspection'):charts(result,result.get('watch',[]))
        from .live_replay import replay_ui
        replay_ui(engine,bank,result)
    if conversation:
        from .ui import save_chat
        st.download_button('Export conversation',json.dumps(save_chat(engine),indent=2),file_name='workshop-conversation.json')
