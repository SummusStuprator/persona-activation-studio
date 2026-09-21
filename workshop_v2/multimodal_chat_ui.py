"""Image and native-tool chat in the regular Chat workspace; explicit transport."""
import json,uuid
import streamlit as st
from . import ollama_bridge as bridge
from .conversation_editor import render_history
from .web_search_tool import search,valid_query

def chat():
    from .runtime import get_engine
    from .live_ui import manager
    engine=get_engine()
    if manager().active():st.warning('Stop the instrumented chat before changing transport.');return
    st.info('Local Ollama image/tool chat uses the selected model’s real image encoder and native tool format. Activation monitoring and activation steering are NOT available in this transport. No image caption is substituted for vision.')
    if engine.handle:
        st.warning('An instrumented model is still resident. Release it before image/tool chat to avoid duplicate resident models.')
        if st.button('Release instrumented model for image/tool chat'):engine.close();st.rerun()
        return
    try:
        records=bridge.inventory();names=[m['name'] for m in records if 'cloud' not in m['name'].lower()]
        if not names:st.info('No locally installed Ollama models found.');return
        name=st.sidebar.selectbox('Ollama chat model',names,key='multimodal_model')
        details=bridge.describe(name);caps=details.get('capabilities',[])
    except Exception as e:st.error(str(e));return
    with st.expander('Ollama template and default system metadata'):
        st.code(details.get('system','(no default system metadata)'))
        st.caption('This API transport may apply the existing Ollama model template. Instrumented steering comparisons use the other transport.')
    st.caption('Advertised capabilities: '+', '.join(caps)+'. CPU-only, two threads, no global Ollama settings changed.')
    if st.session_state.get('external_identity')!=name:
        st.session_state['dialogue']=[];st.session_state['session_id']=uuid.uuid4().hex;st.session_state['external_identity']=name
        for key in ('external_latest','regenerate_after_edit'):st.session_state.pop(key,None)
    messages=st.session_state.setdefault('dialogue',[])
    if st.button('New image/tool chat'):
        st.session_state['dialogue']=[];st.session_state['session_id']=uuid.uuid4().hex;st.session_state.pop('external_latest',None);st.rerun()
    with st.expander('Image / tool generation settings'):
        system=st.text_area('Visible system instruction','',key='external_system')
        thinking=st.checkbox('Enable emitted reasoning',value=False,disabled='thinking' not in caps,key='external_thinking')
        tools=st.checkbox('Allow native web-search requests; approve every execution',disabled='tools' not in caps,key='external_tools')
        budget=st.number_input('Output token budget',16,1024,256,16,key='external_budget')
        st.caption('This API transport uses a 4096-token context. Ollama controls multimodal tokenization and may shorten long histories; start a new chat for long image conversations. The instrumented text transport checks its exact token budget locally.')
    render_history(busy=False)
    resume=st.session_state.pop('regenerate_after_edit',False)
    if tools and messages and messages[-1].get('tool_calls'):
        calls=messages[-1]['tool_calls']
        if len(calls)!=1:st.warning('Only one approved search request per step is supported; no calls were executed.')
        else:
            call=calls[0].get('function',{})
            try:
                if call.get('name')!='web_search':raise ValueError('Unknown tool request; not executed.')
                query=valid_query(call.get('arguments',{}).get('query'));st.code(query)
                provider=st.selectbox('Public search provider',['Wikipedia','DuckDuckGo Instant','Ollama search','Brave','Bing RSS','DuckDuckGo'])
                key=st.text_input('Search API key (session only)',type='password',key='external_search_key') if provider in ('Ollama search','Brave') else None
                if st.button('Approve exact query and continue'):
                    result=search(query,approved=True,provider=provider,api_key=key)
                    messages.append({'role':'tool','tool_name':'web_search','content':json.dumps(result,ensure_ascii=False)})
                    resume=True
            except Exception as e:st.error(str(e))
    images=st.file_uploader('Attach images to the next message',type=['png','jpg','jpeg','webp'],accept_multiple_files=True,disabled='vision' not in caps,key='external_images_'+str(st.session_state.get('image_upload_epoch',0)))
    if images:st.caption('Images are stored locally after orientation correction, metadata removal and optional downscaling. They are replayed in subsequent context until the conversation is branched or cleared.')
    if messages and messages[-1]['role']=='user':
        if st.button('Retry the last message'):resume=True
    prompt=st.chat_input('Message this local model, with optional images',key='external_prompt')
    if prompt or resume:
        if len(images or [])>3:st.error('Attach at most three images per message.');return
        try:
            if prompt:
                attached=[bridge.save_image(x.getvalue()) for x in (images or [])]
                messages.append({'role':'user','content':prompt,'images':attached})
                st.session_state['image_upload_epoch']=st.session_state.get('image_upload_epoch',0)+1
                with st.chat_message('user'):
                    st.markdown(prompt)
                    for image in attached:st.image(bridge.image_bytes(image),width=320)
            effective=([{'role':'system','content':system}] if system.strip() else [])+messages
            result=None
            with st.chat_message('assistant'):
                reason_box=st.empty();answer_box=st.empty()
                for event in bridge.stream_chat(name,effective,int(budget),thinking,tools):
                    if event['type']=='token':
                        answer_box.markdown(event['answer_text']+' ▍')
                        if event['reasoning_text']:
                            with reason_box.container():
                                with st.expander('Model reasoning (emitted text)'):st.markdown(event['reasoning_text'])
                    else:result=event['result'];answer_box.markdown(result['answer_text'])
            if result is None:raise RuntimeError('No completion record.')
            messages.append({'role':'assistant','content':result['answer_text'],'reasoning':result['reasoning_text'],'tool_calls':result['tool_calls'],'run_file':result['saved_to']})
            st.session_state['external_latest']=result
            from .universal_training import atomic_json
            atomic_json(bridge.ROOT/'sessions'/(st.session_state['session_id']+'.json'),dict(model=name,backend='local_ollama_multimodal_tools',messages=messages))
            if result['tool_calls']:st.rerun()
        except Exception as e:st.error(str(e))
    result=st.session_state.get('external_latest')
    if result:
        st.caption(f"Last reply: {result['stop_reason']} / {result['seconds']} seconds / activation steering: OFF")
        if not result['answer_complete'] and not result['tool_calls']:st.warning('No completed final answer; inspect the stop reason and emitted reasoning.')
        st.download_button('Export multimodal audit',json.dumps(result,ensure_ascii=False,indent=2),'multimodal-audit.json')
