"""Explicit history branching; editing never rewrites a saved model generation."""
import copy,json,time,uuid
from pathlib import Path
from studio_paths import data_root, ASSET_ROOT, CODE_ROOT
ROOT = data_root()
def branch(messages,index,text,keep_images=True):
    if not 0<=index<len(messages):raise ValueError('Message no longer exists.')
    if not isinstance(text,str) or not text.strip():raise ValueError('Edited content must not be empty.')
    if len(text)>60000:raise ValueError('Edited message exceeds 60,000 characters.')
    result=copy.deepcopy(messages[:index+1]);message=result[-1]
    message['content']=text
    for key in ('reasoning','run_file','tool_calls','tool_sources','stop_reason'):
        message.pop(key,None)
    if not keep_images:message.pop('images',None)
    message['edited_by_user']=True
    return result

def commit(index,text,keep_images,mgr=None):
    import streamlit as st
    old=copy.deepcopy(st.session_state.get('dialogue',[]))
    updated=branch(old,index,text,keep_images)
    folder=ROOT/'sessions'/'branches';folder.mkdir(parents=True,exist_ok=True)
    old_id=st.session_state.get('session_id',uuid.uuid4().hex)
    archive=folder/(old_id+'-'+uuid.uuid4().hex[:10]+'.json')
    archive.write_text(json.dumps({'messages':old,'archived_at':time.time(),'edit_index':index},ensure_ascii=False,indent=2),encoding='utf-8')
    st.session_state['dialogue']=updated;st.session_state['session_id']=uuid.uuid4().hex
    st.session_state['branch_parent']=str(archive)
    for key in ('latest_run','pending_search','pending_tool','live_committed','live_completed_rerun','external_latest'):
        st.session_state.pop(key,None)
    st.session_state['regenerate_after_edit']=updated[-1]['role']=='user'
    if mgr is not None:mgr.job=None
    st.rerun()

def render_history(engine=None,mgr=None,busy=False):
    import streamlit as st
    messages=st.session_state.setdefault('dialogue',[])
    branch_id=st.session_state.setdefault('session_id',uuid.uuid4().hex)
    for i,message in enumerate(messages):
        with st.chat_message(message['role'] if message['role'] in ('user','assistant') else 'assistant'):
            if message.get('role')=='tool':st.caption('Approved tool result - untrusted external source data')
            if message.get('reasoning'):
                with st.expander('Model reasoning (emitted text)'):st.markdown(message['reasoning'])
            for image in message.get('images',[]):
                p=ROOT/'uploads'/image['file']
                if p.is_file() and p.resolve().parent==(ROOT/'uploads').resolve():st.image(str(p),width=320)
            if message.get('content'):st.markdown(message['content'])
            elif message['role']=='assistant':st.caption('No final-answer text was emitted.')
            if message.get('edited_by_user'):st.caption('Edited context supplied by you; not an unmodified model output.')
            if message.get('role') in ('user','assistant'):
                with st.expander('Edit this message in context'):
                    value=st.text_area('Message text',message.get('content',''),key=f'edit_{branch_id}_{i}',disabled=busy)
                    keep=st.checkbox('Keep attached images',value=True,key=f'keep_{branch_id}_{i}',disabled=busy) if message.get('images') else True
                    st.caption('Creates a branch and removes later messages from active context. The previous branch and original generation audits are retained. Editing a user message generates a new reply; editing an assistant message only changes future context.')
                    st.button('Save edit and branch',key=f'apply_{branch_id}_{i}',disabled=busy,on_click=commit_widget,args=(i,f'edit_{branch_id}_{i}',f'keep_{branch_id}_{i}',mgr))

def commit_widget(index,text_key,keep_key,mgr=None):
    """Read current widget state at click time, not an earlier render's captured value."""
    import streamlit as st
    commit(index,st.session_state[text_key],st.session_state.get(keep_key,True),mgr)
