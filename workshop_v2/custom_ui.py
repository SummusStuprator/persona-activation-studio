"""Inspectable concept/behavior/emotion recipes; no hidden prompt steering."""
import json
from pathlib import Path
import streamlit as st
from .chat_training import STARTERS,make_recipe,train,calibrate,save_recipe,profiles

def concepts(engine,bank):
    st.subheader('Custom concepts and chat behaviors')
    st.write('Define a target, inspect the paired examples, then train a small activation-direction file for the exact loaded checkpoint. Your model weights stay unchanged. Both native GGUF and the persona PEFT backend are supported. The target and examples are NOT added to chat prompts.')
    from .refined_recipes import recipe as refined_recipe
    from .recipe_quality import audit_recipe
    refined_options={'Friendly enthusiasm (40 matched tasks)':'warm','Direct answers (40 matched tasks)':'concise'}
    choice=st.selectbox('Starting recipe',['Custom target']+list(refined_options)+list(STARTERS),key='recipe_start')
    defaults=STARTERS.get(choice,{'name':'concept_custom','category':'Concepts','positive':'volcanoes, magma, lava and volcanic landscapes','negative':'rivers, waterfalls, streams and freshwater landscapes'})
    refined=refined_recipe(refined_options[choice]) if choice in refined_options else None
    if refined:
        defaults={'name':refined['name'],'category':refined['category'],'positive':refined['positive_description'],'negative':refined['negative_description']}
    if st.session_state.get('recipe_bound')!=choice:
        for key in ('name','category','positive','negative'):st.session_state['recipe_'+key]=defaults[key]
        st.session_state['recipe_bound']=choice;st.session_state.pop('recipe_json',None)
    name=st.text_input('Unique direction name',key='recipe_name')
    category=st.selectbox('Control family',['Concepts','Emotions and affect','Behavior and style','Bodily sensations'],key='recipe_category')
    pos=st.text_area('More of this',key='recipe_positive');neg=st.text_area('Less of this / matched contrast',key='recipe_negative')
    use_phrases=choice!='Custom target' and pos==defaults['positive'] and neg==defaults['negative']
    if choice=='Custom target':
        from .custom_concept_recipe import recipe as custom_recipe
        recipe=custom_recipe(name,category,pos,neg)
        st.caption('Custom targets use 40 transparent syntax-matched contexts. Within each pair only your target description changes.')
    elif refined:
        recipe=dict(refined,name=name,category=category,positive_description=pos,negative_description=neg)
        st.caption('This recipe has 40 content-matched task pairs. Description edits do not rewrite the examples; edit pair JSON to change the training texts.')
    else:
        recipe=make_recipe(name,category,pos,neg,defaults.get('positive_phrases') if use_phrases else None,defaults.get('negative_phrases') if use_phrases else None)
    st.caption('Quick recipes are small synthetic contrasts. Repeated phrases can make their separation scores optimistic. Editing diverse matched responses improves specificity.')
    with st.expander('Inspect or edit every training pair',expanded=False):
        st.dataframe(recipe['pairs'],hide_index=True)
        manual=st.checkbox('Use edited JSON rather than the generated recipe')
        edited=st.text_area('Complete recipe JSON',json.dumps(recipe,indent=2),height=320,key='recipe_json')
        if manual:recipe=json.loads(edited)
        upload=st.file_uploader('Import a recipe JSON',type=['json'])
        if upload:
            if upload.size>2*1024*1024:raise ValueError('Recipe must be at most 2 MiB.')
            recipe=json.loads(upload.getvalue());st.info('The uploaded recipe will be used.')
    quality=audit_recipe(recipe)
    with st.expander('Training-data quality audit',expanded=True):
        st.json(quality)
        if quality['responses_reused_across_groups']:st.warning('Repeated response wording occurs in different groups. Held-out context accuracy can overstate generalization.')
    recipe['quality_audit']=quality
    st.download_button('Export editable recipe',json.dumps(recipe,indent=2),file_name=name+'-recipe.json')
    st.info('Calibration now selects settings on up to four development contexts and checks them on separate held-out contexts against three equal-norm random directions. It scores up to 32 response tokens; completed-answer behavior still needs fresh validation.')
    auto=st.checkbox('Also run held-out intervention calibration for Chat',value=True)
    if st.button('Build this control for the loaded model',type='primary'):
        from science import load_bank
        bar=st.progress(0);progress=lambda i,n,t:bar.progress(i/n,text=t+f' ({i}/{n})')
        meta=train(engine,recipe,progress)
        if auto:calibrate(engine,load_bank(engine),meta['name'],progress)
        st.session_state['recent_custom']=meta['name'];st.success('Saved native direction and recipe. Open Chat to use it.');st.rerun()
    recent=st.session_state.get('recent_custom')
    if recent and recent in bank:
        st.success('Ready for chat: '+recent)
        st.caption('This is an on-device trained control, not a promise of the same effect on every prompt. Use live counterfactuals to inspect it.')
    with st.expander('Calibrate an existing custom control'):
        eligible=[n for n,a in bank.items() if a['meta'].get('recipe_file')]
        if eligible:
            n=st.selectbox('Existing recipe direction',eligible)
            if st.button('Recalibrate chat settings'):
                bar=st.progress(0);st.json(calibrate(engine,bank,n,lambda i,m,t:bar.progress(i/m,text=t)))
    st.caption('Recipes can be reused across models; vector files cannot. Build separately for each exact checkpoint. Reusing an existing direction name is rejected instead of silently replacing it.')
