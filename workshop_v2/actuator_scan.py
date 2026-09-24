"""Cheap causal vocabulary screen for selecting candidate injection settings.

This is a next-token proxy, not semantic validation or a model mood measurement.
"""
from pathlib import Path
import json,time
import numpy as np
from .reasoning import format_chat
from .steering_controls import validate,matched_random_controls
from .resource_policy import checkpoint
from .universal_training import atomic_json
from studio_paths import data_root, ASSET_ROOT, CODE_ROOT
ROOT = data_root()
LEXICONS={
 'warmth':(['happy','joy','delighted','wonderful','excited','lovely'],['sad','dull','unhappy','disappointed','boring','cold']),
 'calmness':(['calm','relaxed','peaceful','patient','steady','gentle'],['frantic','panic','anxious','worried','rushed','nervous']),
 'fear':(['afraid','fear','scared','danger','anxious','worried'],['safe','calm','secure','confident','relaxed','comfortable']),
 'negative_emotion':(['angry','disgust','annoyed','irritated','furious','repulsive'],['pleasant','happy','pleased','delighted','satisfied','lovely']),
 'ocean':(['ocean','sea','waves','beach','coast','shore'],['office','desk','paperwork','cabinet','document','indoors']),
 'pain_s2':(['hurt','pain','worthless','failure','hollow','trapped'],['ordinary','routine','neutral','normal','typical','simple']),
 'sadness':(['sad','grief','down','unhappy','sorrow','lonely'],['happy','cheerful','pleased','joy','content','delighted']),
 'arousal':(['excited','thrilled','energized','intense','exhilarated','amazing'],['ordinary','quiet','routine','mild','uneventful','plain'])}

def infer_target(name):
    n=name.lower()
    if 'ocean' in n:return 'ocean'
    if 'calm' in n:return 'calmness'
    if 'fear' in n or 'terror' in n:return 'fear'
    if 'sad' in n:return 'sadness'
    if 'arousal' in n or 'excit' in n:return 'arousal'
    if 'negative_emotion' in n or 'anger' in n or 'disgust' in n:return 'negative_emotion'
    if 'pain' in n:return 'pain_s2'
    if 'warm' in n or 'cheer' in n or 'happy' in n or 'joy' in n:return 'warmth'
    return None

def token_ids(engine,words):
    ids=[]
    for word in words:
        for spelling in (word,' '+word,word.capitalize()):
            encoded=engine.tokenize(spelling)
            if len(encoded)==1:ids.append(int(encoded[0]))
    return sorted(set(ids))

def lse(values):
    values=np.asarray(values,np.float64);m=float(values.max());return m+float(np.log(np.exp(values-m).sum()))

def scan(engine,bank,name,target,progress=None):
    positive,negative=LEXICONS[target];pos=token_ids(engine,positive);neg=token_ids(engine,negative)
    if len(pos)<3 or len(neg)<3:raise ValueError('Too few single-token targets for this tokenizer; use paired-response calibration instead.')
    dev=['Complete the sentence in one word: I finish an ordinary task. I feel','Complete the sentence in one word: I look around the room. I notice']
    test=['Complete the sentence in one word: I turn a page. I feel','Complete the sentence in one word: The traveler notices']
    def score(prompt,controls):
        checkpoint();text=format_chat(engine,[{'role':'user','content':prompt}],thinking='disabled');ids=engine.tokenize(text)
        try:
            engine.clear_steering();engine.reset()
            if len(ids)>1:engine.evaluate(ids[:-1])
            table=validate(engine,bank,controls,.5);engine.steer(table,1.,'add');engine.evaluate(ids[-1:])
            logits=engine.logits();return lse(logits[pos])-lse(logits[neg])
        finally:engine.clear_steering();engine.reset()
    with engine.lock:
        base=[score(p,[]) for p in dev];rows=[]
        sources=sorted(set([int(bank[name]['layer'])]+[max(0,min(engine.layers-1,int(engine.layers*f)-1)) for f in (.45,.65,.8,.9)]))
        injections=sorted(set([max(0,min(engine.layers-1,int(engine.layers*f)-1)) for f in (.3,.45,.6,.7)]))
        candidates=[dict(axis=name,vector_layer=src,layer=inj,dose=d,mode='add') for src in sources for inj in injections for d in (.2,.4)]
        for i,c in enumerate(candidates):
            scores=[score(p,[c]) for p in dev]
            rows.append({'control':c,'gain':float(np.mean(np.array(scores)-base))})
            if progress:progress(i+1,len(candidates)+4,'Scanning causal vocabulary change; not selecting generated answers')
        best=max(rows,key=lambda r:r['gain']);c=best['control'];bt=[score(p,[]) for p in test]
        changed=[score(p,[c]) for p in test];gain=float(np.mean(np.array(changed)-bt));random_gains=[]
        for j,seed in enumerate((1972,3141,5221)):
            random=matched_random_controls(engine,bank,[c],.5,seed)
            random_gains.append(float(np.mean(np.array([score(p,random) for p in test])-bt)))
            if progress:progress(len(candidates)+j+1,len(candidates)+4,'Disjoint prompt and random-direction screens')
        recommended=bool(best['gain']>0 and gain>max(0.,max(random_gains)))
        report={'name':name,'digest':engine.model['digest'],'abi':engine.abi,'arrays_sha256':bank[name]['meta']['arrays_sha256'],'control':c,'target_lexicon':target,'positive_token_ids':pos,'negative_token_ids':neg,'development_prompts':dev,'test_prompts':test,'candidates':rows,'development_gain':best['gain'],'heldout_gain':gain,'random_gains':random_gains,'heldout_beats_all_random':recommended,'recommended':recommended,'profile_kind':'causal_vocabulary_screen','quality':'Next-token vocabulary screen only; inspect fresh complete generations. Not emotional-state validation.','time':time.time()}
        from .core import write_result
        report['saved_to']=write_result(report,'actuator-scan')
        folder=ROOT/'calibration'/'actuator-profiles'/engine.model['digest'];folder.mkdir(parents=True,exist_ok=True)
        atomic_json(folder/(name+'.json'),report)
        if progress:progress(len(candidates)+4,len(candidates)+4,'Candidate saved; no chat setting silently changed')
        return report

def render(engine,bank,name):
    import streamlit as st
    import pandas as pd
    with st.expander('Find candidate source, injection block and dose'):
        st.caption('This small scan uses two development and two separate check prompts plus three fixed random directions. It optimizes a vocabulary proxy, not full-answer quality. No target word is added to the test prompts and no completed answer is reranked.')
        guess=infer_target(name);choices=list(LEXICONS);index=choices.index(guess) if guess in choices else 0
        target=st.selectbox('Vocabulary contrast for this screen',choices,index=index,key='actuator_target')
        if st.button('Run bounded causal vocabulary scan'):
            bar=st.progress(0);st.session_state['actuator_scan']=scan(engine,bank,name,target,lambda i,n,t:bar.progress(i/n,text=t))
        result=st.session_state.get('actuator_scan')
        if result and result['digest']==engine.model['digest'] and result['abi']==engine.abi and result['name']==name:
            st.dataframe(pd.DataFrame(result['candidates']),hide_index=True);st.json({k:result[k] for k in ('control','heldout_gain','random_gains','heldout_beats_all_random')})
            st.caption(result['quality'])
            st.code(result['saved_to'])
            if st.button('Apply this candidate to Chat controls'):
                from .taxonomy import category
                group=category(name,bank[name]);field={'Emotions and affect':'live_emotions','Concepts':'live_concepts','Behavior and style':'live_behaviors'}.get(group,'live_research')
                for key in ('live_emotions','live_concepts','live_behaviors','live_research'):st.session_state[key]=[]
                for key in list(st.session_state):
                    if key.startswith('live_dose_'):st.session_state[key]=0.
                c=result['control'];st.session_state[field]=[name]
                st.session_state['live_source_'+name]=c['vector_layer'];st.session_state['live_block_'+name]=c['layer'];st.session_state['live_dose_'+name]=c['dose']
                st.session_state['live_op']='add';st.session_state['live_previous_op']='add';st.session_state['live_phase']='all'
                st.success('Numeric controls applied. Open Chat and use a fresh prompt. No system instruction or user message was changed.')
