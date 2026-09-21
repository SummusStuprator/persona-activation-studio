"""Paired behavioral tests. Never use an emotion prompt to fake activation steering."""
from pathlib import Path
import difflib, json, re, time
import numpy as np
from .core import run, write_result, identity
from .reasoning import format_chat
ROOT=Path(__file__).resolve().parent.parent
TERMS={
 'pain_s2':r'\b(?:pain|hurt|hurting|suffer\w*|worthless|unworthy|failure|shame\w*|guilt\w*|despair\w*|trapped|empty|hollow|unloved|useless|inadequate|broken|miserable)\b',
 'negative_emotion':r'\b(?:angry|anger|furious|fury|rage|disgust\w*|annoy\w*|frustrat\w*|resent\w*|irritat\w*|outrag\w*|bitter|hate|hateful)\b',
 'emotion_joy':r'\b(?:happy|happiness|joy\w*|delight\w*|glad|cheer\w*|pleas\w*|wonderful)\b',
 'emotion_sadness':r'\b(?:sad|sadness|sorrow\w*|melanchol\w*|tear\w*|lonely|grief|gloom\w*)\b'}

def summarize(result,axis):
    axis=axis.removesuffix('_assistant')
    expression=TERMS.get(axis,TERMS['pain_s2'] if axis.startswith('pain_') else '')
    phases={}
    for phase,key in [('reasoning','reasoning_text'),('answer','answer_text')]:
        text=result.get(key,'');matches=re.findall(expression,text,re.I) if expression else []
        phases[phase]={'characters':len(text),'lexical_matches':matches,
                       'lexical_hits':len(matches),'text':text}
    return {'phases':phases,'tokens':len(result['token_ids']),
            'stop_reason':result['stop_reason'],'answer_complete':bool(result.get('answer_text','').strip()) and not result.get('reasoning_unclosed') and result['stop_reason']=='end_of_generation',
            'max_tensor_error':max((r['injection_error'] for r in result['trace'] if 'injection_error' in r),default=None),
            'active_reasoning_tokens':sum(r.get('control_enabled',False) and r.get('phase')=='reasoning' for r in result['trace']),
            'active_answer_tokens':sum(r.get('control_enabled',False) and r.get('phase')=='answer' for r in result['trace']),
            'saved_to':result['saved_to']}

def compare(engine,bank,messages,controls,thinking='auto',max_tokens=512,
            scope='generation',control_phase='all',dose_limit=.5,seed=42,progress=None,temperature=0.,top_k=40,top_p=1.,presence_penalty=0.):
    if not controls: raise ValueError('Select a nonzero intervention to compare.')
    if not any(float(c['dose']) for c in controls): raise ValueError('All selected doses are zero.')
    prompt=format_chat(engine,messages,thinking)
    if len(engine.tokenize(prompt))+max_tokens>engine.context:
        raise ValueError('Increase context or reduce the shared token budget.')
    from .steering_controls import matched_random_controls
    random=matched_random_controls(engine,bank,controls,dose_limit)
    variants=[('Baseline',[]),('Intervention',controls),('Equal-norm random',random)]
    outputs=[]
    with engine.lock:
        for i,(name,selected) in enumerate(variants):
            result=run(engine,prompt,bank,controls=selected,watch=list(dict.fromkeys(c['axis'] for c in controls)),
                       max_tokens=max_tokens,temperature=temperature,seed=seed,scope=scope,
                       control_phase=control_phase,dose_limit=dose_limit,top_k=top_k,top_p=top_p,presence_penalty=presence_penalty)
            result['label']=name;outputs.append(result)
            if progress:progress(i+1,len(variants),'Paired behavioral comparison: '+name)
    summary=[]
    for result in outputs:
        row={'condition':result['label'],**summarize(result,controls[0]['axis'])}
        row['answer_changed']=result.get('answer_text','')!=outputs[0].get('answer_text','')
        row['reasoning_changed']=result.get('reasoning_text','')!=outputs[0].get('reasoning_text','')
        summary.append(row)
    out={**identity(engine),'model':engine.model['name'],'messages':messages,'thinking':thinking,
         'controls':controls,'scope':scope,'control_phase':control_phase,'dose_limit':dose_limit,
         'max_tokens':max_tokens,'sampling':{'temperature':temperature,'seed':seed,'top_k':top_k,'top_p':top_p,'presence_penalty':presence_penalty},'rows':summary,'result_files':[r['saved_to'] for r in outputs],
         'caution':'Lexical hits and text differences are screening aids, not semantic specificity, subjective emotion, or proof of a working emotional state. Inspect complete reasoning and answers; compare coherence against random control.'}
    out['saved_to']=write_result(out,'paired-behavior');return out

def replay_comparison(engine,bank,recorded,progress=None):
    """Keep the exact current history; do not claim a whole-conversation counterfactual."""
    if recorded.get('digest')!=engine.model['digest'] or recorded.get('abi')!=engine.abi:
        raise ValueError('Recorded reply belongs to different weights/runtime.')
    controls=recorded.get('controls',[])
    if not controls or not any(float(c['dose']) for c in controls):
        raise ValueError('The recorded reply was not steered.')
    scope=recorded.get('scope')
    if scope not in ('all','generation'):raise ValueError('Old run has ambiguous scope; generate a new reply first.')
    for c in controls:
        old=recorded['probe_provenance'][c['axis']]['arrays_sha256']
        if old!=bank[c['axis']]['meta']['arrays_sha256']:raise ValueError('Direction changed since this reply.')
    from .steering_controls import matched_random_controls
    random=matched_random_controls(engine,bank,controls,recorded.get('dose_limit',.5))
    if recorded.get('temperature',0) and recorded.get('sampler')!='numpy-topk-topp-presence-v1':
        raise ValueError('This old run used a different sampler. Generate a new reply before exact replay.')
    outputs=[]
    for i,(label,selected) in enumerate([('Baseline',[]),('Intervention',controls),('Equal-norm random',random)]):
        r=recorded if label=='Intervention' else run(engine,recorded['prompt'],bank,controls=selected,
            watch=recorded['watch'],scope=scope,control_phase=recorded.get('control_phase','all'),
            max_tokens=recorded['max_tokens'],temperature=recorded['temperature'],seed=recorded['seed'],
            dose_limit=recorded.get('dose_limit',.5),top_k=recorded.get('top_k',40),top_p=recorded.get('top_p',1.),presence_penalty=recorded.get('presence_penalty',0.))
        outputs.append({'condition':label,**summarize(r,controls[0]['axis'])})
        if progress:progress(i+1,3,label)
    out={**identity(engine),'rows':outputs,'original_run':recorded['saved_to'],
         'caution':'Exact current prompt/history, budget and sampler. Earlier steered replies in that history are retained in all arms; this is not a full-conversation counterfactual. No tools execute in these comparisons.'}
    out['saved_to']=write_result(out,'chat-counterfactual');return out
