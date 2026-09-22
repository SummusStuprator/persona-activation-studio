"""Incremental native inference with atomically sampled controls before each decode.

No prompt rewriting, logit steering, best-of-N selection or generated-text replacement.
Sampling is the only logit postprocessing, and its settings are recorded separately.
"""
from __future__ import annotations
import codecs
import hashlib
import time
import numpy as np
from .core import configure, identity, write_result
from .sampling import draw
from .reasoning import initial_phase, split_output, phase_summary
from .steering_controls import validate, measurements, check_injection

class Cancelled(Exception):
    """Cooperative cancellation; no intervention survives the finally block."""

def stream(engine, prompt, bank, controller, watch=(), max_tokens=512,
           temperature=.7, seed=42, top_k=40, top_p=.95, presence_penalty=0.,
           scope='generation', token_delay=0.):
    if scope not in ('generation', 'all'):
        raise ValueError('Invalid position scope.')
    if not 1 <= int(max_tokens) <= 4096:
        raise ValueError('Use 1 to 4096 output tokens.')
    if len(watch)>24 or any(n not in bank for n in watch):
        raise ValueError('Choose up to 24 built measurement directions.')
    if not 0<=token_delay<=.5:
        raise ValueError('Inspection delay must be between 0 and 0.5 seconds.')
    from contracts import validate_axis
    for n in watch: validate_axis(engine,n,bank[n])
    started=time.monotonic(); start_phase=initial_phase(prompt); phase=start_phase
    tokens=engine.tokenize(prompt)
    if len(tokens)+max_tokens>engine.context:
        raise ValueError(f'The prompt has {len(tokens)} tokens and the reply reserves {max_tokens}; '
                         f'context is {engine.context}. Lower the reply budget, use a larger context, or start a new chat. Nothing was silently truncated.')
    original_identity=identity(engine)
    initial_settings=None; events=[]; traces=[]; profiles=[]; generated=[]
    output=''; reason='token_limit'; rng=np.random.default_rng(seed)
    decoder=codecs.getincrementaldecoder('utf-8')('replace')
    seen=set(watch); revision=None; controls=[]; table=None; settings=None
    first=None; last=None; terminal=None
    applied_configuration=None
    with engine.lock:
        try:
            engine.clear_steering(); engine.reset()
            # Native prefill is not interrupted halfway; cancellation is checked on return.
            if scope=='generation' and len(tokens)>1: engine.evaluate(tokens[:-1])
            for step in range(int(max_tokens)):
                settings=controller.before_decode(step,phase)
                if settings is None: reason='cancelled'; break
                if initial_settings is None: initial_settings=settings.copy()
                if settings['revision']!=revision:
                    controls=[dict(c) for c in settings['controls']]
                    table=validate(engine,bank,controls,settings['dose_limit'])
                    revision=settings['revision']; seen.update(c['axis'] for c in controls)
                    events.append({'revision':revision,'effective_before_token':step,
                                   'requested_at':settings.get('requested_at'),
                                   'applied_at':time.time(),'phase':settings['phase'],
                                   'dose_limit':settings['dose_limit'],'controls':controls})
                enabled=settings['phase']=='all' or settings['phase']==phase
                if scope=='all' and settings['phase']!='all':
                    raise ValueError('Phase-selective steering requires reply-only scope.')
                configuration=(revision,enabled)
                if configuration != applied_configuration:
                    configure(engine,table,controls,enabled)
                    applied_configuration=configuration
                current=tokens if step==0 and scope=='all' else (tokens[-1:] if step==0 else np.array([generated[-1]],np.int32))
                engine.evaluate(current)
                pre,post=engine.capture(0),engine.capture(1)
                if not np.isfinite(pre).all() or not np.isfinite(post).all():
                    raise ValueError('Non-finite activations; generation stopped.')
                from .tensor_checks import check
                err,precision=check(engine,table,controls,pre,post,enabled)
                logits=engine.logits()
                if first is None:first=pre.copy()
                last=post.copy()
                # Stop even when cancellation arrived during a native operation.
                if controller.cancelled(): reason='cancelled'; break
                p,token,sampling_p=draw(logits,rng,generated,temperature,top_k,top_p,presence_penalty)
                details=measurements(engine,bank,controls,pre,post,enabled)
                control_trace={}
                for item in details:
                    name=item['axis']
                    control_trace[name+'__control_layer']=item['layer']
                    control_trace[name+'__control_source']=item['vector_layer']
                    control_trace[name+'__control_projection_delta']=item['projection_delta']
                    if 'before' in item:
                        control_trace[name+'__control_before_z']=item['before']
                        control_trace[name+'__control_after_z']=item['after']
                eos_id=engine.model.get('eos_id')
                eos_probability=float(p[int(eos_id)]) if eos_id is not None and 0<=int(eos_id)<len(p) else None
                if engine.is_eog(token):
                    reason='end_of_generation'
                    terminal={'step':step,'token_id':token,'control_revision':revision,
                              'injection_error':err,'precision_check':precision,'interventions':details,'phase':phase,
                              'chosen_probability':float(p[token]),'sampling_probability':float(sampling_p),
                              'eos_probability':eos_probability,
                              'entropy_nats':float(-np.sum(p*np.log(p+1e-300)))}
                    break
                raw=engine.piece(token); output+=decoder.decode(raw); generated.append(token)
                top=np.argpartition(p,-min(5,len(p)))[-5:]; top=top[np.argsort(p[top])[::-1]]
                row={'step':step,'token_id':token,'token':raw.decode('utf-8','replace'),
                     'phase':phase,'control_revision':revision,'control_phase':settings['phase'],
                     'control_enabled':bool(enabled and any(float(c['dose']) for c in controls)),
                     'injection_error':err,'precision_check':precision,'interventions':details,
                     'entropy_nats':float(-np.sum(p*np.log(p+1e-300))),
                     'eos_probability':eos_probability,
                     'chosen_probability':float(p[token]),'sampling_probability':float(sampling_p),
                     'top_tokens':[{'token':engine.piece(int(i)).decode('utf-8','replace'),'probability':float(p[i])} for i in top],
                     'residual_norms':np.linalg.norm(post,axis=1).tolist(),**control_trace}
                layer_values={}
                for n in watch:
                    a=bank[n]; k=a['layer']
                    z=(np.einsum('ld,ld->l',post,a['directions'])-a['center'])/a['scale']
                    before=(np.einsum('ld,ld->l',pre,a['directions'])-a['center'])/a['scale']
                    row[n]=float(z[k]);row[n+'__before']=float(before[k]);layer_values[n]=z.tolist()
                traces.append(row);profiles.append(layer_values)
                parsed=split_output(output,start_phase,partial=True);phase=parsed['phase']
                yield {'type':'token','text':output,'trace':row,'profiles':layer_values,**parsed}
                if len(generated)>=24 and generated[-8:]==generated[-16:-8]==generated[-24:-16]:
                    reason='repetition_guard';break
                if token_delay:time.sleep(token_delay)
            output+=decoder.decode(b'',final=True)
        finally:
            if engine.handle:engine.clear_steering();engine.reset()
    final=split_output(output,start_phase)
    result={**original_identity,'model':engine.model['name'],'prompt':prompt,
            'prompt_sha256':hashlib.sha256(prompt.encode('utf-8')).hexdigest(),
            'prompt_tokens':len(tokens),'text':output,**final,'token_ids':generated,
            'trace':traces,'layer_profiles':profiles,'watch':list(watch),
            'controls':controls,'initial_controls':(initial_settings or {}).get('controls',[]),
            'control_events':events,'terminal_prediction':terminal,
            'scope':scope,'control_phase':(settings or {}).get('phase','all'),
            'dose_limit':(settings or {}).get('dose_limit',.5),
            'temperature':temperature,'seed':seed,'top_k':top_k,'top_p':top_p,
            'presence_penalty':presence_penalty,'max_tokens':max_tokens,
            'sampler':'numpy-topk-topp-presence-v1','presence_scope':'generated reply tokens',
            'token_delay':token_delay,'stop_reason':reason,'seconds':round(time.monotonic()-started,3),
            'answer_complete':bool(final['answer_text'] and not final['reasoning_unclosed'] and reason=='end_of_generation'),
            'phase_summary':phase_summary(traces,watch),'reasoning_start_phase':start_phase,
            'live_changes':len(events)>1,'generation_backend':engine.model.get('backend','incremental-native-gguf-r5'),
            'runtime_details':{k:engine.model.get(k) for k in ('adapter_path','base_path','activation_dtype','numerical_backend')},
            'probe_provenance':{n:bank[n]['meta'] for n in seen},
            'interpretation':'Native activation intervention, not an emotion probability or a complete mental-state readout. Earlier text and cached state are not erased when a control is disabled.'}
    result['saved_to']=write_result(result,'live-chat')
    if first is not None:
        from pathlib import Path
        np.savez_compressed(Path(result['saved_to']).with_suffix('.npz'),prompt_residual=first,final_residual=last)
    yield {'type':'done','result':result}
