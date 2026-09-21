"""Inference-time observation and control. Never edits model weights."""
from __future__ import annotations
import codecs
import json
import math
import time
import uuid
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent

def identity(engine):
    return {'digest': engine.model['digest'], 'abi': engine.abi}

def write_result(result, kind='workshop'):
    target = ROOT / 'runs' / f'{time.strftime("%Y%m%d-%H%M%S")}-{uuid.uuid4().hex[:12]}-{kind}.json'
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    return str(target)

def validate_controls(engine, bank, controls, dose_limit=.5):
    from .steering_controls import validate
    return validate(engine,bank,controls,dose_limit)

def configure(engine, table, controls, enabled=True):
    if not enabled or not controls: engine.clear_steering(); return
    if controls[0]['mode'] == 'erase': engine.steer(table, float(controls[0]['dose']), 'erase')
    else: engine.steer(table, 1., 'add')

def probabilities(logits, temperature=0., top_k=40):
    x = np.asarray(logits, dtype=np.float64)
    if not np.isfinite(x).all(): raise ValueError('Non-finite logits; reduce intervention strength.')
    p = np.exp(x - x.max()); p /= p.sum()
    if temperature <= 0: return p, None
    k = min(int(top_k), len(x))
    ids = np.argpartition(x, -k)[-k:]
    q = np.exp((x[ids] - x[ids].max()) / temperature); q /= q.sum()
    return p, (ids, q)

def iter_generate(engine, text, bank, controls=(), watch=(), max_tokens=256,
                  temperature=0., seed=42, scope='generation', repetition_guard=True, control_phase='all', dose_limit=.5, top_k=40, top_p=1., presence_penalty=0.):
    """Stream complete UTF-8 text and real probes; always clear intervention on exit.

    generation scope includes the last prompt position (predicting first output token),
    but not earlier prompt tokens. History is text replay, not persistent hidden state.
    """
    started = time.monotonic()
    if scope not in ('generation', 'all'): raise ValueError('Invalid intervention scope.')
    if not 1 <= int(max_tokens) <= 4096: raise ValueError('Generate 1 to 4096 tokens.')
    if control_phase not in ('all','reasoning','answer'): raise ValueError('Unknown phase scope.')
    if scope=='all' and control_phase!='all': raise ValueError('Phase-selective steering requires reply-only scope.')
    from .reasoning import initial_phase,split_output,phase_summary
    start_phase=initial_phase(text); phase=start_phase
    def phase_enabled(): return control_phase=='all' or control_phase==phase
    if not math.isfinite(temperature) or not 0 <= temperature <= 2: raise ValueError('Invalid temperature.')
    if len(watch) > 24 or any(n not in bank for n in watch): raise ValueError('Watch up to 24 built probes.')
    from contracts import validate_axis
    for name in watch: validate_axis(engine,name,bank[name])
    table = validate_controls(engine, bank, controls, dose_limit)
    rng = np.random.default_rng(seed)
    tokens = engine.tokenize(text)
    if len(tokens) + max_tokens > engine.context:
        raise ValueError(f'Prompt has {len(tokens)} tokens; reserve {max_tokens} within context {engine.context}. Start a new chat or raise context.')
    traces, pieces, ids, profiles = [], [], [], []
    decoder = codecs.getincrementaldecoder('utf-8')('replace')
    output = ''; stop_reason = 'token_limit'; first_h = None; last_h = None
    with engine.lock:
        try:
            engine.clear_steering(); engine.reset()
            if scope == 'generation' and len(tokens) > 1:
                engine.evaluate(tokens[:-1]); configure(engine, table, controls, phase_enabled())
                engine.evaluate(tokens[-1:])
            else:
                configure(engine, table, controls, phase_enabled()); engine.evaluate(tokens)
            for step in range(int(max_tokens)):
                pre, post = engine.capture(0), engine.capture(1)
                if not np.isfinite(post).all(): raise ValueError('Non-finite residual stream.')
                if first_h is None: first_h = pre.copy()
                last_h = post.copy()
                logits = engine.logits()
                from .sampling import draw
                p, token, sampling_probability = draw(logits,rng,ids,temperature,top_k,top_p,presence_penalty)
                if engine.is_eog(token): stop_reason = 'end_of_generation'; break
                raw = engine.piece(token); output += decoder.decode(raw); pieces.append(raw); ids.append(token)
                top = np.argsort(p)[-5:][::-1]
                row = {'step': step, 'token_id': token, 'token': raw.decode('utf-8', 'replace'),
                       'phase': phase, 'control_phase': control_phase, 'control_enabled': bool(controls) and any(float(c['dose']) for c in controls) and phase_enabled(),
                       'entropy_nats': float(-np.sum(p * np.log(p + 1e-300))),
                       'chosen_probability': float(p[token]),'sampling_probability':sampling_probability,
                       'top_tokens': [{'token':engine.piece(int(i)).decode('utf-8','replace'), 'probability':float(p[i])} for i in top],
                       'residual_norms': np.linalg.norm(post, axis=1).tolist()}
                layer_values = {}
                for name in watch:
                    a = bank[name]; k = a['layer']
                    zpre = (np.einsum('ld,ld->l',pre,a['directions'])-a['center'])/a['scale']
                    zpost = (np.einsum('ld,ld->l',post,a['directions'])-a['center'])/a['scale']
                    row[name] = float(zpost[k]); row[name+'__before'] = float(zpre[k])
                    layer_values[name] = zpost.tolist()
                from .steering_controls import measurements, check_injection
                row['interventions'] = measurements(engine,bank,controls,pre,post,phase_enabled())
                from .tensor_checks import check as runtime_check
                row['injection_error'],row['precision_check'] = runtime_check(engine,table,controls,pre,post,phase_enabled())
                traces.append(row); profiles.append(layer_values)
                parsed=split_output(output,start_phase,partial=True)
                yield {'type':'token','text':output,'trace':row,'profiles':layer_values,**parsed}
                phase=parsed['phase']
                if repetition_guard and len(ids)>=24 and ids[-8:]==ids[-16:-8]==ids[-24:-16]:
                    stop_reason='repetition_guard'; break
                if step+1 < max_tokens:
                    configure(engine,table,controls,phase_enabled())
                    engine.evaluate(np.array([token],np.int32))
            output += decoder.decode(b'', final=True)
        finally:
            if engine.handle:
                engine.clear_steering(); engine.reset()
    result = {**identity(engine),'model':engine.model['name'],'prompt':text,'text':output,
              'controls':list(controls),'watch':list(watch),'dose_limit':dose_limit,'scope':scope,'seed':seed,'temperature':temperature,
              'top_k':top_k,'top_p':top_p,'presence_penalty':presence_penalty,'sampler':'numpy-topk-topp-presence-v1','presence_scope':'generated reply tokens',
              'trace':traces,'layer_profiles':profiles,'token_ids':ids,'prompt_tokens':len(tokens),
              'seconds':round(time.monotonic()-started,3),'stop_reason':stop_reason,'max_tokens':max_tokens,
              'probe_provenance':{n:bank[n]['meta'] for n in set(watch)|{c['axis'] for c in controls}},
              'interpretation':'Projections describe learned representations, not emotion probabilities or persistent feelings.'}
    result.update(split_output(output,start_phase))
    result['phase_summary']=phase_summary(traces,watch)
    result['control_phase']=control_phase
    result['reasoning_start_phase']=start_phase
    result['answer_complete']=bool(result['answer_text'] and not result['reasoning_unclosed'] and stop_reason=='end_of_generation')
    result['backend']=engine.model.get('backend','native_gguf')
    result['saved_to'] = write_result(result)
    if first_h is not None:
        np.savez_compressed(Path(result['saved_to']).with_suffix('.npz'),prompt_residual=first_h,final_residual=last_h)
    yield {'type':'done','result':result}

def run(engine, text, bank, **kwargs):
    result = None
    for event in iter_generate(engine,text,bank,**kwargs):
        if event['type']=='done': result=event['result']
    return result
