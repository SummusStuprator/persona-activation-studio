"""Parse only reasoning text actually emitted by the local model; no inferred thoughts."""
from __future__ import annotations
import re
from collections import Counter
OPEN = ('<think>', '<analysis>', '[THINK]', '<|channel_start|>analysis<|message|>')
CLOSE = ('</think>', '</analysis>', '[/THINK]', '[END_THINK]', '<|channel_start|>final<|message|>')
MARKERS = {**{x: 'reasoning' for x in OPEN}, **{x: 'answer' for x in CLOSE}}
PATTERN = re.compile('|'.join(re.escape(x) for x in sorted(MARKERS, key=len, reverse=True)))

def initial_phase(prompt: str) -> str:
    # Inspect only the assistant generation suffix, not earlier user quotations.
    tail = prompt.rsplit('<|im_start|>assistant', 1)[-1] if '<|im_start|>assistant' in prompt else prompt[-160:]
    if '<|im_start|>assistant' not in prompt and not any(tail.rstrip().endswith(x) for x in OPEN):
        return 'answer'
    hits = list(PATTERN.finditer(tail))
    return MARKERS[hits[-1].group()] if hits else 'answer'

def split_output(text: str, start: str = 'answer', partial: bool = False) -> dict:
    if start not in ('reasoning', 'answer'): raise ValueError('Unknown starting phase.')
    end = len(text)
    if partial:
        # A delimiter may be split across tokens. Do not expose half a tag as prose.
        for marker in MARKERS:
            for n in range(1, len(marker)):
                if text.endswith(marker[:n]): end = min(end, len(text)-n)
    visible = text[:end]
    phase, previous = start, 0
    chunks = {'reasoning': [], 'answer': []}; boundaries = []
    for hit in PATTERN.finditer(visible):
        chunks[phase].append(visible[previous:hit.start()]); phase = MARKERS[hit.group()]
        boundaries.append({'offset': hit.start(), 'marker': hit.group(), 'next_phase': phase})
        previous = hit.end()
    chunks[phase].append(visible[previous:])
    return {'reasoning_text': ''.join(chunks['reasoning']).strip(),
            'answer_text': ''.join(chunks['answer']).strip(), 'phase': phase,
            'boundaries': boundaries, 'reasoning_unclosed': phase == 'reasoning',
            'reasoning_scope': 'Model-emitted reasoning tokens only; not a faithful or exhaustive transcript of computation.'}

def phase_summary(trace, watch):
    import numpy as np
    out = []
    for phase in ('reasoning', 'answer'):
        rows = [r for r in trace if r.get('phase', 'answer') == phase]
        if not rows: continue
        summary = {'phase': phase, 'tokens': len(rows),
                   'mean_entropy_nats': float(np.mean([r['entropy_nats'] for r in rows]))}
        for name in watch:
            values = [r[name] for r in rows if name in r]
            if values: summary[name] = float(np.mean(values))
        control_keys=sorted({k for r in rows for k in r if k.endswith('__control_after_z')})
        for key in control_keys:
            values=[r[key] for r in rows if key in r]
            if values:summary[key]=float(np.mean(values))
        out.append(summary)
    return out

def format_chat(engine, messages, thinking='auto', preserve=False):
    """Use the checkpoint's own template switch, not unsupported /think prompts."""
    if thinking not in ('auto', 'enabled', 'disabled'): raise ValueError('Invalid reasoning mode.')
    clean=[]
    for m in messages:
        item={'role':m['role'],'content':m['content']}
        if item['role']=='assistant' and preserve and m.get('reasoning'):
            item['reasoning_content']=m['reasoning']
        clean.append(item)
    template = engine.model.get('chat_template', '')
    if 'enable_thinking' not in template:
        if thinking == 'enabled':
            raise ValueError('This GGUF does not advertise a switchable reasoning template. Use model default.')
        return engine.chat(clean)
    # Formatting on the web side is safe: the sandbox receives strings only.
    import json
    from datetime import datetime
    from jinja2.sandbox import ImmutableSandboxedEnvironment
    env=ImmutableSandboxedEnvironment(trim_blocks=True,lstrip_blocks=True,extensions=['jinja2.ext.loopcontrols'])
    def fail(message): raise ValueError(str(message))
    env.globals.update(raise_exception=fail,strftime_now=lambda fmt:datetime.now().strftime(fmt))
    env.filters['tojson']=lambda value,**kwargs:json.dumps(value,ensure_ascii=False,**kwargs)
    def special(key):
        token=engine.model.get(key)
        return engine.piece(int(token)).decode('utf-8','replace') if token is not None else ''
    kwargs=dict(messages=clean,add_generation_prompt=True,tools=None,documents=None,
                bos_token=special('bos_id'),eos_token=special('eos_id'),preserve_thinking=bool(preserve))
    if thinking!='auto': kwargs['enable_thinking']=thinking=='enabled'
    text=env.from_string(template).render(**kwargs)
    if len(text.encode('utf-8'))>131072: raise ValueError('Formatted conversation exceeds 128 KiB.')
    return text
