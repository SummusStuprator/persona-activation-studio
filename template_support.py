"""Fallback for newer GGUF Jinja chat templates; no model downloads."""
import json
from datetime import datetime
from jinja2.sandbox import ImmutableSandboxedEnvironment
from model_store import metadata

def render_chat(engine, messages):
    info = metadata(engine.model['path'])
    template = info.get('chat_template')
    if not isinstance(template, str) or not template:
        if info.get('architecture') == 'gemma4':
            # Official format: ai.google.dev/gemma/docs/core/prompt-formatting-gemma4
            roles={'system':'system','user':'user','assistant':'model'}
            for special_token in ('<|turn>','<turn|>'):
                ids=engine.tokenize(special_token)
                if engine.piece(int(ids[-1])).decode('utf-8','replace') != special_token:
                    raise ValueError('Gemma 4 turn tokens are not supported by this vocabulary.')
            result=''.join('<|turn>'+roles[m['role']]+'\n'+m['content']+'<turn|>\n' for m in messages)+'<|turn>model\n'
            if len(result.encode('utf-8')) > 131072: raise ValueError('Formatted prompt exceeds 128 KiB.')
            return result
        raise ValueError('This GGUF has no usable embedded chat template. Select raw text mode.')
    def fail(message): raise ValueError(str(message))
    def special(key):
        token = info.get(key)
        return engine.piece(int(token)).decode('utf-8', 'replace') if token is not None else ''
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                        extensions=['jinja2.ext.loopcontrols'])
    env.globals.update(raise_exception=fail, strftime_now=lambda fmt: datetime.now().strftime(fmt))
    env.filters['tojson'] = lambda value, **kwargs: json.dumps(value, ensure_ascii=False, **kwargs)
    try:
        result = env.from_string(template).render(messages=messages, add_generation_prompt=True,
            bos_token=special('bos_id'), eos_token=special('eos_id'), tools=None, documents=None,
            enable_thinking=False, thinking=False)
    except Exception as exc:
        raise ValueError('Embedded GGUF template could not be rendered; use raw mode. '+str(exc)) from exc
    if len(result.encode('utf-8')) > 131072: raise ValueError('Formatted prompt exceeds 128 KiB.')
    return result
