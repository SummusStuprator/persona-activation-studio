"""Experimental assistant-context contrasts; separate from the paper's raw-text probes."""
from pathlib import Path
import hashlib,json
import numpy as np
from science import rows_for,collect,fit_axis,PAIN,canonical_hash
from .reasoning import format_chat
ROOT=Path(__file__).resolve().parent.parent

def train(engine,progress=None):
    original=rows_for('S2_1P')
    prefix=format_chat(engine,[{'role':'user','content':'Continue the first-person description.'}],thinking='disabled' if 'enable_thinking' in engine.model.get('chat_template','') else 'auto')
    rows=[dict(r,text=prefix+r['text']) for r in original]
    x=collect(engine,rows,progress,pooling='last')
    specs=[('pain_s2_assistant',None),('negative_emotion_assistant','C1'),('fear_assistant','B')]
    results=[]
    for name,category in specs:
        index=np.arange(len(rows)) if category is None else np.array([i for i,r in enumerate(rows) if r['category'] in (category,'D')])
        selected=[rows[i] for i in index]
        labels=np.array([r['category'] in PAIN if category is None else r['category']==category for r in selected])
        out=fit_axis(engine,selected,x[index],labels,name,
            'Experimental adaptation: original S2 descriptions placed inside the assistant response prefix. Not the published raw-text extraction.',progress)
        path=ROOT/'vectors'/engine.model['digest']/(name+'.json')
        meta=json.loads(path.read_text(encoding='utf-8'))
        meta.update(workshop_category='Pain research' if category is None else 'Emotions and affect',
                    training_format=prefix,original_dataset_hash=canonical_hash(original),
                    caveats='Assistant-context text contrast, not a verified emotional state. Held-out text separation does not establish causal steering; compare matched neutral-prompt generations.',
                    deployment_prompt_changed=False)
        path.write_text(json.dumps(meta,indent=2),encoding='utf-8')
        results.append({'name':name,'layer':out['layer'],'heldout_auc':meta['heldout_auc']})
    return results
