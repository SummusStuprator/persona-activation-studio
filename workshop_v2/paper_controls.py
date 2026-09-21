"""Additional control directions using the attached Pain Axis paper's own datasets."""
import json,time
import numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
EXTRA={'paper_sadness':'SD_sadness_1P','paper_arousal':'Arousal_1P',
       'paper_numb':'Numb_1P','paper_random':'Random_1P'}

def datasets():
    from science import paper_data
    data=paper_data()
    data.update(json.loads((ROOT/'paper/datasets/3.1_sadness_dataset.json').read_text(encoding='utf-8'))['datasets'])
    return data

def train_extra(engine,targets=None,progress=None):
    from science import rows_for,collect,fit_axis
    neutral=[r for r in rows_for('S2_1P') if r['category']=='D']
    data=datasets()
    for name in (list(EXTRA) if targets is None else targets):
        if name not in EXTRA:raise ValueError('Unknown paper control.')
        source=data[EXTRA[name]]['sentences']
        rows=[{'text':r['prompt'],'category':name,'group':str(r['set'])} for r in source]+neutral
        x=collect(engine,rows,progress,pooling='last')
        labels=np.array([r['category']==name for r in rows])
        a=fit_axis(engine,rows,x,labels,name,
            'Pain Axis v1 section 3.2: '+EXTRA[name]+' vs S2 neutral D; local grouped train-only adaptation.',progress)
        path=ROOT/'vectors'/engine.model['digest']/(name+'.json')
        meta=json.loads(path.read_text(encoding='utf-8'))
        meta.update(paper_section='3.2',dataset=EXTRA[name],
            caveats='First-person control vs S2 neutral; neutral-only denoising. Not a happiness direction. Arousal dataset is positive high-intensity, not valence-independent arousal.',
            workshop_category='Emotions and affect' if name in ('paper_sadness','paper_arousal') else 'Research controls')
        path.write_text(json.dumps(meta,indent=2),encoding='utf-8')
    from science import load_bank
    return load_bank(engine)
