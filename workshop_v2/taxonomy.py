"""Distinct UI categories; a bridge topic is never an emotional measurement."""
EMOTION_NAMES={'fear','negative_emotion','paper_sadness','paper_arousal','state_calmness','state_social_warmth','state_terror'}
BODY_NAMES={'bodily_sensation','state_physical_warmth'}
CONTROL_NAMES={'negative_world','paper_random','paper_numb'}

def category(name, axis=None):
    meta=(axis or {}).get('meta',{})
    if meta.get('recipe_file'):return meta.get('workshop_category','Concepts')
    if name=='golden_gate' or name.startswith('concept_'):return 'Concepts'
    if name.startswith('pain_') or name.startswith('paperrep_'):return 'Pain research'
    if name.startswith('emotion_') or name in EMOTION_NAMES:return 'Emotions and affect'
    if name in BODY_NAMES:return 'Bodily sensations'
    if name in CONTROL_NAMES:return 'Research controls'
    return meta.get('workshop_category','Concepts')

def groups(bank):
    return {group:[n for n,a in bank.items() if category(n,a)==group]
            for group in ('Emotions and affect','Behavior and style','Concepts','Bodily sensations','Pain research','Research controls')}
