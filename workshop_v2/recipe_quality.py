"""Report obvious recipe confounds. No model execution or behavioral claims."""
import numpy as np

def audit_recipe(recipe):
    rows=recipe.get('pairs',[])
    if not rows:raise ValueError('The recipe contains no pairs.')
    sides={side:[r[side].strip().casefold() for r in rows] for side in ('positive','negative')}
    by_text={}
    for r in rows:
        for side in sides:
            by_text.setdefault(r[side].strip().casefold(),set()).add(r['group'])
    duplicates=sum(len(groups)>1 for groups in by_text.values())
    return {'pairs':len(rows),'positive_distinct':len(set(sides['positive'])),
            'negative_distinct':len(set(sides['negative'])),
            'responses_reused_across_groups':duplicates,
            'mean_character_ratio':float(np.mean([len(x) for x in sides['positive']])/max(np.mean([len(x) for x in sides['negative']]),1)),
            'warning':'Repeated response text can leak wording across held-out context groups.' if duplicates else 'No exact cross-group response duplicates; semantic/template confounds still possible.'}
