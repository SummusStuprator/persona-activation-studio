"""Fail-closed validation of checkpoint-specific activation directions."""
import re
import numpy as np

def validate_axis(engine, name, item):
    meta = item['meta']
    if meta.get('name') != name:
        raise ValueError('Direction name and metadata disagree.')
    if meta.get('imported_from'):
        raise ValueError('Imported vectors require native-GGUF retraining before use.')
    if meta.get('model', {}).get('digest') != engine.model['digest']:
        raise ValueError('Direction belongs to different model weights.')
    if meta.get('abi') != engine.abi:
        raise ValueError('Direction belongs to a different native runtime.')
    layers, dim = engine.layers, engine.dim
    if not 0 <= int(item['layer']) < layers:
        raise ValueError('Invalid direction layer.')
    for key, shape in [('directions',(layers,dim)),('center',(layers,)),
                       ('scale',(layers,)),('norm',(layers,))]:
        value = np.asarray(item[key])
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f'Invalid {key} shape or non-finite values.')
    if np.any(item['scale'] <= 0) or np.any(item['norm'] <= 0):
        raise ValueError('Reference scales and residual norms must be positive.')
    if not np.allclose(np.linalg.norm(item['directions'],axis=1),1,atol=2e-5):
        raise ValueError('Directions must be unit vectors at every layer.')

def slug(name):
    return re.sub(r'[^a-z0-9_]+','_',name.lower()).strip('_')[:48]
