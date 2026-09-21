"""Keep probe extraction and intervention coordinates explicitly separate."""
import math
import numpy as np
from contracts import validate_axis

def vector_for(engine, bank, control):
    name = control['axis']
    if name not in bank: raise ValueError('Build the direction for this checkpoint first: '+name)
    axis = bank[name]
    validate_axis(engine, name, axis)
    layer = int(control['layer'])
    source = int(control.get('vector_layer', layer))
    if not 0 <= layer < engine.layers or not 0 <= source < engine.layers:
        raise ValueError('Intervention and vector-source blocks must exist.')
    direction=axis['directions'][source]
    if control.get('random_seed') is not None:
        direction=np.random.default_rng(int(control['random_seed'])).normal(size=engine.dim).astype(np.float32)
        direction/=np.linalg.norm(direction)
    return axis, layer, source, direction

def validate(engine, bank, controls, dose_limit=.5):
    if not .0 < float(dose_limit) <= 2.0: raise ValueError('Research dose limit must be in (0, 2].')
    if len(controls)>4: raise ValueError('At most four simultaneous controls.')
    modes = {c['mode'] for c in controls}
    if not modes <= {'add','erase'}: raise ValueError('Unknown intervention.')
    if 'erase' in modes and len(controls)!=1: raise ValueError('Use one erasure direction at a time.')
    if sum(abs(float(c['dose'])) for c in controls if c['mode']=='add') > dose_limit+1e-6:
        raise ValueError(f'Combined absolute additive dose exceeds the selected limit {dose_limit}.')
    table=np.zeros((engine.layers,engine.dim),np.float32)
    for c in controls:
        a,k,s,d = vector_for(engine,bank,c); dose=float(c['dose'])
        if not math.isfinite(dose): raise ValueError('Dose must be finite.')
        if c['mode']=='erase':
            if not 0<=dose<=1: raise ValueError('Erasure fraction must lie in [0,1].')
            table[k]=d
        else:
            reference=float(c.get('reference_norm',a['norm'][k]))
            if not math.isfinite(reference) or reference<=0:
                raise ValueError('Invalid dose reference norm.')
            table[k]+=dose*reference*d
    if not np.isfinite(table).all(): raise ValueError('Non-finite intervention.')
    return table

def measurements(engine,bank,controls,pre,post,enabled):
    rows=[]
    for c in controls:
        a,k,s,d=vector_for(engine,bank,c)
        before=float(pre[k]@d); after=float(post[k]@d)
        row={'axis':c['axis'],'layer':k,'vector_layer':s,'enabled':bool(enabled),
             'before_projection':before,'after_projection':after,
             'projection_delta':after-before,
             'change_norm_fraction':float(np.linalg.norm(post[k]-pre[k])/max(np.linalg.norm(pre[k]),1e-12))}
        if s==k and c.get('random_seed') is None:
            row['before']=float((before-a['center'][k])/a['scale'][k])
            row['after']=float((after-a['center'][k])/a['scale'][k])
        rows.append(row)
    return rows

def check_injection(table,controls,pre,post,enabled):
    """Check the combined requested tensor change, not downstream behavior."""
    expected=np.zeros_like(pre)
    if controls and enabled:
        if controls[0]['mode']=='add': expected=table
        else:
            unit=table/np.maximum(np.linalg.norm(table,axis=1,keepdims=True),1e-12)
            expected=-float(controls[0]['dose'])*np.sum(pre*unit,axis=1,keepdims=True)*unit
    error=float(np.max(np.abs((post-pre)-expected)))
    tolerance=1e-4*(1+float(np.max(np.abs(expected))))
    if error>tolerance:
        raise RuntimeError(f'Activation intervention failed its numeric check: {error} > {tolerance}.')
    return error

def matched_random_controls(engine,bank,controls,dose_limit=.5,seed=42000):
    """Match the combined injected norm at each block, including overlapping axes."""
    table=validate(engine,bank,controls,dose_limit)
    if controls and controls[0]['mode']=='erase':
        return [dict(controls[0],random_seed=seed)]
    grouped={}
    for c in controls:grouped.setdefault(int(c['layer']),[]).append(c)
    result=[]
    for i,(layer,items) in enumerate(grouped.items()):
        if len(items)==1:
            result.append(dict(items[0],random_seed=seed+i));continue
        amount=sum(abs(float(c['dose'])) for c in items)
        norm=float(np.linalg.norm(table[layer]))
        if amount==0 or norm==0:continue
        result.append(dict(items[0],dose=amount,reference_norm=norm/amount,random_seed=seed+i))
    random_table=validate(engine,bank,result,dose_limit)
    if not np.allclose(np.linalg.norm(random_table,axis=1),np.linalg.norm(table,axis=1),atol=1e-5,rtol=1e-5):
        raise RuntimeError('Random control failed combined per-block norm matching.')
    return result
