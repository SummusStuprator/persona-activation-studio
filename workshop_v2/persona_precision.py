"""Backend-aware numeric validation, without relaxing the GGUF checks.

BF16 rounding is an expected quantization error, not a failure of injection.
The residual difference and rounding discrepancy remain separately measurable.
"""
import numpy as np

def expected_delta(table,controls,pre,enabled):
    if not controls or not enabled:return np.zeros_like(pre)
    if controls[0]['mode']=='add':return table
    unit=table/np.maximum(np.linalg.norm(table,axis=1,keepdims=True),1e-12)
    return -float(controls[0]['dose'])*np.sum(pre*unit,axis=1,keepdims=True)*unit

def rounding_metrics(table,controls,pre,post,enabled,storage='bfloat16'):
    delta=expected_delta(table,controls,pre,enabled)
    wanted=pre+delta
    # BF16 spacing is 2^(floor(log2(abs(x)))-7); allow half a storage ULP.
    exp=np.floor(np.log2(np.maximum(np.abs(wanted),np.finfo(np.float32).tiny)))
    mantissa = {'bfloat16':7, 'float16':10}[storage]
    half_ulp=np.exp2(exp-mantissa-1)
    tolerance=half_ulp+1e-4*(1+np.abs(delta))
    raw=np.abs(post-wanted)
    excess=np.maximum(raw-tolerance,0)
    if np.max(excess)>0:raise RuntimeError('Persona activation differs from requested change beyond BF16 rounding: '+str(float(np.max(excess))))
    return {'max_rounding_error':float(np.max(raw)),'max_excess_error':float(np.max(excess)),
            'storage':storage,'check':'elementwise half-ULP plus float32 arithmetic tolerance'}
