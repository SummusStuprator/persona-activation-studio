"""Native GGUF primitives executed inside the isolated worker, without weight copies."""
import numpy as np

def mean_transport(self, tokens, layer, vector=None, dose=0., skip_first=16):
    """Perturb all valid source positions, average final residuals over valid targets.

    By linearity, the directional derivative equals the reference estimator:
    mean_source sum_target>=source d(h_final_target)/d(h_layer_source).
    The final token is excluded, as in the author fitting code. Central finite
    differences replace backpropagation; no attention patterns are frozen.
    """
    tokens=np.ascontiguousarray(tokens,dtype=np.int32)
    if not 0<=int(layer)<self.layers:raise ValueError('Invalid source layer.')
    if skip_first<0 or len(tokens)<=skip_first+1:raise ValueError('Prompt has no valid source positions.')
    table=np.zeros((self.layers,self.dim),np.float32)
    if vector is not None:
        vector=np.asarray(vector,np.float32)
        if vector.shape!=(self.dim,) or not np.isfinite(vector).all():raise ValueError('Invalid perturbation.')
        table[int(layer)]=vector
    try:
        self.clear_steering();self.reset()
        if skip_first:self.evaluate(tokens[:skip_first])
        self.steer(table,float(dose),'add')
        self.evaluate(tokens[skip_first:-1],all_logits=True)
        h=self.capture(2)
        # capture(2) is before injection; correct a target at source layer.
        if int(layer)==self.layers-1 and vector is not None:h[-1]+=float(dose)*vector
        if not np.isfinite(h).all():raise ValueError('Non-finite mean residual.')
        return {'final_mean':h[-1], 'source_mean':h[int(layer)],'n_valid':len(tokens)-skip_first-1}
    finally:self.clear_steering();self.reset()

def unembed_residual(self, residual):
    """Use the actual graph's final normalization/unembedding on an arbitrary h.

    Replay a fixed carrier and replace its final block output additively. No
    approximation to the checkpoint's normalization, logit scales or tied head.
    """
    vector=np.asarray(residual,dtype=np.float32)
    if vector.shape!=(self.dim,) or not np.isfinite(vector).all():raise ValueError('Invalid transported residual.')
    tokens=self.tokenize('A local readout carrier.')
    try:
        self.clear_steering();self.reset();self.evaluate(tokens)
        base=self.capture(0)[-1]
        self.reset()
        if len(tokens)>1:self.evaluate(tokens[:-1])
        table=np.zeros((self.layers,self.dim),np.float32);table[-1]=vector-base
        self.steer(table,1.,'add');self.evaluate(tokens[-1:])
        actual=self.capture(1)[-1]
        error=float(np.max(np.abs(actual-vector)))
        if error>max(2e-4, float(np.max(np.abs(vector)))*2e-5):raise ValueError('Final residual replacement check failed.')
        return {'logits':self.logits(),'replacement_error':error}
    finally:self.clear_steering();self.reset()

def read_positions(self,text,positions):
    tokens=self.tokenize(text)
    selected=sorted({p if p>=0 else len(tokens)+p for p in positions})
    if not selected or selected[0]<0 or selected[-1]>=len(tokens):raise ValueError('Position out of bounds.')
    values={}
    try:
        self.clear_steering();self.reset();start=0
        for pos in selected:
            self.evaluate(tokens[start:pos+1]);values[pos]=self.capture();start=pos+1
        return {'token_ids':tokens.tolist(),'residuals':values}
    finally:self.clear_steering();self.reset()
