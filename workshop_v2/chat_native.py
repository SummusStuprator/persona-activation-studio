"""Teacher-forced response measurements in the same native GGUF model worker."""
import numpy as np

def _parts(self,prefix,response):
    left=self.tokenize(prefix);full=self.tokenize(prefix+response)
    common=0
    for a,b in zip(left,full):
        if a!=b:break
        common+=1
    if common==len(full):raise ValueError('Empty tokenized continuation.')
    if len(full)>self.context:raise ValueError('Training example exceeds context.')
    return full,common

def extract_response(self,prefix,response):
    tokens,start=_parts(self,prefix,response)
    try:
        self.clear_steering();self.reset()
        if start:self.evaluate(tokens[:start])
        self.evaluate(tokens[start:],all_logits=True)
        return {'mean':self.capture(2),'last':self.capture(0),'response_tokens':len(tokens)-start,'prefix_tokens':start}
    finally:self.clear_steering();self.reset()

def score_response(self,prefix,response,table,max_tokens=8):
    tokens,start=_parts(self,prefix,response)
    if start<1:raise ValueError('Need a nonempty conditioning prefix.')
    scores=[]
    try:
        self.clear_steering();self.reset()
        if start>1:self.evaluate(tokens[:start-1])
        self.steer(np.ascontiguousarray(table,np.float32),1.,'add')
        for j in range(start,min(len(tokens),start+int(max_tokens))):
            self.evaluate(tokens[j-1:j]);logits=self.logits().astype(np.float64)
            z=logits.max();logp=logits[int(tokens[j])]-z-np.log(np.exp(logits-z).sum())
            scores.append(float(logp))
        return {'mean_log_probability':float(np.mean(scores)),'token_log_probabilities':scores,'n':len(scores)}
    finally:self.clear_steering();self.reset()
