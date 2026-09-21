from pathlib import Path
import shutil, threading
import numpy as np
from workshop_v2.chat_training import _paired_response_activations, ROOT

class E:
    layers=3;dim=4;abi='fixture-abi-v1'
    model={'digest':'fixture-pair-cache-smoke','chat_template':''}
    lock=threading.RLock()
    def __init__(self):self.calls=0
    def chat(self,messages):return 'PROMPT:'+messages[-1]['content']+'\nASSISTANT:'
    def extract_response(self,prefix,response):
        self.calls+=1
        base=float(sum(map(ord,response))%17)
        return {'mean':np.full((self.layers,self.dim),base,np.float32)}

rows=[
 {'group':'0','prompt':'p0','positive':'positive zero','negative':'negative zero'},
 {'group':'1','prompt':'p1','positive':'positive one','negative':'negative one'},
]
folder=ROOT/'cache'/'paired-responses'/E.model['digest']/'recipe-fixture'
shutil.rmtree(folder.parent,ignore_errors=True)
e=E()
a=_paired_response_activations(e,rows,'recipe-fixture')
assert e.calls==4,e.calls
e.calls=0
b=_paired_response_activations(e,rows,'recipe-fixture')
assert e.calls==0,e.calls
assert np.array_equal(a,b)
part=folder/'0000.npz'
with np.load(part,allow_pickle=False) as z:arr=z['activations']
arr=arr.copy();arr[0,0,0]+=1
np.savez_compressed(part,activations=arr,sha256='bad')
e.calls=0
c=_paired_response_activations(e,rows,'recipe-fixture')
assert e.calls==2,e.calls
assert np.array_equal(a,c)
shutil.rmtree(folder.parent,ignore_errors=True)
print({'status':'passed','first_extracts':4,'cached_extracts':0,'corrupt_pair_reextracts':2})
