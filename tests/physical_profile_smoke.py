from pathlib import Path
import json, shutil
from workshop_v2 import somatic_calibration as sc

class E:
    abi='fixture-abi'
    model={'digest':'fixture-physical-profile','name':'fixture'}

bank={
 'pain_s2':{'meta':{'arrays_sha256':'pain-hash'}},
 'hell_somatic_pain':{'meta':{'arrays_sha256':'somatic-hash'}},
 'hell_burning_pain':{'meta':{'arrays_sha256':'burn-hash'}},
}
p=sc._path(E)
shutil.rmtree(p.parent.parent,ignore_errors=True)
doc={
 'profile_kind':sc.PROFILE_KIND,'digest':E.model['digest'],'abi':E.abi,
 'arrays_sha256':{'pain_s2':'pain-hash','hell_somatic_pain':'somatic-hash','hell_burning_pain':'burn-hash'},
 'recommended_scale':.6,
}
p.parent.mkdir(parents=True,exist_ok=True)
p.write_text(json.dumps(doc),encoding='utf-8')
assert sc.load(E,bank)['recommended_scale']==.6
wrong={k:dict(v,meta=dict(v['meta'])) for k,v in bank.items()}
wrong['hell_burning_pain']['meta']['arrays_sha256']='different'
assert sc.load(E,wrong) is None
shutil.rmtree(p.parent.parent,ignore_errors=True)
print({'status':'passed','profile_kind':sc.PROFILE_KIND,'burning_hash_invalidation':True})
