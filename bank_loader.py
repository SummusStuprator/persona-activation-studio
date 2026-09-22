"""Validate local direction files before making them selectable."""
import hashlib
import json
from pathlib import Path
import numpy as np
from contracts import validate_axis, slug
from studio_paths import data_root, ASSET_ROOT, CODE_ROOT
ROOT = data_root()
def load_bank(engine):
    bank, diagnostics = {}, []
    if not engine.handle: return bank
    folder = ROOT/'vectors'/engine.model['digest']
    for path in sorted(folder.glob('*.json')):
        try:
            meta = json.loads(path.read_text(encoding='utf-8'))
            name = meta['name']
            if path.stem != slug(name) or meta.get('imported_from'):
                raise ValueError('Noncanonical or imported direction; not loaded.')
            if meta['abi'] != engine.abi:
                raise ValueError('Runtime differs; retrain for this runtime.')
            arrays = path.with_suffix('.npz')
            expected = meta.get('arrays_sha256')
            if not expected or hashlib.sha256(arrays.read_bytes()).hexdigest()!=expected:
                raise ValueError('Unsealed or changed direction arrays; retrain.')
            with np.load(arrays,allow_pickle=False) as data:
                item = {k:data[k] for k in ('directions','center','scale','norm')}
            item.update(layer=meta['layer'],meta=meta)
            validate_axis(engine,name,item)
            if name in bank: raise ValueError('Duplicate direction name.')
            bank[name] = item
        except Exception as exc:
            diagnostics.append({'file':path.name,'reason':str(exc)})
    engine.bank_diagnostics = diagnostics
    return bank
