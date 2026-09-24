"""Opt-in real-model rendering check; never starts training or a generation."""
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import os
from streamlit.testing.v1 import AppTest
from isolated_engine import Engine
from model_store import grouped_inventory
from workshop_v2 import runtime
import runtime_ui
from studio_paths import CODE_ROOT as ROOT
models,_=grouped_inventory()
model=next(m for m in models if m['name']=='llama3.2:1b' or 'llama3.2:1b' in m.get('aliases',[]))
engine=Engine()
try:
    engine.open(model,context=1024,mode='CPU')
    with TemporaryDirectory(prefix='studio-loaded-ui-') as tmp, patch.dict(os.environ,{'STUDIO_HOME':tmp}), patch.object(runtime,'get_engine',return_value=engine), patch.object(runtime_ui,'render',return_value=True):
        at=AppTest.from_file(str(ROOT/'studio_app.py'),default_timeout=60).run()
        pages=('Chat','Hell lab','Activations','Emotion library','Concept builder','Experiments','J-space','Paper reproduction')
        for page in pages:
            next(x for x in at.radio if x.label=='Studio').set_value(page).run()
            assert not at.exception,(page,at.exception)
            assert not at.error,(page,at.error)
            assert at.title,page
            print('LOADED PAGE PASS:',page,flush=True)
        print({'status':'passed','model':model['name'],'pages':list(pages),'scope':'rendering with real loaded model, no action buttons clicked'})
finally:
    engine.close()
