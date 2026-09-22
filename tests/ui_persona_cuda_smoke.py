"""Exercise the actual persona sidebar load action on CUDA."""
import argparse
import json
from pathlib import Path
from unittest.mock import patch
from streamlit.testing.v1 import AppTest
from isolated_engine import Engine
from workshop_v2 import runtime
from workshop_v2.persona_store import discover
from studio_paths import CODE_ROOT

parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True)
parser.add_argument('--output', required=True)
args = parser.parse_args()
model = next(m for m in discover()[0] if m['name'] == args.model)
engine = Engine()
try:
    with patch.object(runtime, 'get_engine', return_value=engine):
        at = AppTest.from_file(str(CODE_ROOT/'studio_app.py'), default_timeout=120).run()
        next(x for x in at.radio if x.label == 'Studio').set_value('Chat').run()
        next(x for x in at.selectbox if x.key == 'model_source').set_value('Persona adapters').run()
        next(x for x in at.checkbox if x.label.startswith('Include installed')).set_value(True).run()
        next(x for x in at.selectbox if x.key == 'persona_choice').set_value(model['selection_id']).run()
        next(x for x in at.selectbox if x.key == 'persona_device').set_value('CUDA').run()
        next(x for x in at.selectbox if x.key == 'persona_precision').set_value('nf4').run()
        next(x for x in at.selectbox if x.key == 'persona_context').set_value(512).run()
        next(x for x in at.button if x.label == 'Load persona model').click().run()
        assert not at.exception and not at.error, (at.exception, at.error)
        assert engine.handle and engine.load_plan['device'] == 'cuda:0'
        assert engine.load_plan['precision'] == 'nf4'
        report = {'status': 'passed', 'model': args.model, 'action': 'Load persona model',
                  'device': engine.load_plan['device'], 'precision': engine.load_plan['precision']}
        Path(args.output).write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps(report))
finally:
    engine.close()
