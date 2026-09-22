"""CUDA direction fitting, bank reload, and live intervention integration."""
import argparse
import json
from pathlib import Path
from isolated_engine import Engine
from workshop_v2.persona_store import discover
from science import train_custom, load_bank
from workshop_v2.core import run

parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True)
parser.add_argument('--output', required=True)
args = parser.parse_args()
model = next(m for m in discover()[0] if m['name'] == args.model)
engine = Engine()
try:
    engine.open(model, context=512, mode='CUDA', precision='nf4')
    objects = 'cup bowl plate chair door window wall box bag coat shirt hat pen book table shelf lamp vase ribbon tile'.split()
    positive = [f'The {item} is red.' for item in objects]
    negative = [f'The {item} is blue.' for item in objects]
    name = 'fixture_color_red'
    train_custom(engine, positive, negative, name)
    bank = load_bank(engine)
    assert name in bank
    control = {'axis': name, 'layer': int(bank[name]['layer']), 'dose': .05, 'mode': 'add'}
    prompt = engine.chat([{'role': 'user', 'content': 'Describe a room in one sentence.'}])
    result = run(engine, prompt, bank, controls=[control], watch=[name], max_tokens=32)
    assert result['token_ids'] and result['trace']
    assert all(row.get('precision_check', {}).get('max_excess_error') == 0 for row in result['trace'])
    report = {'status': 'passed', 'model': model['name'], 'device': engine.load_plan['device'],
              'precision': engine.load_plan['precision'], 'abi': engine.abi,
              'direction': name, 'tokens': len(result['token_ids']),
              'max_excess_error': 0, 'audit': result['saved_to']}
    Path(args.output).write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
finally:
    engine.close()
