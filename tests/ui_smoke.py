from pathlib import Path
from streamlit.testing.v1 import AppTest
ROOT=Path(__file__).resolve().parents[1]
at=AppTest.from_file(str(ROOT/'studio_app.py'),default_timeout=40).run()
assert not at.exception,at.exception
nav=next(x for x in at.radio if x.label=='Studio')
expected={'Overview','Collect X data','Build dataset','Train persona','Chat','Hell lab','Activations','Emotion library','Concept builder','Experiments','J-space','Paper reproduction','Jobs','Guide'}
assert expected.issubset(set(nav.options)),nav.options
for page in ('Collect X data','Build dataset','Train persona','Guide','Hell lab','Activations','Concept builder','Experiments'):
    nav=next(x for x in at.radio if x.label=='Studio')
    nav.set_value(page).run()
    assert not at.exception,(page,at.exception)
print({'status':'passed','pages':len(expected)})