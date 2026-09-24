"""Exercise every navigation destination without loading model weights."""
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import os
from streamlit.testing.v1 import AppTest
from studio_paths import CODE_ROOT as ROOT
PAGES = ('Overview', 'Collect X data', 'Build dataset', 'Train persona',
         'Chat', 'Hell lab', 'Activations', 'Emotion library', 'Concept builder',
         'Experiments', 'J-space', 'Paper reproduction', 'Jobs', 'Guide')
with TemporaryDirectory(prefix='studio-ui-test-') as tmp:
    with patch.dict(os.environ, {'STUDIO_HOME': tmp}):
        at = AppTest.from_file(str(ROOT/'studio_app.py'), default_timeout=60).run()
        assert not at.exception, at.exception
        nav = next(x for x in at.radio if x.label == 'Studio')
        assert set(PAGES) == set(nav.options), nav.options
        visited = []
        for page in PAGES:
            nav = next(x for x in at.radio if x.label == 'Studio')
            nav.set_value(page).run()
            assert not at.exception, (page, at.exception)
            assert at.title, f'{page}: blank page (no rendered title)'
            assert at.markdown or at.info or at.dataframe or at.text_input or at.expander or at.metric, f'{page}: no usable content'
            visited.append(page)
        print({'status':'passed', 'pages_actually_visited':visited,
               'count':len(visited), 'scope':'unloaded-model navigation; no inference'})
