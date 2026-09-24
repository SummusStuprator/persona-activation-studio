from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import json
import os
import unittest
import model_store


class ModelDiscoveryTests(unittest.TestCase):
    def test_explicit_store_does_not_add_default_private_store(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.dict(os.environ, {'OLLAMA_MODELS': str(root / 'models')}), patch.object(model_store, 'ROOT', root):
                self.assertEqual(model_store.store_roots(), [(root / 'models').resolve()])

    def test_additional_stores_remain_explicit_and_deduplicated(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'lab-settings.json').write_text(json.dumps({'model_directories': [str(root / 'extra'), str(root / 'models')]}))
            with patch.dict(os.environ, {'OLLAMA_MODELS': str(root / 'models')}), patch.object(model_store, 'ROOT', root):
                self.assertEqual(model_store.store_roots(), [(root / 'models').resolve(), (root / 'extra').resolve()])
