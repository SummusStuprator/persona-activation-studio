import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import integrity
from studio_paths import ASSET_ROOT
from persona.dataset_builder import InputStore
from persona.benchmark import generate_single
from persona.benchmark_identity import require_compatible
from workshop_v2.paper_reproduction import original_prompts


class ReleaseTests(unittest.TestCase):
    def test_runtime_assets(self):
        for name in ('studio.toml.example', 'native/bridge.cpp', 'native/CMakeLists.txt',
                     'docs/INSTALL.md', 'paper/datasets/4.2_neutral_50.json', 'tests/ui_smoke.py'):
            self.assertTrue((ASSET_ROOT/name).is_file(), name)
        prompts = original_prompts()
        self.assertEqual(len(prompts), 50)
        self.assertEqual(len(set(prompts)), 50)

    def test_benchmark_sampler(self):
        calls = []
        trainer = SimpleNamespace(generate_response=lambda *a, **k: calls.append(k))
        for temperature in (0., .7):
            generate_single(trainer, None, None, [], seed=1, max_new_tokens=8,
                            temperature=temperature, top_p=.9, top_k=20)
        self.assertEqual([call['do_sample'] for call in calls], [False, True])

    def test_external_export_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); exports = root/'external'; exports.mkdir()
            (exports/'all_posts.csv').write_text('id,text\n1,example\n')
            store = InputStore(root/'project', exports)
            self.assertEqual(store.read_csv(['x_exports/all_posts.csv'])[0]['text'], 'example')

    def test_benchmark_rejects_unidentified_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'nll_scores.json').write_text('{}')
            with self.assertRaises(ValueError):
                require_compatible(root, 'new')
            (root/'benchmark_manifest.json').write_text('{"fingerprint":"new"}')
            require_compatible(root, 'new')

    def test_seal_detects_changed_and_added_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root/'module.py'; source.write_text('x=1\n')
            (root/'pyproject.toml').write_text('[tool.setuptools]\npy-modules=["module","new"]\n')
            with patch.object(integrity, 'CODE_ROOT', root), patch.object(integrity, 'ASSET_ROOT', root), patch.object(integrity, 'MANIFEST', root/'seal.json'):
                integrity.seal_release(); integrity.verify_release()
                previous=source.stat().st_mtime_ns
                source.write_text('x=2\n')
                import os
                os.utime(source,ns=(previous,previous))
                with self.assertRaises(RuntimeError): integrity.verify_release()
                integrity.seal_release()
                (root/'new.py').write_text('y=1\n')
                with self.assertRaises(RuntimeError): integrity.verify_release()

    def test_empty_seal_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory)/'seal.json'; manifest.write_text('{}')
            with patch.object(integrity, 'MANIFEST', manifest):
                with self.assertRaises(RuntimeError): integrity.verify_release()

    def test_demo_has_disjoint_splits(self):
        from studio_demo import create_dataset
        with tempfile.TemporaryDirectory() as directory:
            folder = create_dataset(Path(directory))
            identifiers = []
            for name in ('train_authentic.jsonl', 'validation.jsonl', 'test.jsonl'):
                rows = [json.loads(line) for line in (folder/name).read_text().splitlines()]
                self.assertTrue(rows)
                identifiers.extend(row['id'] for row in rows)
            self.assertEqual(len(identifiers), len(set(identifiers)))

    def test_cancel_refuses_reused_pid(self):
        import studio_jobs
        process=SimpleNamespace(pid=999999,create_time=lambda:500.)
        with patch.object(studio_jobs,'read_state',return_value={'status':'running','pid':999999,'process_started':100.}), patch.object(studio_jobs.psutil,'Process',return_value=process):
            with self.assertRaises(ValueError):studio_jobs.cancel('unused')
