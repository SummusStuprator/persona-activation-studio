"""Model-free regressions; all writable state is temporary."""
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from training_runs import resolve_training_run, has_checkpoint
from workshop_v2 import resource_policy as rp
from workshop_v2 import somatic_calibration as sc

class TrainingResumeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='studio-resume-test-')
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.p = SimpleNamespace(workspace=root, datasets=root/'datasets', staging=root/'staging', anchors=root/'anchors')
        for p in (self.p.datasets/'r1', self.p.staging, self.p.anchors):p.mkdir(parents=True)
        (root/'profiles.json').write_text('{"profiles":[{"profile":"example"}]}')
        self.args = Namespace(profile='example', model=None, dataset=None, config=None, anchors=None, max_steps=20, resume=False, run_dir=None)
    def checkpoint(self, folder):
        p=folder/'checkpoint-20';p.mkdir()
        for name in ('trainer_state.json','optimizer.pt','scheduler.pt','adapter_model.safetensors'):(p/name).write_text('{}')
    def test_resume_reuses_output_and_pins_inputs(self):
        folder,spec=resolve_training_run(self.args,self.p);self.checkpoint(folder)
        (self.p.workspace/'profiles.json').write_text('{"profiles":[]}')
        self.args.resume=True;self.args.max_steps=0
        resumed,pinned=resolve_training_run(self.args,self.p)
        self.assertEqual(resumed,folder);self.assertEqual(pinned,spec)
        self.assertIn('example',Path(pinned['config']).read_text())
    def test_no_checkpoint_does_not_restart(self):
        resolve_training_run(self.args,self.p);self.args.resume=True
        with self.assertRaisesRegex(ValueError,'0 resumable'):resolve_training_run(self.args,self.p)
    def test_ambiguous_resume_requires_selection(self):
        a,_=resolve_training_run(self.args,self.p);self.checkpoint(a)
        b,_=resolve_training_run(self.args,self.p);self.checkpoint(b)
        self.args.resume=True
        with self.assertRaisesRegex(ValueError,'2 resumable'):resolve_training_run(self.args,self.p)
        self.args.run_dir=str(a)
        self.assertTrue(resolve_training_run(self.args,self.p)[0].samefile(a))
    def test_changed_dataset_is_rejected(self):
        folder,_=resolve_training_run(self.args,self.p);self.checkpoint(folder)
        self.args.resume=True;self.args.dataset=str(self.p.datasets/'other')
        with self.assertRaisesRegex(ValueError,'dataset differs'):resolve_training_run(self.args,self.p)
    def test_finished_runs_are_not_resumed(self):
        folder,_=resolve_training_run(self.args,self.p);self.checkpoint(folder)
        (folder/'training_report.json').write_text('{}');self.args.resume=True
        with self.assertRaisesRegex(ValueError,'0 resumable'):resolve_training_run(self.args,self.p)
    def test_dataset_content_mutation_is_rejected(self):
        file=self.p.datasets/'r1'/'train.jsonl';file.write_text('first')
        folder,_=resolve_training_run(self.args,self.p);self.checkpoint(folder)
        file.write_text('other');self.args.resume=True
        with self.assertRaisesRegex(ValueError,'Dataset content changed'):resolve_training_run(self.args,self.p)
    def test_pinned_config_mutation_is_rejected(self):
        folder,spec=resolve_training_run(self.args,self.p);self.checkpoint(folder)
        Path(spec['config']).write_text('changed');self.args.resume=True
        with self.assertRaisesRegex(ValueError,'Pinned config content changed'):resolve_training_run(self.args,self.p)
    def test_profile_path_is_rejected(self):
        self.args.profile='../escape'
        with self.assertRaises(ValueError):resolve_training_run(self.args,self.p)
    def test_partial_checkpoint_is_ignored(self):
        folder,_=resolve_training_run(self.args,self.p)
        (folder/'checkpoint-20').mkdir()
        (folder/'checkpoint-20'/'trainer_state.json').write_text('{}')
        self.assertFalse(has_checkpoint(folder))

class MemoryBudgetTests(unittest.TestCase):
    def test_checkpoint_yields_below_ten_percent(self):
        vm=SimpleNamespace(total=32*2**30,available=3*2**30)
        with patch.object(rp,'settings',return_value=rp.DEFAULTS.copy()), patch.object(rp.psutil,'virtual_memory',return_value=vm):
            with self.assertRaises(MemoryError):rp.checkpoint()
    def test_checkpoint_allows_sufficient_headroom(self):
        vm=SimpleNamespace(total=32*2**30,available=4*2**30)
        with patch.object(rp,'settings',return_value=rp.DEFAULTS.copy()), patch.object(rp.psutil,'virtual_memory',return_value=vm):
            rp.checkpoint()
    def test_load_reserves_ten_percent(self):
        vm=SimpleNamespace(total=32*2**30,available=11*2**30)
        model=dict(backend='persona_peft',base_bytes=8*2**30,adapter_bytes=0)
        with patch.object(rp,'settings',return_value=rp.DEFAULTS.copy()), patch.object(rp.psutil,'virtual_memory',return_value=vm):
            plan=rp.load_budget(model,mode='CPU')
            self.assertAlmostEqual(plan['reserve_gib'],3.2)
            self.assertFalse(plan['fits'])

class ProfileOrderingTests(unittest.TestCase):
    def test_versioned_snapshot_wins_equal_timestamp_tie(self):
        tmp=tempfile.TemporaryDirectory(prefix="studio-profile-order-test-")
        self.addCleanup(tmp.cleanup)
        root=Path(tmp.name)
        engine=SimpleNamespace(model={"digest":"fixture-order","name":"fixture"},abi="fixture")
        bank={n:{"meta":{"arrays_sha256":n}} for n in ("pain_s2","hell_somatic_pain","hell_burning_pain")}
        with patch.object(sc,"ROOT",root):
            legacy=sc._path(engine);legacy.parent.mkdir(parents=True,exist_ok=True)
            base=dict(profile_kind=sc.PROFILE_KIND,digest="fixture-order",abi="fixture",arrays_sha256={n:n for n in bank},recommended_scale=.3)
            legacy.write_text(json.dumps(base),encoding="utf-8")
            written=sc._atomic(legacy,dict(base,recommended_scale=.6))
            stamp=legacy.stat().st_mtime_ns
            import os
            os.utime(written,ns=(stamp,stamp))
            loaded=sc.load(engine,bank)
            self.assertEqual(loaded["recommended_scale"],.6)
            self.assertEqual(Path(loaded["_profile_file"]),written)

class ProfileStatusTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='studio-verdict-test-');self.addCleanup(self.tmp.cleanup)
        self.patcher=patch.object(sc,'ROOT',Path(self.tmp.name));self.patcher.start();self.addCleanup(self.patcher.stop)
        self.engine=SimpleNamespace(model={'digest':'fixture','name':'fixture'},abi='fixture')
        self.bank={n:{'meta':{'arrays_sha256':n}} for n in ('pain_s2','hell_somatic_pain','hell_burning_pain')}
        self.doc=dict(profile_kind=sc.PROFILE_KIND,digest='fixture',abi='fixture',arrays_sha256={n:n for n in self.bank},behaviorally_verified=True,behavioral_successes=2,behavioral_validation=[{},{},{}],behavioral_status='Legacy behavioral verdict ignored; re-run localized burn validation.')
    def load_doc(self):
        p=sc._path(self.engine);p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(json.dumps(self.doc),encoding='utf-8')
        return sc.load(self.engine,self.bank)
    def test_current_verdict_does_not_keep_legacy_warning(self):
        self.doc['behavioral_rule_version']='localized_burn_v4'
        result=self.load_doc()
        self.assertTrue(result['behaviorally_verified'])
        self.assertNotIn('Legacy',result['behavioral_status'])
        self.assertIn('2/3',result['behavioral_status'])
    def test_old_rule_does_not_gain_verification(self):
        self.doc['behavioral_rule_version']='localized_burn_v1'
        result=self.load_doc()
        self.assertFalse(result['behaviorally_verified'])
        self.assertIn('Legacy',result['behavioral_status'])
    def test_current_failed_verdict_is_not_promoted(self):
        self.doc.update(behavioral_rule_version='localized_burn_v4',behaviorally_verified=False,behavioral_successes=1)
        result=self.load_doc()
        self.assertFalse(result['behaviorally_verified'])
        self.assertIn('Did not pass',result['behavioral_status'])

if __name__ == '__main__':
    unittest.main(verbosity=2)
