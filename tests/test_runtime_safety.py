"""Worker cleanup and memory checks at generation boundaries."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock,patch
import psutil
from isolated_engine import Engine
from workshop_v2 import core,live_engine
from test_inference_transport import FixtureEngine

class WorkerShutdownTests(unittest.TestCase):
    def test_cleanup_denial_keeps_primary_error(self):
        e=Engine();e.process=Mock(pid=999999,returncode=17)
        e.process.poll.return_value=None;e.process.wait.return_value=17
        e.connection=Mock();e.connection.poll.return_value=True
        e.connection.recv.side_effect=EOFError
        child=Mock();child.is_running.return_value=False
        parent=SimpleNamespace(children=lambda recursive:[child])
        with patch('isolated_engine.psutil.Process',return_value=parent),patch('isolated_engine.psutil.wait_procs',side_effect=psutil.AccessDenied(999999)):
            with self.assertRaisesRegex(RuntimeError,r'worker exited \(17\)'):e._call('evaluate',[1])
        self.assertIsNone(e.process)
        self.assertIn('AccessDenied',e.cleanup_error)
    def test_exited_parent_pid_is_not_inspected(self):
        e=Engine();e.process=SimpleNamespace(pid=999999,poll=lambda:17)
        with patch('isolated_engine.psutil.Process',side_effect=AssertionError('PID reused')):e.close()
        self.assertIsNone(e.process)

class MemoryBoundaryTests(unittest.TestCase):
    def controller(self):
        return SimpleNamespace(before_decode=lambda step,phase:{'revision':0,'controls':[],'phase':'all','dose_limit':.5},cancelled=lambda:False)
    def test_live_checks_each_token(self):
        e=FixtureEngine()
        with tempfile.TemporaryDirectory() as tmp,patch.object(core,'ROOT',Path(tmp)),patch.object(live_engine,'checkpoint') as guard:
            result=list(live_engine.stream(e,'test',{},self.controller(),max_tokens=4,temperature=0.))[-1]['result']
        self.assertEqual(len(result['token_ids']),4)
        self.assertEqual(guard.call_count,5)
    def test_live_memory_error_resets_state(self):
        e=FixtureEngine();e.evaluate=Mock(wraps=e.evaluate)
        with patch.object(live_engine,'checkpoint',side_effect=[None,None,MemoryError('RAM floor')]):
            with self.assertRaisesRegex(MemoryError,'RAM floor'):
                list(live_engine.stream(e,'test',{},self.controller(),max_tokens=4,temperature=0.))
        self.assertEqual(e.evaluate.call_count,2)
        self.assertEqual(e.resets,2);self.assertEqual(e.clears,3)
    def test_comparison_memory_error_resets_state(self):
        e=FixtureEngine();e.evaluate=Mock(wraps=e.evaluate)
        with patch.object(core,'checkpoint',side_effect=[None,None,MemoryError('RAM floor')]):
            with self.assertRaisesRegex(MemoryError,'RAM floor'):
                list(core.iter_generate(e,'test',{},max_tokens=4,temperature=0.))
        self.assertEqual(e.evaluate.call_count,2)
        self.assertEqual(e.resets,2);self.assertEqual(e.clears,3)

if __name__=='__main__':unittest.main()
