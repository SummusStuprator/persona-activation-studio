"""Transport caching and live-control regression tests; no model weights."""
import json, tempfile, threading, unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
import numpy as np
from isolated_engine import Engine
from workshop_v2 import core
from workshop_v2.live_engine import stream

class MetadataCacheTests(unittest.TestCase):
    def engine(self):
        engine=Engine();engine._loaded=True
        engine.process=SimpleNamespace(poll=lambda:None)
        def close():
            engine.process=None;engine.close()
        self.addCleanup(close)
        return engine
    def test_repeated_token_metadata_uses_one_request_each(self):
        e=self.engine();e._call=Mock(side_effect=lambda op,t: b'token' if op=='piece' else False)
        for _ in range(3):
            self.assertEqual(e.piece(7),b'token');self.assertFalse(e.is_eog(7))
        self.assertEqual(e._call.call_count,2)
    def test_failed_request_is_not_cached(self):
        e=self.engine();e._call=Mock(side_effect=[RuntimeError('worker error'),b'ok'])
        with self.assertRaises(RuntimeError):e.piece(1)
        self.assertEqual(e.piece(1),b'ok');self.assertEqual(e.piece(1),b'ok')
        self.assertEqual(e._call.call_count,2)
    def test_close_discards_metadata_before_next_model(self):
        e=self.engine();e._call=Mock(return_value=b'old');e.piece(1)
        e.process=None;e.close()
        self.assertEqual(e._piece_cached.cache_info().currsize,0)
        with self.assertRaises(RuntimeError):e.piece(1)
        e.process=SimpleNamespace(poll=lambda:None);e._loaded=True
        e._call=Mock(return_value=b'new');self.assertEqual(e.piece(1),b'new')
    def test_dead_worker_cannot_return_cached_values(self):
        e=self.engine();e._call=Mock(return_value=b'ok');e.piece(1)
        e.process.poll=lambda:1
        with self.assertRaises(RuntimeError):e.piece(1)
    def test_cache_is_bounded(self):
        e=self.engine();e._call=Mock(return_value=b'x')
        for token in range(4100):e.piece(token)
        self.assertEqual(e._piece_cached.cache_info().currsize,4096)

class FixtureEngine:
    def __init__(self,dtype='bfloat16'):
        self.layers=2;self.dim=3;self.context=128;self.abi='fixture'
        self.model={'digest':'fixture','name':'fixture','activation_dtype':dtype,'backend':'persona_peft'}
        self.handle=True;self.lock=threading.RLock();self.clears=0;self.resets=0
    def clear_steering(self):self.clears+=1
    def reset(self):self.resets+=1
    def tokenize(self,text):return np.array([1,2],np.int32)
    def evaluate(self,tokens):pass
    def capture(self,which):return np.ones((2,3),np.float32)
    def logits(self):return np.array([0.,1.,2.,3.],np.float32)
    def is_eog(self,token):return False
    def piece(self,token):return b't'
class LiveControlTests(unittest.TestCase):
    def run_fixture(self,dtype='bfloat16',revise=False,cancel=False):
        e=FixtureEngine(dtype)
        settings=lambda step,phase: None if cancel and step==1 else {'revision':int(revise and step>=2),'controls':[],'phase':'all','dose_limit':.5}
        controller=SimpleNamespace(before_decode=settings,cancelled=lambda:False)
        with tempfile.TemporaryDirectory(prefix='studio-transport-test-') as tmp,patch.object(core,'ROOT',Path(tmp)):
            events=list(stream(e,'test',{},controller,max_tokens=4,temperature=0.))
            result=events[-1]['result']
        return e,result
    def test_unchanged_controls_are_sent_once(self):
        e,r=self.run_fixture()
        self.assertEqual(e.clears,3)
        self.assertEqual(e.resets,2)
        self.assertEqual(len(r['control_events']),1)
        self.assertEqual(len(r['token_ids']),4)
    def test_control_revision_is_applied(self):
        e,r=self.run_fixture(revise=True)
        self.assertEqual(e.clears,4)
        self.assertEqual(len(r['control_events']),2)
        self.assertEqual(r['control_events'][1]['effective_before_token'],2)
    def test_cancel_resets_and_clears_model(self):
        e,r=self.run_fixture(cancel=True)
        self.assertEqual(r['stop_reason'],'cancelled');self.assertEqual(e.resets,2)
        self.assertEqual(e.clears,3)
    def test_fp16_uses_storage_aware_precision_check(self):
        _,r=self.run_fixture(dtype='float16')
        self.assertEqual(r['trace'][0]['precision_check']['storage'],'float16')

if __name__=='__main__':unittest.main()
