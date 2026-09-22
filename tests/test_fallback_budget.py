import unittest
from unittest.mock import patch
from types import SimpleNamespace
from isolated_engine import Engine
from workshop_v2 import resource_policy as rp

class FallbackBudgetTests(unittest.TestCase):
    def test_cpu_fallback_rechecks_memory_and_updates_estimates(self):
        model={'name':'fixture','digest':'fixture','architecture':'llama','path':'unused.gguf'}
        def plan(model,context,mode,manual=0):
            count=17 if mode=='Auto' else 0 if mode=='CPU' else manual
            return dict(mode=mode,gpu_layers=count,estimated_cpu_weights_gib=2 if count==0 else .5,estimated_gpu_weights_gib=0 if count==0 else 1.5)
        def open_call(method,model,**kwargs):
            if kwargs['gpu_layers']:raise RuntimeError('CUDA runtime unavailable')
        e=Engine();e.model=model
        with patch('isolated_engine.plan_load',side_effect=plan), patch.object(rp,'guard_load',return_value={'cooperative':True}) as guard, patch.object(rp,'ModelLaunchLease',return_value=SimpleNamespace(close=lambda:None)), patch.object(e,'_start'), patch.object(e,'_call',side_effect=open_call), patch.object(e,'_record'):
            e.open(model,context=512,mode='Auto')
        self.assertEqual(e.load_plan['gpu_layers'],0)
        self.assertEqual(e.load_plan['estimated_gpu_weights_gib'],0)
        self.assertEqual(e.load_plan['estimated_cpu_weights_gib'],2)
        self.assertEqual(e.load_plan['requested_mode'],'Auto')
        self.assertEqual([c.kwargs['mode'] for c in guard.call_args_list],['Auto','Manual','Manual','CPU'])
        self.assertEqual(len(e.load_plan['fallback_errors']),2)

if __name__=='__main__':unittest.main()
