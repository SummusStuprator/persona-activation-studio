"""Portable workflow checks; no model downloads or user-data writes."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
import psutil
import studio_jobs as jobs
import studio_cli as cli
from workshop_v2.resource_policy import ModelLease

class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='studio-workflow-')
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
    def test_job_records_nonzero_exit(self):
        path=self.root/'job.json'
        jobs.write_state(path,dict(command=[sys.executable,'-c','raise SystemExit(7)'],cwd=str(self.root)))
        self.assertEqual(jobs.run(path),7)
        self.assertEqual(jobs.read_state(path)['status'],'failed')
    def test_job_records_success(self):
        path=self.root/'job.json'
        jobs.write_state(path,dict(command=[sys.executable,'-c','pass'],cwd=str(self.root)))
        self.assertEqual(jobs.run(path),0)
        self.assertEqual(jobs.read_state(path)['status'],'succeeded')
    def test_reused_pid_is_not_running(self):
        path=self.root/'job.json'
        jobs.write_state(path,dict(status='running',pid=os.getpid(),process_started=0))
        self.assertEqual(jobs.read_state(path)['status'],'interrupted')
    def test_duplicate_job_does_not_launch(self):
        args=['profile','list']
        command=[sys.executable,'-m','studio_cli',*args]
        jobs.write_state(self.root/'job.json',dict(status='running',pid=os.getpid(),
            process_started=psutil.Process().create_time(),command=command,log=str(self.root/'job.log')))
        with patch.object(jobs.subprocess,'Popen') as start:
            jobs.launch(Path.cwd(),self.root,'profile',args)
            start.assert_not_called()
    def test_training_uses_configured_python(self):
        with patch.object(cli,'load_config',return_value={'python':'training-python'}),patch.object(cli.subprocess,'call',return_value=0) as call:
            cli.run_module('persona.trainer_v4',['plan'])
            self.assertEqual(call.call_args.args[0][0],'training-python')
    def test_scrape_uses_core_python(self):
        with patch.object(cli.subprocess,'call',return_value=0) as call:
            cli.run_module('persona.x_scraper',['accounts'])
            self.assertEqual(call.call_args.args[0][0],sys.executable)
    def test_log_tail_is_bounded(self):
        path=self.root/'job.log';path.write_bytes(b'x'*40000)
        self.assertEqual(len(jobs.log_tail(path,100)),100)
    def test_model_lease_blocks_other_process_and_releases(self):
        code='from workshop_v2.resource_policy import ModelLease; lease=ModelLease(); lease.close()'
        env=dict(os.environ,STUDIO_LOCK_DIR=str(self.root/'locks'))
        with patch.dict(os.environ,{'STUDIO_LOCK_DIR':str(self.root/'locks')}):
            with ModelLease():
                blocked=subprocess.run([sys.executable,'-c',code],env=env,capture_output=True,timeout=15)
                self.assertNotEqual(blocked.returncode,0)
            released=subprocess.run([sys.executable,'-c',code],env=env,capture_output=True,timeout=15)
            self.assertEqual(released.returncode,0,released.stderr)

if __name__=='__main__':
    unittest.main()
