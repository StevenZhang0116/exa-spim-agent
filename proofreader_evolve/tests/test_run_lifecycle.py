import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from proofreader_evolve.harness import run_lifecycle as lifecycle
from proofreader_evolve.cli import launch_evolution


class RunLifecycleTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        lifecycle.atomic_json(self.root / 'run_status.json', {'status': 'running', 'generation': 5})
        lifecycle.atomic_json(self.root / 'execution.json', {'slurm_job_id': '123', 'slurm_step_id': '4'})
        reporter = patch('proofreader_evolve.harness.evolution_report.write_report')
        reporter.start()
        self.addCleanup(reporter.stop)

    def test_slurm_exact_step_and_oom(self):
        answer = SimpleNamespace(stdout='123.3|COMPLETED|0:0\n123.4|OUT_OF_MEMORY|0:125\n')
        with patch.object(lifecycle.subprocess, 'run', return_value=answer):
            result = lifecycle.reconcile(self.root)
        self.assertEqual(result['termination']['kind'], 'oom')
        self.assertEqual(result['generation'], 5)
        self.assertEqual(json.loads((self.root / 'run_status.json').read_text())['status'], 'failed')

    def test_active_or_missing_accounting_does_not_guess_from_age(self):
        before = (self.root / 'run_status.json').read_bytes()
        for evidence in ({'state': 'RUNNING'}, {'query_error': 'not ready'}):
            with patch.object(lifecycle, 'slurm_record', return_value=evidence):
                self.assertFalse(lifecycle.reconcile(self.root)['status_changed'])
            self.assertEqual((self.root / 'run_status.json').read_bytes(), before)

    def test_sigkill_is_unconfirmed_until_accounting_proves_oom(self):
        with patch.object(lifecycle, 'slurm_record', return_value={'query_error': 'not ready'}):
            result = lifecycle.reconcile(self.root, srun_exit=137)
        self.assertEqual(result['termination']['kind'], 'unconfirmed_exit')
        with patch.object(lifecycle, 'slurm_record', return_value={'state': 'OUT_OF_MEMORY'}):
            self.assertEqual(lifecycle.reconcile(self.root)['termination']['kind'], 'oom')

    def test_timeout_cancel_and_completed_preservation(self):
        for reason in ('TIMEOUT', 'CANCELLED', 'PREEMPTED'):
            with patch.object(lifecycle, 'slurm_record', return_value={'state': reason}):
                self.assertEqual(lifecycle.reconcile(self.root)['status'], 'interrupted')
        lifecycle.atomic_json(self.root / 'run_status.json', {'status': 'completed'})
        before = (self.root / 'run_status.json').read_bytes()
        with patch.object(lifecycle, 'slurm_record', side_effect=AssertionError('completed run queried')):
            self.assertEqual(lifecycle.reconcile(self.root, srun_exit=137)['status'], 'completed')
        self.assertEqual((self.root / 'run_status.json').read_bytes(), before)

    def test_launcher_reconciles_from_outside_child_and_forwards_args(self):
        with patch.object(launch_evolution.subprocess, 'call', return_value=137) as call, \
                patch.object(launch_evolution, 'reconcile') as reconcile:
            code = launch_evolution.main(['--job-id', '123', '--node', 'n277', '--cpus', '4',
                '--runs-dir', str(self.root / 'runs'), '--python', '/panda/python', '--',
                '--detector-prior', 'raw', '--generations', '10'])
        self.assertEqual(code, 137)
        command = call.call_args.args[0]
        self.assertEqual(command[0], 'srun')
        self.assertIn('--jobid=123', command)
        self.assertEqual(command[-4:], ['--detector-prior', 'raw', '--generations', '10'])
        self.assertEqual(reconcile.call_args.kwargs['srun_exit'], 137)
