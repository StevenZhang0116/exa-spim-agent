"""Admission correctness without allocating large arrays or triggering a real OOM."""
import os
from contextlib import redirect_stderr
import io
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np

from proofreader_evolve.harness import memory_budget as mb
from proofreader_evolve.harness import descriptor_runs as dr
from proofreader_evolve.harness.descriptor_context import DescriptorContexts
from proofreader_evolve.harness.descriptor_contract import descriptor_spec
from proofreader_evolve.harness.model_execution import ModelExecutionError, exit_detail
from proofreader_evolve.tests.test_descriptor_compute import GEOMETRY_DESCRIPTOR, FakeProvider
from proofreader_evolve.tests.test_selection_protocol import brain_table
from proofreader_evolve.tests import label_fixture

setUpModule = label_fixture.setUpModule
tearDownModule = label_fixture.tearDownModule


class MemoryBudgetTests(unittest.TestCase):
    def test_ancestor_headroom_and_no_assumed_page_cache_reclaim(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task = root / 'job' / 'step'
            task.mkdir(parents=True)
            for path, limit, used in ((root, 400, 390), (root / 'job', 100, 40), (task, 80, 10)):
                (path / 'memory.max').write_text(str(limit * mb.GIB))
                (path / 'memory.current').write_text(str(used * mb.GIB))
                (path / 'memory.stat').write_text(f'inactive_file {50 * mb.GIB}')
            (root / 'self').write_text('0::/job/step\n')
            actual = mb.memory_snapshot(root / 'self', root)
            self.assertEqual(actual['headroom_bytes'], 10 * mb.GIB)
            self.assertEqual(actual['limit_bytes'], 80 * mb.GIB)
            self.assertEqual(actual['reserve_bytes'], 8 * mb.GIB)

    def test_slurm_unknown_limit_fails_instead_of_using_entire_node(self):
        with patch.dict(os.environ, {'SLURM_JOB_ID': '42'}):
            with self.assertRaisesRegex(ModelExecutionError, 'finite Slurm'):
                mb.memory_snapshot('/absent/cgroup')

    def test_pending_reservations_prevent_launch_race(self):
        snapshot = {'source': 'test', 'limit_bytes': 100 * mb.GIB,
                    'headroom_bytes': 43 * mb.GIB, 'reserve_bytes': 10 * mb.GIB}
        with patch.object(mb, 'memory_snapshot', return_value=snapshot):
            self.assertEqual(mb.admission(16384, 40)['available_workers'], 2)
            self.assertEqual(mb.admission(16384, 40, pending=2)['available_workers'], 0)

    def test_explicit_worker_count_is_also_capped_by_cpu_affinity(self):
        with patch.object(dr, 'available_workers', return_value=3):
            runs = dr.DescriptorRuns(DescriptorContexts(), '/unused', workers=40)
        self.assertEqual(runs.requested_workers, 40)
        self.assertEqual(runs.workers, 3)

    def test_cli_separates_limits_and_rejects_invalid_descriptor_cap(self):
        from proofreader_evolve.cli.run_precision_evolution import parse_args
        args = parse_args(['--classifier-memory-mb', '98304'])
        self.assertEqual(args.descriptor_memory_mb, 16384)
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args(['--descriptor-memory-mb', '511'])
        self.assertIn('cause unconfirmed', exit_detail(-9, {'memory_mb': 1}, ''))


class DescriptorAdmissionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.table = brain_table('802449').tables['merge']
        self.table.local_context = FakeProvider()
        with patch.object(dr, 'available_workers', return_value=4):
            self.runs = dr.DescriptorRuns(DescriptorContexts(), self.root / 'bank', workers=4, memory_mb=1024)
        self.program, self.spec = descriptor_spec(GEOMETRY_DESCRIPTOR)
        self.snapshot = {'source': 'test', 'limit_bytes': 100 * mb.GIB,
                         'headroom_bytes': 13 * mb.GIB, 'reserve_bytes': 10 * mb.GIB}

    def compute(self, budget=5):
        return self.runs.compute(self.table, self.program, self.spec, 'all_candidates',
                                 k=3, wall_budget=budget, log_dir=self.root / 'logs', force=True)

    def test_serial_first_batch_bounded_concurrency_and_unchanged_values(self):
        active, observed = 0, []
        lock = threading.Lock()
        def batch(root, table, entry, program, spec, rows, deadline, log, **kwargs):
            nonlocal active
            with lock:
                active += 1
                observed.append((int(rows[0]), active))
            time.sleep(.03)
            with lock:
                active -= 1
            return np.column_stack([rows, rows]).astype(float), .03
        with patch.object(mb, 'memory_snapshot', return_value=self.snapshot), patch.object(dr, 'BATCH_ROWS', 4), \
                patch.object(self.runs, '_run_batch', side_effect=batch):
            result = self.compute()
        self.assertEqual(observed[0], (0, 1))
        self.assertEqual(result['workers'], 2)
        self.assertLessEqual(max(n for _, n in observed), 2)
        np.testing.assert_array_equal(result['values'][:, 0], np.arange(16))
        self.assertEqual(result['manifest']['worker_limit'], 4)

    def test_no_capacity_fails_before_input_preparation(self):
        self.snapshot['headroom_bytes'] = 10 * mb.GIB
        with patch.object(mb, 'memory_snapshot', return_value=self.snapshot), \
                patch.object(self.runs, '_run_batch', side_effect=AssertionError('prepared without admission')):
            with self.assertRaisesRegex(ModelExecutionError, 'Insufficient descriptor memory'):
                self.compute()
            with self.assertRaisesRegex(ModelExecutionError, 'Insufficient descriptor memory'):
                self.runs.probe(self.table, self.program, self.spec, [0], log_dir=self.root / 'probe')
        self.assertFalse(self.runs.directory(self.table, self.spec, 'all_candidates').exists())

    def test_memory_wait_uses_call_deadline_and_writes_no_bank(self):
        def batch(root, table, entry, program, spec, rows, deadline, log, **kwargs):
            if rows[0] != 0:
                time.sleep(.08)
            return np.column_stack([rows, rows]).astype(float), 0.
        # Only one batch fits; a pending batch prevents admission of another.
        self.snapshot['headroom_bytes'] = int(11.5 * mb.GIB)
        with patch.object(mb, 'memory_snapshot', return_value=self.snapshot), patch.object(dr, 'BATCH_ROWS', 4), \
                patch.object(self.runs, '_run_batch', side_effect=batch):
            with self.assertRaisesRegex(ModelExecutionError, 'wall budget'):
                self.compute(.04)
        self.assertFalse(self.runs.directory(self.table, self.spec, 'all_candidates').exists())

    def test_plan_uses_memory_capacity_after_probe(self):
        with patch.object(mb, 'memory_snapshot', return_value=self.snapshot), \
                patch.object(self.runs, 'probe', return_value={
                    'rows': np.arange(4), 'values': np.zeros((4, 2)), 'wall_seconds': 4.,  # rows: plan memo since 2026-11-04
                    'preparation_seconds': 1., 'worker_seconds': 3.}):
            plan = self.runs.plan(self.table, self.program, self.spec, k=3, sample_rows=4)
        self.assertEqual(plan['worker_limit'], 4)
        self.assertEqual(plan['workers'], 2)
        self.assertEqual(plan['estimates']['all_candidates']['estimated_seconds'], 16.)
