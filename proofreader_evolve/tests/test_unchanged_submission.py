"""Unchanged submissions (2026-10-12): a generation whose submitted scorer is byte-identical to the
accepted combined scorer skips frozen inference and the outer evaluation, is recorded as
``unchanged`` (ledger flag, trajectory event, report status, handoff host status) and retains the
parent. A changed submission is still evaluated."""

import asyncio
from contextlib import redirect_stdout
import io
import json
import tempfile
import unittest
from unittest.mock import patch

from proofreader_evolve.harness.evolution_report import generation_status
from proofreader_evolve.tests.test_fixed_pool_scoring import fixture, IMPROVED
from proofreader_evolve.tests.label_fixture import setUpModule, tearDownModule  # noqa: F401

HANDOFF = {'evidence': [{'claim': 'the reference was kept', 'verdict': 'mixed'}],
           'open_questions': ['is there a new information source?'], 'next_experiment': 'try one'}


def run_driver(revise):
    """One split generation on the tiny fixture; counts outer evaluations issued after the agent returned."""
    from proofreader_evolve.cli import run_precision_evolution as driver
    with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
        args = driver.parse_args(['--min-positives', '0', '--selection-protocol', 'in_sample', '--promotion-gate', 'margin',
                                  '--kind-schedule', 'alternate', '--train-brains', '1', '--validation-brains', '2',
                                  '--split-k', '1', '--generations', '1', '--runs-dir', tmp, '--target-kind', 'split'])
        state = {'agent_done': False, 'outer_evaluations': 0}
        real_evaluate = driver.scoring.evaluate

        def counting_evaluate(source, tables, *a, **k):
            if state['agent_done'] and sorted(tables) == ['2']:
                state['outer_evaluations'] += 1
            return real_evaluate(source, tables, *a, **k)

        async def wrapped(run_dir, policy, rules, report, model, *, experiments):
            try:
                return await revise(run_dir, policy, rules, report, model, experiments=experiments)
            finally:
                state['agent_done'] = True

        with patch.object(driver.pc, 'resolve_detector_runs', return_value={}), \
                patch.object(driver, 'ensure_native_tables', side_effect=lambda brain, *a, **k: fixture(brain)), \
                patch.object(driver.scoring, 'evaluate', side_effect=counting_evaluate):
            path = asyncio.run(driver.run(args, revise_fn=wrapped))
        records = [json.loads(line) for line in (path / 'ledger.jsonl').read_text().splitlines()]
        trajectory = (path / 'gen001' / 'trajectory.txt').read_text()
        memory = json.loads((path / 'hypothesis_memory.json').read_text())
    return records, trajectory, memory, state['outer_evaluations']


class UnchangedSubmissionTests(unittest.TestCase):
    def test_restoring_the_parent_skips_the_outer_evaluation_and_is_recorded_as_unchanged(self):
        async def revise(run_dir, policy, rules, report, model, *, experiments):
            experiments.restore('parent')
            experiments.proposal_path.write_text(json.dumps({'hypothesis': 'keep', 'strategy': 'reference',
                                                             'family': 'reference', 'parameter_grid': {},
                                                             'handoff': HANDOFF}))
            return {'summary': 'kept the reference unchanged', 'cost_usd': 0}
        records, trajectory, memory, outer = run_driver(revise)
        record = records[0]
        self.assertEqual(outer, 0)  # no frozen inference or Precision@K evaluation on the other brain
        self.assertFalse(record['accepted'])
        self.assertTrue(record['submission_unchanged'])
        self.assertIsNone(record['validation'])
        self.assertIsNone(record.get('failure_stage'))
        self.assertEqual(record['descriptor_inference'], [])
        self.assertTrue(record['reason'].startswith('VALIDATION: Unchanged submission'))
        self.assertIn(record['submitted_experiment'], record['reason'])
        gate = record['validation_gate']
        self.assertEqual(gate['skipped'], 'unchanged_submission')
        self.assertFalse(gate['passed'])
        self.assertEqual(gate['candidate_precision'], gate['parent_precision'])
        self.assertIn(record['submitted_experiment'], trajectory)
        self.assertEqual(generation_status(record, set()), 'unchanged')
        self.assertIn('unchanged_submission', trajectory)
        self.assertIn('Generation 1: unchanged;', trajectory)
        statuses = [item['host']['status'] for rec in memory['records'] for item in rec.get('handoffs', [])]
        self.assertEqual(statuses, ['unchanged_submission'])

    def test_a_changed_submission_is_still_evaluated(self):
        async def revise(run_dir, policy, rules, report, model, *, experiments):
            policy.write_text(IMPROVED)
            experiments.proposal_path.write_text(json.dumps({'hypothesis': 'evidence first', 'strategy': 'use evidence',
                                                             'family': 'evidence', 'parameter_grid': {}}))
            experiments.evaluate()
            return {'summary': 'improved', 'cost_usd': 0}
        records, trajectory, _, outer = run_driver(revise)
        record = records[0]
        self.assertGreaterEqual(outer, 1)
        self.assertTrue(record['accepted'])
        self.assertNotIn('submission_unchanged', record)
        self.assertIsNotNone(record['validation'])
        self.assertEqual(generation_status(record, set()), 'accepted')
        self.assertNotIn('unchanged_submission', trajectory)

    def test_report_status_vocabulary(self):
        self.assertEqual(generation_status({'accepted': False, 'submission_unchanged': True, 'validation': None}, set()), 'unchanged')
        self.assertEqual(generation_status({'accepted': False, 'validation': {'per_kind_precision': {'split': .1}}}, set()), 'rejected')
        self.assertEqual(generation_status({'accepted': True, 'submission_unchanged': False}, set()), 'accepted')


if __name__ == '__main__':
    unittest.main()
