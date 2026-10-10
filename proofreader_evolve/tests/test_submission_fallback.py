"""Submission fallback (2026-10-08): an untested final edit, or a session cut off at the turn
cap, no longer discards a generation's measured work; the best measured candidate is submitted."""

import asyncio
from contextlib import redirect_stdout
import io
import json
import tempfile
import unittest
from unittest.mock import patch

from proofreader_evolve.tests.test_fixed_pool_scoring import fixture, IMPROVED
from proofreader_evolve.tests.label_fixture import setUpModule, tearDownModule  # noqa: F401


def run_driver(revise):
    from proofreader_evolve.cli import run_precision_evolution as driver
    with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
        args = driver.parse_args(['--selection-protocol', 'in_sample', '--promotion-gate', 'margin',
                                  '--kind-schedule', 'alternate', '--train-brains', '1', '--validation-brains', '2',
                                  '--split-k', '1', '--generations', '1', '--runs-dir', tmp, '--target-kind', 'split'])
        with patch.object(driver.pc, 'resolve_detector_runs', return_value={}), \
                patch.object(driver, 'ensure_native_tables', side_effect=lambda brain, *a, **k: fixture(brain)):
            path = asyncio.run(driver.run(args, revise_fn=revise))
        records = [json.loads(line) for line in (path / 'ledger.jsonl').read_text().splitlines()]
        best = (path / 'best_scorer.py').read_text()
        trajectory = (path / 'gen001' / 'trajectory.txt').read_text()
    return records, best, trajectory


class SubmissionFallbackTests(unittest.TestCase):
    def test_untested_edit_after_the_turn_cap_submits_the_best_measured_candidate(self):
        async def revise(run_dir, policy, rules, report, model, *, experiments):
            policy.write_text(IMPROVED)
            experiments.proposal_path.write_text(json.dumps({'hypothesis': 'evidence first', 'strategy': 'use evidence',
                                                             'family': 'evidence', 'parameter_grid': {}}))
            experiments.evaluate()
            policy.write_text(IMPROVED.replace('return ', 'return 3.0 * '))  # untested edit left behind
            return {'summary': 'cut off', 'cost_usd': 0, 'subtype': 'error_max_turns', 'is_error': True, 'num_turns': 61}
        records, best, trajectory = run_driver(revise)
        record = records[0]
        self.assertTrue(record['accepted'])
        self.assertIsNone(record.get('failure_stage'))
        self.assertEqual(record['submitted_experiment'], 'gen001/attempt001')
        self.assertEqual(record['submission_fallback']['restored'], 'gen001/attempt001')
        self.assertIn('exactly match', record['submission_fallback']['reason'])
        self.assertIn("features['evidence'].fillna(0)", best)  # the measured candidate, inside the combined scorer
        self.assertNotIn('3.0 *', best)  # the untested edit never shipped
        self.assertEqual(record['reviser']['subtype'], 'error_max_turns')
        self.assertIn('submission_fallback', trajectory)

    def test_nothing_measured_still_fails_the_generation(self):
        async def revise(run_dir, policy, rules, report, model, *, experiments):
            policy.write_text(IMPROVED)  # never evaluated
            return {'summary': 'no measurement', 'cost_usd': 0}
        records, _, _ = run_driver(revise)
        self.assertFalse(records[0]['accepted'])
        self.assertEqual(records[0]['failure_stage'], 'submission_check')
        self.assertNotIn('submission_fallback', records[0])


if __name__ == '__main__':
    unittest.main()
