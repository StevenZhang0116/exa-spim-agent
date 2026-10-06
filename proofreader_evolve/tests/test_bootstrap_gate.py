"""Bootstrap promotion gate: the mean validation gain must clear the margin AND
the lower bound of the paired row-resampling interval must exceed zero.

Synthetic fixtures only. A single extra Top-K hit (+1/K) on a 4,000-row pool is
the canonical noise-level gain: the legacy margin gate promotes it, the bootstrap
gate rejects it, retains the TRAIN branch and schedules no promotion follow-up.
"""

import asyncio
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from proofreader_evolve.harness import fixed_pool_scoring as scoring
from proofreader_evolve.harness.native_pool import NativeTable, NativeTables, pool_digest

K = 200
DETECTOR = 'def score_candidates(features, ctx):\n    return features["detector_score"].fillna(0).to_numpy(dtype=float)\n'
SIGNAL = 'def score_candidates(features, ctx):\n    return features["signal"].fillna(0).to_numpy(dtype=float)\n'
TWEAK = 'def score_candidates(features, ctx):\n    return features["tweak"].fillna(0).to_numpy(dtype=float)\n'


def pool(brain, n=4000, seed=0):
    """Random detector ranking, a strongly predictive `signal`, and `tweak`: the
    detector ranking with exactly one positive swapped into the Top-K."""
    rng = np.random.default_rng(seed)
    rows = [{'segment_id': i + 1, 'node_id': i} for i in range(n)]
    truth = (rng.random(n) < .2).astype(np.int8)
    detector = rng.random(n)
    signal = truth + rng.normal(0., .3, n)
    order = np.argsort(-detector, kind='stable')
    inside_negative = next(int(i) for i in order[:K][::-1] if truth[i] == 0)
    outside_positive = next(int(i) for i in order[K:] if truth[i] == 1)
    tweak = detector.copy()
    tweak[inside_negative], tweak[outside_positive] = detector[outside_positive], detector[inside_negative]
    frame = pd.DataFrame({'detector_score': detector, 'signal': signal, 'tweak': tweak})
    table = NativeTable('merge', frame, rows, truth,
                        {'pool_sha256': pool_digest('merge', rows), 'training_brain': '794495',
                         'provenance': {'brain': brain}})
    return NativeTables(brain, {'merge': table}, {})


def write_proposal(experiments, hypothesis):
    experiments.proposal_path.write_text(json.dumps({'hypothesis': hypothesis, 'strategy': 'formula',
                                                      'family': 'formula', 'parameter_grid': {}}))


class BootstrapAcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.banks = {'2': pool('2')}
        self.budgets = {'merge': K}
        self.reports, self.states = {}, {}
        for name, source in (('detector', DETECTOR), ('signal', SIGNAL), ('tweak', TWEAK)):
            state = {}
            self.reports[name] = scoring.evaluate(source, self.banks, self.budgets, state=state)
            self.states[name] = state

    def interval(self, before, after, draws=200, alpha=.05):
        return scoring.paired_bootstrap(self.states[before], self.states[after], self.banks, self.budgets,
                                        draws=draws, seed=1, alpha=alpha)

    def test_one_extra_hit_passes_the_margin_gate_but_not_the_bootstrap_gate(self):
        before, after = self.reports['detector'], self.reports['tweak']
        gain = after['macro_precision'] - before['macro_precision']
        self.assertAlmostEqual(gain, 1. / K)
        self.assertTrue(scoring.acceptance(before, after, 0, require_no_cell_regression=False)[0])
        bootstrap = self.interval('detector', 'tweak')
        self.assertLessEqual(bootstrap['macro_delta_ci'][0], 0.)
        self.assertEqual(bootstrap['alpha'], .05)
        passed, reason = scoring.bootstrap_acceptance(before, after, bootstrap, 0, alpha=.05)
        self.assertFalse(passed)
        self.assertIn('within resampling noise', reason)
        self.assertIn(f'delta={gain:+.6f}', reason)
        self.assertIn('200 draws', reason)

    def test_clear_gain_passes_and_point_failures_keep_the_point_reason(self):
        before, after = self.reports['detector'], self.reports['signal']
        bootstrap = self.interval('detector', 'signal')
        self.assertGreater(bootstrap['macro_delta_ci'][0], 0.)
        passed, reason = scoring.bootstrap_acceptance(before, after, bootstrap, 0, alpha=.05)
        self.assertTrue(passed)
        self.assertIn('beyond resampling noise', reason)
        self.assertIn('bootstrap lower bound +', reason)
        passed, reason = scoring.bootstrap_acceptance(after, before, self.interval('signal', 'detector'), 0)
        self.assertFalse(passed)
        self.assertIn('No sufficient mean Precision@K gain', reason)
        self.assertNotIn('bootstrap', reason)

    def test_missing_or_mismatched_interval_fails_closed(self):
        before, after = self.reports['detector'], self.reports['signal']
        for missing in (None, {'status': 'unavailable', 'error': 'boom'}, {}):
            passed, reason = scoring.bootstrap_acceptance(before, after, missing, 0)
            self.assertFalse(passed)
            self.assertIn('unavailable', reason)
        bootstrap = self.interval('detector', 'signal', alpha=.1)
        self.assertEqual(bootstrap['alpha'], .1)
        with self.assertRaises(ValueError):
            scoring.bootstrap_acceptance(before, after, bootstrap, 0, alpha=.05)
        with self.assertRaises(ValueError):
            self.interval('detector', 'signal', alpha=1.5)
        self.assertEqual(scoring.gate_version('bootstrap'), scoring.BOOTSTRAP_GATE_VERSION)
        self.assertEqual(scoring.gate_version('margin'), scoring.VALIDATION_GATE_VERSION)
        with self.assertRaises(ValueError):
            scoring.gate_version('other')

    def test_interval_level_and_legacy_fields(self):
        bootstrap = self.interval('detector', 'signal', alpha=.2)
        self.assertEqual(len(bootstrap['macro_delta_ci95']), 2)
        self.assertGreaterEqual(bootstrap['macro_delta_ci'][0], bootstrap['macro_delta_ci95'][0])
        self.assertLessEqual(bootstrap['macro_delta_ci'][1], bootstrap['macro_delta_ci95'][1])
        self.assertIn('delta_ci', bootstrap['cells']['2/merge'])


class DriverGateTests(unittest.TestCase):
    def run_driver(self, extra, attempts):
        from proofreader_evolve.cli import run_precision_evolution as driver
        banks = {'1': pool('1', n=600, seed=5), '2': pool('2')}
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            seed = Path(tmp) / 'seed.py'
            seed.write_text(DETECTOR)
            args = driver.parse_args(['--selection-protocol', 'in_sample', '--train-brains', '1',
                                      '--validation-brains', '2', '--merge-k', str(K), '--split-k', '1',
                                      '--generations', str(len(attempts)), '--runs-dir', tmp,
                                      '--start-from', str(seed), '--bootstrap-draws', '200', *extra])
            queue = iter(attempts)

            async def revise(run_dir, policy, rules, report, model, *, experiments):
                source = next(queue)
                policy.write_text(source)
                write_proposal(experiments, 'formula ' + str(hash(source)))
                experiments.evaluate()
                return {'summary': 'formula candidate', 'cost_usd': 0}

            with patch.object(driver.pc, 'resolve_detector_runs', return_value={}), \
                    patch.object(driver, 'ensure_native_tables', side_effect=lambda brain, *a, **k: banks[brain]), \
                    patch.object(driver, 'attach_local_context', return_value=None):
                path = asyncio.run(driver.run(args, revise_fn=revise))
            records = [json.loads(line) for line in (path / 'ledger.jsonl').read_text().splitlines()]
            manifest = json.loads((path / 'manifest.json').read_text())
            pool_file = json.loads((path / 'candidate_pool.json').read_text())
            best = (path / 'best_scorer.py').read_text()
        return records, manifest, pool_file, best

    def test_bootstrap_gate_rejects_noise_level_gain_then_accepts_a_clear_one(self):
        records, manifest, pool_file, best = self.run_driver([], [TWEAK, SIGNAL])
        gate = manifest['promotion_gate']
        self.assertEqual(gate['mode'], 'bootstrap')
        self.assertEqual(gate['version'], scoring.BOOTSTRAP_GATE_VERSION)
        self.assertTrue(gate['bootstrap']['affects_decision'])
        self.assertEqual(gate['bootstrap']['draws'], 200)
        first, second = records
        self.assertFalse(first['accepted'])
        self.assertTrue(first['validation_gate']['point_gate']['passed'])
        self.assertFalse(first['validation_gate']['bootstrap_gate']['passed'])
        self.assertTrue(first['validation_gate']['bootstrap_gate']['binding'])
        self.assertIn('within resampling noise', first['reason'])
        self.assertEqual(first['validation_gate']['mode'], 'bootstrap')
        self.assertIsNone(first['search_progress']['pending_followup'])
        self.assertTrue(second['accepted'])
        self.assertIn('beyond resampling noise', second['reason'])
        self.assertGreater(second['validation_gate']['bootstrap_gate']['lower_bound'], 0.)
        self.assertIsNotNone(second['search_progress']['pending_followup'])
        self.assertIn('features["signal"]', best)
        experiments = {entry['experiment'] for entry in pool_file['kinds']['merge']['candidates']}
        self.assertIn('gen001/attempt001', experiments)  # rejected branch stays searchable on TRAIN

    def test_margin_mode_keeps_the_legacy_point_decision(self):
        records, manifest, _, best = self.run_driver(['--promotion-gate', 'margin'], [TWEAK])
        self.assertEqual(manifest['promotion_gate']['mode'], 'margin')
        self.assertFalse(manifest['promotion_gate']['bootstrap']['affects_decision'])
        self.assertTrue(records[0]['accepted'])
        self.assertEqual(records[0]['validation_gate']['version'], scoring.VALIDATION_GATE_VERSION)
        self.assertFalse(records[0]['validation_gate']['bootstrap_gate']['binding'])
        self.assertFalse(records[0]['validation_gate']['bootstrap_gate']['passed'])
        self.assertIn('features["tweak"]', best)

    def test_bootstrap_gate_requires_draws(self):
        from proofreader_evolve.cli import run_precision_evolution as driver
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                driver.parse_args(['--bootstrap-draws', '0'])
            with self.assertRaises(SystemExit):
                driver.parse_args(['--bootstrap-alpha', '1.5'])
        self.assertEqual(driver.parse_args(['--promotion-gate', 'margin', '--kind-schedule', 'alternate', '--bootstrap-draws', '0']).bootstrap_draws, 0)
        self.assertEqual(driver.parse_args([]).promotion_gate, 'bootstrap')
        self.assertEqual(driver.parse_args([]).bootstrap_draws, 1000)


if __name__ == '__main__':
    unittest.main()
