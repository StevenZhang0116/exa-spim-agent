"""--detector-prior no-score (2026-10-12): the frozen detector score is withheld from scorers,
training data, descriptors and the reviser, while the host still measures the frozen baseline.
Synthetic fixtures only; no brain caches, cloud reads or LLM calls."""

import asyncio
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from proofreader_evolve.harness import context_cache as cc
from proofreader_evolve.harness.collect_run import collect
from proofreader_evolve.harness import detector_prior
from proofreader_evolve.harness.image_features import subset_features
from proofreader_evolve.tests.test_descriptor_compute import GEOMETRY_DESCRIPTOR, build_cache
from proofreader_evolve.tests.test_selection_protocol import brain_table, two_brains
from proofreader_evolve.tests.label_fixture import setUpModule, tearDownModule  # noqa: F401

HERE = Path(__file__).resolve().parents[1]
SIGNAL = "def score_candidates(features, ctx):\n    return features['signal'].fillna(0).to_numpy()\n"
# Reads every context it is given: fails if the band flag or the score reaches a descriptor.
BLIND_DESCRIPTOR = GEOMETRY_DESCRIPTOR.replace(
    "    fragment = context['fragment']",
    "    assert 'row_in_band' not in context and 'detector_score' not in context['features']\n"
    "    fragment = context['fragment']")


def driver_banks():
    banks = {**two_brains(), '2': brain_table('2', offset=5000.)}
    banks['2'].tables['merge'].meta['training_brain'] = '794495'
    return banks


def parse(driver, tmp, *extra):
    return driver.parse_args(['--min-positives', '0', '--train-brains', '794495,802449', '--validation-brains', '2',
                              '--merge-k', '3', '--split-k', '1', '--generations', '1', '--runs-dir', tmp,
                              '--skip-auxiliary-ablation', '--bootstrap-draws', '0', '--promotion-gate', 'margin',
                              '--kind-schedule', 'alternate', *extra])


def run_driver(driver, args, banks, revise):
    with patch.object(driver.pc, 'resolve_detector_runs', return_value={}), \
            patch.object(driver, 'ensure_native_tables', side_effect=lambda brain, *a, **k: banks[brain]), \
            patch.object(driver, 'attach_local_context', return_value=None):
        return asyncio.run(driver.run(args, revise_fn=revise))


class ArgumentTests(unittest.TestCase):
    def test_defaults_follow_the_mode_and_the_frozen_bank_is_refused(self):
        from proofreader_evolve.cli import run_precision_evolution as driver
        frozen = driver.parse_args([])
        self.assertEqual(frozen.detector_prior, 'frozen')
        self.assertEqual(frozen.start_from, HERE / 'artifacts' / 'scorer.py')
        self.assertEqual(frozen.descriptor_bank, HERE / 'descriptor_bank')
        self.assertEqual(driver._contract(frozen), driver.CONTRACT)
        ablation = driver.parse_args(['--detector-prior', 'no-score'])
        self.assertEqual(ablation.start_from, HERE / 'artifacts' / 'scorer_no_score.py')
        self.assertEqual(ablation.descriptor_bank, HERE / 'descriptor_bank_no_score')
        self.assertTrue(driver._contract(ablation).endswith(detector_prior.NOTICE))
        with redirect_stdout(io.StringIO()), patch('sys.stderr', io.StringIO()), self.assertRaises(SystemExit):
            driver.parse_args(['--detector-prior', 'no-score', '--descriptor-bank', str(HERE / 'descriptor_bank')])


class HelperTests(unittest.TestCase):
    def test_strip_keeps_a_host_copy_once_per_shared_table(self):
        banks = two_brains()
        banks['alias'] = banks['802449']
        expected = banks['802449'].tables['merge'].features['detector_score'].to_numpy()
        cells = detector_prior.strip(banks)
        self.assertEqual(cells, ['794495/merge', '802449/merge'])
        for bank in banks.values():
            self.assertNotIn('detector_score', bank.tables['merge'].features.columns)
        np.testing.assert_array_equal(banks['alias'].tables['merge'].host_detector_score, expected)
        self.assertEqual(detector_prior.strip(banks), [])

    def test_sources_reading_withheld_names_are_refused(self):
        detector_prior.check_source((HERE / 'artifacts' / 'scorer_no_score.py').read_text(), 'seed')
        detector_prior.check_source(SIGNAL, 'scorer')
        for name in detector_prior.HIDDEN_NAMES:
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, name):
                detector_prior.check_source(f"x = features['{name}']\n", 'scorer')
        with self.assertRaisesRegex(ValueError, 'detector_score'):
            detector_prior.check_source((HERE / 'artifacts' / 'scorer.py').read_text(), 'seed')

    def test_preflight_row_adds_no_score_column_once_it_is_withheld(self):
        frame = brain_table('1').tables['merge'].features
        self.assertEqual(subset_features(frame, [0], missing_row=True)['detector_score'].tolist()[-1], 0.)
        stripped = frame.drop(columns=['detector_score'])
        self.assertNotIn('detector_score', subset_features(stripped, [0], missing_row=True).columns)

    def test_context_coverage_and_band_flag(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            banks = two_brains()
            for bank in banks.values():
                build_cache(Path(tmp), bank.tables['merge'], band_rows=12, images=False)
            cache = cc.ContextCache(tmp, expose_band=False)
            with self.assertRaisesRegex(ValueError, r'794495/merge \(4 of 16 labeled rows uncached\).*--labeled-extension'):
                detector_prior.check_context_coverage(cache, banks, ['merge'])
            detector_prior.check_context_coverage(cache, banks, ['split'])  # other kinds are not checked
            table = banks['802449'].tables['merge']
            hidden, _ = cache.entry(table).contexts(table, [0])
            shown, _ = cc.ContextCache(tmp).entry(table).contexts(table, [0])
            self.assertNotIn('row_in_band', hidden[0])
            self.assertTrue(shown[0]['row_in_band'])


class DriverTests(unittest.TestCase):
    def test_no_score_run_withholds_the_score_and_keeps_the_frozen_baseline(self):
        from proofreader_evolve.cli import run_precision_evolution as driver
        banks = driver_banks()
        seen = {}
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            args = parse(driver, tmp, '--detector-prior', 'no-score')

            async def revise(run_dir, policy, rules, report, model, *, experiments):
                seen['rules'] = rules.read_text()
                seen['seed'] = policy.read_text()
                seen['visible'] = {p.name: p.read_text() for p in policy.parent.glob('*.json')}
                policy.write_text("def score_candidates(features, ctx):\n"
                                  "    assert 'detector_score' not in features.columns\n"
                                  "    return features['signal'].fillna(0).to_numpy()\n")
                experiments.proposal_path.write_text(json.dumps({'hypothesis': 'signal ranks positives',
                    'strategy': 'use signal', 'family': 'signal', 'parameter_grid': {}}))
                experiments.evaluate()
                return {'summary': 'signal', 'cost_usd': 0}

            path = run_driver(driver, args, banks, revise)
            manifest = json.loads((path / 'manifest.json').read_text())
            baseline = json.loads((path / 'baseline.json').read_text())
            seed = json.loads((path / 'seed.json').read_text())
            record = json.loads((path / 'ledger.jsonl').read_text().splitlines()[0])
            split = json.loads((path / 'dataset_split.json').read_text())
            summary = collect(path)
        self.assertEqual(manifest['detector_prior']['mode'], 'no-score')
        self.assertEqual(manifest['detector_prior']['stripped_cells'], ['794495/merge', '802449/merge', '2/merge'])
        self.assertEqual(split['detector_prior'], 'no-score')
        self.assertEqual(summary['detector_prior'], 'no-score')
        self.assertIn('Detector-prior ablation', seen['rules'])
        self.assertIn('np.zeros', seen['seed'])
        for name, text in seen['visible'].items():
            with self.subTest(file=name):
                self.assertNotIn('detector_score', text)
                self.assertNotIn('bank_context_available', text)
        # Frozen scores are descending by row and labels alternate 0 / 1: the baseline's Top 3 hits once.
        self.assertAlmostEqual(baseline['validation']['cells']['2/merge']['precision'], 1 / 3)
        self.assertAlmostEqual(seed['validation']['cells']['2/merge']['precision'], 1 / 3)
        self.assertEqual(record['validation']['cells']['2/merge']['precision'], 1.)
        self.assertTrue(record['accepted'])
        for bank in banks.values():
            self.assertNotIn('detector_score', bank.tables['merge'].features.columns)

    def test_a_seed_reading_the_score_is_refused_before_evaluation(self):
        from proofreader_evolve.cli import run_precision_evolution as driver
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            args = parse(driver, tmp, '--detector-prior', 'no-score', '--start-from', str(HERE / 'artifacts' / 'scorer.py'))
            with self.assertRaisesRegex(ValueError, 'Starting scorer .* references detector_score'):
                run_driver(driver, args, driver_banks(), None)

    def test_context_cache_needs_full_labeled_coverage_and_hides_the_band(self):
        from proofreader_evolve.cli import run_precision_evolution as driver
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root = Path(tmp)
            partial, full = driver_banks(), driver_banks()
            for bank in partial.values():
                build_cache(root / 'partial', bank.tables['merge'], band_rows=12, images=False)
            for bank in full.values():
                build_cache(root / 'full', bank.tables['merge'], band_rows=16, images=False)
            common = ('--detector-prior', 'no-score', '--descriptor-bank', str(root / 'bank'),
                      '--descriptor-workers', '2', '--descriptor-wall-seconds', '120')
            args = parse(driver, tmp, *common, '--context-cache', str(root / 'partial'))
            with self.assertRaisesRegex(ValueError, 'needs every GT-labeled row cached'):
                run_driver(driver, args, partial, None)

            seen = {}

            async def revise(run_dir, policy, rules, report, model, *, experiments):
                (policy.parent / 'descriptor.py').write_text(BLIND_DESCRIPTOR)
                for scope in ('pilot', 'boundary'):
                    with self.assertRaisesRegex(ValueError, r"scope must be one of \('all_cached',\)"):
                        experiments.compute_descriptors(scope)
                seen['result'] = experiments.compute_descriptors('all_cached')
                seen['statistics'] = (policy.parent / 'feature_statistics.json').read_text()
                experiments.restore('parent')
                return {'summary': 'descriptor only', 'cost_usd': 0}

            args = parse(driver, tmp, *common, '--context-cache', str(root / 'full'))
            run_driver(driver, args, full, revise)
        self.assertEqual(seen['result']['status'], 'registered')
        self.assertNotIn('bank_context_available', seen['statistics'])
        self.assertNotIn('detector_score', seen['statistics'])


if __name__ == '__main__':
    unittest.main()
