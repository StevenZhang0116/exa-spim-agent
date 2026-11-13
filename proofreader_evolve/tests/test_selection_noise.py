"""--selection-noise-gate (2026-11-12): the bootstrap interval of a candidate's TRAIN
selection gain, the skipped outer evaluation, and what later sessions are told.
Synthetic fixtures only; no brain caches, cloud reads or LLM calls."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from proofreader_evolve.harness import fixed_pool_scoring as scoring
from proofreader_evolve.harness.candidate_pool import CandidatePool
from proofreader_evolve.harness.selection_protocol import NOISE_VERSION, SelectionProtocol
from proofreader_evolve.tests import label_fixture
from proofreader_evolve.tests.test_detector_prior import driver_banks, run_driver
from proofreader_evolve.tests.test_selection_protocol import brain_table

setUpModule = label_fixture.setUpModule
tearDownModule = label_fixture.tearDownModule

NOISE = "features['signal'].fillna(0) * 0 + 7"  # another constant: the same ranking as the constant seed
BUDGETS = scoring.Budgets({"merge": None}, positive_ratio=1., min_positives=0, objective="average_precision")


def brains(count=3, groups=40):
    return {str(index): brain_table(str(index), groups=groups, offset=1000. * index) for index in range(count)}


def states(train, quality):
    """Scores per cell: the label plus noise of the given size."""
    rng = np.random.default_rng(7)
    return {f"{brain}/merge": {"scores": np.asarray(bank.tables["merge"].truth, dtype=float)
                               + rng.normal(0, quality, len(bank.tables["merge"].truth))}
            for brain, bank in train.items()}


class PairedNoiseTests(unittest.TestCase):
    def test_whole_brain_holdouts_give_the_bootstrap_interval(self):
        train = brains()
        protocol = SelectionProtocol(train, BUDGETS, selection_brains=sorted(train), holdout_brains=sorted(train))
        weak, strong = states(train, 3.), states(train, .3)
        better = protocol.paired_noise("merge", weak, strong, draws=200, seed=1, alpha=.2)
        self.assertEqual((better["status"], better["version"], better["beyond_noise"]), ("ok", NOISE_VERSION, True))
        expected = scoring.paired_bootstrap(weak, strong, train, BUDGETS, target_kind="merge", draws=200, seed=1, alpha=.2)
        self.assertEqual(better["delta_ci"], expected["delta_ci"])
        same = protocol.paired_noise("merge", weak, weak, draws=200, seed=1, alpha=.2)
        self.assertEqual((same["delta_ci"], same["beyond_noise"]), ([0., 0.], False))
        self.assertFalse(protocol.paired_noise("merge", strong, weak, draws=200, seed=1, alpha=.2)["beyond_noise"])

    def test_within_brain_folds_are_unavailable(self):
        train = brains()
        protocol = SelectionProtocol(train, BUDGETS, selection_brains=sorted(train))
        weak, strong = states(train, 3.), states(train, .3)
        result = protocol.paired_noise("merge", weak, strong, draws=50, seed=1)
        self.assertEqual(result["status"], "unavailable")
        self.assertIn("within-brain folds", result["reason"])
        in_sample = SelectionProtocol(train, BUDGETS, mode="in_sample")
        self.assertEqual(in_sample.paired_noise("merge", weak, strong, draws=50, seed=1)["status"], "ok")
        self.assertEqual(in_sample.paired_noise("merge", {}, strong, draws=50, seed=1)["status"], "unavailable")


class HistoryTests(unittest.TestCase):
    def test_the_pool_lists_the_latest_submissions_without_heldout_numbers(self):
        with tempfile.TemporaryDirectory() as tmp:
            pool = CandidatePool(Path(tmp) / "pool.json", ["merge"])
            entry = lambda name: {"target_kind": "merge", "experiment": name, "component_sha256": name,
                                  "scores_sha256": name, "selection_gate": {"delta": .001}}
            pool.mark_validated(entry("seed/merge"), 0, True)
            pool.mark_validated(entry("gen001/attempt001"), 1, True)
            pool.mark_validated(entry("gen002/attempt003"), 2, False, rejection="gain_too_small")
            item = pool.mark_unmeasured(entry("gen003/attempt002"), 3, {"delta_ci": [-.01, .02], "alpha": .2})
            self.assertEqual(item["selection_gain_interval"], [-.01, .02])
            history = pool.outer_history("merge")
            self.assertEqual([(row["generation"], row["outcome"]) for row in history],
                             [(1, "promoted"), (2, "not_promoted"), (3, "not_measured")])
            self.assertEqual(history[1]["rejection"], "gain_too_small")
            self.assertEqual(len(pool.outer_history("merge", limit=2)), 2)
            self.assertEqual(pool.summary()["kinds"]["merge"]["outer_history"], history)
            # An unmeasured candidate is not a known outer measurement.
            self.assertIsNone(pool.validation_of("merge", entry("gen003/attempt002")))

    def test_the_prompt_quotes_the_history(self):
        from proofreader_evolve.cli.run_precision_evolution import outer_history_text
        self.assertEqual(outer_history_text("merge", []), "")
        text = outer_history_text("merge", [
            {"generation": 2, "experiment": "gen002/attempt003", "outcome": "not_promoted", "rejection": "gain_too_small"},
            {"generation": 3, "experiment": "gen003/attempt002", "outcome": "not_measured",
             "selection_gain": .001, "selection_gain_interval": [-.01, .02], "interval_level": .8}],
            {"draws": 100, "alpha": .2})
        self.assertIn("generation 2 gen002/attempt003 not promoted (gain_too_small)", text)
        self.assertIn("generation 3 gen003/attempt002 not measured (TRAIN selection gain +0.001000, 80% interval "
                      "[-0.010000, +0.020000]", text)
        self.assertIn("selection_gate.resampling", text)


class DriverTests(unittest.TestCase):
    def run_once(self, expression, *extra):
        from proofreader_evolve.cli import run_precision_evolution as driver

        async def revise(run_dir, policy, rules, report, model, *, experiments):
            policy.write_text(f"def score_candidates(features, ctx):\n    return ({expression}).to_numpy()\n")
            experiments.proposal_path.write_text(json.dumps({'hypothesis': 'h', 'strategy': 's', 'family': 'f',
                                                             'parameter_grid': {}}))
            experiments.evaluate()
            return {'summary': 's', 'cost_usd': 0}

        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            args = driver.parse_args(['--min-positives', '0', '--selection-protocol', 'in_sample',
                                      '--train-brains', '794495,802449', '--validation-brains', '2',
                                      '--merge-k', '3', '--split-k', '1', '--generations', '1', '--runs-dir', tmp,
                                      '--skip-auxiliary-ablation', '--bootstrap-draws', '50', '--promotion-gate', 'rule',
                                      '--objective', 'average_precision', '--kind-schedule', 'alternate',
                                      '--detector-prior', 'no-score', *extra])
            path = run_driver(driver, args, driver_banks(), revise)
            record = json.loads((path / 'ledger.jsonl').read_text().splitlines()[0])
            record['_manifest'] = json.loads((path / 'manifest.json').read_text())
            record['_pool'] = json.loads((path / 'candidate_pool.json').read_text())
        return record

    def test_a_gain_within_train_noise_is_not_sent_to_the_outer_evaluator(self):
        # A different constant than the seed's: new bytes and scores, an unchanged ranking, a gain of exactly zero.
        record = self.run_once(NOISE, '--selection-noise-gate')
        self.assertEqual(record['_manifest']['promotion_gate']['selection_noise_gate']['alpha'], .2)
        self.assertFalse(record['selection_noise']['beyond_noise'])
        self.assertEqual(record['selection_gate']['resampling'], record['selection_noise'])  # what the session saw
        self.assertEqual(record['validation_gate']['skipped'], 'selection_within_noise')
        self.assertIsNone(record['validation'])
        self.assertFalse(record['accepted'])
        self.assertNotIn('submission_validated', record)
        self.assertEqual(record['submission_within_noise']['selection_gain_interval'], record['selection_noise']['delta_ci'])
        history = record['_pool']['kinds']['merge']['outer_history']
        self.assertEqual([(row['generation'], row['outcome']) for row in history], [(1, 'not_measured')])
        from proofreader_evolve.harness.evolution_report import generation_status
        self.assertEqual(generation_status(record, set()), 'not_measured')

    def test_a_clear_gain_is_measured_and_the_check_is_off_by_default(self):
        record = self.run_once("features['signal'].fillna(0)", '--selection-noise-gate')
        self.assertTrue(record['selection_noise']['beyond_noise'])
        self.assertIsNotNone(record['validation'])
        self.assertNotIn('submission_within_noise', record)
        off = self.run_once(NOISE)
        self.assertNotIn('selection_noise', off)
        self.assertNotIn('resampling', off['selection_gate'])
        self.assertIsNotNone(off['validation'])
        self.assertIsNone(off['_manifest']['promotion_gate']['selection_noise_gate'])
