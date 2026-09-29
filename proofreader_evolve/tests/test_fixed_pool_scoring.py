"""Fixed precision budget, scorer isolation, gates and mocked evolution control flow."""

import asyncio
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from proofreader_evolve.harness import fixed_pool_scoring as scoring
from proofreader_evolve.harness.native_pool import NativeTable, NativeTables, pool_digest


BASELINE = "def score_candidates(features, ctx):\n    return features['detector_score'].to_numpy()\n"
IMPROVED = "def score_candidates(features, ctx):\n    return features['evidence'].to_numpy()\n"


def fixture(brain="1"):
    rows = [{"segment_id_a": 1, "segment_id_b": b} for b in (2, 3, 4)]
    frame = pd.DataFrame({"evidence": [.1, .9, .2], "detector_score": [.9, .1, .2]})
    table = NativeTable("split", frame, rows, np.array([0, 1, 0]),
                        {"pool_sha256": pool_digest("split", rows), "training_brain": "3"})
    return NativeTables(brain, {"split": table}, {})


class FixedScoringTests(unittest.TestCase):
    def test_fixed_k_tie_break_and_small_pool(self):
        metrics, chosen = scoring.rank_metrics([.5, .5], [1, 0], ["z", "a"], 1)
        self.assertEqual(chosen.tolist(), [1])
        self.assertEqual(metrics["precision"], 0)
        metrics, _ = scoring.rank_metrics([.5, .5], [1, 0], ["z", "a"], 100)
        self.assertEqual(metrics["effective_k"], 2)
        self.assertEqual(metrics["precision"], .5)

    def test_abstention_nan_duplicates_and_empty_fail(self):
        for scores, labels, keys in (([], [], []), ([1], [0, 1], ["a", "b"]),
                                     ([float("nan")], [1], ["a"]), ([1, 2], [0, 1], ["a", "a"])):
            with self.subTest(scores=scores), self.assertRaises(ValueError):
                scoring.rank_metrics(scores, labels, keys, 1)

    def test_scorer_inputs_have_no_truth_ids_or_mutable_shared_frame(self):
        bank = fixture()
        source = ("def score_candidates(features, ctx):\n"
                  "    assert set(ctx) == {'kind'}\n"
                  "    assert list(features.columns) == ['evidence', 'detector_score']\n"
                  "    features['evidence'] = 0\n"
                  "    return features['detector_score'].to_numpy()\n")
        report = scoring.evaluate(source, {"1": bank}, {"split": 1})
        self.assertEqual(report["macro_precision"], 0)
        self.assertEqual(bank.tables["split"].features["evidence"].tolist(), [.1, .9, .2])
        self.assertNotIn("training_examples", report["cells"]["1/split"])

    def test_pool_changes_and_enum_params_rejected(self):
        for source in ("ENUM_PARAMS = {}\n" + BASELINE,
                       "def propose_edits(sites, ctx): return []\n" + BASELINE,
                       "def score_candidates(features, ctx): return []"):
            with self.assertRaises(ValueError):
                scoring.evaluate(source, {"1": fixture()}, {"split": 1})

    def test_gate_same_pool_k_strict_gain_and_no_cell_regression(self):
        before = scoring.evaluate(BASELINE, {"1": fixture()}, {"split": 1})
        after = scoring.evaluate(IMPROVED, {"1": fixture()}, {"split": 1})
        self.assertTrue(scoring.acceptance(before, after)[0])
        self.assertFalse(scoring.acceptance(after, after)[0])
        self.assertFalse(scoring.acceptance(after, before)[0])
        changed = deepcopy(after)
        changed["cells"]["1/split"]["pool_sha256"] = "other"
        with self.assertRaises(ValueError):
            scoring.acceptance(before, changed)
        changed = deepcopy(after)
        changed["cells"]["1/split"]["labels_sha256"] = "different truth, same count"
        with self.assertRaises(ValueError):
            scoring.acceptance(before, changed)
        changed = deepcopy(after)
        changed["cells"]["1/split"]["effective_k"] = 2
        with self.assertRaises(ValueError):
            scoring.acceptance(before, changed)

    def test_stateful_scorer_rejected(self):
        source = "counter = 0\ndef score_candidates(features, ctx):\n    global counter\n    counter += 1\n    return features['evidence'].to_numpy() + counter\n"
        with self.assertRaisesRegex(ValueError, "deterministic"):
            scoring.evaluate(source, {"1": fixture()}, {"split": 1})

    def test_three_generations_accept_reject_invalid_and_no_validation_feedback(self):
        from proofreader_evolve.cli import run_precision_evolution as driver
        with tempfile.TemporaryDirectory() as tmp:
            args = driver.parse_args(["--train-brains", "1", "--validation-brains", "2",
                                      "--split-k", "1", "--generations", "3", "--runs-dir", tmp])
            attempts = iter([IMPROVED, BASELINE, "bad syntax!"])

            async def revise(run_dir, policy, rules, report, model):
                text = report.read_text()
                self.assertNotIn('"validation"', text)
                policy.write_text(next(attempts))
                return {"summary": "mock", "usage": {}}

            with patch.object(driver.pc, "resolve_detector_runs", return_value={}), \
                    patch.object(driver, "ensure_native_tables", side_effect=lambda brain, *a, **k: fixture(brain)):
                path = asyncio.run(driver.run(args, revise_fn=revise))
            records = [json.loads(line) for line in (path / "ledger.jsonl").read_text().splitlines()]
            self.assertEqual([r["accepted"] for r in records], [True, False, False])
            self.assertIsNone(records[1]["validation"])
            self.assertIsNone(records[2]["train"])
            self.assertIsNone(records[2]["validation"])
            self.assertEqual((path / "best_scorer.py").read_text(), IMPROVED)
            self.assertEqual(json.loads((path / "final.json").read_text())["validation"]["macro_precision"], 1)

    def test_train_validation_overlap_rejected(self):
        from proofreader_evolve.cli import run_precision_evolution as driver
        with self.assertRaises(SystemExit):
            driver.parse_args(["--train-brains", "1", "--validation-brains", "1"])

    def test_default_entrypoint_dispatches_precision_not_edit_loop(self):
        from proofreader_evolve.cli import run_evolution, run_precision_evolution
        with patch.object(sys, "argv", ["run_evolution", "--generations", "0"]), \
                patch.object(run_precision_evolution, "main", return_value=0) as main:
            self.assertEqual(run_evolution.main(), 0)
        main.assert_called_once_with(["--generations", "0"])

    def test_detector_fitted_brain_cannot_be_validation(self):
        from proofreader_evolve.cli import run_precision_evolution as driver
        args = driver.parse_args(["--train-brains", "1", "--validation-brains", "3", "--generations", "0"])
        with patch.object(driver.pc, "resolve_detector_runs", return_value={}), \
                patch.object(driver, "ensure_native_tables", side_effect=lambda brain, *a, **k: fixture(brain)), \
                self.assertRaisesRegex(ValueError, "Detector-fitted"):
            asyncio.run(driver.run(args))
