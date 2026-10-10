"""Fixed multi-brain validation selection and mean-only promotion."""

import asyncio
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from proofreader_evolve.cli import run_precision_evolution as driver
from proofreader_evolve.harness import fixed_pool_scoring as scoring
from proofreader_evolve.tests.test_fixed_pool_scoring import fixture, IMPROVED
from proofreader_evolve.tests.label_fixture import setUpModule, tearDownModule  # noqa: F401


def report(precisions):
    cells = {}
    for brain, kinds in precisions.items():
        for kind, precision in kinds.items():
            cells[f"{brain}/{kind}"] = {
                "precision": precision, "pool_sha256": f"pool-{brain}-{kind}",
                "labels_sha256": f"labels-{brain}-{kind}", "pool_size": 1000,
                "n_labeled": 1000, "requested_k": 100, "effective_k": 100, "positives": 100,
            }
    return {"version": scoring.SCORING_VERSION, "cells": cells,
            **scoring.aggregate_precision(cells)}


class MeanGateTests(unittest.TestCase):
    def test_equal_brain_and_kind_weights_and_allowed_regression(self):
        before = report({"a": {"merge": .8, "split": .6},
                         "b": {"merge": .1, "split": .2}})
        after = report({"a": {"merge": .7, "split": .6},
                        "b": {"merge": .5, "split": .2}})
        self.assertAlmostEqual(before["per_brain_precision"]["a"], .7)
        self.assertAlmostEqual(after["macro_precision"], .5)
        self.assertTrue(scoring.acceptance(before, after, require_no_cell_regression=False)[0])
        self.assertFalse(scoring.acceptance(before, after)[0])
        self.assertFalse(scoring.acceptance(before, after, .1, require_no_cell_regression=False)[0])
        self.assertFalse(scoring.acceptance(after, after, require_no_cell_regression=False)[0])

    def test_mean_does_not_weight_large_pools_more(self):
        value = report({"a": {"split": .8}, "b": {"split": .2}})
        value["cells"]["a/split"].update(pool_size=10, requested_k=1000, effective_k=10)
        value["cells"]["b/split"].update(pool_size=10000, requested_k=1000, effective_k=1000)
        self.assertAlmostEqual(scoring.aggregate_precision(value["cells"])["macro_precision"], .5)

    def test_changed_benchmark_still_fails_in_mean_mode(self):
        before = report({"a": {"split": .8}, "b": {"split": .1}})
        after = report({"a": {"split": .7}, "b": {"split": .5}})
        for key, changed_value in (("pool_sha256", "other"), ("labels_sha256", "other"),
                                   ("pool_size", 1001), ("n_labeled", 999), ("requested_k", 101),
                                   ("effective_k", 99), ("positives", 101)):
            changed = deepcopy(after)
            changed["cells"]["b/split"][key] = changed_value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "fixed benchmark"):
                scoring.acceptance(before, changed, require_no_cell_regression=False)
        del after["cells"]["b/split"]
        with self.assertRaisesRegex(ValueError, "different evaluation contracts"):
            scoring.acceptance(before, after, require_no_cell_regression=False)

    def test_partial_kind_coverage_is_not_averaged(self):
        with self.assertRaisesRegex(ValueError, "same candidate kinds"):
            report({"a": {"merge": .2, "split": .3}, "b": {"merge": .4}})


class SplitSelectionTests(unittest.TestCase):
    def test_defaults_and_explicit_override(self):
        args = driver.parse_args(['--selection-protocol', 'in_sample'])
        self.assertEqual(args.train_brains, ["794495"])
        self.assertIsNone(args.validation_brains)
        self.assertIsNone(driver.parse_args(['--selection-protocol', 'in_sample', '--promotion-gate', 'margin', '--kind-schedule', 'alternate', "--validation-brains", "auto"]).validation_brains)
        self.assertEqual(driver.parse_args(['--selection-protocol', 'in_sample', '--promotion-gate', 'margin', '--kind-schedule', 'alternate', "--validation-brains", "789202,794491"]).validation_brains,
                         ["789202", "794491"])

    def test_auto_excludes_training_unprepared_and_wrong_mcl_and_freezes_list(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            def cache(brain, mcl=100):
                return root / f"dataset_cache_{brain}_mcl{mcl}_add.pkl"
            for brain in ("794495", "789202", "794491", "802449"):
                cache(brain).touch()
            cache("794493", 10).touch()
            def load(brain, *args, **kwargs):
                if brain == "802449":
                    raise FileNotFoundError("Missing matching split table")
                bank = fixture(brain)
                bank.tables["split"].meta["training_brain"] = "794495"
                return bank
            args = driver.parse_args(['--selection-protocol', 'in_sample'])
            with patch.object(driver, "cache_path", side_effect=cache), \
                    patch.object(driver, "ensure_native_tables", side_effect=load) as loader:
                banks, selection = driver._load_evaluation_banks(args, {}, Mock())
            self.assertEqual(args.validation_brains, ["789202", "794491"])
            self.assertEqual(set(banks), {"794495", "789202", "794491"})
            self.assertEqual(selection["excluded"]["794495"], "TRAIN brain")
            self.assertIn("split", selection["excluded"]["802449"])
            self.assertNotIn("794493", selection["candidates"])
            cache("718162").touch()
            self.assertEqual(args.validation_brains, ["789202", "794491"])
            self.assertTrue(all(call.kwargs["prepare"] is False for call in loader.call_args_list))

    def test_missing_explicit_brain_does_not_silently_shrink_split(self):
        args = driver.parse_args(['--selection-protocol', 'in_sample', '--promotion-gate', 'margin', '--kind-schedule', 'alternate', "--train-brains", "1", "--validation-brains", "2"])
        with patch.object(driver, "ensure_native_tables", side_effect=[fixture("1"), FileNotFoundError("missing")]):
            with self.assertRaises(FileNotFoundError):
                driver._load_evaluation_banks(args, {}, Mock())

    def test_auto_does_not_hide_corrupt_tables_or_allow_empty_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            def cache(brain, mcl=100):
                return root / f"dataset_cache_{brain}_mcl{mcl}_add.pkl"
            cache("2").touch()
            for failure, expected in ((ValueError("Corrupt native table"), "Corrupt"),
                                      (FileNotFoundError("missing"), "No eligible")):
                args = driver.parse_args(['--selection-protocol', 'in_sample', '--promotion-gate', 'margin', '--kind-schedule', 'alternate', "--train-brains", "1"])
                with patch.object(driver, "cache_path", side_effect=cache), \
                        patch.object(driver, "ensure_native_tables", side_effect=[fixture("1"), failure]), \
                        self.subTest(failure=failure), self.assertRaisesRegex(ValueError, expected):
                    driver._load_evaluation_banks(args, {}, Mock())

    def test_empty_candidate_pool_fails_before_scoring(self):
        empty = fixture("2")
        empty.tables["split"].truth = np.array([], dtype=np.int8)
        args = driver.parse_args(['--selection-protocol', 'in_sample', '--promotion-gate', 'margin', '--kind-schedule', 'alternate', "--train-brains", "1", "--validation-brains", "2"])
        with patch.object(driver, "ensure_native_tables", side_effect=[fixture("1"), empty]), \
                self.assertRaisesRegex(ValueError, "Empty candidate pool"):
            driver._load_evaluation_banks(args, {}, Mock())


class PromotionTests(unittest.TestCase):
    def test_train_and_one_validation_brain_can_regress_while_mean_promotes(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            args = driver.parse_args(['--selection-protocol', 'in_sample', '--promotion-gate', 'margin', '--kind-schedule', 'alternate', "--train-brains", "1", "--validation-brains", "2,4,5",
                                      "--split-k", "1", "--generations", "1", "--runs-dir", tmp])
            def load(brain, *args, **kwargs):
                bank = fixture(brain)
                if brain in {"1", "2"}:
                    bank.tables["split"].truth = np.array([1, 0, 0])
                return bank
            async def revise(run_dir, policy, rules, feedback, model, *, experiments):
                self.assertEqual(set(experiments.train), {"1"})
                self.assertNotIn('"validation"', feedback.read_text())
                policy.write_text(IMPROVED)
                experiments.proposal_path.write_text(json.dumps({
                    "hypothesis": "Generalize using geometry", "strategy": "Evidence ranking"}))
                experiments.evaluate()
                return {"summary": "Measured TRAIN candidate", "usage": {}}
            with patch.object(driver.pc, "resolve_detector_runs", return_value={}), \
                    patch.object(driver, "ensure_native_tables", side_effect=load):
                directory = asyncio.run(driver.run(args, revise_fn=revise))
            result = json.loads((directory / "gen001/evaluation.json").read_text())
            self.assertTrue(result["accepted"])
            self.assertFalse(result["train_gate"]["passed"])
            self.assertFalse(result["train_gate"]["required_for_promotion"])
            self.assertAlmostEqual(result["validation_gate"]["parent_mean"], 1 / 3)
            self.assertAlmostEqual(result["validation_gate"]["candidate_mean"], 2 / 3)
            self.assertEqual(result["validation_gate"]["regressed_cells"], ["2/split"])
            manifest = json.loads((directory / "manifest.json").read_text())
            self.assertFalse(manifest["promotion_gate"]["require_train_improvement"])
            self.assertEqual(manifest["dataset_selection"]["validation_brains"], ["2", "4", "5"])


if __name__ == "__main__":
    unittest.main()
