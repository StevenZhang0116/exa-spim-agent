"""Per-pool Top-K from GT positives (objective v3, 2026-10-12; per-kind ratio and capless kinds, v4, 2026-10-26).

Budgets resolve K = ceil(positive_ratio[kind] x positives) per pool, limited to the kind's
cap when it has one, and flag pools below the positive floor; such pools are scored and
reported but excluded from the kind-first mean, the paired bootstrap and kind headroom.
Synthetic fixtures only.
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
from proofreader_evolve.harness import kind_schedule as ks
from proofreader_evolve.harness import run_records as records
from proofreader_evolve.harness.native_pool import NativeTable, NativeTables, pool_digest
from proofreader_evolve.harness.train_feedback import write_train_feedback
from proofreader_evolve.tests.test_fixed_pool_scoring import BASELINE, fixture
from proofreader_evolve.tests.label_fixture import setUpModule, tearDownModule  # noqa: F401


def merge_bank(brain, truth, scores):
    rows = [{"segment_id": 1, "node_id": i} for i in range(len(truth))]
    frame = pd.DataFrame({"detector_score": np.asarray(scores, dtype=float)})
    table = NativeTable("merge", frame, rows, np.asarray(truth), {"pool_sha256": pool_digest("merge", rows),
                                                                  "training_brain": "3"})
    return NativeTables(brain, {"merge": table}, {})


def cell(kind_precision, positives=100, k=100, in_mean=True):
    return {"precision": kind_precision, "pool_sha256": "p", "labels_sha256": "l", "pool_size": 1000,
            "n_labeled": 1000, "requested_k": k, "effective_k": k, "positives": positives, "in_mean": in_mean}


class BudgetsTests(unittest.TestCase):
    def test_k_binds_on_the_ratio_or_the_cap_and_zero_ratio_keeps_the_cap(self):
        budgets = scoring.Budgets({"merge": 300, "split": 500}, positive_ratio=2.0, min_positives=20)
        nine = np.array([1.] * 9 + [0.] * 440 + [np.nan] * 50)
        many = np.array([1.] * 825 + [0.] * 100)
        self.assertEqual(budgets.k_for("merge", nine), 18)
        self.assertFalse(budgets.in_mean(nine))
        self.assertEqual(budgets.k_for("merge", many), 300)
        self.assertTrue(budgets.in_mean(many))
        self.assertEqual(budgets.k_for("merge", np.zeros(5)), 1)  # never below one ranked row
        self.assertEqual(scoring.Budgets({"merge": 300}).k_for("merge", nine), 300)
        self.assertTrue(scoring.Budgets({"merge": 300}).in_mean(np.zeros(5)))
        self.assertEqual(budgets["split"], 500)
        self.assertEqual(dict(budgets), {"merge": 300, "split": 500})
        described = budgets.describe()
        self.assertEqual((described["caps"], described["positive_ratio"], described["min_positives"]),
                         ({"merge": 300, "split": 500}, {"merge": 2.0, "split": 2.0}, 20))
        self.assertIn("ceil(positive_ratio[kind] x GT positives)", described["k_rule"])
        for bad in ({"merge": 0}, {"merge": 1.5}):
            with self.assertRaises(ValueError):
                scoring.Budgets(bad)
        for ratio, floor in ((-1., 0), (float("nan"), 0), (1., -1), (1., 2.5)):
            with self.subTest(ratio=ratio, floor=floor), self.assertRaises(ValueError):
                scoring.Budgets({"merge": 1}, positive_ratio=ratio, min_positives=floor)

    def test_per_kind_ratio_and_a_kind_without_a_cap(self):
        # 2026-10-26 defaults: merge 2 x positives under a cap of 300, split exactly its positives, no cap.
        budgets = scoring.Budgets({"merge": 300, "split": None}, positive_ratio={"merge": 2.0, "split": 1.0},
                                  min_positives=20)
        many = np.array([1.] * 825 + [0.] * 100 + [np.nan] * 40)
        nine = np.array([1.] * 9 + [0.] * 440)
        self.assertEqual(budgets.k_for("split", many), 825)  # K = positives, above the old cap of 500/1000
        self.assertEqual(budgets.k_for("merge", many), 300)
        self.assertEqual(budgets.k_for("split", nine), 9)
        self.assertEqual(budgets.k_for("split", np.zeros(3)), 1)
        self.assertIsNone(budgets["split"])
        self.assertEqual(dict(budgets), {"merge": 300, "split": None})
        self.assertEqual(budgets.positive_ratio, {"merge": 2.0, "split": 1.0})
        self.assertEqual(budgets.kind_budget("merge", [18, 300]), 300)  # the cap, whatever the pools resolve
        self.assertEqual(budgets.kind_budget("split", [9, 825]), 825)  # no cap: the largest resolved pool K
        self.assertIsNone(budgets.kind_budget("split"))
        self.assertEqual(budgets.kind_budgets({"a/split": {"requested_k": 9}, "b/split": {"requested_k": 825},
                                               "a/merge": {"requested_k": 18}}), {"merge": 300, "split": 825})
        described = budgets.describe()
        self.assertEqual((described["caps"], described["positive_ratio"]),
                         ({"merge": 300, "split": None}, {"merge": 2.0, "split": 1.0}))
        self.assertIn("limited to caps[kind] when the kind has a cap", described["k_rule"])
        with self.assertRaisesRegex(ValueError, "no Top-K cap"):
            scoring.Budgets({"split": None})  # fixed K without a cap is undefined
        with self.assertRaisesRegex(ValueError, "exactly the budgeted kinds"):
            scoring.Budgets({"merge": 300, "split": None}, positive_ratio={"merge": 2.0})
        with self.assertRaises(ValueError):
            scoring.Budgets({"merge": 0.5})

    def test_plain_mappings_coerce_to_fixed_k_counting_every_pool(self):
        budgets = scoring.Budgets.coerce({"split": 7})
        self.assertIsInstance(budgets, scoring.Budgets)
        self.assertEqual((budgets.positive_ratio, budgets.min_positives, budgets["split"]), ({"split": 0.}, 0, 7))
        self.assertIs(scoring.Budgets.coerce(budgets), budgets)

    def test_resolve_lists_every_pool_from_labels_only(self):
        banks = {"a": merge_bank("a", [1, 1, 1, 0, 0, 0, 0, 0], np.linspace(1, 0, 8)),
                 "b": merge_bank("b", [1, 0, 0, 0, 0, 0, 0, 0], np.linspace(1, 0, 8))}
        budgets = scoring.Budgets({"merge": 4, "split": 9}, positive_ratio=1.0, min_positives=2)
        resolved = budgets.resolve(banks)
        self.assertEqual(resolved["a/merge"], {"positives": 3, "cap": 4, "requested_k": 3, "in_mean": True})
        self.assertEqual(resolved["b/merge"], {"positives": 1, "cap": 4, "requested_k": 1, "in_mean": False})


class ScoringTests(unittest.TestCase):
    def test_evaluate_ranks_each_pool_at_its_own_k_and_reports_membership(self):
        banks = {"a": merge_bank("a", [1, 1, 0, 0, 1, 0, 0, 0, 0, 0], np.linspace(1, 0, 10)),
                 "b": merge_bank("b", [0, 0, 0, 0, 0, 0, 0, 0, 0, 1], np.linspace(1, 0, 10))}
        budgets = scoring.Budgets({"merge": 8}, positive_ratio=2.0, min_positives=2)
        report = scoring.evaluate(BASELINE, banks, budgets)
        a, b = report["cells"]["a/merge"], report["cells"]["b/merge"]
        self.assertEqual((a["requested_k"], a["effective_k"], a["tp"], a["in_mean"]), (6, 6, 3, True))
        self.assertEqual((b["requested_k"], b["effective_k"], b["tp"], b["in_mean"]), (2, 2, 0, False))
        self.assertEqual(report["excluded_cells"], ["b/merge"])
        self.assertAlmostEqual(report['per_kind_precision']['merge'], .5)  # only a/merge counts: 3 / 6
        self.assertEqual(report["per_kind_precision"], {"merge": .5})
        self.assertNotIn('per_brain_precision', report)
        self.assertNotIn('macro_precision', report)
        self.assertEqual(report["budgets"]["caps"], {"merge": 8})
        self.assertEqual(report["version"], scoring.SCORING_VERSION)
        fixed = scoring.evaluate(BASELINE, banks, {"merge": 8})
        self.assertEqual(fixed["cells"]["b/merge"]["requested_k"], 8)
        self.assertTrue(fixed["cells"]["b/merge"]["in_mean"])
        self.assertEqual(fixed["excluded_cells"], [])

    def test_each_kind_uses_only_its_counted_cells(self):
        cells = {"a/merge": cell(.8), "a/split": cell(.6), "b/merge": cell(.1), "b/split": cell(.2)}
        self.assertEqual(scoring.aggregate_precision(cells)['per_kind_precision'], {'merge': .45, 'split': .4})
        cells["b/merge"] = cell(.9, positives=9, k=18, in_mean=False)
        result = scoring.aggregate_precision(cells)
        self.assertEqual(result['per_kind_precision'], {'merge': .8, 'split': .4})
        self.assertEqual(result["excluded_cells"], ["b/merge"])
        self.assertAlmostEqual(result["per_kind_precision"]["merge"], .8)
        self.assertNotIn('per_brain_precision', result)
        cells["a/merge"] = cell(.8, in_mean=False)
        with self.assertRaisesRegex(ValueError, "No merge pool counts"):
            scoring.aggregate_precision(cells)
        with self.assertRaisesRegex(ValueError, "same candidate kinds"):
            scoring.aggregate_precision({"a/merge": cell(.1), "a/split": cell(.2), "b/split": cell(.3)})

    def test_acceptance_pins_mean_membership(self):
        before = {"version": scoring.SCORING_VERSION, "cells": {"a/merge": cell(.5), "b/merge": cell(.1)}}
        before.update(scoring.aggregate_precision(before["cells"]))
        after = json.loads(json.dumps(before))
        after["cells"]["b/merge"]["in_mean"] = False
        after.update(scoring.aggregate_precision(after["cells"]))
        with self.assertRaisesRegex(ValueError, "fixed benchmark: b/merge/in_mean"):
            scoring.acceptance(before, after, require_no_cell_regression=False, target_kind='merge')

    def test_bootstrap_uses_resolved_k_and_skips_excluded_cells(self):
        truth_a = np.array([1, 0, 1, 0, 0, 0, 1, 0, 0, 0], dtype=float)
        truth_b = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1], dtype=float)
        banks = {"a": merge_bank("a", truth_a, np.linspace(1, 0, 10)),
                 "b": merge_bank("b", truth_b, np.linspace(1, 0, 10))}
        same = np.linspace(1, 0, 10)
        improved_b = same.copy()
        improved_b[-1] = 2.  # the excluded pool's only positive moves to the top
        parent = {"a/merge": {"scores": same}, "b/merge": {"scores": same}}
        candidate = {"a/merge": {"scores": same}, "b/merge": {"scores": improved_b}}
        budgets = scoring.Budgets({"merge": 8}, positive_ratio=2.0, min_positives=2)
        result = scoring.paired_bootstrap(parent, candidate, banks, budgets, draws=50, seed=3, target_kind='merge')
        self.assertEqual(result["excluded_cells"], ["b/merge"])
        self.assertEqual(result["delta_ci"], [0., 0.])
        self.assertGreater(result["cells"]["b/merge"]["delta_ci"][1], 0.)
        self.assertEqual(result["version"], scoring.BOOTSTRAP_VERSION)
        counted = scoring.paired_bootstrap(parent, candidate, banks, {"merge": 8}, draws=50, seed=3, target_kind='merge')
        self.assertEqual(counted["excluded_cells"], [])
        self.assertGreater(counted["delta_ci"][1], 0.)
        with self.assertRaisesRegex(ValueError, "counts toward the bootstrap mean"):
            scoring.paired_bootstrap(parent, candidate, banks,
                                     scoring.Budgets({"merge": 8}, min_positives=50), draws=5, target_kind='merge')


class ScheduleTests(unittest.TestCase):
    def test_headroom_and_saturation_ignore_pools_below_the_floor(self):
        kinds, budgets = ["merge", "split"], {"merge": 300, "split": 500}
        cells = {"a/merge": {"positives": 81, "requested_k": 162, "precision": .2, "in_mean": True},
                 "b/merge": {"positives": 9, "requested_k": 18, "precision": 0., "in_mean": False},
                 "a/split": {"positives": 1476, "requested_k": 500, "precision": .9, "in_mean": True},
                 "b/split": {"positives": 1020, "requested_k": 500, "precision": .93, "in_mean": True}}
        room = ks.headroom(cells, budgets, kinds)
        self.assertAlmostEqual(room["merge"], .5 - .2)  # b/merge's full 0.5 of room is ignored
        self.assertAlmostEqual(room["split"], ((1 - .9) + (1 - .93)) / 2)
        self.assertEqual(ks.saturation_thresholds(cells, budgets, kinds),
                         {"merge": .5 / 162, "split": .5 / 1000})
        self.assertEqual(ks.saturation_thresholds({}, budgets, kinds), {"merge": 0., "split": 0.})
        uncapped = {"merge": 300, "split": None}
        self.assertEqual(ks.saturation_thresholds(cells, uncapped, kinds),
                         {"merge": .5 / 162, "split": .5 / 1000})
        self.assertEqual(ks.saturation_thresholds({}, uncapped, kinds), {"merge": 0., "split": 0.})
        below = {"a/split": {"positives": 9, "requested_k": 9, "precision": .5, "in_mean": False}}
        self.assertEqual(ks.saturation_thresholds(below, uncapped, kinds), {"merge": 0., "split": 0.})
        result = ks.allocate(kinds, cells={}, budgets=uncapped, records=[], paused={}, pending={})
        self.assertFalse(result["kinds"]["split"]["saturated"])
        saturated = {"a/merge": {"positives": 81, "requested_k": 162, "precision": .499, "in_mean": True},
                     "a/split": {"positives": 1476, "requested_k": 500, "precision": .5, "in_mean": True}}
        result = ks.allocate(kinds, cells=saturated, budgets=budgets, records=[], paused={}, pending={})
        self.assertTrue(result["kinds"]["merge"]["saturated"])  # 0.001 < 0.5 / 162
        self.assertEqual(result["chosen"], "split")


class FeedbackAndRecordTests(unittest.TestCase):
    def test_feedback_exposes_caps_rule_and_membership_without_validation_text(self):
        budgets = scoring.Budgets({"split": 3}, positive_ratio=2.0, min_positives=0)
        report = scoring.evaluate(BASELINE, {"1": fixture("1")}, budgets, examples=True)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "feedback.json"
            write_train_feedback(path, report, [], budgets, target_kind="split")
            text = path.read_text()
        feedback = json.loads(text)
        self.assertEqual(feedback["budgets"], {"split": 3})
        self.assertEqual(feedback["k_rule"]["caps"], {"split": 3})
        self.assertEqual(feedback["parent_train"]["cells"]["1/split"]["requested_k"], 2)  # ceil(2 x 1 positive)
        self.assertTrue(feedback["parent_train"]["cells"]["1/split"]["in_mean"])
        self.assertIn("k_rule explains", feedback["reading_guide"])
        self.assertNotIn("validation", text.lower())

    def test_feedback_budget_of_a_capless_kind_is_the_largest_train_pool_k(self):
        budgets = scoring.Budgets({"split": None}, positive_ratio=1.0, min_positives=0)
        report = scoring.evaluate(BASELINE, {"1": fixture("1"), "2": fixture("2")}, budgets, examples=True)
        self.assertEqual(report["cells"]["1/split"]["requested_k"], 1)  # the fixture's one positive
        self.assertIsNone(report["budgets"]["caps"]["split"])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "feedback.json"
            write_train_feedback(path, report, [], budgets, target_kind="split")
            feedback = json.loads(path.read_text())
        self.assertEqual(feedback["budgets"], {"split": 1})
        self.assertIsNone(feedback["k_rule"]["caps"]["split"])
        self.assertIn("one K per kind", feedback["reading_guide"])

    def test_record_readers_accept_known_objective_versions(self):
        self.assertEqual(records.PRECISION, scoring.SCORING_VERSION)
        self.assertEqual(records.PRECISION_VERSIONS,
                         {records.LEGACY_PRECISION, records.PRECISION_V2, records.PRECISION_V3, records.PRECISION_V4,
                          records.PRECISION_V5, records.PRECISION})
        self.assertEqual(records.objective({"objective": records.PRECISION_V2}), records.PRECISION_V2)
        self.assertEqual(records.objective({"objective": records.PRECISION_V3}), records.PRECISION_V3)
        self.assertEqual(records.PRECISION_V3, "native-precision-at-k-v3-positive-scaled-k")
        self.assertEqual(records.PRECISION_V4, "native-precision-at-k-v4-per-kind-k")
        self.assertEqual(records.PRECISION, "native-precision-at-k-v6-per-kind-objective")


class DriverTests(unittest.TestCase):
    def test_defaults_and_argument_validation(self):
        from proofreader_evolve.cli import run_precision_evolution as driver
        args = driver.parse_args([])
        self.assertEqual((args.k_positive_ratio, args.min_positives), ({"merge": 1.0, "split": 1.0}, 20))
        # No cap on either kind since 2026-10-27 (merge 300 before; split 1000 until 2026-10-26).
        self.assertEqual((args.merge_k, args.split_k), (None, None))
        with redirect_stderr(io.StringIO()):
            for bad in (["--k-positive-ratio", "-1"], ["--k-positive-ratio", "nan"], ["--min-positives", "-1"],
                        ["--k-positive-ratio", "0"],  # no kind has a cap, so a zero ratio leaves K undefined
                        ["--k-positive-ratio", "0", "--split-k", "3"],  # merge still capless
                        ["--k-positive-ratio", "merge=2"], ["--k-positive-ratio", "merge=2,omit=1"],
                        ["--k-positive-ratio", "merge=2,merge=1,split=1"], ["--split-k", "0"], ["--merge-k", "x"]):
                with self.subTest(bad=bad), self.assertRaises(SystemExit):
                    driver.parse_args(bad)
        self.assertEqual(driver.parse_args(["--k-positive-ratio", "0", "--merge-k", "300", "--split-k", "3"]).k_positive_ratio,
                         {"merge": 0., "split": 0.})
        self.assertEqual(driver.parse_args(["--k-positive-ratio", "merge=2, split=1.5"]).k_positive_ratio,
                         {"merge": 2., "split": 1.5})
        self.assertEqual(driver.parse_args(["--split-k", "1000"]).split_k, 1000)
        self.assertIsNone(driver.parse_args(["--merge-k", "none"]).merge_k)

    def run_driver(self, extra):
        from proofreader_evolve.cli import run_precision_evolution as driver
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            args = driver.parse_args(['--selection-protocol', 'in_sample', '--promotion-gate', 'margin',
                                      '--kind-schedule', 'alternate', '--train-brains', '1',
                                      '--validation-brains', '2', '--split-k', '3', '--generations', '0',
                                      '--runs-dir', tmp, *extra])
            with patch.object(driver.pc, "resolve_detector_runs", return_value={}), \
                    patch.object(driver, "ensure_native_tables", side_effect=lambda brain, *a, **k: fixture(brain)):
                path = asyncio.run(driver.run(args))
            return (json.loads((path / "manifest.json").read_text()), json.loads((path / "seed.json").read_text()),
                    (path / "trajectory.txt").read_text())

    def test_manifest_records_resolved_pool_budgets_and_the_seed_uses_them(self):
        manifest, seed, trace = self.run_driver(['--min-positives', '0', '--k-positive-ratio', '2'])
        self.assertEqual(manifest["budgets"]["caps"], {"merge": None, "split": 3})
        self.assertEqual(manifest["budgets"]["positive_ratio"], {"merge": 2.0, "split": 2.0})
        self.assertEqual(manifest["pool_budgets"]["1/split"],
                         {"positives": 1, "cap": 3, "requested_k": 2, "in_mean": True})
        self.assertEqual(seed["validation"]["cells"]["2/split"]["requested_k"], 2)
        self.assertIn("pool_budgets:", trace)
        self.assertIn("2/split K=2/3 positives=1", trace)
        self.assertNotIn("label_availability_warning", trace)
        self.assertEqual(manifest["objective"], scoring.SCORING_VERSION)

    def test_a_capless_split_ranks_its_positives(self):
        # The last --split-k wins over run_driver's default of 3: no cap, default ratio 1 for split.
        manifest, seed, trace = self.run_driver(['--min-positives', '0', '--split-k', 'none'])
        self.assertEqual(manifest["budgets"]["caps"], {"merge": None, "split": None})
        self.assertEqual(manifest["budgets"]["positive_ratio"], {"merge": 1.0, "split": 1.0})
        self.assertEqual(manifest["pool_budgets"]["1/split"],
                         {"positives": 1, "cap": None, "requested_k": 1, "in_mean": True})
        self.assertEqual(seed["validation"]["cells"]["2/split"]["requested_k"], 1)
        self.assertIn("2/split K=1/no cap positives=1", trace)
        self.assertNotIn("label_availability_warning", trace)

    def test_fixed_k_mode_keeps_the_crowding_warning(self):
        manifest, seed, trace = self.run_driver(['--min-positives', '0', '--k-positive-ratio', '0', '--merge-k', '3'])
        self.assertEqual(seed["validation"]["cells"]["2/split"]["requested_k"], 3)
        self.assertIn("label_availability_warning", trace)

    def test_a_kind_with_every_pool_below_the_floor_fails_before_scoring(self):
        with self.assertRaisesRegex(ValueError, "below --min-positives 2"):
            self.run_driver(['--min-positives', '2'])


if __name__ == "__main__":
    unittest.main()
