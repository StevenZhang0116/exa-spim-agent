"""Precision run reporting: no brains, APIs, downloads or evolution runs."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from proofreader_evolve.harness import run_records as records
from proofreader_evolve.harness.collect_run import collect, resolve_run_dir


def precision(gen=1, accepted=True, validation=.8, cost=1):
    report = {"version": records.PRECISION, "macro_precision": validation,
              "policy_sha256": "child", "cells": {"1/split": {"precision": validation}}}
    return {"generation": gen, "objective": records.PRECISION,
            "accounting_semantics": "per_generation", "accepted": accepted,
            "parent_validation_precision": .5, "train": report,
            "validation": report if validation is not None else None,
            "reviser": {"cost_usd": cost}, "wall_seconds": 1,
            "reason": "VALIDATION: improved" if accepted else "TRAIN: rejected"}



def save_run(path, rows, version=records.PRECISION):
    (path / "ledger.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (path / "manifest.json").write_text(json.dumps({"objective": version, "train_brains": ["1"],
                                                   "development_validation_brains": ["2"]}))


class RecordTests(unittest.TestCase):
    def test_costs_sum_even_when_increasing_decreasing_or_equal(self):
        for costs, expected in [([1, 2], [1, 3]), ([2, 1], [2, 3]), ([1, 1], [1, 2])]:
            rows = [precision(i + 1, cost=cost) for i, cost in enumerate(costs)]
            self.assertEqual(records.cumulative_accounting(rows), expected)

    def test_missing_cost_propagates_and_invalid_accounting_fails(self):
        self.assertEqual(records.cumulative_accounting([precision(cost=None), precision(2)]), [None, None])
        row = precision()
        row["accounting_semantics"] = "session_cumulative"
        with self.assertRaises(ValueError):
            records.cumulative_accounting([row])
        with self.assertRaises(ValueError):
            records.cumulative_accounting([precision(cost=-1)])

    def test_missing_validation_is_not_zero(self):
        row = precision(accepted=False, validation=None)
        self.assertIsNone(records.validation_value(row))
        self.assertEqual(records.parent_values([row]), [.5])

    def test_old_unknown_and_mixed_objectives_are_rejected(self):
        for row in ({"scoring_version": "structural-v1"},
                    {"heldout_fitness": 1}, {"objective": "future-v42"}):
            with self.subTest(row=row), self.assertRaises(ValueError):
                records.objective(row)
            with self.assertRaises(ValueError):
                records.run_objective(rows=[precision(), row])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            save_run(root, [], "structural-v1")
            with self.assertRaises(ValueError):
                collect(root)

    def test_pairs_require_same_policy_and_completed_validation(self):
        row = precision()
        row["train"] = dict(row["train"], policy_sha256="parent")
        with self.assertRaisesRegex(ValueError, "Mismatched"):
            records.paired_measurements([row])
        self.assertEqual(records.paired_measurements([precision(validation=None)]), [])

    def test_collect_precision_and_baseline_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            save_run(root, [precision(), precision(2, cost=2)])
            (root / "seed.json").write_text(json.dumps({"validation": {"macro_precision": .5}}))
            (root / "best_scorer.py").write_text("# accepted scorer")
            summary = collect(root)
            self.assertEqual(summary["total_cost_usd"], 3)
            self.assertEqual(summary["final_accepted_validation"], .8)
            self.assertEqual(summary["final_policy"], str(root / "best_scorer.py"))
            self.assertEqual(summary["brain_roles"]["train"], ["1"])
            (root / "ledger.jsonl").unlink()
            summary = collect(root)
            self.assertEqual(summary["n_generations"], 0)
            self.assertEqual(summary["final_accepted_validation"], .5)

    def test_best_rejected_score_is_not_final_accepted_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            save_run(root, [precision(), precision(2, accepted=False, validation=.99)])
            self.assertEqual(collect(root)["final_accepted_validation"], .8)

    def test_brain_lookup_ignores_retired_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            live, retired = root / "precision_fixture", root / "1_retired"
            live.mkdir()
            retired.mkdir()
            save_run(live, [precision()])
            save_run(retired, [], "structural-v1")
            with patch("proofreader_evolve.harness.collect_run.RUNS", root):
                self.assertEqual(resolve_run_dir("1"), live)

    def test_duplicate_generations_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            save_run(root, [precision(), precision()])
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                records.load_ledger(root)

    def test_plots_render_and_preserve_missing_measurements(self):
        from proofreader_evolve.plotting import plot_run_performance as performance
        from proofreader_evolve.plotting import plot_search_dynamics as search
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [precision(), precision(2, False, None)]
            save_run(root, rows)
            performance.make_figure(rows, "fixture", root / "performance.png")
            search.make_figure(rows, "fixture", root / "search.png", root)
            search.write_md(rows, "fixture", root, root / "search.md")
            performance.write_csv(rows, records.parent_values(rows), root / "series.csv")
            self.assertGreater((root / "performance.png").stat().st_size, 100)
            self.assertIn("not measured", (root / "search.md").read_text())
            self.assertEqual(search._outcome_reason(rows[-1])[1], records.decision_reason(rows[-1]))


if __name__ == "__main__":
    unittest.main()
