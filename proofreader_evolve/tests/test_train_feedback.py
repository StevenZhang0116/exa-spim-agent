"""Feedback size, sample alignment and TRAIN-only history regressions."""

import asyncio
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

from proofreader_evolve.harness import train_feedback as feedback
from proofreader_evolve.cli import run_precision_evolution as driver


def report(n_features=100):
    examples = [{"label": label, "score": .123456789,
                 "features": {f"feature_{j}_descriptive_name": None if j == 0 else i * .137891234 + j
                              for j in range(n_features)}}
                for label in (0, 1) for i in range(8)]
    return {'per_kind_precision': {'merge': .25, 'split': .25}, "policy_sha256": "parent", "cells": {
        f"1/{kind}": {"precision": .25, "tp": 25, "fp": 75, "effective_k": 100,
                      "training_examples": deepcopy(examples)} for kind in ("merge", "split")}}


class FeedbackTests(unittest.TestCase):
    def test_selection_comparison_exposes_transfer_loss_without_changing_objective(self):
        parent = {'objective': 'auroc', 'aggregation': 'positive-weighted',
                  'per_kind_objective': {'merge': .8}, 'holdout_objective': .9, 'holdout_precision': .3,
                  'cells': {'a/merge': {'auroc': .78, 'precision': .2, 'positives': 90, 'in_mean': True},
                            'b/merge': {'auroc': .9, 'precision': .3, 'positives': 10, 'in_mean': True},
                            'c/merge': {'auroc': .7, 'precision': .1, 'positives': 5, 'in_mean': False}},
                  'holdout_cells': ['b/merge']}
        candidate = deepcopy(parent)
        candidate['per_kind_objective']['merge'] = .81
        candidate['holdout_objective'] = .89
        candidate['holdout_precision'] = .25
        candidate['cells']['b/merge']['auroc'] = .89
        comparison = feedback.selection_comparison(parent, candidate, 'merge')
        self.assertAlmostEqual(comparison['overall']['delta'], .01)
        self.assertAlmostEqual(comparison['holdout']['delta'], -.01)
        self.assertAlmostEqual(comparison['holdout_precision']['delta'], -.05)
        self.assertTrue(comparison['warnings'])
        self.assertTrue(comparison['diagnostic_only'])
        self.assertEqual(comparison['per_brain']['a/merge']['counted_positive_fraction'], .9)
        self.assertEqual(comparison['per_brain']['c/merge']['counted_positive_fraction'], 0.)
        self.assertTrue(comparison['per_brain']['b/merge']['whole_brain_holdout'])
        from proofreader_evolve.harness.train_experiments import TrainingExperiments
        def attempt(value, diagnostic):
            return {'report': {}, 'component': 'code',
                    'entry': {'target_objective': value, 'target_precision': .2, 'sequence': 1,
                              'selection_comparison': diagnostic}}
        worse_transfer = attempt(.81, comparison)
        self.assertIs(TrainingExperiments._best([attempt(.8, {}), worse_transfer]), worse_transfer)

    def test_scalar_holdout_zero_unknown_legacy_and_kind_mapping(self):
        parent = report(1)
        parent.update(holdout_objective=.4, holdout_precision=.2)
        for value in (0., None):
            with self.subTest(value=value):
                candidate = {**parent, 'holdout_objective': value, 'holdout_precision': .1}
                comparison = feedback.selection_comparison(parent, candidate, 'merge')
                self.assertEqual(comparison['holdout']['candidate'], value)
                self.assertEqual(comparison['holdout']['delta'], -.4 if value == 0. else None)
        legacy = {**parent, 'holdout_precision': 0.}
        del legacy['holdout_objective']
        self.assertEqual(feedback.selection_comparison(parent, legacy, 'merge')['holdout']['candidate'], 0.)
        mapped = {**parent, 'holdout_objective': {'merge': .5}, 'holdout_precision': {'merge': 0.}}
        comparison = feedback.selection_comparison(parent, mapped, 'merge')
        self.assertAlmostEqual(comparison['holdout']['delta'], .1)
        self.assertEqual(comparison['holdout_precision']['candidate'], 0.)

    def test_missing_holdout_stays_missing_and_comparison_is_written_to_feedback(self):
        parent = report(1)
        comparison = feedback.selection_comparison(parent, parent, 'merge')
        self.assertIsNone(comparison['holdout']['delta'])
        self.assertEqual(comparison['warnings'], [])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'feedback.json'
            feedback.write_train_feedback(path, parent, [], {}, comparison=comparison)
            self.assertEqual(json.loads(path.read_text())['selection_comparison'], comparison)

    def test_compact_feedback_preserves_columns_alignment_and_full_reports(self):
        parent = report()
        original = deepcopy(parent)
        history = [{"generation": i, "accepted": False, "train": parent,
                    "validation": {"secret": "do not include"},
                    "reason": "VALIDATION: do not include", "reviser": {"summary": "do not include"},
                    "train_gate": {"passed": True, "reason": "Precision improved at fixed K"}}
                   for i in range(1, 8)]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "feedback.json"
            info = feedback.write_train_feedback(path, parent, history, {"merge": 100, "split": 100})
            text = path.read_text()
            data = json.loads(text)
            self.assertLessEqual(path.stat().st_size, feedback.MAX_FEEDBACK_BYTES)
            self.assertEqual(info["bytes"], path.stat().st_size)
        self.assertNotIn("do not include", text)
        self.assertNotIn('"validation"', text)
        self.assertNotIn('"training_examples"', text)
        self.assertEqual(len(data["attempts"]), 5)
        self.assertEqual(data["history_attempts_omitted"], 2)
        cell = data["examples"]["1/merge"]
        self.assertEqual(len(cell["features"]), 100)
        self.assertEqual(set(cell["labels"]), {0, 1})
        self.assertEqual(cell["labels"].count(0), cell["labels"].count(1))
        self.assertEqual(cell["scores"], [.123457] * cell["included"])
        self.assertTrue(all(len(v) == cell["included"] for v in cell["features"].values()))
        self.assertEqual(cell["features"]["feature_0_descriptive_name"], [None] * cell["included"])
        self.assertEqual(cell["features"]["feature_1_descriptive_name"][0], 1)
        self.assertEqual(parent, original)

    def test_missing_examples_and_failed_train_are_explicit(self):
        parent = report(1)
        parent["cells"]["1/merge"]["training_examples"] = []
        history = [{"generation": 1, "accepted": False, "train": None,
                    "reason": "some private failure"}]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "feedback.json"
            feedback.write_train_feedback(path, parent, history, {})
            result = json.loads(path.read_text())
        self.assertEqual(result["examples"]["1/merge"]["included"], 0)
        self.assertIsNone(result["attempts"][0]["train"])
        self.assertIsNone(result["attempts"][0]["train_gate"])

    def test_oversized_feedback_fails_explicitly_without_partial_write(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(feedback, "MAX_FEEDBACK_BYTES", 100):
            path = Path(tmp) / "feedback.json"
            with self.assertRaisesRegex(ValueError, "read budget"):
                feedback.write_train_feedback(path, report(), [], {})
            self.assertFalse(path.exists())

    def test_cells_with_the_fewest_positives_lose_their_examples_before_failing(self):
        # 2026-10-27: five selection cells x about 290 frozen feature columns exceeded the budget at one
        # example per group, so every training experiment of the frozen run failed. Whole cells now lose
        # their examples, fewest GT positives first, and keep their metrics and ranking_delta.
        parent = report(40)
        parent["cells"]["1/merge"]["positives"] = 50
        parent["cells"]["1/split"]["positives"] = 5
        for cell in parent["cells"].values():
            examples = cell["training_examples"]
            cell["training_examples"] = [examples[0], examples[8]]  # one example per group
            for example in cell["training_examples"]:
                example["features"]["feature_0_descriptive_name"] = .5  # nothing to prune
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "feedback.json"
            full = feedback.write_train_feedback(path, parent, [], {})
            self.assertEqual(full["examples_omitted_cells"], [])
            self.assertEqual(sorted(json.loads(path.read_text())["examples"]), ["1/merge", "1/split"])
            # The per-group variants differ by one byte only ("columns_pruned": false/true), so the budget
            # has to sit clearly below them to force a cell out.
            with patch.object(feedback, "MAX_FEEDBACK_BYTES", full["bytes"] - 100):
                info = feedback.write_train_feedback(path, parent, [], {})
                data = json.loads(path.read_text())
            self.assertEqual(info["examples_omitted_cells"], ["1/split"])
            self.assertEqual(data["examples_omitted_cells"], ["1/split"])
            self.assertEqual(list(data["examples"]), ["1/merge"])
            self.assertEqual(data["examples"]["1/merge"]["included"], 2)
            self.assertIn("1/split", data["parent_train"]["cells"])
            self.assertIn("1/split", data["ranking_delta"])
            self.assertIn("examples_omitted_cells", data["reading_guide"])
            self.assertLessEqual(info["bytes"], full["bytes"] - 100)
            # A single cell that still does not fit fails explicitly, as before.
            with patch.object(feedback, "MAX_FEEDBACK_BYTES", info["bytes"] - 100), \
                    self.assertRaisesRegex(ValueError, "read budget"):
                feedback.write_train_feedback(path, parent, [], {})
            # The target kind restricts the candidates before any cell is dropped.
            with patch.object(feedback, "MAX_FEEDBACK_BYTES", info["bytes"]):
                only = feedback.write_train_feedback(path, parent, [], {}, target_kind="split")
            self.assertEqual(only["examples_omitted_cells"], [])
            self.assertEqual(list(json.loads(path.read_text())["examples"]), ["1/split"])

    def test_reviser_turn_budget_is_configurable_and_forwarded(self):
        self.assertEqual(driver.parse_args(['--min-positives', '0', '--selection-protocol', 'in_sample']).reviser_max_turns, 60)
        with self.assertRaises(SystemExit):
            driver.parse_args(['--min-positives', '0', '--selection-protocol', 'in_sample', '--promotion-gate', 'margin', '--kind-schedule', 'alternate', "--reviser-max-turns", "0"])
        with tempfile.TemporaryDirectory() as tmp:
            args = driver.parse_args(['--min-positives', '0', '--selection-protocol', 'in_sample', '--promotion-gate', 'margin', '--kind-schedule', 'alternate', "--reviser-max-turns", "37", "--runs-dir", tmp])

            async def invoke(args, revise_fn, run_dir, trace):
                await revise_fn(run_dir, "policy", "rules", "report", "model")

            with patch.object(driver, "_run", side_effect=invoke), \
                    patch.object(driver, "revise", new_callable=AsyncMock) as revise:
                asyncio.run(driver.run(args))
            self.assertEqual(revise.call_args.kwargs["max_turns"], 37)


if __name__ == "__main__":
    unittest.main()
