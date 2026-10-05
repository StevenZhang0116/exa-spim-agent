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
    return {"macro_precision": .25, "policy_sha256": "parent", "cells": {
        f"1/{kind}": {"precision": .25, "tp": 25, "fp": 75, "effective_k": 100,
                      "training_examples": deepcopy(examples)} for kind in ("merge", "split")}}


class FeedbackTests(unittest.TestCase):
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

    def test_reviser_turn_budget_is_configurable_and_forwarded(self):
        self.assertEqual(driver.parse_args(['--selection-protocol', 'in_sample']).reviser_max_turns, 24)
        with self.assertRaises(SystemExit):
            driver.parse_args(['--selection-protocol', 'in_sample', "--reviser-max-turns", "0"])
        with tempfile.TemporaryDirectory() as tmp:
            args = driver.parse_args(['--selection-protocol', 'in_sample', "--reviser-max-turns", "37", "--runs-dir", tmp])

            async def invoke(args, revise_fn, run_dir, trace):
                await revise_fn(run_dir, "policy", "rules", "report", "model")

            with patch.object(driver, "_run", side_effect=invoke), \
                    patch.object(driver, "revise", new_callable=AsyncMock) as revise:
                asyncio.run(driver.run(args))
            self.assertEqual(revise.call_args.kwargs["max_turns"], 37)


if __name__ == "__main__":
    unittest.main()
