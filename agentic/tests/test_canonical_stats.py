from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_module():
    path = PROJECT_ROOT / "agentic" / "canonical_stats.py"
    spec = importlib.util.spec_from_file_location("test_canonical_stats_module", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


cs = _load_module()


def _merge_companion(n: int = 600, seed: int = 1) -> dict:
    """A small merge-like companion: labels on a third of the rows, 3 folds."""
    rng = np.random.default_rng(seed)
    label = np.full(n, np.nan)
    labeled = rng.choice(n, size=n // 3, replace=False)
    label[labeled] = (rng.random(len(labeled)) < 0.2).astype(float)
    fold = rng.integers(0, 3, size=n).astype(np.int8)
    comp = {
        "kind": "merge",
        "row": np.arange(n),
        "label": label,
        "fold": fold,
        "segment_id": rng.integers(0, 40, size=n),
        "node_id": np.arange(n),
        "degree": rng.choice([2, 3, 3, 3, 4], size=n).astype(np.int16),
    }
    for f in range(3):
        comp[f"fit_mask_fold{f}"] = fold != f
    return comp


def _write_npz(path: Path, comp: dict) -> None:
    np.savez(path, **{k: v for k, v in comp.items() if k != "kind"})


class CanonicalStatsTests(unittest.TestCase):
    def test_average_precision_matches_sklearn_without_ties_and_collapses_ties(self) -> None:
        from sklearn.metrics import average_precision_score

        comp = _merge_companion()
        rng = np.random.default_rng(2)
        scores = rng.random(len(comp["label"]))
        rows = ~np.isnan(comp["label"])
        self.assertAlmostEqual(
            cs.average_precision(comp, scores),
            average_precision_score(comp["label"][rows] == 1, scores[rows]),
            places=12,
        )
        # A constant score ties every row: AP equals the positive rate, whatever
        # the row order inside the tie (sklearn agrees).
        constant = np.zeros(len(scores))
        rate = float((comp["label"][rows] == 1).mean())
        self.assertAlmostEqual(cs.average_precision(comp, constant), rate, places=12)
        self.assertAlmostEqual(
            average_precision_score(comp["label"][rows] == 1, constant[rows]), rate, places=12
        )

    def test_precision_at_fraction_breaks_ties_by_key_and_returns_counts(self) -> None:
        comp = _merge_companion()
        keys = cs.candidate_keys(comp)
        self.assertTrue(keys[0].startswith("merge:"))
        hits, k = cs.precision_at_fraction(comp, np.zeros(len(keys)), 0.1, keys, counts=True)
        self.assertEqual(k, sum(
            max(1, int(round(0.1 * int(((comp["fold"] == f) & ~np.isnan(comp["label"])).sum()))))
            for f in range(3)
        ))
        self.assertAlmostEqual(
            cs.precision_at_fraction(comp, np.zeros(len(keys)), 0.1, keys), hits / k
        )

    def test_baseline_is_the_raw_ranking_of_the_kind(self) -> None:
        comp = _merge_companion()
        name, scores = cs.baseline(comp)
        self.assertEqual(name, "degree")
        np.testing.assert_array_equal(scores, comp["degree"].astype(float))
        split = {"kind": "split", "gap_um": np.array([5.0, 1.0, 3.0])}
        name, scores = cs.baseline(split)
        self.assertEqual(name, "gap")
        np.testing.assert_array_equal(scores, [-5.0, -1.0, -3.0])

    def test_finite_ranks_nan_last(self) -> None:
        s = cs.finite([1.0, np.nan, 3.0, np.inf])
        self.assertEqual(list(np.argsort(-s)), [2, 0, 1, 3] if s[3] == -np.inf else [3, 2, 0, 1])
        self.assertTrue(np.isneginf(s[1]))

    def test_paired_bootstrap_is_zero_for_identical_scores_and_positive_for_a_gain(self) -> None:
        comp = _merge_companion()
        _, base = cs.baseline(comp)
        same = cs.paired_bootstrap_delta_ap(comp, base, base, n_draws=200, seed=3)
        self.assertEqual(same["delta"], 0.0)
        self.assertEqual(same["ci95"], [0.0, 0.0])
        self.assertEqual(same["p_one_sided"], 1.0)
        # The labels themselves rank perfectly: a clear gain over degree.
        oracle = np.where(np.isnan(comp["label"]), 0.0, comp["label"])
        gain = cs.paired_bootstrap_delta_ap(comp, oracle, base, n_draws=200, seed=3)
        self.assertGreater(gain["ci95"][0], 0.0)
        self.assertEqual(gain["p_one_sided"], 0.0)
        self.assertEqual(gain["n_draws"], 200)
        # Same seed, same draws.
        again = cs.paired_bootstrap_delta_ap(comp, oracle, base, n_draws=200, seed=3)
        self.assertEqual(again["mean"], gain["mean"])

    def test_tie_break_null_floor_of_a_tied_baseline(self) -> None:
        comp = _merge_companion()
        name, base = cs.baseline(comp)
        base_ap = cs.average_precision(comp, base)
        null = cs.tie_break_null(comp, base, base_ap, n_draws=50)
        # the tie-aware AP of a heavily tied baseline sits near the floor of its random tie-breaks
        self.assertGreaterEqual(null["share_null_ge_model"], 0.5)
        self.assertGreaterEqual(null["null_mean"], base_ap - 1e-9)
        rows = ~np.isnan(comp["label"])
        oracle = np.where(np.isnan(comp["label"]), 0.0, comp["label"])
        perfect = base + 0.1 * oracle  # breaks degree's ties with the answer key
        self.assertEqual(cs.tie_break_null(comp, base, cs.average_precision(comp, perfect),
                                           n_draws=50)["share_null_ge_model"], 0.0)
        self.assertEqual(null["n_draws"], 50)
        del rows

    def test_report_prints_a_parseable_canonical_block(self) -> None:
        comp = _merge_companion()
        oracle = np.where(np.isnan(comp["label"]), 0.0, comp["label"])
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            result = cs.report(comp, oracle, name="oracle", n_draws=50, seed=0)
        lines = out.getvalue().splitlines()
        self.assertTrue(all(line.startswith("CANONICAL ") for line in lines))
        self.assertIn(f"CANONICAL ap model={result['ap']:.5f} "
                      f"baseline={result['baseline_ap']:.5f}", lines[1])
        self.assertTrue(any(line.startswith("CANONICAL fold 0 ") for line in lines))
        self.assertTrue(any(line.startswith("CANONICAL precision@0.01 ") for line in lines))
        self.assertTrue(any("ci95=[" in line and "draws=50" in line for line in lines))
        self.assertTrue(any(line.startswith("CANONICAL labeled_share@0.01 ") for line in lines))
        self.assertTrue(any(line.startswith("CANONICAL label_shift ks=") for line in lines))
        self.assertGreaterEqual(result["label_shift"]["ks"], 0.0)
        self.assertEqual(result["ap"], 1.0)

    def test_fit_and_held_rows_follow_the_purged_masks(self) -> None:
        comp = _merge_companion()
        fit = cs.fit_rows(comp, 1)
        self.assertTrue(np.all(comp["fit_mask_fold1"][fit]))
        self.assertFalse(np.any(np.isnan(comp["label"][fit])))
        held = cs.held_rows(comp, 1)
        self.assertTrue(np.all(comp["fold"][held] == 1))

    def test_baseline_card_from_an_npz_and_its_text(self) -> None:
        comp = _merge_companion()
        with tempfile.TemporaryDirectory() as tmpdir:
            npz = Path(tmpdir) / "rerank_companion_test_merge.npz"
            _write_npz(npz, comp)
            npz.with_suffix(".json").write_text(json.dumps(
                {"brain": "test", "candidate_policy": "junction|k=2"}))
            card = cs.baseline_card(npz)
            proc = subprocess.run(
                [sys.executable, str(PROJECT_ROOT / "agentic" / "canonical_stats.py"),
                 "card", str(npz)], capture_output=True, text=True, timeout=120,
            )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(json.loads(proc.stdout)["baseline_ap"], card["baseline_ap"])
        self.assertEqual(card["kind"], "merge")
        self.assertEqual(card["baseline"], "degree")
        self.assertEqual(card["brain"], "test")
        self.assertEqual(card["n_labeled"], int((~np.isnan(comp["label"])).sum()))
        self.assertAlmostEqual(card["baseline_ap"],
                               cs.average_precision(comp, comp["degree"].astype(float)))
        text = cs.card_text(card)
        self.assertIn("BASELINE CARD (rerank_companion_test_merge.npz, brain test, merge)", text)
        self.assertIn("node degree, higher first", text)
        self.assertIn(f"{card['baseline_ap']:.4f}", text)
        self.assertIn("agentic/canonical_stats.py", text)


if __name__ == "__main__":
    unittest.main()
