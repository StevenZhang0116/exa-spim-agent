"""Deterministic importance evidence and the short-report driver on a tiny synthetic run."""
from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

from agentic import detector_importance as di
from agentic import run_detector_importance_report as driver


def make_run(root: Path, *, n_rows: int = 240, n_judgeable: int = 160, seed: int = 3) -> tuple[Path, Path]:
    """A merge-site run directory with 3 features from 2 hypotheses and a fitted logistic winner."""
    import joblib
    import pandas as pd
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(seed)
    app = root / "fake-run"
    run = app / "runs" / "1-with-figures"
    run.mkdir(parents=True)
    features = ["h3_strong_signal", "h7_noise", "h7_sparse"]
    label = np.full(n_rows, np.nan)
    judged = rng.choice(n_rows, size=n_judgeable, replace=False)
    label[judged] = (rng.random(n_judgeable) < 0.3).astype(float)
    strong = rng.normal(size=n_rows) + 2.0 * np.nan_to_num(label, nan=0.3)
    noise = rng.normal(size=n_rows)
    sparse = np.where(rng.random(n_rows) < 0.6, np.nan, rng.normal(size=n_rows))
    degree = rng.integers(2, 5, size=n_rows).astype(float)
    frame = pd.DataFrame({
        "candidate_id": np.arange(n_rows), "node_id": np.arange(n_rows) + 100,
        "segment_id": rng.integers(1, 40, size=n_rows), "degree": degree,
        "is_merge_site": label,
        "h3_strong_signal": strong, "h3_strong_signal_is_defined": 1,
        "h7_noise": noise, "h7_noise_is_defined": 1,
        "h7_sparse": sparse, "h7_sparse_is_defined": np.isfinite(sparse).astype(int),
        "merge_site_probability_oof": np.where(np.isfinite(label), rng.random(n_rows), np.nan),
    })
    frame.to_csv(run / "merge_junction_detector_1.csv", index=False)
    pipe = Pipeline([("impute", SimpleImputer(strategy="median")),
                     ("scale", StandardScaler()),
                     ("clf", LogisticRegression(C=1.0, max_iter=500))])
    mask = np.isfinite(label)
    pipe.fit(frame.loc[mask, features].to_numpy(dtype=float), label[mask].astype(int))
    joblib.dump({"pipeline": pipe, "feature_order": features}, run / "merge_junction_detector_1.joblib")
    (run / "merge_junction_detector_1.log.txt").write_text("# merge run\n[detector] done\n# OK after 1.0s\n")
    (run / "model_selection_1.json").write_text(json.dumps({
        "detector_target": "merge_site_detection", "train_brain": "1", "train_mcl": 100,
        "final_winner": "logistic_l2", "final_params": {"C": 1.0},
        "per_family_outer_metrics": {"logistic_l2": {"oof_average_precision": 0.5, "oof_roc_auc": 0.7, "n_scored": n_judgeable}},
        "selector_metrics": {"oof_average_precision": 0.45},
        "feature_order": features,
        "audit": {"n_rows": n_rows, "n_positive": int(np.nansum(label)), "n_judgeable": n_judgeable},
        "heldout": {"test_brain": "2", "held_out_average_precision": 0.41, "held_out_roc_auc": 0.66,
                    "n_test_rows": 500, "n_test_judgeable": 90, "n_test_positive": 20},
    }))
    (app / "feature_inventory.json").write_text(json.dumps({"hypotheses": [
        {"id": 3, "included": True, "statistical_verdict": "SOUND", "post_correction_verdict": None,
         "features": [{"name": "h3_strong_signal", "quantity": "A strong synthetic signal."}]},
        {"id": 7, "included": True, "statistical_verdict": "MAJOR", "post_correction_verdict": "UPHELD",
         "features": [{"name": "h7_noise", "quantity": "Pure noise."},
                      {"name": "h7_sparse", "quantity": "Noise defined on 40% of rows."}]},
    ]}))
    (root / "fake-run.json").write_text(json.dumps([
        {"id": 3, "hypothesis": "Strong signal marks merges."},
        {"id": 7, "hypothesis": "Noise does nothing."},
    ]))
    return run, app


class ImportanceEvidenceTests(unittest.TestCase):
    def test_views_rank_the_signal_feature_and_group(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run, app = make_run(Path(tmp))
            with redirect_stdout(io.StringIO()):
                evidence = di.collect_importance_evidence(
                    run, app, permutation_repeats=2, run_json=Path(tmp) / "fake-run.json")
            json.dumps(evidence)  # serializable
            self.assertEqual(evidence["n_judgeable"], 160)
            self.assertEqual(evidence["baseline"]["column"], "degree")
            views = evidence["views"]
            self.assertEqual(views["coefficients"]["scale"], "standardized (pipeline scaler)")
            self.assertIsNone(views["native_importance"])
            self.assertEqual(views["permutation"]["repeats"], 2)
            ranked = evidence["ranked_features"]
            self.assertEqual(ranked["by_coefficient"][0]["feature"], "h3_strong_signal")
            self.assertEqual(ranked["by_permutation"][0]["feature"], "h3_strong_signal")
            self.assertEqual(ranked["by_univariate_ap"][0]["feature"], "h3_strong_signal")
            top = ranked["by_coefficient"][0]
            self.assertGreater(top["coefficient"], 0.0)
            self.assertGreater(top["ap"], evidence["baseline"]["ap"])
            self.assertEqual(top["quantity"], "A strong synthetic signal.")
            sparse = next(r for r in ranked["by_univariate_ap"] if r["feature"] == "h7_sparse")
            self.assertLess(sparse["defined_share"], 0.6)
            groups = evidence["ranked_groups"]
            self.assertEqual([g["hypothesis_id"] for g in groups], [3, 7])
            self.assertEqual(groups[0]["text"], "Strong signal marks merges.")
            self.assertEqual(groups[1]["n_features"], 2)
            self.assertEqual(groups[1]["post_correction_verdict"], "UPHELD")
            self.assertEqual(evidence["group_primary_view"], "sum_abs_coefficient")
            self.assertEqual(evidence["heldout"]["test_brain"], "2")
            markdown = di.evidence_markdown(evidence)
            self.assertIn("`h3_strong_signal`", markdown)
            self.assertIn("Hypothesis groups by `sum_abs_coefficient`", markdown)
            self.assertIn("Held-out `2`: AP 0.4100", markdown)

    def test_incomplete_log_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run, app = make_run(Path(tmp))
            (run / "merge_junction_detector_1.log.txt").write_text("# merge run\n[extract] progress\n")
            with self.assertRaisesRegex(SystemExit, "not complete"):
                di.collect_importance_evidence(run, app, permutation_repeats=1)

    def test_best_direction_ap_handles_undefined_rows(self) -> None:
        y = np.array([1, 0, 1, 0, 0])
        ap, direction = di.best_direction_ap(y, np.array([np.nan, 0.1, 0.9, 0.2, 0.3]))
        self.assertEqual(direction, "higher")
        self.assertGreater(ap, 0.5)
        ap_none, direction_none = di.best_direction_ap(y, np.full(5, np.nan))
        self.assertTrue(np.isnan(ap_none))
        self.assertEqual(direction_none, "undefined")


class ImportanceReportDriverTests(unittest.TestCase):
    def test_no_agent_run_writes_evidence_and_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run, app = make_run(Path(tmp))
            with redirect_stdout(io.StringIO()):
                code = driver.main([str(run), "--application-dir", str(app), "--no-agent",
                                    "--permutation-repeats", "1",
                                    "--run-json", str(Path(tmp) / "fake-run.json")])
            self.assertEqual(code, 0)
            evidence = json.loads((run / driver.EVIDENCE_NAME).read_text())
            self.assertEqual(evidence["ranked_groups"][0]["hypothesis_id"], 3)
            report = (run / driver.REPORT_NAME).read_text()
            self.assertIn(driver.BLOCK_START, report)
            self.assertNotIn(driver.PLACEHOLDER, report)
            self.assertIn(driver.NO_AGENT_TEXT, report)
            self.assertNotRegex(report, "[\u4e00-\u9fff]")
            log = (run / driver.WORKFLOW_LOG_NAME).read_text()
            self.assertIn("# OK after", log)

    def test_report_check_rejects_placeholders_chinese_length_and_block_edits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run, app = make_run(Path(tmp))
            with redirect_stdout(io.StringIO()):
                evidence = di.collect_importance_evidence(run, app, permutation_repeats=1)
            report = run / driver.REPORT_NAME
            skeleton = driver.write_skeleton(report, evidence, "abc123")
            skeleton_sha = driver.sha256(report)
            block = driver.driver_block(report)

            def check() -> None:
                driver.validate_report(report, skeleton_sha=skeleton_sha,
                                       expected_block=block, evidence_sha="abc123")

            with self.assertRaisesRegex(SystemExit, "skeleton unchanged"):
                check()
            sections = ["The winner reaches AP 0.5 against a baseline of 0.25.",
                        "`h3_strong_signal` measures a strong signal (coefficient 1.2).",
                        "Permutation is in-sample; refit without hypothesis 3 next."]

            def fill(bodies) -> None:
                text = skeleton
                for heading, body in zip(driver.SECTION_HEADINGS, bodies):
                    text = text.replace(f"{heading}\n\n{driver.PLACEHOLDER}", f"{heading}\n\n{body}")
                report.write_text(text, encoding="utf-8")

            fill(sections)
            check()
            fill(sections[:2] + [driver.PLACEHOLDER])
            with self.assertRaisesRegex(SystemExit, "placeholder"):
                check()
            fill(sections[:2] + ["Permutation is in-sample; 下一步 refit without hypothesis 3."])
            with self.assertRaisesRegex(SystemExit, "English only"):
                check()
            fill(sections[:2] + [" ".join(["word"] * (driver.MAX_WORDS + 5))])
            with self.assertRaisesRegex(SystemExit, "cap is"):
                check()
            fill(sections)
            report.write_text(report.read_text().replace(driver.BLOCK_START, driver.BLOCK_START + " edited"))
            with self.assertRaisesRegex(SystemExit, "immutable"):
                check()


if __name__ == "__main__":
    unittest.main()
