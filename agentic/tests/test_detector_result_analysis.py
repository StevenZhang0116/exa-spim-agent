from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from agentic import run_detector_result_analysis as analysis


class DetectorResultAnalysisTests(unittest.TestCase):
    def setUp(self) -> None:
        self.old_project_root = analysis.PROJECT_ROOT
        self.old_application_root = analysis.APPLICATION_ROOT
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        analysis.PROJECT_ROOT = self.root
        analysis.APPLICATION_ROOT = self.root / "autodiscovery-application"

    def tearDown(self) -> None:
        analysis.PROJECT_ROOT = self.old_project_root
        analysis.APPLICATION_ROOT = self.old_application_root
        self.temporary.cleanup()

    def make_completed_result(self) -> tuple[Path, Path]:
        application = analysis.APPLICATION_ROOT / "example-run"
        result = application / "runs" / "completed"
        figures = result / "figures"
        figures.mkdir(parents=True)
        (application / "merge_site_detector.py").write_text("# detector\n")

        model = {
            "train_brain": "123",
            "train_mcl": 100,
            "scope": "full",
            "audit": {
                "n_segments": 4,
                "n_merge": 2,
                "prevalence": 0.5,
                "all_undefined_count": 1,
                "all_undefined_merges": 0,
            },
            "final_winner": "logistic_l2",
            "final_params": {"C": 1.0},
            "per_family_outer_metrics": {
                "logistic_l2": {
                    "oof_average_precision": 0.75,
                    "oof_roc_auc": 0.8,
                }
            },
            "selector_metrics": {"oof_average_precision": 0.75},
            "selection_frequency": {"logistic_l2": 5},
            "hypothesis_selection": None,
            "skipped_candidates": [],
            "dependency_versions": {"python": "test"},
            "seeds": {"random_seed": 42},
            "heldout": None,
            "feature_order": ["feature_a"],
            "all_feature_order": ["feature_a"],
            "train_pkl": "cache/example.pkl",
        }
        (result / "model_selection_123.json").write_text(json.dumps(model))
        (result / "merge_detector_123.csv").write_text(
            "segment_id,is_merge,merge_probability_oof\n"
            "a,1,0.9\n"
            "b,0,0.8\n"
            "c,1,0.7\n"
            "d,0,0.1\n"
        )
        (result / "merge_detector_123.joblib").write_bytes(b"saved-pipeline")
        (result / "merge_detector_123.log.txt").write_text(
            "# argv: detector example.pkl\n# OK after 1.0s\n"
        )
        for number in range(1, 8):
            (figures / f"123_full_{number:02d}_figure.png").write_bytes(
                b"not-needed-for-deterministic-test"
            )
        return application, result

    def test_resolve_accepts_nested_result_and_finds_application(self) -> None:
        application, result = self.make_completed_result()
        resolved_result, resolved_application = analysis.resolve_result_dir(result)
        self.assertEqual(resolved_result, result.resolve())
        self.assertEqual(resolved_application, application.resolve())

    def test_collect_evidence_is_grounded_in_json_csv_log_and_figures(self) -> None:
        application, result = self.make_completed_result()
        evidence = analysis.collect_evidence(result, application)
        self.assertEqual(evidence["brain"], "123")
        self.assertEqual(evidence["csv_summary"]["n_rows"], 4)
        self.assertEqual(evidence["csv_summary"]["n_merge"], 2)
        self.assertEqual(
            evidence["csv_summary"]["review_workload"][0]["merges_found"], 2
        )
        self.assertEqual(len(evidence["figures"]), 7)
        self.assertTrue(evidence["run_complete"])
        artifact_paths = {row["path"] for row in evidence["artifacts"]}
        self.assertTrue(any(path.endswith("merge_detector_123.joblib") for path in artifact_paths))
        self.assertTrue(any(path.endswith("merge_site_detector.py") for path in artifact_paths))

    def test_collect_evidence_rejects_incomplete_log(self) -> None:
        application, result = self.make_completed_result()
        (result / "merge_detector_123.log.txt").write_text("still running\n")
        with self.assertRaisesRegex(SystemExit, "not complete"):
            analysis.collect_evidence(result, application)

    def test_report_validation_preserves_evidence_and_mentions_every_figure(self) -> None:
        application, result = self.make_completed_result()
        evidence = analysis.collect_evidence(result, application)
        evidence_path = result / analysis.EVIDENCE_NAME
        analysis.write_json_atomic(evidence_path, evidence)
        evidence_sha = analysis.sha256(evidence_path)
        report_path = result / analysis.REPORT_NAME
        skeleton = analysis.write_report_skeleton(report_path, evidence, evidence_sha)
        skeleton_sha = hashlib.sha256(skeleton.encode()).hexdigest()
        expected_block = analysis.driver_block(report_path)
        self.assertLess(
            skeleton.index(analysis.ENGLISH_REPORT_HEADING),
            skeleton.index(analysis.CHINESE_REPORT_HEADING),
        )

        with self.assertRaisesRegex(SystemExit, "skeleton unchanged"):
            analysis.validate_agent_report(
                report_path,
                skeleton_sha=skeleton_sha,
                expected_block=expected_block,
                evidence_sha=evidence_sha,
                figure_basenames=[row["basename"] for row in evidence["figures"]],
            )

        english = "\n\n".join(
            f"{heading}\nGrounded interpretation."
            for heading in analysis.ENGLISH_SECTION_HEADINGS
        )
        english += "\n" + "\n".join(
            f"- {figure['basename']}: inspected."
            for figure in evidence["figures"]
        )
        chinese = "\n\n".join(
            f"{heading}\n基于证据的对应翻译。"
            for heading in analysis.CHINESE_SECTION_HEADINGS
        )
        chinese += "\n" + "\n".join(
            f"- {figure['basename']}：已检查。"
            for figure in evidence["figures"]
        )
        completed = skeleton.replace(analysis.REPORT_PLACEHOLDER, english, 1)
        completed = completed.replace(analysis.REPORT_PLACEHOLDER, chinese, 1)
        report_path.write_text(completed)
        analysis.validate_agent_report(
            report_path,
            skeleton_sha=skeleton_sha,
            expected_block=expected_block,
            evidence_sha=evidence_sha,
            figure_basenames=[row["basename"] for row in evidence["figures"]],
        )

        report_path.write_text(completed.replace(
            f"- {evidence['figures'][0]['basename']}：已检查。", ""
        ))
        with self.assertRaisesRegex(SystemExit, "both languages"):
            analysis.validate_agent_report(
                report_path,
                skeleton_sha=skeleton_sha,
                expected_block=expected_block,
                evidence_sha=evidence_sha,
                figure_basenames=[row["basename"] for row in evidence["figures"]],
            )


if __name__ == "__main__":
    unittest.main()
