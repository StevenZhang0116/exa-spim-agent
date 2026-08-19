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

    def test_collect_evidence_supports_split_candidate_rows(self) -> None:
        application, result = self.make_completed_result()
        (application / "merge_site_detector.py").unlink()
        (application / "split_site_detector.py").write_text("# split detector\n")
        model_path = result / "model_selection_123.json"
        model = json.loads(model_path.read_text())
        model.update({
            "detector_target": "split_detection",
            "row_unit": "candidate segment pair",
            "label_name": "is_split",
            "score_prefix": "split_probability",
            "audit": {
                "n_rows": 4,
                "n_positive": 2,
                "n_negative": 2,
                "all_undefined_count": 0,
                "all_undefined_positives": 0,
            },
            "sample_universe_audit": {
                "n_reachable_truth_pairs": 3,
                "n_candidate_truth_pairs": 2,
                "candidate_recall_of_reachable": 2 / 3,
            },
        })
        model_path.write_text(json.dumps(model))
        (result / "merge_detector_123.csv").unlink()
        (result / "merge_detector_123.joblib").rename(
            result / "split_detector_123.joblib")
        (result / "merge_detector_123.log.txt").rename(
            result / "split_detector_123.log.txt")
        (result / "split_detector_123.csv").write_text(
            "candidate_id,segment_id_a,segment_id_b,is_split,"
            "is_merge_creating,split_probability_oof\n"
            "0,1,2,1,0,0.9\n"
            "1,3,4,0,1,0.8\n"
            "2,5,6,1,0,0.7\n"
            "3,7,8,0,0,\n"
        )

        evidence = analysis.collect_evidence(result, application)

        self.assertEqual(evidence["detector_target"], "split_detection")
        self.assertEqual(evidence["csv_summary"]["n_positive"], 2)
        self.assertEqual(evidence["csv_summary"]["n_oof_scored"], 3)
        self.assertEqual(evidence["csv_summary"]["n_oof_unscored"], 1)
        self.assertEqual(evidence["csv_summary"]["oof_coverage"], 0.75)
        self.assertEqual(
            evidence["csv_summary"]["review_workload"][0]["recall_scope"],
            "oof_scored_rows",
        )
        self.assertEqual(
            evidence["csv_summary"]["review_workload"][0]["merge_creating_joins"],
            1,
        )
        self.assertTrue(any(
            row["path"].endswith("split_site_detector.py")
            for row in evidence["artifacts"]
        ))

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
        # The heading contract is now embedded in the skeleton itself: every
        # required heading is pre-written, one placeholder per section, so the
        # analyst never reproduces (or improvises) a title.
        for heading in (*analysis.ENGLISH_SECTION_HEADINGS,
                        *analysis.CHINESE_SECTION_HEADINGS):
            self.assertIn(heading, skeleton)
        self.assertEqual(
            skeleton.count(analysis.REPORT_PLACEHOLDER),
            len(analysis.ENGLISH_SECTION_HEADINGS)
            + len(analysis.CHINESE_SECTION_HEADINGS),
        )

        with self.assertRaisesRegex(SystemExit, "skeleton unchanged"):
            analysis.validate_agent_report(
                report_path,
                skeleton_sha=skeleton_sha,
                expected_block=expected_block,
                evidence_sha=evidence_sha,
                figure_basenames=[row["basename"] for row in evidence["figures"]],
            )

        figures_english = "\n".join(
            f"- {figure['basename']}: inspected."
            for figure in evidence["figures"]
        )
        figures_chinese = "\n".join(
            f"- {figure['basename']}：已检查。"
            for figure in evidence["figures"]
        )

        def fill(part: str, section_texts: list[str]) -> str:
            for text in section_texts:
                part = part.replace(analysis.REPORT_PLACEHOLDER, text, 1)
            return part

        n_en = len(analysis.ENGLISH_SECTION_HEADINGS)
        n_zh = len(analysis.CHINESE_SECTION_HEADINGS)
        english_part, chinese_part = skeleton.split(analysis.CHINESE_REPORT_HEADING)
        completed = (
            fill(english_part,
                 ["Grounded interpretation."] * (n_en - 1) + [figures_english])
            + analysis.CHINESE_REPORT_HEADING
            + fill(chinese_part,
                   ["基于证据的对应翻译。"] * (n_zh - 1) + [figures_chinese])
        )
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

        # A number present in only one language part is rejected; carrying it
        # verbatim into the other part passes again.
        numbered = completed.replace(
            "Grounded interpretation.", "Grounded interpretation (AP 0.0736).", 1
        )
        report_path.write_text(numbered)
        with self.assertRaisesRegex(SystemExit, "numeric mismatch"):
            analysis.validate_agent_report(
                report_path,
                skeleton_sha=skeleton_sha,
                expected_block=expected_block,
                evidence_sha=evidence_sha,
                figure_basenames=[row["basename"] for row in evidence["figures"]],
            )
        report_path.write_text(numbered.replace(
            "基于证据的对应翻译。", "基于证据的对应翻译（AP 0.0736）。", 1
        ))
        analysis.validate_agent_report(
            report_path,
            skeleton_sha=skeleton_sha,
            expected_block=expected_block,
            evidence_sha=evidence_sha,
            figure_basenames=[row["basename"] for row in evidence["figures"]],
        )

    def test_result_contract_derives_from_target_spec_and_matches_runtime(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        source = (
            repo_root / "agentic" / "detector_build" / "target_runtime.py"
        ).read_text(encoding="utf-8")
        for target, prefix in (
            ("merge_detection", "merge_detector"),
            ("split_detection", "split_detector"),
        ):
            contract = analysis._result_contract({"detector_target": target})
            self.assertEqual(contract["output_prefix"], prefix)
            self.assertEqual(contract["score"], contract["output_prefix"].split("_")[0] + "_probability_oof")
            for column in contract["identity"]:
                self.assertIn(f'"{column}"', source, column)
        # contracts resolves legacy unnamed runs to the merge vocabulary.
        legacy = analysis._result_contract({"detector_target": "legacy_unspecified"})
        self.assertEqual(legacy["output_prefix"], "merge_detector")
        with self.assertRaisesRegex(SystemExit, "Unsupported detector_target"):
            analysis._result_contract({"detector_target": "omit_detection"})


if __name__ == "__main__":
    unittest.main()
