from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from agentic.detector_build.candidate_policy import (
    APPLICATION_STATUS,
    OBJECTIVE_NAME,
    compile_candidate_policy,
    validate_candidate_policy_sources,
    validate_frozen_candidate_policy,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class CandidatePolicyTest(unittest.TestCase):
    def _write_bundle(self, root: Path) -> Path:
        tables = root / "tables"
        tables.mkdir(parents=True)
        aggregate = tables / "cross_dataset_aggregate.csv"
        fieldnames = [
            "config_id", "mode", "radius_um", "per_anchor_k", "global_cap",
            "datasets", "total_candidates", "min_recall", "mean_recall",
            "micro_recall", "min_direct_recall", "min_gap_recall",
            "mean_prevalence",
        ]
        rows = [
            {
                "config_id": "tip_to_any_node|r=45|k=2",
                "mode": "tip_to_any_node", "radius_um": "45",
                "per_anchor_k": "2", "global_cap": "none", "datasets": "2",
                "total_candidates": "200", "min_recall": "0.89",
                "mean_recall": "0.94", "micro_recall": "0.93",
                "min_direct_recall": "0.98", "min_gap_recall": "0.80",
                "mean_prevalence": "0.02",
            },
            {
                "config_id": "tip_to_any_node|r=50|k=2",
                "mode": "tip_to_any_node", "radius_um": "50",
                "per_anchor_k": "2", "global_cap": "none", "datasets": "2",
                "total_candidates": "300", "min_recall": "0.91",
                "mean_recall": "0.95", "micro_recall": "0.94",
                "min_direct_recall": "0.99", "min_gap_recall": "0.84",
                "mean_prevalence": "0.018",
            },
            {
                "config_id": "tip_to_any_node|r=55|k=2",
                "mode": "tip_to_any_node", "radius_um": "55",
                "per_anchor_k": "2", "global_cap": "none", "datasets": "2",
                "total_candidates": "400", "min_recall": "0.93",
                "mean_recall": "0.96", "micro_recall": "0.95",
                "min_direct_recall": "0.995", "min_gap_recall": "0.86",
                "mean_prevalence": "0.016",
            },
        ]
        with aggregate.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        evidence = root / "ai_review_evidence.json"
        evidence.write_text(json.dumps({
            "evidence_schema": 1,
            "run": {
                "result_status": "complete",
                "mcl": 100,
                "config_signature": "fixture",
                "brains": ["brain-a", "brain-b"],
                "evaluated_pairing_rules": ["tip_to_any_node", "tip_to_tip"],
                "any_node_to_any_node_evaluated": False,
                "global_cap_swept": False,
                "pre_review_manifest_integrity": {"status": "pass"},
            },
            "source_files": {
                "aggregate": {
                    "path": "tables/cross_dataset_aggregate.csv",
                    "sha256": _sha256(aggregate),
                }
            },
        }, indent=2) + "\n", encoding="utf-8")

        review = root / "AI_REVIEW.md"
        review.write_text(
            "# AI Review\n\n"
            "Evidence file: `ai_review_evidence.json` "
            f"(`sha256={_sha256(evidence)}`)\n",
            encoding="utf-8",
        )
        manifest = root / "artifact_manifest.json"
        manifest.write_text(json.dumps({
            "artifacts": [
                {"path": "AI_REVIEW.md", "sha256": _sha256(review)},
                {"path": "ai_review_evidence.json", "sha256": _sha256(evidence)},
                {
                    "path": "tables/cross_dataset_aggregate.csv",
                    "sha256": _sha256(aggregate),
                },
            ]
        }, indent=2) + "\n", encoding="utf-8")
        return review

    @staticmethod
    def _write_advice(path: Path, selected_config_id: str) -> None:
        path.write_text(json.dumps({
            "schema_version": 1,
            "objective": {
                "name": OBJECTIVE_NAME,
                "minimum_worst_brain_recall": 0.90,
                "tie_breaker": ["total_candidates", "config_id"],
            },
            "selected_config_id": selected_config_id,
            "explanation": "This is the smallest eligible candidate pool.",
        }), encoding="utf-8")

    def test_driver_selects_and_freezes_minimum_eligible_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            review = self._write_bundle(root)
            advice = root / ".candidate_policy_advice.json"
            output = root / "split_candidate_policy.json"
            self._write_advice(advice, "tip_to_any_node|r=50|k=2")

            sources = validate_candidate_policy_sources(
                review, minimum_recall=0.90, expected_mcl=100,
                project_root=root,
            )
            self.assertEqual(
                sources["selected"]["config_id"],
                "tip_to_any_node|r=50|k=2",
            )
            policy = compile_candidate_policy(
                advice, output, review, minimum_recall=0.90,
                expected_mcl=100, project_root=root,
            )
            self.assertEqual(policy["application_status"], APPLICATION_STATUS)
            self.assertEqual(policy["selected_policy"], {
                "config_id": "tip_to_any_node|r=50|k=2",
                "mode": "tip_to_any_node",
                "radius_um": 50,
                "per_anchor_k": 2,
                "global_cap": None,
            })
            self.assertEqual(policy["evidence_metrics"]["total_candidates"], 300)
            validate_frozen_candidate_policy(
                output, review, minimum_recall=0.90, expected_mcl=100,
                project_root=root,
            )

    def test_driver_rejects_agent_disagreement(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            review = self._write_bundle(root)
            advice = root / ".candidate_policy_advice.json"
            self._write_advice(advice, "tip_to_any_node|r=55|k=2")

            with self.assertRaisesRegex(SystemExit, "disagrees"):
                compile_candidate_policy(
                    advice, root / "policy.json", review, minimum_recall=0.90,
                    expected_mcl=100, project_root=root,
                )

    def test_driver_rejects_mcl_mismatch_and_tampered_aggregate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            review = self._write_bundle(root)
            with self.assertRaisesRegex(SystemExit, "mcl100.*mcl10"):
                validate_candidate_policy_sources(
                    review, minimum_recall=0.90, expected_mcl=10,
                    project_root=root,
                )

            aggregate = root / "tables" / "cross_dataset_aggregate.csv"
            aggregate.write_text(
                aggregate.read_text(encoding="utf-8") + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(SystemExit, "hash mismatch"):
                validate_candidate_policy_sources(
                    review, minimum_recall=0.90, expected_mcl=100,
                    project_root=root,
                )


if __name__ == "__main__":
    unittest.main()
