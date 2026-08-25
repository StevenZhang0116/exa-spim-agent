from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from agentic.split_feature_applicability import (
    compile_applicability,
    load_applicability,
)
from agentic.detector_build.inventory import compile_feature_inventory


class SplitFeatureApplicabilityTests(unittest.TestCase):
    def _tree(self, root: Path):
        run = root / "split-error-test.json"
        run.write_text("[]\n")
        rerun = root / "split-error-test.json.predictive.rerun"
        fixed = root / "split-error-test.json.predictive.fixed"
        rerun.mkdir()
        fixed.mkdir()
        (rerun / "hypo_1.py").write_text("RERUN = 1\n")
        (fixed / "hypo_1.py").write_text("FIXED = 1\n")
        draft = root / "draft.json"
        draft.write_text(json.dumps({
            "schema_version": 1,
            "records": [
                {
                    "id": 1,
                    "source": "rerun",
                    "node_role_requirement": "requires_both_tips",
                    "reason": "Compares two terminal directions.",
                },
                {
                    "id": 1,
                    "source": "fixed",
                    "node_role_requirement": "requires_tip_anchor",
                    "reason": "Corrected quantity accepts any partner node.",
                },
            ],
        }))
        return run, rerun, fixed, draft

    def test_compile_binds_exact_sources_and_loads(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run, rerun, fixed, draft = self._tree(root)
            output = root / "labels.json"
            compile_applicability(
                draft,
                output,
                run_json=run,
                selected_ids=[1],
                rerun_dir=rerun,
                fixed_dir=fixed,
                project_root=root,
            )
            index = load_applicability(
                output,
                run_json=run,
                selected_ids=[1],
                rerun_dir=rerun,
                fixed_dir=fixed,
                project_root=root,
            )
            self.assertEqual(len(index), 2)
            requirements = {
                row["source"]: row["node_role_requirement"]
                for row in index.values()
            }
            self.assertEqual(requirements["rerun"], "requires_both_tips")
            self.assertEqual(requirements["fixed"], "requires_tip_anchor")

    def test_changed_source_rejects_stale_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run, rerun, fixed, draft = self._tree(root)
            output = root / "labels.json"
            compile_applicability(
                draft,
                output,
                run_json=run,
                selected_ids=[1],
                rerun_dir=rerun,
                fixed_dir=fixed,
                project_root=root,
            )
            (rerun / "hypo_1.py").write_text("RERUN = 2\n")
            with self.assertRaisesRegex(SystemExit, "stale"):
                load_applicability(
                    output,
                    run_json=run,
                    selected_ids=[1],
                    rerun_dir=rerun,
                    fixed_dir=fixed,
                    project_root=root,
                )

    def test_unclear_is_valid_annotation_for_fail_closed_downstream(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run = root / "split-error-test.json"
            run.write_text("[]\n")
            rerun = root / "split-error-test.json.rerun"
            rerun.mkdir()
            (rerun / "hypo_1.py").write_text("VALUE = 1\n")
            draft = root / "draft.json"
            draft.write_text(json.dumps({
                "schema_version": 1,
                "records": [{
                    "id": 1,
                    "source": "rerun",
                    "node_role_requirement": "unclear",
                    "reason": "The source mixes endpoint assumptions.",
                }],
            }))
            output = root / "labels.json"
            compile_applicability(
                draft,
                output,
                run_json=run,
                selected_ids=[1],
                rerun_dir=rerun,
                fixed_dir=None,
                project_root=root,
            )
            rows = json.loads(output.read_text())["records"]
            self.assertEqual(rows[0]["node_role_requirement"], "unclear")

    def test_split_inventory_inherits_requirement_from_exact_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run = root / "split-error-test.json"
            run.write_text("[]\n")
            rerun = root / "split-error-test.json.rerun"
            rerun.mkdir()
            (rerun / "hypo_1.py").write_text("VALUE = 1\n")
            label_draft = root / "labels.draft.json"
            label_draft.write_text(json.dumps({
                "schema_version": 1,
                "records": [{
                    "id": 1,
                    "source": "rerun",
                    "node_role_requirement": "requires_tip_anchor",
                    "reason": "Uses a terminal anchor and any partner.",
                }],
            }))
            labels = root / "labels.json"
            compile_applicability(
                label_draft,
                labels,
                run_json=run,
                selected_ids=[1],
                rerun_dir=rerun,
                fixed_dir=None,
                project_root=root,
            )
            summary = root / "split-error-test.summary.md"
            summary.write_text(
                "### 1. test\n**ID:** 1\n"
                "- **Reproduction:** REPRODUCED\n"
                "- **Verdict:** SOUND\n"
            )
            semantics = root / "semantics.json"
            semantics.write_text(json.dumps({
                "schema_version": 1,
                "hypotheses": [{
                    "id": 1,
                    "included": True,
                    "exclusion_reason": None,
                    "correction_scope": "none",
                    "feature_source": "rerun",
                    "source_reason": "Reproduced source is authoritative.",
                    "features": [{
                        "name": "direction_alignment",
                        "quantity": "direction alignment",
                        "constants": {},
                        "aggregation": "closest compatible occurrence",
                        "reduction": "one value per candidate pair",
                        "traversal_phase": "candidate_pair_pass",
                        "measurable_condition": "compatible occurrence exists",
                        "historical_undefined_sentinel": None,
                    }],
                }],
            }))
            inventory = root / "inventory.json"
            compile_feature_inventory(
                semantics,
                inventory,
                selected_ids=[1],
                selection_manifest=None,
                summary_path=summary,
                corrected_results_path=None,
                rerun_dir=rerun,
                fixed_dir=None,
                project_root=root,
                split_applicability_path=labels,
                run_json=run,
            )
            payload = json.loads(inventory.read_text())
            self.assertEqual(payload["schema_version"], 3)
            self.assertEqual(
                payload["hypotheses"][0]["features"][0][
                    "node_role_requirement"
                ],
                "requires_tip_anchor",
            )


if __name__ == "__main__":
    unittest.main()
