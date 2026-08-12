from __future__ import annotations

import importlib.util
import hashlib
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_module():
    sdk = types.ModuleType("claude_agent_sdk")
    for name in (
        "AssistantMessage",
        "ClaudeAgentOptions",
        "ClaudeSDKClient",
        "ResultMessage",
        "TextBlock",
        "ToolUseBlock",
    ):
        setattr(sdk, name, type(name, (), {}))
    sys.modules["claude_agent_sdk"] = sdk
    path = PROJECT_ROOT / "agentic" / "run_detector_build_workflow.py"
    spec = importlib.util.spec_from_file_location("test_detector_workflow", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


workflow = _load_module()


class DetectorBuildWorkflowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.old_root = workflow.PROJECT_ROOT

    def tearDown(self) -> None:
        workflow.PROJECT_ROOT = self.old_root

    @staticmethod
    def _valid_model_config(inventory_path: Path) -> dict:
        def candidate(name: str, role: str, grid: dict, *, native_nan=False,
                      requires_package=None) -> dict:
            return {
                "name": name,
                "role": role,
                "reason": f"Exercise the {name} inductive bias.",
                "grid": grid,
                "native_nan": native_nan,
                "requires_package": requires_package,
            }

        return {
            "schema_version": 1,
            "feature_inventory_sha256": hashlib.sha256(
                inventory_path.read_bytes()
            ).hexdigest(),
            "selection_basis": "Three conservative baselines plus one smooth model.",
            "candidates": [
                candidate("logistic_l2", "baseline", {"C": [0.1, 1.0]}),
                candidate(
                    "logistic_elasticnet", "baseline",
                    {"C": [0.1, 1.0], "l1_ratio": [0.25, 0.75]},
                ),
                candidate(
                    "hist_gradient_boosting", "baseline",
                    {"max_leaf_nodes": [7, 15], "learning_rate": [0.03]},
                    native_nan=True,
                ),
                candidate(
                    "spline_logistic", "optional",
                    {"n_knots": [3, 4], "degree": [2], "C": [0.1, 1.0]},
                ),
            ],
        }

    @staticmethod
    def _touch_inputs(root: Path, *, predictive: bool, fixed: bool) -> Path:
        run_json = root / "run.json"
        run_json.write_text("[]")
        (root / "run.summary.md").write_text("## Header\n")
        scope = ".predictive" if predictive else ""
        rerun = root / f"run.json{scope}.rerun"
        rerun.mkdir()
        (rerun / "hypo_1.py").write_text("print('rerun')\n")
        (rerun / "MANIFEST.json").write_text(json.dumps({
            "records": [{"id": 1, "file": "hypo_1.py"}],
        }))
        if predictive:
            source_hash = hashlib.sha256(run_json.read_bytes()).hexdigest()
            (root / "run.predictive-selection.json").write_text(json.dumps({
                "criterion": "predictive",
                "selected_ids": [1],
                "source_sha256": source_hash,
            }))
        if fixed:
            fixed_dir = root / f"run.json{scope}.fixed"
            fixed_dir.mkdir()
            (fixed_dir / "hypo_1.py").write_text("print('fixed')\n")
        return run_json

    def test_resolve_inputs_returns_matching_predictive_fixed_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            workflow.PROJECT_ROOT = root
            run_json = self._touch_inputs(root, predictive=True, fixed=True)
            # A legacy fixed directory must never be paired with predictive rerun.
            legacy_fixed = root / "run.json.fixed"
            legacy_fixed.mkdir()
            (legacy_fixed / "hypo_9.py").write_text("print('legacy')\n")

            summary, rerun, fixed, selection, selected_ids = workflow.resolve_inputs(
                run_json
            )

            self.assertEqual(summary, root / "run.summary.md")
            self.assertEqual(rerun, root / "run.json.predictive.rerun")
            self.assertEqual(fixed, root / "run.json.predictive.fixed")
            self.assertEqual(selection, root / "run.predictive-selection.json")
            self.assertEqual(selected_ids, [1])

    def test_resolve_inputs_allows_no_fixed_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            workflow.PROJECT_ROOT = root
            run_json = self._touch_inputs(root, predictive=False, fixed=False)

            _, rerun, fixed, selection, selected_ids = workflow.resolve_inputs(run_json)

            self.assertEqual(rerun, root / "run.json.rerun")
            self.assertIsNone(fixed)
            self.assertIsNone(selection)
            self.assertEqual(selected_ids, [1])

    def test_resolve_inputs_rejects_stale_predictive_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            workflow.PROJECT_ROOT = root
            run_json = self._touch_inputs(root, predictive=True, fixed=False)
            selection_path = root / "run.predictive-selection.json"
            selection = json.loads(selection_path.read_text())
            selection["selected_ids"] = [2]
            selection_path.write_text(json.dumps(selection))

            with self.assertRaisesRegex(SystemExit, "MANIFEST disagree"):
                workflow.resolve_inputs(run_json)

    def test_build_steps_require_source_audit_and_safe_exclusions(self) -> None:
        steps = workflow.build_steps(
            run_rel="autodiscovery/run.json",
            summary_rel="autodiscovery/run.summary.md",
            rerun_rel="autodiscovery/run.json.predictive.rerun",
            fixed_rel="autodiscovery/run.json.predictive.fixed",
            selection_rel="autodiscovery/run.predictive-selection.json",
            selected_ids=[1],
            out_dir_rel="autodiscovery-application/run",
        )
        by_name = {step["name"]: step["instruction"] for step in steps}
        inventory = by_name["inventory-features"]
        configure = by_name["configure-models"]
        generate = by_name["generate-detector"]
        verify = by_name["verify-and-document"]

        self.assertEqual([step["name"] for step in steps], [
            "inventory-features", "configure-models", "generate-detector",
            "verify-and-document",
        ])
        self.assertIn("FAILED/UNUSABLE", inventory)
        self.assertIn("uncorrected CRITICAL", inventory)
        self.assertIn("feature_semantics_changed", inventory)
        self.assertIn("stale_fixed_ignored", inventory)
        self.assertIn("schema_version=2", inventory)
        self.assertIn("authoritative hypothesis set", inventory)
        self.assertIn("feature_source_path", inventory)
        self.assertIn("SHA-256", inventory)
        self.assertIn("zero, one, or two", configure)
        self.assertIn("Never write", configure)
        self.assertIn("feature_inventory_sha256", configure)
        self.assertIn("native_nan is", configure)
        self.assertIn("--model-config", generate)
        self.assertIn("Never eval", generate)
        self.assertIn("parameter value types/ranges", generate)
        self.assertIn("feature_source_path", generate)
        self.assertIn("Never silently fall back", generate)
        self.assertIn("rerun-vs-fixed choice", verify)

    def test_validate_model_config_accepts_bounded_allowlisted_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            workflow.PROJECT_ROOT = root
            inventory_path = root / "feature_inventory.json"
            inventory_path.write_text('{"schema_version": 2}')
            config_path = root / workflow.MODEL_CONFIG_NAME
            config_path.write_text(json.dumps(self._valid_model_config(inventory_path)))

            workflow.validate_model_config(config_path, inventory_path)

    def test_validate_model_config_rejects_missing_baseline_and_stale_hash(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            workflow.PROJECT_ROOT = root
            inventory_path = root / "feature_inventory.json"
            inventory_path.write_text('{"schema_version": 2}')
            config_path = root / workflow.MODEL_CONFIG_NAME
            config = self._valid_model_config(inventory_path)
            config["candidates"] = [
                row for row in config["candidates"]
                if row["name"] != "logistic_l2"
            ]
            config_path.write_text(json.dumps(config))
            with self.assertRaisesRegex(SystemExit, "missing required baselines"):
                workflow.validate_model_config(config_path, inventory_path)

            config = self._valid_model_config(inventory_path)
            config["feature_inventory_sha256"] = "0" * 64
            config_path.write_text(json.dumps(config))
            with self.assertRaisesRegex(SystemExit, "is stale"):
                workflow.validate_model_config(config_path, inventory_path)

    def test_validate_model_config_rejects_unsafe_or_excessive_extensions(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            workflow.PROJECT_ROOT = root
            inventory_path = root / "feature_inventory.json"
            inventory_path.write_text('{"schema_version": 2}')
            config_path = root / workflow.MODEL_CONFIG_NAME

            config = self._valid_model_config(inventory_path)
            config["candidates"][0]["grid"]["class_path"] = ["evil.Model"]
            config_path.write_text(json.dumps(config))
            with self.assertRaisesRegex(SystemExit, "disallowed parameters"):
                workflow.validate_model_config(config_path, inventory_path)

            config = self._valid_model_config(inventory_path)
            for name in ("extra_trees", "random_forest"):
                config["candidates"].append({
                    "name": name,
                    "role": "optional",
                    "reason": "Test bounded extension count.",
                    "grid": {"n_estimators": [100]},
                    "native_nan": False,
                    "requires_package": None,
                })
            config_path.write_text(json.dumps(config))
            with self.assertRaisesRegex(SystemExit, "maximum is 2"):
                workflow.validate_model_config(config_path, inventory_path)

    def test_validate_model_config_rejects_false_metadata_and_bad_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            workflow.PROJECT_ROOT = root
            inventory_path = root / "feature_inventory.json"
            inventory_path.write_text('{"schema_version": 2}')
            config_path = root / workflow.MODEL_CONFIG_NAME

            self.assertEqual(
                set(workflow.MODEL_PARAMETER_ALLOWLIST),
                workflow.ALLOWED_MODEL_FAMILIES,
            )
            self.assertEqual(
                set(workflow.MODEL_NATIVE_NAN),
                workflow.ALLOWED_MODEL_FAMILIES,
            )

            config = self._valid_model_config(inventory_path)
            config["candidates"][0]["native_nan"] = True
            config_path.write_text(json.dumps(config))
            with self.assertRaisesRegex(SystemExit, "native_nan must be False"):
                workflow.validate_model_config(config_path, inventory_path)

            config = self._valid_model_config(inventory_path)
            config["candidates"][0]["grid"]["C"] = [None, "large"]
            config_path.write_text(json.dumps(config))
            with self.assertRaisesRegex(SystemExit, "has invalid values"):
                workflow.validate_model_config(config_path, inventory_path)

            config = self._valid_model_config(inventory_path)
            config["candidates"][0]["name"] = "arbitrary.Estimator"
            config_path.write_text(json.dumps(config))
            with self.assertRaisesRegex(SystemExit, "disallowed family"):
                workflow.validate_model_config(config_path, inventory_path)

    def test_validate_inventory_checks_selected_source_hash(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            workflow.PROJECT_ROOT = root
            rerun = root / "run.json.predictive.rerun"
            rerun.mkdir()
            source = rerun / "hypo_1.py"
            source.write_text("FEATURE = 1\n")
            source_rel = workflow._rel_to_root(source)
            source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
            inventory_path = root / "feature_inventory.json"
            inventory = {
                "schema_version": 2,
                "selection_manifest": "run.predictive-selection.json",
                "selected_ids": [1],
                "hypotheses": [{
                    "id": 1,
                    "included": True,
                    "exclusion_reason": None,
                    "reproduction_status": "REPRODUCED",
                    "statistical_verdict": "SOUND",
                    "corrected_result_status": None,
                    "post_correction_verdict": None,
                    "correction_scope": "none",
                    "rerun_path": source_rel,
                    "rerun_sha256": source_hash,
                    "fixed_path": None,
                    "fixed_sha256": None,
                    "feature_source": "rerun",
                    "feature_source_path": source_rel,
                    "feature_source_sha256": source_hash,
                    "source_reason": "No correction was required.",
                    "features": [{
                        "name": "feature",
                        "quantity": "one",
                        "constants": {},
                        "aggregation": "segment",
                        "reduction": "identity",
                        "traversal_phase": "segment",
                        "measurable_condition": "always",
                        "historical_undefined_sentinel": None,
                    }],
                }],
            }
            inventory_path.write_text(json.dumps(inventory))

            workflow.validate_inventory(
                inventory_path,
                [1],
                "run.predictive-selection.json",
                rerun,
                None,
            )
            inventory["hypotheses"][0]["correction_scope"] = "unclear"
            inventory_path.write_text(json.dumps(inventory))
            with self.assertRaisesRegex(SystemExit, "cannot be included"):
                workflow.validate_inventory(
                    inventory_path,
                    [1],
                    "run.predictive-selection.json",
                    rerun,
                    None,
                )
            inventory["hypotheses"][0]["correction_scope"] = "none"
            inventory["hypotheses"][0]["feature_source_sha256"] = "0" * 64
            inventory_path.write_text(json.dumps(inventory))
            with self.assertRaisesRegex(SystemExit, "source SHA-256 is stale"):
                workflow.validate_inventory(
                    inventory_path,
                    [1],
                    "run.predictive-selection.json",
                    rerun,
                    None,
                )


if __name__ == "__main__":
    unittest.main()
