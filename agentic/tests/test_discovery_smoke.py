from __future__ import annotations

import asyncio
import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_workflow_module():
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
    path = PROJECT_ROOT / "agentic" / "run_discovery_workflow.py"
    spec = importlib.util.spec_from_file_location("smoke_workflow", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


workflow = _load_workflow_module()


def _records() -> list[dict]:
    return [
        {
            "id": 1,
            "status": "SUCCEEDED",
            "hypothesis": "Positive feature\nwith wrapped text.",
            "surprisal": 0.8,
            "prior": 0.1,
            "posterior": 0.9,
        },
        {
            "id": 2,
            "status": "SUCCEEDED",
            "hypothesis": "Inverse feature.",
            "surprisal": -0.7,
            "prior": 0.9,
            "posterior": 0.2,
        },
        {
            "id": 3,
            "status": "SUCCEEDED",
            "hypothesis": "Excluded constant feature.",
            "surprisal": -0.6,
            "prior": 0.8,
            "posterior": 0.2,
        },
    ]


class DiscoverySmokeTests(unittest.TestCase):
    def setUp(self) -> None:
        workflow.PROJECT_ROOT = PROJECT_ROOT
        workflow.PREDICTIVE_MANIFEST = None

    def test_positive_smoke_respects_top_k_and_prints_one_line(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_json = Path(tmpdir) / "run.json"
            run_json.write_text(json.dumps(_records()))
            workflow.DIRECTION = "positive"
            workflow.TOP_K = 1

            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                asyncio.run(workflow.run_smoke(run_json))

        self.assertEqual(
            output.getvalue().strip(),
            "ID 1: Positive feature with wrapped text.",
        )

    def test_predictive_smoke_saves_and_reuses_selection_manifest(self) -> None:
        class FakeClient:
            def __init__(self, options=None):
                self.options = options

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

        async def fake_run_step(client, step):
            manifest = {
                "source_file": "run.json",
                "criterion": "predictive",
                "selected_ids": [1, 2],
                "excluded": [{"id": 3, "reason": "constant"}],
            }
            path = workflow.PROJECT_ROOT / workflow.PREDICTIVE_MANIFEST
            path.write_text(json.dumps(manifest))
            return ""

        with tempfile.TemporaryDirectory() as tmpdir:
            run_json = Path(tmpdir) / "run.json"
            run_json.write_text(json.dumps(_records()))
            workflow.DIRECTION = "predictive"
            workflow.TOP_K = None
            old_client = workflow.ClaudeSDKClient
            old_options = workflow.build_options
            old_run_step = workflow.run_step
            workflow.ClaudeSDKClient = FakeClient
            workflow.build_options = lambda: None
            workflow.run_step = fake_run_step
            try:
                output = io.StringIO()
                with contextlib.redirect_stdout(output), contextlib.redirect_stderr(
                    io.StringIO()
                ):
                    asyncio.run(workflow.run_smoke(run_json))

                manifest_path = workflow.predictive_manifest_path(run_json)
                manifest = json.loads(manifest_path.read_text())
                self.assertEqual(manifest["policy_version"], "exclusion-only-v1")
                self.assertEqual(
                    manifest["source_sha256"], workflow._source_sha256(run_json)
                )
                self.assertEqual(
                    manifest["one_line_summaries"],
                    output.getvalue().strip().splitlines(),
                )

                async def fail_if_called(client, step):
                    raise AssertionError("cached predictive selection was not reused")

                workflow.run_step = fail_if_called
                cached_output = io.StringIO()
                with contextlib.redirect_stdout(
                    cached_output
                ), contextlib.redirect_stderr(io.StringIO()):
                    asyncio.run(workflow.run_smoke(run_json))
                cached_manifest = json.loads(manifest_path.read_text())
                self.assertEqual(
                    cached_manifest["one_line_summaries"],
                    cached_output.getvalue().strip().splitlines(),
                )
                manifest_rel = workflow._rel_to_root(manifest_path)
                self.assertTrue(
                    workflow._predictive_manifest_is_current(
                        manifest_rel, workflow._rel_to_root(run_json)
                    )
                )
                changed = _records()
                changed[0]["hypothesis"] = "Changed source hypothesis."
                run_json.write_text(json.dumps(changed))
                self.assertFalse(
                    workflow._predictive_manifest_is_current(
                        manifest_rel, workflow._rel_to_root(run_json)
                    )
                )
            finally:
                workflow.ClaudeSDKClient = old_client
                workflow.build_options = old_options
                workflow.run_step = old_run_step

        lines = output.getvalue().strip().splitlines()
        self.assertEqual({line.split(":", 1)[0] for line in lines}, {"ID 1", "ID 2"})
        self.assertEqual(cached_output.getvalue(), output.getvalue())
        self.assertIsNone(workflow.PREDICTIVE_MANIFEST)

    def test_compute_steps_use_selective_and_extra_only_work_items(self) -> None:
        workflow.DIRECTION = "both"
        workflow.TOP_K = 2
        steps = workflow.build_steps(
            "autodiscovery/run.json",
            "cache/origin.pkl",
            "autodiscovery/run.summary.md",
            ["cache/extra-a.pkl", "cache/extra-b.pkl"],
        )
        by_name = {step["name"]: step for step in steps}

        remeasure = by_name["reproduce-remeasure"]
        self.assertIn("--only-changed", remeasure["argv"])
        self.assertIn("--base-results", remeasure["argv"])
        self.assertIn("--checkpoint", remeasure["argv"])
        self.assertNotIn("skip_if_no_marker", remeasure)

        extrapolate = by_name["extrapolate-run"]
        self.assertIn("--extra-only", extrapolate["argv"])
        self.assertIn("--origin-results", extrapolate["argv"])
        self.assertEqual(extrapolate["argv"].count("--extra-pkl"), 2)

        corrected = by_name["fix-tests-measure"]
        self.assertIn("--only-corrected", corrected["argv"])
        self.assertEqual(workflow.COMPUTE_TIMEOUT_S, 43200)


if __name__ == "__main__":
    unittest.main()
