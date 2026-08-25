from __future__ import annotations

import asyncio
import contextlib
import hashlib
import importlib.util
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

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


class _TtyInput(io.StringIO):
    def isatty(self) -> bool:
        return True


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

    def test_workflow_cost_summary_diffs_session_cumulative_costs(self) -> None:
        costs = workflow.WorkflowCostSummary()
        for _ in range(4):
            costs.start_turn()
        self.assertEqual(costs.add(1.25, "session-a"), 1.25)
        self.assertEqual(costs.add(1.75, "session-a"), 0.5)
        self.assertEqual(costs.add(0.4, "session-b"), 0.4)
        self.assertIsNone(costs.add(None, "session-b"))
        self.assertEqual(costs.total_usd, 2.15)
        self.assertIn("$2.1500", costs.describe())

    def test_usage_formatting_supports_sdk_key_styles(self) -> None:
        self.assertEqual(
            workflow._format_token_usage({
                "input_tokens": 10,
                "output_tokens": 20,
                "cache_read_input_tokens": 30,
                "cache_creation_input_tokens": 40,
            }),
            "input=10, output=20, cache_read=30, cache_write=40",
        )
        formatted = workflow._format_model_usage({
            "claude-sonnet": {
                "inputTokens": 11,
                "outputTokens": 22,
                "cacheReadInputTokens": 33,
                "cacheCreationInputTokens": 44,
                "costUSD": 0.125,
            },
        })
        self.assertEqual(
            formatted,
            "claude-sonnet (input=11, output=22, cache_read=33, "
            "cache_write=44, cost=$0.1250)",
        )

    def test_run_step_uses_a_fresh_sdk_session_each_time(self) -> None:
        class SessionClient:
            def __init__(self) -> None:
                self.queries = []

            async def query(self, prompt, session_id="default"):
                self.queries.append((prompt, session_id))

            async def receive_response(self):
                if False:
                    yield None

        client = SessionClient()
        workflow.WORKFLOW_COSTS.reset()
        asyncio.run(workflow.run_step(client, {
            "name": "first stage",
            "instruction": "first prompt",
        }))
        asyncio.run(workflow.run_step(client, {
            "name": "second/stage",
            "instruction": "second prompt",
        }))

        self.assertEqual([prompt for prompt, _ in client.queries], [
            "first prompt", "second prompt",
        ])
        session_ids = [session_id for _, session_id in client.queries]
        self.assertEqual(len(set(session_ids)), 2)
        self.assertTrue(session_ids[0].startswith("discovery-first-stage-"))
        self.assertTrue(session_ids[1].startswith("discovery-second-stage-"))
        self.assertNotIn("default", session_ids)

    def test_mechanical_stages_use_opus_at_medium_effort(self) -> None:
        workflow.DIRECTION = "predictive"
        workflow.TOP_K = None
        workflow.PREDICTIVE_MANIFEST = "autodiscovery/selection.json"
        steps = {
            step["name"]: step
            for step in workflow.build_steps(
                "autodiscovery/split-error-test.json",
                "cache/dataset_cache_1_mcl100_add.pkl",
                "autodiscovery/split-error-test.summary.md",
                ["cache/dataset_cache_2_mcl100_add.pkl"],
            )
        }
        self.assertIn(
            "discovery-predictive-selector",
            steps["select-predictive-hypotheses"]["instruction"],
        )
        self.assertIn(
            "discovery-feature-applicability-labeler",
            steps["label-split-feature-applicability"]["instruction"],
        )
        self.assertIn(
            "discovery-test-fixer",
            steps["fix-tests-author"]["instruction"],
        )
        self.assertIn(
            "discovery-test-result-folder",
            steps["fix-tests-fold"]["instruction"],
        )

        for name in (
            "discovery-reproducer.md",
            "discovery-extrapolator.md",
            "discovery-predictive-selector.md",
            "discovery-feature-applicability-labeler.md",
            "discovery-test-result-folder.md",
        ):
            frontmatter = (PROJECT_ROOT / ".claude" / "agents" / name).read_text()
            self.assertIn("model: inherit", frontmatter)
            self.assertIn("effort: medium", frontmatter)

        driver = (PROJECT_ROOT / "agentic" / "run_discovery_workflow.py").read_text()
        self.assertIn('model="claude-opus-4-8"', driver)

        statistical_author = (
            PROJECT_ROOT / ".claude" / "agents" / "discovery-test-fixer.md"
        ).read_text()
        self.assertIn("model: inherit", statistical_author)
        self.assertIn("effort: xhigh", statistical_author)

    def test_split_workflow_appends_source_bound_feature_label_stage(self) -> None:
        workflow.DIRECTION = "predictive"
        workflow.PREDICTIVE_MANIFEST = (
            "autodiscovery/split-error-test.predictive-selection.json"
        )
        steps = workflow.build_steps(
            "autodiscovery/split-error-test.json",
            "cache/dataset_cache_1_mcl100_add.pkl",
            "autodiscovery/split-error-test.summary.md",
        )
        label = steps[-1]
        self.assertEqual(label["name"], "label-split-feature-applicability")
        self.assertEqual(label["phase"], "feature-applicability")
        self.assertIn("requires_both_tips", label["instruction"])
        self.assertIn("requires_tip_anchor", label["instruction"])
        self.assertIn("do not run experiments", label["instruction"])
        self.assertTrue(
            str(label["expects_file"]).endswith(
                ".split-feature-applicability.draft.json"
            )
        )

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

    def test_rerun_manifest_must_match_selection_and_scripts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            old_root = workflow.PROJECT_ROOT
            workflow.PROJECT_ROOT = root
            try:
                rerun_dir = root / "run.rerun"
                rerun_dir.mkdir()
                (rerun_dir / "hypo_1.py").write_text("print(1)")
                (rerun_dir / "MANIFEST.json").write_text(json.dumps({
                    "records": [{"id": 1}],
                }))
                selection = root / "selection.json"
                selection.write_text(json.dumps({"selected_ids": [1]}))

                self.assertTrue(workflow._rerun_manifest_matches_selection(
                    "run.rerun", "selection.json"
                ))
                selection.write_text(json.dumps({"selected_ids": [2]}))
                self.assertFalse(workflow._rerun_manifest_matches_selection(
                    "run.rerun", "selection.json"
                ))
            finally:
                workflow.PROJECT_ROOT = old_root

    def test_split_label_inputs_use_selection_order_not_manifest_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            old_root = workflow.PROJECT_ROOT
            workflow.PROJECT_ROOT = root
            try:
                run_json = root / "split-error-run.json"
                run_json.write_text(json.dumps([{"id": 1}, {"id": 2}]))
                rerun_dir = root / "split-error-run.json.predictive.rerun"
                rerun_dir.mkdir()
                for hypothesis_id in (1, 2):
                    (rerun_dir / f"hypo_{hypothesis_id}.py").write_text(
                        f"VALUE = {hypothesis_id}\n"
                    )
                (rerun_dir / "MANIFEST.json").write_text(json.dumps({
                    "records": [{"id": 2}, {"id": 1}],
                }))
                workflow.predictive_manifest_path(run_json).write_text(json.dumps({
                    "selected_ids": [1, 2],
                    "excluded": [],
                }))

                selected_ids, resolved_rerun, fixed = (
                    workflow._resolve_split_label_inputs(run_json)
                )

                self.assertEqual(selected_ids, [1, 2])
                self.assertEqual(resolved_rerun, rerun_dir)
                self.assertIsNone(fixed)
            finally:
                workflow.PROJECT_ROOT = old_root

    def test_labels_only_compiles_existing_draft_without_agent(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            old_root = workflow.PROJECT_ROOT
            workflow.PROJECT_ROOT = root
            try:
                run_json = root / "split-error-run.json"
                run_json.write_text(json.dumps([{"id": 1}, {"id": 2}]))
                rerun_dir = root / "split-error-run.json.predictive.rerun"
                rerun_dir.mkdir()
                for hypothesis_id in (1, 2):
                    (rerun_dir / f"hypo_{hypothesis_id}.py").write_text(
                        f"VALUE = {hypothesis_id}\n"
                    )
                (rerun_dir / "MANIFEST.json").write_text(json.dumps({
                    "records": [{"id": 2}, {"id": 1}],
                }))
                workflow.predictive_manifest_path(run_json).write_text(json.dumps({
                    "selected_ids": [1, 2],
                    "excluded": [],
                }))
                draft = workflow.split_applicability_draft_path(run_json)
                draft.write_text(json.dumps({
                    "schema_version": 1,
                    "records": [
                        {
                            "id": hypothesis_id,
                            "source": "rerun",
                            "node_role_requirement": "no_tip_requirement",
                            "reason": "The feature is defined for any node pair.",
                        }
                        for hypothesis_id in (1, 2)
                    ],
                }))

                with mock.patch.object(
                    workflow,
                    "run_step",
                    side_effect=AssertionError("agent should not run"),
                ):
                    asyncio.run(workflow.run_split_feature_labeling_only(run_json))

                output = workflow.split_applicability_path(run_json)
                payload = json.loads(output.read_text())
                self.assertEqual(payload["selected_ids"], [1, 2])
                self.assertEqual(
                    [record["id"] for record in payload["records"]], [1, 2]
                )
                self.assertFalse(draft.exists())
            finally:
                workflow.PROJECT_ROOT = old_root

    def test_prior_artifact_survey_reports_mismatch_and_old_analysis(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            old_root = workflow.PROJECT_ROOT
            workflow.PROJECT_ROOT = root
            try:
                run_json = root / "run.json"
                run_json.write_text(json.dumps([
                    {"id": 1}, {"id": 2},
                ]))
                (root / "run.summary.md").write_text("old report")
                (root / "run.json.predictive.reproduce.json").write_text("{}")
                rerun_dir = root / "run.json.predictive.rerun"
                rerun_dir.mkdir()
                (rerun_dir / "hypo_1.py").write_text("print(1)")
                (rerun_dir / "MANIFEST.json").write_text(json.dumps({
                    "records": [{"id": 1}],
                }))
                fixed_dir = root / "run.json.predictive.fixed"
                fixed_dir.mkdir()
                (fixed_dir / "hypo_2.py").write_text("print(2)")
                selection_path = root / "run.predictive-selection.json"
                selection_path.write_text(json.dumps({
                    "criterion": "predictive",
                    "selected_ids": [2],
                    "excluded": [{"id": 1, "reason": "not predictive"}],
                    "source_sha256": workflow._source_sha256(run_json),
                    "policy_version": workflow.PREDICTIVE_POLICY_VERSION,
                }))

                findings = workflow.survey_prior_artifacts(run_json, "predictive")
                by_path = {path: reason for path, reason in findings}

                self.assertIn(root / "run.summary.md", by_path)
                self.assertIn(root / "run.json.predictive.reproduce.json", by_path)
                self.assertIn(rerun_dir, by_path)
                self.assertIn(fixed_dir, by_path)
                self.assertIn(rerun_dir / "MANIFEST.json", by_path)
                self.assertIn("STALE/MISMATCHED", by_path[rerun_dir / "MANIFEST.json"])
                self.assertNotIn(selection_path, by_path)
            finally:
                workflow.PROJECT_ROOT = old_root

    def test_prior_artifact_confirmation_requires_explicit_yes(self) -> None:
        findings = [(Path("old.json"), "previous measured-analysis JSON")]

        with (
            mock.patch.object(workflow.sys, "stdin", _TtyInput("y\n")),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            workflow.confirm_prior_artifacts(findings, assume_yes=False)

        with (
            mock.patch.object(workflow.sys, "stdin", _TtyInput("n\n")),
            contextlib.redirect_stderr(io.StringIO()),
            self.assertRaisesRegex(SystemExit, "left untouched"),
        ):
            workflow.confirm_prior_artifacts(findings, assume_yes=False)

        with (
            mock.patch.object(workflow.sys, "stdin", io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
            self.assertRaisesRegex(SystemExit, "--yes-stale"),
        ):
            workflow.confirm_prior_artifacts(findings, assume_yes=False)

        with contextlib.redirect_stderr(io.StringIO()):
            workflow.confirm_prior_artifacts(findings, assume_yes=True)

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

        export = by_name["reproduce-export"]
        self.assertEqual(
            export["skip_if_scripts_match_selection"],
            workflow.PREDICTIVE_MANIFEST,
        )

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
        # The translate step was removed: fix-tests-fold is now the last step.
        self.assertNotIn("translate-report", by_name)
        self.assertEqual(steps[-1]["name"], "fix-tests-fold")
        self.assertEqual(workflow.COMPUTE_TIMEOUT_S, 86400)

        # Repair rounds 2..MAX sit between the first remeasure and the single
        # fold, roll their merge through the same .reproduce.json, and carry
        # the self-terminating skip/stall rules.
        self.assertEqual(workflow.MAX_FIX_LOADING_ROUNDS, 3)
        for round_no in range(2, workflow.MAX_FIX_LOADING_ROUNDS + 1):
            fix = by_name[f"reproduce-fix-loading-{round_no}"]
            self.assertEqual(fix["kind"], "agent")
            self.assertTrue(fix["skip_if_stalled"])
            self.assertTrue(
                str(fix["skip_if_no_unusable_in"]).endswith(".reproduce.json")
            )
            repair = by_name[f"reproduce-remeasure-{round_no}"]
            self.assertTrue(repair["skip_if_stalled"])
            self.assertIn("--only-changed", repair["argv"])
            base = repair["argv"][repair["argv"].index("--base-results") + 1]
            self.assertEqual(base, repair["out"])  # rolling merge in place
            self.assertEqual(
                repair["stall_if_code_unchanged"],
                {"code_dir": repair["argv"][repair["argv"].index("--code-dir") + 1],
                 "results": repair["out"]},
            )
        names = [str(s["name"]) for s in steps]
        self.assertLess(
            names.index("reproduce-remeasure"),
            names.index("reproduce-fix-loading-2"),
        )
        self.assertLess(
            names.index(f"reproduce-remeasure-{workflow.MAX_FIX_LOADING_ROUNDS}"),
            names.index("reproduce-fold"),
        )

        # Every fix round carries the driver-side syntax gate and the embedded
        # failure digest: round 1 baselines against the raw results, repair
        # rounds against the rolling merged results.
        first_fix = by_name["reproduce-fix-loading"]
        self.assertTrue(
            str(first_fix["syntax_check"]["results"]).endswith(".reproduce-raw.json")
        )
        self.assertTrue(
            str(first_fix["failure_brief"]["results"]).endswith(".reproduce-raw.json")
        )
        for round_no in range(2, workflow.MAX_FIX_LOADING_ROUNDS + 1):
            fix = by_name[f"reproduce-fix-loading-{round_no}"]
            self.assertTrue(
                str(fix["syntax_check"]["results"]).endswith(".reproduce.json")
            )
            self.assertTrue(
                str(fix["failure_brief"]["results"]).endswith(".reproduce.json")
            )

    def test_syntax_gate_and_failure_brief_helpers(self) -> None:
        old_root = workflow.PROJECT_ROOT
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                root = Path(tmpdir)
                workflow.PROJECT_ROOT = root
                rerun_dir = root / "run.json.rerun"
                rerun_dir.mkdir()
                unchanged_code = "print('old')\n"
                (rerun_dir / "hypo_1.py").write_text("def broken(:\n")   # changed + broken
                (rerun_dir / "hypo_2.py").write_text("print('fine')\n")  # changed + valid
                (rerun_dir / "hypo_3.py").write_text(unchanged_code)      # unchanged
                results_rel = "run.json.reproduce.json"
                (root / results_rel).write_text(json.dumps({"results": [
                    {
                        "id": 1,
                        "result_status": "UNUSABLE",
                        "result_failure_reason": "dataset-loading",
                        "code_source": "revised",
                        "rerun_stderr": "Traceback ...\nFileNotFoundError: nope",
                        "recorded_code_sha256": "r1",
                        "executed_code_sha256": "e1",
                    },
                    {
                        "id": 2,
                        "result_status": "UNUSABLE",
                        "result_failure_reason": "timeout",
                        "code_source": "recorded",
                        "rerun_stderr": "",
                        "recorded_code_sha256": "r2",
                        "executed_code_sha256": "e2",
                    },
                    {
                        "id": 3,
                        "result_status": "USABLE",
                        "recorded_code_sha256": "r3",
                        "executed_code_sha256": hashlib.sha256(
                            unchanged_code.encode("utf-8")
                        ).hexdigest(),
                    },
                ]}))

                problems = workflow._broken_changed_scripts(
                    "run.json.rerun", results_rel
                )
                self.assertEqual(len(problems), 1)
                self.assertIn("hypo_1.py", problems[0])
                repair = workflow._syntax_repair_instruction(
                    problems, "run.json.rerun"
                )
                self.assertIn("hypo_1.py", repair)
                self.assertIn("do not parse", repair)

                brief = workflow._failure_brief(results_rel, "run.json.rerun")
                self.assertIn("2 UNUSABLE result(s)", brief)
                self.assertIn(
                    "id 1: dataset-loading (revised) — FileNotFoundError: nope",
                    brief,
                )
                self.assertIn("id 2: timeout (recorded)", brief)
                self.assertNotIn("id 3", brief)
                self.assertEqual(
                    workflow._failure_brief("missing.json", "run.json.rerun"), ""
                )
        finally:
            workflow.PROJECT_ROOT = old_root

    def test_repair_loop_helpers_detect_change_unusable_and_stall(self) -> None:
        old_root = workflow.PROJECT_ROOT
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                root = Path(tmpdir)
                workflow.PROJECT_ROOT = root
                rerun = root / "run.json.rerun"
                rerun.mkdir()
                code = "print('x')\n"
                code_hash = hashlib.sha256(code.encode("utf-8")).hexdigest()
                (rerun / "hypo_1.py").write_text(code)
                results_rel = "run.json.reproduce.json"
                (root / results_rel).write_text(json.dumps({"results": [
                    {
                        "id": 1,
                        "result_status": "UNUSABLE",
                        "result_failure_reason": "nonzero-exit",
                        "code_source": "revised",
                        "rerun_stderr": "Traceback ...\nKeyError: 'gt_graph'",
                        "recorded_code_sha256": "recorded-1",
                        "executed_code_sha256": code_hash,
                    },
                    {
                        "id": 2,
                        "result_status": "USABLE",
                        "recorded_code_sha256": "recorded-2",
                        "executed_code_sha256": "recorded-2",
                    },
                ]}))

                self.assertTrue(workflow._has_unusable_results(results_rel))
                self.assertFalse(workflow._has_unusable_results("missing.json"))
                # hypo_1 matches its executed hash; id 2 has no script and its
                # recorded hash matches — nothing new to measure.
                self.assertFalse(
                    workflow._scripts_changed_since("run.json.rerun", results_rel)
                )
                # Unreadable results cannot prove convergence.
                self.assertTrue(
                    workflow._scripts_changed_since("run.json.rerun", "missing.json")
                )

                step = {
                    "name": "reproduce-remeasure-2",
                    "kind": "compute",
                    "stall_if_code_unchanged": {
                        "code_dir": "run.json.rerun",
                        "results": results_rel,
                    },
                    "argv": ["python", "should-not-run"],
                    "out": results_rel,
                }
                with mock.patch.object(
                    workflow, "run_compute",
                    side_effect=AssertionError("stalled round must not launch"),
                ):
                    self.assertEqual(workflow.run_compute_step(step), "stalled")

                # After a real edit the same round measures again.
                (rerun / "hypo_1.py").write_text(code + "# fixed\n")
                self.assertTrue(
                    workflow._scripts_changed_since("run.json.rerun", results_rel)
                )
                with mock.patch.object(workflow, "run_compute") as run_compute:
                    self.assertEqual(workflow.run_compute_step(step), "ran")
                    run_compute.assert_called_once()

                # The failure digest surfaces id, reason, source, stderr tail.
                digest = io.StringIO()
                with contextlib.redirect_stderr(digest):
                    workflow._log_failure_summary(results_rel, "reproduce-remeasure-2")
                text = digest.getvalue()
                self.assertIn("1 USABLE, 1 UNUSABLE", text)
                self.assertIn("id 1: nonzero-exit (revised)", text)
                self.assertIn("KeyError: 'gt_graph'", text)
        finally:
            workflow.PROJECT_ROOT = old_root


if __name__ == "__main__":
    unittest.main()
