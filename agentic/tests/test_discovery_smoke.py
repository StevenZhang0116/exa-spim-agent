from __future__ import annotations

import asyncio
import contextlib
import hashlib
import importlib.util
import io
import json
import re
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
            workflow.fix_tests_author_instruction(
                **{
                    "summary_rel": steps["fix-tests-author"]["fix_tests_batches"]["summary"],
                    "rerun_dir_rel": "r", "fixed_dir_rel": "f",
                    "ids": ["1"], "drafts": [],
                }
            ),
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
        self.assertIn('model="claude-opus-5-5"', driver)

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

    def test_rerun_argv_forwards_only_the_companions_of_the_pkls_it_passes(self) -> None:
        """The reproduce step runs the origin pkl alone, so the companion of an
        --extra-pkl brain must not reach it: rerun_experiments.py rejects a companion
        whose manifest names a pkl absent from the call (the 2026-10-14 abort)."""
        saved = (workflow.COMPANIONS_BY_PKL, workflow.DIRECTION, workflow.TOP_K,
                 workflow.PREDICTIVE_MANIFEST)
        try:
            workflow.COMPANIONS_BY_PKL = {
                "cache/a.pkl": ["m/rerank_companion_111_split.npz", "m/rerank_companion_111_split.json"],
                "cache/b.pkl": ["m/rerank_companion_222_split.npz", "m/rerank_companion_222_split.json"],
            }
            workflow.DIRECTION, workflow.TOP_K, workflow.PREDICTIVE_MANIFEST = "predictive", None, "sel.json"

            def companions(argv: list[str]) -> list[str]:
                return [argv[i + 1] for i, a in enumerate(argv) if a == "--companion"]

            reproduce = workflow.rerun_argv("run.json", "cache/a.pkl")
            self.assertEqual(companions(reproduce),
                             ["m/rerank_companion_111_split.npz", "m/rerank_companion_111_split.json"])
            self.assertNotIn("--extra-pkl", reproduce)
            extrapolate = workflow.rerun_argv(
                "run.json", "cache/a.pkl", extra_pkls_rel=["cache/b.pkl"],
                origin_results_rel="origin.json", extra_only=True)
            self.assertEqual(companions(extrapolate), [
                "m/rerank_companion_111_split.npz", "m/rerank_companion_111_split.json",
                "m/rerank_companion_222_split.npz", "m/rerank_companion_222_split.json"])
            self.assertEqual(extrapolate[extrapolate.index("--extra-pkl") + 1], "cache/b.pkl")
            # No companions at all: the flag is absent (pre-companion runs unchanged).
            workflow.COMPANIONS_BY_PKL = {}
            self.assertNotIn("--companion", workflow.rerun_argv("run.json", "cache/a.pkl"))
        finally:
            (workflow.COMPANIONS_BY_PKL, workflow.DIRECTION, workflow.TOP_K,
             workflow.PREDICTIVE_MANIFEST) = saved

    def test_rerun_argv_lets_runs_install_missing_packages_unless_disabled(self) -> None:
        saved = (workflow.INSTALL_MISSING, workflow.DIRECTION, workflow.TOP_K,
                 workflow.PREDICTIVE_MANIFEST)
        try:
            workflow.DIRECTION, workflow.TOP_K, workflow.PREDICTIVE_MANIFEST = "predictive", None, "sel.json"
            self.assertIn("--install-missing", workflow.rerun_argv("run.json", "cache/a.pkl"))
            self.assertNotIn("--install-missing", workflow.rerun_argv(
                "run.json", "cache/a.pkl", export_dir_rel="run.rerun"))
            workflow.INSTALL_MISSING = False
            self.assertNotIn("--install-missing", workflow.rerun_argv("run.json", "cache/a.pkl"))
        finally:
            (workflow.INSTALL_MISSING, workflow.DIRECTION, workflow.TOP_K,
             workflow.PREDICTIVE_MANIFEST) = saved

    def test_rerun_argv_passes_jobs_to_every_run_but_not_the_export(self) -> None:
        saved = (workflow.JOBS, workflow.DIRECTION, workflow.TOP_K,
                 workflow.PREDICTIVE_MANIFEST)
        try:
            workflow.DIRECTION, workflow.TOP_K, workflow.PREDICTIVE_MANIFEST = "predictive", None, "sel.json"
            self.assertEqual(workflow.JOBS, workflow.DEFAULT_JOBS)
            argv = workflow.rerun_argv("run.json", "cache/a.pkl")
            self.assertEqual(argv[argv.index("--jobs") + 1], str(workflow.DEFAULT_JOBS))
            self.assertNotIn("--jobs", workflow.rerun_argv(
                "run.json", "cache/a.pkl", export_dir_rel="run.rerun"))
            workflow.JOBS = 1
            argv = workflow.rerun_argv("run.json", "cache/a.pkl", extra_only=True)
            self.assertEqual(argv[argv.index("--jobs") + 1], "1")
        finally:
            (workflow.JOBS, workflow.DIRECTION, workflow.TOP_K,
             workflow.PREDICTIVE_MANIFEST) = saved

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
        # Round 1 may touch only UNUSABLE scripts, so it skips itself without any.
        self.assertTrue(
            str(first_fix["skip_if_no_unusable_in"]).endswith(".reproduce-raw.json")
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
        # The numeric diff runs right before the fold, which starts from it, and
        # every compute step writes its full outputs beside its JSON.
        names = [str(s["name"]) for s in steps]
        self.assertEqual(names.index("reproduce-diff") + 1, names.index("reproduce-fold"))
        diff = by_name["reproduce-diff"]
        self.assertEqual(diff["argv"][:2], ["python", "agentic/output_diff.py"])
        self.assertTrue(str(diff["out"]).endswith(".reproduce.json.diff.json"))
        self.assertEqual(diff["expects_file"], diff["out"])
        self.assertEqual(by_name["verify-slice"]["expects_file"], by_name["verify-slice"]["out"])
        self.assertIn(diff["argv"][diff["argv"].index("--outputs-dir") + 1],
                      by_name["reproduce-run"]["argv"])
        for step_name in ("reproduce-run", "reproduce-remeasure", "reproduce-remeasure-2",
                          "extrapolate-run", "fix-tests-measure"):
            argv = by_name[step_name]["argv"]
            self.assertIn("--outputs-dir", argv, step_name)
        self.assertNotIn("--outputs-dir", export["argv"])
        fold = by_name["reproduce-fold"]
        self.assertIn(str(diff["out"]), fold["instruction"])
        # Every fold step declares its verdicts file and asks the agent for it.
        for step_name, phase in (("reproduce-fold", "reproduce"),
                                 ("extrapolate-fold", "generalize"),
                                 ("verify-statistics-and-logic", "verify"),
                                 ("fix-tests-fold", "fix_tests")):
            spec = by_name[step_name]["verdicts"]
            self.assertEqual(spec["phase"], phase)
            self.assertTrue(str(spec["file"]).endswith(f".{phase}-verdicts.json"))
            if "instruction" in by_name[step_name]:
                self.assertIn(str(spec["file"]), by_name[step_name]["instruction"])
        self.assertTrue(by_name["fix-tests-fold"]["verdicts"]["corrected_only"])
        self.assertEqual(by_name["fix-tests-author"]["fix_tests_batches"]["verdicts"],
                         workflow.verdicts_path("autodiscovery/run.json"))
        verify = by_name["verify-statistics-and-logic"]
        self.assertEqual(names.index("verify-slice") + 1, names.index("verify-statistics-and-logic"))
        self.assertEqual(by_name["verify-slice"]["argv"][:2], ["python", "agentic/rank_by_surprise.py"])
        self.assertIn("--include-code", by_name["verify-slice"]["argv"])
        self.assertEqual(verify["verify_batches"]["slice"], by_name["verify-slice"]["out"])
        self.assertEqual(verify["verify_batches"]["verdicts_file"], verify["verdicts"]["file"])
        self.assertNotIn("instruction", verify)
        batch = workflow.verify_batch_instruction(verify["verify_batches"], ["3", "4"], 1, 2)
        self.assertIn(workflow.CANONICAL_STATS_REL, batch)
        self.assertIn(str(verify["verdicts"]["file"]), batch)
        # Every fix round tells the agent to repair a swallowed read and a
        # library form this environment rejects.
        for name in ["reproduce-fix-loading"] + [
            f"reproduce-fix-loading-{n}"
            for n in range(2, workflow.MAX_FIX_LOADING_ROUNDS + 1)
        ]:
            self.assertIn(workflow.VERSION_SKEW_RULE, by_name[name]["instruction"])
            self.assertIn(workflow.MISSING_PACKAGE_RULE, by_name[name]["instruction"])

    def test_baseline_cards_reach_the_verify_and_author_instructions(self) -> None:
        saved = (workflow.BASELINE_CARDS, workflow.DIRECTION, workflow.TOP_K,
                 workflow.PREDICTIVE_MANIFEST)
        card = {"companion": "rerank_companion_x_merge.npz", "brain": "x", "kind": "merge",
                "baseline": "degree", "n_rows": 100, "n_labeled": 40, "n_pos": 4,
                "random_rate": 0.1, "baseline_ap": 0.0278,
                "folds": {"0": {"ap": 0.02, "n_labeled": 20, "n_pos": 2}},
                "precision": {"0.01": [1, 2]}, "labeled_share": {"0.01": 0.2},
                "labeled_share_overall": 0.1}
        try:
            workflow.DIRECTION, workflow.TOP_K, workflow.PREDICTIVE_MANIFEST = "predictive", None, "sel.json"
            workflow.BASELINE_CARDS = {}
            self.assertEqual(workflow.baseline_rule(), "")
            workflow.BASELINE_CARDS = {"cache/x.pkl": card}
            rule = workflow.baseline_rule()
            self.assertIn("BASELINE CARD (rerank_companion_x_merge.npz, brain x, merge)", rule)
            self.assertIn("0.0278", rule)
            steps = workflow.build_steps("run.json", "cache/x.pkl", "run.summary.md", [])
            by_name = {step["name"]: step for step in steps}
            self.assertIn("BASELINE CARD", workflow.verify_batch_instruction(
                by_name["verify-statistics-and-logic"]["verify_batches"], ["3"], 1, 1))
            author = workflow.fix_tests_author_instruction(
                "run.summary.md", "run.json.rerun", "run.json.fixed", ["3"], [])
            self.assertIn("BASELINE CARD", author)
            self.assertIn("import canonical_stats", author)
            self.assertIn("canonical_stats.report(", author)
        finally:
            (workflow.BASELINE_CARDS, workflow.DIRECTION, workflow.TOP_K,
             workflow.PREDICTIVE_MANIFEST) = saved

    def test_recorded_audit_reaches_selection_and_verify_instructions(self) -> None:
        saved = workflow.RECORDED_AUDIT, workflow.RECORDED_AUDIT_REL
        try:
            workflow.RECORDED_AUDIT, workflow.RECORDED_AUDIT_REL = {}, None
            self.assertEqual(workflow.recorded_audit_rule(), "")
            workflow.RECORDED_AUDIT = {"n": 3, "n_suspected": 1, "suspected_ids": ["46"],
                                       "entries": {}}
            workflow.RECORDED_AUDIT_REL = "run.json.predictive.recorded-audit.json"
            selection = workflow.predictive_selection_instruction("run.json", "sel.json")
            self.assertIn("RECORDED-DATA AUDIT", selection)
            self.assertIn("Flagged ids: 46 (1 of 3)", selection)
            spec = {"summary": "s.md", "slice": "slice.json", "rerun_dir": "d",
                    "verdicts_file": "v.json"}
            self.assertIn("Flagged ids: 46", workflow.verify_batch_instruction(spec, ["46"], 1, 1))
        finally:
            workflow.RECORDED_AUDIT, workflow.RECORDED_AUDIT_REL = saved

    def test_failure_brief_names_a_swallowed_read(self) -> None:
        old_root = workflow.PROJECT_ROOT
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                workflow.PROJECT_ROOT = Path(tmpdir)
                (Path(tmpdir) / "r.json").write_text(json.dumps({"results": [{
                    "id": 33,
                    "result_status": "UNUSABLE",
                    "result_failure_reason": "swallowed-input-error",
                    "code_source": "recorded",
                    "rerun_stderr": "warning\n[runner] swallowed input error: "
                                    "tensorstore: ValueError: extra members\n",
                }]}))
                brief = workflow._failure_brief("r.json", "run.json.rerun")
        finally:
            workflow.PROJECT_ROOT = old_root
        self.assertIn(
            "id 33: swallowed-input-error (recorded) — [runner] swallowed input "
            "error: tensorstore: ValueError: extra members",
            brief,
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


_FIX_REPORT = """# Run

## Header

## Ranked Conclusions (highest priority first)

### 1. (Priority 0.5 · Surprise 0.7) Severe
- **Run:** r · **ID:** 10 · **Belief:** x
- **Verdict:** MAJOR
- **Statistical issues:** Row-level bootstrap.

### 2. (Priority 0.4 · Surprise 0.6) Minor with a test fault
- **Run:** r · **ID:** 11 · **Belief:** x
- **Verdict:** MINOR
- **Statistical issues:** Concrete test fault: no paired bootstrap.

### 3. (Priority 0.3 · Surprise 0.5) Minor without one
- **Run:** r · **ID:** 12 · **Belief:** x
- **Verdict:** MINOR
- **Statistical issues:** Run-wide caveats only; no concrete test fault.

### 4. (Priority 0.2 · Surprise 0.4) Sound
- **Run:** r · **ID:** 13 · **Belief:** x
- **Verdict:** SOUND
- **Statistical issues:** Concrete test fault: none that matters.

### 5. (Priority 0.1 · Surprise 0.3) Critical
- **Run:** r · **ID:** 14 · **Belief:** x
- **Verdict:** CRITICAL

### 6. (Priority 0.0 · Surprise 0.2) Not audited
- **Run:** r · **ID:** 15 · **Belief:** x

## Statistical Verification — Summary

### Not an entry
- **ID:** 99
- **Verdict:** MAJOR
"""


class StructuredVerdictTests(unittest.TestCase):
    """The fold steps' verdicts JSON: schema, vocabulary, ids, report agreement."""

    REPORT = """# Run

## Header

## Ranked Conclusions (highest priority first)

### 1. (Priority 0.5 · Surprise 0.7) one
- **Run:** r · **ID:** 10 · **Belief:** x
- **Reproduction:** REPRODUCED (code: recorded)
- **Verdict:** MAJOR
- **Deployability:** BLIND-COMPUTABLE — graph only

### 2. (Priority 0.4 · Surprise 0.6) two
- **Run:** r · **ID:** 11 · **Belief:** x
- **Reproduction:** FAILED (code: revised-loading)
- **Verdict:** MINOR
- **Statistical issues:** Concrete test fault: no paired bootstrap.
- **Deployability:** GT-REFERENCING — distance to gt_graph

## Reproduction — Summary
"""

    def setUp(self) -> None:
        self._old = (workflow.PROJECT_ROOT, workflow.DIRECTION)
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        workflow.PROJECT_ROOT = self.root
        workflow.DIRECTION = "predictive"
        (self.root / "run.summary.md").write_text(self.REPORT)
        (self.root / "run.json.predictive.reproduce.json").write_text(json.dumps(
            {"results": [{"id": 10, "result_status": "USABLE"},
                         {"id": 11, "result_status": "UNUSABLE"}]}))
        self.spec = {"phase": "reproduce", "file": "run.json.predictive.reproduce-verdicts.json",
                     "ids_from": "run.json.predictive.reproduce.json"}

    def tearDown(self) -> None:
        workflow.PROJECT_ROOT, workflow.DIRECTION = self._old
        self._tmp.cleanup()

    def _write(self, payload: dict, name: str | None = None) -> None:
        (self.root / (name or self.spec["file"])).write_text(json.dumps(payload))

    def test_good_file_passes_and_merges(self) -> None:
        self._write({"phase": "reproduce", "entries": {
            "10": {"verdict": "REPRODUCED", "code_source": "recorded"},
            "11": {"verdict": "FAILED", "code_source": "revised"}}})
        self.assertEqual(workflow._check_phase_verdicts(self.spec, self.root / "run.summary.md"), [])
        with contextlib.redirect_stdout(io.StringIO()):
            workflow._merge_phase_verdicts(self.spec, "run.json")
        merged = json.loads((self.root / "run.json.predictive.verdicts.json").read_text())
        self.assertEqual(merged["schema_version"], workflow.VERDICTS_SCHEMA_VERSION)
        self.assertEqual(merged["entries"]["11"]["reproduce"]["verdict"], "FAILED")
        self.assertEqual(merged["phases"]["reproduce"]["n"], 2)
        # a second phase merges beside the first
        verify_spec = {"phase": "verify", "file": "run.json.predictive.verify-verdicts.json",
                       "ids_from": "run.json.predictive.reproduce.json"}
        self._write({"phase": "verify", "entries": {
            "10": {"verdict": "MAJOR", "test_fault": False, "deployability": "BLIND-COMPUTABLE",
                   "headline": {"p": 0.002, "delta_ap": 0.024, "ci95": [0.01, 0.04],
                                "baseline": "raw degree"}},
            "11": {"verdict": "MINOR", "test_fault": True, "deployability": "GT-REFERENCING",
                   "headline": None}}}, verify_spec["file"])
        self.assertEqual(workflow._check_phase_verdicts(verify_spec, self.root / "run.summary.md"), [])
        with contextlib.redirect_stdout(io.StringIO()):
            workflow._merge_phase_verdicts(verify_spec, "run.json")
        merged = json.loads((self.root / "run.json.predictive.verdicts.json").read_text())
        self.assertEqual(merged["entries"]["10"]["reproduce"]["verdict"], "REPRODUCED")
        self.assertEqual(merged["entries"]["10"]["verify"]["verdict"], "MAJOR")
        # fix-tests selection now comes from the merged file
        selected, unjudged = workflow.fix_tests_selection(
            self.root / "run.summary.md", self.root / "run.json.predictive.verdicts.json")
        self.assertEqual(selected, [("10", "MAJOR"), ("11", "MINOR")])
        self.assertEqual(unjudged, [])

    def test_fresh_run_resets_previous_verdict_files_and_survey_lists_them(self) -> None:
        merged = self.root / "run.json.predictive.verdicts.json"
        merged.write_text(json.dumps({"schema_version": 1, "run": "run.json",
                                      "entries": {"99": {"verify": {"verdict": "MAJOR"}}}}))
        phase = self.root / "run.json.predictive.verify-verdicts.json"
        phase.write_text("{}")
        (self.root / "run.json.predictive.reproduce.json.outputs").mkdir()
        (self.root / "run.json.predictive.reproduce.json.diff.json").write_text("{}")
        (self.root / "run.json").write_text("[]")
        reasons = {path.name: reason for path, reason in
                   workflow.survey_prior_artifacts(self.root / "run.json", "predictive")}
        self.assertIn("run.json.predictive.verdicts.json", reasons)
        self.assertIn("run.json.predictive.verify-verdicts.json", reasons)
        self.assertIn("run.json.predictive.reproduce.json.outputs", reasons)
        self.assertIn("run.json.predictive.reproduce.json.diff.json", reasons)
        with contextlib.redirect_stdout(io.StringIO()):
            workflow._reset_verdicts("run.json")
        self.assertFalse(merged.exists())
        self.assertFalse(phase.exists())
        # selection never picks an id the report no longer carries
        merged.write_text(json.dumps({"schema_version": 1, "run": "run.json", "entries": {
            "10": {"verify": {"verdict": "MAJOR", "test_fault": False}},
            "99": {"verify": {"verdict": "CRITICAL", "test_fault": False}}}}))
        selected, _ = workflow.fix_tests_selection(self.root / "run.summary.md", merged)
        self.assertEqual(selected, [("10", "MAJOR")])

    def test_problems_are_named(self) -> None:
        self.assertEqual(workflow._check_phase_verdicts(self.spec, self.root / "run.summary.md"),
                         [f"{self.spec['file']} was not written"])
        self._write({"phase": "reproduce", "entries": {
            "10": {"verdict": "DIVERGED"},            # report says REPRODUCED
            "12": {"verdict": "REPRODUCED"}}})        # not measured; 11 missing
        problems = workflow._check_phase_verdicts(self.spec, self.root / "run.summary.md")
        self.assertTrue(any("missing entries for ids 11" in p for p in problems))
        self.assertTrue(any("did not measure: 12" in p for p in problems))
        self.assertTrue(any("JSON says DIVERGED but the report's Reproduction bullet says "
                            "REPRODUCED" in p for p in problems))
        self._write({"phase": "reproduce", "entries": {
            "10": {"verdict": "REPRODUCED"}, "11": {"verdict": "MAYBE"}}})
        problems = workflow._check_phase_verdicts(self.spec, self.root / "run.summary.md")
        self.assertEqual(len(problems), 1)
        self.assertIn("'MAYBE' is not one of", problems[0])
        repair = workflow._verdicts_repair_instruction(self.spec, problems, "run.summary.md")
        self.assertIn("'MAYBE' is not one of", repair)
        self.assertIn("report's bullets are authoritative", repair)

    def test_verify_phase_checks_test_fault_and_deployability(self) -> None:
        spec = {"phase": "verify", "file": "run.json.predictive.verify-verdicts.json",
                "ids_from": "run.json.predictive.reproduce.json"}
        self._write({"phase": "verify", "entries": {
            "10": {"verdict": "MAJOR", "test_fault": "no", "deployability": "GT-REFERENCING"},
            "11": {"verdict": "MINOR", "test_fault": True, "deployability": "BLIND"}}},
            spec["file"])
        problems = workflow._check_phase_verdicts(spec, self.root / "run.summary.md")
        self.assertTrue(any("test_fault must be true or false" in p for p in problems))
        self.assertTrue(any("JSON deployability GT-REFERENCING but the report says "
                            "BLIND-COMPUTABLE" in p for p in problems))
        self.assertTrue(any("deployability 'BLIND' is not one of" in p for p in problems))

    def test_fix_tests_phase_covers_corrected_results_only_and_allows_null(self) -> None:
        (self.root / "run.json.predictive.corrected.json").write_text(json.dumps(
            {"results": [{"id": 10, "code_source": "corrected", "result_status": "UNUSABLE"},
                         {"id": 11, "code_source": "recorded", "result_status": "USABLE"}]}))
        spec = {"phase": "fix_tests", "file": "run.json.predictive.fix_tests-verdicts.json",
                "ids_from": "run.json.predictive.corrected.json", "corrected_only": True}
        self._write({"phase": "fix_tests", "entries": {
            "10": {"verdict": None, "generalization": None}}}, spec["file"])
        self.assertEqual(workflow._check_phase_verdicts(spec, self.root / "run.summary.md"), [])
        self._write({"phase": "fix_tests", "entries": {
            "10": {"verdict": "UPHELD", "generalization": "SOMETIMES"},
            "11": {"verdict": "UPHELD"}}}, spec["file"])
        problems = workflow._check_phase_verdicts(spec, self.root / "run.summary.md")
        self.assertTrue(any("did not measure: 11" in p for p in problems))
        self.assertTrue(any("generalization 'SOMETIMES' is not one of" in p for p in problems))
        self.assertTrue(any("the report has no Post-correction verdict bullet" in p
                            for p in problems))


class _VerifyClient:
    """Audits every id a batch turn names; the synthesis turn writes the section."""

    def __init__(self, root: Path, verdict: str = "WEAK", p: float = 0.2) -> None:
        self.root, self.verdict, self.p, self.prompts = root, verdict, p, []

    async def query(self, prompt, session_id="default"):
        self.prompts.append(prompt)
        summary = self.root / "run.summary.md"
        text = summary.read_text()
        match = re.search(r"exactly these hypothesis ids: ([^(]+) \(batch", prompt)
        if match:
            ids = match.group(1).strip().split(", ")
            verdicts_path = self.root / "run.json.predictive.verify-verdicts.json"
            payload = (json.loads(verdicts_path.read_text()) if verdicts_path.is_file()
                       else {"phase": "verify", "entries": {}})
            for rid in ids:
                text = text.replace(
                    f"- **Run:** r · **ID:** {rid} · x\n",
                    f"- **Run:** r · **ID:** {rid} · x\n- **Verdict:** {self.verdict}\n"
                    "- **Deployability:** BLIND-COMPUTABLE — graph\n",
                )
                payload["entries"][rid] = {
                    "verdict": self.verdict, "test_fault": False,
                    "deployability": "BLIND-COMPUTABLE",
                    "headline": {"p": self.p, "delta_ap": 0.01, "ci95": [0.0, 0.02],
                                 "baseline": "raw degree"},
                }
            verdicts_path.write_text(json.dumps(payload))
            summary.write_text(text)
        elif "run-wide synthesis" in prompt:
            summary.write_text(text + "\n## Statistical Verification — Summary\n\nfigures\n")

    async def receive_response(self):
        if False:
            yield None


class VerifyBatchTests(unittest.TestCase):
    REPORT = "# Run\n\n## Header\n\n## Ranked Conclusions (highest priority first)\n\n" + "".join(
        f"### {k}. (Priority 0.{9 - k} · Surprise 0.5) hypothesis {rid}\n"
        f"- **Run:** r · **ID:** {rid} · x\n\n"
        for k, rid in enumerate(("10", "11", "12", "13", "14"), start=1)
    ) + "## Reproduction — Summary\n"

    def setUp(self) -> None:
        self._old = (workflow.PROJECT_ROOT, workflow.DIRECTION, workflow.VERIFY_BATCH_SIZE)
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        workflow.PROJECT_ROOT = self.root
        workflow.DIRECTION = "predictive"
        workflow.VERIFY_BATCH_SIZE = 2
        (self.root / "run.summary.md").write_text(self.REPORT)
        (self.root / "run.json.predictive.reproduce.json").write_text(json.dumps(
            {"results": [{"id": int(rid), "result_status": "USABLE"}
                         for rid in ("10", "11", "12", "13", "14")]}))
        (self.root / "run.json.predictive.verify-slice.json").write_text(json.dumps(
            {"n_total": 8, "records": []}))
        self.spec = {
            "summary": "run.summary.md", "slice": "run.json.predictive.verify-slice.json",
            "rerun_dir": "run.json.predictive.rerun",
            "verdicts_file": "run.json.predictive.verify-verdicts.json",
            "ids_from": "run.json.predictive.reproduce.json",
            "ledger": "run.json.predictive.verify-ledger.json",
            "mc": "run.json.predictive.multiple-comparisons.json",
        }

    def tearDown(self) -> None:
        workflow.PROJECT_ROOT, workflow.DIRECTION, workflow.VERIFY_BATCH_SIZE = self._old
        self._tmp.cleanup()

    def _run(self, client) -> None:
        with contextlib.redirect_stdout(io.StringIO()):
            asyncio.run(workflow.run_verify_batches(client, self.spec))

    def test_batches_then_synthesis_with_the_drivers_bh_figures(self) -> None:
        client = _VerifyClient(self.root, p=0.004)
        self._run(client)
        batch_prompts = [p for p in client.prompts if "(batch " in p]
        self.assertEqual(len(batch_prompts), 3)  # 2 + 2 + 1
        self.assertIn("exactly these hypothesis ids: 10, 11 (batch 1 of 3", batch_prompts[0])
        self.assertIn("exactly these hypothesis ids: 14 (batch 3 of 3", batch_prompts[2])
        self.assertIn("do NOT write the run-wide summary section", batch_prompts[0])
        self.assertIn("run.json.predictive.verify-slice.json", batch_prompts[0])
        synthesis = client.prompts[-1]
        self.assertIn("run-wide synthesis", synthesis)
        self.assertIn("5 audited: SOUND 0, WEAK 5, MINOR 0, MAJOR 0, CRITICAL 0", synthesis)
        self.assertIn("BH at q = 0.05 over m = 5", synthesis)
        self.assertIn("5 survive", synthesis)
        self.assertIn("With m = 8", synthesis)
        mc = json.loads((self.root / self.spec["mc"]).read_text())
        self.assertEqual(sorted(mc["survivors"]), ["10", "11", "12", "13", "14"])
        self.assertEqual(mc["m_total"], 8)
        ledger = json.loads((self.root / self.spec["ledger"]).read_text())["ids"]
        self.assertEqual(sorted(ledger), ["10", "11", "12", "13", "14"])
        self.assertIn("## Statistical Verification — Summary",
                      (self.root / "run.summary.md").read_text())

    def test_resume_audits_only_the_open_ids(self) -> None:
        first = _VerifyClient(self.root)
        self._run(first)
        # drop one entry's bullets and ledger line: only that id is audited again
        summary = self.root / "run.summary.md"
        text = summary.read_text().replace(
            "- **Run:** r · **ID:** 12 · x\n- **Verdict:** WEAK\n"
            "- **Deployability:** BLIND-COMPUTABLE — graph\n",
            "- **Run:** r · **ID:** 12 · x\n")
        summary.write_text(text.split("## Statistical Verification — Summary")[0])
        ledger_path = self.root / self.spec["ledger"]
        ledger = json.loads(ledger_path.read_text())
        ledger["ids"].pop("12")
        ledger_path.write_text(json.dumps(ledger))
        second = _VerifyClient(self.root)
        self._run(second)
        batch_prompts = [p for p in second.prompts if "(batch " in p]
        self.assertEqual(len(batch_prompts), 1)
        self.assertIn("exactly these hypothesis ids: 12 (batch 1 of 1", batch_prompts[0])

    def test_multiple_comparisons_is_benjamini_hochberg(self) -> None:
        entries = {rid: {"verify": {"headline": {"p": p}}} for rid, p in
                   (("a", 0.001), ("b", 0.012), ("c", 0.03), ("d", 0.2), ("e", 0.6))}
        entries["f"] = {"verify": {"headline": None}}
        mc = workflow.multiple_comparisons(entries, q=0.05, m_total=10)
        # thresholds k*q/m: 0.01, 0.02, 0.03, 0.04, 0.05 → c passes at k=3
        self.assertEqual(mc["survivors"], ["a", "b", "c"])
        self.assertEqual(mc["m"], 5)
        self.assertEqual(mc["missing_ids"], ["f"])
        self.assertEqual(mc["n_nominal"], 3)
        self.assertEqual(mc["borderline"], [])
        # with m = 10 the thresholds halve: 0.005, 0.01, 0.015 → only a
        self.assertEqual(mc["survivors_m_total"], ["a"])
        self.assertEqual(mc["n_at_floor"], 1)
        text = workflow.multiple_comparisons_text(mc)
        self.assertIn("3 survive (largest passing p 0.03): a, b, c", text)
        self.assertIn("With m = 10", text)
        none = workflow.multiple_comparisons({"x": {"verify": {"headline": {"p": 0.5}}}})
        self.assertEqual(none["survivors"], [])
        self.assertEqual(none["borderline"], [])


class _AuthorClient:
    """Writes a corrected script for every id an author turn names, except ``skip``."""

    def __init__(self, fixed_dir: Path, skip=()) -> None:
        self.fixed_dir, self.skip, self.prompts = fixed_dir, set(skip), []

    async def query(self, prompt, session_id="default"):
        self.prompts.append(prompt)
        named = re.search(r"exactly these hypothesis ids: ([^.]+)\.", prompt).group(1)
        for rid in named.split(", "):
            if rid not in self.skip:
                (self.fixed_dir / f"hypo_{rid}.py").write_text(f"print('fixed {rid}')\n")

    async def receive_response(self):
        if False:
            yield None


class FixTestsBatchTests(unittest.TestCase):
    def setUp(self) -> None:
        self._old_root = workflow.PROJECT_ROOT
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        workflow.PROJECT_ROOT = self.root
        (self.root / "run.summary.md").write_text(_FIX_REPORT)
        self.fixed = self.root / "run.json.fixed"
        self.spec = {"summary": "run.summary.md", "rerun_dir": "run.json.rerun",
                     "fixed_dir": "run.json.fixed"}

    def tearDown(self) -> None:
        workflow.PROJECT_ROOT = self._old_root
        self._tmp.cleanup()

    def _author(self, client) -> None:
        with mock.patch.object(workflow, "FIX_TESTS_BATCH_SIZE", 2), \
                contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            asyncio.run(workflow.run_fix_tests_author(client, self.spec))

    def test_split_runs_are_named_by_the_word_split(self) -> None:
        for name in ("split-error.json", "split-error-x.json",
                     "autodiscovery/new-split-1_2026-10-19.json"):
            self.assertTrue(workflow._is_split_run(name), name)
        for name in ("new-merge-1_2026-10-19.json", "splitting.json",
                     "split-vs-merge.json"):
            self.assertFalse(workflow._is_split_run(name), name)

    def test_selection_reads_verdicts_and_the_test_fault_marker(self) -> None:
        selected, unjudged = workflow.fix_tests_selection(self.root / "run.summary.md")
        self.assertEqual(selected, [("10", "MAJOR"), ("11", "MINOR"), ("14", "CRITICAL")])
        self.assertEqual(unjudged, ["15"])

    def test_author_runs_batches_records_them_and_resumes_open_ids_only(self) -> None:
        self.fixed.mkdir()
        (self.fixed / "hypo_12.py").write_text("print('not selected')\n")
        (self.fixed / "hypo_14.py").write_text("print('interrupted draft')\n")
        client = _AuthorClient(self.fixed, skip={"11"})
        self._author(client)

        self.assertEqual(len(client.prompts), 2)  # [10, 11] then [14]
        self.assertIn("ids: 10, 11.", client.prompts[0])
        self.assertIn("ids: 14.", client.prompts[1])
        self.assertIn("draft from an interrupted earlier turn", client.prompts[1])
        self.assertFalse((self.fixed / "hypo_12.py").exists())
        ledger = json.loads((self.fixed / workflow.FIX_TESTS_LEDGER).read_text())["ids"]
        self.assertEqual(ledger["11"], {"status": "no-fix"})
        self.assertEqual(sorted(ledger), ["10", "11", "14"])
        self.assertEqual(ledger["10"]["status"], "written")

        rerun_client = _AuthorClient(self.fixed)
        self._author(rerun_client)
        self.assertEqual(rerun_client.prompts, [])

        (self.fixed / "hypo_10.py").write_text("print('edited')\n")
        edited_client = _AuthorClient(self.fixed)
        self._author(edited_client)
        self.assertEqual(len(edited_client.prompts), 1)
        self.assertIn("ids: 10.", edited_client.prompts[0])

    def test_resume_index_requires_the_earlier_steps_to_have_finished(self) -> None:
        steps = [
            {"name": "summarize", "kind": "agent", "expects": ["## Ranked"]},
            {"name": "measure", "kind": "compute"},
            {"name": "fold", "kind": "agent", "expects": ["## Never written"]},
            {"name": "author", "kind": "agent"},
        ]
        summary = self.root / "run.summary.md"
        self.assertEqual(workflow.resume_index(steps, "measure", summary), 1)
        with self.assertRaisesRegex(SystemExit, "not a step"):
            workflow.resume_index(steps, "nope", summary)
        with self.assertRaisesRegex(SystemExit, "fold: '## Never written'"):
            workflow.resume_index(steps, "author", summary)

    def test_timed_out_turn_names_the_step_to_resume_from(self) -> None:
        class SlowClient:
            async def query(self, prompt, session_id="default"):
                pass

            async def receive_response(self):
                await asyncio.sleep(30)
                yield None

        with self.assertRaisesRegex(SystemExit, "--resume-from fix-tests-author"):
            asyncio.run(workflow.run_step(SlowClient(), {
                "name": "fix-tests-author-3", "resume_from": "fix-tests-author",
                "timeout_s": 1, "instruction": "x",
            }))


if __name__ == "__main__":
    unittest.main()
