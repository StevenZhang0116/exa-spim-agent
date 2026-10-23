"""The agent discovery driver: records, batching, the runner, both engines' glue and the budget loop."""
from __future__ import annotations

import io
import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from agentic import run_agent_discovery as ad

AD_FIELDS = ["id", "status", "hypothesis", "surprisal", "isSurprising", "prior", "posterior",
             "priorBelief", "posteriorBelief", "analysis", "review", "experimentPlan", "code",
             "codeOutput", "richOutputs", "runtimeMs", "createdAt"]


def write_record(work: Path, rid: int, *, result=True, exit_code=0, code="print('x')\n",
                 stdout="INPUTS READ fragment_nodes=3\nAP 0.5\n", stderr=""):
    (work / "records").mkdir(exist_ok=True)
    (work / "outputs").mkdir(exist_ok=True)
    plan = ({"objective": "rank and score", "steps": "1. load 2. score", "deliverables": "AP lines"}
            if rid % 2 else "rank and score")   # odd ids: AD-shaped object; even ids: a plain string
    (work / "records" / f"{rid}.proposal.json").write_text(json.dumps(
        {"hypothesis": f"Hypothesis {rid} says more is better.", "experimentPlan": plan}))
    (work / f"hypo_{rid}.py").write_text(code)
    (work / "outputs" / f"hypo_{rid}.stdout.txt").write_text(stdout)
    (work / "outputs" / f"hypo_{rid}.stderr.txt").write_text(stderr)
    (work / "outputs" / f"hypo_{rid}.meta.json").write_text(json.dumps(
        {"id": rid, "exit_code": exit_code, "runtime_ms": 1234, "started_at": "2026-10-22T10:00:00+00:00",
         "finished_at": "2026-10-22T10:00:01+00:00", "timeout_s": 60}))
    if result:
        (work / "records" / f"{rid}.result.json").write_text(json.dumps(
            {"analysis": "AP 0.5 beats nothing.", "review": "small n"}))


def fake_materials(root: Path, kind="merge", brain="1", mcl=100) -> None:
    (root / "markdowns").mkdir()
    (root / "cache").mkdir()
    (root / "markdowns" / f"{kind}_features_for_proofreader_evolve.md").write_text("# task\n")
    (root / "cache" / f"dataset_cache_{brain}_mcl{mcl}_add.pkl").write_bytes(b"pkl")
    (root / "markdowns" / f"rerank_companion_{brain}_{kind}.npz").write_bytes(b"npz")
    (root / "markdowns" / f"rerank_companion_{brain}_{kind}.json").write_text("{}")


class RecordTests(unittest.TestCase):
    def test_belief_fields_are_neutral_constants_in_ad_shape(self) -> None:
        b = ad.belief(ad.NEUTRAL_BELIEF)
        self.assertEqual(b["_type"], "boolean_cat")
        self.assertEqual(b["uncertain"], 5)
        self.assertEqual(sum(b[name] for name, _ in ad.BELIEF_BUCKETS), 5)
        self.assertEqual(b["mean"], 0.5)
        self.assertEqual(ad.belief(0.9167)["definitely_true"], 5)

    def test_assembled_record_has_every_ad_field(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            write_record(work, 1)
            write_record(work, 2, result=False, exit_code=1, stderr="Traceback: boom")
            write_record(work, 3, exit_code=124)
            (work / "outputs" / "hypo_1_figure.png").write_bytes(b"png")
            records = ad.assemble_export(work, 5, {"engine": "claude", "model": "claude-opus-5-5", "effort": "high"})
            self.assertEqual([r["id"] for r in records], [1, 2, 3])
            one = records[0]
            self.assertEqual(list(one.keys())[:17], AD_FIELDS)
            self.assertEqual((one["engine"], one["model"], one["effort"]), ("claude", "claude-opus-5-5", "high"))
            self.assertEqual(one["status"], "SUCCEEDED")
            self.assertEqual(one["code"], "print('x')\n")
            self.assertIn("INPUTS READ", one["codeOutput"])
            self.assertEqual(one["runtimeMs"], 1234)
            self.assertEqual(one["createdAt"], "2026-10-22T10:00:01+00:00")
            self.assertEqual(one["richOutputs"], [{"image/png": "cG5n"}])   # base64 of b"png", AD-style
            self.assertEqual(one["experimentPlan"], {"objective": "rank and score", "steps": "1. load 2. score",
                                                     "deliverables": "AP lines"})
            self.assertEqual(records[1]["experimentPlan"],
                             {"objective": "rank and score", "steps": "", "deliverables": ""})
            self.assertEqual((one["prior"], one["posterior"], one["surprisal"], one["isSurprising"]),
                             (0.5, 0.5, 0.0, False))        # neutral constants, never asked of the agent
            two = records[1]
            self.assertEqual(two["status"], "FAILED")      # no result file, non-zero exit
            self.assertIn("STDERR:\nTraceback", two["codeOutput"])
            self.assertEqual(two["analysis"], "")
            self.assertEqual(records[2]["status"], "FAILED")  # killed by the timeout despite a result file
            json.dumps(records)


class BatchingTests(unittest.TestCase):
    def test_next_batch_finishes_proposed_ids_before_new_ones(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            write_record(work, 1)
            write_record(work, 2, result=False)
            self.assertEqual(ad.record_state(work, 1), "complete")
            self.assertEqual(ad.record_state(work, 2), "proposed")
            self.assertEqual(ad.record_state(work, 3), "absent")
            self.assertEqual(ad.next_batch(work, 6, 3), [2, 3, 4])
            self.assertEqual(ad.next_batch(work, 2, 3), [2])
            write_record(work, 2)
            self.assertEqual(ad.next_batch(work, 2, 3), [])
            ledger = ad.write_ledger(work, 6).read_text()
            self.assertIn("1 [done]: Hypothesis 1", ledger)
            self.assertNotIn("prior", ledger)
            instruction = ad.batch_instruction(work, [3, 4], "merge", "1", 100)
            self.assertIn("ids [3, 4]", instruction)
            self.assertIn("rerank_companion_1_merge.npz", instruction)
            self.assertIn(f"{work}/run_hypo.sh <id>", instruction)
            self.assertIn(f"{work}/records/<id>.proposal.json", instruction)
            self.assertNotIn("subagent", instruction)   # engine-independent wording

    def test_export_is_never_emptied_or_shrunk(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake_materials(root)
            auto = root / "autodiscovery"
            work = auto / "claude-merge-t.claude"
            (work / "records").mkdir(parents=True)
            export = auto / "claude-merge-t.json"
            export.write_text(json.dumps([{"id": 1}, {"id": 2}]))
            args = ["--kind", "merge", "--brain", "1", "--run-name", "claude-merge-t", "--hypotheses", "3",
                    "--assemble-only", "--autodiscovery-dir", str(auto)]
            with patch.object(ad, "PROJECT_ROOT", root), redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(SystemExit, "no records"):
                    ad.main(args)
                write_record(work, 1)
                with self.assertRaisesRegex(SystemExit, "refusing to shrink"):
                    ad.main(args)
                write_record(work, 2)
                write_record(work, 3)
                ad.main(args)
            self.assertEqual(len(json.loads(export.read_text())), 3)

    def test_run_name_kind_follows_the_downstream_token_rule(self) -> None:
        self.assertEqual(ad.run_name_kind("claude-merge-1_2026-10-22"), "merge")
        self.assertEqual(ad.run_name_kind("codex-split-1_2026-10-22"), "split")
        self.assertEqual(ad.run_name_kind("new-split-1_2026-10-20"), "split")
        self.assertIsNone(ad.run_name_kind("claudemerge1"))      # not a token
        self.assertIsNone(ad.run_name_kind("split-merge-1"))     # ambiguous
        for bad in ("claude-1", "claudesplit-1", "split-merge-1"):
            with redirect_stdout(io.StringIO()), self.assertRaises(SystemExit):
                ad.main(["--kind", "split", "--brain", "1", "--run-name", bad, "--assemble-only"])


class RunnerTests(unittest.TestCase):
    def _install(self, work: Path, timeout: int = 30) -> None:
        (work / "run_hypo.sh").write_text(ad.RUN_HYPO_SH.replace("@TIMEOUT@", str(timeout)).replace("@PYTHON@", sys.executable))
        (work / "run_hypo.sh").chmod(0o755)
        (work / "wait_hypo.sh").write_text(ad.WAIT_HYPO_SH)
        (work / "wait_hypo.sh").chmod(0o755)

    def test_runner_starts_detached_and_waiter_reports_the_outcome(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            self._install(work)
            (work / "hypo_7.py").write_text(
                "import sys, time\ntime.sleep(2)\nprint('INPUTS READ ok')\nprint('warn', file=sys.stderr)\nsys.exit(3)\n")
            started = subprocess.run([str(work / "run_hypo.sh"), "7"], capture_output=True, text=True, cwd=tmp)
            self.assertEqual(started.returncode, 0)
            self.assertIn("started hypo_7.py", started.stdout)
            self.assertTrue((work / "outputs" / "hypo_7.pid").is_file())
            again = subprocess.run([str(work / "run_hypo.sh"), "7"], capture_output=True, text=True, cwd=tmp)
            self.assertEqual(again.returncode, 2)          # already running: refuse a second start
            waited = subprocess.run([str(work / "wait_hypo.sh"), "7", "60"], capture_output=True, text=True, cwd=tmp)
            self.assertEqual(waited.returncode, 0, waited.stdout + waited.stderr)
            self.assertIn("done:", waited.stdout)
            self.assertIn("INPUTS READ ok", waited.stdout)
            meta = json.loads((work / "outputs" / "hypo_7.meta.json").read_text())
            self.assertEqual(meta["exit_code"], 3)
            self.assertGreaterEqual(meta["runtime_ms"], 2000)
            self.assertTrue(meta["finished_at"].endswith("+00:00"))
            self.assertEqual((work / "outputs" / "hypo_7.stdout.txt").read_text(), "INPUTS READ ok\n")
            self.assertEqual((work / "outputs" / "hypo_7.stderr.txt").read_text(), "warn\n")
            self.assertFalse((work / "outputs" / "hypo_7.pid").exists())
            never = subprocess.run([str(work / "wait_hypo.sh"), "8", "5"], capture_output=True, text=True, cwd=tmp)
            self.assertEqual(never.returncode, 4)          # never started

    def test_waiter_returns_still_running_within_its_budget(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            self._install(work)
            (work / "hypo_9.py").write_text("import time\ntime.sleep(12)\n")
            subprocess.run([str(work / "run_hypo.sh"), "9"], capture_output=True, text=True, cwd=tmp, check=True)
            waited = subprocess.run([str(work / "wait_hypo.sh"), "9", "5"], capture_output=True, text=True, cwd=tmp)
            self.assertEqual(waited.returncode, 3)
            self.assertIn("still running", waited.stdout)
            done = subprocess.run([str(work / "wait_hypo.sh"), "9", "60"], capture_output=True, text=True, cwd=tmp)
            self.assertEqual(done.returncode, 0)


class CodexGlueTests(unittest.TestCase):
    def test_codex_command_and_prompt(self) -> None:
        argv = ad.codex_command(Path("/w"), "gpt-6-astra", "high", Path("/w/codex/last.txt"), "bypass")
        self.assertEqual(argv[1:3], ["exec", "--json"])
        self.assertIn("--skip-git-repo-check", argv)
        self.assertEqual(argv[argv.index("-C") + 1], "/w")
        self.assertEqual(argv[argv.index("-m") + 1], "gpt-6-astra")
        self.assertIn('model_reasoning_effort="high"', argv)
        self.assertIn("--dangerously-bypass-approvals-and-sandbox", argv)
        self.assertIn("project_doc_max_bytes=0", argv)
        self.assertEqual(argv[-1], "-")
        sandboxed = ad.codex_command(Path("/w"), "m", "low", Path("/w/l.txt"), "workspace-write")
        self.assertNotIn("--dangerously-bypass-approvals-and-sandbox", sandboxed)
        self.assertEqual(sandboxed[sandboxed.index("-s") + 1], "workspace-write")
        instructions = ad.researcher_instructions()
        self.assertFalse(instructions.startswith("---"))
        self.assertIn("# Discovery Researcher", instructions)
        self.assertIn("proposal.json", instructions)
        prompt = ad.session_prompt(Path("/w"), [1, 2], "merge", "1", 100)
        self.assertTrue(prompt.startswith("# Discovery Researcher"))
        self.assertIn("# This batch", prompt)
        self.assertIn("ids [1, 2]", prompt)
        self.assertIn("no subagent, no background shell command, no file watch", prompt)

    def test_codex_events_are_summed_and_priced(self) -> None:
        lines = [
            '{"type":"thread.started","thread_id":"t"}',
            '{"type":"item.completed","item":{"type":"command_execution","command":"ls","exit_code":0}}',
            '{"type":"item.completed","item":{"type":"agent_message","text":"done 1"}}',
            '{"type":"turn.completed","usage":{"input_tokens":18759,"cached_input_tokens":0,'
            '"cache_write_input_tokens":0,"output_tokens":5,"reasoning_output_tokens":2}}',
            'not json',
            '{"type":"turn.completed","usage":{"input_tokens":1,"output_tokens":1}}',
        ]
        parsed = ad.parse_codex_events(lines)
        self.assertEqual(parsed["usage"]["input_tokens"], 18760)
        self.assertEqual(parsed["usage"]["output_tokens"], 6)
        self.assertEqual(parsed["usage"]["reasoning_output_tokens"], 2)
        self.assertEqual(parsed["turns"], 2)
        self.assertEqual(parsed["commands"], 1)
        self.assertEqual(parsed["messages"], ["done 1"])
        self.assertAlmostEqual(ad.codex_cost(parsed["usage"], 2.0, 10.0), (18760 * 2.0 + 8 * 10.0) / 1e6)
        self.assertEqual(ad.codex_cost(parsed["usage"], 0.0, 0.0), 0.0)


class DriverLoopTests(unittest.TestCase):
    def test_loop_batches_resumes_and_stops_on_budget(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake_materials(root)
            calls = []

            async def fake_batch(work_dir, ids, kind, brain, mcl, *, model, effort, verbose,
                                 max_budget_usd, timeout_s):
                calls.append((list(ids), round(max_budget_usd, 2), model, effort))
                for rid in ids[:-1] if len(ids) > 1 else ids:  # last id of a batch left unfinished
                    write_record(work_dir, rid)
                if len(ids) > 1:
                    write_record(work_dir, ids[-1], result=False)
                return {"cost_usd": 4.0, "tokens": None}

            with patch.object(ad, "PROJECT_ROOT", root), patch.object(ad, "run_batch_claude", fake_batch), \
                    redirect_stdout(io.StringIO()):
                code = ad.main(["--kind", "merge", "--brain", "1", "--run-name", "claude-merge-t",
                                "--hypotheses", "7", "--batch", "3", "--max-cost-usd", "10",
                                "--autodiscovery-dir", str(root / "autodiscovery")])
            self.assertEqual(code, 0)
            # batch 1: 1,2,3 (3 unfinished) -> batch 2: 3,4,5 (5 unfinished) -> batch 3: 5,6,7 (7 unfinished);
            # $4 per batch, the third batch starts at $8 < $10, a fourth would start at $12 and is not run
            self.assertEqual([ids for ids, *_ in calls], [[1, 2, 3], [3, 4, 5], [5, 6, 7]])
            self.assertEqual([budget for _, budget, *_ in calls], [10.0, 6.0, 2.0])
            self.assertEqual(calls[0][2:], ("claude-opus-5-5", "high"))
            work = root / "autodiscovery" / "claude-merge-t.claude"
            self.assertTrue((work / "run_hypo.sh").is_file())
            self.assertTrue((work / "wait_hypo.sh").is_file())
            self.assertTrue((work / "task.md").is_symlink())
            manifest = json.loads((work / "manifest.json").read_text())
            self.assertEqual(manifest["cost_usd"], 12.0)
            self.assertEqual(manifest["engine"], "claude")
            self.assertEqual(len(manifest["sessions"]), 3)
            export = json.loads((root / "autodiscovery" / "claude-merge-t.json").read_text())
            self.assertEqual([r["id"] for r in export], [1, 2, 3, 4, 5, 6, 7])
            self.assertEqual(export[0]["model"], "claude-opus-5-5")
            self.assertEqual(sum(r["status"] == "SUCCEEDED" for r in export), 6)
            self.assertEqual(export[6]["status"], "FAILED")

            # assemble-only rebuilds the export without any session
            with patch.object(ad, "PROJECT_ROOT", root), patch.object(ad, "run_batch_claude", fake_batch), \
                    redirect_stdout(io.StringIO()):
                ad.main(["--kind", "merge", "--brain", "1", "--run-name", "claude-merge-t",
                         "--hypotheses", "7", "--assemble-only",
                         "--autodiscovery-dir", str(root / "autodiscovery")])
            self.assertEqual(len(calls), 3)

    def test_loop_continues_after_partial_progress_and_stops_when_nothing_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake_materials(root)
            calls = []

            async def fake_batch(work_dir, ids, kind, brain, mcl, *, model, effort, verbose,
                                 max_budget_usd, timeout_s):
                calls.append(list(ids))
                if len(calls) == 1:          # proposals and scripts only: progress without a result
                    for rid in ids:
                        write_record(work_dir, rid, result=False)
                elif len(calls) == 2:        # finishes them
                    for rid in ids:
                        write_record(work_dir, rid)
                # third call: touches nothing -> the driver must stop
                return {"cost_usd": 1.0, "tokens": None}

            with patch.object(ad, "PROJECT_ROOT", root), patch.object(ad, "run_batch_claude", fake_batch), \
                    redirect_stdout(io.StringIO()) as out:
                ad.main(["--kind", "merge", "--brain", "1", "--run-name", "claude-merge-p",
                         "--hypotheses", "6", "--batch", "3", "--max-cost-usd", "100",
                         "--autodiscovery-dir", str(root / "autodiscovery")])
            self.assertEqual(calls, [[1, 2, 3], [1, 2, 3], [4, 5, 6]])
            self.assertIn("left new work behind", out.getvalue())
            self.assertIn("produced nothing new", out.getvalue())

    def test_codex_engine_runs_unpriced_until_the_hypotheses_are_done(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake_materials(root, kind="split")
            calls = []

            def fake_codex(work_dir, ids, kind, brain, mcl, *, model, effort, verbose, timeout_s,
                           price_in, price_out, sandbox):
                calls.append((list(ids), model, effort, sandbox))
                for rid in ids:
                    write_record(work_dir, rid)
                return {"cost_usd": 0.0, "tokens": {"input_tokens": 100, "output_tokens": 10,
                                                    "cached_input_tokens": 0, "reasoning_output_tokens": 0},
                        "exit_code": 0}

            with patch.object(ad, "PROJECT_ROOT", root), patch.object(ad, "run_batch_codex", fake_codex), \
                    patch.object(ad.shutil, "which", lambda name: "/fake/codex"), \
                    redirect_stdout(io.StringIO()):
                code = ad.main(["--engine", "codex", "--kind", "split", "--brain", "1",
                                "--run-name", "codex-split-t", "--hypotheses", "4", "--batch", "2",
                                "--max-cost-usd", "1", "--autodiscovery-dir", str(root / "autodiscovery")])
            self.assertEqual(code, 0)
            self.assertEqual([ids for ids, *_ in calls], [[1, 2], [3, 4]])
            self.assertEqual(calls[0][1:], ("gpt-6-astra", "high", "bypass"))
            work = root / "autodiscovery" / "codex-split-t.codex"
            manifest = json.loads((work / "manifest.json").read_text())
            self.assertEqual(manifest["engine"], "codex")
            self.assertEqual(manifest["cost_usd"], 0.0)
            self.assertEqual(manifest["sessions"][0]["tokens"]["input_tokens"], 100)
            export = json.loads((root / "autodiscovery" / "codex-split-t.json").read_text())
            self.assertEqual(len(export), 4)


if __name__ == "__main__":
    unittest.main()
