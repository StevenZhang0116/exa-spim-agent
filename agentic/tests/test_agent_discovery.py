"""The agent discovery driver: records, batching, the runner, both engines' glue and the budget loop."""
from __future__ import annotations

import io
import json
import os
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


def write_record(work: Path, rid: int, *, prior=0.6, posterior=None, exit_code=0, code="print('x')\n",
                 stdout="INPUTS READ fragment_nodes=3\nAP 0.5\n", stderr=""):
    (work / "records").mkdir(exist_ok=True)
    (work / "outputs").mkdir(exist_ok=True)
    (work / "records" / f"{rid}.proposal.json").write_text(json.dumps(
        {"hypothesis": f"Hypothesis {rid} says more is better.", "experimentPlan": "rank and score", "prior": prior}))
    (work / f"hypo_{rid}.py").write_text(code)
    (work / "outputs" / f"hypo_{rid}.stdout.txt").write_text(stdout)
    (work / "outputs" / f"hypo_{rid}.stderr.txt").write_text(stderr)
    (work / "outputs" / f"hypo_{rid}.meta.json").write_text(json.dumps(
        {"id": rid, "exit_code": exit_code, "runtime_ms": 1234, "started_at": "2026-10-22T10:00:00+00:00",
         "finished_at": "2026-10-22T10:00:01+00:00", "timeout_s": 60}))
    if posterior is not None:
        (work / "records" / f"{rid}.result.json").write_text(json.dumps(
            {"analysis": "AP 0.5 beats nothing.", "review": "small n", "posterior": posterior}))


def fake_materials(root: Path, kind="merge", brain="1", mcl=100) -> None:
    (root / "markdowns").mkdir()
    (root / "cache").mkdir()
    (root / "markdowns" / f"{kind}_features_for_proofreader_evolve.md").write_text("# task\n")
    (root / "cache" / f"dataset_cache_{brain}_mcl{mcl}_add.pkl").write_bytes(b"pkl")
    (root / "markdowns" / f"rerank_companion_{brain}_{kind}.npz").write_bytes(b"npz")
    (root / "markdowns" / f"rerank_companion_{brain}_{kind}.json").write_text("{}")


class RecordTests(unittest.TestCase):
    def test_belief_and_surprisal_follow_the_export_conventions(self) -> None:
        b = ad.belief(0.9167)
        self.assertEqual(b["_type"], "boolean_cat")
        self.assertEqual(b["definitely_true"], 5)
        self.assertEqual(sum(b[name] for name, _ in ad.BELIEF_BUCKETS), 5)
        self.assertAlmostEqual(b["mean"], 0.9167)
        self.assertEqual(ad.belief(0.45)["uncertain"], 5)
        s, surprising = ad.surprisal(0.9167, 0.40625)
        self.assertAlmostEqual(s, -0.7953, places=3)   # the 2026-10-19 export's first record
        self.assertTrue(surprising)
        s2, surprising2 = ad.surprisal(0.7083, 0.734375)
        self.assertAlmostEqual(s2, 0.0406, places=3)
        self.assertFalse(surprising2)

    def test_assembled_record_has_every_ad_field(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            write_record(work, 1, prior=0.6, posterior=0.9)
            write_record(work, 2, prior=0.5, posterior=None, exit_code=1, stderr="Traceback: boom")
            write_record(work, 3, prior=0.5, posterior=0.3, exit_code=124)
            (work / "outputs" / "hypo_1_figure.png").write_bytes(b"png")
            records = ad.assemble_export(work, 5)
            self.assertEqual([r["id"] for r in records], [1, 2, 3])
            one = records[0]
            self.assertEqual(list(one.keys())[:17], AD_FIELDS)
            self.assertEqual(one["status"], "SUCCEEDED")
            self.assertEqual(one["code"], "print('x')\n")
            self.assertIn("INPUTS READ", one["codeOutput"])
            self.assertEqual(one["runtimeMs"], 1234)
            self.assertEqual(one["createdAt"], "2026-10-22T10:00:01+00:00")
            self.assertEqual(one["richOutputs"], [{"path": "outputs/hypo_1_figure.png", "type": "image/png"}])
            self.assertTrue(one["isSurprising"])
            two = records[1]
            self.assertEqual(two["status"], "FAILED")      # no result file, non-zero exit
            self.assertIn("STDERR:\nTraceback", two["codeOutput"])
            self.assertEqual(two["posterior"], two["prior"])  # unchanged belief when never judged
            self.assertEqual(records[2]["status"], "FAILED")  # killed by the timeout despite a result file
            json.dumps(records)

    def test_bad_probability_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            write_record(work, 1, prior=1.4, posterior=0.5)
            with self.assertRaisesRegex(SystemExit, "prior"):
                ad.assemble_export(work, 1)


class BatchingTests(unittest.TestCase):
    def test_next_batch_finishes_proposed_ids_before_new_ones(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            write_record(work, 1, posterior=0.7)
            write_record(work, 2, posterior=None)
            self.assertEqual(ad.record_state(work, 1), "complete")
            self.assertEqual(ad.record_state(work, 2), "proposed")
            self.assertEqual(ad.record_state(work, 3), "absent")
            self.assertEqual(ad.next_batch(work, 6, 3), [2, 3, 4])
            self.assertEqual(ad.next_batch(work, 2, 3), [2])
            write_record(work, 2, posterior=0.2)
            self.assertEqual(ad.next_batch(work, 2, 3), [])
            ledger = ad.write_ledger(work, 6).read_text()
            self.assertIn("1 [done] prior=0.6 posterior=0.7: Hypothesis 1", ledger)
            instruction = ad.batch_instruction(work, [3, 4], "merge", "1", 100)
            self.assertIn("ids [3, 4]", instruction)
            self.assertIn("rerank_companion_1_merge.npz", instruction)
            self.assertIn(f"{work}/run_hypo.sh <id>", instruction)
            self.assertIn(f"{work}/records/<id>.proposal.json", instruction)
            self.assertNotIn("subagent", instruction)   # engine-independent wording

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
    def test_runner_captures_output_exit_code_and_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            (work / "run_hypo.sh").write_text(ad.RUN_HYPO_SH.format(timeout=30, python=sys.executable))
            (work / "run_hypo.sh").chmod(0o755)
            (work / "hypo_7.py").write_text("import sys\nprint('INPUTS READ ok')\nprint('warn', file=sys.stderr)\nsys.exit(3)\n")
            proc = subprocess.run([str(work / "run_hypo.sh"), "7"], capture_output=True, text=True,
                                  cwd=tmp, env={**os.environ})
            self.assertEqual(proc.returncode, 3)
            self.assertIn("exit=3", proc.stdout)
            meta = json.loads((work / "outputs" / "hypo_7.meta.json").read_text())
            self.assertEqual(meta["exit_code"], 3)
            self.assertGreaterEqual(meta["runtime_ms"], 0)
            self.assertTrue(meta["finished_at"].endswith("+00:00"))
            self.assertEqual((work / "outputs" / "hypo_7.stdout.txt").read_text(), "INPUTS READ ok\n")
            self.assertEqual((work / "outputs" / "hypo_7.stderr.txt").read_text(), "warn\n")


class CodexGlueTests(unittest.TestCase):
    def test_codex_command_and_prompt(self) -> None:
        argv = ad.codex_command(Path("/w"), "gpt-6-astra", "high", Path("/w/codex/last.txt"), "bypass")
        self.assertEqual(argv[1:3], ["exec", "--json"])
        self.assertIn("--skip-git-repo-check", argv)
        self.assertEqual(argv[argv.index("-C") + 1], "/w")
        self.assertEqual(argv[argv.index("-m") + 1], "gpt-6-astra")
        self.assertIn('model_reasoning_effort="high"', argv)
        self.assertIn("--dangerously-bypass-approvals-and-sandbox", argv)
        self.assertEqual(argv[-1], "-")
        sandboxed = ad.codex_command(Path("/w"), "m", "low", Path("/w/l.txt"), "workspace-write")
        self.assertNotIn("--dangerously-bypass-approvals-and-sandbox", sandboxed)
        self.assertEqual(sandboxed[sandboxed.index("-s") + 1], "workspace-write")
        instructions = ad.researcher_instructions()
        self.assertFalse(instructions.startswith("---"))
        self.assertIn("# Discovery Researcher", instructions)
        self.assertIn("proposal.json", instructions)

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
                    write_record(work_dir, rid, posterior=0.8)
                if len(ids) > 1:
                    write_record(work_dir, ids[-1], posterior=None)
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
            self.assertTrue((work / "task.md").is_symlink())
            manifest = json.loads((work / "manifest.json").read_text())
            self.assertEqual(manifest["cost_usd"], 12.0)
            self.assertEqual(manifest["engine"], "claude")
            self.assertEqual(len(manifest["sessions"]), 3)
            export = json.loads((root / "autodiscovery" / "claude-merge-t.json").read_text())
            self.assertEqual([r["id"] for r in export], [1, 2, 3, 4, 5, 6, 7])
            self.assertEqual(sum(r["status"] == "SUCCEEDED" for r in export), 6)
            self.assertEqual(export[6]["status"], "FAILED")

            # assemble-only rebuilds the export without any session
            with patch.object(ad, "PROJECT_ROOT", root), patch.object(ad, "run_batch_claude", fake_batch), \
                    redirect_stdout(io.StringIO()):
                ad.main(["--kind", "merge", "--brain", "1", "--run-name", "claude-merge-t",
                         "--hypotheses", "7", "--assemble-only",
                         "--autodiscovery-dir", str(root / "autodiscovery")])
            self.assertEqual(len(calls), 3)

    def test_codex_engine_runs_unpriced_until_the_hypotheses_are_done(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake_materials(root, kind="split")
            calls = []

            def fake_codex(work_dir, ids, kind, brain, mcl, *, model, effort, verbose, timeout_s,
                           price_in, price_out, sandbox):
                calls.append((list(ids), model, effort, sandbox))
                for rid in ids:
                    write_record(work_dir, rid, posterior=0.5)
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
