"""
Unit tests for the multi-candidate generation machinery
(--candidates-per-gen K > 1): the candidate fan-out, the train-side screen's
cheap gates + ranking, the attempts-memory rendering of screened-out candidates,
and the isolation guard's per-candidate sandboxes.

No brain data needed — everything runs on temp files and fakes (same policy as
test_detector_seed.py). Run in the panda env:

    python -m unittest discover -s proofreader_evolve/tests -t .
"""

from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path


VALID_POLICY = "def propose_edits(sites, ctx):\n    return []\n"


class TestDiffstat(unittest.TestCase):

    def test_added_and_removed_counts(self):
        from proofreader_evolve.cli.run_evolution import _diffstat
        with tempfile.TemporaryDirectory() as td:
            a = Path(td) / "a.py"
            b = Path(td) / "b.py"
            a.write_text("one\ntwo\nthree\n")
            b.write_text("one\nTWO\nthree\nfour\n")
            self.assertEqual(_diffstat(a, b), "+2 -1")  # TWO+four added, two removed

    def test_identical_files(self):
        from proofreader_evolve.cli.run_evolution import _diffstat
        with tempfile.TemporaryDirectory() as td:
            a = Path(td) / "a.py"
            a.write_text(VALID_POLICY)
            self.assertEqual(_diffstat(a, a), "+0 -0")

    def test_missing_file_is_question_mark(self):
        from proofreader_evolve.cli.run_evolution import _diffstat
        with tempfile.TemporaryDirectory() as td:
            a = Path(td) / "a.py"
            a.write_text(VALID_POLICY)
            self.assertEqual(_diffstat(a, Path(td) / "missing.py"), "?")


class TestPickScreenWinner(unittest.TestCase):

    @staticmethod
    def _rec(cand, status="scored", fitness=0.0, false_policy=0, diffstat="+1 -1"):
        return {"cand": cand, "status": status, "train_fitness": fitness,
                "false_policy": false_policy, "diffstat": diffstat}

    def test_highest_fitness_wins(self):
        from proofreader_evolve.cli.run_evolution import _pick_screen_winner
        recs = [self._rec(1, fitness=5), self._rec(2, fitness=9),
                self._rec(3, fitness=7)]
        self.assertEqual(_pick_screen_winner(recs), 2)

    def test_tie_prefers_fewer_false_then_smaller_diff(self):
        from proofreader_evolve.cli.run_evolution import _pick_screen_winner
        # Same fitness: fewer policy-caused false merges wins.
        recs = [self._rec(1, fitness=5, false_policy=2),
                self._rec(2, fitness=5, false_policy=0)]
        self.assertEqual(_pick_screen_winner(recs), 2)
        # Same fitness and false count: the smaller diff wins.
        recs = [self._rec(1, fitness=5, diffstat="+30 -10"),
                self._rec(2, fitness=5, diffstat="+3 -1")]
        self.assertEqual(_pick_screen_winner(recs), 2)

    def test_non_scored_records_never_win(self):
        from proofreader_evolve.cli.run_evolution import _pick_screen_winner
        recs = [self._rec(1, status="import-failed", fitness=99),
                self._rec(2, status="budget-timeout", fitness=99),
                self._rec(3, fitness=-5)]
        self.assertEqual(_pick_screen_winner(recs), 3)

    def test_no_survivor_returns_zero(self):
        from proofreader_evolve.cli.run_evolution import _pick_screen_winner
        recs = [self._rec(1, status="lint-failed"),
                self._rec(2, status="session-error")]
        self.assertEqual(_pick_screen_winner(recs), 0)
        self.assertEqual(_pick_screen_winner([]), 0)


class TestGenerateCandidates(unittest.TestCase):

    def test_fanout_focus_rotation_and_error_containment(self):
        import proofreader_evolve.cli.run_evolution as rev

        with tempfile.TemporaryDirectory() as td:
            gen_dir = Path(td) / "gen01"
            wh = Path(td) / "heuristics.py"
            wr = Path(td) / "rules.md"
            wh.write_text(VALID_POLICY)
            wr.write_text("# rules\n")

            seen = []

            async def fake_ask(options, report_path, hpath, rpath, verbose,
                               **kw):
                seen.append({"heuristics": hpath, "focus": kw.get("focus"),
                             "verbose": verbose})
                if "cand_2" in hpath:
                    raise RuntimeError("boom")  # one bad session, contained
                Path(hpath).write_text(VALID_POLICY + "# edited\n")
                return ("diag line", 10, 20, 0.5, None, 30, 40)

            orig = rev.ask_reviser
            rev.ask_reviser = fake_ask
            try:
                res = asyncio.run(rev.generate_candidates(
                    3, gen_dir, wh, wr, None, "report.md", True,
                    [], None, False, False, []))
            finally:
                rev.ask_reviser = orig

        self.assertEqual([c["idx"] for c in res], [1, 2, 3])
        # Sandboxes: each candidate got its own copy of the parent files.
        for c in res:
            self.assertTrue(c["heuristics"].endswith(
                f"cand_{c['idx']}/heuristics.py"))
        # Focus rotation: cand 1 free choice, then CANDIDATE_FOCI order.
        foci = {s["heuristics"].split("/")[-2]: s["focus"] for s in seen}
        self.assertIsNone(foci["cand_1"])
        self.assertEqual(foci["cand_2"], rev.CANDIDATE_FOCI[1])
        self.assertEqual(foci["cand_3"], rev.CANDIDATE_FOCI[2])
        # Only candidate 1 streams live.
        verbose_by = {s["heuristics"].split("/")[-2]: s["verbose"] for s in seen}
        self.assertTrue(verbose_by["cand_1"])
        self.assertFalse(verbose_by["cand_2"] or verbose_by["cand_3"])
        # The failed session is contained: error set, usage zeroed, others intact.
        by_idx = {c["idx"]: c for c in res}
        self.assertIsNotNone(by_idx[2]["error"])
        self.assertEqual(by_idx[2]["in_tok"], 0)
        self.assertIsNone(by_idx[1]["error"])
        self.assertEqual(by_idx[1]["diagnosis"], "diag line")
        self.assertEqual((by_idx[3]["in_tok"], by_idx[3]["out_tok"]), (10, 20))

    def test_focus_pool_shape(self):
        from proofreader_evolve.cli.run_evolution import CANDIDATE_FOCI
        self.assertIsNone(CANDIDATE_FOCI[0])          # candidate 1 = free choice
        self.assertTrue(all(isinstance(f, str) and f
                            for f in CANDIDATE_FOCI[1:]))


class TestScreenCheapGates(unittest.TestCase):
    """The screen's pre-scoring gates (session error / import / lint) and the
    scored path, exercised with NO brain data (empty train_ctxs -> a scored
    candidate gets the neutral fitness 0)."""

    def _cand(self, td, idx, code):
        d = Path(td) / f"cand_{idx}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "heuristics.py").write_text(code)
        (d / "rules.md").write_text("# rules\n")
        return {"idx": idx, "dir": str(d),
                "heuristics": str(d / "heuristics.py"),
                "rules": str(d / "rules.md"), "focus": None,
                "diagnosis": f"cand {idx} diagnosis", "error": None}

    def test_gates_and_winner(self):
        from proofreader_evolve.cli.run_evolution import screen_candidates
        with tempfile.TemporaryDirectory() as td:
            parent = Path(td) / "heuristics.py"
            parent.write_text(VALID_POLICY)
            ok = self._cand(td, 1, VALID_POLICY + "# tweak\n")
            broken = self._cand(td, 2, "def propose_edits(:\n")   # SyntaxError
            hardcoded = self._cand(
                td, 3, VALID_POLICY.replace(
                    "return []", 'return [("888777", "999666")]'))
            errored = self._cand(td, 4, VALID_POLICY)
            errored["error"] = "RuntimeError('boom')"

            winner, records, secs = screen_candidates(
                [ok, broken, hardcoded, errored], parent,
                report_labels={"888777", "999666"},
                train_ctxs=[], screen_names_attr="train_names",
                max_class_size=None, verbose=False, splits_only=False,
                two_phase=False, policy_time_budget=300.0,
                merge_penalty=100.0, parent_train_fitness=0.0)

        by = {r["cand"]: r for r in records}
        self.assertEqual(by[1]["status"], "scored")
        self.assertEqual(by[1]["train_fitness"], 0.0)  # no brains -> neutral
        self.assertEqual(by[2]["status"], "import-failed")
        self.assertEqual(by[3]["status"], "lint-failed")
        self.assertEqual(by[4]["status"], "session-error")
        self.assertEqual(winner, 1)
        self.assertTrue(by[1]["winner"])
        self.assertEqual(secs, 0.0)

    def test_no_survivor(self):
        from proofreader_evolve.cli.run_evolution import screen_candidates
        with tempfile.TemporaryDirectory() as td:
            parent = Path(td) / "heuristics.py"
            parent.write_text(VALID_POLICY)
            broken = self._cand(td, 1, "def propose_edits(:\n")
            winner, records, _ = screen_candidates(
                [broken], parent, report_labels=set(),
                train_ctxs=[], screen_names_attr="train_names",
                max_class_size=None, verbose=False, splits_only=False,
                two_phase=False, policy_time_budget=300.0,
                merge_penalty=100.0, parent_train_fitness=0.0)
        self.assertEqual(winner, 0)
        self.assertFalse(records[0]["winner"])


class TestAttemptsRendering(unittest.TestCase):

    def test_screen_stage_entries_render_as_train_screen(self):
        from proofreader_evolve.cli.run_evolution import _format_attempts
        text = _format_attempts([
            {"gen": 3, "stage": "screen",
             "summary": "[cand 2, focus: ENUMERATION CEILING] +12 -3 lines; widened max_gap",
             "heldout": -4, "correct": 8, "false": 0, "accepted": False},
            {"gen": 3, "summary": "+5 -2 lines; tightened colinearity",
             "heldout": -1, "correct": 12, "false": 1, "accepted": False},
        ])
        self.assertIn("TRAIN-screen fitness -4", text)
        self.assertIn("screened out before the held-out gate", text)
        self.assertIn("held-out fitness -1", text)
        self.assertIn("rejected", text)


class TestIsolationGuardCandidateSandboxes(unittest.TestCase):

    def _guard(self, run_dir: Path):
        from proofreader_evolve.cli.run_evolution import _make_isolation_guard
        return _make_isolation_guard(run_dir, run_dir / "tool_audit.jsonl")

    @staticmethod
    def _behavior(result) -> str:
        return getattr(result, "behavior", type(result).__name__)

    def test_cand_sandbox_rw_allowed_but_siblings_denied(self):
        from proofreader_evolve.cli.run_evolution import HERE
        runs_root = HERE / "runs"
        with tempfile.TemporaryDirectory(dir=str(runs_root)) as td:
            run_dir = Path(td)
            (run_dir / "gen01" / "cand_2").mkdir(parents=True)
            guard, state = self._guard(run_dir)

            async def ask(tool, path):
                return await guard(tool, {"file_path": str(path)}, None)

            # Own candidate sandbox: read AND write allowed.
            p = run_dir / "gen01" / "cand_2" / "heuristics.py"
            self.assertIn("allow", self._behavior(asyncio.run(ask("Write", p))))
            self.assertIn("allow", self._behavior(asyncio.run(ask("Read", p))))
            r = run_dir / "gen01" / "cand_2" / "rules.md"
            self.assertIn("allow", self._behavior(asyncio.run(ask("Edit", r))))
            # Failure report: read allowed, write denied.
            fr = run_dir / "gen01" / "failure_report.md"
            self.assertIn("allow", self._behavior(asyncio.run(ask("Read", fr))))
            self.assertIn("deny", self._behavior(asyncio.run(ask("Write", fr))))
            # Another candidate's screen record / arbitrary cand file: denied.
            other = run_dir / "gen01" / "cand_2" / "notes.txt"
            self.assertIn("deny", self._behavior(asyncio.run(ask("Write", other))))
            # A SIBLING run's candidate sandbox: denied.
            sibling = runs_root / "some_other_run" / "gen01" / "cand_1" / "heuristics.py"
            self.assertIn("deny",
                          self._behavior(asyncio.run(ask("Read", sibling))))
            self.assertEqual(len(state["violations"]), 3)


if __name__ == "__main__":
    unittest.main()
