"""Precision-only entrypoint and SDK isolation regressions."""

import asyncio
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from proofreader_evolve.harness.reviser_session import build_options, bind_session_options


class EntrypointTests(unittest.TestCase):
    def test_precision_help_and_preflight_do_not_import_legacy_or_sdk(self):
        code = '''
import importlib.abc
import sys
class BlockLegacy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in {
            "proofreader_evolve.cli.run_structural_evolution",
            "proofreader_evolve.harness.candidate",
            "claude_agent_sdk",
        }:
            raise AssertionError("Unexpected dependency: " + fullname)
sys.meta_path.insert(0, BlockLegacy())
from proofreader_evolve.cli import run_evolution, preflight
try:
    run_evolution.main(["--help"])
except SystemExit as exc:
    assert exc.code == 0
else:
    raise AssertionError("Help must exit without starting a run")
'''
        result = subprocess.run([sys.executable, "-c", code], text=True,
                                capture_output=True, timeout=60,
                                cwd=Path(__file__).resolve().parents[2])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--merge-k", result.stdout)

    def test_removed_objective_is_rejected_before_work(self):
        from contextlib import redirect_stderr
        import io
        from proofreader_evolve.cli import run_evolution, run_precision_evolution
        with patch.object(run_precision_evolution, "run") as run:
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                run_evolution.main(["--objective", "structural_repair"])
        self.assertEqual(error.exception.code, 2)
        run.assert_not_called()

    def test_legacy_helpers_are_not_exposed(self):
        from proofreader_evolve.cli import run_evolution
        self.assertFalse(hasattr(run_evolution, "run_evolution"))


class ReviserSessionTests(unittest.TestCase):
    @patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-only-not-a-real-key"})
    def test_scorer_session_has_exact_read_write_scope_and_shared_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            generation = root / "gen001"
            generation.mkdir()
            report = generation / "train_feedback.json"
            options, state = build_options(run_dir=root, system_prompt="scorer contract",
                                            max_turns=12)
            bound = bind_session_options(options, generation / "scorer.py",
                                         generation / "rules.md", report)
            self.assertEqual(bound.system_prompt, "scorer contract")
            self.assertEqual(bound.max_turns, 12)
            self.assertEqual(bound.setting_sources, [])
            self.assertIs(bound._isolation_state, state)
            for tool, path, expected in [
                ("Edit", generation / "scorer.py", "allow"),
                ("Write", generation / "rules.md", "allow"),
                ("Read", report, "allow"),
                ("Edit", generation / "heuristics.py", "deny"),
                ("Read", root / "baseline.json", "deny"),
                ("Read", generation / "evaluation.json", "deny"),
                ("Read", root / "gen002/scorer.py", "deny"),
            ]:
                result = asyncio.run(bound.can_use_tool(tool, {"file_path": str(path)}, None))
                self.assertEqual(result.behavior, expected, str(path))
            self.assertEqual(len(state["violations"]), 4)
            self.assertIn("PreToolUse", bound.hooks)

    @patch.dict("os.environ", {}, clear=True)
    def test_live_session_requires_explicit_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError, "ANTHROPIC_API_KEY"):
                build_options(run_dir=Path(tmp), system_prompt="contract")


if __name__ == "__main__":
    unittest.main()
