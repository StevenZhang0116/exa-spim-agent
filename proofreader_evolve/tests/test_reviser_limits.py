"""Reviser I/O limits (2026-10-05): larger MCP output cap, larger feedback budget,
and read access to tool results the SDK parks when a result still overflows."""

import asyncio
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from proofreader_evolve.harness import reviser_access as access
from proofreader_evolve.harness import reviser_session as session
from proofreader_evolve.harness import train_feedback as feedback


class LimitTests(unittest.TestCase):
    def test_environment_carries_the_mcp_output_cap(self):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "k"}):
            env = session.anthropic_api_env()
        self.assertEqual(env["MAX_MCP_OUTPUT_TOKENS"], str(session.MCP_OUTPUT_TOKENS))
        self.assertGreaterEqual(session.MCP_OUTPUT_TOKENS, 100_000)

    def test_feedback_budget(self):
        self.assertEqual(feedback.MAX_FEEDBACK_BYTES, 96_000)


class ParkedToolResultTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.run_dir = root / "runs" / "precision_x"
        (self.run_dir / "gen001").mkdir(parents=True)
        self.config = root / "claude_config"
        self.env = patch.dict(os.environ, {"CLAUDE_CONFIG_DIR": str(self.config)})
        self.env.start()
        self.addCleanup(self.env.stop)
        project = access.sdk_project_dir(self.run_dir)
        self.results = project / "session-1" / "tool-results"
        self.results.mkdir(parents=True)
        self.parked = self.results / "mcp-training-search_memory-1.txt"
        self.parked.write_text("[]")
        self.outside = root / "secret.txt"
        self.outside.write_text("x")
        self.guard, _ = access.make_guard(self.run_dir, self.run_dir / "gen001", [], self.run_dir / "audit.jsonl")

    def decision(self, tool, path):
        return asyncio.run(self.guard(tool, {"file_path": str(path)}, None)).behavior

    def test_project_dir_escapes_every_non_alphanumeric_character(self):
        name = access.sdk_project_dir(self.run_dir).name
        self.assertTrue(name.startswith("-"))
        self.assertNotIn("/", name)
        self.assertNotIn("_", name)
        self.assertIn("precision-x", name)

    def test_parked_results_are_readable_but_never_writable(self):
        self.assertEqual(self.decision("Read", self.parked), "allow")
        self.assertEqual(self.decision("Write", self.parked), "deny")
        self.assertEqual(self.decision("Edit", self.parked), "deny")

    def test_symlinks_other_projects_and_nested_paths_are_denied(self):
        link = self.results / "link.txt"
        link.symlink_to(self.outside)
        self.assertEqual(self.decision("Read", link), "deny")
        other = self.config / "projects" / "-other-run" / "session-9" / "tool-results"
        other.mkdir(parents=True)
        (other / "a.txt").write_text("x")
        self.assertEqual(self.decision("Read", other / "a.txt"), "deny")
        nested = self.results / "deeper"
        nested.mkdir()
        (nested / "b.txt").write_text("x")
        self.assertEqual(self.decision("Read", nested / "b.txt"), "deny")
        self.assertEqual(self.decision("Read", self.results), "deny")
        self.assertEqual(self.decision("Read", self.outside), "deny")

    def test_parked_access_can_be_disabled(self):
        guard, _ = access.make_guard(self.run_dir, self.run_dir / "gen001", [], self.run_dir / "audit.jsonl",
                                     sdk_tool_results=False)
        self.assertEqual(asyncio.run(guard("Read", {"file_path": str(self.parked)}, None)).behavior, "deny")


if __name__ == "__main__":
    unittest.main()
