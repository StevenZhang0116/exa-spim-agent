"""Reviser I/O limits (2026-10-05, raised again 2026-10-07): MCP output and file-read caps,
feedback budget, read access to SDK-parked tool results and to this generation's own TRAIN
artefact directories."""

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
        # Parked results must be readable in one Read: the file-read cap matches the MCP cap.
        self.assertEqual(env["CLAUDE_CODE_FILE_READ_MAX_OUTPUT_TOKENS"], str(session.FILE_READ_TOKENS))
        self.assertGreaterEqual(session.FILE_READ_TOKENS, session.MCP_OUTPUT_TOKENS)

    def test_feedback_budget(self):
        self.assertEqual(feedback.MAX_FEEDBACK_BYTES, 128_000)


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


class ReadableDirectoryTests(unittest.TestCase):
    """This generation's experiments/, classifier_fits/ and descriptor_runs/ are readable, never writable."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.run_dir = Path(self.temp.name) / "runs" / "precision-x"
        self.gen = self.run_dir / "gen001"
        self.attempt = self.gen / "experiments" / "attempt001"
        self.attempt.mkdir(parents=True)
        (self.attempt / "result.json").write_text("{}")
        (self.gen / "scorer.py").write_text("")
        self.outside = Path(self.temp.name) / "secret.txt"
        self.outside.write_text("x")
        guard, _ = access.make_guard(self.run_dir, self.gen, [], self.run_dir / "audit.jsonl", sdk_tool_results=False,
                                     readable_dirs=[self.gen / "experiments", self.gen / "classifier_fits"])
        self.guard = guard

    def decision(self, tool, path):
        return asyncio.run(self.guard(tool, {"file_path": str(path)}, None)).behavior

    def test_files_below_the_directories_are_readable_only(self):
        self.assertEqual(self.decision("Read", self.attempt / "result.json"), "allow")
        self.assertEqual(self.decision("Write", self.attempt / "result.json"), "deny")
        self.assertEqual(self.decision("Edit", self.attempt / "result.json"), "deny")
        self.assertEqual(self.decision("Read", self.attempt), "deny")  # directories themselves are not files
        self.assertEqual(self.decision("Read", self.gen / "classifier_fits" / "fit001" / "worker.log"), "deny")  # absent
        self.assertEqual(self.decision("Read", self.gen / "search_plan.json"), "deny")  # not listed, not in a tree

    def test_symlinks_out_of_the_tree_and_other_runs_are_denied(self):
        link = self.attempt / "escape.txt"
        link.symlink_to(self.outside)
        self.assertEqual(self.decision("Read", link), "deny")
        self.assertEqual(self.decision("Read", self.outside), "deny")
        with self.assertRaises(ValueError):
            access.make_guard(self.run_dir, self.gen, [], self.run_dir / "audit.jsonl",
                              readable_dirs=[Path(self.temp.name)])


if __name__ == "__main__":
    unittest.main()
