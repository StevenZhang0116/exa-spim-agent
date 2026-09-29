"""Scorer file-tool isolation; no live API or brain loads."""

import asyncio
from pathlib import Path
import tempfile
import unittest

from proofreader_evolve.harness.reviser_access import make_guard, pre_tool_hook


class AccessTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.own = self.root / "gen01" / "cand_1"
        self.own.mkdir(parents=True)
        self.report = self.root / "gen01" / "train_feedback.json"
        self.source = self.root / "evaluator.py"
        self.guard, self.state = make_guard(self.root, self.own, [self.report, self.source],
                                            self.root / "audit.jsonl")

    def decision(self, tool, path):
        return asyncio.run(self.guard(tool, {"file_path": str(path)}, None)).behavior

    def test_own_files_allowed_relative_paths_use_session_cwd(self):
        for tool in ("Read", "Write", "Edit"):
            self.assertEqual(self.decision(tool, "gen01/cand_1/scorer.py"), "allow")

    def test_readonly_sources_never_writable(self):
        self.assertEqual(self.decision("Read", self.source), "allow")
        self.assertEqual(self.decision("Write", self.source), "deny")
        self.assertEqual(self.decision("Edit", self.report), "deny")

    def test_other_candidate_parent_and_external_files_denied(self):
        for path in (self.root / "gen01/cand_2/scorer.py",
                     self.root / "artifacts/scorer.py", "/etc/passwd",
                     self.root / "manifest.json", self.root / "ledger.jsonl"):
            for tool in ("Read", "Write"):
                self.assertEqual(self.decision(tool, path), "deny")

    def test_symlink_escape_after_configuration_denied(self):
        (self.own / "scorer.py").symlink_to(self.source)
        self.assertEqual(self.decision("Write", self.own / "scorer.py"), "deny")
        self.source.symlink_to("/etc/passwd")
        self.assertEqual(self.decision("Read", self.source), "deny")

    def test_unknown_tool_and_missing_path_fail_closed(self):
        self.assertEqual(self.decision("Bash", self.own / "scorer.py"), "deny")
        self.assertEqual(asyncio.run(self.guard("Read", {}, None)).behavior, "deny")

    def test_run_mount_alias_supported_but_candidate_directory_retarget_denied(self):
        alias = self.root / "mount_alias"
        alias.symlink_to(self.root.resolve(), target_is_directory=True)
        guard, _ = make_guard(alias, alias / "gen01/cand_1", [alias / "evaluator.py"],
                              self.root / "audit.jsonl")
        request = {"file_path": str(alias / "gen01/cand_1/scorer.py")}
        self.assertEqual(asyncio.run(guard("Write", request, None)).behavior, "allow")
        other = self.root / "gen01/cand_2"
        other.mkdir()
        self.own.rmdir()  # empty fixture only
        self.own.symlink_to(other.resolve(), target_is_directory=True)
        self.assertEqual(asyncio.run(guard("Write", request, None)).behavior, "deny")

    def test_preexisting_leaf_symlink_is_not_added_to_allowlist(self):
        (self.own / "scorer.py").symlink_to(self.source)
        guard, _ = make_guard(self.root, self.own, [], self.root / "audit.jsonl")
        request = {"file_path": str(self.own / "scorer.py")}
        self.assertEqual(asyncio.run(guard("Write", request, None)).behavior, "deny")

    def test_pretool_hook_reuses_guard_even_if_tool_is_auto_allowed(self):
        hook = pre_tool_hook(self.guard)
        result = asyncio.run(hook({"tool_name": "Write", "tool_input":
                                   {"file_path": str(self.source)}}, None, None))
        self.assertEqual(result["hookSpecificOutput"]["permissionDecision"], "deny")
        self.assertEqual(len(self.state["violations"]), 1)


if __name__ == "__main__":
    unittest.main()
