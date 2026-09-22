"""Tests for the durable central log mirror (proofreader_evolve/log/)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path


class RunLogCaptureTests(unittest.TestCase):

    def test_mirrors_run_log_to_central_log_dir(self) -> None:
        from proofreader_evolve.cli import run_evolution as re_mod

        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td) / "runs" / "794495_demo_20260101"
            run_dir.mkdir(parents=True)
            old_log_dir = re_mod.LOG_DIR
            re_mod.LOG_DIR = Path(td) / "log"
            try:
                with re_mod.run_log_capture(run_dir):
                    print("hello-run-log")           # stdout leg of the tee
                    import sys
                    print("hello-stderr", file=sys.stderr)
            finally:
                re_mod.LOG_DIR = old_log_dir

            run_log = (run_dir / "run.log").read_text()
            self.assertIn("hello-run-log", run_log)
            self.assertIn("hello-stderr", run_log)
            self.assertIn("# argv:", run_log)

            mirror = Path(td) / "log" / "794495_demo_20260101.log"
            self.assertTrue(mirror.exists())
            self.assertEqual(mirror.read_text(), run_log)  # byte-identical copy

            index = (Path(td) / "log" / "runs_index.tsv").read_text()
            self.assertIn("794495_demo_20260101", index)

    def test_run_survives_unwritable_mirror(self) -> None:
        from proofreader_evolve.cli import run_evolution as re_mod

        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td) / "runs" / "789202_demo"
            run_dir.mkdir(parents=True)
            blocker = Path(td) / "log"
            blocker.write_text("a FILE where the log dir should be")
            old_log_dir = re_mod.LOG_DIR
            re_mod.LOG_DIR = blocker  # mkdir/open will fail -> warn, not crash
            try:
                with re_mod.run_log_capture(run_dir):
                    print("still-logging")
            finally:
                re_mod.LOG_DIR = old_log_dir
            self.assertIn("still-logging", (run_dir / "run.log").read_text())


if __name__ == "__main__":
    unittest.main()
