"""Launcher tests: no brain loads, table writes, or API requests."""

from contextlib import redirect_stdout, redirect_stderr
import io
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

from proofreader_evolve import prepare_feature_tables as launcher


class PrepareTablesScriptTests(unittest.TestCase):
    def test_defaults_match_evolution_and_preflight(self):
        args = launcher.parse_args([])
        planned = launcher.commands(args)
        self.assertEqual(len(planned), 3)
        for brain, command in zip(("789202", "794491"), planned):
            self.assertIn(f"dataset_cache_{brain}_mcl100_add.pkl", " ".join(command))
            self.assertNotIn("--candidate-mode", command)
            self.assertNotIn("--split-max-sites", command)
            self.assertNotIn("--merge-max-sites", command)
        self.assertIn("proofreader_evolve.cli.preflight", planned[-1])
        self.assertNotIn("--check-api", planned[-1])

    def test_dry_run_never_starts_processes_or_checks_inputs(self):
        with patch.object(launcher.subprocess, "run") as run, \
                patch.object(Path, "is_file") as exists, redirect_stdout(io.StringIO()):
            self.assertEqual(launcher.main(["--dry-run"]), 0)
        run.assert_not_called()
        exists.assert_not_called()

    def test_sequential_success_and_thread_limits(self):
        with patch.object(Path, "is_file", return_value=True), \
                patch.object(launcher.subprocess, "run") as run, redirect_stdout(io.StringIO()):
            self.assertEqual(launcher.main(["--threads", "2"]), 0)
        self.assertEqual(run.call_count, 3)
        self.assertTrue(all(c.kwargs["check"] for c in run.call_args_list))
        self.assertTrue(all(c.kwargs["env"]["OPENBLAS_NUM_THREADS"] == "2" for c in run.call_args_list))

    def test_all_uses_affinity_and_logs_resolved_threads(self):
        output = io.StringIO()
        with patch.object(launcher.os, "sched_getaffinity", return_value=set(range(96)), create=True), \
                patch.object(launcher.os, "cpu_count", return_value=192), \
                patch.object(Path, "is_file", return_value=True), \
                patch.object(launcher.subprocess, "run") as run, redirect_stdout(output):
            self.assertEqual(launcher.main(["--threads", "all"]), 0)
        self.assertIn("Threads: 96 (requested: all; available CPUs: 96)", output.getvalue())
        for call in run.call_args_list:
            for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                         "NUMEXPR_NUM_THREADS", "NUMEXPR_MAX_THREADS"):
                self.assertEqual(call.kwargs["env"][name], "96")

    def test_all_cpu_count_fallback(self):
        for count, expected in ((8, 8), (None, 1)):
            with self.subTest(count=count), \
                    patch.object(launcher.os, "sched_getaffinity", side_effect=OSError, create=True), \
                    patch.object(launcher.os, "cpu_count", return_value=count):
                self.assertEqual(launcher.parse_args(["--threads", "all"]).threads, expected)

    def test_failure_stops_before_next_brain(self):
        with patch.object(Path, "is_file", return_value=True), \
                patch.object(launcher.subprocess, "run", side_effect=subprocess.CalledProcessError(7, ["test"])) as run, \
                redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.assertEqual(launcher.main([]), 7)
        self.assertEqual(run.call_count, 1)

    def test_missing_cache_stops_before_any_work(self):
        with patch.object(Path, "is_file", return_value=False), \
                patch.object(launcher.subprocess, "run") as run, redirect_stderr(io.StringIO()):
            self.assertEqual(launcher.main([]), 1)
        run.assert_not_called()

    def test_invalid_arguments(self):
        for arguments in (["--brains", "../cache"], ["--brains", "1", "1"],
                          ["--mcl", "-1"], ["--threads", "0"], ["--threads", "-1"],
                          ["--threads", "1.5"], ["--threads", "invalid"]):
            with self.subTest(arguments=arguments), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                launcher.parse_args(arguments)
