from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_module():
    path = PROJECT_ROOT / "agentic" / "rerun_experiments.py"
    spec = importlib.util.spec_from_file_location("test_rerun_module", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


rerun = _load_module()


def _record(rid: int, code: str) -> dict:
    return {
        "id": rid,
        "status": "SUCCEEDED",
        "hypothesis": f"hypothesis {rid}",
        "code": code,
        "codeOutput": "recorded",
        "_surprisal": 0.5,
        "_priority": 0.4,
    }


def _prior_result(rid: int, code: str, *, usable: bool = True) -> dict:
    return {
        "rank": rid,
        "id": rid,
        "status": "SUCCEEDED",
        "code_source": "recorded",
        "result_status": "USABLE" if usable else "UNUSABLE",
        "result_failure_reason": None if usable else "timeout",
        "rerun_exitcode": 0 if usable else None,
        "rerun_timed_out": not usable,
        "rerun_failure_kind": None,
        "rerun_runtime_ms": 1,
        "rerun_stdout": "metric=1\n" if usable else "",
        "rerun_stderr": "",
        "extrapolations": [],
        "recorded_code_sha256": rerun.code_sha256(code),
        "executed_code_sha256": rerun.code_sha256(code),
        "code_changed": False,
        "sentinel": f"base-{rid}",
    }


def _prior_payload(run_json: Path, dataset: Path, results: list[dict]) -> dict:
    return {
        "run_sha256": rerun.hashlib.sha256(run_json.read_bytes()).hexdigest(),
        "pkl_fingerprint": rerun.file_fingerprint(dataset),
        "results": results,
    }


class RerunOneCredentialTests(unittest.TestCase):
    _PROBE = (
        "import json, os\n"
        "from pathlib import Path\n"
        "name = 'zihan_gcs_token.json'\n"
        "new = 'allen-nd-goog-f5d46dbfa2cd.json'\n"
        "print(json.dumps({\n"
        "    'tok_env': os.environ.get('RERUN_GCS_TOKEN'),\n"
        "    'gac': os.environ.get('GOOGLE_APPLICATION_CREDENTIALS'),\n"
        "    'aws': os.environ.get('AWS_EC2_METADATA_DISABLED'),\n"
        "    'cwd_link': (Path.cwd() / name).is_file(),\n"
        "    'parent_link': (Path.cwd().parent / name).is_file(),\n"
        "    'pkl_link': (Path(os.environ['RERUN_PKL']).parent / name)"
        ".is_file(),\n"
        "    'cwd_new': (Path.cwd() / new).is_file(),\n"
        "    'pkl_new': (Path(os.environ['RERUN_PKL']).parent / new)"
        ".is_file(),\n"
        "}))\n"
    )

    def test_rerun_one_exposes_gcs_token_and_cloud_env(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            token = root / "allen-nd-goog-f5d46dbfa2cd.json"
            token.write_text("{}")
            pkl_dir = root / "delivery"
            pkl_dir.mkdir()
            pkl = pkl_dir / "dataset_cache_test_add.pkl"
            pkl.write_bytes(b"")
            with mock.patch.dict(rerun.os.environ):
                rerun.os.environ.pop("GOOGLE_APPLICATION_CREDENTIALS", None)
                rerun.os.environ.pop("AWS_EC2_METADATA_DISABLED", None)
                with mock.patch.object(rerun, "GCS_TOKEN_PATH", token):
                    result = rerun.rerun_one(self._PROBE, pkl, timeout=120)
            self.assertEqual(result["exitcode"], 0, result["stderr"])
            probe = json.loads(result["stdout"])
            self.assertEqual(probe["tok_env"], str(token))
            self.assertEqual(probe["gac"], str(token))
            self.assertEqual(probe["aws"], "true")
            self.assertTrue(probe["cwd_link"])
            self.assertTrue(probe["parent_link"])
            self.assertTrue(probe["pkl_link"])
            self.assertTrue(probe["cwd_new"])
            self.assertTrue(probe["pkl_new"])
            self.assertTrue((pkl_dir / token.name).is_symlink())
            self.assertTrue((pkl_dir / "zihan_gcs_token.json").is_symlink())

    def test_stale_legacy_token_link_is_repointed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            token = root / "allen-nd-goog-f5d46dbfa2cd.json"
            token.write_text("{}")
            revoked = root / "revoked.json"
            revoked.write_text("{}")
            delivery = root / "delivery"
            delivery.mkdir()
            (delivery / "zihan_gcs_token.json").symlink_to(revoked)
            user_file = root / "user"
            user_file.mkdir()
            (user_file / "zihan_gcs_token.json").write_text("user")
            with mock.patch.object(rerun, "GCS_TOKEN_PATH", token):
                rerun._link_gcs_token(delivery)
                rerun._link_gcs_token(user_file)
            self.assertEqual((delivery / "zihan_gcs_token.json").resolve(), token.resolve())
            self.assertEqual((user_file / "zihan_gcs_token.json").read_text(), "user")

    def test_rerun_one_without_token_still_disables_ec2_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pkl_dir = root / "delivery"
            pkl_dir.mkdir()
            pkl = pkl_dir / "dataset_cache_test_add.pkl"
            pkl.write_bytes(b"")
            missing = root / "allen-nd-goog-f5d46dbfa2cd.json"  # never written
            with mock.patch.dict(rerun.os.environ):
                rerun.os.environ.pop("GOOGLE_APPLICATION_CREDENTIALS", None)
                rerun.os.environ.pop("AWS_EC2_METADATA_DISABLED", None)
                with mock.patch.object(rerun, "GCS_TOKEN_PATH", missing):
                    result = rerun.rerun_one(self._PROBE, pkl, timeout=120)
            self.assertEqual(result["exitcode"], 0, result["stderr"])
            probe = json.loads(result["stdout"])
            self.assertIsNone(probe["tok_env"])
            self.assertIsNone(probe["gac"])
            self.assertEqual(probe["aws"], "true")
            self.assertFalse(probe["pkl_link"])
            self.assertFalse((pkl_dir / missing.name).exists())


class RerunForkedPreloadTests(unittest.TestCase):
    def setUp(self) -> None:
        if not hasattr(rerun.os, "fork"):
            self.skipTest("fork is unavailable on this platform")

    @staticmethod
    def _dataset(root: Path) -> Path:
        pkl = root / "dataset_cache_forktest_add.pkl"
        pkl.write_bytes(rerun.pickle.dumps({"marker": 41, "mut": []}))
        return pkl

    def test_forked_run_uses_resident_payload_and_isolates_mutations(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pkl = self._dataset(root)
            resident = {"marker": 42, "mut": []}
            code = (
                "import os, pickle\n"
                "with open(os.environ['RERUN_PKL'], 'rb') as f:\n"
                "    p = pickle.load(f)\n"
                "p['mut'].append(1)\n"
                "print('marker', p['marker'], 'mut', len(p['mut']))\n"
            )
            with mock.patch.object(rerun, "_PRELOAD_ENABLED", True), \
                    mock.patch.dict(rerun._PAYLOAD_CACHE,
                                    {str(pkl.resolve()): resident}, clear=True):
                first = rerun.rerun_one(code, pkl, timeout=60)
                second = rerun.rerun_one(code, pkl, timeout=60)
            self.assertEqual(first["exitcode"], 0, first["stderr"])
            # marker 42 (resident), not 41 (on disk): pickle.load short-circuited.
            self.assertIn("marker 42 mut 1", first["stdout"])
            # The first child's mutation never reaches the parent or the second
            # child: copy-on-write keeps each fork's payload private.
            self.assertIn("marker 42 mut 1", second["stdout"])
            self.assertEqual(resident["mut"], [])

    def test_forked_run_times_out_and_kills_the_child(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pkl = self._dataset(root)
            code = "import time\nprint('started', flush=True)\ntime.sleep(60)\n"
            start = rerun.time.monotonic()
            with mock.patch.object(rerun, "_PRELOAD_ENABLED", True), \
                    mock.patch.dict(rerun._PAYLOAD_CACHE,
                                    {str(pkl.resolve()): {}}, clear=True):
                run = rerun.rerun_one(code, pkl, timeout=2)
            self.assertTrue(run["timed_out"])
            self.assertIsNone(run["exitcode"])
            self.assertIn("started", run["stdout"])
            self.assertIn("timed out after 2s", run["stderr"])
            self.assertLess(rerun.time.monotonic() - start, 30)

    def test_forked_run_reports_exception_and_exitcode(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pkl = self._dataset(root)
            with mock.patch.object(rerun, "_PRELOAD_ENABLED", True), \
                    mock.patch.dict(rerun._PAYLOAD_CACHE,
                                    {str(pkl.resolve()): {}}, clear=True):
                run = rerun.rerun_one("raise RuntimeError('boom-fork')\n",
                                      pkl, timeout=60)
            self.assertEqual(run["exitcode"], 1)
            self.assertFalse(run["timed_out"])
            self.assertIn("boom-fork", run["stderr"])

    def test_preload_failure_falls_back_to_subprocess_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pkl = root / "dataset_cache_broken_add.pkl"
            pkl.write_bytes(b"")  # unpicklable — preload must fail gracefully
            captured = io.StringIO()
            with mock.patch.object(rerun, "_PRELOAD_ENABLED", True), \
                    mock.patch.dict(rerun._PAYLOAD_CACHE, {}, clear=True), \
                    contextlib.redirect_stderr(captured):
                run = rerun.rerun_one("print('subprocess-ok')\n", pkl, timeout=60)
                cached = rerun._PAYLOAD_CACHE.get(str(pkl.resolve()), "unset")
            self.assertEqual(run["exitcode"], 0, run["stderr"])
            self.assertIn("subprocess-ok", run["stdout"])
            self.assertIn("preload failed", captured.getvalue())
            self.assertIsNone(cached)  # failure is cached, so no retry storms


class RerunWorkItemTests(unittest.TestCase):
    def test_export_scripts_removes_unselected_stale_scripts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            export_dir = root / "rerun"
            export_dir.mkdir()
            (export_dir / "hypo_9.py").write_text("stale")
            (export_dir / "keep.txt").write_text("user file")

            rerun.export_scripts(
                [_record(1, "print('one')")], root / "data.pkl", export_dir
            )

            self.assertFalse((export_dir / "hypo_9.py").exists())
            self.assertTrue((export_dir / "hypo_1.py").is_file())
            self.assertEqual((export_dir / "keep.txt").read_text(), "user file")
            manifest = json.loads((export_dir / "MANIFEST.json").read_text())
            self.assertEqual([row["id"] for row in manifest["records"]], [1])

    def test_zero_exit_dataset_gate_is_unusable(self) -> None:
        status, reason = rerun.result_usability(
            {
                "timed_out": False,
                "exitcode": 0,
                "stdout": "No dataset found.\n",
                "stderr": "",
            }
        )
        self.assertEqual((status, reason), ("UNUSABLE", "dataset-loading"))

    def test_checkpoint_key_invalidates_on_code_and_dataset_change(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset = root / "data.pkl"
            dataset.write_bytes(b"one")
            checkpoint_path = root / "checkpoint.json"
            checkpoint = rerun._load_checkpoint(checkpoint_path)
            calls = []

            def fake_run(code, path, timeout, companions=()):
                calls.append((code, path.read_bytes(), timeout))
                return {
                    "timed_out": False,
                    "exitcode": 0,
                    "stdout": "metric=1\n",
                    "stderr": "",
                    "runtime_ms": 1,
                }

            with mock.patch.object(rerun, "rerun_one", side_effect=fake_run):
                _, reused = rerun._run_work_item(
                    "code-a", dataset, 10, run_sha256="run", record_id=1,
                    checkpoint=checkpoint, checkpoint_path=checkpoint_path,
                )
                self.assertFalse(reused)
                _, reused = rerun._run_work_item(
                    "code-a", dataset, 10, run_sha256="run", record_id=1,
                    checkpoint=checkpoint, checkpoint_path=checkpoint_path,
                )
                self.assertTrue(reused)
                rerun._run_work_item(
                    "code-b", dataset, 10, run_sha256="run", record_id=1,
                    checkpoint=checkpoint, checkpoint_path=checkpoint_path,
                )
                dataset.write_bytes(b"two-two")
                rerun._run_work_item(
                    "code-b", dataset, 10, run_sha256="run", record_id=1,
                    checkpoint=checkpoint, checkpoint_path=checkpoint_path,
                )

            self.assertEqual(len(calls), 3)
            saved = json.loads(checkpoint_path.read_text())
            self.assertEqual(len(saved["work_items"]), 3)

    def test_checkpointed_environment_failure_is_run_again(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset = root / "data.pkl"
            dataset.write_bytes(b"one")
            checkpoint_path = root / "checkpoint.json"
            checkpoint = rerun._load_checkpoint(checkpoint_path)
            runs = [
                {"timed_out": False, "exitcode": 1, "stdout": "", "runtime_ms": 1,
                 "stderr": "ModuleNotFoundError: No module named 'lightgbm'"},
                {"timed_out": False, "exitcode": 0, "stdout": "metric=1\n",
                 "stderr": "", "runtime_ms": 1},
            ]
            item = dict(run_sha256="run", record_id=1, checkpoint=checkpoint,
                        checkpoint_path=checkpoint_path)
            with mock.patch.object(rerun, "rerun_one", side_effect=runs) as run_one:
                first, reused_first = rerun._run_work_item("code", dataset, 10, **item)
                second, reused_second = rerun._run_work_item("code", dataset, 10, **item)
                third, reused_third = rerun._run_work_item("code", dataset, 10, **item)

            self.assertEqual(run_one.call_count, 2)
            self.assertEqual((first["exitcode"], reused_first), (1, False))
            self.assertEqual((second["exitcode"], reused_second), (0, False))
            self.assertEqual((third["exitcode"], reused_third), (0, True))

    def test_missing_module_names_only_absent_top_level_packages(self) -> None:
        absent = "No module named 'surely_absent_pkg_xyz'"
        self.assertEqual(rerun.missing_module(absent), "surely_absent_pkg_xyz")
        self.assertEqual(
            rerun.missing_module("No module named 'surely_absent_pkg_xyz.sub'"),
            "surely_absent_pkg_xyz",
        )
        self.assertIsNone(rerun.missing_module("No module named 'numpy._core'"))
        self.assertIsNone(rerun.missing_module("ValueError: bad shape"))
        self.assertEqual(
            rerun.missing_module(f"{absent}\nduring handling\nNo module named 'json.x'"),
            "surely_absent_pkg_xyz",
        )

    def test_install_missing_installs_allowlisted_package_once_and_reruns(self) -> None:
        fail = {"timed_out": False, "exitcode": 1, "stdout": "", "runtime_ms": 1,
                "stderr": "ModuleNotFoundError: No module named 'surely_absent_pkg_xyz'"}
        ok = {"timed_out": False, "exitcode": 0, "stdout": "metric=1\n",
              "stderr": "", "runtime_ms": 1}
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = Path(tmpdir) / "data.pkl"
            dataset.write_bytes(b"one")
            with (
                mock.patch.object(rerun, "_INSTALL_MISSING", True),
                mock.patch.dict(rerun._INSTALLED, clear=True),
                mock.patch.dict(rerun.INSTALLABLE_MODULES,
                                {"surely_absent_pkg_xyz": "absent-pkg"}),
                mock.patch.object(rerun, "_pip_install",
                                  return_value={"version": "1.0"}) as pip,
                mock.patch.object(rerun, "rerun_one",
                                  side_effect=[fail, ok, ok]) as run_one,
                contextlib.redirect_stderr(io.StringIO()),
            ):
                first = rerun._run_installing_missing("code-a", dataset, 10)
                second = rerun._run_installing_missing("code-b", dataset, 10)
                installed = dict(rerun._INSTALLED)

        self.assertEqual((first["exitcode"], second["exitcode"]), (0, 0))
        self.assertEqual(run_one.call_count, 3)
        pip.assert_called_once_with("absent-pkg")
        self.assertEqual(installed["surely_absent_pkg_xyz"]["version"], "1.0")

    def test_install_missing_never_fetches_unlisted_modules_or_runs_when_off(self) -> None:
        fail = {"timed_out": False, "exitcode": 1, "stdout": "", "runtime_ms": 1,
                "stderr": "ModuleNotFoundError: No module named 'surely_absent_pkg_xyz'"}
        dataset = Path("unused.pkl")
        for enabled in (True, False):
            with (
                mock.patch.object(rerun, "_INSTALL_MISSING", enabled),
                mock.patch.dict(rerun._INSTALLED, clear=True),
                mock.patch.object(rerun, "_pip_install") as pip,
                mock.patch.object(rerun, "rerun_one", return_value=fail) as run_one,
                contextlib.redirect_stderr(io.StringIO()),
            ):
                run = rerun._run_installing_missing("code", dataset, 10)
                installed = dict(rerun._INSTALLED)
            self.assertEqual(run["exitcode"], 1)
            self.assertEqual(run_one.call_count, 1)
            pip.assert_not_called()
            if enabled:
                self.assertIn("not in INSTALLABLE_MODULES",
                              installed["surely_absent_pkg_xyz"]["error"])
            else:
                self.assertEqual(installed, {})

    def test_pip_install_pins_every_installed_distribution(self) -> None:
        seen = {}

        def fake_pip(cmd, **kwargs):
            seen["cmd"] = cmd
            seen["constraints"] = Path(cmd[cmd.index("-c") + 1]).read_text()
            return subprocess.CompletedProcess(cmd, 0, "", "")

        with (
            tempfile.TemporaryDirectory() as prefix,
            mock.patch.object(rerun.subprocess, "run", side_effect=fake_pip),
            mock.patch.object(rerun.sys, "prefix", prefix),
        ):
            record = rerun._pip_install("numpy")

        import numpy
        self.assertEqual(seen["cmd"][-1], "numpy")
        self.assertIn(f"numpy=={numpy.__version__}", seen["constraints"].splitlines())
        self.assertEqual(record, {"version": numpy.__version__})

    def test_only_changed_reruns_subset_and_merges_base_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_json = root / "run.json"
            run_json.write_text("[]")
            dataset = root / "data.pkl"
            dataset.write_bytes(b"data")
            code_dir = root / "code"
            code_dir.mkdir()
            records = [_record(1, "print('one')"), _record(2, "print('two')")]
            (code_dir / "hypo_1.py").write_text("print('one')")
            (code_dir / "hypo_2.py").write_text("print('two-fixed')")
            base = root / "base.json"
            base.write_text(
                json.dumps(
                    _prior_payload(
                        run_json,
                        dataset,
                        [
                            _prior_result(1, "print('one')"),
                            _prior_result(2, "print('two')"),
                        ],
                    )
                )
            )

            def fake_run(code, path, timeout, companions=()):
                return {
                    "timed_out": False,
                    "exitcode": 0,
                    "stdout": "metric=2\n",
                    "stderr": "",
                    "runtime_ms": 2,
                }

            output = io.StringIO()
            with (
                mock.patch.object(rerun, "load_records", return_value=records),
                mock.patch.object(
                    rerun, "rank_records", return_value=(records, None, [])
                ),
                mock.patch.object(rerun, "preflight_env", return_value=(True, "2")),
                mock.patch.object(rerun, "rerun_one", side_effect=fake_run) as run_one,
                contextlib.redirect_stdout(output),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                rc = rerun.main(
                    [
                        str(run_json), "--pkl", str(dataset),
                        "--code-dir", str(code_dir), "--only-changed",
                        "--base-results", str(base),
                    ]
                )

            payload = json.loads(output.getvalue())
            self.assertEqual(rc, 0)
            self.assertEqual(run_one.call_count, 1)
            self.assertEqual(payload["n_executed"], 1)
            self.assertEqual(payload["n_base_reused"], 1)
            self.assertEqual([item["id"] for item in payload["results"]], [1, 2])
            self.assertEqual(payload["results"][0]["sentinel"], "base-1")
            self.assertTrue(payload["results"][1]["code_changed"])

    def test_only_changed_retries_environment_failures_with_unchanged_code(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_json = root / "run.json"
            run_json.write_text("[]")
            dataset = root / "data.pkl"
            dataset.write_bytes(b"data")
            code_dir = root / "code"
            code_dir.mkdir()
            records = [_record(1, "import lightgbm"), _record(2, "print('two')")]
            (code_dir / "hypo_1.py").write_text("import lightgbm")
            (code_dir / "hypo_2.py").write_text("print('two')")
            env_failed = _prior_result(1, "import lightgbm", usable=False)
            env_failed.update(
                rerun_timed_out=False, rerun_exitcode=1, rerun_failure_kind="environment",
                result_failure_reason="nonzero-exit",
                rerun_stderr="ModuleNotFoundError: No module named 'lightgbm'",
            )
            base = root / "base.json"
            base.write_text(json.dumps(_prior_payload(
                run_json, dataset, [env_failed, _prior_result(2, "print('two')")]
            )))

            def fake_run(code, path, timeout, companions=()):
                return {"timed_out": False, "exitcode": 0, "stdout": "metric=2\n",
                        "stderr": "", "runtime_ms": 2}

            output = io.StringIO()
            with (
                mock.patch.object(rerun, "load_records", return_value=records),
                mock.patch.object(
                    rerun, "rank_records", return_value=(records, None, [])
                ),
                mock.patch.object(rerun, "preflight_env", return_value=(True, "2")),
                mock.patch.object(rerun, "rerun_one", side_effect=fake_run) as run_one,
                contextlib.redirect_stdout(output),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                rc = rerun.main(
                    [
                        str(run_json), "--pkl", str(dataset),
                        "--code-dir", str(code_dir), "--only-changed",
                        "--base-results", str(base),
                    ]
                )

            payload = json.loads(output.getvalue())
            self.assertEqual(rc, 0)
            self.assertEqual(run_one.call_count, 1)
            self.assertEqual(payload["results"][0]["result_status"], "USABLE")
            self.assertFalse(payload["results"][0]["code_changed"])
            self.assertEqual(payload["results"][1]["sentinel"], "base-2")

    def test_extra_only_never_executes_origin_and_skips_unusable_origin(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_json = root / "run.json"
            run_json.write_text("[]")
            origin = root / "origin.pkl"
            extra = root / "extra.pkl"
            origin.write_bytes(b"origin")
            extra.write_bytes(b"extra")
            records = [_record(1, "print('one')"), _record(2, "print('two')")]
            origin_results = root / "origin.json"
            origin_results.write_text(
                json.dumps(
                    _prior_payload(
                        run_json,
                        origin,
                        [
                            _prior_result(1, "print('one')"),
                            _prior_result(2, "print('two')", usable=False),
                        ],
                    )
                )
            )
            called_paths = []

            def fake_run(code, path, timeout, companions=()):
                called_paths.append(path.resolve())
                return {
                    "timed_out": False,
                    "exitcode": 0,
                    "stdout": "metric=3\n",
                    "stderr": "",
                    "runtime_ms": 3,
                }

            output = io.StringIO()
            with (
                mock.patch.object(rerun, "load_records", return_value=records),
                mock.patch.object(
                    rerun, "rank_records", return_value=(records, None, [])
                ),
                mock.patch.object(rerun, "preflight_env", return_value=(True, "2")),
                mock.patch.object(rerun, "rerun_one", side_effect=fake_run),
                contextlib.redirect_stdout(output),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                rc = rerun.main(
                    [
                        str(run_json), "--pkl", str(origin),
                        "--extra-pkl", str(extra), "--extra-only",
                        "--origin-results", str(origin_results),
                    ]
                )

            payload = json.loads(output.getvalue())
            self.assertEqual(rc, 0)
            self.assertEqual(called_paths, [extra.resolve()])
            self.assertEqual(len(payload["results"][0]["extrapolations"]), 1)
            self.assertEqual(payload["results"][1]["extrapolations"], [])

    def test_only_corrected_runs_only_scripts_present_in_corrected_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_json = root / "run.json"
            run_json.write_text("[]")
            dataset = root / "data.pkl"
            dataset.write_bytes(b"data")
            corrected_dir = root / "corrected"
            corrected_dir.mkdir()
            (corrected_dir / "hypo_2.py").write_text("print('corrected')")
            records = [_record(1, "print('one')"), _record(2, "print('two')")]

            output = io.StringIO()
            with (
                mock.patch.object(rerun, "load_records", return_value=records),
                mock.patch.object(
                    rerun, "rank_records", return_value=(records, None, [])
                ),
                mock.patch.object(rerun, "preflight_env", return_value=(True, "2")),
                mock.patch.object(
                    rerun,
                    "rerun_one",
                    return_value={
                        "timed_out": False,
                        "exitcode": 0,
                        "stdout": "metric=4\n",
                        "stderr": "",
                        "runtime_ms": 4,
                    },
                ) as run_one,
                contextlib.redirect_stdout(output),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                rc = rerun.main(
                    [
                        str(run_json), "--pkl", str(dataset),
                        "--corrected-dir", str(corrected_dir), "--only-corrected",
                    ]
                )

            payload = json.loads(output.getvalue())
            self.assertEqual(rc, 0)
            self.assertEqual(run_one.call_count, 1)
            self.assertEqual([item["id"] for item in payload["results"]], [2])
            self.assertEqual(payload["results"][0]["code_source"], "corrected")


class CompanionTests(unittest.TestCase):
    """--companion: binding by manifest, digest check, and per-brain linking."""

    @staticmethod
    def _companion(root: Path, brain: str, kind: str, pkl_name: str, marker: int) -> Path:
        import hashlib
        import numpy as np
        npz = root / f"rerank_companion_{brain}_{kind}.npz"
        np.savez(npz, marker=np.array([marker]))
        (root / f"rerank_companion_{brain}_{kind}.json").write_text(json.dumps({
            "brain": brain, "kind": kind, "source_cache": {"file": pkl_name},
            "npz_sha256": hashlib.sha256(npz.read_bytes()).hexdigest()}))
        return npz

    def _datasets(self, root: Path) -> tuple[Path, Path]:
        a, b = root / "dataset_cache_111_mcl100_add.pkl", root / "dataset_cache_222_mcl100_add.pkl"
        for pkl in (a, b):
            pkl.write_bytes(rerun.pickle.dumps({}))
        return a, b

    def test_bind_by_manifest_and_link_under_code_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            a, b = self._datasets(root)
            ca = self._companion(root, "111", "split", a.name, 1)
            cb = self._companion(root, "222", "split", b.name, 2)
            bound = rerun.bind_companions([ca, cb.with_suffix(".json")], [a, b])
            self.assertEqual([f.name for f in bound[a]],
                             ["rerank_companion_111_split.npz", "rerank_companion_111_split.json"])
            self.assertEqual(bound[b][0].name, "rerank_companion_222_split.npz")
            # Code written for brain 111, re-run on 222: the 111 name points at 222's file.
            links = rerun.companion_links("np.load('rerank_companion_111_split.npz')", bound[b])
            self.assertEqual(links["rerank_companion_111_split.npz"], str(bound[b][0]))
            self.assertEqual(links["rerank_companion_222_split.json"], str(bound[b][1]))

    def test_bind_rejects_unknown_pkl_digest_mismatch_and_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            a, b = self._datasets(root)
            c = self._companion(root, "222", "split", b.name, 2)
            with self.assertRaisesRegex(ValueError, "not among the dataset pkls"):
                rerun.bind_companions([c], [a])
            d = self._companion(root, "333", "merge", a.name, 3)
            d.write_bytes(d.read_bytes() + b"x")
            with self.assertRaisesRegex(ValueError, "digest"):
                rerun.bind_companions([d], [a])
            e = self._companion(root, "111", "split", a.name, 1)
            f = self._companion(root, "111b", "split", a.name, 4)
            with self.assertRaisesRegex(ValueError, "Two split companions"):
                rerun.bind_companions([e, f], [a])
            with self.assertRaisesRegex(ValueError, "Not a companion"):
                rerun.bind_companions([a], [a])

    def test_rerun_one_reads_the_bound_companion_on_both_paths(self) -> None:
        code = ("import numpy as np, os\n"
                "m = int(np.load('rerank_companion_111_split.npz')['marker'][0])\n"
                "n = int(np.load('/elsewhere/rerank_companion_111_split.npz')['marker'][0])\n"
                "print('marker', m, n, os.path.exists('/elsewhere/rerank_companion_111_split.json'))\n")
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            a, b = self._datasets(root)
            self._companion(root, "111", "split", a.name, 1)
            cb = self._companion(root, "222", "split", b.name, 2)
            bound = rerun.bind_companions([cb], [a, b])
            for preload in (False, True):
                if preload and not hasattr(rerun.os, "fork"):
                    continue
                with mock.patch.object(rerun, "_PRELOAD_ENABLED", preload), \
                        mock.patch.dict(rerun._PAYLOAD_CACHE, {str(b.resolve()): {}}, clear=True):
                    run = rerun.rerun_one(code, b, timeout=60, companions=bound[b])
                self.assertEqual(run["exitcode"], 0, run["stderr"])
                self.assertIn("marker 2 2 True", run["stdout"])

    def test_work_key_depends_on_companions_only_when_given(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            a, _ = self._datasets(root)
            c = self._companion(root, "111", "split", a.name, 1)
            base = dict(run_sha256="r", record_id=1, code_hash="c", dataset=a, timeout=5)
            plain = rerun._work_key(**base)
            self.assertEqual(plain, rerun._work_key(**base, companions=()))
            self.assertNotEqual(plain, rerun._work_key(**base, companions=[c]))

    @staticmethod
    def _set_image(companion: Path, folder: str, scale: list[float]) -> None:
        manifest = companion.with_suffix(".json")
        payload = json.loads(manifest.read_text())
        payload["image"] = {
            "uri": f"s3://open-bucket/{folder}/fusion/fused.zarr/",
            "levels": {"0": {"scale_xyz_um": scale, "translation_xyz_um": [0, 0, 0]}},
        }
        manifest.write_text(json.dumps(payload))

    def test_retarget_image_rewrites_same_geometry_and_refuses_otherwise(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            a, b = self._datasets(root)
            ca = self._companion(root, "111", "merge", a.name, 1)
            cb = self._companion(root, "222", "merge", b.name, 2)
            self._set_image(ca, "exaSPIM_111_x", [0.748, 0.748, 1.0])
            self._set_image(cb, "exaSPIM_222_y", [0.748, 0.748, 1.0])
            bound = rerun.bind_companions([ca, cb], [a, b])
            code = "path = 'exaSPIM_111_x/fusion/fused.zarr/0'"

            self.assertEqual(
                rerun.retarget_image(code, bound[a], bound[b]),
                ("path = 'exaSPIM_222_y/fusion/fused.zarr/0'",
                 "exaSPIM_111_x -> exaSPIM_222_y", None),
            )
            self.assertEqual(
                rerun.retarget_image("np.load('x.npz')", bound[a], bound[b]),
                ("np.load('x.npz')", None, None),
            )
            self.assertIn("no companion", rerun.retarget_image(code, bound[a], [])[2])
            self._set_image(cb, "exaSPIM_222_y", [1.0, 1.0, 1.0])
            unchanged, note, refusal = rerun.retarget_image(code, bound[a], bound[b])
            self.assertEqual((unchanged, note), (code, None))
            self.assertIn("level geometry", refusal)

    def test_extrapolation_runs_retargeted_code_and_skips_refused_scripts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            a, b = self._datasets(root)
            ca = self._companion(root, "111", "merge", a.name, 1)
            cb = self._companion(root, "222", "merge", b.name, 2)
            self._set_image(ca, "exaSPIM_111_x", [0.748, 0.748, 1.0])
            self._set_image(cb, "exaSPIM_222_y", [0.748, 0.748, 1.0])
            run_json = root / "run.json"
            run_json.write_text("[]")
            hardcoded = "print('exaSPIM_111_x/fusion/fused.zarr/0')"
            records = [_record(1, hardcoded), _record(2, "print('manifest')")]
            origin_results = root / "origin.json"
            origin_results.write_text(json.dumps(_prior_payload(
                run_json, a, [_prior_result(1, hardcoded), _prior_result(2, "print('manifest')")],
            )))
            executed = []

            def fake_run(code, path, timeout, companions=()):
                executed.append(code)
                return {"timed_out": False, "exitcode": 0, "stdout": "metric=3\n",
                        "stderr": "", "runtime_ms": 3}

            def run_main() -> dict:
                output = io.StringIO()
                with (
                    mock.patch.object(rerun, "load_records", return_value=records),
                    mock.patch.object(rerun, "rank_records", return_value=(records, None, [])),
                    mock.patch.object(rerun, "preflight_env", return_value=(True, "2")),
                    mock.patch.object(rerun, "rerun_one", side_effect=fake_run),
                    contextlib.redirect_stdout(output),
                    contextlib.redirect_stderr(io.StringIO()),
                ):
                    rc = rerun.main([
                        str(run_json), "--pkl", str(a), "--extra-pkl", str(b),
                        "--extra-only", "--origin-results", str(origin_results),
                        "--companion", str(ca), str(cb),
                    ])
                self.assertEqual(rc, 0)
                return json.loads(output.getvalue())

            payload = run_main()
            self.assertEqual(executed, [
                "print('exaSPIM_222_y/fusion/fused.zarr/0')", "print('manifest')",
            ])
            first = payload["results"][0]["extrapolations"][0]
            self.assertEqual(first["image_retargeted"], "exaSPIM_111_x -> exaSPIM_222_y")
            self.assertNotIn("image_retargeted", payload["results"][1]["extrapolations"][0])

            executed.clear()
            self._set_image(cb, "exaSPIM_222_y", [1.0, 1.0, 1.0])
            payload = run_main()
            self.assertEqual(executed, ["print('manifest')"])
            refused = payload["results"][0]["extrapolations"][0]
            self.assertEqual(
                (refused["result_status"], refused["result_failure_reason"]),
                ("UNUSABLE", "origin-image-path"),
            )


# A stand-in tensorstore: ``open`` rejects a spec marked bad, like panda's 0.1.71
# rejecting {"type": "anonymous"} S3 credentials (new-merge-1 id 33).
_FAKE_TENSORSTORE = '''
class Future:
    def __init__(self, value):
        self._value = value

    def result(self, timeout=None):
        return self._value


class TensorStore:
    def __array__(self, dtype=None, copy=None):
        import numpy
        return numpy.zeros(2)


def open(spec, **kwargs):
    if spec.get("bad"):
        raise ValueError('Object includes extra members: "type"')
    return Future(TensorStore())
'''

_READ_SPECS = '''
import os, sys
sys.path.insert(0, os.environ["FAKE_TS_DIR"])
import numpy as np
import tensorstore as ts
patches = 0
for spec in SPECS:
    try:
        patches += int(np.array(ts.open(spec).result()).size > 0)
    except Exception as exc:
        print("open failed:", exc)
print("patches", patches)
print("delta AP +0.00000")
'''


class InputAuditTests(unittest.TestCase):
    def _run(self, code: str, root: Path, *, preload: bool) -> dict:
        fake = root / "fake"
        (fake / "tensorstore").mkdir(parents=True, exist_ok=True)
        (fake / "tensorstore" / "__init__.py").write_text(_FAKE_TENSORSTORE)
        pkl = root / "dataset_cache_audit_add.pkl"
        pkl.write_bytes(rerun.pickle.dumps({}))
        resident = {str(pkl.resolve()): {}} if preload else {}
        with mock.patch.object(rerun, "_PRELOAD_ENABLED", preload), \
                mock.patch.dict(rerun._PAYLOAD_CACHE, resident, clear=True), \
                mock.patch.dict(rerun.os.environ, {"FAKE_TS_DIR": str(fake)}), \
                mock.patch.dict(rerun.sys.modules), \
                contextlib.redirect_stderr(io.StringIO()):
            # A forked child inherits this process's modules: hide a real
            # tensorstore another test imported so the child loads the fake.
            for name in [m for m in rerun.sys.modules if m.split(".")[0] == "tensorstore"]:
                del rerun.sys.modules[name]
            return rerun.rerun_one(code, pkl, timeout=120)

    @staticmethod
    def _paths():
        return (False, True) if hasattr(rerun.os, "fork") else (False,)

    def test_swallowed_read_is_unusable_on_both_paths(self) -> None:
        for preload in self._paths():
            with self.subTest(preload=preload), tempfile.TemporaryDirectory() as tmpdir:
                run = self._run("SPECS = [{'bad': True}]\n" + _READ_SPECS,
                                Path(tmpdir), preload=preload)
                self.assertEqual(run["exitcode"], 0, run["stderr"])
                self.assertEqual(
                    {k: run["input_audit"]["tensorstore"][k] for k in ("ok", "failed")},
                    {"ok": 0, "failed": 1},
                )
                self.assertEqual(rerun.result_usability(run),
                                 ("UNUSABLE", "swallowed-input-error"))
                self.assertEqual(
                    run["stderr"].strip().splitlines()[-1],
                    "[runner] swallowed input error: tensorstore: ValueError: "
                    "Object includes extra members: \"type\"",
                )

    def test_partial_read_failures_stay_usable(self) -> None:
        for preload in self._paths():
            with self.subTest(preload=preload), tempfile.TemporaryDirectory() as tmpdir:
                run = self._run("SPECS = [{'bad': True}, {}, {}]\n" + _READ_SPECS,
                                Path(tmpdir), preload=preload)
                self.assertIn("patches 2", run["stdout"])
                counts = run["input_audit"]["tensorstore"]
                # two opens, two results and two array reads succeeded
                self.assertEqual((counts["ok"], counts["failed"]), (6, 1))
                self.assertEqual(rerun.result_usability(run), ("USABLE", None))
                self.assertNotIn("[runner] swallowed", run["stderr"])

    def test_corrupt_npz_array_is_swallowed_but_absent_keys_are_not(self) -> None:
        import numpy as np
        import zipfile

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            npz = root / "companion.npz"
            np.savez(npz, a=np.arange(3), b=np.arange(4))
            info = zipfile.ZipFile(npz).getinfo("b.npy")
            data = bytearray(npz.read_bytes())
            extra = int.from_bytes(data[info.header_offset + 28:info.header_offset + 30],
                                   "little")
            start = info.header_offset + 30 + len(info.filename.encode()) + extra
            data[start:start + 6] = b"\0" * 6  # break b.npy's magic string
            npz.write_bytes(bytes(data))
            code = (
                "import numpy as np\n"
                f"d = np.load({str(npz)!r})\n"
                "a = d['a']\n"
                "try:\n    d['missing']\nexcept KeyError:\n    pass\n"
                "try:\n    d['b']\nexcept Exception as exc:\n"
                "    print('b unreadable:', type(exc).__name__)\n"
                "print('rows', len(a))\n"
            )
            run = self._run(code, root, preload=False)
        audit = run["input_audit"]
        self.assertEqual(run["exitcode"], 0, run["stderr"])
        self.assertEqual(audit["np.load:companion.npz"]["ok"], 1)
        self.assertEqual(audit["npz:companion.npz[a]"]["ok"], 1)
        self.assertEqual(audit["npz:companion.npz[missing]"]["absent"], 1)
        self.assertEqual(audit["npz:companion.npz[b]"]["failed"], 1)
        self.assertEqual(rerun.swallowed_inputs(audit), ["npz:companion.npz[b]"])
        self.assertEqual(rerun.result_usability(run),
                         ("UNUSABLE", "swallowed-input-error"))

    def test_zero_image_inputs_line_is_unusable(self) -> None:
        def usability(stdout: str, audit=None) -> tuple:
            return rerun.result_usability({"timed_out": False, "exitcode": 0,
                                           "stdout": stdout, "input_audit": audit})

        zero = ("UNUSABLE", "zero-inputs")
        self.assertEqual(usability(
            "INPUTS READ fragment_nodes=10 companion_rows=5 image_patches=0/120\n"), zero)
        self.assertEqual(usability(
            "INPUTS READ: 0 fragment nodes, 0 companion rows, 0 image patches read.\n"),
            zero)
        self.assertEqual(usability(
            "INPUTS READ fragment_nodes=10 companion_rows=5 image_patches=0/0\n"),
            ("USABLE", None))
        self.assertEqual(usability("INPUTS READ: 11245 image patches read\n"),
                         ("USABLE", None))
        self.assertEqual(
            usability("metric=1\n", {"tensorstore": {"ok": 0, "failed": 2}}),
            ("UNUSABLE", "swallowed-input-error"),
        )


class CompatRewriteTests(unittest.TestCase):
    CODE = (
        "spec = {'kvstore': {'driver': 's3', "
        "'aws_credentials': {'type': 'anonymous'}}}\n"
        "other = dict(aws_credentials={\"type\": \"anonymous\"})\n"
    )

    def test_rewrite_applies_only_when_the_probe_says_so(self) -> None:
        with mock.patch.object(rerun, "_compat_needed", return_value=True):
            code, names = rerun.apply_compat_rewrites(self.CODE)
        self.assertEqual(names, ["tensorstore-anonymous-credentials"])
        self.assertIn("'aws_credentials': {'anonymous': True}", code)
        self.assertIn('aws_credentials={"anonymous": True}', code)
        with mock.patch.object(rerun, "_compat_needed", return_value=False):
            self.assertEqual(rerun.apply_compat_rewrites(self.CODE), (self.CODE, []))
        with mock.patch.object(rerun, "_compat_needed") as needed:
            self.assertEqual(rerun.apply_compat_rewrites("print(1)\n"), ("print(1)\n", []))
        needed.assert_not_called()  # no probe unless the pattern occurs

    def test_work_item_runs_and_keys_the_rewritten_code(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = Path(tmpdir) / "data.pkl"
            dataset.write_bytes(b"one")
            checkpoint = rerun._load_checkpoint(None)
            executed = []

            def fake_run(code, path, timeout, companions=()):
                executed.append(code)
                return {"timed_out": False, "exitcode": 0, "stdout": "metric=1\n",
                        "stderr": "", "runtime_ms": 1}

            with mock.patch.object(rerun, "_compat_needed", return_value=True), \
                    mock.patch.object(rerun, "rerun_one", side_effect=fake_run), \
                    contextlib.redirect_stderr(io.StringIO()):
                run, reused = rerun._run_work_item(
                    self.CODE, dataset, 10, run_sha256="run", record_id=33,
                    checkpoint=checkpoint, checkpoint_path=None,
                )
        self.assertFalse(reused)
        self.assertEqual(run["compat_rewrites"], ["tensorstore-anonymous-credentials"])
        self.assertNotIn("'type'", executed[0])
        (item,) = checkpoint["work_items"].values()
        self.assertEqual(item["code_sha256"], rerun.code_sha256(executed[0]))

    def test_result_keeps_the_script_hash_and_lists_rewrites(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_json = root / "run.json"
            run_json.write_text("[]")
            origin = root / "origin.pkl"
            origin.write_bytes(b"origin")
            records = [_record(33, self.CODE)]

            def fake_run(code, path, timeout, companions=()):
                return {"timed_out": False, "exitcode": 0, "stdout": "metric=1\n",
                        "stderr": "", "runtime_ms": 1,
                        "input_audit": {"tensorstore": {"ok": 3, "failed": 0}}}

            output = io.StringIO()
            with (
                mock.patch.object(rerun, "load_records", return_value=records),
                mock.patch.object(rerun, "rank_records", return_value=(records, None, [])),
                mock.patch.object(rerun, "preflight_env", return_value=(True, "2")),
                mock.patch.object(rerun, "_compat_needed", return_value=True),
                mock.patch.object(rerun, "rerun_one", side_effect=fake_run),
                contextlib.redirect_stdout(output),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                rc = rerun.main([str(run_json), "--pkl", str(origin)])
        self.assertEqual(rc, 0)
        (result,) = json.loads(output.getvalue())["results"]
        self.assertEqual(result["executed_code_sha256"], rerun.code_sha256(self.CODE))
        self.assertEqual(result["compat_rewrites"], ["tensorstore-anonymous-credentials"])
        self.assertEqual(result["input_audit"]["tensorstore"]["ok"], 3)
        self.assertEqual(result["result_status"], "USABLE")

    def test_rewritten_spec_parses_with_the_installed_tensorstore(self) -> None:
        if importlib.util.find_spec("tensorstore") is None:
            self.skipTest("tensorstore is not installed")
        rule = rerun.COMPAT_REWRITES[0]
        rerun._compat_needed.cache_clear()
        needed = rerun._compat_needed(rule)
        code = (
            "import tensorstore as ts\n"
            "spec = {'driver': 'zarr', 'kvstore': {'driver': 's3', "
            "'bucket': 'aind-open-data', 'path': 'p/', "
            "'aws_credentials': {'type': 'anonymous'}}}\n"
            "ts.Spec(spec)\n"
        )
        rewritten, names = rerun.apply_compat_rewrites(code)
        self.assertEqual(names, [rule.name] if needed else [])
        proc = subprocess.run([rerun.sys.executable, "-c", rewritten],
                              capture_output=True, text=True, timeout=120)
        self.assertEqual(proc.returncode, 0, proc.stderr)

    def test_work_items_from_the_previous_runner_version_are_not_reused(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = Path(tmpdir) / "data.pkl"
            dataset.write_bytes(b"one")
            with mock.patch.object(rerun, "_RUNNER_VERSION", "work-items-v1"):
                old_key = rerun._work_key(
                    run_sha256="run", record_id=1, code_hash=rerun.code_sha256("code"),
                    dataset=dataset, timeout=10,
                )
            stale = {"timed_out": False, "exitcode": 0, "stdout": "metric=0\n",
                     "stderr": "", "runtime_ms": 1}
            checkpoint = {"schema_version": 1, "work_items": {old_key: {"run": stale}}}
            fresh = dict(stale, stdout="metric=1\n")
            with mock.patch.object(rerun, "rerun_one", return_value=fresh) as run_one:
                run, reused = rerun._run_work_item(
                    "code", dataset, 10, run_sha256="run", record_id=1,
                    checkpoint=checkpoint, checkpoint_path=None,
                )
        self.assertEqual((run["stdout"], reused, run_one.call_count),
                         ("metric=1\n", False, 1))



class _FakeHandle:
    """A started script that ends ``seconds`` after it starts with ``run``."""

    def __init__(self, run: dict, seconds: float, live: list) -> None:
        self._run, self._end, self._live = run, rerun.time.monotonic() + seconds, live
        self.peak_private_bytes = 0
        live.append(self)

    def poll(self):
        if rerun.time.monotonic() < self._end:
            return None
        if self in self._live:
            self._live.remove(self)
        return dict(self._run)


def _ok_run(stdout: str = "metric=1\n") -> dict:
    return {"timed_out": False, "exitcode": 0, "stdout": stdout, "stderr": "",
            "runtime_ms": 1, "input_audit": {}}


class ParallelPoolTests(unittest.TestCase):
    TIMEOUT_S = 8  # roomy enough for a cold interpreter start on a busy node
    # One record per outcome the runner distinguishes; id 3 times out (TIMEOUT_S).
    CODES = {
        1: "print('metric=1')\n",
        2: "raise SystemExit(3)\n",
        3: "import time\ntime.sleep(120)\n",
        4: ("import io, numpy as np\n"
            "try:\n    np.load(io.BytesIO(b'not an array'))\nexcept Exception:\n    pass\n"
            "print('delta AP +0.00000')\n"),
        5: "print('No dataset found.')\n",
        6: "import os\nprint('threads', os.environ.get('OMP_NUM_THREADS'))\n",
    }

    def _main(self, root: Path, jobs: int, *, preload: bool) -> dict:
        run_json = root / "run.json"
        run_json.write_text("[]")
        origin = root / "dataset_cache_origin_add.pkl"
        extra = root / "dataset_cache_extra_add.pkl"
        for path in (origin, extra):
            path.write_bytes(rerun.pickle.dumps({}))
        records = [_record(rid, code) for rid, code in self.CODES.items()]
        output = io.StringIO()
        argv = [str(run_json), "--pkl", str(origin), "--extra-pkl", str(extra),
                "--timeout", str(self.TIMEOUT_S), "--jobs", str(jobs)]
        if not preload:
            argv.append("--no-preload")
        with (
            mock.patch.object(rerun, "load_records", return_value=records),
            mock.patch.object(rerun, "rank_records", return_value=(records, None, [])),
            mock.patch.object(rerun, "preflight_env", return_value=(True, "2")),
            mock.patch.object(rerun, "_PRELOAD_ENABLED", preload),
            mock.patch.dict(rerun._PAYLOAD_CACHE, {}, clear=True),
            contextlib.redirect_stdout(output),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            self.assertEqual(rerun.main(argv), 0)
        return json.loads(output.getvalue())

    @staticmethod
    def _comparable(payload: dict) -> list[dict]:
        results = json.loads(json.dumps(payload["results"]))
        for result in results:
            result.pop("rerun_runtime_ms")
            for extra in result["extrapolations"]:
                extra.pop("runtime_ms")
            if result["id"] == 6:  # the thread cap is the one intended difference
                result.pop("rerun_stdout")
                for extra in result["extrapolations"]:
                    extra.pop("stdout")
        return results

    def test_jobs_1_and_3_write_the_same_results_on_both_paths(self) -> None:
        paths = (False, True) if hasattr(rerun.os, "fork") else (False,)
        for preload in paths:
            with self.subTest(preload=preload), tempfile.TemporaryDirectory() as tmpdir:
                serial = self._main(Path(tmpdir), 1, preload=preload)
                parallel = self._main(Path(tmpdir), 3, preload=preload)
                self.assertEqual((serial["jobs"], parallel["jobs"]), (1, 3))
                self.assertEqual(self._comparable(serial), self._comparable(parallel))
                by_id = {r["id"]: r for r in parallel["results"]}
                self.assertEqual(
                    {rid: by_id[rid]["result_failure_reason"] for rid in by_id},
                    {1: None, 2: "nonzero-exit", 3: "timeout",
                     4: "swallowed-input-error", 5: "dataset-loading", 6: None},
                )
                # Extrapolation follows a USABLE origin only.
                self.assertEqual([len(by_id[rid]["extrapolations"]) for rid in by_id],
                                 [1, 0, 0, 0, 0, 1])
                threads = str(max(1, rerun._cpu_count() // 3))
                self.assertIn(f"threads {threads}", by_id[6]["rerun_stdout"])
                self.assertIn(f"threads {rerun.os.environ.get('OMP_NUM_THREADS')}",
                              {r["id"]: r for r in serial["results"]}[6]["rerun_stdout"])

    def test_pool_caps_concurrency_runs_longest_first_and_keeps_order(self) -> None:
        live, started, peak = [], [], [0]
        durations = {1: 0.1, 2: 0.6, 3: 0.3, 4: 0.5, 5: 0.2}

        def fake_start(code, dataset, timeout, companions=(), *, threads=None):
            started.append(code)
            handle = _FakeHandle(_ok_run(code), durations[int(code)], live)
            peak[0] = max(peak[0], len(live))
            return handle

        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = Path(tmpdir) / "data.pkl"
            dataset.write_bytes(b"x")
            checkpoint_path = Path(tmpdir) / "checkpoint.json"
            checkpoint = rerun._load_checkpoint(checkpoint_path)
            work = [{"code": str(rid), "dataset": dataset, "companions": (),
                     "record_id": rid, "label": f"id={rid}", "predicted_s": seconds}
                    for rid, seconds in durations.items()]
            # id 5 is already in the checkpoint and must never start.
            _, _, key = rerun._work_item("5", dataset, 10, run_sha256="run", record_id=5)
            rerun._store_run(checkpoint, None, key, record_id=5, code="5",
                             dataset=dataset, run=_ok_run("cached\n"))
            with mock.patch.object(rerun, "start_run", side_effect=fake_start), \
                    mock.patch.object(rerun, "_memory_headroom", return_value=None), \
                    contextlib.redirect_stderr(io.StringIO()):
                rerun._run_pool(work, jobs=2, timeout=10, run_sha256="run",
                                checkpoint=checkpoint, checkpoint_path=checkpoint_path)
            saved = json.loads(checkpoint_path.read_text())
        self.assertEqual(started, ["2", "4", "3", "1"])
        self.assertEqual(peak[0], 2)
        self.assertEqual([entry["run"]["stdout"] for entry in work],
                         ["1", "2", "3", "4", "cached\n"])
        self.assertEqual([entry["reused"] for entry in work],
                         [False, False, False, False, True])
        self.assertEqual(len(saved["work_items"]), 5)
        self.assertEqual(saved["work_items"][key]["run"]["stdout"], "cached\n")

    def test_memory_gate_holds_back_new_scripts(self) -> None:
        live, peak = [], [0]

        def fake_start(code, dataset, timeout, companions=(), *, threads=None):
            handle = _FakeHandle(_ok_run(), 0.1, live)
            peak[0] = max(peak[0], len(live))
            return handle

        work = [{"code": str(i), "dataset": Path(__file__), "companions": (),
                 "record_id": i, "label": f"id={i}"} for i in range(4)]
        with mock.patch.object(rerun, "start_run", side_effect=fake_start), \
                mock.patch.object(rerun, "_memory_headroom", return_value=1 << 20), \
                contextlib.redirect_stderr(io.StringIO()):
            rerun._run_pool(work, jobs=4, timeout=10, run_sha256="run",
                            checkpoint=rerun._load_checkpoint(None), checkpoint_path=None)
        self.assertEqual(peak[0], 1)  # 1 MiB of headroom admits one script at a time
        self.assertTrue(all(entry["run"]["exitcode"] == 0 for entry in work))

    def test_memory_headroom_reads_the_tightest_cgroup_then_meminfo(self) -> None:
        gib = 1 << 30
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "cgroup"
            leaf = root / "job" / "step" / "task"
            leaf.mkdir(parents=True)
            (leaf / "memory.max").write_text("max\n")

            def limit(node: Path, maximum: int, current: int, inactive: int) -> None:
                (node / "memory.max").write_text(f"{maximum}\n")
                (node / "memory.current").write_text(f"{current}\n")
                (node / "memory.stat").write_text(f"anon 1\ninactive_file {inactive}\n")

            limit(root / "job" / "step", 100 * gib, 60 * gib, 10 * gib)  # 50 GiB free
            limit(root / "job", 80 * gib, 75 * gib, 5 * gib)             # 10 GiB free
            cgroup = Path(tmpdir) / "self_cgroup"
            cgroup.write_text("0::/job/step/task\n")
            meminfo = Path(tmpdir) / "meminfo"
            meminfo.write_text("MemTotal: 100 kB\nMemAvailable: 64 kB\n")
            self.assertEqual(rerun._memory_headroom(str(cgroup), str(root), str(meminfo)),
                             10 * gib)
            cgroup.write_text("0::/elsewhere\n")
            self.assertEqual(rerun._memory_headroom(str(cgroup), str(root), str(meminfo)),
                             64 * 1024)
            self.assertIsNone(rerun._memory_headroom(str(cgroup), str(root),
                                                     str(Path(tmpdir) / "none")))

    def test_a_killed_script_is_run_again_alone(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pkl = root / "dataset_cache_kill_add.pkl"
            pkl.write_bytes(rerun.pickle.dumps({}))
            marker = root / "killed-once"
            once = ("import os, signal\nfrom pathlib import Path\n"
                    f"m = Path({str(marker)!r})\n"
                    "if not m.exists():\n    m.touch()\n"
                    "    os.kill(os.getpid(), signal.SIGKILL)\n"
                    "print('metric=1')\n")
            always = "import os, signal\nos.kill(os.getpid(), signal.SIGKILL)\n"
            work = [{"code": code, "dataset": pkl, "companions": (), "record_id": rid,
                     "label": f"id={rid}"}
                    for rid, code in ((1, once), (2, always), (3, "print('metric=3')\n"))]
            log = io.StringIO()
            with mock.patch.object(rerun, "_PRELOAD_ENABLED", False), \
                    contextlib.redirect_stderr(log):
                rerun._run_pool(work, jobs=3, timeout=60, run_sha256="run",
                                checkpoint=rerun._load_checkpoint(None),
                                checkpoint_path=None)
        self.assertEqual(work[0]["run"]["exitcode"], 0)
        self.assertIn("metric=1", work[0]["run"]["stdout"])
        self.assertEqual(work[1]["run"]["exitcode"], -rerun.signal.SIGKILL)
        self.assertEqual(work[2]["run"]["exitcode"], 0)
        self.assertEqual(log.getvalue().count("requeued alone"), 2)

    def test_pool_installs_a_missing_package_and_requeues(self) -> None:
        live = []
        runs = [{"timed_out": False, "exitcode": 1, "stdout": "", "runtime_ms": 1,
                 "stderr": "ModuleNotFoundError: No module named 'lightgbm'"}, _ok_run()]

        def fake_start(code, dataset, timeout, companions=(), *, threads=None):
            return _FakeHandle(runs.pop(0), 0.0, live)

        work = [{"code": "x", "dataset": Path(__file__), "companions": (),
                 "record_id": 1, "label": "id=1"}]
        with mock.patch.object(rerun, "start_run", side_effect=fake_start), \
                mock.patch.object(rerun, "_INSTALL_MISSING", True), \
                mock.patch.object(rerun, "missing_module", return_value="lightgbm"), \
                mock.patch.object(rerun, "install_missing",
                                  return_value={"version": "4.7.0"}) as install, \
                mock.patch.object(rerun, "_memory_headroom", return_value=None), \
                contextlib.redirect_stderr(io.StringIO()):
            rerun._run_pool(work, jobs=2, timeout=10, run_sha256="run",
                            checkpoint=rerun._load_checkpoint(None), checkpoint_path=None)
        install.assert_called_once_with("lightgbm")
        self.assertEqual(work[0]["run"]["exitcode"], 0)

    def test_thread_caps_reach_both_paths_and_only_when_asked(self) -> None:
        import importlib
        importlib.import_module("numpy")  # loads a BLAS the forked child inherits

        code = ("import os\nfrom threadpoolctl import threadpool_info\n"
                "print(os.environ.get('OMP_NUM_THREADS'),"
                " sorted({p['num_threads'] for p in threadpool_info()}))\n")
        with tempfile.TemporaryDirectory() as tmpdir:
            pkl = Path(tmpdir) / "dataset_cache_threads_add.pkl"
            pkl.write_bytes(rerun.pickle.dumps({}))
            paths = (False, True) if hasattr(rerun.os, "fork") else (False,)
            for preload in paths:
                with self.subTest(preload=preload), \
                        mock.patch.object(rerun, "_PRELOAD_ENABLED", preload), \
                        mock.patch.dict(rerun._PAYLOAD_CACHE, {}, clear=True), \
                        contextlib.redirect_stderr(io.StringIO()):
                    capped = rerun.start_run(code, pkl, 60, threads=2).wait()
                    plain = rerun.rerun_one(code, pkl, 60)
                env, pools = capped["stdout"].split(" ", 1)
                self.assertEqual(env, "2")
                self.assertTrue(set(json.loads(pools)) <= {2}, capped["stdout"])
                self.assertTrue(plain["stdout"].startswith(
                    str(rerun.os.environ.get("OMP_NUM_THREADS"))))


if __name__ == "__main__":
    unittest.main()
