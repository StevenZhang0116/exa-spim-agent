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


if __name__ == "__main__":
    unittest.main()
