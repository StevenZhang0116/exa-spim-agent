from __future__ import annotations

import contextlib
import importlib.util
import io
import json
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


class RerunWorkItemTests(unittest.TestCase):
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

            def fake_run(code, path, timeout):
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

            def fake_run(code, path, timeout):
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

            def fake_run(code, path, timeout):
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


if __name__ == "__main__":
    unittest.main()
