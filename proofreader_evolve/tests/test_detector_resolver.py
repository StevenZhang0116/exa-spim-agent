"""Stage 3 detector resolution: a deliverable directory or one of its run directories.

RUN_COMMANDS.md writes every detector run into ``<deliverable>/runs/<brain>-<purpose>/``,
so the fitted model and its model_selection JSON live one level below the script. The
resolver accepts such a run directory when the run's ``data_provenance.detector_sha256``
names the script; no real detector or data is involved.
"""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from proofreader_evolve.cli import precompute_error_scores as pc


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _deliverable(root: Path, kind: str) -> tuple[Path, str, str]:
    script = "merge_junction_detector.py" if kind == "merge" else "split_site_detector.py"
    prefix = "merge_junction_detector" if kind == "merge" else "split_detector"
    directory = root / f"{kind}-app"
    directory.mkdir()
    (directory / script).write_text(f"DETECTOR_TARGET = '{kind}'\n")
    return directory, script, prefix


def _run(directory: Path, prefix: str, brain: str, detector_sha256) -> Path:
    import joblib
    run_dir = directory / "runs" / f"{brain}-default"
    run_dir.mkdir(parents=True)
    joblib.dump({"pipeline": None, "feature_order": ["f0"]}, run_dir / f"{prefix}_{brain}.joblib")
    record = {"train_brain": brain, "train_mcl": 100, "feature_order": ["f0"]}
    if detector_sha256 is not None:
        record["data_provenance"] = {"detector_sha256": detector_sha256}
    (run_dir / f"model_selection_{brain}.json").write_text(json.dumps(record))
    return run_dir


class DetectorResolverTests(unittest.TestCase):
    def test_run_directory_resolves_to_parent_script_when_bound_by_hash(self):
        for kind in ("merge", "split"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as tmp:
                directory, script, prefix = _deliverable(Path(tmp), kind)
                run_dir = _run(directory, prefix, "794495", _sha(directory / script))
                found_script, model = pc._resolve_detector_dir(str(run_dir), kind)
                self.assertEqual(found_script, (directory / script).resolve())
                self.assertEqual(model, (run_dir / f"{prefix}_794495.joblib").resolve())
                record = {"run_id": f"{directory.name}/runs/794495-default", "dir": str(run_dir.resolve()),
                          "script_dir": str(directory.resolve()), "script": script,
                          "script_sha256": _sha(directory / script),
                          "file": f"{prefix}_794495.joblib",
                          "sha256": _sha(run_dir / f"{prefix}_794495.joblib")}
                self.assertEqual(pc._script_path(record), (directory / script).resolve())
                self.assertEqual(pc._model_path(record), (run_dir / f"{prefix}_794495.joblib").resolve())
                pc._assert_detector_unchanged(record)  # must not raise

    def test_resolve_detector_runs_records_run_and_script_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            merge_dir, merge_script, merge_prefix = _deliverable(Path(tmp), "merge")
            split_dir, split_script, split_prefix = _deliverable(Path(tmp), "split")
            merge_run = _run(merge_dir, merge_prefix, "794495", _sha(merge_dir / merge_script))
            split_run = _run(split_dir, split_prefix, "794495", _sha(split_dir / split_script))
            selected = pc.resolve_detector_runs([str(merge_run)], [str(split_run)])
            for kind, directory, run_dir, script in (("merge", merge_dir, merge_run, merge_script),
                                                     ("split", split_dir, split_run, split_script)):
                record = selected[kind][0]
                self.assertEqual(record["dir"], str(run_dir.resolve()))
                self.assertEqual(record["script_dir"], str(directory.resolve()))
                self.assertEqual(record["run_id"], f"{directory.name}/runs/794495-default")
                self.assertEqual(record["script"], script)
                pc._assert_detector_unchanged(record)
            # a record without script_dir (the one-directory layout) still resolves
            legacy = {**selected["merge"][0]}
            legacy.pop("script_dir")
            legacy["dir"] = str(merge_dir.resolve())
            self.assertEqual(pc._script_path(legacy), (merge_dir / merge_script).resolve())

    def test_run_directory_without_provenance_or_with_another_detector_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory, script, prefix = _deliverable(Path(tmp), "split")
            unbound = _run(directory, prefix, "1", None)
            with self.assertRaisesRegex(SystemExit, "does not record"):
                pc._resolve_detector_dir(str(unbound), "split")
            other = _run(directory / "other", prefix, "2", "f" * 64)
            (directory / "other" / script).write_text("DETECTOR_TARGET = 'split'\n")
            with self.assertRaisesRegex(SystemExit, "was produced by a detector"):
                pc._resolve_detector_dir(str(other), "split")

    def test_one_directory_layout_still_accepted_and_hash_checked_when_recorded(self):
        with tempfile.TemporaryDirectory() as tmp:
            import joblib
            directory, script, prefix = _deliverable(Path(tmp), "merge")
            joblib.dump({"pipeline": None, "feature_order": ["f0"]}, directory / f"{prefix}_1.joblib")
            (directory / "model_selection_1.json").write_text(json.dumps(
                {"train_brain": "1", "train_mcl": 100, "feature_order": ["f0"]}))
            found, model = pc._resolve_detector_dir(str(directory), "merge")
            self.assertEqual(found, (directory / script).resolve())
            self.assertEqual(model, (directory / f"{prefix}_1.joblib").resolve())
            (directory / "model_selection_1.json").write_text(json.dumps(
                {"train_brain": "1", "train_mcl": 100, "feature_order": ["f0"],
                 "data_provenance": {"detector_sha256": "0" * 64}}))
            with self.assertRaisesRegex(SystemExit, "was produced by a detector"):
                pc._resolve_detector_dir(str(directory), "merge")


if __name__ == "__main__":
    unittest.main()
