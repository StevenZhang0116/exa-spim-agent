"""Prepare immutable detector-native tables from trusted labeled _add.pkl caches.

Exactly one frozen merge-junction detector and one split-site detector are used.
Their own universe and feature code define the pool; no fitting or graph edits.
Run in panda on a compute node. Tables bind source/model/code provenance and
separate predictor features from evaluator labels. Existing matching tables are
reused; corrupt or mismatched tables are never silently replaced.
"""

import argparse
import hashlib
import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.insert(0, str(PROJECT_ROOT))

from proofreader_evolve.harness import dataset as ds

HERE = Path(__file__).resolve().parents[1]
DEFAULT_OUT = HERE / "feature_tables"
APP_DIR = PROJECT_ROOT / "autodiscovery-application"
DEFAULT_MERGE_DIR = APP_DIR / "merge-error-794495-mcl100_2026-08-04"
DEFAULT_SPLIT_DIR = APP_DIR / "split-error-794495-mcl100-run-3_2026-08-24"


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _import_module(path: Path, name: str):
    """Import a frozen detector script by file path (its dir name has hyphens)."""
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # so any internal pickling/introspection resolves
    spec.loader.exec_module(module)
    return module


def _score(estimator, X: np.ndarray) -> np.ndarray:
    """Ranking score from a fitted pipeline (same convention as the detectors)."""
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(X)[:, 1]
    if hasattr(estimator, "decision_function"):
        return estimator.decision_function(X)
    raise TypeError(f"estimator {type(estimator).__name__} has no scoring method")


def _load_joblib(path: Path) -> dict:
    import joblib
    bundle = joblib.load(path)
    for key in ("pipeline", "feature_order"):
        if key not in bundle:
            raise KeyError(f"{path} lacks '{key}' — not a detector joblib bundle")
    return bundle


def _resolve_detector_dir(dir_arg: str, kind: str) -> tuple[Path, Path]:
    """Resolve one detector deliverable folder to ``(script_path, joblib_path)``.

    Accept merge-junction and split-site detectors. Exactly one
    supported script and one matching fitted model must exist in the folder.
    """
    if kind not in ("merge", "split"):
        raise ValueError("Detector kind must be merge or split")
    d = Path(dir_arg).expanduser()
    if not d.is_dir() and len(d.parts) == 1:
        d = APP_DIR / d
    d = d.resolve()
    if not d.is_dir():
        raise SystemExit(f"--{kind}-dir is not a directory: {d}")
    names = (["merge_junction_detector.py"]
             if kind == "merge" else ["split_site_detector.py"])
    scripts = [d / name for name in names if (d / name).is_file()]
    if len(scripts) != 1:
        raise SystemExit(
            f"--{kind}-dir must contain exactly one of {names}; found {len(scripts)}: {d}")
    script = scripts[0]
    prefix = "merge_junction_detector" if script.name == "merge_junction_detector.py" else f"{kind}_detector"
    joblibs = sorted(path for path in d.glob(f"{prefix}_*.joblib") if path.is_file())
    if not joblibs:
        raise SystemExit(
            f"--{kind}-dir has no {prefix}_*.joblib — run the detector "
            f"first (see {d / 'RUN_COMMANDS.md'})")
    if len(joblibs) > 1:
        raise SystemExit(
            f"--{kind}-dir holds {len(joblibs)} {prefix}_*.joblib files "
            f"({', '.join(p.name for p in joblibs)}) — remove all but the "
            f"intended one: {d}")
    return script, joblibs[0]


def _model_analysis_keys(module, feature_order):
    """Select native computation groups from the fitted model's input columns.

    A group may own several inseparable features. Definedness columns use the
    same owner as their numeric feature. Never fall back to enabling everything
    when the model schema cannot be mapped to the detector's groups.
    """
    required = list(feature_order)
    if not required or any(not isinstance(name, str) or not name for name in required):
        raise ValueError("Model feature_order must contain non-empty feature names")
    if len(set(required)) != len(required):
        raise ValueError("Model feature_order contains duplicate features")
    groups = getattr(module, "ANALYSIS_TIMING_GROUPS", None)
    if not isinstance(groups, list) or not groups:
        raise ValueError("Detector lacks ANALYSIS_TIMING_GROUPS for model feature selection")
    owners, keys = {}, set()
    for group in groups:
        key = group["key"]
        if key in keys:
            raise ValueError(f"Duplicate analysis group key: {key}")
        keys.add(key)
        for name in group["feature_names"]:
            for column in (name, name + "_is_defined"):
                if column in owners:
                    raise ValueError(f"Ambiguous analysis group ownership for feature: {column}")
                owners[column] = key
    missing = sorted(set(required) - owners.keys())
    if missing:
        raise ValueError(f"Model features have no analysis group: {missing}")
    return {owners[name] for name in required}


def _extract_candidate_features(payload, module, samples, feature_order):
    """Extract only model-required groups on the unchanged GT-free universe."""
    enabled = _model_analysis_keys(module, feature_order)
    skipped = sorted(group["key"] for group in module.ANALYSIS_TIMING_GROUPS
                     if group["key"] not in enabled)
    log(f"[extract] model feature selection: {len(enabled)}/{len(module.ANALYSIS_TIMING_GROUPS)} "
        f"groups enabled; skipped={skipped}")
    working = {key: value for key, value in payload.items()
               if not key.startswith("gt_") and not key.startswith("__detector_")}
    working["__detector_sample_universe_cache__"] = (samples, np.zeros(len(samples), dtype=np.int64))
    extractor = getattr(module, "_extract_features_runtime", None)
    if extractor is not None:
        rows, labels, accumulator = extractor(working, verbose=True, enabled_analysis_keys=enabled)
    else:
        view = module._BlindPayloadView(working, samples) if hasattr(module, "_BlindPayloadView") else working
        rows, labels, accumulator = module.extract_features(view, verbose=True,
                                                            enabled_analysis_keys=enabled)
    identity = module.sample_display
    if [identity(row) for row in rows] != [identity(row) for row in samples]:
        raise RuntimeError("Detector extraction changed candidate identity/order")
    if len(labels) != len(samples) or len(accumulator.to_frame()) != len(samples):
        raise RuntimeError("Detector extraction returned misaligned labels/features")
    return rows, accumulator


def resolve_detector_runs(merge_dirs=None, split_dirs=None):
    """Resolve exactly one trusted frozen detector per kind."""
    selected = {}
    for kind, directories, default in (("merge", merge_dirs, DEFAULT_MERGE_DIR),
                                       ("split", split_dirs, DEFAULT_SPLIT_DIR)):
        directories = [default] if directories is None else list(directories)
        if len(directories) != 1:
            raise ValueError(f"Exactly one {kind} detector run is required")
        script, model = _resolve_detector_dir(directories[0], kind)
        selected[kind] = [{"run_id": script.parent.name, "dir": str(script.parent),
                           "script": script.name, "script_sha256": _sha256(script),
                           "file": model.name, "sha256": _sha256(model)}]
    return selected


def _cache_identity(path):
    path = Path(path)
    stat = path.stat()
    return {"path": str(path.resolve()), "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns, "inode": stat.st_ino}


def _assert_detector_unchanged(record):
    directory = Path(record["dir"])
    if (_sha256(directory / record["script"]) != record["script_sha256"]
            or _sha256(directory / record["file"]) != record["sha256"]):
        raise RuntimeError(f"Detector artifacts changed during preparation: {directory}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--brain", required=True)
    parser.add_argument("--pkl", required=True, help="Matching trusted labeled _add.pkl cache")
    parser.add_argument("--mcl", type=int, default=100)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--merge-dir", action="append", help="One merge-junction detector directory")
    parser.add_argument("--split-dir", action="append", help="One split-site detector directory")
    args = parser.parse_args(argv)
    from proofreader_evolve.harness.native_pool import ensure_native_tables
    selected = resolve_detector_runs(args.merge_dir, args.split_dir)
    bank = ensure_native_tables(args.brain, args.pkl, selected, args.out_dir,
                                prepare=True, mcl=args.mcl)
    log(json.dumps(bank.meta, indent=2))
    log("# OK (fixed native pool)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
