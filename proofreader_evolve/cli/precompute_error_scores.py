"""Prepare immutable detector-native tables from trusted labeled _add.pkl caches.

Exactly one frozen merge-junction detector and one split-site detector are used.
The pool is the rule declared in candidate_pools.json, enumerated by candidate_rules.py;
the detectors compute features and scores for its rows; no fitting or graph edits.
Run in panda on a compute node. Tables bind source/model/code provenance and
separate predictor features from evaluator labels. Existing matching tables are
reused; corrupt or mismatched tables are never silently replaced.
"""

import argparse
import hashlib
import importlib.util
import inspect
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
# Default frozen detectors (a directory under APP_DIR). None since 2026-10-17: the 2026-08
# detectors were deleted; set these to the new builds (rebuild plan, section 3 step 4).
DEFAULT_MERGE_DIR = None
DEFAULT_SPLIT_DIR = None


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


def _run_detector_sha256(run_dir: Path, joblib_path: Path) -> str | None:
    """``data_provenance.detector_sha256`` of the run that wrote ``joblib_path``, or None.

    The detector runtime records the SHA-256 of its own script in every
    ``model_selection_<brain>.json`` (``data_provenance.detector_sha256``), which is what
    binds a run directory to the script that produced it.
    """
    brain = joblib_path.stem.rsplit("_", 1)[-1]
    path = run_dir / f"model_selection_{brain}.json"
    if not path.is_file():
        return None
    record = json.loads(path.read_text())
    digest = (record.get("data_provenance") or {}).get("detector_sha256")
    return digest if isinstance(digest, str) and len(digest) == 64 else None


def _resolve_detector_dir(dir_arg: str, kind: str) -> tuple[Path, Path]:
    """Resolve a detector deliverable folder or one of its run folders to ``(script, joblib)``.

    Accept merge-junction and split-site detectors. The folder holds exactly one
    supported script and exactly one matching fitted model; or it is a run directory
    (``RUN_COMMANDS.md`` writes every run into ``<deliverable>/runs/<brain>-<purpose>/``)
    that holds the model while the deliverable directory above it holds the script. A run
    directory is accepted
    only when its ``model_selection_<brain>.json`` names the script's SHA-256, so a
    model can never be paired with a detector that did not produce it.
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
    # A run directory sits at <deliverable>/runs/<brain>-<purpose>/ (RUN_COMMANDS.md);
    # the script is two levels up. One level up is accepted too for a flat layout.
    for ancestor in (d.parent, d.parent.parent):
        if scripts or ancestor == d:
            break
        scripts = [ancestor / name for name in names if (ancestor / name).is_file()]
    if len(scripts) != 1:
        raise SystemExit(
            f"--{kind}-dir must contain exactly one of {names}, or be a run directory "
            f"under a deliverable directory that does; found {len(scripts)}: {d}")
    script = scripts[0]
    prefix = "merge_junction_detector" if script.name == "merge_junction_detector.py" else f"{kind}_detector"
    joblibs = sorted(path for path in d.glob(f"{prefix}_*.joblib") if path.is_file())
    if not joblibs:
        raise SystemExit(
            f"--{kind}-dir has no {prefix}_*.joblib — run the detector "
            f"first (see {script.parent / 'RUN_COMMANDS.md'})")
    if len(joblibs) > 1:
        raise SystemExit(
            f"--{kind}-dir holds {len(joblibs)} {prefix}_*.joblib files "
            f"({', '.join(p.name for p in joblibs)}) — remove all but the "
            f"intended one: {d}")
    model = joblibs[0]
    recorded = _run_detector_sha256(d, model)
    if script.parent != d and recorded is None:
        raise SystemExit(
            f"{d} is a run directory but its model_selection JSON does not record "
            "data_provenance.detector_sha256; run the detector again (any detector "
            "assembled since 2026-10 records it) or pass the deliverable directory")
    if recorded is not None and recorded != _sha256(script):
        raise SystemExit(
            f"{model.name} in {d} was produced by a detector with SHA-256 {recorded}, "
            f"not by {script} ({_sha256(script)}); keep each run with its own detector")
    return script, model


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


def _extract_candidate_features(payload, module, samples, feature_order, image_workers=0):
    """Extract only model-required groups on the unchanged GT-free universe.

    ``image_workers`` reaches a detector whose extraction entry accepts it (the
    runtime's image pass: 0 = auto capped at 8, -1 = all CPUs, N = N threads);
    older detectors without the parameter run as before.
    """
    enabled = _model_analysis_keys(module, feature_order)
    skipped = sorted(group["key"] for group in module.ANALYSIS_TIMING_GROUPS
                     if group["key"] not in enabled)
    log(f"[extract] model feature selection: {len(enabled)}/{len(module.ANALYSIS_TIMING_GROUPS)} "
        f"groups enabled; skipped={skipped}")
    working = {key: value for key, value in payload.items()
               if not key.startswith("gt_") and not key.startswith("__detector_")}
    working["__detector_sample_universe_cache__"] = (samples, np.zeros(len(samples), dtype=np.int64))
    extractor = getattr(module, "_extract_features_runtime", None)
    direct = extractor is None
    if direct:
        extractor = module.extract_features
    kwargs = {"verbose": True, "enabled_analysis_keys": enabled}
    try:
        accepts_workers = "image_workers" in inspect.signature(extractor).parameters
    except (TypeError, ValueError):
        accepts_workers = False
    if accepts_workers:
        kwargs["image_workers"] = int(image_workers)
        log(f"[extract] image_workers={int(image_workers)} (0 = auto capped at 8, -1 = all CPUs)")
    if direct:
        view = module._BlindPayloadView(working, samples) if hasattr(module, "_BlindPayloadView") else working
        rows, labels, accumulator = extractor(view, **kwargs)
    else:
        rows, labels, accumulator = extractor(working, **kwargs)
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
        if directories is None and default is None:
            raise SystemExit(f"No default frozen {kind} detector: pass --{kind}-dir, or set "
                             f"DEFAULT_{kind.upper()}_DIR in proofreader_evolve/cli/precompute_error_scores.py "
                             "once the new detectors are built")
        directories = [default] if directories is None else list(directories)
        if len(directories) != 1:
            raise ValueError(f"Exactly one {kind} detector run is required")
        script, model = _resolve_detector_dir(directories[0], kind)
        run_dir = model.parent
        run_id = (script.parent.name if run_dir == script.parent
                  else f"{script.parent.name}/{run_dir.relative_to(script.parent).as_posix()}")
        # "dir" is where the run outputs live (joblib, model_selection JSON); the
        # script may sit one level up. Readers use _script_path / _model_path.
        selected[kind] = [{"run_id": run_id, "dir": str(run_dir), "script_dir": str(script.parent),
                           "script": script.name, "script_sha256": _sha256(script),
                           "file": model.name, "sha256": _sha256(model)}]
    return selected


def _cache_identity(path):
    path = Path(path)
    stat = path.stat()
    return {"path": str(path.resolve()), "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns, "inode": stat.st_ino}


def _script_path(record) -> Path:
    """The detector script of a resolved record (beside the run outputs, or one level up)."""
    return Path(record.get("script_dir") or record["dir"]) / record["script"]


def _model_path(record) -> Path:
    return Path(record["dir"]) / record["file"]


def _assert_detector_unchanged(record):
    if (_sha256(_script_path(record)) != record["script_sha256"]
            or _sha256(_model_path(record)) != record["sha256"]):
        raise RuntimeError(f"Detector artifacts changed during preparation: {record['dir']}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--brain", required=True)
    parser.add_argument("--pkl", required=True, help="Matching trusted labeled _add.pkl cache")
    parser.add_argument("--mcl", type=int, default=100)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--merge-dir", action="append", help="One merge-junction detector directory")
    parser.add_argument("--split-dir", action="append", help="One split-site detector directory")
    parser.add_argument("--image-workers", type=int, default=0,
                        help="threads for the detectors' image pass (S3-bound, may exceed the CPU "
                             "count): 0 = auto capped at 8, -1 = all CPUs, N = N threads")
    args = parser.parse_args(argv)
    if args.image_workers < -1:
        parser.error("--image-workers must be -1, zero, or a positive integer")
    from proofreader_evolve.harness.native_pool import ensure_native_tables
    selected = resolve_detector_runs(args.merge_dir, args.split_dir)
    bank = ensure_native_tables(args.brain, args.pkl, selected, args.out_dir,
                                prepare=True, mcl=args.mcl, image_workers=args.image_workers)
    log(json.dumps(bank.meta, indent=2))
    log("# OK (fixed native pool)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
