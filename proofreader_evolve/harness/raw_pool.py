"""Detector-free candidate pools for ``--detector-prior raw`` (2026-11-06).

A raw pool holds the same candidate universe as a detector-native table: rows and binary
evaluator labels from ``candidate_rules.py`` under the rule declared in ``candidate_pools.json``
(``native_pool.build_native_universe``, the one enumeration path), with the label-blind raw
candidate attributes (``detector_prior.raw_columns``) as its only feature frame. No frozen
detector is resolved, imported, hashed or run, so a build is one cache load plus the
enumeration (minutes per brain), not the detectors' image pass (hours). The layout mirrors the
native tables, so availability sidecars, selection, scoring and descriptor contexts (keyed by
``pool_sha256``, which hashes the rows only and is therefore shared with a detector-native table
built from the same cache) are unchanged:

    feature_tables/<brain>/<digest>/{features.pkl, candidates.pkl, evaluator_labels.npy, meta.json}

The digest covers the cache identity, the declared rule, the raw column definition
(``detector_prior.RAW_COLUMNS_VERSION``) and the hashes of the enumeration implementation; it
never names a detector. Detector-native tables keep their own digests
(``native_pool.ensure_native_tables``) and are needed only by the ``frozen`` and ``no-score``
priors. Bump ``RAW_POOL_VERSION`` when the on-disk layout changes.
"""

import gc
import hashlib
import json
from pathlib import Path
import pickle
import tempfile

import numpy as np
import pandas as pd

from . import candidate_pools, detector_prior, native_pool
from .detector_metadata import inference_metadata
from .native_pool import NativeTable, NativeTables, build_native_universe, candidate_keys, pool_digest

RAW_POOL_VERSION = "raw-pool-v1"
FILES = ("features.pkl", "candidates.pkl", "evaluator_labels.npy")
REQUIRED_PAYLOAD = {"gt_graph", "gt_node_canonical_label", "gt_edge_error", "gt_merge_sites"}


def _implementation_hashes():
    from proofreader_evolve.cli import precompute_error_scores as pc
    return {path.name: pc._sha256(path) for path in
            (Path(native_pool.__file__), Path(native_pool.__file__).with_name("detector_metadata.py"),
             candidate_pools.RULES_PATH)}


def provenance(brain, kind, mcl, identity, pool_policy=None):
    """Provenance of one raw pool; its digest names the table directory. No detector field is set."""
    policy = candidate_pools.load_pool_policies()[kind] if pool_policy is None else pool_policy
    return {"version": RAW_POOL_VERSION, "detector_prior": "raw", "brain": str(brain), "mcl": int(mcl),
            "kind": kind, "pool_policy": policy, "source_cache": identity,
            "raw_columns": {"version": detector_prior.RAW_COLUMNS_VERSION,
                            "columns": list(detector_prior.RAW_COLUMNS[kind])},
            "detector": None, "training": None, "implementation": _implementation_hashes()}


def digest(record):
    return hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()


def build_raw_table(payload, kind, pool_policy=None):
    """The shared universe enumeration plus the raw columns; no detector, no image pass."""
    rows, truth, meta = build_native_universe(payload, kind, None, pool_policy)
    features = detector_prior.raw_columns(kind, rows)
    return NativeTable(kind, features, rows, truth, {**meta, "training_brain": None, "detector_prior": "raw"})


def _write(table, record, directory):
    from proofreader_evolve.cli import precompute_error_scores as pc
    directory.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".raw_", dir=directory.parent) as tmp:
        staging = Path(tmp)
        table.features.to_pickle(staging / "features.pkl")
        with (staging / "candidates.pkl").open("wb") as stream:
            pickle.dump(table.candidates, stream, protocol=pickle.HIGHEST_PROTOCOL)
        np.save(staging / "evaluator_labels.npy", table.truth, allow_pickle=False)
        metadata = {**table.meta, "provenance": record,
                    "files": {name: pc._sha256(staging / name) for name in FILES}}
        (staging / "meta.json").write_text(json.dumps(metadata, indent=2))
        staging.rename(directory)


def _load(kind, record, directory):
    from proofreader_evolve.cli import precompute_error_scores as pc
    meta = json.loads((directory / "meta.json").read_text())
    if meta.get("provenance") != record or set(meta.get("files", {})) != set(FILES):
        raise ValueError(f"Invalid raw pool metadata: {directory}")
    if any(pc._sha256(directory / name) != value for name, value in meta["files"].items()):
        raise ValueError(f"Corrupt raw pool: {directory}")
    with (directory / "candidates.pkl").open("rb") as stream:
        rows = pickle.load(stream)
    features = pd.read_pickle(directory / "features.pkl")
    truth = np.load(directory / "evaluator_labels.npy", allow_pickle=False)
    if (pool_digest(kind, rows) != meta["pool_sha256"] or len(rows) != len(features)
            or list(features.columns) != list(detector_prior.RAW_COLUMNS[kind])
            or truth.shape != (len(rows),) or not np.isin(truth, [0, 1]).all()):
        raise ValueError("Raw pool candidate/feature/label alignment failed")
    candidate_keys(kind, rows)
    return NativeTable(kind, features, rows, truth, meta)


def ensure_raw_tables(brain, source_path, tables_dir, prepare=False, mcl=100):
    """Build once (``prepare=True``) or load the exact raw pools of both kinds for one brain.

    Same contract as ``native_pool.ensure_native_tables`` without the detector argument: a
    missing pool is an error unless ``prepare`` is set, and a mismatched or corrupt pool is
    never silently replaced.
    """
    from proofreader_evolve.cli import precompute_error_scores as pc
    brain = str(brain)
    if not brain.isascii() or not brain.isdigit():
        raise ValueError("Brain ID must be numeric")
    inference_metadata({}, mcl)
    identity = pc._cache_identity(source_path)
    plans = []
    for kind in ("merge", "split"):
        record = provenance(brain, kind, mcl, identity)
        plans.append((kind, record, Path(tables_dir) / brain / digest(record)))
    missing = [plan for plan in plans if not plan[2].exists()]
    if missing and not prepare:
        raise FileNotFoundError(f"Missing matching raw pools for brain {brain}: "
                                + ", ".join(plan[0] for plan in missing)
                                + ". Run proofreader_evolve/prepare_feature_tables.py --detector-prior raw "
                                "with the same MCL.")
    if missing:
        _, _, payload = pc.ds.load_cached_graphs(str(source_path), expect_brain=brain, expect_mcl=mcl)
        payload = {**payload, **inference_metadata(payload, mcl)}
        if not REQUIRED_PAYLOAD.issubset(payload):
            raise ValueError("Raw pools need the matching labeled _add.pkl cache")
        try:
            for kind, record, directory in missing:
                pc.log(f"[{brain}/{kind}] preparing raw candidate pool (declared rule, no detector)")
                table = build_raw_table(payload, kind, pool_policy=record["pool_policy"])
                if pc._cache_identity(source_path) != identity:
                    raise RuntimeError("Source cache changed while preparing raw pools")
                _write(table, record, directory)
                pc.log(f"[{brain}/{kind}] saved {len(table.truth)} candidates to {directory}")
                del table
                gc.collect()
        finally:
            del payload
            gc.collect()
    tables, paths = {}, {}
    for kind, record, directory in plans:
        tables[kind] = _load(kind, record, directory)
        paths[kind] = str(directory.resolve())
    if pc._cache_identity(source_path) != identity:
        raise RuntimeError("Source cache changed while loading raw pools")
    return NativeTables(brain, tables, {"table_paths": paths, "version": RAW_POOL_VERSION,
                                        "detector_prior": "raw"})
