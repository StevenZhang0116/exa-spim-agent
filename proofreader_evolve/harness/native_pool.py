"""Immutable detector-native candidate pools for scorer evolution.

The frozen detector is the single source of candidate enumeration AND evaluator
labels. Its universe builder runs only in the trusted preparation process.
Only whitelisted geometric rows enter extraction; GT labels/audit fields are
stored separately and never passed to a scorer. No repair-site conversion,
occurrence truncation, geometry cap, post-edit enumeration, or fitting occurs.
"""

from dataclasses import dataclass
import gc
import hashlib
import json
from pathlib import Path
import pickle
import tempfile

import numpy as np
import pandas as pd

from .detector_metadata import inference_metadata


POOL_VERSION = "detector-native-v1"
SAFE_KEYS = {
    "merge": ("candidate_id", "node_id", "segment_id", "degree", "x_um", "y_um", "z_um"),
    "split": ("candidate_id", "segment_id_a", "segment_id_b", "node_id_a", "node_id_b",
              "gap_um", "anchor_side", "partner_rank", "node_role_a", "node_role_b"),
}
OCCURRENCE_KEYS = ("node_id_a", "node_id_b", "gap_um", "anchor_side", "partner_rank",
                   "node_role_a", "node_role_b")


def cache_path(brain, mcl=100):
    from .dataset import default_cache_path
    return Path(default_cache_path(brain, mcl))


def training_provenance(record):
    """Read the model-selection record; do not infer training brains from run IDs."""
    from proofreader_evolve.cli import precompute_error_scores as pc
    paths = list(Path(record["dir"]).glob("model_selection_*.json"))
    matches = []
    for path in paths:
        metadata = json.loads(path.read_text())
        brain = str(metadata.get("train_brain", ""))
        if brain.isascii() and brain.isdigit() and record["file"].endswith(f"_{brain}.joblib"):
            matches.append((path, metadata))
    if len(matches) != 1:
        raise ValueError("One matching model_selection_<brain>.json is required for training provenance")
    path, metadata = matches[0]
    return {"path": str(path.resolve()), "sha256": pc._sha256(path),
            "brain": str(metadata["train_brain"]), "mcl": metadata["train_mcl"],
            "scope": metadata.get("scope"), "feature_order": metadata["feature_order"]}


def blind_rows(kind, rows):
    result = []
    for row in rows:
        safe = {key: row[key] for key in SAFE_KEYS[kind]}
        if kind == "split":
            safe["occurrences"] = tuple({key: occ[key] for key in OCCURRENCE_KEYS}
                                        for occ in row["occurrences"])
        result.append(safe)
    return result


def candidate_keys(kind, rows):
    if kind == "merge":
        keys = [f"merge:{int(r['segment_id'])}:{int(r['node_id'])}" for r in rows]
    else:
        keys = [f"split:{int(r['segment_id_a'])}:{int(r['segment_id_b'])}" for r in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("Native candidate identities must be unique")
    return keys


def pool_digest(kind, rows):
    return hashlib.sha256(json.dumps([kind, rows], sort_keys=True, allow_nan=False).encode()).hexdigest()


def build_native_table(payload, kind, module, bundle):
    """Native universe and native feature aggregation, not evolution approximations."""
    from proofreader_evolve.cli import precompute_error_scores as pc
    target = "merge_site_detection" if kind == "merge" else "split_detection"
    if module.DETECTOR_TARGET != target:
        raise ValueError(f"Fixed-pool {kind} requires {target}, got {module.DETECTOR_TARGET}")
    # Do not reuse sample caches supplied by other detectors or earlier policies.
    source = {k: v for k, v in payload.items() if not k.startswith("__detector_")}
    rows, truth = module.build_sample_universe(source)
    safe = blind_rows(kind, rows)
    keys = candidate_keys(kind, safe)
    truth = np.asarray(truth)
    if truth.shape != (len(safe),) or not np.isin(truth, [0, 1]).all():
        raise ValueError("Native evaluator labels must be aligned binary labels")
    # Keep feature input separate even though the trusted universe builder above
    # used annotations to attach evaluator-only labels after geometric selection.
    working = {"fragments_graph": payload["fragments_graph"], **inference_metadata(payload)}
    if safe:
        _, accumulator = pc._extract_candidate_features(working, module, safe, bundle["feature_order"])
        features = accumulator.to_frame().reset_index(drop=True)[list(bundle["feature_order"])].copy()
        scores = pc._score(bundle["pipeline"], features.to_numpy(dtype=float))
    else:
        features = pd.DataFrame(columns=list(bundle["feature_order"]), dtype=float)
        scores = np.empty(0)
    if "detector_score" in features.columns or not features.columns.is_unique:
        raise ValueError("Reserved or duplicate feature columns")
    if np.asarray(scores).shape != (len(safe),) or not np.isfinite(scores).all():
        raise ValueError("Native detector scores must be finite and aligned")
    features["detector_score"] = np.asarray(scores, dtype=float)
    return NativeTable(kind, features, safe, truth.astype(np.int8),
                       {"pool_sha256": pool_digest(kind, safe), "rows": len(keys),
                        "training_brain": bundle.get("train_brain"),
                        "candidate_policy": getattr(module, "CANDIDATE_POLICY_CONFIG_ID", None),
                        "candidate_policy_sha256": getattr(module, "CANDIDATE_POLICY_SHA256", None)})


@dataclass
class NativeTable:
    kind: str
    features: pd.DataFrame
    candidates: list
    truth: np.ndarray
    meta: dict

    @property
    def keys(self):
        return candidate_keys(self.kind, self.candidates)


@dataclass
class NativeTables:
    brain_id: str
    tables: dict
    meta: dict


def ensure_native_tables(brain, source_path, selected, tables_dir, prepare=False, mcl=100):
    """Build once or load exact native pools; old capped tables never match."""
    from proofreader_evolve.cli import precompute_error_scores as pc
    brain = str(brain)
    if not brain.isascii() or not brain.isdigit():
        raise ValueError("Brain ID must be numeric")
    inference_metadata({}, mcl)
    # A single universe per kind makes the benchmark contract unambiguous.
    # Ensembles can be added later only after explicit universe identity checks.
    if any(len(selected.get(kind, [])) != 1 for kind in ("merge", "split")):
        raise ValueError("Fixed native pools currently require exactly one detector per kind")
    identity = pc._cache_identity(source_path)
    plans = []
    for kind in ("merge", "split"):
        record = selected[kind][0]
        pc._assert_detector_unchanged(record)
        training = training_provenance(record)
        if training["mcl"] != mcl:
            raise ValueError("Native precision mode requires the detector's training MCL")
        provenance = {"version": POOL_VERSION, "brain": brain, "mcl": mcl,
                      "training": training,
                      "source_cache": identity, "detector": record, "kind": kind,
                      "implementation": {str(path.name): pc._sha256(path) for path in
                          (Path(__file__), Path(pc.__file__), Path(__file__).with_name("detector_metadata.py"))}}
        digest = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()
        directory = Path(tables_dir) / brain / digest
        plans.append((kind, record, provenance, directory))
    missing = [p for p in plans if not p[3].exists()]
    if missing and not prepare:
        raise FileNotFoundError(f"Missing matching detector-native tables for brain {brain}: "
                                + ", ".join(p[0] for p in missing)
                                + ". Run proofreader_evolve/prepare_feature_tables.py with the same MCL.")
    if missing:
        _, _, payload = pc.ds.load_cached_graphs(str(source_path), expect_brain=brain, expect_mcl=mcl)
        payload = {**payload, **inference_metadata(payload, mcl)}
        required = {"gt_graph", "gt_node_canonical_label", "gt_edge_error", "gt_merge_sites"}
        if not required.issubset(payload):
            raise ValueError("Native precision evaluation needs the matching labeled _add.pkl cache")
        try:
            for kind, record, provenance, directory in missing:
                pc.log(f"[{brain}/{kind}] preparing full detector-native universe (no evolution cap)")
                module = pc._import_module(Path(record["dir"]) / record["script"],
                                           f"native_{kind}_{record['script_sha256']}")
                bundle = pc._load_joblib(Path(record["dir"]) / record["file"])
                if list(bundle["feature_order"]) != provenance["training"]["feature_order"]:
                    raise ValueError("Model-selection metadata and fitted feature schema disagree")
                table = build_native_table(payload, kind, module, bundle)
                table.meta["training_brain"] = provenance["training"]["brain"]
                pc._assert_detector_unchanged(record)
                if training_provenance(record) != provenance["training"]:
                    raise RuntimeError("Training provenance changed during preparation")
                if pc._cache_identity(source_path) != identity:
                    raise RuntimeError("Source cache changed while preparing native tables")
                directory.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.TemporaryDirectory(prefix=".native_", dir=directory.parent) as tmp:
                    staging = Path(tmp)
                    table.features.to_pickle(staging / "features.pkl")
                    with (staging / "candidates.pkl").open("wb") as stream:
                        pickle.dump(table.candidates, stream, protocol=pickle.HIGHEST_PROTOCOL)
                    np.save(staging / "evaluator_labels.npy", table.truth, allow_pickle=False)
                    metadata = {**table.meta, "provenance": provenance,
                                "files": {name: pc._sha256(staging / name) for name in
                                          ("features.pkl", "candidates.pkl", "evaluator_labels.npy")}}
                    (staging / "meta.json").write_text(json.dumps(metadata, indent=2))
                    staging.rename(directory)
                pc.log(f"[{brain}/{kind}] saved {len(table.truth)} candidates to {directory}")
                del table, module, bundle
                gc.collect()
        finally:
            del payload
            gc.collect()
    tables, paths = {}, {}
    for kind, record, provenance, directory in plans:
        pc._assert_detector_unchanged(record)
        if training_provenance(record) != provenance["training"]:
            raise RuntimeError("Training provenance changed while loading")
        meta = json.loads((directory / "meta.json").read_text())
        if meta["provenance"] != provenance or set(meta["files"]) != {
                "features.pkl", "candidates.pkl", "evaluator_labels.npy"}:
            raise ValueError(f"Invalid native table metadata: {directory}")
        if any(pc._sha256(directory / name) != digest for name, digest in meta["files"].items()):
            raise ValueError(f"Corrupt native table: {directory}")
        with (directory / "candidates.pkl").open("rb") as stream:
            rows = pickle.load(stream)
        features = pd.read_pickle(directory / "features.pkl")
        truth = np.load(directory / "evaluator_labels.npy", allow_pickle=False)
        if (pool_digest(kind, rows) != meta["pool_sha256"] or len(rows) != len(features)
                or truth.shape != (len(rows),) or not np.isin(truth, [0, 1]).all()):
            raise ValueError("Native candidate/feature/label alignment failed")
        candidate_keys(kind, rows)
        tables[kind] = NativeTable(kind, features, rows, truth, meta)
        paths[kind] = str(directory.resolve())
    if pc._cache_identity(source_path) != identity:
        raise RuntimeError("Source cache changed while loading native tables")
    return NativeTables(brain, tables, {"table_paths": paths, "version": POOL_VERSION})
