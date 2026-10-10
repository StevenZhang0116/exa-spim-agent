"""Host-only label availability for detector-native candidate tables (three-valued labels).

``evaluator_labels.npy`` is binary, but GT covers only a few traced neurons per brain,
so most label-0 rows mean "no GT here", not "no error". An availability sidecar marks,
row by row, whether the label is supported by GT. Together with the binary label it
gives a three-valued label: 1 = error, 0 = no error (GT available), unavailable = NaN.

The sidecar is stored NEXT TO the table directory, never inside it, so the table
digest, the tables themselves and every context cache built from them stay valid:

    feature_tables/<brain>/<table_digest>.availability/
        availability.npz   one boolean array per definition, in table row order
        meta.json          definition version, table bindings, fractions, provenance

Bindings checked when writing AND when loading (a mismatch raises; ``load`` returns None
for a missing sidecar, and ``apply``, which evolution uses, raises):

- row count and candidate identity/order (``pool_sha256`` and a hash of the keys);
- the exact labels (``labels_sha256`` of ``evaluator_labels.npy``);
- the source ``_add.pkl`` identity recorded in the table provenance;
- every positive row is available, for every stored definition.

``apply`` swaps each loaded table's ``truth`` for the three-valued float array before
any scoring, so every metric, fit and diagnostic uses GT-labeled rows only.

Availability is derived from GT. It must never reach a scorer, a fitted model's
``predict``, feature frames, descriptors or reviser-visible files: unavailable rows are
never positive, so exposing the mask would let a scorer win by ranking GT coverage.
"""
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np

AVAILABILITY_VERSION = "gt-available-v1"
NEAR_GT_UM = 150.0  # the frozen merge policy's claim radius (CANDIDATE_CLAIM_RADIUS_UM)
DEFINITIONS = {"merge": ("segment_on_gt", "near_gt_150"), "split": ("segment_on_gt",)}
DEFAULT_DEFINITION = {"merge": "near_gt_150", "split": "segment_on_gt"}
SUFFIX = ".availability"


class AvailabilityMismatch(ValueError):
    """The sidecar does not belong to (or contradicts) the loaded table."""


def sidecar_dir(table_dir):
    table_dir = Path(table_dir)
    return table_dir.parent / (table_dir.name + SUFFIX)


def labels_digest(truth):
    """SHA-256 of labels as int8 (NaN -> -1); identical to the binary digest for 0/1 labels."""
    truth = np.asarray(truth)
    if truth.dtype.kind == "f":
        truth = np.where(np.isnan(truth), -1, truth)
    return hashlib.sha256(truth.astype(np.int8).tobytes()).hexdigest()


def labeled_rows(truth):
    """Indices of GT-judgeable rows (label 0 or 1); raises on any other value."""
    truth = np.asarray(truth, dtype=float)
    labeled = ~np.isnan(truth)
    if truth.ndim != 1 or not np.isin(truth[labeled], [0., 1.]).all():
        raise ValueError("Labels must be a 1-D array of 1 / 0 / NaN")
    return np.flatnonzero(labeled)


def label_counts(truth):
    truth = np.asarray(truth, dtype=float)
    nan = int(np.isnan(truth).sum())
    ones = int((truth == 1).sum())
    return {"1": ones, "0": int(len(truth) - nan - ones), "nan": nan}


def keys_digest(keys):
    return hashlib.sha256("\n".join(map(str, keys)).encode()).hexdigest()


def gt_segments(gt_node_label):
    labels = np.asarray(gt_node_label)
    return {int(x) for x in np.unique(labels) if int(x) != 0}


def three_valued(truth, available):
    """Three-valued label: 1 = error, 0 = GT-confirmed no error, NaN = not judgeable by GT.

    Raises if a positive row is marked unavailable (the inputs contradict each other).
    """
    truth, available = np.asarray(truth), np.asarray(available)
    if available.dtype != bool or available.shape != truth.shape or not np.isin(truth, [0, 1]).all():
        raise AvailabilityMismatch("Labels and availability must be aligned binary / boolean arrays")
    if np.any((truth == 1) & ~available):
        raise AvailabilityMismatch("A positive row is marked unavailable")
    labels = truth.astype(float)
    labels[~available] = np.nan
    return labels


def compute(kind, candidates, gt_node_label, gt_node_xyz=None):
    """Boolean arrays (table row order) for every definition of ``kind``."""
    segments = gt_segments(gt_node_label)
    if kind == "merge":
        on_gt = np.fromiter((int(c["segment_id"]) in segments for c in candidates), bool, len(candidates))
        if gt_node_xyz is None:
            raise ValueError("near_gt_150 needs GT node coordinates")
        from scipy.spatial import cKDTree
        xyz = np.array([[c["x_um"], c["y_um"], c["z_um"]] for c in candidates], dtype=float).reshape(-1, 3)
        dist, _ = cKDTree(np.asarray(gt_node_xyz, dtype=float)).query(xyz) if len(xyz) else (np.empty(0), None)
        return {"segment_on_gt": on_gt, "near_gt_150": on_gt & (np.asarray(dist) <= NEAR_GT_UM)}
    if kind == "split":
        on_gt = np.fromiter((int(c["segment_id_a"]) in segments and int(c["segment_id_b"]) in segments
                             for c in candidates), bool, len(candidates))
        return {"segment_on_gt": on_gt}
    raise ValueError(f"Unknown candidate kind {kind!r}")


def check(arrays, kind, truth, keys):
    """Raise AvailabilityMismatch unless the arrays are a valid sidecar for these rows."""
    truth = np.asarray(truth)
    expected = set(DEFINITIONS[kind])
    if set(arrays) != expected:
        raise AvailabilityMismatch(f"{kind} sidecar definitions {sorted(arrays)} != {sorted(expected)}")
    for name, mask in arrays.items():
        mask = np.asarray(mask)
        if mask.dtype != bool or mask.shape != truth.shape or len(keys) != len(truth):
            raise AvailabilityMismatch(f"{kind}/{name}: availability is not aligned with the table rows")
        bad = np.flatnonzero((truth == 1) & ~mask)
        if len(bad):
            sample = [str(keys[i]) for i in bad[:5]]
            raise AvailabilityMismatch(f"{kind}/{name}: {len(bad)} positive rows are marked unavailable "
                                       f"(e.g. {sample}); the sidecar contradicts the table labels")


def write(table_dir, kind, arrays, *, truth, keys, pool_sha256, source_cache, provenance=None):
    """Validate, then write the sidecar atomically beside ``table_dir``."""
    check(arrays, kind, truth, keys)
    target = sidecar_dir(table_dir)
    truth = np.asarray(truth)
    meta = {
        "version": AVAILABILITY_VERSION, "kind": kind, "table": Path(table_dir).name,
        "rows": int(len(truth)), "pool_sha256": pool_sha256, "candidate_keys_sha256": keys_digest(keys),
        "labels_sha256": labels_digest(truth), "source_cache": source_cache,
        "positives": int(truth.sum()), "near_gt_um": NEAR_GT_UM,
        "definitions": {name: {"available_rows": int(np.asarray(m).sum()),
                               "available_fraction": float(np.asarray(m).mean()) if len(truth) else 0.0,
                               "positives_available": int(((truth == 1) & np.asarray(m)).sum()),
                               "three_valued_counts": {"1": int(((truth == 1) & np.asarray(m)).sum()),
                                                       "0": int(((truth == 0) & np.asarray(m)).sum()),
                                                       "nan": int((~np.asarray(m)).sum())}}
                        for name, m in arrays.items()},
        "provenance": provenance or {},
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".availability_", dir=target.parent) as tmp:
        staging = Path(tmp) / "out"
        staging.mkdir()
        np.savez(staging / "availability.npz", **{k: np.asarray(v, dtype=bool) for k, v in arrays.items()})
        meta["files"] = {"availability.npz": hashlib.sha256((staging / "availability.npz").read_bytes()).hexdigest()}
        (staging / "meta.json").write_text(json.dumps(meta, indent=1, sort_keys=True))
        if target.exists():
            old = Path(tmp) / "old"
            os.replace(target, old)
        os.replace(staging, target)
    return target


def load(table, table_dir, definition=None):
    """The availability mask for one loaded NativeTable, or None if no sidecar exists.

    Every binding is re-checked against the table as loaded; any mismatch raises.
    """
    directory = sidecar_dir(table_dir)
    if not directory.exists():
        return None
    meta = json.loads((directory / "meta.json").read_text())
    if meta.get("version") != AVAILABILITY_VERSION or meta.get("kind") != table.kind:
        raise AvailabilityMismatch(f"{directory}: version/kind {meta.get('version')}/{meta.get('kind')} "
                                   f"does not match {AVAILABILITY_VERSION}/{table.kind}")
    npz_path = directory / "availability.npz"
    if hashlib.sha256(npz_path.read_bytes()).hexdigest() != meta.get("files", {}).get("availability.npz"):
        raise AvailabilityMismatch(f"{directory}: availability.npz checksum mismatch")
    keys = table.keys
    expected = {"rows": len(table.truth), "pool_sha256": table.meta.get("pool_sha256"),
                "candidate_keys_sha256": keys_digest(keys), "labels_sha256": labels_digest(table.truth),
                "source_cache": table.meta.get("provenance", {}).get("source_cache")}
    for field, value in expected.items():
        if meta.get(field) != value:
            raise AvailabilityMismatch(f"{directory}: {field} does not match the loaded table "
                                       f"(regenerate with cli.precompute_availability)")
    with np.load(npz_path, allow_pickle=False) as data:
        arrays = {name: data[name] for name in data.files}
    check(arrays, table.kind, table.truth, keys)
    name = definition or DEFAULT_DEFINITION[table.kind]
    if name not in arrays:
        raise AvailabilityMismatch(f"{directory}: no definition {name!r} for {table.kind}")
    return {"mask": arrays[name], "definition": name, "version": AVAILABILITY_VERSION,
            "path": str(directory.resolve())}


def apply(banks):
    """Replace every table's binary ``truth`` with the three-valued label, in place.

    The sidecar is required: a missing or mismatched sidecar raises. Tables shared by
    several bank dicts are converted once. The mask itself is not kept on the table;
    NaN in ``table.truth`` is the only trace, and ``truth`` never reaches scorers,
    ``predict``, feature frames or descriptors.
    Returns {brain/kind: {definition, version, path, counts}}.
    """
    summary = {}
    for brain, bank in banks.items():
        for kind, table in bank.tables.items():
            if getattr(table, "availability", None) is None:
                table_dir = (bank.meta or {}).get("table_paths", {}).get(kind)
                record = None if table_dir is None else load(table, table_dir)
                if record is None:
                    raise AvailabilityMismatch(
                        f"{brain}/{kind}: no availability sidecar beside {table_dir}; run "
                        f"python -m proofreader_evolve.cli.precompute_availability --brains {brain}")
                table.truth = three_valued(table.truth, record["mask"])
                table.availability = {k: v for k, v in record.items() if k != "mask"}
                table.availability["counts"] = label_counts(table.truth)
                if not table.availability["counts"]["1"] or not table.availability["counts"]["0"]:
                    raise AvailabilityMismatch(f"{brain}/{kind}: GT-labeled rows lack one of the classes")
            summary[f"{brain}/{kind}"] = dict(table.availability)
    return summary
