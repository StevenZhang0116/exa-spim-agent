"""Export the AutoDiscovery companion files for one brain (dataset-derived only).

The companion accompanies `dataset_cache_<brain>_mcl<N>_add.pkl` in an AutoDiscovery
run that follows `markdowns/{merge,split}_features_for_proofreader_evolve.md`. It fixes
where features are evaluated and how they are judged, so findings transfer to
proofreader_evolve:

- the frozen candidate pool of each kind, in table row order (identities only);
- the three-valued label (1 error / 0 GT-confirmed no error / NaN not judgeable);
- label-free grouped folds and per-fold fitting masks (`internal_validation`);
- simple geometric columns the specification names as baselines (merge degree,
  split gap).

Nothing produced by a detector or an earlier run is written: no detector score, no
frozen features, no model output, no evolve K.

Run with panda on a compute node (the split folds load the fragment graph):
    python -u -m proofreader_evolve.cli.export_rerank_companion --brain 794495
"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from . import precompute_error_scores as pc
from ..harness import label_availability as la
from ..harness.internal_validation import FOLDS, fold_assignments, grouped_folds
from ..harness.native_pool import cache_path, ensure_native_tables

REPO = Path(__file__).resolve().parents[2]
# Next to the specification, so the files handed to AutoDiscovery sit together (gitignored).
DEFAULT_OUT = REPO / "markdowns"
ALIGNMENT = REPO / "configs" / "image_alignment.json"
COMPANION_VERSION = "rerank-companion-v1"
# Level geometry of the reviewed fused volumes (identical across the current brains;
# from the OME-Zarr multiscales metadata): world = voxel * scale + translation.
IMAGE_LEVELS = {"0": {"dataset": "0", "scale_xyz_um": [0.748, 0.748, 1.0], "translation_xyz_um": [0.0, 0.0, 0.0]},
                "1": {"dataset": "1", "scale_xyz_um": [1.496, 1.496, 2.0], "translation_xyz_um": [0.374, 0.374, 0.5]}}
RULES = {
    "merge": ("Candidates: every fragment node of degree >= 3 on a valid segment; greedy non-maximum "
              "suppression in order (-degree, node id) removes same-segment junctions within 20 um "
              "(Euclidean). Positive: geodesic distance (fragment cable) <= 20 um to a GT merge site, "
              "after snapping the site to the nearest node of its own segment within 50 um. Label 0 only "
              "where GT can judge: the segment carries a GT node label and the junction lies within "
              "150 um of a GT node; otherwise NaN."),
    "split": ("Candidates: from every degree-1 fragment tip, the nearest node of each other segment within "
              "50 um; the 2 nearest partner segments are kept (tip_to_any_node, r=50, k=2). One row per "
              "unordered segment pair; occurrences list every anchor/partner realisation, the first being "
              "the shortest gap. Positive: the pair is a GT split key (distinct non-zero labels across a GT "
              "split edge, or the two flanks of a zero-label GT stretch on one neuron). Label 0 only where "
              "GT can judge: both segments carry a GT node label; otherwise NaN."),
}


def _graph_provider(brain, mcl):
    from ..harness.dataset import load_cached_graphs
    state = {}

    def graph(provenance):
        if "graph" not in state:
            graph_, gt, payload = load_cached_graphs(provenance["source_cache"]["path"],
                                                     expect_brain=provenance["brain"], expect_mcl=provenance["mcl"])
            del gt, payload
            state["graph"] = graph_
        return state["graph"], None
    return SimpleNamespace(store=SimpleNamespace(graph=graph))


def _columns(kind, table):
    rows = table.candidates
    if kind == "merge":
        return {"segment_id": np.array([int(r["segment_id"]) for r in rows], dtype=np.int64),
                "node_id": np.array([int(r["node_id"]) for r in rows], dtype=np.int64),
                "degree": np.array([int(r["degree"]) for r in rows], dtype=np.int16),
                "x_um": np.array([float(r["x_um"]) for r in rows]),
                "y_um": np.array([float(r["y_um"]) for r in rows]),
                "z_um": np.array([float(r["z_um"]) for r in rows])}
    columns = {"segment_id_a": np.array([int(r["segment_id_a"]) for r in rows], dtype=np.int64),
               "segment_id_b": np.array([int(r["segment_id_b"]) for r in rows], dtype=np.int64),
               "node_id_a": np.array([int(r["node_id_a"]) for r in rows], dtype=np.int64),
               "node_id_b": np.array([int(r["node_id_b"]) for r in rows], dtype=np.int64),
               "gap_um": np.array([float(r["gap_um"]) for r in rows]),
               "anchor_side": np.array([r["anchor_side"] for r in rows], dtype="U1"),
               "partner_rank": np.array([int(r["partner_rank"]) for r in rows], dtype=np.int16),
               "node_role_a": np.array([r["node_role_a"] for r in rows], dtype="U7"),
               "node_role_b": np.array([r["node_role_b"] for r in rows], dtype="U7")}
    occurrence = [(i, o) for i, r in enumerate(rows) for o in r.get("occurrences", ())]
    columns.update({
        "occ_row": np.array([i for i, _ in occurrence], dtype=np.int64),
        "occ_node_id_a": np.array([int(o["node_id_a"]) for _, o in occurrence], dtype=np.int64),
        "occ_node_id_b": np.array([int(o["node_id_b"]) for _, o in occurrence], dtype=np.int64),
        "occ_gap_um": np.array([float(o["gap_um"]) for _, o in occurrence]),
        "occ_anchor_side": np.array([o["anchor_side"] for _, o in occurrence], dtype="U1"),
        "occ_partner_rank": np.array([int(o["partner_rank"]) for _, o in occurrence], dtype=np.int16),
        "occ_node_role_a": np.array([o["node_role_a"] for _, o in occurrence], dtype="U7"),
        "occ_node_role_b": np.array([o["node_role_b"] for _, o in occurrence], dtype="U7"),
    })
    return columns


def export(brain, mcl, out_dir, kinds):
    selected = pc.resolve_detector_runs(None, None)
    bank = ensure_native_tables(brain, cache_path(brain, mcl), selected, pc.DEFAULT_OUT, prepare=False, mcl=mcl)
    summary = la.apply({brain: bank})
    review = json.loads(ALIGNMENT.read_text())
    review = review.get("brains", review)[brain]
    if review.get("status") != "reviewed":
        raise ValueError(f"Image source of {brain} is not reviewed in {ALIGNMENT}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for kind in kinds:
        table = bank.tables[kind]
        if kind == "split":
            table.local_context = _graph_provider(brain, mcl)
        partitions, description = grouped_folds({brain: bank}, kind, selection_brains=[brain])
        folds = fold_assignments(partitions, brain, len(table.truth))
        if (folds < 0).any():
            raise ValueError("Every row must belong to exactly one fold")
        arrays = {"row": np.arange(len(table.truth), dtype=np.int64),
                  "label": np.asarray(table.truth, dtype=float),
                  "fold": folds.astype(np.int8),
                  **_columns(kind, table)}
        for f, (fit, _held) in enumerate(partitions):
            mask = np.zeros(len(table.truth), dtype=bool)
            mask[np.asarray(fit[brain], dtype=int)] = True
            arrays[f"fit_mask_fold{f}"] = mask
        path = out_dir / f"rerank_companion_{brain}_{kind}.npz"
        np.savez_compressed(path, **arrays)
        counts = la.label_counts(table.truth)
        manifest = {
            "version": COMPANION_VERSION, "created": datetime.now().astimezone().isoformat(timespec="seconds"),
            "brain": brain, "kind": kind, "mcl": mcl, "rows": int(len(table.truth)),
            "label_counts": counts, "labeled_rows": counts["1"] + counts["0"],
            "candidate_rule": RULES[kind], "candidate_policy": table.meta.get("candidate_policy"),
            "pool_sha256": table.meta["pool_sha256"], "labels_sha256": la.labels_digest(table.truth),
            "label_availability": {k: v for k, v in summary[f"{brain}/{kind}"].items() if k != "path"},
            "folds": {"n_folds": FOLDS, "version": description["version"], "grouping": description["grouping"],
                      "spatial_block_um": description["spatial_block_um"],
                      "partition_sha256": description["partition_sha256"],
                      "fit_masks": "fit_mask_fold<f>: rows usable for fitting when fold f is held out "
                                   "(rows sharing a fragment with fold f are purged)"},
            "source_cache": {"file": Path(table.meta["provenance"]["source_cache"]["path"]).name,
                             "size": table.meta["provenance"]["source_cache"]["size"]},
            "image": {"uri": review["image_uri"], "access": "public S3 (anonymous), OME-Zarr 0.4, axes t,c,z,y,x",
                      "graph_to_world": "identity (graph xyz are world microns)",
                      "levels": IMAGE_LEVELS,
                      "note": "verify scale/translation against the zarr multiscales metadata before use"},
            "columns": {name: str(value.dtype) for name, value in arrays.items()},
            "identity_columns_note": "segment/node ids and x/y/z locate candidates only; never feature inputs",
            "npz_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        (out_dir / f"rerank_companion_{brain}_{kind}.json").write_text(json.dumps(manifest, indent=1))
        written.append((path, manifest))
        print(f"{brain}/{kind}: {len(table.truth)} rows, labels {counts}, "
              f"{path.stat().st_size / 1e6:.1f} MB -> {path}", flush=True)
    return written


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--brain", required=True)
    p.add_argument("--mcl", type=int, default=100)
    p.add_argument("--kinds", nargs="+", default=["merge", "split"], choices=["merge", "split"])
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    a = p.parse_args(argv)
    export(a.brain, a.mcl, a.out_dir, a.kinds)


if __name__ == "__main__":
    main()
