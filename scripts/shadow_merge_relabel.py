#!/usr/bin/env python
"""Shadow merge relabeling: GT-node-count thresholds rescaled to the cache's node spacing.

``canonical_labeling`` ports three merge thresholds from ``segmentation_skeleton_metrics``
as GT NODE COUNTS (50 nodes): the node-count merge rule, the walk's pass-through test
and the two-GT junction audit. The canonical package reads raw GT SWCs (~1 um between
nodes), so 50 nodes is ~50 um there; the cache's GT graph is resampled (~4 um between
nodes), so the same 50 nodes is ~200 um and short genuine merges are dropped.

This script recomputes merge labels and sites from an existing ``_add.pkl`` with those
thresholds expressed as the node count that spans the same cable length in the cache:

    threshold_nodes = round(threshold_um / median GT edge length)    (threshold_um = 50)

It reuses the stored GT node labels (no segmentation or cloud reads), NEVER writes the
``_add.pkl`` and never imports ``proofreader_evolve``. For each brain it writes a JSON
with the stored labels, optionally a recomputation with the original thresholds (a
reproduction check), the rescaled variant, and each variant's agreement with the
canonical ``metrics_out/<brain>/*/merge_sites.csv`` (reference point Voxel * anisotropy,
see notebooks/verify_add_cache_merge_info.py) and ``results.csv``.

Run with panda on a compute node, one cache at a time:

    python -u scripts/shadow_merge_relabel.py --brains 789202 750318
    python -u scripts/shadow_merge_relabel.py --all --check-reproduce
"""
import argparse
import ast
from collections import Counter, defaultdict
from datetime import datetime
import functools
import gc
import glob
import json
import os
from pathlib import Path
import pickle
import re
import socket
import sys
import time

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT.parent / "agentic-neuron-proofreader" / "src"))

import agentic_neuron_proofreader  # noqa: E402,F401  (registers SkeletonGraph for pickle)
from agentic_neuron_proofreader.data_modules import canonical_labeling as cl  # noqa: E402

CANONICAL_THRESHOLD_UM = 50.0  # 50 nodes at the canonical ~1 um GT node spacing
ORIGINAL_NODES = 50


# --- threshold override --------------------------------------------------------------
class Thresholds:
    """Temporarily set every GT-node-count merge threshold in canonical_labeling.

    The module reads MERGE_MIN_NODES / MERGE_PASSTHRU_MIN_CC as globals at call time,
    but ``merge_labels`` and ``audit_junction_gt_connections`` bind MERGE_MIN_NODES as
    a default argument at import, so those two are wrapped as well.
    """

    def __init__(self, nodes):
        self.nodes = int(nodes)

    def __enter__(self):
        self.saved = {name: getattr(cl, name) for name in
                      ("MERGE_MIN_NODES", "MERGE_PASSTHRU_MIN_CC", "merge_labels",
                       "audit_junction_gt_connections")}
        cl.MERGE_MIN_NODES = self.nodes
        cl.MERGE_PASSTHRU_MIN_CC = self.nodes
        cl.merge_labels = functools.partial(self.saved["merge_labels"], min_nodes=self.nodes)
        cl.audit_junction_gt_connections = functools.partial(
            self.saved["audit_junction_gt_connections"], min_gt_nodes=self.nodes)
        return self

    def __exit__(self, *exc):
        for name, value in self.saved.items():
            setattr(cl, name, value)
        return False


def relabel(fragments, gt, node_label, nodes):
    """Merge labels + combined sites with every node-count threshold set to ``nodes``."""
    with Thresholds(nodes):
        merge_set = set(cl.merge_labels(gt, node_label))
        walk_labels, walk_sites = cl.geometric_merge_sites(fragments, gt, node_label, verbose=False)
        merge_set |= set(walk_labels)
        sites, metadata = cl.combined_merge_sites(fragments, gt, node_label, walk_sites, verbose=False)
        merge_set |= {int(s["segment_id"]) for s in sites}
        summary = cl.summarize(gt, node_label, merge_label_set=merge_set)
    return {"merge_labels": sorted(int(x) for x in merge_set), "sites": sites,
            "metadata": metadata, "summary": summary}


# --- canonical comparison -------------------------------------------------------------
def load_canonical(metrics_dir, brain, anisotropy):
    sites_csv = sorted(glob.glob(str(metrics_dir / brain / "*" / "merge_sites.csv")))
    results_csv = sorted(glob.glob(str(metrics_dir / brain / "*" / "results.csv")))
    if len(sites_csv) != 1 or len(results_csv) != 1:
        raise FileNotFoundError(f"need exactly one canonical run under {metrics_dir / brain}")
    raw = pd.read_csv(sites_csv[0])
    xyz = np.array([ast.literal_eval(str(v)) for v in raw["Voxel"]], float) * np.asarray(anisotropy)
    results = pd.read_csv(results_csv[0], index_col=0)
    return {"xyz": xyz, "segment": raw["Segment_ID"].astype(np.int64).to_numpy(),
            "neuron": raw["GroundTruth_ID"].astype(str).to_numpy(), "results": results,
            "paths": {"merge_sites": sites_csv[0], "results": results_csv[0]}}


def compare(sites, merge_labels, canon, tol_um):
    from scipy.spatial import cKDTree
    out = {"n_sites": len(sites), "n_merge_labels": len(merge_labels),
           "n_canonical_sites": int(len(canon["xyz"]))}
    if not sites or not len(canon["xyz"]):
        return out
    xyz = np.array([s["xyz"] for s in sites], float)
    d_canon, _ = cKDTree(xyz).query(canon["xyz"])
    d_cache, _ = cKDTree(canon["xyz"]).query(xyz)
    recall, precision = float((d_canon <= tol_um).mean()), float((d_cache <= tol_um).mean())
    canon_segments = set(int(s) for s in canon["segment"])
    labels = set(merge_labels)
    out.update({
        "site_recall": recall, "site_precision": precision,
        "site_f1": 2 * recall * precision / (recall + precision) if recall + precision else 0.0,
        # Are the canonical merge segments flagged at all (location aside)?
        "canonical_segment_recall": len(canon_segments & labels) / len(canon_segments),
        "canonical_segments": len(canon_segments),
    })
    # Per-neuron merge counts, as in results.csv '# Merges'.
    cache_counts = Counter(str(s["gt_neuron"]) for s in sites)
    canon_counts = canon["results"]["# Merges"].fillna(0).astype(int)
    diffs = [abs(cache_counts.get(name, 0) - int(n)) for name, n in canon_counts.items()]
    out["per_neuron_merges_mean_abs_diff"] = float(np.mean(diffs)) if diffs else None
    return out


def source_counts(sites):
    counts = Counter()
    for s in sites:
        counts["+".join(s.get("sources") or [s.get("source", "geometric_walk")])] += 1
    return dict(counts)


def strip(sites):
    keep = ("segment_id", "gt_neuron", "gt_neurons", "xyz", "source", "sources")
    return [{k: (list(map(float, v)) if k == "xyz" else v) for k, v in s.items() if k in keep}
            for s in sites]


# --- per brain ------------------------------------------------------------------------
def run_brain(brain, args):
    cache = args.cache_dir / f"dataset_cache_{brain}_mcl{args.mcl}_add.pkl"
    print(f"\n=== {brain}: loading {cache}", flush=True)
    started = time.monotonic()
    with open(cache, "rb") as f:
        payload = pickle.load(f)
    fragments, gt = payload["fragments_graph"], payload["gt_graph"]
    anisotropy = tuple(float(v) for v in payload["anisotropy"])
    node_label = np.asarray(payload["gt_node_canonical_label"])
    stored = {"merge_labels": sorted(int(x) for x in payload["gt_merge_labels"]),
              "sites": list(payload["gt_merge_sites"] or [])}
    stat = cache.stat()
    identity = {"path": str(cache.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    del payload
    gc.collect()

    xyz = np.asarray(gt.node_xyz)
    edges = np.asarray(list(gt.edges))
    spacing = float(np.median(np.linalg.norm(xyz[edges[:, 0]] - xyz[edges[:, 1]], axis=1)))
    scaled_nodes = max(1, int(round(args.threshold_um / spacing)))
    print(f"GT median node spacing {spacing:.2f} um -> {args.threshold_um:g} um = "
          f"{scaled_nodes} nodes (original {ORIGINAL_NODES})", flush=True)
    gt.set_kdtree()
    canon = load_canonical(args.metrics_dir, brain, anisotropy)

    variants = {"stored": stored}
    if args.check_reproduce:
        t = time.monotonic()
        variants["recomputed_original"] = relabel(fragments, gt, node_label, ORIGINAL_NODES)
        print(f"original thresholds recomputed in {time.monotonic() - t:.0f}s", flush=True)
    t = time.monotonic()
    variants["scaled"] = relabel(fragments, gt, node_label, scaled_nodes)
    print(f"scaled thresholds computed in {time.monotonic() - t:.0f}s", flush=True)

    record = {"brain": brain, "cache": identity, "anisotropy": anisotropy,
              "gt_median_node_spacing_um": spacing, "threshold_um": args.threshold_um,
              "original_nodes": ORIGINAL_NODES, "scaled_nodes": scaled_nodes,
              "tolerance_um": args.tol, "canonical": canon["paths"], "variants": {}}
    for name, v in variants.items():
        stats = compare(v["sites"], v["merge_labels"], canon, args.tol)
        stats["sources"] = source_counts(v["sites"])
        if "summary" in v:
            stats["summary"] = v["summary"]
        record["variants"][name] = {"stats": stats, "merge_labels": v["merge_labels"],
                                    "sites": strip(v["sites"]), "metadata": v.get("metadata")}
        print(f"  {name:<20} sites {stats['n_sites']:>5} | labels {stats['n_merge_labels']:>5} | "
              f"recall {stats.get('site_recall', float('nan')):.3f} | "
              f"precision {stats.get('site_precision', float('nan')):.3f} | "
              f"canonical segments flagged {stats.get('canonical_segment_recall', float('nan')):.3f}",
              flush=True)
    if args.check_reproduce:
        a = {(int(s["segment_id"]), tuple(np.round(s["xyz"], 3))) for s in stored["sites"]}
        b = {(int(s["segment_id"]), tuple(np.round(s["xyz"], 3)))
             for s in variants["recomputed_original"]["sites"]}
        record["reproduction"] = {"sites_equal": a == b, "labels_equal":
                                  stored["merge_labels"] == variants["recomputed_original"]["merge_labels"],
                                  "only_stored": len(a - b), "only_recomputed": len(b - a)}
        print(f"  reproduction: {record['reproduction']}", flush=True)
    record["elapsed_seconds"] = time.monotonic() - started
    path = args.out_dir / f"{brain}_mcl{args.mcl}.json"
    path.write_text(json.dumps(record, indent=1, default=float))
    print(f"saved {path}", flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--brains", nargs="+")
    parser.add_argument("--all", action="store_true", help="every *_mcl<N>_add.pkl in --cache-dir")
    parser.add_argument("--mcl", type=int, default=100)
    parser.add_argument("--threshold-um", type=float, default=CANONICAL_THRESHOLD_UM)
    parser.add_argument("--tol", type=float, default=50.0, help="site match tolerance (um)")
    parser.add_argument("--check-reproduce", action="store_true",
                        help="also recompute with the original thresholds (doubles the walk time)")
    parser.add_argument("--cache-dir", type=Path, default=PROJECT_ROOT / "cache")
    parser.add_argument("--metrics-dir", type=Path, default=PROJECT_ROOT / "metrics_out")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "notebooks" / "merge_label_shadow")
    args = parser.parse_args()
    if args.all:
        pattern = re.compile(rf"dataset_cache_(\d+)_mcl{args.mcl}_add\.pkl$")
        brains = sorted(m.group(1) for m in map(pattern.search, os.listdir(args.cache_dir)) if m)
    elif args.brains:
        brains = list(dict.fromkeys(args.brains))
    else:
        parser.error("give --brains or --all")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "run.json").write_text(json.dumps({
        "started": datetime.now().astimezone().isoformat(), "host": socket.gethostname(),
        "argv": sys.argv, "brains": brains, "threshold_um": args.threshold_um}, indent=1))

    rows = []
    for brain in brains:
        try:
            rec = run_brain(brain, args)
            for name, v in rec["variants"].items():
                s = v["stats"]
                rows.append({"brain": brain, "variant": name, "nodes": rec["scaled_nodes"] if name == "scaled" else ORIGINAL_NODES,
                             "sites": s["n_sites"], "canon_sites": s["n_canonical_sites"],
                             "labels": s["n_merge_labels"], "recall": s.get("site_recall"),
                             "precision": s.get("site_precision"),
                             "canon_seg_flagged": s.get("canonical_segment_recall"),
                             "pct_merged_edges": (s.get("summary") or {}).get("pct_merged_edges")})
        except Exception as exc:
            print(f"ERROR {brain}: {type(exc).__name__}: {exc}", flush=True)
            rows.append({"brain": brain, "variant": "ERROR"})
        gc.collect()
    table = pd.DataFrame(rows)
    table.to_csv(args.out_dir / "summary.csv", index=False)
    print("\nSummary\n" + table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    return 1 if (table["variant"] == "ERROR").any() else 0


if __name__ == "__main__":
    sys.exit(main())
