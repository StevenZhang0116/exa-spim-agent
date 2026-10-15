#!/usr/bin/env python3
"""Two gaps between canonical split events and the split pipeline, per brain.

The canonical metrics (``segmentation_skeleton_metrics``) publish no split sites, only per-neuron
counts, so this script derives the canonical split events itself: it labels the raw 1 um GT
tracings with the segmentation exactly as ``evaluate_skeleton_metrics.py`` does
(``DataLoader.load_groundtruth``, misalignment fix included) and turns every GT edge whose two
ends carry different non-zero segment labels into a split event, plus the pairs of labels that
flank a run of unlabeled (omitted) GT nodes. The same procedure on the cache's resampled GT
graph (``gt_node_canonical_label``) gives the cache's split events. A split event is identified
by its unordered segment pair, which is also how the split candidate pool identifies candidates.

For every canonical split pair the script measures

1. the **cache gap**: is the pair among the cache's split pairs (and how far is the nearest cache
   event with that pair);
2. the **candidate gap**: is the pair a row of the proofreader_evolve split table
   (``tip_to_any_node|r=50|k=2``, the frozen candidate rule);

and classifies the pairs missing from the cache: one segment absent from the fragments graph
(below the 100 um minimum cable length), a sliver segment touched by at most ``--sliver-nodes``
raw GT nodes (about that many um of GT, invisible to 4 um resampling), or other. As a
cross-check the candidate coverage of the cache's own pairs is reported too (the split sweep's
measure).

Read-only. Needs GCS credentials for the segmentation and a compute node (the cache plus the
labeled GT graphs); labeling the GT of one brain takes about half an hour.

    python -u notebooks/split_canonical_gap.py --measure --brains 794495
    python -u notebooks/split_canonical_gap.py --plot

Outputs, all under ``notebooks/split_canonical_gap/``: ``<brain>_mcl<N>.json``, ``summary.csv``,
``split_canonical_gap.svg``.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
import agentic_neuron_proofreader  # noqa: E402,F401  (registers SkeletonGraph for pickle)

DEFAULT_BRAINS = ["794495", "802449", "789202", "794493", "794491",
                  "747807", "750318", "751473", "754610", "754612", "754613"]
OUT_DIR = PROJECT_ROOT / "notebooks" / "split_canonical_gap"
CONFIG_DIR = PROJECT_ROOT / "configs"
MATCH_RADIUS_UM = 50.0


# ----------------------------------------------------------------------------- events
def split_events(edges, labels, xyz, neuron_of_node):
    """Split events of one labeled skeleton.

    ``edges``: iterable of (u, v) node ids; ``labels``: int array per node, 0 = unlabeled;
    ``xyz``: float array (n, 3) in um. Returns a list of dicts {pair, xyz, kind, neuron}.
    Direct events: an edge whose ends carry different non-zero labels. Flank events: the
    distinct non-zero labels adjacent to one connected run of zero-labeled nodes, paired.
    """
    events = []
    zero_adjacency = defaultdict(set)
    zero_boundary = defaultdict(set)   # zero node -> neighbouring non-zero labels (with positions)
    for u, v in edges:
        u, v = int(u), int(v)
        lu, lv = int(labels[u]), int(labels[v])
        if lu and lv:
            if lu != lv:
                events.append({"pair": (min(lu, lv), max(lu, lv)), "xyz": ((xyz[u] + xyz[v]) / 2).tolist(),
                               "kind": "direct", "neuron": neuron_of_node(u)})
        elif lu == 0 and lv == 0:
            zero_adjacency[u].add(v); zero_adjacency[v].add(u)
        else:
            zero, labelled = (u, v) if lu == 0 else (v, u)
            zero_adjacency.setdefault(zero, set())
            zero_boundary[zero].add((int(labels[labelled]), labelled))
    seen = set()
    for start in list(zero_adjacency):
        if start in seen:
            continue
        stack, component = [start], []
        seen.add(start)
        while stack:
            node = stack.pop(); component.append(node)
            for nxt in zero_adjacency[node]:
                if nxt not in seen:
                    seen.add(nxt); stack.append(nxt)
        boundary = {}
        for node in component:
            for label, labelled in zero_boundary.get(node, ()):
                boundary.setdefault(label, labelled)
        if len(boundary) >= 2:
            centre = np.mean([xyz[n] for n in component], axis=0).tolist()
            labels_sorted = sorted(boundary)
            for i, a in enumerate(labels_sorted):
                for b in labels_sorted[i + 1:]:
                    events.append({"pair": (a, b), "xyz": centre, "kind": "flank", "neuron": neuron_of_node(component[0])})
    return events


def canonical_events(brain, anisotropy):
    """Label the raw GT tracings with the segmentation as the canonical metrics do."""
    from dataset_config import get_segmentation_path
    from segmentation_skeleton_metrics.data_handling.graph_loading import DataLoader
    from segmentation_skeleton_metrics.utils.img_util import TensorStoreImage
    os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS", str(CONFIG_DIR / "allen-nd-goog-f5d46dbfa2cd.json"))
    segmentation_path = get_segmentation_path(brain, rtf_path=str(CONFIG_DIR / "segmentation_datasets.rtf"))
    gt_path = f"gs://allen-nd-goog/ground_truth_tracings/{brain}/voxel"
    segmentation = TensorStoreImage(segmentation_path, swap_axes=True)   # as in evaluate_skeleton_metrics.py
    loader = DataLoader(anisotropy=tuple(anisotropy), use_anisotropy=False, verbose=True)
    graphs = loader.load_groundtruth(gt_path, segmentation)
    events, label_nodes, n_nodes = [], defaultdict(int), 0
    for name, graph in graphs.items():
        nodes = list(graph.nodes)
        index = {n: i for i, n in enumerate(nodes)}
        labels = np.array([int(float(str(graph.node_label[n]))) if str(graph.node_label[n]) not in ("", "None") else 0
                           for n in nodes], dtype=np.int64)
        xyz = np.asarray(graph.node_voxel, dtype=float)[nodes] * np.asarray(anisotropy, float)
        for label in labels[labels != 0]:
            label_nodes[int(label)] += 1
        n_nodes += len(nodes)
        events.extend(split_events(((index[u], index[v]) for u, v in graph.edges), labels, xyz, lambda i, name=name: name))
    return events, dict(label_nodes), n_nodes, {"gt_path": gt_path, "segmentation": segmentation_path, "neurons": len(graphs)}


def cache_events(payload):
    gt = payload["gt_graph"]
    labels = np.asarray(payload["gt_node_canonical_label"]).astype(np.int64)
    xyz = np.asarray(gt.node_xyz, dtype=float)
    comp = np.asarray(gt.node_component_id)
    swc = {int(c): str(s) for c, s in gt.component_id_to_swc_id.items()}
    events = split_events(gt.edges, labels, xyz, lambda i: swc.get(int(comp[i]), str(comp[i])))
    return events, int(len(xyz))


def pool_pairs(tables_dir: Path, brain: str):
    for directory in sorted(tables_dir.glob(f"{brain}/*")):
        if directory.suffix == ".availability" or not (directory / "candidates.pkl").is_file():
            continue
        candidates = pickle.load(open(directory / "candidates.pkl", "rb"))
        if candidates and "segment_id_a" in candidates[0]:
            pairs = {(min(int(c["segment_id_a"]), int(c["segment_id_b"])), max(int(c["segment_id_a"]), int(c["segment_id_b"])))
                     for c in candidates}
            return pairs, len(candidates), str(directory)
    raise FileNotFoundError(f"no split table with candidates.pkl under {tables_dir / brain}")


# ----------------------------------------------------------------------------- per brain
def derive(rows):
    """Numbers over the pairs whose two segments both exist in the fragments graph: only those can
    be candidates (fragments below the MCL are absent by design), so they are the pipeline-relevant set."""
    present = [r for r in rows if not r["segment_absent_from_fragments"]]
    absent = [r for r in rows if r["segment_absent_from_fragments"]]
    def share(sub, pred):
        return float(np.mean([pred(r) for r in sub])) if sub else float("nan")
    direct = [r for r in present if r["kinds"] == ["direct"]]
    flank = [r for r in present if "flank" in r["kinds"]]
    return {
        "present_pairs": len(present), "absent_pairs": len(absent),
        "absent_cache_recall": share(absent, lambda r: r["in_cache"]),
        "present_cache_recall": share(present, lambda r: r["in_cache"]),
        "present_candidate_recall": share(present, lambda r: r["in_pool"]),
        "present_direct_pairs": len(direct), "present_direct_candidate_recall": share(direct, lambda r: r["in_pool"]),
        "present_flank_pairs": len(flank), "present_flank_candidate_recall": share(flank, lambda r: r["in_pool"]),
        "crosstab_present": {"cache_and_pool": sum(r["in_cache"] and r["in_pool"] for r in present),
                             "pool_only": sum((not r["in_cache"]) and r["in_pool"] for r in present),
                             "cache_only": sum(r["in_cache"] and not r["in_pool"] for r in present),
                             "neither": sum((not r["in_cache"]) and not r["in_pool"] for r in present)},
    }


def measure_brain(brain: str, args) -> dict:
    started = time.monotonic()
    cache_path = PROJECT_ROOT / "cache" / f"dataset_cache_{brain}_mcl{args.mcl}_add.pkl"
    payload = pickle.load(open(cache_path, "rb"))
    anisotropy = payload["anisotropy"]
    fragments = payload["fragments_graph"]
    fragment_segments = {int(float(str(s))) for s in fragments.component_id_to_swc_id.values()}
    c_events, c_nodes = cache_events(payload)
    del payload
    print(f"[{brain}] cache: {len(c_events)} split events on {c_nodes} GT nodes", flush=True)
    k_events, label_nodes, k_nodes, provenance = canonical_events(brain, anisotropy)
    print(f"[{brain}] canonical: {len(k_events)} split events on {k_nodes} GT nodes ({provenance['neurons']} neurons)", flush=True)
    pairs_pool, n_candidates, table_dir = pool_pairs(args.tables_dir, brain)

    def by_pair(events):
        out = defaultdict(list)
        for e in events:
            out[e["pair"]].append(e)
        return out
    k_by, c_by = by_pair(k_events), by_pair(c_events)
    rows = []
    for pair, evs in k_by.items():
        in_cache = pair in c_by
        dist = float("inf")
        if in_cache:
            tree = cKDTree(np.array([e["xyz"] for e in c_by[pair]]))
            dist = float(min(tree.query(np.array(e["xyz"]))[0] for e in evs))
        min_label_nodes = min(label_nodes.get(pair[0], 0), label_nodes.get(pair[1], 0))
        absent = not (pair[0] in fragment_segments and pair[1] in fragment_segments)
        rows.append({"pair": list(pair), "events": len(evs), "kinds": sorted({e["kind"] for e in evs}),
                     "neurons": sorted({e["neuron"] for e in evs}), "in_cache": in_cache,
                     "cache_distance_um": None if not np.isfinite(dist) else dist, "in_pool": pair in pairs_pool,
                     "min_label_gt_nodes": int(min_label_nodes), "segment_absent_from_fragments": absent,
                     "sliver": (not absent) and min_label_nodes <= args.sliver_nodes})
    n = len(rows)
    def share(pred):
        return float(np.mean([pred(r) for r in rows])) if rows else float("nan")
    missing = [r for r in rows if not r["in_cache"]]
    summary = {
        "canonical_pairs": n, "canonical_events": len(k_events), "cache_pairs": len(c_by), "cache_events": len(c_events),
        "pool_candidates": n_candidates,
        "cache_recall_pairs": share(lambda r: r["in_cache"]),
        "cache_recall_pairs_within_50um": share(lambda r: r["in_cache"] and r["cache_distance_um"] <= MATCH_RADIUS_UM),
        "cache_precision_pairs": float(np.mean([p in k_by for p in c_by])) if c_by else float("nan"),
        "candidate_recall_canonical_pairs": share(lambda r: r["in_pool"]),
        "candidate_reach_cache_pairs": float(np.mean([p in pairs_pool for p in c_by])) if c_by else float("nan"),
        "missing_from_cache": len(missing),
        "missing_segment_absent_from_fragments": sum(r["segment_absent_from_fragments"] for r in missing),
        "missing_sliver": sum(r["sliver"] for r in missing),
        "missing_other": sum((not r["segment_absent_from_fragments"]) and (not r["sliver"]) for r in missing),
        "missing_other_in_pool": sum((not r["segment_absent_from_fragments"]) and (not r["sliver"]) and r["in_pool"] for r in missing),
        "crosstab": {"cache_and_pool": sum(r["in_cache"] and r["in_pool"] for r in rows),
                     "pool_only": sum((not r["in_cache"]) and r["in_pool"] for r in rows),
                     "cache_only": sum(r["in_cache"] and not r["in_pool"] for r in rows),
                     "neither": sum((not r["in_cache"]) and not r["in_pool"] for r in rows)},
    }
    summary.update(derive(rows))
    s = summary
    print(f"[{brain}] present pairs {s['present_pairs']} (both segments in the fragments graph): in cache "
          f"{s['present_cache_recall']:.3f}, in pool {s['present_candidate_recall']:.3f} (direct {s['present_direct_candidate_recall']:.3f}, "
          f"flank {s['present_flank_candidate_recall']:.3f}); {s['absent_pairs']} pairs with a segment below the MCL, in cache "
          f"{s['absent_cache_recall']:.3f}", flush=True)
    print(f"[{brain}] canonical pairs {n}: in cache {s['cache_recall_pairs']:.3f} (within 50 um {s['cache_recall_pairs_within_50um']:.3f}), "
          f"in pool {s['candidate_recall_canonical_pairs']:.3f} | cache pairs {len(c_by)}: precision {s['cache_precision_pairs']:.3f}, "
          f"in pool {s['candidate_reach_cache_pairs']:.3f} (sweep check) | missing from cache {len(missing)}: absent segment "
          f"{s['missing_segment_absent_from_fragments']}, sliver {s['missing_sliver']}, other {s['missing_other']} "
          f"(of which in pool {s['missing_other_in_pool']}) | {time.monotonic() - started:.0f}s", flush=True)
    return {"brain": brain, "mcl": args.mcl, "cache": str(cache_path), "split_table": table_dir, "provenance": provenance,
            "sliver_nodes": args.sliver_nodes, "canonical_gt_nodes": k_nodes, "cache_gt_nodes": c_nodes,
            "summary": summary, "pairs": rows, "seconds": time.monotonic() - started}


# ----------------------------------------------------------------------------- figure
SERIES = [("cache_and_pool", "in cache labels and in the candidate pool", "#2a78d6"),
          ("pool_only", "candidate exists, cache label missing", "#eb6834"),
          ("cache_only", "cache label exists, no candidate", "#1baf7a"),
          ("neither", "neither", "#eda100")]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e8e7e3", "#fcfcfb"


def plot(results, fig_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for r in results.values():  # older JSON files may predate derive()
        r["summary"].update(derive(r["pairs"]))
    brains = sorted(results, key=lambda b: -results[b]["summary"]["canonical_pairs"])
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 0.5 * len(brains) + 2.6), gridspec_kw={"width_ratios": [3, 2]})
    fig.patch.set_facecolor(SURFACE)
    for a in (ax, ax2):
        a.set_facecolor(SURFACE)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            a.spines[side].set_color(GRID)
        a.grid(True, axis="x", color=GRID, linewidth=0.8); a.set_axisbelow(True)
        a.tick_params(colors=INK2, labelsize=8.5, length=0)
    y = np.arange(len(brains))
    left = np.zeros(len(brains))
    for key, label, colour in SERIES:
        vals = np.array([results[b]["summary"]["crosstab_present"][key] / max(1, results[b]["summary"]["present_pairs"]) for b in brains])
        ax.barh(y, vals, left=left, color=colour, edgecolor=SURFACE, linewidth=2, height=0.62, label=label)
        for i, (v, l) in enumerate(zip(vals, left)):
            if v >= 0.07:
                ax.text(l + v / 2, i, f"{v:.0%}", ha="center", va="center", fontsize=8, color=SURFACE if key in ("cache_and_pool",) else INK)
        left += vals
    ax.set_yticks(y); ax.set_yticklabels([f"{b}  ({results[b]['summary']['present_pairs']} of {results[b]['summary']['canonical_pairs']} pairs)" for b in brains], fontsize=8.5, color=INK)
    ax.invert_yaxis(); ax.set_xlim(0, 1); ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0]); ax.set_xticklabels(["0", "25%", "50%", "75%", "100%"])
    ax.set_title("Pairs with both segments in the fragments graph", fontsize=9.5, color=INK, loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.9 / (len(brains) + 1.5)), ncol=2, frameon=False, fontsize=8, labelcolor=INK)
    # right: why pairs are missing from the cache
    causes = [("missing_segment_absent_from_fragments", "segment below the MCL (absent from fragments)", "#2a78d6"),
              ("missing_sliver", "sliver segment (few raw GT nodes)", "#eb6834"),
              ("missing_other", "other", "#1baf7a")]
    left = np.zeros(len(brains))
    for key, label, colour in causes:
        vals = np.array([results[b]["summary"][key] / max(1, results[b]["summary"]["missing_from_cache"]) for b in brains])
        ax2.barh(y, vals, left=left, color=colour, edgecolor=SURFACE, linewidth=2, height=0.62, label=label)
        for i, (v, l) in enumerate(zip(vals, left)):
            if v >= 0.08:
                ax2.text(l + v / 2, i, f"{v:.0%}", ha="center", va="center", fontsize=8, color=SURFACE if key.endswith("fragments") else INK)
        left += vals
    ax2.set_yticks(y); ax2.set_yticklabels([f"{results[b]['summary']['missing_from_cache']} missing" for b in brains], fontsize=8.5, color=INK2)
    ax2.invert_yaxis(); ax2.set_xlim(0, 1); ax2.set_xticks([0, 0.5, 1.0]); ax2.set_xticklabels(["0", "50%", "100%"])
    ax2.set_title("Why pairs are missing from the cache labels (all pairs)", fontsize=9.5, color=INK, loc="left")
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.9 / (len(brains) + 1.5)), ncol=1, frameon=False, fontsize=8, labelcolor=INK)
    fig.suptitle("Canonical split pairs: in the cache labels? a candidate in the split pool (tip_to_any_node, r 50 um, k 2)?", fontsize=11, color=INK, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(fig_path, bbox_inches="tight", facecolor=SURFACE)
    print(f"figure -> {fig_path}")


def write_summary(results, out_dir: Path):
    rows = []
    for brain, r in sorted(results.items()):
        s = dict(r["summary"]); s.update(derive(r["pairs"])); ct = s.pop("crosstab"); ctp = s.pop("crosstab_present")
        rows.append({"brain": brain, **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items()},
                     **{f"all_{k}": v for k, v in ct.items()}, **{f"present_{k}": v for k, v in ctp.items()}})
    frame = pd.DataFrame(rows)
    frame.to_csv(out_dir / "summary.csv", index=False)
    print(frame.to_string(index=False))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--measure", action="store_true")
    p.add_argument("--plot", action="store_true")
    p.add_argument("--brains", nargs="+", default=DEFAULT_BRAINS)
    p.add_argument("--mcl", type=int, default=100)
    p.add_argument("--sliver-nodes", type=int, default=4, help="a label on at most this many raw GT nodes is a sliver")
    p.add_argument("--tables-dir", type=Path, default=PROJECT_ROOT / "proofreader_evolve" / "feature_tables")
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = p.parse_args(argv)
    if not (args.measure or args.plot):
        p.error("pass --measure and/or --plot")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.measure:
        for brain in args.brains:
            result = measure_brain(brain, args)
            (args.out_dir / f"{brain}_mcl{args.mcl}.json").write_text(json.dumps(result, indent=1))
    if args.plot:
        results = {}
        for brain in args.brains:
            path = args.out_dir / f"{brain}_mcl{args.mcl}.json"
            if path.is_file():
                results[brain] = json.loads(path.read_text())
        if not results:
            p.error(f"no measurements under {args.out_dir}")
        write_summary(results, args.out_dir)
        plot(results, args.out_dir / "split_canonical_gap.svg")


if __name__ == "__main__":
    main()
