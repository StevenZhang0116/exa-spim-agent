#!/usr/bin/env python3
"""Two gaps between canonical merge sites and the merge pipeline, per brain.

For every canonical merge site (``metrics_out/<brain>/*/merge_sites.csv``, reference point
``Voxel * anisotropy``) this script measures

1. the **cache gap**: the Euclidean distance to the nearest merge site baked into the
   dataset cache (``payload["gt_merge_sites"]``); the share within 50 um is the site recall
   of ``notebooks/merge_label_shadow_findings.md`` section 1;
2. the **candidate gap**: the geodesic cable distance, along the fragment skeleton, to the
   nearest merge candidate of the frozen junction rule (``junction|nms=20|r=150``), using the
   candidates of the proofreader_evolve merge table and the detector's own site rule (snap the
   site to the nearest node of its own segment within 50 um, bounded Dijkstra to 150 um).
   The share within 20 um is the fraction of canonical merges the current pool labels
   positive; within 150 um, the fraction it can claim at all.

As a cross-check the candidate distances are also measured from the cache's own sites, which
is what ``notebooks/merge_candidate_pool_sweep_outputs_with_oldscope`` reports (denominator:
cache sites); at r = 20 um it must reproduce that sweep's ``site_recall``.

Read-only: no cache, table or label is modified. Run in panda on a compute node (one cache at
a time, > 20 GB RAM):

    python -u notebooks/merge_canonical_gap.py --measure                  # all brains with tables
    python -u notebooks/merge_canonical_gap.py --measure --brains 794495
    python -u notebooks/merge_canonical_gap.py --plot                     # figure + summary.csv

Outputs, all under ``notebooks/merge_canonical_gap/``: ``<brain>_mcl<N>.json`` (per-site
distances), ``summary.csv`` and ``merge_canonical_gap.svg``.
"""
from __future__ import annotations

import argparse
import ast
import glob
import heapq
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
import agentic_neuron_proofreader  # noqa: E402,F401  (registers SkeletonGraph for pickle)

DEFAULT_BRAINS = ["794495", "802449", "789202", "794493", "794491",
                  "747807", "750318", "751473", "754610", "754612", "754613"]
RADII_UM = [5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 75.0, 100.0, 150.0]
SNAP_MAX_UM = 50.0          # detector: CANDIDATE_SITE_SNAP_MAX_UM
CLAIM_RADIUS_UM = 150.0     # detector: CANDIDATE_CLAIM_RADIUS_UM
POSITIVE_RADIUS_UM = 20.0   # detector: CANDIDATE_POSITIVE_LABEL_RADIUS_UM
MATCH_RADIUS_UM = 50.0      # site matching tolerance of the shadow findings
OUT_DIR = PROJECT_ROOT / "notebooks" / "merge_canonical_gap"


# ----------------------------------------------------------------------------- inputs
def load_canonical(metrics_dir: Path, brain: str, anisotropy) -> dict:
    sites_csv = sorted(glob.glob(str(metrics_dir / brain / "*" / "merge_sites.csv")))
    if len(sites_csv) != 1:
        raise FileNotFoundError(f"need exactly one canonical run under {metrics_dir / brain}")
    raw = pd.read_csv(sites_csv[0])
    xyz = np.array([ast.literal_eval(str(v)) for v in raw["Voxel"]], float) * np.asarray(anisotropy, float)
    return {"xyz": xyz, "segment": raw["Segment_ID"].astype(np.int64).to_numpy(), "path": sites_csv[0]}


def load_merge_candidates(tables_dir: Path, brain: str) -> tuple[np.ndarray, np.ndarray, str]:
    """Node ids and segment ids of the frozen merge pool (the proofreader_evolve merge table)."""
    for directory in sorted(tables_dir.glob(f"{brain}/*")):
        if directory.suffix == ".availability" or not (directory / "candidates.pkl").is_file():
            continue
        candidates = pickle.load(open(directory / "candidates.pkl", "rb"))
        if candidates and "node_id" in candidates[0] and "degree" in candidates[0]:
            nodes = np.fromiter((int(c["node_id"]) for c in candidates), dtype=np.int64, count=len(candidates))
            segments = np.fromiter((int(c["segment_id"]) for c in candidates), dtype=np.int64, count=len(candidates))
            return nodes, segments, str(directory)
    raise FileNotFoundError(f"no merge table with candidates.pkl under {tables_dir / brain}")


def node_arrays(frag):
    xyz = np.asarray(frag.node_xyz, dtype=float)
    components = np.asarray(frag.node_component_id, dtype=np.int64)
    # swc ids are stored as strings such as '10870650356.0'
    comp_to_seg = {int(c): int(float(str(s))) for c, s in frag.component_id_to_swc_id.items()}
    segments = np.fromiter((comp_to_seg.get(int(c), -1) for c in components), dtype=np.int64, count=len(components))
    return xyz, segments


# ----------------------------------------------------------------------------- distances
def geodesic_to_candidates(frag, xyz, node_segments, tree, kept, sites_xyz, sites_segment):
    """Per site: (snapped?, min geodesic cable distance to a kept candidate node, inf if none
    within CLAIM_RADIUS_UM). Mirrors the detector's _site_distances_to_kept."""
    snapped = np.zeros(len(sites_xyz), dtype=bool)
    distance = np.full(len(sites_xyz), np.inf)
    for i, (point, segment) in enumerate(zip(sites_xyz, sites_segment)):
        neighbours = np.asarray(tree.query_ball_point(point, r=SNAP_MAX_UM), dtype=np.int64)
        own = neighbours[node_segments[neighbours] == int(segment)] if len(neighbours) else neighbours
        if not len(own):
            continue
        start = int(own[int(np.argmin(np.linalg.norm(xyz[own] - point, axis=1)))])
        snapped[i] = True
        best = {start: 0.0}
        heap = [(0.0, start)]
        site_best = np.inf
        while heap:
            d, node = heapq.heappop(heap)
            if d > best.get(node, np.inf):
                continue
            if kept[node]:
                site_best = min(site_best, d)
            for neighbour in frag.neighbors(node):
                neighbour = int(neighbour)
                step = d + float(np.linalg.norm(xyz[neighbour] - xyz[node]))
                if step <= CLAIM_RADIUS_UM and step < best.get(neighbour, np.inf):
                    best[neighbour] = step
                    heapq.heappush(heap, (step, neighbour))
        distance[i] = site_best
    return snapped, distance


def nearest(points_a, points_b):
    if not len(points_a):
        return np.zeros(0)
    if not len(points_b):
        return np.full(len(points_a), np.inf)
    d, _ = cKDTree(points_b).query(points_a)
    return d


# ----------------------------------------------------------------------------- per brain
def measure_brain(brain: str, args) -> dict:
    started = time.monotonic()
    cache_path = PROJECT_ROOT / "cache" / f"dataset_cache_{brain}_mcl{args.mcl}_add.pkl"
    payload = pickle.load(open(cache_path, "rb"))
    frag = payload["fragments_graph"]
    anisotropy = payload["anisotropy"]
    cache_sites = list(payload.get("gt_merge_sites") or [])
    cache_xyz = np.array([s["xyz"] for s in cache_sites], float).reshape(-1, 3)
    cache_segment = np.array([int(s["segment_id"]) for s in cache_sites], dtype=np.int64)
    canonical = load_canonical(args.metrics_dir, brain, anisotropy)
    xyz, node_segments = node_arrays(frag)
    cand_nodes, cand_segments, table_dir = load_merge_candidates(args.tables_dir, brain)
    # The table must belong to this cache: candidate node ids must be junctions of the named segments.
    degrees = np.fromiter((frag.degree(int(n)) for n in cand_nodes), dtype=np.int64, count=len(cand_nodes))
    mismatch = int(np.sum((node_segments[cand_nodes] != cand_segments) | (degrees < 3)))
    if mismatch:
        raise ValueError(f"{brain}: {mismatch} merge candidates do not match the cache graph "
                         f"(table {table_dir}); rebuild the table from this cache first")
    kept = np.zeros(len(xyz), dtype=bool)
    kept[cand_nodes] = True
    tree = frag.kdtree if getattr(frag, "kdtree", None) is not None else cKDTree(xyz)
    print(f"[{brain}] loaded in {time.monotonic() - started:.0f}s: {len(xyz)} nodes, {len(cand_nodes)} merge "
          f"candidates, {len(cache_xyz)} cache sites, {len(canonical['xyz'])} canonical sites", flush=True)

    canon_to_cache = nearest(canonical["xyz"], cache_xyz)          # cache gap (Euclidean)
    cache_to_canon = nearest(cache_xyz, canonical["xyz"])          # reverse: cache-site precision
    canon_snapped, canon_geo = geodesic_to_candidates(frag, xyz, node_segments, tree, kept,
                                                      canonical["xyz"], canonical["segment"])
    cache_snapped, cache_geo = geodesic_to_candidates(frag, xyz, node_segments, tree, kept,
                                                      cache_xyz, cache_segment)
    cand_xyz = xyz[cand_nodes]
    canon_to_cand_euclid = nearest(canonical["xyz"], cand_xyz)     # looser reference, any segment

    def share(values, r):
        return float(np.mean(np.asarray(values) <= r)) if len(values) else float("nan")

    result = {
        "brain": brain, "mcl": args.mcl, "cache": str(cache_path), "merge_table": table_dir,
        "canonical_sites_csv": canonical["path"], "radii_um": RADII_UM,
        "n_canonical": int(len(canonical["xyz"])), "n_cache_sites": int(len(cache_xyz)),
        "n_candidates": int(len(cand_nodes)),
        "canonical_unsnappable": int((~canon_snapped).sum()), "cache_unsnappable": int((~cache_snapped).sum()),
        "canonical_to_cache_site_um": [float(v) for v in canon_to_cache],
        "cache_site_to_canonical_um": [float(v) for v in cache_to_canon],
        "canonical_to_candidate_geodesic_um": [None if not np.isfinite(v) else float(v) for v in canon_geo],
        "cache_site_to_candidate_geodesic_um": [None if not np.isfinite(v) else float(v) for v in cache_geo],
        "canonical_to_candidate_euclid_um": [float(v) for v in canon_to_cand_euclid],
        "summary": {
            "cache_recall_at_50": share(canon_to_cache, MATCH_RADIUS_UM),
            "cache_precision_at_50": share(cache_to_canon, MATCH_RADIUS_UM),
            "candidate_recall_canonical_at_20": share(canon_geo, POSITIVE_RADIUS_UM),
            "candidate_recall_canonical_at_50": share(canon_geo, 50.0),
            "candidate_recall_canonical_at_150": share(canon_geo, CLAIM_RADIUS_UM),
            "candidate_reach_cache_at_20": share(cache_geo, POSITIVE_RADIUS_UM),
            "candidate_reach_cache_at_150": share(cache_geo, CLAIM_RADIUS_UM),
        },
        "seconds": time.monotonic() - started,
    }
    s = result["summary"]
    s["product_approximation_at_20"] = s["cache_recall_at_50"] * s["candidate_reach_cache_at_20"]
    print(f"[{brain}] cache recall@50 {s['cache_recall_at_50']:.3f} | candidate recall of canonical @20 "
          f"{s['candidate_recall_canonical_at_20']:.3f} @150 {s['candidate_recall_canonical_at_150']:.3f} | "
          f"reach of cache sites @20 {s['candidate_reach_cache_at_20']:.3f} (sweep check) | product approx "
          f"{s['product_approximation_at_20']:.3f} | unsnappable canonical {result['canonical_unsnappable']} "
          f"| {result['seconds']:.0f}s", flush=True)
    return result


# ----------------------------------------------------------------------------- figure
SERIES = [  # fixed categorical order (validated reference palette, light surface)
    ("cache", "Cache merge site within d of the canonical site (Euclidean)", "#2a78d6"),
    ("candidate", "Junction candidate within r of the canonical site (geodesic)", "#eb6834"),
    ("reach", "Junction candidate within r of a cache site (sweep measure)", "#1baf7a"),
]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e8e7e3", "#fcfcfb"


def curves(result):
    radii = np.asarray(result["radii_um"], float)

    def curve(values):
        v = np.asarray([np.inf if x is None else x for x in values], float)
        return np.array([np.mean(v <= r) if len(v) else np.nan for r in radii])

    return radii, {"cache": curve(result["canonical_to_cache_site_um"]),
                   "candidate": curve(result["canonical_to_candidate_geodesic_um"]),
                   "reach": curve(result["cache_site_to_candidate_geodesic_um"])}


def plot(results, fig_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    brains = sorted(results, key=lambda b: -results[b]["n_canonical"])
    n_panels = len(brains) + 1
    cols = 4
    rows = -(-n_panels // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.0 * cols, 3.1 * rows), sharex=True, sharey=True)
    fig.patch.set_facecolor(SURFACE)
    axes = np.atleast_2d(axes)

    def style(ax, title):
        ax.set_facecolor(SURFACE)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(colors=INK2, labelsize=8, length=0)
        ax.set_ylim(0, 1.02)
        ax.set_xlim(0, 155)
        ax.set_xticks([20, 50, 100, 150])
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["0", "25%", "50%", "75%", "100%"])
        ax.set_title(title, fontsize=9.5, color=INK, loc="left")
        for x, label in ((POSITIVE_RADIUS_UM, "positive 20"), (MATCH_RADIUS_UM, "match 50")):
            ax.axvline(x, color=GRID, linewidth=1.2, zorder=0)

    def draw(ax, radii, series, counts_text):
        for key, _, colour in SERIES:
            y = series[key]
            ax.plot(radii, y, color=colour, linewidth=2, linestyle="--" if key == "reach" else "-", zorder=3)
            mark_r = MATCH_RADIUS_UM if key == "cache" else POSITIVE_RADIUS_UM
            yi = float(np.interp(mark_r, radii, y))
            ax.plot([mark_r], [yi], marker="o", markersize=7, color=colour, markeredgecolor=SURFACE,
                    markeredgewidth=1.5, zorder=4)
            if key != "reach":
                ax.annotate(f"{yi:.0%}", (mark_r, yi), textcoords="offset points",
                            xytext=(6, -12 if key == "candidate" else 6), fontsize=8, color=INK)
        ax.text(0.98, 0.04, counts_text, transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5, color=INK2)

    for ax, brain in zip(axes.flat, brains):
        r = results[brain]
        radii, series = curves(r)
        style(ax, f"{brain}")
        draw(ax, radii, series, f"{r['n_canonical']} canonical, {r['n_cache_sites']} cache sites, "
                                f"{r['n_candidates']:,} candidates")
    # pooled panel: micro-average over all sites
    pooled = {k: [] for k in ("canonical_to_cache_site_um", "canonical_to_candidate_geodesic_um",
                              "cache_site_to_candidate_geodesic_um")}
    for r in results.values():
        for k in pooled:
            pooled[k].extend(r[k])
    pooled_result = {"radii_um": RADII_UM, **pooled}
    radii, series = curves(pooled_result)
    ax = axes.flat[len(brains)]
    style(ax, f"All {len(brains)} brains, pooled sites")
    draw(ax, radii, series, f"{len(pooled['canonical_to_cache_site_um'])} canonical, "
                            f"{len(pooled['cache_site_to_candidate_geodesic_um'])} cache sites")
    # direct labels on the pooled panel: short names, placed off the curves
    placements = {"cache": (100.0, -0.10, "cache sites"), "candidate": (100.0, -0.07, "candidates vs canonical"),
                  "reach": (60.0, 0.06, "candidates vs cache sites")}
    for key, _, colour in SERIES:
        x, dy, text = placements[key]
        y = float(np.interp(x, radii, series[key]))
        ax.text(x, y + dy, text, fontsize=7.5, color=INK, ha="center", va="center")
    for ax in axes.flat[n_panels:]:
        ax.set_visible(False)
    fig.supxlabel("radius in um: Euclidean d for the cache gap, geodesic cable r for the candidate gap", fontsize=9, color=INK2)
    for ax in axes[:, 0]:
        ax.set_ylabel("share of sites", fontsize=8.5, color=INK2)
    handles = [plt.Line2D([], [], color=c, linewidth=2, linestyle="--" if k == "reach" else "-") for k, _, c in SERIES]
    fig.legend(handles, [s[1] for s in SERIES], loc="upper center", ncol=3, frameon=False, fontsize=8.5,
               labelcolor=INK, bbox_to_anchor=(0.5, 0.995))
    fig.suptitle("Canonical merge sites: how many the cache labels hold, how many the junction candidate rule reaches",
                 fontsize=11, color=INK, y=1.03)
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    fig.savefig(fig_path, bbox_inches="tight", facecolor=SURFACE)
    print(f"figure -> {fig_path}")


def write_summary(results, out_dir: Path):
    rows = []
    for brain, r in sorted(results.items()):
        s = r["summary"]
        rows.append({"brain": brain, "canonical_sites": r["n_canonical"], "cache_sites": r["n_cache_sites"],
                     "merge_candidates": r["n_candidates"], "canonical_unsnappable": r["canonical_unsnappable"],
                     **{k: round(v, 4) for k, v in s.items()}})
    frame = pd.DataFrame(rows)
    pooled = {"brain": "pooled", "canonical_sites": int(frame["canonical_sites"].sum()),
              "cache_sites": int(frame["cache_sites"].sum()), "merge_candidates": int(frame["merge_candidates"].sum()),
              "canonical_unsnappable": int(frame["canonical_unsnappable"].sum())}
    allc = sum((r["canonical_to_cache_site_um"] for r in results.values()), [])
    allg = [np.inf if v is None else v for r in results.values() for v in r["canonical_to_candidate_geodesic_um"]]
    allr = [np.inf if v is None else v for r in results.values() for v in r["cache_site_to_candidate_geodesic_um"]]
    allp = sum((r["cache_site_to_canonical_um"] for r in results.values()), [])
    pooled.update(cache_recall_at_50=round(float(np.mean(np.asarray(allc) <= 50)), 4),
                  cache_precision_at_50=round(float(np.mean(np.asarray(allp) <= 50)), 4),
                  candidate_recall_canonical_at_20=round(float(np.mean(np.asarray(allg) <= 20)), 4),
                  candidate_recall_canonical_at_150=round(float(np.mean(np.asarray(allg) <= 150)), 4),
                  candidate_reach_cache_at_20=round(float(np.mean(np.asarray(allr) <= 20)), 4))
    frame = pd.concat([frame, pd.DataFrame([pooled])], ignore_index=True)
    frame.to_csv(out_dir / "summary.csv", index=False)
    print(frame.to_string(index=False))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--measure", action="store_true")
    p.add_argument("--plot", action="store_true")
    p.add_argument("--brains", nargs="+", default=DEFAULT_BRAINS)
    p.add_argument("--mcl", type=int, default=100)
    p.add_argument("--metrics-dir", type=Path, default=PROJECT_ROOT / "metrics_out")
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
        plot(results, args.out_dir / "merge_canonical_gap.svg")


if __name__ == "__main__":
    main()
