#!/usr/bin/env python
"""Follow-ups to merge_topology_diagnosis: bridge dissection + multi-site labels.

(A) TRUE-BRIDGE DISSECTION -- every census site classified ``bridge`` at the
    walk bound gets:
      * an EXTENDED junction walk (150 um): finite = the 30-um bound was the
        artifact; inf = structurally junction-free at any useful scale;
      * ``segment_n_junction_nodes``: 0 = the whole segment is an unbranched
        path. (The "mcl pruning removed the arms" hypothesis was REFUTED by the
        mcl10 run: pure-path bridges are MORE prevalent at mcl10 (61/185) than
        at mcl100 (34/148), so they are real structure, not a pruning artifact.);
      * geodesic distance to the nearest ENDPOINT (degree 1): merges at
        fragment ends are a different geometry than mid-shaft necks;
      * segment size, to compare against branch-site segments.

(B) MULTI-SITE LABELS -- every gt_merge_label with >=2 census sites gets
    pairwise Euclidean (site xyz) and geodesic (snapped nodes) distances:
    close pairs (<20 um) are double-counted single events; far-apart pairs on
    the same component are chain merges; cross-component pairs mean the label's
    merges live on disconnected skeletons.

Reads merge_diagnosis/mcl<MCL>/per_brain/*_sites.csv plus the same _add caches
as the census; writes to merge_diagnosis/mcl<MCL>/followup/:
    bridge_sites.csv       one row per bridge site (extended columns)
    multisite_labels.csv   one row per label with >=2 sites
    multisite_pairs.csv    one row per same-label site pair
    followup_summary.json  per-brain + pooled headline numbers
    followup_figure.png    bridge dissection + multi-site structure

Usage (panda env, from the project root):
    python notebooks/merge_diagnosis_followup.py --mcl 100
    python notebooks/merge_diagnosis_followup.py --mcl 100 --plot-only
"""

from __future__ import annotations

import argparse
import gc
import heapq
import json
import time
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from merge_topology_diagnosis import (
    CACHE_DIR,
    OUTPUT_ROOT,
    THIS_FILE as CENSUS_FILE,
    atomic_to_csv,
    atomic_write_text,
    brain_id_from_cache,
    build_fragment_index,
    discover_add_caches,
    load_payload,
)

THIS_FILE = Path(__file__).resolve()
EXTENDED_WALK_UM = 150.0
PAIR_WALK_UM = 500.0
DUPLICATE_PAIR_UM = 20.0


def _walk_to(index: dict, start_node: int, max_um: float, is_target) -> float:
    """Bounded Dijkstra over cable length; inf when no target within max_um."""
    xyz = index["xyz"]
    graph = index["graph"]
    if is_target(start_node):
        return 0.0
    best: dict[int, float] = {int(start_node): 0.0}
    heap: list[tuple[float, int]] = [(0.0, int(start_node))]
    while heap:
        distance, node = heapq.heappop(heap)
        if distance > best.get(node, float("inf")):
            continue
        if is_target(node):
            return float(distance)
        for neighbor in graph.neighbors(node):
            neighbor = int(neighbor)
            candidate = distance + float(
                np.linalg.norm(xyz[neighbor] - xyz[node]))
            if candidate <= max_um and candidate < best.get(
                    neighbor, float("inf")):
                best[neighbor] = candidate
                heapq.heappush(heap, (candidate, neighbor))
    return float("inf")


def segment_stats(index: dict, segments: set[int]) -> dict[int, dict]:
    """Node count / max degree / junction count per segment, one sorted pass."""
    interest = np.asarray(sorted(int(s) for s in segments), dtype=np.int64)
    mask = np.isin(index["node_segments"], interest)
    segs = index["node_segments"][mask]
    degs = index["degrees"][mask]
    if len(segs) == 0:
        return {}
    order = np.argsort(segs, kind="stable")
    segs, degs = segs[order], degs[order]
    uniq, starts, counts = np.unique(segs, return_index=True, return_counts=True)
    max_deg = np.maximum.reduceat(degs, starts)
    n_junction = np.add.reduceat((degs >= 3).astype(np.int64), starts)
    return {
        int(s): {"n_nodes": int(c), "max_degree": int(m),
                 "n_junction_nodes": int(j)}
        for s, c, m, j in zip(uniq, counts, max_deg, n_junction)
    }


def analyze_brain(cache_path: Path, census_dir: Path) -> dict:
    brain = brain_id_from_cache(cache_path)
    sites_csv = census_dir / "per_brain" / f"{brain}_sites.csv"
    if not sites_csv.exists():
        raise FileNotFoundError(f"run the census first: missing {sites_csv}")
    sites = pd.read_csv(sites_csv)

    payload = load_payload(cache_path)
    index = build_fragment_index(payload["fragments_graph"])
    gt_sites = list(payload.get("gt_merge_sites") or [])
    degrees = index["degrees"]
    node_components = index["node_components"]

    seg_stats = segment_stats(index, set(int(s) for s in sites["segment_id"]))
    sites["segment_n_nodes"] = [
        seg_stats.get(int(s), {}).get("n_nodes", 0) for s in sites["segment_id"]]

    # (A) bridge dissection
    bridge_rows = []
    for row in sites.itertuples():
        if row.topology != "bridge" or int(row.snap_node) < 0:
            continue
        snap = int(row.snap_node)
        stats = seg_stats.get(int(row.segment_id), {})
        bridge_rows.append({
            "brain": brain,
            "site_index": int(row.site_index),
            "segment_id": int(row.segment_id),
            "degree_at_snap": int(row.degree_at_snap),
            "junction_um_extended": _walk_to(
                index, snap, EXTENDED_WALK_UM, lambda n: degrees[n] >= 3),
            "endpoint_um": _walk_to(
                index, snap, EXTENDED_WALK_UM, lambda n: degrees[n] == 1),
            "segment_n_nodes": stats.get("n_nodes", 0),
            "segment_max_degree": stats.get("max_degree", 0),
            "segment_n_junction_nodes": stats.get("n_junction_nodes", 0),
        })
    bridge_df = pd.DataFrame(bridge_rows)

    # (B) multi-site labels
    pair_rows, label_rows = [], []
    for segment_id, group in sites.groupby("segment_id"):
        if len(group) < 2:
            continue
        entries = []
        for row in group.itertuples():
            xyz = np.asarray(gt_sites[int(row.site_index)]["xyz"], dtype=float)
            entries.append((int(row.site_index), int(row.snap_node), xyz))
        euclids = []
        n_cross = 0
        for (ia, na, xa), (ib, nb, xb) in combinations(entries, 2):
            euclid = float(np.linalg.norm(xa - xb))
            same_component = bool(
                na >= 0 and nb >= 0
                and node_components[na] == node_components[nb])
            geodesic = (_walk_to(index, na, PAIR_WALK_UM, lambda n: n == nb)
                        if same_component else float("inf"))
            n_cross += int(not same_component)
            euclids.append(euclid)
            pair_rows.append({
                "brain": brain,
                "segment_id": int(segment_id),
                "site_a": ia,
                "site_b": ib,
                "euclid_um": euclid,
                "same_component": same_component,
                "geodesic_um": geodesic,
            })
        snapped_components = {node_components[n] for _, n, _ in entries if n >= 0}
        label_rows.append({
            "brain": brain,
            "segment_id": int(segment_id),
            "n_sites": int(len(group)),
            "n_components_spanned": len(snapped_components),
            "n_cross_component_pairs": n_cross,
            "min_pair_euclid_um": float(min(euclids)),
            "max_pair_euclid_um": float(max(euclids)),
        })

    branch_seg_nodes = sites.loc[sites["topology"] == "branch",
                                 "segment_n_nodes"]
    summary = {
        "brain": brain,
        "n_sites": int(len(sites)),
        "n_labels": int(sites["segment_id"].nunique()),
        # (A)
        "n_bridge_sites": int(len(bridge_df)),
        "bridge_junction_in_30_150um": (
            int(np.isfinite(bridge_df["junction_um_extended"]).sum())
            if len(bridge_df) else 0),
        "bridge_segment_pure_path": (
            int((bridge_df["segment_n_junction_nodes"] == 0).sum())
            if len(bridge_df) else 0),
        "bridge_degree_at_snap_counts": (
            {int(k): int(v) for k, v in
             bridge_df["degree_at_snap"].value_counts().items()}
            if len(bridge_df) else {}),
        "bridge_endpoint_um_median": (
            float(np.median(bridge_df["endpoint_um"].replace(
                np.inf, np.nan).dropna()))
            if len(bridge_df) and np.isfinite(bridge_df["endpoint_um"]).any()
            else float("nan")),
        "bridge_endpoint_within_10um": (
            int((bridge_df["endpoint_um"] <= 10).sum())
            if len(bridge_df) else 0),
        "bridge_segment_n_nodes_median": (
            float(bridge_df["segment_n_nodes"].median())
            if len(bridge_df) else float("nan")),
        "branch_segment_n_nodes_median": (
            float(branch_seg_nodes.median())
            if len(branch_seg_nodes) else float("nan")),
        # (B)
        "n_multisite_labels": int(len(label_rows)),
        "n_pairs": int(len(pair_rows)),
        "n_duplicate_suspect_pairs": int(sum(
            1 for p in pair_rows if p["euclid_um"] < DUPLICATE_PAIR_UM)),
        "n_cross_component_pairs": int(sum(
            1 for p in pair_rows if not p["same_component"])),
        "multisite_min_pair_euclid_median": (
            float(np.median([l["min_pair_euclid_um"] for l in label_rows]))
            if label_rows else float("nan")),
    }

    del payload, index
    gc.collect()
    return {
        "bridge": bridge_df,
        "pairs": pd.DataFrame(pair_rows),
        "labels": pd.DataFrame(label_rows),
        "summary": summary,
    }


def save_followup_figure(bridge: pd.DataFrame, pairs: pd.DataFrame,
                         labels: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # (A1) Where do bridge sites really stand? Extended-walk distance,
    # with the structurally junction-free mass shown as an "inf" bar.
    ax = axes[0, 0]
    if len(bridge):
        ext = bridge["junction_um_extended"].to_numpy()
        finite = ext[np.isfinite(ext)]
        bins = np.linspace(30, EXTENDED_WALK_UM, 25)
        ax.hist(finite, bins=bins, alpha=0.75,
                label=f"junction at 30\u2013{EXTENDED_WALK_UM:g} µm "
                      f"(n={len(finite)})")
        n_inf = int(np.isinf(ext).sum())
        ax.bar([EXTENDED_WALK_UM * 1.06], [n_inf],
               width=EXTENDED_WALK_UM * 0.05, color="crimson", alpha=0.8,
               label=f"none within {EXTENDED_WALK_UM:g} µm (n={n_inf})")
        ax.legend(fontsize=8)
    ax.set_xlabel("extended geodesic distance to junction (µm)")
    ax.set_ylabel("bridge sites")
    ax.set_title("(A1) Bridge sites: was 30 µm the artifact?")
    ax.grid(True, alpha=0.25)

    # (A2) Pure-path fragments vs segments that do have junctions elsewhere,
    # split by whether the merge sits at a fragment end.
    ax = axes[0, 1]
    if len(bridge):
        pure = bridge["segment_n_junction_nodes"] == 0
        at_end = bridge["endpoint_um"] <= 10
        groups = {
            "pure path\n@ end": (pure & at_end).sum(),
            "pure path\nmid-shaft": (pure & ~at_end).sum(),
            "has junctions\n@ end": (~pure & at_end).sum(),
            "has junctions\nmid-shaft": (~pure & ~at_end).sum(),
        }
        ax.bar(list(groups), [int(v) for v in groups.values()],
               color=["crimson", "salmon", "steelblue", "lightsteelblue"])
        for i, v in enumerate(groups.values()):
            ax.text(i, int(v), str(int(v)), ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("bridge sites")
    ax.set_title("(A2) Bridge anatomy: segment type × endpoint")
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(True, axis="y", alpha=0.25)

    # (B1) Same-label site-pair separation; the <20 µm mass is the
    # double-counting suspect population.
    ax = axes[1, 0]
    if len(pairs):
        euclid = pairs["euclid_um"].to_numpy()
        ax.hist(np.clip(euclid, 0, 500), bins=40, alpha=0.75)
        ax.axvline(DUPLICATE_PAIR_UM, color="crimson", ls=":",
                   label=f"duplicate suspect < {DUPLICATE_PAIR_UM:g} µm "
                         f"({(euclid < DUPLICATE_PAIR_UM).mean():.0%})")
        ax.legend(fontsize=8)
    ax.set_xlabel("pair Euclidean separation (µm, clipped at 500)")
    ax.set_ylabel("same-label site pairs")
    ax.set_title("(B1) Multi-site labels: pair separation")
    ax.grid(True, alpha=0.25)

    # (B2) Sites per label and how many skeleton components they span.
    ax = axes[1, 1]
    if len(labels):
        n_sites = labels["n_sites"].clip(upper=6)
        span = labels["n_components_spanned"].clip(upper=3)
        for spanned, color in ((1, "steelblue"), (2, "salmon"), (3, "crimson")):
            sub = n_sites[span == spanned]
            if len(sub):
                label = (f"{spanned} component" if spanned < 3
                         else "\u22653 components")
                ax.hist(sub, bins=np.arange(1.5, 8) - 0.0, alpha=0.7,
                        label=f"{label} (n={len(sub)})", color=color,
                        histtype="bar", stacked=False)
        ax.legend(fontsize=8)
    ax.set_xlabel("sites per label (clipped at 6)")
    ax.set_ylabel("labels")
    ax.set_title("(B2) Chain merges vs cross-component labels")
    ax.grid(True, axis="y", alpha=0.25)

    fig.suptitle("Merge-diagnosis follow-up — bridge dissection & "
                 "multi-site labels", fontsize=13)
    fig.tight_layout()
    temporary = output_path.with_name(output_path.name + ".tmp")
    fig.savefig(temporary, dpi=180, bbox_inches="tight", format="png")
    temporary.replace(output_path)
    plt.close(fig)


def replot_from_artifacts(followup_dir: Path) -> int:
    """Rebuild the follow-up figure from saved CSVs; no cache loading."""
    frames = {}
    for name in ("bridge_sites", "multisite_pairs", "multisite_labels"):
        path = followup_dir / f"{name}.csv"
        if not path.exists():
            raise FileNotFoundError(f"run the analysis first: missing {path}")
        try:
            frames[name] = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            frames[name] = pd.DataFrame()
    save_followup_figure(frames["bridge_sites"], frames["multisite_pairs"],
                         frames["multisite_labels"],
                         followup_dir / "followup_figure.png")
    print(f"figure: {followup_dir / 'followup_figure.png'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mcl", type=int, required=True)
    parser.add_argument("--brains", nargs="*", default=None)
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    parser.add_argument("--diagnosis-dir", type=Path, default=OUTPUT_ROOT,
                        help="census artifact root (default merge_diagnosis)")
    parser.add_argument("--plot-only", action="store_true",
                        help="rebuild the figure from saved followup CSVs "
                             "(no cache loading)")
    args = parser.parse_args(argv)

    census_dir = args.diagnosis_dir.resolve() / f"mcl{args.mcl}"
    followup_dir = census_dir / "followup"
    if args.plot_only:
        return replot_from_artifacts(followup_dir)
    caches = discover_add_caches(args.cache_dir.resolve(), args.mcl, args.brains)
    print(f"census inputs: {census_dir}")
    print(f"artifacts: {followup_dir}")

    started = time.monotonic()
    all_bridge, all_pairs, all_labels, summaries = [], [], [], []
    for cache in caches:
        result = analyze_brain(cache, census_dir)
        all_bridge.append(result["bridge"])
        all_pairs.append(result["pairs"])
        all_labels.append(result["labels"])
        summaries.append(result["summary"])
        print(f"{result['summary']['brain']}: done", flush=True)

    bridge = pd.concat(all_bridge, ignore_index=True) if all_bridge else pd.DataFrame()
    pairs = pd.concat(all_pairs, ignore_index=True) if all_pairs else pd.DataFrame()
    labels = pd.concat(all_labels, ignore_index=True) if all_labels else pd.DataFrame()
    atomic_to_csv(bridge, followup_dir / "bridge_sites.csv")
    atomic_to_csv(pairs, followup_dir / "multisite_pairs.csv")
    atomic_to_csv(labels, followup_dir / "multisite_labels.csv")
    save_followup_figure(bridge, pairs, labels,
                         followup_dir / "followup_figure.png")

    pooled = {
        "n_bridge_sites": int(len(bridge)),
        "bridge_junction_in_30_150um": (
            int(np.isfinite(bridge["junction_um_extended"]).sum())
            if len(bridge) else 0),
        "bridge_segment_pure_path": (
            int((bridge["segment_n_junction_nodes"] == 0).sum())
            if len(bridge) else 0),
        "bridge_endpoint_within_10um": (
            int((bridge["endpoint_um"] <= 10).sum()) if len(bridge) else 0),
        "n_multisite_labels": int(len(labels)),
        "n_pairs": int(len(pairs)),
        "n_duplicate_suspect_pairs": (
            int((pairs["euclid_um"] < DUPLICATE_PAIR_UM).sum())
            if len(pairs) else 0),
        "n_cross_component_pairs": (
            int((~pairs["same_component"]).sum()) if len(pairs) else 0),
    }
    atomic_write_text(followup_dir / "followup_summary.json", json.dumps({
        "mcl": args.mcl,
        "extended_walk_um": EXTENDED_WALK_UM,
        "pair_walk_um": PAIR_WALK_UM,
        "duplicate_pair_um": DUPLICATE_PAIR_UM,
        "elapsed_seconds": time.monotonic() - started,
        "census_script": str(CENSUS_FILE),
        "analysis_script": str(THIS_FILE),
        "pooled": pooled,
        "per_brain": summaries,
    }, indent=2) + "\n")

    print("\n(A) POOLED bridge dissection:")
    print(f"  bridge sites: {pooled['n_bridge_sites']}")
    if pooled["n_bridge_sites"]:
        n = pooled["n_bridge_sites"]
        print(f"  junction found at 30-150 um: "
              f"{pooled['bridge_junction_in_30_150um']} "
              f"({pooled['bridge_junction_in_30_150um'] / n:.0%})"
              f"  -> bound artifact, recoverable by a wider walk")
        print(f"  segment has NO junction anywhere: "
              f"{pooled['bridge_segment_pure_path']} "
              f"({pooled['bridge_segment_pure_path'] / n:.0%})"
              f"  -> pure-path fragments (mcl pruning suspect)")
        print(f"  endpoint within 10 um: {pooled['bridge_endpoint_within_10um']} "
              f"({pooled['bridge_endpoint_within_10um'] / n:.0%})"
              f"  -> merges at fragment ends")
    print("\n(B) POOLED multi-site labels:")
    print(f"  labels with >=2 sites: {pooled['n_multisite_labels']}, "
          f"pairs: {pooled['n_pairs']}")
    if pooled["n_pairs"]:
        print(f"  duplicate-suspect pairs (<{DUPLICATE_PAIR_UM:g} um): "
              f"{pooled['n_duplicate_suspect_pairs']} "
              f"({pooled['n_duplicate_suspect_pairs'] / pooled['n_pairs']:.0%})")
        print(f"  cross-component pairs: {pooled['n_cross_component_pairs']} "
              f"({pooled['n_cross_component_pairs'] / pooled['n_pairs']:.0%})")
    print(f"\nall outputs: {followup_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
