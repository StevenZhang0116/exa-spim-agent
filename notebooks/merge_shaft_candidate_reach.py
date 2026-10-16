"""Read-only: what a tip-to-shaft merge candidate family adds to the junction rule's reach.

Family J (the current merge pool): degree>=3 fragment nodes on a valid segment, same-segment
greedy NMS within 20 um Euclidean in order (-degree, node id), i.e. the frozen merge
detector's ``junction|nms=20`` enumeration (``merge_candidate_pool_sweep``).

Family S (tip-to-shaft): in the occurrences of the declared split pool
(``proofreader_evolve/candidate_pools.json``, ``tip_to_any_node|r=80|k=2``), every partner node
whose role is ``non_tip``: a fragment tip of another segment comes within the gap of this
segment's shaft there. A merge often leaves such a tip behind (the rest of the fused neuron
ends where it was fused into the other segment). Same-segment greedy NMS within 20 um
Euclidean in order (gap, node id), and a shaft node within 20 um (same segment, Euclidean)
of a kept junction is dropped, so S counts only rows J does not already have. Variants by
the largest tip gap allowed: 10, 20, 40, 80 um.

Per brain and family, all GT-blind except the measurement:
- candidate count;
- reach: cache ``gt_merge_sites`` and canonical sites (``metrics_out/<brain>/*/merge_sites.csv``),
  each snapped to the nearest node of its own segment within 50 um, then the geodesic cable
  distance to the nearest candidate; covered at 20 um (the positive-label radius), 30 um and
  150 um (the claim radius); every site's nearest distance per family is stored too, so other
  radii need no rerun;
- positives (candidates within 20 um of a snapped cache site) and GT-judgeable rows
  (``near_gt_150``: segment carries a GT node label and the candidate lies within 150 um of a
  GT node), hence the prevalence each family would bring to the benchmark;
- coverage of the census bridge sites (``merge_diagnosis/mcl100/followup/bridge_sites.csv``)
  by category.

Validation printed per brain: J must reproduce ``notebooks/merge_site_counts/mcl100_length_v1``
(candidates, positives, reach at 20 um).

Run with panda on a compute node, one process per group of brains (each loads one cache):
    python -u notebooks/merge_shaft_candidate_reach.py --brains 789202 754613
    python -u notebooks/merge_shaft_candidate_reach.py --summarize      # tables from all JSONs
Outputs: notebooks/merge_shaft_reach/<brain>_mcl<N>.json and summary.csv (rebuilt from every
per-brain JSON in the directory, never only from the brains of one invocation).
"""
from __future__ import annotations

import argparse
import gc
import glob
import heapq
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "notebooks"))

from merge_candidate_pool_sweep import enumerate_junctions, nms_keep_mask  # noqa: E402
from merge_canonical_gap import load_canonical  # noqa: E402
from merge_topology_diagnosis import build_fragment_index, load_payload  # noqa: E402

ALL_BRAINS = ["794495", "802449", "789202", "794493", "794491", "747807", "750318",
              "751473", "754610", "754612", "754613"]
NMS_UM = 20.0
SNAP_MAX_UM = 50.0
POSITIVE_UM = 20.0
CLAIM_UM = 150.0
REACH_RADII_UM = (POSITIVE_UM, 30.0, CLAIM_UM)
NEAR_GT_UM = 150.0
GAP_VARIANTS = (10.0, 20.0, 40.0, 80.0)
OUT_DIR = PROJECT_ROOT / "notebooks" / "merge_shaft_reach"
SITE_COUNTS = PROJECT_ROOT / "notebooks" / "merge_site_counts" / "mcl100_length_v1" / "summary.csv"
BRIDGE_SITES = PROJECT_ROOT / "notebooks" / "merge_diagnosis" / "mcl100" / "followup" / "bridge_sites.csv"


# ----------------------------------------------------------------------------- families
def split_shaft_nodes(payload: dict, mcl: int) -> tuple[np.ndarray, np.ndarray, str]:
    """Non-tip partner nodes of the declared split pool's occurrences, with their smallest gap."""
    from proofreader_evolve.cli import precompute_error_scores as pc
    from proofreader_evolve.harness import candidate_pools, native_pool
    from proofreader_evolve.harness.detector_metadata import inference_metadata

    record = pc.resolve_detector_runs(None, None)["split"][0]
    module = pc._import_module(Path(record["dir"]) / record["script"], f"shaft_reach_split_{record['script_sha256']}")
    policy = candidate_pools.load_pool_policies()["split"]
    rows, _, meta = native_pool.build_native_universe({**payload, **inference_metadata(payload, mcl)},
                                                      "split", module, policy)
    best: dict[int, float] = {}
    for row in rows:
        for occ in row["occurrences"]:
            partner = "b" if occ["anchor_side"] == "a" else "a"
            if occ[f"node_role_{partner}"] != "non_tip":
                continue
            node, gap = int(occ[f"node_id_{partner}"]), float(occ["gap_um"])
            if gap < best.get(node, math.inf):
                best[node] = gap
    nodes = np.fromiter(best.keys(), dtype=np.int64, count=len(best))
    gaps = np.fromiter(best.values(), dtype=float, count=len(best))
    return nodes, gaps, f"{meta['candidate_policy']} ({len(rows)} pairs)"


def shaft_keep_mask(index: dict, nodes: np.ndarray, gaps: np.ndarray, kept_junctions: np.ndarray,
                    max_gap: float) -> np.ndarray:
    """Boolean over all nodes: shaft candidates kept after junction priority and same-segment NMS."""
    kept = np.zeros(index["n_nodes"], dtype=bool)
    select = (gaps <= max_gap) & (index["node_segments"][nodes] > 0)
    nodes, gaps = nodes[select], gaps[select]
    if not len(nodes):
        return kept
    xyz, segments = index["xyz"], index["node_segments"]
    junction_nodes = np.flatnonzero(kept_junctions)
    if len(junction_nodes):
        near = cKDTree(xyz[junction_nodes]).query_ball_point(xyz[nodes], r=NMS_UM)
        covered = np.fromiter((any(segments[junction_nodes[j]] == segments[n] for j in hits)
                               for n, hits in zip(nodes, near)), dtype=bool, count=len(nodes))
        nodes, gaps = nodes[~covered], gaps[~covered]
    if not len(nodes):
        return kept
    order = np.lexsort((nodes, gaps))
    nodes = nodes[order]
    neighbours = cKDTree(xyz[nodes]).query_ball_point(xyz[nodes], r=NMS_UM)
    suppressed = np.zeros(len(nodes), dtype=bool)
    for i, node in enumerate(nodes):
        if suppressed[i]:
            continue
        kept[int(node)] = True
        for j in neighbours[i]:
            if segments[nodes[j]] == segments[node]:
                suppressed[j] = True
    return kept


# ----------------------------------------------------------------------------- measurement
def snap(index: dict, segment: int, point) -> int | None:
    hits = np.asarray(index["kdtree"].query_ball_point(np.asarray(point, float), r=SNAP_MAX_UM), dtype=np.int64)
    own = hits[index["node_segments"][hits] == int(segment)] if len(hits) else hits
    if not len(own):
        return None
    return int(own[int(np.argmin(np.linalg.norm(index["xyz"][own] - np.asarray(point, float), axis=1)))])


def geodesic_profile(index: dict, start: int, max_um: float, masks: dict) -> tuple[dict, dict]:
    """One bounded Dijkstra from ``start``: per mask, the nearest kept node's cable distance
    (inf beyond ``max_um``) and the set of kept nodes within POSITIVE_UM."""
    graph, xyz = index["graph"], index["xyz"]
    nearest = {name: math.inf for name in masks}
    within = {name: set() for name in masks}
    best = {start: 0.0}
    heap = [(0.0, start)]
    while heap:
        distance, node = heapq.heappop(heap)
        if distance > best.get(node, math.inf):
            continue
        for name, mask in masks.items():
            if mask[node]:
                if distance < nearest[name]:
                    nearest[name] = distance
                if distance <= POSITIVE_UM:
                    within[name].add(node)
        for neighbour in graph.neighbors(node):
            neighbour = int(neighbour)
            step = distance + float(np.linalg.norm(xyz[neighbour] - xyz[node]))
            if step <= max_um and step < best.get(neighbour, math.inf):
                best[neighbour] = step
                heapq.heappush(heap, (step, neighbour))
    return nearest, within


def measure_brain(brain: str, mcl: int, out_dir: Path) -> dict:
    started = time.monotonic()
    cache = PROJECT_ROOT / "cache" / f"dataset_cache_{brain}_mcl{mcl}_add.pkl"
    payload = load_payload(cache)
    index = build_fragment_index(payload["fragments_graph"])
    t = time.monotonic()
    kept_j = nms_keep_mask(index, enumerate_junctions(index), NMS_UM)
    shaft_nodes, shaft_gaps, split_pool = split_shaft_nodes(payload, mcl)
    masks = {"J": kept_j}
    for gap in GAP_VARIANTS:
        masks[f"S{gap:g}"] = shaft_keep_mask(index, shaft_nodes, shaft_gaps, kept_j, gap)
    for gap in GAP_VARIANTS:
        masks[f"J+S{gap:g}"] = kept_j | masks[f"S{gap:g}"]
    print(f"[{brain}] families built in {time.monotonic() - t:.0f}s from {split_pool}: "
          + ", ".join(f"{k} {int(m.sum())}" for k, m in masks.items() if not k.startswith("J+")), flush=True)

    # GT-judgeable rows (near_gt_150) and the GT segment set, for prevalence
    gt_segments = {int(x) for x in np.unique(np.asarray(payload["gt_node_canonical_label"])) if int(x) != 0}
    gt_tree = cKDTree(np.asarray(payload["gt_graph"].node_xyz, dtype=float))

    def judgeable(mask):
        nodes = np.flatnonzero(mask)
        on_gt = np.fromiter((int(s) in gt_segments for s in index["node_segments"][nodes]), bool, len(nodes))
        dist, _ = gt_tree.query(index["xyz"][nodes]) if len(nodes) else (np.empty(0), None)
        return set(nodes[on_gt & (np.asarray(dist) <= NEAR_GT_UM)].tolist())

    sites = list(payload.get("gt_merge_sites") or [])
    canonical = load_canonical(PROJECT_ROOT / "metrics_out", brain, payload["anisotropy"])
    site_rows, positives = [], {name: set() for name in masks}
    for kind, points, segments in (("cache", [s["xyz"] for s in sites], [int(s["segment_id"]) for s in sites]),
                                   ("canonical", canonical["xyz"], canonical["segment"])):
        for i, (point, segment) in enumerate(zip(points, segments)):
            start = snap(index, segment, point)
            if start is None:
                site_rows.append({"kind": kind, "site_index": i, "snapped": False,
                                  **{f"d_{k}": math.inf for k in masks}})
                continue
            nearest, within = geodesic_profile(index, start, CLAIM_UM, masks)
            if kind == "cache":
                for name in masks:
                    positives[name] |= within[name]
            site_rows.append({"kind": kind, "site_index": i, "snapped": True,
                              **{f"d_{k}": v for k, v in nearest.items()}})
    frame = pd.DataFrame(site_rows)

    families = {}
    for name, mask in masks.items():
        available = judgeable(mask)
        pos = positives[name]
        record = {"candidates": int(mask.sum()), "positives": len(pos),
                  "judgeable": len(available), "positives_judgeable": len(pos & available),
                  "prevalence": (len(pos & available) / len(available)) if available else float("nan")}
        for kind in ("cache", "canonical"):
            d = frame.loc[frame["kind"] == kind, f"d_{name}"].to_numpy(dtype=float)
            record[f"{kind}_sites"] = int(len(d))
            for radius in REACH_RADII_UM:
                record[f"{kind}_reach_{radius:g}"] = float(np.mean(d <= radius)) if len(d) else float("nan")
                record[f"{kind}_covered_{radius:g}"] = int(np.sum(d <= radius))
        families[name] = record

    # bridge-site categories of the census (cache sites only)
    bridge = {}
    if BRIDGE_SITES.exists():
        table = pd.read_csv(BRIDGE_SITES)
        table = table[table["brain"].astype(str) == brain]
        cache_frame = frame[frame["kind"] == "cache"].set_index("site_index")
        def category(row):
            if int(row["segment_n_junction_nodes"]) == 0:
                return "pure_path"
            return "junction_30_150" if np.isfinite(row["junction_um_extended"]) else "junction_beyond_150"
        table = table.assign(category=[category(row) for _, row in table.iterrows()])
        for cat, group in table.groupby("category"):
            idx = group["site_index"].astype(int)
            bridge[cat] = {"n": int(len(group)),
                           **{f"covered_20_{k}": int((cache_frame.loc[idx, f"d_{k}"] <= POSITIVE_UM).sum())
                              for k in ("J", "S80", "J+S80", "S20", "J+S20")}}

    # validation against the site-count run (J must reproduce it)
    check = {}
    if SITE_COUNTS.exists():
        counts = pd.read_csv(SITE_COUNTS)
        counts = counts[counts["brain_id"].astype(str) == brain]
        if len(counts):
            c = counts.iloc[0]
            check = {"candidates": [int(c["n_candidates"]), families["J"]["candidates"]],
                     "positives": [int(c["n_is_merge_site_1"]), families["J"]["positives"]],
                     "reach_20": [round(float(c["gt_site_recall_at_positive_radius"]), 4),
                                  round(families["J"]["cache_reach_20"], 4)]}
    # nearest-candidate cable distance of every site per family (None: none within CLAIM_UM / unsnappable)
    site_distances = {kind: {name: [None if not np.isfinite(v) else round(float(v), 3)
                                    for v in frame.loc[frame["kind"] == kind, f"d_{name}"]]
                             for name in masks}
                      for kind in ("cache", "canonical")}
    result = {"brain": brain, "mcl": mcl, "cache": str(cache), "split_pool": split_pool,
              "canonical_sites_csv": canonical["path"], "families": families, "bridge": bridge,
              "site_distances_um": site_distances,
              "validation_against_site_counts": check, "seconds": round(time.monotonic() - started, 1)}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{brain}_mcl{mcl}.json").write_text(json.dumps(result, indent=1, default=float))
    j, s, u = families["J"], families["S80"], families["J+S80"]
    print(f"[{brain}] validation (expected, measured): {check}", flush=True)
    print(f"[{brain}] J {j['candidates']} cand, reach@20 {j['cache_reach_20']:.3f} | S80 +{s['candidates']} cand, "
          f"prevalence {s['prevalence']:.4f} vs J {j['prevalence']:.4f} | J+S80 reach@20 {u['cache_reach_20']:.3f}, "
          f"canonical@20 {j['canonical_reach_20']:.3f} -> {u['canonical_reach_20']:.3f} | bridge {bridge} | "
          f"{result['seconds']:.0f}s", flush=True)
    del payload, index, masks
    gc.collect()
    return result


# ----------------------------------------------------------------------------- summary
def summarize(out_dir: Path, mcl: int) -> pd.DataFrame:
    rows = []
    for path in sorted(glob.glob(str(out_dir / f"*_mcl{mcl}.json"))):
        result = json.loads(Path(path).read_text())
        for name, record in result["families"].items():
            rows.append({"brain": result["brain"], "family": name, **record})
    frame = pd.DataFrame(rows)
    if frame.empty:
        print("no per-brain results yet")
        return frame
    pooled = []
    for name, group in frame.groupby("family", sort=False):
        total = {"brain": "pooled", "family": name}
        for column in ("candidates", "positives", "judgeable", "positives_judgeable", "cache_sites",
                       "canonical_sites", "cache_covered_20", "cache_covered_30", "cache_covered_150",
                       "canonical_covered_20", "canonical_covered_30", "canonical_covered_150"):
            total[column] = int(group[column].sum())
        total["prevalence"] = total["positives_judgeable"] / total["judgeable"] if total["judgeable"] else float("nan")
        for kind in ("cache", "canonical"):
            for radius in ("20", "30", "150"):
                total[f"{kind}_reach_{radius}"] = total[f"{kind}_covered_{radius}"] / total[f"{kind}_sites"]
                total[f"{kind}_worst_reach_{radius}"] = float(group[f"{kind}_reach_{radius}"].min())
        pooled.append(total)
    frame = pd.concat([frame, pd.DataFrame(pooled)], ignore_index=True)
    frame.to_csv(out_dir / "summary.csv", index=False)
    return frame


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--brains", nargs="+", default=ALL_BRAINS)
    parser.add_argument("--mcl", type=int, default=100)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--summarize", action="store_true", help="Only rebuild summary.csv from the per-brain JSONs")
    args = parser.parse_args(argv)
    if not args.summarize:
        for brain in args.brains:
            measure_brain(str(brain), args.mcl, args.out_dir)
    frame = summarize(args.out_dir, args.mcl)
    if not frame.empty:
        pooled = frame[frame["brain"] == "pooled"]
        print(pooled[["family", "candidates", "prevalence", "cache_reach_20", "cache_worst_reach_20",
                      "cache_reach_150", "cache_worst_reach_150", "canonical_reach_20"]].round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
