"""Read-only: how many GT merge sites sit at the end of a GT tracing.

The geometric walk that labels most merge sites (``canonical_labeling.geometric_merge_sites``,
a port of segmentation_skeleton_metrics' MergeCountMetric) starts at a fragment leaf more than
50 um from every GT node and walks inward until a fragment node comes within 6 um of a GT node;
that node is the site. It does not ask where on the GT the walk arrives. When a fragment runs
along GT neuron A and carries on past the end of A's tracing, the walk from its far leaf meets A
where the tracing ends: the site sits at a GT tip, the fragment has no junction there and
usually no foreign tip next to it. Whether the continuation is another neuron (a merge) or A
itself beyond the end of the tracing is not in the labels.

Per cache site (``gt_merge_sites``):
- gt_um: Euclidean distance to the nearest GT node;
- gt_tip_um: cable distance along the GT graph from that node to the nearest GT tip (degree 1),
  searched to 150 um (empty beyond);
- turn_deg (sites within 20 um of a GT tip): angle between the GT's direction into its tip
  (the GT node 20 um of cable back -> the tip) and the fragment's direction away from the GT
  (the snapped fragment node -> 20 um of cable further, on the side that ends farther from the
  GT); 0 = the fragment carries straight on past the end of the tracing;
- group, from the per-site distances of ``merge_shaft_candidate_reach.py``: ``junction``
  (a junction candidate within 20 um), ``tip_to_shaft`` (added by J+S20), ``blind_bridge``
  (a census bridge site neither reaches: the 199) and ``missed_branch`` (the rest).
Per brain, the base rate: the share of GT nodes within 10 / 20 um of cable of a GT tip.

Run with panda on a compute node, one process per group of brains (each loads one cache):
    python -u notebooks/merge_site_gt_tip.py --brains 794495 802449
    python -u notebooks/merge_site_gt_tip.py --summarize       # tables from all CSVs
Outputs: notebooks/merge_site_gt_tip/<brain>_mcl<N>.csv (one row per site) and
<brain>_mcl<N>.json (base rates); the summary is rebuilt from every CSV in the directory.
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

from merge_shaft_candidate_reach import snap  # noqa: E402
from merge_topology_diagnosis import build_fragment_index, load_payload  # noqa: E402

ALL_BRAINS = ["794495", "802449", "789202", "794493", "794491", "747807", "750318",
              "751473", "754610", "754612", "754613"]
OUT_DIR = PROJECT_ROOT / "notebooks" / "merge_site_gt_tip"
REACH_DIR = PROJECT_ROOT / "notebooks" / "merge_shaft_reach"
BRIDGE_SITES = PROJECT_ROOT / "notebooks" / "merge_diagnosis" / "mcl100" / "followup" / "bridge_sites.csv"
POSITIVE_UM = 20.0
TIP_SEARCH_UM = 150.0
DIRECTION_UM = 20.0
TIP_RADII_UM = (10.0, 20.0)
GROUPS = ("junction", "tip_to_shaft", "missed_branch", "blind_bridge")


def bounded_search(graph, xyz, start: int, max_um: float, blocked: int | None = None, stop=None):
    """Dijkstra from ``start`` up to ``max_um`` of cable, never entering ``blocked``. Returns the
    cable distances reached and the first node popped for which ``stop`` holds (or None)."""
    best = {start: 0.0}
    heap = [(0.0, start)]
    while heap:
        distance, node = heapq.heappop(heap)
        if distance > best[node]:
            continue
        if stop is not None and stop(node):
            return best, node
        for neighbour in graph.neighbors(node):
            neighbour = int(neighbour)
            if neighbour == blocked:
                continue
            step = distance + float(np.linalg.norm(xyz[neighbour] - xyz[node]))
            if step <= max_um and step < best.get(neighbour, math.inf):
                best[neighbour] = step
                heapq.heappush(heap, (step, neighbour))
    return best, None


def farthest(best: dict) -> int:
    return max(best, key=best.get)


def angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return math.nan
    return math.degrees(math.acos(float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))))


def tip_base_rate(gt, gt_xyz: np.ndarray, gt_degree: np.ndarray) -> dict:
    """Share of GT nodes within each radius of cable of a GT tip (multi-source search)."""
    radius = max(TIP_RADII_UM)
    best = {int(t): 0.0 for t in np.flatnonzero(gt_degree == 1)}
    heap = [(0.0, t) for t in best]
    heapq.heapify(heap)
    while heap:
        distance, node = heapq.heappop(heap)
        if distance > best[node]:
            continue
        for neighbour in gt.neighbors(node):
            neighbour = int(neighbour)
            step = distance + float(np.linalg.norm(gt_xyz[neighbour] - gt_xyz[node]))
            if step <= radius and step < best.get(neighbour, math.inf):
                best[neighbour] = step
                heapq.heappush(heap, (step, neighbour))
    values = np.fromiter(best.values(), float, len(best))
    return {"gt_nodes": int(len(gt_xyz)), "gt_tips": int(np.sum(gt_degree == 1)),
            **{f"within_{r:g}": float(np.sum(values <= r) / len(gt_xyz)) for r in TIP_RADII_UM}}


def site_groups(brain: str, mcl: int, n_sites: int) -> list[str]:
    result = json.loads((REACH_DIR / f"{brain}_mcl{mcl}.json").read_text())
    distances = result.get("site_distances_um", {}).get("cache")
    if distances is None or len(distances["J"]) != n_sites:
        raise SystemExit(f"{brain}: the reach JSON has no per-site distances for {n_sites} cache sites; "
                         f"rerun merge_shaft_candidate_reach.py --brains {brain}")
    bridge = pd.read_csv(BRIDGE_SITES, dtype={"brain": str})
    bridge = set(bridge.loc[bridge["brain"] == brain, "site_index"].astype(int))
    within = lambda d: d is not None and d <= POSITIVE_UM  # noqa: E731
    groups = []
    for i, (j, js) in enumerate(zip(distances["J"], distances["J+S20"])):
        groups.append("junction" if within(j) else "tip_to_shaft" if within(js)
                      else "blind_bridge" if i in bridge else "missed_branch")
    return groups


def measure_brain(brain: str, mcl: int, out_dir: Path) -> pd.DataFrame:
    started = time.monotonic()
    payload = load_payload(PROJECT_ROOT / "cache" / f"dataset_cache_{brain}_mcl{mcl}_add.pkl")
    sites = list(payload.get("gt_merge_sites") or [])
    groups = site_groups(brain, mcl, len(sites))
    index = build_fragment_index(payload["fragments_graph"])
    frag, frag_xyz = index["graph"], index["xyz"]
    gt = payload["gt_graph"]
    gt_xyz = np.asarray(gt.node_xyz, dtype=float)
    gt_degree = np.zeros(len(gt_xyz), dtype=np.int64)
    for node, degree in gt.degree():
        gt_degree[int(node)] = int(degree)
    gt_tree = cKDTree(gt_xyz)
    base = tip_base_rate(gt, gt_xyz, gt_degree)

    rows = []
    for i, site in enumerate(sites):
        point = np.asarray(site["xyz"], dtype=float)
        gt_um, g = gt_tree.query(point)
        best, tip = bounded_search(gt, gt_xyz, int(g), TIP_SEARCH_UM, stop=lambda n: gt_degree[n] == 1)
        tip_um = best[tip] if tip is not None else math.nan
        turn = math.nan
        start = snap(index, int(site["segment_id"]), point)
        if tip is not None and tip_um <= max(TIP_RADII_UM) and start is not None:
            back, _ = bounded_search(gt, gt_xyz, tip, DIRECTION_UM)
            into_tip = gt_xyz[tip] - gt_xyz[farthest(back)]
            ends = []
            for neighbour in frag.neighbors(start):
                neighbour = int(neighbour)
                step = float(np.linalg.norm(frag_xyz[neighbour] - frag_xyz[start]))
                reach, _ = bounded_search(frag, frag_xyz, neighbour, max(DIRECTION_UM - step, 0.0), blocked=start)
                end = farthest(reach)
                ends.append((float(gt_tree.query(frag_xyz[end])[0]), end))
            if ends:
                away = max(ends)[1]
                turn = angle_deg(into_tip, frag_xyz[away] - frag_xyz[start])
        sources = sorted(set(site.get("sources") or []) | ({site["source"]} if site.get("source") else set()))
        rows.append({"brain": brain, "site_index": i, "segment_id": int(site["segment_id"]),
                     "sources": "+".join(sources) or "geometric_walk", "gt_neuron": site.get("gt_neuron"),
                     "group": groups[i], "gt_um": round(float(gt_um), 3),
                     "gt_tip_um": round(tip_um, 3) if np.isfinite(tip_um) else math.nan,
                     "turn_deg": round(turn, 1) if np.isfinite(turn) else math.nan})
    frame = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_dir / f"{brain}_mcl{mcl}.csv", index=False)
    (out_dir / f"{brain}_mcl{mcl}.json").write_text(json.dumps(
        {"brain": brain, "mcl": mcl, "sites": len(frame), "base_rate": base,
         "seconds": round(time.monotonic() - started, 1)}, indent=1))
    print(f"[{brain}] {len(frame)} sites, GT tips {base['gt_tips']}, base rate "
          f"{base['within_10']:.3f} / {base['within_20']:.3f} | {time.monotonic() - started:.0f}s", flush=True)
    print(group_table(frame).to_string(), flush=True)
    del payload, index, gt, frag
    gc.collect()
    return frame


def group_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group in GROUPS + ("all",):
        part = frame if group == "all" else frame[frame["group"] == group]
        tip = part["gt_tip_um"]
        record = {"group": group, "sites": len(part)}
        for r in TIP_RADII_UM:
            n = int((tip <= r).sum())
            record[f"at_gt_tip_{r:g}"] = n
            record[f"share_{r:g}"] = round(n / len(part), 3) if len(part) else math.nan
        turns = part.loc[tip <= max(TIP_RADII_UM), "turn_deg"].dropna()
        record["turn_median_deg"] = round(float(turns.median()), 1) if len(turns) else math.nan
        record["turn_le_30"] = int((turns <= 30).sum())
        rows.append(record)
    return pd.DataFrame(rows).set_index("group")


def summarize(out_dir: Path, mcl: int) -> None:
    paths = sorted(glob.glob(str(out_dir / f"*_mcl{mcl}.csv")))
    if not paths:
        raise SystemExit(f"no per-brain CSVs under {out_dir}")
    frame = pd.concat([pd.read_csv(p, dtype={"brain": str}) for p in paths], ignore_index=True)
    table = group_table(frame)
    table.to_csv(out_dir / "summary.csv")
    print(f"pooled over {frame['brain'].nunique()} brains:\n{table.to_string()}")
    bases = [json.loads(Path(p).read_text())["base_rate"] for p in sorted(glob.glob(str(out_dir / f"*_mcl{mcl}.json")))]
    nodes = sum(b["gt_nodes"] for b in bases)
    print("GT nodes within 10 / 20 um of a GT tip (base rate, pooled): "
          + " / ".join(f"{sum(b[f'within_{r:g}'] * b['gt_nodes'] for b in bases) / nodes:.3f}" for r in TIP_RADII_UM))
    blind = frame[frame["group"] == "blind_bridge"]
    print("blind sites at a GT tip (<= 20 um) by brain:\n"
          + blind.groupby("brain")["gt_tip_um"].apply(lambda s: f"{int((s <= 20).sum())}/{len(s)}").to_string())


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--brains", nargs="+", default=ALL_BRAINS)
    parser.add_argument("--mcl", type=int, default=100)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--summarize", action="store_true", help="Only rebuild the pooled tables from the CSVs")
    args = parser.parse_args(argv)
    if not args.summarize:
        for brain in dict.fromkeys(map(str, args.brains)):
            measure_brain(brain, args.mcl, args.out_dir)
    summarize(args.out_dir, args.mcl)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
