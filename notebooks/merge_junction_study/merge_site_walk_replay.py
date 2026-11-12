#!/usr/bin/env python3
"""Why does the dataset cache record no merge site where segmentation-skeleton-metrics does?

Background (``merge_canonical_site_labels.py``): 90 of the 537 canonical merge sites on the TRAIN
brains have candidates within 20 um that are all labeled 0, and for 62 of them the cache holds no
site within 150 um. The cache's sites come from ``canonical_labeling.geometric_merge_sites``, a
port of ``MergeCountMetric`` (``segmentation_skeleton_metrics.skeleton_metrics``). Reading the two
side by side shows two differences:

1. **One GT neuron at a time.** The metric loops over the GT neurons; for each it builds a
   KD-tree of that neuron alone, takes the fragments whose label appears on it, starts a walk at
   every fragment leaf more than 50 um from *that neuron*, and stops at the first node within
   6 um of *that neuron* (the visited set is new for every neuron and fragment). The port uses
   one KD-tree of all GT neurons in a single pass: a leaf near any other traced neuron starts no
   walk, and a walk stops at whatever tracing it meets first.
2. **Site moved to a branching node.** The metric moves an accepted site to the nearest fragment
   node of degree > 2 within 40 um of cable that lies within 16 um of the neuron
   (``graph_util.search_branching_node``); the port keeps the node where the walk stopped.

This script replays the metric's procedure on the cache's own graphs, with and without the move,
and compares the result with the canonical sites and with the cache's sites.

``sites``
    Per brain, three site lists computed on the cache: ``per_neuron`` (difference 1 applied),
    ``per_neuron_moved`` (both applied) and, as a check of the replay itself, ``all_neurons``
    (the port's structure: one tree, one pass), which should reproduce the cache's
    ``geometric_walk`` sites. The pass-through test uses the cache's own threshold
    (``gt_label_policy.passthru_min_cc``); sites within 30 um are de-duplicated in the order found.

``summarize``
    Per brain: the size of every list, the share of canonical sites with a site of the list
    within 20 um and 50 um (Euclidean), and the share of the list's sites with a canonical site
    within 50 um.

Run on a compute node in the ``panda`` environment from the repository root::

    python -u notebooks/merge_junction_study/merge_site_walk_replay.py sites --brains 747807 802449
    python -u notebooks/merge_junction_study/merge_site_walk_replay.py summarize

Outputs under ``notebooks/merge_junction_study/merge_site_walk_replay/`` (gitignored):
``sites_<brain>.npz``, ``summary.md``. Read-only for caches and ``metrics_out``.

Limits. TRAIN brains only. The metric ran on its own fragment and GT graphs (other node
spacing, no minimum cable length), and the order in which leaves and neighbours are visited
decides which contact a walk records, so the replay cannot match site for site.
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for entry in (str(ROOT), str(ROOT / "notebooks")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from merge_junction_fragment_context import TRAIN_BRAINS, log  # noqa: E402
from merge_junction_label_localization import same_label_run, step  # noqa: E402

OUT = Path(__file__).resolve().parent / "merge_site_walk_replay"
AWAY_UM, APPROACH_UM, DEDUP_UM = 50.0, 6.0, 30.0   # MergeCountMetric / canonical_labeling
MOVE_RADIUS_UM, MOVE_GT_UM = 40.0, 16.0            # graph_util.search_branching_node
LISTS = ["canonical", "cache", "cache_geometric", "all_neurons", "per_neuron", "per_neuron_moved"]


def walks(graph, xyz, gt, gt_label, components, distance, nearest, passthru, move):
    """The metric's ``search_for_merges`` over ``components`` (``{segment: [node lists]}``) against
    the GT nodes ``distance`` / ``nearest`` were measured to. Returns ``[(node, segment)]``."""
    found = []
    for segment_id, parts in components.items():
        for nodes in parts:
            visited = set()
            for leaf in nodes:
                if len(graph.adj[leaf]) != 1 or leaf in visited:
                    continue
                visited.add(leaf)
                if not distance[leaf] > AWAY_UM:
                    continue
                queue = deque([leaf])
                while queue:
                    node = queue.pop()
                    if distance[node] < APPROACH_UM:
                        if same_label_run(gt, gt_label, int(nearest[node]), segment_id, cap=passthru) >= passthru:
                            found.append((branching_node(graph, xyz, distance, node) if move else node, segment_id))
                        break
                    for following in graph.adj[node]:
                        following = int(following)
                        if following not in visited:
                            queue.append(following)
                            visited.add(following)
    return found


def branching_node(graph, xyz, distance, root):
    queue, visited = deque([(root, 0.0)]), {root}
    while queue:
        node, travelled = queue.popleft()
        if len(graph.adj[node]) > 2 and distance[node] < MOVE_GT_UM:
            return node
        for following in graph.adj[node]:
            following = int(following)
            reached = travelled + step(xyz, following, node)
            if following not in visited and reached < MOVE_RADIUS_UM:
                queue.append((following, reached))
                visited.add(following)
    return root


def dedup(points):
    from scipy.spatial import cKDTree
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if not len(points):
        return points
    tree, removed = cKDTree(points), set()
    for index in range(len(points)):
        if index not in removed:
            removed |= {j for j in tree.query_ball_point(points[index], DEDUP_UM) if j != index}
    return points[[i for i in range(len(points)) if i not in removed]]


def brain_sites(brain):
    from scipy.spatial import cKDTree
    from candidate_rules import segment_of_component
    from merge_canonical_gap import load_canonical
    from proofreader_evolve.harness.dataset import load_cached_graphs
    from proofreader_evolve.harness.native_pool import cache_path

    started = time.monotonic()
    graph, gt, payload = load_cached_graphs(str(cache_path(brain, 100)), expect_brain=brain, expect_mcl=100)
    xyz = np.asarray(graph.node_xyz, dtype=np.float64)
    component = np.asarray(graph.node_component_id, dtype=np.int64)
    to_segment = np.full(int(component.max()) + 1, -1, dtype=np.int64)
    for key, value in segment_of_component(graph).items():
        to_segment[int(key)] = int(value)
    gt_xyz = np.asarray(gt.node_xyz, dtype=np.float64)
    gt_neuron = np.asarray(gt.node_component_id, dtype=np.int64)
    gt_label = np.asarray(payload["gt_node_canonical_label"], dtype=np.int64)
    passthru = int(payload["gt_label_policy"]["passthru_min_cc"])
    on_gt = np.unique(gt_label[gt_label != 0])
    kept = np.flatnonzero(np.isin(to_segment[component], on_gt))
    order = kept[np.argsort(component[kept], kind="stable")]
    bounds = np.flatnonzero(np.diff(component[order])) + 1
    components = {}
    for nodes in np.split(order, bounds):
        components.setdefault(int(to_segment[component[nodes[0]]]), []).append(nodes.tolist())
    log(f"[{brain}] {len(kept)} fragment nodes in {sum(len(v) for v in components.values())} fragments of "
        f"{len(components)} segments on GT; {len(np.unique(gt_neuron))} GT neurons; pass-through {passthru} "
        f"[{time.monotonic() - started:.0f}s]")

    distance = np.full(len(xyz), np.inf)
    nearest = np.full(len(xyz), -1, dtype=np.int64)
    distance[kept], nearest[kept] = cKDTree(gt_xyz).query(xyz[kept], workers=8)
    lists = {"all_neurons": walks(graph, xyz, gt, gt_label, components, distance, nearest, passthru, move=False)}
    per_neuron = {False: [], True: []}
    for neuron in np.unique(gt_neuron):
        members = np.flatnonzero(gt_neuron == neuron)
        labels = set(np.unique(gt_label[members]).tolist()) - {0}
        chosen = {segment: components[segment] for segment in labels if segment in components}
        nodes = np.array([n for parts in chosen.values() for part in parts for n in part], dtype=np.int64)
        if not len(nodes):
            continue
        near, index = cKDTree(gt_xyz[members]).query(xyz[nodes], workers=8)
        distance[nodes], nearest[nodes] = near, members[index]
        for move in (False, True):
            per_neuron[move] += walks(graph, xyz, gt, gt_label, chosen, distance, nearest, passthru, move)
    lists["per_neuron"], lists["per_neuron_moved"] = per_neuron[False], per_neuron[True]
    canonical = load_canonical(ROOT / "metrics_out", brain, payload["anisotropy"])
    cache_sites = list(payload.get("gt_merge_sites") or [])
    arrays = {name: dedup(xyz[[node for node, _ in found]]) for name, found in lists.items()}
    arrays["canonical"] = np.asarray(canonical["xyz"], dtype=np.float64)
    arrays["cache"] = np.array([s["xyz"] for s in cache_sites], dtype=np.float64).reshape(-1, 3)
    arrays["cache_geometric"] = np.array([s["xyz"] for s in cache_sites if s.get("source", "geometric_walk")
                                          == "geometric_walk"], dtype=np.float64).reshape(-1, 3)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"sites_{brain}.npz", **arrays)
    log(f"[{brain}] sites: " + ", ".join(f"{name} {len(arrays[name])}" for name in LISTS)
        + f" [{time.monotonic() - started:.0f}s]")


def share(points_a, points_b, radius):
    from scipy.spatial import cKDTree
    if not len(points_a) or not len(points_b):
        return 0
    return int((cKDTree(points_b).query(points_a)[0] <= radius).sum())


def summarize(args):
    lines = ["# Canonical merge sites against the cache's sites and a replay of the metric on the cache", "",
             "Script `notebooks/merge_junction_study/merge_site_walk_replay.py`. `cache`: all sites in the cache; "
             "`cache_geometric`: its geometric-walk sites; `all_neurons`: replay with one KD-tree of all GT neurons "
             "(the port's structure, a check of the replay); `per_neuron`: one GT neuron at a time as in the metric; "
             "`per_neuron_moved`: also moved to a branching node as in the metric. Distances Euclidean.", ""]
    header = ["brain", "canonical sites"]
    for name in LISTS[1:]:
        header += [f"{name}: sites", "canonical within 20 um", "within 50 um", "of its sites, canonical within 50 um"]
    body, totals = [], None
    for brain in args.brains:
        stored = np.load(OUT / f"sites_{brain}.npz", allow_pickle=False)
        canonical = stored["canonical"]
        row = [len(canonical)]
        for name in LISTS[1:]:
            points = stored[name]
            row += [len(points), share(canonical, points, 20.0), share(canonical, points, 50.0),
                    share(points, canonical, 50.0)]
        totals = np.array(row) if totals is None else totals + np.array(row)
        body.append([brain, *row])
    body.append(["all", *totals.tolist()])
    formatted = []
    for row in body:
        cells, canonical = [row[0], row[1]], row[1]
        for start in range(2, len(row), 4):
            sites, at20, at50, precise = row[start:start + 4]
            cells += [sites, f"{at20} ({100 * at20 / canonical:.0f}%)", f"{at50} ({100 * at50 / canonical:.0f}%)",
                      f"{precise} ({100 * precise / max(sites, 1):.0f}%)"]
        formatted.append(cells)
    lines += ["| " + " | ".join(header) + " |", "|" + "---|" * len(header),
              *["| " + " | ".join(str(c) for c in row) + " |" for row in formatted], ""]
    (OUT / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="stage", required=True)
    p = sub.add_parser("sites", help="replay the metric's walk on the cache graphs")
    p.add_argument("--brains", nargs="+", default=TRAIN_BRAINS, help="TRAIN brains only")
    p = sub.add_parser("summarize", help="compare the site lists")
    p.add_argument("--brains", nargs="+", default=TRAIN_BRAINS)
    args = parser.parse_args()
    unknown = set(args.brains) - set(TRAIN_BRAINS)
    if unknown:
        parser.error(f"Not TRAIN brains: {sorted(unknown)}; this analysis must not open validation or reserved brains")
    if args.stage == "sites":
        for brain in args.brains:
            brain_sites(brain)
    else:
        summarize(args)


if __name__ == "__main__":
    main()
