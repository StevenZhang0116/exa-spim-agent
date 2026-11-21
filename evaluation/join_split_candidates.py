"""Join fragments of the cache's graph at split candidates, after the merge cuts, and write the
result for the capsule.

The split stage works on the graph that the merge stage left: the merge cuts are applied first
(``cut_merge_sites.py``, same options), then the split candidates are enumerated again on the
cut graph with the declared rule (``candidate_pools.json``, ``candidate_rules.pair_occurrences``),
then chosen candidates are joined by an edge between their two nodes.

A cut fragment is several pieces of one segment, so a candidate here is a pair of pieces, not
a pair of segments: the occurrences of the rule are grouped by the two pieces they connect and
the closest occurrence stands for the pair. By default the rule runs on segment IDs as it does
for the pools, so two pieces of one segment are never paired and no join crosses a cut;
``--across-cuts`` gives every piece of a cut fragment its own ID. Anchors are the tips of the
listed segments; every node of the graph is a partner.

Which candidates are joined, ``--joins``:

* ``oracle``: the pairs that ground truth puts next to each other on one neuron. It measures
  what the candidates can reach at best; no policy takes part. Two readings, ``--truth``:

  * ``labels``: the repository's split truth (``candidate_rules.split_truth``: the two sides of
    a split edge of the cache, or two flanks of an unlabeled stretch) carried to pieces. A
    ground-truth node keeps the segment the cache gives it; a segment that the cache has no
    fragment of is a label nothing can be joined to.
  * ``graph``: the same rule read on the fragments that exist. A ground-truth node belongs to
    the nearest piece of its segment within 20 um, as in the capsule, and to nothing when the
    cache has no fragment of the segment, so two fragments on either side of a missing short
    one are a pair. Every edge between two pieces counts, whatever class the cache gave it.

  Neither reading asks how long a piece follows the neuron: a fragment that touches it at a few
  ground-truth nodes and runs on elsewhere is a true partner, and joining it makes a merge in
  the capsule's count. ``--min-support-um 50`` drops the pairs in which a piece labels less than
  50 um of the neuron's ground truth, the length the capsule's merge count
  (``MergeCountMetric.min_intersection_length``) asks before a fragment counts as on the neuron.

Joins are made closest first; a join between two pieces that earlier joins already connect is
skipped. The cache is not changed.

Run on a compute node in panda, from the repository root:

    python -u evaluation/join_split_candidates.py --brain 841260 --merge-rule remove --radius-um 12 \\
        --joins oracle --truth graph --segments evaluation/results/<run>/segments_under_gt.txt \\
        --out evaluation/results/<experiment>/<name>/fragments.zip

Written beside ``<out>``: ``<out>.json`` (counts), ``<out stem>_joins.csv`` and
``<out stem>_connections.txt``, the joined segment pairs in the form the capsule reads. Evaluate
with ``run_capsule_original.py --fragments <zip> --connections <txt>`` in arborist_eval.
"""
import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import networkx as nx
import numpy as np
from scipy.spatial import cKDTree

from cut_merge_sites import ARBORIST_RADIUS_UM, apply_cuts, oracle_sites
from export_cache_fragments import REPO, cache_record, load_cache, read_segments, segment_of, write_fragments

GT_NODE_TO_PIECE_UM = 20.0          # the capsule's limit when it gives a ground-truth node to a fragment


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--brain", required=True)
    parser.add_argument("--mcl", type=int, default=100)
    parser.add_argument("--segments", required=True, type=Path, help="Text file, one segment ID per line")
    parser.add_argument("--merge-rule", required=True, choices=("none", "remove", "detach"),
                        help="Merge cuts applied first, at the oracle sites of cut_merge_sites.py")
    parser.add_argument("--radius-um", type=float, default=ARBORIST_RADIUS_UM)
    parser.add_argument("--arm-um", type=float, default=30.0)
    parser.add_argument("--joins", required=True, choices=("oracle",))
    parser.add_argument("--truth", required=True, choices=("labels", "graph"))
    parser.add_argument("--min-support-um", type=float, default=0.0,
                        help="A pair is true only when each piece labels at least this length of the neuron's "
                             "ground truth (default 0; 50 is the length the capsule's merge count asks of a "
                             "fragment before it takes it as following the neuron)")
    parser.add_argument("--across-cuts", action="store_true",
                        help="Let two pieces of one cut fragment be paired (default: never)")
    parser.add_argument("--out", required=True, type=Path, help="ZIP file to write")
    return parser.parse_args()


def node_state(graph, sites):
    """Per node of the cache: segment, whether the edited graph still has it, its degree there,
    and its connected piece (0 for a deleted node). Pieces of untouched fragments are numbered
    from the cache's components; the pieces of a cut fragment get new numbers."""
    component_of = np.asarray(graph.node_component_id, dtype=np.int64)
    if component_of.min() < 0:
        sys.exit("negative component id in the cache")
    component_ids, inverse = np.unique(component_of, return_inverse=True)
    names = graph.component_id_to_swc_id
    segment = np.array([int(segment_of(names[c])) for c in component_ids.tolist()], dtype=np.int64)[inverse]
    present = np.zeros(len(component_of), dtype=bool)
    degree = np.zeros(len(component_of), dtype=np.int64)
    nodes = np.fromiter(graph.nodes, dtype=np.int64, count=graph.number_of_nodes())
    degree[nodes] = np.fromiter((len(neighbors) for _, neighbors in graph.adjacency()), dtype=np.int64,
                                count=graph.number_of_nodes())
    present[nodes] = True

    piece = component_of + 1
    cut = np.zeros(len(component_of), dtype=bool)
    following = int(piece.max()) + 1
    for component in sorted({int(component_of[row["node_id"]]) for row in sites}):
        members = np.flatnonzero((component_of == component) & present)
        cut[members] = True
        for part in nx.connected_components(nx.Graph.subgraph(graph, members.tolist())):
            piece[np.fromiter(part, dtype=np.int64, count=len(part))] = following
            following += 1
    piece[~present] = 0
    return component_of, segment, present, degree, piece, cut


def ground_truth_pieces(payload, xyz, segment, present, piece, truth):
    """The piece each ground-truth node belongs to (0 = none; a negative number stands for a
    segment without a fragment under ``labels``)."""
    gt_xyz = np.asarray(payload["gt_graph"].node_xyz, dtype=float)
    canonical = np.asarray(payload["gt_node_canonical_label"]).astype(np.int64)
    gt_piece = np.zeros(len(canonical), dtype=np.int64)
    labels = np.unique(canonical[canonical != 0])
    held = np.flatnonzero(present & np.isin(segment, labels))
    held = held[np.argsort(segment[held], kind="stable")]
    starts = np.searchsorted(segment[held], labels, side="left")
    ends = np.searchsorted(segment[held], labels, side="right")
    by_label = np.argsort(canonical, kind="stable")
    first = np.searchsorted(canonical[by_label], labels, side="left")
    last = np.searchsorted(canonical[by_label], labels, side="right")
    counts = Counter()
    for label, start, end, a, b in zip(labels.tolist(), starts.tolist(), ends.tolist(), first.tolist(), last.tolist()):
        gt_nodes = by_label[a:b]
        if start == end:                                   # the cache has no fragment of this segment
            gt_piece[gt_nodes] = -label if truth == "labels" else 0
            counts["gt_nodes_on_segments_without_fragment"] += len(gt_nodes)
            continue
        fragment_nodes = held[start:end]
        distance, hit = cKDTree(xyz[fragment_nodes]).query(gt_xyz[gt_nodes])
        pieces = piece[fragment_nodes[hit]]
        far = distance > GT_NODE_TO_PIECE_UM
        counts["gt_nodes_farther_than_limit_from_their_segment"] += int(far.sum())
        gt_piece[gt_nodes] = np.where(far, 0, pieces) if truth == "graph" else pieces
        counts["gt_nodes_on_a_piece"] += int(len(gt_nodes) - (far.sum() if truth == "graph" else 0))
    counts["gt_nodes_unlabeled"] = int((canonical == 0).sum())
    return canonical, gt_piece, dict(counts)


def piece_truth(rules, payload, canonical, gt_piece, truth, min_support_um):
    """``{(piece_a, piece_b): kind}``, the repository's split truth on pieces, and how many pairs
    the support condition dropped. The support of a piece on a neuron is the length of that
    neuron's ground truth it labels (nodes times the median edge)."""
    gt = payload["gt_graph"]
    edges = np.array(list(gt.edges()), dtype=np.int64)
    u, v = gt_piece[edges[:, 0]], gt_piece[edges[:, 1]]
    differ = (u != v) & (u != 0) & (v != 0)
    if truth == "graph":
        error = differ.astype(np.uint8)
    else:                                                  # the cache's classes, plus the splits a cut made
        error = np.asarray(payload["gt_edge_error"]).copy()
        error[differ & (canonical[edges[:, 0]] == canonical[edges[:, 1]])] = 1

    neuron_of = rules.gt_neuron_membership(gt)
    neuron = np.fromiter((neuron_of[node] for node in range(len(gt_piece))), dtype=np.int64, count=len(gt_piece))
    gt_xyz = np.asarray(gt.node_xyz, dtype=float)
    spacing = float(np.median(np.linalg.norm(gt_xyz[edges[:, 0]] - gt_xyz[edges[:, 1]], axis=1)))
    on_piece = gt_piece > 0
    keys, counts = np.unique(np.stack([neuron[on_piece], gt_piece[on_piece]], axis=1), axis=0, return_counts=True)
    support = {(int(n), int(p)): count * spacing for (n, p), count in zip(keys.tolist(), counts.tolist())}

    pairs, dropped = {}, 0
    for (neuron_id, pair), kind in rules.split_truth({"gt_graph": gt, "gt_node_canonical_label": gt_piece,
                                                      "gt_edge_error": error}).items():
        if min(support.get((neuron_id, pair[0]), 0.0), support.get((neuron_id, pair[1]), 0.0)) < min_support_um:
            dropped += 1
            continue
        pairs[pair] = kind if pairs.get(pair) in (None, kind) else "both"
    return pairs, dropped


def find(parent, item):
    while parent.setdefault(item, item) != item:
        parent[item] = parent[parent[item]]
        item = parent[item]
    return item


def main():
    options = parse_args()
    sys.path.insert(0, str(REPO))
    from proofreader_evolve.harness import candidate_pools

    cache, payload = load_cache(options.brain, options.mcl)
    graph = payload["fragments_graph"]
    wanted = read_segments(options.segments)
    sites, merge_record, edit = [], {}, {"rule": "none"}
    if options.merge_rule != "none":
        sites, merge_record = oracle_sites(options.brain, cache, graph, wanted)
        _, edit = apply_cuts(graph, sites, options.merge_rule, options.radius_um, options.arm_um)

    component_of, segment, present, degree, piece, cut = node_state(graph, sites)
    xyz = np.asarray(graph.node_xyz, dtype=float)
    listed = np.isin(segment, np.array(sorted(int(item) for item in wanted), dtype=np.int64))
    print(f"edited graph: {int(present.sum())} nodes, {len(np.unique(piece[cut]))} pieces of "
          f"{len(np.unique(component_of[cut]))} cut fragments", flush=True)

    # The declared rule on the edited graph. Nodes of other segments stay partners but are no anchors.
    rule = candidate_pools.load_pool_policies()["split"]
    rules = candidate_pools.rules()
    group = segment.copy()
    if options.across_cuts:
        group[cut] = int(segment.max()) + piece[cut]
    group[~present] = -1
    anchor_degree = np.where(listed & present, degree, 2)
    grouped = rules.pair_occurrences(xyz, group, anchor_degree, rule["mode"], rule["radius_um"],
                                     rule["per_anchor_k"]) or {}
    candidates, occurrences = {}, 0
    for rows in grouped.values():
        for row in rows:
            occurrences += 1
            pair = tuple(sorted((int(piece[row["node_id_a"]]), int(piece[row["node_id_b"]]))))
            if pair[0] != pair[1] and (pair not in candidates or row["gap_um"] < candidates[pair]["gap_um"]):
                candidates[pair] = row
    print(f"split rule {rule['config_id']}: {int(((anchor_degree == 1) & (group > 0)).sum())} anchors, "
          f"{occurrences} occurrences, {len(candidates)} pairs of pieces", flush=True)

    canonical, gt_piece, gt_counts = ground_truth_pieces(payload, xyz, segment, present, piece, options.truth)
    truth, dropped = piece_truth(rules, payload, canonical, gt_piece, options.truth, options.min_support_um)
    reachable = {pair for pair in truth if pair[0] > 0 and pair[1] > 0}
    chosen = sorted((pair for pair in candidates if pair in truth), key=lambda pair: (candidates[pair]["gap_um"], pair))

    names, parent, joins, loops = graph.component_id_to_swc_id, {}, [], 0
    for pair in chosen:
        row = candidates[pair]
        a, b = find(parent, pair[0]), find(parent, pair[1])
        if a == b:
            loops += 1
            continue
        parent[a] = b
        nx.Graph.add_edge(graph, row["node_id_a"], row["node_id_b"])
        joins.append({"node_id_a": row["node_id_a"], "node_id_b": row["node_id_b"], "gap_um": round(row["gap_um"], 3),
                      "swc_a": names[int(component_of[row["node_id_a"]])],
                      "swc_b": names[int(component_of[row["node_id_b"]])],
                      "same_segment": int(segment[row["node_id_a"]] == segment[row["node_id_b"]]),
                      "truth": truth[pair], "x_um": round(float(xyz[row["node_id_a"], 0]), 1),
                      "y_um": round(float(xyz[row["node_id_a"], 1]), 1), "z_um": round(float(xyz[row["node_id_a"], 2]), 1)})

    summary = {
        **cache_record(options.brain, cache, payload), **merge_record, "merge_edit": edit,
        "split_rule": rule["config_id"], "across_cuts": options.across_cuts,
        "truth": {"reading": options.truth, "min_support_um": options.min_support_um,
                  "neuron_pairs_dropped_for_support": dropped, **gt_counts, "pairs": len(truth),
                  "pairs_between_existing_pieces": len(reachable), "by_kind": dict(Counter(truth.values()))},
        "candidates": {"occurrences": occurrences, "pairs_of_pieces": len(candidates),
                       "pairs_in_truth": len(chosen),
                       "truth_pairs_between_existing_pieces_without_candidate": len(reachable - set(candidates))},
        "joins": {"selection": options.joins, "made": len(joins), "skipped_as_loops": loops,
                  "within_one_segment": sum(join["same_segment"] for join in joins),
                  "by_truth_kind": dict(Counter(join["truth"] for join in joins)),
                  "gap_um_quantiles": ([round(float(q), 1) for q in np.quantile([j["gap_um"] for j in joins],
                                                                               [0, .25, .5, .75, 1])] if joins else None)},
        **write_fragments(graph, wanted, options.out),
    }
    with options.out.with_name(options.out.stem + "_joins.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(joins[0]) if joins else ["node_id_a"])
        writer.writeheader()
        writer.writerows(joins)
    options.out.with_name(options.out.stem + "_connections.txt").write_text(
        "".join(f"{join['swc_a']}, {join['swc_b']}\n" for join in joins))
    options.out.with_suffix(".json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
