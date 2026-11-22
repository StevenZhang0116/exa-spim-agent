"""Cut the cache's fragment graph at merge candidates and write the result for the capsule.

A merge candidate of this repository is a node of the cache's fragment graph (a junction, or a
shaft node next to another segment's tip). It says where a merge is, not which arm belongs to
the other neuron, so a cut needs a rule:

* ``remove``: delete the nodes within ``--radius-um`` of path length from the site, as the
  Arborist pipeline does (``FragmentsGraph.remove_merge_sites`` of neuron-proofreader, 12 um:
  the neighbourhood of the site, extended around every other branching node inside it). Every
  arm of the site becomes a fragment of its own. With radius 0 only the site's node goes.
* ``detach``: remove one edge, the one to the arm that continues no other arm (the direction
  of an arm is taken over ``--arm-um`` of path; the arm whose best opposite arm is the least
  collinear is detached). The straightest pair of arms stays connected through the site. At a
  shaft node, which has two arms, one of its two edges is removed.

Which candidates are cut, ``--sites``:

* ``oracle``: the candidates that ground truth labels as merges (``evaluator_labels.npy`` of
  the brain's raw pool). It measures what the candidates and a cut rule can reach at best; no
  policy takes part and nothing here may be used to choose a policy.
* ``policy``: the ``--top`` candidates that a scorer ranks highest, read from a file of an
  evolution run's ``validation_scores/`` (``--scores``: the scores of one measured submission
  on the development-validation pools). Such a file holds scores only for the rows that ground
  truth can judge (a candidate on a segment under ground truth, within 150 um of a
  ground-truth node), so the ranking is over those rows and no candidate elsewhere is cut.
  Labels take no part in the choice; each site's label is written to the cuts file for the
  analysis afterwards.

Only candidates on the segments of ``--segments`` are cut, and only those segments are written
(``segments_under_gt.txt`` of a run of ``run_capsule_original.py``). The cache is not changed.
``join_split_candidates.py`` applies the same cuts (``oracle_sites`` or ``policy_sites``, then
``apply_cuts``) and reads a policy's split scores with the same ``ranked_rows``, before it
joins.

Run on a compute node in panda, from the repository root:

    python -u evaluation/cut_merge_sites.py --brain 841260 --sites oracle --rule remove \\
        --radius-um 12 --segments evaluation/results/<run>/segments_under_gt.txt \\
        --out evaluation/results/<experiment>/remove12/fragments.zip

For a policy's cuts, replace ``--sites oracle`` by
``--sites policy --scores <run>/validation_scores/gen<NNN>.npz --top <K>``.

Written beside ``<out>``: ``<out>.json`` (counts) and ``<out stem>_cuts.csv`` (one row per
site; with ``policy`` also its rank, score and label). Evaluate with ``run_capsule_original.py --fragments <zip>`` in arborist_eval and compare
with the unedited export with ``compare_capsule_runs.py --cuts <csv>``.
"""
import argparse
import csv
import hashlib
import json
import pickle
import sys
from collections import Counter
from pathlib import Path

import networkx as nx
import numpy as np

from export_cache_fragments import REPO, cache_record, load_cache, read_segments, segment_of, write_fragments

ARBORIST_RADIUS_UM = 12.0


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--brain", required=True)
    parser.add_argument("--mcl", type=int, default=100)
    parser.add_argument("--segments", required=True, type=Path, help="Text file, one segment ID per line")
    parser.add_argument("--sites", required=True, choices=("oracle", "policy"))
    parser.add_argument("--scores", type=Path, help="policy: a file of a run's validation_scores/")
    parser.add_argument("--top", type=int, help="policy: how many of the highest-ranked candidates to cut")
    parser.add_argument("--rule", required=True, choices=("remove", "detach"))
    parser.add_argument("--radius-um", type=float, default=ARBORIST_RADIUS_UM,
                        help=f"remove: path length around a site that is deleted (default {ARBORIST_RADIUS_UM:g}, "
                             "the Arborist pipeline's; 0 deletes the site's node only)")
    parser.add_argument("--arm-um", type=float, default=30.0,
                        help="detach: path length over which an arm's direction is taken (default 30)")
    parser.add_argument("--out", required=True, type=Path, help="ZIP file to write")
    options = parser.parse_args()
    if options.sites == "policy" and (options.scores is None or options.top is None or options.top < 1):
        parser.error("--sites policy needs --scores and a positive --top")
    return options


def pool_table(brain, cache, kind):
    """Rows and evaluator labels of the brain's raw pool of one kind, which must descend from the cache."""
    found = []
    for meta_path in sorted((REPO / "proofreader_evolve/feature_tables" / brain).glob("*/meta.json")):
        meta = json.loads(meta_path.read_text())
        if meta.get("provenance", {}).get("kind") == kind:
            found.append((meta_path.parent, meta))
    if len(found) != 1:
        sys.exit(f"expected one {kind} pool of {brain}, found {len(found)}")
    table, meta = found[0]
    source, stat = meta["provenance"]["source_cache"], cache.stat()
    if (source["size"], source["mtime_ns"]) != (stat.st_size, stat.st_mtime_ns):
        sys.exit(f"the {kind} pool {table.name[:12]} was not built from the present {cache.name}")
    with (table / "candidates.pkl").open("rb") as stream:
        rows = pickle.load(stream)
    labels = np.load(table / "evaluator_labels.npy")
    if len(rows) != len(labels):
        sys.exit(f"the {kind} pool's rows and labels differ in length")
    return table, meta, rows, labels


def ranked_rows(brain, cache, kind, scores_path, top):
    """The ``top`` rows of the brain's pool that a stored scorer ranks highest among the rows ground
    truth can judge (a file of a run's ``validation_scores/`` holds no other rows). Each row carries
    its rank, its score and its label; the label is not used to choose. Also returns the record of
    the pool and of the file. Refuses a file measured on another pool or other labels."""
    sys.path.insert(0, str(REPO))
    from proofreader_evolve.harness import label_availability

    table, meta, rows, labels = pool_table(brain, cache, kind)
    with np.load(label_availability.sidecar_dir(table) / "availability.npz", allow_pickle=False) as data:
        truth = label_availability.three_valued(labels, data[label_availability.DEFAULT_DEFINITION[kind]])
    cell = f"{brain}/{kind}"
    with np.load(scores_path, allow_pickle=False) as data:
        recorded = json.loads(str(data["meta"]))["cells"].get(cell)
        if recorded is None:
            sys.exit(f"{scores_path} holds no scores of {cell}")
        scored, scores = data[f"{cell}:rows"], data[f"{cell}:scores"]
    same_tables = (recorded["pool_sha256"], recorded["labels_sha256"]) == (
        meta["pool_sha256"], label_availability.labels_digest(truth))
    if not same_tables or not np.array_equal(scored, label_availability.labeled_rows(truth)):
        sys.exit(f"the scores of {cell} were measured on another pool or other labels")
    order = np.argsort(-scores, kind="stable")[:top]
    chosen = [{**rows[int(scored[index])], "rank": rank, "score": float(scores[index]),
               "evaluator_label": int(truth[scored[index]])} for rank, index in enumerate(order, start=1)]
    record = {"pool": {"table": table.name, "pool_sha256": meta["pool_sha256"], "rows": len(rows),
                       "candidate_policy": meta["candidate_policy"]},
              "scores": {"scores": str(scores_path),
                         "scores_sha256": hashlib.sha256(scores_path.read_bytes()).hexdigest(),
                         "ranked_rows": len(scores), "top": top, "chosen": len(chosen)}}
    return chosen, record


def check_rows(graph, rows):
    """The rows must name nodes of this graph: same position, degree and segment."""
    xyz, component_of, names = graph.node_xyz, graph.node_component_id, graph.component_id_to_swc_id
    for row in rows:
        node = row["node_id"]
        offset = float(np.abs(xyz[node] - np.array([row["x_um"], row["y_um"], row["z_um"]])).max())
        segment = segment_of(names[int(component_of[node])])
        if offset > 0.01 or len(graph.adj[node]) != row["degree"] or segment != str(row["segment_id"]):
            sys.exit(f"candidate {row['candidate_id']} does not match node {node} of the cache graph "
                     f"(offset {offset:.3f} um, degree {len(graph.adj[node])} vs {row['degree']}, "
                     f"segment {segment} vs {row['segment_id']})")


def within(graph, xyz, root, radius):
    """Nodes within ``radius`` um of path length from ``root``, the root included."""
    reached, stack = {root: 0.0}, [root]
    while stack:
        node = stack.pop()
        for other in graph.adj[node]:
            length = reached[node] + float(np.linalg.norm(xyz[node] - xyz[other]))
            if length <= radius and length < reached.get(other, np.inf):
                reached[other] = length
                stack.append(other)
    return set(reached)


def removal(graph, xyz, site, radius):
    """Nodes that the Arborist pipeline's rule deletes for one site."""
    nodes = within(graph, xyz, site, radius)
    for node in list(nodes):
        if node != site and len(graph.adj[node]) >= 3:
            nodes |= within(graph, xyz, node, radius)
    return nodes


def arm_direction(graph, xyz, site, neighbor, length):
    """Unit vector from the site to the point ``length`` um along the arm that starts at ``neighbor``
    (or to the arm's first branching node or its tip, when that comes first)."""
    previous, node = site, neighbor
    walked = float(np.linalg.norm(xyz[site] - xyz[neighbor]))
    while walked < length and len(graph.adj[node]) == 2:
        following = next(other for other in graph.adj[node] if other != previous)
        walked += float(np.linalg.norm(xyz[node] - xyz[following]))
        previous, node = node, following
    vector = (xyz[node] - xyz[site]).astype(np.float64)
    return vector / max(float(np.linalg.norm(vector)), 1e-9)


def arm_to_detach(graph, xyz, site, arm_um):
    """The neighbour of the site whose arm continues no other arm, and that arm's best collinearity
    (1 = it runs straight on from another arm)."""
    neighbors = sorted(graph.adj[site])
    if len(neighbors) == 2:
        return neighbors[0], None
    directions = [arm_direction(graph, xyz, site, neighbor, arm_um) for neighbor in neighbors]
    straight = [max(-float(directions[i] @ directions[j]) for j in range(len(neighbors)) if j != i)
                for i in range(len(neighbors))]
    arm = int(np.argmin(straight))
    return neighbors[arm], straight[arm]


def oracle_sites(brain, cache, graph, wanted):
    """The merge candidates that ground truth labels as merges, on the listed segments, with
    the record of where they came from."""
    table, meta, rows, labels = pool_table(brain, cache, "merge")
    chosen = [row for row, label in zip(rows, labels) if label == 1]
    sites = [row for row in chosen if str(row["segment_id"]) in wanted]
    check_rows(graph, sites)
    print(f"merge pool {table.name[:12]}: {len(rows)} rows, {len(chosen)} chosen (oracle), "
          f"{len(sites)} on the {len(wanted)} listed segments", flush=True)
    record = {"merge_pool": {"table": table.name, "pool_sha256": meta["pool_sha256"], "rows": len(rows),
                             "candidate_policy": meta["candidate_policy"]},
              "sites": {"selection": "oracle", "chosen": len(chosen), "on_listed_segments": len(sites),
                        "by_family": dict(Counter(row["candidate_family"] for row in sites)),
                        "by_degree": dict(sorted(Counter(int(row["degree"]) for row in sites).items()))}}
    return sites, record


def policy_sites(brain, cache, graph, wanted, scores_path, top):
    """The ``top`` merge candidates that a stored scorer ranks highest among the rows ground truth
    can judge, on the listed segments, with the record of where they came from. Each site
    carries its rank, its score and its label; the label is not used to choose."""
    chosen, ranked = ranked_rows(brain, cache, "merge", scores_path, top)
    sites = [row for row in chosen if str(row["segment_id"]) in wanted]
    check_rows(graph, sites)
    print(f"merge pool {ranked['pool']['table'][:12]}: {ranked['pool']['rows']} rows, "
          f"{ranked['scores']['ranked_rows']} ranked by {scores_path.name}, "
          f"top {top} chosen (policy), {len(sites)} on the {len(wanted)} listed segments, "
          f"{sum(row['evaluator_label'] for row in sites)} of them labeled as merges", flush=True)
    record = {"merge_pool": ranked["pool"],
              "sites": {"selection": "policy", **ranked["scores"],
                        "on_listed_segments": len(sites),
                        "labeled_as_merges": sum(row["evaluator_label"] for row in sites),
                        "lowest_score": chosen[-1]["score"],
                        "by_family": dict(Counter(row["candidate_family"] for row in sites)),
                        "by_degree": dict(sorted(Counter(int(row["degree"]) for row in sites).items()))}}
    return sites, record


def apply_cuts(graph, sites, rule, radius_um=ARBORIST_RADIUS_UM, arm_um=30.0):
    """Cut the graph in place at the sites. Returns one record per site and the edit's summary."""
    xyz, records = graph.node_xyz, []
    if rule == "remove":
        plans = [(row, removal(graph, xyz, row["node_id"], radius_um)) for row in sites]
        deleted = set()
        for row, nodes in plans:
            records.append({"action": "removed", "nodes_removed": len(nodes), "arms": row["degree"]})
            deleted |= nodes
        nx.Graph.remove_nodes_from(graph, deleted)
        return records, {"rule": "remove", "radius_um": radius_um, "nodes_removed": len(deleted)}
    for row in sites:
        site = row["node_id"]
        if len(graph.adj[site]) < 2:
            records.append({"action": "skipped: fewer than two arms left", "arms": len(graph.adj[site])})
            continue
        neighbor, collinearity = arm_to_detach(graph, xyz, site, arm_um)
        records.append({"action": "detached", "arms": len(graph.adj[site]),
                        "detached_arm_collinearity": None if collinearity is None else round(collinearity, 4)})
        nx.Graph.remove_edge(graph, site, neighbor)
    return records, {"rule": "detach", "arm_um": arm_um,
                     "edges_removed": sum(record["action"] == "detached" for record in records)}


def main():
    options = parse_args()
    cache, payload = load_cache(options.brain, options.mcl)
    graph = payload["fragments_graph"]
    wanted = read_segments(options.segments)
    if options.sites == "oracle":
        sites, record = oracle_sites(options.brain, cache, graph, wanted)
    else:
        sites, record = policy_sites(options.brain, cache, graph, wanted, options.scores, options.top)
    records, edit = apply_cuts(graph, sites, options.rule, options.radius_um, options.arm_um)

    summary = {**cache_record(options.brain, cache, payload), **record, "edit": edit,
               **write_fragments(graph, wanted, options.out)}
    site_fields = ["candidate_id", "node_id", "segment_id", "candidate_family", "degree", "x_um", "y_um", "z_um"]
    if options.sites == "policy":
        site_fields += ["rank", "score", "evaluator_label"]
    fields = site_fields + ["action", "arms", "nodes_removed", "detached_arm_collinearity"]
    with options.out.with_name(options.out.stem + "_cuts.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, restval="")
        writer.writeheader()
        for row, record in zip(sites, records):
            writer.writerow({**{field: row[field] for field in site_fields}, **record})
    options.out.with_suffix(".json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
