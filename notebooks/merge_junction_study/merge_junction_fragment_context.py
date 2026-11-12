#!/usr/bin/env python3
"""Does fragment-level context separate merge junctions from true branch points?

Background (``proofreader_evolve/TODO.md`` section 7): nearly every merge positive is a
``junction`` candidate, and inside that family the accepted scorers reach a whole-brain holdout
AUROC of only 0.63-0.74. A descriptor sees the fragment within 50 um of the candidate (256
nodes at most), so it cannot tell how large the fragment is or what each arm of the junction
leads to. This script measures, offline and on TRAIN brains only, whether that longer-range
information carries signal, before anything is added to the evolution workflow.

``features``
    For every GT-labeled merge candidate of a brain, from the cached fragment graph (a forest):

    * fragment (connected component) totals: cable length, tips, branch points;
    * for junction candidates, every arm leaving the junction: the cable length, tips and
      branch points of the whole subtree behind it, however far it extends;
    * a local description of the arms from a 30 um walk (direction, mean radius): the most
      collinear ("through") pair and the remaining "side" arm, so that the long-range size of
      the side arm can be read together with the local geometry.

    Subtree totals come from one pass over the irreducible forest (degree-2 chains contracted),
    so the cost is linear in the fragment sizes. Nothing here reads a label: the labels are
    attached only to choose the rows and are stored beside the features for the next stage.

``evaluate``
    Junction family, GT-labeled rows, whole-brain holdout over the TRAIN brains (each brain is
    scored by a model fitted on the others). Compared, with the same small gradient-boosting
    model: the accepted scorer's own holdout score (read from a finished run), the fragment
    features alone, the scorer score plus fragment totals, and the scorer score plus all
    fragment features. Also the one-feature AUROC of every feature per brain.

Run on a compute node in the ``panda`` environment from the repository root::

    python -u notebooks/merge_junction_study/merge_junction_fragment_context.py features --brains 747807 802449
    python -u notebooks/merge_junction_study/merge_junction_fragment_context.py evaluate --run proofreader_evolve/runs/<run>

Outputs under ``notebooks/merge_junction_study/merge_junction_fragment_context/`` (gitignored): ``features_<brain>.npz``,
``univariate.csv``, ``holdout.csv``, ``summary.md``. Read-only for caches, tables and runs.

Limits. TRAIN brains only: the validation brains are not opened, so the result says whether the
information is worth giving to the reviser, not what a scorer would reach on validation. Fragment
size is related to where ground truth exists (longer fragments are more often traced), so a
fragment-total feature can carry that relation; the comparison "scorer + totals" against
"scorer + all" separates the fragment-wide effect from the junction-specific one. The model is
fixed, not tuned, and the brains differ in junction base rate (1.4% to 9%).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = Path(__file__).resolve().parent / "merge_junction_fragment_context"
TRAIN_BRAINS = ["802449", "747807", "750318", "754610", "754612", "754613"]
WALK_UM = 30.0
TOTALS = ["comp_cable_um", "comp_tips", "comp_branches"]
ARMS = ["arm_max_um", "arm_mid_um", "arm_min_um", "arm_min_frac", "arm_mid_frac",
        "arm_min_tips", "arm_mid_tips", "arm_min_branches", "arms_ge_100um", "arms_ge_500um",
        "side_arm_um", "side_arm_frac", "side_arm_tips", "side_arm_branches",
        "through_cos", "side_abs_cos", "side_radius_ratio", "n_arms"]
FEATURES = TOTALS + ARMS


def log(message):
    print(time.strftime("%H:%M:%S"), message, flush=True)


# --------------------------------------------------------------------------------------------
# fragment graph -> irreducible forest -> subtree totals
# --------------------------------------------------------------------------------------------

def irreducible_forest(xyz, edges, degree, keep):
    """Contract degree-2 chains of the kept nodes. Returns the irreducible nodes' adjacency
    ``{node: [(neighbor, length_um), ...]}`` and, for every chain node, its two chain ends
    (the irreducible nodes the chain connects)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    a, b = edges[keep[edges[:, 0]]].T
    length = np.linalg.norm(xyz[a].astype(np.float64) - xyz[b].astype(np.float64), axis=1)
    irr = degree != 2
    two = np.flatnonzero(keep & ~irr)
    position = np.full(len(degree), -1, dtype=np.int64)
    position[two] = np.arange(len(two))
    inner = ~irr[a] & ~irr[b]
    count, chain = connected_components(
        coo_matrix((np.ones(inner.sum(), dtype=np.int8), (position[a[inner]], position[b[inner]])),
                   shape=(len(two), len(two))), directed=False)
    chain_of = np.full(len(degree), -1, dtype=np.int64)
    chain_of[two] = chain
    chain_length = np.bincount(chain[position[a[inner]]], weights=length[inner], minlength=count)
    boundary = irr[a] ^ irr[b]
    end = np.where(irr[a], a, b)[boundary]
    chain_id = chain_of[np.where(irr[a], b, a)[boundary]]
    chain_length += np.bincount(chain_id, weights=length[boundary], minlength=count)
    order = np.argsort(chain_id, kind="stable")
    chain_id, end = chain_id[order], end[order]
    # In a forest every chain meets exactly two irreducible nodes; anything else is skipped.
    pairs = np.flatnonzero(np.bincount(chain_id, minlength=count) == 2)
    starts = np.searchsorted(chain_id, pairs)
    ends = np.full((count, 2), -1, dtype=np.int64)
    ends[pairs, 0], ends[pairs, 1] = end[starts], end[starts + 1]
    adjacency = {}

    def link(u, v, w):
        adjacency.setdefault(int(u), []).append((int(v), float(w)))
        adjacency.setdefault(int(v), []).append((int(u), float(w)))

    direct = irr[a] & irr[b]
    for u, v, w in zip(a[direct], b[direct], length[direct]):
        link(u, v, w)
    for c in pairs:
        link(ends[c, 0], ends[c, 1], chain_length[c])
    return adjacency, chain_of, ends


def subtree_totals(adjacency, degree):
    """Root every tree of the irreducible forest; per node the parent, the cable, tips and branch
    points of its subtree (the node included for tips and branch points, the edge to its parent
    excluded for cable), and its tree's root."""
    parent, root_of, cable, tips, branches = {}, {}, {}, {}, {}
    for start in adjacency:
        if start in parent:
            continue
        parent[start] = -1
        order, stack = [], [start]
        while stack:
            node = stack.pop()
            order.append(node)
            root_of[node] = start
            for neighbor, _ in adjacency[node]:
                if neighbor not in parent:
                    parent[neighbor] = node
                    stack.append(neighbor)
        for node in reversed(order):
            total_cable, total_tips, total_branches = 0.0, int(degree[node] == 1), int(degree[node] >= 3)
            for neighbor, weight in adjacency[node]:
                if parent.get(neighbor) == node:
                    total_cable += weight + cable[neighbor]
                    total_tips += tips[neighbor]
                    total_branches += branches[neighbor]
            cable[node], tips[node], branches[node] = total_cable, total_tips, total_branches
    return parent, root_of, cable, tips, branches


def walk(graph, xyz, radius, degree, junction, first, limit_um):
    """Unit direction and mean radius of one arm over about ``limit_um`` of path from the junction."""
    previous, node, travelled, radii = junction, first, 0.0, []
    while True:
        travelled += float(np.linalg.norm(xyz[node].astype(np.float64) - xyz[previous].astype(np.float64)))
        if np.isfinite(radius[node]):
            radii.append(float(radius[node]))
        if travelled >= limit_um or degree[node] != 2:
            break
        following = next(n for n in graph.adj[node] if n != previous)
        previous, node = node, following
    vector = xyz[node].astype(np.float64) - xyz[junction].astype(np.float64)
    norm = float(np.linalg.norm(vector))
    return (vector / norm if norm > 0 else np.zeros(3)), (float(np.mean(radii)) if radii else np.nan)


def brain_features(brain):
    from proofreader_evolve.harness import label_availability
    from proofreader_evolve.harness.dataset import load_cached_graphs
    from proofreader_evolve.harness.native_pool import cache_path
    from proofreader_evolve.harness.raw_pool import ensure_raw_tables

    started = time.monotonic()
    bank = ensure_raw_tables(brain, cache_path(brain, 100), ROOT / "proofreader_evolve" / "feature_tables",
                             prepare=False, mcl=100)
    label_availability.apply({brain: bank})
    table = bank.tables["merge"]
    truth = np.asarray(table.truth, dtype=float)
    rows = np.flatnonzero(np.isfinite(truth))
    node = np.array([table.candidates[i]["node_id"] for i in rows], dtype=np.int64)
    family = np.array([table.candidates[i]["candidate_family"] for i in rows])
    log(f"[{brain}] {len(rows)} labeled merge rows ({int(truth[rows].sum())} positives); loading the fragment graph")
    graph, gt, payload = load_cached_graphs(str(cache_path(brain, 100)), expect_brain=brain, expect_mcl=100)
    del gt, payload
    xyz, radius, component = graph.node_xyz, graph.node_radius, graph.node_component_id
    edges = np.array(list(graph.edges()), dtype=np.int64)
    degree = np.bincount(edges.ravel(), minlength=len(xyz))
    if len(edges) - len(xyz) + int(component.max()) + 1 != 0:
        raise ValueError(f"{brain}: the fragment graph is not a forest")
    keep = np.isin(component, np.unique(component[node]))
    log(f"[{brain}] graph {len(xyz)} nodes, {int(keep.sum())} in the {len(np.unique(component[node]))} fragments "
        f"of labeled candidates [{time.monotonic() - started:.0f}s]")
    adjacency, chain_of, chain_ends = irreducible_forest(xyz, edges, degree, keep)
    parent, root_of, cable, tips, branches = subtree_totals(adjacency, degree)
    values = np.full((len(rows), len(FEATURES)), np.nan)
    column = {name: i for i, name in enumerate(FEATURES)}
    skipped = 0
    for index, (candidate, kind) in enumerate(zip(node, family)):
        candidate = int(candidate)
        if degree[candidate] != 2:
            anchor = candidate
        else:  # a shaft node (tip_to_shaft): fragment totals only, through either end of its chain
            anchor = int(chain_ends[chain_of[candidate], 0])
        if anchor < 0 or anchor not in root_of:
            skipped += 1
            continue
        root = root_of[anchor]
        total_cable, total_tips, total_branches = cable[root], tips[root], branches[root]
        values[index, column["comp_cable_um"]] = total_cable
        values[index, column["comp_tips"]] = total_tips
        values[index, column["comp_branches"]] = total_branches
        if kind != "junction" or degree[candidate] < 3:
            continue
        arms = []
        for first in graph.adj[candidate]:
            first = int(first)
            if degree[first] != 2:
                behind = first
            else:
                ends = chain_ends[chain_of[first]]
                behind = int(ends[0] if ends[1] == candidate else ends[1])
            if behind < 0 or behind not in root_of:
                arms = None
                break
            if parent.get(behind) == candidate:
                weight = next(w for n, w in adjacency[candidate] if n == behind)
                arm = (weight + cable[behind], tips[behind], branches[behind])
            else:  # the arm that leads to the root: everything outside the junction's subtree
                arm = (total_cable - cable[candidate], total_tips - tips[candidate],
                       total_branches - branches[candidate])
            direction, mean_radius = walk(graph, xyz, radius, degree, candidate, first, WALK_UM)
            arms.append((*arm, direction, mean_radius))
        if not arms or len(arms) < 3:
            skipped += 1
            continue
        by_size = sorted(arms, key=lambda arm: arm[0], reverse=True)
        sizes = [arm[0] for arm in by_size]
        # Through pair: the two arms whose directions are most nearly opposite; side arm: the largest other.
        best = min(((float(np.dot(arms[i][3], arms[j][3])), i, j)
                    for i in range(len(arms)) for j in range(i + 1, len(arms))), key=lambda item: item[0])
        through = (arms[best[1]], arms[best[2]])
        side = max((arms[i] for i in range(len(arms)) if i not in best[1:]), key=lambda arm: arm[0])
        axis = through[0][3] - through[1][3]
        axis_norm = float(np.linalg.norm(axis))
        through_radius = np.nanmean([through[0][4], through[1][4]])
        row = {"arm_max_um": sizes[0], "arm_mid_um": sizes[1], "arm_min_um": sizes[-1],
               "arm_min_frac": sizes[-1] / total_cable if total_cable else np.nan,
               "arm_mid_frac": sizes[1] / total_cable if total_cable else np.nan,
               "arm_min_tips": by_size[-1][1], "arm_mid_tips": by_size[1][1], "arm_min_branches": by_size[-1][2],
               "arms_ge_100um": sum(s >= 100 for s in sizes), "arms_ge_500um": sum(s >= 500 for s in sizes),
               "side_arm_um": side[0], "side_arm_frac": side[0] / total_cable if total_cable else np.nan,
               "side_arm_tips": side[1], "side_arm_branches": side[2], "through_cos": best[0],
               "side_abs_cos": abs(float(np.dot(side[3], axis / axis_norm))) if axis_norm > 0 else np.nan,
               "side_radius_ratio": side[4] / through_radius if through_radius and np.isfinite(through_radius) else np.nan,
               "n_arms": len(arms)}
        for name, value in row.items():
            values[index, column[name]] = value
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"features_{brain}.npz", rows=rows, truth=truth[rows].astype(np.int8), family=family,
                        node=node, values=values, names=np.array(FEATURES))
    junction = family == "junction"
    log(f"[{brain}] wrote features: {int(junction.sum())} junction rows ({int(truth[rows][junction].sum())} positives), "
        f"{skipped} rows without arms [{time.monotonic() - started:.0f}s]")


# --------------------------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------------------------

def rank_statistics(scores, labels):
    """Tie-aware AUROC and average precision, and hits in the Top K with K = positives."""
    from sklearn.metrics import average_precision_score, roc_auc_score
    positives = int(labels.sum())
    order = np.argsort(-scores, kind="stable")
    return (float(roc_auc_score(labels, scores)), float(average_precision_score(labels, scores)),
            int(labels[order[:positives]].sum()), positives)


def accepted_merge_scores(run_dir):
    rows = [json.loads(line) for line in (run_dir / "ledger.jsonl").open()]
    accepted = [row for row in rows if row["target_kind"] == "merge" and row["accepted"]][-1]
    path = run_dir / accepted["submitted_experiment"].replace("/", "/experiments/", 1) / "selection_state.npz"
    return accepted["submitted_experiment"], np.load(path)


def evaluate(args):
    from scipy.stats import rankdata
    from sklearn.ensemble import HistGradientBoostingClassifier

    experiment, state = accepted_merge_scores(Path(args.run))
    data = {}
    for brain in args.brains:
        stored = np.load(OUT / f"features_{brain}.npz", allow_pickle=False)
        junction = (stored["family"] == "junction") & np.isfinite(stored["values"][:, FEATURES.index("arm_max_um")])
        scores = state[f"{brain}/merge:scores"][stored["rows"]]
        # Holdout scores of different fold models are on different scales: use the within-brain rank.
        data[brain] = {"X": stored["values"][junction], "y": stored["truth"][junction].astype(int),
                       "score": rankdata(scores[junction]) / int(junction.sum()),
                       "dropped": int(((stored["family"] == "junction") & ~junction).sum())}
        log(f"[{brain}] junction rows {len(data[brain]['y'])}, positives {int(data[brain]['y'].sum())}, "
            f"without arm features {data[brain]['dropped']}")
    arms = {
        "scorer holdout score (reference)": None,
        "fragment features alone": FEATURES,
        "scorer score + fragment totals": ["score"] + TOTALS,
        "scorer score + all fragment features": ["score"] + FEATURES,
    }

    def matrix(brain, names):
        columns = [data[brain]["score"] if name == "score" else data[brain]["X"][:, FEATURES.index(name)]
                   for name in names]
        return np.column_stack(columns)

    holdout = []
    for label, names in arms.items():
        for brain in args.brains:
            y = data[brain]["y"]
            if names is None:
                predicted = data[brain]["score"]
            else:
                others = [b for b in args.brains if b != brain]
                model = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_depth=3,
                                                       min_samples_leaf=20, l2_regularization=1.0, random_state=0)
                model.fit(np.vstack([matrix(b, names) for b in others]), np.concatenate([data[b]["y"] for b in others]))
                predicted = model.predict_proba(matrix(brain, names))[:, 1]
            auroc, precision, hits, positives = rank_statistics(predicted, y)
            holdout.append({"arm": label, "brain": brain, "rows": len(y), "positives": positives,
                            "auroc": auroc, "average_precision": precision, "hits_at_k": hits})
    univariate = []
    for name in FEATURES:
        row = {"feature": name}
        for brain in args.brains:
            values, y = data[brain]["X"][:, FEATURES.index(name)], data[brain]["y"]
            finite = np.isfinite(values)
            row[brain] = rank_statistics(values[finite], y[finite])[0] if y[finite].sum() and (1 - y[finite]).sum() else np.nan
        univariate.append(row)
    with (OUT / "holdout.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(holdout[0]))
        writer.writeheader()
        writer.writerows(holdout)
    with (OUT / "univariate.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(univariate[0]))
        writer.writeheader()
        writer.writerows(univariate)

    lines = ["# Fragment-level context for merge junctions", "",
             f"Junction family, GT-labeled rows, whole-brain holdout over TRAIN brains {', '.join(args.brains)}. "
             f"Reference scorer: `{experiment}` of `{Path(args.run).name}` (its own holdout scores). "
             "Model for the other rows: HistGradientBoostingClassifier (200 trees, depth 3, fixed, not tuned). "
             "Hits are counted in the Top K with K = the brain's junction positives.", "",
             "| arm | " + " | ".join(args.brains) + " | positive-weighted mean |", "|---|" + "---|" * (len(args.brains) + 1)]
    for statistic, title in (("auroc", "AUROC"), ("average_precision", "average precision"), ("hits_at_k", "hits")):
        lines.append(f"| **{title}** |" + " |" * (len(args.brains) + 1))
        for label in arms:
            cells = [row for row in holdout if row["arm"] == label]
            weights = np.array([row["positives"] for row in cells], dtype=float)
            if statistic == "hits_at_k":
                values = [f"{row['hits_at_k']}/{row['positives']}" for row in cells]
                total = f"{sum(row['hits_at_k'] for row in cells)}/{int(weights.sum())}"
            else:
                values = [f"{row[statistic]:.3f}" for row in cells]
                total = f"{float(np.dot([row[statistic] for row in cells], weights) / weights.sum()):.3f}"
            lines.append(f"| {label} | " + " | ".join(values) + f" | {total} |")
    lines += ["", "## One-feature AUROC within the junction family (0.5 = no information; below 0.5 = lower values on positives)",
              "", "| feature | " + " | ".join(args.brains) + " |", "|---|" + "---|" * len(args.brains)]
    for row in univariate:
        lines.append(f"| {row['feature']} | " + " | ".join(f"{row[b]:.3f}" for b in args.brains) + " |")
    lines += ["", "Junction rows per brain (labeled / positives): "
              + ", ".join(f"{b} {len(data[b]['y'])} / {int(data[b]['y'].sum())}" for b in args.brains) + "."]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="stage", required=True)
    p = sub.add_parser("features", help="fragment totals and arm subtrees of the labeled merge candidates")
    p.add_argument("--brains", nargs="+", default=TRAIN_BRAINS, help="TRAIN brains only")
    p = sub.add_parser("evaluate", help="whole-brain holdout comparison within the junction family")
    p.add_argument("--brains", nargs="+", default=TRAIN_BRAINS)
    p.add_argument("--run", required=True, help="finished run whose accepted merge scorer is the reference")
    args = parser.parse_args()
    unknown = set(args.brains) - set(TRAIN_BRAINS)
    if unknown:
        parser.error(f"Not TRAIN brains: {sorted(unknown)}; this analysis must not open validation or reserved brains")
    if args.stage == "features":
        for brain in args.brains:
            brain_features(brain)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
