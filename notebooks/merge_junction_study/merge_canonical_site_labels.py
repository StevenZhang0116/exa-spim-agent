#!/usr/bin/env python3
"""Which merge candidates sit at a canonical merge site without being labeled positive?

Background (``notebooks/merge_canonical_gap.py``, ``proofreader_evolve/TODO.md`` section 7): a
merge candidate is positive when it lies within 20 um of fragment cable of a merge site stored
in the dataset cache (``gt_merge_sites``). Those sites are recomputed on the cache; they are not
the sites reported by segmentation-skeleton-metrics (``metrics_out/<brain>/*/merge_sites.csv``,
"canonical" here). Of 1,299 canonical sites on twelve brains, 81% have a cache site within
50 um and 83% a candidate within 20 um of cable, so some candidates sit at a canonical merge
and carry the label 0 or no label. This script finds them, on TRAIN brains only, and measures
what would change if the canonical sites were a label source as well.

``rows``
    Every canonical site is snapped to the nearest node of its own segment within 50 um (the
    rule of ``candidate_rules``) and the fragment is searched to ``REACH_UM`` of cable. Per site:
    the candidates within 20 um with their current labels, the cable distance to the nearest
    cache site, and, at the snapped node, the distance to the nearest GT node and how many
    connected GT nodes there carry the segment's label (the cache's walk drops a contact below
    ``passthru_min_cc``). Per candidate: the cable distance to the nearest canonical site.

``summarize``
    Sites by outcome (a positive candidate, candidates but none positive, no candidate); the
    candidates that would change to positive, by family and by the reading of the GT around
    them (``merge_junction_label_localization``); where the cache's nearest site is for the
    sites it misses; and, with the holdout scores of a finished run's accepted merge scorer,
    the scorer's rank of those candidates and the junction AUROC, average precision and Top-K
    hits under the current labels and with the canonical sites added.

Run on a compute node in the ``panda`` environment from the repository root::

    python -u notebooks/merge_junction_study/merge_canonical_site_labels.py rows --brains 747807 802449
    python -u notebooks/merge_junction_study/merge_canonical_site_labels.py summarize --run proofreader_evolve/runs/<run>

Outputs under ``notebooks/merge_junction_study/merge_canonical_site_labels/`` (gitignored):
``sites_<brain>.npz``, ``summary.md``. Read-only for caches, tables, runs and ``metrics_out``.

Limits. TRAIN brains only. The scorer was fitted with the current labels; only the labels it
is measured against change here. A canonical site is taken as given: whether it is a merge
rests on the same tracing as every other label.
"""
from __future__ import annotations

import argparse
import heapq
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for entry in (str(ROOT), str(ROOT / "notebooks")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from merge_junction_fragment_context import TRAIN_BRAINS, accepted_merge_scores, log  # noqa: E402
from merge_junction_label_localization import CATEGORIES, same_label_run, step  # noqa: E402

OUT = Path(__file__).resolve().parent / "merge_canonical_site_labels"
EVIDENCE = Path(__file__).resolve().parent / "merge_junction_label_localization"
REACH_UM, LABEL_UM = 150.0, 20.0


def snap(tree, kept, xyz, segment, point, segment_id, limit_um):
    near = kept[np.asarray(tree.query_ball_point(point, r=limit_um), dtype=np.int64)]
    own = near[segment[near] == segment_id]
    return int(own[int(np.argmin(np.linalg.norm(xyz[own] - point, axis=1)))]) if len(own) else -1


def cable_from(graph, xyz, start, limit_um):
    best, heap = {start: 0.0}, [(0.0, start)]
    while heap:
        distance, node = heapq.heappop(heap)
        if distance > best.get(node, np.inf):
            continue
        for following in graph.adj[node]:
            following = int(following)
            reached = distance + step(xyz, following, node)
            if reached <= limit_um and reached < best.get(following, np.inf):
                best[following] = reached
                heapq.heappush(heap, (reached, following))
    return best


def brain_rows(brain):
    from scipy.spatial import cKDTree
    from candidate_rules import SITE_SNAP_MAX_UM, segment_of_component
    from merge_canonical_gap import load_canonical
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
    row_of = {int(c["node_id"]): i for i, c in enumerate(table.candidates)}
    if len(row_of) != len(table.candidates):
        raise ValueError(f"{brain}: merge candidates share nodes")
    graph, gt, payload = load_cached_graphs(str(cache_path(brain, 100)), expect_brain=brain, expect_mcl=100)
    xyz = np.asarray(graph.node_xyz, dtype=np.float64)
    component = np.asarray(graph.node_component_id, dtype=np.int64)
    to_segment = np.full(int(component.max()) + 1, -1, dtype=np.int64)
    for key, value in segment_of_component(graph).items():
        to_segment[int(key)] = int(value)
    segment = to_segment[component]
    gt_xyz = np.asarray(gt.node_xyz, dtype=np.float64)
    gt_label = np.asarray(payload["gt_node_canonical_label"], dtype=np.int64)
    passthru = int(payload["gt_label_policy"]["passthru_min_cc"])
    gt_tree = cKDTree(gt_xyz)
    canonical = load_canonical(ROOT / "metrics_out", brain, payload["anisotropy"])
    cache_sites = list(payload.get("gt_merge_sites") or [])
    wanted = set(canonical["segment"].tolist()) | {int(s["segment_id"]) for s in cache_sites}
    kept = np.flatnonzero(np.isin(segment, list(wanted)))
    tree = cKDTree(xyz[kept])
    cache_nodes = {}
    for site in cache_sites:
        node = snap(tree, kept, xyz, segment, np.asarray(site["xyz"], dtype=np.float64), int(site["segment_id"]),
                    SITE_SNAP_MAX_UM)
        if node >= 0:
            cache_nodes[node] = str(site.get("source", "geometric_walk"))
    cache_xyz = np.array([s["xyz"] for s in cache_sites], dtype=np.float64).reshape(-1, 3)
    log(f"[{brain}] {len(canonical['xyz'])} canonical sites, {len(cache_sites)} cache sites ({len(cache_nodes)} snapped), "
        f"{len(table.candidates)} merge candidates [{time.monotonic() - started:.0f}s]")

    names = ["snapped", "candidate_um", "n_rows", "n_positive", "n_zero", "n_unlabeled", "cache_cable_um",
             "cache_euclid_um", "gt_um", "gt_run", "added_cable_um"]
    sites = np.full((len(canonical["xyz"]), len(names)), np.nan)
    column = {name: i for i, name in enumerate(names)}
    canon_cable = {}
    import pandas as pd
    added = pd.read_csv(canonical["path"]).filter(like="Added Cable").iloc[:, 0].to_numpy(dtype=float)
    for index, (point, segment_id) in enumerate(zip(canonical["xyz"], canonical["segment"].tolist())):
        record = sites[index]
        record[column["added_cable_um"]] = added[index]
        record[column["cache_euclid_um"]] = (float(np.linalg.norm(cache_xyz - point, axis=1).min())
                                             if len(cache_xyz) else np.inf)
        node = snap(tree, kept, xyz, segment, point, int(segment_id), SITE_SNAP_MAX_UM)
        record[column["snapped"]] = float(node >= 0)
        if node < 0:
            continue
        reach = cable_from(graph, xyz, node, REACH_UM)
        rows = [(distance, row_of[n]) for n, distance in reach.items() if n in row_of]
        for distance, row in rows:
            if distance < canon_cable.get(row, np.inf):
                canon_cable[row] = distance
        close = [row for distance, row in rows if distance <= LABEL_UM]
        labels = truth[close] if close else np.empty(0)
        record[column["candidate_um"]] = min((distance for distance, _ in rows), default=np.inf)
        record[column["n_rows"]] = len(close)
        record[column["n_positive"]] = int((labels == 1).sum())
        record[column["n_zero"]] = int((labels == 0).sum())
        record[column["n_unlabeled"]] = int(np.isnan(labels).sum())
        record[column["cache_cable_um"]] = min((reach[n] for n in cache_nodes if n in reach), default=np.inf)
        distance, nearest = gt_tree.query(xyz[node])
        record[column["gt_um"]] = distance
        record[column["gt_run"]] = same_label_run(gt, gt_label, int(nearest), int(segment_id), cap=max(passthru, 60))
    reached = np.array(sorted(canon_cable), dtype=np.int64)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"sites_{brain}.npz", sites=sites, names=np.array(names), passthru=passthru,
                        rows=reached, canon_cable=np.array([canon_cable[r] for r in reached]),
                        truth=truth, family=np.array([c["candidate_family"] for c in table.candidates]))
    close = np.array([canon_cable[r] <= LABEL_UM for r in reached], dtype=bool)
    labels = truth[reached[close]]
    log(f"[{brain}] candidates within {LABEL_UM:g} um of a canonical site: {int(close.sum())} "
        f"(label 1: {int((labels == 1).sum())}, label 0: {int((labels == 0).sum())}, unlabeled: {int(np.isnan(labels).sum())}) "
        f"[{time.monotonic() - started:.0f}s]")


def table(header, body):
    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + ["| " + " | ".join(str(c) for c in row) + " |"
                                                                               for row in body]


def summarize(args):
    from sklearn.metrics import average_precision_score, roc_auc_score

    experiment, state = accepted_merge_scores(Path(args.run))
    lines = ["# Merge candidates at a canonical merge site that are not labeled positive", "",
             f"Script `notebooks/merge_junction_study/merge_canonical_site_labels.py`; canonical sites from "
             f"`metrics_out/<brain>/*/merge_sites.csv`; a candidate counts within {LABEL_UM:g} um of fragment cable. "
             f"Scores: holdout scores of `{experiment}` in `{Path(args.run).name}`.", ""]
    site_body, row_body, metric_body, why_body = [], [], [], []
    category_counts = {c: 0 for c in CATEGORIES}
    ranks = {"labeled positive": [], "label 0 at a canonical site": [], "unlabeled at a canonical site": [],
             "other label 0": []}
    totals = np.zeros(7, dtype=int)
    row_totals = {}
    for brain in args.brains:
        stored = np.load(OUT / f"sites_{brain}.npz", allow_pickle=False)
        site = {name: stored["sites"][:, i] for i, name in enumerate(stored["names"].tolist())}
        truth, family, passthru = stored["truth"], stored["family"], int(stored["passthru"])
        snapped = site["snapped"] == 1
        with_rows = snapped & (site["n_rows"] > 0)
        positive = with_rows & (site["n_positive"] > 0)
        zero = with_rows & ~positive & (site["n_zero"] > 0)
        unlabeled = with_rows & ~positive & ~zero
        counts = np.array([len(snapped), int((~snapped).sum()), int(positive.sum()), int(zero.sum()),
                           int(unlabeled.sum()), int((snapped & ~with_rows).sum()),
                           int((snapped & ~with_rows & (site["candidate_um"] <= 50)).sum())])
        totals += counts
        site_body.append([brain, *counts.tolist()])

        missed = zero | unlabeled
        far = missed & ~(site["cache_cable_um"] <= REACH_UM)
        why_body.append([brain, int(missed.sum()),
                         int((missed & (site["cache_cable_um"] <= 50)).sum()),
                         int((missed & (site["cache_cable_um"] > 50) & (site["cache_cable_um"] <= REACH_UM)).sum()),
                         int(far.sum()), int((far & (site["gt_run"] < passthru)).sum()),
                         int((far & (site["gt_um"] > 6)).sum()),
                         f"{np.nanmedian(site['added_cable_um'][missed]):.0f}" if missed.any() else "",
                         f"{np.nanmedian(site['added_cable_um'][positive]):.0f}" if positive.any() else ""])

        at_site = np.zeros(len(truth), dtype=bool)
        at_site[stored["rows"][stored["canon_cable"] <= LABEL_UM]] = True
        scores = state[f"{brain}/merge:scores"]
        for name in ("junction", "tip_to_shaft"):
            chosen = family == name
            cells = [int((chosen & at_site & (truth == 1)).sum()), int((chosen & at_site & (truth == 0)).sum()),
                     int((chosen & at_site & np.isnan(truth)).sum()), int((chosen & ~at_site & (truth == 1)).sum())]
            row_body.append([brain, name, int((chosen & (truth == 1)).sum()), *cells])
            row_totals[name] = row_totals.get(name, np.zeros(5, dtype=int)) + np.array(
                [int((chosen & (truth == 1)).sum()), *cells])

        junction = family == "junction"
        labeled = junction & np.isfinite(truth)
        reference = np.sort(scores[labeled])
        rank = np.searchsorted(reference, scores, side="right") / len(reference)
        ranks["labeled positive"].append(rank[junction & (truth == 1)])
        ranks["label 0 at a canonical site"].append(rank[junction & at_site & (truth == 0)])
        ranks["unlabeled at a canonical site"].append(rank[junction & at_site & np.isnan(truth)])
        ranks["other label 0"].append(rank[junction & ~at_site & (truth == 0)])

        evidence = np.load(EVIDENCE / f"evidence_{brain}.npz", allow_pickle=False)
        flips = set(np.flatnonzero(junction & at_site & (truth == 0)).tolist())
        for row, category in zip(evidence["rows"].tolist(), evidence["category"].tolist()):
            if row in flips:
                category_counts[category] += 1

        revised = truth.copy()
        revised[at_site] = 1.0
        cells = [brain]
        for labels in (truth, revised):
            use = junction & np.isfinite(labels)
            y, s = labels[use].astype(int), scores[use]
            k = int(y.sum())
            hits = int(y[np.argsort(-s, kind="stable")[:k]].sum())
            cells += [f"{k} / {int((y == 0).sum())}", f"{roc_auc_score(y, s):.3f}", f"{average_precision_score(y, s):.3f}",
                      f"{hits} / {k}"]
        metric_body.append(cells)

    site_body.append(["all", *totals.tolist()])
    lines += ["## Canonical sites by what the candidate pool and the labels make of them", "",
              *table(["brain", "canonical sites", "not on a fragment of their segment (50 um)",
                      f"a positive candidate within {LABEL_UM:g} um", "candidates, none positive, some labeled 0",
                      "candidates, all unlabeled", f"no candidate within {LABEL_UM:g} um",
                      "of those, a candidate within 50 um"], site_body), ""]
    for name, values in row_totals.items():
        row_body.append(["all", name, *values.tolist()])
    lines += ["## Candidates and their current labels", "",
              *table(["brain", "family", "positives now", "at a canonical site: label 1", "label 0", "unlabeled",
                      "positive now, no canonical site within 20 um"], row_body), ""]
    lines += ["## Junction candidates labeled 0 at a canonical site, by the reading of the GT around them", "",
              *table(CATEGORIES, [[category_counts[c] for c in CATEGORIES]]), ""]
    body = []
    for name, values in ranks.items():
        values = np.concatenate(values)
        body.append([name, len(values), f"{values.mean():.2f}" if len(values) else "",
                     f"{np.median(values):.2f}" if len(values) else "",
                     f"{100 * (values >= 0.9).mean():.0f}%" if len(values) else ""])
    lines += ["## Rank of the accepted scorer among the labeled junction rows of the brain (1 = top)", "",
              *table(["junction rows", "rows", "mean rank", "median rank", "in the top 10%"], body), ""]
    lines += ["## Junction family, accepted scorer: current labels against labels with the canonical sites added", "",
              *table(["brain", "now: positives / negatives", "AUROC", "average precision", "Top-K hits",
                      "added: positives / negatives", "AUROC", "average precision", "Top-K hits"], metric_body), "",
              "The scorer was fitted with the current labels; only the labels it is measured against change.", ""]
    lines += ["## Canonical sites with candidates but no positive: where the cache's nearest site is", "",
              *table(["brain", "sites", "cache site within 50 um of cable", "50-150 um", "none within 150 um",
                      "of those: contact below the pass-through threshold", "of those: GT node farther than 6 um",
                      "median added cable (um), these sites", "median added cable, sites with a positive"], why_body), ""]
    (OUT / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="stage", required=True)
    p = sub.add_parser("rows", help="candidates around every canonical merge site")
    p.add_argument("--brains", nargs="+", default=TRAIN_BRAINS, help="TRAIN brains only")
    p = sub.add_parser("summarize", help="tables of sites, candidates and the scorer's ranks")
    p.add_argument("--brains", nargs="+", default=TRAIN_BRAINS)
    p.add_argument("--run", required=True, help="finished run whose accepted merge scorer is read")
    args = parser.parse_args()
    unknown = set(args.brains) - set(TRAIN_BRAINS)
    if unknown:
        parser.error(f"Not TRAIN brains: {sorted(unknown)}; this analysis must not open validation or reserved brains")
    if args.stage == "rows":
        for brain in args.brains:
            brain_rows(brain)
    else:
        summarize(args)


if __name__ == "__main__":
    main()
