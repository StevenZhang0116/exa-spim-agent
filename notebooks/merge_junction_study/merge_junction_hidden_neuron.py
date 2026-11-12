#!/usr/bin/env python3
"""How often is a merge candidate that is labeled 0 away from any tracing really a merge?

Background (``proofreader_evolve/TODO.md`` section 7, ``merge_junction_label_localization.py``):
the merge label is 0 when the candidate's segment carries a GT label and a GT node lies within
150 um (``label_availability``, definition ``near_gt_150``); it does not ask whether a tracing
runs through the candidate. About 13% of the junction negatives sit on a part of the fragment
that no tracing follows. Whether those rows should stay 0 or become NaN depends on how many of
them are real merges, and the tracing cannot say so for the rows themselves.

It can say so for stand-ins. Take a candidate that lies on the tracing of exactly one GT neuron
B (no other neuron's tracing within reach). Its label is known. Now suppose B had not been
traced: the candidate would have no tracing at all, and the current rule would still label it 0
if its segment also carries the label of another GT neuron and a node of another GT neuron lies
within 150 um. Those candidates are what the off-tracing negatives look like when the truth is
known: junctions on the far side of a segment shared with a traced neuron. The share of
positives among them estimates the share of unrecorded merges among the off-tracing negatives.

``rows``
    For every GT-labeled merge candidate: the GT neurons with a node within ``GT_UM`` of a
    fragment node within ``REACH_UM`` of cable of the candidate ("near"), and those of them
    whose tracing carries the candidate's segment label over at least ``passthru_min_cc``
    connected GT nodes ("on", the cache's own pass-through threshold). For a candidate on
    exactly one neuron and near no other: whether another GT neuron labels the same segment,
    and the distance to the nearest node of another GT neuron.

``summarize``
    Per brain and family: the positive rate of all candidates on a tracing, of the candidates
    on exactly one neuron, and of the stand-ins (those that the current rule would label 0 with
    that neuron hidden), with a Wilson 95% interval; and, for scale, the number of off-tracing
    negatives the stand-ins speak for.

Run on a compute node in the ``panda`` environment from the repository root::

    python -u notebooks/merge_junction_study/merge_junction_hidden_neuron.py rows --brains 747807 802449
    python -u notebooks/merge_junction_study/merge_junction_hidden_neuron.py summarize

Outputs under ``notebooks/merge_junction_study/merge_junction_hidden_neuron/`` (gitignored):
``rows_<brain>.npz``, ``summary.csv``, ``summary.md``. Read-only for caches and tables.

Limits. TRAIN brains only. A stand-in sits on a neuron somebody chose to trace, on a segment
that two traced neurons share; an untraced neurite beside a traced one need not merge at the
same rate. Its label follows the labeling convention (positive within 20 um of cable of a
recorded site), so the estimate is of labels the convention would give, not of independently
verified merges. The positives among the stand-ins exclude the junction where the two traced
neurons meet (it is near both tracings), as they should: that junction keeps its tracing when
one neuron is hidden.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from merge_junction_fragment_context import TRAIN_BRAINS, log  # noqa: E402
from merge_junction_label_localization import GT_UM, same_label_run, step  # noqa: E402

OUT = Path(__file__).resolve().parent / "merge_junction_hidden_neuron"
REACH_UM = 20.0     # cable from the candidate within which a tracing counts (the positive label radius)
NEAR_GT_UM = 150.0  # label_availability.NEAR_GT_UM


def tracings_at(graph, xyz, gt, gt_tree, gt_neuron, gt_label, junction, segment_id, passthru):
    """GT neurons near the candidate, and those whose tracing carries its segment there."""
    nodes, stack = [junction], [(junction, -1, 0.0)]
    while stack:  # the fragment graph is a forest
        node, previous, travelled = stack.pop()
        for following in graph.adj[node]:
            following = int(following)
            reached = travelled + step(xyz, following, node)
            if following != previous and reached <= REACH_UM:
                nodes.append(following)
                stack.append((following, node, reached))
    near, on = set(), set()
    for found in gt_tree.query_ball_point(xyz[nodes], GT_UM):
        for g in found:
            neuron = int(gt_neuron[g])
            near.add(neuron)
            if neuron not in on and same_label_run(gt, gt_label, int(g), segment_id, cap=passthru) >= passthru:
                on.add(neuron)
    return near, on


def brain_rows(brain):
    from scipy.spatial import cKDTree
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
    candidates = [table.candidates[i] for i in rows]
    graph, gt, payload = load_cached_graphs(str(cache_path(brain, 100)), expect_brain=brain, expect_mcl=100)
    xyz = np.asarray(graph.node_xyz, dtype=np.float64)
    gt_xyz = np.asarray(gt.node_xyz, dtype=np.float64)
    gt_neuron = np.asarray(gt.node_component_id, dtype=np.int64)
    gt_label = np.asarray(payload["gt_node_canonical_label"], dtype=np.int64)
    passthru = int(payload["gt_label_policy"]["passthru_min_cc"])
    gt_tree = cKDTree(gt_xyz)
    neurons = np.unique(gt_neuron)
    trees = {int(n): cKDTree(gt_xyz[gt_neuron == n]) for n in neurons}
    labelled = gt_label != 0
    segment_neurons = {}
    for segment, neuron in set(zip(gt_label[labelled].tolist(), gt_neuron[labelled].tolist())):
        segment_neurons.setdefault(segment, set()).add(neuron)
    log(f"[{brain}] {len(rows)} labeled merge rows ({int(truth[rows].sum())} positives), {len(neurons)} GT neurons, "
        f"pass-through threshold {passthru} [{time.monotonic() - started:.0f}s]")

    n_near = np.zeros(len(rows), dtype=np.int16)
    n_on = np.zeros(len(rows), dtype=np.int16)
    neuron_of = np.full(len(rows), -1, dtype=np.int64)       # the one neuron a candidate is on, if exactly one
    shared = np.zeros(len(rows), dtype=bool)                 # another GT neuron labels the same segment
    other_um = np.full(len(rows), np.inf)                    # nearest node of another GT neuron
    for position, candidate in enumerate(candidates):
        junction, segment_id = int(candidate["node_id"]), int(candidate["segment_id"])
        near, on = tracings_at(graph, xyz, gt, gt_tree, gt_neuron, gt_label, junction, segment_id, passthru)
        n_near[position], n_on[position] = len(near), len(on)
        if len(near) == 1 and len(on) == 1:
            hidden = next(iter(on))
            neuron_of[position] = hidden
            shared[position] = bool(segment_neurons.get(segment_id, set()) - {hidden})
            others = [tree.query(xyz[junction])[0] for neuron, tree in trees.items() if neuron != hidden]
            other_um[position] = min(others) if others else np.inf
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"rows_{brain}.npz", rows=rows, truth=truth[rows].astype(np.int8),
                        family=np.array([c["candidate_family"] for c in candidates]),
                        segment=np.array([c["segment_id"] for c in candidates], dtype=np.int64),
                        n_near=n_near, n_on=n_on, neuron=neuron_of, shared=shared, other_um=other_um)
    stand_in = (neuron_of >= 0) & shared & (other_um <= NEAR_GT_UM)
    y = truth[rows].astype(int)
    log(f"[{brain}] on a tracing {int((n_on > 0).sum())}, on exactly one neuron {int((neuron_of >= 0).sum())}, "
        f"stand-ins {int(stand_in.sum())} ({int(y[stand_in].sum())} positives) [{time.monotonic() - started:.0f}s]")


def wilson(positives, total):
    if not total:
        return ""
    z, p = 1.96, positives / total
    centre = (p + z * z / (2 * total)) / (1 + z * z / total)
    half = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / (1 + z * z / total)
    return f"{100 * p:.1f}% ({100 * max(centre - half, 0):.1f}-{100 * min(centre + half, 1):.1f})"


def summarize(args):
    records, lines = [], [
        "# Positive rate of merge candidates that would be off-tracing negatives if their neuron were not traced", "",
        f"Script `notebooks/merge_junction_study/merge_junction_hidden_neuron.py`; a tracing counts within {GT_UM:g} um "
        f"of a fragment node within {REACH_UM:g} um of cable of the candidate; the hidden-neuron rule is "
        f"`near_gt_150` (segment labeled by another GT neuron, a node of another GT neuron within {NEAR_GT_UM:g} um). "
        "Rates with Wilson 95% intervals.", ""]
    pooled = {}
    for family in ("junction", "tip_to_shaft"):
        body = []
        for brain in [*args.brains, "all"]:
            if brain == "all":
                data = pooled[family]
            else:
                stored = np.load(OUT / f"rows_{brain}.npz", allow_pickle=False)
                chosen = stored["family"] == family
                data = {key: stored[key][chosen] for key in ("truth", "n_on", "neuron", "shared", "other_um", "segment")}
                data["brain"] = np.full(int(chosen.sum()), brain)
                store = pooled.setdefault(family, {key: [] for key in data})
                for key, value in data.items():
                    store[key].append(value)
            if brain == "all":
                data = {key: np.concatenate(value) for key, value in data.items()}
            y = data["truth"].astype(int)
            on, one = data["n_on"] > 0, data["neuron"] >= 0
            stand_in = one & data["shared"] & (data["other_um"] <= NEAR_GT_UM)
            groups = {"on a tracing": on, "on exactly one neuron": one, "stand-ins": stand_in,
                      "one neuron, not a stand-in": one & ~stand_in}
            hidden = len(set(zip(data["brain"][stand_in].tolist(), data["neuron"][stand_in].tolist())))
            segments = len(set(zip(data["brain"][stand_in].tolist(), data["segment"][stand_in].tolist())))
            record = {"family": family, "brain": brain, "off_tracing_negatives": int((~on & (y == 0)).sum()),
                      "off_tracing_positives": int((~on & (y == 1)).sum()),
                      "stand_in_neurons": hidden, "stand_in_segments": segments}
            row = [brain]
            for name, mask in groups.items():
                key = name.replace(" ", "_").replace(",", "").replace("-", "_")
                record[f"{key}_rows"], record[f"{key}_positives"] = int(mask.sum()), int(y[mask].sum())
                row += [f"{int(y[mask].sum())} / {int(mask.sum())}", wilson(int(y[mask].sum()), int(mask.sum()))]
            row += [hidden, segments, record["off_tracing_negatives"]]
            records.append(record)
            body.append(row)
        header = ["brain"]
        for name in ("on a tracing", "on exactly one neuron", "stand-ins", "one neuron, not a stand-in"):
            header += [f"{name}: positives / rows", "rate"]
        header += ["hidden neurons", "segments", "off-tracing negatives (current labels)"]
        lines += [f"## {family}", "", "| " + " | ".join(header) + " |", "|" + "---|" * len(header),
                  *["| " + " | ".join(str(c) for c in row) + " |" for row in body], ""]
    with (OUT / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    (OUT / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="stage", required=True)
    p = sub.add_parser("rows", help="the tracings at every labeled merge candidate")
    p.add_argument("--brains", nargs="+", default=TRAIN_BRAINS, help="TRAIN brains only")
    p = sub.add_parser("summarize", help="positive rates of the stand-ins")
    p.add_argument("--brains", nargs="+", default=TRAIN_BRAINS)
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
