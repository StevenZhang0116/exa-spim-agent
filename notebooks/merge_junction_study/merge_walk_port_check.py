#!/usr/bin/env python3
"""What the ``length_v2`` label policy would change, computed in memory before any cache is rewritten.

``length_v2`` (``labeled_cache.policy``, 2026-11-11) relabels the merge sites with the metric's
walk: one GT neuron at a time, the site moved to a branching node
(``canonical_labeling.geometric_merge_sites(walk="per_neuron")``). ``merge_site_walk_replay.py``
predicted the effect with a replay written for that purpose; this script runs the package's own
functions, the way ``BrainDataset.refresh_merge_labels`` does, on each TRAIN cache held in memory,
labels the benchmark candidates from the result with ``candidate_rules.build_merge_universe`` and
compares with the stored labels. Nothing is written to a cache or a table.

``check``
    Per brain: the stored and the recomputed sites by source; the candidates' labels now and
    after (positives per family, rows that change in either direction); and, per canonical site
    (``metrics_out/<brain>/*/merge_sites.csv``), whether a positive candidate lies within 20 um
    of fragment cable now and after.

``summarize``
    The table over the brains.

Run on a compute node in the ``panda`` environment from the repository root::

    python -u notebooks/merge_junction_study/merge_walk_port_check.py check --brains 747807 802449
    python -u notebooks/merge_junction_study/merge_walk_port_check.py summarize

Outputs under ``notebooks/merge_junction_study/merge_walk_port_check/`` (gitignored):
``check_<brain>.json``, ``summary.md``. TRAIN brains only; read-only.

The stored outputs are from 2026-11-11, before the relabel. The twelve caches carry ``length_v2``
labels since that evening, so a rerun compares those labels with themselves and reports no change.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for entry in (str(ROOT), str(ROOT / "notebooks")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from merge_canonical_site_labels import LABEL_UM, cable_from, snap  # noqa: E402
from merge_junction_fragment_context import TRAIN_BRAINS, log  # noqa: E402

OUT = Path(__file__).resolve().parent / "merge_walk_port_check"


def brain_check(brain):
    from scipy.spatial import cKDTree
    from agentic_neuron_proofreader.data_modules import canonical_labeling
    from candidate_rules import SITE_SNAP_MAX_UM, build_merge_universe, segment_of_component
    from labeled_cache.policy import POLICY_LENGTH_V2, recorded_thresholds, walk_for
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
    family = np.array([c["candidate_family"] for c in table.candidates])
    row_of = {int(c["node_id"]): i for i, c in enumerate(table.candidates)}
    graph, gt, payload = load_cached_graphs(str(cache_path(brain, 100)), expect_brain=brain, expect_mcl=100)
    labels = np.asarray(payload["gt_node_canonical_label"])
    thresholds = recorded_thresholds(payload["gt_label_policy"])
    stored = list(payload["gt_merge_sites"])
    gt.set_kdtree() if getattr(gt, "kdtree", None) is None else None
    walk = walk_for(POLICY_LENGTH_V2)
    _, geometric = canonical_labeling.geometric_merge_sites(graph, gt, labels, verbose=False, thresholds=thresholds,
                                                            walk=walk)
    sites, _ = canonical_labeling.combined_merge_sites(graph, gt, labels, geometric, verbose=False,
                                                       thresholds=thresholds)
    log(f"[{brain}] sites stored {len(stored)}, recomputed {len(sites)} ({len(geometric)} from the walk) "
        f"[{time.monotonic() - started:.0f}s]")

    rule = json.loads((ROOT / "proofreader_evolve" / "candidate_pools.json").read_text())["pools"]["merge"]
    rule = {key: rule[key] for key in ("config_id", "mode", "nms_um", "tip_gap_um", "tip_partner_k",
                                       "claim_radius_um", "positive_label_radius_um")}
    samples, new_labels, _ = build_merge_universe({**payload, "gt_merge_sites": sites}, rule)
    if len(samples) != len(table.candidates):
        raise ValueError(f"{brain}: {len(samples)} candidates enumerated, the table holds {len(table.candidates)}")
    positive = np.zeros(len(truth), dtype=bool)
    for sample, label in zip(samples, new_labels):
        positive[row_of[int(sample["node_id"])]] = bool(label)
    result = {"brain": brain, "walk": walk, "thresholds": thresholds.as_record(),
              "sites": {"stored": dict(Counter(str(s.get("source", "geometric_walk")) for s in stored)),
                        "recomputed": dict(Counter(str(s.get("source", "geometric_walk")) for s in sites))}}
    for name in ("junction", "tip_to_shaft"):
        chosen = family == name
        result[name] = {"positive_now": int((chosen & (truth == 1)).sum()), "positive_after": int((chosen & positive).sum()),
                        "zero_to_positive": int((chosen & positive & (truth == 0)).sum()),
                        "unlabeled_to_positive": int((chosen & positive & np.isnan(truth)).sum()),
                        "positive_to_not": int((chosen & ~positive & (truth == 1)).sum())}

    xyz = np.asarray(graph.node_xyz, dtype=np.float64)
    component = np.asarray(graph.node_component_id, dtype=np.int64)
    to_segment = np.full(int(component.max()) + 1, -1, dtype=np.int64)
    for key, value in segment_of_component(graph).items():
        to_segment[int(key)] = int(value)
    segment = to_segment[component]
    canonical = load_canonical(ROOT / "metrics_out", brain, payload["anisotropy"])
    kept = np.flatnonzero(np.isin(segment, np.unique(canonical["segment"])))
    tree = cKDTree(xyz[kept])
    counts = Counter()
    for point, segment_id in zip(canonical["xyz"], canonical["segment"].tolist()):
        node = snap(tree, kept, xyz, segment, point, int(segment_id), SITE_SNAP_MAX_UM)
        if node < 0:
            counts["unsnappable"] += 1
            continue
        rows = [row_of[n] for n, distance in cable_from(graph, xyz, node, LABEL_UM).items() if n in row_of]
        if not rows:
            counts["no_candidate"] += 1
            continue
        now, after = bool((truth[rows] == 1).any()), bool(positive[rows].any())
        counts[f"positive_now_{now}_after_{after}"] += 1
    result["canonical_sites"] = {"total": len(canonical["xyz"]), **dict(counts)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"check_{brain}.json").write_text(json.dumps(result, indent=1))
    log(f"[{brain}] junction {result['junction']}; canonical {result['canonical_sites']} [{time.monotonic() - started:.0f}s]")


def summarize(args):
    results = [json.loads((OUT / f"check_{brain}.json").read_text()) for brain in args.brains]
    lines = ["# Merge labels under `length_v2` (the metric's walk), computed in memory on the TRAIN caches", "",
             "Script `notebooks/merge_junction_study/merge_walk_port_check.py`; the package's "
             "`geometric_merge_sites(walk=\"per_neuron\")` and `combined_merge_sites`, labels from "
             "`candidate_rules.build_merge_universe`. No cache or table was written.", "",
             "| brain | sites stored | recomputed (walk + two-GT only) | junction positives now | after | 0 to 1 | 1 to not "
             "| tip_to_shaft positives now | after | 0 to 1 | 1 to not | canonical sites | a positive candidate now "
             "| after | gained | lost | candidates, none positive after | no candidate / unsnappable |",
             "|" + "---|" * 18]
    total = np.zeros(16, dtype=int)
    for result in results:
        recomputed = result["sites"]["recomputed"]
        junction, tip, canonical = result["junction"], result["tip_to_shaft"], result["canonical_sites"]
        both, gained = canonical.get("positive_now_True_after_True", 0), canonical.get("positive_now_False_after_True", 0)
        lost, neither = canonical.get("positive_now_True_after_False", 0), canonical.get("positive_now_False_after_False", 0)
        row = [sum(result["sites"]["stored"].values()), sum(recomputed.values()),
               junction["positive_now"], junction["positive_after"],
               junction["zero_to_positive"] + junction["unlabeled_to_positive"], junction["positive_to_not"],
               tip["positive_now"], tip["positive_after"], tip["zero_to_positive"] + tip["unlabeled_to_positive"],
               tip["positive_to_not"], canonical["total"], both + lost, both + gained, gained, lost, neither]
        total += np.array(row)
        other = canonical.get("no_candidate", 0) + canonical.get("unsnappable", 0)
        lines.append(f"| {result['brain']} | {row[0]} | {row[1]} ({recomputed.get('geometric_walk', 0)} + "
                     f"{recomputed.get('two_gt_junction', 0)}) | " + " | ".join(str(v) for v in row[2:]) + f" | {other} |")
    lines.append("| all | " + " | ".join(str(v) for v in total.tolist()) + " | |")
    lines.append("")
    (OUT / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="stage", required=True)
    for name in ("check", "summarize"):
        p = sub.add_parser(name)
        p.add_argument("--brains", nargs="+", default=TRAIN_BRAINS, help="TRAIN brains only")
    args = parser.parse_args()
    unknown = set(args.brains) - set(TRAIN_BRAINS)
    if unknown:
        parser.error(f"Not TRAIN brains: {sorted(unknown)}; this analysis must not open validation or reserved brains")
    if args.stage == "check":
        for brain in args.brains:
            brain_check(brain)
    else:
        summarize(args)


if __name__ == "__main__":
    main()
