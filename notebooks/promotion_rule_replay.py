#!/usr/bin/env python3
"""Replay of the rule gate and the TRAIN selection-noise check on finished evolution runs.

Questions (2026-11-12): what would ``--promotion-gate rule`` have decided on the tables the
promotion judge voted on, and which submissions would ``--selection-noise-gate`` have kept away
from the validation pools?

For every generation of a run the script reads the ledger and

* applies ``promotion_judge.rule_check`` and the guard rail to the recorded table
  (``validation_gate.judge.statistics``) at each ``--margin``, and derives the rejection
  category the votes would get today (``promotion_judge.rejection_category``);
* recomputes ``SelectionProtocol.paired_noise`` of the submitted candidate against the then
  accepted parent from the saved selection scores (``gen*/experiments/attempt*/selection_state.npz``)
  at each ``--alpha``. The seed parent is a constant scorer.

Each decision is replayed with the run's own parent sequence; another gate would have changed
the later parents, so this is a comparison of single decisions, not of whole runs. The
validation numbers come from the host-side ledger; no validation pool is opened. The TRAIN
tables of the run's selection brains are loaded (raw pools, about two minutes for six brains).

Run on a compute node in the ``panda`` environment from the repository root::

    python -u notebooks/promotion_rule_replay.py \
        --runs proofreader_evolve/runs/precision_20261112_010152_bfg7kacp \
               proofreader_evolve/runs/precision_20261112_010306_tuap47i8

Output: ``notebooks/promotion_rule_replay/replay.json`` and ``summary.md`` (gitignored).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = Path(__file__).resolve().parent / "promotion_rule_replay"


def log(message):
    print(message, flush=True)


def replay(args):
    from proofreader_evolve.harness import fixed_pool_scoring as scoring, label_availability, promotion_judge
    from proofreader_evolve.harness.native_pool import cache_path
    from proofreader_evolve.harness.raw_pool import ensure_raw_tables
    from proofreader_evolve.harness.selection_protocol import SelectionProtocol, load_selection

    rows = []
    for run in map(Path, args.runs):
        manifest = json.loads((run / "manifest.json").read_text())
        split = json.loads((run / "dataset_split.json").read_text())
        if split.get("detector_prior") != "raw":
            raise SystemExit(f"{run.name}: this replay loads raw pools; the run used {split.get('detector_prior')!r}")
        brains = sorted(split["selection_brains"])
        objective = manifest["objective_statistic"]
        described = manifest["budgets"]
        if set(described["positive_ratio"].values()) != {1.} or any(described["caps"].values()):
            raise SystemExit(f"{run.name}: this replay assumes K = GT positives without caps")
        budgets = scoring.Budgets({"merge": None, "split": None}, positive_ratio=1.,
                                  min_positives=described["min_positives"], aggregation=described["aggregation"],
                                  objective=objective)
        started = time.monotonic()
        train = {brain: ensure_raw_tables(brain, cache_path(brain, 100), ROOT / "proofreader_evolve" / "feature_tables",
                                          prepare=False, mcl=100) for brain in brains}
        label_availability.apply(train)
        protocol = SelectionProtocol(train, budgets, selection_brains=brains,
                                     holdout_brains=sorted(split.get("selection_holdout_brains") or ()))
        log(f"[{run.name}] {len(brains)} selection brains loaded [{time.monotonic() - started:.0f}s]")
        parent = {kind: {f"{brain}/{kind}": {"scores": np.zeros(len(train[brain].tables[kind].truth))} for brain in brains}
                  for kind in ("merge", "split")}
        for row in map(json.loads, (run / "ledger.jsonl").read_text().splitlines()):
            kind, generation, experiment = row["target_kind"], row["generation"], row.get("submitted_experiment")
            gate = row.get("validation_gate") or {}
            judge = gate.get("judge") or {}
            line = {"run": run.name, "generation": generation, "kind": kind, "experiment": experiment,
                    "accepted": bool(row["accepted"]), "gate_status": judge.get("status") or gate.get("skipped"),
                    "accept_votes": judge.get("accept_votes"), "votes_cast": judge.get("votes_cast"),
                    "recorded_category": judge.get("rejection_category"),
                    "selection_gain": (row.get("selection_gate") or {}).get("delta")}
            statistics = judge.get("statistics")
            if statistics:
                name = promotion_judge.rule_check(statistics, objective)["statistic"]
                line["validation_gain"] = statistics[name]["delta"]
                line["validation_interval"] = [statistics[name]["low"], statistics[name]["high"]]
                line["votes_category_now"] = (None if row["accepted"] or not judge.get("votes_cast")
                                              and judge.get("status") != "guard_rail"
                                              else promotion_judge.rejection_category(statistics))
                line["rule"] = {}
                for margin in args.margin:
                    rule = promotion_judge.rule_check(statistics, objective, margin)
                    passed = promotion_judge.guard_rail(statistics) and rule["passed"]
                    line["rule"][f"{margin:g}"] = "accept" if passed else promotion_judge.rejection_category(statistics, rule)
            if experiment:
                generation_dir, attempt = experiment.split("/")
                _, state = load_selection(run / generation_dir / "experiments" / attempt)
                line["selection_noise"] = {}
                for alpha in args.alpha:
                    tick = time.monotonic()
                    noise = protocol.paired_noise(kind, parent[kind], state, draws=args.draws, seed=generation, alpha=alpha)
                    line["selection_noise"][f"{alpha:g}"] = {key: noise.get(key) for key in
                                                             ("status", "delta_ci", "beyond_noise", "reason")}
                    line["noise_seconds"] = round(time.monotonic() - tick, 1)
                if row["accepted"]:
                    parent[kind] = state
            rows.append(line)
            log(json.dumps(line))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "replay.json").write_text(json.dumps(rows, indent=1))
    summarize(rows, args)


def summarize(rows, args):
    def number(value):
        return "" if value is None else f"{value:+.4f}"

    lines = ["# Rule gate and TRAIN selection-noise check replayed on finished runs", "",
             "Script `notebooks/promotion_rule_replay.py`. Each decision uses the run's own parent sequence.", "",
             "| run | gen | kind | recorded outcome | votes | validation gain of the objective | 95% interval | "
             + " | ".join(f"rule, margin {margin:g}" for margin in args.margin)
             + " | category under the votes, today | TRAIN selection gain | "
             + " | ".join(f"TRAIN {100 * (1 - alpha):g}% interval | beyond noise" for alpha in args.alpha) + " |",
             "|" + "---|" * (10 + len(args.margin) + 2 * len(args.alpha) - 1)]
    for row in rows:
        outcome = "accepted" if row["accepted"] else (row["gate_status"] or "failed")
        votes = "" if row["accept_votes"] is None or row["gate_status"] != "voted" else f"{row['accept_votes']}/{row['votes_cast']}"
        interval = row.get("validation_interval")
        cells = [row["run"].rsplit("_", 1)[-1], str(row["generation"]), row["kind"], outcome, votes,
                 number(row.get("validation_gain")),
                 "" if not interval or interval[0] is None else f"[{interval[0]:+.4f}, {interval[1]:+.4f}]"]
        cells += [(row.get("rule") or {}).get(f"{margin:g}", "") for margin in args.margin]
        cells += [row.get("votes_category_now") or "", number(row.get("selection_gain"))]
        for alpha in args.alpha:
            noise = (row.get("selection_noise") or {}).get(f"{alpha:g}") or {}
            ci = noise.get("delta_ci")
            cells += ["" if not ci else f"[{ci[0]:+.4f}, {ci[1]:+.4f}]",
                      "" if noise.get("beyond_noise") is None else ("yes" if noise["beyond_noise"] else "no")]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.md").write_text("\n".join(lines))
    log("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="+", required=True, help="finished raw-run directories")
    parser.add_argument("--margin", nargs="+", type=float, default=[0., .002], help="minimum effects of the rule")
    parser.add_argument("--alpha", nargs="+", type=float, default=[.2, .05], help="levels of the TRAIN interval")
    parser.add_argument("--draws", type=int, default=1000)
    replay(parser.parse_args())


if __name__ == "__main__":
    main()
