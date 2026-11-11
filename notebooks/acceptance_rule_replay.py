#!/usr/bin/env python3
"""Offline replay of promotion rules on the validated candidates of finished evolution runs.

Question: the promotion gate compares one statistic (AUROC since 2026-11-02) of the candidate
with the accepted scorer on the development-validation pools. Would another fixed rule, or an
LLM judge shown several statistics, decide better? This script replays four kinds of rule on
candidates that finished runs already validated, and measures every rule's false-acceptance
rate on null pairs whose true difference is zero by construction.

Three stages, each a subcommand writing under ``notebooks/acceptance_rule_replay/`` (gitignored):

``rescore``
    Validation scores are held in memory during a run and are not saved, so every validated
    candidate (``ledger.jsonl`` rows with a validation report) and the seed are scored again on
    the validation pools with the driver's own code path: raw pools, three-valued labels,
    descriptor columns registered from the descriptor bank (cache hits; nothing is recomputed
    unless ``--allow-compute``), the submitted ``combined_scorer.py`` and the run's
    ``model_artifacts``. Saves the scores of the GT-labeled rows per cell and checks every
    recomputed kind statistic against the ledger (``check_<kind>.json``).

``replay``
    For each candidate against the scorer that was accepted when it was submitted: one joint
    paired Poisson(1) bootstrap (the row weights of ``fixed_pool_scoring.paired_bootstrap``, the
    statistics of ``rank_statistics.ranking_statistics``) gives the delta and its 95% interval
    for AUROC, pAUC, average precision, precision at K/2, K, 2K and their mean, under the run's
    positive-weighted aggregation. With ``seed = generation`` the AUROC interval reproduces the
    gate's recorded bounds. Fixed rules are then applied (``RULES``). Null pairs: for a real
    pair (A, B) the within-cell rank-normalized scores are swapped row by row with probability
    one half, giving two hybrids C1 = (A or B) and C2 = (B or A) that are exchangeable, so
    their expected difference is exactly zero while their row-level disagreement is that of two
    real models; every acceptance of C2 over C1 is a false acceptance.

``judge``
    The promotion judge of ``proofreader_evolve/harness/promotion_judge.py`` (the module behind
    ``--promotion-gate judge``; this script imports its prompt, statistics and guard rail, so
    what is replayed here is what a run applies). An LLM (``claude-opus-5-5``) sees only a table of the statistics (parent, candidate, delta,
    and in one variant the bootstrap intervals), never run ids, code or the agent's summary,
    and votes accept / reject several times per item. Reported: the majority with intervals
    behind a host guard rail (no statistic significantly worse), the raw majority with
    intervals, and the majority without intervals. Needs ``ANTHROPIC_API_KEY``.

Run on a compute node in the ``panda`` environment from the repository root, for example::

    python -u notebooks/acceptance_rule_replay.py rescore --runs <run_dir> <run_dir> --kinds merge
    python -u notebooks/acceptance_rule_replay.py replay
    python -u notebooks/acceptance_rule_replay.py judge
    python -u notebooks/acceptance_rule_replay.py summary

Read-only for the runs, the caches and the tables. The descriptor bank is read; with
``--allow-compute`` a missing entry would be computed and written there like in a run.
The validation brains are development data: this replay is a study of decision rules on
candidates the gate already judged, not an independent test of any scorer.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from proofreader_evolve.harness import promotion_judge  # noqa: E402  (the gate's own judge and statistics)

OUT = Path(__file__).resolve().parent / "acceptance_rule_replay"
STATISTICS = promotion_judge.STATISTICS
HEAD_TOLERANCE = 0.01
JUDGE_MODEL = promotion_judge.DEFAULT_MODEL


def log(message):
    print(time.strftime("%H:%M:%S"), message, flush=True)


# --------------------------------------------------------------------------------------------
# Stage 1: rescore the validated candidates on the validation pools
# --------------------------------------------------------------------------------------------

def validated_rows(run_dir):
    rows = [json.loads(line) for line in (run_dir / "ledger.jsonl").open()]
    return [row for row in rows if (row.get("validation") or {}).get("cells")]


def descriptor_programs(run_dir):
    """code_sha256 -> {program, spec, scope, generation} from every generation's descriptor runs,
    plus the run-level registry of the last accepted scorer."""
    from proofreader_evolve.harness.descriptor_contract import descriptor_spec
    programs = {}
    for directory in sorted(run_dir.glob("gen*/descriptor_runs/run*")):
        if not (directory / "descriptor.py").is_file() or not (directory / "result.json").is_file():
            continue
        result = json.loads((directory / "result.json").read_text())
        if result.get("status") != "registered":
            continue
        program, spec = descriptor_spec((directory / "descriptor.py").read_text())
        if spec["code_sha256"] != result["code_sha256"]:
            raise ValueError(f"{directory}: descriptor.py does not hash to the recorded code")
        programs.setdefault(spec["code_sha256"], {
            "program": program, "spec": spec, "scope": result["scope"],
            "generation": int(directory.parts[-3][3:])})
    registry_path = run_dir / "descriptors.json"
    registry = json.loads(registry_path.read_text()) if registry_path.is_file() else {"columns": {}, "programs": {}}
    for code, stored in registry["programs"].items():
        program, spec = descriptor_spec(stored["program"])
        scope = next(r["scope"] for r in registry["columns"].values() if r["code_sha256"] == code)
        programs.setdefault(code, {"program": program, "spec": spec, "scope": scope, "generation": 0})
    return programs, {column: record["code_sha256"] for column, record in registry["columns"].items()}


def column_codes(columns, kind, programs, registry_columns, generation):
    """Code of every referenced column of ``kind``: the registry's when it names the column, else
    the latest registered program up to this generation that produces it."""
    codes = {}
    for column in columns:
        owners = [code for code, info in programs.items()
                  if column in info["spec"]["columns"] and info["generation"] <= generation]
        if not owners:
            raise ValueError(f"No registered descriptor program produces {column}")
        if programs[owners[0]]["spec"]["kind"] != kind:
            continue
        code = registry_columns.get(column)
        if code not in owners:
            code = max(owners, key=lambda c: programs[c]["generation"])
        codes[column] = code
    return codes


def rescore(args):
    from proofreader_evolve.cli import run_precision_evolution as driver
    from proofreader_evolve.harness import detector_prior, label_availability
    from proofreader_evolve.harness import fixed_pool_scoring as scoring
    from proofreader_evolve.harness.descriptor_context import DescriptorContexts
    from proofreader_evolve.harness.descriptor_runs import DescriptorRuns
    from proofreader_evolve.harness.image_context import attach_image_context
    from proofreader_evolve.harness.local_context import attach_local_context
    from proofreader_evolve.harness.native_pool import cache_path
    from proofreader_evolve.harness.rank_statistics import labeled_rows
    from proofreader_evolve.harness.raw_pool import ensure_raw_tables
    from proofreader_evolve.harness.trajectory import Trajectory

    scratch = args.scratch.resolve()
    scratch.mkdir(parents=True, exist_ok=True)
    (OUT / "scores").mkdir(parents=True, exist_ok=True)
    runs = [Path(r).resolve() for r in args.runs]
    launch = json.loads((runs[0] / "launch.json").read_text())
    for run in runs[1:]:
        other = json.loads((run / "launch.json").read_text())["evolution_args"]
        if other != launch["evolution_args"]:
            raise ValueError("The runs were launched with different arguments; replay them separately")
    run_args = driver.parse_args(list(launch["evolution_args"]))
    if run_args.detector_prior != "raw":
        raise ValueError("This replay loads raw pools; the runs must use --detector-prior raw")
    trace = Trajectory(scratch)
    banks = {}
    for brain in run_args.validation_brains:
        log(f"loading raw pools of {brain}")
        banks[brain] = ensure_raw_tables(brain, cache_path(brain, run_args.mcl), run_args.feature_tables_dir,
                                         prepare=False, mcl=run_args.mcl)
    label_availability.apply(banks)
    attach_local_context(banks, scratch / "local_feature_cache", trace)
    attach_image_context(banks, scratch / "image_cache", run_args.image_alignment, trace)
    detector_prior.strip_to_raw(banks)
    head_guard = (None if run_args.head_guard_statistic == "none" else
                  {"statistic": run_args.head_guard_statistic, "tolerance": run_args.head_guard_tolerance})
    budgets = scoring.Budgets({"merge": run_args.merge_k, "split": run_args.split_k},
                              positive_ratio=run_args.k_positive_ratio, min_positives=run_args.min_positives,
                              aggregation=run_args.aggregation, objective=run_args.objective,
                              pauc_fpr=run_args.pauc_fpr, head_guard=head_guard)
    descriptor_runs = DescriptorRuns(DescriptorContexts(), run_args.descriptor_bank,
                                     workers=run_args.descriptor_workers,
                                     memory_mb=run_args.descriptor_memory_mb, trace=trace)
    raw_frames = {(brain, kind): table.features for brain, bank in banks.items()
                  for kind, table in bank.tables.items()}

    def view(kind):
        out = {}
        for brain, bank in banks.items():
            shallow = copy.copy(bank)
            shallow.tables = {kind: bank.tables[kind]}
            out[brain] = shallow
        return out

    def reset(kind):
        for brain, bank in banks.items():
            bank.tables[kind].features = raw_frames[(brain, kind)].copy()

    def save(name, kind, state, report):
        arrays = {}
        for brain, bank in banks.items():
            table = bank.tables[kind]
            rows = labeled_rows(np.asarray(table.truth, dtype=float))
            arrays[brain] = np.asarray(state[f"{brain}/{kind}"]["scores"], dtype=float)[rows]
        np.savez_compressed(OUT / "scores" / f"{name}_{kind}.npz", **arrays)
        return {s: report["per_kind_statistics"][kind].get(s) for s in STATISTICS}

    for kind in args.kinds:
        check = {}
        cells = {}
        arrays = {}
        for brain, bank in banks.items():
            table = bank.tables[kind]
            truth = np.asarray(table.truth, dtype=float)
            rows = labeled_rows(truth)
            arrays[f"{brain}:truth"] = truth[rows].astype(np.int8)
            arrays[f"{brain}:keys"] = np.asarray(table.keys, dtype=str)[rows]
            cells[brain] = {"k": int(budgets.k_for(kind, table.truth)), "in_mean": bool(budgets.in_mean(table.truth)),
                            "positives": int(np.nansum(truth)), "labeled": int(len(rows)), "pool": int(len(truth))}
        np.savez_compressed(OUT / f"cells_{kind}.npz", **arrays)
        (OUT / f"cells_{kind}.json").write_text(json.dumps(
            {"brains": list(banks), "cells": cells, "pauc_fpr": budgets.pauc_fpr,
             "aggregation": run_args.aggregation, "min_positives": run_args.min_positives}, indent=1))
        log(f"[{kind}] cells: " + ", ".join(f"{b} K={c['k']} labeled={c['labeled']}" for b, c in cells.items()))
        reset(kind)
        state = {}
        started = time.monotonic()
        report = scoring.evaluate(run_args.start_from.read_text(), view(kind), budgets, run_args.policy_time_budget,
                                  state=state, memory_mb=run_args.classifier_memory_mb)
        check["seed"] = {"recomputed": save("seed", kind, state, report)}
        log(f"[{kind}] seed: AUROC {check['seed']['recomputed']['auroc']} [{time.monotonic() - started:.0f}s]")
        for run in runs:
            programs, registry_columns = descriptor_programs(run)
            for row in validated_rows(run):
                if row["target_kind"] != kind:
                    continue
                name = f"{run.name[-8:]}_gen{row['generation']:02d}"
                started = time.monotonic()
                reset(kind)
                codes = column_codes(row.get("descriptors_referenced") or [], kind, programs,
                                     registry_columns, row["generation"])
                cached = computed = 0
                for code in sorted(set(codes.values())):
                    info = programs[code]
                    for brain, bank in banks.items():
                        table = bank.tables[kind]
                        if not args.allow_compute and not descriptor_runs.directory(table, info["spec"], info["scope"]).is_dir():
                            raise RuntimeError(f"{name}: descriptor {code[:12]} is not in the bank for {brain}/{kind}; "
                                               "pass --allow-compute to compute it")
                        result = descriptor_runs.compute(
                            table, info["program"], info["spec"], info["scope"], k=budgets.k_for(kind, table.truth),
                            wall_budget=run_args.descriptor_wall_seconds,
                            log_dir=scratch / "descriptor_logs" / name / brain / code[:12])
                        cached += bool(result["cached"])
                        computed += not result["cached"]
                        DescriptorRuns.register(table, info["spec"]["columns"], result["rows"], result["values"])
                source = (run / row["submitted_experiment"].replace("/", "/experiments/", 1) / "combined_scorer.py").read_text()
                state = {}
                report = scoring.evaluate(source, view(kind), budgets, run_args.policy_time_budget, state=state,
                                          artifact_store=run / "model_artifacts", memory_mb=run_args.classifier_memory_mb)
                recomputed = save(name, kind, state, report)
                ledger = {s: row["validation"]["per_kind_statistics"][kind].get(s) for s in STATISTICS}
                worst = max(abs(recomputed[s] - ledger[s]) for s in STATISTICS)
                check[name] = {"recomputed": recomputed, "ledger": ledger, "max_abs_difference": worst,
                               "descriptor_codes": len(set(codes.values())), "bank_hits": cached, "computed": computed}
                log(f"[{kind}] {name}: AUROC {recomputed['auroc']:.6f} (ledger {ledger['auroc']:.6f}), "
                    f"max |difference| over statistics {worst:.2e}, descriptor bank hits {cached}, computed {computed} "
                    f"[{time.monotonic() - started:.0f}s]")
                (OUT / f"check_{kind}.json").write_text(json.dumps(check, indent=1))
        (OUT / f"check_{kind}.json").write_text(json.dumps(check, indent=1))
        (OUT / f"runs_{kind}.json").write_text(json.dumps([str(r) for r in runs]))
    log("rescore done")


# --------------------------------------------------------------------------------------------
# Stage 2: joint bootstrap, fixed rules, null pairs
# --------------------------------------------------------------------------------------------

def load_kind(kind):
    meta = json.loads((OUT / f"cells_{kind}.json").read_text())
    arrays = np.load(OUT / f"cells_{kind}.npz", allow_pickle=False)
    cells = {}
    for brain in meta["brains"]:
        cells[brain] = {**meta["cells"][brain], "truth": arrays[f"{brain}:truth"].astype(float),
                        "keys": arrays[f"{brain}:keys"]}
    return meta, cells


def load_scores(name, kind, brains):
    data = np.load(OUT / "scores" / f"{name}_{kind}.npz", allow_pickle=False)
    return {brain: data[brain] for brain in brains}


def paired_intervals(job):
    """Point statistics of both arms and the paired bootstrap interval of every delta, by the
    promotion judge's own implementation (``promotion_judge.intervals_from_rankings``)."""
    from proofreader_evolve.harness.rank_statistics import ranking
    cells, parent, candidate, draws, seed, pauc_fpr = job
    prepared = {}
    for brain, cell in cells.items():
        arms = []
        for scores in (parent[brain], candidate[brain]):
            order, labels, starts = ranking(scores, cell["truth"], cell["keys"])
            arms.append((order, labels.astype(float), starts))  # every stored row is labeled: order is the position
        prepared[brain] = {"arms": tuple(arms), "size": len(cell["truth"]), "k": cell["k"], "counted": cell["in_mean"]}
    return promotion_judge.intervals_from_rankings(prepared, draws=draws, seed=seed, alpha=.05, pauc_fpr=pauc_fpr)


def rank_normalized(scores):
    from scipy.stats import rankdata
    return rankdata(scores, method="average") / len(scores)


def null_pair(parent, candidate, cells, seed):
    """Two exchangeable hybrids of a real pair: every labeled row takes its rank-normalized
    score from one model in C1 and from the other in C2, by a fair coin per row."""
    rng = np.random.default_rng(seed)
    first, second = {}, {}
    for brain in cells:
        a, b = rank_normalized(parent[brain]), rank_normalized(candidate[brain])
        coin = rng.random(len(a)) < .5
        first[brain], second[brain] = np.where(coin, a, b), np.where(coin, b, a)
    return first, second


def rule_auroc(stats):
    """The current gate: AUROC point gain, bootstrap lower bound above zero, head guard."""
    return _objective_rule(stats, "auroc")


def rule_average_precision(stats):
    return _objective_rule(stats, "average_precision")


def rule_depth_mean(stats):
    return _objective_rule(stats, "precision_depth_mean")


def _objective_rule(stats, objective):
    head = stats["precision_depth_mean"]
    return bool(stats[objective]["delta"] > 0 and stats[objective]["low"] > 0
                and head["candidate"] >= head["parent"] - HEAD_TOLERANCE)


guard_rail = promotion_judge.guard_rail  # no guarded statistic significantly worse


def rule_multi(stats):
    """Fixed multi-statistic rule: AUROC and average precision both not lower, at least one of
    them improved beyond resampling noise, and no guarded statistic significantly worse."""
    return bool(guard_rail(stats) and stats["auroc"]["delta"] >= 0 and stats["average_precision"]["delta"] >= 0
                and (stats["auroc"]["low"] > 0 or stats["average_precision"]["low"] > 0))


RULES = {"auroc_gate": rule_auroc, "average_precision_gate": rule_average_precision,
         "depth_mean_gate": rule_depth_mean, "multi_statistic": rule_multi}


def replay(args):
    result = {"draws": args.draws, "kinds": {}}
    for kind in args.kinds:
        meta, cells = load_kind(kind)
        brains = meta["brains"]
        runs = [Path(r) for r in json.loads((OUT / f"runs_{kind}.json").read_text())]
        check = json.loads((OUT / f"check_{kind}.json").read_text())
        seed_scores = load_scores("seed", kind, brains)
        pairs = []
        for run in runs:
            accepted = "seed"
            for row in validated_rows(run):
                if row["target_kind"] != kind:
                    continue
                name = f"{run.name[-8:]}_gen{row['generation']:02d}"
                gate = (row.get("validation_gate") or {}).get("bootstrap_gate") or {}
                pairs.append({"name": name, "run": run.name[-8:], "generation": row["generation"],
                              "parent": accepted, "ledger_accepted": bool(row["accepted"]),
                              "ledger_lower": gate.get("lower_bound"), "ledger_upper": gate.get("upper_bound"),
                              "recheck": check[name]["max_abs_difference"]})
                if row["accepted"]:
                    accepted = name
        scores = {"seed": seed_scores}
        for pair in pairs:
            scores[pair["name"]] = load_scores(pair["name"], kind, brains)
        jobs = [(cells, scores[p["parent"]], scores[p["name"]], args.draws, p["generation"], meta["pauc_fpr"])
                for p in pairs]
        nulls = []
        null_jobs = []
        for pair in pairs:
            if pair["parent"] == "seed":
                continue  # the seed is a constant scorer: its ranks carry no information to mix
            for index in range(args.null_per_pair):
                seed = zlib.crc32(f"{kind}/{pair['name']}/{index}".encode())
                first, second = null_pair(scores[pair["parent"]], scores[pair["name"]], cells, seed)
                nulls.append({"source": pair["name"], "index": index})
                null_jobs.append((cells, first, second, args.draws, seed + 1, meta["pauc_fpr"]))
        log(f"[{kind}] {len(jobs)} real pairs, {len(null_jobs)} null pairs, {args.draws} draws, {args.workers} workers")
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            real_stats = list(pool.map(paired_intervals, jobs))
            null_stats = list(pool.map(paired_intervals, null_jobs, chunksize=4))
        for pair, stats in zip(pairs, real_stats):
            pair["statistics"] = stats
            pair["rules"] = {name: rule(stats) for name, rule in RULES.items()}
            pair["guard_rail"] = guard_rail(stats)
            if pair["ledger_lower"] is not None:
                pair["auroc_interval_matches_ledger"] = bool(
                    abs(stats["auroc"]["low"] - pair["ledger_lower"]) < 1e-9
                    and abs(stats["auroc"]["high"] - pair["ledger_upper"]) < 1e-9)
            log(f"[{kind}] {pair['name']} vs {pair['parent']}: "
                + " ".join(f"{s}={stats[s]['delta']:+.4f}[{stats[s]['low']:+.4f},{stats[s]['high']:+.4f}]"
                           for s in ("auroc", "average_precision", "precision_depth_mean", "precision"))
                + f" | ledger {'A' if pair['ledger_accepted'] else 'r'} | "
                + " ".join(f"{n}={'A' if v else 'r'}" for n, v in pair["rules"].items())
                + f" | AUROC interval matches the gate: {pair.get('auroc_interval_matches_ledger')}")
        for null, stats in zip(nulls, null_stats):
            null["statistics"] = stats
            null["rules"] = {name: rule(stats) for name, rule in RULES.items()}
            null["guard_rail"] = guard_rail(stats)
        rates = {name: sum(n["rules"][name] for n in nulls) / len(nulls) for name in RULES} if nulls else {}
        log(f"[{kind}] false acceptance on {len(nulls)} null pairs: "
            + ", ".join(f"{n} {r:.3f}" for n, r in rates.items()))
        result["kinds"][kind] = {"pairs": pairs, "nulls": nulls, "null_false_acceptance": rates,
                                 "cells": {b: {k: v for k, v in c.items() if k not in ("truth", "keys")}
                                           for b, c in cells.items()}}
    (OUT / "replay.json").write_text(json.dumps(result, indent=1))
    log(f"wrote {OUT / 'replay.json'}")


# --------------------------------------------------------------------------------------------
# Stage 3: LLM judge
# --------------------------------------------------------------------------------------------

def judge_prompt(kind, cells, stats, with_intervals):
    counted = [c for c in cells.values() if c["in_mean"]]
    pools = {"pools": len(counted), "labeled": sum(c["labeled"] for c in counted),
             "positives": sum(c["positives"] for c in counted)}
    return promotion_judge.prompt(kind, pools, stats, with_intervals=with_intervals)


def judge(args):
    import anthropic
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY is not set")
    replayed = json.loads((OUT / "replay.json").read_text())
    items = []
    for kind, data in replayed["kinds"].items():
        cells = data["cells"]
        for pair in data["pairs"]:
            if pair["parent"] != "seed":
                items.append((f"real/{kind}/{pair['name']}", kind, cells, pair["statistics"]))
        for null in data["nulls"]:
            if null["index"] < args.null_per_pair:
                items.append((f"null/{kind}/{null['source']}/{null['index']}", kind, cells, null["statistics"]))
    votes_path = OUT / "judge_votes.jsonl"
    done = set()
    if votes_path.is_file():
        for line in votes_path.open():
            record = json.loads(line)
            if "decision" in record:
                done.add((record["item"], record["variant"], record["vote"]))
    tasks = [(item, variant, vote, judge_prompt(kind, cells, stats, variant == "intervals"))
             for item, kind, cells, stats in items for variant in ("intervals", "no_intervals")
             for vote in range(args.votes) if (item, variant, vote) not in done]
    log(f"{len(items)} items, {len(tasks)} calls to {JUDGE_MODEL} at effort {args.effort} ({len(done)} already recorded)")
    client = anthropic.Anthropic(max_retries=6)

    def run(task):
        item, variant, vote, prompt = task
        return {"item": item, "variant": variant, "vote": vote,
                **promotion_judge.vote(client, prompt, model=JUDGE_MODEL, effort=args.effort)}

    spent = {"input_tokens": 0, "output_tokens": 0}
    with votes_path.open("a") as stream, ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        for count, record in enumerate(pool.map(run, tasks), 1):
            stream.write(json.dumps(record) + "\n")
            stream.flush()
            for key in spent:
                spent[key] += (record.get("usage") or {}).get(key, 0)
            if count % 50 == 0 or count == len(tasks):
                log(f"{count}/{len(tasks)} calls; tokens in {spent['input_tokens']} out {spent['output_tokens']}")
    if tasks:
        (OUT / "judge_prompt_example.txt").write_text(promotion_judge.SYSTEM + "\n\n---\n\n" + tasks[0][3])
    log(f"wrote {votes_path}")


# --------------------------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------------------------

def wilson(hits, total):
    if not total:
        return (float("nan"), float("nan"))
    z = 1.96
    p = hits / total
    centre = (p + z * z / (2 * total)) / (1 + z * z / total)
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / (1 + z * z / total)
    return centre - half, centre + half


def summary(args):
    replayed = json.loads((OUT / "replay.json").read_text())
    votes = {}
    usage = {"input_tokens": 0, "output_tokens": 0}
    errors = 0
    votes_path = OUT / "judge_votes.jsonl"
    if votes_path.is_file():
        for line in votes_path.open():
            record = json.loads(line)
            for key in usage:
                usage[key] += (record.get("usage") or {}).get(key, 0)
            if "decision" not in record:
                errors += 1
                continue
            votes.setdefault((record["item"], record["variant"]), {})[record["vote"]] = record["decision"] == "accept"

    def majority(item, variant):
        cast = list(votes.get((item, variant), {}).values())
        if not cast:
            return None, None
        return sum(cast) * 2 > len(cast), f"{sum(cast)}/{len(cast)}"

    lines = ["# Acceptance rule replay", "",
             f"Paired bootstrap with {replayed['draws']} draws; intervals are two-sided 95%. `A` accept, `r` reject. "
             "Judge columns: majority of the votes (accept votes / votes cast); `judge+guard` is the majority with "
             "intervals behind the host guard rail (no guarded statistic significantly worse).", ""]
    for kind, data in replayed["kinds"].items():
        lines += [f"## {kind}: validated candidates against the scorer accepted at the time", "",
                  "| candidate | parent | AUROC delta [95%] | AP delta [95%] | depth-mean delta [95%] | P@K delta [95%] | run's gate | "
                  + " | ".join(RULES) + " | judge+guard | judge (intervals) | judge (no intervals) |",
                  "|---|---|---|---|---|---|---|" + "---|" * (len(RULES) + 3)]
        for pair in data["pairs"]:
            stats = pair["statistics"]
            cell = lambda s: f"{stats[s]['delta']:+.4f} [{stats[s]['low']:+.4f}, {stats[s]['high']:+.4f}]"
            item = f"real/{kind}/{pair['name']}"
            with_ci, with_ci_votes = majority(item, "intervals")
            without, without_votes = majority(item, "no_intervals")
            flag = lambda v: "-" if v is None else ("A" if v else "r")
            guarded = None if with_ci is None else (with_ci and pair["guard_rail"])
            lines.append(f"| {pair['name']} | {pair['parent']} | {cell('auroc')} | {cell('average_precision')} | "
                         f"{cell('precision_depth_mean')} | {cell('precision')} | {'A' if pair['ledger_accepted'] else 'r'} | "
                         + " | ".join("A" if pair["rules"][n] else "r" for n in RULES)
                         + f" | {flag(guarded)} | {flag(with_ci)} {with_ci_votes or ''} | {flag(without)} {without_votes or ''} |")
        nulls = data["nulls"]
        lines += ["", f"### {kind}: false acceptance on null pairs (true difference zero by construction)", "",
                  "| rule | null pairs | accepted | rate | 95% interval (Wilson) |", "|---|---|---|---|---|"]
        for name in RULES:
            hits = sum(n["rules"][name] for n in nulls)
            low, high = wilson(hits, len(nulls))
            lines.append(f"| {name} | {len(nulls)} | {hits} | {hits / len(nulls):.3f} | [{low:.3f}, {high:.3f}] |")
        judged = [n for n in nulls if (f"null/{kind}/{n['source']}/{n['index']}", "intervals") in votes]
        for label, decide in (
                ("judge+guard", lambda n: majority(f"null/{kind}/{n['source']}/{n['index']}", "intervals")[0] and n["guard_rail"]),
                ("judge (intervals)", lambda n: majority(f"null/{kind}/{n['source']}/{n['index']}", "intervals")[0]),
                ("judge (no intervals)", lambda n: majority(f"null/{kind}/{n['source']}/{n['index']}", "no_intervals")[0])):
            if judged:
                hits = sum(bool(decide(n)) for n in judged)
                low, high = wilson(hits, len(judged))
                lines.append(f"| {label} | {len(judged)} | {hits} | {hits / len(judged):.3f} | [{low:.3f}, {high:.3f}] |")
        if judged:
            split = [0, 0]
            for n in judged:
                for variant, slot in (("intervals", 0), ("no_intervals", 1)):
                    cast = list(votes[(f"null/{kind}/{n['source']}/{n['index']}", variant)].values())
                    split[slot] += 0 < sum(cast) < len(cast)
            lines += ["", f"Null pairs on which the judge's own votes disagreed: {split[0]} of {len(judged)} with intervals, "
                      f"{split[1]} of {len(judged)} without."]
        lines.append("")
    if votes:
        cost = usage["input_tokens"] * 4e-6 + usage["output_tokens"] * 20e-6
        lines += [f"Judge: {JUDGE_MODEL}; {sum(len(v) for v in votes.values())} votes recorded, {errors} calls without a vote; "
                  f"{usage['input_tokens']} input and {usage['output_tokens']} output tokens (about ${cost:.2f} at list price).", ""]
    (OUT / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="stage", required=True)
    p = sub.add_parser("rescore", help="score the validated candidates again on the validation pools")
    p.add_argument("--runs", nargs="+", required=True, help="finished run directories launched with the same arguments")
    p.add_argument("--kinds", nargs="+", default=["merge", "split"], choices=["merge", "split"])
    p.add_argument("--scratch", type=Path, required=True, help="directory outside the repository for worker logs")
    p.add_argument("--allow-compute", action="store_true", help="compute descriptor entries missing from the bank")
    p = sub.add_parser("replay", help="joint bootstrap, fixed rules and null pairs")
    p.add_argument("--kinds", nargs="+", default=["merge", "split"], choices=["merge", "split"])
    p.add_argument("--draws", type=int, default=1000)
    p.add_argument("--null-per-pair", type=int, default=25)
    p.add_argument("--workers", type=int, default=max(1, len(os.sched_getaffinity(0)) - 2))
    p = sub.add_parser("judge", help="LLM votes on the real pairs and a subset of the null pairs")
    p.add_argument("--votes", type=int, default=5)
    p.add_argument("--null-per-pair", type=int, default=4)
    p.add_argument("--effort", default="medium", choices=["low", "medium", "high", "xhigh", "max"])
    p.add_argument("--concurrency", type=int, default=8)
    sub.add_parser("summary", help="write and print summary.md")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    {"rescore": rescore, "replay": replay, "judge": judge, "summary": summary}[args.stage](args)


if __name__ == "__main__":
    main()
