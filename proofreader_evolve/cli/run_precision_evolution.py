"""Evolve a scorer on immutable AutoDiscovery detector-native candidate pools.

No graph edits, candidate expansion, model retraining or table preparation.
Default objective: macro Precision@K at fixed K per brain/kind, no cell regression.
Validation is repeatedly queried development data, never an untouched final test.
"""

import argparse
import asyncio
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import tempfile
import time

from . import precompute_error_scores as pc
from ..harness import fixed_pool_scoring as scoring
from ..harness.native_pool import cache_path, ensure_native_tables
from ..harness.reviser_session import DEFAULT_MODEL, build_options, bind_session_options


HERE = Path(__file__).resolve().parents[1]
CONTRACT = (HERE / "artifacts" / "scorer_rules.md").read_text()


async def revise(run_dir, policy_path, rules_path, report_path, model):
    """A fresh SDK session with the existing per-file access guard; no subagents."""
    from claude_agent_sdk import ResultMessage, ClaudeSDKClient
    options, _ = build_options(model=model, run_dir=run_dir,
                               system_prompt=CONTRACT, max_turns=12)
    options = bind_session_options(options, policy_path, rules_path, report_path)
    result = None
    prompt = (
        f"Read TRAIN feedback at {report_path}. Improve {policy_path} and update {rules_path}. "
        "Follow the fixed-pool scorer contract. Do not change the candidate pool, budgets or evaluator. "
        "Return a short explanation of the revision."
    )
    async with ClaudeSDKClient(options=options) as client:
        await client.query(prompt)
        async for message in client.receive_response():
            if isinstance(message, ResultMessage):
                if message.is_error:
                    raise RuntimeError("Scorer reviser SDK reported an error")
                result = {"summary": message.result, "usage": message.usage,
                          "cost_usd": message.total_cost_usd}
    if result is None:
        raise RuntimeError("Scorer reviser ended without a result")
    return result


def _write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False))


def _csv_brains(value):
    brains = value.split(",")
    if any(not b.isascii() or not b.isdigit() for b in brains) or len(set(brains)) != len(brains):
        raise argparse.ArgumentTypeError("Use distinct comma-separated numeric brain IDs")
    return brains


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-brains", type=_csv_brains, default=["789202"])
    p.add_argument("--validation-brains",
                   type=_csv_brains, default=["794491"],
                   help="Development validation, not final testing")
    p.add_argument("--mcl", type=int, default=100)
    p.add_argument("--generations", type=int, default=5)
    p.add_argument("--merge-k", type=int, default=100)
    p.add_argument("--split-k", type=int, default=100)
    p.add_argument("--precision-margin", type=float, default=0)
    p.add_argument("--policy-time-budget", type=float, default=120)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--feature-tables-dir", type=Path, default=pc.DEFAULT_OUT)
    p.add_argument("--merge-dir", action="append")
    p.add_argument("--split-dir", action="append")
    p.add_argument("--start-from", type=Path, default=HERE / "artifacts" / "scorer.py")
    p.add_argument("--runs-dir", type=Path, default=HERE / "runs")
    args = p.parse_args(argv)
    if set(args.train_brains) & set(args.validation_brains):
        p.error("Train and validation brains must be disjoint")
    if args.generations < 0 or args.mcl < 0 or min(args.merge_k, args.split_k) < 1:
        p.error("generations/MCL must be >= 0; K must be >= 1")
    if (not math.isfinite(args.precision_margin) or args.precision_margin < 0
            or not math.isfinite(args.policy_time_budget) or args.policy_time_budget <= 0):
        p.error("Invalid precision margin or policy time budget")
    return args


async def run(args, revise_fn=None):
    """One-child generations with immutable pools, train screening and rollback."""
    revise_fn = revise if revise_fn is None else revise_fn
    selected = pc.resolve_detector_runs(args.merge_dir, args.split_dir)
    brains = args.train_brains + args.validation_brains
    banks = {b: ensure_native_tables(b, cache_path(b, args.mcl), selected,
                                    args.feature_tables_dir, prepare=False, mcl=args.mcl) for b in brains}
    for brain in args.validation_brains:
        for table in banks[brain].tables.values():
            fitted = table.meta.get("training_brain")
            if fitted is None:
                raise ValueError("Detector training-brain provenance is required for independent validation")
            if str(fitted) == brain:
                raise ValueError("Detector-fitted brain cannot be development validation")
    train = {b: banks[b] for b in args.train_brains}
    validation = {b: banks[b] for b in args.validation_brains}
    budgets = {"merge": args.merge_k, "split": args.split_k}
    source = args.start_from.read_text()
    parent_train = scoring.evaluate(source, train, budgets, args.policy_time_budget, examples=True)
    parent_validation = scoring.evaluate(source, validation, budgets, args.policy_time_budget)
    # Record the original frozen-score comparator even when resuming from a custom policy.
    baseline_source = (HERE / "artifacts" / "scorer.py").read_text()
    baseline = {"train": scoring.evaluate(baseline_source, train, budgets, args.policy_time_budget),
                "validation": scoring.evaluate(baseline_source, validation, budgets, args.policy_time_budget)}
    args.runs_dir.mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix="precision_" + datetime.now().strftime("%Y%m%d_%H%M%S_"),
                                   dir=args.runs_dir)).resolve()
    manifest = {"objective": scoring.SCORING_VERSION, "accounting_semantics": "per_generation", "budgets": budgets,
                "precision_margin": args.precision_margin, "train_brains": args.train_brains,
                "development_validation_brains": args.validation_brains, "selected_detectors": selected,
                "pools": {b: {k: t.meta for k, t in bank.tables.items()} for b, bank in banks.items()},
                "evaluator_sha256": pc._sha256(Path(scoring.__file__)),
                "note": "Native-label precision, not proof of repair efficacy; Python is not OS-sandboxed"}
    _write(run_dir / "manifest.json", manifest)
    _write(run_dir / "baseline.json", baseline)
    _write(run_dir / "seed.json", {"train": parent_train, "validation": parent_validation})
    rules = CONTRACT
    history = []
    (run_dir / "best_scorer.py").write_text(source)
    for generation in range(1, args.generations + 1):
        generation_started = time.monotonic()
        gen_dir = run_dir / f"gen{generation:03d}"
        gen_dir.mkdir()
        policy_path, rules_path = gen_dir / "scorer.py", gen_dir / "rules.md"
        policy_path.write_text(source)
        rules_path.write_text(rules)
        report_path = gen_dir / "train_feedback.json"
        _write(report_path, {"contract": CONTRACT, "budgets": budgets, "parent_train": parent_train,
                            "attempts": [{"generation": h["generation"], "accepted": h["accepted"],
                                          "train": h["train"]} for h in history[-5:]]})
        record = {"generation": generation, "accepted": False,
                  "objective": scoring.SCORING_VERSION, "accounting_semantics": "per_generation",
                  "parent_validation_precision": parent_validation["macro_precision"],
                  "parent_policy_sha256": hashlib.sha256(source.encode()).hexdigest(),
                  "candidate_policy_sha256": None, "train": None, "validation": None}
        try:
            record["reviser"] = await revise_fn(run_dir, policy_path, rules_path, report_path, args.model)
            candidate_source = policy_path.read_text()
            record["candidate_policy_sha256"] = hashlib.sha256(candidate_source.encode()).hexdigest()
            candidate_train = scoring.evaluate(candidate_source, train, budgets,
                                               args.policy_time_budget, examples=True)
            record["train"] = candidate_train
            improves_train, reason = scoring.acceptance(parent_train, candidate_train, args.precision_margin)
            record["reason"] = "TRAIN: " + reason
            if improves_train:
                candidate_validation = scoring.evaluate(candidate_source, validation, budgets, args.policy_time_budget)
                record["validation"] = candidate_validation
                accepted, reason = scoring.acceptance(parent_validation, candidate_validation, args.precision_margin)
                record.update(accepted=accepted, reason="VALIDATION: " + reason)
                if accepted:
                    source, rules = candidate_source, rules_path.read_text()
                    parent_train, parent_validation = candidate_train, candidate_validation
                    (run_dir / "best_scorer.py").write_text(source)
        except Exception as exc:
            # A failed/timeout policy is never substituted with a baseline measurement.
            record["reason"] = f"Failed: {type(exc).__name__}: {exc}"
        record["wall_seconds"] = time.monotonic() - generation_started
        history.append(record)
        _write(gen_dir / "evaluation.json", record)
        with (run_dir / "ledger.jsonl").open("a") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        print(f"Generation {generation}: {'accepted' if record['accepted'] else 'rejected'}; {record['reason']}", flush=True)
    _write(run_dir / "final.json", {"train": parent_train, "validation": parent_validation})
    print(f"Fixed-pool scorer run: {run_dir}", flush=True)
    return run_dir


def main(argv=None):
    asyncio.run(run(parse_args(argv)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
