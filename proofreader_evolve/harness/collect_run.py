"""Summarize a fixed-pool scorer run without executing policies or calling an LLM.

Usage: python -m proofreader_evolve.harness.collect_run RUN_OR_PATH
A numeric brain ID selects the newest matching precision run by metadata.
"""

import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from proofreader_evolve.harness import run_records as records

RUNS = Path(__file__).resolve().parents[1] / "runs"


def _roles(run_dir):
    metadata = records.read_json(run_dir / "manifest.json", {})
    if metadata.get("objective") != records.PRECISION:
        return {"train": [], "validation": []}
    return {"train": metadata.get("train_brains") or [],
            "validation": metadata.get("development_validation_brains") or []}


def resolve_run_dir(arg):
    for path in (Path(arg), RUNS / arg):
        if path.is_dir():
            return path
    matches = [path for path in RUNS.iterdir() if path.is_dir()
               and str(arg) in {str(b) for group in _roles(path).values() for b in group}]
    if not matches:
        raise FileNotFoundError(f"No precision run matches {arg!r}")
    return max(matches, key=lambda p: p.stat().st_mtime)


def collect(run_dir):
    run_dir = Path(run_dir)
    rows = records.load_ledger(run_dir)
    version = records.run_objective(run_dir, rows)
    accepted = [r for r in rows if r.get("accepted")]
    seed = records.read_json(run_dir / "seed.json", {})
    baseline = records.measurement_value(seed.get("validation"))
    final_record = records.read_json(run_dir / "final.json", {})
    final = records.measurement_value(final_record.get("validation"))
    if final is None:
        final = records.validation_value(accepted[-1]) if accepted else baseline
    policy = run_dir / "best_scorer.py"
    rules = run_dir / f"gen{accepted[-1]['generation']:03d}" / "rules.md" if accepted else None
    costs = records.cumulative_accounting(rows)
    return {
        "run_id": run_dir.name, "run_dir": str(run_dir), "objective": version,
        "metric": records.METRIC_NAME, "brain_roles": _roles(run_dir),
        "n_generations": len(rows), "n_accepted": len(accepted),
        "accepted_generations": [r["generation"] for r in accepted],
        "baseline_validation": baseline, "final_accepted_validation": final,
        "net_validation_gain": final - baseline if final is not None and baseline is not None else None,
        "total_cost_usd": costs[-1] if costs else 0.0,
        "accounting_note": "Per-generation session costs are summed; missing amounts stay null.",
        "generations": [
            {"generation": r["generation"], "accepted": r.get("accepted"),
             "validation": records.validation_value(r), "reason": records.decision_reason(r)}
            for r in rows
        ],
        "final_policy": str(policy) if policy.exists() else None,
        "final_rules": str(rules) if rules is not None and rules.exists() else None,
        "paired_measurements": records.paired_measurements(rows, run_dir),
        "note": "Development validation is not final testing; ranking gains do not prove safe graph repairs.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run")
    args = parser.parse_args(argv)
    print(json.dumps(collect(resolve_run_dir(args.run)), indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
