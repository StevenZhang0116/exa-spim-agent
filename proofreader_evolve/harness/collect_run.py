"""
Collect the ground-truth facts of one evolution run into a single JSON blob.

POST-ANALYSIS ONLY — not part of run_evolution.py's pipeline. The run-summarizer
agent calls this so its Markdown narrative is grounded in real numbers (ledger,
diffstats, accepted lineage, final policy params) rather than re-derived/guessed.

Usage:
    python proofreader_evolve/harness/collect_run.py 789202_20260611_002615
    python proofreader_evolve/harness/collect_run.py 789202        # newest run for a brain
The argument is either a full run-id folder name, or just a brain id (then the
newest matching runs/<brain>_* folder is used).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

RUNS = Path(__file__).resolve().parent.parent / "runs"


def resolve_run_dir(arg: str) -> Path:
    """Accept a full run-id folder, or a brain id (-> newest runs/<brain>_* dir)."""
    exact = RUNS / arg
    if exact.is_dir():
        return exact
    matches = sorted(RUNS.glob(f"{arg}_*"), key=lambda p: p.stat().st_mtime)
    if not matches:
        raise SystemExit(f"No run folder matches {arg!r} under {RUNS}")
    return matches[-1]


def _read_ledger(run_dir: Path) -> list[dict]:
    path = run_dir / "ledger.jsonl"
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def _final_accepted(run_dir: Path) -> dict:
    """The last accepted policy's files + extracted parameters (the 'learned' policy)."""
    accepted = sorted(run_dir.glob("gen*/heuristics.accepted.py"))
    if not accepted:
        return {"gen": None, "heuristics_path": "", "rules_path": "", "params": {}}
    last = accepted[-1]
    gen = last.parent.name
    rules = last.parent / "rules.accepted.md"
    # Pull top-level UPPER_CASE = number params (the tunables the policy exposes).
    params = {}
    for line in last.read_text().splitlines():
        m = re.match(r"^([A-Z][A-Z0-9_]+)\s*=\s*([-\d.]+)", line)
        if m:
            params[m.group(1)] = m.group(2)
    return {
        "gen": gen,
        "heuristics_path": str(last),
        "rules_path": str(rules) if rules.exists() else "",
        "params": params,
    }


def _rules_changelog(run_dir: Path, final: dict) -> str:
    """The '## Change log' section of the final accepted rules.md (the per-gen story)."""
    rp = final.get("rules_path")
    if not rp or not Path(rp).exists():
        return ""
    text = Path(rp).read_text()
    idx = text.find("## Change log")
    return text[idx:].strip() if idx != -1 else ""


def collect(run_dir: Path) -> dict:
    ledger = _read_ledger(run_dir)
    final = _final_accepted(run_dir)

    # Backward-compatible field reads: current ledgers use train_edge_accuracy /
    # heldout_edge_accuracy / parent_split_repair_score; older ones used
    # train_primary / heldout_primary / parent_heldout (the last MISLABELLED — it
    # always held the split-repair score, never an Edge Accuracy).
    def _train_ea(row):
        return row.get("train_edge_accuracy", row.get("train_primary"))
    def _heldout_ea(row):
        return row.get("heldout_edge_accuracy", row.get("heldout_primary"))
    def _parent_sr(row):
        return row.get("parent_split_repair_score", row.get("parent_heldout"))

    gens = []
    for row in ledger:
        gens.append({
            "generation": row.get("generation"),
            "train_edge_accuracy": _train_ea(row),
            "heldout_edge_accuracy": _heldout_ea(row),
            "parent_split_repair_score": _parent_sr(row),
            "heldout_split_repair_score": row.get("heldout_split_repair_score"),
            "accepted": row.get("accepted"),
            "diffstat": row.get("heuristics_diffstat", ""),
            "note": row.get("note", ""),
            "cost_usd": row.get("cost_usd"),
            "output_tokens": row.get("output_tokens"),
            # candidate file is kept regardless of accept/reject — point at it
            "candidate_path": row.get("candidate_path", ""),
            "diagnosis": (row.get("diagnosis") or "")[:600],
        })

    accepted_gens = [g["generation"] for g in gens if g["accepted"]]

    # Edge Accuracy is a DIAGNOSTIC (not the gate); report its baseline->final as
    # such, comparing like with like (gen-1 baseline EA vs best EA over the run).
    heldout_eas = [g["heldout_edge_accuracy"] for g in gens
                   if g["heldout_edge_accuracy"] is not None]
    baseline_ea = gens[0]["heldout_edge_accuracy"] if gens else None
    final_ea = max(heldout_eas) if heldout_eas else None

    # The GATE's real progress = split-repair score. baseline = the bar gen 1 faced
    # (the seed's score, recorded as gen 1's parent_split_repair_score); final = the
    # best accepted score over the run.
    accepted_sr = [g["heldout_split_repair_score"] for g in gens
                   if g["accepted"] and g["heldout_split_repair_score"] is not None]
    baseline_sr = gens[0]["parent_split_repair_score"] if gens else None
    final_sr = max(accepted_sr) if accepted_sr else None

    return {
        "run_id": run_dir.name,
        "run_dir": str(run_dir),
        "brain_id": run_dir.name.split("_")[0],
        "n_generations": len(gens),
        "n_accepted": len(accepted_gens),
        "accepted_generations": accepted_gens,
        # Gate metric (split-repair score) — the number the run actually optimized.
        "baseline_split_repair_score": baseline_sr,
        "final_split_repair_score": final_sr,
        "net_split_repair_gain": (final_sr - baseline_sr)
                                 if (final_sr is not None and baseline_sr is not None) else None,
        # Edge Accuracy — diagnostic only (NOT the gate); like-with-like baseline->final.
        "baseline_heldout_edge_accuracy": baseline_ea,
        "final_heldout_edge_accuracy": final_ea,
        "net_heldout_edge_accuracy_gain": (final_ea - baseline_ea)
                            if (final_ea is not None and baseline_ea is not None) else None,
        # cost_usd is the running SESSION total (cumulative), so the run total is the
        # MAX value, NOT a sum over generations.
        "total_cost_usd": round(max((g["cost_usd"] or 0 for g in gens), default=0), 4),
        "generations": gens,
        "final_policy": final,
        "rules_changelog": _rules_changelog(run_dir, final),
        "artifacts": {
            # ledger.jsonl is the single source of truth for per-generation state.
            # The human-readable attempts timeline is rendered on demand from it via
            # ``python -m proofreader_evolve.harness.ledger <run_dir>`` (there is no
            # longer a standalone attempts.md file).
            "ledger": str(run_dir / "ledger.jsonl"),
            "final_heuristics": final.get("heuristics_path", ""),
            "final_rules": final.get("rules_path", ""),
        },
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("usage: collect_run.py <run_id | brain_id>")
    out = collect(resolve_run_dir(sys.argv[1]))
    print(json.dumps(out, indent=2))
