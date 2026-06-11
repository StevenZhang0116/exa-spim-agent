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

    gens = []
    for row in ledger:
        gens.append({
            "generation": row.get("generation"),
            "train_primary": row.get("train_primary"),
            "heldout_primary": row.get("heldout_primary"),
            "parent_heldout": row.get("parent_heldout"),
            "accepted": row.get("accepted"),
            "diffstat": row.get("heuristics_diffstat", ""),
            "note": row.get("note", ""),
            "cost_usd": row.get("cost_usd"),
            "output_tokens": row.get("output_tokens"),
            # candidate file is kept regardless of accept/reject — point at it
            "candidate_path": row.get("candidate_path", ""),
            "diagnosis": (row.get("diagnosis") or "")[:600],
        })

    heldouts = [g["heldout_primary"] for g in gens if g["heldout_primary"] is not None]
    accepted_gens = [g["generation"] for g in gens if g["accepted"]]
    baseline = gens[0]["parent_heldout"] if gens else None  # seed bar = no-op baseline
    final_heldout = max(heldouts) if heldouts else None

    return {
        "run_id": run_dir.name,
        "run_dir": str(run_dir),
        "brain_id": run_dir.name.split("_")[0],
        "n_generations": len(gens),
        "n_accepted": len(accepted_gens),
        "accepted_generations": accepted_gens,
        "baseline_heldout": baseline,
        "final_heldout": final_heldout,
        "net_heldout_gain": (final_heldout - baseline)
                            if (final_heldout is not None and baseline is not None) else None,
        "total_cost_usd": round(sum(g["cost_usd"] or 0 for g in gens), 4),
        "generations": gens,
        "final_policy": final,
        "rules_changelog": _rules_changelog(run_dir, final),
        "artifacts": {
            "ledger": str(run_dir / "ledger.jsonl"),
            "attempts": str(run_dir / "attempts.md"),
            "final_heuristics": final.get("heuristics_path", ""),
            "final_rules": final.get("rules_path", ""),
        },
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("usage: collect_run.py <run_id | brain_id>")
    out = collect(resolve_run_dir(sys.argv[1]))
    print(json.dumps(out, indent=2))
