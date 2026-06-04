"""
Rank AutoDiscovery hypotheses by surprise across one or more run exports.

Deterministic helper for the ``discovery-summarizer`` agent
(``.claude/agents/discovery-summarizer.md``). Not a general-purpose tool — its
record schema and stdout contract are what that agent reads. If you change the
output shape here, update the agent prompt to match.

Each AutoDiscovery run export is a JSON file in ``autodiscovery/``: a list of
tested-hypothesis objects with (at least) the fields ``id``, ``status``,
``hypothesis``, ``surprisal`` (a SIGNED belief-shift score, roughly -1..1),
``isSurprising``, ``prior`` / ``posterior`` (belief probability before/after),
``priorBelief`` / ``posteriorBelief`` (the full belief distributions),
``analysis``, ``review``, and ``experimentPlan`` (with ``objective``,
``steps``, ``deliverables``).

This script gathers the hypotheses from ALL given JSON files (or, with no
paths, every ``autodiscovery/*.json`` next to it) and prints the ordered
records as JSON to stdout for the agent to turn into a single combined,
cross-run scientific summary. Two ranking modes are supported (``--rank-by``):

* ``surprise`` (default) — rank by surprise MAGNITUDE (``abs(surprisal)``)
  descending: the most belief-shifting results, regardless of which way belief
  moved, come first.
* ``posterior-surprise`` — rank by a combined priority score
  ``posterior * abs(surprisal)`` descending: results that are BOTH strongly
  believed true after the experiment (high ``posterior``) AND highly
  belief-shifting (high surprise) come first. This surfaces well-established
  surprising findings rather than surprising-but-now-disbelieved ones.

It is intentionally deterministic: it does NOT summarize or call any model. It
only parses, validates, sorts, and reshapes the run exports so the agent's job
is purely the scientific-conclusion writing. By default nothing is written to
disk — pass ``--out PATH`` only if you explicitly want to persist the JSON.

Usage (paths shown relative to the ``exa-spim-agent/`` project root)
--------------------------------------------------------------------
    python agentic/rank_by_surprise.py                              # all autodiscovery/*.json
    python agentic/rank_by_surprise.py autodiscovery/RUN.json       # a single run
    python agentic/rank_by_surprise.py autodiscovery/*.json         # several explicit runs
    python agentic/rank_by_surprise.py --top 20                     # most surprising N overall
    python agentic/rank_by_surprise.py --rank-by posterior-surprise # high posterior AND high surprise
    python agentic/rank_by_surprise.py --out ranked.json            # also write a file
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# The folder run exports live in, relative to this script (agentic/ -> ../autodiscovery).
DEFAULT_DIR = Path(__file__).resolve().parent.parent / "autodiscovery"


def parse_surprisal(raw) -> float | None:
    """Parse a surprisal value to a float, or None if missing/unparseable."""
    if raw is None:
        return None
    try:
        return float(raw)
    except (ValueError, TypeError):
        return None


def belief_label(p: float | None) -> str:
    """Map a belief probability to a coarse, human-readable label.

    Mirrors the boolean-category buckets the discovery loop uses: a probability
    near 1 means the hypothesis is believed true, near 0 believed false, and the
    middle is genuine uncertainty.
    """
    if p is None:
        return "Unknown"
    if p >= 0.80:
        return "Likely True"
    if p >= 0.60:
        return "Leaning True"
    if p > 0.40:
        return "Uncertain"
    if p > 0.20:
        return "Leaning False"
    return "Likely False"


def direction(prior: float | None, posterior: float | None) -> str:
    """Which way the evidence moved belief: Positive / Negative / Neutral.

    Positive = the experiment raised belief in the hypothesis; Negative =
    lowered it; Neutral = essentially unchanged.
    """
    if prior is None or posterior is None:
        return "Unknown"
    delta = posterior - prior
    if delta > 0.02:
        return "Positive"
    if delta < -0.02:
        return "Negative"
    return "Neutral"


def load_records(json_path: Path) -> list[dict]:
    """Read one run export into a list of hypothesis dicts, tagged with source.

    Tolerates either a bare list of hypotheses or an object wrapping them under
    a ``records``/``hypotheses``/``results`` key.
    """
    with json_path.open(encoding="utf-8-sig") as f:
        data = json.load(f)

    if isinstance(data, dict):
        for key in ("records", "hypotheses", "results", "data"):
            if isinstance(data.get(key), list):
                data = data[key]
                break
        else:
            raise ValueError(
                f"{json_path} is a JSON object with no recognized list of "
                f"hypotheses (looked for records/hypotheses/results/data)."
            )
    if not isinstance(data, list):
        raise ValueError(f"{json_path} is not a JSON list of hypotheses.")
    if not data:
        raise ValueError(f"{json_path} contains no hypotheses.")

    run = json_path.stem  # e.g. "exa-spim-run-1_2026-06-03"
    for r in data:
        r["_source_file"] = json_path.name
        r["_run"] = run
    return data


def priority_score(r: dict) -> float:
    """Combined ``posterior * abs(surprisal)`` priority for a ranked record.

    Rewards hypotheses that are BOTH strongly believed true after the
    experiment (high ``posterior``) AND highly belief-shifting (high surprise
    magnitude). A missing/unparseable posterior is treated as 0 so such records
    sink rather than crash the sort.
    """
    posterior = r.get("posterior")
    p = float(posterior) if isinstance(posterior, (int, float)) else 0.0
    return p * abs(r["_surprisal"])


def rank_records(
    records: list[dict], rank_by: str = "surprise"
) -> tuple[list[dict], list[dict]]:
    """Split into (ranked-with-surprisal, dropped-missing-surprisal).

    ``rank_by`` selects the ordering key, both descending:

    * ``"surprise"`` — surprise MAGNITUDE (|surprisal|).
    * ``"posterior-surprise"`` — combined ``posterior * |surprisal|`` priority,
      so confidently-held surprising findings rank above surprising-but-now-
      disbelieved ones.

    Ties are broken by run then ID so the ordering is stable and reproducible.
    """
    ranked, dropped = [], []
    for r in records:
        s = parse_surprisal(r.get("surprisal"))
        if s is None:
            dropped.append(r)
        else:
            r["_surprisal"] = s
            r["_priority"] = priority_score(r)
            ranked.append(r)

    def id_key(r: dict):
        raw = r.get("id")
        try:
            return (0, int(raw), "")
        except (ValueError, TypeError):
            return (1, 0, str(raw))

    if rank_by == "posterior-surprise":
        primary = lambda r: -r["_priority"]
    else:
        primary = lambda r: -abs(r["_surprisal"])

    ranked.sort(key=lambda r: (primary(r), r.get("_run", ""), id_key(r)))
    return ranked, dropped


def to_records(ranked: list[dict]) -> list[dict]:
    """Reshape ranked hypotheses into compact records for the agent."""
    out = []
    for rank, r in enumerate(ranked, start=1):
        s = r["_surprisal"]
        prior = r.get("prior")
        posterior = r.get("posterior")
        plan = r.get("experimentPlan") or {}
        out.append(
            {
                "rank": rank,
                "run": r.get("_run", ""),
                "source_file": r.get("_source_file", ""),
                "id": r.get("id"),
                "status": r.get("status", ""),
                "surprisal": round(s, 4),
                "surprise_magnitude": round(abs(s), 4),
                "priority_score": round(r.get("_priority", 0.0), 4),
                "is_surprising": bool(r.get("isSurprising", abs(s) >= 0.3)),
                "prior": round(prior, 4) if isinstance(prior, (int, float)) else None,
                "posterior": round(posterior, 4)
                if isinstance(posterior, (int, float))
                else None,
                "belief_before": belief_label(prior),
                "belief_after": belief_label(posterior),
                "direction": direction(prior, posterior),
                "hypothesis": (r.get("hypothesis") or "").strip(),
                "objective": (plan.get("objective") or "").strip(),
                "analysis": (r.get("analysis") or "").strip(),
                "review": (r.get("review") or "").strip(),
            }
        )
    return out


def resolve_paths(raw_paths: list[Path]) -> list[Path]:
    """Resolve CLI paths to a sorted list of run JSON files.

    With no paths given, default to every ``*.json`` in the autodiscovery dir.
    Directory arguments are expanded to their ``*.json`` contents.
    """
    if not raw_paths:
        raw_paths = [DEFAULT_DIR]

    files: list[Path] = []
    for p in raw_paths:
        if p.is_dir():
            files.extend(sorted(p.glob("*.json")))
        else:
            files.append(p)

    # De-duplicate while preserving a stable, sorted order.
    seen, unique = set(), []
    for f in sorted(files):
        if f not in seen:
            seen.add(f)
            unique.append(f)
    return unique


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        type=Path,
        nargs="*",
        help="Run JSON files or directories. Default: all autodiscovery/*.json.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional path to also persist the ranked JSON. Default: stdout only.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=None,
        help="Keep only the N top-ranked hypotheses overall.",
    )
    parser.add_argument(
        "--rank-by",
        choices=["surprise", "posterior-surprise"],
        default="surprise",
        help=(
            "Ranking key: 'surprise' = |surprisal| only (default); "
            "'posterior-surprise' = posterior * |surprisal| (high posterior AND "
            "high surprise first)."
        ),
    )
    args = parser.parse_args(argv)

    files = resolve_paths(args.paths)
    if not files:
        parser.error(f"No run JSON files found (looked in {DEFAULT_DIR}).")

    all_records: list[dict] = []
    per_file: list[dict] = []
    for f in files:
        recs = load_records(f)
        per_file.append({"file": f.name, "n": len(recs)})
        all_records.extend(recs)

    ranked, dropped = rank_records(all_records, rank_by=args.rank_by)
    records = to_records(ranked)
    if args.top is not None:
        records = records[: args.top]

    payload = {
        "source_files": [str(f) for f in files],
        "per_file_counts": per_file,
        "rank_by": args.rank_by,
        "n_total": len(all_records),
        "n_ranked": len(ranked),
        "n_returned": len(records),
        "n_dropped_missing_surprisal": len(dropped),
        "dropped": [
            {"run": r.get("_run", ""), "id": r.get("id")} for r in dropped
        ],
        "surprise_magnitude_max": records[0]["surprise_magnitude"]
        if records
        else None,
        "surprise_magnitude_min": round(abs(ranked[-1]["_surprisal"]), 4)
        if ranked
        else None,
        "priority_score_max": records[0]["priority_score"] if records else None,
        "records": records,
    }
    json_text = json.dumps(payload, indent=2, ensure_ascii=False)

    # Default: ranked JSON to stdout for the agent to consume directly. Only
    # touch the filesystem when --out is explicitly given.
    if args.out is not None:
        args.out.write_text(json_text)
        print(f"Wrote ordered records: {args.out}", file=sys.stderr)
    print(json_text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
