"""
Rank AutoDiscovery hypotheses by surprise.

Deterministic helper for the ``discovery-summarizer`` agent
(``.claude/agents/discovery-summarizer.md``). Not a general-purpose tool — its
record schema and stdout contract are what that agent reads. If you change the
output shape here, update the agent prompt to match.

Parses an AutoDiscovery run CSV (one row per tested hypothesis, with columns
ID, Hypothesis, Surprisal, Belief Before, Belief After, Direction, Analysis,
Review, Objective, Steps, Deliverables), sorts the rows by ``Surprisal`` in
descending order, and prints the ordered records as JSON to stdout for the
agent to read and turn into per-hypothesis scientific conclusions.

This script is intentionally deterministic: it does NOT summarize or call any
model. It only parses, validates, sorts, and reshapes the run export so the
agent's job is purely the scientific-conclusion writing.

By default nothing is written to disk — the ranked JSON goes to stdout so the
agent can consume it directly. Pass ``--out PATH`` only if you explicitly want
to persist it.

Usage (paths shown relative to the ``exa-spim-agent/`` project root)
--------------------------------------------------------------------
    python scripts/rank_by_surprise.py autodiscovery/RUN.csv           # JSON to stdout
    python scripts/rank_by_surprise.py autodiscovery/RUN.csv --top 10  # most surprising N
    python scripts/rank_by_surprise.py autodiscovery/RUN.csv --out o.json  # also write a file
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

# Columns we expect in an AutoDiscovery run export. Missing columns are
# tolerated (filled with "") so the script survives schema drift, but the
# absence of the ranking key (Surprisal) is a hard error.
EXPECTED_COLUMNS = [
    "ID",
    "Hypothesis",
    "Surprisal",
    "Belief Before",
    "Belief After",
    "Direction",
    "Analysis",
    "Review",
    "Objective",
    "Steps",
    "Deliverables",
]


def parse_surprisal(raw: str) -> float | None:
    """Parse the Surprisal cell to a float, or None if blank/unparseable."""
    raw = (raw or "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def load_rows(csv_path: Path) -> list[dict]:
    """Read the run CSV into a list of dicts.

    Uses ``utf-8-sig`` so the leading BOM some exports carry on the first
    header (``﻿ID``) is stripped automatically.
    """
    with csv_path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{csv_path} has no header row.")
        if "Surprisal" not in reader.fieldnames:
            raise ValueError(
                f"{csv_path} is missing the 'Surprisal' column; found "
                f"{reader.fieldnames}. Cannot rank by surprise."
            )
        rows = [dict(r) for r in reader]
    if not rows:
        raise ValueError(f"{csv_path} has a header but no data rows.")
    return rows


def rank_rows(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Split rows into (ranked-with-surprisal, dropped-missing-surprisal).

    Ranked rows are sorted by surprisal descending; ties broken by ID so the
    ordering is stable and reproducible across runs.
    """
    ranked, dropped = [], []
    for r in rows:
        s = parse_surprisal(r.get("Surprisal", ""))
        if s is None:
            dropped.append(r)
        else:
            r = dict(r)
            r["_surprisal"] = s
            ranked.append(r)

    def id_key(r: dict):
        raw = (r.get("ID") or "").strip()
        try:
            return (0, int(raw))
        except ValueError:
            return (1, raw)

    ranked.sort(key=lambda r: (-r["_surprisal"], id_key(r)))
    return ranked, dropped


def to_records(ranked: list[dict]) -> list[dict]:
    """Reshape ranked rows into compact records for the agent."""
    records = []
    for rank, r in enumerate(ranked, start=1):
        records.append(
            {
                "rank": rank,
                "id": (r.get("ID") or "").strip(),
                "surprisal": round(r["_surprisal"], 4),
                "belief_before": (r.get("Belief Before") or "").strip(),
                "belief_after": (r.get("Belief After") or "").strip(),
                "direction": (r.get("Direction") or "").strip(),
                "hypothesis": (r.get("Hypothesis") or "").strip(),
                "objective": (r.get("Objective") or "").strip(),
                "analysis": (r.get("Analysis") or "").strip(),
                "review": (r.get("Review") or "").strip(),
            }
        )
    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path, help="AutoDiscovery run CSV.")
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
        help="Keep only the N most surprising hypotheses.",
    )
    args = parser.parse_args(argv)

    rows = load_rows(args.csv_path)
    ranked, dropped = rank_rows(rows)
    records = to_records(ranked)
    if args.top is not None:
        records = records[: args.top]

    payload = {
        "source_csv": str(args.csv_path),
        "n_total": len(rows),
        "n_ranked": len(ranked),
        "n_dropped_missing_surprisal": len(dropped),
        "dropped_ids": [(r.get("ID") or "").strip() for r in dropped],
        "surprisal_max": records[0]["surprisal"] if records else None,
        "surprisal_min": ranked[-1]["_surprisal"] if ranked else None,
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
