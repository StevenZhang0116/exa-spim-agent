"""
Collect and parse every AutoDiscovery ``.summary.md`` report into one structured
JSON payload, pooled across files, for the ``discovery-consolidator`` agent.

Deterministic helper for ``.claude/agents/discovery-consolidator.md`` (the
cross-report consolidation step). Not a general-purpose tool — its record schema
and stdout contract are what that agent reads. If you change the output shape
here, update the agent prompt to match. It is the read-side analogue of
``rank_by_surprise.py``: that script ranks the raw JSON run exports; this one
ranks/reshapes the *finished* per-run Markdown reports the workflow already
produced, so the consolidator can cluster equivalent findings ACROSS reports.

Each input is a workflow report written by ``run_discovery_workflow.py``:
``autodiscovery/<RUN>.summary.md``. Every ``### N. (Priority X · Surprise Y)
<title>`` block is one ranked hypothesis, followed by ``- **Field:** value``
bullets (Tested, Conclusion, Caveats, Reproduction, Generalization, Verdict,
Post-correction verdict, …). The compound ``- **Run:** <run> · **ID:** <id> ·
**Belief:** … · **Direction:** …`` bullet carries the entry's stable identity.

This script gathers the entries from ALL matched reports and prints them as one
JSON object to stdout. It does NOT cluster, dedupe, judge, or summarize — that
semantic work is the agent's job. It only finds files, parses entries, and
reshapes them so the agent's job is purely the cross-report consolidation
writing.

File selection
--------------
By default it takes every ``autodiscovery/*.md`` EXCEPT the ``*.zh.md`` Chinese
translations (which are not independent results — they are localizations of
their English sibling) and any consolidated output it would itself write
(``*combined*.md``). Pass explicit paths to override, or ``--include-zh`` /
``--include-combined`` to keep those.

Usage (paths shown relative to the ``exa-spim-agent/`` project root)
--------------------------------------------------------------------
    python agentic/collect_summaries.py                      # all autodiscovery/*.summary.md
    python agentic/collect_summaries.py autodiscovery/A.summary.md autodiscovery/B.summary.md
    python agentic/collect_summaries.py --glob '*.summary.md'  # explicit glob
    python agentic/collect_summaries.py --out collected.json   # also write a file
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# The folder reports live in, relative to this script (agentic/ -> ../autodiscovery).
DEFAULT_DIR = Path(__file__).resolve().parent.parent / "autodiscovery"

# A ranked-entry header, e.g.
#   ### 1. (Priority 0.324 · Surprise 0.351) Split-gap connections obey a …
#   ### 7. (Surprise 0.307) Merge sites have a 75% higher …      (no priority)
_ENTRY_RE = re.compile(r"^###\s+(\d+)\.\s*(?:\((?P<scores>[^)]*)\))?\s*(?P<title>.*)$")
# A bolded field label inside a bullet: **Tested:** … or **Run:** … · **ID:** …
_FIELD_RE = re.compile(r"\*\*(?P<key>[^*]+?):\*\*\s*(?P<val>.*?)\s*(?=\*\*[^*]+?:\*\*|$)")
_PRIORITY_RE = re.compile(r"Priority\s+([0-9.]+)")
_SURPRISE_RE = re.compile(r"Surprise\s+([0-9.]+)")


def _to_float(text: str) -> float | None:
    try:
        return float(text)
    except (ValueError, TypeError):
        return None


def parse_fields(line: str) -> dict[str, str]:
    """Extract every ``**Key:** value`` pair from one bullet line.

    Handles both a single-field bullet (``- **Tested:** …``) and the compound
    identity bullet (``- **Run:** … · **ID:** … · **Belief:** … · **Direction:**
    …``) by scanning for all bold labels on the line.
    """
    out: dict[str, str] = {}
    for m in _FIELD_RE.finditer(line):
        key = m.group("key").strip()
        val = m.group("val").strip().rstrip("·").strip()
        if key:
            out[key] = val
    return out


def parse_report(path: Path) -> dict:
    """Parse one ``<RUN>.summary.md`` into a header blob + list of entry dicts.

    Returns ``{"file", "run", "title", "entries": [...]}`` where each entry is
    ``{num, priority_score, surprise_magnitude, title, run, id, belief,
    direction, fields}``. ``fields`` is the full ``label -> text`` map of every
    bullet under the entry (Tested, Conclusion, Caveats, Reproduction,
    Generalization, Verdict, Post-correction verdict, …), so nothing analytical
    is dropped. Identity (``run``/``id``) is read from the compound first bullet.
    """
    text = path.read_text(encoding="utf-8-sig")
    lines = text.splitlines()

    # Report-level H1 title (first "# " heading), informational only.
    report_title = ""
    for ln in lines:
        if ln.startswith("# "):
            report_title = ln[2:].strip()
            break

    entries: list[dict] = []
    current: dict | None = None

    def close(entry: dict | None) -> None:
        if entry is not None:
            entries.append(entry)

    for ln in lines:
        m = _ENTRY_RE.match(ln)
        if m:
            close(current)
            scores = m.group("scores") or ""
            pr = _PRIORITY_RE.search(scores)
            su = _SURPRISE_RE.search(scores)
            current = {
                "num": int(m.group(1)),
                "priority_score": _to_float(pr.group(1)) if pr else None,
                "surprise_magnitude": _to_float(su.group(1)) if su else None,
                "title": m.group("title").strip(),
                "run": "",
                "id": None,
                "belief": "",
                "direction": "",
                "fields": {},
            }
            continue
        if current is None:
            continue
        # A new top-level "## " section ends the ranked-entries region.
        if ln.startswith("## "):
            close(current)
            current = None
            continue
        stripped = ln.lstrip()
        if stripped.startswith("- ") and "**" in stripped:
            fields = parse_fields(stripped)
            for k, v in fields.items():
                # First bullet usually carries Run/ID/Belief/Direction inline.
                lk = k.lower()
                if lk == "run":
                    current["run"] = v
                elif lk == "id":
                    current["id"] = v
                elif lk == "belief":
                    current["belief"] = v
                elif lk == "direction":
                    current["direction"] = v
                else:
                    current["fields"][k] = v

    close(current)

    run = entries[0]["run"] if entries and entries[0].get("run") else path.stem
    return {
        "file": path.name,
        "run": run,
        "title": report_title,
        "entries": entries,
    }


def resolve_paths(
    raw_paths: list[Path],
    glob: str,
    include_zh: bool,
    include_combined: bool,
) -> list[Path]:
    """Resolve CLI paths/globs to a sorted, de-duplicated list of report files.

    With no explicit paths, glob the autodiscovery dir. Drop ``*.zh.md``
    translations and ``*combined*`` consolidated outputs unless asked to keep
    them, so the consolidator never ingests a localization or its own product.
    """
    files: list[Path] = []
    if raw_paths:
        for p in raw_paths:
            if p.is_dir():
                files.extend(sorted(p.glob(glob)))
            else:
                files.append(p)
    else:
        files.extend(sorted(DEFAULT_DIR.glob(glob)))

    def keep(f: Path) -> bool:
        name = f.name.lower()
        if not include_zh and name.endswith(".zh.md"):
            return False
        if not include_combined and "combined" in name:
            return False
        return True

    seen, unique = set(), []
    for f in sorted(files):
        if f in seen or not keep(f):
            continue
        seen.add(f)
        unique.append(f)
    return unique


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        type=Path,
        nargs="*",
        help="Report .md files or directories. Default: all autodiscovery/*.summary.md.",
    )
    parser.add_argument(
        "--glob",
        default="*.summary.md",
        help="Glob used when scanning a directory / the default dir (default: *.summary.md).",
    )
    parser.add_argument(
        "--include-zh",
        action="store_true",
        help="Also include *.zh.md Chinese translations (excluded by default).",
    )
    parser.add_argument(
        "--include-combined",
        action="store_true",
        help="Also include *combined* consolidated reports (excluded by default).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional path to also persist the collected JSON. Default: stdout only.",
    )
    args = parser.parse_args(argv)

    files = resolve_paths(args.paths, args.glob, args.include_zh, args.include_combined)
    if not files:
        parser.error(
            f"No report files found (looked for '{args.glob}' in {DEFAULT_DIR})."
        )

    reports: list[dict] = []
    per_file: list[dict] = []
    n_entries_total = 0
    for f in files:
        rep = parse_report(f)
        reports.append(rep)
        per_file.append({"file": rep["file"], "run": rep["run"], "n_entries": len(rep["entries"])})
        n_entries_total += len(rep["entries"])

    # Flat, pooled list of every entry tagged with its source report, so the
    # agent can cluster equivalent findings across reports without re-reading
    # the files. Each keeps source_file/run/id as a stable cross-report key.
    entries: list[dict] = []
    for rep in reports:
        for e in rep["entries"]:
            entries.append(
                {
                    "source_file": rep["file"],
                    "run": e.get("run") or rep["run"],
                    "id": e.get("id"),
                    "num": e.get("num"),
                    "priority_score": e.get("priority_score"),
                    "surprise_magnitude": e.get("surprise_magnitude"),
                    "title": e.get("title", ""),
                    "belief": e.get("belief", ""),
                    "direction": e.get("direction", ""),
                    "fields": e.get("fields", {}),
                }
            )

    payload = {
        "source_files": [str(f) for f in files],
        "per_file_counts": per_file,
        "n_files": len(files),
        "n_entries_total": n_entries_total,
        "entries": entries,
    }
    json_text = json.dumps(payload, indent=2, ensure_ascii=False)

    if args.out is not None:
        args.out.write_text(json_text, encoding="utf-8")
        print(f"Wrote collected records: {args.out}", file=sys.stderr)
    print(json_text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
