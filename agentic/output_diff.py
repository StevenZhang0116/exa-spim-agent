"""
Numeric diff of a recorded AutoDiscovery output against its rerun output.

The reproduction fold judges REPRODUCED / DIVERGED by comparing the numbers a
script printed in the AutoDiscovery sandbox with the numbers it prints here.
Done by reading 91 pairs of outputs by eye, that misses lines; and the results
JSON truncates long outputs, so the fold of new-merge-1 (2026-10-19) had to dig
the full text out of the checkpoint. This tool gives the fold the complete list
of numeric differences per record, deterministically. It does not judge: the
agent decides which differences are material.

How it compares
---------------
Each output is split into lines. A line's *template* is the line with every
number replaced by ``#`` (whitespace collapsed); its *values* are the numbers in
order. Lines whose template matches a noise pattern (timings, throughput,
paths, dates, progress counters, pip notices, glog lines) are ignored. Lines of
the two outputs are paired by template, repeated templates in order of
occurrence (so ``Fold #: AP=#`` pairs fold 0 with fold 0). Paired values are
exact, tiny (within ``--rel-tol`` or ``--abs-tol``: fit noise, rounding) or
material. Templates present on one side only are listed too.

Usage
-----
    python agentic/output_diff.py RUN.json RESULTS.json [--outputs-dir DIR] > DIFF.json

``RESULTS.json`` is a ``rerun_experiments.py`` payload. The rerun text comes
from ``--outputs-dir`` (the runner's full per-record stdout files) when given,
else from the payload's possibly truncated ``rerun_stdout``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_NUMBER = re.compile(r"(?<![\w.])[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?(?![\w.])")
_NOISE = [re.compile(p, re.IGNORECASE) for p in (
    r"#\s*(?:s|sec|secs|seconds|ms|min|minutes|h|hours)\b",      # durations
    r"\belapsed|\blatency|\bruntime|\bwall[- ]?time|\btook\b|\btime[:=]",
    r"/s\b|\bper second|it/s|patches/s|rows/s",                  # throughput
    r"[A-Za-z0-9_.-]*/[A-Za-z0-9_.-]+",                          # paths and URIs
    r"(?<!\S)#:#(?::#)?(?!\S)|#-#-#",                            # clock times, dates
    r"^progress\b|\bprogress:|\[#/#\]|\b#\s*/\s*#\s+(?:candidates|patches|rows)\b",
    r"\[notice\]|^[EWIF]#\s|\bpid\b|\bthreads?\b|\bworkers?\b",  # pip, glog, process info
    r"\b(?:GB|GiB|MB|MiB|KB|KiB|bytes)\b|\bmemory\b|\brss\b",   # memory sizes
    r"\bloading\b|\bloaded in\b|\bpreload|\bcache(?:d)? (?:hit|miss)",
    r"^code output:$|^exitcode:",                                 # the AD wrapper
)]
_MAX_LISTED = 60


def recorded_stdout(code_output: str) -> str:
    """The script's stdout inside AutoDiscovery's ``codeOutput`` wrapper."""
    text = code_output or ""
    head, sep, _ = text.partition("\nSTDERR:")
    text = head if sep else text
    lines = text.splitlines()
    if lines and lines[0].startswith("exitcode:"):
        lines = lines[1:]
    if lines and lines[0].startswith("Code output:"):
        lines[0] = lines[0][len("Code output:"):].lstrip()
    return "\n".join(lines)


def _template(line: str) -> tuple[str, list[float]]:
    values = [float(m.group(0)) for m in _NUMBER.finditer(line)]
    template = " ".join(_NUMBER.sub("#", line).split())
    return template, values


def numeric_lines(text: str) -> list[tuple[str, list[float], str]]:
    """``(template, values, original line)`` of every informative line with a number."""
    out = []
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        template, values = _template(line)
        if not values or any(p.search(template) for p in _NOISE):
            continue
        out.append((template, values, line))
    return out


def _close(a: float, b: float, rel_tol: float, abs_tol: float) -> bool:
    return abs(a - b) <= max(abs_tol, rel_tol * max(abs(a), abs(b)))


def diff_outputs(recorded: str, rerun: str, *, rel_tol: float = 1e-3,
                 abs_tol: float = 1e-4) -> dict:
    """Compare two outputs; see the module docstring for the pairing rules."""
    left, right = numeric_lines(recorded), numeric_lines(rerun)
    by_template_right: dict[str, list[tuple[list[float], str]]] = {}
    for template, values, line in right:
        by_template_right.setdefault(template, []).append((values, line))
    used: dict[str, int] = {}
    counts = {"compared": 0, "exact": 0, "tiny": 0, "material": 0}
    mismatches, only_recorded = [], []
    for template, values, line in left:
        pool = by_template_right.get(template)
        index = used.get(template, 0)
        if pool is None or index >= len(pool):
            only_recorded.append(line)
            continue
        used[template] = index + 1
        other_values, other_line = pool[index]
        if len(values) != len(other_values):
            counts["material"] += 1
            mismatches.append({"recorded": line, "rerun": other_line, "note": "value count"})
            continue
        for a, b in zip(values, other_values):
            counts["compared"] += 1
            if a == b:
                counts["exact"] += 1
            elif _close(a, b, rel_tol, abs_tol):
                counts["tiny"] += 1
            else:
                counts["material"] += 1
                rel = abs(a - b) / max(abs(a), abs(b), 1e-12)
                mismatches.append({"recorded": line, "rerun": other_line,
                                   "recorded_value": a, "rerun_value": b,
                                   "rel_diff": round(rel, 6)})
    only_rerun = [line for template, pool in by_template_right.items()
                  for _, line in pool[used.get(template, 0):]]
    if counts["material"]:
        hint = "numeric-differences"
    elif only_recorded or only_rerun:
        hint = "structural-differences"
    else:
        hint = "identical"
    return {
        "hint": hint,
        "counts": counts,
        "mismatches": mismatches[:_MAX_LISTED],
        "n_mismatches": len(mismatches),
        "only_recorded": only_recorded[:_MAX_LISTED],
        "n_only_recorded": len(only_recorded),
        "only_rerun": only_rerun[:_MAX_LISTED],
        "n_only_rerun": len(only_rerun),
    }


def _rerun_text(result: dict, outputs_dir: Path | None) -> tuple[str, str]:
    """The rerun stdout and where it came from (full file or the payload)."""
    if outputs_dir is not None:
        rel = result.get("rerun_stdout_path")
        path = Path(rel) if rel else outputs_dir / f"hypo_{result.get('id')}.stdout.txt"
        if not path.is_absolute():
            path = Path.cwd() / path
        try:
            return path.read_text(encoding="utf-8", errors="replace"), "file"
        except OSError:
            pass
    return str(result.get("rerun_stdout") or ""), "payload"


def diff_results(run_json: Path, results_json: Path, outputs_dir: Path | None = None,
                 **tolerances) -> dict:
    records = {str(r.get("id")): r for r in json.loads(run_json.read_text(encoding="utf-8"))}
    payload = json.loads(results_json.read_text(encoding="utf-8"))
    entries, summary = {}, {"identical": 0, "numeric-differences": 0,
                            "structural-differences": 0, "not-usable": 0}
    for result in payload.get("results") or []:
        rid = str(result.get("id"))
        if result.get("result_status") != "USABLE":
            entries[rid] = {"hint": "not-usable",
                            "reason": result.get("result_failure_reason")}
            summary["not-usable"] += 1
            continue
        record = records.get(rid) or {}
        rerun, source = _rerun_text(result, outputs_dir)
        entry = diff_outputs(recorded_stdout(record.get("codeOutput") or ""), rerun,
                             **tolerances)
        entry["rerun_source"] = source
        entries[rid] = entry
        summary[entry["hint"]] += 1
    return {
        "run_json": str(run_json),
        "results_json": str(results_json),
        "outputs_dir": str(outputs_dir) if outputs_dir else None,
        "tolerances": tolerances,
        "summary": summary,
        "entries": entries,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("Usage")[0])
    parser.add_argument("run_json", type=Path)
    parser.add_argument("results_json", type=Path)
    parser.add_argument("--outputs-dir", type=Path, default=None,
                        help="Directory of the runner's full per-record stdout files.")
    parser.add_argument("--rel-tol", type=float, default=1e-3)
    parser.add_argument("--abs-tol", type=float, default=1e-4)
    args = parser.parse_args(argv)
    payload = diff_results(args.run_json, args.results_json, args.outputs_dir,
                           rel_tol=args.rel_tol, abs_tol=args.abs_tol)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"[diff] {payload['summary']}", file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
