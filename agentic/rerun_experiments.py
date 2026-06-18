"""
Re-execute AutoDiscovery experiments against a provided dataset pkl.

Deterministic helper for the ``discovery-reproducer`` agent
(``.claude/agents/discovery-reproducer.md``). Not a general-purpose tool — its
record schema and stdout contract are what that agent reads. If you change the
output shape here, update the agent prompt to match.

Each AutoDiscovery run export (one JSON file) is a list of tested-hypothesis
objects. Each carries a ``code`` field: a self-contained Python script that
loads a dataset ``.pkl``, runs a statistical test, and prints its results (the
recorded ``codeOutput``). This helper takes ONE run JSON and ONE dataset pkl,
selects the SAME top-ranked records the summarizer reported (using the same
``--rank-by`` / ``--top`` flags), re-runs each record's ``code`` against the
given pkl, and prints the fresh output alongside the recorded one as JSON to
stdout. It does NOT judge reproduction — that is the agent's job; the helper
only re-executes deterministically and reports raw old-vs-new output.

Two execution paths
-------------------
Run directly (forced-load runner): each record's recorded ``code`` runs in a
fresh temp dir through a runner that monkeypatches ``open`` / ``os.path.exists``
/ ``glob.glob`` so any ``.pkl`` reference resolves to the provided pkl. This
works when the host environment can unpickle the dataset and the script's
bootstrap is benign.

Export → revise → rerun (agent-in-the-loop): the recorded scripts often fail to
reproduce for DATA-LOADING / ENVIRONMENT reasons, not analysis reasons — e.g. a
NumPy-2-written pkl that the host's NumPy 1.x can't unpickle, a "dataset not
found" gate the monkeypatch doesn't intercept, or a ``pip install`` retry loop
that times out. For those, ``--export-dir`` dumps each top-K record's code as an
editable ``hypo_<id>.py`` (plus ``MANIFEST.json`` and ``REVISION_GUIDE.md``) so
the agent can revise ONLY the loading/bootstrap part — load the pkl directly
from ``$RERUN_PKL``, pin a compatible NumPy, drop pip-retry loops — keeping the
analysis identical. ``--code-dir`` then executes those revised scripts (falling
back to the recorded ``code`` for any record without a revised file) and reports
which source each result used.

Usage (paths relative to the ``exa-spim-agent/`` project root)
--------------------------------------------------------------
    python agentic/rerun_experiments.py autodiscovery/RUN.json --pkl ../data/RUN.pkl
    python agentic/rerun_experiments.py autodiscovery/RUN.json --pkl DATA.pkl \
        --rank-by posterior-surprise --top 20
    # Agent-in-the-loop: export editable scripts, revise them, then rerun:
    python agentic/rerun_experiments.py autodiscovery/RUN.json --pkl DATA.pkl \
        --top 20 --export-dir autodiscovery/RUN.rerun
    #   (agent edits autodiscovery/RUN.rerun/hypo_<id>.py loading sections)
    python agentic/rerun_experiments.py autodiscovery/RUN.json --pkl DATA.pkl \
        --top 20 --code-dir autodiscovery/RUN.rerun
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# Reuse the exact ranking the summarizer uses so the rerun set matches the
# report set. rank_by_surprise.py lives next to this file.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from rank_by_surprise import load_records, rank_records  # noqa: E402

# Every ``*.pkl`` filename literal a recorded script might open.
_PKL_LITERAL = re.compile(r"""["']([^"']*?\.pkl)["']""")


def pkl_basenames(code: str) -> list[str]:
    """Distinct ``*.pkl`` basenames referenced as literals in a record's code."""
    names = {Path(m).name for m in _PKL_LITERAL.findall(code or "")}
    return sorted(n for n in names if n)


def hypo_filename(rid) -> str:
    """Stable per-record script filename for the export/code dir."""
    safe = re.sub(r"[^0-9A-Za-z_.-]", "_", str(rid))
    return f"hypo_{safe}.py"


# Guidance written into each export dir so the agent knows exactly what it may
# change (the data-loading/bootstrap) and what it must NOT (the analysis).
_REVISION_GUIDE = """\
# Rerun revision guide

Each `hypo_<id>.py` here is one hypothesis's recorded experiment script. The
reruns fail mostly for DATA-LOADING / ENVIRONMENT reasons, not analysis reasons.
Revise ONLY the data-loading and bootstrap part of each script; keep the
statistical ANALYSIS and what it prints byte-for-byte identical so the rerun is
a faithful reproduction.

You MAY change, near the top of each script:
- Replace the dataset search (hardcoded paths, `os.path.exists`, `glob.glob`,
  "dataset not found" gates) with a DIRECT load of the provided pkl:

      import os, pickle
      with open(os.environ["RERUN_PKL"], "rb") as f:
          payload = pickle.load(f)

  Keep the SAME variable name the rest of the script uses (e.g. `payload`,
  `data`) and the same downstream keys (`fragments_graph`, `gt_graph`,
  `gt_edge_error`, `gt_node_canonical_label`, `gt_merge_sites`, ...).
- Fix environment/version problems that block the load, e.g. a pkl written with
  NumPy 2 that won't unpickle under NumPy 1.x — pin/upgrade inside the script
  (`subprocess.check_call([sys.executable,"-m","pip","install","-q","numpy>=2"])`
  BEFORE importing numpy), or otherwise make the unpickle succeed.
- Remove `pip install` RETRY LOOPS that re-install on every iteration and cause
  timeouts; install each dependency at most once, quietly, up front.

You MUST NOT change:
- The statistical test, its parameters, the sampling/grouping logic, the effect
  the script computes, or the lines it prints. The whole point is to re-measure
  the SAME analysis on the real data.

Print a clear `Loading dataset from: $RERUN_PKL` line so the log shows the fix
took effect. `MPLBACKEND=Agg` is already set (headless plotting is fine).
"""


def export_scripts(ranked: list[dict], pkl: Path, export_dir: Path) -> dict:
    """Write each ranked record's code as an editable script + manifest/guide.

    Returns a small summary dict for stdout. The agent edits the ``hypo_<id>.py``
    files (loading section only) and then a ``--code-dir`` rerun executes them.
    """
    export_dir.mkdir(parents=True, exist_ok=True)
    (export_dir / "REVISION_GUIDE.md").write_text(_REVISION_GUIDE, encoding="utf-8")

    manifest = []
    for rank, r in enumerate(ranked, start=1):
        rid = r.get("id")
        fname = hypo_filename(rid)
        code = r.get("code") or ""
        (export_dir / fname).write_text(code, encoding="utf-8")
        manifest.append(
            {
                "rank": rank,
                "id": rid,
                "file": fname,
                "status": r.get("status", ""),
                "surprisal": round(r["_surprisal"], 4),
                "priority_score": round(r.get("_priority", 0.0), 4),
                "hypothesis": (r.get("hypothesis") or "").strip(),
                "pkl_basenames_in_code": pkl_basenames(code),
                "has_code": bool(code.strip()),
            }
        )
    (export_dir / "MANIFEST.json").write_text(
        json.dumps(
            {"pkl": str(pkl), "n": len(manifest), "records": manifest},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return {
        "export_dir": str(export_dir),
        "n_exported": len(manifest),
        "guide": str(export_dir / "REVISION_GUIDE.md"),
        "manifest": str(export_dir / "MANIFEST.json"),
    }


def resolve_code(r: dict, code_dir: Path | None) -> tuple[str, str]:
    """Return ``(code, source)`` for a record.

    With ``--code-dir``, prefer the agent's revised ``hypo_<id>.py`` when present
    (``source='revised'``); otherwise fall back to the recorded JSON ``code``
    (``source='recorded'``). Without ``--code-dir`` always use the recorded code.
    """
    recorded = r.get("code") or ""
    if code_dir is not None:
        revised = code_dir / hypo_filename(r.get("id"))
        if revised.is_file():
            text = revised.read_text(encoding="utf-8")
            if text.strip():
                return text, "revised"
    return recorded, "recorded"


# The recorded scripts locate the dataset through a wide variety of hardcoded
# paths and relative globs (``../data/<name>.pkl``, ``../*_add.pkl``,
# ``**/*_add.pkl``, ``cache/*_add.pkl``, ``/data/...`` …) — there is no single
# path or env hook to set. So rather than trying to make every path-search
# *find* the provided pkl, we force the load: a small runner monkeypatches the
# handful of filesystem calls the scripts use to locate and open a ``.pkl`` so
# that ANY ``.pkl`` reference resolves to the provided dataset, whatever path
# logic the script runs. All 100 records in the run export load via
# ``pickle.load(open(<path>, "rb"))`` after an ``os.path.exists`` / ``glob.glob``
# search, which these patches cover.
_RUNNER = r'''
import builtins, glob, os, runpy, sys

_PKL = os.environ["RERUN_PKL"]


def _is_pkl(x):
    try:
        s = os.fspath(x)
    except TypeError:
        return False
    return isinstance(s, str) and s.endswith(".pkl")


# Any open() of a *.pkl path -> open the provided dataset instead.
_orig_open = builtins.open
def _open(file, *a, **k):
    return _orig_open(_PKL if _is_pkl(file) else file, *a, **k)
builtins.open = _open

# A *.pkl path always "exists" so the script takes its load branch.
_orig_exists = os.path.exists
os.path.exists = lambda p: True if _is_pkl(p) else _orig_exists(p)
_orig_isfile = os.path.isfile
os.path.isfile = lambda p: True if _is_pkl(p) else _orig_isfile(p)

# Any glob for *.pkl resolves to exactly the provided dataset.
_orig_glob = glob.glob
def _glob(pathname, *a, **k):
    if isinstance(pathname, str) and pathname.endswith(".pkl"):
        return [_PKL]
    return _orig_glob(pathname, *a, **k)
glob.glob = _glob

# pathlib.Path(...).exists()/is_file() for *.pkl, in case a script uses it.
from pathlib import Path as _Path
_orig_p_exists, _orig_p_isfile = _Path.exists, _Path.is_file
_Path.exists = lambda self: True if str(self).endswith(".pkl") else _orig_p_exists(self)
_Path.is_file = lambda self: True if str(self).endswith(".pkl") else _orig_p_isfile(self)

runpy.run_path("experiment.py", run_name="__main__")
'''


def rerun_one(code: str, pkl: Path, timeout: int) -> dict:
    """Execute one record's recorded ``code``, forcing every pkl load to ``pkl``.

    The recorded script is written verbatim to ``experiment.py`` and launched
    through ``_runner.py``, which monkeypatches ``open`` / ``os.path.exists`` /
    ``glob.glob`` (and the ``pathlib`` equivalents) so the script's own
    path-finding always resolves to the provided dataset (passed via the
    ``RERUN_PKL`` env var) regardless of the hardcoded paths or globs it uses.
    """
    with tempfile.TemporaryDirectory(prefix="rerun_") as tmp:
        run = Path(tmp)
        (run / "experiment.py").write_text(code, encoding="utf-8")
        (run / "_runner.py").write_text(_RUNNER, encoding="utf-8")
        start = time.monotonic()
        env = {
            **os.environ,
            "MPLBACKEND": "Agg",  # headless plotting
            "RERUN_PKL": str(pkl),
        }
        try:
            proc = subprocess.run(
                [sys.executable, "_runner.py"],
                cwd=str(run),
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
            )
            return {
                "timed_out": False,
                "exitcode": proc.returncode,
                "stdout": proc.stdout,
                "stderr": proc.stderr,
                "runtime_ms": round((time.monotonic() - start) * 1000),
            }
        except subprocess.TimeoutExpired as e:
            # On timeout the partial stdout/stderr may come back as bytes even
            # with text=True; decode defensively before concatenating.
            def _txt(x):
                if isinstance(x, bytes):
                    return x.decode("utf-8", "replace")
                return x or ""

            return {
                "timed_out": True,
                "exitcode": None,
                "stdout": _txt(e.stdout),
                "stderr": _txt(e.stderr) + f"\n[rerun] timed out after {timeout}s",
                "runtime_ms": round((time.monotonic() - start) * 1000),
            }


def truncate(text: str | None, limit: int) -> str:
    """Trim long captured output to keep the stdout payload manageable."""
    text = text or ""
    if len(text) <= limit:
        return text
    head = text[: limit // 2]
    tail = text[-limit // 2 :]
    return f"{head}\n…[{len(text) - limit} chars omitted]…\n{tail}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "json_file", type=Path, help="The run JSON export to re-execute."
    )
    parser.add_argument(
        "--pkl",
        type=Path,
        required=True,
        help="The dataset .pkl this run's experiments load.",
    )
    parser.add_argument(
        "--rank-by",
        choices=["surprise", "posterior-surprise"],
        default="posterior-surprise",
        help="Ranking key (must match the summarizer's) for picking the top-K.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=20,
        help="Re-run only the N top-ranked records (the ones in the report).",
    )
    parser.add_argument(
        "--export-dir",
        type=Path,
        default=None,
        help=(
            "Write each top-K record's code as an editable hypo_<id>.py (plus "
            "MANIFEST.json and REVISION_GUIDE.md) into this dir, then exit "
            "without running. Edit the loading sections and rerun with --code-dir."
        ),
    )
    parser.add_argument(
        "--code-dir",
        type=Path,
        default=None,
        help=(
            "Execute the revised hypo_<id>.py from this dir instead of the JSON "
            "code (per record; falls back to the recorded code when no revised "
            "file exists). Typically the same dir a prior --export-dir wrote."
        ),
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=1800,
        help="Per-experiment timeout in seconds (default 1800 = 30 min).",
    )
    parser.add_argument(
        "--max-output-chars",
        type=int,
        default=6000,
        help="Truncate each recorded/fresh output to this many chars.",
    )
    args = parser.parse_args(argv)

    if not args.json_file.is_file():
        parser.error(f"No such run JSON: {args.json_file}")
    if not args.pkl.is_file():
        parser.error(f"No such dataset pkl: {args.pkl}")
    if args.export_dir is not None and args.code_dir is not None:
        parser.error("--export-dir and --code-dir are mutually exclusive.")
    pkl = args.pkl.resolve()

    records = load_records(args.json_file)
    ranked, _ = rank_records(records, rank_by=args.rank_by)
    if args.top is not None:
        ranked = ranked[: args.top]

    # Export mode: dump editable scripts for the agent to revise, then exit.
    if args.export_dir is not None:
        summary = export_scripts(ranked, pkl, args.export_dir)
        print(
            f"[rerun] exported {summary['n_exported']} script(s) to "
            f"{summary['export_dir']}",
            file=sys.stderr,
            flush=True,
        )
        print(json.dumps({"mode": "export", **summary}, indent=2, ensure_ascii=False))
        return 0

    results = []
    n_ok = n_failed = n_timeout = 0
    n_revised = 0
    for rank, r in enumerate(ranked, start=1):
        code, source = resolve_code(r, args.code_dir)
        if source == "revised":
            n_revised += 1
        rid = r.get("id")
        print(
            f"[rerun {rank}/{len(ranked)}] id={rid} ({source}) …",
            file=sys.stderr,
            flush=True,
        )
        if not code.strip():
            run = {
                "timed_out": False,
                "exitcode": None,
                "stdout": "",
                "stderr": "[rerun] record has no 'code' to execute",
                "runtime_ms": 0,
            }
            outcome = "NO-CODE"
            n_failed += 1
        else:
            run = rerun_one(code, pkl, args.timeout)
            if run["timed_out"]:
                outcome = "TIMEOUT"
                n_timeout += 1
            elif run["exitcode"] == 0:
                outcome = "OK"
                n_ok += 1
            else:
                outcome = f"FAILED(exit={run['exitcode']})"
                n_failed += 1

        # Progress log after each hypothesis: outcome, this run's wall time, and
        # the running tally so a long rerun is observable as it goes.
        print(
            f"[rerun {rank}/{len(ranked)}] id={rid} {outcome} "
            f"in {run['runtime_ms'] / 1000:.0f}s "
            f"(ok={n_ok} failed={n_failed} timeout={n_timeout})",
            file=sys.stderr,
            flush=True,
        )

        results.append(
            {
                "rank": rank,
                "id": rid,
                "status": r.get("status", ""),
                "code_source": source,
                "surprisal": round(r["_surprisal"], 4),
                "priority_score": round(r.get("_priority", 0.0), 4),
                "hypothesis": (r.get("hypothesis") or "").strip(),
                "pkl_basenames_in_code": pkl_basenames(code),
                "recorded_output": truncate(
                    str(r.get("codeOutput") or ""), args.max_output_chars
                ),
                "rerun_exitcode": run["exitcode"],
                "rerun_timed_out": run["timed_out"],
                "rerun_runtime_ms": run["runtime_ms"],
                "rerun_stdout": truncate(run["stdout"], args.max_output_chars),
                "rerun_stderr": truncate(run["stderr"], args.max_output_chars),
            }
        )

    payload = {
        "json_file": str(args.json_file),
        "pkl": str(pkl),
        "rank_by": args.rank_by,
        "top": args.top,
        "timeout_s": args.timeout,
        "code_dir": str(args.code_dir) if args.code_dir is not None else None,
        "n_rerun": len(results),
        "n_revised": n_revised,
        "n_recorded": len(results) - n_revised,
        "n_ok": n_ok,
        "n_failed": n_failed,
        "n_timeout": n_timeout,
        "results": results,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
