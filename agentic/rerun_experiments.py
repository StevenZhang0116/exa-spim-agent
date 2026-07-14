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
reproduce because they cannot LOCATE the dataset — a "dataset not found" gate the
monkeypatch doesn't intercept, or a hardcoded path/glob that finds nothing. For
those, ``--export-dir`` dumps each top-K record's code as an editable
``hypo_<id>.py`` (plus ``MANIFEST.json`` and ``REVISION_GUIDE.md``) so the agent
can revise ONLY the loading part — load the pkl directly from ``$RERUN_PKL`` —
keeping the analysis identical. ``--code-dir`` then executes those revised
scripts (falling back to the recorded ``code`` for any record without a revised
file) and reports which source each result used.

The driver owns the ENVIRONMENT, not the scripts. The host interpreter is the
single source of truth and is pre-provisioned with the scientific stack; the
runner neutralizes any ``pip``/``apt``/``conda`` install the recorded code
attempts (they become logged no-ops) so a script can neither swap the NumPy
version mid-run nor hang in an install retry loop. Before running anything, a
one-time preflight imports numpy/pandas/scipy/statsmodels and aborts loudly
(exit 3) if the env itself is broken; if every script fails with an
import/native-load error and none succeed, the run is flagged
``environment_failure`` and also exits 3 — so a broken env can never be folded
as a batch of "could not run" statistical verdicts.

Corrected statistical test (fix a wrong test, then re-measure): when the
verifier flags a hypothesis whose STATISTICAL TEST is wrong (wrong test for the
data, violated assumptions, p-value misread, huge-n significance of a trivial
effect), ``--corrected-dir`` runs ``hypo_<id>.py`` scripts whose ANALYSIS has
been rewritten to the correct test (built on the loading-fixed script, keeping
the data and quantities the same). It takes precedence over ``--code-dir`` per
record, so the corrected result is real measured numbers, not a paper proposal.

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
    # Extrapolate: run the reproduced code on OTHER datasets to test generalization
    python agentic/rerun_experiments.py autodiscovery/RUN.json --pkl DATA.pkl \
        --code-dir autodiscovery/RUN.rerun \
        --extra-pkl OTHER1.pkl --extra-pkl OTHER2.pkl
    # Corrected tests: re-measure the flagged hypotheses with the right test
    python agentic/rerun_experiments.py autodiscovery/RUN.json --pkl DATA.pkl \
        --code-dir autodiscovery/RUN.rerun --corrected-dir autodiscovery/RUN.fixed

Extrapolation (generalization to other datasets)
------------------------------------------------
The dataset a hypothesis was generated on is the ORIGIN pkl (``--pkl``). Pass
one or more OTHER datasets via ``--extra-pkl`` (repeatable) to test whether each
finding GENERALIZES: the SAME code that ran on the origin (the revised code from
``--code-dir`` when present, else the recorded code) is executed again with the
load redirected to each extra pkl. All pkls are assumed to share the same
payload structure (same keys). Each result then carries an ``extrapolations``
list — one entry per extra pkl with its own fresh output — so the agent can
compare the conclusion across datasets and judge generalization.
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
reruns fail mostly because the script cannot LOCATE the dataset (hardcoded paths
/ globs / "dataset not found" gates), not for analysis reasons. Revise ONLY the
data-loading part of each script; keep the statistical ANALYSIS and what it
prints byte-for-byte identical so the rerun is a faithful reproduction.

You MAY change, near the top of each script:
- Replace the dataset search (hardcoded paths, `os.path.exists`, `glob.glob`,
  "dataset not found" gates) with a DIRECT load of the provided pkl:

      import os, pickle
      with open(os.environ["RERUN_PKL"], "rb") as f:
          payload = pickle.load(f)

  Keep the SAME variable name the rest of the script uses (e.g. `payload`,
  `data`) and the same downstream keys (`fragments_graph`, `gt_graph`,
  `gt_edge_error`, `gt_node_canonical_label`, `gt_merge_sites`, ...).

The ENVIRONMENT is owned by the driver, NOT by your script. Do NOT manage
packages or interpreters from inside the script:
- Do NOT `pip install` / `apt install` / `conda install` anything, and do NOT
  add or keep install retry loops — the runner already turns every install
  command into a logged no-op, so they only waste time. The host env is
  pre-provisioned with numpy, pandas, scipy, statsmodels, sklearn, networkx,
  matplotlib, tensorstore and the proofreader package; just `import` them.
- Do NOT touch NumPy: never `pip install numpy...`, never `del sys.modules[...]`
  to reload it, and NEVER put a numpy/site-packages/source directory on
  `sys.path` (e.g. `sys.path.insert(0, "/tmp/np2")`). Importing numpy from a
  source tree raises "you should not try to import numpy from its source
  directory" and breaks the whole script. Just `import numpy as np`.
- If an import genuinely fails, that is an ENVIRONMENT problem for the driver to
  fix (provision the package once, globally) — not something to patch per
  script. Leave it; the driver reports it as an environment failure.

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


def resolve_code(
    r: dict, code_dir: Path | None, corrected_dir: Path | None = None
) -> tuple[str, str]:
    """Return ``(code, source)`` for a record, by precedence.

    ``corrected_dir`` (the test-fixer's rewritten-analysis scripts) wins when it
    has a ``hypo_<id>.py`` (``source='corrected'``); else ``code_dir`` (the
    loading-revised scripts) is used (``source='revised'``); else the recorded
    JSON ``code`` (``source='recorded'``). This lets a corrected statistical test
    supersede the loading-only revision, which supersedes the raw recorded code.
    """
    rid = r.get("id")
    for d, src in ((corrected_dir, "corrected"), (code_dir, "revised")):
        if d is not None:
            f = d / hypo_filename(rid)
            if f.is_file():
                text = f.read_text(encoding="utf-8")
                if text.strip():
                    return text, src
    return r.get("code") or "", "recorded"


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
import builtins, glob, os, runpy, subprocess, sys

_PKL = os.environ["RERUN_PKL"]


# --- Driver owns the environment --------------------------------------------
# The host env is the single source of truth: it is pre-provisioned with the
# scientific stack the recorded scripts need (numpy, pandas, scipy, statsmodels,
# sklearn, networkx, matplotlib, tensorstore, the proofreader package). The
# recorded scripts, however, routinely try to `pip install` those at run time
# (sometimes in retry loops that hang until the timeout) or to SWAP the NumPy
# version mid-script — both mutate the interpreter environment unpredictably and
# were a real source of failures (a NumPy source tree shoved onto sys.path made
# `import numpy` fail for every corrected script in one run). So we neutralize
# package installs HERE, in the driver's runner, rather than trusting each
# script (or an agent) to get the bootstrap right: pip / apt / conda install
# commands become logged no-ops, and the script runs against the one known-good
# env. A package that is genuinely missing then surfaces as a loud import error
# (which the driver classifies as an environment failure), not as a silent
# version swap or a half-hour hang.
def _is_install_cmd(a):
    if isinstance(a, str):
        s = a.split()
    elif isinstance(a, (list, tuple)):
        s = [str(x) for x in a]
    else:
        return False
    if not s:
        return False
    prog = os.path.basename(s[0])
    if prog.startswith("pip") and "install" in s:
        return True
    if "pip" in s and "install" in s:          # python -m pip install ...
        return True
    if prog in ("apt", "apt-get", "conda", "mamba", "sudo") and "install" in s:
        return True
    return False


_orig_check_call = subprocess.check_call
def _check_call(a, *args, **kw):
    if _is_install_cmd(a):
        print("[runner] suppressed package install:", a, file=sys.stderr, flush=True)
        return 0
    return _orig_check_call(a, *args, **kw)
subprocess.check_call = _check_call

_orig_check_output = subprocess.check_output
def _check_output(a, *args, **kw):
    if _is_install_cmd(a):
        print("[runner] suppressed package install:", a, file=sys.stderr, flush=True)
        return "" if (kw.get("text") or kw.get("encoding")) else b""
    return _orig_check_output(a, *args, **kw)
subprocess.check_output = _check_output

_orig_run = subprocess.run
def _run(a, *args, **kw):
    if _is_install_cmd(a):
        print("[runner] suppressed package install:", a, file=sys.stderr, flush=True)
        return subprocess.CompletedProcess(a, 0, "" if kw.get("text") else b"",
                                           "" if kw.get("text") else b"")
    return _orig_run(a, *args, **kw)
subprocess.run = _run

_orig_popen = subprocess.Popen
class _Popen(_orig_popen):
    def __init__(self, a, *args, **kw):
        if _is_install_cmd(a):
            print("[runner] suppressed package install:", a, file=sys.stderr, flush=True)
            a = [sys.executable, "-c", "pass"]
        super().__init__(a, *args, **kw)
subprocess.Popen = _Popen


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


# A failure is an ENVIRONMENT failure (the interpreter could not import the
# scientific stack) rather than an ANALYSIS failure (the script ran but the
# statistics/data logic raised, or a "dataset not found" gate exited) when its
# stderr carries an import/native-load signature. We classify these distinctly so
# a broken env can NEVER again masquerade as a batch of INCONCLUSIVE statistical
# verdicts: the driver aborts on a systematic env failure instead of letting an
# agent fold per-hypothesis "could not run" calls. The patterns are deliberately
# narrow — a "dataset not found" gate or a genuine analysis exception is NOT an
# environment failure (the reproducer is meant to fix loading and rerun those).
_ENV_FAILURE_RE = re.compile(
    r"ModuleNotFoundError"
    r"|ImportError"
    r"|cannot import name"
    r"|you should not try to import numpy from its source directory"
    r"|numpy\._core"
    r"|_multiarray_umath"
    r"|undefined symbol"
    r"|DLL load failed"
    r"|cannot open shared object file"
    r"|\.so: cannot open",
    re.IGNORECASE,
)


def classify_failure(stderr: str | None) -> str | None:
    """Tag a non-zero run as ``"environment"`` vs ``"analysis"`` (None if clean).

    Used both per-result (so the agent sees why a script failed) and to decide
    whether a whole compute step is a systematic environment failure worth
    aborting on, rather than folding as statistical verdicts.
    """
    if not stderr:
        return "analysis"
    return "environment" if _ENV_FAILURE_RE.search(stderr) else "analysis"


def preflight_env(python: str, *, timeout: int = 60) -> tuple[bool, str]:
    """Cheap one-time check that the run interpreter can import the core stack.

    Catches a broken base environment ONCE, up front, with a clear message —
    instead of discovering it as N identical per-script import failures after
    spending time launching each. Returns ``(ok, detail)``; ``detail`` is the
    captured error when not ok. The recorded scripts also use statsmodels for the
    corrected tests, so it is included.
    """
    probe = (
        "import importlib, sys\n"
        "mods = ['numpy', 'pandas', 'scipy', 'statsmodels']\n"
        "bad = []\n"
        "for m in mods:\n"
        "    try:\n"
        "        importlib.import_module(m)\n"
        "    except Exception as e:\n"
        "        bad.append('%s: %s: %s' % (m, type(e).__name__, e))\n"
        "if bad:\n"
        "    sys.stderr.write('\\n'.join(bad))\n"
        "    sys.exit(1)\n"
        "import numpy\n"
        "print(numpy.__version__)\n"
    )
    try:
        proc = subprocess.run(
            [python, "-c", probe],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (subprocess.TimeoutExpired, OSError) as e:
        return False, f"preflight could not launch the interpreter: {e}"
    if proc.returncode != 0:
        return False, (proc.stderr or proc.stdout or "unknown import error").strip()
    return True, (proc.stdout or "").strip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "json_file", type=Path, help="The run JSON export to re-execute."
    )
    parser.add_argument(
        "--pkl",
        type=Path,
        required=True,
        help="The ORIGIN dataset .pkl this run's experiments load.",
    )
    parser.add_argument(
        "--extra-pkl",
        type=Path,
        action="extend",
        nargs="+",
        default=[],
        metavar="PKL",
        help=(
            "Other dataset .pkl(s) to EXTRAPOLATE onto. Accepts several paths "
            "after one flag and/or the flag repeated. The same code is re-run "
            "with its load redirected to each, to test generalization. Assumes "
            "the same payload structure as --pkl."
        ),
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
        "--corrected-dir",
        type=Path,
        default=None,
        help=(
            "Dir of hypo_<id>.py with a CORRECTED statistical test (the "
            "test-fixer's output). Takes precedence over --code-dir per record, "
            "so a fixed test supersedes the loading-only revision. Records with "
            "no corrected file fall back to --code-dir then the recorded code."
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
    if args.export_dir is not None and (
        args.code_dir is not None or args.corrected_dir is not None
    ):
        parser.error("--export-dir cannot combine with --code-dir/--corrected-dir.")
    if args.export_dir is not None and args.extra_pkl:
        parser.error("--extra-pkl has no effect with --export-dir (export only).")
    pkl = args.pkl.resolve()
    extra_pkls = []
    for ep in args.extra_pkl:
        if not ep.is_file():
            parser.error(f"No such extra dataset pkl: {ep}")
        extra_pkls.append(ep.resolve())

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

    # Preflight (#4 guardrail): verify ONCE, up front, that the run interpreter
    # can import the core scientific stack. A globally broken env then fails here
    # with one clear message instead of as N identical per-script import errors,
    # and the non-zero exit stops the workflow before any agent folds the result.
    ok, detail = preflight_env(sys.executable)
    if not ok:
        print(
            f"[rerun] PREFLIGHT FAILED: the interpreter ({sys.executable}) cannot "
            f"import the core scientific stack — aborting before running any "
            f"script.\n{detail}",
            file=sys.stderr,
            flush=True,
        )
        print(
            json.dumps(
                {
                    "json_file": str(args.json_file),
                    "pkl": str(pkl),
                    "preflight_ok": False,
                    "environment_failure": True,
                    "preflight_error": detail,
                    "results": [],
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 3
    print(f"[rerun] preflight OK (numpy {detail})", file=sys.stderr, flush=True)

    results = []
    n_ok = n_failed = n_timeout = n_env_failed = 0
    n_revised = n_corrected = 0
    for rank, r in enumerate(ranked, start=1):
        code, source = resolve_code(r, args.code_dir, args.corrected_dir)
        if source == "revised":
            n_revised += 1
        elif source == "corrected":
            n_corrected += 1
        rid = r.get("id")
        print(
            f"[rerun {rank}/{len(ranked)}] id={rid} ({source}) …",
            file=sys.stderr,
            flush=True,
        )
        failure_kind = None  # None | "environment" | "analysis"
        if not code.strip():
            run = {
                "timed_out": False,
                "exitcode": None,
                "stdout": "",
                "stderr": "[rerun] record has no 'code' to execute",
                "runtime_ms": 0,
            }
            outcome = "NO-CODE"
            failure_kind = "analysis"
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
                failure_kind = classify_failure(run["stderr"])
                if failure_kind == "environment":
                    n_env_failed += 1
                    outcome = f"ENV-FAILED(exit={run['exitcode']})"
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

        # Extrapolation: run the SAME code on each other dataset to test whether
        # the finding generalizes. Skip when the code didn't even run on origin
        # (no point — there's no analysis to carry over).
        extrapolations = []
        if extra_pkls and code.strip():
            for ei, ep in enumerate(extra_pkls, start=1):
                xrun = rerun_one(code, ep, args.timeout)
                xoutcome = (
                    "TIMEOUT"
                    if xrun["timed_out"]
                    else ("OK" if xrun["exitcode"] == 0 else f"FAILED(exit={xrun['exitcode']})")
                )
                print(
                    f"[rerun {rank}/{len(ranked)}] id={rid} extrapolate "
                    f"{ei}/{len(extra_pkls)} {ep.name} {xoutcome} "
                    f"in {xrun['runtime_ms'] / 1000:.0f}s",
                    file=sys.stderr,
                    flush=True,
                )
                extrapolations.append(
                    {
                        "pkl": str(ep),
                        "pkl_name": ep.name,
                        "exitcode": xrun["exitcode"],
                        "timed_out": xrun["timed_out"],
                        "runtime_ms": xrun["runtime_ms"],
                        "stdout": truncate(xrun["stdout"], args.max_output_chars),
                        "stderr": truncate(xrun["stderr"], args.max_output_chars),
                    }
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
                "rerun_failure_kind": failure_kind,
                "rerun_runtime_ms": run["runtime_ms"],
                "rerun_stdout": truncate(run["stdout"], args.max_output_chars),
                "rerun_stderr": truncate(run["stderr"], args.max_output_chars),
                "extrapolations": extrapolations,
            }
        )

    payload = {
        "json_file": str(args.json_file),
        "pkl": str(pkl),
        "extra_pkls": [str(p) for p in extra_pkls],
        "rank_by": args.rank_by,
        "top": args.top,
        "timeout_s": args.timeout,
        "code_dir": str(args.code_dir) if args.code_dir is not None else None,
        "corrected_dir": str(args.corrected_dir)
        if args.corrected_dir is not None
        else None,
        "n_rerun": len(results),
        "n_revised": n_revised,
        "n_corrected": n_corrected,
        "n_recorded": len(results) - n_revised - n_corrected,
        "n_ok": n_ok,
        "n_failed": n_failed,
        "n_env_failed": n_env_failed,
        "n_timeout": n_timeout,
        "preflight_ok": True,
        # A systematic environment failure: nothing ran AND every failure is an
        # import/native-load error. This is the signature of a broken env (the
        # NumPy-source-tree regression), NOT a batch of legitimately-inconclusive
        # statistics — flag it so the downstream fold step / orchestrator does not
        # record N "could not run" verdicts as if they were real findings.
        "environment_failure": (
            n_ok == 0 and n_env_failed > 0 and n_env_failed == n_failed
        ),
        "results": results,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    # Exit 3 (distinct from argparse's 2) on a systematic env failure so the
    # driver's run_compute() aborts loudly BEFORE an agent folds a bad result,
    # rather than treating it as a normal completion.
    if payload["environment_failure"]:
        print(
            f"[rerun] ENVIRONMENT FAILURE: {n_env_failed}/{len(results)} scripts "
            f"failed to import the scientific stack and none succeeded. This is an "
            f"environment problem, not a statistical result — aborting so it is not "
            f"folded as 'could not run' verdicts. Inspect a rerun_stderr above.",
            file=sys.stderr,
            flush=True,
        )
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
