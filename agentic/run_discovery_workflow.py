"""
Run the AutoDiscovery summarization workflow via the Claude Agent SDK.

Processes ONE AutoDiscovery run export (a single JSON file) together with the
dataset ``.pkl`` its experiments were run against, and produces a Markdown
deliverable next to the input: ``<stem>.summary.md``. Subagents live in
``.claude/agents/`` and are auto-discovered via ``setting_sources``.

Architecture — driver-owned compute, agents operate on artifacts
---------------------------------------------------------------
The workflow is a sequence of typed STEPS of two kinds, so a long re-execution
NEVER runs inside an agent turn (the failure that previously wedged the run —
an hour-long ``rerun_experiments.py`` killed at turn-end / lost to ``nohup &``):

  * compute step — the DRIVER runs ``rerun_experiments.py`` as a blocking
    foreground subprocess and captures its stdout JSON beside the export. In
    predictive mode, generated artifact names include a ``.predictive`` scope
    so they cannot be confused with positive/both runs. Deterministic,
    observable, resumable; a non-zero exit aborts before an agent folds bad data.
  * agent step — an instruction to the persistent SDK session; the agent reads
    source/result artifacts already on disk and selects findings, writes or
    updates reports, edits loading, or authors corrected scripts. It never owns
    a long experiment run.

The PHASES, in order:
  1. summarize   — [agent] optionally select predictive candidates, rank the
                   retained hypotheses, and write the report.
  2. reproduce   — [compute] run recorded code on ``--pkl`` → [compute] export
                   editable scripts → [agent] fix data-loading failures →
                   [compute] hash-detect and re-measure only changed scripts,
                   merging them with the first pass → [agent] fold verdicts.
  3. extrapolate — (only with ``--extra-pkl``) [compute] run the reproduced code
                   on each OTHER dataset → [agent] fold GENERALIZES/PARTIAL/
                   DOES-NOT-GENERALIZE/INCONCLUSIVE verdicts.
  4. verify      — [agent] audit statistics/logic from recorded + reproduced
                   results; assign SOUND/WEAK/MINOR/MAJOR/CRITICAL verdicts.
  5. fix-tests   — [agent] author corrected scripts for flagged tests →
                   [compute] re-measure them (+extras) → [agent] fold
                   UPHELD/WEAKENED/OVERTURNED. (Compute+fold skipped if nothing
                   was flagged.)
  6. feature-applicability — for split-error runs only, [agent] classifies the
                   minimum node-role requirement of each exact rerun/fixed
                   source → [driver] binds those judgments to source hashes.

Every step folds into ONE report whose top-level layout is fixed by
``REPORT_SECTION_ORDER`` (Header → Ranked Conclusions → Reproduction →
Generalization → Statistical Verification → Statistical Test Corrections →
Excluded). ``## Excluded`` is the trailing appendix, so each step inserts before
it. ``_scale_timeouts`` sizes agent turns from the hypothesis count, because
predictive mode removed the top-K that the old flat budget assumed. Compute
steps have a fixed 24-hour budget for large experiment reruns.

A full (non-smoke) run also tees its entire console output to
``<RUN>.json[.predictive].workflow.log.txt`` beside the export (override with
``--log-txt``, ``/dev/null`` to skip): the dataset guards, step progress, every
subagent tool call, each step's final reply, the compute subprocesses' live
``[rerun k/N]`` output, validation WARNINGs, and any fatal diagnostic. It is
flushed as it goes and closed with an ``# OK/FAILED after Ns`` footer, so a log
with no footer means the process was killed rather than finished.

Before that log can be overwritten or any workflow step begins, the driver
surveys reports, result JSONs, checkpoints, rerun/fixed scripts, selection state,
and the previous log. If prior or likely stale artifacts exist, an interactive
run requires explicit y/n confirmation. Batch runs abort unless the reviewed
invocation includes ``--yes-stale``.

``RANK_BY`` controls ordering. Positive/both modes keep a percentage-based
top-K. Predictive mode removes only explicit exclusions and keeps every
remaining candidate; ranking changes display order but never inclusion. Its
selection is cached as ``autodiscovery/<RUN>.predictive-selection.json`` and
reused while both the run JSON hash and selection-policy version still match.

Usage (from the ``exa-spim-agent/`` project root):
    python agentic/run_discovery_workflow.py autodiscovery/RUN.json --pkl ../data/RUN.pkl --direction both
    python agentic/run_discovery_workflow.py autodiscovery/RUN.json --pkl DATA.pkl --direction positive
    # Keep every feature not explicitly excluded, regardless of direction:
    python agentic/run_discovery_workflow.py autodiscovery/RUN.json --pkl DATA.pkl \
        --direction predictive
    # Preview selected IDs; predictive mode creates/reuses its cached manifest:
    python agentic/run_discovery_workflow.py autodiscovery/RUN.json \
        --direction predictive --smoke
    # Also test generalization onto other datasets:
    python agentic/run_discovery_workflow.py autodiscovery/RUN.json --pkl ORIGIN.pkl \
        --extra-pkl OTHER1.pkl --extra-pkl OTHER2.pkl --direction predictive
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import json
import math
import os
import platform
import re
import shlex
import subprocess
import sys
import threading
import time
import traceback
import uuid
from collections import Counter
from datetime import datetime
from pathlib import Path

try:
    from agentic.split_feature_applicability import (
        NODE_ROLE_REQUIREMENTS,
        artifact_path as split_applicability_path,
        compile_applicability,
        draft_path as split_applicability_draft_path,
        load_applicability,
    )
except ModuleNotFoundError:  # direct: python agentic/run_discovery_workflow.py
    from split_feature_applicability import (  # type: ignore[no-redef]
        NODE_ROLE_REQUIREMENTS,
        artifact_path as split_applicability_path,
        compile_applicability,
        draft_path as split_applicability_draft_path,
        load_applicability,
    )

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ResultMessage,
    TextBlock,
)

# ToolUseBlock is what gives us live progress (which tool/subagent is running).
# Import defensively so a minor SDK version mismatch doesn't break the script.
try:
    from claude_agent_sdk import ToolUseBlock
except ImportError:  # pragma: no cover - depends on installed SDK version
    ToolUseBlock = ()  # type: ignore[assignment]


def log(msg: str) -> None:
    """Timestamped progress line to stderr (kept separate from step output)."""
    print(f"[{datetime.now():%H:%M:%S}] {msg}", file=sys.stderr, flush=True)


class _Tee:
    """Fan writes out to several streams at once (console + log file).

    Stands in for ``sys.stdout`` / ``sys.stderr`` so the whole workflow is
    captured to a txt file while still streaming live to the terminal. Flushes on
    every write: this run takes hours, so a buffered log that only lands at exit
    would be useless while it is going — and empty if the process is killed.
    """

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            s.write(data)
            s.flush()
        return len(data)

    def flush(self):
        for s in self._streams:
            s.flush()

    def isatty(self):
        # Never claim a TTY: a tee is not one, and libraries that probe this to
        # decide on progress bars / ANSI codes would otherwise pollute the log.
        return False


@contextlib.contextmanager
def driver_log(log_txt: Path):
    """Capture this driver's whole console output to ``log_txt`` for the block.

    Mirrors ``run_detector_build_workflow.py``: the workflow runs for hours over
    several agent turns and several long compute subprocesses, and until now none
    of that output was persisted anywhere — a run that died mid-step left no
    record of which step it was on, which validation WARNINGs had fired, or what
    the compute child printed before it failed. Everything written through
    ``sys.stdout`` / ``sys.stderr`` inside the block is tee'd here as it goes.

    The header ties the log back to its invocation; the footer is written last on
    purpose, so a log with no footer means the process was KILLED (an OOM, a
    Slurm timeout) rather than having finished.
    """
    log_txt.parent.mkdir(parents=True, exist_ok=True)
    orig_out, orig_err = sys.stdout, sys.stderr
    started = time.monotonic()
    with log_txt.open("w", encoding="utf-8") as log_fh:
        log_fh.write(f"# {' '.join(sys.argv)}\n")
        log_fh.write(f"# started {datetime.now():%Y-%m-%d %H:%M:%S}\n")
        log_fh.write(f"# host {platform.node()}  python {platform.python_version()}\n\n")
        log_fh.flush()
        sys.stdout = _Tee(orig_out, log_fh)
        sys.stderr = _Tee(orig_err, log_fh)
        outcome = "FAILED"
        try:
            yield log_fh
            outcome = "OK"
        except SystemExit as exc:
            # Step failures and argparse guards raise SystemExit carrying the
            # diagnostic as its message. Python prints that to the real stderr on
            # the way out — but only after the finally below has restored it, so
            # the log would end with no hint of WHY. Write it straight to the
            # file rather than through the tee, which would duplicate it on the
            # console.
            if exc.code not in (0, None):
                log_fh.write(f"FATAL: {exc.code}\n")
                log_fh.flush()
            raise
        except BaseException:
            traceback.print_exc()
            raise
        finally:
            sys.stdout, sys.stderr = orig_out, orig_err
            log_fh.write(
                f"\n# {outcome} after {time.monotonic() - started:.0f}s "
                f"(ended {datetime.now():%Y-%m-%d %H:%M:%S})\n"
            )
            log_fh.flush()


def describe_tool(block) -> str:
    """One-line, human-readable summary of a tool-use block for progress logs."""
    name = getattr(block, "name", "tool")
    args = getattr(block, "input", {}) or {}
    if name == "Task":
        sub = args.get("subagent_type") or args.get("description") or "?"
        return f"Task → subagent '{sub}'"
    if name == "Bash":
        cmd = " ".join(str(args.get("command", "")).split())
        return f"Bash: {cmd[:100]}" + ("…" if len(cmd) > 100 else "")
    if name in ("Write", "Edit", "Read", "Glob"):
        target = args.get("file_path") or args.get("path") or args.get("pattern") or ""
        return f"{name}: {target}"
    return name

# Project root = the directory that holds .claude/, agentic/, autodiscovery/.
# agentic/run_discovery_workflow.py -> parent.parent is the project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- Ranking configuration ---------------------------------------------------
# TOP_K_MAX: hard ceiling on hypotheses kept in the report.
# TOP_K_PCT: fraction of the total hypothesis count to keep.
# For positive/both, effective TOP_K = min(TOP_K_MAX, floor(N × TOP_K_PCT));
# computed at startup from the actual record count in the run JSON and stored in
# the module global TOP_K. Example: 50 hypotheses → 10. Predictive mode keeps
# every hypothesis selected by its semantic inclusion pass (TOP_K=None).
# RANK_BY: which deterministic ordering rank_by_surprise.py uses.
#   "posterior-surprise" ranks by posterior * |surprisal| so hypotheses that
#   are BOTH strongly believed true AND highly belief-shifting come first;
#   "surprise" ranks by |surprisal| alone.
# DIRECTION: direction filter applied before ranking.
#   "both"     — keep positive AND negative surprisal (any strong belief shift).
#   "positive" — keep only hypotheses where the experiment RAISED belief
#                (posterior > prior by more than 0.02); discards surprising-but-
#                now-disbelieved findings.
#   "predictive" — start with every hypothesis, remove only explicit exclusions
#                  (invalid, constant, non-predictive, or wholly confounded), and
#                  keep every remaining candidate without a score/top-K cutoff.
# Set from --direction (required CLI argument) at startup; see main().
TOP_K_MAX: int = 20
TOP_K_PCT: float = 0.20
RANK_BY: str = "posterior-surprise"

TOP_K: int | None = TOP_K_MAX  # overridden at startup; see main()
DIRECTION: str = "both"       # overridden at startup from --direction; see main()
PREDICTIVE_MANIFEST: str | None = None
# Bump this whenever predictive_selection_instruction changes semantically.
PREDICTIVE_POLICY_VERSION: str = "exclusion-only-v1"
PREDICTIVE_MANIFEST_SUFFIX: str = ".predictive-selection.json"
# Default name for this driver's captured console log (see driver_log_path).
DRIVER_LOG_SUFFIX: str = ".workflow.log.txt"


def _is_split_run(path: Path | str) -> bool:
    stem = Path(path).name.lower()
    return stem == "split-error.json" or stem.startswith(
        ("split-error-", "split-error_")
    )

# --- Timeout budgets ---------------------------------------------------------
# Agent budgets scale with how many hypotheses the report carries, because the
# summarize step authors one full entry per record in a single turn. Compute
# steps use a fixed 24-hour budget: rerun_experiments.py executes one script per
# record (each with its own 3600-second cap), and large predictive runs can take
# substantially longer than the old record-count estimate allowed.
AGENT_STEP_BASE_S: int = 600        # 10 min of fixed overhead per agent turn
AGENT_STEP_PER_RECORD_S: int = 30   # + writing/folding one entry
AGENT_STEP_CEILING_S: int = 3600    # 1 h — a hung session must still abort

AGENT_STEP_TIMEOUT_S: int = 900     # overridden at startup; see _scale_timeouts()
COMPUTE_TIMEOUT_S: int = 86400      # 24 h per compute step

# The reproduce phase's fix-loading → remeasure pair runs up to this many
# rounds. One revision often uncovers the next loading failure underneath (fix
# the pkl load, then hit a credential gate), and with a single shot every
# still-UNUSABLE hypothesis was permanently folded as FAILED. Rounds after the
# first skip themselves as soon as nothing is UNUSABLE, and the loop ends the
# moment a fix round changes no script (code-hash comparison, no dataset load),
# so a converged or stalled loop costs no extra compute or agent turns.
MAX_FIX_LOADING_ROUNDS: int = 3


def _scale_timeouts(n_records: int) -> tuple[int, int]:
    """Size the agent budget from the hypothesis count and report both budgets.

    ``n_records`` is an upper bound on what the report will carry: the effective
    top-K for positive/both, or the export's full record count for predictive
    (whose selection manifest may not exist yet on a first run).
    """
    global AGENT_STEP_TIMEOUT_S
    n = max(0, n_records)
    AGENT_STEP_TIMEOUT_S = min(
        AGENT_STEP_CEILING_S, AGENT_STEP_BASE_S + AGENT_STEP_PER_RECORD_S * n
    )
    return AGENT_STEP_TIMEOUT_S, COMPUTE_TIMEOUT_S


def _top_flags() -> str:
    """The shared ``--rank-by``/``--top``/``--direction`` flags both helpers must agree on."""
    flags = f"--rank-by {RANK_BY} --direction {DIRECTION}"
    if DIRECTION == "predictive":
        if PREDICTIVE_MANIFEST is None:
            raise RuntimeError("predictive direction requires a selection manifest")
        if TOP_K is not None:
            raise RuntimeError("predictive direction must not apply a top-K cutoff")
        flags += f" --predictive-manifest {PREDICTIVE_MANIFEST}"
    if TOP_K is not None:
        flags += f" --top {TOP_K}"
    return flags


def _top_k_phrase() -> str:
    """Human-readable effective top-K for agent instructions."""
    return f"the top {TOP_K} hypotheses" if TOP_K is not None else "all ranked hypotheses"


def _compute_top_k(n_hypotheses: int) -> int:
    """Effective top-K: min(TOP_K_MAX, floor(N × TOP_K_PCT)), at least 1.

    Falls back to TOP_K_MAX when ``n_hypotheses`` is 0 (e.g. JSON read error).
    """
    if n_hypotheses <= 0:
        return TOP_K_MAX
    return max(1, min(TOP_K_MAX, int(n_hypotheses * TOP_K_PCT)))


def predictive_manifest_path(json_path: Path) -> Path:
    """Persistent predictive-selection cache beside the run export."""
    return json_path.with_name(f"{json_path.stem}{PREDICTIVE_MANIFEST_SUFFIX}")


def driver_log_path(json_path: Path) -> Path:
    """Default console-log path beside the run export.

    Scoped by ``--direction`` exactly like the compute artifacts
    (``.predictive.rerun``, ``.predictive.reproduce.json``, …) so a predictive run
    never overwrites the log of a positive/both run on the same export.
    """
    scope = ".predictive" if DIRECTION == "predictive" else ""
    return json_path.with_name(f"{json_path.name}{scope}{DRIVER_LOG_SUFFIX}")


def predictive_selection_instruction(json_rel: str, manifest_rel: str) -> str:
    """Instruction shared by full and smoke predictive selection."""
    return (
        "Use the discovery-predictive-selector subagent. "
        f"Read every hypothesis in the single AutoDiscovery export at {json_rel} "
        "using an EXCLUSION-ONLY policy for downstream detector candidates. "
        "START with every hypothesis and ignore whether its belief-shift "
        "direction is Positive or Negative. Do NOT impose any AUC, PR-AUC, "
        "p-value, effect-size, priority-score, rank, or top-K cutoff. Do NOT "
        "exclude a feature merely because its standalone evidence is weak, "
        "low-ranked, inverse, specialist, or uncertain; keep it for downstream "
        "cross-validation. EXCLUDE only features that the available result "
        "clearly establishes are constant/unavailable, invalidly computed, "
        "approximately random with no useful subgroup enrichment, or entirely "
        "explained by a known confounder. A statistically significant but "
        "explicitly non-discriminative effect may also be excluded. Write a "
        f"JSON manifest to {manifest_rel} with this top-level selection schema: "
        "{\"source_file\": <path>, \"criterion\": \"predictive\", "
        "\"selected_ids\": [<original hypothesis ids>], "
        "\"excluded\": [{\"id\": <id>, \"reason\": <short reason>}]} . "
        "Every original hypothesis ID must appear exactly once, either in "
        "selected_ids or excluded. Preserve each ID's original JSON type, make "
        "selected_ids unique, and write valid JSON only. The driver will add "
        "source-hash and policy-version cache metadata after validation. Report "
        "the selected IDs and output path."
    )


def rank_cmd(json_rel: str, *, include_code: bool = False) -> str:
    """Deterministic ranking command for ONE run export.

    Built from the free parameters above so the helper, the report, and the
    verifier stay in sync. ``json_rel`` is the run file path relative to the
    project root, so the helper ranks only that single file. With
    ``include_code`` the helper also emits each returned record's truncated
    ``code`` / ``codeOutput`` so the verifier can audit only the selected report
    set instead of loading the full multi-MB export.
    """
    cmd = f"python agentic/rank_by_surprise.py {json_rel} {_top_flags()}"
    return cmd + " --include-code" if include_code else cmd


def rerun_argv(
    json_rel: str,
    pkl_rel: str,
    *,
    code_dir_rel: str | None = None,
    corrected_dir_rel: str | None = None,
    extra_pkls_rel: list[str] | None = None,
    export_dir_rel: str | None = None,
    base_results_rel: str | None = None,
    origin_results_rel: str | None = None,
    only_changed: bool = False,
    only_corrected: bool = False,
    extra_only: bool = False,
    checkpoint_rel: str | None = None,
) -> list[str]:
    """Build the ``rerun_experiments.py`` argv for ONE run export + dataset pkl.

    The DRIVER (not an agent) runs this as a blocking foreground subprocess, so
    no long re-execution ever lives inside an agent turn. All compute phases —
    reproduce, extrapolate, and the corrected-test re-measurement — go through
    this one builder so they always agree on ``--rank-by``/``--direction``/``--top``:

    - ``export_dir_rel``  : export editable hypo_<id>.py scripts and exit (no run).
    - ``code_dir_rel``    : run the loading-fixed scripts instead of recorded code.
    - ``corrected_dir_rel``: run the test-fixer's corrected scripts (takes
      precedence per record over ``--code-dir``).
    - ``extra_pkls_rel``  : also run each executed record on these OTHER datasets (the
      ``extrapolations`` list), to judge generalization.
    - ``base_results_rel`` / ``only_changed``: merge unchanged first-pass
      results and execute only scripts whose resolved code hash changed.
    - ``origin_results_rel`` / ``extra_only``: reuse origin measurements and
      execute only code×extra-dataset work items.
    - ``only_corrected``  : restrict execution to scripts present in the
      corrected directory.
    - ``checkpoint_rel``  : atomically persist and resume fingerprint-matching
      work items.
    """
    argv = ["python", "agentic/rerun_experiments.py", json_rel,
            "--pkl", pkl_rel, "--rank-by", RANK_BY, "--direction", DIRECTION]
    if DIRECTION == "predictive":
        if PREDICTIVE_MANIFEST is None:
            raise RuntimeError("predictive direction requires a selection manifest")
        if TOP_K is not None:
            raise RuntimeError("predictive direction must not apply a top-K cutoff")
        argv += ["--predictive-manifest", PREDICTIVE_MANIFEST]
    if TOP_K is not None:
        argv += ["--top", str(TOP_K)]
    # Export mode is mutually exclusive with running (the helper enforces this).
    if export_dir_rel is not None:
        return argv + ["--export-dir", export_dir_rel]
    if code_dir_rel is not None:
        argv += ["--code-dir", code_dir_rel]
    if corrected_dir_rel is not None:
        argv += ["--corrected-dir", corrected_dir_rel]
    if base_results_rel is not None:
        argv += ["--base-results", base_results_rel]
    if origin_results_rel is not None:
        argv += ["--origin-results", origin_results_rel]
    if only_changed:
        argv.append("--only-changed")
    if only_corrected:
        argv.append("--only-corrected")
    if extra_only:
        argv.append("--extra-only")
    if checkpoint_rel is not None:
        argv += ["--checkpoint", checkpoint_rel]
    for p in extra_pkls_rel or []:
        argv += ["--extra-pkl", p]
    return argv


def run_compute(argv: list[str], out_rel: str | None) -> None:
    """Run a deterministic helper subprocess to completion in the FOREGROUND.

    This is the heart of the driver-owned design: the long ``rerun_experiments.py``
    invocations run here, as a blocking child of the driver, NOT inside an agent
    turn — so nothing is ever killed at turn-end or lost to a ``nohup &`` detach.
    The child's stdout (its JSON payload) is captured to ``out_rel`` when given;
    its stderr (the live ``[rerun k/N]`` progress) is relayed line by line so a
    long run stays observable. Output is written to a sibling temporary file and
    atomically replaces the final JSON only after a successful exit; failures
    leave any prior valid result untouched and abort before an agent can fold it.

    The relay is what puts the child's progress in the driver log: inheriting this
    process's file descriptors would send it to the real terminal only, bypassing
    the ``_Tee`` installed on ``sys.stderr`` — and the compute steps are where the
    hours go, so their output is exactly what a post-mortem needs.
    """
    pretty = " ".join(shlex.quote(a) for a in argv)
    log(f"  [compute] {pretty}")
    if out_rel is not None:
        log(f"  [compute] → stdout JSON to {out_rel}")
    start = time.monotonic()
    out_path = PROJECT_ROOT / out_rel if out_rel is not None else None
    tmp_out_path = (
        out_path.with_name(f".{out_path.name}.{os.getpid()}.tmp")
        if out_path is not None
        else None
    )
    out_fh = open(tmp_out_path, "w") if tmp_out_path is not None else None

    def _pump(pipe, sink) -> None:
        """Relay a child pipe to one of our (possibly tee'd) streams, live."""
        try:
            for line in pipe:
                sink.write(line)
                sink.flush()
        except (ValueError, OSError):
            pass  # pipe closed under us on kill — nothing left to relay
        finally:
            pipe.close()

    pumps: list[threading.Thread] = []
    try:
        # cwd=PROJECT_ROOT so the relative paths in argv resolve. stderr is always
        # piped and relayed; stdout goes to the JSON file when there is one, and
        # is relayed as console output when there is not.
        proc = subprocess.Popen(
            argv,
            cwd=str(PROJECT_ROOT),
            stdout=out_fh if out_fh is not None else subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        # Daemon threads, and the timeout stays on proc.wait() rather than on the
        # reads: a child that hangs while printing nothing must still be killed.
        for pipe, sink in ((proc.stderr, sys.stderr), (proc.stdout, sys.stdout)):
            if pipe is None:
                continue
            t = threading.Thread(target=_pump, args=(pipe, sink), daemon=True)
            t.start()
            pumps.append(t)
        try:
            proc.wait(timeout=COMPUTE_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            if tmp_out_path is not None:
                tmp_out_path.unlink(missing_ok=True)
            raise SystemExit(
                f"Compute step TIMED OUT after {COMPUTE_TIMEOUT_S}s: {pretty}. "
                "Aborting — increase COMPUTE_TIMEOUT_S if the dataset is large."
            )
    finally:
        # Join before closing the output file so no relayed line is lost, but
        # bounded: a wedged pump thread must not hold the workflow hostage.
        for t in pumps:
            t.join(timeout=10)
        if out_fh is not None:
            out_fh.close()
    dur = time.monotonic() - start
    if proc.returncode != 0:
        # The child wrote only to the temporary path, so discard it and preserve
        # any prior final JSON. Per-work-item checkpoints retain completed work.
        if tmp_out_path is not None:
            tmp_out_path.unlink(missing_ok=True)
        raise SystemExit(
            f"Compute step FAILED (exit {proc.returncode}) after {dur:.0f}s: {pretty}. "
            "Aborting before any agent folds a bad result."
        )
    if tmp_out_path is not None and out_path is not None:
        os.replace(tmp_out_path, out_path)
    log(f"  [compute] done in {dur:.0f}s.")


# --- Canonical report layout -------------------------------------------------
# Every fold step appends a file-wide "— Summary" section to the SAME report, and
# each step used to describe its position only relative to its predecessor —
# nothing anchored the summaries to the ranked list. Three past runs therefore
# produced three different layouts (in one, Reproduction and Generalization landed
# ABOVE the ranked list while Verification landed below it, and the top section was
# titled "Synthesis" instead of "Header"). Every instruction now quotes this one
# order, and `## Excluded` is the trailing appendix: later steps insert BEFORE it
# rather than at the end of the file.
HEADER_SECTION = "## Header"
RANKED_SECTION = "## Ranked Conclusions (highest priority first)"
EXCLUDED_SECTION = "## Excluded"
REPORT_SECTION_ORDER: tuple[str, ...] = (
    HEADER_SECTION,
    RANKED_SECTION,
    "## Reproduction — Summary",
    "## Generalization — Summary",
    "## Statistical Verification — Summary",
    "## Statistical Test Corrections — Summary",
    EXCLUDED_SECTION,
)


def layout_rule(section: str) -> str:
    """The placement sentence every fold step appends to its instruction.

    One shared sentence rather than per-step prose, so the sections cannot drift
    apart again: a step that only knows its predecessor cannot keep the file in a
    stable order.
    """
    order = " → ".join(REPORT_SECTION_ORDER)
    return (
        f"PLACEMENT: the report uses one canonical top-level order shared by every "
        f"step of this workflow — {order}. Insert '{section}' at exactly that "
        f"position; sections for phases that did not run are simply absent. Never "
        f"reorder or re-title sections already in the file, and keep "
        f"'{EXCLUDED_SECTION} …' LAST as the appendix (insert before it, not at the "
        f"end of the file). "
    )


def excluded_rule() -> str:
    """What the summarize step must put in the trailing Excluded appendix.

    Previously unspecified: in predictive mode a past run improvised the section by
    reading the selection manifest itself, which no instruction asked for — so the
    appendix was not reproducible. The ranking helper now reports each exclusion's
    reason, so the content is pinned to data the agent is already given.
    """
    if DIRECTION == "predictive":
        return (
            f"Close the report with an '{EXCLUDED_SECTION} (predictive-selection)' "
            "appendix built from the helper's `excluded_direction` list "
            "(`n_excluded_direction` entries, each with `id` and the selection "
            "`reason`): one bullet per excluded hypothesis, quoting its reason "
            "verbatim. This documents what was dropped and why, so the kept set is "
            "auditable against the full export. "
        )
    return (
        f"If the helper reported dropped hypotheses (`n_dropped_missing_surprisal` "
        f"> 0), close the report with an '{EXCLUDED_SECTION} (no surprisal score)' "
        "appendix listing their run + id so coverage is transparent. "
    )


def split_feature_label_step(
    json_rel: str,
    rerun_dir_rel: str,
    fixed_dir_rel: str,
) -> dict[str, object]:
    """One source-semantic labeling turn shared by full and labels-only runs."""
    run_path = PROJECT_ROOT / json_rel
    draft_rel = _rel_to_root(split_applicability_draft_path(run_path))
    output_rel = _rel_to_root(split_applicability_path(run_path))
    requirements = ", ".join(sorted(NODE_ROLE_REQUIREMENTS))
    order_source = (
        f"authoritative selected_ids order in {PREDICTIVE_MANIFEST}"
        if PREDICTIVE_MANIFEST is not None
        else f"records order in {rerun_dir_rel}/MANIFEST.json"
    )
    return {
        "name": "label-split-feature-applicability",
        "phase": "feature-applicability",
        "kind": "agent",
        "expects_file": draft_rel,
        "compile_split_applicability": {
            "run": json_rel,
            "draft": draft_rel,
            "output": output_rel,
        },
        "instruction": f"""
Use the discovery-feature-applicability-labeler subagent to classify the node-role requirement
of every available selected split-feature source. This is a small semantic
annotation step; do not run experiments, load a pkl, edit a hypothesis script,
or change the report.

Use the {order_source}. For each id, read {rerun_dir_rel}/hypo_<id>.py. If a
same-id source exists under {fixed_dir_rel}/, classify it as a separate `fixed`
record immediately after the corresponding `rerun` record. Write only
{draft_rel} as valid JSON with exactly schema_version=1 and records. Every
record has exactly id, source, node_role_requirement, and reason. `source` is
`rerun` or `fixed`. node_role_requirement must be one of: {requirements}.

Classify the minimum node-role precondition of the feature quantities, not the
candidate-generation policy:
- requires_both_tips: the measurement needs both participating nodes to be
  degree-1 tips, for example a two-terminal direction comparison;
- requires_tip_anchor: the anchor must be a degree-1 tip but the partner may be
  a tip, shaft, or branch node;
- no_tip_requirement: the measurement is valid without either node being a tip;
- unclear: the source mixes incompatible requirements or the minimum safe
  requirement cannot be established without guessing.

The categories are requirements, not mutually exclusive candidate pools. A
requires_tip_anchor feature remains valid on the tip-tip subset. Give each row a
short concrete reason grounded in its source math. Do not include paths, hashes,
selected_ids, feature scores, or candidate-policy values; the driver owns those
fields and will bind this draft to exact source SHA-256 values.
""".strip(),
    }


def build_steps(
    json_rel: str,
    pkl_rel: str,
    summary_rel: str,
    extra_pkls_rel: list[str] | None = None,
) -> list[dict[str, object]]:
    """Build the ordered workflow steps for a SINGLE run export + dataset pkl.

    Two KINDS of step are interleaved so no long re-execution ever lives inside
    an agent turn (the failure mode that wedged the per-turn design):

    - ``"compute"`` — the DRIVER runs ``rerun_experiments.py`` as a blocking
      foreground subprocess and captures its stdout JSON to a file next to the
      run export. Deterministic, no LLM, fully observable, resumable.
    - ``"agent"`` — an instruction sent to the persistent SDK session; the
      agent operates on source/result artifacts already on disk: selecting or
      summarizing findings, folding verdicts, editing loading, or authoring
      corrected scripts. It never owns a long experiment run.

    Each step carries a ``"phase"`` (summarize / reproduce / extrapolate /
    verify / fix-tests). ``json_rel`` is the run JSON to digest,
    ``pkl_rel`` the origin dataset,
    ``summary_rel`` the Markdown deliverable — all relative to the project root.
    Order: summarize → reproduce (run→export→fix-loading→remeasure, repeated as
    a repair loop up to ``MAX_FIX_LOADING_ROUNDS`` rounds that end early once no
    result is UNUSABLE or a fix round changes no script, then a single fold) →
    [extrapolate (run→fold), only with ``extra_pkls_rel``] → verify →
    fix-tests (author→measure→fold).
    """
    extra_pkls_rel = extra_pkls_rel or []
    scope_suffix = ".predictive" if DIRECTION == "predictive" else ""
    rerun_dir_rel = f"{json_rel}{scope_suffix}.rerun"
    fixed_dir_rel = f"{json_rel}{scope_suffix}.fixed"
    # Compute-output JSONs live next to the run export so they survive across
    # runs and a resume can reuse the expensive ones instead of recomputing.
    repro_raw_rel = f"{json_rel}{scope_suffix}.reproduce-raw.json"   # recorded code on origin
    repro_rel = f"{json_rel}{scope_suffix}.reproduce.json"           # loading-fixed code on origin
    extrap_rel = f"{json_rel}{scope_suffix}.extrapolate.json"        # +extra datasets
    corrected_rel = f"{json_rel}{scope_suffix}.corrected.json"       # corrected tests +extras
    repro_raw_checkpoint_rel = f"{repro_raw_rel}.checkpoint.json"
    repro_checkpoint_rel = f"{repro_rel}.checkpoint.json"
    extrap_checkpoint_rel = f"{extrap_rel}.checkpoint.json"
    corrected_checkpoint_rel = f"{corrected_rel}.checkpoint.json"

    steps: list[dict[str, object]] = []
    if DIRECTION == "predictive":
        if PREDICTIVE_MANIFEST is None:
            raise RuntimeError("predictive direction requires a selection manifest")
        steps.append(
            {
                "name": "select-predictive-hypotheses",
                "phase": "summarize",
                "kind": "agent",
                "expects_file": PREDICTIVE_MANIFEST,
                "validate_predictive_manifest_for": json_rel,
                "reuse_predictive_manifest_for": json_rel,
                "instruction": predictive_selection_instruction(
                    json_rel, PREDICTIVE_MANIFEST
                ),
            }
        )

    steps.extend([
        {
            "name": "summarize-discoveries",
            "phase": "summarize",
            "kind": "agent",
            # Check the two headings this step OWNS and every later step anchors
            # to. The previous markers could not catch a layout problem:
            # "priority_score" is a code identifier a report need never print
            # (one past run contains it zero times) and "##" is true of any
            # Markdown file.
            "expects": [HEADER_SECTION, RANKED_SECTION],
            "instruction": (
                "Use the discovery-summarizer subagent to digest the single "
                f"AutoDiscovery run export at {json_rel}. It must run "
                f"`{rank_cmd(json_rel)}` to rank that file's hypotheses by the "
                "combined posterior-and-surprise priority (posterior * "
                f"|surprisal|), keeping only {_top_k_phrase()}. The report MUST "
                "contain ONLY the records the helper returns (do NOT add entries "
                "beyond what the helper returns). "
                f"Write the ranked report to {summary_rel}, ordered by the "
                "helper's ranking (highest priority first), and display each "
                "entry's priority_score alongside its surprise magnitude. "
                f"Use exactly these two top-level headings, verbatim: "
                f"'{HEADER_SECTION}' for the source/ranking/count preamble and "
                f"one-paragraph synthesis, then '{RANKED_SECTION}' for the ranked "
                "entries — later steps insert their own sections around these two "
                "and depend on the exact titles. " + excluded_rule() + "Report "
                "the path it wrote and a short executive summary of the "
                "highest-priority conclusions (high posterior and high surprise)."
            ),
        },
        # --- Reproduce phase: run recorded code, fix loading, re-measure, fold ---
        {
            "name": "reproduce-run",
            "phase": "reproduce",
            "kind": "compute",
            "argv": rerun_argv(
                json_rel, pkl_rel, checkpoint_rel=repro_raw_checkpoint_rel
            ),
            "out": repro_raw_rel,
            "failure_summary": True,
        },
        {
            "name": "reproduce-export",
            "phase": "reproduce",
            "kind": "compute",
            # Export editable hypo_<id>.py scripts to rerun_dir_rel so the
            # fix-loading agent can edit them directly — no subprocess inside
            # an agent turn. Resume preserves existing edits only while the
            # exported MANIFEST/files still match the predictive selection.
            "argv": rerun_argv(json_rel, pkl_rel, export_dir_rel=rerun_dir_rel),
            "out": None,
            "skip_if_scripts_exist_in": rerun_dir_rel,
            "skip_if_scripts_match_selection": PREDICTIVE_MANIFEST,
        },
        {
            "name": "reproduce-fix-loading",
            "phase": "reproduce",
            "kind": "agent",
            "failure_brief": {"results": repro_raw_rel, "code_dir": rerun_dir_rel},
            "syntax_check": {"code_dir": rerun_dir_rel, "results": repro_raw_rel},
            "instruction": (
                "Use the discovery-reproducer subagent to FIX ONLY data-loading / "
                "environment failures so the experiments can run — do NOT re-measure "
                "or fold verdicts yet (a later step does that). The driver has "
                f"already executed each selected record's recorded `code` on "
                f"{pkl_rel} and exported editable scripts to {rerun_dir_rel}; "
                f"read the results JSON at {repro_raw_rel} "
                "(top-level counts + a `results` list; each has `id`, `status`, "
                "`rerun_exitcode`, `rerun_stdout`, `rerun_stderr`, and the separate "
                "`result_status`/`result_failure_reason`). For every result whose "
                "result is UNUSABLE for a DATA-LOADING / ENVIRONMENT reason — even "
                "when the script incorrectly exited 0 — "
                "(a NumPy-2-written pkl the host can't unpickle, a 'dataset not "
                "found' gate, a pip-install retry loop), revise ONLY the "
                f"loading/bootstrap of the failing {rerun_dir_rel}/hypo_<id>.py "
                "(load the pkl directly from $RERUN_PKL, fix the NumPy version, drop "
                "pip-retry loops) while keeping the ANALYSIS byte-for-byte identical. "
                "Do NOT touch scripts whose result is USABLE. No marker file is "
                "needed: the driver compares code hashes and selectively reruns only "
                "the scripts whose contents actually changed. Report which ids you "
                "fixed and why."
            ),
        },
        {
            "name": "reproduce-remeasure",
            "phase": "reproduce",
            "kind": "compute",
            "argv": rerun_argv(
                json_rel,
                pkl_rel,
                code_dir_rel=rerun_dir_rel,
                base_results_rel=repro_raw_rel,
                only_changed=True,
                checkpoint_rel=repro_checkpoint_rel,
            ),
            "out": repro_rel,
            "failure_summary": True,
        },
    ])

    # Loading-repair rounds 2..MAX_FIX_LOADING_ROUNDS. One revision often only
    # uncovers the next loading failure underneath, so the fix → remeasure pair
    # repeats until nothing is UNUSABLE (the fix round skips itself), a round
    # changes no script (the remeasure round detects the stall via code hashes
    # and every later round skips), or the cap is reached. Folding stays a
    # single step AFTER the loop so verdicts are judged once, on the converged
    # results.
    for round_no in range(2, MAX_FIX_LOADING_ROUNDS + 1):
        steps.append(
            {
                "name": f"reproduce-fix-loading-{round_no}",
                "phase": "reproduce",
                "kind": "agent",
                "skip_if_stalled": True,
                "skip_if_no_unusable_in": repro_rel,
                "failure_brief": {"results": repro_rel, "code_dir": rerun_dir_rel},
                "syntax_check": {"code_dir": rerun_dir_rel, "results": repro_rel},
                "instruction": (
                    "Use the discovery-reproducer subagent for loading-repair "
                    f"round {round_no}/{MAX_FIX_LOADING_ROUNDS}: some hypotheses "
                    "are STILL UNUSABLE after the previous loading fix and "
                    "re-measurement. Read the merged results JSON at "
                    f"{repro_rel} (each result has `code_source` "
                    "recorded|revised, `result_status`, `result_failure_reason`, "
                    "`rerun_exitcode`, `rerun_stdout`, `rerun_stderr`). For "
                    "every result still UNUSABLE for a DATA-LOADING / "
                    "ENVIRONMENT reason — even when the script exited 0 — "
                    "revise ONLY the loading/bootstrap of the failing "
                    f"{rerun_dir_rel}/hypo_<id>.py (load the pkl directly from "
                    "$RERUN_PKL, fix the NumPy version, drop pip-retry loops). "
                    "A failing `code_source: revised` result means the PREVIOUS "
                    "revision is itself suspect — re-read that script first "
                    "instead of assuming the recorded code is at fault. Keep "
                    "the ANALYSIS byte-for-byte identical; never weaken or stub "
                    "an analysis to make it run, and do NOT touch scripts whose "
                    "result is USABLE. If none of the remaining failures are "
                    "loading/environment problems, change NOTHING and say so — "
                    "the driver compares code hashes and ends the repair loop "
                    "when a round changes no script. Report which ids you "
                    "revised and why."
                ),
            }
        )
        steps.append(
            {
                "name": f"reproduce-remeasure-{round_no}",
                "phase": "reproduce",
                "kind": "compute",
                "skip_if_stalled": True,
                "stall_if_code_unchanged": {
                    "code_dir": rerun_dir_rel,
                    "results": repro_rel,
                },
                "failure_summary": True,
                "argv": rerun_argv(
                    json_rel,
                    pkl_rel,
                    code_dir_rel=rerun_dir_rel,
                    base_results_rel=repro_rel,
                    only_changed=True,
                    checkpoint_rel=repro_checkpoint_rel,
                ),
                "out": repro_rel,
            }
        )

    steps.append(
        {
            "name": "reproduce-fold",
            "phase": "reproduce",
            "kind": "agent",
            "expects": ["Reproduction — Summary"],
            "instruction": (
                "Use the discovery-reproducer subagent to fold the reproduction "
                "verdicts into the report. The driver has produced the final "
                f"reproduction results JSON at {repro_rel} (each result marks its "
                "`code_source` as recorded|revised, carries `result_status` / "
                "`result_failure_reason`, and has `rerun_stdout` plus the recorded "
                "`recorded_output`). Do NOT re-run anything. For each "
                "record compare the fresh key numbers (test statistic, p-value, "
                "effect size, n) to the recorded ones and assign a REPRODUCED / "
                f"DIVERGED / FAILED verdict. Read {summary_rel} and fold the result "
                "INTO each hypothesis's existing ranked entry (append Reproduction / "
                "Rerun result bullets in place, keeping prior bullets; note whether "
                "the code was recorded or revised-loading), plus one "
                "'## Reproduction — Summary' section. "
                + layout_rule("## Reproduction — Summary")
                + "Report the path and which findings did NOT reproduce."
            ),
        }
    )

    # Optional extrapolation phase: only when other datasets were provided. The
    # driver runs the pre-exported (and loading-fixed where needed) scripts via
    # --code-dir on each extra pkl; the agent only folds the generalization verdicts.
    if extra_pkls_rel:
        extra_list = ", ".join(extra_pkls_rel)
        steps.append(
            {
                "name": "extrapolate-run",
                "phase": "extrapolate",
                "kind": "compute",
                "argv": rerun_argv(
                    json_rel, pkl_rel,
                    code_dir_rel=rerun_dir_rel, extra_pkls_rel=extra_pkls_rel,
                    origin_results_rel=repro_rel,
                    extra_only=True,
                    checkpoint_rel=extrap_checkpoint_rel,
                ),
                "out": extrap_rel,
                # reproduce-export always runs before this step, so rerun_dir_rel
                # is normally populated. It can still be empty if that export
                # produced nothing (no selected records, or a prior export that
                # failed after creating the dir) — passing --code-dir to an empty
                # dir has implicit behaviour, so fall back to recorded code.
                "check_scripts_in": rerun_dir_rel,
                "argv_if_no_scripts": rerun_argv(
                    json_rel,
                    pkl_rel,
                    extra_pkls_rel=extra_pkls_rel,
                    origin_results_rel=repro_rel,
                    extra_only=True,
                    checkpoint_rel=extrap_checkpoint_rel,
                ),
            }
        )
        steps.append(
            {
                "name": "extrapolate-fold",
                "phase": "extrapolate",
                "kind": "agent",
                "expects": ["Generalization — Summary"],
                "instruction": (
                    "Use the discovery-extrapolator subagent to fold GENERALIZATION "
                    "verdicts into the report. The driver has already re-run each "
                    "reported finding's reproduced code on the extra datasets it was "
                    f"NOT generated on ({extra_list}); read the results JSON at "
                    f"{extrap_rel}. Each result has the origin fields PLUS an "
                    "`extrapolations` list — one entry per extra pkl with `pkl`, "
                    "`pkl_name`, `exitcode`, `timed_out`, `runtime_ms`, "
                    "`result_status`, `result_failure_reason`, `stdout`, `stderr`. "
                    "Do NOT re-run anything. For each hypothesis compare "
                    "the origin numbers to each extra dataset's numbers (direction, "
                    "significance, effect size) and assign GENERALIZES / PARTIAL / "
                    "DOES-NOT-GENERALIZE / INCONCLUSIVE (INCONCLUSIVE when a script "
                    f"could not run on an extra pkl). Read {summary_rel} and fold "
                    "the result INTO each hypothesis's existing ranked entry (append "
                    "Generalization / Across datasets bullets in place, keeping prior "
                    "bullets), plus one '## Generalization — Summary' section. "
                    + layout_rule("## Generalization — Summary")
                    + "Report the path and which findings do NOT generalize."
                ),
            }
        )

    steps.append(
        {
            "name": "verify-statistics-and-logic",
            "phase": "verify",
            "kind": "agent",
            "expects": ["Statistical Verification — Summary"],
            "instruction": (
                "Use the discovery-verifier subagent to audit whether each "
                "hypothesis's statistical test and its inductive/deductive "
                "reasoning are correct. For the recorded code, codeOutput and "
                f"analysis, run `{rank_cmd(json_rel, include_code=True)}` and read "
                "its stdout JSON — that already contains ONLY the reported "
                "records selected for this report, each with the "
                "recorded `code`, `codeOutput`, `analysis` and `review` "
                "(code/codeOutput middle-truncated); audit from that SLICE, do "
                f"NOT read the full multi-MB export {json_rel}. Also use the "
                "freshly reproduced results the previous rerun step folded into "
                f"{summary_rel} (do NOT re-run experiments yourself; use the "
                "rerun output already in the report). Check test choice, "
                "assumptions, power, effect size, p-value interpretation, the "
                "'failed-to-reject != null-is-true' fallacy, conclusion "
                "overreach, whether a finding that FAILED or DIVERGED on rerun "
                "should be downgraded, and a file-wide multiple-comparisons (FDR) "
                f"analysis. Read {summary_rel} and fold the audit INTO each "
                "hypothesis's existing ranked entry (append Verdict / Test / "
                "Statistical issues / Logic issues / Verdict rationale bullets in "
                "place, keeping the summary, reproduction and generalization "
                "bullets). Use the 5-level verdict scheme SOUND | WEAK | MINOR | "
                "MAJOR | CRITICAL (the downstream fix-tests step keys off "
                "MAJOR/CRITICAL and concrete test-fault bullets). Add one "
                "file-wide '## Statistical Verification — Summary' section with the "
                "verdict breakdown and a Benjamini–Hochberg FDR subsection. "
                + layout_rule("## Statistical Verification — Summary")
                + "Report the path and the most serious problems found."
            ),
        }
    )

    # Corrective phase: re-test the hypotheses the verifier flagged with a WRONG
    # statistical test. Split into author (agent writes corrected scripts) →
    # measure (DRIVER runs them) → fold (agent reads the JSON and judges), so the
    # re-measurement never runs inside an agent turn.
    corrected_note = (
        " (each `code_source: corrected` result also carries an `extrapolations` "
        "list — the SAME corrected test run on each extra dataset)."
        if extra_pkls_rel
        else "."
    )
    steps.append(
        {
            "name": "fix-tests-author",
            "phase": "fix-tests",
            "kind": "agent",
            "instruction": (
                "Use the discovery-test-fixer subagent to AUTHOR corrected scripts "
                "for the hypotheses whose statistical TEST the verifier flagged as "
                "wrong/unsound — do NOT re-measure or fold verdicts yet (the driver "
                f"re-measures, then a later step folds). Read {summary_rel} and "
                "select entries with a MAJOR/CRITICAL verdict or a 'Statistical "
                "issues' bullet naming a concrete test fault (wrong test for the "
                "data, violated independence/normality assumptions, huge-n "
                "significance of a trivial effect, p-value misuse, double-counting). "
                "Do NOT touch entries judged SOUND, or whose only problem is "
                "generalization. For each selected id, start from the loading-fixed "
                f"(or recorded) script {rerun_dir_rel}/hypo_<id>.py — the driver "
                "pre-exports every reported script there during the reproduce phase so "
                "they are always available — and write a corrected hypo_<id>.py into "
                f"{fixed_dir_rel} that changes ONLY the statistical test (print an "
                "effect size with a CI, and a cluster/permutation p-value where "
                "independence is violated), keeping the same data and quantities. "
                "Echo the original recorded numbers in a comment. If NO entry needs "
                f"a test fix, leave {fixed_dir_rel} empty and say so — the driver "
                "will skip the re-measurement. Report which ids you wrote corrected "
                "scripts for and the right test for each."
            ),
        }
    )
    steps.append(
        {
            "name": "fix-tests-measure",
            "phase": "fix-tests",
            "kind": "compute",
            "argv": rerun_argv(
                json_rel, pkl_rel,
                code_dir_rel=rerun_dir_rel, corrected_dir_rel=fixed_dir_rel,
                extra_pkls_rel=extra_pkls_rel,
                only_corrected=True,
                checkpoint_rel=corrected_checkpoint_rel,
            ),
            "out": corrected_rel,
            # No corrected scripts → nothing to re-measure; skip the run entirely.
            "skip_if_no_scripts_in": fixed_dir_rel,
            # Same empty-.rerun/ fallback as extrapolate-run: this step also
            # passes --code-dir rerun_dir_rel, so it needs the same guard rather
            # than relying on --code-dir's implicit empty-dir behaviour.
            "check_scripts_in": rerun_dir_rel,
            "argv_if_no_scripts": rerun_argv(
                json_rel, pkl_rel,
                corrected_dir_rel=fixed_dir_rel, extra_pkls_rel=extra_pkls_rel,
                only_corrected=True,
                checkpoint_rel=corrected_checkpoint_rel,
            ),
        }
    )
    steps.append(
        {
            "name": "fix-tests-fold",
            "phase": "fix-tests",
            "kind": "agent",
            # Folding is a no-op when the measure step was skipped; the driver
            # detects that and skips this step too. BOTH guards are needed:
            # skip_if_missing alone let a stale .corrected.json from an earlier
            # run be folded as if it belonged to this one, so this step skips
            # unless the author step actually wrote corrected scripts.
            "skip_if_no_scripts_in": fixed_dir_rel,
            "skip_if_missing": corrected_rel,
            "expects": ["Statistical Test Corrections — Summary"],
            "instruction": (
                "Use the discovery-test-result-folder subagent to fold CORRECTED-test "
                "verdicts into the report. The driver has re-measured the corrected "
                f"scripts; read the results JSON at {corrected_rel} and use the "
                f"`code_source: corrected` results and their `result_status` / "
                f"`result_failure_reason` fields{corrected_note} Do NOT re-run "
                "anything. For each corrected hypothesis assign UPHELD / WEAKENED / "
                "OVERTURNED by comparing the corrected numbers (statistic, p, effect "
                "size + CI) to the original"
                + (
                    ", plus a corrected-test generalization call (does the finding "
                    "still hold across the extra datasets under the right test?)"
                    if extra_pkls_rel
                    else ""
                )
                + ". Assign a verdict ONLY where the corrected `result_status` is "
                "USABLE. For a non-USABLE corrected result (FAILED/UNUSABLE, e.g. a "
                "timeout), describe the failure in the 'Corrected result' bullet and "
                "write NO 'Post-correction verdict' bullet at all — never "
                "INCONCLUSIVE or any other substitute token: the detector-build "
                "workflow rejects any verdict token that lacks a USABLE "
                "measurement, and the entry's original verifier verdict stands. "
                + f"Read {summary_rel} and fold Corrected test / Corrected result "
                "/ Post-correction verdict"
                + (" / Corrected generalization" if extra_pkls_rel else "")
                + " bullets INTO each fixed entry in place (keep all prior bullets; "
                "the verdict bullet only where allowed above), "
                "plus one '## Statistical Test Corrections — Summary' section. "
                + layout_rule("## Statistical Test Corrections — Summary")
                + "Report the path and which findings changed under the "
                "correct test."
            ),
        }
    )

    if _is_split_run(json_rel):
        steps.append(
            split_feature_label_step(json_rel, rerun_dir_rel, fixed_dir_rel)
        )

    return steps


def build_options() -> ClaudeAgentOptions:
    """Configure the SDK session for this project.

    ``setting_sources=["project"]`` is what makes the SDK auto-discover the
    filesystem subagents in ``.claude/agents/`` (and project settings) relative
    to ``cwd`` — without it the discovery-summarizer agent would not be loaded.
    """
    return ClaudeAgentOptions(
        cwd=str(PROJECT_ROOT),
        # Load .claude/agents/*.md (incl. discovery-summarizer) and project settings.
        setting_sources=["project"],
        # The orchestrator delegates to subagents (Task), which need filesystem
        # tools to rank records and update the combined Markdown deliverable.
        allowed_tools=["Task", "Bash", "Read", "Write", "Edit", "Glob"],
        # Non-interactive: don't prompt for permission on each tool call. Drop to
        # "acceptEdits" if you'd rather review/limit what runs.
        permission_mode="bypassPermissions",
        # Pin Opus 4.8 explicitly so the model is not left to the ambient session
        # default. Subagents with `model: inherit` follow this; agents that name a
        # cheaper model in their frontmatter override it.
        model="claude-opus-4-8",
        # Default reasoning effort for the ORCHESTRATOR turns (levels:
        # low|medium|high|xhigh|max). Per-subagent effort is set in each
        # .claude/agents/*.md frontmatter and takes precedence for that subagent's
        # turns — the mechanical fold agents run at medium, the analytical ones
        # (summarize/verify/test-fixer) at xhigh. We set the
        # session default via the SDK's native `effort` field, NOT the old
        # `CLAUDE_EFFORT` env var — that var is an OUTPUT the CLI exports to hooks
        # to report the active level, not an input it reads, so setting it did
        # nothing. (The env var that WOULD override is CLAUDE_CODE_EFFORT_LEVEL,
        # which we deliberately do not set so frontmatter effort wins.)
        effort="xhigh",
    )


class WorkflowCostSummary:
    """Track turn cost without double-counting session-cumulative reports."""

    def __init__(self) -> None:
        self.started_turns = 0
        self.reported_turns = 0
        self.unreported_turns = 0
        self._session_usd: dict[object, float] = {}

    @property
    def total_usd(self) -> float:
        return sum(self._session_usd.values())

    def reset(self) -> None:
        self.__init__()

    def start_turn(self) -> None:
        self.started_turns += 1

    def add(self, cost_usd: object, session_id: object = None) -> float | None:
        """Record one SDK cost report and return this turn's increment."""
        if cost_usd is None:
            self.unreported_turns += 1
            return None
        try:
            cost = float(cost_usd)
        except (TypeError, ValueError):
            self.unreported_turns += 1
            return None
        if not math.isfinite(cost) or cost < 0:
            self.unreported_turns += 1
            return None
        previous = self._session_usd.get(session_id, 0.0)
        self._session_usd[session_id] = max(previous, cost)
        self.reported_turns += 1
        return max(0.0, cost - previous)

    def describe(self) -> str:
        unfinished = max(
            0,
            self.started_turns - self.reported_turns - self.unreported_turns,
        )
        coverage = (
            f"{self.reported_turns} reported turn(s), "
            f"{self.unreported_turns} completed turn(s) without a cost, "
            f"{unfinished} started turn(s) without a ResultMessage"
        )
        if self.reported_turns == 0:
            return f"unavailable ({coverage})"
        return f"${self.total_usd:.4f} ({coverage})"


WORKFLOW_COSTS = WorkflowCostSummary()


def _first_usage_value(usage: dict[str, object], *keys: str) -> object | None:
    for key in keys:
        if key in usage:
            return usage[key]
    return None


def _format_token_usage(usage: object) -> str | None:
    """Compact the SDK's snake_case or camelCase token counters for logs."""
    if not isinstance(usage, dict):
        return None
    fields = (
        ("input", ("input_tokens", "inputTokens")),
        ("output", ("output_tokens", "outputTokens")),
        ("cache_read", ("cache_read_input_tokens", "cacheReadInputTokens")),
        (
            "cache_write",
            ("cache_creation_input_tokens", "cacheCreationInputTokens"),
        ),
    )
    parts = []
    for label, keys in fields:
        value = _first_usage_value(usage, *keys)
        if value is not None:
            parts.append(f"{label}={value}")
    return ", ".join(parts) if parts else None


def _format_model_usage(model_usage: object) -> str | None:
    """Return per-model token and cost details from one SDK result."""
    if not isinstance(model_usage, dict):
        return None
    models = []
    for model, details in sorted(model_usage.items(), key=lambda item: str(item[0])):
        if not isinstance(details, dict):
            models.append(f"{model}: {details}")
            continue
        parts = []
        token_usage = _format_token_usage(details)
        if token_usage:
            parts.append(token_usage)
        cost = _first_usage_value(details, "cost_usd", "costUSD")
        if cost is not None:
            try:
                parts.append(f"cost=${float(cost):.4f}")
            except (TypeError, ValueError):
                parts.append(f"cost={cost}")
        models.append(f"{model} ({', '.join(parts) or 'usage unavailable'})")
    return "; ".join(models) if models else None


def _new_step_session_id(step_name: str) -> str:
    """Return a unique readable SDK session id for one isolated agent turn."""
    safe_name = re.sub(r"[^A-Za-z0-9_-]+", "-", step_name).strip("-")
    return f"discovery-{safe_name[:48] or 'step'}-{uuid.uuid4().hex}"


async def _run_step_inner(
    client: ClaudeSDKClient,
    step: dict[str, object],
    session_id: str,
) -> str:
    """Core logic for one agent step — called by run_step inside a timeout guard."""
    WORKFLOW_COSTS.start_turn()
    log(f"  [session] {step['name']}: {session_id}")
    await client.query(str(step["instruction"]), session_id=session_id)

    chunks: list[str] = []
    n_tools = 0
    async for message in client.receive_response():
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    chunks.append(block.text)
                elif ToolUseBlock and isinstance(block, ToolUseBlock):
                    n_tools += 1
                    log(f"  → {describe_tool(block)}")
        elif isinstance(message, ResultMessage):
            cost = getattr(message, "total_cost_usd", None)
            delta = WORKFLOW_COSTS.add(
                cost, getattr(message, "session_id", None)
            )
            dur_ms = getattr(message, "duration_ms", None)
            parts = [f"{n_tools} tool call(s)"]
            if dur_ms is not None:
                parts.append(f"{dur_ms / 1000:.0f}s")
            if delta is not None:
                parts.append(
                    f"${delta:.4f} this turn "
                    f"(${float(cost):.4f} session cumulative)"
                )
            log(f"  step turn finished — {', '.join(parts)}")
            usage = _format_token_usage(getattr(message, "usage", None))
            if usage:
                log(f"  [usage] {step['name']}: {usage}")
            models = _format_model_usage(getattr(message, "model_usage", None))
            if models:
                log(f"  [model-usage] {step['name']}: {models}")
    return "".join(chunks)


async def run_step(client: ClaudeSDKClient, step: dict[str, object]) -> str:
    """Send one workflow step in an isolated SDK session and return its text.

    ``step`` is one entry from ``build_steps`` (plus the lighter dict ``run_smoke``
    builds), so its values are heterogeneous — ``instruction`` is a str, but
    siblings hold argv lists, ``None``, and skip flags.

    The client transport is shared, but every call receives a fresh logical
    session id, so a later stage cannot inherit an earlier stage's conversation.
    Files remain the sole cross-stage state. Wraps ``_run_step_inner`` in the
    ``AGENT_STEP_TIMEOUT_S`` guard so a hung session aborts the run instead of
    stalling it forever.
    """
    timeout_s = AGENT_STEP_TIMEOUT_S
    session_id = _new_step_session_id(str(step["name"]))
    try:
        return await asyncio.wait_for(
            _run_step_inner(client, step, session_id),
            timeout=timeout_s,
        )
    except asyncio.TimeoutError:
        raise SystemExit(
            f"Agent step '{step['name']}' timed out after {timeout_s}s. "
            "Aborting — increase AGENT_STEP_TIMEOUT_S if it legitimately "
            "needs more time."
        )


def _validate_step_output(step: dict, summary_path: Path) -> None:
    """Warn if expected section markers are absent from the summary after a fold step.

    Each fold step declares ``"expects": [marker, ...]`` — strings that must
    appear in the summary file if the step ran correctly. A missing marker does
    not abort the workflow (the agent may have legitimately phrased things
    differently) but logs a visible WARNING so the operator can inspect the file.
    ``"expects_file"`` checks that a separate output file exists (used by the
    predictive-selection step, which writes a JSON manifest rather than editing
    the summary).
    """
    expects = step.get("expects", [])
    expects_file = step.get("expects_file")

    if expects:
        if not summary_path.is_file():
            log(f"  [validate] WARNING: summary {summary_path.name} not found "
                f"after step '{step['name']}'.")
        else:
            text = summary_path.read_text(encoding="utf-8", errors="replace")
            missing = [m for m in expects if m not in text]
            if missing:
                log(f"  [validate] WARNING: step '{step['name']}' may be incomplete — "
                    f"expected markers absent from {summary_path.name}: {missing}")
            else:
                log(f"  [validate] OK: expected markers present after '{step['name']}'.")

    if expects_file:
        ef = PROJECT_ROOT / expects_file
        if not ef.is_file() or ef.stat().st_size == 0:
            log(f"  [validate] WARNING: step '{step['name']}' expected output file "
                f"{expects_file} is missing or empty.")
        else:
            log(f"  [validate] OK: output file {expects_file} exists after '{step['name']}'.")


def _rel_to_root(path: Path) -> str:
    """Path relative to PROJECT_ROOT (the session cwd), tolerating ``..``.

    The dataset pkl often lives outside the project (e.g. ``../data/``), so a
    plain ``relative_to`` would raise — ``os.path.relpath`` handles parent dirs.
    """
    return Path(os.path.relpath(path, PROJECT_ROOT)).as_posix()


def _resolve_split_label_inputs(
    run_json: Path,
) -> tuple[list[int], Path, Path | None]:
    """Resolve current selected sources without loading a dataset.

    Predictive artifacts take precedence, matching detector-build input
    resolution. Legacy rerun artifacts remain labelable for older completed
    discovery runs.
    """
    predictive_rerun = run_json.with_name(f"{run_json.name}.predictive.rerun")
    legacy_rerun = run_json.with_name(f"{run_json.name}.rerun")
    predictive_ready = predictive_rerun.is_dir() and any(
        predictive_rerun.glob("hypo_*.py")
    )
    rerun_dir = predictive_rerun if predictive_ready else legacy_rerun
    manifest_path = rerun_dir / "MANIFEST.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        records = manifest["records"]
        rerun_ids = [row["id"] for row in records]
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise SystemExit(
            f"Cannot resolve split feature sources from "
            f"{_rel_to_root(manifest_path)}: {exc}"
        ) from exc
    if (
        not rerun_ids
        or any(isinstance(value, bool) or not isinstance(value, int) for value in rerun_ids)
        or len(rerun_ids) != len(set(rerun_ids))
    ):
        raise SystemExit("Split feature labeling requires unique integer rerun IDs.")

    if predictive_ready:
        selection = predictive_manifest_path(run_json)
        _validate_predictive_manifest(
            _rel_to_root(selection), _rel_to_root(run_json), log_success=False
        )
        selected_payload = json.loads(selection.read_text(encoding="utf-8"))
        selected_ids = selected_payload["selected_ids"]
        if (
            not selected_ids
            or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in selected_ids
            )
            or len(selected_ids) != len(set(selected_ids))
        ):
            raise SystemExit(
                "Split feature labeling requires unique integer selected IDs."
            )
        if set(selected_ids) != set(rerun_ids):
            missing = sorted(set(selected_ids) - set(rerun_ids))
            extra = sorted(set(rerun_ids) - set(selected_ids))
            raise SystemExit(
                "Predictive selection and rerun MANIFEST must contain the same "
                f"hypothesis IDs; missing from rerun={missing}, extra in rerun={extra}."
            )
        fixed_candidate = run_json.with_name(
            f"{run_json.name}.predictive.fixed"
        )
    else:
        selected_ids = rerun_ids
        fixed_candidate = run_json.with_name(f"{run_json.name}.fixed")
    fixed_dir = (
        fixed_candidate
        if fixed_candidate.is_dir() and any(fixed_candidate.glob("hypo_*.py"))
        else None
    )
    return selected_ids, rerun_dir, fixed_dir


def _compile_split_label_step(spec: dict[str, object]) -> None:
    run_json = PROJECT_ROOT / str(spec["run"])
    selected_ids, rerun_dir, fixed_dir = _resolve_split_label_inputs(run_json)
    draft = PROJECT_ROOT / str(spec["draft"])
    output = PROJECT_ROOT / str(spec["output"])
    compile_applicability(
        draft,
        output,
        run_json=run_json,
        selected_ids=selected_ids,
        rerun_dir=rerun_dir,
        fixed_dir=fixed_dir,
        project_root=PROJECT_ROOT,
    )
    load_applicability(
        output,
        run_json=run_json,
        selected_ids=selected_ids,
        rerun_dir=rerun_dir,
        fixed_dir=fixed_dir,
        project_root=PROJECT_ROOT,
    )
    draft.unlink()
    log(
        f"  [validate] compiled source-bound split feature applicability at "
        f"{_rel_to_root(output)}."
    )


def _compile_existing_split_label_draft(spec: dict[str, object]) -> bool:
    """Compile a valid draft left by an interrupted run before relabeling."""
    draft = PROJECT_ROOT / str(spec["draft"])
    if not draft.is_file():
        return False
    try:
        _compile_split_label_step(spec)
    except SystemExit as exc:
        log(
            f"  [agent] existing split applicability draft is stale or invalid; "
            f"relabeling: {exc}"
        )
        return False
    log("  [agent] reused and compiled the existing split applicability draft.")
    return True


def _has_hypo_scripts(dir_rel: str) -> bool:
    """True if ``dir_rel`` holds at least one ``hypo_<id>.py``.

    Used in three contexts: ``skip_if_scripts_exist_in`` (driver-exported
    scripts, avoid clobbering on resume), ``skip_if_no_scripts_in`` (agent-
    authored corrected scripts, nothing to re-measure), and ``check_scripts_in``
    (select fallback argv when the reproduce phase was skipped).
    """
    d = PROJECT_ROOT / dir_rel
    return d.is_dir() and any(d.glob("hypo_*.py"))


def _load_results(results_rel: str) -> list[dict]:
    """The ``results`` list of a compute-output JSON, or [] when unreadable."""
    try:
        payload = json.loads((PROJECT_ROOT / results_rel).read_text())
    except (OSError, ValueError):
        return []
    results = payload.get("results") if isinstance(payload, dict) else None
    if not isinstance(results, list):
        return []
    return [r for r in results if isinstance(r, dict)]


def _has_unusable_results(results_rel: str) -> bool:
    """Whether any result in ``results_rel`` is still marked UNUSABLE."""
    return any(
        r.get("result_status") == "UNUSABLE" for r in _load_results(results_rel)
    )


def _scripts_changed_since(code_dir_rel: str, results_rel: str) -> bool:
    """Whether any exported script differs from what ``results_rel`` measured.

    Mirrors ``rerun_experiments.resolve_code``: a present, non-blank
    ``hypo_<id>.py`` supplies the code (hash of its exact text), otherwise the
    recorded code does (the result's ``recorded_code_sha256``). Comparing that
    candidate against ``executed_code_sha256`` — without loading any dataset —
    tells the repair loop whether re-measuring could produce anything new.
    """
    results = _load_results(results_rel)
    if not results:
        return True  # unreadable results cannot prove convergence — run.
    for r in results:
        safe = re.sub(r"[^0-9A-Za-z_.-]", "_", str(r.get("id")))
        script = PROJECT_ROOT / code_dir_rel / f"hypo_{safe}.py"
        candidate = r.get("recorded_code_sha256")
        try:
            text = script.read_text(encoding="utf-8")
        except OSError:
            text = ""
        if text.strip():
            candidate = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if candidate != r.get("executed_code_sha256"):
            return True
    return False


def _log_failure_summary(results_rel: str, step_name: str) -> None:
    """Per-id failure digest into the driver log after a measuring step.

    The results JSON already records why each hypothesis is UNUSABLE; surfacing
    that here means the log alone shows what the repair loop is still chasing
    (and what it finally gave up on) without opening the artifact.
    """
    results = _load_results(results_rel)
    unusable = [r for r in results if r.get("result_status") == "UNUSABLE"]
    log(
        f"  [results] {step_name}: {len(results) - len(unusable)} USABLE, "
        f"{len(unusable)} UNUSABLE ({results_rel})."
    )
    for r in unusable:
        stderr_tail = ""
        for line in reversed(str(r.get("rerun_stderr") or "").splitlines()):
            if line.strip():
                stderr_tail = line.strip()[:200]
                break
        reason = r.get("result_failure_reason") or "unknown"
        source = r.get("code_source") or "recorded"
        log(
            f"  [results]   id {r.get('id')}: {reason} ({source})"
            + (f" — {stderr_tail}" if stderr_tail else "")
        )


def _broken_changed_scripts(code_dir_rel: str, results_rel: str) -> list[str]:
    """Compile-check every script that differs from what ``results_rel`` measured.

    A syntax-broken loading fix would otherwise cost a full re-measurement run
    (dataset load included) just to fail at import; this driver-side gate
    catches it in milliseconds so the fix agent can repair it for free.
    """
    problems: list[str] = []
    for r in _load_results(results_rel):
        safe = re.sub(r"[^0-9A-Za-z_.-]", "_", str(r.get("id")))
        script = PROJECT_ROOT / code_dir_rel / f"hypo_{safe}.py"
        try:
            text = script.read_text(encoding="utf-8")
        except OSError:
            continue
        if not text.strip():
            continue
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if digest == r.get("executed_code_sha256"):
            continue  # unchanged since it was last measured — already ran once
        try:
            compile(text, str(script), "exec")
        except SyntaxError as exc:
            problems.append(f"{script.name} line {exc.lineno}: {exc.msg}")
    return problems


def _syntax_repair_instruction(problems: list[str], code_dir_rel: str) -> str:
    """The follow-up prompt for a fix turn that left scripts unparseable."""
    bullets = "\n".join(f"- {problem}" for problem in problems)
    return (
        "The driver compile-checked every script you changed BEFORE spending a "
        "re-measurement run, and these do not parse:\n"
        f"{bullets}\n"
        f"Fix ONLY these syntax errors in place under {code_dir_rel}. Keep the "
        "statistical analysis unchanged and do not touch any other script. "
        "Report which files you repaired."
    )


def _failure_brief(results_rel: str, code_dir_rel: str, limit: int = 20) -> str:
    """Digest of still-UNUSABLE results for embedding into a fix instruction.

    The same facts live in the results JSON, but handing them to the agent up
    front saves it a read-and-filter pass per repair round and anchors the turn
    on the actual failures.
    """
    unusable = [
        r for r in _load_results(results_rel)
        if r.get("result_status") == "UNUSABLE"
    ]
    if not unusable:
        return ""
    lines = [
        f"DRIVER DIGEST — {len(unusable)} UNUSABLE result(s) in {results_rel} "
        f"(scripts in {code_dir_rel}; full stdout/stderr in the JSON):"
    ]
    for r in unusable[:limit]:
        tail = ""
        for line in reversed(str(r.get("rerun_stderr") or "").splitlines()):
            if line.strip():
                tail = line.strip()[:200]
                break
        reason = r.get("result_failure_reason") or "unknown"
        source = r.get("code_source") or "recorded"
        entry = f"- id {r.get('id')}: {reason} ({source})"
        if tail:
            entry += f" — {tail}"
        lines.append(entry)
    if len(unusable) > limit:
        lines.append(
            f"- … and {len(unusable) - limit} more; read the JSON for the rest."
        )
    return "\n".join(lines)


def _rerun_manifest_matches_selection(dir_rel: str, selection_rel: str) -> bool:
    """Whether an exported rerun directory contains exactly the selected ids."""
    try:
        rerun_manifest = json.loads(
            (PROJECT_ROOT / dir_rel / "MANIFEST.json").read_text()
        )
        selection = json.loads((PROJECT_ROOT / selection_rel).read_text())
        records = rerun_manifest.get("records")
        selected_ids = selection.get("selected_ids")
        if not isinstance(records, list) or not isinstance(selected_ids, list):
            return False
        rerun_ids = [
            record.get("id") if isinstance(record, dict) else None
            for record in records
        ]
        if None in rerun_ids or len(rerun_ids) != len(set(rerun_ids)):
            return False
        if len(selected_ids) != len(set(selected_ids)):
            return False
        if set(rerun_ids) != set(selected_ids):
            return False
        return all(
            (PROJECT_ROOT / dir_rel / f"hypo_{hypothesis_id}.py").is_file()
            for hypothesis_id in selected_ids
        )
    except (OSError, TypeError, ValueError):
        return False


def _source_sha256(source_path: Path) -> str:
    """Content hash used to invalidate a cached predictive selection."""
    return hashlib.sha256(source_path.read_bytes()).hexdigest()


def _validate_predictive_manifest(
    manifest_rel: str, source_rel: str, *, log_success: bool = True
) -> None:
    """Fail fast unless a predictive manifest partitions every source ID once."""
    manifest_path = PROJECT_ROOT / manifest_rel
    source_path = PROJECT_ROOT / source_rel
    try:
        manifest = json.loads(manifest_path.read_text())
        source = json.loads(source_path.read_text())
    except (OSError, ValueError) as exc:
        raise SystemExit(f"Invalid predictive selection manifest: {exc}")

    if isinstance(source, dict):
        for key in ("records", "hypotheses", "results", "data"):
            if isinstance(source.get(key), list):
                source = source[key]
                break
    if not isinstance(source, list) or not isinstance(manifest, dict):
        raise SystemExit("Predictive selection source/manifest has an invalid schema.")

    selected = manifest.get("selected_ids")
    excluded = manifest.get("excluded")
    if not isinstance(selected, list) or not isinstance(excluded, list):
        raise SystemExit(
            "Predictive manifest must contain selected_ids and excluded lists."
        )
    excluded_ids = [row.get("id") for row in excluded if isinstance(row, dict)]
    if len(excluded_ids) != len(excluded):
        raise SystemExit("Every predictive manifest excluded entry must contain an id.")

    source_ids = [row.get("id") for row in source if isinstance(row, dict)]
    source_keys = [json.dumps(value, sort_keys=True) for value in source_ids]
    selected_keys = [json.dumps(value, sort_keys=True) for value in selected]
    excluded_keys = [json.dumps(value, sort_keys=True) for value in excluded_ids]
    combined = selected_keys + excluded_keys
    if len(combined) != len(set(combined)):
        raise SystemExit(
            "Predictive manifest contains duplicate or overlapping hypothesis IDs."
        )
    if set(combined) != set(source_keys):
        missing = sorted(set(source_keys) - set(combined))
        extra = sorted(set(combined) - set(source_keys))
        raise SystemExit(
            f"Predictive manifest does not partition all source IDs; "
            f"missing={missing}, extra={extra}."
        )
    if log_success:
        log(
            f"  [validate] predictive manifest selected {len(selected)} of "
            f"{len(source_ids)} hypotheses."
        )


def _stamp_predictive_manifest(manifest_rel: str, source_rel: str) -> None:
    """Add cache metadata after the agent's selection has been validated."""
    manifest_path = PROJECT_ROOT / manifest_rel
    source_path = PROJECT_ROOT / source_rel
    payload = json.loads(manifest_path.read_text())
    payload["source_sha256"] = _source_sha256(source_path)
    payload["policy_version"] = PREDICTIVE_POLICY_VERSION
    payload["generated_at"] = datetime.now().astimezone().isoformat()
    manifest_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _predictive_manifest_is_current(manifest_rel: str, source_rel: str) -> bool:
    """Whether a cached manifest matches the source and current policy."""
    manifest_path = PROJECT_ROOT / manifest_rel
    source_path = PROJECT_ROOT / source_rel
    try:
        payload = json.loads(manifest_path.read_text())
        if payload.get("policy_version") != PREDICTIVE_POLICY_VERSION:
            return False
        if payload.get("source_sha256") != _source_sha256(source_path):
            return False
        _validate_predictive_manifest(
            manifest_rel, source_rel, log_success=False
        )
    except (OSError, ValueError, SystemExit):
        return False
    return True


def survey_prior_artifacts(json_path: Path, direction: str) -> list[tuple[Path, str]]:
    """Describe prior-run artifacts that could be reused or folded accidentally.

    This is deliberately read-only and conservative. Checkpoints are listed as
    validated caches rather than called stale: ``rerun_experiments.py`` verifies
    their source/code/data fingerprints before reuse. Reports, result JSONs and
    editable scripts can influence agent decisions, so a human sees them before
    the driver overwrites its log or starts any workflow step.
    """
    scope = ".predictive" if direction == "predictive" else ""
    findings: list[tuple[Path, str]] = []

    def add(path: Path, reason: str) -> None:
        if path.exists():
            findings.append((path, reason))

    add(json_path.with_suffix(".summary.md"), "existing analysis report")
    add(
        split_applicability_path(json_path),
        "source-bound split feature applicability from a previous invocation",
    )
    add(
        split_applicability_draft_path(json_path),
        "unfinished split feature applicability draft",
    )
    prior_log = json_path.with_name(
        f"{json_path.name}{scope}{DRIVER_LOG_SUFFIX}"
    )
    add(prior_log, "previous workflow log; the default path will be overwritten")

    result_suffixes = (
        "reproduce-raw.json", "reproduce.json", "extrapolate.json", "corrected.json"
    )
    for suffix in result_suffixes:
        result_path = json_path.with_name(f"{json_path.name}{scope}.{suffix}")
        add(result_path, "previous measured-analysis JSON")
        add(
            result_path.with_name(f"{result_path.name}.checkpoint.json"),
            "prior compute checkpoint; fingerprint-validated before reuse",
        )

    rerun_dir = json_path.with_name(f"{json_path.name}{scope}.rerun")
    fixed_dir = json_path.with_name(f"{json_path.name}{scope}.fixed")
    if _has_hypo_scripts(_rel_to_root(rerun_dir)):
        findings.append((rerun_dir, "editable rerun scripts from a previous invocation"))
    if _has_hypo_scripts(_rel_to_root(fixed_dir)):
        findings.append((fixed_dir, "corrected-analysis scripts from a previous invocation"))

    if direction == "predictive":
        selection_path = predictive_manifest_path(json_path)
        if selection_path.exists():
            selection_rel = _rel_to_root(selection_path)
            source_rel = _rel_to_root(json_path)
            if not _predictive_manifest_is_current(selection_rel, source_rel):
                findings.append((
                    selection_path,
                    "STALE predictive selection (source hash, policy, or schema changed)",
                ))

            if rerun_dir.is_dir() and not _rerun_manifest_matches_selection(
                _rel_to_root(rerun_dir), selection_rel
            ):
                findings.append((
                    rerun_dir / "MANIFEST.json",
                    "STALE/MISMATCHED rerun manifest or selected script set",
                ))

    # Keep one entry per concrete path, preferring the later/more-specific reason.
    deduplicated: dict[Path, str] = {}
    for path, reason in findings:
        deduplicated[path] = reason
    return sorted(deduplicated.items(), key=lambda item: item[0].as_posix())


def confirm_prior_artifacts(
    findings: list[tuple[Path, str]], *, assume_yes: bool
) -> None:
    """Require explicit y/n confirmation before a run can reuse prior artifacts."""
    if not findings:
        return

    print("\nWARNING: prior workflow artifacts were found:", file=sys.stderr)
    for path, reason in findings:
        print(f"  - {_rel_to_root(path)}: {reason}", file=sys.stderr)
    print(
        "These files may be reused, refreshed, overwritten, or folded into this "
        "run. Review the list before proceeding.",
        file=sys.stderr,
    )

    if assume_yes:
        print("[stale-artifact-check] proceeding via --yes-stale", file=sys.stderr)
        return
    if not sys.stdin.isatty():
        raise SystemExit(
            "Prior workflow artifacts require human confirmation, but stdin is "
            "not a terminal. Re-run interactively or pass --yes-stale after review."
        )

    while True:
        print("Proceed with these existing artifacts? [y/n]: ", end="", file=sys.stderr, flush=True)
        answer = sys.stdin.readline().strip().lower()
        if answer in {"y", "yes"}:
            print("[stale-artifact-check] confirmed by user", file=sys.stderr)
            return
        if answer in {"n", "no"}:
            raise SystemExit("Aborted — existing workflow artifacts were left untouched.")
        print("Please answer y or n.", file=sys.stderr)


def run_compute_step(step: dict[str, object]) -> str:
    """Execute one ``kind == "compute"`` step in the driver (no agent involved).

    Honors skip rules so a resume reuses expensive results and the common path
    stays cheap. Checked in order:
    - ``stall_if_code_unchanged``: a repair-round remeasure skips — and reports
      the repair loop as stalled — when no exported script differs from the
      code hashes its results JSON already measured (the preceding fix round
      changed nothing). Checked driver-side so a stalled round never pays the
      dataset load.
    - ``skip_if_scripts_exist_in``: skip (without overwriting) when the target
      dir already holds hypo_*.py scripts. Predictive exports additionally use
      ``skip_if_scripts_match_selection``: a stale/missing MANIFEST or script
      forces a refresh instead of silently reusing the wrong hypothesis set.
    - ``skip_if_no_scripts_in``: if the preceding agent wrote no hypo_<id>.py
      there, this run would have no work, so skip it.
    - all result-producing steps enter the helper, which cheaply reuses its
      per-work-item checkpoint and notices changed code/data/configuration.
    - ``check_scripts_in`` / ``argv_if_no_scripts``: when the dir is empty, use
      the fallback argv (without ``--code-dir``) so the step runs against the
      recorded code instead of an empty directory.

    Skip rules deliberately run before launching the helper so an empty corrected
    phase cannot make a fold step consume stale output from an earlier run.

    Returns ``"ran"``, ``"skipped"``, or ``"stalled"`` so the step loop can end
    a repair loop whose latest fix round produced no change.
    """
    out_rel = step.get("out")  # type: ignore[assignment]
    # 0) Repair-round remeasure with nothing new to measure — mark the stall.
    stall_spec = step.get("stall_if_code_unchanged")
    if isinstance(stall_spec, dict):
        code_dir = str(stall_spec["code_dir"])
        results_rel = str(stall_spec["results"])
        if not _scripts_changed_since(code_dir, results_rel):
            log(
                f"  [compute] skipped: no script in {code_dir} differs from the "
                f"code hashes already measured in {results_rel} — the preceding "
                "fix round changed nothing, so the repair loop ends here."
            )
            return "stalled"
    # 1) Scripts already on disk — skip export to avoid clobbering edits.
    skip_exist = step.get("skip_if_scripts_exist_in")
    if skip_exist and _has_hypo_scripts(str(skip_exist)):
        selection = step.get("skip_if_scripts_match_selection")
        if not selection or _rerun_manifest_matches_selection(
            str(skip_exist), str(selection)
        ):
            log(f"  [compute] skipped (scripts already in {skip_exist}; "
                "not overwriting).")
            return "skipped"
        log(
            f"  [compute] {skip_exist} does not match {selection}; refreshing "
            "the export and removing stale workflow-owned hypo_*.py files."
        )
    # 2) Nothing for the preceding agent to have fixed/corrected → skip the run.
    skip_dir = step.get("skip_if_no_scripts_in")
    if skip_dir and not _has_hypo_scripts(str(skip_dir)):
        log(f"  [compute] skipped (no scripts in {skip_dir}); "
            "nothing to re-measure.")
        if out_rel and (PROJECT_ROOT / str(out_rel)).is_file():
            log(f"  [compute] NOTE: {out_rel} on disk is from an EARLIER run "
                "and is NOT part of this one; the fold step skips with this "
                "step, so it will not be read.")
        return "skipped"
    # 3) Select argv: use the no-code-dir fallback when the scripts dir is empty
    #    (e.g. reproduce phase was skipped) to avoid passing --code-dir to an
    #    empty directory whose behaviour is implicit in rerun_experiments.py.
    check_dir = step.get("check_scripts_in")
    if check_dir and not _has_hypo_scripts(str(check_dir)):
        argv = list(step.get("argv_if_no_scripts", step["argv"]))
        log(f"  [compute] {check_dir} is empty — using fallback argv "
            "(no --code-dir).")
    else:
        argv = list(step["argv"])
    run_compute(argv, str(out_rel) if out_rel else None)
    return "ran"


def _print_smoke_selection(json_rel: str) -> list[str]:
    """Print and return one line per selected hypothesis."""
    command = shlex.split(rank_cmd(json_rel))
    proc = subprocess.run(
        command,
        cwd=str(PROJECT_ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        raise SystemExit(
            f"Smoke selection failed (exit {proc.returncode}): "
            f"{' '.join(shlex.quote(part) for part in command)}\n{proc.stderr}"
        )
    try:
        payload = json.loads(proc.stdout)
        records = payload["records"]
    except (ValueError, KeyError, TypeError) as exc:
        raise SystemExit(f"Smoke selection returned invalid JSON: {exc}")

    summaries = []
    for record in records:
        hypothesis = " ".join(str(record.get("hypothesis", "")).split())
        summary = f"ID {record.get('id')}: {hypothesis}"
        summaries.append(summary)
        print(summary)
    return summaries


def _save_smoke_summaries(manifest_rel: str, summaries: list[str]) -> None:
    """Persist the exact human-readable smoke output in its manifest."""
    manifest_path = PROJECT_ROOT / manifest_rel
    try:
        payload = json.loads(manifest_path.read_text())
    except (OSError, ValueError) as exc:
        raise SystemExit(f"Unable to save smoke summaries to manifest: {exc}")
    if not isinstance(payload, dict):
        raise SystemExit("Unable to save smoke summaries: manifest is not an object.")
    payload["one_line_summaries"] = summaries
    manifest_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


async def run_smoke(json_path: Path) -> None:
    """Select, print, and stop without running the discovery workflow."""
    WORKFLOW_COSTS.reset()
    global PREDICTIVE_MANIFEST
    json_rel = _rel_to_root(json_path)

    if DIRECTION != "predictive":
        PREDICTIVE_MANIFEST = None
        _print_smoke_selection(json_rel)
        return

    manifest_path = predictive_manifest_path(json_path)
    PREDICTIVE_MANIFEST = _rel_to_root(manifest_path)
    try:
        if not _predictive_manifest_is_current(PREDICTIVE_MANIFEST, json_rel):
            step = {
                "name": "smoke-select-predictive-hypotheses",
                "instruction": predictive_selection_instruction(
                    json_rel, PREDICTIVE_MANIFEST
                ),
            }
            async with ClaudeSDKClient(options=build_options()) as client:
                await run_step(client, step)
            _validate_predictive_manifest(PREDICTIVE_MANIFEST, json_rel)
            _stamp_predictive_manifest(PREDICTIVE_MANIFEST, json_rel)
        summaries = _print_smoke_selection(json_rel)
        _save_smoke_summaries(PREDICTIVE_MANIFEST, summaries)
    finally:
        PREDICTIVE_MANIFEST = None


async def run_split_feature_labeling_only(json_path: Path) -> None:
    """Backfill/refresh applicability metadata without rerunning discovery."""
    WORKFLOW_COSTS.reset()
    if not _is_split_run(json_path):
        raise SystemExit("--split-feature-labels-only requires a split-error run.")
    selected_ids, rerun_dir, fixed_dir = _resolve_split_label_inputs(json_path)
    global PREDICTIVE_MANIFEST
    PREDICTIVE_MANIFEST = (
        _rel_to_root(predictive_manifest_path(json_path))
        if ".predictive.rerun" in rerun_dir.name
        else None
    )
    output = split_applicability_path(json_path)
    if output.is_file():
        try:
            load_applicability(
                output,
                run_json=json_path,
                selected_ids=selected_ids,
                rerun_dir=rerun_dir,
                fixed_dir=fixed_dir,
                project_root=PROJECT_ROOT,
            )
        except SystemExit as exc:
            log(f"Existing split feature applicability is stale; refreshing: {exc}")
        else:
            log(
                f"Reusing current split feature applicability "
                f"{_rel_to_root(output)}."
            )
            return
    fixed_candidate = fixed_dir or json_path.with_name(
        f"{json_path.name}{'.predictive' if PREDICTIVE_MANIFEST else ''}.fixed"
    )
    step = split_feature_label_step(
        _rel_to_root(json_path),
        _rel_to_root(rerun_dir),
        _rel_to_root(fixed_candidate),
    )
    if _compile_existing_split_label_draft(
        dict(step["compile_split_applicability"])
    ):
        return
    async with ClaudeSDKClient(options=build_options()) as client:
        final_text = await run_step(client, step)
        print(final_text.strip())
    _validate_step_output(step, json_path.with_suffix(".summary.md"))
    _compile_split_label_step(dict(step["compile_split_applicability"]))


async def run_workflow(
    json_path: Path,
    pkl_path: Path,
    extra_pkl_paths: list[Path],
) -> None:
    WORKFLOW_COSTS.reset()
    global PREDICTIVE_MANIFEST
    options = build_options()
    # Paths handed to the agent are relative to PROJECT_ROOT (the session cwd).
    json_rel = _rel_to_root(json_path)
    if DIRECTION == "predictive":
        PREDICTIVE_MANIFEST = _rel_to_root(predictive_manifest_path(json_path))
    else:
        PREDICTIVE_MANIFEST = None
    pkl_rel = _rel_to_root(pkl_path)
    extra_pkls_rel = [_rel_to_root(p) for p in extra_pkl_paths]
    summary_rel = _rel_to_root(json_path.with_suffix(".summary.md"))
    steps = build_steps(json_rel, pkl_rel, summary_rel, extra_pkls_rel)

    extra_note = f", extrapolate onto {len(extra_pkls_rel)} dataset(s)" if extra_pkls_rel else ""
    n_compute = sum(1 for s in steps if s["kind"] == "compute")
    log(
        f"Starting workflow on {json_rel} (pkl {pkl_rel}{extra_note}): "
        f"{len(steps)} step(s), {n_compute} driver-run compute "
        f"[{', '.join(str(s['name']) for s in steps)}]."
    )
    wf_start = time.monotonic()
    # Reuse one transport connection; run_step assigns every agent turn a fresh
    # logical SDK session so only filesystem artifacts carry state across stages.
    client_cm = ClaudeSDKClient(options=options)

    summary_path = PROJECT_ROOT / summary_rel

    async def _drive(client) -> None:
        log("SDK transport opened; agent turns use isolated sessions.")
        # Set when a repair-round remeasure finds its fix round changed no
        # script: every later step carrying skip_if_stalled is then pointless
        # (the same inputs would produce the same "nothing to fix" turn).
        repair_stalled = False
        for i, step in enumerate(steps, start=1):
            name, kind = str(step["name"]), str(step["kind"])
            print(f"\n=== Step {i}/{len(steps)}: {name} [{kind}] ===")
            log(f"Step {i}/{len(steps)} '{name}' ({kind}) started.")
            step_start = time.monotonic()
            if repair_stalled and step.get("skip_if_stalled"):
                log(f"  skipped ('{name}'): an earlier repair round changed no "
                    "script — the fix-loading loop already converged/stalled.")
                continue
            if kind == "compute":
                status = run_compute_step(step)
                if status == "stalled":
                    repair_stalled = True
                elif status == "ran" and step.get("failure_summary"):
                    _log_failure_summary(str(step["out"]), name)
            else:
                # A repair fix round with nothing left to repair has no work.
                no_unusable = step.get("skip_if_no_unusable_in")
                if no_unusable and not _has_unusable_results(str(no_unusable)):
                    log(f"  [agent] skipped ('{name}'): no UNUSABLE result in "
                        f"{no_unusable} — nothing left for a repair round.")
                    continue
                # An agent fold step whose compute input was skipped has no work.
                # Checked before skip_if_missing: an output file left by an
                # earlier run can exist while THIS run's phase produced nothing.
                no_scripts = step.get("skip_if_no_scripts_in")
                if no_scripts and not _has_hypo_scripts(str(no_scripts)):
                    log(f"  [agent] skipped ('{name}'): no hypo_*.py in "
                        f"{no_scripts} (nothing was authored this run — any "
                        "output JSON on disk belongs to an earlier run).")
                    continue
                miss = step.get("skip_if_missing")
                if miss and not (PROJECT_ROOT / str(miss)).is_file():
                    log(f"  [agent] skipped ('{name}'): {miss} absent "
                        "(its compute step was skipped — nothing to fold).")
                    continue
                cached_source = step.get("reuse_predictive_manifest_for")
                if cached_source and _predictive_manifest_is_current(
                    str(step["expects_file"]), str(cached_source)
                ):
                    log(
                        f"  [agent] reusing predictive selection manifest "
                        f"{step['expects_file']}."
                    )
                    continue
                applicability_spec = step.get("compile_split_applicability")
                if applicability_spec:
                    spec_dict = dict(applicability_spec)
                    run_for_labels = PROJECT_ROOT / str(spec_dict["run"])
                    output_for_labels = PROJECT_ROOT / str(spec_dict["output"])
                    if output_for_labels.is_file():
                        try:
                            ids, rerun_for_labels, fixed_for_labels = (
                                _resolve_split_label_inputs(run_for_labels)
                            )
                            load_applicability(
                                output_for_labels,
                                run_json=run_for_labels,
                                selected_ids=ids,
                                rerun_dir=rerun_for_labels,
                                fixed_dir=fixed_for_labels,
                                project_root=PROJECT_ROOT,
                            )
                        except SystemExit as exc:
                            log(
                                f"  [agent] existing split applicability is stale; "
                                f"refreshing: {exc}"
                            )
                        else:
                            log(
                                f"  [agent] reusing current split feature "
                                f"applicability {spec_dict['output']}."
                            )
                            continue
                    if _compile_existing_split_label_draft(spec_dict):
                        continue
                brief_spec = step.get("failure_brief")
                if brief_spec:
                    brief = _failure_brief(
                        str(brief_spec["results"]), str(brief_spec["code_dir"])
                    )
                    if brief:
                        step = {
                            **step,
                            "instruction": f"{step['instruction']}\n\n{brief}",
                        }
                final_text = await run_step(client, step)
                print(final_text.strip())
                _validate_step_output(step, summary_path)
                # Syntax gate: a broken loading fix must not cost a full
                # re-measurement run (dataset load included) to be discovered.
                syntax_spec = step.get("syntax_check")
                if syntax_spec:
                    problems = _broken_changed_scripts(
                        str(syntax_spec["code_dir"]), str(syntax_spec["results"])
                    )
                    if problems:
                        log(
                            f"  [syntax-gate] {len(problems)} changed script(s) "
                            "do not compile; requesting an immediate fix before "
                            "any re-measurement."
                        )
                        repair_step = {
                            "name": f"{name}-syntax-repair",
                            "instruction": _syntax_repair_instruction(
                                problems, str(syntax_spec["code_dir"])
                            ),
                        }
                        final_text = await run_step(client, repair_step)
                        print(final_text.strip())
                        problems = _broken_changed_scripts(
                            str(syntax_spec["code_dir"]),
                            str(syntax_spec["results"]),
                        )
                        if problems:
                            raise SystemExit(
                                "fix-loading left syntactically broken scripts "
                                "after a repair turn: " + "; ".join(problems)
                            )
                manifest_source = step.get("validate_predictive_manifest_for")
                if manifest_source:
                    _validate_predictive_manifest(
                        str(step["expects_file"]), str(manifest_source)
                    )
                    _stamp_predictive_manifest(
                        str(step["expects_file"]), str(manifest_source)
                    )
                applicability_spec = step.get("compile_split_applicability")
                if applicability_spec:
                    _compile_split_label_step(dict(applicability_spec))
            log(
                f"Step {i}/{len(steps)} '{name}' done in "
                f"{time.monotonic() - step_start:.0f}s."
            )

    async with client_cm as client:
        await _drive(client)
    log(
        f"Workflow complete in {time.monotonic() - wf_start:.0f}s; "
        f"agent cost {WORKFLOW_COSTS.describe()}."
    )
    print("\n=== Workflow complete ===")


# --- Dataset consistency guards ----------------------------------------------
# _CACHE_STEM_RE: matches the full stem (brain + mcl + add/sub) from a concrete
# dataset_cache_<stem>.pkl path.  Wildcards (*) are not word chars so glob
# patterns in experiment code are silently skipped.
# _MCL_RE / _SUFFIX_RE: extract the mcl level and add/sub suffix for granularity
# matching between --pkl and --extra-pkl.
_CACHE_STEM_RE = re.compile(r"dataset_cache_([\w]+)\.pkl")
_MCL_RE = re.compile(r"_mcl(\d+)(?:_|\.pkl)")
_SUFFIX_RE = re.compile(r"_(add|sub)\.pkl$")


def _pkl_stem(pkl_path: Path) -> str | None:
    """Full cache stem from a pkl filename.

    Returns ``'794495_mcl100_add'`` for
    ``'dataset_cache_794495_mcl100_add.pkl'``, or ``None`` if the name does not
    follow the ``dataset_cache_<stem>.pkl`` convention.
    """
    m = _CACHE_STEM_RE.search(pkl_path.name)
    return m.group(1) if m else None


def _pkl_granularity(pkl_path: Path) -> tuple[str, str] | None:
    """``(mcl_level, suffix)`` from a pkl filename for cross-dataset matching.

    Returns ``('100', 'add')`` for ``dataset_cache_794495_mcl100_add.pkl``, or
    ``None`` if either token cannot be parsed.  Extra datasets used for
    extrapolation must share both tokens with the origin ``--pkl``.
    """
    mcl_m = _MCL_RE.search(pkl_path.name)
    suf_m = _SUFFIX_RE.search(pkl_path.name)
    if mcl_m and suf_m:
        return mcl_m.group(1), suf_m.group(1)
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path",
        type=Path,
        help="The run JSON file to process (relative to the project root or absolute).",
    )
    parser.add_argument(
        "--pkl",
        type=Path,
        default=None,
        help=(
            "The ORIGIN dataset .pkl this run's experiments load, used to "
            "re-execute and reproduce the findings (relative to the project root "
            "or absolute). Required unless --smoke is used."
        ),
    )
    parser.add_argument(
        "--extra-pkl",
        type=Path,
        action="extend",
        nargs="+",
        default=[],
        metavar="PKL",
        help=(
            "Other dataset .pkl(s) to EXTRAPOLATE the findings onto. Accepts "
            "several paths after one flag and/or the flag repeated. When given, "
            "an extra step re-runs the reproduced code on each to test whether "
            "the conclusions generalize. Assumes the same payload structure as "
            "--pkl."
        ),
    )
    parser.add_argument(
        "--direction",
        required=False,
        choices=["both", "positive", "predictive"],
        help=(
            "Which hypothesis directions to include in the report. "
            "'positive' keeps only hypotheses where the experiment RAISED belief "
            "(posterior > prior by more than 0.02) — surprising confirmations only. "
            "'both' keeps positive AND negative direction hypotheses (any strong "
            "belief shift, regardless of whether belief went up or down). "
            "'predictive' starts with every hypothesis, removes only explicit "
            "invalid/constant/non-predictive/confounded exclusions, and applies no "
            "score or top-K cutoff to the remaining candidates."
        ),
    )
    parser.add_argument(
        "--split-feature-labels-only",
        action="store_true",
        help=(
            "For a completed split-error run, read existing selection/rerun/fixed "
            "artifacts and generate the source-bound split feature applicability "
            "JSON only. Does not require --pkl or rerun experiments."
        ),
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Print each selected hypothesis as 'ID <id>: <one-line hypothesis>' "
            "using the current direction and top-K policy, then stop. Does not "
            "require --pkl or run later workflow phases. Predictive smoke creates "
            "or reuses <RUN>.predictive-selection.json beside the run export."
        ),
    )
    parser.add_argument(
        "--log-txt",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Path for this driver's captured console log. Default: "
            f"<RUN>.json[.predictive]{DRIVER_LOG_SUFFIX} beside the run export. "
            "Everything printed — the dataset guards, step progress, each "
            "subagent tool call, every step's final reply, the compute "
            "subprocesses' live progress, validation WARNINGs, and any fatal "
            "diagnostic — is tee'd there as well as to the terminal, flushed as "
            "it goes, so a workflow that dies mid-step still leaves a record. "
            "Pass /dev/null to skip. Ignored with --smoke, whose output is "
            "already persisted in the predictive-selection manifest."
        ),
    )
    parser.add_argument(
        "--yes-stale",
        action="store_true",
        help=(
            "Proceed non-interactively after printing the prior/stale artifact "
            "survey. Use only after reviewing the listed files; without this "
            "flag, findings require an interactive y/n confirmation."
        ),
    )
    args = parser.parse_args()

    def _resolve_existing(p: Path, what: str) -> Path:
        rp = p if p.is_absolute() else (PROJECT_ROOT / p).resolve()
        if not rp.is_file():
            parser.error(f"No such {what}: {rp}")
        return rp

    json_path = _resolve_existing(args.path, "run JSON file")
    if args.direction is None and not args.split_feature_labels_only:
        parser.error("--direction is required unless --split-feature-labels-only is used.")
    if args.split_feature_labels_only and args.smoke:
        parser.error("--split-feature-labels-only cannot be combined with --smoke.")
    if args.split_feature_labels_only and (args.pkl is not None or args.extra_pkl):
        parser.error(
            "--split-feature-labels-only does not accept --pkl or --extra-pkl."
        )
    if args.smoke and args.extra_pkl:
        parser.error("--smoke cannot be combined with --extra-pkl.")
    if not args.smoke and not args.split_feature_labels_only and args.pkl is None:
        parser.error("--pkl is required unless --smoke is used.")
    pkl_path = (
        _resolve_existing(args.pkl, "dataset pkl")
        if args.pkl is not None
        else None
    )
    extra_pkl_paths = (
        [_resolve_existing(p, "extra dataset pkl") for p in args.extra_pkl]
        if not args.smoke
        else []
    )

    global DIRECTION
    DIRECTION = args.direction or "predictive"

    if args.split_feature_labels_only:
        asyncio.run(run_split_feature_labeling_only(json_path))
        return 0

    def _announce_selection() -> list:
        """Print the direction, resolve TOP_K, and return the parsed records.

        Called INSIDE the driver-log block for a full run so these lines — the
        policy the whole run is scored under — are the first thing in the log.
        """
        direction_labels = {
            "positive": "positive only (posterior > prior)",
            "both": "both positive and negative",
            "predictive": "exclusion-only predictive candidates (no cutoff)",
        }
        print(
            f"[direction] {direction_labels[DIRECTION]}",
            file=sys.stderr,
        )

        # Positive/both use the usual percentage-based top-K. Predictive mode keeps
        # every semantically selected candidate. This is resolved once here so all
        # subsequent helpers and agent instructions agree.
        global TOP_K
        try:
            _raw = json.loads(json_path.read_text())
            records = (
                _raw if isinstance(_raw, list)
                else _raw.get("records", []) if isinstance(_raw, dict)
                else []
            )
            n_hypo = sum(1 for r in records if isinstance(r, dict))
        except Exception:
            records = []
            n_hypo = 0
        # Predictive is exclusion-only: every non-excluded feature continues. The
        # other modes retain the existing percentage cap.
        TOP_K = None if DIRECTION == "predictive" else _compute_top_k(n_hypo)
        _pct_val = int(n_hypo * TOP_K_PCT) if n_hypo > 0 else 0
        if TOP_K is None:
            print(
                f"[selection] {n_hypo} hypotheses → keep every candidate not "
                "explicitly excluded (no score/top-K cutoff)",
                file=sys.stderr,
            )
        else:
            print(
                f"[top-k] {n_hypo} hypotheses → TOP_K={TOP_K} "
                f"(min({TOP_K_MAX}, floor({n_hypo}×{TOP_K_PCT:.0%}))="
                f"min({TOP_K_MAX},{_pct_val})={TOP_K})",
                file=sys.stderr,
            )

        # Per-step budgets follow the record count, not a fixed top-K.
        n_budget = n_hypo if TOP_K is None else TOP_K
        agent_s, compute_s = _scale_timeouts(n_budget)
        print(
            f"[timeouts] {n_budget} record(s) → agent step {agent_s}s, "
            f"compute step {compute_s}s",
            file=sys.stderr,
        )
        return records

    if args.smoke:
        # Smoke is a print-and-stop: its selection is already persisted in the
        # predictive-selection manifest, so it gets no driver log.
        _announce_selection()
        asyncio.run(run_smoke(json_path))
        return 0

    assert pkl_path is not None  # enforced above unless smoke returned

    prior_artifacts = survey_prior_artifacts(json_path, DIRECTION)
    confirm_prior_artifacts(prior_artifacts, assume_yes=args.yes_stale)

    # A full run is hours long across several agent turns and compute
    # subprocesses, so everything from here on is tee'd to a file: the dataset
    # guards (including a fatal mismatch), step progress, each step's final reply,
    # the compute children's live output, and validation WARNINGs.
    log_txt = args.log_txt or driver_log_path(json_path)
    log_txt = log_txt if log_txt.is_absolute() else (PROJECT_ROOT / log_txt).resolve()

    with driver_log(log_txt):
        _records = _announce_selection()

        # GUARD 1: --pkl full stem must match the dominant dataset_cache_<stem>.pkl
        # reference in the run JSON.  The stem encodes brain id, mcl level, and
        # add/sub suffix, so a mismatch means every reproduced number is on wrong data.
        # Reuse the already-parsed records instead of reading the JSON a second time.
        hits: Counter = Counter()
        for _it in _records:
            if not isinstance(_it, dict):
                continue
            _blob = f"{_it.get('code', '')}\n{_it.get('codeOutput', '')}"
            for _m in _CACHE_STEM_RE.finditer(_blob):
                hits[_m.group(1)] += 1
        json_stem = hits.most_common(1)[0][0] if hits else None
        json_tally = dict(hits)
        pkl_stem_val = _pkl_stem(pkl_path)
        if json_stem is None:
            parser.error(
                f"No concrete dataset_cache_<stem>.pkl reference found in "
                f"{json_path.name}. Cannot verify --pkl matches the origin dataset.")
        if pkl_stem_val is None:
            parser.error(
                f"--pkl {pkl_path.name} does not follow the "
                f"dataset_cache_<stem>.pkl naming convention.")
        if pkl_stem_val != json_stem:
            parser.error(
                f"Dataset MISMATCH: the run JSON references "
                f"dataset_cache_{json_stem}.pkl (tally: {json_tally}) but --pkl is "
                f"dataset_cache_{pkl_stem_val}.pkl. Pass "
                f"--pkl dataset_cache_{json_stem}.pkl.")
        print(f"[dataset-check] OK: --pkl matches JSON origin ({pkl_stem_val})",
              file=sys.stderr)

        # GUARD 2: --extra-pkl files must share the same mcl level and add/sub suffix
        # as --pkl.  Extra datasets are expected to be DIFFERENT brains (the
        # extrapolation set), but must use identical processing parameters.
        if extra_pkl_paths:
            pkl_gran = _pkl_granularity(pkl_path)
            if pkl_gran is None:
                parser.error(
                    f"Could not parse mcl level / add|sub suffix from "
                    f"--pkl {pkl_path.name}.")
            for ep in extra_pkl_paths:
                ep_gran = _pkl_granularity(ep)
                if ep_gran is None:
                    parser.error(
                        f"Could not parse mcl level / add|sub suffix from "
                        f"--extra-pkl {ep.name}.")
                if ep_gran != pkl_gran:
                    mcl_p, suf_p = pkl_gran
                    mcl_e, suf_e = ep_gran
                    parser.error(
                        f"Granularity MISMATCH: --pkl uses mcl{mcl_p}_{suf_p} but "
                        f"--extra-pkl {ep.name} uses mcl{mcl_e}_{suf_e}. "
                        f"Extra datasets must share the same mcl level and suffix.")
            mcl_ok, suf_ok = pkl_gran
            print(
                f"[granularity-check] OK: all --extra-pkl files match --pkl "
                f"(mcl{mcl_ok}_{suf_ok})",
                file=sys.stderr,
            )

        asyncio.run(
            run_workflow(
                json_path,
                pkl_path,
                extra_pkl_paths,
            )
        )

    log(f"Driver console log → {_rel_to_root(log_txt)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
