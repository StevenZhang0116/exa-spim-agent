"""
Run the AutoDiscovery summarization workflow via the Claude Agent SDK.

Processes ONE AutoDiscovery run export (a single JSON file) together with the
dataset ``.pkl`` its experiments were run against, and produces a Markdown
deliverable next to the input: ``<stem>.summary.md`` (plus a 简体中文
``<stem>.summary.zh.md``). Subagents live in ``.claude/agents/`` and are
auto-discovered via ``setting_sources``.

Architecture — driver-owned compute, agents only read + fold
------------------------------------------------------------
The workflow is a sequence of typed STEPS of two kinds, so a long re-execution
NEVER runs inside an agent turn (the failure that previously wedged the run —
an hour-long ``rerun_experiments.py`` killed at turn-end / lost to ``nohup &``):

  * compute step — the DRIVER runs ``rerun_experiments.py`` as a blocking
    foreground subprocess and captures its stdout JSON to a file next to the
    export (``<stem>.json.reproduce.json`` / ``.extrapolate.json`` /
    ``.corrected.json``). Deterministic, observable, resumable; a non-zero exit
    aborts loudly before any agent folds a bad result.
  * agent step — an instruction to the persistent SDK session; the subagent only
    READS the compute JSON already on disk and folds verdicts into the report
    (or authors corrected scripts). Always fast and turn-safe.

The PHASES (the unit ``--steps`` / ``--from`` select), in order:
  1. summarize   — [agent] rank the hypotheses and write the top-K report.
  2. reproduce   — [compute] run recorded code on ``--pkl`` → [agent] fix only
                   data-loading failures → [compute] re-measure the fixed code →
                   [agent] fold REPRODUCED/DIVERGED/FAILED verdicts. (The
                   re-measure is skipped when no loading fix was needed.)
  3. extrapolate — (only with ``--extra-pkl``) [compute] run the reproduced code
                   on each OTHER dataset → [agent] fold GENERALIZES/PARTIAL/
                   DOES-NOT-GENERALIZE/INCONCLUSIVE verdicts.
  4. verify      — [agent] audit statistics/logic from recorded + reproduced
                   results; assign SOUND/WEAK/MINOR/MAJOR/CRITICAL verdicts.
  5. fix-tests   — [agent] author corrected scripts for flagged tests →
                   [compute] re-measure them (+extras) → [agent] fold
                   UPHELD/WEAKENED/OVERTURNED. (Compute+fold skipped if nothing
                   was flagged.)
  6. translate   — [agent] faithful 简体中文 localization of the finished report.
                   Pure localization (no re-analysis), so it ALWAYS runs last.

Two free parameters near the top control what the report contains:
``RANK_BY`` (``"posterior-surprise"`` ranks by ``posterior * |surprisal|`` so
findings that are both strongly believed and highly belief-shifting come first;
``"surprise"`` ranks by ``|surprisal|`` alone) and ``TOP_K`` (keep only the K
top-ranked hypotheses in the final Markdown; ``None`` keeps all).

Usage (from the ``exa-spim-agent/`` project root):
    python agentic/run_discovery_workflow.py autodiscovery/RUN.json --pkl ../data/RUN.pkl
    python agentic/run_discovery_workflow.py autodiscovery/RUN.json --pkl DATA.pkl --verbose
    # Also test generalization onto other datasets:
    python agentic/run_discovery_workflow.py autodiscovery/RUN.json --pkl ORIGIN.pkl \
        --extra-pkl OTHER1.pkl --extra-pkl OTHER2.pkl
    # Inspect the resolved plan (which phases are compute vs agent) without running:
    python agentic/run_discovery_workflow.py autodiscovery/RUN.json --pkl ORIGIN.pkl \
        --extra-pkl OTHER1.pkl --plan
    # Resume / run a subset by PHASE, reusing earlier phases' on-disk artifacts
    # (<RUN>.summary.md, <RUN>.json.rerun, the compute JSONs) — e.g. the corrective phase:
    python agentic/run_discovery_workflow.py autodiscovery/RUN.json --pkl ORIGIN.pkl \
        --steps fix-tests
    python agentic/run_discovery_workflow.py autodiscovery/RUN.json --pkl ORIGIN.pkl \
        --from verify          # run verify and everything after it
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

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

# --- Free parameters for the summarization step ------------------------------
# TOP_K: how many top-ranked hypotheses to keep in the final Markdown report.
#   Only these K entries are written out (and later verified). Set to None to
#   keep every ranked hypothesis.
# RANK_BY: which deterministic ordering rank_by_surprise.py uses.
#   "posterior-surprise" ranks by posterior * |surprisal| so hypotheses that
#   are BOTH strongly believed true AND highly belief-shifting come first;
#   "surprise" ranks by |surprisal| alone.
TOP_K: int | None = 20
RANK_BY: str = "posterior-surprise"

_TOP_K_PHRASE = (
    f"the top {TOP_K} hypotheses" if TOP_K is not None else "all ranked hypotheses"
)


def _top_flags() -> str:
    """The shared ``--rank-by``/``--top`` flags both helpers must agree on."""
    return f"--rank-by {RANK_BY}" + (f" --top {TOP_K}" if TOP_K is not None else "")


def rank_cmd(json_rel: str, *, include_code: bool = False) -> str:
    """Deterministic ranking command for ONE run export.

    Built from the free parameters above so the helper, the report, and the
    verifier stay in sync. ``json_rel`` is the run file path relative to the
    project root, so the helper ranks only that single file. With
    ``include_code`` the helper also emits each returned record's truncated
    ``code`` / ``codeOutput`` so the verifier can audit the top-K SLICE instead
    of loading the full multi-MB export (~6x less to read for TOP_K=20).
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
) -> list[str]:
    """Build the ``rerun_experiments.py`` argv for ONE run export + dataset pkl.

    The DRIVER (not an agent) runs this as a blocking foreground subprocess, so
    no long re-execution ever lives inside an agent turn. All compute phases —
    reproduce, extrapolate, and the corrected-test re-measurement — go through
    this one builder so they always agree on ``--rank-by``/``--top``:

    - ``export_dir_rel``  : export editable hypo_<id>.py scripts and exit (no run).
    - ``code_dir_rel``    : run the loading-fixed scripts instead of recorded code.
    - ``corrected_dir_rel``: run the test-fixer's corrected scripts (takes
      precedence per record over ``--code-dir``).
    - ``extra_pkls_rel``  : also run each record on these OTHER datasets (the
      ``extrapolations`` list), to judge generalization.
    """
    argv = ["python", "agentic/rerun_experiments.py", json_rel,
            "--pkl", pkl_rel, "--rank-by", RANK_BY]
    if TOP_K is not None:
        argv += ["--top", str(TOP_K)]
    # Export mode is mutually exclusive with running (the helper enforces this).
    if export_dir_rel is not None:
        return argv + ["--export-dir", export_dir_rel]
    if code_dir_rel is not None:
        argv += ["--code-dir", code_dir_rel]
    if corrected_dir_rel is not None:
        argv += ["--corrected-dir", corrected_dir_rel]
    for p in extra_pkls_rel or []:
        argv += ["--extra-pkl", p]
    return argv


def run_compute(argv: list[str], out_rel: str | None, verbose: bool) -> None:
    """Run a deterministic helper subprocess to completion in the FOREGROUND.

    This is the heart of the driver-owned design: the long ``rerun_experiments.py``
    invocations run here, as a blocking child of the driver, NOT inside an agent
    turn — so nothing is ever killed at turn-end or lost to a ``nohup &`` detach.
    The child's stdout (its JSON payload) is captured to ``out_rel`` when given;
    its stderr (the live ``[rerun k/N]`` progress) streams straight through so a
    long run stays observable. A non-zero exit aborts the workflow loudly rather
    than letting a later agent step fold a half-written or empty JSON.
    """
    pretty = " ".join(shlex.quote(a) for a in argv)
    log(f"  [compute] {pretty}")
    if out_rel is not None:
        log(f"  [compute] → stdout JSON to {out_rel}")
    start = time.monotonic()
    out_fh = open(PROJECT_ROOT / out_rel, "w") if out_rel is not None else None
    try:
        # cwd=PROJECT_ROOT so the relative paths in argv resolve; stderr inherits
        # this process's stderr so progress is visible live; stdout → the file.
        proc = subprocess.run(
            argv,
            cwd=str(PROJECT_ROOT),
            stdout=out_fh if out_fh is not None else None,
            check=False,
        )
    finally:
        if out_fh is not None:
            out_fh.close()
    dur = time.monotonic() - start
    if proc.returncode != 0:
        # Drop a non-empty hint into the (now-suspect) output file's neighbour so
        # a resume can tell the JSON is not trustworthy, then abort.
        raise SystemExit(
            f"Compute step FAILED (exit {proc.returncode}) after {dur:.0f}s: {pretty}. "
            "Aborting before any agent folds a bad result. Fix the cause and resume "
            "with --from on this phase."
        )
    log(f"  [compute] done in {dur:.0f}s.")


def zh_path_for(summary_rel: str) -> str:
    """The 简体中文 sibling path for a report: insert ``.zh`` before ``.md``.

    ``foo.summary.md`` -> ``foo.summary.zh.md``. Used by the final translate
    step so its output sits next to the English report.
    """
    if summary_rel.endswith(".md"):
        return summary_rel[: -len(".md")] + ".zh.md"
    return summary_rel + ".zh.md"


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
      subagent only READS the compute JSON already on disk and folds verdicts
      into the Markdown report (or authors corrected scripts). Always fast.

    Each step carries a ``"phase"`` (summarize / reproduce / extrapolate /
    verify / fix-tests / translate) — the unit ``--steps`` / ``--from`` select
    on. ``json_rel`` is the run JSON to digest, ``pkl_rel`` the origin dataset,
    ``summary_rel`` the Markdown deliverable — all relative to the project root.
    Order: summarize → reproduce (run→fix-loading→remeasure→fold) →
    [extrapolate (run→fold), only with ``extra_pkls_rel``] → verify →
    fix-tests (author→measure→fold) → translate. Translate is purely cosmetic and
    always runs last.
    """
    extra_pkls_rel = extra_pkls_rel or []
    rerun_dir_rel = f"{json_rel}.rerun"
    fixed_dir_rel = f"{json_rel}.fixed"
    # Compute-output JSONs live next to the run export so they survive across
    # runs and a resume can reuse the expensive ones instead of recomputing.
    repro_raw_rel = f"{json_rel}.reproduce-raw.json"   # recorded code on origin
    repro_rel = f"{json_rel}.reproduce.json"           # loading-fixed code on origin
    extrap_rel = f"{json_rel}.extrapolate.json"        # +extra datasets
    corrected_rel = f"{json_rel}.corrected.json"       # corrected tests +extras

    steps: list[dict[str, object]] = [
        {
            "name": "summarize-discoveries",
            "phase": "summarize",
            "kind": "agent",
            "instruction": (
                "Use the discovery-summarizer subagent to digest the single "
                f"AutoDiscovery run export at {json_rel}. It must run "
                f"`{rank_cmd(json_rel)}` to rank that file's hypotheses by the "
                "combined posterior-and-surprise priority (posterior * "
                f"|surprisal|), keeping only {_TOP_K_PHRASE}. The report MUST "
                f"contain ONLY those top {TOP_K if TOP_K is not None else 'N'} "
                "records (do NOT add entries beyond what the helper returns). "
                f"Write the ranked report to {summary_rel}, ordered by the "
                "helper's ranking (highest priority first), and display each "
                "entry's priority_score alongside its surprise magnitude. Report "
                "the path it wrote and a short executive summary of the "
                "highest-priority conclusions (high posterior and high surprise)."
            ),
        },
        # --- Reproduce phase: run recorded code, fix loading, re-measure, fold ---
        {
            "name": "reproduce-run",
            "phase": "reproduce",
            "kind": "compute",
            "argv": rerun_argv(json_rel, pkl_rel),
            "out": repro_raw_rel,
        },
        {
            "name": "reproduce-fix-loading",
            "phase": "reproduce",
            "kind": "agent",
            "instruction": (
                "Use the discovery-reproducer subagent to FIX ONLY data-loading / "
                "environment failures so the experiments can run — do NOT re-measure "
                "or fold verdicts yet (a later step does that). The driver has "
                f"already executed each top-ranked record's recorded `code` on "
                f"{pkl_rel}; read the results JSON it wrote at {repro_raw_rel} "
                "(top-level counts + a `results` list; each has `id`, `status`, "
                "`rerun_exitcode`, `rerun_stdout`, `rerun_stderr`). For every result "
                "that FAILED or TIMED OUT for a DATA-LOADING / ENVIRONMENT reason "
                "(a NumPy-2-written pkl the host can't unpickle, a 'dataset not "
                "found' gate, a pip-install retry loop), export the editable scripts "
                f"with `python agentic/rerun_experiments.py {json_rel} --pkl "
                f"{pkl_rel} {_top_flags()} --export-dir {rerun_dir_rel}` (this only "
                "writes files, it does not run anything), then revise ONLY the "
                f"loading/bootstrap of the failing {rerun_dir_rel}/hypo_<id>.py "
                "(load the pkl directly from $RERUN_PKL, fix the NumPy version, drop "
                "pip-retry loops) while keeping the ANALYSIS byte-for-byte identical. "
                "Do NOT touch scripts that already succeeded. If NOTHING failed for "
                f"a loading reason, leave {rerun_dir_rel} empty and say so — the "
                "driver will then reuse the recorded-code results directly. Report "
                "which ids you fixed and why."
            ),
        },
        {
            "name": "reproduce-remeasure",
            "phase": "reproduce",
            "kind": "compute",
            "argv": rerun_argv(json_rel, pkl_rel, code_dir_rel=rerun_dir_rel),
            "out": repro_rel,
            # If the agent wrote no loading-fix scripts, the --code-dir run would
            # equal the recorded-code run, so skip it and reuse the raw JSON.
            "skip_if_no_scripts_in": rerun_dir_rel,
            "reuse_out_from": repro_raw_rel,
        },
        {
            "name": "reproduce-fold",
            "phase": "reproduce",
            "kind": "agent",
            "instruction": (
                "Use the discovery-reproducer subagent to fold the reproduction "
                "verdicts into the report. The driver has produced the final "
                f"reproduction results JSON at {repro_rel} (each result marks its "
                "`code_source` as recorded|revised and carries `rerun_stdout` plus "
                "the recorded `recorded_output`). Do NOT re-run anything. For each "
                "record compare the fresh key numbers (test statistic, p-value, "
                "effect size, n) to the recorded ones and assign a REPRODUCED / "
                f"DIVERGED / FAILED verdict. Read {summary_rel} and fold the result "
                "INTO each hypothesis's existing ranked entry (append Reproduction / "
                "Rerun result bullets in place, keeping prior bullets; note whether "
                "the code was recorded or revised-loading), plus one "
                "'Reproduction — Summary' section. Report the path and which "
                "findings did NOT reproduce."
            ),
        },
    ]

    # Optional extrapolation phase: only when other datasets were provided. The
    # driver runs the reproduced code (the reviser's --code-dir) on each extra pkl;
    # the agent only folds the generalization verdicts.
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
                ),
                "out": extrap_rel,
            }
        )
        steps.append(
            {
                "name": "extrapolate-fold",
                "phase": "extrapolate",
                "kind": "agent",
                "instruction": (
                    "Use the discovery-extrapolator subagent to fold GENERALIZATION "
                    "verdicts into the report. The driver has already re-run each "
                    "reported finding's reproduced code on the extra datasets it was "
                    f"NOT generated on ({extra_list}); read the results JSON at "
                    f"{extrap_rel}. Each result has the origin fields PLUS an "
                    "`extrapolations` list — one entry per extra pkl with `pkl`, "
                    "`pkl_name`, `exitcode`, `timed_out`, `runtime_ms`, `stdout`, "
                    "`stderr`. Do NOT re-run anything. For each hypothesis compare "
                    "the origin numbers to each extra dataset's numbers (direction, "
                    "significance, effect size) and assign GENERALIZES / PARTIAL / "
                    "DOES-NOT-GENERALIZE / INCONCLUSIVE (INCONCLUSIVE when a script "
                    f"could not run on an extra pkl). Read {summary_rel} and fold "
                    "the result INTO each hypothesis's existing ranked entry (append "
                    "Generalization / Across datasets bullets in place, keeping prior "
                    "bullets), plus one 'Generalization — Summary' section placed "
                    "AFTER 'Reproduction — Summary' and BEFORE any 'Statistical "
                    "Verification' section. Report the path and which findings do "
                    "NOT generalize."
                ),
            }
        )

    steps.append(
        {
            "name": "verify-statistics-and-logic",
            "phase": "verify",
            "kind": "agent",
            "instruction": (
                "Use the discovery-verifier subagent to audit whether each "
                "hypothesis's statistical test and its inductive/deductive "
                "reasoning are correct. For the recorded code, codeOutput and "
                f"analysis, run `{rank_cmd(json_rel, include_code=True)}` and read "
                "its stdout JSON — that already contains ONLY the reported top "
                f"{TOP_K if TOP_K is not None else 'N'} records, each with the "
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
                "MAJOR/CRITICAL and concrete test-fault bullets). Place one "
                "file-wide 'Statistical Verification — Summary' section AFTER the "
                "'Generalization — Summary' section (when present) and BEFORE any "
                "'Excluded' section; include the verdict breakdown and a "
                "Benjamini–Hochberg FDR subsection. Report the path and the most "
                "serious problems found."
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
                f"script {rerun_dir_rel}/hypo_<id>.py (if it has none, export it "
                f"first with `python agentic/rerun_experiments.py {json_rel} --pkl "
                f"{pkl_rel} {_top_flags()} --export-dir {rerun_dir_rel}` and apply "
                "the same loading fix) and write a corrected hypo_<id>.py into "
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
            ),
            "out": corrected_rel,
            # No corrected scripts → nothing to re-measure; skip the run entirely.
            "skip_if_no_scripts_in": fixed_dir_rel,
        }
    )
    steps.append(
        {
            "name": "fix-tests-fold",
            "phase": "fix-tests",
            "kind": "agent",
            # Folding is a no-op when the measure step was skipped (no corrected
            # JSON on disk); the driver detects that and skips this step too.
            "skip_if_missing": corrected_rel,
            "instruction": (
                "Use the discovery-test-fixer subagent to fold CORRECTED-test "
                "verdicts into the report. The driver has re-measured the corrected "
                f"scripts; read the results JSON at {corrected_rel} and use the "
                f"`code_source: corrected` results{corrected_note} Do NOT re-run "
                "anything. For each corrected hypothesis assign UPHELD / WEAKENED / "
                "OVERTURNED by comparing the corrected numbers (statistic, p, effect "
                "size + CI) to the original"
                + (
                    ", plus a corrected-test generalization call (does the finding "
                    "still hold across the extra datasets under the right test?)"
                    if extra_pkls_rel
                    else ""
                )
                + f". Read {summary_rel} and fold Corrected test / Corrected result "
                "/ Post-correction verdict"
                + (" / Corrected generalization" if extra_pkls_rel else "")
                + " bullets INTO each fixed entry in place (keep all prior bullets), "
                "plus one 'Statistical Test Corrections — Summary' section at the "
                "very end. Report the path and which findings changed under the "
                "correct test."
            ),
        }
    )

    # Final, purely-cosmetic step: translate the finished report into 简体中文.
    # It reads only the completed Markdown deliverable and writes a sibling
    # .zh.md, so it has NO analytical dependency and must always run LAST — after
    # every step above has folded its verdicts into the report. If more
    # analytical steps are added, append them BEFORE this block so translate
    # stays the final step.
    summary_zh_rel = zh_path_for(summary_rel)
    steps.append(
        {
            "name": "translate-report",
            "phase": "translate",
            "kind": "agent",
            "instruction": (
                "Use the discovery-translator subagent to produce a faithful "
                f"简体中文 translation of the finished report at {summary_rel}, "
                f"writing it to {summary_zh_rel}. This is a pure localization pass: "
                "translate the prose and section/field labels but keep the EXACT "
                "same structure, ordering, facts and verdicts, and copy ALL numbers "
                "(p-values, coefficients, counts, ratios, CIs, scores, belief "
                "probabilities) verbatim. Keep verbatim (do NOT translate) every "
                "identifier, hypothesis id, run/dataset id, file path, code span, "
                "metric/test name, unit, and verdict token (REPRODUCED, DIVERGED, "
                "GENERALIZES, DOES-NOT-GENERALIZE, PARTIAL, OK, MINOR, MAJOR, "
                "CRITICAL, UPHELD, WEAKENED, OVERTURNED). Do NOT re-run or re-judge "
                "anything. Write exactly that one file and report its path."
            ),
        }
    )
    return steps


# The selectable unit for --steps / --from is the PHASE (a phase may expand into
# several compute+agent steps internally). These are the phase tokens, in order.
# ``rerun`` is kept as a back-compat alias for the renamed ``reproduce`` phase.
PHASE_ORDER: list[str] = [
    "summarize", "reproduce", "extrapolate", "verify", "fix-tests", "translate",
]
PHASE_ALIASES: dict[str, str] = {"rerun": "reproduce"}


def _canonical_phase(token: str, available: list[str]) -> str:
    """Resolve a user token (phase name or alias) to a phase present in this run."""
    phase = PHASE_ALIASES.get(token, token)
    if phase in available:
        return phase
    raise SystemExit(
        f"Unknown step/phase '{token}'. Choose from: {', '.join(available)} "
        f"(alias: {', '.join(f'{k}→{v}' for k, v in PHASE_ALIASES.items())})."
    )


def select_steps(
    steps: list[dict[str, object]],
    only: list[str] | None,
    from_step: str | None,
) -> list[dict[str, object]]:
    """Filter the built steps by ``--steps`` / ``--from``, selecting whole PHASES.

    Selection is by phase, not by the internal compute/agent sub-steps: keeping a
    phase keeps all of its steps in order. Resuming relies on earlier phases'
    artifacts already being on disk — the ``<RUN>.summary.md`` report (with its
    folded-in verdicts), the ``<RUN>.json.rerun`` loading-fixed scripts, and the
    compute-output JSONs (``<RUN>.json.reproduce.json`` etc.). ``--steps`` keeps
    exactly the named phases (in pipeline order); ``--from`` keeps that phase and
    everything after it. They are mutually exclusive; with neither, all run.
    """
    # Phases present in THIS run, in pipeline order (extrapolate may be absent).
    present: list[str] = []
    for s in steps:
        ph = str(s["phase"])
        if ph not in present:
            present.append(ph)
    if only and from_step:
        raise SystemExit("--steps and --from are mutually exclusive.")
    if only:
        wanted = {_canonical_phase(t, present) for t in only}
        return [s for s in steps if str(s["phase"]) in wanted]
    if from_step:
        start = _canonical_phase(from_step, present)
        start_idx = present.index(start)
        keep = set(present[start_idx:])
        return [s for s in steps if str(s["phase"]) in keep]
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
        # cheaper model in their frontmatter (e.g. the translator → sonnet) override it.
        model="claude-opus-4-8",
        # Default reasoning effort for the ORCHESTRATOR turns (levels:
        # low|medium|high|xhigh|max). Per-subagent effort is set in each
        # .claude/agents/*.md frontmatter and takes precedence for that subagent's
        # turns — the mechanical fold agents run at medium, the analytical ones
        # (summarize/verify/test-fixer) at xhigh, the translator at low. We set the
        # session default via the SDK's native `effort` field, NOT the old
        # `CLAUDE_EFFORT` env var — that var is an OUTPUT the CLI exports to hooks
        # to report the active level, not an input it reads, so setting it did
        # nothing. (The env var that WOULD override is CLAUDE_CODE_EFFORT_LEVEL,
        # which we deliberately do not set so frontmatter effort wins.)
        effort="xhigh",
    )


async def run_step(client: ClaudeSDKClient, step: dict[str, str], verbose: bool) -> str:
    """Send one workflow step to the session and return its final text.

    Logs live progress (each tool call / subagent launch) to stderr so a
    long-running step is observable, streams assistant text as it arrives (when
    ``verbose``), and returns the concatenated assistant text for the step so a
    caller could chain on it.
    """
    await client.query(step["instruction"])

    chunks: list[str] = []
    n_tools = 0
    async for message in client.receive_response():
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    chunks.append(block.text)
                    if verbose:
                        print(block.text, end="", flush=True)
                elif ToolUseBlock and isinstance(block, ToolUseBlock):
                    n_tools += 1
                    log(f"  → {describe_tool(block)}")
        elif isinstance(message, ResultMessage):
            # End of this turn. Surface timing / cost / token usage when present.
            if verbose:
                print()  # newline after the streamed text
            cost = getattr(message, "total_cost_usd", None)
            dur_ms = getattr(message, "duration_ms", None)
            parts = [f"{n_tools} tool call(s)"]
            if dur_ms is not None:
                parts.append(f"{dur_ms / 1000:.0f}s")
            if cost is not None:
                parts.append(f"${cost:.4f}")
            log(f"  step turn finished — {', '.join(parts)}")
    return "".join(chunks)


def _rel_to_root(path: Path) -> str:
    """Path relative to PROJECT_ROOT (the session cwd), tolerating ``..``.

    The dataset pkl often lives outside the project (e.g. ``../data/``), so a
    plain ``relative_to`` would raise — ``os.path.relpath`` handles parent dirs.
    """
    return Path(os.path.relpath(path, PROJECT_ROOT)).as_posix()


def _has_hypo_scripts(dir_rel: str) -> bool:
    """True if ``dir_rel`` holds at least one ``hypo_<id>.py`` the agent wrote.

    The loading-fix and test-fix agent steps may legitimately write NOTHING (no
    failure to fix / no test to correct). The following compute step then has no
    work to do, so the driver uses this to skip the redundant re-run.
    """
    d = PROJECT_ROOT / dir_rel
    return d.is_dir() and any(d.glob("hypo_*.py"))


def _valid_compute_json(out_rel: str) -> bool:
    """True if ``out_rel`` already holds a usable ``rerun_experiments.py`` payload.

    Used to REUSE an expensive compute result on resume instead of recomputing.
    A previous run can leave a half-written / empty / non-JSON file behind (e.g.
    a killed process) — exactly what cost an hour last time — so a bare
    ``exists()`` is not enough. We require the file to parse as JSON and carry a
    non-empty ``results`` list, the contract every downstream fold step relies on.
    """
    p = PROJECT_ROOT / out_rel
    if not p.is_file() or p.stat().st_size == 0:
        return False
    try:
        payload = json.loads(p.read_text())
    except (ValueError, OSError):
        return False
    return bool(isinstance(payload, dict) and payload.get("results"))


async def run_compute_step(
    step: dict[str, object], verbose: bool, force: bool = False
) -> None:
    """Execute one ``kind == "compute"`` step in the driver (no agent involved).

    Honors skip rules so a resume reuses expensive results and the common path
    stays cheap. Checked in order:
    - reuse-if-output-exists: if this step's ``out`` JSON is already on disk and
      VALID (parses, non-empty ``results``), reuse it instead of recomputing —
      this is what makes ``--from extrapolate`` cheap when ``.extrapolate.json``
      already exists from a prior run. ``--force`` (``force=True``) overrides it.
    - ``skip_if_no_scripts_in``: if the preceding agent wrote no hypo_<id>.py
      there, this run would duplicate an earlier one — skip it, and when
      ``reuse_out_from`` is given, copy that earlier JSON to this step's ``out``
      so the downstream fold step finds its expected input.
    """
    out_rel = step.get("out")  # type: ignore[assignment]
    # 1) Reuse a valid existing output (resume without recomputing).
    if out_rel and not force and _valid_compute_json(str(out_rel)):
        log(f"  [compute] reusing existing {out_rel} "
            "(valid; pass --force to recompute).")
        return
    # 2) Nothing for the preceding agent to have fixed/corrected → skip the run.
    skip_dir = step.get("skip_if_no_scripts_in")
    if skip_dir and not _has_hypo_scripts(str(skip_dir)):
        reuse = step.get("reuse_out_from")
        if reuse and out_rel:
            shutil.copyfile(PROJECT_ROOT / str(reuse), PROJECT_ROOT / str(out_rel))
            log(f"  [compute] skipped (no scripts in {skip_dir}); "
                f"reused {reuse} → {out_rel}.")
        else:
            log(f"  [compute] skipped (no scripts in {skip_dir}); "
                "nothing to re-measure.")
        return
    run_compute(list(step["argv"]), str(out_rel) if out_rel else None, verbose)


def print_plan(steps: list[dict[str, object]]) -> None:
    """Print the resolved step list (kind + phase + command/output) and exit.

    Lets you see exactly what the driver will run — which phases are compute vs
    agent and which long re-executions happen — without starting anything.
    """
    print(f"Resolved plan — {len(steps)} step(s):")
    for i, s in enumerate(steps, start=1):
        kind = str(s["kind"])
        if kind == "compute":
            cmd = " ".join(shlex.quote(a) for a in s["argv"])  # type: ignore[arg-type]
            extra = f"  → {s['out']}" if s.get("out") else ""
            skip = f"  [skip if no scripts in {s['skip_if_no_scripts_in']}]" if s.get("skip_if_no_scripts_in") else ""
            reuse = "  [reuse if output exists, unless --force]" if s.get("out") else ""
            print(f"  {i:2}. [{s['phase']}/compute] {s['name']}")
            print(f"        $ {cmd}{extra}{skip}{reuse}")
        else:
            skip = f"  [skip if missing {s['skip_if_missing']}]" if s.get("skip_if_missing") else ""
            print(f"  {i:2}. [{s['phase']}/agent]   {s['name']}{skip}")


async def run_workflow(
    json_path: Path,
    pkl_path: Path,
    extra_pkl_paths: list[Path],
    verbose: bool,
    only_steps: list[str] | None = None,
    from_step: str | None = None,
    plan_only: bool = False,
    force: bool = False,
) -> None:
    options = build_options()
    # Paths handed to the agent are relative to PROJECT_ROOT (the session cwd).
    json_rel = _rel_to_root(json_path)
    pkl_rel = _rel_to_root(pkl_path)
    extra_pkls_rel = [_rel_to_root(p) for p in extra_pkl_paths]
    summary_rel = _rel_to_root(json_path.with_suffix(".summary.md"))
    steps = build_steps(json_rel, pkl_rel, summary_rel, extra_pkls_rel)

    first_phase = str(steps[0]["phase"])
    steps = select_steps(steps, only_steps, from_step)
    if not steps:
        log("No steps selected to run.")
        return
    if plan_only:
        print_plan(steps)
        return
    # Resuming a later phase relies on earlier phases' artifacts already on disk.
    if str(steps[0]["phase"]) != first_phase:
        summary_path = json_path.with_suffix(".summary.md")
        if not summary_path.is_file():
            log(
                f"WARNING: resuming at phase '{steps[0]['phase']}' but {summary_rel} "
                "does not exist yet — the earlier steps that write it were skipped."
            )

    extra_note = f", extrapolate onto {len(extra_pkls_rel)} dataset(s)" if extra_pkls_rel else ""
    n_compute = sum(1 for s in steps if s["kind"] == "compute")
    log(
        f"Starting workflow on {json_rel} (pkl {pkl_rel}{extra_note}): "
        f"{len(steps)} step(s), {n_compute} driver-run compute "
        f"[{', '.join(str(s['name']) for s in steps)}]."
    )
    wf_start = time.monotonic()
    # Open the SDK session only if at least one agent step is selected, so a
    # compute-only selection (e.g. --steps reproduce on a fresh export) needs no
    # model session at all.
    needs_session = any(s["kind"] == "agent" for s in steps)
    client_cm = ClaudeSDKClient(options=options) if needs_session else None

    async def _drive(client) -> None:
        if client is not None:
            log("SDK session opened.")
        for i, step in enumerate(steps, start=1):
            name, kind = str(step["name"]), str(step["kind"])
            print(f"\n=== Step {i}/{len(steps)}: {name} [{kind}] ===")
            log(f"Step {i}/{len(steps)} '{name}' ({kind}) started.")
            step_start = time.monotonic()
            if kind == "compute":
                await run_compute_step(step, verbose, force=force)
            else:
                # An agent fold step whose compute input was skipped has no work.
                miss = step.get("skip_if_missing")
                if miss and not (PROJECT_ROOT / str(miss)).is_file():
                    log(f"  [agent] skipped ('{name}'): {miss} absent "
                        "(its compute step was skipped — nothing to fold).")
                    continue
                final_text = await run_step(client, step, verbose)
                if not verbose:
                    print(final_text.strip())
            log(
                f"Step {i}/{len(steps)} '{name}' done in "
                f"{time.monotonic() - step_start:.0f}s."
            )

    if client_cm is not None:
        async with client_cm as client:
            await _drive(client)
    else:
        await _drive(None)
    log(f"Workflow complete in {time.monotonic() - wf_start:.0f}s.")
    print("\n=== Workflow complete ===")


# --- Brain-id consistency guard ---------------------------------------------
# The run JSON's experiments were generated against ONE brain; the --pkl passed at
# rerun time must be that SAME brain, or every reproduced number is computed on the
# wrong data while looking superficially fine. The brain id is the 6-digit dataset
# number. We recover it two ways and require agreement.
_BRAIN_RE = re.compile(r"(?<!\d)(\d{6})(?!\d)")
_CACHE_BRAIN_RE = re.compile(r"dataset_cache_(\d{6})_")


def _brain_from_pkl(pkl_path: Path) -> str | None:
    """Brain id from a cache filename like ``dataset_cache_789202_mcl100_add.pkl``."""
    m = _CACHE_BRAIN_RE.search(pkl_path.name)
    if m:
        return m.group(1)
    m = _BRAIN_RE.search(pkl_path.name)   # fallback: any lone 6-digit token
    return m.group(1) if m else None


def _brain_from_json(json_path: Path) -> tuple[str | None, dict]:
    """Brain id inferred from the run JSON's experiment code.

    The JSON has no structured dataset field, so we count 6-digit ids inside each
    record's ``code``/``codeOutput`` (preferring ``dataset_cache_<brain>_`` refs,
    which are unambiguous) and return the dominant one. Returns (brain_id, counts)
    where counts is the full {brain: n} tally for diagnostics. (brain_id None if the
    file has no recoverable id.)
    """
    try:
        data = json.loads(json_path.read_text())
    except Exception:
        return None, {}
    records = data if isinstance(data, list) else data.get("records", []) if isinstance(data, dict) else []
    cache_hits: Counter = Counter()   # from dataset_cache_<brain>_ (authoritative)
    loose_hits: Counter = Counter()   # any lone 6-digit token (fallback)
    for it in records:
        if not isinstance(it, dict):
            continue
        blob = f"{it.get('code', '')}\n{it.get('codeOutput', '')}"
        cache_hits.update(_CACHE_BRAIN_RE.findall(blob))
        loose_hits.update(_BRAIN_RE.findall(blob))
    tally = cache_hits or loose_hits
    if not tally:
        return None, {}
    return tally.most_common(1)[0][0], dict(tally)


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
        required=True,
        help=(
            "The ORIGIN dataset .pkl this run's experiments load, used to "
            "re-execute and reproduce the findings (relative to the project root "
            "or absolute)."
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
        "--steps",
        nargs="+",
        default=None,
        metavar="PHASE",
        help=(
            "Run ONLY these phases (summarize, reproduce [alias: rerun], "
            "extrapolate, verify, fix-tests, translate), reusing earlier phases' "
            "on-disk artifacts (<RUN>.summary.md, <RUN>.json.rerun, the compute "
            "JSONs <RUN>.json.reproduce.json / .extrapolate.json / .corrected.json). "
            "A phase may expand into several compute+agent steps. E.g. --steps "
            "translate to only (re)generate the 简体中文 <RUN>.summary.zh.md."
        ),
    )
    parser.add_argument(
        "--from",
        dest="from_step",
        default=None,
        metavar="PHASE",
        help=(
            "Resume from this phase and run everything after it, reusing prior "
            "artifacts. E.g. --from verify. Mutually exclusive with --steps."
        ),
    )
    parser.add_argument(
        "--plan",
        action="store_true",
        help=(
            "Print the resolved step list (which phases are driver-run compute vs "
            "agent, and the exact commands) and exit without running anything."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Recompute every compute step even when a valid output JSON "
            "(<RUN>.json.reproduce.json / .extrapolate.json / .corrected.json) "
            "already exists. By default such results are REUSED on resume instead "
            "of re-running the (slow) rerun_experiments.py."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Stream every assistant text block as it arrives.",
    )
    args = parser.parse_args()

    def _resolve_existing(p: Path, what: str) -> Path:
        rp = p if p.is_absolute() else (PROJECT_ROOT / p).resolve()
        if not rp.is_file():
            parser.error(f"No such {what}: {rp}")
        return rp

    json_path = _resolve_existing(args.path, "run JSON file")
    pkl_path = _resolve_existing(args.pkl, "dataset pkl")
    extra_pkl_paths = [
        _resolve_existing(p, "extra dataset pkl") for p in args.extra_pkl
    ]

    # --plan needs no model session and no expensive checks — show and exit.
    if args.plan:
        asyncio.run(
            run_workflow(
                json_path, pkl_path, extra_pkl_paths, args.verbose,
                only_steps=args.steps, from_step=args.from_step, plan_only=True,
            )
        )
        return 0

    # GUARD: the --pkl brain id (6-digit dataset number) MUST match the brain the run
    # JSON's experiments were generated on. Reproducing a run against the wrong brain
    # silently computes every number on the wrong data while looking fine, so a
    # mismatch is a hard error. (--extra-pkl is expected to be OTHER brains — it is
    # the extrapolation set — so it is NOT checked here.)
    json_brain, json_tally = _brain_from_json(json_path)
    pkl_brain = _brain_from_pkl(pkl_path)
    if json_brain is None:
        parser.error(
            f"Could not infer a brain id (6-digit dataset number) from {json_path} — "
            f"no dataset_cache_<brain>_ reference or lone 6-digit token found in its "
            f"experiment code. Cannot verify it matches --pkl; aborting.")
    if pkl_brain is None:
        parser.error(
            f"Could not infer a brain id from --pkl {pkl_path.name} — expected a name "
            f"like dataset_cache_<brain>_mcl<N>.pkl. Aborting.")
    if json_brain != pkl_brain:
        parser.error(
            f"Brain-id MISMATCH: the run JSON {json_path.name} was generated on brain "
            f"{json_brain} (code references {json_tally}), but --pkl is brain "
            f"{pkl_brain} ({pkl_path.name}). Reproducing one brain's run against "
            f"another brain's data computes every number on the wrong dataset. Pass "
            f"the matching --pkl (dataset_cache_{json_brain}_*.pkl).")
    print(f"[brain-check] OK: run JSON and --pkl are both brain {pkl_brain}",
          file=sys.stderr)

    asyncio.run(
        run_workflow(
            json_path,
            pkl_path,
            extra_pkl_paths,
            args.verbose,
            only_steps=args.steps,
            from_step=args.from_step,
            force=args.force,
        )
    )
    return 0


# NOTE on the brain-check + --plan ordering: --plan is handled above, BEFORE the
# brain-id guard, so you can inspect a plan without a matching pkl on hand. The
# real run below still enforces the guard.


if __name__ == "__main__":
    raise SystemExit(main())
