"""
Run the AutoDiscovery summarization workflow via the Claude Agent SDK.

Processes ONE AutoDiscovery run export (a single JSON file) together with the
dataset ``.pkl`` its experiments were run against. A persistent Claude session
is opened and the ordered STEPS run through it (summarize → rerun → verify) so
later steps share earlier context. Subagents live in ``.claude/agents/`` and are
auto-discovered via ``setting_sources``. The file produces a Markdown
deliverable next to the input: ``<stem>.summary.md``.

The steps:
  1. summarize   — rank the file's hypotheses and write the top-K report.
  2. rerun       — re-execute each reported hypothesis's recorded code against
                   the provided ``--pkl`` and fold a REPRODUCED/DIVERGED/FAILED
                   verdict into each entry.
  3. extrapolate — (only when ``--extra-pkl`` is given) re-run the reproduced
                   code on each OTHER dataset to test whether the conclusions
                   generalize; fold a GENERALIZES/PARTIAL/DOES-NOT/INCONCLUSIVE
                   verdict into each entry.
  4. verify      — audit the statistics/logic using the recorded and freshly
                   reproduced results; assign OK/MINOR/MAJOR/CRITICAL verdicts.
  5. fix-tests   — for hypotheses the verifier flagged with a WRONG statistical
                   test, rewrite ONLY the test, re-measure on the real data, and
                   fold an UPHELD/WEAKENED/OVERTURNED corrected verdict in.
  6. translate   — faithfully translate the finished ``<stem>.summary.md`` report
                   into 简体中文, writing ``<stem>.summary.zh.md``. This is a pure
                   localization pass (no re-analysis), so it ALWAYS runs last,
                   after every analytical step, however many are added.

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
    # Resume / run a subset, reusing prior steps' on-disk artifacts
    # (<RUN>.summary.md, <RUN>.json.rerun) — e.g. only the corrective step:
    python agentic/run_discovery_workflow.py autodiscovery/RUN.json --pkl ORIGIN.pkl \
        --steps fix-tests
    python agentic/run_discovery_workflow.py autodiscovery/RUN.json --pkl ORIGIN.pkl \
        --from verify          # rerun verify and everything after it
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
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


def rank_cmd(json_rel: str) -> str:
    """Deterministic ranking command for ONE run export.

    Built from the free parameters above so the helper, the report, and the
    verifier stay in sync. ``json_rel`` is the run file path relative to the
    project root, so the helper ranks only that single file.
    """
    return f"python agentic/rank_by_surprise.py {json_rel} {_top_flags()}"


def rerun_cmd(json_rel: str, pkl_rel: str) -> str:
    """Deterministic re-execution command for ONE run export + its dataset pkl.

    Uses the SAME ``--rank-by``/``--top`` flags as ``rank_cmd`` so the rerun set
    is exactly the set of records the summarizer put in the report.
    """
    return (
        f"python agentic/rerun_experiments.py {json_rel} "
        f"--pkl {pkl_rel} {_top_flags()}"
    )


def extrapolate_cmd(
    json_rel: str, pkl_rel: str, rerun_dir_rel: str, extra_pkls_rel: list[str]
) -> str:
    """Deterministic extrapolation command: rerun the reproduced code on others.

    Same ``--rank-by``/``--top`` as the rerun, reuses the reproducer's revised
    ``--code-dir`` so the working code runs, and adds one ``--extra-pkl`` per
    other dataset to test generalization.
    """
    extra = " ".join(f"--extra-pkl {p}" for p in extra_pkls_rel)
    return (
        f"python agentic/rerun_experiments.py {json_rel} --pkl {pkl_rel} "
        f"{_top_flags()} --code-dir {rerun_dir_rel} {extra}"
    )


def fix_tests_cmd(
    json_rel: str,
    pkl_rel: str,
    rerun_dir_rel: str,
    fixed_dir_rel: str,
    extra_pkls_rel: list[str] | None = None,
) -> str:
    """Deterministic re-measurement command for corrected statistical tests.

    Reuses the reproducer's loading-fix ``--code-dir`` and adds
    ``--corrected-dir`` (the test-fixer's rewritten-analysis scripts), which take
    precedence per record so flagged hypotheses run the corrected test. When
    extra datasets are given, the SAME corrected test is also run on each (via
    ``--extra-pkl``), so generalization is judged with the right test rather than
    the original flawed one.
    """
    extra = " ".join(f"--extra-pkl {p}" for p in (extra_pkls_rel or []))
    return (
        f"python agentic/rerun_experiments.py {json_rel} --pkl {pkl_rel} "
        f"{_top_flags()} --code-dir {rerun_dir_rel} --corrected-dir {fixed_dir_rel}"
        + (f" {extra}" if extra else "")
    )


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
) -> list[dict[str, str]]:
    """Build the ordered workflow steps for a SINGLE run export + dataset pkl.

    Each step is an instruction sent to the same persistent session, so step N
    can build on the results of step N-1. ``json_rel`` is the run JSON to digest,
    ``pkl_rel`` is the dataset the experiments re-execute against, and
    ``summary_rel`` is the per-file Markdown deliverable — all relative to the
    project root. Order: summarize → rerun (reproduce against the pkl) →
    [extrapolate onto other datasets, only when ``extra_pkls_rel`` is given] →
    verify (audits using the recorded and freshly-reproduced results) →
    fix-tests (re-measure hypotheses the verifier flagged with a wrong test) →
    translate (faithful 简体中文 localization of the finished report). The
    translate step is purely cosmetic — it depends on nothing analytical — so it
    is ALWAYS appended last, after any step added above it.
    """
    extra_pkls_rel = extra_pkls_rel or []
    rerun_dir_rel = f"{json_rel}.rerun"
    fixed_dir_rel = f"{json_rel}.fixed"
    steps: list[dict[str, str]] = [
        {
            "name": "summarize-discoveries",
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
        {
            "name": "rerun-experiments",
            "instruction": (
                "Use the discovery-reproducer subagent to RE-EXECUTE the "
                "experiment code of the reported hypotheses against the provided "
                f"dataset {pkl_rel} and check whether each finding reproduces. "
                f"First run `{rerun_cmd(json_rel, pkl_rel)}` (same --rank-by/--top "
                "as the summarizer, so the rerun set matches the report) to run "
                "each top-ranked record's recorded `code` on the pkl and print the "
                "fresh output next to the recorded codeOutput. Then, for any "
                "result that FAILED or TIMED OUT for a DATA-LOADING / ENVIRONMENT "
                "reason (e.g. a NumPy-2-written pkl the host can't unpickle, a "
                "'dataset not found' gate, or a pip-install retry loop), export "
                f"the editable scripts with `--export-dir {json_rel}.rerun`, "
                "revise ONLY the loading/bootstrap of those scripts (load the pkl "
                "directly from $RERUN_PKL, fix the NumPy version, drop pip-retry "
                "loops) while keeping the analysis identical, and rerun with "
                f"`--code-dir {json_rel}.rerun`. Compare the fresh key numbers "
                "(test statistic, p-value, effect size, n) to the recorded ones "
                "and assign a REPRODUCED / DIVERGED / FAILED verdict (note whether "
                f"the code was recorded or revised-loading). Read {summary_rel} "
                "and fold the reproduction result INTO each hypothesis's existing "
                "ranked entry (append Reproduction / Rerun result bullets in "
                "place, keeping prior bullets), plus one 'Reproduction — Summary' "
                "section. Report the path and which findings did NOT reproduce."
            ),
        },
    ]

    # Optional extrapolation step: only when other datasets were provided. Runs
    # the reproduced code (the reviser's --code-dir) on each extra pkl to test
    # whether the conclusions generalize beyond the origin dataset.
    if extra_pkls_rel:
        extra_list = ", ".join(extra_pkls_rel)
        steps.append(
            {
                "name": "extrapolate-generalization",
                "instruction": (
                    "Use the discovery-extrapolator subagent to test whether each "
                    "reported finding GENERALIZES to other datasets that the "
                    "hypotheses were NOT generated on: "
                    f"{extra_list}. Run "
                    f"`{extrapolate_cmd(json_rel, pkl_rel, rerun_dir_rel, extra_pkls_rel)}` "
                    "(same --rank-by/--top as the summarizer, reusing the "
                    f"reproducer's revised --code-dir {rerun_dir_rel} so the "
                    "working code runs). The helper runs each record's code on the "
                    "origin pkl and, with the load redirected, on each extra pkl; "
                    "each result carries an `extrapolations` list with the fresh "
                    "output per extra dataset. For each hypothesis, compare the "
                    "origin numbers to each extra dataset's numbers (direction, "
                    "significance, effect size) and assign GENERALIZES / PARTIAL / "
                    "DOES-NOT-GENERALIZE / INCONCLUSIVE. If a revised script still "
                    "fails to LOAD an extra pkl, apply the same loading-only fix to "
                    f"its hypo_<id>.py in {rerun_dir_rel} and re-run. Read "
                    f"{summary_rel} and fold the generalization result INTO each "
                    "hypothesis's existing ranked entry (append Generalization / "
                    "Across datasets bullets in place, keeping prior bullets), plus "
                    "one 'Generalization — Summary' section. Report the path and "
                    "which findings do NOT generalize."
                ),
            }
        )

    steps.append(
        {
            "name": "verify-statistics-and-logic",
            "instruction": (
                "Use the discovery-verifier subagent to audit whether each "
                "hypothesis's statistical test and its inductive/deductive "
                "reasoning are correct, judging from the recorded code, "
                f"codeOutput, and analysis in {json_rel} AND the freshly "
                "reproduced results the previous rerun step folded into "
                f"{summary_rel} (do NOT re-run experiments yourself; use the "
                "rerun output already in the report). Check test choice, "
                "assumptions, power, effect size, p-value interpretation, the "
                "'failed-to-reject != null-is-true' fallacy, conclusion "
                "overreach, whether a finding that FAILED or DIVERGED on rerun "
                "should be downgraded, and a file-wide multiple-comparisons (FDR) "
                f"analysis. Read {summary_rel} and fold the audit INTO each "
                "hypothesis's existing ranked entry (append Verdict / Test / "
                "Statistical issues / Logic issues bullets in place, keeping the "
                "summary and reproduction bullets), plus one file-wide "
                "'Statistical Verification — Summary' section at the end. Report "
                "the path and the most serious problems found."
            ),
        }
    )

    # Corrective step: re-test the hypotheses the verifier flagged with a WRONG
    # statistical test. Runs after verify because it needs the verdicts.
    steps.append(
        {
            "name": "fix-statistical-tests",
            "instruction": (
                "Use the discovery-test-fixer subagent to CORRECT and re-measure "
                "the hypotheses whose statistical TEST the verifier flagged as "
                f"wrong/unsound. Read {summary_rel} and select the entries with a "
                "MAJOR/CRITICAL verdict or a 'Statistical issues' bullet naming a "
                "concrete test fault (wrong test for the data, violated "
                "independence/normality assumptions, huge-n significance of a "
                "trivial effect, p-value misuse, double-counting). For each, start "
                f"from the loading-fixed script in {rerun_dir_rel}/hypo_<id>.py and "
                f"write a corrected hypo_<id>.py into {fixed_dir_rel} that changes "
                "ONLY the statistical test (and prints an effect size with a CI, "
                "and a cluster/permutation p-value where independence is "
                "violated), keeping the same data and quantities. Re-measure by "
                f"running `{fix_tests_cmd(json_rel, pkl_rel, rerun_dir_rel, fixed_dir_rel, extra_pkls_rel)}` "
                "and read the `code_source: corrected` results"
                + (
                    " (each carries an `extrapolations` list — the SAME corrected "
                    "test run on each extra dataset, so you can re-judge "
                    "generalization with the CORRECT test). "
                    if extra_pkls_rel
                    else ". "
                )
                + "For each fixed hypothesis assign UPHELD / WEAKENED / OVERTURNED "
                "by comparing the corrected numbers to the original"
                + (
                    ", and a corrected-test generalization note (does the finding "
                    "still hold across the extra datasets under the right test?). "
                    if extra_pkls_rel
                    else ". "
                )
                + "Fold Corrected test / "
                "Corrected result / Post-correction verdict bullets INTO each "
                "fixed entry in place (keep all prior bullets), plus one "
                "'Statistical Test Corrections — Summary' section at the very end. "
                "Report the path and which findings changed under the correct test."
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


# Short aliases for the step names, for --steps / --from selection on the CLI.
# Each maps to the canonical step ``name`` build_steps() produces.
STEP_ALIASES: dict[str, str] = {
    "summarize": "summarize-discoveries",
    "rerun": "rerun-experiments",
    "extrapolate": "extrapolate-generalization",
    "verify": "verify-statistics-and-logic",
    "fix-tests": "fix-statistical-tests",
    "translate": "translate-report",
}


def _canonical_step(token: str, available: list[str]) -> str:
    """Resolve a user token (alias or full name) to a canonical step name."""
    if token in STEP_ALIASES:
        return STEP_ALIASES[token]
    if token in available:
        return token
    raise SystemExit(
        f"Unknown step '{token}'. Choose from: "
        + ", ".join(STEP_ALIASES) + " (or full names: " + ", ".join(available) + ")."
    )


def select_steps(
    steps: list[dict[str, str]],
    only: list[str] | None,
    from_step: str | None,
) -> list[dict[str, str]]:
    """Filter the built steps by ``--steps`` (explicit set) or ``--from`` (suffix).

    Resuming relies on prior steps' artifacts already being on disk: the
    ``<RUN>.summary.md`` report (and its folded-in verdicts) and the
    ``<RUN>.json.rerun`` loading-fixed scripts. ``--steps`` keeps exactly the
    named steps (in pipeline order); ``--from`` keeps that step and everything
    after it. They are mutually exclusive; with neither, all steps run.
    """
    available = [s["name"] for s in steps]
    if only and from_step:
        raise SystemExit("--steps and --from are mutually exclusive.")
    if only:
        wanted = {_canonical_step(t, available) for t in only}
        return [s for s in steps if s["name"] in wanted]
    if from_step:
        start = _canonical_step(from_step, available)
        idx = available.index(start)
        return steps[idx:]
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
        # default. Subagents are `model: inherit`, so they follow this too.
        model="claude-opus-4-8",
        # Run Opus 4.8 at maximum reasoning effort for the summarize/rerun/verify
        # work. Levels: low|medium|high|xhigh|max; Opus 4.8 thinks adaptively and
        # at xhigh almost always reasons deeply. Applies to the session + subagents.
        env={**os.environ, "CLAUDE_EFFORT": "xhigh"},
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


async def run_workflow(
    json_path: Path,
    pkl_path: Path,
    extra_pkl_paths: list[Path],
    verbose: bool,
    only_steps: list[str] | None = None,
    from_step: str | None = None,
) -> None:
    options = build_options()
    # Paths handed to the agent are relative to PROJECT_ROOT (the session cwd).
    json_rel = _rel_to_root(json_path)
    pkl_rel = _rel_to_root(pkl_path)
    extra_pkls_rel = [_rel_to_root(p) for p in extra_pkl_paths]
    summary_rel = _rel_to_root(json_path.with_suffix(".summary.md"))
    steps = build_steps(json_rel, pkl_rel, summary_rel, extra_pkls_rel)

    all_names = [s["name"] for s in steps]
    steps = select_steps(steps, only_steps, from_step)
    if not steps:
        log("No steps selected to run.")
        return
    # Resuming a later step relies on earlier steps' artifacts already on disk.
    if steps[0]["name"] != all_names[0]:
        summary_path = json_path.with_suffix(".summary.md")
        if not summary_path.is_file():
            log(
                f"WARNING: resuming at '{steps[0]['name']}' but {summary_rel} does "
                "not exist yet — the earlier steps that write it were skipped."
            )

    extra_note = f", extrapolate onto {len(extra_pkls_rel)} dataset(s)" if extra_pkls_rel else ""
    log(
        f"Starting workflow on {json_rel} (pkl {pkl_rel}{extra_note}): "
        f"{len(steps)} step(s) [{', '.join(s['name'] for s in steps)}]."
    )
    wf_start = time.monotonic()
    async with ClaudeSDKClient(options=options) as client:
        log("SDK session opened.")
        for i, step in enumerate(steps, start=1):
            print(f"\n=== Step {i}/{len(steps)}: {step['name']} ===")
            log(f"Step {i}/{len(steps)} '{step['name']}' started.")
            step_start = time.monotonic()
            final_text = await run_step(client, step, verbose)
            log(
                f"Step {i}/{len(steps)} '{step['name']}' done in "
                f"{time.monotonic() - step_start:.0f}s."
            )
            if not verbose:
                # In quiet mode, print only each step's final summary.
                print(final_text.strip())
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
        metavar="STEP",
        help=(
            "Run ONLY these steps (aliases: summarize, rerun, extrapolate, "
            "verify, fix-tests, translate), reusing prior steps' on-disk artifacts "
            "(<RUN>.summary.md, <RUN>.json.rerun). E.g. --steps translate to only "
            "(re)generate the 简体中文 <RUN>.summary.zh.md from an existing report."
        ),
    )
    parser.add_argument(
        "--from",
        dest="from_step",
        default=None,
        metavar="STEP",
        help=(
            "Resume from this step and run everything after it, reusing prior "
            "artifacts. E.g. --from verify. Mutually exclusive with --steps."
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
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
