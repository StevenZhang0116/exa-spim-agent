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
                   reproduced results.

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
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
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
    verify (audits using the recorded and freshly-reproduced results).
    """
    extra_pkls_rel = extra_pkls_rel or []
    rerun_dir_rel = f"{json_rel}.rerun"
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
) -> None:
    options = build_options()
    # Paths handed to the agent are relative to PROJECT_ROOT (the session cwd).
    json_rel = _rel_to_root(json_path)
    pkl_rel = _rel_to_root(pkl_path)
    extra_pkls_rel = [_rel_to_root(p) for p in extra_pkl_paths]
    summary_rel = _rel_to_root(json_path.with_suffix(".summary.md"))
    steps = build_steps(json_rel, pkl_rel, summary_rel, extra_pkls_rel)

    extra_note = f", extrapolate onto {len(extra_pkls_rel)} dataset(s)" if extra_pkls_rel else ""
    log(
        f"Starting workflow on {json_rel} (pkl {pkl_rel}{extra_note}): "
        f"{len(steps)} step(s)."
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
    asyncio.run(run_workflow(json_path, pkl_path, extra_pkl_paths, args.verbose))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
