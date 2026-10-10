"""
Run the cross-report consolidation workflow via the Claude Agent SDK.

Where ``run_discovery_workflow.py`` digests ONE AutoDiscovery run export into a
``<RUN>.summary.md`` report, this workflow goes one level up: it reads the
FINISHED, independent per-run reports (every ``autodiscovery/*.summary.md``) and
folds them into ONE combined cross-report summary. It reads through each
report's hypotheses, clusters findings that are the same / near-duplicate across
reports, drops the redundant copies, and highlights the hypotheses that are
genuinely unique and new. A persistent Claude session runs the ordered STEPS so
later steps share earlier context. Subagents live in ``.claude/agents/`` and are
auto-discovered via ``setting_sources``. The deliverable is a single Markdown
file: ``autodiscovery/all-runs.combined.md`` (override with ``--out``).

The steps:
  1. consolidate — read every input report's ranked hypotheses, cluster
                   equivalent findings ACROSS reports, dedupe each cluster to one
                   canonical finding, and write the combined report with a
                   "Unique & New Findings" section first, then "Corroborated
                   Findings (consolidated)", then an "Excluded as Redundant"
                   audit. Pure consolidation — no experiment is re-run and no
                   finding is re-judged; verdicts/caveats are carried over.
  2. translate   — faithfully translate the finished combined report into
                   Simplified Chinese, writing ``<out-stem>.zh.md``. Pure localization (no
                   re-analysis), so it ALWAYS runs last.

By default the workflow consolidates EVERY ``autodiscovery/*.summary.md`` except
the ``*.zh.md`` translations and any prior ``*combined*`` output. Pass explicit
report paths to consolidate only those.

Usage (from the ``exa-spim-agent/`` project root):
    python agentic/run_consolidation_workflow.py
    python agentic/run_consolidation_workflow.py --verbose
    # Consolidate only named reports into a custom output path:
    python agentic/run_consolidation_workflow.py \
        autodiscovery/A.summary.md autodiscovery/B.summary.md \
        --out autodiscovery/A-vs-B.combined.md
    # Re-run only one step, reusing the other's on-disk artifact:
    python agentic/run_consolidation_workflow.py --steps translate
    python agentic/run_consolidation_workflow.py --from translate
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
# agentic/run_consolidation_workflow.py -> parent.parent is the project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Default deliverable: one combined report next to the per-run summaries.
DEFAULT_OUT_REL = "autodiscovery/all-runs.combined.md"


def collect_cmd(report_rels: list[str]) -> str:
    """Deterministic collection command for the consolidator.

    With no explicit reports it ingests every ``autodiscovery/*.summary.md``
    (excluding ``*.zh.md`` and ``*combined*``); otherwise it ingests exactly the
    named reports. Same flags the consolidator agent must echo.
    """
    if report_rels:
        return "python agentic/collect_summaries.py " + " ".join(report_rels)
    return "python agentic/collect_summaries.py"


def zh_path_for(out_rel: str) -> str:
    """The Simplified Chinese sibling path: insert ``.zh`` before ``.md``."""
    if out_rel.endswith(".md"):
        return out_rel[: -len(".md")] + ".zh.md"
    return out_rel + ".zh.md"


def build_steps(out_rel: str, report_rels: list[str]) -> list[dict[str, str]]:
    """Build the ordered workflow steps.

    ``report_rels`` is the explicit list of per-run reports to consolidate
    (empty = all autodiscovery/*.summary.md); ``out_rel`` is the combined
    Markdown deliverable. Order: consolidate → translate. The translate step is
    purely cosmetic (no analytical dependency) so it is ALWAYS appended last; add
    any new analytical step BEFORE it.
    """
    explicit = (
        "the reports " + ", ".join(report_rels)
        if report_rels
        else "EVERY per-run report under autodiscovery/ (all *.summary.md, "
        "excluding *.zh.md translations and any prior *combined* output)"
    )
    steps: list[dict[str, str]] = [
        {
            "name": "consolidate-reports",
            "instruction": (
                "Use the discovery-consolidator subagent to combine the finished, "
                "independent per-run AutoDiscovery reports into ONE cross-report "
                f"summary. Consolidate {explicit}. First run "
                f"`{collect_cmd(report_rels)}` to get every report's ranked "
                "hypotheses parsed into one pooled JSON (read it from stdout; it "
                "carries each entry's source_file, run, id, priority_score, "
                "surprise_magnitude, title, belief, direction, and the full "
                "fields map of every bullet). Read through every entry and CLUSTER "
                "the ones that express the SAME underlying scientific finding "
                "across reports — judging sameness by what was tested and "
                "concluded, NOT by id (ids do not correspond across runs). Within "
                "each cluster, dedupe to one canonical finding (prefer the "
                "best-verified, highest-priority entry), fold the redundant copies "
                "away, and record which reports corroborated it and any "
                "disagreement in verdict/direction across them. Then identify the "
                "findings that are UNIQUE (appear in only one report) or NEW "
                "(first appear in the newer run) and foreground them. Do NOT "
                "re-run experiments or re-judge statistics — carry over each "
                "source entry's verdicts and caveats faithfully. Write the "
                f"combined report to {out_rel} with a 'Unique & New Findings' "
                "section FIRST, then 'Corroborated Findings (consolidated)', then "
                "an 'Excluded as Redundant' audit list. Report the path and a "
                "short executive summary: how many distinct findings remain, the "
                "most important unique/new ones, and any cross-report disagreements."
            ),
        }
    ]

    # Final, purely-cosmetic step: translate the finished combined report into
    # Simplified Chinese. It reads only the completed Markdown deliverable and writes a
    # sibling .zh.md, so it has NO analytical dependency and must always run
    # LAST. If more analytical steps are added, append them BEFORE this block.
    out_zh_rel = zh_path_for(out_rel)
    steps.append(
        {
            "name": "translate-report",
            "instruction": (
                "Use the discovery-translator subagent to produce a faithful "
                f"Simplified Chinese translation of the finished combined report at {out_rel}, "
                f"writing it to {out_zh_rel}. This is a pure localization pass: "
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
STEP_ALIASES: dict[str, str] = {
    "consolidate": "consolidate-reports",
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

    Resuming relies on the prior step's artifact already on disk (the combined
    report). ``--steps`` keeps exactly the named steps (in pipeline order);
    ``--from`` keeps that step and everything after it. Mutually exclusive; with
    neither, all steps run.
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
    filesystem subagents in ``.claude/agents/`` (incl. discovery-consolidator)
    and project settings relative to ``cwd``.
    """
    return ClaudeAgentOptions(
        cwd=str(PROJECT_ROOT),
        setting_sources=["project"],
        # The orchestrator delegates to subagents (Task), which need filesystem
        # tools to collect the reports and write the combined Markdown.
        allowed_tools=["Task", "Bash", "Read", "Write", "Edit", "Glob"],
        permission_mode="bypassPermissions",
        # Pin Opus 5.5 explicitly so the model is not left to the ambient session
        # default. Subagents are `model: inherit`, so they follow this too.
        model="claude-opus-5-5",
        # Run Opus 5.5 at maximum reasoning effort for the cross-report
        # clustering/dedup work. Applies to the session + subagents.
        env={**os.environ, "CLAUDE_EFFORT": "xhigh"},
    )


async def run_step(client: ClaudeSDKClient, step: dict[str, str], verbose: bool) -> str:
    """Send one workflow step to the session and return its final text."""
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
    """Path relative to PROJECT_ROOT (the session cwd), tolerating ``..``."""
    return Path(os.path.relpath(path, PROJECT_ROOT)).as_posix()


async def run_workflow(
    report_paths: list[Path],
    out_path: Path,
    verbose: bool,
    only_steps: list[str] | None = None,
    from_step: str | None = None,
) -> None:
    options = build_options()
    report_rels = [_rel_to_root(p) for p in report_paths]
    out_rel = _rel_to_root(out_path)
    steps = build_steps(out_rel, report_rels)

    all_names = [s["name"] for s in steps]
    steps = select_steps(steps, only_steps, from_step)
    if not steps:
        log("No steps selected to run.")
        return
    # Resuming a later step relies on the earlier step's artifact already on disk.
    if steps[0]["name"] != all_names[0] and not out_path.is_file():
        log(
            f"WARNING: resuming at '{steps[0]['name']}' but {out_rel} does not "
            "exist yet — the consolidate step that writes it was skipped."
        )

    scope = f"{len(report_rels)} named report(s)" if report_rels else "all autodiscovery/*.summary.md"
    log(
        f"Starting consolidation on {scope} → {out_rel}: "
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
                print(final_text.strip())
    log(f"Workflow complete in {time.monotonic() - wf_start:.0f}s.")
    print("\n=== Workflow complete ===")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "reports",
        type=Path,
        nargs="*",
        help=(
            "Per-run report .md files to consolidate (relative to the project "
            "root or absolute). Default: every autodiscovery/*.summary.md "
            "(excluding *.zh.md and any *combined* output)."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(DEFAULT_OUT_REL),
        metavar="PATH",
        help=(
            "Path for the combined report (relative to the project root or "
            f"absolute). Default: {DEFAULT_OUT_REL}."
        ),
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        default=None,
        metavar="STEP",
        help=(
            "Run ONLY these steps (aliases: consolidate, translate), reusing the "
            "other step's on-disk artifact. E.g. --steps translate to only "
            "(re)generate the Simplified Chinese version from an existing combined report."
        ),
    )
    parser.add_argument(
        "--from",
        dest="from_step",
        default=None,
        metavar="STEP",
        help=(
            "Resume from this step and run everything after it, reusing prior "
            "artifacts. E.g. --from translate. Mutually exclusive with --steps."
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

    report_paths = [_resolve_existing(p, "report .md file") for p in args.reports]
    # The output need not exist yet; just resolve it against the project root.
    out_path = (
        args.out if args.out.is_absolute() else (PROJECT_ROOT / args.out).resolve()
    )

    asyncio.run(
        run_workflow(
            report_paths,
            out_path,
            args.verbose,
            only_steps=args.steps,
            from_step=args.from_step,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
