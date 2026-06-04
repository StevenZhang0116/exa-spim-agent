"""
Run the AutoDiscovery summarization workflow via the Claude Agent SDK.

Opens one persistent Claude session and runs the ordered STEPS through it
(summarize, then verify), so later steps share earlier context. Subagents
live in ``.claude/agents/`` and are auto-discovered via ``setting_sources``.
Both steps write into one combined Markdown deliverable:
``autodiscovery/all-runs.summary.md``.

Usage (from the ``exa-spim-agent/`` project root):
    python agentic/run_discovery_workflow.py
    python agentic/run_discovery_workflow.py --verbose   # stream assistant text
"""

from __future__ import annotations

import argparse
import asyncio
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

# The ordered workflow. Each step is an instruction sent to the same persistent
# session, so step N can build on the results of step N-1. Step 1 kicks off the
# discovery-summarizer subagent; add further steps below as the workflow grows.
STEPS: list[dict[str, str]] = [
    {
        "name": "summarize-discoveries",
        "instruction": (
            "Use the discovery-summarizer subagent to digest every AutoDiscovery "
            "run export under the autodiscovery/ folder. It must run "
            "agentic/rank_by_surprise.py to pool and rank all hypotheses by "
            "surprise magnitude, then write the collective ranked report to "
            "autodiscovery/all-runs.summary.md. Report the path it wrote and a "
            "short executive summary of the most surprising conclusions."
        ),
    },
    {
        "name": "verify-statistics-and-logic",
        "instruction": (
            "Use the discovery-verifier subagent to audit whether each "
            "hypothesis's statistical test and its inductive/deductive reasoning "
            "are correct, judging ONLY from the recorded code, codeOutput, and "
            "analysis in the autodiscovery/ JSON exports (do NOT re-run any "
            "experiment). Check test choice, assumptions, power, effect size, "
            "p-value interpretation, the 'failed-to-reject != null-is-true' "
            "fallacy, conclusion overreach, and a run-wide multiple-comparisons "
            "(FDR) analysis. Read autodiscovery/all-runs.summary.md and fold the "
            "audit INTO each hypothesis's existing ranked entry (append Verdict / "
            "Test / Statistical issues / Logic issues bullets in place, keeping "
            "the summary bullets), plus one run-wide 'Statistical Verification — "
            "Run-wide Summary' section at the end. Report the path and the most "
            "serious problems found."
        ),
    },
    # --- Add later steps here, e.g.: ---
    # {
    #     "name": "propose-followups",
    #     "instruction": (
    #         "From all-runs.summary.md, take the 5 most surprising belief flips "
    #         "and draft a concrete follow-up experiment plan for each."
    #     ),
    # },
]


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


async def run_workflow(verbose: bool) -> None:
    options = build_options()
    log(f"Starting workflow: {len(STEPS)} step(s), project root {PROJECT_ROOT}")
    wf_start = time.monotonic()
    async with ClaudeSDKClient(options=options) as client:
        log("SDK session opened.")
        for i, step in enumerate(STEPS, start=1):
            print(f"\n=== Step {i}/{len(STEPS)}: {step['name']} ===")
            log(f"Step {i}/{len(STEPS)} '{step['name']}' started.")
            step_start = time.monotonic()
            final_text = await run_step(client, step, verbose)
            elapsed = time.monotonic() - step_start
            log(f"Step {i}/{len(STEPS)} '{step['name']}' done in {elapsed:.0f}s.")
            if not verbose:
                # In quiet mode, print only each step's final summary.
                print(final_text.strip())
    total = time.monotonic() - wf_start
    log(f"All {len(STEPS)} step(s) complete in {total:.0f}s.")
    print("\n=== Workflow complete ===")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Stream every assistant text block as it arrives.",
    )
    args = parser.parse_args()
    asyncio.run(run_workflow(args.verbose))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
