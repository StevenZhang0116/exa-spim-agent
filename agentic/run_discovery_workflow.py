"""
Run the AutoDiscovery summarization workflow via the Claude Agent SDK.

This is an extensible, sequential workflow scaffold. It opens one persistent
Claude session and runs an ordered list of STEPS through it, so later steps
share the conversation/context of earlier ones. As shipped there is a single
step — it delegates to the project's ``discovery-summarizer`` subagent (defined
in ``.claude/agents/discovery-summarizer.md``) to rank every ``autodiscovery/``
run export by surprise and write the collective ``all-runs.summary.md`` report.

To grow the workflow, append more entries to STEPS (e.g. "propose follow-up
experiments from the top-ranked hypotheses", "critique the most surprising
conclusions"). Each step is just a natural-language instruction sent to the
same session in order.

Prerequisites
-------------
- The Claude Code CLI must be installed and on PATH (the SDK drives it):
      claude --version
- The Python SDK:
      pip install claude-agent-sdk
- Auth: an ``ANTHROPIC_API_KEY`` in the environment, or an already-authenticated
  Claude Code CLI login.

Usage (from the ``exa-spim-agent/`` project root)
-------------------------------------------------
    python agentic/run_discovery_workflow.py
    python agentic/run_discovery_workflow.py --verbose   # stream every text block

The subagent ranks deterministically via ``agentic/rank_by_surprise.py`` and
writes ``autodiscovery/all-runs.summary.md``; that file is the deliverable.
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ResultMessage,
    TextBlock,
)

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
        # The orchestrator delegates to the subagent (Task) and the subagent
        # itself needs Bash/Read/Write/Glob; allow them so the run is non-interactive.
        allowed_tools=["Task", "Bash", "Read", "Write", "Glob"],
        # Non-interactive: don't prompt for permission on each tool call. Drop to
        # "acceptEdits" if you'd rather review/limit what runs.
        permission_mode="bypassPermissions",
    )


async def run_step(client: ClaudeSDKClient, step: dict[str, str], verbose: bool) -> str:
    """Send one workflow step to the session and return its final text.

    Streams assistant text as it arrives (when ``verbose``) and returns the
    concatenated assistant text for the step so a caller could chain on it.
    """
    await client.query(step["instruction"])

    chunks: list[str] = []
    async for message in client.receive_response():
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    chunks.append(block.text)
                    if verbose:
                        print(block.text, end="", flush=True)
        elif isinstance(message, ResultMessage):
            # End of this turn. ResultMessage carries cost/usage if you want it.
            if verbose:
                print()  # newline after the streamed text
    return "".join(chunks)


async def run_workflow(verbose: bool) -> None:
    options = build_options()
    async with ClaudeSDKClient(options=options) as client:
        for i, step in enumerate(STEPS, start=1):
            print(f"\n=== Step {i}/{len(STEPS)}: {step['name']} ===")
            final_text = await run_step(client, step, verbose)
            if not verbose:
                # In quiet mode, print only each step's final summary.
                print(final_text.strip())
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
