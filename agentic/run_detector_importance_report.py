#!/usr/bin/env python3
"""Short feature-importance report for one completed detector run.

Two steps, the first deterministic:

1. ``agentic/detector_importance.py`` reads the run's model-selection JSON, fitted
   joblib and CSV (GT-judgeable rows only) plus the application's feature inventory
   and the origin AutoDiscovery export, and writes ``importance_evidence.json`` and
   an immutable Markdown evidence block: standardized coefficients or native
   importances, in-sample permutation drops, univariate AP against the raw baseline,
   and the same views summed per hypothesis.
2. The ``detector-importance-summarizer`` subagent (Claude Agent SDK) turns that
   evidence into a short English summary in ``IMPORTANCE_REPORT.md``; the driver
   checks the result (immutable block, every placeholder replaced, headings, length
   cap, English only) and allows one repair turn. ``--no-agent`` writes the evidence
   and the tables only.

Run on a compute node under ``panda`` (the fitted model and the CSV's judgeable rows
are loaded; the dataset pkl is never read); the agent step needs ``ANTHROPIC_API_KEY``:

    python agentic/run_detector_importance_report.py \\
        autodiscovery-application/<RUN>/runs/<brain>-with-figures
    python agentic/run_detector_importance_report.py <run dir> --no-agent

Outputs inside the run directory: ``importance_evidence.json``,
``IMPORTANCE_REPORT.md``, ``importance_report_workflow.log.txt``.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import platform
import re
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

try:
    from agentic.detector_importance import (
        PROJECT_ROOT, Tee, collect_importance_evidence, evidence_markdown, rel,
        resolve_result_dir, sha256, write_json_atomic)
    from agentic.detector_build.agent_session import load_claude_sdk, open_agent_session
except ImportError:  # pragma: no cover - script-style import
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from detector_importance import (  # type: ignore
        PROJECT_ROOT, Tee, collect_importance_evidence, evidence_markdown, rel,
        resolve_result_dir, sha256, write_json_atomic)
    from detector_build.agent_session import load_claude_sdk, open_agent_session  # type: ignore

EVIDENCE_NAME = "importance_evidence.json"
REPORT_NAME = "IMPORTANCE_REPORT.md"
WORKFLOW_LOG_NAME = "importance_report_workflow.log.txt"
BLOCK_START = "<!-- BEGIN DRIVER-GENERATED IMPORTANCE EVIDENCE — preserve this block -->"
BLOCK_END = "<!-- END DRIVER-GENERATED IMPORTANCE EVIDENCE -->"
SUMMARY_HEADING = "# Summary"
SECTION_HEADINGS = (
    "## 1. What drives the ranking",
    "## 2. The top features in plain language",
    "## 3. Caveats and next checks",
)
PLACEHOLDER = "<!-- IMPORTANCE-SUMMARIZER MUST REPLACE THIS PLACEHOLDER -->"
NO_AGENT_TEXT = "No summarizer turn was run (`--no-agent`); read the evidence block above."
MAX_WORDS = 600
AGENT_MODEL = os.environ.get("DETECTOR_IMPORTANCE_AGENT_MODEL", "claude-opus-5-5")
AGENT_EFFORT = os.environ.get("DETECTOR_IMPORTANCE_AGENT_EFFORT", "high")
AGENT_TIMEOUT_S = int(os.environ.get("DETECTOR_IMPORTANCE_TIMEOUT_S", "3600"))


def write_skeleton(report_path: Path, evidence: dict, evidence_sha: str) -> str:
    sections = "\n\n".join(f"{h}\n\n{PLACEHOLDER}" for h in SECTION_HEADINGS)
    content = f"""# Feature importance — `{evidence['brain']}` `{evidence['detector_target']}`

> Short report: which features and hypotheses drive this detector's ranking. The
> evidence block is generated deterministically; the summary is written by
> `detector-importance-summarizer` from that block only.

{BLOCK_START}

Evidence file: `{evidence['result_dir']}/{EVIDENCE_NAME}` — SHA-256 `{evidence_sha}`

{evidence_markdown(evidence)}
{BLOCK_END}

{SUMMARY_HEADING}

{sections}
"""
    report_path.write_text(content, encoding="utf-8")
    return content


def driver_block(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    start, end = text.find(BLOCK_START), text.find(BLOCK_END)
    if start < 0 or end < start:
        raise SystemExit(f"Report {path} is missing its driver evidence block.")
    return text[start:end + len(BLOCK_END)]


def validate_report(report_path: Path, *, skeleton_sha: str, expected_block: str,
                    evidence_sha: str) -> None:
    if not report_path.is_file() or report_path.stat().st_size == 0:
        raise SystemExit(f"The summarizer did not write {report_path}.")
    if sha256(report_path) == skeleton_sha:
        raise SystemExit("The summarizer left the IMPORTANCE_REPORT.md skeleton unchanged.")
    if driver_block(report_path) != expected_block:
        raise SystemExit("The summarizer modified the immutable importance-evidence block.")
    text = report_path.read_text(encoding="utf-8")
    if evidence_sha not in text:
        raise SystemExit("The report lost its evidence SHA-256.")
    if PLACEHOLDER in text:
        raise SystemExit("A section placeholder was left unchanged.")
    summary_start = text.find(SUMMARY_HEADING)
    if summary_start < 0 or summary_start < text.find(BLOCK_END):
        raise SystemExit("The summary must follow the evidence block under its heading.")
    summary = text[summary_start:]
    missing = [h for h in SECTION_HEADINGS if h not in summary]
    if missing:
        raise SystemExit(f"Required sections are missing: {missing}.")
    if re.search(r"[一-鿿]", summary):
        raise SystemExit("The summary must be English only.")
    words = len(re.findall(r"[A-Za-z0-9][\w.%+-]*", summary))
    if words > MAX_WORDS:
        raise SystemExit(
            f"The summary has {words} words; the cap is {MAX_WORDS}. "
            "This is a short report: cut, do not pad.")


def fill_without_agent(report_path: Path) -> None:
    text = report_path.read_text(encoding="utf-8")
    report_path.write_text(text.replace(PLACEHOLDER, NO_AGENT_TEXT), encoding="utf-8")


async def run_agent(result_dir: Path, evidence_path: Path, report_path: Path,
                    inventory_path: Path, verbose: bool, check) -> None:
    """One summarizer turn, a driver check, and at most one repair turn."""
    sdk = load_claude_sdk()
    options = sdk.agent_options(
        cwd=str(PROJECT_ROOT),
        setting_sources=["project"],
        allowed_tools=["Task", "Read", "Write", "Edit", "Glob"],
        # The orchestrating session must not shell out: the first run showed Bash
        # calls although the subagent itself is limited to Read/Write/Edit/Glob.
        disallowed_tools=["Bash"],
        permission_mode="bypassPermissions",
        model=AGENT_MODEL,
        effort=AGENT_EFFORT,
    )
    instruction = f"""
Use the detector-importance-summarizer subagent to write the short feature-importance
summary for the completed detector run {rel(result_dir)}. Read {rel(evidence_path)}
as the only source of numbers. The skeleton at {rel(report_path)} already holds the
immutable BEGIN/END DRIVER-GENERATED IMPORTANCE EVIDENCE block and the exact required
section headings, each followed by a placeholder comment: replace EVERY placeholder and
do not renumber, retitle, add or remove any heading. Write in English only, at most
{MAX_WORDS} words in total: this is a short report. For every feature or hypothesis you
name, say in plain language what it measures and which hypothesis it came from, using
{rel(inventory_path)} and the origin export named in the evidence (read-only). Do not
run the detector, read the dataset pkl, load the joblib or the CSV, or modify any other
file. This turn writes only {rel(report_path)}.
""".strip()
    async with open_agent_session(sdk.client, options=options, connect_timeout_s=60) as client:
        async def receive() -> None:
            async for message in client.receive_response():
                if isinstance(message, sdk.assistant_message):
                    for block in message.content:
                        if isinstance(block, sdk.text_block) and verbose:
                            print(block.text, end="", flush=True)
                        elif sdk.tool_use_block and isinstance(block, sdk.tool_use_block):
                            print(f"[importance-agent] {getattr(block, 'name', 'tool')}", flush=True)
                elif isinstance(message, sdk.result_message):
                    cost = getattr(message, "total_cost_usd", None)
                    print("[importance-agent] turn completed"
                          + (f"; session cost ${cost:.4f}" if cost is not None else ""), flush=True)

        async def one_turn(prompt: str) -> None:
            await client.query(prompt)
            try:
                await asyncio.wait_for(receive(), timeout=AGENT_TIMEOUT_S)
            except asyncio.TimeoutError as exc:
                raise SystemExit(f"Importance summary timed out after {AGENT_TIMEOUT_S}s.") from exc

        await one_turn(instruction)
        try:
            check()
            return
        except SystemExit as exc:
            print(f"[importance-agent] report rejected; requesting one repair: {exc}", flush=True)
            repair = f"""
The driver check rejected {rel(report_path)} with:
{exc}
Repair that file in place: keep the BEGIN/END DRIVER-GENERATED IMPORTANCE EVIDENCE
block byte-for-byte, keep every pre-written heading exactly as given, replace every
remaining placeholder, stay within {MAX_WORDS} English words, and satisfy the stated
requirement. Modify only {rel(report_path)}.
""".strip()
        await one_turn(repair)
        check()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", type=Path, help="completed detector run directory")
    parser.add_argument("--application-dir", type=Path, default=None,
                        help="the deliverable directory holding feature_inventory.json "
                             "(default: the first directory under autodiscovery-application/)")
    parser.add_argument("--run-json", type=Path, default=None,
                        help="origin AutoDiscovery export for hypothesis texts "
                             "(default: autodiscovery/<application dir name>.json)")
    parser.add_argument("--no-agent", action="store_true",
                        help="write the evidence and the tables only; no SDK turn")
    parser.add_argument("--top-features", type=int, default=20)
    parser.add_argument("--top-groups", type=int, default=12)
    parser.add_argument("--permutation-repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    if args.application_dir is not None:
        result_dir = args.path.resolve()
        application_dir = args.application_dir.resolve()
        if not result_dir.is_dir() or not application_dir.is_dir():
            parser.error("path and --application-dir must be directories")
    else:
        result_dir, application_dir = resolve_result_dir(args.path)

    evidence = collect_importance_evidence(
        result_dir, application_dir, top_features=args.top_features,
        top_groups=args.top_groups, permutation_repeats=args.permutation_repeats,
        seed=args.seed, run_json=args.run_json)
    evidence_path = result_dir / EVIDENCE_NAME
    report_path = result_dir / REPORT_NAME
    log_path = result_dir / WORKFLOW_LOG_NAME
    write_json_atomic(evidence_path, evidence)
    evidence_sha = sha256(evidence_path)
    write_skeleton(report_path, evidence, evidence_sha)
    skeleton_sha = sha256(report_path)
    expected_block = driver_block(report_path)

    def check() -> None:
        validate_report(report_path, skeleton_sha=skeleton_sha,
                        expected_block=expected_block, evidence_sha=evidence_sha)

    original_out, original_err = sys.stdout, sys.stderr
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(f"# argv: {' '.join(sys.argv)}\n")
        handle.write(f"# started: {datetime.now():%Y-%m-%d %H:%M:%S}\n")
        handle.write(f"# host: {platform.node()} python: {platform.python_version()}\n")
        handle.flush()
        sys.stdout, sys.stderr = Tee(original_out, handle), Tee(original_err, handle)
        outcome = "FAILED"
        try:
            print(f"[importance] evidence: {rel(evidence_path)} sha256={evidence_sha}")
            top = evidence["ranked_groups"][:5]
            print("[importance] top hypothesis groups: " + ", ".join(
                f"h{g['hypothesis_id']} ({g[evidence['group_primary_view']]:.4f})" for g in top))
            if args.no_agent:
                fill_without_agent(report_path)
                check()
            else:
                asyncio.run(run_agent(result_dir, evidence_path, report_path,
                                      application_dir / "feature_inventory.json",
                                      args.verbose, check))
            outcome = "OK"
        except BaseException:
            traceback.print_exc()
            raise
        finally:
            sys.stdout, sys.stderr = original_out, original_err
            handle.write(f"# {outcome} after {time.monotonic() - started:.1f}s\n")
            handle.flush()
    print(f"Importance report: {rel(report_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
