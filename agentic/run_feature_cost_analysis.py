#!/usr/bin/env python3
"""Theory-driven feature-cost categorization for one assembled detector.

Where ``--measuretime`` OBSERVES seconds on a small deterministic sample, this
workflow asks the ``detector-feature-cost-analyst`` subagent to READ the
detector's feature implementation and classify every ANALYSIS_TIMING_GROUPS
entry into demanding / moderate / cheap from its algorithmic structure
(per-call full-array scans, subgraph copies, unbounded Dijkstra, global
pre-passes, memoization opportunities). The two views complement each other:
profiling misses cost that scales with structures the sample lacked; code
analysis misses constant factors. Neither loads a dataset.

Usage, from the project root::

    python agentic/run_feature_cost_analysis.py \
        autodiscovery-application/<RUN>/split_site_detector.py

PATH may also be the application directory itself when it contains exactly one
``*_site_detector.py``.

Deliverables, written beside the detector:

    feature_cost_analysis.json       validated per-group cost tiers
    FEATURE_COST_ANALYSIS.md         readable ranking, demanding tier first

The driver AST-parses the detector to extract the authoritative timing-group
keys, then validates the agent draft for exact coverage, closed tier
vocabulary, and per-record schema before compiling the public artifacts. The
agent never sees measured timings and cannot edit the detector.
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import hashlib
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    from agentic.detector_build.agent_session import load_claude_sdk, open_agent_session
except ImportError:  # direct ``python agentic/...`` execution
    from detector_build.agent_session import load_claude_sdk, open_agent_session


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DRAFT_NAME = ".feature_cost_analysis.draft.json"
ARTIFACT_NAME = "feature_cost_analysis.json"
REPORT_NAME = "FEATURE_COST_ANALYSIS.md"
COST_TIERS = ("demanding", "moderate", "cheap")
GROUP_FIELDS = {
    "key", "cost_tier", "per_row_complexity", "scaling_variables",
    "reason", "shared_computation_with", "optimization_note",
}

AGENT_MODEL = os.environ.get("FEATURE_COST_AGENT_MODEL", "claude-opus-5-5")
AGENT_EFFORT = os.environ.get("FEATURE_COST_AGENT_EFFORT", "high")
AGENT_TIMEOUT_S = int(os.environ.get("FEATURE_COST_TIMEOUT_S", "3600"))


def rel(path: Path) -> str:
    return str(path.resolve().relative_to(PROJECT_ROOT))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_detector(path: Path) -> Path:
    """Accept a detector script or an application directory holding one."""
    path = path.resolve()
    if path.is_file():
        if not path.name.endswith("_site_detector.py"):
            raise SystemExit(f"{path} is not a *_site_detector.py script.")
        return path
    if path.is_dir():
        candidates = sorted(path.glob("*_site_detector.py"))
        if len(candidates) != 1:
            raise SystemExit(
                f"{path} must contain exactly one *_site_detector.py "
                f"(found {len(candidates)})."
            )
        return candidates[0]
    raise SystemExit(f"{path} does not exist.")


def extract_timing_group_keys(detector_path: Path) -> list[str]:
    """AST-extract the authoritative ANALYSIS_TIMING_GROUPS keys."""
    tree = ast.parse(detector_path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "ANALYSIS_TIMING_GROUPS"
            for t in node.targets
        ):
            groups = ast.literal_eval(node.value)
            keys = [group["key"] for group in groups]
            if not keys or len(keys) != len(set(keys)):
                raise SystemExit(
                    f"{detector_path} has empty/duplicate timing group keys."
                )
            return keys
    raise SystemExit(f"{detector_path} defines no literal ANALYSIS_TIMING_GROUPS.")


def validate_draft(draft_path: Path, expected_keys: list[str]) -> dict:
    """Enforce exact coverage and the closed schema; return the parsed draft."""
    try:
        draft = json.loads(draft_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Cannot load agent draft {draft_path}: {exc}") from exc
    if not isinstance(draft, dict) or set(draft) != {
        "schema_version", "analysis_basis", "groups",
    }:
        raise SystemExit(
            "Draft must contain exactly schema_version, analysis_basis, groups."
        )
    if draft["schema_version"] != 1:
        raise SystemExit("Draft schema_version must be 1.")
    if not isinstance(draft["analysis_basis"], str) or not draft[
        "analysis_basis"
    ].strip():
        raise SystemExit("Draft needs a non-empty analysis_basis.")
    groups = draft["groups"]
    if not isinstance(groups, list):
        raise SystemExit("Draft groups must be a list.")
    seen: list[str] = []
    key_set = set(expected_keys)
    for index, record in enumerate(groups):
        if not isinstance(record, dict) or set(record) != GROUP_FIELDS:
            raise SystemExit(
                f"Group record {index} must contain exactly: "
                + ", ".join(sorted(GROUP_FIELDS))
            )
        key = record["key"]
        if key not in key_set:
            raise SystemExit(f"Group record {index} has unknown key {key!r}.")
        if key in seen:
            raise SystemExit(f"Group key {key!r} is duplicated.")
        seen.append(key)
        if record["cost_tier"] not in COST_TIERS:
            raise SystemExit(
                f"Group {key} cost_tier must be one of {COST_TIERS}."
            )
        for field in ("per_row_complexity", "reason"):
            if not isinstance(record[field], str) or not record[field].strip():
                raise SystemExit(f"Group {key} needs a non-empty {field}.")
        variables = record["scaling_variables"]
        if not isinstance(variables, list) or not all(
            isinstance(v, str) and v for v in variables
        ):
            raise SystemExit(f"Group {key} scaling_variables must be strings.")
        shared = record["shared_computation_with"]
        if not isinstance(shared, list) or not all(
            s in key_set and s != key for s in shared
        ):
            raise SystemExit(
                f"Group {key} shared_computation_with must name other known keys."
            )
        note = record["optimization_note"]
        if note is not None and (not isinstance(note, str) or not note.strip()):
            raise SystemExit(
                f"Group {key} optimization_note must be null or non-empty."
            )
    missing = key_set - set(seen)
    if missing:
        raise SystemExit(
            "Draft is missing timing groups: " + ", ".join(sorted(missing))
        )
    return draft


def write_artifacts(
    draft: dict, detector_path: Path, out_dir: Path,
) -> tuple[Path, Path]:
    artifact_path = out_dir / ARTIFACT_NAME
    artifact = {
        "schema_version": 1,
        "detector": rel(detector_path),
        "detector_sha256": sha256(detector_path),
        "generated": f"{datetime.now():%Y-%m-%d %H:%M:%S}",
        "method": (
            "code-analysis by detector-feature-cost-analyst; no measured "
            "timings were used"
        ),
        "analysis_basis": draft["analysis_basis"],
        "groups": draft["groups"],
    }
    artifact_path.write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    tier_rank = {tier: i for i, tier in enumerate(COST_TIERS)}
    ordered = sorted(
        draft["groups"], key=lambda r: (tier_rank[r["cost_tier"]], r["key"])
    )
    lines = [
        f"# Theory-driven feature cost analysis: {detector_path.name}",
        "",
        "Derived by reading the feature implementation (algorithmic structure),",
        "not by measuring seconds. Complements `--measuretime`, which observes a",
        "small sample and can miss cost that scales with structures the sample",
        "did not contain. Exclusion decisions should combine both views with",
        "each feature's scientific value.",
        "",
        f"Basis: {draft['analysis_basis']}",
        "",
        "| timing group | tier | per-row complexity | scales with | shared with | reason |",
        "|---|---|---|---|---|---|",
    ]
    for record in ordered:
        lines.append(
            "| {key} | {tier} | {complexity} | {variables} | {shared} | {reason} |".format(
                key=record["key"],
                tier=record["cost_tier"],
                complexity=record["per_row_complexity"],
                variables=", ".join(record["scaling_variables"]) or "—",
                shared=", ".join(record["shared_computation_with"]) or "—",
                reason=record["reason"].replace("|", "\\|"),
            )
        )
    notes = [
        f"- **{r['key']}**: {r['optimization_note']}"
        for r in ordered if r["optimization_note"]
    ]
    if notes:
        lines += ["", "## Semantics-preserving optimization opportunities", "", *notes]
    report_path = out_dir / REPORT_NAME
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return artifact_path, report_path


async def run_agent(
    detector_path: Path,
    inventory_path: Path | None,
    draft_path: Path,
    expected_keys: list[str],
    verbose: bool,
) -> None:
    """One analysis turn plus at most one validation-driven repair turn."""
    sdk = load_claude_sdk()
    options = sdk.agent_options(
        cwd=str(PROJECT_ROOT),
        setting_sources=["project"],
        allowed_tools=["Task", "Read", "Write", "Glob"],
        permission_mode="bypassPermissions",
        model=AGENT_MODEL,
        effort=AGENT_EFFORT,
    )
    inventory_note = (
        f"Read {rel(inventory_path)} (read-only) for what each feature measures."
        if inventory_path is not None
        else "No feature inventory is present; classify from the code alone."
    )
    instruction = f"""
Use the detector-feature-cost-analyst subagent to classify the computational
cost of every feature timing group in {rel(detector_path)} (read-only).
{inventory_note}
The authoritative timing group keys are exactly: {json.dumps(expected_keys)}.
Cover each exactly once. Classify from the code's algorithmic structure, not
from any measured timing artifact. Write exactly one JSON draft to
{rel(draft_path)} following your output contract, then return. Do not run the
detector, load a pkl, or modify any other file.
""".strip()

    async with open_agent_session(
        sdk.client, options=options, connect_timeout_s=60
    ) as client:
        async def receive() -> None:
            async for message in client.receive_response():
                if isinstance(message, sdk.assistant_message):
                    for block in message.content:
                        if isinstance(block, sdk.text_block) and verbose:
                            print(block.text, end="", flush=True)
                        elif sdk.tool_use_block and isinstance(block, sdk.tool_use_block):
                            print(f"[cost-agent] {getattr(block, 'name', 'tool')}",
                                  flush=True)
                elif isinstance(message, sdk.result_message):
                    cost = getattr(message, "total_cost_usd", None)
                    print(
                        "[cost-agent] turn completed"
                        + (f"; session cost ${cost:.4f}" if cost is not None else ""),
                        flush=True,
                    )

        async def one_turn(prompt: str) -> None:
            await client.query(prompt)
            try:
                await asyncio.wait_for(receive(), timeout=AGENT_TIMEOUT_S)
            except asyncio.TimeoutError as exc:
                raise SystemExit(
                    f"Feature cost analysis timed out after {AGENT_TIMEOUT_S}s."
                ) from exc

        await one_turn(instruction)
        try:
            validate_draft(draft_path, expected_keys)
            return
        except SystemExit as exc:
            print(f"[cost-agent] draft rejected; requesting one repair: {exc}",
                  flush=True)
            repair = f"""
The driver validator rejected {rel(draft_path)} with:
{exc}
Repair that file in place to satisfy the output contract (exact key coverage,
closed tier vocabulary, exact per-record fields). Modify only {rel(draft_path)}.
""".strip()
        await one_turn(repair)
        validate_draft(draft_path, expected_keys)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path", type=Path,
        help="*_site_detector.py or the application directory containing one.",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    detector_path = resolve_detector(args.path)
    out_dir = detector_path.parent
    expected_keys = extract_timing_group_keys(detector_path)
    inventory_path = out_dir / "feature_inventory.json"
    inventory = inventory_path if inventory_path.is_file() else None
    draft_path = out_dir / DRAFT_NAME

    print(f"[driver] detector: {rel(detector_path)}")
    print(f"[driver] timing groups: {len(expected_keys)}")
    started = time.monotonic()
    asyncio.run(
        run_agent(detector_path, inventory, draft_path, expected_keys,
                  args.verbose)
    )
    draft = validate_draft(draft_path, expected_keys)
    artifact_path, report_path = write_artifacts(draft, detector_path, out_dir)
    draft_path.unlink(missing_ok=True)

    tiers = {tier: 0 for tier in COST_TIERS}
    for record in draft["groups"]:
        tiers[record["cost_tier"]] += 1
    print(f"[driver] tiers: {tiers}")
    print(f"[driver] wrote {rel(artifact_path)} and {rel(report_path)} "
          f"in {time.monotonic() - started:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
