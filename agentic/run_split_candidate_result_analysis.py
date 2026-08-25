#!/usr/bin/env python3
"""Generate a grounded English AI review for one split candidate-pool sweep.

The driver deterministically collects numeric evidence from a completed
``mcl<value>_<signature>`` bundle, writes an immutable report skeleton, and asks
the ``split-candidate-result-analyst`` filesystem subagent to fill the prose.

Usage from the project root::

    python agentic/run_split_candidate_result_analysis.py \
        notebooks/split_candidate_pool_sweep_outputs/mcl100_<signature>

Outputs are written inside the bundle:

    ai_review_evidence.json
    AI_REVIEW.md
    ai_review_workflow.log.txt
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

try:
    from agentic.detector_build.agent_session import load_claude_sdk, open_agent_session
except ImportError:  # direct ``python agentic/...`` execution
    from detector_build.agent_session import load_claude_sdk, open_agent_session


PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVIDENCE_NAME = "ai_review_evidence.json"
REPORT_NAME = "AI_REVIEW.md"
LOG_NAME = "ai_review_workflow.log.txt"
REVIEW_OWNED_FILES = {EVIDENCE_NAME, REPORT_NAME, LOG_NAME, "artifact_manifest.json"}
BLOCK_START = "<!-- BEGIN DRIVER-GENERATED AI REVIEW EVIDENCE — preserve this block -->"
BLOCK_END = "<!-- END DRIVER-GENERATED AI REVIEW EVIDENCE -->"
PLACEHOLDER = "<!-- SPLIT-CANDIDATE-ANALYST MUST REPLACE THIS PLACEHOLDER -->"
HEADINGS = (
    "## Executive summary",
    "## Experimental scope and integrity",
    "## Cross-dataset findings",
    "## Dataset-specific observations",
    "## Recall-cost trade-offs",
    "## Direct versus gap truth",
    "## Limitations and next experiment",
)
MODEL = os.environ.get("SPLIT_CANDIDATE_REVIEW_MODEL", "claude-opus-4-8")
EFFORT = os.environ.get("SPLIT_CANDIDATE_REVIEW_EFFORT", "xhigh")
TIMEOUT_S = int(os.environ.get("SPLIT_CANDIDATE_REVIEW_TIMEOUT_S", "3600"))
MODE_ALIASES = {
    "leaf_leaf": "tip_to_tip",
    "leaf_any": "tip_to_any_node",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def resolve_run_dir(path: Path) -> Path:
    run_dir = (path if path.is_absolute() else PROJECT_ROOT / path).resolve()
    if not run_dir.is_dir():
        raise SystemExit(f"Candidate-pool run directory does not exist: {run_dir}")
    required = (
        run_dir / "run_summary.json",
        run_dir / "tables" / "all_dataset_configurations.csv",
        run_dir / "tables" / "cross_dataset_aggregate.csv",
        run_dir / "tables" / "current_detector_baseline.csv",
        run_dir / "tables" / "dataset_summary.csv",
        run_dir / "tables" / "recommended_minimal_pools.csv",
    )
    missing = [str(item) for item in required if not item.is_file()]
    if missing:
        raise SystemExit("Incomplete candidate-pool bundle; missing: " + ", ".join(missing))
    summary = load_json(run_dir / "run_summary.json")
    expected_name = f"mcl{summary.get('mcl')}_{summary.get('config_signature')}"
    if run_dir.name != expected_name:
        raise SystemExit(
            f"Run directory name {run_dir.name!r} does not match run_summary "
            f"identity {expected_name!r}"
        )
    return run_dir


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SystemExit(f"Expected a JSON object in {path}")
    return value


def canonical_config_id(value: Any) -> str:
    result = str(value)
    for old, new in MODE_ALIASES.items():
        result = result.replace(old, new)
    return result


def normalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    rename = {}
    if "per_tip_k" in frame.columns and "per_anchor_k" not in frame.columns:
        rename["per_tip_k"] = "per_anchor_k"
    if "per_tip_k_value" in frame.columns and "per_anchor_k_value" not in frame.columns:
        rename["per_tip_k_value"] = "per_anchor_k_value"
    frame = frame.rename(columns=rename)
    if "mode" not in frame.columns or "per_anchor_k" not in frame.columns:
        raise SystemExit("Sweep table lacks mode/per-anchor-k columns")
    frame["mode"] = frame["mode"].astype(str).replace(MODE_ALIASES)
    frame["per_anchor_k"] = frame["per_anchor_k"].astype(str)
    frame["global_cap"] = frame["global_cap"].astype(str)
    if "config_id" in frame.columns:
        frame["config_id"] = frame["config_id"].map(canonical_config_id)
    return frame


def scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def record(row: pd.Series, columns: tuple[str, ...]) -> dict[str, Any]:
    return {column: scalar(row[column]) for column in columns if column in row.index}


CONFIG_COLUMNS = (
    "config_id",
    "mode",
    "radius_um",
    "per_anchor_k",
    "global_cap",
)
AGG_METRICS = (
    "datasets",
    "total_candidates",
    "mean_candidates",
    "min_recall",
    "mean_recall",
    "micro_recall",
    "min_direct_recall",
    "min_gap_recall",
    "mean_prevalence",
)
BRAIN_METRICS = (
    "brain",
    "n_reachable_truth_pairs",
    "n_candidate_pairs",
    "n_candidate_truth_pairs",
    "candidate_recall",
    "direct_candidate_recall",
    "gap_candidate_recall",
    "candidate_prevalence",
)


def match_config(frame: pd.DataFrame, config: pd.Series) -> pd.DataFrame:
    return frame[
        (frame["mode"] == str(config["mode"]))
        & np.isclose(frame["radius_um"].astype(float), float(config["radius_um"]))
        & (frame["per_anchor_k"].astype(str) == str(config["per_anchor_k"]))
        & (frame["global_cap"].astype(str) == str(config["global_cap"]))
    ]


def verify_existing_manifest(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "artifact_manifest.json"
    if not path.is_file():
        return {"status": "not_available_before_review", "artifacts_checked": 0}
    manifest = load_json(path)
    failures = []
    checked = 0
    for item in manifest.get("artifacts", []):
        artifact = run_dir / str(item.get("path", ""))
        if not artifact.is_file():
            failures.append({"path": str(item.get("path")), "reason": "missing"})
            continue
        checked += 1
        if artifact.stat().st_size != int(item.get("size_bytes", -1)):
            failures.append({"path": str(item.get("path")), "reason": "size_mismatch"})
        elif sha256(artifact) != item.get("sha256"):
            failures.append({"path": str(item.get("path")), "reason": "sha256_mismatch"})
    return {
        "status": "pass" if not failures else "fail",
        "artifacts_checked": checked,
        "failures": failures,
    }


def collect_evidence(run_dir: Path) -> dict[str, Any]:
    summary = load_json(run_dir / "run_summary.json")
    if summary.get("result_status") != "complete":
        raise SystemExit("run_summary.json does not mark this run complete")

    tables = run_dir / "tables"
    all_results = normalize_frame(
        pd.read_csv(tables / "all_dataset_configurations.csv", keep_default_na=False)
    )
    aggregate = normalize_frame(
        pd.read_csv(tables / "cross_dataset_aggregate.csv", keep_default_na=False)
    )
    baseline = normalize_frame(
        pd.read_csv(tables / "current_detector_baseline.csv", keep_default_na=False)
    )
    dataset_summary = pd.read_csv(tables / "dataset_summary.csv", keep_default_na=False)
    recommendations = normalize_frame(
        pd.read_csv(tables / "recommended_minimal_pools.csv", keep_default_na=False)
    )

    brains = sorted(str(value) for value in all_results["brain"].unique())
    summary_brains = sorted(str(value) for value in summary.get("brains", []))
    if brains != summary_brains:
        raise SystemExit(
            f"Dataset mismatch: run_summary has {summary_brains}, tables have {brains}"
        )
    checkpoint_brains = sorted(
        path.name
        for path in (run_dir / "per_dataset").iterdir()
        if path.is_dir() and (path / "sweep.csv").is_file()
    )
    if brains != checkpoint_brains:
        raise SystemExit(
            f"Dataset mismatch: tables have {brains}, checkpoints have {checkpoint_brains}"
        )
    for brain in brains:
        metadata = load_json(run_dir / "per_dataset" / brain / "metadata.json")
        if str(metadata.get("brain")) != brain:
            raise SystemExit(f"Checkpoint metadata brain mismatch for {brain}")
        if metadata.get("config_signature") != summary.get("config_signature"):
            raise SystemExit(f"Checkpoint configuration signature mismatch for {brain}")
    modes = sorted(str(value) for value in all_results["mode"].unique())
    global_cap_values = sorted(str(value) for value in all_results["global_cap"].unique())
    config = dict(summary.get("config", {}))
    original_modes = [str(value) for value in config.get("modes", [])]
    canonical_modes = [MODE_ALIASES.get(value, value) for value in original_modes]
    config["modes_original"] = original_modes
    config["modes_canonical"] = canonical_modes
    if "per_tip_k" in config and "per_anchor_k" not in config:
        config["per_anchor_k"] = config["per_tip_k"]

    baseline_aggregate = None
    if not baseline.empty:
        baseline_config = baseline.iloc[0]
        matched = match_config(aggregate, baseline_config)
        if len(matched) == 1:
            baseline_aggregate = record(
                matched.iloc[0], CONFIG_COLUMNS + AGG_METRICS
            )

    recommendation_rows = []
    for _, recommendation in recommendations.iterrows():
        matched_aggregate = match_config(aggregate, recommendation)
        matched_brains = match_config(all_results, recommendation).sort_values("brain")
        recommendation_rows.append(
            {
                "target_worst_brain_recall": scalar(
                    recommendation.get("target_worst_brain_recall")
                ),
                "status": str(recommendation.get("status")),
                "configuration": record(
                    recommendation, CONFIG_COLUMNS + AGG_METRICS
                ),
                "aggregate": (
                    record(matched_aggregate.iloc[0], CONFIG_COLUMNS + AGG_METRICS)
                    if len(matched_aggregate) == 1
                    else None
                ),
                "per_dataset": [
                    record(row, BRAIN_METRICS) for _, row in matched_brains.iterrows()
                ],
            }
        )

    best_by_cap = []
    for cap, group in aggregate.groupby("global_cap", sort=False):
        best = group.sort_values(
            ["min_recall", "total_candidates"], ascending=[False, True]
        ).iloc[0]
        best_by_cap.append(record(best, CONFIG_COLUMNS + AGG_METRICS))

    best_by_mode = []
    for mode, group in aggregate.groupby("mode", sort=False):
        best = group.sort_values(
            ["min_recall", "total_candidates"], ascending=[False, True]
        ).iloc[0]
        best_by_mode.append(record(best, CONFIG_COLUMNS + AGG_METRICS))

    tip_any = aggregate[
        (aggregate["mode"] == "tip_to_any_node")
        & (aggregate["global_cap"] == "none")
    ]
    radius_series = []
    k_series = []
    if not tip_any.empty:
        k2 = tip_any[tip_any["per_anchor_k"] == "2"].sort_values("radius_um")
        radius_series = [record(row, CONFIG_COLUMNS + AGG_METRICS) for _, row in k2.iterrows()]
        max_radius = float(tip_any["radius_um"].max())
        at_max = tip_any[np.isclose(tip_any["radius_um"], max_radius)].copy()
        at_max["_k_sort"] = pd.to_numeric(at_max["per_anchor_k"], errors="coerce")
        at_max = at_max.sort_values("_k_sort")
        k_series = [record(row, CONFIG_COLUMNS + AGG_METRICS) for _, row in at_max.iterrows()]

    figures = []
    for path in sorted(run_dir.rglob("*.png")):
        figures.append(
            {
                "path": str(path.relative_to(run_dir)),
                "basename": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )

    return {
        "evidence_schema": 1,
        "evidence_for_run_created_at_utc": summary.get("created_at_utc"),
        "run": {
            "directory": str(run_dir),
            "result_status": summary.get("result_status"),
            "created_at_utc": summary.get("created_at_utc"),
            "elapsed_seconds": scalar(summary.get("elapsed_seconds")),
            "mcl": scalar(summary.get("mcl")),
            "config_signature": summary.get("config_signature"),
            "brains": brains,
            "config": config,
            "evaluated_pairing_rules": modes,
            "global_cap_values": global_cap_values,
            "global_cap_swept": len(global_cap_values) > 1,
            "any_node_to_any_node_evaluated": "any_node_to_any_node" in modes,
            "pre_review_manifest_integrity": verify_existing_manifest(run_dir),
        },
        "definitions": {
            "candidate_recall": "Pre-feature-scoring enumeration ceiling over reachable truth segment pairs.",
            "per_anchor_k": "Maximum distinct partner segments retained per anchor; it is not a node count.",
            "candidate_prevalence": "Truth-pair fraction inside the candidate pool; it is not classifier precision.",
            "global_cap": "Post-enumeration closest-pair truncation; it does not avoid neighborhood search.",
            "truth_kind_overlap": "Direct and gap truth sets may overlap, so missed counts are not additive.",
        },
        "baseline": {
            "aggregate": baseline_aggregate,
            "per_dataset": [record(row, BRAIN_METRICS) for _, row in baseline.sort_values("brain").iterrows()],
        },
        "recommendations": recommendation_rows,
        "dataset_best": [
            {
                **record(
                    row,
                    (
                        "brain",
                        "n_reachable_truth_pairs",
                        "best_candidate_recall",
                        "fewest_candidates_at_best_recall",
                    ),
                ),
                "minimal_best_config_id": canonical_config_id(
                    row.get("minimal_best_config_id")
                ),
            }
            for _, row in dataset_summary.sort_values("brain").iterrows()
        ],
        "best_by_global_cap": best_by_cap,
        "best_by_pairing_rule": best_by_mode,
        "tip_to_any_node_radius_series_at_k2": radius_series,
        "tip_to_any_node_k_series_at_max_radius": k_series,
        "figures": figures,
        "source_files": {
            name: {
                "path": str(path.relative_to(run_dir)),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for name, path in {
                "run_summary": run_dir / "run_summary.json",
                "all_configurations": tables / "all_dataset_configurations.csv",
                "aggregate": tables / "cross_dataset_aggregate.csv",
                "baseline": tables / "current_detector_baseline.csv",
                "dataset_summary": tables / "dataset_summary.csv",
                "recommendations": tables / "recommended_minimal_pools.csv",
            }.items()
        },
    }


def fmt_percent(value: Any) -> str:
    return "N/A" if value is None else f"{100 * float(value):.2f}%"


def fmt_int(value: Any) -> str:
    return "N/A" if value is None else f"{int(round(float(value))):,}"


def evidence_block(evidence: dict[str, Any], evidence_sha: str) -> str:
    run = evidence["run"]
    lines = [
        BLOCK_START,
        "This block is deterministic evidence, not AI-authored interpretation.",
        f"Evidence file: `{EVIDENCE_NAME}` (`sha256={evidence_sha}`)",
        "",
        "| Run fact | Value |",
        "|---|---|",
        f"| Status | `{run['result_status']}` |",
        f"| Configuration signature | `{run['config_signature']}` |",
        f"| MCL | `{run['mcl']}` |",
        f"| Completed datasets | {', '.join(run['brains'])} |",
        f"| Evaluated pairing rules | {', '.join(f'`{x}`' for x in run['evaluated_pairing_rules'])} |",
        f"| `any_node_to_any_node` evaluated | `{str(run['any_node_to_any_node_evaluated']).lower()}` |",
        f"| Global-cap values | {', '.join(f'`{x}`' for x in run['global_cap_values'])} |",
        f"| Global cap swept | `{str(run['global_cap_swept']).lower()}` |",
        f"| Pre-review manifest integrity | `{run['pre_review_manifest_integrity']['status']}` |",
        "",
        "| Recall target | Status | Configuration | Worst recall | Mean recall | Micro recall | Total candidates |",
        "|---:|---|---|---:|---:|---:|---:|",
    ]
    for item in evidence["recommendations"]:
        config = item["aggregate"] or item["configuration"]
        lines.append(
            "| "
            + " | ".join(
                [
                    fmt_percent(item["target_worst_brain_recall"]),
                    f"`{item['status']}`",
                    f"`{config.get('config_id')}`",
                    fmt_percent(config.get("min_recall")),
                    fmt_percent(config.get("mean_recall")),
                    fmt_percent(config.get("micro_recall")),
                    fmt_int(config.get("total_candidates")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "| Dataset | Reachable truth pairs | Best candidate recall | Fewest candidates at that recall | Best configuration |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for item in evidence["dataset_best"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(item["brain"]),
                    fmt_int(item.get("n_reachable_truth_pairs")),
                    fmt_percent(item.get("best_candidate_recall")),
                    fmt_int(item.get("fewest_candidates_at_best_recall")),
                    f"`{item.get('minimal_best_config_id')}`",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Definitions: candidate recall is a pre-scoring enumeration ceiling; candidate prevalence is not classifier precision; direct and gap truth sets may overlap.",
            BLOCK_END,
        ]
    )
    return "\n".join(lines)


def write_skeleton(path: Path, block: str) -> None:
    sections = "\n\n".join(f"{heading}\n\n{PLACEHOLDER}" for heading in HEADINGS)
    atomic_write(
        path,
        f"# AI Review — Split Candidate-Pool Sweep\n\n{block}\n\n{sections}\n",
    )


def validate_report(path: Path, expected_block: str, evidence: dict[str, Any]) -> None:
    if not path.is_file():
        raise SystemExit(f"AI analyst did not write {path}")
    text = path.read_text(encoding="utf-8")
    if PLACEHOLDER in text:
        raise SystemExit("AI review still contains an unreplaced placeholder")
    start = text.find(BLOCK_START)
    end = text.find(BLOCK_END)
    if start < 0 or end < start:
        raise SystemExit("AI review lost the immutable evidence block")
    actual_block = text[start : end + len(BLOCK_END)]
    if actual_block != expected_block:
        raise SystemExit("AI review modified the immutable evidence block")
    positions = []
    for heading in HEADINGS:
        if text.count(heading) != 1:
            raise SystemExit(f"AI review must contain heading exactly once: {heading}")
        positions.append(text.index(heading))
    if positions != sorted(positions):
        raise SystemExit("AI review section headings are out of order")
    prose = text[end + len(BLOCK_END) :]
    if len(prose.split()) < 450:
        raise SystemExit("AI review is too short to satisfy the analysis contract")
    for brain in evidence["run"]["brains"]:
        if str(brain) not in prose:
            raise SystemExit(f"AI review prose does not discuss dataset {brain}")
    for figure in evidence["figures"]:
        if figure["path"] not in prose:
            raise SystemExit(
                f"AI review prose does not mention figure path {figure['path']}"
            )


def source_snapshot(run_dir: Path) -> dict[str, str]:
    """Hash scientific artifacts that the review process does not own."""

    return {
        str(path.relative_to(run_dir)): sha256(path)
        for path in sorted(item for item in run_dir.rglob("*") if item.is_file())
        if path.name not in REVIEW_OWNED_FILES
    }


def validate_source_snapshot(run_dir: Path, expected: dict[str, str]) -> None:
    actual = source_snapshot(run_dir)
    if actual != expected:
        changed = sorted(
            path
            for path in set(expected) | set(actual)
            if expected.get(path) != actual.get(path)
        )
        raise SystemExit(
            "AI review modified or added non-review artifacts: " + ", ".join(changed)
        )


class Tee:
    def __init__(self, stream, handle):
        self.stream = stream
        self.handle = handle

    def write(self, text):
        self.stream.write(text)
        self.handle.write(text)
        self.flush()
        return len(text)

    def flush(self):
        self.stream.flush()
        self.handle.flush()

    def isatty(self):
        return False


async def run_agent(
    run_dir: Path,
    evidence_path: Path,
    report_path: Path,
    validate: Callable[[], None],
    verbose: bool,
) -> None:
    if importlib.util.find_spec("claude_agent_sdk") is None:
        await asyncio.to_thread(
            run_agent_cli,
            run_dir,
            evidence_path,
            report_path,
            validate,
            verbose,
        )
        return

    sdk = load_claude_sdk()
    options = sdk.agent_options(
        cwd=str(PROJECT_ROOT),
        setting_sources=["project"],
        allowed_tools=["Task", "Read", "Write", "Edit", "Glob"],
        permission_mode="bypassPermissions",
        model=MODEL,
        effort=EFFORT,
    )
    instruction = f"""
Use the split-candidate-result-analyst subagent to review the completed split
candidate-pool bundle {relative(run_dir)}. Read {relative(evidence_path)} as the
numeric source of truth and visually inspect every PNG listed there. Fill the
existing skeleton {relative(report_path)} in place. Preserve its driver-generated
evidence block byte-for-byte, keep every heading exactly, and replace every
placeholder. Write a concise English scientific report. Do not load source pkl
caches, rerun candidate generation, or modify any other file. This turn writes
only {relative(report_path)}.
""".strip()

    async with open_agent_session(
        sdk.client, options=options, connect_timeout_s=60
    ) as client:
        async def receive() -> None:
            async for message in client.receive_response():
                if isinstance(message, sdk.assistant_message):
                    for content in message.content:
                        if isinstance(content, sdk.text_block) and verbose:
                            print(content.text, end="", flush=True)
                        elif sdk.tool_use_block and isinstance(content, sdk.tool_use_block):
                            print(f"[split-candidate-review] {getattr(content, 'name', 'tool')}", flush=True)
                elif isinstance(message, sdk.result_message):
                    cost = getattr(message, "total_cost_usd", None)
                    suffix = f"; session cost ${cost:.4f}" if cost is not None else ""
                    print(f"[split-candidate-review] turn completed{suffix}", flush=True)

        async def one_turn(prompt: str) -> None:
            await client.query(prompt)
            try:
                await asyncio.wait_for(receive(), timeout=TIMEOUT_S)
            except asyncio.TimeoutError as exc:
                raise SystemExit(f"AI review timed out after {TIMEOUT_S}s") from exc

        await one_turn(instruction)
        try:
            validate()
            return
        except SystemExit as exc:
            print(f"[split-candidate-review] report rejected; requesting one repair: {exc}", flush=True)
            repair = f"""
The driver rejected {relative(report_path)} with: {exc}
Repair only that file. Preserve the evidence block byte-for-byte, keep all seven
headings exactly once and in order, replace every placeholder, discuss every
dataset, and mention every listed PNG relative path in the prose.
""".strip()
        await one_turn(repair)
        validate()


def run_agent_cli(
    run_dir: Path,
    evidence_path: Path,
    report_path: Path,
    validate: Callable[[], None],
    verbose: bool,
) -> None:
    """Use the installed Claude CLI when the optional Python SDK is absent."""

    executable = shutil.which("claude")
    if executable is None:
        raise SystemExit(
            "Neither claude_agent_sdk nor the claude CLI is available for AI review"
        )

    instruction = f"""
Review the completed split candidate-pool bundle {relative(run_dir)}. Read
{relative(evidence_path)} as the numeric source of truth and visually inspect
every PNG listed there. Fill the existing skeleton {relative(report_path)} in
place. Preserve its driver-generated evidence block byte-for-byte, keep every
heading exactly, replace every placeholder, and write a concise English
scientific report. Do not load source pkl caches, rerun candidate generation,
or modify any other file. Write only {relative(report_path)}.
""".strip()

    def one_turn(prompt: str) -> None:
        command = [
            executable,
            "--print",
            "--agent",
            "split-candidate-result-analyst",
            "--setting-sources",
            "project",
            "--permission-mode",
            "bypassPermissions",
            "--allowedTools",
            "Read,Write,Edit,Glob",
            "--model",
            MODEL,
            "--effort",
            EFFORT,
            "--no-session-persistence",
            prompt,
        ]
        try:
            completed = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                text=True,
                capture_output=True,
                timeout=TIMEOUT_S,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise SystemExit(f"AI review timed out after {TIMEOUT_S}s") from exc
        if verbose and completed.stdout:
            print(completed.stdout, end="" if completed.stdout.endswith("\n") else "\n")
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "unknown CLI failure").strip()
            raise SystemExit(
                f"Claude CLI review failed with exit code {completed.returncode}: {detail}"
            )
        print("[split-candidate-review] Claude CLI turn completed", flush=True)

    one_turn(instruction)
    try:
        validate()
        return
    except SystemExit as exc:
        print(
            f"[split-candidate-review] report rejected; requesting one repair: {exc}",
            flush=True,
        )
        repair = f"""
Repair only {relative(report_path)}. The driver rejected it with: {exc}
Preserve the evidence block byte-for-byte, keep all seven headings exactly once
and in order, replace every placeholder, discuss every dataset, and mention
every listed PNG relative path in the prose. Read {relative(evidence_path)} for
all numeric facts and modify no other file.
""".strip()
    one_turn(repair)
    validate()


def write_manifest(run_dir: Path) -> Path:
    path = run_dir / "artifact_manifest.json"
    artifacts = []
    for artifact in sorted(item for item in run_dir.rglob("*") if item.is_file()):
        if artifact == path:
            continue
        suffix = artifact.suffix.lower()
        role = "figure" if suffix == ".png" else "table" if suffix == ".csv" else "metadata"
        if suffix == ".md":
            role = "report"
        elif suffix == ".py":
            role = "source"
        artifacts.append(
            {
                "path": str(artifact.relative_to(run_dir)),
                "role": role,
                "size_bytes": artifact.stat().st_size,
                "sha256": sha256(artifact),
            }
        )
    atomic_write(path, json.dumps({"artifacts": artifacts}, indent=2) + "\n")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="completed candidate-pool run bundle")
    parser.add_argument("--prepare-only", action="store_true", help="write evidence and skeleton without calling an AI agent")
    parser.add_argument("--force", action="store_true", help="regenerate an already-current AI review")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    run_dir = resolve_run_dir(args.path)
    evidence = collect_evidence(run_dir)
    evidence_path = run_dir / EVIDENCE_NAME
    report_path = run_dir / REPORT_NAME
    evidence_text = json.dumps(evidence, indent=2, allow_nan=False) + "\n"
    evidence_sha = hashlib.sha256(evidence_text.encode("utf-8")).hexdigest()
    block = evidence_block(evidence, evidence_sha)

    if not args.force and evidence_path.is_file() and report_path.is_file():
        if evidence_path.read_text(encoding="utf-8") == evidence_text:
            try:
                validate_report(report_path, block, evidence)
            except SystemExit:
                pass
            else:
                print(f"AI review is already current: {relative(report_path)}")
                write_manifest(run_dir)
                return 0

    atomic_write(evidence_path, evidence_text)
    write_skeleton(report_path, block)
    if args.prepare_only:
        write_manifest(run_dir)
        print(f"Prepared AI review evidence and skeleton: {relative(report_path)}")
        return 0

    scientific_snapshot = source_snapshot(run_dir)
    log_path = run_dir / LOG_NAME
    original_out, original_err = sys.stdout, sys.stderr
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(f"# argv: {' '.join(sys.argv)}\n")
        handle.write(f"# started: {datetime.now():%Y-%m-%d %H:%M:%S}\n")
        handle.write(f"# host: {platform.node()} python: {platform.python_version()}\n")
        handle.flush()
        sys.stdout = Tee(original_out, handle)
        sys.stderr = Tee(original_err, handle)
        outcome = "FAILED"
        try:
            print(f"[split-candidate-review] evidence sha256={evidence_sha}")

            def validate() -> None:
                validate_report(report_path, block, evidence)

            asyncio.run(run_agent(run_dir, evidence_path, report_path, validate, args.verbose))
            validate_source_snapshot(run_dir, scientific_snapshot)
            outcome = "OK"
        except BaseException:
            traceback.print_exc()
            raise
        finally:
            sys.stdout, sys.stderr = original_out, original_err
            handle.write(f"# {outcome} after {time.monotonic() - started:.1f}s\n")
            handle.flush()

    write_manifest(run_dir)
    print(f"AI review report: {relative(report_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
