#!/usr/bin/env python3
"""Generate a grounded bilingual Markdown analysis for one completed detector result.

The positional PATH may be an ``autodiscovery-application/<RUN>/`` directory or
one of its nested result directories such as ``runs/<brain>-with-figures``. The
driver validates a complete detector run, deterministically extracts numeric
evidence from JSON/CSV/log metadata, hashes every relevant artifact, writes an
immutable evidence block, and delegates only figure interpretation to the
``detector-result-analyst`` subagent.

Usage, from the project root::

    python agentic/run_detector_result_analysis.py \
        autodiscovery-application/<RUN>/runs/<COMPLETED-RESULT>

The deliverables are written inside PATH:

    result_analysis_evidence.json
    RESULT_ANALYSIS.md  (complete English, then complete Chinese translation)
    result_analysis_workflow.log.txt

The factual evidence is reproducible for identical inputs. Agent prose can vary
slightly across invocations and must never override the immutable evidence.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import math
import os
import platform
import re
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

try:
    from agentic.detector_build.agent_session import load_claude_sdk, open_agent_session
except ImportError:  # direct ``python agentic/...`` execution
    from detector_build.agent_session import load_claude_sdk, open_agent_session


PROJECT_ROOT = Path(__file__).resolve().parent.parent
APPLICATION_ROOT = PROJECT_ROOT / "autodiscovery-application"
EVIDENCE_NAME = "result_analysis_evidence.json"
REPORT_NAME = "RESULT_ANALYSIS.md"
WORKFLOW_LOG_NAME = "result_analysis_workflow.log.txt"
BLOCK_START = "<!-- BEGIN DRIVER-GENERATED RESULT EVIDENCE — preserve this block -->"
BLOCK_END = "<!-- END DRIVER-GENERATED RESULT EVIDENCE -->"
ENGLISH_REPORT_HEADING = "# Part I — English Report"
CHINESE_REPORT_HEADING = "# 第二部分——中文报告"
ENGLISH_SECTION_HEADINGS = (
    "## Executive summary",
    "## Model comparison and selection",
    "## Manual-review workload",
    "## Figure-by-figure interpretation",
    "## Missingness and confounding",
    "## Feature interpretation",
    "## Limitations and next steps",
)
CHINESE_SECTION_HEADINGS = (
    "## 结论摘要",
    "## 模型比较与选择",
    "## 人工复核工作量",
    "## 逐图解读",
    "## 缺失性与混杂",
    "## 特征解释",
    "## 限制与下一步",
)
REPORT_PLACEHOLDER = "<!-- RESULT-ANALYST MUST REPLACE THIS PLACEHOLDER -->"
AGENT_MODEL = os.environ.get("DETECTOR_RESULT_ANALYSIS_AGENT_MODEL", "claude-opus-4-8")
AGENT_EFFORT = os.environ.get("DETECTOR_RESULT_ANALYSIS_AGENT_EFFORT", "xhigh")
AGENT_TIMEOUT_S = int(os.environ.get("DETECTOR_RESULT_ANALYSIS_TIMEOUT_S", "10800"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def resolve_result_dir(path: Path) -> tuple[Path, Path]:
    result_dir = (path if path.is_absolute() else PROJECT_ROOT / path).resolve()
    if not result_dir.is_dir():
        raise SystemExit(f"Detector result path is not a directory: {result_dir}")
    try:
        relative = result_dir.relative_to(APPLICATION_ROOT.resolve())
    except ValueError as exc:
        raise SystemExit(
            f"Detector result path must be under {APPLICATION_ROOT.resolve()}: "
            f"{result_dir}"
        ) from exc
    if not relative.parts:
        raise SystemExit("Pass one application/result subdirectory, not autodiscovery-application itself.")
    application_dir = APPLICATION_ROOT.resolve() / relative.parts[0]
    return result_dir, application_dir


def exactly_one(directory: Path, pattern: str, label: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if len(matches) != 1:
        raise SystemExit(
            f"Expected exactly one {label} matching {pattern!r} in {directory}; "
            f"found {len(matches)}."
        )
    return matches[0]


def load_json_object(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SystemExit(f"Expected a JSON object in {path}.")
    return value


def _float(value: str, label: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"CSV has invalid {label}: {value!r}") from exc


def _oof_score(value: str | None, label: str) -> float:
    """Parse an OOF score, accepting blank/NaN as deliberately unscored."""
    if value is None or not str(value).strip():
        return float("nan")
    parsed = _float(value, label)
    if math.isinf(parsed):
        raise SystemExit(f"CSV has infinite {label}: {value!r}")
    return parsed


def _result_contract(model: dict) -> dict:
    target = model.get("detector_target") or "merge_detection"
    if target == "split_detection":
        return {
            "target": target,
            "output_prefix": "split_detector",
            "detector_name": "split_site_detector.py",
            "label": "is_split",
            "score": "split_probability_oof",
            "identity": ("candidate_id", "segment_id_a", "segment_id_b"),
            "positive": "splits",
            "negative": "non-split candidates",
            "row_unit": "candidate segment pair",
        }
    if target != "merge_detection":
        raise SystemExit(f"Unsupported detector_target in model JSON: {target!r}")
    return {
        "target": target,
        "output_prefix": "merge_detector",
        "detector_name": "merge_site_detector.py",
        "label": "is_merge",
        "score": "merge_probability_oof",
        "identity": ("segment_id",),
        "positive": "merges",
        "negative": "clean segments",
        "row_unit": "segment",
    }


def summarize_csv(
    csv_path: Path,
    review_budgets: tuple[int, ...],
    contract: dict | None = None,
) -> dict:
    contract = contract or _result_contract({})
    rows: list[tuple[float, str, int, int]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = set(contract["identity"]) | {contract["label"], contract["score"]}
        missing = sorted(required - set(reader.fieldnames or ()))
        if missing:
            raise SystemExit(f"CSV {csv_path} is missing required columns: {missing}")
        for row in reader:
            label_raw = _float(row[contract["label"]], contract["label"])
            if label_raw not in (0.0, 1.0):
                raise SystemExit(
                    f"CSV has non-binary {contract['label']}: {label_raw!r}")
            identity = ":".join(str(row[name]) for name in contract["identity"])
            rows.append((
                _oof_score(row[contract["score"]], contract["score"]),
                identity,
                int(label_raw),
                int(_float(row.get("is_merge_creating", "0"),
                           "is_merge_creating")),
            ))
    if not rows:
        raise SystemExit(f"CSV {csv_path} has no data rows.")
    n_positive = sum(item[2] for item in rows)
    if n_positive == 0:
        raise SystemExit(f"CSV {csv_path} has no positive {contract['positive']} rows.")
    scored_rows = [item for item in rows if math.isfinite(item[0])]
    if not scored_rows:
        raise SystemExit(f"CSV {csv_path} has no finite {contract['score']} values.")
    if contract["target"] == "merge_detection" and len(scored_rows) != len(rows):
        raise SystemExit(
            f"Merge CSV {csv_path} unexpectedly has unscored OOF rows.")
    scored_rows.sort(key=lambda item: (-item[0], item[1]))
    n_scored_positive = sum(item[2] for item in scored_rows)
    if n_scored_positive == 0:
        raise SystemExit(
            f"CSV {csv_path} has no positive rows with finite {contract['score']}.")
    workload = []
    for requested in review_budgets:
        k = min(requested, len(scored_rows))
        found = sum(item[2] for item in scored_rows[:k])
        merge_creating = sum(item[3] for item in scored_rows[:k])
        workload.append({
            "requested_k": requested,
            "k": k,
            "positives_found": found,
            "merges_found": found if contract["target"] == "merge_detection" else None,
            "precision": found / k,
            "recall": found / n_scored_positive,
            "recall_scope": "oof_scored_rows",
            "merge_creating_joins": (
                merge_creating if contract["target"] == "split_detection" else None),
        })
    return {
        "n_rows": len(rows),
        "row_unit": contract["row_unit"],
        "n_positive": n_positive,
        "n_negative": len(rows) - n_positive,
        "n_oof_scored": len(scored_rows),
        "n_oof_unscored": len(rows) - len(scored_rows),
        "n_oof_scored_positive": n_scored_positive,
        "oof_coverage": len(scored_rows) / len(rows),
        "oof_scored_prevalence": n_scored_positive / len(scored_rows),
        "positive_name": contract["positive"],
        "negative_name": contract["negative"],
        "score_column": contract["score"],
        "n_merge": n_positive if contract["target"] == "merge_detection" else None,
        "n_clean": (len(rows) - n_positive
                    if contract["target"] == "merge_detection" else None),
        "prevalence": n_positive / len(rows),
        "review_workload": workload,
    }


def collect_evidence(result_dir: Path, application_dir: Path) -> dict:
    model_json_path = exactly_one(result_dir, "model_selection_*.json", "model-selection JSON")
    model = load_json_object(model_json_path)
    contract = _result_contract(model)
    brain = str(model.get("train_brain") or "")
    if not brain:
        raise SystemExit(f"{model_json_path} has no train_brain.")
    prefix = contract["output_prefix"]
    csv_path = result_dir / f"{prefix}_{brain}.csv"
    joblib_path = result_dir / f"{prefix}_{brain}.joblib"
    log_path = result_dir / f"{prefix}_{brain}.log.txt"
    for path, label in ((csv_path, "detector CSV"), (joblib_path, "fitted joblib"),
                        (log_path, "detector log")):
        if not path.is_file():
            raise SystemExit(f"Missing {label}: {path}")

    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    final_line = next((line for line in reversed(log_text.splitlines()) if line.strip()), "")
    if not final_line.startswith("# OK after "):
        raise SystemExit(
            f"Detector log is not complete (last line is {final_line!r}); analyze only "
            "after a successful run."
        )

    figures_dir = result_dir / "figures"
    figures = sorted(figures_dir.glob("*.png")) if figures_dir.is_dir() else []
    missing_numbers = [
        number for number in range(1, 8)
        if not any(f"_{number:02d}_" in path.name for path in figures)
    ]
    if missing_numbers:
        raise SystemExit(
            "Result analysis requires the complete training figure set 01–07; "
            f"missing {missing_numbers} under {figures_dir}. Rerun without --no-figures."
        )

    csv_summary = summarize_csv(csv_path, (50, 100, 200, 500), contract)
    audit = model.get("audit") if isinstance(model.get("audit"), dict) else {}
    audit_rows = audit.get("n_rows", audit.get("n_segments"))
    audit_positive = audit.get("n_positive", audit.get("n_merge"))
    if (audit_rows != csv_summary["n_rows"] or
            audit_positive != csv_summary["n_positive"]):
        raise SystemExit("Model-selection audit counts do not match the detector CSV.")

    detector_path = application_dir / contract["detector_name"]
    inventory_path = application_dir / "feature_inventory.json"
    model_config_path = application_dir / "model_candidates.json"
    model_policy_path = PROJECT_ROOT / "agentic" / "detector_model_policy.json"
    artifacts = [model_json_path, csv_path, joblib_path, log_path, *figures]
    for path in (detector_path, inventory_path, model_config_path, model_policy_path):
        if path.is_file():
            artifacts.append(path)

    recorded_hashes = {
        inventory_path: model.get("feature_inventory_sha256"),
        model_config_path: model.get("model_config_sha256"),
        model_policy_path: model.get("model_policy_sha256"),
    }
    for path, recorded_hash in recorded_hashes.items():
        if recorded_hash is None:
            continue
        if not path.is_file():
            raise SystemExit(f"Training provenance names a hash but {path} is missing.")
        actual_hash = sha256(path)
        if actual_hash != recorded_hash:
            raise SystemExit(
                f"Training provenance hash mismatch for {path}: "
                f"recorded {recorded_hash}, current {actual_hash}."
            )
    artifact_records = [
        {"path": rel(path), "sha256": sha256(path), "bytes": path.stat().st_size}
        for path in artifacts
    ]
    argv_line = next((line for line in log_text.splitlines() if line.startswith("# argv: ")), None)
    per_family = model.get("per_family_outer_metrics")
    if not isinstance(per_family, dict) or not per_family:
        raise SystemExit(f"{model_json_path} has no per_family_outer_metrics.")

    return {
        "schema_version": 1,
        "detector_target": contract["target"],
        "row_unit": contract["row_unit"],
        "label_name": contract["label"],
        "score_column": contract["score"],
        "positive_name": contract["positive"],
        "negative_name": contract["negative"],
        "result_dir": rel(result_dir),
        "application_dir": rel(application_dir),
        "brain": brain,
        "train_mcl": model.get("train_mcl"),
        "scope": model.get("scope"),
        "run_complete": True,
        "completion_line": final_line,
        "argv": argv_line.removeprefix("# argv: ") if argv_line else None,
        "csv_summary": csv_summary,
        "audit": audit,
        "sample_universe_audit": model.get("sample_universe_audit"),
        "final_winner": model.get("final_winner"),
        "final_params": model.get("final_params"),
        "per_family_outer_metrics": per_family,
        "selector_metrics": model.get("selector_metrics"),
        "selection_frequency": model.get("selection_frequency"),
        "hypothesis_selection": model.get("hypothesis_selection"),
        "skipped_candidates": model.get("skipped_candidates"),
        "dependency_versions": model.get("dependency_versions"),
        "seeds": model.get("seeds"),
        "heldout": model.get("heldout"),
        "feature_order": model.get("feature_order"),
        "all_feature_order": model.get("all_feature_order"),
        "convergence_warning_count": log_text.count("ConvergenceWarning"),
        "training_provenance": {
            "train_pkl_path": model.get("train_pkl"),
            "train_pkl_sha256": model.get("train_pkl_sha256"),
            "train_pkl_sha256_recorded": bool(model.get("train_pkl_sha256")),
            "detector_sha256": model.get("detector_sha256"),
            "detector_sha256_recorded_by_training": bool(model.get("detector_sha256")),
            "model_config_sha256": model.get("model_config_sha256"),
            "model_policy_sha256": model.get("model_policy_sha256"),
            "feature_inventory_sha256": model.get("feature_inventory_sha256"),
        },
        "figures": [
            {"path": rel(path), "basename": path.name, "sha256": sha256(path)}
            for path in figures
        ],
        "artifacts": artifact_records,
    }


def write_json_atomic(path: Path, value: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _fmt(value: object, digits: int = 4) -> str:
    return f"{value:.{digits}f}" if isinstance(value, (int, float)) else "—"


def write_report_skeleton(report_path: Path, evidence: dict, evidence_sha: str) -> str:
    family_lines = [
        "| Model family | OOF rows | OOF AP | OOF ROC-AUC |",
        "|---|---:|---:|---:|",
    ]
    for family, metrics in sorted(evidence["per_family_outer_metrics"].items()):
        family_lines.append(
            f"| `{family}` | {metrics.get('n_scored', '—')} | "
            f"{_fmt(metrics.get('oof_average_precision'))} | "
            f"{_fmt(metrics.get('oof_roc_auc'))} |"
        )
    split_result = evidence["detector_target"] == "split_detection"
    workload_lines = [
        (f"| Review k | {evidence['positive_name'].title()} found | Precision@k | "
         "Recall@k within OOF-scored rows |" +
         (" Merge-creating joins |" if split_result else "")),
        "|---:|---:|---:|---:|" + ("---:|" if split_result else ""),
    ]
    for row in evidence["csv_summary"]["review_workload"]:
        workload_lines.append(
            f"| {row['k']} | {row['positives_found']} | {_fmt(row['precision'])} | "
            f"{_fmt(row['recall'])} |" +
            (f" {row['merge_creating_joins']} |" if split_result else "")
        )
    figure_lines = "\n".join(
        f"- `{item['basename']}` — `{item['path']}` — SHA-256 `{item['sha256']}`"
        for item in evidence["figures"]
    )
    audit = evidence["audit"]
    csv_summary = evidence["csv_summary"]
    content = f"""# Bilingual detector result analysis — `{evidence['brain']}`

> This report is generated by a deterministic evidence extractor and
> `detector-result-analyst`. English comes first, followed by a faithful Chinese
> translation. / 本报告由确定性证据提取器和 `detector-result-analyst` 共同生成；
> 先写完整英文，再附忠实的完整中文翻译。

{BLOCK_START}

## Reproducible evidence / 可复现证据

- Result directory: `{evidence['result_dir']}`
- Evidence manifest: `{rel(report_path.parent / EVIDENCE_NAME)}`
- Evidence SHA-256: `{evidence_sha}`
- Run completion: `{evidence['completion_line']}`
- Detector target / row unit: `{evidence['detector_target']}` / `{evidence['row_unit']}`
- Brain / MCL / scope: `{evidence['brain']}` / `{evidence['train_mcl']}` / `{evidence['scope']}`
- Final winner: `{evidence['final_winner']}` with `{json.dumps(evidence['final_params'], sort_keys=True)}`
- Rows: {csv_summary['n_rows']}; positives ({evidence['positive_name']}): {csv_summary['n_positive']}; negatives ({evidence['negative_name']}): {csv_summary['n_negative']}; prevalence: {_fmt(csv_summary['prevalence'])}
- Finite OOF scores: {csv_summary['n_oof_scored']} / {csv_summary['n_rows']} ({_fmt(csv_summary['oof_coverage'])}); unscored rows: {csv_summary['n_oof_unscored']}. Review-workload recall is conditional on the OOF-scored subset.
- All features undefined: {audit.get('all_undefined_count', '—')}; positives in that group: {audit.get('all_undefined_positives', audit.get('all_undefined_merges', '—'))}
- Convergence warnings in detector log: {evidence['convergence_warning_count']}
- Held-out result present: {'yes' if evidence['heldout'] is not None else 'no'}
- Training provenance records config/policy/inventory hashes, but does not record the input-pkl SHA-256 or detector SHA-256.

### Nested OOF model comparison

{chr(10).join(family_lines)}

Selector metrics: `{json.dumps(evidence['selector_metrics'], sort_keys=True)}`  
Selection frequency: `{json.dumps(evidence['selection_frequency'], sort_keys=True)}`

### Deterministic review workload from `{evidence['score_column']}`

{chr(10).join(workload_lines)}

### Figures inspected by the analyst

{figure_lines}

{BLOCK_END}

{ENGLISH_REPORT_HEADING}

{REPORT_PLACEHOLDER}

{CHINESE_REPORT_HEADING}

{REPORT_PLACEHOLDER}
"""
    temporary = report_path.with_name(f".{report_path.name}.skeleton.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(report_path)
    return content


def driver_block(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    start = text.find(BLOCK_START)
    end = text.find(BLOCK_END)
    if start < 0 or end < start:
        raise SystemExit(f"Report {path} is missing its driver evidence block.")
    return text[start:end + len(BLOCK_END)]


def validate_agent_report(
    report_path: Path,
    *,
    skeleton_sha: str,
    expected_block: str,
    evidence_sha: str,
    figure_basenames: list[str],
) -> None:
    if not report_path.is_file() or report_path.stat().st_size == 0:
        raise SystemExit(f"Analysis agent did not write {report_path}.")
    if sha256(report_path) == skeleton_sha:
        raise SystemExit("Analysis agent left RESULT_ANALYSIS.md skeleton unchanged.")
    if driver_block(report_path) != expected_block:
        raise SystemExit("Analysis agent modified the immutable result-evidence block.")
    text = report_path.read_text(encoding="utf-8")
    if evidence_sha not in text:
        raise SystemExit("Analysis report lost its evidence SHA-256.")
    if REPORT_PLACEHOLDER in text:
        raise SystemExit("Analysis agent left a bilingual report placeholder unchanged.")
    english_start = text.find(ENGLISH_REPORT_HEADING)
    chinese_start = text.find(CHINESE_REPORT_HEADING)
    if english_start < 0 or chinese_start < 0 or english_start >= chinese_start:
        raise SystemExit("Analysis report must contain English first, then Chinese.")
    english = text[english_start:chinese_start]
    chinese = text[chinese_start:]
    missing_english_headings = [h for h in ENGLISH_SECTION_HEADINGS if h not in english]
    missing_chinese_headings = [h for h in CHINESE_SECTION_HEADINGS if h not in chinese]
    if missing_english_headings or missing_chinese_headings:
        raise SystemExit(
            "Analysis report is missing required bilingual sections: "
            f"English={missing_english_headings}, Chinese={missing_chinese_headings}."
        )
    if re.search(r"[\u4e00-\u9fff]", english):
        raise SystemExit("The English report contains Chinese prose.")
    if not re.search(r"[\u4e00-\u9fff]", chinese):
        raise SystemExit("The Chinese report contains no Chinese prose.")
    missing_english = [name for name in figure_basenames if name not in english]
    missing_chinese = [name for name in figure_basenames if name not in chinese]
    if missing_english or missing_chinese:
        raise SystemExit(
            "Analysis report did not discuss every figure in both languages: "
            f"English={missing_english}, Chinese={missing_chinese}."
        )


class Tee:
    def __init__(self, stream, file_handle):
        self.stream = stream
        self.file_handle = file_handle

    def write(self, text):
        self.stream.write(text)
        self.file_handle.write(text)
        self.flush()
        return len(text)

    def flush(self):
        self.stream.flush()
        self.file_handle.flush()

    def isatty(self):
        return False


async def run_agent(result_dir: Path, evidence_path: Path, report_path: Path, verbose: bool) -> None:
    sdk = load_claude_sdk()
    options = sdk.agent_options(
        cwd=str(PROJECT_ROOT),
        setting_sources=["project"],
        allowed_tools=["Task", "Read", "Write", "Edit", "Glob"],
        permission_mode="bypassPermissions",
        model=AGENT_MODEL,
        effort=AGENT_EFFORT,
    )
    instruction = f"""
Use the detector-result-analyst subagent to analyze the completed detector result
directory {rel(result_dir)}. Read {rel(evidence_path)} as the numeric source of
truth and visually inspect every PNG listed there. Enrich the existing skeleton
at {rel(report_path)} while preserving its BEGIN/END DRIVER-GENERATED RESULT
EVIDENCE block byte-for-byte. First write the complete English report, then
translate that completed report faithfully and fully into Simplified Chinese.
Mention every figure basename in each language part. Do not run the detector,
read the source pkl, load the joblib, or modify any other file. This turn writes
only {rel(report_path)}.
""".strip()
    async with open_agent_session(
        sdk.client, options=options, connect_timeout_s=60
    ) as client:
        await client.query(instruction)
        async def receive() -> None:
            async for message in client.receive_response():
                if isinstance(message, sdk.assistant_message):
                    for block in message.content:
                        if isinstance(block, sdk.text_block) and verbose:
                            print(block.text, end="", flush=True)
                        elif sdk.tool_use_block and isinstance(block, sdk.tool_use_block):
                            name = getattr(block, "name", "tool")
                            print(f"[analysis-agent] {name}", flush=True)
                elif isinstance(message, sdk.result_message):
                    cost = getattr(message, "total_cost_usd", None)
                    print(
                        "[analysis-agent] completed"
                        + (f"; reported cost ${cost:.4f}" if cost is not None else ""),
                        flush=True,
                    )
        try:
            await asyncio.wait_for(receive(), timeout=AGENT_TIMEOUT_S)
        except asyncio.TimeoutError as exc:
            raise SystemExit(
                f"Detector result analysis timed out after {AGENT_TIMEOUT_S}s."
            ) from exc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path",
        type=Path,
        help="Completed detector result directory under autodiscovery-application/.",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    result_dir, application_dir = resolve_result_dir(args.path)
    evidence = collect_evidence(result_dir, application_dir)
    evidence_path = result_dir / EVIDENCE_NAME
    report_path = result_dir / REPORT_NAME
    log_path = result_dir / WORKFLOW_LOG_NAME
    write_json_atomic(evidence_path, evidence)
    evidence_sha = sha256(evidence_path)
    write_report_skeleton(report_path, evidence, evidence_sha)
    skeleton_sha = sha256(report_path)
    expected_block = driver_block(report_path)

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
            print(f"[analysis] evidence: {rel(evidence_path)} sha256={evidence_sha}")
            asyncio.run(run_agent(result_dir, evidence_path, report_path, args.verbose))
            validate_agent_report(
                report_path,
                skeleton_sha=skeleton_sha,
                expected_block=expected_block,
                evidence_sha=evidence_sha,
                figure_basenames=[item["basename"] for item in evidence["figures"]],
            )
            outcome = "OK"
        except BaseException:
            traceback.print_exc()
            raise
        finally:
            sys.stdout, sys.stderr = original_out, original_err
            handle.write(
                f"# {outcome} after {time.monotonic() - started:.1f}s\n"
            )
            handle.flush()

    print(f"Result analysis report: {rel(report_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
