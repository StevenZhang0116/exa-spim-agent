"""Deterministic input discovery and provenance parsing.

No code-agent judgment belongs here.  AutoDiscovery evidence is joined by the
explicit hypothesis ID and every predictive artifact is cross-checked before an
output directory can be removed.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from pathlib import Path
from .contracts import DetectorTarget, RunContext
try:
    from agentic.split_feature_applicability import (
        artifact_path as split_applicability_path,
        load_applicability,
    )
except ModuleNotFoundError:  # direct workflow-script import via detector_build
    from split_feature_applicability import (  # type: ignore[no-redef]
        artifact_path as split_applicability_path,
        load_applicability,
    )


def run_stem(run_json: Path) -> str:
    return (
        run_json.name[: -len(".json")]
        if run_json.name.endswith(".json")
        else run_json.stem
    )


def origin_cache_hint(run_json: Path) -> str | None:
    match = re.search(r"(\d{5,7})[-_]mcl(\d+)", run_stem(run_json))
    if not match:
        return None
    return f"cache/dataset_cache_{match.group(1)}_mcl{match.group(2)}_add.pkl"


def rel_to_root(path: Path, project_root: Path) -> str:
    rel = Path(os.path.relpath(path, project_root))
    return path.as_posix() if rel.parts and rel.parts[0] == ".." else rel.as_posix()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json_object(path: Path, label: str, project_root: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(
            f"Cannot read {label} {rel_to_root(path, project_root)}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise SystemExit(
            f"{label.capitalize()} {rel_to_root(path, project_root)} must be a "
            "JSON object."
        )
    return value


def integer_ids(value, label: str, path: Path, project_root: Path) -> list[int]:
    if not isinstance(value, list) or not value:
        raise SystemExit(
            f"{label} in {rel_to_root(path, project_root)} must be a non-empty list."
        )
    if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        raise SystemExit(
            f"{label} in {rel_to_root(path, project_root)} must contain only integers."
        )
    if len(value) != len(set(value)):
        raise SystemExit(
            f"{label} in {rel_to_root(path, project_root)} contains duplicate ids."
        )
    return value


def manifest_ids(manifest_path: Path, project_root: Path) -> list[int]:
    manifest = load_json_object(manifest_path, "rerun manifest", project_root)
    records = manifest.get("records")
    if not isinstance(records, list) or not records:
        raise SystemExit(
            f"Rerun manifest {rel_to_root(manifest_path, project_root)} has no "
            "non-empty records list."
        )
    ids = [record.get("id") if isinstance(record, dict) else None for record in records]
    validated_ids = integer_ids(ids, "records[].id", manifest_path, project_root)
    for index, (record, hypothesis_id) in enumerate(zip(records, validated_ids)):
        expected_file = f"hypo_{hypothesis_id}.py"
        if record.get("file") != expected_file:
            raise SystemExit(
                f"Rerun manifest {rel_to_root(manifest_path, project_root)} record "
                f"{index} maps id {hypothesis_id} to {record.get('file')!r}; "
                f"expected {expected_file!r}."
            )
    return validated_ids


def report_evidence_by_id(
    summary_path: Path,
    project_root: Path,
) -> dict[int, dict[str, str | None]]:
    try:
        text = summary_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SystemExit(
            f"Cannot read finished report {rel_to_root(summary_path, project_root)}: "
            f"{exc}"
        ) from exc

    evidence: dict[int, dict[str, str | None]] = {}
    for block in re.split(r"(?m)(?=^###\s+\d+\.)", text):
        id_match = re.search(r"\*\*ID:\*\*\s*(\d+)", block)
        if id_match is None:
            continue
        hypothesis_id = int(id_match.group(1))
        if hypothesis_id in evidence:
            raise SystemExit(
                f"Finished report {rel_to_root(summary_path, project_root)} repeats "
                f"explicit hypothesis ID {hypothesis_id}."
            )

        def field(label: str) -> str | None:
            match = re.search(
                rf"(?m)^-\s+\*\*{re.escape(label)}:\*\*\s*([A-Z]+)",
                block,
            )
            return match.group(1) if match else None

        evidence[hypothesis_id] = {
            "reproduction_status": field("Reproduction"),
            "statistical_verdict": field("Verdict"),
            "post_correction_verdict": field("Post-correction verdict"),
        }
    return evidence


def corrected_status_by_id(
    corrected_path: Path,
    rerun_dir: Path,
    fixed_dir: Path | None,
    project_root: Path,
) -> dict[int, str]:
    payload = load_json_object(corrected_path, "corrected results", project_root)
    expected_code_dir = rel_to_root(rerun_dir, project_root)
    expected_corrected_dir = (
        rel_to_root(fixed_dir, project_root) if fixed_dir else None
    )
    if payload.get("code_dir") != expected_code_dir:
        raise SystemExit(
            f"Corrected results {rel_to_root(corrected_path, project_root)} are "
            f"stale: code_dir must be {expected_code_dir!r}."
        )
    if payload.get("corrected_dir") != expected_corrected_dir:
        raise SystemExit(
            f"Corrected results {rel_to_root(corrected_path, project_root)} are "
            f"stale: corrected_dir must be {expected_corrected_dir!r}."
        )
    results = payload.get("results")
    if not isinstance(results, list):
        raise SystemExit(
            f"Corrected results {rel_to_root(corrected_path, project_root)} needs "
            "a results list."
        )
    statuses: dict[int, str] = {}
    allowed = {"USABLE", "FAILED", "UNUSABLE"}
    for index, result in enumerate(results):
        if not isinstance(result, dict):
            raise SystemExit(
                f"Corrected results {rel_to_root(corrected_path, project_root)} "
                f"item {index} must be an object."
            )
        hypothesis_id = result.get("id")
        status = result.get("result_status")
        if isinstance(hypothesis_id, bool) or not isinstance(hypothesis_id, int):
            raise SystemExit(
                f"Corrected results {rel_to_root(corrected_path, project_root)} "
                f"item {index} needs an integer id."
            )
        if hypothesis_id in statuses:
            raise SystemExit(
                f"Corrected results {rel_to_root(corrected_path, project_root)} "
                f"repeats hypothesis ID {hypothesis_id}."
            )
        if status not in allowed:
            raise SystemExit(
                f"Corrected result {hypothesis_id} has invalid result_status "
                f"{status!r}; expected one of {sorted(allowed)}."
            )
        statuses[hypothesis_id] = status
    return statuses


def unbacked_correction_verdicts(
    report_evidence: dict[int, dict[str, str | None]],
    corrected_statuses: dict[int, str],
    hypothesis_ids,
) -> list[tuple[int, str, str | None]]:
    """Report verdicts with no USABLE corrected measurement behind them.

    Returns ``(id, verdict token, corrected status or None)`` for every listed
    hypothesis whose report row carries a post-correction verdict while the
    corrected results record no USABLE measurement for that same ID (a missing
    corrected entry — or a missing corrected JSON, when ``corrected_statuses``
    is empty — reports the status as ``None``). This is the single definition
    of "unbacked" shared by the input gate, the inventory compiler, and the
    inventory validator, so reconciliation cannot drift between them.
    """
    offenders: list[tuple[int, str, str | None]] = []
    for hypothesis_id in hypothesis_ids:
        row = report_evidence.get(hypothesis_id)
        if row is None:
            continue
        verdict = row.get("post_correction_verdict")
        if verdict is None:
            continue
        status = corrected_statuses.get(hypothesis_id)
        if status != "USABLE":
            offenders.append((hypothesis_id, verdict, status))
    return offenders


def infer_target(run_json: Path) -> DetectorTarget:
    """Infer the legacy task type without changing the input artifact format."""
    stem = run_stem(run_json).lower()
    if stem == "split-error" or stem.startswith(("split-error-", "split-error_")):
        return DetectorTarget.SPLIT
    if stem == "merge-error" or stem.startswith(("merge-error-", "merge-error_")):
        return DetectorTarget.MERGE
    return DetectorTarget.LEGACY_UNSPECIFIED


def resolve_run_context(
    run_json: Path,
    project_root: Path,
    predictive_policy_version: str,
    *,
    reconcile_unbacked_verdicts: bool = False,
) -> RunContext:
    """Resolve and cross-check every input before any destructive output action.

    ``reconcile_unbacked_verdicts`` downgrades one class of inconsistency from a
    hard abort to a warning: a report post-correction verdict with no USABLE
    corrected measurement behind it (e.g. the discovery fold step wrote a
    verdict for a corrected script that timed out). When set, such verdicts are
    treated as null throughout the build — the correction is ignored and the
    hypothesis is judged on its pre-correction evidence — and the affected ids
    are recorded on the returned context.
    """
    target = infer_target(run_json)

    stem = run_stem(run_json)
    summary_path = run_json.with_name(f"{stem}.summary.md")
    predictive_rerun_dir = run_json.with_name(f"{run_json.name}.predictive.rerun")
    legacy_rerun_dir = run_json.with_name(f"{run_json.name}.rerun")
    predictive_fixed_dir = run_json.with_name(f"{run_json.name}.predictive.fixed")
    legacy_fixed_dir = run_json.with_name(f"{run_json.name}.fixed")
    selection_path = run_json.with_name(f"{stem}.predictive-selection.json")

    predictive_ready = predictive_rerun_dir.is_dir() and any(
        predictive_rerun_dir.glob("hypo_*.py")
    )
    legacy_ready = legacy_rerun_dir.is_dir() and any(
        legacy_rerun_dir.glob("hypo_*.py")
    )
    rerun_dir = predictive_rerun_dir if predictive_ready else legacy_rerun_dir
    candidate_fixed_dir = predictive_fixed_dir if predictive_ready else legacy_fixed_dir
    fixed_dir = (
        candidate_fixed_dir
        if candidate_fixed_dir.is_dir() and any(candidate_fixed_dir.glob("hypo_*.py"))
        else None
    )

    if not summary_path.is_file():
        raise SystemExit(
            f"No finished report at {rel_to_root(summary_path, project_root)}. Run "
            f"`python agentic/run_discovery_workflow.py "
            f"{rel_to_root(run_json, project_root)} --pkl "
            f"{origin_cache_hint(run_json) or '<ORIGIN>_add.pkl'} "
            "--direction predictive` first — this workflow builds on its output."
        )
    if not (predictive_ready or legacy_ready):
        raise SystemExit(
            "No loading-fixed scripts found. Expected predictive artifacts at "
            f"{rel_to_root(predictive_rerun_dir, project_root)} (or legacy "
            f"artifacts at {rel_to_root(legacy_rerun_dir, project_root)}). Run "
            "the discovery workflow with --direction predictive first; detector "
            "feature definitions come from those scripts, not from the report."
        )
    rerun_manifest_path = rerun_dir / "MANIFEST.json"
    if not rerun_manifest_path.is_file():
        raise SystemExit(
            f"No rerun manifest at {rel_to_root(rerun_manifest_path, project_root)}; "
            "refuse to infer hypothesis membership from possibly stale loose scripts."
        )
    rerun_ids = manifest_ids(rerun_manifest_path, project_root)

    authoritative_selection = selection_path if predictive_ready else None
    if predictive_ready:
        if not selection_path.is_file():
            raise SystemExit(
                f"Predictive rerun sources require "
                f"{rel_to_root(selection_path, project_root)} as the authoritative "
                "hypothesis selection."
            )
        selection = load_json_object(
            selection_path, "predictive selection", project_root
        )
        if selection.get("criterion") != "predictive":
            raise SystemExit(
                f"Predictive selection {rel_to_root(selection_path, project_root)} "
                "has the wrong criterion."
            )
        if selection.get("policy_version") != predictive_policy_version:
            raise SystemExit(
                f"Predictive selection {rel_to_root(selection_path, project_root)} "
                f"uses stale policy {selection.get('policy_version')!r}; expected "
                f"{predictive_policy_version!r}. Regenerate discovery outputs."
            )
        if selection.get("source_file") != rel_to_root(run_json, project_root):
            raise SystemExit(
                f"Predictive selection {rel_to_root(selection_path, project_root)} "
                f"points to source_file {selection.get('source_file')!r}, not "
                f"{rel_to_root(run_json, project_root)!r}."
            )
        selected_ids = integer_ids(
            selection.get("selected_ids"),
            "selected_ids",
            selection_path,
            project_root,
        )
        if selection.get("source_sha256") != sha256(run_json):
            raise SystemExit(
                f"Predictive selection {rel_to_root(selection_path, project_root)} "
                f"is stale for {rel_to_root(run_json, project_root)}; regenerate "
                "the discovery outputs."
            )
        selected_set, rerun_set = set(selected_ids), set(rerun_ids)
        if selected_set != rerun_set:
            raise SystemExit(
                "Predictive selection and rerun MANIFEST disagree; refuse stale "
                f"sources. Missing from MANIFEST: "
                f"{sorted(selected_set - rerun_set)}; not selected: "
                f"{sorted(rerun_set - selected_set)}. Regenerate the predictive "
                "rerun artifacts before building a detector."
            )
    else:
        selected_ids = rerun_ids

    missing_scripts = [
        hypothesis_id
        for hypothesis_id in selected_ids
        if not (rerun_dir / f"hypo_{hypothesis_id}.py").is_file()
    ]
    if missing_scripts:
        raise SystemExit(
            f"Rerun MANIFEST selects ids with no script: {missing_scripts}."
        )

    scope_suffix = ".predictive" if authoritative_selection is not None else ""
    corrected_candidate = run_json.with_name(
        f"{run_json.name}{scope_suffix}.corrected.json"
    )
    corrected_results_path = (
        corrected_candidate if corrected_candidate.is_file() else None
    )
    report_evidence = report_evidence_by_id(summary_path, project_root)
    corrected_statuses = (
        corrected_status_by_id(
            corrected_results_path,
            rerun_dir,
            fixed_dir,
            project_root,
        )
        if corrected_results_path is not None
        else {}
    )
    offenders = unbacked_correction_verdicts(
        report_evidence, corrected_statuses, selected_ids
    )
    unbacked_ids: tuple[int, ...] = ()
    if offenders:
        remedy = (
            "Regenerate the corrected discovery outputs before building the "
            "detector, or rerun with --reconcile-unbacked-verdicts to treat "
            "these verdicts as null (each such correction is then ignored and "
            "its hypothesis judged on the pre-correction evidence alone)."
        )
        if not reconcile_unbacked_verdicts:
            if corrected_results_path is None:
                offending_ids = [hid for hid, _, _ in offenders]
                raise SystemExit(
                    "Finished report records a post-correction verdict for "
                    f"selected hypothesis id(s) {offending_ids}, but "
                    f"{rel_to_root(corrected_candidate, project_root)} is "
                    f"missing. {remedy}"
                )
            detail = "; ".join(
                f"id {hid}: report verdict {verdict!r} vs corrected result "
                f"{status or 'ABSENT'}"
                for hid, verdict, status in offenders
            )
            raise SystemExit(
                "Finished report records post-correction verdicts that "
                "corrected results do not back with a USABLE measurement for "
                f"the same ID: {detail}. {remedy}"
            )
        for hid, verdict, status in offenders:
            print(
                f"[inputs] WARNING: post-correction verdict {verdict!r} for "
                f"hypothesis {hid} has no USABLE corrected measurement "
                f"(corrected result: {status or 'ABSENT'}); treating the "
                "verdict as null and ignoring the correction.",
                file=sys.stderr,
            )
        unbacked_ids = tuple(hid for hid, _, _ in offenders)
    applicability_path: Path | None = None
    if target is DetectorTarget.SPLIT:
        applicability_path = split_applicability_path(run_json)
        if not applicability_path.is_file():
            raise SystemExit(
                f"No split feature applicability artifact at "
                f"{rel_to_root(applicability_path, project_root)}. Generate it "
                "without rerunning experiments via `python "
                f"agentic/run_discovery_workflow.py "
                f"{rel_to_root(run_json, project_root)} "
                "--split-feature-labels-only`."
            )
        load_applicability(
            applicability_path,
            run_json=run_json,
            selected_ids=selected_ids,
            rerun_dir=rerun_dir,
            fixed_dir=fixed_dir,
            project_root=project_root,
        )
    return RunContext(
        target=target,
        run_json=run_json,
        summary_path=summary_path,
        rerun_dir=rerun_dir,
        fixed_dir=fixed_dir,
        selection_path=authoritative_selection,
        corrected_results_path=corrected_results_path,
        split_feature_applicability_path=applicability_path,
        selected_ids=tuple(selected_ids),
        unbacked_verdict_ids=unbacked_ids,
    )


def protected_source_paths(context: RunContext) -> tuple[Path, ...]:
    """Return every authoritative input that agent stages must treat as read-only."""
    paths: list[Path] = [
        context.run_json,
        context.summary_path,
        context.rerun_dir / "MANIFEST.json",
    ]
    if context.selection_path is not None:
        paths.append(context.selection_path)
    if context.corrected_results_path is not None:
        paths.append(context.corrected_results_path)
    if context.split_feature_applicability_path is not None:
        paths.append(context.split_feature_applicability_path)
    paths.extend(
        context.rerun_dir / f"hypo_{hypothesis_id}.py"
        for hypothesis_id in context.selected_ids
    )
    if context.fixed_dir is not None:
        paths.extend(sorted(context.fixed_dir.glob("hypo_*.py")))
    # Preserve order while avoiding duplicate paths.
    return tuple(dict.fromkeys(paths))
