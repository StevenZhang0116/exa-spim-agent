"""Validated split-candidate policy selection for detector-build workflows.

The agent supplies a small semantic recommendation.  Deterministic driver code
independently resolves the minimum-candidate configuration that meets the
declared worst-brain recall target, verifies the sweep bundle and hashes, and
writes the frozen public artifact. Detector assembly then requires that artifact,
embeds its selected generator fields and SHA-256, and refuses an implicit split
candidate default.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from pathlib import Path

from .contracts import (
    CANDIDATE_POLICY_ADVICE_DRAFT_NAME,
    CANDIDATE_POLICY_NAME,
    MERGE_CANDIDATE_POLICY_NAME,
)

CANDIDATE_POLICY_SCHEMA_VERSION = 1
CANDIDATE_POLICY_ADVICE_SCHEMA_VERSION = 1
OBJECTIVE_NAME = "minimum_candidate_count_subject_to_worst_brain_recall"
APPLICATION_STATUS = "selected_for_detector_runtime"
SUPPORTED_RUNTIME_PAIRING_RULES = frozenset({
    "tip_to_tip", "tip_to_any_node", "any_node_to_any_node",
})

# --- merge-site (junction) candidate policy ---------------------------------
# The merge candidate-pool sweep (notebooks/merge_candidate_pool_sweep.py)
# already writes a DETERMINISTIC recommended_policy.json, so unlike the split
# flow there is no agent advice stage: the driver validates the sweep bundle,
# independently re-derives the argmin selection from the aggregate table, and
# freezes merge_candidate_policy.json for assembly.
MERGE_SITE_OBJECTIVE_NAME = (
    "minimum_candidate_count_subject_to_worst_brain_site_recall"
)
MERGE_SITE_SOURCE_ARTIFACT_TYPE = "merge_candidate_policy_selection"
MERGE_SITE_FROZEN_ARTIFACT_TYPE = "merge_site_candidate_policy_selection"
SUPPORTED_MERGE_SITE_MODES = frozenset({"junction"})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path, label: str) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SystemExit(f"{label} {path} must contain a JSON object.")
    return value


def _relative(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _finite_probability(value: object, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"{label} must be numeric.") from exc
    if not math.isfinite(number) or not 0 < number <= 1:
        raise SystemExit(f"{label} must be finite and in (0, 1].")
    return number


def expected_mcl_from_run_path(run_json: Path) -> int | None:
    match = re.search(r"(?:^|[-_])mcl(\d+)(?:[-_.]|$)", run_json.name.lower())
    return int(match.group(1)) if match else None


def candidate_policy_source_paths(review_path: Path) -> tuple[Path, ...]:
    directory = review_path.resolve().parent
    return (
        review_path.resolve(),
        directory / "ai_review_evidence.json",
        directory / "artifact_manifest.json",
        directory / "tables" / "cross_dataset_aggregate.csv",
    )


def _manifest_entries(manifest_path: Path) -> dict[str, dict]:
    manifest = _load_json(manifest_path, "candidate-sweep artifact manifest")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise SystemExit(f"Manifest {manifest_path} needs an artifacts list.")
    entries: dict[str, dict] = {}
    for entry in artifacts:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise SystemExit(f"Manifest {manifest_path} has an invalid artifact entry.")
        artifact_path = entry["path"]
        if artifact_path in entries:
            raise SystemExit(f"Manifest {manifest_path} repeats {artifact_path!r}.")
        entries[artifact_path] = entry
    return entries


def _require_manifest_hash(
    entries: dict[str, dict], relative_path: str, actual_path: Path
) -> str:
    entry = entries.get(relative_path)
    if entry is None:
        raise SystemExit(f"Candidate-sweep manifest does not list {relative_path}.")
    expected = entry.get("sha256")
    actual = _sha256(actual_path)
    if expected != actual:
        raise SystemExit(
            f"Candidate-sweep manifest hash mismatch for {relative_path}: "
            f"expected {expected!r}, got {actual!r}."
        )
    return actual


def _read_aggregate_rows(path: Path) -> list[dict]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as exc:
        raise SystemExit(f"Cannot read candidate aggregate table {path}: {exc}") from exc
    required = {
        "config_id", "mode", "radius_um", "per_anchor_k", "global_cap",
        "datasets", "total_candidates", "min_recall", "mean_recall",
        "micro_recall", "min_direct_recall", "min_gap_recall",
        "mean_prevalence",
    }
    if not rows:
        raise SystemExit(f"Candidate aggregate table {path} is empty.")
    missing = required - set(rows[0])
    if missing:
        raise SystemExit(
            f"Candidate aggregate table {path} is missing columns {sorted(missing)}."
        )
    return rows


def _parse_selected_row(row: dict) -> dict:
    try:
        radius = float(row["radius_um"])
        total_candidates = int(row["total_candidates"])
        datasets = int(row["datasets"])
        metrics = {
            key: float(row[key])
            for key in (
                "min_recall", "mean_recall", "micro_recall",
                "min_direct_recall", "min_gap_recall", "mean_prevalence",
            )
        }
    except (TypeError, ValueError, KeyError) as exc:
        raise SystemExit(f"Invalid numeric candidate aggregate row: {row}") from exc
    if not math.isfinite(radius) or radius <= 0 or total_candidates < 1 or datasets < 1:
        raise SystemExit(f"Invalid candidate aggregate row: {row}")
    if any(
        not math.isfinite(value) or not 0 <= value <= 1
        for value in metrics.values()
    ):
        raise SystemExit(f"Candidate aggregate metrics must be finite in [0, 1]: {row}")
    quota_text = str(row["per_anchor_k"])
    if quota_text == "all":
        quota: int | None = None
    else:
        try:
            quota = int(quota_text)
        except ValueError as exc:
            raise SystemExit(f"Invalid per_anchor_k {quota_text!r}.") from exc
        if quota < 1:
            raise SystemExit(f"per_anchor_k must be positive, got {quota}.")
    global_cap_text = str(row["global_cap"])
    if global_cap_text != "none":
        raise SystemExit(
            "This policy selector currently accepts only global_cap=none; "
            f"got {global_cap_text!r}."
        )
    mode = str(row["mode"]).strip()
    config_id = str(row["config_id"]).strip()
    expected_config_id = (
        f"{mode}|r={radius:g}|k={'all' if quota is None else quota}"
    )
    if not mode or config_id != expected_config_id:
        raise SystemExit(
            "Candidate aggregate config_id is inconsistent with its policy "
            f"columns: {config_id!r} != {expected_config_id!r}."
        )
    return {
        "config_id": config_id,
        "mode": mode,
        "radius_um": int(radius) if radius.is_integer() else radius,
        "per_anchor_k": quota,
        "global_cap": None,
        "datasets": datasets,
        "total_candidates": total_candidates,
        **metrics,
    }


def _deterministic_selection(aggregate_path: Path, minimum_recall: float) -> dict:
    eligible: list[dict] = []
    for raw in _read_aggregate_rows(aggregate_path):
        parsed = _parse_selected_row(raw)
        if parsed["min_recall"] + 1e-12 >= minimum_recall:
            eligible.append(parsed)
    if not eligible:
        raise SystemExit(
            "No candidate configuration meets worst-brain recall >= "
            f"{minimum_recall:.6g}."
        )
    return min(eligible, key=lambda row: (row["total_candidates"], row["config_id"]))


def _runtime_policy_from_artifact(policy: dict, policy_path: Path) -> dict:
    """Return the strictly validated candidate-generator fields to embed."""
    if policy.get("schema_version") != CANDIDATE_POLICY_SCHEMA_VERSION:
        raise SystemExit(f"Candidate policy {policy_path} has a stale schema_version.")
    if policy.get("artifact_type") != "split_candidate_policy_selection":
        raise SystemExit(f"Candidate policy {policy_path} has the wrong artifact_type.")
    if policy.get("application_status") != APPLICATION_STATUS:
        raise SystemExit(
            f"Candidate policy {policy_path} is not selected for detector runtime."
        )
    selected = policy.get("selected_policy")
    required = {"config_id", "mode", "radius_um", "per_anchor_k", "global_cap"}
    if not isinstance(selected, dict) or set(selected) != required:
        raise SystemExit(
            f"Candidate policy {policy_path} selected_policy must contain exactly "
            f"{sorted(required)}."
        )
    mode = selected.get("mode")
    if mode not in SUPPORTED_RUNTIME_PAIRING_RULES:
        raise SystemExit(
            f"Candidate policy {policy_path} has unsupported runtime mode {mode!r}."
        )
    try:
        radius = float(selected.get("radius_um"))
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"Candidate policy {policy_path} radius_um must be numeric.") from exc
    if not math.isfinite(radius) or radius <= 0:
        raise SystemExit(f"Candidate policy {policy_path} radius_um must be positive.")
    quota = selected.get("per_anchor_k")
    if quota is not None and (isinstance(quota, bool) or not isinstance(quota, int) or quota < 1):
        raise SystemExit(
            f"Candidate policy {policy_path} per_anchor_k must be null or a positive integer."
        )
    if selected.get("global_cap") is not None:
        raise SystemExit(
            f"Candidate policy {policy_path} must use global_cap=null."
        )
    quota_label = "all" if quota is None else str(quota)
    expected_id = f"{mode}|r={radius:g}|k={quota_label}"
    if selected.get("config_id") != expected_id:
        raise SystemExit(
            f"Candidate policy {policy_path} config_id disagrees with its fields: "
            f"{selected.get('config_id')!r} != {expected_id!r}."
        )
    return {
        "config_id": expected_id,
        "mode": mode,
        "radius_um": int(radius) if radius.is_integer() else radius,
        "per_anchor_k": quota,
        "global_cap": None,
    }


def load_runtime_candidate_policy(policy_path: Path) -> tuple[dict, str]:
    """Load the exact policy fields and artifact hash consumed by assembly."""
    policy = _load_json(policy_path, "frozen split candidate policy")
    return _runtime_policy_from_artifact(policy, policy_path), _sha256(policy_path)


def validate_candidate_policy_advice(
    advice_path: Path, *, minimum_recall: float
) -> dict:
    advice = _load_json(advice_path, "candidate-policy advice draft")
    expected_keys = {
        "schema_version", "objective", "selected_config_id", "explanation"
    }
    if set(advice) != expected_keys:
        raise SystemExit(
            "Candidate-policy advice must contain exactly "
            f"{sorted(expected_keys)}."
        )
    if advice["schema_version"] != CANDIDATE_POLICY_ADVICE_SCHEMA_VERSION:
        raise SystemExit(
            "Candidate-policy advice has unsupported schema_version "
            f"{advice['schema_version']!r}."
        )
    objective = advice["objective"]
    if not isinstance(objective, dict) or set(objective) != {
        "name", "minimum_worst_brain_recall", "tie_breaker"
    }:
        raise SystemExit("Candidate-policy advice has an invalid objective object.")
    if objective["name"] != OBJECTIVE_NAME:
        raise SystemExit(f"Candidate-policy advice must use objective {OBJECTIVE_NAME!r}.")
    draft_minimum = _finite_probability(
        objective["minimum_worst_brain_recall"],
        "advice objective minimum_worst_brain_recall",
    )
    if not math.isclose(draft_minimum, minimum_recall, rel_tol=0, abs_tol=1e-12):
        raise SystemExit(
            "Candidate-policy advice changed the driver-owned recall target: "
            f"{draft_minimum} != {minimum_recall}."
        )
    if objective["tie_breaker"] != ["total_candidates", "config_id"]:
        raise SystemExit(
            "Candidate-policy advice tie_breaker must be "
            "['total_candidates', 'config_id']."
        )
    if not isinstance(advice["selected_config_id"], str) or not advice[
        "selected_config_id"
    ].strip():
        raise SystemExit("Candidate-policy advice needs a selected_config_id.")
    if not isinstance(advice["explanation"], str) or not advice["explanation"].strip():
        raise SystemExit("Candidate-policy advice needs a non-empty explanation.")
    return advice


def _validated_source_bundle(
    review_path: Path,
    *,
    minimum_recall: float,
    expected_mcl: int | None,
    project_root: Path,
) -> tuple[dict, dict]:
    minimum_recall = _finite_probability(
        minimum_recall, "minimum worst-brain candidate recall"
    )
    review_path, evidence_path, manifest_path, aggregate_path = (
        candidate_policy_source_paths(review_path)
    )
    for path in (review_path, evidence_path, manifest_path, aggregate_path):
        if not path.is_file():
            raise SystemExit(f"Missing split candidate-policy source: {path}")

    entries = _manifest_entries(manifest_path)
    review_sha = _require_manifest_hash(entries, "AI_REVIEW.md", review_path)
    evidence_sha = _require_manifest_hash(
        entries, "ai_review_evidence.json", evidence_path
    )
    aggregate_sha = _require_manifest_hash(
        entries, "tables/cross_dataset_aggregate.csv", aggregate_path
    )
    review_text = review_path.read_text(encoding="utf-8")
    declared = re.search(r"Evidence file: `ai_review_evidence\.json` \(`sha256=([0-9a-f]{64})`\)", review_text)
    if declared is None or declared.group(1) != evidence_sha:
        raise SystemExit(
            "AI_REVIEW.md does not declare the current ai_review_evidence.json SHA-256."
        )

    evidence = _load_json(evidence_path, "candidate-policy evidence")
    if evidence.get("evidence_schema") != 1:
        raise SystemExit("Candidate-policy evidence must use evidence_schema=1.")
    run = evidence.get("run")
    if not isinstance(run, dict) or run.get("result_status") != "complete":
        raise SystemExit("Candidate-policy evidence run is not complete.")
    integrity = run.get("pre_review_manifest_integrity")
    if not isinstance(integrity, dict) or integrity.get("status") != "pass":
        raise SystemExit("Candidate-policy evidence manifest integrity did not pass.")
    evidence_mcl = run.get("mcl")
    if not isinstance(evidence_mcl, int):
        raise SystemExit("Candidate-policy evidence run needs an integer mcl.")
    if expected_mcl is not None and evidence_mcl != expected_mcl:
        raise SystemExit(
            f"Candidate-policy evidence is mcl{evidence_mcl}, but the detector "
            f"run name indicates mcl{expected_mcl}."
        )
    sources = evidence.get("source_files")
    aggregate_source = sources.get("aggregate") if isinstance(sources, dict) else None
    if not isinstance(aggregate_source, dict) or (
        aggregate_source.get("path") != "tables/cross_dataset_aggregate.csv"
        or aggregate_source.get("sha256") != aggregate_sha
    ):
        raise SystemExit(
            "Candidate-policy evidence does not bind the current aggregate table."
        )

    selected = _deterministic_selection(aggregate_path, minimum_recall)
    brains = run.get("brains")
    pairing_rules = run.get("evaluated_pairing_rules")
    if (
        not isinstance(brains, list)
        or not brains
        or any(not isinstance(brain, str) or not brain for brain in brains)
        or selected["datasets"] != len(brains)
    ):
        raise SystemExit(
            "Candidate-policy evidence brain scope disagrees with the selected "
            "aggregate row."
        )
    if (
        not isinstance(pairing_rules, list)
        or selected["mode"] not in pairing_rules
    ):
        raise SystemExit(
            "Candidate-policy evidence pairing-rule scope does not include the "
            "selected mode."
        )
    scope = {
        "mcl": evidence_mcl,
        "config_signature": run.get("config_signature"),
        "brains": brains,
        "evaluated_pairing_rules": pairing_rules,
        "any_node_to_any_node_evaluated": run.get(
            "any_node_to_any_node_evaluated"
        ),
        "global_cap_swept": run.get("global_cap_swept"),
    }
    provenance = {
        "review_path": _relative(review_path, project_root),
        "review_sha256": review_sha,
        "evidence_path": _relative(evidence_path, project_root),
        "evidence_sha256": evidence_sha,
        "aggregate_path": _relative(aggregate_path, project_root),
        "aggregate_sha256": aggregate_sha,
        "manifest_path": _relative(manifest_path, project_root),
        "manifest_sha256": _sha256(manifest_path),
    }
    return selected, {"scope": scope, "provenance": provenance}


def validate_candidate_policy_sources(
    review_path: Path,
    *,
    minimum_recall: float,
    expected_mcl: int | None,
    project_root: Path,
) -> dict:
    """Validate the sweep bundle before any output directory can be deleted."""
    selected, metadata = _validated_source_bundle(
        review_path,
        minimum_recall=minimum_recall,
        expected_mcl=expected_mcl,
        project_root=project_root,
    )
    return {"selected": selected, **metadata}


def compile_candidate_policy(
    advice_path: Path,
    output_path: Path,
    review_path: Path,
    *,
    minimum_recall: float,
    expected_mcl: int | None,
    project_root: Path,
) -> dict:
    advice = validate_candidate_policy_advice(
        advice_path, minimum_recall=minimum_recall
    )
    selected, metadata = _validated_source_bundle(
        review_path,
        minimum_recall=minimum_recall,
        expected_mcl=expected_mcl,
        project_root=project_root,
    )
    if advice["selected_config_id"] != selected["config_id"]:
        raise SystemExit(
            "Agent candidate-policy advice disagrees with deterministic selection: "
            f"{advice['selected_config_id']!r} != {selected['config_id']!r}."
        )
    policy = {
        "schema_version": CANDIDATE_POLICY_SCHEMA_VERSION,
        "artifact_type": "split_candidate_policy_selection",
        "application_status": APPLICATION_STATUS,
        "objective": {
            "name": OBJECTIVE_NAME,
            "minimum_worst_brain_recall": minimum_recall,
            "tie_breaker": ["total_candidates", "config_id"],
        },
        "selected_policy": {
            key: selected[key]
            for key in (
                "config_id", "mode", "radius_um", "per_anchor_k", "global_cap"
            )
        },
        "evidence_metrics": {
            key: selected[key]
            for key in (
                "datasets", "total_candidates", "min_recall", "mean_recall",
                "micro_recall", "min_direct_recall", "min_gap_recall",
                "mean_prevalence",
            )
        },
        "evidence_scope": metadata["scope"],
        "agent_interpretation": {
            "selected_config_id": advice["selected_config_id"],
            "explanation": advice["explanation"].strip(),
        },
        "provenance": metadata["provenance"],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(policy, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return policy


def validate_frozen_candidate_policy(
    policy_path: Path,
    review_path: Path,
    *,
    minimum_recall: float,
    expected_mcl: int | None,
    project_root: Path,
) -> dict:
    policy = _load_json(policy_path, "frozen split candidate policy")
    runtime_policy = _runtime_policy_from_artifact(policy, policy_path)
    selected, metadata = _validated_source_bundle(
        review_path,
        minimum_recall=minimum_recall,
        expected_mcl=expected_mcl,
        project_root=project_root,
    )
    expected_objective = {
        "name": OBJECTIVE_NAME,
        "minimum_worst_brain_recall": minimum_recall,
        "tie_breaker": ["total_candidates", "config_id"],
    }
    if policy.get("objective") != expected_objective:
        raise SystemExit("Frozen split candidate policy objective is stale.")
    expected_selected = {
        key: selected[key]
        for key in (
            "config_id", "mode", "radius_um", "per_anchor_k", "global_cap"
        )
    }
    if runtime_policy != expected_selected:
        raise SystemExit("Frozen split candidate policy selection is stale.")
    expected_metrics = {
        key: selected[key]
        for key in (
            "datasets", "total_candidates", "min_recall", "mean_recall",
            "micro_recall", "min_direct_recall", "min_gap_recall",
            "mean_prevalence",
        )
    }
    if policy.get("evidence_metrics") != expected_metrics:
        raise SystemExit("Frozen split candidate policy metrics are stale.")
    if policy.get("evidence_scope") != metadata["scope"]:
        raise SystemExit("Frozen split candidate policy evidence scope is stale.")
    if policy.get("provenance") != metadata["provenance"]:
        raise SystemExit("Frozen split candidate policy provenance is stale.")
    interpretation = policy.get("agent_interpretation")
    if not isinstance(interpretation, dict) or set(interpretation) != {
        "selected_config_id", "explanation"
    }:
        raise SystemExit("Frozen split candidate policy lacks agent interpretation.")
    if interpretation["selected_config_id"] != selected["config_id"]:
        raise SystemExit("Frozen split candidate policy agent selection is stale.")
    if not isinstance(interpretation["explanation"], str) or not interpretation[
        "explanation"
    ].strip():
        raise SystemExit("Frozen split candidate policy explanation is empty.")
    return policy


# --------------------------------------------------------------------------- #
# Merge-site (junction) candidate policy
# --------------------------------------------------------------------------- #
def merge_candidate_policy_source_paths(source_policy_path: Path) -> tuple[Path, ...]:
    """The sweep-bundle files the merge-site policy validation reads."""
    directory = source_policy_path.resolve().parent
    source = _load_json(source_policy_path, "merge candidate sweep policy")
    scope = source.get("evidence_scope")
    brains = scope.get("brains") if isinstance(scope, dict) else None
    if (not isinstance(brains, list) or not brains
            or any(not isinstance(brain, str) or not brain.isascii() or not brain.isdigit()
                   for brain in brains) or len(set(brains)) != len(brains)):
        raise SystemExit("Merge candidate sweep needs unique numeric brain IDs in evidence_scope.")
    return (
        source_policy_path.resolve(),
        directory / "artifact_manifest.json",
        directory / "tables" / "cross_dataset_aggregate.csv",
        *(directory / "per_brain" / f"{brain}_meta.json" for brain in brains),
    )


def _merge_positive_label_radius(value: object) -> float:
    try:
        radius = float(value)
    except (TypeError, ValueError) as exc:
        raise SystemExit("positive_label_radius_um must be numeric.") from exc
    if not math.isfinite(radius) or radius <= 0:
        raise SystemExit("positive_label_radius_um must be positive and finite.")
    return radius


def _read_merge_aggregate_rows(path: Path) -> list[dict]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as exc:
        raise SystemExit(
            f"Cannot read merge candidate aggregate table {path}: {exc}") from exc
    required = {
        "config_id", "mode", "nms_um", "claim_radius_um", "total_candidates",
        "min_site_recall", "mean_site_recall", "micro_site_recall",
        "min_label_recall", "n_brains",
    }
    if not rows:
        raise SystemExit(f"Merge candidate aggregate table {path} is empty.")
    missing = required - set(rows[0])
    if missing:
        raise SystemExit(
            f"Merge candidate aggregate table {path} is missing columns "
            f"{sorted(missing)}."
        )
    return rows


def _parse_merge_selected_row(row: dict) -> dict:
    try:
        nms_um = float(row["nms_um"])
        claim_radius = float(row["claim_radius_um"])
        total_candidates = int(row["total_candidates"])
        n_brains = int(row["n_brains"])
        metrics = {
            key: float(row[key])
            for key in (
                "min_site_recall", "mean_site_recall", "micro_site_recall",
                "min_label_recall",
            )
        }
    except (TypeError, ValueError, KeyError) as exc:
        raise SystemExit(f"Invalid numeric merge candidate aggregate row: {row}") from exc
    if (not math.isfinite(nms_um) or nms_um < 0
            or not math.isfinite(claim_radius) or claim_radius <= 0
            or total_candidates < 1 or n_brains < 1):
        raise SystemExit(f"Invalid merge candidate aggregate row: {row}")
    if any(
        not math.isfinite(value) or not 0 <= value <= 1
        for value in metrics.values()
    ):
        raise SystemExit(
            f"Merge candidate aggregate metrics must be finite in [0, 1]: {row}")
    mode = str(row["mode"]).strip()
    if mode not in SUPPORTED_MERGE_SITE_MODES:
        raise SystemExit(
            f"Merge candidate aggregate row has unsupported mode {mode!r}.")
    config_id = str(row["config_id"]).strip()
    expected_config_id = f"{mode}|nms={nms_um:g}|r={claim_radius:g}"
    if config_id != expected_config_id:
        raise SystemExit(
            "Merge candidate aggregate config_id is inconsistent with its "
            f"policy columns: {config_id!r} != {expected_config_id!r}."
        )
    return {
        "config_id": config_id,
        "mode": mode,
        "nms_um": int(nms_um) if nms_um.is_integer() else nms_um,
        "claim_radius_um": (
            int(claim_radius) if claim_radius.is_integer() else claim_radius),
        "n_brains": n_brains,
        "total_candidates": total_candidates,
        **metrics,
    }


def _merge_deterministic_selection(
    aggregate_path: Path, minimum_site_recall: float
) -> dict:
    eligible: list[dict] = []
    for raw in _read_merge_aggregate_rows(aggregate_path):
        parsed = _parse_merge_selected_row(raw)
        if parsed["min_site_recall"] + 1e-12 >= minimum_site_recall:
            eligible.append(parsed)
    if not eligible:
        raise SystemExit(
            "No merge candidate configuration meets worst-brain site recall >= "
            f"{minimum_site_recall:.6g}."
        )
    return min(eligible, key=lambda row: (row["total_candidates"], row["config_id"]))


def _validate_merge_cache_evidence(meta_path, entries, brain, signature, mcl, project_root):
    """Reject changed cache files without deserializing multi-GB payloads."""
    relative = f"per_brain/{brain}_meta.json"
    if not meta_path.is_file():
        raise SystemExit(f"Missing merge sweep cache evidence {meta_path}; rerun the sweep.")
    meta_sha = _require_manifest_hash(entries, relative, meta_path)
    meta = _load_json(meta_path, "merge sweep brain metadata")
    if meta.get("brain") != brain or meta.get("config_signature") != signature:
        raise SystemExit(f"Merge sweep metadata scope mismatch for brain {brain}.")
    identity = meta.get("cache_identity")
    if (not isinstance(identity, dict) or not isinstance(identity.get("path"), str)
            or not identity["path"]
            or any(type(identity.get(key)) is not int or identity[key] < 0
                   for key in ("size", "mtime_ns", "inode"))):
        raise SystemExit(f"Merge sweep lacks valid cache identity for brain {brain}; rerun the sweep.")
    recorded = meta.get("cache")
    if not isinstance(recorded, str) or not recorded:
        raise SystemExit(f"Merge sweep lacks cache path for brain {brain}.")
    cache_path = Path(identity["path"])
    cache_path = (cache_path if cache_path.is_absolute() else project_root / cache_path).resolve()
    recorded_path = Path(recorded)
    recorded_path = (recorded_path if recorded_path.is_absolute() else project_root / recorded_path).resolve()
    if cache_path != recorded_path or cache_path.name != f"dataset_cache_{brain}_mcl{mcl}_add.pkl":
        raise SystemExit(f"Merge sweep cache path/brain/MCL mismatch for brain {brain}.")
    try:
        stat = cache_path.stat()
    except OSError as exc:
        raise SystemExit(f"Cannot verify merge sweep cache {cache_path}: {exc}") from exc
    actual = {"path": str(cache_path), "size": stat.st_size,
              "mtime_ns": stat.st_mtime_ns, "inode": stat.st_ino}
    expected = {**identity, "path": str(cache_path)}
    if actual != expected:
        raise SystemExit(f"Merge sweep cache identity is stale for brain {brain}; rerun the sweep "
                         "against the current cache before building the detector.")
    return {"brain": brain, "metadata_path": _relative(meta_path, project_root),
            "metadata_sha256": meta_sha, "cache_identity": actual}


def _validated_merge_source_bundle(
    source_policy_path: Path,
    *,
    minimum_site_recall: float,
    expected_mcl: int | None,
    project_root: Path,
) -> tuple[dict, dict]:
    minimum_site_recall = _finite_probability(
        minimum_site_recall, "minimum worst-brain merge site recall"
    )
    source_policy_path, manifest_path, aggregate_path, *meta_paths = (
        merge_candidate_policy_source_paths(source_policy_path)
    )
    for path in (source_policy_path, manifest_path, aggregate_path):
        if not path.is_file():
            raise SystemExit(f"Missing merge candidate-policy source: {path}")

    entries = _manifest_entries(manifest_path)
    source_policy_sha = _require_manifest_hash(
        entries, "recommended_policy.json", source_policy_path
    )
    aggregate_sha = _require_manifest_hash(
        entries, "tables/cross_dataset_aggregate.csv", aggregate_path
    )
    source = _load_json(source_policy_path, "merge candidate sweep policy")
    if source.get("artifact_type") != MERGE_SITE_SOURCE_ARTIFACT_TYPE:
        raise SystemExit(
            f"Merge candidate sweep policy {source_policy_path} has the wrong "
            "artifact_type."
        )
    provenance_block = source.get("provenance")
    if (not isinstance(provenance_block, dict)
            or provenance_block.get("aggregate_sha256") != aggregate_sha):
        raise SystemExit(
            "Merge candidate sweep policy does not bind the current aggregate "
            "table."
        )
    scope_block = source.get("evidence_scope")
    if not isinstance(scope_block, dict):
        raise SystemExit("Merge candidate sweep policy lacks evidence_scope.")
    evidence_mcl = scope_block.get("mcl")
    if not isinstance(evidence_mcl, int):
        raise SystemExit("Merge candidate sweep policy needs an integer mcl.")
    if expected_mcl is not None and evidence_mcl != expected_mcl:
        raise SystemExit(
            f"Merge candidate sweep evidence is mcl{evidence_mcl}, but the "
            f"detector run name indicates mcl{expected_mcl}."
        )
    brains = scope_block.get("brains")
    if (not isinstance(brains, list) or not brains
            or any(not isinstance(brain, str) or not brain for brain in brains)):
        raise SystemExit("Merge candidate sweep evidence needs a brains list.")

    cache_evidence = [
        _validate_merge_cache_evidence(path, entries, brain, scope_block.get("config_signature"),
                                       evidence_mcl, project_root)
        for brain, path in zip(brains, meta_paths)
    ]

    selected = _merge_deterministic_selection(aggregate_path, minimum_site_recall)
    if selected["n_brains"] != len(brains):
        raise SystemExit(
            "Merge candidate sweep brain scope disagrees with the selected "
            "aggregate row."
        )
    scope = {
        "mcl": evidence_mcl,
        "config_signature": scope_block.get("config_signature"),
        "brains": brains,
        "segment_detector_gating_evaluated": scope_block.get(
            "segment_detector_gating_evaluated"
        ),
        "bridge_candidate_family_evaluated": scope_block.get(
            "bridge_candidate_family_evaluated"
        ),
    }
    structural_ceilings = source.get("structural_ceilings")
    provenance = {
        "source_policy_path": _relative(source_policy_path, project_root),
        "source_policy_sha256": source_policy_sha,
        "aggregate_path": _relative(aggregate_path, project_root),
        "aggregate_sha256": aggregate_sha,
        "manifest_path": _relative(manifest_path, project_root),
        "manifest_sha256": _sha256(manifest_path),
        "cache_evidence": cache_evidence,
    }
    return selected, {
        "scope": scope,
        "provenance": provenance,
        "structural_ceilings": structural_ceilings,
    }


def validate_merge_candidate_policy_sources(
    source_policy_path: Path,
    *,
    minimum_site_recall: float,
    expected_mcl: int | None,
    project_root: Path,
) -> dict:
    """Validate the merge sweep bundle before any output directory is deleted."""
    selected, metadata = _validated_merge_source_bundle(
        source_policy_path,
        minimum_site_recall=minimum_site_recall,
        expected_mcl=expected_mcl,
        project_root=project_root,
    )
    return {"selected": selected, **metadata}


def _merge_objective(minimum_site_recall: float,
                     positive_label_radius_um: float) -> dict:
    return {
        "name": MERGE_SITE_OBJECTIVE_NAME,
        "minimum_worst_brain_site_recall": minimum_site_recall,
        "positive_label_radius_um": positive_label_radius_um,
        "tie_breaker": ["total_candidates", "config_id"],
    }


def compile_merge_candidate_policy(
    output_path: Path,
    source_policy_path: Path,
    *,
    minimum_site_recall: float,
    positive_label_radius_um: float,
    expected_mcl: int | None,
    project_root: Path,
) -> dict:
    """Freeze merge_candidate_policy.json from the validated sweep bundle.

    The claim radius is the sweep's COVERAGE accounting tolerance; the tighter
    driver-owned ``positive_label_radius_um`` defines the runtime's positive
    training label so a far-fetched claim does not dilute site supervision.
    """
    positive_label_radius_um = _merge_positive_label_radius(
        positive_label_radius_um)
    selected, metadata = _validated_merge_source_bundle(
        source_policy_path,
        minimum_site_recall=minimum_site_recall,
        expected_mcl=expected_mcl,
        project_root=project_root,
    )
    if positive_label_radius_um > float(selected["claim_radius_um"]):
        raise SystemExit(
            "positive_label_radius_um must not exceed the selected claim "
            f"radius {selected['claim_radius_um']} um."
        )
    policy = {
        "schema_version": CANDIDATE_POLICY_SCHEMA_VERSION,
        "artifact_type": MERGE_SITE_FROZEN_ARTIFACT_TYPE,
        "application_status": APPLICATION_STATUS,
        "objective": _merge_objective(
            minimum_site_recall, positive_label_radius_um),
        "selected_policy": {
            "config_id": selected["config_id"],
            "mode": selected["mode"],
            "nms_um": selected["nms_um"],
            "claim_radius_um": selected["claim_radius_um"],
            "positive_label_radius_um": positive_label_radius_um,
        },
        "evidence_metrics": {
            key: selected[key]
            for key in (
                "n_brains", "total_candidates", "min_site_recall",
                "mean_site_recall", "micro_site_recall", "min_label_recall",
            )
        },
        "evidence_scope": metadata["scope"],
        "structural_ceilings": metadata["structural_ceilings"],
        "provenance": metadata["provenance"],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(policy, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return policy


def _merge_runtime_policy_from_artifact(policy: dict, policy_path: Path) -> dict:
    """Return the strictly validated merge-site generator fields to embed."""
    if policy.get("schema_version") != CANDIDATE_POLICY_SCHEMA_VERSION:
        raise SystemExit(
            f"Merge candidate policy {policy_path} has a stale schema_version.")
    if policy.get("artifact_type") != MERGE_SITE_FROZEN_ARTIFACT_TYPE:
        raise SystemExit(
            f"Merge candidate policy {policy_path} has the wrong artifact_type.")
    if policy.get("application_status") != APPLICATION_STATUS:
        raise SystemExit(
            f"Merge candidate policy {policy_path} is not selected for "
            "detector runtime."
        )
    selected = policy.get("selected_policy")
    required = {
        "config_id", "mode", "nms_um", "claim_radius_um",
        "positive_label_radius_um",
    }
    if not isinstance(selected, dict) or set(selected) != required:
        raise SystemExit(
            f"Merge candidate policy {policy_path} selected_policy must "
            f"contain exactly {sorted(required)}."
        )
    mode = selected.get("mode")
    if mode not in SUPPORTED_MERGE_SITE_MODES:
        raise SystemExit(
            f"Merge candidate policy {policy_path} has unsupported runtime "
            f"mode {mode!r}."
        )
    try:
        nms_um = float(selected.get("nms_um"))
        claim_radius = float(selected.get("claim_radius_um"))
    except (TypeError, ValueError) as exc:
        raise SystemExit(
            f"Merge candidate policy {policy_path} radii must be numeric."
        ) from exc
    if not math.isfinite(nms_um) or nms_um < 0:
        raise SystemExit(
            f"Merge candidate policy {policy_path} nms_um must be >= 0.")
    if not math.isfinite(claim_radius) or claim_radius <= 0:
        raise SystemExit(
            f"Merge candidate policy {policy_path} claim_radius_um must be "
            "positive."
        )
    label_radius = _merge_positive_label_radius(
        selected.get("positive_label_radius_um"))
    if label_radius > claim_radius:
        raise SystemExit(
            f"Merge candidate policy {policy_path} positive_label_radius_um "
            "must not exceed claim_radius_um."
        )
    expected_id = f"{mode}|nms={nms_um:g}|r={claim_radius:g}"
    if selected.get("config_id") != expected_id:
        raise SystemExit(
            f"Merge candidate policy {policy_path} config_id disagrees with "
            f"its fields: {selected.get('config_id')!r} != {expected_id!r}."
        )
    return {
        "config_id": expected_id,
        "mode": mode,
        "nms_um": int(nms_um) if nms_um.is_integer() else nms_um,
        "claim_radius_um": (
            int(claim_radius) if claim_radius.is_integer() else claim_radius),
        "positive_label_radius_um": label_radius,
    }


def load_runtime_merge_candidate_policy(policy_path: Path) -> tuple[dict, str]:
    """Load the exact merge-site policy fields and hash consumed by assembly."""
    policy = _load_json(policy_path, "frozen merge candidate policy")
    return (
        _merge_runtime_policy_from_artifact(policy, policy_path),
        _sha256(policy_path),
    )


def validate_frozen_merge_candidate_policy(
    policy_path: Path,
    source_policy_path: Path,
    *,
    minimum_site_recall: float,
    positive_label_radius_um: float,
    expected_mcl: int | None,
    project_root: Path,
) -> dict:
    positive_label_radius_um = _merge_positive_label_radius(
        positive_label_radius_um)
    policy = _load_json(policy_path, "frozen merge candidate policy")
    runtime_policy = _merge_runtime_policy_from_artifact(policy, policy_path)
    selected, metadata = _validated_merge_source_bundle(
        source_policy_path,
        minimum_site_recall=minimum_site_recall,
        expected_mcl=expected_mcl,
        project_root=project_root,
    )
    if policy.get("objective") != _merge_objective(
            minimum_site_recall, positive_label_radius_um):
        raise SystemExit("Frozen merge candidate policy objective is stale.")
    expected_selected = {
        "config_id": selected["config_id"],
        "mode": selected["mode"],
        "nms_um": selected["nms_um"],
        "claim_radius_um": selected["claim_radius_um"],
        "positive_label_radius_um": positive_label_radius_um,
    }
    if runtime_policy != expected_selected:
        raise SystemExit("Frozen merge candidate policy selection is stale.")
    expected_metrics = {
        key: selected[key]
        for key in (
            "n_brains", "total_candidates", "min_site_recall",
            "mean_site_recall", "micro_site_recall", "min_label_recall",
        )
    }
    if policy.get("evidence_metrics") != expected_metrics:
        raise SystemExit("Frozen merge candidate policy metrics are stale.")
    if policy.get("evidence_scope") != metadata["scope"]:
        raise SystemExit("Frozen merge candidate policy evidence scope is stale.")
    if policy.get("provenance") != metadata["provenance"]:
        raise SystemExit("Frozen merge candidate policy provenance is stale.")
    return policy
