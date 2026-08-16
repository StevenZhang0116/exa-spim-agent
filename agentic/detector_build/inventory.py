"""Compile agent semantic judgments into the public inventory artifact."""

from __future__ import annotations

import json
from pathlib import Path

from .inputs import (
    corrected_status_by_id,
    load_json_object,
    rel_to_root,
    report_evidence_by_id,
    sha256,
)


SEMANTIC_ROW_FIELDS = {
    "id",
    "included",
    "exclusion_reason",
    "correction_scope",
    "feature_source",
    "source_reason",
    "features",
}


def compile_feature_inventory(
    semantics_path: Path,
    inventory_path: Path,
    *,
    selected_ids: list[int],
    selection_manifest: str | None,
    summary_path: Path,
    corrected_results_path: Path | None,
    rerun_dir: Path,
    fixed_dir: Path | None,
    project_root: Path,
) -> None:
    """Merge semantic judgments with deterministic same-ID provenance.

    The resulting on-disk schema is inventory v2 and is intentionally identical
    to the workflow's previous public output.  Only its construction changes.
    """
    draft = load_json_object(semantics_path, "feature semantics draft", project_root)
    if set(draft) != {"schema_version", "hypotheses"}:
        raise SystemExit(
            "Feature semantics draft must contain exactly schema_version and "
            "hypotheses."
        )
    if draft.get("schema_version") != 1:
        raise SystemExit("Feature semantics draft must set schema_version to 1.")
    semantic_rows = draft.get("hypotheses")
    if not isinstance(semantic_rows, list):
        raise SystemExit("Feature semantics draft hypotheses must be a list.")
    semantic_ids = [
        row.get("id") if isinstance(row, dict) else None for row in semantic_rows
    ]
    if semantic_ids != selected_ids:
        raise SystemExit(
            "Feature semantics draft must contain exactly one row per selected "
            "hypothesis id, in selected_ids order."
        )

    evidence = report_evidence_by_id(summary_path, project_root)
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
    compiled_rows: list[dict] = []
    for row in semantic_rows:
        hypothesis_id = row["id"]
        if set(row) != SEMANTIC_ROW_FIELDS:
            raise SystemExit(
                f"Feature semantics hypothesis {hypothesis_id} must contain "
                "exactly: " + ", ".join(sorted(SEMANTIC_ROW_FIELDS))
            )
        report_row = evidence.get(hypothesis_id)
        if report_row is None:
            raise SystemExit(
                f"Finished report has no row with explicit hypothesis ID "
                f"{hypothesis_id}."
            )
        rerun_path = rerun_dir / f"hypo_{hypothesis_id}.py"
        fixed_path = fixed_dir / f"hypo_{hypothesis_id}.py" if fixed_dir else None
        fixed_exists = fixed_path is not None and fixed_path.is_file()
        included = row["included"]
        exclusion_reason = row["exclusion_reason"]
        correction_scope = row["correction_scope"]
        source_reason = row["source_reason"]

        # Once a hypothesis is excluded, no feature source reaches the detector.
        # Do not spend another agent turn classifying an unused fixed script or
        # restating the same exclusion in a second prose field.  Keep included
        # rows strict: their source choice and correction scope affect feature
        # semantics and must remain explicit agent judgments.
        if not included:
            if source_reason is None and isinstance(exclusion_reason, str):
                source_reason = f"Excluded: {exclusion_reason}"
            if fixed_exists and correction_scope == "none":
                correction_scope = "unclear"

        feature_source = row["feature_source"]
        if feature_source == "rerun":
            source_path = rerun_path
        elif feature_source == "fixed" and fixed_exists:
            source_path = fixed_path
        else:
            source_path = None

        compiled_rows.append({
            "id": hypothesis_id,
            "included": included,
            "exclusion_reason": exclusion_reason,
            "reproduction_status": report_row["reproduction_status"],
            "statistical_verdict": report_row["statistical_verdict"],
            "corrected_result_status": corrected_statuses.get(hypothesis_id),
            "post_correction_verdict": report_row["post_correction_verdict"],
            "correction_scope": correction_scope,
            "rerun_path": rel_to_root(rerun_path, project_root),
            "rerun_sha256": sha256(rerun_path),
            "fixed_path": (
                rel_to_root(fixed_path, project_root) if fixed_exists else None
            ),
            "fixed_sha256": sha256(fixed_path) if fixed_exists else None,
            "feature_source": feature_source,
            "feature_source_path": (
                rel_to_root(source_path, project_root) if source_path else None
            ),
            "feature_source_sha256": sha256(source_path) if source_path else None,
            "source_reason": source_reason,
            "features": row["features"],
        })

    inventory = {
        "schema_version": 2,
        "selection_manifest": selection_manifest,
        "selected_ids": selected_ids,
        "hypotheses": compiled_rows,
    }
    temporary = inventory_path.with_name(f".{inventory_path.name}.compile.tmp")
    try:
        temporary.write_text(
            json.dumps(inventory, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(inventory_path)
    except OSError as exc:
        raise SystemExit(
            f"Cannot compile feature inventory "
            f"{rel_to_root(inventory_path, project_root)}: {exc}"
        ) from exc
