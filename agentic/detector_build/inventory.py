"""Compile agent semantic judgments into the public inventory artifact."""

from __future__ import annotations

import json
from pathlib import Path
from .feature_scope import (
    MERGE_SITE_FEATURE_FIELDS,
    inventory_feature_scopes,
    validate_feature_scope,
)
from .split_feature_scope import SPLIT_FEATURE_FIELDS, validate_split_feature, split_inventory_specs

from .inputs import (
    corrected_status_by_id,
    load_json_object,
    rel_to_root,
    report_evidence_by_id,
    sha256,
    unbacked_correction_verdicts,
)
try:
    from agentic.split_feature_applicability import load_applicability
except ModuleNotFoundError:  # direct workflow-script import via detector_build
    from split_feature_applicability import load_applicability  # type: ignore[no-redef]


SEMANTIC_ROW_FIELDS = {
    "id",
    "included",
    "exclusion_reason",
    "correction_scope",
    "feature_source",
    "source_reason",
    "features",
}

SEMANTIC_FEATURE_FIELDS = {
    "name",
    "quantity",
    "constants",
    "aggregation",
    "reduction",
    "traversal_phase",
    "measurable_condition",
    "historical_undefined_sentinel",
}

SEMANTIC_DRAFT_CONTRACT = """\
The semantic draft is canonical JSON with exactly schema_version=1 and
hypotheses. It contains one row per selected ID in the supplied order. Every row
has exactly id, included, exclusion_reason, correction_scope, feature_source,
source_reason, and features.

Conditional invariants:
- included=true: exclusion_reason MUST be JSON null (not an empty string),
  feature_source is "rerun" or "fixed", source_reason is a non-empty string,
  and features is non-empty.
- included=false: exclusion_reason is a non-empty string, feature_source is JSON
  null, source_reason is JSON null or a non-empty string, and features is an
  empty list.
- correction_scope is one of none, test_only, feature_semantics_changed,
  stale_fixed_ignored, or unclear.
- Every feature has exactly name, quantity, constants, aggregation, reduction,
  traversal_phase, measurable_condition, and historical_undefined_sentinel.
"""


def canonicalize_semantic_draft(
    semantics_path: Path,
    project_root: Path,
) -> list[str]:
    """Canonicalize only representation changes with no semantic ambiguity."""
    draft = load_json_object(
        semantics_path, "feature semantics draft", project_root)
    rows = draft.get("hypotheses")
    if not isinstance(rows, list):
        return []
    changes: list[str] = []
    for index, row in enumerate(rows):
        if (isinstance(row, dict) and row.get("included") is True
                and row.get("exclusion_reason") == ""):
            row["exclusion_reason"] = None
            changes.append(
                f"hypotheses[{index}] id={row.get('id')} exclusion_reason: "
                '"" -> null (included=true)'
            )
    if changes:
        temporary = semantics_path.with_name(
            f".{semantics_path.name}.canonicalize.tmp")
        temporary.write_text(
            json.dumps(draft, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(semantics_path)
    return changes


def validate_semantic_draft(
    semantics_path: Path,
    *,
    selected_ids: list[int],
    project_root: Path,
    merge_site: bool = False,
    split: bool = False,
) -> dict:
    """Validate the agent-owned semantic schema before provenance compilation."""
    draft = load_json_object(
        semantics_path, "feature semantics draft", project_root)
    if merge_site and split:
        raise SystemExit("A semantic draft cannot be both merge-site and split.")
    if set(draft) != {"schema_version", "hypotheses"}:
        raise SystemExit(
            "Feature semantics draft must contain exactly schema_version and "
            "hypotheses."
        )
    if draft.get("schema_version") != 1:
        raise SystemExit("Feature semantics draft must set schema_version to 1.")
    rows = draft.get("hypotheses")
    if not isinstance(rows, list):
        raise SystemExit("Feature semantics draft hypotheses must be a list.")
    ids = [row.get("id") if isinstance(row, dict) else None for row in rows]
    if ids != selected_ids:
        raise SystemExit(
            "Feature semantics draft must contain exactly one row per selected "
            "hypothesis id, in selected_ids order."
        )
    allowed_scopes = {
        "none", "test_only", "feature_semantics_changed",
        "stale_fixed_ignored", "unclear",
    }
    feature_names: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != SEMANTIC_ROW_FIELDS:
            raise SystemExit(
                f"Feature semantics row {index} must contain exactly: "
                + ", ".join(sorted(SEMANTIC_ROW_FIELDS))
            )
        hypothesis_id = row["id"]
        if not isinstance(row["included"], bool):
            raise SystemExit(
                f"Semantic hypothesis {hypothesis_id} included must be boolean.")
        if row["correction_scope"] not in allowed_scopes:
            raise SystemExit(
                f"Semantic hypothesis {hypothesis_id} has invalid correction_scope.")
        if row["included"]:
            if (not isinstance(row["source_reason"], str)
                    or not row["source_reason"].strip()):
                raise SystemExit(
                    f"Included hypothesis {hypothesis_id} needs a source_reason.")
            if row["exclusion_reason"] is not None:
                raise SystemExit(
                    f"Included hypothesis {hypothesis_id} must have JSON null "
                    "exclusion_reason, not an empty string."
                )
            if row["feature_source"] not in {"rerun", "fixed"}:
                raise SystemExit(
                    f"Included hypothesis {hypothesis_id} needs rerun/fixed "
                    "feature_source."
                )
            if not isinstance(row["features"], list) or not row["features"]:
                raise SystemExit(
                    f"Included hypothesis {hypothesis_id} needs non-empty features."
                )
        else:
            if (not isinstance(row["exclusion_reason"], str)
                    or not row["exclusion_reason"].strip()):
                raise SystemExit(
                    f"Excluded hypothesis {hypothesis_id} needs a non-empty "
                    "exclusion_reason."
                )
            if row["feature_source"] is not None:
                raise SystemExit(
                    f"Excluded hypothesis {hypothesis_id} must have null "
                    "feature_source."
                )
            if row["source_reason"] is not None and (
                not isinstance(row["source_reason"], str)
                or not row["source_reason"].strip()
            ):
                raise SystemExit(
                    f"Excluded hypothesis {hypothesis_id} source_reason must be "
                    "JSON null or a non-empty string."
                )
            if row["features"] != []:
                raise SystemExit(
                    f"Excluded hypothesis {hypothesis_id} must have empty features."
                )
        for feature_index, feature in enumerate(row["features"]):
            fields = SEMANTIC_FEATURE_FIELDS | (MERGE_SITE_FEATURE_FIELDS if merge_site else set())
            if split:
                fields |= SPLIT_FEATURE_FIELDS
            if not isinstance(feature, dict) or set(feature) != fields:
                raise SystemExit(
                    f"Semantic hypothesis {hypothesis_id} feature {feature_index} "
                    "must contain exactly: "
                    + ", ".join(sorted(fields))
                )
            name = feature["name"]
            if not isinstance(name, str) or not name.strip():
                raise SystemExit(
                    f"Semantic hypothesis {hypothesis_id} has an empty feature name.")
            if name in feature_names:
                raise SystemExit(f"Semantic draft duplicates feature name {name!r}.")
            feature_names.add(name)
            if merge_site:
                validate_feature_scope(feature)
            if split:
                validate_split_feature(feature)
    if merge_site:
        inventory_feature_scopes({"schema_version": 4, "hypotheses": rows})
    return draft


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
    reconcile_unbacked_verdicts: bool = False,
    split_applicability_path: Path | None = None,
    run_json: Path | None = None,
    merge_site: bool = False,
) -> list[str]:
    """Merge semantic judgments with deterministic same-ID provenance.

    Legacy merge output is v2; merge-site output is v4 with explicit scope.
    Split output is v5 with source-validated applicability, explicit anchoring
    and occurrence reduction. Applicability provenance remains driver-owned.

    ``reconcile_unbacked_verdicts`` mirrors the input gate's flag: a report
    post-correction verdict with no USABLE corrected measurement is transcribed
    as null, so the compiled inventory obeys the verdict-requires-measurement
    invariant even when the report does not.
    """
    changes = canonicalize_semantic_draft(semantics_path, project_root)
    draft = validate_semantic_draft(
        semantics_path, selected_ids=selected_ids, project_root=project_root,
        merge_site=merge_site, split=split_applicability_path is not None)
    if merge_site and split_applicability_path is not None:
        raise SystemExit("Merge-site scope and split applicability are mutually exclusive.")
    semantic_rows = draft["hypotheses"]
    if (split_applicability_path is None) != (run_json is None):
        raise SystemExit(
            "Split inventory compilation requires both applicability path and run JSON."
        )
    applicability = (
        load_applicability(
            split_applicability_path,
            run_json=run_json,
            selected_ids=selected_ids,
            rerun_dir=rerun_dir,
            fixed_dir=fixed_dir,
            project_root=project_root,
        )
        if split_applicability_path is not None and run_json is not None
        else None
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
    unbacked_ids = (
        {
            hypothesis_id
            for hypothesis_id, _, _ in unbacked_correction_verdicts(
                evidence, corrected_statuses, [row["id"] for row in semantic_rows]
            )
        }
        if reconcile_unbacked_verdicts
        else set()
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

        features = [dict(feature) for feature in row["features"]]
        if included and applicability is not None:
            assert source_path is not None
            annotation = applicability.get((hypothesis_id, sha256(source_path)))
            if annotation is None:
                raise SystemExit(
                    f"Included split hypothesis {hypothesis_id} has no applicability "
                    "record for its selected feature source."
                )
            requirement = annotation["node_role_requirement"]
            if requirement == "unclear":
                raise SystemExit(
                    f"Included split hypothesis {hypothesis_id} has unclear node-role "
                    "requirements; exclude it rather than generating guessed feature code."
                )
            for feature in features:
                feature["node_role_requirement"] = requirement

        compiled_rows.append({
            "id": hypothesis_id,
            "included": included,
            "exclusion_reason": exclusion_reason,
            "reproduction_status": report_row["reproduction_status"],
            "statistical_verdict": report_row["statistical_verdict"],
            "corrected_result_status": corrected_statuses.get(hypothesis_id),
            "post_correction_verdict": (
                None
                if hypothesis_id in unbacked_ids
                else report_row["post_correction_verdict"]
            ),
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
            "features": features,
        })

    inventory = {
        "schema_version": 4 if merge_site else (5 if applicability is not None else 2),
        "selection_manifest": selection_manifest,
        "selected_ids": selected_ids,
        "hypotheses": compiled_rows,
    }
    if applicability is not None:
        split_inventory_specs(inventory)
        assert split_applicability_path is not None
        inventory["split_feature_applicability"] = rel_to_root(
            split_applicability_path, project_root
        )
        inventory["split_feature_applicability_sha256"] = sha256(
            split_applicability_path
        )
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
    return changes
