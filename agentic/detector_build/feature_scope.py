"""Structured merge-site feature semantics and the reviewed storage runtime."""

from __future__ import annotations

import json
from pathlib import Path


MERGE_SITE_FEATURE_FIELDS = {"scope", "source_feature", "source_aggregation", "adaptation"}
MERGE_SITE_SEMANTIC_CONTRACT = """
For merge-site detection every feature additionally has EXACTLY these fields:
- scope: "candidate" or "segment". This is independent of traversal_phase.
- source_feature: the name of the scalar quantity in the selected source.
- source_aggregation: a non-empty description of the ORIGINAL aggregation.
- adaptation: "identity" or "candidate_local".
An identity feature preserves the source definition, including its aggregation.
A candidate_local feature evaluates the source's local statistic at THIS
candidate node, omitting the original across-junction/segment reduction. It
must have scope=candidate and a NEW name different from source_feature. Keep
the local formula, units, radii and other constants unchanged. Describe the
change explicitly in quantity/aggregation/reduction and the candidate-specific
measurable_condition. Do not describe removing a reduction as value-preserving.
Keep an original whole-segment aggregate only as an explicit scope=segment,
adaptation=identity feature. Never call a segment minimum a candidate feature.
Do not infer scope from traversal phase: even a junction traversal can compute
a segment density. Exclude features without an honest blind implementation.
At least one included candidate feature is required for a localization model.
""".strip()


def validate_feature_scope(feature: dict) -> None:
    name = feature.get("name")
    if feature.get("scope") not in {"candidate", "segment"}:
        raise SystemExit(f"Feature {name!r} needs scope=candidate or segment.")
    for field in ("source_feature", "source_aggregation"):
        if not isinstance(feature.get(field), str) or not feature[field].strip():
            raise SystemExit(f"Feature {name!r} needs non-empty {field}.")
    adaptation = feature.get("adaptation")
    if adaptation not in {"identity", "candidate_local"}:
        raise SystemExit(f"Feature {name!r} has invalid adaptation.")
    if adaptation == "candidate_local":
        if feature["scope"] != "candidate" or name == feature["source_feature"]:
            raise SystemExit(
                f"Feature {name!r}: candidate_local requires candidate scope "
                "and a new name distinct from source_feature.")


def inventory_feature_scopes(inventory: dict) -> dict[str, str]:
    """Read authoritative scopes; reject legacy inventories instead of guessing."""
    if inventory.get("schema_version") != 4:
        raise SystemExit("Merge-site features require inventory schema_version=4; rebuild inventory.")
    scopes = {}
    for row in inventory.get("hypotheses", []):
        if not row.get("included"):
            continue
        for feature in row.get("features", []):
            validate_feature_scope(feature)
            name = feature.get("name")
            if not isinstance(name, str) or not name or name.endswith("_is_defined"):
                raise SystemExit("Invalid merge-site feature name.")
            if name in scopes:
                raise SystemExit(f"Duplicate merge-site feature {name!r}.")
            scopes[name] = feature["scope"]
    if "candidate" not in scopes.values():
        raise SystemExit("Merge-site inventory requires at least one candidate feature.")
    return scopes


def feature_scope_runtime_source(inventory_path: Path) -> tuple[str, dict[str, str]]:
    try:
        inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SystemExit(f"Cannot read merge-site inventory: {exc}") from exc
    scopes = inventory_feature_scopes(inventory)
    runtime = Path(__file__).with_name("templates") / "merge_feature_accumulator.py.tmpl"
    return (
        "MERGE_FEATURE_SCOPES = " + repr(scopes) + "\n" + runtime.read_text(encoding="utf-8"),
        scopes,
    )
