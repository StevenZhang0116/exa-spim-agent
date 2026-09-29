"""Source semantics and runtime-owned anchoring for split candidate features."""
from __future__ import annotations

import json
from pathlib import Path


SPLIT_FEATURE_FIELDS = {
    "scope", "anchor", "occurrence_reduction", "source_feature",
    "source_formula", "source_aggregation", "adaptation",
}
SPLIT_ANCHORS = {"tip_anchor", "endpoint_pair", "gap_midpoint", "component_pair", "segment_pair"}
SPLIT_REDUCTIONS = {"first_compatible", "min", "max", "mean", "sum", "all_compatible", "pair_once"}
SPLIT_SEMANTIC_CONTRACT = """
For split detection every feature additionally has EXACTLY these fields:
- scope: "candidate" (one value per canonical unordered segment pair).
- anchor: "tip_anchor", "endpoint_pair", "gap_midpoint", "component_pair",
  or "segment_pair". Choose from the SOURCE formula, not its traversal phase.
- occurrence_reduction: "first_compatible", "min", "max", "mean", "sum",
  "all_compatible", or "pair_once". first_compatible uses the closest compatible
  occurrence even if its formula is undefined; do not silently try later ones.
  min/max/mean/sum reduce defined scalar results over compatible occurrences.
  all_compatible passes their full tuple to a joint formula (e.g. variance).
  pair_once is required only for segment_pair context, with no_tip_requirement.
- source_feature, source_formula, source_aggregation: non-empty descriptions
  of the ORIGINAL source quantity, exact mathematical expression and reduction.
  Check units, numerator/denominator order, signs and constants against source
  code; source_formula and quantity must not contradict each other.
- adaptation: "identity" or "candidate_pair". identity preserves the source
  formula and reduction. An explicit change of anchor or reduction is a
  candidate_pair adaptation, must use a NEW name distinct from source_feature,
  and must describe the change in quantity/aggregation/reduction. Never repair
  a performance problem by altering the scientific definition.
The driver still supplies node_role_requirement from source-bound annotations.
tip_anchor must have requires_tip_anchor or requires_both_tips applicability.
Do not equate canonical side a with the tip: the recorded anchor may be side b.
Context features are composed into each candidate PAIR; no single-segment value
may be implicitly broadcast to every pair sharing that segment.
""".strip()


def validate_split_feature(feature: dict, *, compiled: bool = False) -> None:
    name = feature.get("name")
    if feature.get("scope") != "candidate":
        raise SystemExit(f"Split feature {name!r} must have scope=candidate.")
    anchor = feature.get("anchor")
    reduction = feature.get("occurrence_reduction")
    if anchor not in SPLIT_ANCHORS or reduction not in SPLIT_REDUCTIONS:
        raise SystemExit(f"Split feature {name!r} has invalid anchor or occurrence_reduction.")
    if (anchor == "segment_pair") != (reduction == "pair_once"):
        raise SystemExit(f"Split feature {name!r}: segment_pair requires pair_once and vice versa.")
    for field in ("source_feature", "source_formula", "source_aggregation"):
        if not isinstance(feature.get(field), str) or not feature[field].strip():
            raise SystemExit(f"Split feature {name!r} needs non-empty {field}.")
    if feature.get("adaptation") not in {"identity", "candidate_pair"}:
        raise SystemExit(f"Split feature {name!r} has invalid adaptation.")
    if feature["adaptation"] == "candidate_pair" and name == feature["source_feature"]:
        raise SystemExit(f"Split feature {name!r}: candidate_pair adaptation requires a new name.")
    if compiled:
        requirement = feature.get("node_role_requirement")
        if requirement not in {"requires_tip_anchor", "requires_both_tips", "no_tip_requirement"}:
            raise SystemExit(f"Split feature {name!r} lacks validated node-role applicability.")
        if anchor == "tip_anchor" and requirement == "no_tip_requirement":
            raise SystemExit(f"Split feature {name!r}: tip_anchor needs a tip requirement.")
        if anchor == "segment_pair" and requirement != "no_tip_requirement":
            raise SystemExit(f"Split feature {name!r}: segment_pair must have no_tip_requirement.")


def split_inventory_specs(inventory: dict) -> dict[str, dict]:
    if inventory.get("schema_version") != 5:
        raise SystemExit("Split features require inventory schema_version=5; rebuild inventory.")
    specs = {}
    for row in inventory.get("hypotheses", []):
        if not row.get("included"):
            continue
        for feature in row.get("features", []):
            validate_split_feature(feature, compiled=True)
            name = feature.get("name")
            if not isinstance(name, str) or not name or name.endswith("_is_defined") or name in specs:
                raise SystemExit("Invalid or duplicate split feature name.")
            specs[name] = {key: feature[key] for key in (
                "anchor", "occurrence_reduction", "node_role_requirement")}
    if not specs:
        raise SystemExit("Split inventory must include at least one feature.")
    return specs


def split_scope_runtime_source(inventory_path: Path) -> tuple[str, dict[str, str]]:
    try:
        inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SystemExit(f"Cannot read split inventory: {exc}") from exc
    specs = split_inventory_specs(inventory)
    runtime = Path(__file__).with_name("templates") / "split_feature_accumulator.py.tmpl"
    return ("SPLIT_FEATURE_SPECS = " + repr(specs) + "\n" + runtime.read_text(encoding="utf-8"),
            {name: "candidate" for name in specs})
