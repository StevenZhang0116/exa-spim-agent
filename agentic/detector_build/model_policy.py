"""Versioned, non-executable model policy and configuration validation."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from .inputs import load_json_object, sha256


PARAMETER_RULES = frozenset({
    "positive_number",
    "nonnegative_number",
    "unit_interval",
    "positive_unit_interval",
    "positive_integer",
    "nonnegative_integer",
    "integer_min_2",
    "nullable_positive_integer",
    "max_features",
})

# Policy may choose among implemented adapters; it cannot create executable
# model behavior merely by naming a new class or package.
SAFE_MODEL_ADAPTERS = frozenset({
    "logistic_l2",
    "logistic_elasticnet",
    "hist_gradient_boosting",
    "spline_logistic",
    "explainable_boosting",
    "extra_trees",
    "random_forest",
    "xgboost",
})


def load_model_policy(path: Path) -> dict:
    try:
        policy = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Cannot load detector model policy {path}: {exc}") from exc
    if not isinstance(policy, dict) or set(policy) != {
        "schema_version", "selection", "families",
    } or policy.get("schema_version") != 1:
        raise RuntimeError("Detector model policy must use the exact schema v1 fields.")
    selection = policy.get("selection")
    if not isinstance(selection, dict) or set(selection) != {
        "max_optional_models", "max_grid_combinations", "simplicity_order",
        "primary_metric", "selection_rule", "outer_folds", "inner_folds",
        "random_seed", "review_budget",
    }:
        raise RuntimeError("Detector model policy has an invalid selection section.")
    for key in (
        "max_optional_models", "max_grid_combinations", "outer_folds",
        "inner_folds", "review_budget",
    ):
        if (
            isinstance(selection[key], bool)
            or not isinstance(selection[key], int)
            or selection[key] < 1
        ):
            raise RuntimeError(
                f"Detector model policy {key} must be a positive integer."
            )
    if selection["outer_folds"] < 2 or selection["inner_folds"] < 2:
        raise RuntimeError("Detector model policy CV fold counts must be at least 2.")
    if (
        isinstance(selection["random_seed"], bool)
        or not isinstance(selection["random_seed"], int)
    ):
        raise RuntimeError("Detector model policy random_seed must be an integer.")
    if selection["primary_metric"] not in {"average_precision", "roc_auc"}:
        raise RuntimeError("Detector model policy uses an unsupported primary_metric.")
    if selection["selection_rule"] != "one_standard_error":
        raise RuntimeError("Detector model policy uses an unsupported selection_rule.")

    families = policy.get("families")
    if not isinstance(families, dict) or not families:
        raise RuntimeError("Detector model policy needs a non-empty families object.")
    if not set(families) <= SAFE_MODEL_ADAPTERS:
        raise RuntimeError(
            "Detector model policy names families without reviewed safe adapters: "
            + ", ".join(sorted(set(families) - SAFE_MODEL_ADAPTERS))
        )
    family_fields = {"required", "native_nan", "requires_package", "parameters"}
    for name, family in families.items():
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(family, dict)
            or set(family) != family_fields
        ):
            raise RuntimeError(f"Detector model policy family {name!r} is malformed.")
        if not isinstance(family["required"], bool) or not isinstance(
            family["native_nan"], bool
        ):
            raise RuntimeError(
                f"Detector model policy family {name} has invalid booleans."
            )
        if family["requires_package"] is not None and not isinstance(
            family["requires_package"], str
        ):
            raise RuntimeError(
                f"Detector model policy family {name} has invalid package."
            )
        parameters = family["parameters"]
        if not isinstance(parameters, dict) or not parameters:
            raise RuntimeError(
                f"Detector model policy family {name} needs parameters."
            )
        unknown_rules = set(parameters.values()) - PARAMETER_RULES
        if unknown_rules:
            raise RuntimeError(
                f"Detector model policy family {name} uses unknown rules: "
                + ", ".join(sorted(unknown_rules))
            )
    simplicity_order = selection["simplicity_order"]
    if (
        not isinstance(simplicity_order, list)
        or simplicity_order != list(dict.fromkeys(simplicity_order))
        or set(simplicity_order) != set(families)
    ):
        raise RuntimeError(
            "Detector model policy simplicity_order must list every family once."
        )
    return policy


@dataclass(frozen=True)
class ModelPolicyContract:
    """Convenient immutable view of the JSON policy."""

    payload: dict

    @classmethod
    def from_path(cls, path: Path) -> "ModelPolicyContract":
        return cls(load_model_policy(path))

    @property
    def families(self) -> dict:
        return self.payload["families"]

    @property
    def selection(self) -> dict:
        return self.payload["selection"]

    @property
    def allowed_families(self) -> frozenset[str]:
        return frozenset(self.families)

    @property
    def required_families(self) -> frozenset[str]:
        return frozenset(
            name for name, spec in self.families.items() if spec["required"]
        )

    @property
    def optional_families(self) -> frozenset[str]:
        return self.allowed_families - self.required_families

    @property
    def parameter_rules(self) -> dict[str, dict[str, str]]:
        return {
            name: dict(spec["parameters"])
            for name, spec in self.families.items()
        }

    @property
    def native_nan(self) -> dict[str, bool]:
        return {name: spec["native_nan"] for name, spec in self.families.items()}

    @property
    def required_package(self) -> dict[str, str]:
        return {
            name: spec["requires_package"]
            for name, spec in self.families.items()
            if spec["requires_package"] is not None
        }


def valid_grid_value(rule: str, value) -> bool:
    is_int = isinstance(value, int) and not isinstance(value, bool)
    is_number = (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and (not isinstance(value, float) or math.isfinite(value))
    )
    if rule == "positive_number":
        return is_number and value > 0
    if rule == "nonnegative_number":
        return is_number and value >= 0
    if rule == "unit_interval":
        return is_number and 0 <= value <= 1
    if rule == "positive_unit_interval":
        return is_number and 0 < value <= 1
    if rule == "positive_integer":
        return is_int and value >= 1
    if rule == "nonnegative_integer":
        return is_int and value >= 0
    if rule == "integer_min_2":
        return is_int and value >= 2
    if rule == "nullable_positive_integer":
        return value is None or (is_int and value >= 1)
    if rule == "max_features":
        return (
            value is None
            or value in {"sqrt", "log2"}
            or (is_int and value >= 1)
            or (is_number and not is_int and 0 < value <= 1)
        )
    return False


def validate_model_config(
    config_path: Path,
    inventory_path: Path,
    policy_path: Path,
    contract: ModelPolicyContract,
    schema_version: int,
    project_root: Path,
) -> tuple[int, int]:
    """Validate config and return ``(candidate_count, optional_count)``."""
    config = load_json_object(config_path, "model candidate configuration", project_root)
    top_level_fields = {
        "schema_version",
        "feature_inventory_sha256",
        "model_policy_sha256",
        "selection_basis",
        "candidates",
    }
    if set(config) != top_level_fields:
        raise SystemExit(
            "Model candidate configuration must contain exactly: "
            + ", ".join(sorted(top_level_fields))
        )
    if config.get("schema_version") != schema_version:
        raise SystemExit(
            f"Model candidate configuration must set schema_version to {schema_version}."
        )
    if config.get("feature_inventory_sha256") != sha256(inventory_path):
        raise SystemExit(
            "Model candidate configuration feature_inventory_sha256 is stale."
        )
    if config.get("model_policy_sha256") != sha256(policy_path):
        raise SystemExit(
            "Model candidate configuration model_policy_sha256 is stale."
        )
    if not isinstance(config.get("selection_basis"), str) or not config[
        "selection_basis"
    ].strip():
        raise SystemExit("Model candidate configuration needs a selection_basis.")
    candidates = config.get("candidates")
    if not isinstance(candidates, list):
        raise SystemExit("Model candidate configuration candidates must be a list.")

    required_fields = {
        "name", "role", "reason", "grid", "native_nan", "requires_package",
    }
    names: list[str] = []
    optional_count = 0
    parameter_rules = contract.parameter_rules
    parameter_allowlist = {
        name: frozenset(rules) for name, rules in parameter_rules.items()
    }
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict) or set(candidate) != required_fields:
            raise SystemExit(
                f"Model candidate {index} must contain exactly: "
                + ", ".join(sorted(required_fields))
            )
        name = candidate["name"]
        if not isinstance(name, str) or name not in contract.allowed_families:
            raise SystemExit(
                f"Model candidate {index} has disallowed family {name!r}."
            )
        if name in names:
            raise SystemExit(f"Model candidate configuration duplicates {name}.")
        names.append(name)
        expected_role = "baseline" if name in contract.required_families else "optional"
        if candidate["role"] != expected_role:
            raise SystemExit(f"Model candidate {name} must have role={expected_role}.")
        optional_count += expected_role == "optional"
        if not isinstance(candidate["reason"], str) or not candidate["reason"].strip():
            raise SystemExit(f"Model candidate {name} needs a non-empty reason.")
        expected_native_nan = contract.native_nan[name]
        if candidate["native_nan"] is not expected_native_nan:
            raise SystemExit(
                f"Model candidate {name} native_nan must be {expected_native_nan}."
            )
        expected_package = contract.required_package.get(name)
        if candidate["requires_package"] != expected_package:
            raise SystemExit(
                f"Model candidate {name} requires_package must be "
                f"{expected_package!r}."
            )

        grid = candidate["grid"]
        if not isinstance(grid, dict) or not grid:
            raise SystemExit(
                f"Model candidate {name} grid must be a non-empty object."
            )
        unknown = set(grid) - parameter_allowlist[name]
        if unknown:
            raise SystemExit(
                f"Model candidate {name} has disallowed parameters: "
                + ", ".join(sorted(unknown))
            )
        combinations = 1
        for parameter, values in grid.items():
            if not isinstance(values, list) or not values:
                raise SystemExit(
                    f"Model candidate {name} parameter {parameter} needs a "
                    "non-empty list."
                )
            if any(isinstance(value, (dict, list)) for value in values):
                raise SystemExit(
                    f"Model candidate {name} parameter {parameter} values must "
                    "be JSON scalars."
                )
            invalid_values = [
                value
                for value in values
                if not valid_grid_value(parameter_rules[name][parameter], value)
            ]
            if invalid_values:
                raise SystemExit(
                    f"Model candidate {name} parameter {parameter} has invalid "
                    f"values: {invalid_values!r}."
                )
            combinations *= len(values)
        if combinations > contract.selection["max_grid_combinations"]:
            raise SystemExit(
                f"Model candidate {name} grid has {combinations} combinations; "
                f"maximum is {contract.selection['max_grid_combinations']}."
            )

    missing = contract.required_families - set(names)
    if missing:
        raise SystemExit(
            "Model candidate configuration is missing required baselines: "
            + ", ".join(sorted(missing))
        )
    maximum_optional = contract.selection["max_optional_models"]
    if optional_count > maximum_optional:
        raise SystemExit(
            f"Model candidate configuration has {optional_count} optional models; "
            f"maximum is {maximum_optional}."
        )
    return len(names), optional_count


def compile_model_config(
    advice_path: Path,
    config_path: Path,
    inventory_path: Path,
    policy_path: Path,
    contract: ModelPolicyContract,
    schema_version: int,
    project_root: Path,
) -> None:
    """Compile agent choices into the unchanged public model-config schema."""
    advice = load_json_object(advice_path, "model advice draft", project_root)
    if set(advice) != {"schema_version", "selection_basis", "candidates"}:
        raise SystemExit(
            "Model advice draft must contain exactly schema_version, "
            "selection_basis, and candidates."
        )
    if advice.get("schema_version") != 1:
        raise SystemExit("Model advice draft must set schema_version to 1.")
    if not isinstance(advice.get("selection_basis"), str) or not advice[
        "selection_basis"
    ].strip():
        raise SystemExit("Model advice draft needs a selection_basis.")
    advice_candidates = advice.get("candidates")
    if not isinstance(advice_candidates, list):
        raise SystemExit("Model advice draft candidates must be a list.")

    candidates: list[dict] = []
    for index, candidate in enumerate(advice_candidates):
        if not isinstance(candidate, dict) or set(candidate) != {
            "name", "reason", "grid",
        }:
            raise SystemExit(
                f"Model advice candidate {index} must contain exactly name, "
                "reason, and grid."
            )
        name = candidate["name"]
        if not isinstance(name, str) or name not in contract.allowed_families:
            raise SystemExit(
                f"Model advice candidate {index} has disallowed family {name!r}."
            )
        candidates.append({
            "name": name,
            "role": "baseline" if name in contract.required_families else "optional",
            "reason": candidate["reason"],
            "grid": candidate["grid"],
            "native_nan": contract.native_nan[name],
            "requires_package": contract.required_package.get(name),
        })

    config = {
        "schema_version": schema_version,
        "feature_inventory_sha256": sha256(inventory_path),
        "model_policy_sha256": sha256(policy_path),
        "selection_basis": advice["selection_basis"],
        "candidates": candidates,
    }
    temporary = config_path.with_name(f".{config_path.name}.compile.tmp")
    try:
        temporary.write_text(
            json.dumps(config, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(config_path)
    except OSError as exc:
        raise SystemExit(
            f"Cannot compile model configuration {config_path}: {exc}"
        ) from exc
