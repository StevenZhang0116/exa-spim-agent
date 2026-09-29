"""Deterministically assemble a self-contained detector from feature code."""

from __future__ import annotations

import ast
import builtins
import symtable
import warnings
from pathlib import Path

from .candidate_policy import (
    load_runtime_candidate_policy,
    load_runtime_merge_candidate_policy,
)
from .contracts import DetectorTarget
from .cost_validation import analyze_row_cost
from .feature_scope import feature_scope_runtime_source
from .split_feature_scope import split_scope_runtime_source
from .target_runtime import TARGET_ADAPTER_MARKER, target_adapter_source


FEATURE_MARKER = "# __DETECTOR_FEATURE_IMPLEMENTATION__"
REQUIRED_SYMBOLS = frozenset({
    "FEATURE_REGISTRY", "ANALYSIS_TIMING_GROUPS", "COMPUTATION_PLAN",
    "SegmentAccumulator", "extract_features",
})
COMPUTATION_COST_CLASSES = frozenset({
    "constant", "bounded_local", "density_scaled", "component_scaled",
    "global_scan",
})
COST_CLASSES_REQUIRING_BOUND = frozenset({"density_scaled", "component_scaled"})
COST_CLASSES_REQUIRING_AMORTIZATION = frozenset({"density_scaled", "global_scan"})
_PLAN_PLACEHOLDER_DECLARATIONS = frozenset({
    "", "none", "none required", "n/a", "na", "unbounded", "not required",
})
RUNTIME_OWNED_SYMBOLS = frozenset({
    "EMBEDDED_MODEL_POLICY", "EMBEDDED_MODEL_POLICY_SHA256", "RANDOM_SEED",
    "ALLOWED_FAMILIES", "NATIVE_NAN_FAMILIES", "MODEL_FIT_JOBS",
    "build_estimator", "run_nested_selection", "fit_final_winner",
    "run_smoke_test", "build_arg_parser", "run_detector", "main",
    "validate_model_config", "validate_analysis_timing_groups",
    "load_hypothesis_selection", "AnalysisTimingRecorder",
    "write_hypothesis_cost_artifacts", "_collect_measuretime_versions",
    "run_measuretime", "_available_cpu_count", "_resolve_image_worker_count",
    "_bounded_thread_map", "_memoized", "_extract_features_runtime",
    "DETECTOR_TARGET", "DETECTOR_FILENAME", "OUTPUT_PREFIX", "ROW_UNIT",
    "ROW_UNIT_PLURAL", "LABEL_NAME", "POSITIVE_NAME", "NEGATIVE_NAME",
    "SCORE_PREFIX", "CANDIDATE_POLICY_CONFIG_ID", "CANDIDATE_PAIRING_RULE",
    "CANDIDATE_MAX_DISTANCE_UM", "CANDIDATE_PER_ANCHOR_K",
    "CANDIDATE_GLOBAL_CAP", "CANDIDATE_POLICY_SHA256",
    "CANDIDATE_MODE", "CANDIDATE_NMS_UM", "CANDIDATE_CLAIM_RADIUS_UM",
    "CANDIDATE_POSITIVE_LABEL_RADIUS_UM", "CANDIDATE_SITE_SNAP_MAX_UM",
    "build_sample_universe",
    "compatible_occurrences",
    "sample_output_frame", "sample_display", "sample_universe_audit",
    "ascertainment_covariates",
    "_derive_split_truth", "_gt_neuron_membership",
    "_merge_site_node_arrays", "_nms_kept_junctions",
    "_site_distances_to_kept", "_merge_site_provenance",
})

# GT payload keys no feature fragment may read, for EITHER target. These are the
# answer key: a feature computed from them cannot run blind at inference. Labels /
# universe derivation read them only inside the runtime-owned target adapter
# (build_sample_universe / _derive_split_truth), never in feature math. This is
# the static half of the blindness gate; the runtime half is the
# _BlindPayloadView guard in the detector runtime template.
FEATURE_FORBIDDEN_PAYLOAD_KEYS = frozenset({
    "gt_edge_error", "gt_graph", "gt_node_canonical_label", "gt_merge_labels",
    "gt_merge_sites", "gt_junction_audit", "gt_merge_site_metadata",
    "__detector_universe_site_audit__",
})
SPLIT_FEATURE_FORBIDDEN_SAMPLE_KEYS = frozenset({
    "is_split", "split_kind", "is_merge_creating",
    "contains_known_merge_segment",
})
MERGE_FEATURE_FORBIDDEN_SAMPLE_KEYS = frozenset({
    "is_merge",
})
MERGE_SITE_FEATURE_FORBIDDEN_SAMPLE_KEYS = frozenset({
    "is_merge_site", "distance_to_nearest_gt_site_um", "in_ambiguous_ring",
})
FEATURE_FORBIDDEN_RUNTIME_NAMES = frozenset({
    "_derive_split_truth", "_gt_neuron_membership", "ascertainment_covariates",
    "_site_distances_to_kept", "_merge_site_provenance",
})


def _top_level_defined_symbols(tree: ast.Module) -> set[str]:
    """Return names whose module-level definitions can shadow another fragment."""
    defined: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defined.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    defined.add(target.id)
    return defined


def _assigned_value(tree: ast.Module, name: str) -> ast.expr | None:
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name
            for target in node.targets
        ):
            return node.value
        if (isinstance(node, ast.AnnAssign)
                and isinstance(node.target, ast.Name)
                and node.target.id == name):
            return node.value
    return None


def _feature_registry_names(tree: ast.Module, path: Path) -> list[str]:
    value = _assigned_value(tree, "FEATURE_REGISTRY")
    if not isinstance(value, (ast.List, ast.Tuple)):
        raise SystemExit(f"Feature implementation {path} has a non-literal FEATURE_REGISTRY.")
    names: list[str] = []
    for item in value.elts:
        first = None
        if isinstance(item, ast.Call) and item.args:
            first = item.args[0]
        elif isinstance(item, (ast.List, ast.Tuple)) and item.elts:
            first = item.elts[0]
        if not isinstance(first, ast.Constant) or not isinstance(first.value, str):
            raise SystemExit(
                f"Feature implementation {path} has a FEATURE_REGISTRY entry "
                "without a literal feature name."
            )
        names.append(first.value)
    if not names or len(names) != len(set(names)):
        raise SystemExit(f"Feature implementation {path} has empty/duplicate feature names.")
    return names


def _validate_accumulator_registry_usage(tree: ast.Module, path: Path) -> None:
    """Keep registry metadata tuples out of accumulator column keys."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            constructor = node.func.id
        elif isinstance(node.func, ast.Attribute):
            constructor = node.func.attr
        else:
            continue
        if not constructor.endswith("Accumulator"):
            continue
        values = [*node.args, *(keyword.value for keyword in node.keywords)]
        if any(
            isinstance(value, ast.Name) and value.id == "FEATURE_REGISTRY"
            for value in values
        ):
            raise SystemExit(
                f"Feature implementation {path} passes FEATURE_REGISTRY to "
                f"{constructor} on line {node.lineno}. Accumulators require "
                "flat string FEATURE_NAMES; FEATURE_REGISTRY contains metadata "
                "entries."
            )


def _validate_accumulator_set_calls(tree: ast.Module, path: Path) -> None:
    """Bind statically resolvable accumulator writes to their declared methods.

    Constructor-bound locals and their aliases are tracked within lexical
    scopes, including closures. Dynamic dispatch and expanded arguments are
    left to extraction tests, not claimed as statically verified.
    """
    import inspect

    feature_names = set(_feature_registry_names(tree, path))
    signatures = {}
    for definition in tree.body:
        if not isinstance(definition, ast.ClassDef) or not definition.name.endswith("Accumulator"):
            continue
        method = next((node for node in definition.body
                       if isinstance(node, ast.FunctionDef) and node.name == "set"), None)
        if method is None or method.decorator_list:
            continue
        positional = [*method.args.posonlyargs, *method.args.args]
        if not positional:
            continue
        defaults = len(positional) - len(method.args.defaults)
        parameters = []
        for index, parameter in enumerate(positional[1:], 1):
            kind = (inspect.Parameter.POSITIONAL_ONLY if index < len(method.args.posonlyargs)
                    else inspect.Parameter.POSITIONAL_OR_KEYWORD)
            default = None if index >= defaults else inspect.Parameter.empty
            parameters.append(inspect.Parameter(parameter.arg, kind, default=default))
        if method.args.vararg:
            parameters.append(inspect.Parameter(method.args.vararg.arg, inspect.Parameter.VAR_POSITIONAL))
        for parameter, default in zip(method.args.kwonlyargs, method.args.kw_defaults):
            parameters.append(inspect.Parameter(parameter.arg, inspect.Parameter.KEYWORD_ONLY,
                default=None if default is not None else inspect.Parameter.empty))
        if method.args.kwarg:
            parameters.append(inspect.Parameter(method.args.kwarg.arg, inspect.Parameter.VAR_KEYWORD))
        signatures[definition.name] = inspect.Signature(parameters)

    class WriteVisitor(ast.NodeVisitor):
        def __init__(self):
            self.bindings = {}

        def visit_FunctionDef(self, node):
            outer = self.bindings
            self.bindings = dict(outer)
            arguments = [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
            arguments += [argument for argument in (node.args.vararg, node.args.kwarg) if argument]
            for argument in arguments:
                self.bindings.pop(argument.arg, None)
            for statement in node.body:
                self.visit(statement)
            self.bindings = outer

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_ClassDef(self, node):
            return

        def assign(self, target, value):
            if not isinstance(target, ast.Name):
                return
            kind = None
            if isinstance(value, ast.Call) and isinstance(value.func, ast.Name):
                kind = value.func.id if value.func.id in signatures else None
            elif isinstance(value, ast.Name):
                kind = self.bindings.get(value.id)
            if kind is None:
                self.bindings.pop(target.id, None)
            else:
                self.bindings[target.id] = kind

        def visit_Assign(self, node):
            self.visit(node.value)
            for target in node.targets:
                self.assign(target, node.value)

        def visit_AnnAssign(self, node):
            if node.value is not None:
                self.visit(node.value)
            self.assign(node.target, node.value)

        def visit_Call(self, node):
            self.generic_visit(node)
            if not isinstance(node.func, ast.Attribute) or node.func.attr != "set":
                return
            owner = node.func.value
            kind = self.bindings.get(owner.id) if isinstance(owner, ast.Name) else None
            if kind is None:
                return
            if any(isinstance(argument, ast.Starred) for argument in node.args) or any(
                    keyword.arg is None for keyword in node.keywords):
                return
            signature = signatures[kind]
            try:
                bound = signature.bind(*node.args, **{keyword.arg: keyword.value for keyword in node.keywords})
            except TypeError as exc:
                raise SystemExit(f"Feature implementation {path}: accumulator set() call on line "
                                 f"{node.lineno} does not match {kind}.set{signature}: {exc}") from exc
            feature_parameter = next((name for name in ("feature_name", "feature", "name")
                                      if name in signature.parameters), None)
            if feature_parameter is None:
                return
            feature = bound.arguments.get(feature_parameter)
            if isinstance(feature, ast.Constant):
                if not isinstance(feature.value, str) or feature.value not in feature_names:
                    raise SystemExit(f"Feature implementation {path}: accumulator set() on line "
                                     f"{node.lineno} binds unknown feature {feature.value!r} to "
                                     f"{feature_parameter}; expected a FEATURE_NAMES entry.")
            for parameter, value in bound.arguments.items():
                if (parameter != feature_parameter and isinstance(value, ast.Constant)
                        and isinstance(value.value, str) and value.value in feature_names):
                    raise SystemExit(f"Feature implementation {path}: accumulator set() argument order "
                                     f"mismatch on line {node.lineno}; feature {value.value!r} binds to "
                                     f"{parameter}, not {feature_parameter}, in {kind}.set{signature}.")

    WriteVisitor().visit(tree)


def _scope_assignment_lines(
    statements: list[ast.stmt], variable_name: str
) -> list[int]:
    """Find assignments in one function scope, including its control flow."""

    class AssignmentVisitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.lines: list[int] = []

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            return

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            return

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            return

        def visit_Assign(self, node: ast.Assign) -> None:
            if any(
                isinstance(target, ast.Name) and target.id == variable_name
                for target in node.targets
            ):
                self.lines.append(node.lineno)
            self.generic_visit(node.value)

        def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
            if isinstance(node.target, ast.Name) and node.target.id == variable_name:
                self.lines.append(node.lineno)
            if node.value is not None:
                self.generic_visit(node.value)

    visitor = AssignmentVisitor()
    for statement in statements:
        visitor.visit(statement)
    return visitor.lines


def _validate_image_patch_cache_scope(tree: ast.Module, path: Path) -> None:
    """Reject image arrays retained by an extraction-wide patch cache."""
    extract = next(
        (
            node for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "extract_features"
        ),
        None,
    )
    if extract is None:
        return
    outer_lines = _scope_assignment_lines(extract.body, "patch_cache")
    if outer_lines:
        raise SystemExit(
            f"Feature implementation {path} creates extraction-wide patch_cache "
            f"on line(s) {', '.join(str(line) for line in outer_lines)}. Image "
            "patch arrays must be cached inside the per-row worker so they are "
            "released after each row."
        )


def _validate_timing_contract(tree: ast.Module, path: Path) -> set[str]:
    feature_names = _feature_registry_names(tree, path)
    value = _assigned_value(tree, "ANALYSIS_TIMING_GROUPS")
    try:
        groups = ast.literal_eval(value) if value is not None else None
    except (ValueError, TypeError, SyntaxError) as exc:
        raise SystemExit(
            f"Feature implementation {path} has non-literal ANALYSIS_TIMING_GROUPS."
        ) from exc
    if not isinstance(groups, list) or not groups:
        raise SystemExit("ANALYSIS_TIMING_GROUPS must be a non-empty literal list.")
    keys: set[str] = set()
    hypothesis_owner: dict[int, str] = {}
    covered: list[str] = []
    for group in groups:
        if not isinstance(group, dict):
            raise SystemExit("Each ANALYSIS_TIMING_GROUPS entry must be an object.")
        key = group.get("key")
        ids = group.get("hypothesis_ids")
        names = group.get("feature_names")
        phase = group.get("phase")
        if not isinstance(key, str) or not key or key in keys:
            raise SystemExit("Analysis timing group keys must be unique non-empty strings.")
        keys.add(key)
        if (not isinstance(ids, list) or not ids or
                any(not isinstance(x, int) or isinstance(x, bool) or x <= 0 for x in ids)):
            raise SystemExit(f"Analysis timing group {key!r} has invalid hypothesis_ids.")
        if len(ids) != len(set(ids)):
            raise SystemExit(f"Analysis timing group {key!r} repeats hypothesis_ids.")
        for hypothesis_id in ids:
            if hypothesis_id in hypothesis_owner:
                raise SystemExit(
                    f"Hypothesis {hypothesis_id} belongs to multiple analysis "
                    f"groups: {hypothesis_owner[hypothesis_id]!r} and {key!r}."
                )
            hypothesis_owner[hypothesis_id] = key
        if (not isinstance(names, list) or not names or
                any(not isinstance(x, str) or not x for x in names)):
            raise SystemExit(f"Analysis timing group {key!r} has invalid feature_names.")
        if not isinstance(phase, str) or not phase:
            raise SystemExit(f"Analysis timing group {key!r} has invalid phase.")
        covered.extend(names)
    duplicates = sorted({name for name in covered if covered.count(name) > 1})
    missing = sorted(set(feature_names) - set(covered))
    unknown = sorted(set(covered) - set(feature_names))
    if duplicates or missing or unknown or len(covered) != len(feature_names):
        raise SystemExit(
            "Analysis timing coverage mismatch: "
            f"duplicate={duplicates}, missing={missing}, unknown={unknown}."
        )

    extract = next(
        (node for node in tree.body
         if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
         and node.name == "extract_features"),
        None,
    )
    if extract is None:
        return keys
    positional = extract.args.args
    if len(positional) < 5 or [arg.arg for arg in positional[:5]] != [
        "payload", "verbose", "timing", "enabled_analysis_keys",
        "profile_segment_limit"
    ]:
        raise SystemExit(
            "extract_features must begin with (payload, verbose=True, timing=None, "
            "enabled_analysis_keys=None, profile_segment_limit=None)."
        )
    default_start = len(positional) - len(extract.args.defaults)
    timing_default_index = 2 - default_start
    if (timing_default_index < 0
            or timing_default_index >= len(extract.args.defaults)
            or not isinstance(extract.args.defaults[timing_default_index], ast.Constant)
            or extract.args.defaults[timing_default_index].value is not None):
        raise SystemExit("extract_features timing must default to None.")
    selection_default_index = 3 - default_start
    if (selection_default_index < 0
            or selection_default_index >= len(extract.args.defaults)
            or not isinstance(extract.args.defaults[selection_default_index], ast.Constant)
            or extract.args.defaults[selection_default_index].value is not None):
        raise SystemExit("extract_features enabled_analysis_keys must default to None.")
    profile_default_index = 4 - default_start
    if (profile_default_index < 0
            or profile_default_index >= len(extract.args.defaults)
            or not isinstance(extract.args.defaults[profile_default_index], ast.Constant)
            or extract.args.defaults[profile_default_index].value is not None):
        raise SystemExit("extract_features profile_segment_limit must default to None.")

    image_hypothesis_ids = {
        hypothesis_id
        for group in groups
        if group["phase"] == "image_patch_pass"
        for hypothesis_id in group["hypothesis_ids"]
    }
    image_heavy = len(image_hypothesis_ids) * 2 > len(hypothesis_owner)
    if image_heavy:
        if len(positional) < 6 or positional[5].arg != "image_workers":
            raise SystemExit(
                "Image-heavy extract_features must accept image_workers after "
                "profile_segment_limit."
            )
        worker_default_index = 5 - default_start
        if (worker_default_index < 0
                or worker_default_index >= len(extract.args.defaults)
                or not isinstance(
                    extract.args.defaults[worker_default_index], ast.Constant)
                or extract.args.defaults[worker_default_index].value != 1):
            raise SystemExit(
                "Image-heavy extract_features image_workers must default to 1."
            )
        called_names = {
            node.func.id for node in ast.walk(extract)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        required_helpers = {
            "_bounded_thread_map", "_resolve_image_worker_count"
        }
        if not required_helpers.issubset(called_names):
            raise SystemExit(
                "Image-heavy extract_features must use the runtime bounded "
                "thread-map and automatic worker-count helpers."
            )
        accumulator_classes = [
            node for node in tree.body
            if isinstance(node, ast.ClassDef)
            and node.name in {"FeatureAccumulator", "SegmentAccumulator"}
        ]
        for accumulator in accumulator_classes:
            set_method = next(
                (node for node in accumulator.body
                 if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                 and node.name == "set"),
                None,
            )
            lock_constructors = {
                node.func.id
                for node in ast.walk(accumulator)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in {"Lock", "RLock"}
            } | {
                node.func.attr
                for node in ast.walk(accumulator)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in {"Lock", "RLock"}
            }
            set_uses_context_manager = set_method is not None and any(
                isinstance(node, (ast.With, ast.AsyncWith))
                for node in ast.walk(set_method)
            )
            if lock_constructors or set_uses_context_manager:
                raise SystemExit(
                    "Image-heavy accumulator set() must not acquire a shared "
                    "per-feature write lock; give each row one worker owner or "
                    "commit row-local results in the parent thread."
                )
    if not any(
        isinstance(node, ast.Name)
        and node.id == "profile_segment_limit"
        and isinstance(node.ctx, ast.Load)
        for node in ast.walk(extract)
    ):
        raise SystemExit(
            "extract_features must use profile_segment_limit to restrict "
            "profiling work."
        )
    called_attrs = {
        node.func.attr for node in ast.walk(extract)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    if not {"start_phase", "start_analysis"}.issubset(called_attrs):
        raise SystemExit(
            "extract_features must call timing start_phase and start_analysis "
            "through observational helpers."
        )
    if not any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "_analysis_enabled"
        for node in ast.walk(extract)
    ):
        raise SystemExit(
            "extract_features must gate selectable computation through an "
            "_analysis_enabled helper."
        )
    gated_keys = {
        node.args[0].value
        for node in ast.walk(extract)
        if (isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_analysis_enabled"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str))
    }
    missing_gates = sorted(keys - gated_keys)
    if missing_gates:
        raise SystemExit(
            "extract_features has timing groups without explicit selection "
            f"gates: {missing_gates}."
        )

    class _DirectReturnVisitor(ast.NodeVisitor):
        """Collect returns owned by extract_features, excluding nested scopes."""

        def __init__(self) -> None:
            self.returns: list[ast.Return] = []

        def visit_Return(self, node: ast.Return) -> None:  # noqa: N802
            self.returns.append(node)

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
            return

        def visit_AsyncFunctionDef(  # noqa: N802
            self, node: ast.AsyncFunctionDef
        ) -> None:
            return

        def visit_Lambda(self, node: ast.Lambda) -> None:  # noqa: N802
            return

        def visit_ClassDef(self, node: ast.ClassDef) -> None:  # noqa: N802
            return

    return_visitor = _DirectReturnVisitor()
    for statement in extract.body:
        return_visitor.visit(statement)
    if not return_visitor.returns:
        raise SystemExit(
            "extract_features must explicitly return exactly "
            "(row_records, labels, accumulator)."
        )
    if not isinstance(extract.body[-1], ast.Return):
        raise SystemExit(
            "extract_features must end with an explicit return of exactly "
            "(row_records, labels, accumulator), so no control-flow path can "
            "fall through to None."
        )
    invalid_return_lines = [
        node.lineno
        for node in return_visitor.returns
        if not isinstance(node.value, ast.Tuple) or len(node.value.elts) != 3
    ]
    if invalid_return_lines:
        raise SystemExit(
            "extract_features must return exactly the three-item tuple "
            "(row_records, labels, accumulator) on every path; never return a "
            "DataFrame or accumulator.to_frame(). Invalid return line(s): "
            + ", ".join(str(line) for line in invalid_return_lines)
            + "."
        )
    return keys


def _validate_computation_plan(
    tree: ast.Module, path: Path, timing_group_keys: set[str]
) -> None:
    """Enforce the declared cost-discipline contract for feature computation."""
    value = _assigned_value(tree, "COMPUTATION_PLAN")
    try:
        plan = ast.literal_eval(value) if value is not None else None
    except (ValueError, TypeError, SyntaxError) as exc:
        raise SystemExit(
            f"Feature implementation {path} has non-literal COMPUTATION_PLAN."
        ) from exc
    if not isinstance(plan, list) or not plan:
        raise SystemExit("COMPUTATION_PLAN must be a non-empty literal list.")
    expected_keys = {"primitive", "cost_class", "bound", "amortization", "consumers"}
    primitives: set[str] = set()
    consumed: set[str] = set()
    needs_memoized = False
    for entry in plan:
        if not isinstance(entry, dict) or set(entry) != expected_keys:
            raise SystemExit(
                "Each COMPUTATION_PLAN entry must be an object with exactly "
                "the keys primitive, cost_class, bound, amortization, "
                "consumers."
            )
        primitive = entry["primitive"]
        if not isinstance(primitive, str) or not primitive or primitive in primitives:
            raise SystemExit(
                "COMPUTATION_PLAN primitive names must be unique non-empty "
                "strings."
            )
        primitives.add(primitive)
        cost_class = entry["cost_class"]
        if cost_class not in COMPUTATION_COST_CLASSES:
            raise SystemExit(
                f"COMPUTATION_PLAN primitive {primitive!r} has unknown "
                f"cost_class {cost_class!r}; allowed: "
                + ", ".join(sorted(COMPUTATION_COST_CLASSES)) + "."
            )
        bound = entry["bound"]
        amortization = entry["amortization"]
        if not isinstance(bound, str) or not isinstance(amortization, str):
            raise SystemExit(
                f"COMPUTATION_PLAN primitive {primitive!r} bound and "
                "amortization must be strings."
            )
        if (cost_class in COST_CLASSES_REQUIRING_BOUND
                and bound.strip().lower() in _PLAN_PLACEHOLDER_DECLARATIONS):
            raise SystemExit(
                f"COMPUTATION_PLAN primitive {primitive!r} is {cost_class} and "
                "must declare a concrete bound (cutoff, node cap, or per-"
                "component memoization)."
            )
        if (cost_class in COST_CLASSES_REQUIRING_AMORTIZATION
                and amortization.strip().lower() in _PLAN_PLACEHOLDER_DECLARATIONS):
            raise SystemExit(
                f"COMPUTATION_PLAN primitive {primitive!r} is {cost_class} and "
                "must declare how its cost is amortized (memoization key or "
                "one-time pre-pass)."
            )
        if (cost_class == "global_scan"
                and "pre-pass" not in amortization.lower()
                and "prepass" not in amortization.lower()):
            raise SystemExit(
                f"COMPUTATION_PLAN primitive {primitive!r} is global_scan and "
                "its amortization must declare a one-time pre-pass; full-array "
                "scans are forbidden on the per-row path."
            )
        if (cost_class == "density_scaled"
                or "memo" in amortization.lower()
                or "memo" in bound.lower()):
            needs_memoized = True
        consumers = entry["consumers"]
        if (not isinstance(consumers, list) or not consumers
                or any(not isinstance(x, str) or not x for x in consumers)):
            raise SystemExit(
                f"COMPUTATION_PLAN primitive {primitive!r} has invalid "
                "consumers; expected a non-empty list of analysis timing "
                "group keys."
            )
        if len(consumers) != len(set(consumers)):
            raise SystemExit(
                f"COMPUTATION_PLAN primitive {primitive!r} repeats consumers."
            )
        unknown = sorted(set(consumers) - timing_group_keys)
        if unknown:
            raise SystemExit(
                f"COMPUTATION_PLAN primitive {primitive!r} consumers reference "
                f"unknown analysis timing group keys: {unknown}."
            )
        consumed.update(consumers)
    uncovered = sorted(timing_group_keys - consumed)
    if uncovered:
        raise SystemExit(
            "COMPUTATION_PLAN coverage mismatch: analysis timing groups with "
            f"no declared computation primitive: {uncovered}."
        )
    if needs_memoized and not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_memoized"
        for node in ast.walk(tree)
    ):
        raise SystemExit(
            "COMPUTATION_PLAN declares memoized amortization but the "
            "implementation never calls the runtime _memoized helper; route "
            "each declared memoization through _memoized(cache, key, compute)."
        )


def _direct_function_definitions(owner: ast.AST) -> dict[str, ast.AST]:
    """Return function definitions belonging to one lexical scope."""
    definitions: dict[str, ast.AST] = {}

    class Visitor(ast.NodeVisitor):
        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
            definitions[node.name] = node

        def visit_AsyncFunctionDef(  # noqa: N802
            self, node: ast.AsyncFunctionDef
        ) -> None:
            definitions[node.name] = node

        def visit_Lambda(self, node: ast.Lambda) -> None:  # noqa: N802
            return

        def visit_ClassDef(self, node: ast.ClassDef) -> None:  # noqa: N802
            return

    for statement in getattr(owner, "body", ()):
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            definitions[statement.name] = statement
        else:
            Visitor().visit(statement)
    return definitions


def _executed_nodes(statements: list[ast.stmt] | tuple[ast.stmt, ...]):
    """Yield nodes executed by statements without entering nested callables."""

    class Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.nodes: list[ast.AST] = []

        def generic_visit(self, node: ast.AST) -> None:
            self.nodes.append(node)
            super().generic_visit(node)

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
            self.nodes.append(node)

        def visit_AsyncFunctionDef(  # noqa: N802
            self, node: ast.AsyncFunctionDef
        ) -> None:
            self.nodes.append(node)

        def visit_Lambda(self, node: ast.Lambda) -> None:  # noqa: N802
            self.nodes.append(node)

        def visit_ClassDef(self, node: ast.ClassDef) -> None:  # noqa: N802
            self.nodes.append(node)

    visitor = Visitor()
    for statement in statements:
        visitor.visit(statement)
    return visitor.nodes


def _call_name(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


def _literal_expression(node: ast.AST) -> bool:
    try:
        ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError):
        return False
    return True


def _contains_full_universe_reference(node: ast.AST) -> bool:
    full_names = {
        "node_component", "node_components", "node_component_id",
        "node_xyz", "node_radius", "all_nodes", "all_edges",
    }
    full_attributes = {"node_component_id", "node_xyz", "node_radius"}
    found = False

    class Visitor(ast.NodeVisitor):
        def visit_Name(self, item: ast.Name) -> None:  # noqa: N802
            nonlocal found
            if item.id in full_names:
                found = True

        def visit_Attribute(self, item: ast.Attribute) -> None:  # noqa: N802
            nonlocal found
            if item.attr in full_attributes:
                found = True
            else:
                self.generic_visit(item)

        def visit_Call(self, item: ast.Call) -> None:  # noqa: N802
            nonlocal found
            if (isinstance(item.func, ast.Attribute)
                    and item.func.attr in {"number_of_nodes", "number_of_edges"}
                    and not item.args and not item.keywords):
                found = True
                return
            self.generic_visit(item)

        def visit_Subscript(self, item: ast.Subscript) -> None:  # noqa: N802
            # A scalar/indexed row is local even when its backing array is
            # universe-sized. A full slice still represents the whole array.
            full_slice = (
                isinstance(item.slice, ast.Slice)
                and item.slice.lower is None
                and item.slice.upper is None
                and item.slice.step is None
            ) or (
                isinstance(item.slice, ast.Constant)
                and item.slice.value is Ellipsis
            )
            if full_slice:
                self.visit(item.value)
            else:
                self.visit(item.slice)

    Visitor().visit(node)
    return found


def _row_path_violation(nodes: list[ast.AST]) -> tuple[int, str] | None:
    """Return the first high-confidence setup/global operation in row work."""
    index_builders = {
        "KDTree", "cKDTree", "BallTree", "NearestNeighbors",
        "KMeans", "MiniBatchKMeans", "GaussianMixture",
    }
    global_reductions = {
        "amin", "amax", "argmin", "argmax", "min", "max", "sum",
        "mean", "median", "percentile", "quantile", "sort", "argsort",
        "where", "flatnonzero", "unique",
    }
    unbounded_graph_calls = {
        "all_pairs_shortest_path", "all_pairs_dijkstra",
        "floyd_warshall", "shortest_path", "shortest_path_length",
        "single_source_shortest_path", "single_source_shortest_path_length",
        "single_source_dijkstra", "single_source_dijkstra_path_length",
    }

    for node in nodes:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            return node.lineno, "import/module initialization"
        if isinstance(node, (ast.For, ast.AsyncFor)):
            if _contains_full_universe_reference(node.iter):
                return node.lineno, "full-universe iteration"
        if isinstance(node, ast.comprehension):
            if _contains_full_universe_reference(node.iter):
                return getattr(node, "lineno", 0), "full-universe comprehension"
        if not isinstance(node, ast.Call):
            continue
        name = _call_name(node)
        if (name in index_builders
                and any(_contains_full_universe_reference(arg) for arg in node.args)):
            return node.lineno, f"full-universe {name} index/model construction"
        if (name in global_reductions
                and any(_contains_full_universe_reference(arg) for arg in node.args)):
            return node.lineno, f"full-universe {name} operation"
        if (name in {"array", "asarray", "fromiter"}
                and any(_contains_full_universe_reference(arg) for arg in node.args)):
            return node.lineno, f"full-universe {name} materialization"
        if name in unbounded_graph_calls:
            keywords = {keyword.arg for keyword in node.keywords}
            if "cutoff" not in keywords and "node_cap" not in keywords:
                return node.lineno, f"unbounded graph call {name}"
    return None


_GRAPH_CONSTRUCTOR_NAMES = frozenset({
    "Graph", "DiGraph", "MultiGraph", "MultiDiGraph",
})
_PER_ELEMENT_FORBIDDEN_ATTRS = frozenset({
    "subgraph", "connected_components", "edge_betweenness_centrality",
    "betweenness_centrality",
}) | _GRAPH_CONSTRUCTOR_NAMES


def _graph_element_iterator(iter_node: ast.AST) -> str | None:
    """Return 'edges'/'nodes' when a for-loop iterates graph elements."""
    for sub in ast.walk(iter_node):
        if (isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute)
                and sub.func.attr in ("edges", "nodes")):
            return sub.func.attr
    return None


def _validate_no_per_element_graph_rebuilds(tree: ast.Module, path: Path) -> None:
    """Reject graph reconstruction / whole-graph algorithms per edge or node.

    The cautionary tale is a generated merge-site fragment whose "tree fast
    path" for edge betweenness copied the WHOLE component graph and re-ran
    connected_components for EVERY edge — an O(E x (V+E)) blow-up that turned
    a bounded per-component feature into hours per large component, even
    though the declared bound (k-sampled betweenness) was honored on the
    non-tree branch. An O(E)-iteration loop with an O(V+E) body defeats every
    declared bound, so it is rejected statically: per-element quantities must
    be derived in one pass (tree edge betweenness is subtree-size products
    from a single DFS, exactly as the bounded discovery sources implement it).
    """
    violations: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.For):
            continue
        element = _graph_element_iterator(node.iter)
        if element is None:
            continue
        for statement in node.body:
            for inner in ast.walk(statement):
                if not isinstance(inner, ast.Call):
                    continue
                func = inner.func
                name = None
                if (isinstance(func, ast.Attribute)
                        and func.attr in _PER_ELEMENT_FORBIDDEN_ATTRS):
                    name = func.attr
                elif (isinstance(func, ast.Name)
                        and func.id in _GRAPH_CONSTRUCTOR_NAMES):
                    name = func.id
                if name is not None:
                    violations.append(
                        f"{name} at line {inner.lineno} inside a "
                        f"per-{element[:-1]} loop (line {node.lineno})"
                    )
    if violations:
        raise SystemExit(
            f"Feature implementation {path} rebuilds a graph or runs a "
            "whole-graph algorithm inside a per-edge/per-node loop — a "
            "quadratic blow-up that defeats every declared cost bound. "
            "Derive per-element quantities in one pass instead: "
            + "; ".join(sorted(violations))
        )


def _validate_row_path_cost(tree: ast.Module, path: Path) -> None:
    """Combine existing structural guards with scope-aware data-flow checks."""
    _validate_legacy_row_path_cost(tree, path)
    report = analyze_row_cost(tree)
    if report.violations:
        raise SystemExit(
            f"Feature implementation {path} has setup/global work reachable "
            "from per-row analysis or invalid memoization:\n"
            + "\n".join(report.violations[:8])
        )
    if report.unverified:
        warnings.warn(
            f"Partial row-cost verification for {path}: "
            f"{len(report.unverified)} unverified path(s); this is not a full complexity guarantee.\n"
            + "\n".join(report.unverified[:5]),
            RuntimeWarning, stacklevel=2,
        )


def _validate_legacy_row_path_cost(tree: ast.Module, path: Path) -> None:
    """Reject setup/global work reachable from timing-instrumented row loops.

    COMPUTATION_PLAN is a useful declaration but cannot prove complexity. This
    check follows generated local helpers and callbacks from loops that perform
    timed analysis work. A whole-universe computation hidden behind memoization
    by row or component is still rejected. A constant-key memoized helper is
    permitted only when the same helper is explicitly warmed before row work.
    """
    extract = next(
        (node for node in tree.body
         if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
         and node.name == "extract_features"),
        None,
    )
    if extract is None:
        return

    module_defs = _direct_function_definitions(tree)
    extract_defs = _direct_function_definitions(extract)
    all_defs = list(module_defs.values()) + list(extract_defs.values())

    starter_names: set[str] = set()
    calls_by_name: dict[str, set[str]] = {}
    for function in all_defs:
        executed = _executed_nodes(tuple(function.body))
        calls = {
            name for node in executed
            if isinstance(node, ast.Call) and (name := _call_name(node))
        }
        calls_by_name[function.name] = calls
        if "start_analysis" in calls:
            starter_names.add(function.name)
    changed = True
    while changed:
        changed = False
        for name, calls in calls_by_name.items():
            if name not in starter_names and calls & starter_names:
                starter_names.add(name)
                changed = True

    class MainLoopVisitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.loops: list[ast.AST] = []

        def _visit_loop(self, node: ast.AST) -> None:
            executed = _executed_nodes(tuple(getattr(node, "body", ())))
            calls = {
                name for item in executed
                if isinstance(item, ast.Call) and (name := _call_name(item))
            }
            if "start_analysis" in calls or calls & starter_names:
                self.loops.append(node)
            self.generic_visit(node)

        def visit_For(self, node: ast.For) -> None:  # noqa: N802
            self._visit_loop(node)

        def visit_AsyncFor(self, node: ast.AsyncFor) -> None:  # noqa: N802
            self._visit_loop(node)

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
            return

        def visit_AsyncFunctionDef(  # noqa: N802
            self, node: ast.AsyncFunctionDef
        ) -> None:
            return

        def visit_Lambda(self, node: ast.Lambda) -> None:  # noqa: N802
            return

    loop_visitor = MainLoopVisitor()
    for statement in extract.body:
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        loop_visitor.visit(statement)
    if not loop_visitor.loops:
        return

    row_loop_ids = {id(loop) for loop in loop_visitor.loops}
    parent: dict[int, ast.AST] = {}
    for node in ast.walk(extract):
        for child in ast.iter_child_nodes(node):
            parent[id(child)] = node

    def inside_row_loop(node: ast.AST) -> bool:
        current: ast.AST | None = node
        while current is not None:
            if id(current) in row_loop_ids:
                return True
            current = parent.get(id(current))
        return False

    first_row_line = min(loop.lineno for loop in loop_visitor.loops)
    prewarmed_names: set[str] = set()
    for node in _executed_nodes(tuple(extract.body)):
        if (isinstance(node, ast.Call) and node.lineno < first_row_line
                and not inside_row_loop(node)
                and isinstance(node.func, ast.Name)
                and node.func.id in extract_defs):
            prewarmed_names.add(node.func.id)

    def resolve(owner: ast.AST, name: str) -> ast.AST | None:
        local = _direct_function_definitions(owner)
        return local.get(name) or extract_defs.get(name) or module_defs.get(name)

    def callback_nodes(owner: ast.AST, call: ast.Call) -> list[ast.AST]:
        name = _call_name(call)
        positions: tuple[int, ...] = ()
        if name == "_memoized" and len(call.args) >= 3:
            if (getattr(owner, "name", None) in prewarmed_names
                    and _literal_expression(call.args[1])):
                return []
            positions = (2,)
        elif name == "_bounded_thread_map" and call.args:
            positions = (0,)
        callbacks: list[ast.AST] = []
        for position in positions:
            callback = call.args[position]
            if isinstance(callback, ast.Name):
                definition = resolve(owner, callback.id)
                if definition is not None:
                    callbacks.append(definition)
            elif isinstance(callback, ast.Lambda):
                callbacks.append(callback)
        return callbacks

    queue: list[tuple[ast.AST, list[ast.stmt] | tuple[ast.stmt, ...], list[str]]] = []
    for loop in loop_visitor.loops:
        queue.append((extract, tuple(loop.body), [f"row loop line {loop.lineno}"]))
    for node in _executed_nodes(tuple(extract.body)):
        if isinstance(node, ast.Call) and _call_name(node) == "_bounded_thread_map":
            for callback in callback_nodes(extract, node):
                body = ([ast.Expr(value=callback.body)] if isinstance(callback, ast.Lambda)
                        else tuple(callback.body))
                queue.append((callback, body, [getattr(callback, "name", "lambda")]))

    seen: set[tuple[int, tuple[int, ...]]] = set()
    while queue:
        owner, statements, chain = queue.pop(0)
        marker = (id(owner), tuple(id(statement) for statement in statements))
        if marker in seen:
            continue
        seen.add(marker)
        executed = _executed_nodes(statements)
        violation = _row_path_violation(executed)
        if violation is not None:
            line, operation = violation
            raise SystemExit(
                f"Feature implementation {path} has setup/global work reachable "
                f"from per-row analysis: {operation} on line {line}; call path "
                f"{' -> '.join(chain)}. Move whole-universe work before the "
                "timed row loop and materialize the derived lookup. Memoizing a "
                "whole-universe scan by row/component key is not sufficient."
            )
        for node in executed:
            if not isinstance(node, ast.Call):
                continue
            name = _call_name(node)
            if isinstance(node.func, ast.Name) and name:
                definition = resolve(owner, name)
                if definition is not None:
                    queue.append((definition, tuple(definition.body), [*chain, name]))
            for callback in callback_nodes(owner, node):
                body = ([ast.Expr(value=callback.body)] if isinstance(callback, ast.Lambda)
                        else tuple(callback.body))
                queue.append((
                    callback, body,
                    [*chain, getattr(callback, "name", "lambda")],
                ))


def _literal_mapping_key(node: ast.AST) -> str | None:
    """Return a literal key used by ``obj[key]`` or ``obj.get(key)``."""
    if isinstance(node, ast.Subscript):
        key = node.slice
    elif (isinstance(node, ast.Call) and node.args
          and isinstance(node.func, ast.Attribute)
          and node.func.attr == "get"):
        key = node.args[0]
    else:
        return None
    if isinstance(key, ast.Constant) and isinstance(key.value, str):
        return key.value
    return None


def _mapping_owner_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Subscript):
        owner = node.value
    elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        owner = node.func.value
    else:
        return None
    return owner.id if isinstance(owner, ast.Name) else None


def _validate_feature_no_gt_access(tree: ast.Module, path: Path,
                                   target: DetectorTarget) -> None:
    """Reject direct GT/label/audit access from generated feature math.

    Applies to BOTH targets. The merge image-only-v2 build (2026-08-21) is the
    cautionary tale: this gate used to run for split only, so a merge fragment
    shipped ``fragment_to_gt_distance`` (payload["gt_graph"] in feature math) and
    that GT-ascertainment artifact became the model's dominant feature. Feature
    fragments read candidate identity from build_sample_universe's cache and the
    fragments graph/image only; every GT payload key access is a hard reject.
    """
    violations: set[str] = set()
    if target is DetectorTarget.SPLIT:
        forbidden_sample_keys = SPLIT_FEATURE_FORBIDDEN_SAMPLE_KEYS
    elif target is DetectorTarget.MERGE_SITE:
        forbidden_sample_keys = MERGE_SITE_FEATURE_FORBIDDEN_SAMPLE_KEYS
    else:
        forbidden_sample_keys = MERGE_FEATURE_FORBIDDEN_SAMPLE_KEYS
    for node in ast.walk(tree):
        if (isinstance(node, ast.Name)
                and isinstance(node.ctx, ast.Load)
                and node.id in FEATURE_FORBIDDEN_RUNTIME_NAMES):
            violations.add(node.id)
        key = _literal_mapping_key(node)
        if key is None:
            continue
        owner = _mapping_owner_name(node)
        if key in FEATURE_FORBIDDEN_PAYLOAD_KEYS:
            # Forbid GT payload keys on ANY owner name, not just the literal
            # ``payload`` variable — aliasing (p = payload; p["gt_graph"]) must
            # not slip past the static gate.
            violations.add(f"[{key!r}]" if owner is None else f"{owner}[{key!r}]")
        elif key in forbidden_sample_keys:
            violations.add(key)
    if violations:
        raise SystemExit(
            f"{target.value} feature implementation {path} reads GT-only "
            "label/audit state: " + ", ".join(sorted(violations))
        )


_IMPLICIT_SCOPE_NAMES = frozenset({
    "__name__", "__file__", "__doc__", "__package__", "__spec__",
    "__loader__", "__builtins__", "__debug__", "__annotations__",
    "__class__",
})


def _validate_assembled_name_resolution(source: str, path: Path) -> None:
    """Reject assembled detectors referencing names no scope defines.

    A leftover call to a renamed helper is a latent NameError that only
    surfaces at runtime on code paths the smoke test may never execute.
    """
    table = symtable.symtable(source, str(path), "exec")
    module_defined: set[str] = set()

    def collect(scope: symtable.SymbolTable) -> None:
        for symbol in scope.get_symbols():
            if scope.get_type() == "module":
                if symbol.is_assigned() or symbol.is_imported():
                    module_defined.add(symbol.get_name())
            elif symbol.is_declared_global() and symbol.is_assigned():
                module_defined.add(symbol.get_name())
        for child in scope.get_children():
            collect(child)

    collect(table)
    unresolved: set[tuple[str, str]] = set()

    def check(scope: symtable.SymbolTable) -> None:
        for symbol in scope.get_symbols():
            if not symbol.is_referenced():
                continue
            name = symbol.get_name()
            if scope.get_type() == "module":
                unbound = not (symbol.is_assigned() or symbol.is_imported())
            else:
                unbound = symbol.is_global()
            if (unbound and name not in module_defined
                    and not hasattr(builtins, name)
                    and name not in _IMPLICIT_SCOPE_NAMES):
                unresolved.add((scope.get_name(), name))
        for child in scope.get_children():
            check(child)

    check(table)
    if unresolved:
        details = ", ".join(
            f"{name} (in {scope})"
            for scope, name in sorted(unresolved, key=lambda i: (i[1], i[0])))
        raise SystemExit(
            f"Assembled detector {path} references undefined names — a latent "
            f"NameError on paths the smoke test may not reach: {details}."
        )


def validate_feature_implementation(
    path: Path,
    runtime_owned_symbols: set[str] | frozenset[str] = RUNTIME_OWNED_SYMBOLS,
    required_symbols: set[str] | frozenset[str] = REQUIRED_SYMBOLS,
    target: DetectorTarget = DetectorTarget.MERGE,
) -> str:
    """Return a parsed feature fragment after enforcing its ownership boundary."""
    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
    except (OSError, SyntaxError) as exc:
        raise SystemExit(f"Feature implementation {path} does not parse: {exc}") from exc

    defined = _top_level_defined_symbols(tree)
    for node in tree.body:
        if isinstance(node, ast.If) and any(
            isinstance(item, ast.Name) and item.id == "__name__"
            for item in ast.walk(node.test)
        ):
            raise SystemExit("Feature implementation must not own a __main__ entry point.")

    missing = sorted(set(required_symbols) - defined)
    if missing:
        raise SystemExit("Feature implementation is missing: " + ", ".join(missing))
    overlap = sorted(set(runtime_owned_symbols) & defined)
    if overlap:
        raise SystemExit("Feature implementation redefines runtime-owned symbols: " + ", ".join(overlap))
    _validate_accumulator_registry_usage(tree, path)
    _validate_accumulator_set_calls(tree, path)
    _validate_image_patch_cache_scope(tree, path)
    timing_group_keys = _validate_timing_contract(tree, path)
    _validate_computation_plan(tree, path, timing_group_keys)
    _validate_row_path_cost(tree, path)
    _validate_no_per_element_graph_rebuilds(tree, path)
    _validate_feature_no_gt_access(tree, path, target)
    return source.rstrip() + "\n"


def _validate_scoped_feature_writes(tree: ast.Module, scopes: dict[str, str]) -> None:
    """Catch literal scope mistakes at build time; dynamic names are checked at runtime."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        expected = {"set_candidate": "candidate", "set_segment": "segment"}.get(node.func.attr)
        if expected is None:
            continue
        feature = node.args[0] if node.args else next(
            (k.value for k in node.keywords if k.arg == "feature_name"), None)
        if isinstance(feature, ast.Constant):
            if not isinstance(feature.value, str) or feature.value not in scopes:
                raise SystemExit(f"Unknown feature in {node.func.attr} at line {node.lineno}.")
            if scopes[feature.value] != expected:
                raise SystemExit(
                    f"Feature {feature.value!r} has scope={scopes[feature.value]}, "
                    f"cannot use {node.func.attr} (line {node.lineno}).")


def _validate_split_feature_writes(tree: ast.Module, scopes: dict[str, str]) -> None:
    """Split writes must pass through runtime anchoring and occurrence reduction."""
    has_compute = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr in {"set", "set_candidate", "set_segment", "bind_rows"}:
            raise SystemExit("Split features must use acc.compute with runtime-owned anchoring/reduction.")
        if node.func.attr != "compute":
            continue
        has_compute = True
        feature = node.args[0] if node.args else next(
            (k.value for k in node.keywords if k.arg == "feature_name"), None)
        if isinstance(feature, ast.Constant) and feature.value not in scopes:
            raise SystemExit(f"Unknown split feature in compute at line {node.lineno}.")
    if not has_compute:
        raise SystemExit("Split extraction must use runtime FeatureAccumulator.compute.")


def assemble_detector(
    template_path: Path,
    feature_path: Path,
    output_path: Path,
    *,
    target: DetectorTarget = DetectorTarget.MERGE,
    candidate_policy_path: Path | None = None,
    inventory_path: Path | None = None,
) -> None:
    """Inject one validated feature fragment into the reviewed runtime template."""
    try:
        template = template_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SystemExit(f"Cannot read detector runtime template {template_path}: {exc}") from exc
    if template.count(FEATURE_MARKER) != 1:
        raise SystemExit("Detector runtime template must contain exactly one feature marker.")
    adapter_markers = template.count(TARGET_ADAPTER_MARKER)
    policy_targets = (DetectorTarget.SPLIT, DetectorTarget.MERGE_SITE)
    if adapter_markers > 1 or (
            target in policy_targets and adapter_markers != 1):
        raise SystemExit(
            "Split/merge-site detector runtime templates must contain exactly "
            "one target adapter marker (merge-only compatibility templates may "
            "omit it).")
    try:
        template_tree = ast.parse(template, filename=str(template_path))
    except SyntaxError as exc:
        raise SystemExit(f"Detector runtime template does not parse: {exc}") from exc
    runtime_owned = RUNTIME_OWNED_SYMBOLS | _top_level_defined_symbols(template_tree)
    required_symbols = set(REQUIRED_SYMBOLS)
    if target in policy_targets:
        required_symbols.remove("SegmentAccumulator")
    scope_source = ""
    feature_scopes = None
    if target is DetectorTarget.SPLIT:
        if candidate_policy_path is None:
            raise SystemExit("Split detector assembly requires split_candidate_policy.json.")
        if inventory_path is None:
            raise SystemExit("Split detector assembly requires feature_inventory.json (v5).")
        scope_source, feature_scopes = split_scope_runtime_source(inventory_path)
        runtime_owned = runtime_owned | _top_level_defined_symbols(ast.parse(scope_source))
    if target is DetectorTarget.MERGE_SITE:
        if candidate_policy_path is None:
            raise SystemExit("Merge-site detector assembly requires merge_candidate_policy.json.")
        if inventory_path is None:
            raise SystemExit("Merge-site detector assembly requires feature_inventory.json (v4).")
        scope_source, feature_scopes = feature_scope_runtime_source(inventory_path)
        runtime_owned = runtime_owned | _top_level_defined_symbols(ast.parse(scope_source))
    feature_source = validate_feature_implementation(
        feature_path, runtime_owned, required_symbols, target)
    if feature_scopes is not None:
        feature_tree = ast.parse(feature_source)
        names = _feature_registry_names(feature_tree, feature_path)
        if names != list(feature_scopes):
            raise SystemExit("FEATURE_REGISTRY must match inventory names and order exactly.")
        _validate_scoped_feature_writes(feature_tree, feature_scopes)
        if target is DetectorTarget.SPLIT:
            _validate_split_feature_writes(feature_tree, feature_scopes)
    candidate_policy = None
    candidate_policy_sha256 = None
    if target is DetectorTarget.SPLIT:
        if candidate_policy_path is None:
            raise SystemExit(
                "Split detector assembly requires split_candidate_policy.json."
            )
        candidate_policy, candidate_policy_sha256 = load_runtime_candidate_policy(
            candidate_policy_path
        )
    elif target is DetectorTarget.MERGE_SITE:
        if candidate_policy_path is None:
            raise SystemExit(
                "Merge-site detector assembly requires merge_candidate_policy.json."
            )
        candidate_policy, candidate_policy_sha256 = (
            load_runtime_merge_candidate_policy(candidate_policy_path)
        )
    elif candidate_policy_path is not None:
        raise SystemExit("Merge detector assembly does not accept a candidate policy.")
    if adapter_markers:
        template = template.replace(
            TARGET_ADAPTER_MARKER,
            target_adapter_source(
                target,
                candidate_policy=candidate_policy,
                candidate_policy_sha256=candidate_policy_sha256,
            ).rstrip() + "\n" + scope_source,
        )
    assembled = template.replace(FEATURE_MARKER, feature_source)
    try:
        ast.parse(assembled, filename=str(output_path))
    except SyntaxError as exc:
        raise SystemExit(f"Assembled detector does not parse: {exc}") from exc
    _validate_assembled_name_resolution(assembled, output_path)
    temporary = output_path.with_name(f".{output_path.name}.assemble.tmp")
    temporary.write_text(assembled, encoding="utf-8")
    temporary.replace(output_path)
