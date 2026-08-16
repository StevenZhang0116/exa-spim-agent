"""Deterministically assemble a self-contained detector from feature code."""

from __future__ import annotations

import ast
from pathlib import Path


FEATURE_MARKER = "# __DETECTOR_FEATURE_IMPLEMENTATION__"
REQUIRED_SYMBOLS = frozenset({
    "FEATURE_REGISTRY", "ANALYSIS_TIMING_GROUPS", "SegmentAccumulator",
    "extract_features",
})
RUNTIME_OWNED_SYMBOLS = frozenset({
    "EMBEDDED_MODEL_POLICY", "EMBEDDED_MODEL_POLICY_SHA256", "RANDOM_SEED",
    "ALLOWED_FAMILIES", "NATIVE_NAN_FAMILIES",
    "build_estimator", "run_nested_selection", "fit_final_winner",
    "run_smoke_test", "build_arg_parser", "run_detector", "main",
    "validate_model_config", "validate_analysis_timing_groups",
    "load_hypothesis_selection", "AnalysisTimingRecorder",
    "write_hypothesis_cost_artifacts", "_collect_measuretime_versions",
    "run_measuretime",
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


def _validate_timing_contract(tree: ast.Module, path: Path) -> None:
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
        return
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


def validate_feature_implementation(
    path: Path,
    runtime_owned_symbols: set[str] | frozenset[str] = RUNTIME_OWNED_SYMBOLS,
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

    missing = sorted(REQUIRED_SYMBOLS - defined)
    if missing:
        raise SystemExit("Feature implementation is missing: " + ", ".join(missing))
    overlap = sorted(set(runtime_owned_symbols) & defined)
    if overlap:
        raise SystemExit("Feature implementation redefines runtime-owned symbols: " + ", ".join(overlap))
    _validate_timing_contract(tree, path)
    return source.rstrip() + "\n"


def assemble_detector(template_path: Path, feature_path: Path, output_path: Path) -> None:
    """Inject one validated feature fragment into the reviewed runtime template."""
    try:
        template = template_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SystemExit(f"Cannot read detector runtime template {template_path}: {exc}") from exc
    if template.count(FEATURE_MARKER) != 1:
        raise SystemExit("Detector runtime template must contain exactly one feature marker.")
    try:
        template_tree = ast.parse(template, filename=str(template_path))
    except SyntaxError as exc:
        raise SystemExit(f"Detector runtime template does not parse: {exc}") from exc
    runtime_owned = RUNTIME_OWNED_SYMBOLS | _top_level_defined_symbols(template_tree)
    feature_source = validate_feature_implementation(feature_path, runtime_owned)
    assembled = template.replace(FEATURE_MARKER, feature_source)
    try:
        ast.parse(assembled, filename=str(output_path))
    except SyntaxError as exc:
        raise SystemExit(f"Assembled detector does not parse: {exc}") from exc
    temporary = output_path.with_name(f".{output_path.name}.assemble.tmp")
    temporary.write_text(assembled, encoding="utf-8")
    temporary.replace(output_path)
