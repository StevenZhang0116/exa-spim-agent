"""Deterministic feature-importance evidence for one completed detector run.

Reads the run's model-selection JSON, fitted joblib and CSV (GT-judgeable rows
only), the application's ``feature_inventory.json`` and, when present, the origin
AutoDiscovery export, and produces ``importance_evidence.json`` plus a Markdown
evidence block. Four views are computed, each named for what it is:

- ``coefficients``: the winner's standardized signed coefficients (linear winners
  only; the pipeline's scaler standardizes the imputed features, so a coefficient
  is the change in log-odds per standard deviation of that feature);
- ``native_importance``: the estimator's own unsigned global importance when it
  has one (xgboost ``feature_importances_``, EBM ``term_importances()``);
- ``permutation``: in-sample permutation importance of the fitted winner on the
  judgeable rows (AP drop when one column is shuffled). Optimistic, because the
  model saw these rows, but defined for every family and comparable across them;
- ``univariate``: each feature alone, best direction, AP on the judgeable rows,
  next to the raw baseline (node degree for merge sites, gap distance for splits),
  with its defined share and class medians.

Features are grouped by hypothesis (``h<id>_`` prefix); the group table sums the
unsigned views and keeps the best univariate AP. Nothing here reads the dataset
pkl or reruns extraction.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import warnings
from pathlib import Path

import numpy as np

try:
    from agentic.detector_build.contracts import DetectorTarget, target_spec
except ImportError:  # pragma: no cover - script-style import
    from detector_build.contracts import DetectorTarget, target_spec  # type: ignore

PROJECT_ROOT = Path(__file__).resolve().parent.parent
APPLICATION_ROOT = PROJECT_ROOT / "autodiscovery-application"
EVIDENCE_SCHEMA = 1
HYPOTHESIS_RE = re.compile(r"^h(\d+)_")
BASELINE_COLUMNS = {
    "merge_detection": ("degree", 1.0),
    "merge_site_detection": ("degree", 1.0),
    "legacy_unspecified": ("degree", 1.0),
    "split_detection": ("gap_um", -1.0),
}


# --------------------------------------------------------------------------- run files
def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rel(path: Path) -> str:
    try:
        return Path(path).resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def resolve_result_dir(path: Path) -> tuple[Path, Path]:
    """``(run directory, deliverable directory)`` for a path under autodiscovery-application/."""
    result_dir = (path if path.is_absolute() else PROJECT_ROOT / path).resolve()
    if not result_dir.is_dir():
        raise SystemExit(f"Detector result path is not a directory: {result_dir}")
    try:
        relative = result_dir.relative_to(APPLICATION_ROOT.resolve())
    except ValueError as exc:
        raise SystemExit(
            f"Detector result path must be under {APPLICATION_ROOT.resolve()}: {result_dir}"
        ) from exc
    if not relative.parts:
        raise SystemExit("Pass one application/result subdirectory, not autodiscovery-application itself.")
    return result_dir, APPLICATION_ROOT.resolve() / relative.parts[0]


def exactly_one(directory: Path, pattern: str, label: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if len(matches) != 1:
        raise SystemExit(f"Expected exactly one {label} matching {pattern!r} in {directory}; "
                         f"found {len(matches)}.")
    return matches[0]


def load_json_object(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SystemExit(f"Expected a JSON object in {path}.")
    return value


def write_json_atomic(path: Path, value: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def result_contract(model: dict) -> dict:
    """Target vocabulary of one result from ``detector_build.contracts.target_spec``."""
    target = model.get("detector_target") or "merge_detection"
    try:
        spec = target_spec(DetectorTarget(target))
    except ValueError as exc:
        raise SystemExit(f"Unsupported detector_target in model JSON: {target!r}") from exc
    return {
        "target": spec.target.value, "output_prefix": spec.output_prefix,
        "detector_name": spec.detector_name, "label": spec.label_name,
        "score": f"{spec.score_prefix}_oof", "identity": spec.identity_columns,
        "positive": spec.positive_plural, "negative": spec.negative_plural,
        "row_unit": spec.row_unit,
    }


class Tee:
    """Mirror a stream into a log file handle."""

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


# --------------------------------------------------------------------------- evidence
def feature_hypothesis_id(name: str) -> int | None:
    match = HYPOTHESIS_RE.match(name)
    return int(match.group(1)) if match else None


def _finite_or_none(value) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _average_precision(y: np.ndarray, scores: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score

    return float(average_precision_score(y, scores))


def best_direction_ap(y: np.ndarray, values: np.ndarray) -> tuple[float, str]:
    """AP of one column ranked high-first or low-first, undefined rows last."""
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    if not finite.any():
        return float("nan"), "undefined"
    low = float(values[finite].min()) - 1.0
    high = float(values[finite].max()) + 1.0
    up = np.where(finite, values, low)
    down = np.where(finite, -values, -high)
    ap_up, ap_down = _average_precision(y, up), _average_precision(y, down)
    return (ap_up, "higher") if ap_up >= ap_down else (ap_down, "lower")


def load_run(result_dir: Path, application_dir: Path) -> dict:
    """Paths, model JSON and target contract of one completed run."""
    model_json_path = exactly_one(result_dir, "model_selection_*.json", "model-selection JSON")
    model = load_json_object(model_json_path)
    contract = result_contract(model)
    brain = str(model.get("train_brain") or "")
    if not brain:
        raise SystemExit(f"{model_json_path} has no train_brain.")
    prefix = contract["output_prefix"]
    paths = {
        "model_json": model_json_path,
        "csv": result_dir / f"{prefix}_{brain}.csv",
        "joblib": result_dir / f"{prefix}_{brain}.joblib",
        "log": result_dir / f"{prefix}_{brain}.log.txt",
        "inventory": application_dir / "feature_inventory.json",
    }
    for label, path in paths.items():
        if not path.is_file():
            raise SystemExit(f"Missing {label}: {path}")
    log_lines = paths["log"].read_text(encoding="utf-8", errors="replace").splitlines()
    final_line = next((line for line in reversed(log_lines) if line.strip()), "")
    if not final_line.startswith("# OK after "):
        raise SystemExit(
            f"Detector log is not complete (last line is {final_line!r}); summarize only "
            "after a successful run."
        )
    feature_order = list(model.get("feature_order") or [])
    if not feature_order:
        raise SystemExit(f"{model_json_path} has no feature_order.")
    return {"model": model, "contract": contract, "brain": brain, "paths": paths,
            "feature_order": feature_order, "completion_line": final_line}


def load_judgeable_frame(csv_path: Path, feature_order: list[str], label: str,
                         baseline_column: str | None):
    """Judgeable rows (0/1 label) of the CSV with the model features and the baseline."""
    import pandas as pd

    wanted = [label, *feature_order] + ([baseline_column] if baseline_column else [])
    header = pd.read_csv(csv_path, nrows=0).columns
    missing = [c for c in wanted if c not in header]
    if missing:
        raise SystemExit(f"CSV {csv_path} lacks columns {missing[:8]}")
    dtype = {name: "float32" for name in feature_order}
    frame = pd.read_csv(csv_path, usecols=wanted, dtype=dtype)
    n_rows = int(len(frame))
    labels = pd.to_numeric(frame[label], errors="coerce")
    judged = labels.notna().to_numpy()
    frame = frame.loc[judged].reset_index(drop=True)
    y = labels[judged].to_numpy(dtype=float)
    if not np.isin(y, (0.0, 1.0)).all():
        raise SystemExit(f"CSV {csv_path} has non-binary judgeable labels.")
    return frame, y.astype(int), n_rows


def _estimator_and_scaler(pipeline):
    steps = getattr(pipeline, "steps", None)
    if not steps:
        return pipeline, None, []
    estimator = steps[-1][1]
    scaler = next((step for _, step in steps if hasattr(step, "scale_")), None)
    return estimator, scaler, [name for name, _ in steps]


def model_views(pipeline, X: np.ndarray, y: np.ndarray, feature_order: list[str],
                *, repeats: int, seed: int) -> dict:
    """Coefficient, native and in-sample permutation views of the fitted winner."""
    from sklearn.inspection import permutation_importance

    estimator, scaler, step_names = _estimator_and_scaler(pipeline)
    views: dict = {"pipeline_steps": step_names, "estimator": type(estimator).__name__,
                   "coefficients": None, "native_importance": None, "permutation": None}
    coef = getattr(estimator, "coef_", None)
    if coef is not None:
        coef = np.asarray(coef, dtype=float).ravel()
        if coef.shape[0] == len(feature_order):
            intercept = getattr(estimator, "intercept_", None)
            views["coefficients"] = {
                "scale": ("standardized (pipeline scaler)" if scaler is not None
                          else "raw feature units (no scaler in the pipeline)"),
                "intercept": _finite_or_none(np.asarray(intercept).ravel()[0]) if intercept is not None else None,
                "values": {name: float(v) for name, v in zip(feature_order, coef)},
            }
    native = getattr(estimator, "feature_importances_", None)
    if native is not None:
        native = np.asarray(native, dtype=float).ravel()
        if native.shape[0] == len(feature_order):
            views["native_importance"] = {
                "kind": f"{type(estimator).__name__}.feature_importances_"
                        + (f" ({getattr(estimator, 'importance_type', None)})"
                           if getattr(estimator, "importance_type", None) else ""),
                "values": {name: float(v) for name, v in zip(feature_order, native)},
            }
    elif hasattr(estimator, "term_importances") and hasattr(estimator, "term_names_"):
        try:
            importances = np.asarray(estimator.term_importances(), dtype=float)
            names = list(estimator.term_names_)
        except Exception as exc:  # pragma: no cover - library specific
            importances, names = np.zeros(0), []
            views["native_importance_error"] = f"{type(exc).__name__}: {exc}"
        values, interactions = {}, {}
        for name, value in zip(names, importances):
            if " & " in name:
                interactions[name] = float(value)
                continue
            key = name
            if name not in feature_order and name.startswith("feature_") and name[8:].isdigit():
                index = int(name[8:])
                if index < len(feature_order):
                    key = feature_order[index]
            values[key] = float(value)
        if values:
            views["native_importance"] = {"kind": f"{type(estimator).__name__}.term_importances()",
                                          "values": values, "interactions": interactions}
    if len(y) and y.sum() and (y == 0).sum():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fitted_scores = pipeline.predict_proba(X)[:, 1]
            in_sample_ap = _average_precision(y, fitted_scores)
            result = permutation_importance(
                pipeline, X, y, scoring="average_precision", n_repeats=int(repeats),
                random_state=int(seed), n_jobs=1)
        views["permutation"] = {
            "scope": "in-sample: the fitted winner scored on the judgeable rows it was fitted on",
            "metric": "average_precision drop when one column is shuffled",
            "repeats": int(repeats), "seed": int(seed),
            "in_sample_ap": in_sample_ap,
            "mean": {name: float(v) for name, v in zip(feature_order, result.importances_mean)},
            "std": {name: float(v) for name, v in zip(feature_order, result.importances_std)},
        }
    return views


def univariate_view(frame, y: np.ndarray, feature_order: list[str]) -> dict:
    out = {}
    pos, neg = y == 1, y == 0
    for name in feature_order:
        values = frame[name].to_numpy(dtype=float)
        finite = np.isfinite(values)
        ap, direction = best_direction_ap(y, values)
        out[name] = {
            "ap": _finite_or_none(ap), "direction": direction,
            "defined_share": float(finite.mean()) if len(values) else 0.0,
            "defined_share_positives": float(finite[pos].mean()) if pos.any() else None,
            "median_positive": _finite_or_none(np.nanmedian(values[pos])) if finite[pos].any() else None,
            "median_negative": _finite_or_none(np.nanmedian(values[neg])) if finite[neg].any() else None,
        }
    return out


def hypothesis_context(application_dir: Path, inventory: dict, run_json: Path | None) -> dict:
    """Plain-language context per hypothesis id and per feature, read-only."""
    features: dict[str, dict] = {}
    hypotheses: dict[int, dict] = {}
    for record in inventory.get("hypotheses", []):
        try:
            hid = int(record.get("id"))
        except (TypeError, ValueError):
            continue
        hypotheses[hid] = {
            "included": bool(record.get("included")),
            "statistical_verdict": record.get("statistical_verdict"),
            "post_correction_verdict": record.get("post_correction_verdict"),
            "feature_source": record.get("feature_source"),
            "text": None,
        }
        for feature in record.get("features") or []:
            name = feature.get("name")
            if name:
                features[name] = {
                    "hypothesis_id": hid,
                    "quantity": str(feature.get("quantity") or "")[:320],
                    "measurable_condition": str(feature.get("measurable_condition") or "")[:200],
                }
    candidates = [run_json] if run_json else []
    candidates.append(PROJECT_ROOT / "autodiscovery" / f"{application_dir.name}.json")
    source = next((p for p in candidates if p and p.is_file()), None)
    if source is not None:
        try:
            records = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            records = []
        for record in records if isinstance(records, list) else []:
            try:
                hid = int(record.get("id"))
            except (TypeError, ValueError, AttributeError):
                continue
            text = str(record.get("hypothesis") or "").strip()
            if hid in hypotheses and text:
                hypotheses[hid]["text"] = text[:300]
    return {"features": features, "hypotheses": hypotheses,
            "origin_run_json": rel(source) if source is not None else None}


def _top(values: dict[str, float], n: int, key=abs) -> list[str]:
    return sorted(values, key=lambda name: -key(values[name]))[:n]


def collect_importance_evidence(result_dir: Path, application_dir: Path, *,
                                top_features: int = 20, top_groups: int = 12,
                                permutation_repeats: int = 5, seed: int = 0,
                                run_json: Path | None = None) -> dict:
    import joblib

    run = load_run(result_dir, application_dir)
    model, contract, feature_order = run["model"], run["contract"], run["feature_order"]
    baseline_column, baseline_sign = BASELINE_COLUMNS.get(contract["target"], (None, 1.0))
    frame, y, n_rows = load_judgeable_frame(run["paths"]["csv"], feature_order,
                                            contract["label"], baseline_column)
    if not len(y) or not y.sum():
        raise SystemExit("The CSV has no judgeable positive rows; nothing to rank.")
    audit = model.get("audit") if isinstance(model.get("audit"), dict) else {}
    if "n_judgeable" in audit and int(audit["n_judgeable"]) != int(len(y)):
        raise SystemExit("Model-selection audit n_judgeable does not match the CSV's 0/1 rows.")
    bundle = joblib.load(run["paths"]["joblib"])
    pipeline = bundle["pipeline"] if isinstance(bundle, dict) else bundle
    saved_order = list(bundle.get("feature_order", feature_order)) if isinstance(bundle, dict) else feature_order
    if saved_order != feature_order:
        raise SystemExit("joblib feature_order differs from the model-selection JSON.")
    X = frame[feature_order].to_numpy(dtype=float)
    views = model_views(pipeline, X, y, feature_order, repeats=permutation_repeats, seed=seed)
    univariate = univariate_view(frame, y, feature_order)
    baseline = None
    if baseline_column:
        values = frame[baseline_column].to_numpy(dtype=float) * baseline_sign
        baseline = {"column": baseline_column,
                    "direction": "higher first" if baseline_sign > 0 else "lower first",
                    "ap": _average_precision(y, np.where(np.isfinite(values), values,
                                                         np.nanmin(values) - 1.0))}
    inventory = load_json_object(run["paths"]["inventory"])
    context = hypothesis_context(application_dir, inventory, run_json)

    def record(name: str) -> dict:
        hid = feature_hypothesis_id(name)
        meta = context["features"].get(name, {})
        item = {"feature": name, "hypothesis_id": hid,
                "quantity": meta.get("quantity"), **univariate[name]}
        if views["coefficients"]:
            item["coefficient"] = views["coefficients"]["values"].get(name)
        if views["native_importance"]:
            item["native_importance"] = views["native_importance"]["values"].get(name)
        if views["permutation"]:
            item["permutation_drop"] = views["permutation"]["mean"].get(name)
            item["permutation_sd"] = views["permutation"]["std"].get(name)
        return item

    ranked: dict[str, list[dict]] = {}
    if views["coefficients"]:
        ranked["by_coefficient"] = [record(n) for n in _top(views["coefficients"]["values"], top_features)]
    if views["native_importance"]:
        ranked["by_native_importance"] = [record(n) for n in _top(views["native_importance"]["values"], top_features)]
    if views["permutation"]:
        ranked["by_permutation"] = [record(n) for n in _top(views["permutation"]["mean"], top_features, key=float)]
    ranked["by_univariate_ap"] = [record(n) for n in sorted(
        feature_order, key=lambda n: -(univariate[n]["ap"] or 0.0))[:top_features]]

    groups: dict[int, dict] = {}
    for name in feature_order:
        hid = feature_hypothesis_id(name)
        if hid is None:
            continue
        group = groups.setdefault(hid, {
            "hypothesis_id": hid, "n_features": 0, "sum_abs_coefficient": 0.0,
            "sum_native_importance": 0.0, "sum_permutation_drop": 0.0,
            "best_univariate_ap": 0.0, "best_univariate_feature": None,
            "text": context["hypotheses"].get(hid, {}).get("text"),
            "statistical_verdict": context["hypotheses"].get(hid, {}).get("statistical_verdict"),
            "post_correction_verdict": context["hypotheses"].get(hid, {}).get("post_correction_verdict"),
        })
        group["n_features"] += 1
        if views["coefficients"]:
            group["sum_abs_coefficient"] += abs(views["coefficients"]["values"].get(name, 0.0))
        if views["native_importance"]:
            group["sum_native_importance"] += views["native_importance"]["values"].get(name, 0.0)
        if views["permutation"]:
            group["sum_permutation_drop"] += views["permutation"]["mean"].get(name, 0.0)
        ap = univariate[name]["ap"] or 0.0
        if ap > group["best_univariate_ap"]:
            group["best_univariate_ap"], group["best_univariate_feature"] = ap, name
    primary = ("sum_abs_coefficient" if views["coefficients"] else
               "sum_native_importance" if views["native_importance"] else
               "sum_permutation_drop" if views["permutation"] else "best_univariate_ap")
    group_rank = sorted(groups.values(), key=lambda g: -g[primary])[:top_groups]

    heldout = model.get("heldout") if isinstance(model.get("heldout"), dict) else None
    return {
        "schema_version": EVIDENCE_SCHEMA,
        "result_dir": rel(result_dir), "application_dir": rel(application_dir),
        "brain": run["brain"], "detector_target": contract["target"],
        "row_unit": contract["row_unit"], "label_name": contract["label"],
        "positive_name": contract["positive"],
        "completion_line": run["completion_line"],
        "n_rows": n_rows, "n_judgeable": int(len(y)), "n_positive": int(y.sum()),
        "prevalence": float(y.mean()),
        "final_winner": model.get("final_winner"), "final_params": model.get("final_params"),
        "per_family_outer_metrics": model.get("per_family_outer_metrics"),
        "selector_metrics": model.get("selector_metrics"),
        "baseline": baseline,
        "heldout": None if heldout is None else {
            "test_brain": heldout.get("test_brain"),
            "held_out_average_precision": heldout.get("held_out_average_precision"),
            "held_out_roc_auc": heldout.get("held_out_roc_auc"),
            "n_test_rows": heldout.get("n_test_rows"),
            "n_test_judgeable": heldout.get("n_test_judgeable"),
            "n_test_positive": heldout.get("n_test_positive"),
        },
        "n_features": len(feature_order),
        "views": {k: v for k, v in views.items() if k != "permutation"}
                 | ({"permutation": {k: v for k, v in views["permutation"].items() if k not in ("mean", "std")}}
                    if views["permutation"] else {}),
        "group_primary_view": primary,
        "ranked_features": ranked,
        "ranked_groups": group_rank,
        "hypothesis_context": {"origin_run_json": context["origin_run_json"],
                               "n_hypotheses": len(context["hypotheses"])},
        "artifacts": {label: {"path": rel(path), "sha256": sha256(path)}
                      for label, path in run["paths"].items()},
    }


def _f(value, digits=4) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "—"
    return f"{value:.{digits}f}"


def _short(text: str | None, limit: int = 110) -> str:
    text = (text or "").replace("|", "/").replace("\n", " ").strip()
    return text if len(text) <= limit else text[:limit - 1] + "…"


def evidence_markdown(evidence: dict) -> str:
    """Deterministic tables the summarizer must not alter."""
    lines = [f"- Run: `{evidence['result_dir']}` (`{evidence['detector_target']}`, brain `{evidence['brain']}`)",
             f"- Winner: `{evidence['final_winner']}` `{json.dumps(evidence['final_params'], sort_keys=True)}`; "
             f"{evidence['n_features']} features; estimator `{evidence['views']['estimator']}`",
             f"- Judgeable rows: {evidence['n_judgeable']} of {evidence['n_rows']}; positives "
             f"({evidence['positive_name']}): {evidence['n_positive']}; prevalence {evidence['prevalence']:.4f}"]
    fam = evidence.get("per_family_outer_metrics") or {}
    if fam:
        lines.append("- Nested OOF AP per family: " + ", ".join(
            f"`{k}` {_f(v.get('oof_average_precision'))}" for k, v in sorted(fam.items())))
    if evidence.get("baseline"):
        b = evidence["baseline"]
        lines.append(f"- Raw baseline `{b['column']}` ({b['direction']}): AP {_f(b['ap'])} on the judgeable rows")
    perm = evidence["views"].get("permutation")
    if perm:
        lines.append(f"- In-sample AP of the fitted winner on the judgeable rows: {_f(perm['in_sample_ap'])} "
                     f"(permutation repeats {perm['repeats']})")
    if evidence.get("heldout"):
        h = evidence["heldout"]
        lines.append(f"- Held-out `{h['test_brain']}`: AP {_f(h['held_out_average_precision'])}, "
                     f"ROC-AUC {_f(h['held_out_roc_auc'])}, {h['n_test_judgeable']} judgeable rows, "
                     f"{h['n_test_positive']} positives")
    lines.append("")
    ranked = evidence["ranked_features"]
    primary_key = ("by_coefficient" if "by_coefficient" in ranked else
                   "by_native_importance" if "by_native_importance" in ranked else
                   "by_permutation" if "by_permutation" in ranked else "by_univariate_ap")
    title = {"by_coefficient": "standardized coefficient",
             "by_native_importance": "native importance",
             "by_permutation": "in-sample permutation drop",
             "by_univariate_ap": "univariate AP"}[primary_key]
    lines.append(f"**Top features by {title}** (univariate AP is the feature alone, best direction, judgeable rows)")
    lines.append("")
    lines.append("| # | feature | hyp | coef | native | perm drop ± sd | univariate AP | defined | what it measures |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---|")
    for i, item in enumerate(ranked[primary_key], 1):
        perm_text = ("—" if item.get("permutation_drop") is None
                     else f"{item['permutation_drop']:+.4f} ± {item.get('permutation_sd') or 0:.4f}")
        coef_text = "—" if item.get("coefficient") is None else f"{item['coefficient']:+.4f}"
        lines.append(f"| {i} | `{item['feature']}` | {item['hypothesis_id'] if item['hypothesis_id'] is not None else '—'} | "
                     f"{coef_text} | {_f(item.get('native_importance'))} | {perm_text} | "
                     f"{_f(item['ap'])} ({item['direction']}) | {item['defined_share']:.2f} | {_short(item.get('quantity'))} |")
    if "by_permutation" in ranked and primary_key != "by_permutation":
        lines.append("")
        lines.append("**Top features by in-sample permutation drop**")
        lines.append("")
        lines.append("| # | feature | hyp | perm drop ± sd | coef | univariate AP |")
        lines.append("|---:|---|---:|---:|---:|---:|")
        for i, item in enumerate(ranked["by_permutation"], 1):
            coef_text = "—" if item.get("coefficient") is None else f"{item['coefficient']:+.4f}"
            lines.append(f"| {i} | `{item['feature']}` | {item['hypothesis_id'] if item['hypothesis_id'] is not None else '—'} | "
                         f"{item['permutation_drop']:+.4f} ± {item.get('permutation_sd') or 0:.4f} | {coef_text} | "
                         f"{_f(item['ap'])} ({item['direction']}) |")
    lines.append("")
    lines.append(f"**Hypothesis groups by `{evidence['group_primary_view']}`** (sum over the group's features)")
    lines.append("")
    lines.append("| # | hyp | features | Σ|coef| | Σ native | Σ perm drop | best univariate AP (feature) | verdicts | hypothesis |")
    lines.append("|---:|---:|---:|---:|---:|---:|---|---|---|")
    has_coef = bool(evidence["views"].get("coefficients"))
    has_native = bool(evidence["views"].get("native_importance"))
    has_perm = bool(evidence["views"].get("permutation"))
    for i, g in enumerate(evidence["ranked_groups"], 1):
        verdicts = "/".join(str(v) for v in (g.get("statistical_verdict"), g.get("post_correction_verdict")) if v)
        lines.append(f"| {i} | {g['hypothesis_id']} | {g['n_features']} | "
                     f"{g['sum_abs_coefficient']:.4f} | " if has_coef else f"| {i} | {g['hypothesis_id']} | {g['n_features']} | — | ")
        lines[-1] += (f"{g['sum_native_importance']:.4f} | " if has_native else "— | ")
        lines[-1] += (f"{g['sum_permutation_drop']:+.4f} | " if has_perm else "— | ")
        lines[-1] += (f"{g['best_univariate_ap']:.4f} (`{g['best_univariate_feature']}`) | {verdicts or '—'} | "
                      f"{_short(g.get('text'), 140)} |")
    return "\n".join(lines) + "\n"
