"""Fixed-budget ranking benchmark. Scorers never receive labels or candidate IDs.

Labels are three-valued (harness/label_availability.py): 1 = error, 0 = GT-confirmed
no error, NaN = not judgeable by GT. Scorers score every row; ranking, Precision@K,
recall and the bootstrap use GT-labeled rows only, so NaN rows never count as hits
or false positives.
"""

import hashlib
import math

import numpy as np

from .isolated_scoring import score
from .scorer_components import components
from .training_diagnostics import describe
from .local_features import augmented_features
from .model_execution import DEFAULT_MEMORY_MB
from .label_availability import labeled_rows, labels_digest


SCORING_VERSION = "native-precision-at-k-v2-three-valued"
AGGREGATION = "equal-brain-mean-kind-precision"
VALIDATION_GATE_VERSION = "mean-validation-precision-v2-three-valued"
BOOTSTRAP_GATE_VERSION = "mean-validation-bootstrap-v2-three-valued"
GATE_MODES = ("bootstrap", "margin")


def gate_version(mode):
    if mode not in GATE_MODES:
        raise ValueError(f"Unknown promotion gate mode {mode!r}")
    return BOOTSTRAP_GATE_VERSION if mode == "bootstrap" else VALIDATION_GATE_VERSION


def aggregate_precision(cells):
    """Give each brain equal weight, with the same kinds represented in each."""
    grouped = {}
    for name, cell in cells.items():
        brain, kind = name.rsplit("/", 1)
        precision = float(cell["precision"])
        if not math.isfinite(precision) or not 0 <= precision <= 1:
            raise ValueError(f"Invalid precision on {name}")
        grouped.setdefault(brain, {})[kind] = precision
    if not grouped:
        raise ValueError("No candidate pools supplied")
    if len({tuple(sorted(values)) for values in grouped.values()}) != 1:
        raise ValueError("Every evaluated brain must have the same candidate kinds")
    per_brain = {brain: math.fsum(values.values()) / len(values)
                 for brain, values in grouped.items()}
    result = {"aggregation": AGGREGATION, "per_brain_precision": per_brain,
              "macro_precision": math.fsum(per_brain.values()) / len(per_brain)}
    return result


def labeled_order(scores, truth, keys):
    """GT-labeled rows in ranking order; candidate identity resolves ties reproducibly."""
    rows = labeled_rows(truth)
    return rows[np.lexsort((np.asarray(keys, dtype=str)[rows], -np.asarray(scores, dtype=float)[rows]))]


def rank_metrics(scores, truth, keys, k):
    """Precision@K over GT-labeled rows: the Top-K is taken among rows labeled 1 or 0."""
    scores, truth = np.asarray(scores, dtype=float), np.asarray(truth, dtype=float)
    if type(k) is not int or k < 1:
        raise ValueError("K must be a positive integer")
    if (scores.shape != truth.shape or scores.ndim != 1 or len(keys) != len(scores)
            or not len(scores) or len(set(keys)) != len(keys) or not np.isfinite(scores).all()):
        raise ValueError("Return one finite score per candidate in original row order; no dropping/abstention")
    order = labeled_order(scores, truth, keys)
    if not len(order):
        raise ValueError("No GT-labeled rows in this pool")
    chosen = order[:min(k, len(order))]
    tp = int(truth[chosen].sum())
    positives = int(np.nansum(truth))
    return {"requested_k": k, "effective_k": len(chosen), "pool_size": len(scores),
            "n_labeled": len(order), "n_unlabeled": len(scores) - len(order),
            "tp": tp, "fp": len(chosen) - tp, "precision": tp / len(chosen),
            "recall": tp / positives if positives else None,
            "positives": positives}, chosen


def evaluate(source, tables_by_brain, budgets, timeout=120, examples=False, *,
             state=None, parent_state=None, generation=0, artifact_store=None, memory_mb=DEFAULT_MEMORY_MB):
    """Score features in a restricted worker; labels and ranking stay here."""
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Scorer timeout must be positive and finite")
    report = {"version": SCORING_VERSION,
              "policy_sha256": hashlib.sha256(source.encode()).hexdigest(), "cells": {}}
    sources = components(source)
    for brain, banks in tables_by_brain.items():
        for kind, table in banks.tables.items():
            features = augmented_features(sources[kind], table, timeout)
            scores = (score(sources[kind], features, kind, timeout, memory_mb=memory_mb, artifact_store=artifact_store)
                      if artifact_store is not None else score(sources[kind], features, kind, timeout, memory_mb=memory_mb))
            metrics, chosen = rank_metrics(scores, table.truth, table.keys, budgets[kind])
            if 'local_context' in features.attrs:
                metrics['local_context'] = features.attrs['local_context']
            if 'image_context' in features.attrs:
                from .image_selection import scoring_coverage
                metrics['image_context'] = {**features.attrs['image_context'],
                    'scoring_coverage': scoring_coverage(features, chosen, table.truth, budgets[kind])}
            metrics["pool_sha256"] = table.meta["pool_sha256"]
            metrics["labels_sha256"] = labels_digest(table.truth)
            cell = f"{brain}/{kind}"
            report["cells"][cell] = metrics
            if state is not None:
                state[cell] = {'scores': scores, 'chosen': chosen}
            if examples:
                metrics.update(describe(table, scores, chosen, generation, cell,
                                        (parent_state or {}).get(cell), features=features))
    report.update(aggregate_precision(report["cells"]))
    return report


def acceptance(parent, candidate, margin=0, *, require_no_cell_regression=True):
    """Compare means at identical K/pools; optionally veto individual regressions."""
    if not math.isfinite(margin) or margin < 0:
        raise ValueError("Precision margin must be finite and nonnegative")
    if parent["version"] != candidate["version"] or parent["cells"].keys() != candidate["cells"].keys():
        raise ValueError("Cannot compare different evaluation contracts")
    for cell, before in parent["cells"].items():
        after = candidate["cells"][cell]
        for key in ("pool_sha256", "labels_sha256", "pool_size", "n_labeled", "requested_k", "effective_k",
                    "positives"):
            if before[key] != after[key]:
                raise ValueError(f"Candidate changed fixed benchmark: {cell}/{key}")
    before_mean = aggregate_precision(parent["cells"])["macro_precision"]
    after_mean = aggregate_precision(candidate["cells"])["macro_precision"]
    if require_no_cell_regression:
        for cell, before in parent["cells"].items():
            if candidate["cells"][cell]["precision"] < before["precision"]:
                return False, f"Precision regressed on {cell}"
    gain = after_mean - before_mean
    passed = gain > max(margin, 1e-12)
    reason = "Mean Precision@K improved" if passed else "No sufficient mean Precision@K gain"
    return passed, (f"{reason}: {before_mean:.6f} -> {after_mean:.6f}; "
                    f"delta={gain:+.6f}, margin={margin:g}")


BOOTSTRAP_VERSION = "paired-poisson-bootstrap-v2-three-valued"


def _prefix_hits(order, truth, counts, k):
    """TP and effective K of the Top-K over a resampled multiset in ranking order."""
    weights = counts[order]
    cumulative = np.cumsum(weights)
    if not len(cumulative) or cumulative[-1] <= k:
        return int((truth[order] * weights).sum()), int(cumulative[-1]) if len(cumulative) else 0
    stop = int(np.searchsorted(cumulative, k))
    partial = k - (int(cumulative[stop - 1]) if stop else 0)
    hits = int((truth[order[:stop]] * weights[:stop]).sum()) + int(truth[order[stop]]) * partial
    return hits, k


def paired_bootstrap(parent_state, candidate_state, tables_by_brain, budgets, *, draws=1000, seed=0, alpha=.05):
    """Paired resampling of GT-labeled candidate rows for the promotion delta.

    Both scorers are re-ranked on the same Poisson(1)-weighted multiset per cell,
    so the interval reflects row sampling noise, not scorer nondeterminism. The
    two-sided (1 - alpha) interval is what `bootstrap_acceptance` reads; the 95%
    interval is always recorded as well.
    """
    if type(draws) is not int or draws < 1:
        raise ValueError("Bootstrap draws must be a positive integer")
    if not (isinstance(alpha, float) and math.isfinite(alpha) and 0. < alpha < 1.):
        raise ValueError("Bootstrap alpha must be a float in (0, 1)")
    rng = np.random.default_rng(seed)
    cells = {}
    for brain, banks in tables_by_brain.items():
        for kind, table in banks.tables.items():
            cell = f"{brain}/{kind}"
            truth = np.asarray(table.truth, dtype=float)
            before = np.asarray(parent_state[cell]["scores"], dtype=float)
            after = np.asarray(candidate_state[cell]["scores"], dtype=float)
            if before.shape != truth.shape or after.shape != truth.shape:
                raise ValueError(f"Bootstrap scores do not align with {cell}")
            # Resample labeled rows only: weights are drawn per labeled row and unlabeled
            # rows never enter either ranking, matching rank_metrics.
            rows = labeled_rows(truth)
            position = np.full(len(truth), -1, dtype=np.int64)
            position[rows] = np.arange(len(rows))
            labels = truth[rows].astype(np.int64)
            cells[cell] = (labels, position[labeled_order(before, truth, table.keys)],
                           position[labeled_order(after, truth, table.keys)], budgets[kind])
    deltas = np.empty(draws)
    per_cell = {cell: np.empty(draws) for cell in cells}
    for draw in range(draws):
        sampled = {}
        for cell, (truth, order_before, order_after, k) in cells.items():
            counts = rng.poisson(1., len(truth)).astype(np.int64)
            hits_before, k_before = _prefix_hits(order_before, truth, counts, k)
            hits_after, k_after = _prefix_hits(order_after, truth, counts, k)
            precision_before = hits_before / k_before if k_before else 0.
            precision_after = hits_after / k_after if k_after else 0.
            per_cell[cell][draw] = precision_after - precision_before
            sampled[cell] = precision_after - precision_before
        grouped = {}
        for cell, value in sampled.items():
            brain, _ = cell.rsplit("/", 1)
            grouped.setdefault(brain, []).append(value)
        deltas[draw] = math.fsum(math.fsum(v) / len(v) for v in grouped.values()) / len(grouped)
    bounds = [100. * alpha / 2., 100. * (1. - alpha / 2.)]
    low, high = np.percentile(deltas, bounds)
    return {"version": BOOTSTRAP_VERSION, "draws": draws, "seed": seed, "alpha": alpha,
            "macro_delta_ci": [float(low), float(high)],
            "macro_delta_ci95": [float(v) for v in np.percentile(deltas, [2.5, 97.5])],
            "macro_delta_point": float(np.mean(deltas)),
            "p_delta_le_0": float(np.mean(deltas <= 0)), "p_delta_ge_0": float(np.mean(deltas >= 0)),
            "cells": {cell: {"delta_ci": [float(v) for v in np.percentile(values, bounds)],
                             "delta_ci95": [float(v) for v in np.percentile(values, [2.5, 97.5])]}
                      for cell, values in per_cell.items()},
            "scope": "Row-resampling uncertainty of the paired Precision@K delta over the fixed pools; "
                     "the configured promotion gate mode decides whether it is binding."}


def bootstrap_acceptance(parent, candidate, bootstrap, margin=0, *, alpha=.05):
    """Promote only when the point gain clears the margin AND the lower bound of the
    paired bootstrap interval of the mean delta is above zero.

    A missing or failed bootstrap never promotes (fail closed). The point-estimate
    comparison reuses `acceptance`, so benchmark identity checks still apply.
    """
    passed_point, point_reason = acceptance(parent, candidate, margin, require_no_cell_regression=False)
    details = point_reason.split(": ", 1)[1]
    if not passed_point:
        return False, point_reason
    if not isinstance(bootstrap, dict) or bootstrap.get("status") == "unavailable" \
            or "macro_delta_ci" not in bootstrap:
        return False, f"Bootstrap interval unavailable; candidate not promoted: {details}"
    if bootstrap.get("alpha") != alpha:
        raise ValueError("Bootstrap interval level differs from the gate alpha")
    lower = float(bootstrap["macro_delta_ci"][0])
    if not math.isfinite(lower):
        return False, f"Bootstrap interval not finite; candidate not promoted: {details}"
    detail = (f"{details}; bootstrap lower bound {lower:+.6f} at alpha={alpha:g} "
              f"({bootstrap['draws']} draws)")
    if lower > 0.:
        return True, f"Mean Precision@K improved beyond resampling noise: {detail}"
    return False, f"Mean Precision@K gain within resampling noise: {detail}"
