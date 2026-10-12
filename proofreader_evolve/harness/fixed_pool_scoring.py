"""Fixed-budget ranking benchmark. Scorers never receive labels or candidate IDs.

Labels are three-valued (harness/label_availability.py): 1 = error, 0 = GT-confirmed
no error, NaN = not judgeable by GT. Scorers score every row; ranking, Precision@K,
recall and the bootstrap use GT-labeled rows only, so NaN rows never count as hits
or false positives.

Since 2026-10-12 (v3) each pool's K is resolved from its own GT positives through
``Budgets``: ``K = min(cap, ceil(positive_ratio * positives))``, so every pool that
counts has the same attainable ceiling (1 / positive_ratio when the ratio binds), and
pools with fewer than ``min_positives`` GT positives are scored and reported but
excluded from the promotion mean, the bootstrap and kind headroom. Both inputs are
label counts, so every K is fixed for a run. The mean is taken per kind over the
counted brains and then over kinds; with every pool counted this equals the former
equal-brain mean of per-brain kind means.
"""

from collections.abc import Mapping
import hashlib
import math

import numpy as np

from .isolated_scoring import score
from .scorer_components import components
from .training_diagnostics import describe
from .local_features import augmented_features
from .model_execution import DEFAULT_MEMORY_MB
from .label_availability import labeled_rows, labels_digest


SCORING_VERSION = "native-precision-at-k-v3-positive-scaled-k"
AGGREGATION = "equal-kind-mean-of-equal-brain-precision"
VALIDATION_GATE_VERSION = "mean-validation-precision-v3-positive-scaled-k"
BOOTSTRAP_GATE_VERSION = "mean-validation-bootstrap-v3-positive-scaled-k"
GATE_MODES = ("bootstrap", "margin")


class Budgets(Mapping):
    """Per-kind Top-K caps plus the rule resolving each pool's K from its GT positives.

    ``budgets[kind]`` is the cap (the historical fixed K). ``k_for(kind, truth)`` is the
    K ranked on one pool: ``min(cap, ceil(positive_ratio * positives))`` when
    ``positive_ratio > 0``, otherwise the cap. ``in_mean(truth)`` is False for pools with
    fewer than ``min_positives`` GT positives; those pools are scored and reported but
    never enter the promotion mean, the bootstrap or kind headroom. Both depend on
    label counts only, so every pool's K is fixed for a run. A plain ``{kind: K}``
    mapping coerces to fixed-K budgets that count every pool (the pre-v3 behaviour).
    """

    def __init__(self, caps, positive_ratio=0., min_positives=0):
        caps = dict(caps)
        for kind, cap in caps.items():
            if type(cap) is not int or cap < 1:
                raise ValueError(f"Top-K cap for {kind} must be a positive integer")
        if (isinstance(positive_ratio, bool) or not isinstance(positive_ratio, (int, float))
                or not math.isfinite(positive_ratio) or positive_ratio < 0):
            raise ValueError("positive_ratio must be a finite nonnegative number (0 keeps the fixed cap)")
        if type(min_positives) is not int or min_positives < 0:
            raise ValueError("min_positives must be a nonnegative integer (0 counts every pool)")
        self.caps, self.positive_ratio, self.min_positives = caps, float(positive_ratio), min_positives

    @classmethod
    def coerce(cls, value):
        return value if isinstance(value, cls) else cls(value)

    def __getitem__(self, kind):
        return self.caps[kind]

    def __iter__(self):
        return iter(self.caps)

    def __len__(self):
        return len(self.caps)

    @staticmethod
    def positives(truth):
        return int(np.nansum(np.asarray(truth, dtype=float)))

    def k_for(self, kind, truth):
        cap = self.caps[kind]
        if self.positive_ratio <= 0:
            return cap
        return max(1, min(cap, math.ceil(self.positive_ratio * self.positives(truth))))

    def in_mean(self, truth):
        return self.positives(truth) >= self.min_positives

    def describe(self):
        return {"caps": dict(self.caps), "positive_ratio": self.positive_ratio,
                "min_positives": self.min_positives,
                "k_rule": ("K per pool = min(cap, ceil(positive_ratio x GT positives))"
                           if self.positive_ratio > 0 else "K per pool = cap (fixed)"),
                "mean_rule": (f"pools with fewer than {self.min_positives} GT positives are scored and "
                              "reported but excluded from the mean"
                              if self.min_positives > 0 else "every pool counts toward the mean")}

    def resolve(self, tables_by_brain):
        """Per-pool K and mean membership for a bank mapping; depends on labels only."""
        cells = {}
        for brain, banks in tables_by_brain.items():
            for kind, table in banks.tables.items():
                if kind in self.caps:
                    cells[f"{brain}/{kind}"] = {"positives": self.positives(table.truth), "cap": self.caps[kind],
                                                "requested_k": self.k_for(kind, table.truth),
                                                "in_mean": self.in_mean(table.truth)}
        return cells


def gate_version(mode):
    if mode not in GATE_MODES:
        raise ValueError(f"Unknown promotion gate mode {mode!r}")
    return BOOTSTRAP_GATE_VERSION if mode == "bootstrap" else VALIDATION_GATE_VERSION


def aggregate_precision(cells):
    """Equal-kind mean of the equal-brain mean precision over the pools that count.

    Pools flagged ``in_mean: false`` (too few GT positives) are reported but skipped.
    Every evaluated brain must still carry the same kinds, so the grid stays complete;
    when every pool counts this equals the former equal-brain mean of per-brain means.
    """
    grouped, by_kind, excluded = {}, {}, []
    for name, cell in cells.items():
        brain, kind = name.rsplit("/", 1)
        precision = float(cell["precision"])
        if not math.isfinite(precision) or not 0 <= precision <= 1:
            raise ValueError(f"Invalid precision on {name}")
        grouped.setdefault(brain, set()).add(kind)
        if not cell.get("in_mean", True):
            excluded.append(name)
            continue
        by_kind.setdefault(kind, {})[brain] = precision
    if not grouped:
        raise ValueError("No candidate pools supplied")
    if len({tuple(sorted(values)) for values in grouped.values()}) != 1:
        raise ValueError("Every evaluated brain must have the same candidate kinds")
    missing = [kind for kind in sorted(next(iter(grouped.values()))) if kind not in by_kind]
    if missing:
        raise ValueError(f"No {missing[0]} pool counts toward the mean: every {missing[0]} pool is below "
                         "the GT-positive floor")
    per_kind = {kind: math.fsum(values.values()) / len(values) for kind, values in sorted(by_kind.items())}
    per_brain = {}
    for values in by_kind.values():
        for brain, precision in values.items():
            per_brain.setdefault(brain, []).append(precision)
    per_brain = {brain: math.fsum(v) / len(v) for brain, v in sorted(per_brain.items())}
    return {"aggregation": AGGREGATION, "per_kind_precision": per_kind, "per_brain_precision": per_brain,
            "excluded_cells": sorted(excluded),
            "macro_precision": math.fsum(per_kind.values()) / len(per_kind)}


def labeled_order(scores, truth, keys):
    """GT-labeled rows in ranking order; candidate identity resolves ties reproducibly."""
    rows = labeled_rows(truth)
    return rows[np.lexsort((np.asarray(keys, dtype=str)[rows], -np.asarray(scores, dtype=float)[rows]))]


def rank_metrics(scores, truth, keys, k, *, in_mean=True):
    """Precision@K over GT-labeled rows: the Top-K is taken among rows labeled 1 or 0.

    ``k`` is this pool's resolved K (see ``Budgets.k_for``); ``in_mean`` records whether
    the pool counts toward the mean (``Budgets.in_mean``).
    """
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
            "positives": positives, "in_mean": bool(in_mean)}, chosen


def evaluate(source, tables_by_brain, budgets, timeout=120, examples=False, *,
             state=None, parent_state=None, generation=0, artifact_store=None, memory_mb=DEFAULT_MEMORY_MB):
    """Score features in a restricted worker; labels and ranking stay here."""
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Scorer timeout must be positive and finite")
    budgets = Budgets.coerce(budgets)
    report = {"version": SCORING_VERSION, "budgets": budgets.describe(),
              "policy_sha256": hashlib.sha256(source.encode()).hexdigest(), "cells": {}}
    sources = components(source)
    for brain, banks in tables_by_brain.items():
        for kind, table in banks.tables.items():
            features = augmented_features(sources[kind], table, timeout)
            scores = (score(sources[kind], features, kind, timeout, memory_mb=memory_mb, artifact_store=artifact_store)
                      if artifact_store is not None else score(sources[kind], features, kind, timeout, memory_mb=memory_mb))
            k = budgets.k_for(kind, table.truth)
            metrics, chosen = rank_metrics(scores, table.truth, table.keys, k, in_mean=budgets.in_mean(table.truth))
            if 'local_context' in features.attrs:
                metrics['local_context'] = features.attrs['local_context']
            if 'image_context' in features.attrs:
                from .image_selection import scoring_coverage
                metrics['image_context'] = {**features.attrs['image_context'],
                    'scoring_coverage': scoring_coverage(features, chosen, table.truth, k)}
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
        if before.get("in_mean", True) != after.get("in_mean", True):
            raise ValueError(f"Candidate changed fixed benchmark: {cell}/in_mean")
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


BOOTSTRAP_VERSION = "paired-poisson-bootstrap-v3-positive-scaled-k"


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
    interval is always recorded as well. Each cell uses its own resolved K; cells
    below the GT-positive floor get a per-cell interval but never enter the mean
    delta, matching `aggregate_precision`.
    """
    if type(draws) is not int or draws < 1:
        raise ValueError("Bootstrap draws must be a positive integer")
    if not (isinstance(alpha, float) and math.isfinite(alpha) and 0. < alpha < 1.):
        raise ValueError("Bootstrap alpha must be a float in (0, 1)")
    budgets = Budgets.coerce(budgets)
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
                           position[labeled_order(after, truth, table.keys)],
                           budgets.k_for(kind, table.truth), budgets.in_mean(table.truth))
    counted = {cell for cell, values in cells.items() if values[4]}
    if not counted:
        raise ValueError("No candidate pool counts toward the bootstrap mean")
    deltas = np.empty(draws)
    per_cell = {cell: np.empty(draws) for cell in cells}
    for draw in range(draws):
        sampled = {}
        for cell, (truth, order_before, order_after, k, _) in cells.items():
            counts = rng.poisson(1., len(truth)).astype(np.int64)
            hits_before, k_before = _prefix_hits(order_before, truth, counts, k)
            hits_after, k_after = _prefix_hits(order_after, truth, counts, k)
            precision_before = hits_before / k_before if k_before else 0.
            precision_after = hits_after / k_after if k_after else 0.
            per_cell[cell][draw] = precision_after - precision_before
            sampled[cell] = precision_after - precision_before
        by_kind = {}
        for cell in counted:
            _, kind = cell.rsplit("/", 1)
            by_kind.setdefault(kind, []).append(sampled[cell])
        deltas[draw] = math.fsum(math.fsum(v) / len(v) for v in by_kind.values()) / len(by_kind)
    bounds = [100. * alpha / 2., 100. * (1. - alpha / 2.)]
    low, high = np.percentile(deltas, bounds)
    return {"version": BOOTSTRAP_VERSION, "draws": draws, "seed": seed, "alpha": alpha,
            "aggregation": AGGREGATION, "excluded_cells": sorted(set(cells) - counted),
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
