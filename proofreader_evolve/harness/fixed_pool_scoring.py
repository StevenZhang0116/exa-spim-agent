"""Fixed-budget ranking benchmark. Scorers never receive labels or candidate IDs."""

import hashlib
import math

import numpy as np

from .isolated_scoring import score
from .scorer_components import components
from .training_diagnostics import describe


SCORING_VERSION = "native-precision-at-k-v1"
AGGREGATION = "equal-brain-mean-kind-precision"
VALIDATION_GATE_VERSION = "mean-validation-precision-v1"


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
    return {"aggregation": AGGREGATION, "per_brain_precision": per_brain,
            "macro_precision": math.fsum(per_brain.values()) / len(per_brain)}


def rank_metrics(scores, truth, keys, k):
    scores, truth = np.asarray(scores, dtype=float), np.asarray(truth)
    if type(k) is not int or k < 1:
        raise ValueError("K must be a positive integer")
    if (scores.shape != truth.shape or scores.ndim != 1 or len(keys) != len(scores)
            or not len(scores) or len(set(keys)) != len(keys)
            or not np.isfinite(scores).all() or not np.isin(truth, [0, 1]).all()):
        raise ValueError("Return one finite score per candidate in original row order; no dropping/abstention")
    # Candidate identity, not policy output order, resolves ties reproducibly.
    order = np.lexsort((np.asarray(keys, dtype=str), -scores))
    chosen = order[:min(k, len(order))]
    tp = int(truth[chosen].sum())
    positives = int(truth.sum())
    return {"requested_k": k, "effective_k": len(chosen), "pool_size": len(order),
            "tp": tp, "fp": len(chosen) - tp, "precision": tp / len(chosen),
            "recall": tp / positives if positives else None,
            "positives": positives}, chosen


def evaluate(source, tables_by_brain, budgets, timeout=120, examples=False, *,
             state=None, parent_state=None, generation=0, artifact_store=None):
    """Score features in a restricted worker; labels and ranking stay here."""
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Scorer timeout must be positive and finite")
    report = {"version": SCORING_VERSION,
              "policy_sha256": hashlib.sha256(source.encode()).hexdigest(), "cells": {}}
    sources = components(source)
    for brain, banks in tables_by_brain.items():
        for kind, table in banks.tables.items():
            scores = (score(sources[kind], table.features, kind, timeout, artifact_store=artifact_store)
                      if artifact_store is not None else score(sources[kind], table.features, kind, timeout))
            metrics, chosen = rank_metrics(scores, table.truth, table.keys, budgets[kind])
            metrics["pool_sha256"] = table.meta["pool_sha256"]
            metrics["labels_sha256"] = hashlib.sha256(np.asarray(table.truth, dtype=np.int8).tobytes()).hexdigest()
            cell = f"{brain}/{kind}"
            report["cells"][cell] = metrics
            if state is not None:
                state[cell] = {'scores': scores, 'chosen': chosen}
            if examples:
                metrics.update(describe(table, scores, chosen, generation, cell,
                                        (parent_state or {}).get(cell)))
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
        for key in ("pool_sha256", "labels_sha256", "pool_size", "requested_k", "effective_k", "positives"):
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
