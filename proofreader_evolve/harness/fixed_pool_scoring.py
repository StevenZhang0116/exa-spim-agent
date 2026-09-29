"""Fixed-budget ranking benchmark. Scorers never receive labels or candidate IDs."""

import hashlib
import math
from types import ModuleType

import numpy as np

from .policy_runtime import policy_time_budget


SCORING_VERSION = "native-precision-at-k-v1"


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


def evaluate(source, tables_by_brain, budgets, timeout=120, examples=False):
    """Fresh module each evaluation; copied feature-only inputs; deterministic scores.

    This is an API/data boundary, NOT an OS sandbox for hostile Python. Generated
    code must be trusted before execution. Timeouts use the existing harness.
    """
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Scorer timeout must be positive and finite")
    report = {"version": SCORING_VERSION,
              "policy_sha256": hashlib.sha256(source.encode()).hexdigest(), "cells": {}}
    for brain, banks in tables_by_brain.items():
        for kind, table in banks.tables.items():
            with policy_time_budget(timeout):
                module = ModuleType("evolved_fixed_pool_scorer")
                exec(compile(source, "<fixed-pool-scorer>", "exec"), module.__dict__)
                if hasattr(module, "ENUM_PARAMS") or hasattr(module, "propose_edits"):
                    raise ValueError("Fixed-pool policies define score_candidates, not ENUM_PARAMS/propose_edits")
                scorer = getattr(module, "score_candidates", None)
                if not callable(scorer):
                    raise ValueError("Policy must define score_candidates(features, ctx)")
                scores = np.asarray(scorer(table.features.copy(deep=True), {"kind": kind}), dtype=float)
                repeat = np.asarray(scorer(table.features.copy(deep=True), {"kind": kind}), dtype=float)
                if not np.array_equal(scores, repeat):
                    raise ValueError("Scorer must be deterministic")
                metrics, chosen = rank_metrics(scores, table.truth, table.keys, budgets[kind])
                metrics["pool_sha256"] = table.meta["pool_sha256"]
                metrics["labels_sha256"] = hashlib.sha256(np.asarray(table.truth, dtype=np.int8).tobytes()).hexdigest()
                report["cells"][f"{brain}/{kind}"] = metrics
                if examples:
                    # TRAIN only: bounded, anonymous feature/label examples. The
                    # evaluator owns labels; no candidate IDs or GT geometry go in.
                    missed = np.flatnonzero((table.truth == 1) & ~np.isin(np.arange(len(scores)), chosen))
                    indices = list(chosen[table.truth[chosen] == 0][:8]) + list(missed[:8])
                    metrics["training_examples"] = [
                        {"label": int(table.truth[i]), "score": float(scores[i]),
                         "features": {name: (float(value) if math.isfinite(float(value)) else None)
                                      for name, value in table.features.iloc[i].items()}}
                        for i in indices]
    if not report["cells"]:
        raise ValueError("No candidate pools supplied")
    report["macro_precision"] = float(np.mean([c["precision"] for c in report["cells"].values()]))
    return report


def acceptance(parent, candidate, margin=0):
    """Strict macro gain at identical K/pools, with no brain/kind regression."""
    if not math.isfinite(margin) or margin < 0:
        raise ValueError("Precision margin must be finite and nonnegative")
    if parent["version"] != candidate["version"] or parent["cells"].keys() != candidate["cells"].keys():
        raise ValueError("Cannot compare different evaluation contracts")
    for cell, before in parent["cells"].items():
        after = candidate["cells"][cell]
        for key in ("pool_sha256", "labels_sha256", "pool_size", "requested_k", "effective_k", "positives"):
            if before[key] != after[key]:
                raise ValueError(f"Candidate changed fixed benchmark: {cell}/{key}")
        if after["precision"] < before["precision"]:
            return False, f"Precision regressed on {cell}"
    gain = candidate["macro_precision"] - parent["macro_precision"]
    return (True, "Precision improved at fixed K") if gain > max(margin, 1e-12) else (False, "No sufficient precision gain")
