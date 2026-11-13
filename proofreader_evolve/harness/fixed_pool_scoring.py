"""Fixed-budget ranking benchmark. Scorers never receive labels or candidate IDs.

Labels are three-valued (harness/label_availability.py): 1 = error, 0 = GT-confirmed
no error, NaN = not judgeable by GT. Scorers score every row; ranking, Precision@K,
recall and the bootstrap use GT-labeled rows only, so NaN rows never count as hits
or false positives.

Since 2026-10-12 (v3) each pool's K is resolved from its own GT positives through
``Budgets``: ``K = min(cap, ceil(positive_ratio * positives))``, so every pool that
counts has the same attainable ceiling (1 / positive_ratio when the ratio binds), and
pools with fewer than ``min_positives`` GT positives are scored and reported but
excluded from the promotion mean, the bootstrap and kind headroom (the CLI floor
``DEFAULT_MIN_POSITIVES`` is 10 since 2026-11-08, 20 before). Both inputs are
label counts, so every K is fixed for a run. Since v6 the objective is independent
for each kind across its counted brains. Selection, ablation and validation use
the same aggregation; each generation promotes only its target kind.

Since 2026-10-26 (v4) the ratio is per kind and a kind may have no cap. The defaults
are merge 2.0 under a cap of 300 (unchanged) and split 1.0 with no cap, so a split pool
ranks exactly its GT positives (R-precision: precision equals recall at that depth).
The fixed split cap of 1000 had saturated: the frozen detector alone reached
Precision@1000 of 0.92 to 0.97 on the validation pools while the labeled ranking still
held 1,759 to 2,783 positives, and precision at K = positives was 0.83 to 0.88.

Since 2026-10-27 (v5) the default rule is uniform: no kind has a cap and both ratios are
1.0, so every pool of either kind, in TRAIN, selection and validation alike, ranks exactly
its GT positives (R-precision). The merge cap of 300 bound on one pool only, the selection
brain 802449 (1,038 merge positives, so K = 300 measured the head of the ranking at recall
0.13), while the validation merge pools (41 to 144 positives, K = 2 x positives) measured a
depth where recall was 0.24 to 0.40. Over the 20 validated merge candidates of runs
`precision_20261026_210812_yqeacq5s` and `precision_20261027_000457_5d0_pbkb`, selection
Precision@300 ranked the candidates against their validation merge means with Spearman
0.67; Precision@K at K = positives did so with 0.88 (`notebooks/k_rule_check/`).
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

# CLI default of ``Budgets.min_positives`` (``--min-positives`` of the driver and preflight). 10 since
# 2026-11-08: under positive-weighted aggregation a pool near the floor carries a few percent of its
# kind's weight, so the floor only has to drop pools whose rank statistics are degenerate. It was 20
# from 2026-10-12, chosen under equal-brain aggregation, where merge pools of 9 to 17 positives held
# a third of the merge weight. The library default below stays 0 (every pool counts).
DEFAULT_MIN_POSITIVES = 10
from .rank_statistics import (DEFAULT_PAUC_FPR, OBJECTIVE_LABELS, aggregate_families, aggregate_statistics,
                              cell_statistics, family_statistics, cell_value, check_objective, check_pauc_fpr, prefix_hits, ranking,
                              auroc as weighted_auroc, average_precision as weighted_average_precision,
                              pauc as weighted_pauc)


# v7 (2026-11-02): the objective is a configurable rank statistic (``Budgets.objective``: precision at K,
# AUROC, partial AUROC or average precision over the same GT-labeled rows), every cell records all of
# them (the optional point-estimate head guard of v7 was removed on 2026-11-12; the guard rail of
# ``promotion_gate`` protects the head). Selection, ablation, the bootstrap and promotion all
# read the objective; per_kind_precision stays in every report for comparison with earlier runs.
SCORING_VERSION = "native-rank-objective-v7-configurable-statistic"
AGGREGATIONS = ("equal-brain", "positive-weighted")
AGGREGATION_LABELS = {"equal-brain": "per-kind-equal-brain-precision",
                      "positive-weighted": "per-kind-positive-weighted-precision"}


def aggregation_mode(value):
    """The aggregation mode for a mode name or a report's aggregation label."""
    if value is None:
        return "equal-brain"
    if value in AGGREGATIONS:
        return value
    for mode, label in AGGREGATION_LABELS.items():
        if value == label:
            return mode
    raise ValueError(f"Unknown aggregation {value!r}; use one of {AGGREGATIONS}")
VALIDATION_GATE_VERSION = "target-kind-validation-objective-v8"
# The point-estimate gate. The gate with intervals is ``promotion_gate`` (mode ``rule``).
GATE_MODES = ("margin",)


class Budgets(Mapping):
    """Per-kind Top-K caps plus the rule resolving each pool's K from its GT positives.

    ``budgets[kind]`` is the cap (the historical fixed K), or ``None`` for a kind without a
    cap. ``k_for(kind, truth)`` is the K ranked on one pool:
    ``ceil(positive_ratio[kind] * positives)``, limited to the cap when the kind has one; a
    zero ratio keeps the cap (fixed K), so a kind without a cap needs a positive ratio.
    ``positive_ratio`` is one number for every kind or a ``{kind: ratio}`` mapping naming
    exactly the budgeted kinds (defaults since 2026-10-27: ratio 1.0 and no cap for both
    kinds, so every pool ranks exactly its GT positives; from 2026-10-26 merge was 2.0 under
    a cap of 300 and split 1.0 without a cap). ``in_mean(truth)`` is
    False for pools with fewer than ``min_positives`` GT positives; those pools are scored
    and reported but never enter the promotion mean, the bootstrap or kind headroom.
    Everything depends on label counts only, so every pool's K is fixed for a run. A plain
    ``{kind: K}`` mapping coerces to fixed-K budgets that count every pool (the pre-v3
    behaviour).

    ``objective`` (v7) names the rank statistic that selection, ablation, the bootstrap and
    promotion compare: ``precision_at_k`` (the rule of every run before 2026-11-02),
    ``auroc``, ``pauc`` (partial AUROC up to a false-positive rate of ``pauc_fpr``) or
    ``average_precision``. The head of the ranking is protected by the guard rail of
    ``promotion_gate`` on the validation pools; the point-estimate head guard of v7 was removed
    on 2026-11-12.
    """

    def __init__(self, caps, positive_ratio=0., min_positives=0, aggregation="equal-brain", *,
                 objective="precision_at_k", pauc_fpr=DEFAULT_PAUC_FPR):
        caps = dict(caps)
        for kind, cap in caps.items():
            if cap is not None and (type(cap) is not int or cap < 1):
                raise ValueError(f"Top-K cap for {kind} must be a positive integer or None (no cap)")
        if isinstance(positive_ratio, Mapping):
            ratios = dict(positive_ratio)
            if set(ratios) != set(caps):
                raise ValueError("positive_ratio must name exactly the budgeted kinds: "
                                 f"{sorted(caps)} vs {sorted(ratios)}")
        else:
            ratios = {kind: positive_ratio for kind in caps}
        for kind, ratio in ratios.items():
            if (isinstance(ratio, bool) or not isinstance(ratio, (int, float))
                    or not math.isfinite(ratio) or ratio < 0):
                raise ValueError(f"positive_ratio for {kind} must be a finite nonnegative number "
                                 "(0 keeps the fixed cap)")
            if caps[kind] is None and ratio <= 0:
                raise ValueError(f"{kind} has no Top-K cap, so its positive_ratio must be > 0")
        if type(min_positives) is not int or min_positives < 0:
            raise ValueError("min_positives must be a nonnegative integer (0 counts every pool)")
        if aggregation not in AGGREGATIONS:
            raise ValueError(f"aggregation must be one of {AGGREGATIONS}, got {aggregation!r}")
        self.caps, self.min_positives, self.aggregation = caps, min_positives, aggregation
        self.positive_ratio = {kind: float(ratios[kind]) for kind in caps}
        self.objective = check_objective(objective)
        self.pauc_fpr = check_pauc_fpr(pauc_fpr)

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
        cap, ratio = self.caps[kind], self.positive_ratio[kind]
        if ratio <= 0:
            return cap
        k = math.ceil(ratio * self.positives(truth))
        if cap is not None:
            k = min(cap, k)
        return max(1, k)

    def kind_budget(self, kind, resolved_ks=()):
        """One literal K for a kind (agent-facing ``budgets``, image ``selection_k``): the cap
        when the kind has one, otherwise the largest of the given resolved pool Ks; None when
        neither exists."""
        cap = self.caps[kind]
        if cap is not None:
            return cap
        ks = [int(k) for k in resolved_ks]
        return max(ks) if ks else None

    def kind_budgets(self, cells):
        """``kind_budget`` for every budgeted kind from a report's cells (``requested_k``)."""
        return {kind: self.kind_budget(kind, (cell["requested_k"] for name, cell in cells.items()
                                              if name.rsplit("/", 1)[-1] == kind))
                for kind in self.caps}

    def in_mean(self, truth):
        return self.positives(truth) >= self.min_positives

    def describe(self):
        scaled = any(ratio > 0 for ratio in self.positive_ratio.values())
        pooled = self.aggregation == "positive-weighted"
        return {"caps": dict(self.caps), "positive_ratio": dict(self.positive_ratio),
                "min_positives": self.min_positives,
                "aggregation": self.aggregation, "aggregation_label": AGGREGATION_LABELS[self.aggregation],
                "k_rule": ("K per pool = ceil(positive_ratio[kind] x GT positives), limited to caps[kind] "
                           "when the kind has a cap; a zero ratio keeps the cap (fixed K)"
                           if scaled else "K per pool = cap (fixed)"),
                "precision_rule": ((f"pools with fewer than {self.min_positives} GT positives are scored and "
                                    "reported but excluded from the objective"
                                    if self.min_positives > 0 else "every pool counts toward the objective")
                                   + ("; separately for each kind: sum(TP) / sum(effective K) across counted pools"
                                      if pooled else "; separately for each kind: equal-weight precision across counted pools")),
                "objective": self.objective, "objective_label": OBJECTIVE_LABELS[self.objective],
                "pauc_fpr": self.pauc_fpr,
                "objective_rule": self.objective_rule()}

    def objective_rule(self):
        pooled = self.aggregation == "positive-weighted"
        weighting = ("weighted by its GT positives" if pooled else "with equal weight per counted pool")
        rules = {"precision_at_k": "the precision_rule above (precision at K is the objective)",
                 "auroc": f"separately for each kind: AUROC of each counted pool over its GT-labeled rows "
                          f"(probability that a positive outranks a negative, ties one half), {weighting}",
                 "pauc": f"separately for each kind: partial AUROC of each counted pool (mean true-positive rate "
                         f"while the false-positive rate is at most {self.pauc_fpr:g}, ties one half), {weighting}",
                 "average_precision": f"separately for each kind: tie-aware average precision of each counted "
                                      f"pool over its GT-labeled rows, {weighting}"}
        return rules[self.objective]

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
    return VALIDATION_GATE_VERSION


def aggregate_precision(cells, aggregation="equal-brain"):
    """Independent per-kind precision over the pools that count.

    ``aggregation`` (a mode or a report label) sets the per-kind rule: ``equal-brain``
    averages the counted pools' precision with equal weight (the rule of every run before
    2026-10-27); ``positive-weighted`` pools hits divided by effective K summed over pools
    (equivalent to GT-positive weighting and pooled recall when K = positives), so a pool
    with 41 positives no longer carries the weight of one with 144.
    Pools flagged ``in_mean: false`` (too few GT positives) are reported but skipped.
    Every evaluated brain must still carry the same kinds, so the grid stays complete.
    """
    mode = aggregation_mode(aggregation)
    grouped, by_kind, counts, excluded = {}, {}, {}, []
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
        if mode == "positive-weighted":
            tp, k = int(cell["tp"]), int(cell["effective_k"])
            if tp < 0 or k < 0 or tp > k:
                raise ValueError(f"Invalid hit count on {name}")
            counts.setdefault(kind, []).append((tp, k))
    if not grouped:
        raise ValueError("No candidate pools supplied")
    if len({tuple(sorted(values)) for values in grouped.values()}) != 1:
        raise ValueError("Every evaluated brain must have the same candidate kinds")
    missing = [kind for kind in sorted(next(iter(grouped.values()))) if kind not in by_kind]
    if missing:
        raise ValueError(f"No {missing[0]} pool counts toward the mean: every {missing[0]} pool is below "
                         "the GT-positive floor")
    if mode == "positive-weighted":
        per_kind = {}
        for kind in sorted(by_kind):
            pooled_k = sum(k for _, k in counts[kind])
            per_kind[kind] = sum(tp for tp, _ in counts[kind]) / pooled_k if pooled_k else 0.
    else:
        per_kind = {kind: math.fsum(values.values()) / len(values) for kind, values in sorted(by_kind.items())}
    return {"aggregation": AGGREGATION_LABELS[mode], "per_kind_precision": per_kind,
            "excluded_cells": sorted(excluded)}


def precision_for_kind(report, kind):
    """Precision at K of this kind, recomputed from its cells (the objective before v7; a
    reported statistic since)."""
    cells = {name: cell for name, cell in report['cells'].items() if name.rsplit('/', 1)[-1] == kind}
    if not cells:
        raise ValueError(f"No candidate pools for target kind {kind!r}")
    return aggregate_precision(cells, report.get('aggregation'))['per_kind_precision'][kind]


def report_objective(report):
    """The objective a report was measured under; reports before v7 carry none and are precision at K."""
    return check_objective(report.get('objective') or 'precision_at_k')


def aggregate_objective(cells, aggregation="equal-brain", objective="precision_at_k"):
    """Per-kind objective and every rank statistic over the pools that count.

    Extends ``aggregate_precision``: ``per_kind_objective`` is the configured statistic
    (identical to ``per_kind_precision`` for ``precision_at_k``), ``per_kind_statistics`` holds
    all of them under the same aggregation rule.
    """
    result = aggregate_precision(cells, aggregation)
    objective = check_objective(objective)
    by_kind = {}
    for name, cell in cells.items():
        by_kind.setdefault(name.rsplit('/', 1)[-1], {})[name] = cell
    statistics = {kind: aggregate_statistics(group, result['aggregation']) for kind, group in sorted(by_kind.items())}
    per_kind = {}
    for kind, summary in statistics.items():
        value = result['per_kind_precision'][kind] if objective == 'precision_at_k' else summary[objective]
        if value is None:
            raise ValueError(f"No {OBJECTIVE_LABELS[objective]} on the counted {kind} pools")
        per_kind[kind] = float(value)
    # Reporting only: within-family figures for kinds whose candidates come from several rules.
    families = {kind: summary for kind, summary in ((kind, aggregate_families(group))
                                                    for kind, group in sorted(by_kind.items())) if summary}
    if families:
        result['per_kind_by_family'] = families
    result.update(objective=objective, objective_label=OBJECTIVE_LABELS[objective], per_kind_objective=per_kind,
                  per_kind_statistics={kind: {key: value for key, value in summary.items() if key not in ('note', 'version')}
                                       for kind, summary in statistics.items()})
    return result


def _kind_cells(report, kind):
    cells = {name: cell for name, cell in report['cells'].items() if name.rsplit('/', 1)[-1] == kind}
    if not cells:
        raise ValueError(f"No candidate pools for target kind {kind!r}")
    return cells


def objective_for_kind(report, kind):
    """The shared selection/promotion objective of this kind, recomputed from its cells under the
    objective the report was measured with."""
    objective = report_objective(report)
    return aggregate_objective(_kind_cells(report, kind), report.get('aggregation'), objective)['per_kind_objective'][kind]


def statistic_for_kind(report, kind, statistic):
    """One rank statistic of this kind over its counted cells (None when no cell carries it)."""
    if statistic == 'precision_at_k':
        statistic = 'precision'
    return aggregate_statistics(_kind_cells(report, kind), report.get('aggregation'))[statistic]


def gate_details(parent, candidate, kind):
    """Objective and precision numbers of a parent/candidate pair for the ledger."""
    objective = report_objective(parent)
    details = {'objective': objective, 'objective_label': OBJECTIVE_LABELS[objective],
               'parent_objective': objective_for_kind(parent, kind), 'candidate_objective': None,
               'parent_precision': precision_for_kind(parent, kind), 'candidate_precision': None}
    if candidate is not None:
        details.update(candidate_objective=objective_for_kind(candidate, kind),
                       candidate_precision=precision_for_kind(candidate, kind))
    return details


def labeled_order(scores, truth, keys):
    """GT-labeled rows in ranking order; candidate identity resolves ties reproducibly."""
    rows = labeled_rows(truth)
    return rows[np.lexsort((np.asarray(keys, dtype=str)[rows], -np.asarray(scores, dtype=float)[rows]))]


def rank_metrics(scores, truth, keys, k, *, in_mean=True, pauc_fpr=DEFAULT_PAUC_FPR):
    """Precision@K over GT-labeled rows, plus every other rank statistic of the same ranking.

    The Top-K is taken among rows labeled 1 or 0. ``k`` is this pool's resolved K (see
    ``Budgets.k_for``); ``in_mean`` records whether the pool counts toward the mean
    (``Budgets.in_mean``); ``pauc_fpr`` bounds the partial AUROC (``rank_statistics``).
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
    metrics = {"requested_k": k, "effective_k": len(chosen), "pool_size": len(scores),
               "n_labeled": len(order), "n_unlabeled": len(scores) - len(order),
               "tp": tp, "fp": len(chosen) - tp, "precision": tp / len(chosen),
               "recall": tp / positives if positives else None,
               "positives": positives, "in_mean": bool(in_mean)}
    metrics.update(cell_statistics(scores, truth, keys, k, pauc_fpr=pauc_fpr))
    return metrics, chosen


def evaluate(source, tables_by_brain, budgets, timeout=120, examples=False, *,
             state=None, parent_state=None, generation=0, artifact_store=None, memory_mb=DEFAULT_MEMORY_MB):
    """Score features in a restricted worker; labels and ranking stay here."""
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Scorer timeout must be positive and finite")
    budgets = Budgets.coerce(budgets)
    report = {"version": SCORING_VERSION, "budgets": budgets.describe(),
              "objective": budgets.objective,
              "policy_sha256": hashlib.sha256(source.encode()).hexdigest(), "cells": {}}
    sources = components(source)
    for brain, banks in tables_by_brain.items():
        for kind, table in banks.tables.items():
            features = augmented_features(sources[kind], table, timeout)
            scores = (score(sources[kind], features, kind, timeout, memory_mb=memory_mb, artifact_store=artifact_store)
                      if artifact_store is not None else score(sources[kind], features, kind, timeout, memory_mb=memory_mb))
            k = budgets.k_for(kind, table.truth)
            metrics, chosen = rank_metrics(scores, table.truth, table.keys, k, in_mean=budgets.in_mean(table.truth),
                                           pauc_fpr=budgets.pauc_fpr)
            by_family = family_statistics(scores, table.truth, table.keys, getattr(table, 'candidates', ()), chosen)
            if by_family:
                metrics['by_family'] = by_family
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
    report.update(aggregate_objective(report["cells"], budgets.aggregation, budgets.objective))
    return report


def acceptance(parent, candidate, margin=0, *, target_kind, require_no_cell_regression=True):
    """Compare the target kind's objective at identical K/pools; optionally veto its cell
    regressions."""
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
    mode = aggregation_mode(parent.get("aggregation"))
    if aggregation_mode(candidate.get("aggregation")) != mode:
        raise ValueError("Cannot compare evaluations with different aggregations")
    objective = report_objective(parent)
    if report_objective(candidate) != objective:
        raise ValueError("Cannot compare evaluations with different objectives")
    label = OBJECTIVE_LABELS[objective]
    before_value = objective_for_kind(parent, target_kind)
    after_value = objective_for_kind(candidate, target_kind)
    if require_no_cell_regression:
        for cell, before in parent["cells"].items():
            if cell.rsplit('/', 1)[-1] == target_kind and before.get('in_mean', True):
                was, now = cell_value(before, objective), cell_value(candidate["cells"][cell], objective)
                if was is not None and now is not None and now < was:
                    return False, f"{label} regressed on {cell}"
    gain = after_value - before_value
    details = f"{before_value:.6f} -> {after_value:.6f}; delta={gain:+.6f}, margin={margin:g}"
    passed = gain > max(margin, 1e-12)
    reason = f"{target_kind} {label} improved" if passed else f"No sufficient {target_kind} {label} gain"
    return passed, f"{reason}: {details}"


BOOTSTRAP_VERSION = "paired-poisson-bootstrap-v8-fragment-groups"
RESAMPLING_UNIT = ("fragment segment (merge: the candidate's segment; split: the segment of the pair's "
                   "anchor-side fragment); one row per group when a table carries no candidate rows")


def resampling_groups(kind, table):
    """Group index per row for the bootstrap (2026-11-12).

    Candidates on one fragment are not independent draws: they share the fragment's geometry,
    image and tracing, and a GT neuron covers many of them. The bootstrap therefore resamples
    fragments, not rows. Merge: the candidate's segment. Split: the segment of the anchor-side
    fragment of the pair (``anchor_side``), the fragment whose tip produced the candidate. A
    table without candidate rows (synthetic fixtures) gets one group per row, which is row
    resampling.
    """
    candidates = getattr(table, "candidates", None)
    size = len(table.truth)
    if not candidates:
        return np.arange(size, dtype=np.int64)
    if len(candidates) != size:
        raise ValueError("Candidate rows do not align with the labels")
    if kind == "merge":
        keys = [int(row["segment_id"]) for row in candidates]
    elif kind == "split":
        keys = [int(row["segment_id_b"] if row.get("anchor_side") == "b" else row["segment_id_a"]) for row in candidates]
    else:
        raise ValueError(f"Unknown candidate kind {kind!r}")
    return np.unique(np.asarray(keys, dtype=np.int64), return_inverse=True)[1].astype(np.int64)


def labeled_groups(kind, table, rows):
    """(group index of each labeled row, number of groups) among the labeled ``rows``."""
    groups = resampling_groups(kind, table)[rows]
    present, inverse = np.unique(groups, return_inverse=True)
    return inverse.astype(np.int64), int(len(present))


def paired_bootstrap(parent_state, candidate_state, tables_by_brain, budgets, *, target_kind, draws=1000, seed=0, alpha=.05):
    """Paired resampling of the GT-labeled candidates for the target-kind delta of the objective.

    Both scorers are re-measured on the same Poisson(1)-weighted multiset per cell (one weight
    per fragment group of ``resampling_groups``, shared by its labeled rows) with the weighted
    statistics of ``rank_statistics``, so the interval reflects the sampling noise of the
    fragments, not scorer nondeterminism: precision at the cell's own K, or
    AUROC, partial AUROC and average precision with the cell's score-tie groups. The two-sided
    (1 - alpha) interval is ``delta_ci``; the 95% interval is always
    recorded as well. Cells below the GT-positive floor get a per-cell interval but never enter
    the target-kind delta, matching `aggregate_objective`; positive-weighted aggregation of a
    rank statistic uses each cell's resampled positive weight.
    """
    if type(draws) is not int or draws < 1:
        raise ValueError("Bootstrap draws must be a positive integer")
    if not (isinstance(alpha, float) and math.isfinite(alpha) and 0. < alpha < 1.):
        raise ValueError("Bootstrap alpha must be a float in (0, 1)")
    budgets = Budgets.coerce(budgets)
    pooled = budgets.aggregation == "positive-weighted"
    objective = budgets.objective
    rng = np.random.default_rng(seed)
    cells = {}
    for brain, banks in tables_by_brain.items():
        for kind, table in banks.tables.items():
            if kind != target_kind:
                continue
            cell = f"{brain}/{kind}"
            truth = np.asarray(table.truth, dtype=float)
            before = np.asarray(parent_state[cell]["scores"], dtype=float)
            after = np.asarray(candidate_state[cell]["scores"], dtype=float)
            if before.shape != truth.shape or after.shape != truth.shape:
                raise ValueError(f"Bootstrap scores do not align with {cell}")
            # Labeled rows only: weights are drawn per fragment group of the labeled rows, and
            # unlabeled rows never enter either ranking, matching rank_metrics.
            rows = labeled_rows(truth)
            position = np.full(len(truth), -1, dtype=np.int64)
            position[rows] = np.arange(len(rows))
            arms = []
            for scores in (before, after):
                order, labels, starts = ranking(scores, truth, table.keys)
                arms.append((position[order], labels.astype(float), starts))
            groups, n_groups = labeled_groups(kind, table, rows)
            cells[cell] = {"arms": arms, "size": len(rows), "k": budgets.k_for(kind, table.truth),
                           "counted": budgets.in_mean(table.truth), "groups": groups, "n_groups": n_groups}
    counted = {cell for cell, info in cells.items() if info["counted"]}
    if not counted:
        raise ValueError("No candidate pool counts toward the bootstrap mean")
    statistic = {"auroc": weighted_auroc, "average_precision": weighted_average_precision,
                 "pauc": lambda labels, weights, starts: weighted_pauc(labels, weights, starts, budgets.pauc_fpr)}

    def measure(arm, counts, k):
        """(numerator, denominator) of one arm on one resample: hits and depth for precision,
        the statistic and the resampled positive weight otherwise (None when undefined)."""
        positions, labels, starts = arm
        weights = counts[positions].astype(float)
        if objective == "precision_at_k":
            return prefix_hits(labels, weights, k)
        return statistic[objective](labels, weights, starts), float((labels * weights).sum())

    def combine(values):
        """Aggregate (numerator, denominator) pairs of the counted cells for one arm."""
        if objective == "precision_at_k":
            if pooled:
                depth = sum(d for _, d in values)
                return sum(n for n, _ in values) / depth if depth else 0.
            return math.fsum((n / d if d else 0.) for n, d in values) / len(values)
        defined = [(v, w) for v, w in values if v is not None]
        if not defined:
            return None
        if pooled:
            total = sum(w for _, w in defined)
            return sum(v * w for v, w in defined) / total if total else None
        return math.fsum(v for v, _ in defined) / len(defined)

    deltas = np.full(draws, np.nan)
    per_cell = {cell: np.full(draws, np.nan) for cell in cells}
    for draw in range(draws):
        sampled = {}
        for cell, info in cells.items():
            counts = rng.poisson(1., info["n_groups"]).astype(np.int64)[info["groups"]]
            before_pair, after_pair = (measure(arm, counts, info["k"]) for arm in info["arms"])
            if objective == "precision_at_k":
                value_before = before_pair[0] / before_pair[1] if before_pair[1] else 0.
                value_after = after_pair[0] / after_pair[1] if after_pair[1] else 0.
            else:
                value_before, value_after = before_pair[0], after_pair[0]
            if value_before is not None and value_after is not None:
                per_cell[cell][draw] = value_after - value_before
            sampled[cell] = (before_pair, after_pair)
        pairs = [sampled[cell] for cell in sorted(counted)]
        before_total = combine([pair[0] for pair in pairs])
        after_total = combine([pair[1] for pair in pairs])
        if before_total is not None and after_total is not None:
            deltas[draw] = after_total - before_total
    if not np.isfinite(deltas).any():
        raise ValueError(f"The {OBJECTIVE_LABELS[objective]} delta is undefined on every resample")
    bounds = [100. * alpha / 2., 100. * (1. - alpha / 2.)]
    low, high = np.nanpercentile(deltas, bounds)
    defined_deltas = deltas[np.isfinite(deltas)]

    def interval(values, percentiles):
        return ([float(v) for v in np.nanpercentile(values, percentiles)] if np.isfinite(values).any()
                else [None, None])
    return {"version": BOOTSTRAP_VERSION, "target_kind": target_kind, "objective": objective,
            "objective_label": OBJECTIVE_LABELS[objective], "draws": draws, "seed": seed, "alpha": alpha,
            "aggregation": AGGREGATION_LABELS[budgets.aggregation], "excluded_cells": sorted(set(cells) - counted),
            "undefined_draws": int(np.count_nonzero(~np.isfinite(deltas))),
            "delta_ci": [float(low), float(high)],
            "delta_ci95": interval(deltas, [2.5, 97.5]),
            "bootstrap_mean_delta": float(defined_deltas.mean()),
            "p_delta_le_0": float(np.mean(defined_deltas <= 0)), "p_delta_ge_0": float(np.mean(defined_deltas >= 0)),
            "cells": {cell: {"delta_ci": interval(values, bounds), "delta_ci95": interval(values, [2.5, 97.5])}
                      for cell, values in per_cell.items()},
            "resampling_unit": RESAMPLING_UNIT,
            "groups": {cell: info["n_groups"] for cell, info in cells.items()},
            "scope": f"Fragment-resampling uncertainty of the paired {OBJECTIVE_LABELS[objective]} delta over the fixed "
                     "pools."}
