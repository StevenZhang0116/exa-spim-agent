"""Rank statistics over GT-labeled rows: Precision@depth, average precision, AUROC, partial AUROC.

Every statistic is a function of one ranking (labels in ranking order, score-tie groups) and
nonnegative row weights, so the point estimate (unit weights) and the paired Poisson row
bootstrap (resampled weights) share one implementation. Which statistic is the objective of a
run is the ``Budgets.objective`` setting (``OBJECTIVES``); the others are recorded in every
cell and report as diagnostics. Since 2026-11-02 the driver's default objective is AUROC with a
head guard on the mean precision at ceil(K/2), K and 2K (see ``fixed_pool_scoring``): on pools
shaped like the validation merge cells a broad improvement of the ranking gives AUROC about
four times the paired signal of Precision@K, head-only changes the same signal, and
degradations two to three times (simulation in WORKFLOW_REVISION, 2026-11-02).

Ranking rule (``labeled_order``): score descending, candidate key ascending, GT-labeled rows
only. Precision at a depth credits the key-broken order; AUROC, partial AUROC and average
precision treat tied scores as one threshold (as sklearn does), so a score that ties many rows
is not credited for the row order inside its ties.
"""
import math

import numpy as np

from .label_availability import labeled_rows


STATISTICS_VERSION = 'rank-statistics-v3-by-family'
OBJECTIVES = ('precision_at_k', 'auroc', 'pauc', 'average_precision')
OBJECTIVE_LABELS = {'precision_at_k': 'Precision@K', 'auroc': 'AUROC', 'pauc': 'partial AUROC',
                    'average_precision': 'average precision'}
HEAD_STATISTICS = ('precision_depth_mean', 'precision_at_k')
DEFAULT_PAUC_FPR = .1
# Cell fields produced by cell_statistics (beside the official precision/tp/fp of the cell).
CELL_KEYS = ('half_k', 'half_k_tp', 'precision_at_half_k', 'double_k', 'double_k_tp',
             'precision_at_double_k', 'precision_depth_mean', 'average_precision', 'auroc', 'pauc', 'pauc_fpr')
STATISTIC_KEYS = ('precision', 'precision_at_half_k', 'precision_at_double_k', 'precision_depth_mean',
                  'average_precision', 'auroc', 'pauc')
NOTE = ('All statistics use the same GT-labeled rows and fold partition as precision at K: precision at '
        'ceil(K/2) and 2K (half_k and double_k are the labeled rows actually ranked), their mean with '
        'precision at K (precision_depth_mean), tie-aware average precision, AUROC and partial AUROC '
        '(mean true-positive rate while the false-positive rate is at most pauc_fpr). The objective named '
        'by the report is the one candidate ranking, ablation and promotion use; the others are diagnostic.')


def check_objective(objective):
    if objective not in OBJECTIVES:
        raise ValueError(f'Unknown objective {objective!r}; use one of {OBJECTIVES}')
    return objective


def check_pauc_fpr(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value <= 1:
        raise ValueError('pauc_fpr must be a number in (0, 1]')
    return float(value)


def fold_budgets(held_counts, k):
    """Largest-remainder split of K across folds, capped by each fold's held rows."""
    counts = [int(c) for c in held_counts]
    if any(c < 0 for c in counts) or type(k) is not int or k < 1:
        raise ValueError('Fold sizes must be nonnegative and K a positive integer')
    total = sum(counts)
    if total == 0:
        return [0] * len(counts)
    k = min(k, total)
    shares = [k * c / total for c in counts]
    budgets = [min(math.floor(s), c) for s, c in zip(shares, counts)]
    remaining = k - sum(budgets)
    order = sorted(range(len(counts)), key=lambda i: (-(shares[i] - math.floor(shares[i])), i))
    while remaining > 0:
        progressed = False
        for index in order:
            if remaining == 0:
                break
            if budgets[index] < counts[index]:
                budgets[index] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            break
    return budgets


# --- one ranking -----------------------------------------------------------------------------

def tie_starts(ordered_scores):
    """Start index of every score-tie group along a ranking (scores in ranking order)."""
    s = np.asarray(ordered_scores, dtype=float)
    if not len(s):
        return np.zeros(0, dtype=np.int64)
    return np.flatnonzero(np.r_[True, s[1:] != s[:-1]])


def prefix_hits(labels, weights, k):
    """Weighted hits and effective depth of the Top-K of a ranking (labels and weights in
    ranking order; a boundary row counts with the part of its weight that fits)."""
    labels, weights = np.asarray(labels), np.asarray(weights)
    cumulative = np.cumsum(weights)
    if not len(cumulative) or cumulative[-1] <= k:
        return float((labels * weights).sum()), float(cumulative[-1]) if len(cumulative) else 0.
    stop = int(np.searchsorted(cumulative, k))
    partial = k - (float(cumulative[stop - 1]) if stop else 0.)
    hits = float((labels[:stop] * weights[:stop]).sum()) + float(labels[stop]) * partial
    return hits, float(k)


def _group_sums(labels, weights, starts):
    labels, weights = np.asarray(labels, dtype=float), np.asarray(weights, dtype=float)
    positives = np.add.reduceat(labels * weights, starts) if len(starts) else np.zeros(0)
    negatives = np.add.reduceat((1. - labels) * weights, starts) if len(starts) else np.zeros(0)
    return positives, negatives


def auroc(labels, weights, starts):
    """Weighted, tie-aware AUROC: the probability that a positive outranks a negative, ties
    counting one half. None without a positive or without a negative of positive weight."""
    positives, negatives = _group_sums(labels, weights, starts)
    total_p, total_n = positives.sum(), negatives.sum()
    if total_p <= 0 or total_n <= 0:
        return None
    below = total_n - np.cumsum(negatives)
    return float((positives * below + .5 * positives * negatives).sum() / (total_p * total_n))


def average_precision(labels, weights, starts):
    """Weighted, tie-aware average precision (each tie group is one threshold)."""
    positives, negatives = _group_sums(labels, weights, starts)
    total_p = positives.sum()
    if total_p <= 0:
        return None
    cum_p, cum_n = np.cumsum(positives), np.cumsum(negatives)
    depth = cum_p + cum_n
    # Leading tie groups of zero weight (rows a bootstrap draw left out) are no threshold;
    # their positives are zero, so any finite precision there contributes nothing.
    precision = np.divide(cum_p, depth, out=np.zeros_like(cum_p), where=depth > 0)
    return float((positives * precision).sum() / total_p)


def pauc(labels, weights, starts, fpr_max=DEFAULT_PAUC_FPR):
    """Weighted, tie-aware partial AUROC: area under the ROC polyline for false-positive rates
    up to ``fpr_max``, divided by ``fpr_max`` (the mean true-positive rate over that range, in
    [0, 1]). Tie groups are ROC vertices, so ties contribute diagonal segments."""
    fpr_max = check_pauc_fpr(fpr_max)
    positives, negatives = _group_sums(labels, weights, starts)
    total_p, total_n = positives.sum(), negatives.sum()
    if total_p <= 0 or total_n <= 0:
        return None
    fpr = np.r_[0., np.cumsum(negatives) / total_n]
    tpr = np.r_[0., np.cumsum(positives) / total_p]
    inside = fpr <= fpr_max
    xs, ys = fpr[inside], tpr[inside]
    beyond = np.flatnonzero(~inside)
    if len(beyond):
        j = int(beyond[0])  # first vertex past fpr_max; interpolate on the segment (j - 1, j)
        x0, x1, y0, y1 = fpr[j - 1], fpr[j], tpr[j - 1], tpr[j]
        y = y0 + (y1 - y0) * (fpr_max - x0) / (x1 - x0) if x1 > x0 else y1
        xs, ys = np.r_[xs, fpr_max], np.r_[ys, y]
    return float(np.sum((xs[1:] - xs[:-1]) * (ys[1:] + ys[:-1]) / 2.) / fpr_max)


def ranking(scores, truth, keys, rows=None):
    """Labeled rows of one pool in ranking order, with their labels and tie-group starts.

    ``rows`` restricts the ranking to a subset of row indices (a fold); the order inside is
    score descending then candidate key ascending, the rule shared with Precision@K.
    """
    scores, truth = np.asarray(scores, dtype=float), np.asarray(truth, dtype=float)
    keys = np.asarray(keys, dtype=str)
    if rows is None:
        rows = labeled_rows(truth)
    else:
        rows = np.asarray(rows, dtype=int)
        rows = rows[np.isfinite(truth[rows])]
    order = rows[np.lexsort((keys[rows], -scores[rows]))]
    return order, truth[order].astype(np.int64), tie_starts(scores[order])


def ranking_statistics(labels, starts, k, *, weights=None, pauc_fpr=DEFAULT_PAUC_FPR):
    """Every statistic of one ranking at depth ``k`` (labels in ranking order)."""
    labels = np.asarray(labels, dtype=float)
    weights = np.ones(len(labels)) if weights is None else np.asarray(weights, dtype=float)
    out = {}
    for name, depth in (('half_k', max(1, math.ceil(k / 2))), ('k', k), ('double_k', 2 * k)):
        hits, effective = prefix_hits(labels, weights, depth)
        out[name] = (hits, effective)
    out['average_precision'] = average_precision(labels, weights, starts)
    out['auroc'] = auroc(labels, weights, starts)
    out['pauc'] = pauc(labels, weights, starts, pauc_fpr)
    return out


# --- one cell, optionally partitioned into folds ---------------------------------------------

def cell_statistics(scores, truth, keys, k, *, fold_ids=None, pauc_fpr=DEFAULT_PAUC_FPR):
    """Diagnostic statistics of one pool beside its official precision: depths ceil(K/2) and 2K
    under the fold rule (largest-remainder budgets per depth), their mean with precision at K,
    and the positive-weighted mean over folds of average precision, AUROC and partial AUROC.
    Fold scores are never pooled (fold models emit different scales)."""
    scores, truth = np.asarray(scores, dtype=float), np.asarray(truth, dtype=float)
    if type(k) is not int or k < 1:
        raise ValueError('K must be a positive integer')
    fold_ids = np.zeros(len(scores), dtype=np.int64) if fold_ids is None else np.asarray(fold_ids)
    labeled = np.zeros(len(truth), dtype=bool)
    labeled[labeled_rows(truth)] = True
    n_folds = int(fold_ids.max()) + 1 if len(fold_ids) else 0
    folds = [np.flatnonzero(labeled & (fold_ids == fold)) for fold in range(n_folds)]
    held = [len(rows) for rows in folds]
    rankings = [ranking(scores, truth, keys, rows) for rows in folds]
    out = {}
    precisions = []
    for name, depth in (('half_k', max(1, math.ceil(k / 2))), ('k', k), ('double_k', 2 * k)):
        budgets = fold_budgets(held, depth)
        tp = effective = 0
        for (order, labels, _), budget in zip(rankings, budgets):
            tp += int(labels[:budget].sum())
            effective += min(budget, len(labels))
        precision = tp / effective if effective else None
        precisions.append(precision)
        if name != 'k':
            out[name], out[f'{name}_tp'], out[f'precision_at_{name}'] = effective, tp, precision
    out['precision_depth_mean'] = (math.fsum(precisions) / 3 if all(p is not None for p in precisions) else None)
    for statistic, function in (('average_precision', average_precision), ('auroc', auroc),
                                ('pauc', lambda l, w, s: pauc(l, w, s, pauc_fpr))):
        weighted = positives = 0.
        for _, labels, starts in rankings:
            value = function(labels, np.ones(len(labels)), starts) if len(labels) else None
            if value is not None:
                count = float(labels.sum())
                weighted += count * value
                positives += count
        out[statistic] = weighted / positives if positives else None
    out['pauc_fpr'] = check_pauc_fpr(pauc_fpr)
    return out


# --- candidate families (reporting only) -------------------------------------------------------

FAMILY_FIELD = 'candidate_family'
FAMILY_NOTE = ('by_family splits a pool by the rule that produced each candidate (merge pools: junction and '
               'tip_to_shaft). Per family: labeled rows and positives, positives_in_pool_top_k (how many of its '
               'positives the whole-pool Top-K holds), and, ranking the family alone, top_k_hits of k (k = its '
               'positives), average_precision and auroc. A pooled statistic can be high only because one family '
               'ranks above the other; the within-family figures show whether errors are told from non-errors '
               'inside a family. Reporting only: the objective and the gate use the whole pool.')


def family_statistics(scores, truth, keys, candidates, chosen=None, fold_ids=None):
    """Per-family statistics of one pool, or None when the candidates carry no family or the
    labeled rows span several folds (fold scores are never pooled). ``chosen`` is the pool's
    Top-K (row indices). Reporting only: nothing here enters the objective or a gate."""
    if not len(candidates) or FAMILY_FIELD not in candidates[0]:
        return None
    scores, truth = np.asarray(scores, dtype=float), np.asarray(truth, dtype=float)
    labeled = np.isfinite(truth)
    if fold_ids is not None and len(np.unique(np.asarray(fold_ids)[labeled])) > 1:
        return None
    families = np.array([candidate[FAMILY_FIELD] for candidate in candidates])
    in_top = np.zeros(len(truth), dtype=bool)
    if chosen is not None:
        in_top[np.asarray(chosen, dtype=int)] = True
    out = {}
    for name in sorted(set(families.tolist())):
        member = families == name
        _, labels, starts = ranking(scores, truth, keys, np.flatnonzero(member))
        labels = labels.astype(float)
        positives = int(labels.sum())
        entry = {'labeled': int(len(labels)), 'positives': positives,
                 'positives_in_pool_top_k': None if chosen is None else int((member & in_top & (truth == 1)).sum()),
                 'k': None, 'top_k_hits': None, 'average_precision': None, 'auroc': None}
        if 0 < positives < len(labels):
            weights = np.ones(len(labels))
            hits, depth = prefix_hits(labels, weights, positives)
            entry.update(k=int(depth), top_k_hits=int(round(hits)),
                         average_precision=average_precision(labels, weights, starts),
                         auroc=auroc(labels, weights, starts))
        out[str(name)] = entry
    return out


def aggregate_families(cells):
    """Family statistics over the counted cells of one kind: counts and hits are summed, the rank
    statistics weighted by the family's positives. None when no cell carries ``by_family``."""
    totals = {}
    for cell in cells.values():
        if not cell.get('in_mean', True) or not cell.get('by_family'):
            continue
        for name, entry in cell['by_family'].items():
            total = totals.setdefault(name, {'pools': 0, 'labeled': 0, 'positives': 0, 'positives_in_pool_top_k': 0,
                                             'k': 0, 'top_k_hits': 0, 'average_precision': [], 'auroc': []})
            total['pools'] += 1
            for key in ('labeled', 'positives', 'positives_in_pool_top_k', 'k', 'top_k_hits'):
                total[key] += int(entry[key] or 0)
            for key in ('average_precision', 'auroc'):
                if entry[key] is not None:
                    total[key].append((float(entry[key]), entry['positives']))
    for total in totals.values():
        total['precision_at_k'] = total['top_k_hits'] / total['k'] if total['k'] else None
        for key in ('average_precision', 'auroc'):
            weight = sum(count for _, count in total[key])
            total[key] = sum(value * count for value, count in total[key]) / weight if weight else None
    return totals or None


def cell_value(cell, statistic):
    """One statistic of a scored cell (``precision`` = the official precision at K)."""
    if statistic == 'precision_at_k':
        statistic = 'precision'
    if statistic not in STATISTIC_KEYS:
        raise ValueError(f'Unknown statistic {statistic!r}')
    value = cell.get(statistic)
    return None if value is None else float(value)


def aggregate_statistics(cells, aggregation='positive-weighted'):
    """Per-kind summary of every statistic over the counted cells (``in_mean``), under the
    objective's aggregation rule: positive-weighted pools hits over depths for precisions and
    weights the rank statistics by GT positives; equal-brain averages the cells."""
    counted = {name: cell for name, cell in cells.items() if cell.get('in_mean', True)}
    pooled = str(aggregation).startswith('positive-weighted') or aggregation == 'per-kind-positive-weighted-precision'
    result = {'version': STATISTICS_VERSION}
    for statistic, hits_key, depth_key in (('precision', 'tp', 'effective_k'),
                                           ('precision_at_half_k', 'half_k_tp', 'half_k'),
                                           ('precision_at_double_k', 'double_k_tp', 'double_k')):
        rows = [cell for cell in counted.values() if cell.get(hits_key) is not None and cell.get(depth_key)]
        if not rows:
            result[statistic] = None
        elif pooled:
            depth = sum(int(cell[depth_key]) for cell in rows)
            result[statistic] = sum(int(cell[hits_key]) for cell in rows) / depth if depth else None
        else:
            result[statistic] = math.fsum(int(cell[hits_key]) / int(cell[depth_key]) for cell in rows) / len(rows)
    depths = [result['precision_at_half_k'], result['precision'], result['precision_at_double_k']]
    result['precision_depth_mean'] = math.fsum(depths) / 3 if all(v is not None for v in depths) else None
    for statistic in ('average_precision', 'auroc', 'pauc'):
        scored = [(float(cell[statistic]), float(cell.get('positives') or 0.))
                  for cell in counted.values() if cell.get(statistic) is not None]
        if not scored:
            result[statistic] = None
        elif pooled:
            total = sum(weight for _, weight in scored)
            result[statistic] = sum(value * weight for value, weight in scored) / total if total else None
        else:
            result[statistic] = math.fsum(value for value, _ in scored) / len(scored)
    result['note'] = NOTE
    return result
