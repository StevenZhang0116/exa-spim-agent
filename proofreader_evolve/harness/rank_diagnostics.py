"""Diagnostic rank statistics beside the official Precision@K (2026-11-01).

Precision at ceil(K/2), precision at 2K and tie-aware average precision over the
same GT-labeled rows and the same fold partition as the official selection score.
They describe where a candidate's gains sit on the ranked list: head hits, deeper
coverage and the whole ranking. Candidate ranking, feature ablation and the
promotion gate keep using Precision@K; nothing here reads validation brains.
"""
import math

import numpy as np

from .label_availability import labeled_rows


DIAGNOSTICS_VERSION = 'rank-diagnostics-v1'
CELL_KEYS = ('half_k', 'half_k_tp', 'precision_at_half_k', 'double_k', 'double_k_tp',
             'precision_at_double_k', 'average_precision')
NOTE = ('Diagnostic only: precision at ceil(K/2) and at 2K (capped by the labeled rows, see half_k '
        'and double_k) and tie-aware average precision, on the same labeled rows and fold partition as '
        'precision; candidate ranking, ablation and promotion use precision at K.')


def average_precision(hits, scores):
    """Tie-aware average precision of boolean ``hits`` ranked by ``scores``, high first.

    Tied scores form one threshold (as in sklearn's average_precision_score), so a score
    that ties many rows is not credited for the row order inside its ties; the official
    Precision@K breaks ties by candidate id instead. Copied from agentic/canonical_stats.
    None when there is no positive.
    """
    hits = np.asarray(hits, dtype=bool)
    if not hits.any():
        return None
    s = -np.asarray(scores, dtype=float)
    order = np.argsort(s, kind='stable')
    s, hits = s[order], hits[order]
    last = np.flatnonzero(np.r_[s[1:] != s[:-1], True])  # last row of every tie group
    tp = np.cumsum(hits)[last]
    return float(np.sum(tp / (last + 1) * np.diff(np.r_[0, tp])) / hits.sum())


def cell_diagnostics(scores, truth, keys, fold_ids, k, *, fold_budgets):
    """Per-cell diagnostics under the selection partition; ``fold_budgets`` is the
    protocol's largest-remainder split so every depth uses the official fold rule."""
    scores, truth = np.asarray(scores, dtype=float), np.asarray(truth, dtype=float)
    fold_ids, keys = np.asarray(fold_ids), np.asarray(keys, dtype=str)
    labeled = np.zeros(len(truth), dtype=bool)
    labeled[labeled_rows(truth)] = True
    n_folds = int(fold_ids.max()) + 1 if len(fold_ids) else 0
    held = [int(np.count_nonzero(labeled & (fold_ids == fold))) for fold in range(n_folds)]
    out = {}
    for name, depth in (('half_k', max(1, math.ceil(k / 2))), ('double_k', 2 * k)):
        budgets = fold_budgets(held, depth)
        tp = effective = 0
        for fold in range(n_folds):
            rows = np.flatnonzero(labeled & (fold_ids == fold))
            picked = rows[np.lexsort((keys[rows], -scores[rows]))][:budgets[fold]]
            tp += int(truth[picked].sum())
            effective += len(picked)
        out[name], out[f'{name}_tp'] = effective, tp
        out[f'precision_at_{name}'] = tp / effective if effective else None
    weighted = positives = 0.
    for fold in range(n_folds):
        rows = np.flatnonzero(labeled & (fold_ids == fold))
        count = float(truth[rows].sum())
        if count > 0:
            weighted += count * average_precision(truth[rows] == 1, scores[rows])
            positives += count
    out['average_precision'] = weighted / positives if positives else None
    return out


def aggregate_diagnostics(cells, aggregation='positive-weighted'):
    """Per-kind summary over the counted cells, under the objective's aggregation rule."""
    counted = [cell for cell in cells.values() if cell.get('in_mean', True) and 'average_precision' in cell]
    if not counted:
        return None
    pooled = str(aggregation).startswith('positive-weighted') or aggregation == 'per-kind-positive-weighted-precision'
    result = {'version': DIAGNOSTICS_VERSION, 'role': 'diagnostic'}
    for name in ('half_k', 'double_k'):
        if pooled:
            depth = sum(int(cell[name]) for cell in counted)
            result[f'precision_at_{name}'] = sum(int(cell[f'{name}_tp']) for cell in counted) / depth if depth else None
        else:
            values = [cell[f'precision_at_{name}'] for cell in counted if cell[f'precision_at_{name}'] is not None]
            result[f'precision_at_{name}'] = math.fsum(values) / len(values) if values else None
    scored = [(cell['average_precision'], float(cell.get('positives') or 0.))
              for cell in counted if cell['average_precision'] is not None]
    if not scored:
        result['average_precision'] = None
    elif pooled:
        total = sum(weight for _, weight in scored)
        result['average_precision'] = sum(value * weight for value, weight in scored) / total if total else None
    else:
        result['average_precision'] = math.fsum(value for value, _ in scored) / len(scored)
    result['note'] = NOTE
    return result
