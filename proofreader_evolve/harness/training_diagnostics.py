"""Stratified TRAIN diagnostics and parent-relative top-K changes.

Labels are three-valued (1 / 0 / NaN); every example group is drawn from GT-labeled
rows, and the boundary panel follows the labeled-row ranking used by Precision@K. The
``deep_positive`` group (2026-11-02) samples positives in the lower half of the labeled
ranking, the region a whole-ranking objective such as AUROC also scores. With ``fold_ids``
(grouped out-of-fold selection scores, one model per fold) both panels are taken inside
each fold's own ranking, as the official statistics are; pooling the folds would order
rows by the scale of their fold's model instead of by rank.
"""

from copy import deepcopy
import hashlib
import numpy as np
from .local_context import candidate_ref


def _sample(indices, seed, limit=4):
    indices = np.asarray(indices, dtype=int)
    if not len(indices):
        return []
    # One fixed anchor, plus reproducibly rotating examples.
    rest = indices[1:]
    rng = np.random.default_rng(seed)
    return [int(indices[0]), *map(int, rng.choice(rest, min(limit - 1, len(rest)), replace=False))]


def example(table, scores, index, group, features=None):
    features = table.features if features is None else features
    return {'group': group, 'label': int(table.truth[index]), 'score': float(scores[index]),
            'candidate_ref': candidate_ref(table, index),
            'features': {name: float(value) if np.isfinite(value) else None
                         for name, value in features.iloc[index].items()}}


def describe(table, scores, chosen, generation=0, cell='', parent=None, *, features=None, fold_ids=None):
    from .fixed_pool_scoring import labeled_order
    order = labeled_order(scores, table.truth, table.keys)
    mask = np.zeros(len(scores), dtype=bool)
    mask[chosen] = True
    # One ranking, or one per fold: the pooled order restricted to a fold keeps the fold's
    # own order (score descending, key ascending), the rule of the fold-wise statistics.
    folds = np.unique(np.asarray(fold_ids)[order]) if fold_ids is not None else []
    rankings = [order[np.asarray(fold_ids)[order] == fold] for fold in folds] or [order]
    boundary, deep = [], []
    for ranking in rankings:
        selected = int(np.count_nonzero(mask[ranking]))
        # Boundary panel is outside top-K, distinct from currently selected rows.
        boundary.append(ranking[selected:min(len(ranking), selected + max(20, selected // 4))])
        deep.append(ranking[len(ranking) // 2:])
    boundary, deep = np.concatenate(boundary), np.concatenate(deep)
    groups = {
        'selected_positive': chosen[table.truth[chosen] == 1],
        'selected_label0': chosen[table.truth[chosen] == 0],
        'boundary_positive': boundary[table.truth[boundary] == 1],
        'boundary_label0': boundary[table.truth[boundary] == 0],
        'missed_positive': np.flatnonzero((table.truth == 1) & ~mask),
        'deep_positive': deep[table.truth[deep] == 1],
    }
    delta = None
    if parent is not None:
        before = np.zeros(len(scores), dtype=bool)
        before[parent['chosen']] = True
        entered = np.flatnonzero(mask & ~before)
        exited = np.flatnonzero(before & ~mask)
        gained = entered[table.truth[entered] == 1]
        lost = exited[table.truth[exited] == 1]
        delta = {'entered': len(entered), 'exited': len(exited),
                 'gained_positives': len(gained), 'lost_positives': len(lost),
                 'net_tp': len(gained) - len(lost),
                 'top_k_overlap': int(np.count_nonzero(mask & before)),
                 'same_top_k': bool(np.array_equal(mask, before))}
        groups.update(gained_positive=gained, lost_positive=lost,
                      newly_selected_label0=entered[table.truth[entered] == 0])
    examples = []
    for group, indices in groups.items():
        seed = int.from_bytes(hashlib.sha256(f'{cell}/{generation}/{group}'.encode()).digest()[:8], 'big')
        examples.extend(example(table, scores, i, group, features) for i in _sample(indices, seed))
    return {'training_examples': examples, 'group_sizes': {g: len(i) for g, i in groups.items()},
            'ranking_delta': delta}


def selection_examples(report, state, protocol, tables_by_brain, kind, generation=0, parent_state=None, *,
                       features_by_brain=None):
    """The selection report with example panels per cell: the one entry for both the
    per-generation feedback and candidate evaluations, so grouped out-of-fold scores are
    always ranked inside their folds (``protocol.fold_ids``) and in-sample scores as one ranking."""
    feedback = deepcopy(report)
    for cell, values in state.items():
        brain = cell.rsplit('/', 1)[0]
        fold_ids = protocol.fold_ids(kind, brain) if protocol.requires_fold_fits else None
        feedback['cells'][cell].update(describe(
            tables_by_brain[brain].tables[kind], values['scores'], values['chosen'], generation, cell,
            (parent_state or {}).get(cell), features=(features_by_brain or {}).get(brain), fold_ids=fold_ids))
    return feedback


def feature_statistics(tables_by_brain, target_kind):
    """Full TRAIN pool summaries, not correlations inferred from error samples."""
    result = {}
    for brain, bank in tables_by_brain.items():
        table = bank.tables[target_kind]
        cells = {}
        unlabeled = int(np.isnan(np.asarray(table.truth, dtype=float)).sum())
        for name in table.features.columns:
            values = table.features[name].to_numpy(dtype=float)
            entry = {'unlabeled_count': unlabeled}
            for label in (0, 1):
                selected = values[table.truth == label]
                finite = selected[np.isfinite(selected)]
                entry[str(label)] = {'count': len(selected), 'finite': len(finite),
                                    'q25_q50_q75': [float(format(v, '.6g')) for v in np.quantile(finite, [.25, .5, .75])]
                                    if len(finite) else None}
            cells[name] = entry
        result[f'{brain}/{target_kind}'] = cells
    return {'note': 'Whole TRAIN pool grouped by three-valued labels: 1 = error, 0 = GT-confirmed no error; '
                    'NaN rows (not judgeable by GT) are only counted. Quartiles rounded to 6 significant digits.',
            'cells': result}
