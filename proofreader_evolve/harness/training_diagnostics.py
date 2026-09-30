"""Stratified TRAIN diagnostics and parent-relative top-K changes."""

import hashlib
import numpy as np


def _sample(indices, seed, limit=4):
    indices = np.asarray(indices, dtype=int)
    if not len(indices):
        return []
    # One fixed anchor, plus reproducibly rotating examples.
    rest = indices[1:]
    rng = np.random.default_rng(seed)
    return [int(indices[0]), *map(int, rng.choice(rest, min(limit - 1, len(rest)), replace=False))]


def example(table, scores, index, group):
    return {'group': group, 'label': int(table.truth[index]), 'score': float(scores[index]),
            'features': {name: float(value) if np.isfinite(value) else None
                         for name, value in table.features.iloc[index].items()}}


def describe(table, scores, chosen, generation=0, cell='', parent=None):
    order = np.lexsort((np.asarray(table.keys, dtype=str), -scores))
    mask = np.zeros(len(scores), dtype=bool)
    mask[chosen] = True
    # Boundary panel is outside top-K, distinct from currently selected rows.
    boundary = order[len(chosen):min(len(order), len(chosen) + max(20, len(chosen) // 4))]
    groups = {
        'selected_positive': chosen[table.truth[chosen] == 1],
        'selected_label0': chosen[table.truth[chosen] == 0],
        'boundary_positive': boundary[table.truth[boundary] == 1],
        'boundary_label0': boundary[table.truth[boundary] == 0],
        'missed_positive': np.flatnonzero((table.truth == 1) & ~mask),
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
        examples.extend(example(table, scores, i, group) for i in _sample(indices, seed))
    return {'training_examples': examples, 'group_sizes': {g: len(i) for g, i in groups.items()},
            'ranking_delta': delta}


def feature_statistics(tables_by_brain, target_kind):
    """Full TRAIN pool summaries, not correlations inferred from error samples."""
    result = {}
    for brain, bank in tables_by_brain.items():
        table = bank.tables[target_kind]
        cells = {}
        for name in table.features.columns:
            values = table.features[name].to_numpy(dtype=float)
            entry = {}
            for label in (0, 1):
                selected = values[table.truth == label]
                finite = selected[np.isfinite(selected)]
                entry[str(label)] = {'count': len(selected), 'finite': len(finite),
                                    'q25_q50_q75': [float(format(v, '.6g')) for v in np.quantile(finite, [.25, .5, .75])]
                                    if len(finite) else None}
            cells[name] = entry
        result[f'{brain}/{target_kind}'] = cells
    return {'note': 'Whole TRAIN pool grouped by native labels; quartiles rounded to 6 significant digits. '
                    'Label 0 is not verified biological truth.',
            'cells': result}
