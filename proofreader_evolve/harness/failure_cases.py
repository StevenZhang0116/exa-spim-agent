"""Bounded, matched TRAIN counterexamples from the selection measurement; no graph loading or heldout input."""

import numpy as np

from .local_context import candidate_ref


CASE_VERSION = 'matched-train-failures-v1'
COMPARISON_VERSION = 'branch-vs-accepted-v1'


def matched_failure_cases(train, kind, state, *, max_pairs=4):
    """Match missed positives to selected label-0 rows using base observations.

    `train` and `state` are the selection protocol's brains and chosen rows (out of
    fold under grouped_oof), so the pairs describe errors the official score sees.
    Pairing is diagnostic, not a causal match or a representative evaluation set.
    At most four pairs can be inspected within the existing eight-query budget.
    """
    panels = {}
    for brain, bank in sorted(train.items()):
        table = bank.tables[kind]
        scores = np.asarray(state[f'{brain}/{kind}']['scores'])
        chosen = np.asarray(state[f'{brain}/{kind}']['chosen'], dtype=int)
        selected = np.zeros(len(scores), dtype=bool)
        selected[chosen] = True
        positives = np.flatnonzero((table.truth == 1) & ~selected)
        negatives = chosen[table.truth[chosen] == 0]
        order = positives[np.argsort(-scores[positives], kind='stable')]
        # Include near-boundary and farther missed cases, rather than only extremes.
        anchors = order[np.unique(np.linspace(0, len(order) - 1,
                         min(max_pairs, len(order)), dtype=int))] if len(order) else []
        fields, vectors = [], []
        if 'detector_score' in table.features:
            fields.append('detector_score')
            vectors.append(table.features['detector_score'].to_numpy(dtype=float))
        geometry = 'gap_um' if kind == 'split' else 'degree'
        vectors.append(np.asarray([row.get(geometry, np.nan) for row in table.candidates], dtype=float))
        fields.append(geometry)
        values = np.column_stack(vectors)
        medians, scales = [], []
        for column in values.T:
            finite = column[np.isfinite(column)]
            q = np.quantile(finite, [.25, .5, .75]) if len(finite) else [0., 0., 1.]
            medians.append(q[1])
            scales.append(max(float(q[2] - q[0]), 1e-6))
        missing = ~np.isfinite(values)
        standardized = (np.where(missing, medians, values) - medians) / scales
        available, pairs = list(map(int, negatives)), []
        for positive in anchors:
            if not available:
                break
            distances = np.mean(np.minimum(np.abs(standardized[available] - standardized[positive]), 20.), axis=1)
            distances += np.mean(missing[available] != missing[positive], axis=1)
            offset = int(np.argmin(distances))
            negative = available.pop(offset)
            def row(index, role):
                return {'candidate_ref': candidate_ref(table, index), 'role': role,
                        'label': int(table.truth[index]), 'score': float(scores[index]),
                        'matching_values': {name: float(v) if np.isfinite(v) else None
                                            for name, v in zip(fields, values[index])}}
            pairs.append({'brain': brain, 'kind': kind, 'matching_fields': fields,
                          'distance_in_robust_units': float(distances[offset]),
                          'cases': [row(int(positive), 'missed_positive'), row(negative, 'selected_label0')]})
        panels[brain] = pairs
    pairs = []
    for index in range(max_pairs):
        for brain in panels:
            if index < len(panels[brain]) and len(pairs) < max_pairs:
                pairs.append({'pair_id': f'pair{len(pairs) + 1:02d}', **panels[brain][index]})
    return {'version': CASE_VERSION, 'target_kind': kind, 'pairs': pairs,
            'scope': 'Selection-brain TRAIN only; errors of the assigned search branch under the official '
                     'selection measurement (the accepted parent when the branch is the reference); '
                     'native label 0 is not verified biological absence',
            'note': 'Pairs are diagnostic examples, not an evaluation dataset or proof of a mechanism. '
                    'Inspect local geometry, propose a falsifiable feature, then measure its incremental value.'}


def branch_comparison(train, kind, branch_state, accepted_state, *, same_as_accepted=False, max_refs=8):
    """Positives the assigned branch gains or loses against the accepted scorer, and
    positives both miss, from the two selection states (out of fold on selection brains).

    Candidate handles only; scores are TRAIN selection scores. This is a TRAIN
    diagnostic for steering the next edit, not a promotion signal.
    """
    groups = {'branch_new_positives': [], 'branch_lost_positives': [], 'shared_missed_positives': []}
    cells = {}
    for brain, bank in sorted(train.items()):
        table = bank.tables[kind]
        cell = f'{brain}/{kind}'
        if cell not in branch_state or cell not in accepted_state:
            continue
        branch_scores = np.asarray(branch_state[cell]['scores'], dtype=float)
        accepted_scores = np.asarray(accepted_state[cell]['scores'], dtype=float)
        chosen_branch = set(map(int, branch_state[cell]['chosen']))
        chosen_accepted = set(map(int, accepted_state[cell]['chosen']))
        positives = set(np.flatnonzero(np.asarray(table.truth) == 1).tolist())
        new = sorted((chosen_branch - chosen_accepted) & positives, key=lambda i: -branch_scores[i])
        lost = sorted((chosen_accepted - chosen_branch) & positives, key=lambda i: -accepted_scores[i])
        shared = sorted(positives - chosen_branch - chosen_accepted, key=lambda i: -branch_scores[i])
        cells[cell] = {'branch_tp': len(chosen_branch & positives), 'accepted_tp': len(chosen_accepted & positives),
                       'branch_new_positives': len(new), 'branch_lost_positives': len(lost),
                       'shared_missed_positives': len(shared)}
        for name, rows in (('branch_new_positives', new), ('branch_lost_positives', lost),
                           ('shared_missed_positives', shared)):
            for index in rows[:max_refs]:
                groups[name].append({'candidate_ref': candidate_ref(table, int(index)), 'brain': brain,
                                     'branch_score': float(branch_scores[index]),
                                     'accepted_score': float(accepted_scores[index])})
    for name in groups:
        groups[name] = groups[name][:max_refs]
    return {'version': COMPARISON_VERSION, 'same_as_accepted': bool(same_as_accepted), 'cells': cells, **groups,
            'note': ('The assigned branch versus the accepted scorer on the selection brains: keep the positives '
                     'it newly finds, understand the ones it loses, and treat positives both miss as the open gap. '
                     'Handles are TRAIN candidate_refs for inspect_candidate; counts are not a promotion signal.')}
