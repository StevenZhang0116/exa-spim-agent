"""Host-owned TRAIN folds: fragment groups or purged spatial blocks.

Candidate identities and grouping coordinates remain in the host. Workers receive
selected predictor rows and declared patches; fitting also receives TRAIN labels.

Selection brains are partitioned into folds. Auxiliary TRAIN brains supply all of
their rows to every fold's fitting set and are never held out, so an auxiliary
brain that the frozen detector was fitted on cannot leak an optimistic in-sample
measurement into the out-of-fold selection score.
"""

import hashlib

import numpy as np


SPLIT_VERSION = 'train-fragment-spatial-purged-v2'
FOLDS = 3
SPATIAL_BLOCK_UM = 500.


def _fragments(kind, row):
    fields = ('segment_id',) if kind == 'merge' else ('segment_id_a', 'segment_id_b')
    return tuple(str(row[field]) for field in fields)


def grouped_folds(train, kind, *, selection_brains=None):
    """Fixed, label-independent assignment; purge shared split fragments per fold.

    Each selection-brain row receives one out-of-fold prediction, retaining the
    complete selection pool. Auxiliary brains (TRAIN brains outside
    ``selection_brains``) are fully included in every fitting set. Too few groups
    or unusable fit labels make the diagnostic unavailable; there is no silent
    fallback to a random row split.
    """
    selection = sorted(train) if selection_brains is None else sorted(str(b) for b in selection_brains)
    if not selection or set(selection) - set(train):
        raise ValueError('Selection brains must be a nonempty subset of the TRAIN brains')
    auxiliary = sorted(set(train) - set(selection))
    groups, fragment_rows, counts = {}, {}, {}
    for brain, bank in sorted(train.items()):
        table = bank.tables[kind]
        fragments = [_fragments(kind, row) for row in table.candidates]
        fragment_rows[brain] = fragments
        if brain not in selection:
            continue
        if kind == 'merge':
            values = [f'{brain}/fragment/{ids[0]}' for ids in fragments]
        else:
            provider = getattr(table, 'local_context', None)
            if provider is None:
                raise ValueError('Split internal validation needs the TRAIN fragment provider for spatial blocks')
            graph, _ = provider.store.graph(table.meta['provenance'])
            anchors = np.asarray([[int(row['node_id_a']), int(row['node_id_b'])]
                                  for row in table.candidates], dtype=int)
            if anchors.shape != (len(table.features), 2) or (anchors < 0).any() or (anchors >= len(graph.node_xyz)).any():
                raise ValueError('Invalid split anchor geometry for internal folds')
            centers = np.asarray(graph.node_xyz[anchors], dtype=float).mean(axis=1)
            if not np.isfinite(centers).all():
                raise ValueError('Nonfinite TRAIN spatial coordinates')
            blocks = np.floor(centers / SPATIAL_BLOCK_UM).astype(np.int64)
            values = [f'{brain}/block/{x}/{y}/{z}' for x, y, z in blocks]
        groups[brain] = values
        for value in values:
            counts[value] = counts.get(value, 0) + 1
    if len(counts) < FOLDS:
        raise ValueError(f'Internal validation requires at least {FOLDS} independent groups')
    loads, assignments = [0] * FOLDS, {}
    # Largest groups first, stable hash for ties; never inspect labels to assign folds.
    # Only selection-brain groups are balanced; auxiliary rows join every fold.
    for group in sorted(counts, key=lambda g: (-counts[g], hashlib.sha256(g.encode()).hexdigest())):
        fold = min(range(FOLDS), key=lambda f: (loads[f], f))
        assignments[group] = fold
        loads[fold] += counts[group]
    fold_ids = {brain: np.asarray([assignments[g] for g in values], dtype=np.int8)
                for brain, values in groups.items()}
    partitions, summary = [], []
    for fold in range(FOLDS):
        fit_rows, held_rows, cells, fit_labels = {}, {}, {}, []
        for brain, bank in sorted(train.items()):
            table = bank.tables[kind]
            if brain in selection:
                held = np.flatnonzero(fold_ids[brain] == fold)
                fitting = np.flatnonzero(fold_ids[brain] != fold)
                held_fragments = {segment for index in held for segment in fragment_rows[brain][index]}
                keep = np.asarray([not any(segment in held_fragments for segment in fragment_rows[brain][index])
                                   for index in fitting], dtype=bool)
                fitted = fitting[keep]
            else:
                held = np.zeros(0, dtype=int)
                fitted = np.arange(len(table.truth))
                keep = np.ones(len(fitted), dtype=bool)
            fit_rows[brain], held_rows[brain] = fitted, held
            truth = np.asarray(table.truth, dtype=float)
            fit_labels.extend(truth[fitted][~np.isnan(truth[fitted])])
            cells[brain] = {'role': 'selection' if brain in selection else 'auxiliary',
                            'fit_rows': len(fitted), 'held_rows': len(held),
                            'held_labeled_rows': int((~np.isnan(truth[held])).sum()),
                            'purged_rows': int((~keep).sum()),
                            'held_positives': int(np.nansum(truth[held]))}
        if set(map(int, fit_labels)) != {0, 1}:
            raise ValueError(f'Internal fold {fold} lacks both fitting classes after grouping/purging')
        partitions.append((fit_rows, held_rows))
        summary.append({'fold': fold, 'brains': cells})
    identity = hashlib.sha256()
    for brain in sorted(train):
        identity.update(brain.encode())
        identity.update(train[brain].tables[kind].meta['pool_sha256'].encode())
        identity.update(('selection' if brain in selection else 'auxiliary').encode())
        if brain in fold_ids:
            identity.update(fold_ids[brain].tobytes())
        for fitted, held in partitions:
            identity.update(fitted[brain].astype('<i8').tobytes())
            identity.update(held[brain].astype('<i8').tobytes())
    return partitions, {'version': SPLIT_VERSION, 'folds': FOLDS,
                        'selection_brains': selection, 'auxiliary_brains': auxiliary,
                        'grouping': 'fragment' if kind == 'merge' else 'spatial_blocks_with_fragment_purge',
                        'spatial_block_um': SPATIAL_BLOCK_UM if kind == 'split' else None,
                        'partition_sha256': identity.hexdigest(), 'fold_details': summary,
                        'scope': 'TRAIN-internal selection/diagnostic; no fitting/held-row fragment overlap within '
                                 'each fold; auxiliary brains are never held out. '
                                 'Repeated adaptive use is not an independent final test.'}


def fold_assignments(partitions, brain, size):
    """Recover each selection row's fold from the held sets; auxiliary brains have none."""
    ids = np.full(int(size), -1, dtype=np.int8)
    for fold, (_, held) in enumerate(partitions):
        rows = np.asarray(held.get(brain, ()), dtype=int)
        if len(rows):
            if (ids[rows] != -1).any():
                raise ValueError('A TRAIN row is held out in more than one fold')
            ids[rows] = fold
    return ids
