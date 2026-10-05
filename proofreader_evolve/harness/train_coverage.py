"""Host-owned TRAIN positive identities and bounded specialist diagnostics.

Only aggregate coverage diagnostics are exposed to the reviser. Positive row
identities stay in the host. Coverage is defined on the selection protocol's
TRAIN brains (selection brains under grouped_oof, every TRAIN brain under
in_sample) and uses the protocol's chosen rows, never heldout data.
"""

from copy import deepcopy
import hashlib
import math

import numpy as np


COVERAGE_VERSION = 'train-positive-coverage-v1'
SLICE_DEFINITIONS = {
    'all': 'All GT-positive candidates in this fixed TRAIN pool',
    'gap_le_25um': 'Split representative gap <= 25 microns',
    'gap_gt_25um': 'Split representative gap > 25 microns',
    'degree_3': 'Merge candidate node degree = 3',
    'degree_ge_4': 'Merge candidate node degree >= 4',
}


class TrainingCoverage:
    def __init__(self, train):
        self._cells, self._hits = {}, {}
        for brain, bank in train.items():
            for kind, table in bank.tables.items():
                truth = np.asarray(table.truth)
                if truth.ndim != 1 or not np.isin(truth, [0, 1]).all():
                    raise ValueError('Specialist coverage requires binary TRAIN labels')
                if len(table.candidates) != len(truth):
                    raise ValueError('TRAIN geometry and labels must have identical rows')
                positives = frozenset(map(int, np.flatnonzero(truth == 1)))
                slices = {'all': positives}
                # Read geometry only for positives; no whole-brain graph load or
                # new feature extraction is needed, even for million-row pools.
                groups = {}
                field = 'gap_um' if kind == 'split' else 'degree'
                for index in positives:
                    try:
                        value = float(table.candidates[index].get(field))
                    except (ValueError, TypeError):
                        continue
                    if not math.isfinite(value):
                        continue
                    if kind == 'split' and value >= 0:
                        group = 'gap_le_25um' if value <= 25 else 'gap_gt_25um'
                    elif kind == 'merge' and (value == 3 or value >= 4):
                        group = 'degree_3' if value == 3 else 'degree_ge_4'
                    else:
                        continue
                    groups.setdefault(group, set()).add(index)
                slices.update({name: frozenset(rows) for name, rows in groups.items()})
                self._cells[f'{brain}/{kind}'] = {
                    'kind': kind, 'pool_size': len(truth), 'positive_rows': positives,
                    'slices': slices, 'pool_sha256': table.meta['pool_sha256'],
                    'labels_sha256': hashlib.sha256(truth.astype(np.int8).tobytes()).hexdigest(),
                }

    def record(self, component_sha256, kind, cells, state):
        expected = {name for name, cell in self._cells.items() if cell['kind'] == kind}
        if set(cells) != expected or not expected:
            raise ValueError('Specialist coverage accepts the configured TRAIN cells only')
        hits, profile = set(), {}
        for name, metrics in cells.items():
            cell = self._cells[name]
            for key in ('pool_size', 'pool_sha256', 'labels_sha256'):
                if metrics[key] != cell[key]:
                    raise ValueError(f'TRAIN coverage benchmark changed: {name}/{key}')
            if metrics['positives'] != len(cell['positive_rows']):
                raise ValueError('TRAIN positive count changed')
            budget = (metrics['requested_k'], metrics['effective_k'])
            if cell.get('budget', budget) != budget:
                raise ValueError('TRAIN coverage requires the same fixed K for all candidates')
            indices = np.asarray(state[name]['chosen'])
            if (indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer)
                    or len(indices) != metrics['effective_k']
                    or len(set(map(int, indices))) != len(indices)
                    or (indices < 0).any() or (indices >= cell['pool_size']).any()):
                raise ValueError('Invalid measured TRAIN Top-K indices')
            selected = set(map(int, indices)) & cell['positive_rows']
            if len(selected) != metrics['tp']:
                raise ValueError('TRAIN Top-K positive identities disagree with measured TP')
            cell.setdefault('budget', budget)
            hits.update((name, index) for index in selected)
            for group, positives in cell['slices'].items():
                count = len(selected & positives)
                profile[f'{name}/{group}'] = {
                    'positives': len(positives), 'hits': count,
                    'recall': count / len(positives) if positives else None,
                }
        key = (kind, component_sha256)
        frozen = frozenset(hits)
        if key in self._hits and self._hits[key] != frozen:
            raise ValueError('Identical scorer produced inconsistent TRAIN positive coverage')
        self._hits[key] = frozen
        return {'version': COVERAGE_VERSION, 'slices': profile}

    def hits(self, entry):
        return self._hits[(entry['target_kind'], entry['component_sha256'])]

    def union(self, entries):
        return set().union(*(self.hits(entry) for entry in entries))

    @staticmethod
    def slice_bests(entries):
        best = {}
        for entry in entries:
            for name, metrics in entry['specialist_profile']['slices'].items():
                if metrics['positives']:
                    best[name] = max(best.get(name, 0), metrics['hits'])
        return best

    def compare(self, entry, references):
        """Marginal coverage under the SAME global Top K, not a new per-slice K."""
        own, others = self.hits(entry), self.union(references)
        extra = own - others
        counts = {}
        for cell, _ in extra:
            counts[cell] = counts.get(cell, 0) + 1
        cells = [c for c in self._cells.values() if c['kind'] == entry['target_kind']]
        gain = sum(count / len(self._cells[name]['positive_rows']) for name, count in counts.items())
        previous = self.slice_bests(references)
        wins = [name for name, metrics in entry['specialist_profile']['slices'].items()
                if metrics['positives'] and metrics['hits'] > previous.get(name, 0)]
        return {'additional_positive_hits': len(extra), 'additional_hits_by_cell': counts,
                'additional_macro_recall': gain / len(cells),
                'lost_positive_hits': len(others - own), 'improved_slices': sorted(wins)}

    def definitions(self):
        return {'version': COVERAGE_VERSION, 'scope': 'TRAIN fixed candidate pools only',
                'slice_definitions': deepcopy(SLICE_DEFINITIONS),
                'note': 'Slice hits count positives inside the same global Top K. '
                        'Missing geometry is included in all only. Zero-positive slices do not select specialists. '
                        'These diagnostics do not expand the candidate pool or establish biological truth.'}
