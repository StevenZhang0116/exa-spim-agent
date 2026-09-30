"""TRAIN-only branch retention and per-component plateau scheduling."""

from copy import deepcopy
import hashlib
import json

import numpy as np

from .search_proposals import formula_info


def candidate_metadata(component, kind, report, state):
    info, _ = formula_info(component)
    cells = {name: value for name, value in report['cells'].items() if name.endswith('/' + kind)}
    membership = hashlib.sha256()
    for name in sorted(cells):
        membership.update(name.encode())
        membership.update(np.sort(state[name]['chosen']).astype('<i8').tobytes())
    return {**info, 'target_precision': sum(c['precision'] for c in cells.values()) / len(cells),
            'top_k_sha256': membership.hexdigest()}


class CandidatePool:
    def __init__(self, path, kinds, *, size=5, score_tolerance=.02, plateau_patience=2,
                 exploration_patience=2, explore_every=3, early_stop=True):
        self.path, self.kinds, self.cursor = path, list(kinds), 0
        self.size, self.score_tolerance = size, score_tolerance
        self.plateau_patience, self.exploration_patience = plateau_patience, exploration_patience
        self.explore_every, self.early_stop = explore_every, early_stop
        self.members = {kind: [] for kind in kinds}
        self.progress = {kind: {'rounds': 0, 'stalled_rounds': 0, 'stalled_explorations': 0,
                                'paused': False, 'last_used': {}} for kind in kinds}

    def add(self, entry):
        """Retain the TRAIN champion and different Top-K sets near its score."""
        kind = entry['target_kind']
        candidates = {e['component_sha256']: e for e in self.members[kind]}
        candidates.setdefault(entry['component_sha256'], deepcopy(entry))
        ranked = sorted(candidates.values(), key=lambda e: (-e['target_precision'], e['sequence']))
        best = ranked[0]['target_precision']
        remaining = [e for e in ranked if e['target_precision'] >= best - self.score_tolerance - 1e-12]
        selected, rankings, formulas = [], set(), set()
        while remaining and len(selected) < self.size:
            # First keep the champion, then prefer distinct formula structures among eligible branches.
            if selected:
                remaining.sort(key=lambda e: (e['formula_sha256'] in formulas,
                                               -e['target_precision'], e['sequence']))
            candidate = remaining.pop(0)
            if candidate['top_k_sha256'] in rankings:
                continue
            selected.append(candidate)
            rankings.add(candidate['top_k_sha256'])
            formulas.add(candidate['formula_sha256'])
        self.members[kind] = selected

    def next_plan(self):
        for _ in self.kinds:
            kind = self.kinds[self.cursor % len(self.kinds)]
            self.cursor += 1
            progress = self.progress[kind]
            if progress['paused']:
                continue
            champion = self.members[kind][0]
            plateau = progress['stalled_rounds'] >= self.plateau_patience
            periodic = (progress['rounds'] + 1) % self.explore_every == 0
            explore = plateau or periodic or not champion['parameters']
            parent = champion
            if explore and len(self.members[kind]) > 1:
                parent = min(self.members[kind][1:], key=lambda e: (
                    progress['last_used'].get(e['experiment'], -1), -e['target_precision'], e['sequence']))
            progress['last_used'][parent['experiment']] = progress['rounds']
            return {'target_kind': kind, 'mode': 'explore' if explore else 'tune',
                    'reason': ('plateau' if plateau else 'periodic_exploration' if periodic
                               else 'introduce_parameters' if not champion['parameters'] else 'parameter_tuning'),
                    'search_parent': deepcopy(parent), 'best_precision_before': champion['target_precision'],
                    'stalled_rounds': progress['stalled_rounds']}
        return None

    def finish(self, plan, entries):
        kind, progress = plan['target_kind'], self.progress[plan['target_kind']]
        for entry in entries:
            if entry['status'] == 'evaluated':
                self.add(entry)
        # Cached repeats and infrastructure/format failures do not supply a new plateau observation.
        measured = any(e['status'] == 'evaluated' and not e.get('cached', False) for e in entries)
        if measured:
            progress['rounds'] += 1
            improved = self.members[kind][0]['target_precision'] > plan['best_precision_before'] + 1e-12
            if improved:
                progress.update(stalled_rounds=0, stalled_explorations=0)
            else:
                progress['stalled_rounds'] += 1
                if plan['reason'] == 'plateau':
                    progress['stalled_explorations'] += 1
                if self.early_stop and progress['stalled_explorations'] >= self.exploration_patience:
                    progress['paused'] = True
        self.save()
        return deepcopy(progress)

    def summary(self):
        # These records are constructed from TRAIN only, never the outer acceptance ledger.
        return {'kinds': {kind: {'progress': deepcopy(self.progress[kind]), 'candidates': [
                    {key: e.get(key) for key in ('experiment', 'family', 'formula_sha256', 'parameters',
                                                 'candidate_type', 'classifier',
                                                 'target_precision', 'top_k_sha256', 'train')}
                    for e in self.members[kind]]} for kind in self.kinds},
                'settings': {'size': self.size, 'score_tolerance': self.score_tolerance,
                             'plateau_patience': self.plateau_patience,
                             'exploration_patience': self.exploration_patience,
                             'explore_every': self.explore_every, 'early_stop': self.early_stop}}

    def save(self):
        self.path.write_text(json.dumps(self.summary(), indent=2, allow_nan=False))
