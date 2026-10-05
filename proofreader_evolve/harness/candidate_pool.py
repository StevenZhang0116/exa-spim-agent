"""TRAIN branches plus a protected reference supplied by the outer evaluator."""

from copy import deepcopy
import hashlib
import json

import numpy as np

from .search_proposals import formula_info


REFERENCE_EVERY = 3
EVIDENCE_PATIENCE = 2
FOLLOWUP_ATTEMPTS = 2
REFERENCE_FIELDS = (
    'experiment', 'target_kind', 'sequence', 'component_sha256', 'family',
    'hypothesis', 'strategy', 'parameter_grid', 'snapshot', 'status', 'train',
    'cached', 'parameters', 'formula_sha256', 'target_precision', 'top_k_sha256',
    'specialist_profile', 'candidate_type', 'classifier', 'training',
    'research', 'feature_evidence', 'selection', 'in_sample_precision', 'selection_gate',
)


def candidate_metadata(component, kind, report, state, *, coverage=None):
    """Rank/coverage identity from the report the protocol designates for selection."""
    info, _ = formula_info(component)
    cells = {name: value for name, value in report['cells'].items() if name.endswith('/' + kind)}
    membership = hashlib.sha256()
    for name in sorted(cells):
        membership.update(name.encode())
        membership.update(np.sort(state[name]['chosen']).astype('<i8').tobytes())
    metadata = {**info, 'target_precision': sum(c['precision'] for c in cells.values()) / len(cells),
                'top_k_sha256': membership.hexdigest()}
    if coverage is not None:
        metadata['specialist_profile'] = coverage.record(
            hashlib.sha256(component.encode()).hexdigest(), kind, cells, state)
    return metadata


class CandidatePool:
    def __init__(self, path, kinds, *, size=5, score_tolerance=.02, plateau_patience=2,
                 exploration_patience=2, explore_every=3, early_stop=True, coverage=None, hypotheses=None,
                 research=None, image_kinds=()):
        self.path, self.kinds, self.cursor = path, list(kinds), 0
        self.size, self.score_tolerance = size, score_tolerance
        self.plateau_patience, self.exploration_patience = plateau_patience, exploration_patience
        self.explore_every, self.early_stop = explore_every, early_stop
        self.coverage = coverage
        self.hypotheses = hypotheses
        self.research, self.image_kinds = research, set(image_kinds)
        self.members = {kind: [] for kind in kinds}
        # Separate from the bounded TRAIN archive: a weaker TRAIN score cannot
        # evict the component currently used by the outer evaluator.
        self.references = {kind: None for kind in kinds}
        self.covered_history = {kind: set() for kind in kinds}
        self.best_slice_hits = {kind: {} for kind in kinds}
        self.progress = {kind: {'rounds': 0, 'train_rounds': 0, 'stalled_rounds': 0, 'stalled_explorations': 0,
                                'paused': False, 'last_used': {}, 'scheduled_rounds': 0,
                                'reference_measured': False, 'pending_followup': None,
                                'no_evidence_rounds': 0, 'incomplete_investigations': 0,
                                'execution_blocked_rounds': 0, 'pending_investigation': None,
                                'pause_reason': None} for kind in kinds}

    def add(self, entry):
        self._retain(entry['target_kind'], [entry])

    def set_reference(self, entry):
        """Pin a measured TRAIN snapshot; callers never pass heldout measurements."""
        kind = entry['target_kind']
        if kind not in self.references or entry.get('status') != 'evaluated':
            raise ValueError('Reference must be a measured candidate of a configured kind')
        current = self.references[kind]
        if current is not None and current['component_sha256'] == entry['component_sha256']:
            return
        self.references[kind] = {key: deepcopy(entry[key]) for key in REFERENCE_FIELDS if key in entry}
        if current is not None:
            self.progress[kind]['pending_followup'] = {
                'experiment': entry['experiment'], 'component_sha256': entry['component_sha256'], 'attempts': 0}
        # A newly supplied reference needs its own exploration opportunity even
        # when its TRAIN score did not improve. Keep the per-kind cadence intact.
        self.progress[kind].update(stalled_rounds=0, stalled_explorations=0,
                                   paused=False, pause_reason=None, reference_measured=False)

    def _retain(self, kind, entries):
        """Keep the champion, then specialists, then near-best alternatives.

        Specialist slots are bounded by the existing pool size, but are exempt
        from the mean-score tolerance. Compare every trial in a generation at
        once so trial order cannot discard an intermediate specialist early.
        """
        candidates = {e['component_sha256']: e for e in self.members[kind]}
        for entry in entries:
            if entry['target_kind'] != kind:
                raise ValueError('Candidate kind differs from the exploration plan')
            candidates.setdefault(entry['component_sha256'], deepcopy(entry))
        ranked = sorted(candidates.values(), key=lambda e: (-e['target_precision'], e['sequence']))
        if not ranked:
            return
        best = ranked[0]['target_precision']
        selected = [ranked[0]]
        selected[0]['retention'] = {'reason': 'train_champion'}
        rankings, formulas = {ranked[0]['top_k_sha256']}, {ranked[0]['formula_sha256']}
        remaining = ranked[1:]
        while remaining and len(selected) < self.size:
            remaining = [e for e in remaining if e['top_k_sha256'] not in rankings]
            specialists = []
            if self.coverage is not None:
                for entry in remaining:
                    comparison = self.coverage.compare(entry, selected)
                    if comparison['additional_positive_hits'] or comparison['improved_slices']:
                        specialists.append((entry, comparison))
            if specialists:
                candidate, comparison = max(specialists, key=lambda item: (
                    len(item[1]['improved_slices']), item[1]['additional_macro_recall'],
                    item[0]['target_precision'], item[0]['formula_sha256'] not in formulas,
                    -item[0]['sequence']))
                candidate['retention'] = {'reason': 'train_specialist', **comparison}
            else:
                eligible = [e for e in remaining
                            if e['target_precision'] >= best - self.score_tolerance - 1e-12]
                if not eligible:
                    break
                candidate = min(eligible, key=lambda e: (
                    e['formula_sha256'] in formulas, -e['target_precision'], e['sequence']))
                candidate['retention'] = {'reason': 'near_best_alternative'}
            remaining = [e for e in remaining if e['component_sha256'] != candidate['component_sha256']]
            selected.append(candidate)
            rankings.add(candidate['top_k_sha256'])
            formulas.add(candidate['formula_sha256'])
        self.members[kind] = selected
        if self.coverage is not None:
            self.covered_history[kind].update(self.coverage.union(selected))
            for name, count in self.coverage.slice_bests(selected).items():
                self.best_slice_hits[kind][name] = max(self.best_slice_hits[kind].get(name, 0), count)

    def _complements(self, parent):
        if self.coverage is None:
            return []
        results = []
        kind = parent['target_kind']
        candidates = {e['component_sha256']: e for e in self.members[kind]}
        reference = self.references[kind]
        if reference is not None:
            candidates.setdefault(reference['component_sha256'], reference)
        for entry in candidates.values():
            if entry['component_sha256'] == parent['component_sha256']:
                continue
            comparison = self.coverage.compare(entry, [parent])
            if comparison['additional_positive_hits'] or comparison['improved_slices']:
                results.append({'experiment': entry['experiment'], 'family': entry['family'],
                                'target_precision': entry['target_precision'], **comparison})
        return sorted(results, key=lambda item: (
            -len(item['improved_slices']), -item['additional_macro_recall'], -item['target_precision']))

    def next_plan(self):
        for _ in self.kinds:
            kind = self.kinds[self.cursor % len(self.kinds)]
            self.cursor += 1
            progress = self.progress[kind]
            if progress['paused']:
                continue
            progress['scheduled_rounds'] += 1
            champion = self.members[kind][0]
            plateau = progress['stalled_rounds'] >= self.plateau_patience
            hypothesis_plateau = self.hypotheses is not None and self.hypotheses.penalty(champion) >= 2
            evidence_due = progress['no_evidence_rounds'] >= EVIDENCE_PATIENCE
            investigation = deepcopy(progress['pending_investigation'])
            if self.research is not None and not investigation and (plateau or hypothesis_plateau or evidence_due):
                sources = self.research.sources(kind)
                investigation = {'required_source': ('raw_image' if kind in self.image_kinds
                    and 'raw_image' not in sources['compared'] else None),
                    'trigger': 'no_new_evidence' if evidence_due else 'performance_stall',
                    'completion': 'New paired TRAIN feature ablation or numerical 3D comparison on '
                                  'at least two distinct TRAIN candidates; negative results count.'}
            # Reserved reference rounds must not consume every periodic TRAIN
            # alternative slot when both intervals are three.
            periodic = (progress['train_rounds'] + 1) % self.explore_every == 0
            explore = bool(investigation) or plateau or hypothesis_plateau or periodic or not champion['parameters']
            parent = champion
            reference = self.references[kind]
            reference_due = reference is not None and progress['scheduled_rounds'] % REFERENCE_EVERY == 0
            followup = progress['pending_followup'] is not None
            if followup or reference_due:
                parent, explore = reference, True
            elif explore and len(self.members[kind]) > 1:
                parent = min(self.members[kind][1:], key=lambda e: (
                    self.hypotheses.penalty(e) if self.hypotheses is not None else 0,
                    progress['last_used'].get(e['experiment'], -1), -e['target_precision'], e['sequence']))
            progress['last_used'][parent['experiment']] = progress['rounds']
            return {'target_kind': kind, 'mode': 'explore' if explore else 'tune',
                    'branch_role': ('reference' if followup or reference_due else 'train_champion'
                                    if parent['component_sha256'] == champion['component_sha256']
                                    else 'train_alternative'),
                    'reason': ('promotion_followup' if followup else 'reference_branch' if reference_due
                               else 'evidence_required' if evidence_due else 'plateau' if plateau
                               else 'hypothesis_plateau' if hypothesis_plateau
                               else 'periodic_exploration' if periodic
                               else 'introduce_parameters' if not champion['parameters'] else 'parameter_tuning'),
                    'scheduled_round': progress['scheduled_rounds'],
                    'followup_required': followup, 'investigation': investigation,
                    'observed_sources': self.research.sources(kind) if self.research is not None else None,
                    'search_parent': deepcopy(parent), 'best_precision_before': champion['target_precision'],
                    'complementary_candidates': self._complements(parent),
                    'research_focus': ('failure_driven_feature_discovery' if explore else 'controlled_parameter_tuning'),
                    'hypothesis_advice': self.hypotheses.advice(kind) if self.hypotheses is not None else None,
                    'stalled_rounds': progress['stalled_rounds']}
        return None

    def finish(self, plan, entries, *, reference_entry=None, research_status=None):
        kind, progress = plan['target_kind'], self.progress[plan['target_kind']]
        previous_coverage = len(self.covered_history[kind])
        previous_slices = dict(self.best_slice_hits[kind])
        self._retain(kind, [entry for entry in entries if entry['status'] == 'evaluated'])
        # Cached repeats and infrastructure/format failures do not supply a new plateau observation.
        measured = any(e['status'] == 'evaluated' and not e.get('cached', False) for e in entries)
        receipt = research_status or {}
        investigation = plan.get('investigation')
        if self.research is not None:
            fresh = receipt.get('new_measurements', 0)
            measured = measured and fresh > 0
            blocked = receipt.get('execution_failures', 0) > 0 and not fresh
            progress['execution_blocked_rounds'] = progress['execution_blocked_rounds'] + 1 if blocked else 0
            if fresh:
                progress['no_evidence_rounds'] = 0
            elif not blocked:
                progress['no_evidence_rounds'] += 1
            if investigation:
                complete = receipt.get('investigation_complete', False)
                progress['pending_investigation'] = None if complete else deepcopy(investigation)
                if complete:
                    progress['incomplete_investigations'] = 0
                elif not blocked:
                    progress['incomplete_investigations'] += 1
        pending = progress['pending_followup']
        if plan.get('followup_required') and pending:
            pending['attempts'] += 1
            done = receipt.get('complete' if investigation else 'followup_complete', False)
            progress['followup_outcome'] = ('measured' if done else 'retry_pending'
                                           if pending['attempts'] < FOLLOWUP_ATTEMPTS else 'incomplete')
            if done or pending['attempts'] >= FOLLOWUP_ATTEMPTS:
                progress['pending_followup'] = None
        progress['new_positive_hits'] = 0
        progress['improved_slices'] = []
        compared = bool(investigation and receipt.get('investigation_complete'))
        if measured or compared:
            progress['rounds'] += 1
            if plan.get('branch_role') != 'reference':
                progress['train_rounds'] += 1
            reference = self.references[kind]
            if (reference is not None and plan['search_parent']['component_sha256'] == reference['component_sha256']
                    and (not plan.get('followup_required') or receipt.get('followup_complete'))):
                progress['reference_measured'] = True
            progress['new_positive_hits'] = len(self.covered_history[kind]) - previous_coverage
            progress['improved_slices'] = sorted(name for name, count in self.best_slice_hits[kind].items()
                                                 if count > previous_slices.get(name, 0))
            improved = (self.members[kind][0]['target_precision'] > plan['best_precision_before'] + 1e-12
                        or progress['new_positive_hits'] > 0 or bool(progress['improved_slices']))
            if improved:
                progress.update(stalled_rounds=0, stalled_explorations=0)
            else:
                progress['stalled_rounds'] += 1
                forced = bool(investigation) or plan['reason'] in {'plateau', 'hypothesis_plateau'} or (
                    plan.get('branch_role') == 'reference' and plan['stalled_rounds'] >= self.plateau_patience)
                if forced and (self.research is None or compared):
                    progress['stalled_explorations'] += 1
                if (self.early_stop and progress['stalled_explorations'] >= self.exploration_patience
                        and (reference is None or progress['reference_measured'])
                        and progress['pending_followup'] is None):
                    progress['paused'] = True
                    progress['pause_reason'] = 'measured_stagnation'
        if self.early_stop and self.research is not None:
            if progress['execution_blocked_rounds'] >= self.exploration_patience:
                progress.update(paused=True, pause_reason='execution_blocked')
            elif progress['incomplete_investigations'] >= self.exploration_patience:
                progress.update(paused=True, pause_reason='investigation_incomplete')
        if reference_entry is not None:
            if reference_entry['target_kind'] != kind:
                raise ValueError('Reference kind differs from the exploration plan')
            self.set_reference(reference_entry)
        self.save()
        return deepcopy(progress)

    def summary(self):
        # Metrics and diagnostics remain TRAIN-only. Reference identity is supplied
        # by the host; no heldout scores, labels or decision reasons are included.
        return {'selection': 'train-champion-and-specialists-v1' if self.coverage else 'near-best-train',
                'scheduling': 'measured-investigation-promotion-followup-v3',
                'coverage': self.coverage.definitions() if self.coverage else None,
                'kinds': {kind: {'progress': deepcopy(self.progress[kind]),
                    'reference_branch': self._reference_summary(kind),
                    'retained_positive_hits': len(self.coverage.union(self.members[kind])) if self.coverage else None,
                    'ever_retained_positive_hits': len(self.covered_history[kind]) if self.coverage else None,
                    'candidates': [self._candidate_summary(e) for e in self.members[kind]]} for kind in self.kinds},
                'settings': {'size': self.size, 'score_tolerance': self.score_tolerance,
                             'reference_every': REFERENCE_EVERY,
                             'evidence_patience': EVIDENCE_PATIENCE,
                             'followup_attempt_limit': FOLLOWUP_ATTEMPTS,
                             'reference_slots_per_kind': 1,
                             'reference_counts_toward_size': False,
                             'tolerance_scope': 'near-best fallback only; specialists are exempt',
                             'plateau_patience': self.plateau_patience,
                             'exploration_patience': self.exploration_patience,
                             'explore_every': self.explore_every, 'early_stop': self.early_stop}}

    def _reference_summary(self, kind):
        entry = self.references[kind]
        if entry is None:
            return None
        return {**self._candidate_summary(entry), 'retention': {'reason': 'protected_reference'}}

    def _candidate_summary(self, entry):
        result = {key: deepcopy(entry.get(key)) for key in (
            'experiment', 'component_sha256', 'family', 'formula_sha256', 'parameters', 'candidate_type', 'classifier',
            'target_precision', 'top_k_sha256', 'train', 'specialist_profile', 'retention', 'research', 'feature_evidence',
            'selection', 'in_sample_precision', 'selection_gate')}
        if self.coverage is not None:
            members = self.members[entry['target_kind']]
            result['vs_champion'] = self.coverage.compare(entry, members[:1])
            result['vs_other_retained'] = self.coverage.compare(
                entry, [e for e in members if e['component_sha256'] != entry['component_sha256']])
        return result

    def save(self):
        self.path.write_text(json.dumps(self.summary(), indent=2, allow_nan=False))
