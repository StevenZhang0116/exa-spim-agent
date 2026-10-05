"""Host-owned hypothesis evidence and scheduling hints, derived only from TRAIN."""

from copy import deepcopy
import hashlib
import json
import re


MEMORY_VERSION = 'train-hypothesis-evidence-v1'
RESEARCH_FIELDS = {'hypothesis_id', 'failure_mode', 'information_source', 'prediction', 'feature_columns'}


def validate_research(value):
    if not isinstance(value, dict) or set(value) - RESEARCH_FIELDS:
        raise ValueError('research accepts hypothesis_id, failure_mode, information_source, prediction and feature_columns')
    for name in ('hypothesis_id', 'information_source'):
        if not isinstance(value.get(name), str) or not re.fullmatch(r'[a-zA-Z0-9_-]{1,64}', value[name]):
            raise ValueError(f'research.{name} must be a short identifier (letters, digits, _ or -)')
    for name in ('failure_mode', 'prediction'):
        if not isinstance(value.get(name), str) or not 1 <= len(value[name].strip()) <= 600:
            raise ValueError(f'research.{name} requires a testable description of 1..600 characters')
    columns = value.get('feature_columns', [])
    if (not isinstance(columns, list) or len(columns) > 64
            or any(not isinstance(c, str) or not 1 <= len(c) <= 128 for c in columns)
            or len(set(columns)) != len(columns)):
        raise ValueError('research.feature_columns must contain at most 64 distinct predictor names')
    return {**value, 'feature_columns': columns}


def identity(entry):
    research = entry.get('research') or {}
    identifier = research.get('hypothesis_id')
    if not identifier:
        identifier = 'legacy-' + hashlib.sha256(entry.get('hypothesis', '').encode()).hexdigest()[:16]
    return f"{entry['target_kind']}/{identifier}"


class HypothesisMemory:
    """Descriptions are agent claims; results and stagnation are host measurements."""

    def __init__(self, path):
        self.path = path
        self.records, self.seen_components, self.seen_rankings = {}, set(), set()
        self.ablation_cache = {}
        self.pending_progress = {}

    def _record(self, entry):
        key = identity(entry)
        research = entry.get('research') or {}
        record = self.records.setdefault(key, {
            'key': key, 'target_kind': entry['target_kind'],
            'hypothesis_id': research.get('hypothesis_id', key.split('/', 1)[1]),
            'agent_claim': entry.get('hypothesis', ''), 'declarations': [],
            'information_source': research.get('information_source', 'unspecified'),
            'family': entry.get('family', 'unspecified'), 'fresh_measurements': 0,
            'cached_repeats': 0, 'execution_failures': 0, 'new_train_rankings': 0,
            'best_train_precision': None, 'stalled_generations': 0, 'measured_generations': 0,
            'recent_evidence': [], 'feature_evidence': [], 'last_generation': 0})
        if research and research not in record['declarations']:
            record['declarations'].append(deepcopy(research))
            record['declarations'] = record['declarations'][-3:]
        return record

    def register_seed(self, entry):
        """A seed is already known behavior, not a new hypothesis measurement."""
        self.seen_components.add((entry['target_kind'], entry['component_sha256']))
        self.seen_rankings.add((entry['target_kind'], entry['top_k_sha256']))

    def observe(self, entry):
        """Record a measured candidate; `target_precision` is the run's official selection score.

        Field names keep the `train_precision` schema of memory version v1; under the
        grouped_oof protocol the value is out-of-fold precision on selection brains.
        """
        record = self._record(entry)
        component = (entry['target_kind'], entry['component_sha256'])
        if entry.get('cached') or component in self.seen_components:
            record['cached_repeats'] += 1
        elif entry.get('status') != 'evaluated':
            record['execution_failures'] += 1
        else:
            self.seen_components.add(component)
            record['fresh_measurements'] += 1
            ranking = (entry['target_kind'], entry.get('top_k_sha256'))
            novel = ranking not in self.seen_rankings
            self.seen_rankings.add(ranking)
            record['new_train_rankings'] += int(novel)
            precision = entry['target_precision']
            prior = record['best_train_precision']
            improved = prior is None or precision > prior + 1e-12
            self.pending_progress[record['key']] = self.pending_progress.get(record['key'], False) or improved
            record['best_train_precision'] = max(precision, prior if prior is not None else precision)
            evidence = {'experiment': entry['experiment'], 'train_precision': precision,
                        'improved_hypothesis_best': improved, 'new_train_ranking': novel,
                        'ranking_delta': {k: v for k, v in (entry.get('ranking_delta') or {}).items()
                                          if k.endswith('/' + entry['target_kind'])},
                        'scope': 'Selection-score ranking vs accepted parent (see selection_protocol); '
                                 'not a causal feature attribution'}
            record['recent_evidence'] = [*record['recent_evidence'], evidence][-4:]
        self.save()

    def observe_ablation(self, entry, report):
        record = self._record(entry)
        record['last_generation'] = max(record['last_generation'], report.get('generation', 0))
        fields = ('experiment', 'target_kind', 'status', 'protocol', 'feature_columns', 'signature',
                  'program_sha256', 'parameters_sha256', 'delta_precision', 'cells',
                  'conclusion', 'error', 'scope', 'wall_seconds', 'evaluations_used')
        evidence = {k: deepcopy(report[k]) for k in fields if k in report}
        record['feature_evidence'] = [*record['feature_evidence'], evidence][-3:]
        if report.get('status') == 'measured':
            self.ablation_cache[report['signature']] = deepcopy(report)
        self.save()

    def finish_generation(self, generation, entries):
        grouped = {}
        for entry in entries:
            if entry.get('status') == 'evaluated' and not entry.get('cached'):
                grouped.setdefault(identity(entry), []).append(entry)
        for key, entries in grouped.items():
            record = self.records[key]
            improved = self.pending_progress.pop(key, False)
            record['stalled_generations'] = 0 if improved else record['stalled_generations'] + 1
            record['measured_generations'] += 1
            record['last_generation'] = generation
        self.save()

    @staticmethod
    def _penalty(record):
        no_increment = 0
        for evidence in reversed(record.get('feature_evidence', [])):
            if evidence.get('status') != 'measured' or evidence.get('protocol') != 'TRAIN_grouped_out_of_fold':
                continue
            if evidence['delta_precision'] > 1e-12:
                break
            no_increment += 1
        return min(3, max(record.get('stalled_generations', 0), 2 if no_increment >= 2 else 0))

    def penalty(self, entry):
        return self._penalty(self.records.get(identity(entry), {}))

    def candidate_evidence(self, entry):
        """Attach measured evidence by code/parameter identity, never an agent claim."""
        program = entry.get('program_sha256', entry.get('component_sha256'))
        config = entry.get('classifier') or {'parameters': entry.get('parameters', {})}
        parameters = hashlib.sha256(json.dumps(config, sort_keys=True, allow_nan=False).encode()).hexdigest()
        matches = [report for report in self.ablation_cache.values()
                   if report.get('target_kind') == entry['target_kind']
                   and report.get('program_sha256') == program and report.get('parameters_sha256') == parameters]
        return [{k: deepcopy(report[k]) for k in ('experiment', 'protocol', 'feature_columns',
                    'delta_precision', 'conclusion', 'signature')} for report in matches[-3:]]

    def snapshot(self, kind, query='', limit=5):
        tokens = set(re.findall(r'\w+', query.lower()))
        records = [r for r in self.records.values() if r['target_kind'] == kind]
        def key(record):
            text = json.dumps([record['agent_claim'], record['declarations'], record['family']]).lower()
            return len(tokens & set(re.findall(r'\w+', text))), record['last_generation'], record['fresh_measurements']
        selected = deepcopy(sorted(records, key=key, reverse=True)[:limit])
        for record in selected:
            record['declarations'] = record['declarations'][-1:]
            record['recent_evidence'] = record['recent_evidence'][-2:]
            record['feature_evidence'] = record['feature_evidence'][-1:]
        while selected and len(json.dumps(selected).encode()) > 20_000:
            selected.pop()
        return {'version': MEMORY_VERSION, 'scope': 'TRAIN only; no heldout metrics, labels or decisions',
                'records': selected, 'records_omitted': len(records) - len(selected),
                'note': 'A failed configuration does not refute a whole hypothesis. Feature evidence binds '
                        'to the exact program, parameters, columns and internal split. Ranking novelty is not '
                        'proof of new information. Infrastructure failures do not imply a scientific failure.'}

    def advice(self, kind):
        records = [r for r in self.records.values() if r['target_kind'] == kind]
        sources = sorted({r['information_source'] for r in records})
        stale = sorted((r for r in records if self._penalty(r) >= 2),
                       key=lambda r: (-self._penalty(r), -r['last_generation']))[:5]
        return {'declared_information_sources': sources[:32],
                'information_sources_omitted': max(0, len(sources) - 32),
                'deprioritized_hypotheses': [r['hypothesis_id'] for r in stale],
                'recent_feature_findings': [{k: e.get(k) for k in ('experiment', 'protocol', 'feature_columns',
                                             'delta_precision', 'conclusion')}
                    for record in sorted(records, key=lambda r: -r['last_generation'])[:3]
                    for e in record['feature_evidence'][-1:]],
                'suggestion': ('Test a different failure mechanism or information source; local_geometry has not '
                               'yet been declared.' if 'local_geometry' not in sources else
                               'Use measured feature ablations and complementary failure cases to choose a new direction.'),
                'rule': 'Soft scheduling preference only; reference cadence and outer promotion gate are unchanged.'}

    def save(self):
        self.path.write_text(json.dumps({'version': MEMORY_VERSION, 'scope': 'TRAIN only',
                                        'records': list(self.records.values())}, indent=2, allow_nan=False))
