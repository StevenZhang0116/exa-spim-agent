"""Host-owned hypothesis evidence and scheduling hints, derived only from TRAIN, plus the
agent-written research handoffs stored verbatim next to the host's measured outcomes."""

from copy import deepcopy
import hashlib
import json
import re


MEMORY_VERSION = 'train-hypothesis-evidence-v1'
RESEARCH_FIELDS = {'hypothesis_id', 'failure_mode', 'information_source', 'prediction', 'feature_columns'}
HANDOFF_FIELDS = {'open_questions', 'evidence', 'next_experiment'}
HANDOFF_VERDICTS = ('supports', 'contradicts', 'mixed')
HANDOFF_TEXT = 600  # 300 clipped 5-8 texts per generation in run precision_20261007_112554


HANDOFF_LIMIT = 5  # agents wrote four or five questions or evidence items in most sessions
HANDOFF_UNITS_MAX = 64
RESEARCH_TEXT = 1500  # same bound as proposal hypothesis/strategy; 600 was a submission-time hard check
SNAPSHOT_BYTES = 40_000  # the 20 KB snapshot omitted 1-5 records per generation in the same run
SNAPSHOT_RECORDS = 8


def _clip_text(value, name, notes):
    """Stripped text of at most HANDOFF_TEXT characters; non-text or empty entries are dropped."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        value = str(value)
    if not isinstance(value, str) or not value.strip():
        notes.append(f'{name}: dropped an empty or non-text entry')
        return None
    text = value.strip()
    if len(text) > HANDOFF_TEXT:
        notes.append(f'{name}: clipped to {HANDOFF_TEXT} characters')
        text = text[:HANDOFF_TEXT]
    return text


def _clip_list(value, name, notes):
    if value is None:
        return []
    if isinstance(value, (str, dict)):
        value = [value]
    if not isinstance(value, list):
        notes.append(f'{name}: ignored (not a list)')
        return []
    if len(value) > HANDOFF_LIMIT:
        notes.append(f'{name}: kept the first {HANDOFF_LIMIT} of {len(value)}')
    return value[:HANDOFF_LIMIT]


def _verdict(value, notes):
    """Canonical verdict; common variants map by stem, anything else is recorded as mixed."""
    text = str(value or '').strip().lower()
    if text in HANDOFF_VERDICTS:
        return text
    if text.startswith(('support', 'confirm')):
        return 'supports'
    if text.startswith(('contradict', 'refut', 'reject', 'disconfirm')):
        return 'contradicts'
    notes.append(f"evidence.verdict {value!r} recorded as 'mixed'")
    return 'mixed'


def sanitize_handoff(value):
    """Bounded form of an agent-written research handoff, plus notes on what was clipped or dropped.

    Lenient on purpose. The handoff is optional advice for the next session, so a
    malformed one must never fail the generation that produced it: with strict
    validation, run precision_20261007_001936 lost 8 of 15 generations at the
    submission check to four open questions, an extra `do_not_repeat` field and a
    `measurement` key in evidence items. Unknown fields are dropped, lists are cut to
    HANDOFF_LIMIT items, texts to HANDOFF_TEXT characters, unknown verdicts become
    `mixed`, and extra evidence fields are folded into `conditions` so the agent's
    words are kept. Agent text is never rewritten beyond clipping.

    Returns (handoff, notes); handoff is None when nothing usable remains.
    """
    notes = []
    if isinstance(value, str):
        value = {'open_questions': [value]}
    if not isinstance(value, dict):
        notes.append('handoff ignored: not an object')
        return None, notes
    extra = sorted(set(value) - HANDOFF_FIELDS)
    if extra:
        notes.append(f'dropped unknown handoff fields {extra}')
    result = {'open_questions': [], 'evidence': []}
    for question in _clip_list(value.get('open_questions'), 'open_questions', notes):
        if isinstance(question, dict):
            question = question.get('question', question.get('text'))
        text = _clip_text(question, 'open_questions', notes)
        if text:
            result['open_questions'].append(text)
    for item in _clip_list(value.get('evidence'), 'evidence', notes):
        if isinstance(item, str):
            item = {'claim': item}
        if not isinstance(item, dict):
            notes.append('evidence: dropped a non-object item')
            continue
        claim = _clip_text(item.get('claim'), 'evidence.claim', notes)
        if not claim:
            continue
        verdict = _verdict(item.get('verdict'), notes)
        entry = {'claim': claim, 'verdict': verdict}
        conditions = item.get('conditions')
        others = {k: v for k, v in item.items() if k not in ('claim', 'verdict', 'conditions')}
        if conditions is None and others:
            conditions = '; '.join(f'{k}: {v}' for k, v in others.items() if isinstance(v, (str, int, float)))
            notes.append(f'evidence: folded {sorted(others)} into conditions')
        elif others:
            notes.append(f'evidence: dropped fields {sorted(others)}')
        if conditions is not None:
            text = _clip_text(conditions, 'evidence.conditions', notes)
            if text:
                entry['conditions'] = text
        result['evidence'].append(entry)
    nxt = value.get('next_experiment')
    if isinstance(nxt, str):
        nxt = {'what': nxt}
    if isinstance(nxt, dict):
        what = _clip_text(nxt.get('what'), 'next_experiment.what', notes)
        if what:
            result['next_experiment'] = {'what': what}
            if nxt.get('discriminates') is not None:
                text = _clip_text(nxt['discriminates'], 'next_experiment.discriminates', notes)
                if text:
                    result['next_experiment']['discriminates'] = text
            units = nxt.get('units')
            if units is not None:
                if type(units) is int and 0 <= units <= HANDOFF_UNITS_MAX:
                    result['next_experiment']['units'] = units
                else:
                    notes.append(f'next_experiment.units {units!r} dropped (integer 0..{HANDOFF_UNITS_MAX})')
            extra = sorted(set(nxt) - {'what', 'discriminates', 'units'})
            if extra:
                notes.append(f'next_experiment: dropped fields {extra}')
    elif nxt is not None:
        notes.append('next_experiment ignored: not an object')
    if not (result['open_questions'] or result['evidence'] or result.get('next_experiment')):
        notes.append('handoff ignored: no usable question, evidence item or next experiment')
        return None, notes
    return result, notes


RESEARCH_ID = re.compile(r'[^a-zA-Z0-9_-]+')


def _research_id(value, name, notes):
    """Identifier of letters, digits, _ and -, at most 64 characters; None when nothing usable remains."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        value = str(value)
    if not isinstance(value, str):
        return None
    slug = RESEARCH_ID.sub('-', value.strip()).strip('-')[:64]
    if slug != value:
        notes.append(f'research.{name}: normalised to {slug!r}')
    return slug or None


def sanitize_research(value):
    """Bounded form of an agent-written research declaration, plus notes on what was changed.

    Lenient on purpose, like `sanitize_handoff`: generation 16 of run
    precision_20261008_001127 finished with a measured candidate and was discarded only
    because the agent added an `outcome` key here. Unknown fields are dropped, identifiers
    are normalised, texts clipped to RESEARCH_TEXT characters, feature columns deduplicated
    and cut to 64. Returns (research, notes); research is None when there is no usable
    `hypothesis_id`, in which case the proposal simply carries no declaration.
    """
    notes = []
    if not isinstance(value, dict):
        notes.append('research ignored: not an object')
        return None, notes
    extra = sorted(set(value) - RESEARCH_FIELDS)
    if extra:
        notes.append(f'dropped unknown research fields {extra}')
    hypothesis_id = _research_id(value.get('hypothesis_id'), 'hypothesis_id', notes)
    if hypothesis_id is None:
        notes.append('research ignored: no usable hypothesis_id')
        return None, notes
    result = {'hypothesis_id': hypothesis_id,
              'information_source': _research_id(value.get('information_source'), 'information_source', notes) or 'unspecified'}
    for name in ('failure_mode', 'prediction'):
        text = value.get(name)
        if isinstance(text, (int, float)) and not isinstance(text, bool):
            text = str(text)
        text = text.strip() if isinstance(text, str) else ''
        if not text:
            notes.append(f'research.{name}: missing, recorded as unspecified')
            text = 'unspecified'
        if len(text) > RESEARCH_TEXT:
            notes.append(f'research.{name}: clipped to {RESEARCH_TEXT} characters')
            text = text[:RESEARCH_TEXT]
        result[name] = text
    columns = value.get('feature_columns', [])
    if isinstance(columns, str):
        columns = [columns]
    if not isinstance(columns, list):
        notes.append('research.feature_columns: ignored (not a list)')
        columns = []
    cleaned = []
    for column in columns:
        if not isinstance(column, str) or not column.strip():
            notes.append('research.feature_columns: dropped a non-text entry')
            continue
        column = column.strip()[:128]
        if column not in cleaned:
            cleaned.append(column)
    if len(cleaned) > 64:
        notes.append(f'research.feature_columns: kept the first 64 of {len(cleaned)}')
        cleaned = cleaned[:64]
    result['feature_columns'] = cleaned
    return result, notes


def identity(entry):
    research = entry.get('research') or {}
    identifier = research.get('hypothesis_id')
    if not identifier:
        identifier = 'legacy-' + hashlib.sha256(entry.get('hypothesis', '').encode()).hexdigest()[:16]
    return f"{entry['target_kind']}/{identifier}"


class HypothesisMemory:
    """Descriptions and handoffs are agent claims; results, stagnation and promotion are host measurements."""

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
            'recent_evidence': [], 'feature_evidence': [], 'last_generation': 0, 'handoffs': []})
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

    def record_handoff(self, entry, generation, handoff, host):
        """Store an agent handoff next to host-measured facts about the same generation."""
        record = self._record(entry)
        item = {'generation': int(generation), 'experiment': host.get('experiment'),
                'agent': deepcopy(handoff), 'host': {**deepcopy(host), 'source': 'host measurements'},
                'note': 'agent = claims from that session; host = measured facts; promoted means the submission '
                        'replaced the reference scorer'}
        record.setdefault('handoffs', [])
        record['handoffs'] = [*record['handoffs'], item][-3:]
        record['last_generation'] = max(record.get('last_generation', 0), int(generation))
        self.save()
        return item

    def mark_promoted(self, experiment):
        for record in self.records.values():
            for item in record.get('handoffs', []):
                if item.get('experiment') == experiment:
                    item['host']['promoted'] = True
        self.save()

    def handoffs(self, kind, parent=None, limit=3):
        """Most relevant handoffs for the next session: the assigned branch's lineage first."""
        parent = parent or {}
        parent_key = identity({**parent, 'target_kind': kind}) if parent else None
        parent_experiment = parent.get('experiment')
        items = []
        for record in self.records.values():
            if record['target_kind'] != kind:
                continue
            for item in record.get('handoffs', []):
                related = record['key'] == parent_key or item.get('experiment') == parent_experiment
                items.append((0 if related else 1, -item['generation'], record['hypothesis_id'], item))
        items.sort(key=lambda t: t[:3])
        return [{'hypothesis_id': hid, 'related_to_assigned_branch': rank == 0, **deepcopy(item)}
                for rank, _, hid, item in items[:limit]]

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

    def snapshot(self, kind, query='', limit=SNAPSHOT_RECORDS, parent=None):
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
        while selected and len(json.dumps(selected).encode()) > SNAPSHOT_BYTES:
            selected.pop()
        return {'version': MEMORY_VERSION, 'scope': 'TRAIN only; no heldout metrics, labels or decisions',
                'handoff': self.handoffs(kind, parent),
                'handoff_guide': 'Previous sessions\' open questions, evidence verdicts and proposed next experiment, '
                                 'each next to host-measured facts; related_to_assigned_branch marks the branch you '
                                 'start from. Fill proposal.json.handoff before finishing so the next session can continue.',
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
