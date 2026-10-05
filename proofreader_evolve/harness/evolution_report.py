"""Offline, human-only evolution reports. Standard library; never execute artifacts.

Read an explicit set of JSON/text artifacts, including partially completed runs.
No pickle/NPZ loading, scoring, SDK calls, or inference of missing ancestry.
"""

from datetime import datetime
import json
import math
import os
from pathlib import Path
import re
import tempfile

from .run_records import PRECISION


ASSETS = Path(__file__).resolve().parents[1] / 'plotting' / 'report_assets'
CELL_FIELDS = ('pool_size', 'positives', 'requested_k', 'effective_k', 'tp', 'fp',
               'precision', 'recall')
ATTEMPT_FIELDS = ('experiment', 'sequence', 'target_kind', 'status', 'hypothesis', 'strategy',
                  'family', 'parameters', 'classifier', 'candidate_type', 'component_sha256',
                  'candidate_sha256', 'search_parent', 'search_mode', 'lineage', 'cached',
                  'cached_from', 'error', 'failure_stage', 'wall_seconds', 'evaluations_used',
                  'target_precision', 'in_sample_precision', 'ranking_delta', 'specialist_profile',
                  'selection_gate')
TEXT_FILES = ('scorer.py.from_edit_base.diff', 'training.py.from_edit_base.diff',
              'scorer.py.diff', 'training.py.diff', 'scorer.py', 'training.py', 'rules.md',
              'proposal.json', 'parameters.json', 'worker.log')


def _number(value):
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else None


def _metrics(value):
    if not isinstance(value, dict):
        return None
    return {'macro_precision': _number(value.get('macro_precision')),
            'policy_sha256': value.get('policy_sha256'),
            'cells': {name: {key: _number(cell.get(key)) for key in CELL_FIELDS}
                      | ({'local_context': cell['local_context']} if isinstance(cell.get('local_context'), dict) else {})
                      | ({'image_context': cell['image_context']} if isinstance(cell.get('image_context'), dict) else {})
                      for name, cell in value.get('cells', {}).items()}}


def _pair(value):
    return {scope: _metrics((value or {}).get(scope)) for scope in ('train', 'validation')}


class ArtifactReader:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.warnings = []
        self.preview_remaining = 12 * 1024 * 1024

    def path(self, relative):
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError(f'Artifact escapes run directory: {relative}')
        return path

    def json(self, relative, default=None):
        try:
            path = self.path(relative)
            if not path.exists():
                return default
            if path.stat().st_size > 64 * 1024 * 1024:
                raise ValueError('JSON artifact exceeds 64 MiB')
            return json.loads(path.read_text(encoding='utf-8'))
        except (OSError, ValueError) as exc:
            self.warnings.append(f'{relative}: {exc}')
            return default

    def jsonl(self, relative):
        rows = []
        try:
            path = self.path(relative)
            if not path.exists():
                return rows
            with path.open(encoding='utf-8') as stream:
                for number, line in enumerate(stream, 1):
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                        if not isinstance(row, dict):
                            raise ValueError('Expected JSON object')
                        rows.append(row)
                    except ValueError:
                        self.warnings.append(f'{relative}:{number}: incomplete/invalid record skipped')
        except (OSError, ValueError) as exc:
            self.warnings.append(f'{relative}: {exc}')
        return rows

    def text(self, relative, limit=32_000):
        try:
            path = self.path(relative)
            if not path.is_file():
                return None
            limit = max(0, min(limit, self.preview_remaining))
            with path.open('rb') as stream:
                raw = stream.read(limit)
                truncated = bool(stream.read(1))
            self.preview_remaining -= len(raw)
            return {'path': str(relative), 'text': raw.decode('utf-8', errors='replace'),
                    'truncated': truncated}
        except (OSError, ValueError) as exc:
            self.warnings.append(f'{relative}: {exc}')
            return None

    def event_names(self, directory='.'):
        """Read a bounded tail for status recovery; event payloads stay on disk."""
        relative = Path(directory) / 'trajectory.jsonl'
        names = set()
        try:
            path = self.path(relative)
            if not path.is_file():
                return names
            size = path.stat().st_size
            limit = 256_000
            with path.open('rb') as stream:
                offset = max(0, size - limit)
                stream.seek(offset)
                if offset:
                    stream.readline()  # Discard a potentially partial first record.
                raw = stream.read(limit)
            for line in raw.decode('utf-8', errors='replace').splitlines():
                try:
                    event = json.loads(line)
                    if isinstance(event, dict) and isinstance(event.get('event'), str):
                        names.add(event['event'])
                except ValueError:
                    continue
        except (OSError, ValueError) as exc:
            self.warnings.append(f'{relative}: {exc}')
        return names

    def files(self, directory, names=TEXT_FILES):
        return [value for name in names if (value := self.text(Path(directory) / name)) is not None]


def _pool(value):
    if not isinstance(value, dict):
        return None
    def candidate(entry):
        return {key: entry.get(key) for key in (
            'experiment', 'component_sha256', 'family', 'target_precision', 'in_sample_precision', 'retention',
            'specialist_profile', 'vs_champion', 'vs_other_retained')} | {'selection': _metrics(entry.get('selection'))} \
            if isinstance(entry, dict) else None
    return {'selection': value.get('selection'), 'scheduling': value.get('scheduling'),
            'settings': value.get('settings'),
            'kinds': {kind: {**{key: item.get(key) for key in (
                'progress', 'retained_positive_hits', 'ever_retained_positive_hits')},
                'reference_branch': candidate(item.get('reference_branch')),
                'candidates': [candidate(entry) for entry in item.get('candidates', [])]}
                for kind, item in value.get('kinds', {}).items()}}


def generation_status(row, event_names):
    if row.get('failure_stage'):
        if row['failure_stage'] == 'revision':
            return 'agent_error'
        if row['failure_stage'] == 'submission_check':
            return 'submission_error'
        if row['failure_stage'] == 'research_check':
            return 'research_incomplete'
        return 'evaluation_error'
    if row.get('accepted') is True:
        return 'accepted'
    if row.get('validation') is not None and row.get('accepted') is False:
        return 'rejected'
    if 'generation_interrupted' in event_names:
        return 'interrupted'
    if row.get('reason', '').startswith('Failed:') or 'generation_error' in event_names:
        return 'error'
    return 'incomplete'


def _attempt(reader, entry, directory):
    result = {key: entry.get(key) for key in ATTEMPT_FIELDS}
    result['train'] = _metrics(entry.get('train'))
    result['selection'] = _metrics(entry.get('selection'))
    result['status'] = entry.get('status', 'incomplete')
    result['files'] = reader.files(directory)
    return result


def build_report_data(run_dir):
    reader = ArtifactReader(run_dir)
    manifest = reader.json('manifest.json', {})
    state = reader.json('run_status.json', {})
    ledger = reader.jsonl('ledger.jsonl')
    versions = {row['objective'] for row in [manifest, state, *ledger] if row.get('objective')}
    if versions and versions != {PRECISION}:
        raise ValueError(f'Only fixed-pool precision runs are supported; found {versions}')
    if not versions:
        # Older precision ledgers sometimes omit objective but have both scopes.
        if not ledger or not all('train' in row and 'validation' in row for row in ledger):
            raise ValueError('No identifiable fixed-pool precision run at this path')
    archived = {row['experiment']: row for row in reader.jsonl('train_experiments.jsonl')
                if row.get('experiment')}
    by_gen = {}
    for row in ledger:
        number = row.get('generation')
        if not isinstance(number, int):
            reader.warnings.append('Ledger entry without integer generation skipped')
            continue
        if number in by_gen:
            raise ValueError(f'Duplicate generation in ledger: {number}')
        by_gen[number] = row
    directories = {int(path.name[3:]): path.name for path in reader.root.glob('gen*')
                   if path.is_dir() and re.fullmatch(r'gen\d+', path.name)}
    for number in by_gen:
        directories.setdefault(number, f'gen{number:03d}')
    generations = []
    seed = _pair(reader.json('seed.json'))
    accepted = dict(seed)
    for number, directory in sorted(directories.items()):
        base = Path(directory)
        row = by_gen.get(number) or reader.json(base / 'evaluation.json', {})
        plan = reader.json(base / 'search_plan.json', {})
        event_names = reader.event_names(directory)
        entries = {key: value for key, value in archived.items() if key.startswith(directory + '/')}
        entries.update({item['experiment']: item for item in row.get('experiments', [])})
        # Also recover persisted attempts from an interrupted generation.
        for folder in ('experiments', 'classifier_fits'):
            for path in sorted(reader.path(base / folder).glob('*')):
                if not path.is_dir() or not re.fullmatch(r'(attempt|fit)\d+', path.name):
                    continue
                rel = base / folder / path.name
                item = reader.json(rel / 'result.json')
                if item and item.get('experiment'):
                    entries.setdefault(item['experiment'], item)
                elif folder == 'experiments' or reader.json(rel / 'training_summary.json') is None:
                    proposal = reader.json(rel / 'proposal.json', {})
                    identifier = directory + '/' + path.name
                    entries.setdefault(identifier, {**proposal, 'experiment': identifier,
                                       'target_kind': plan.get('target_kind'), 'status': 'incomplete'})
        attempts = []
        for identifier, entry in sorted(entries.items(), key=lambda pair: (pair[1].get('sequence', 10**9), pair[0])):
            leaf = identifier.rsplit('/', 1)[-1]
            if not re.fullmatch(r'(attempt|fit)\d+', leaf):
                reader.warnings.append(f'Unknown experiment path: {identifier}')
                continue
            folder = 'classifier_fits' if leaf.startswith('fit') else 'experiments'
            attempts.append(_attempt(reader, entry, base / folder / leaf))
        reviser = row.get('reviser') or reader.json(base / 'reviser_result.json', {})
        after = reader.json(base / 'candidate_pool_after.json')
        pair = _pair(row)
        hashes = [(pair[scope] or {}).get('policy_sha256') for scope in ('train', 'validation')]
        if all(hashes) and hashes[0] != hashes[1]:
            reader.warnings.append(f'{directory}: TRAIN/validation policies do not match')
            # Preserve the individual measurements for inspection, but mark the
            # pair so the UI cannot imply they describe one submitted policy.
        policy_match = None if not all(hashes) else hashes[0] == hashes[1]
        parent = {}
        for scope, measurement in accepted.items():
            expected, actual = row.get('parent_policy_sha256'), (measurement or {}).get('policy_sha256')
            parent[scope] = None if expected and actual and expected != actual else measurement
        generations.append({
            'generation': number, 'status': generation_status(row, event_names),
            'target_kind': row.get('target_kind') or plan.get('target_kind'),
            'mode': row.get('search_mode') or plan.get('mode'),
            'search_reason': row.get('search_reason') or plan.get('reason'),
            'search_branch_role': row.get('search_branch_role') or plan.get('branch_role'),
            'search_parent': row.get('search_parent') or plan.get('search_parent', {}).get('experiment'),
            'parent_policy_sha256': row.get('parent_policy_sha256'),
            'submitted_experiment': row.get('submitted_experiment'),
            'reason': row.get('reason'), 'failure_stage': row.get('failure_stage'),
            'parent_validation_precision': _number(row.get('parent_validation_precision')),
            'train_evaluations_used': row.get('train_evaluations_used'),
            'descriptor_runs': [{key: run.get(key) for key in ('run', 'status', 'columns', 'scope', 'inputs', 'cached', 'wall_seconds')}
                                for run in row.get('descriptor_runs') or []],
            'wall_seconds': _number(row.get('wall_seconds')),
            'cost_usd': _number(reviser.get('cost_usd')), 'agent_summary': reviser.get('summary'),
            'usage': reviser.get('usage'), 'progress': row.get('search_progress'),
            'image_evidence': row.get('image_evidence') or reader.json(base / 'image_evidence.json'),
            'research_status': row.get('research_status') or reader.json(base / 'research_status.json'),
            **pair, 'policy_match': policy_match, 'parent': parent,
            'pool_after': _pool(after), 'attempts': attempts,
        })
        if row.get('accepted') is True:
            accepted = dict(pair)
    seeds = []
    for kind in ('merge', 'split'):
        entry = reader.json(Path('train_seed') / kind / 'result.json')
        if entry:
            seeds.append(_attempt(reader, entry, Path('train_seed') / kind))
    final = reader.json('final.json')
    if not state:
        names = reader.event_names()
        state = {'status': 'completed' if 'run_complete' in names else
                 'failed' if 'run_failed' in names else 'unknown'}
    costs = [g['cost_usd'] for g in generations]
    counts = [g['train_evaluations_used'] for g in generations]
    return {'schema_version': 2, 'run_name': reader.root.name,
            'built_at': datetime.now().astimezone().isoformat(timespec='seconds'),
            'state': state, 'manifest': {key: manifest.get(key) for key in (
                'train_brains', 'development_validation_brains', 'budgets', 'promotion_gate',
                'search_version', 'classifier_training_protocol', 'selection_protocol')},
            'baseline': _pair(reader.json('baseline.json')), 'seed': seed,
            'final': _pair(final) if final else None,
            'pool': _pool(reader.json('candidate_pool.json')), 'seeds': seeds,
            'generations': generations,
            'totals': {'recorded_cost_usd': sum(c for c in costs if c is not None),
                       'cost_complete': bool(costs) and all(c is not None for c in costs),
                       'train_evaluations': sum(c for c in counts if isinstance(c, int)),
                       'evaluations_complete': all(isinstance(c, int) for c in counts),
                       'generation_seconds': sum(g['wall_seconds'] or 0 for g in generations),
                       'timing_complete': bool(generations) and all(g['wall_seconds'] is not None for g in generations)},
            'warnings': list(dict.fromkeys(reader.warnings))}


def _atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                         prefix='.' + path.name + '.', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(text)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_report(run_dir, output=None):
    data = build_report_data(run_dir)
    # Escaping '<' also neutralizes closing script tags embedded in agent output.
    payload = json.dumps(data, ensure_ascii=False, allow_nan=False).replace('<', '\\u003c')
    payload = payload.replace('\u2028', '\\u2028').replace('\u2029', '\\u2029')
    html = (ASSETS / 'report.html').read_text(encoding='utf-8')
    html = html.replace('@@CSS@@', (ASSETS / 'report.css').read_text(encoding='utf-8'))
    html = html.replace('@@JS@@', (ASSETS / 'report.js').read_text(encoding='utf-8'))
    html = html.replace('@@DATA@@', payload)
    path = Path(output) if output is not None else Path(run_dir) / 'report.html'
    if path.suffix.lower() != '.html':
        raise ValueError('Report output must have an .html suffix')
    _atomic_text(path, html)
    return path


def update_report(run_dir, trace, *, status='running', **details):
    """Best-effort host output; a reporting failure cannot reject a scorer."""
    try:
        state = {'objective': PRECISION, 'status': status,
                 'updated_at': datetime.now().astimezone().isoformat(timespec='seconds'), **details}
        _atomic_text(Path(run_dir) / 'run_status.json', json.dumps(state, ensure_ascii=False, allow_nan=False))
        return write_report(run_dir)
    except Exception as exc:
        # Logging a disk failure must not mask the experiment's original outcome.
        try:
            trace.emit('report_warning', f'Report update failed: {type(exc).__name__}: {exc}')
        except Exception:
            pass
        return None
