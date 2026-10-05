"""Host-observed TRAIN experiments, independent of agent hypothesis declarations."""

import ast
from copy import deepcopy
import hashlib
import json
import math


EVIDENCE_VERSION = 'measured-research-v1'


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def program_identity(source):
    """Ignore formatting, comments and docstrings; do not claim semantic equivalence."""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.body and isinstance(node.body[0], ast.Expr):
                value = node.body[0].value
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    node.body = node.body[1:]
    return hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest()


def has_measurement(value):
    if isinstance(value, dict):
        return any(has_measurement(v) for v in value.values())
    if isinstance(value, list):
        return any(has_measurement(v) for v in value)
    return type(value) in (int, float) and math.isfinite(value)


def registered_image_available(table):
    """Check existing registration receipts without loading graphs or cloud images."""
    provenance = table.meta.get('provenance', {})
    provider = getattr(table, 'image_context', None)
    review = getattr(provider, 'reviews', {}).get(provenance.get('brain'), {})
    return bool(getattr(table, 'local_context', None) is not None
                and 'source_cache' in provenance and review.get('status') == 'reviewed'
                and review.get('source_cache') == provenance['source_cache'])


def scoring_source(report, kind):
    """Inputs actually supplied by the host, not proof that a program used them."""
    cells = [cell for name, cell in (report or {}).get('cells', {}).items() if name.endswith('/' + kind)]
    if any(cell.get('image_context', {}).get('selected_rows', 0) > 0 for cell in cells):
        return 'raw_image'
    if any(cell.get('local_context') for cell in cells):
        return 'local_geometry'
    return 'table'


def observed_descendant(base, parent, candidates):
    """Follow workspace checkpoints; restoring a cache follows its actual origin."""
    visited = set()
    while base and base not in visited:
        if base == parent:
            return True
        visited.add(base)
        entry = candidates.get(base, {})
        base = (entry.get('cached_from') if entry.get('cached') else
                entry.get('lineage', {}).get('edit_base_experiment'))
    return False


class ResearchEvidence:
    def __init__(self, path):
        self.path, self.events, self.seen = path, [], set()

    def record(self, *, kind, generation, tool, source, category, identity, success,
               artifact, edit_base=None, cached=False, comparison=False, outcome=None):
        signature = fingerprint({'kind': kind, 'tool': tool, 'identity': identity})
        fresh = bool(success and not cached and signature not in self.seen)
        if success:
            self.seen.add(signature)
        event = dict(version=EVIDENCE_VERSION, event=len(self.events) + 1, target_kind=kind,
                     generation=generation, tool=tool, source=source, category=category,
                     signature=signature, identity=deepcopy(identity), success=bool(success), fresh=fresh,
                     cached=bool(cached), comparison=bool(comparison and success),
                     artifact=str(artifact), edit_base_experiment=edit_base, outcome=deepcopy(outcome))
        self.events.append(event)
        with self.path.open('a') as stream:
            stream.write(json.dumps(event, allow_nan=False) + '\n')
        return deepcopy(event)

    def sources(self, kind):
        events = [e for e in self.events if e['target_kind'] == kind and e['success']]
        return {'accessed': sorted({e['source'] for e in events}),
                'measured': sorted({e['source'] for e in events if e['category'] == 'measurement'}),
                'compared': sorted({e['source'] for e in events if e['comparison']})}

    def status(self, kind, generation, plan, candidates):
        events = [e for e in self.events if e['target_kind'] == kind and e['generation'] == generation]
        fresh = [e for e in events if e['fresh'] and e['category'] == 'measurement']
        task = plan.get('investigation') or {}
        source = task.get('required_source')
        comparisons = [e for e in fresh if e['comparison'] and (source is None or e['source'] == source)]
        parent = plan.get('search_parent', {}).get('experiment')
        followup = bool(plan.get('followup_required'))
        branch = [e for e in fresh if observed_descendant(e['edit_base_experiment'], parent, candidates)]
        # A combined assignment must measure the requested comparison on the assigned branch.
        qualifying = [e for e in comparisons if not followup or e in branch]
        investigation_complete = not task or bool(qualifying)
        followup_complete = not followup or bool(branch)
        complete = investigation_complete and followup_complete
        failed = sum(not e['success'] for e in events)
        state = ('complete' if complete else 'execution_blocked' if failed and not fresh else 'incomplete')
        return {'version': EVIDENCE_VERSION, 'status': state, 'complete': complete,
                'investigation_required': bool(task), 'investigation_complete': investigation_complete,
                'followup_required': followup, 'followup_complete': followup_complete,
                'required_source': source, 'new_measurements': len(fresh),
                'new_observations': sum(e['fresh'] and e['category'] == 'observation' for e in events),
                'reused_results': sum(e['success'] and not e['fresh'] for e in events),
                'execution_failures': failed, 'qualifying_events': [e['event'] for e in qualifying],
                'branch_measurements': [e['event'] for e in branch],
                'observed_sources': self.sources(kind),
                'events': [{k: e[k] for k in ('event', 'tool', 'source', 'category', 'fresh',
                           'success', 'comparison', 'artifact', 'signature')} for e in events],
                'requirement': ('Run a new paired TRAIN feature ablation or numerical 3D comparison '
                    'on at least two distinct TRAIN candidates. Required source: ' + (source or 'any')
                    if task else 'No additional comparison is required'),
                'note': 'Fresh execution is not proof of scientific novelty or benefit. Negative results count. '
                        'Plans, previews, repeated observations and cached restores do not satisfy a comparison.'}
