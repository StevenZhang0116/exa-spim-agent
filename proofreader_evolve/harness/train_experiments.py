"""Budgeted TRAIN tools, parameter search, immutable snapshots and experiment memory."""

import asyncio
from copy import deepcopy
import difflib
import hashlib
import json
from pathlib import Path
import re
import time
from types import SimpleNamespace

import numpy as np

from . import fixed_pool_scoring as scoring
from .candidate_pool import candidate_metadata
from .classifier_contract import (MAX_MODEL_SOURCE_BYTES, frozen_model, classifier_configs,
                                  classifier_info, config_identity, normalize_config, validate_program,
                                  verify_artifacts, MODEL_VERSION)
from .classifier_training import fit_classifier
from .isolated_scoring import preflight, ScorerExecutionError
from .scorer_components import compose
from .search_proposals import read_proposal, formula_info, parameter_candidates
from .train_feedback import metrics_only, write_train_feedback
from .trajectory import Trajectory, log_metrics


def digest(source):
    return hashlib.sha256(source.encode()).hexdigest()


def save_state(path, state):
    np.savez(path, **{f'{cell}:{key}': values[key] for cell, values in state.items()
                     for key in ('scores', 'chosen')})


class ExperimentMemory:
    """Host-owned archive: no validation values or validation decision reasons."""
    def __init__(self, run_dir):
        self.path = run_dir / 'train_experiments.jsonl'
        self.artifact_store = run_dir / 'model_artifacts'
        self.entries, self.cache, self.candidates = [], {}, {}
        self.fit_cache, self.fitted_sources = {}, {}

    def register(self, entry):
        self.cache[(entry['target_kind'], entry['component_sha256'])] = entry
        self.candidates[entry['experiment']] = entry

    def seed(self, kind, component, rules, report, state):
        directory = self.path.parent / 'train_seed' / kind
        directory.mkdir(parents=True)
        (directory / 'scorer.py').write_text(component)
        (directory / 'rules.md').write_text(rules)
        (directory / 'train_evaluation.json').write_text(json.dumps(report, allow_nan=False))
        save_state(directory / 'train_state.npz', state)
        entry = {'experiment': f'seed/{kind}', 'target_kind': kind, 'sequence': 0,
                 'component_sha256': digest(component), 'family': 'seed',
                 'hypothesis': 'Initial measured scorer', 'strategy': 'Starting component',
                 'parameter_grid': {}, 'snapshot': str(directory), 'status': 'evaluated',
                 'train': metrics_only(report), 'cached': True,
                 **candidate_metadata(component, kind, report, state)}
        model = frozen_model(component)
        if model is not None:
            summary = {key: model[key] for key in ('config', 'training_brains', 'training_rows', 'training_fingerprint')}
            if model.get('program') is not None:
                verify_artifacts(model, self.artifact_store)
                (directory / 'training.py').write_text(model['program'])
            summary['resumed_frozen_model'] = True
            self.fitted_sources[digest(component)] = summary
            entry['training'] = summary
        self.register(entry)
        return entry

    def append(self, entry):
        self.entries.append(deepcopy(entry))
        with self.path.open('a') as stream:
            stream.write(json.dumps(entry, allow_nan=False) + '\n')
        if entry['status'] == 'evaluated':
            self.register(deepcopy(entry))

    def search(self, target_kind, query='', limit=5):
        tokens = set(re.findall(r'\w+', query.lower()))
        candidates = [entry for entry in self.entries if entry['target_kind'] == target_kind]
        def relevance(entry):
            text = (entry['hypothesis'] + ' ' + entry['strategy']).lower()
            return len(tokens & set(re.findall(r'\w+', text))), entry['sequence']
        ranked = sorted(candidates, key=relevance, reverse=True)
        chosen, seen = [], set()
        for entry in ranked:
            if entry['component_sha256'] not in seen:
                chosen.append(entry)
                seen.add(entry['component_sha256'])
            if len(chosen) == limit:
                break
        return [deepcopy(e) for e in chosen]


class TrainingExperiments:
    def __init__(self, gen_dir, policy_path, rules_path, target_kind, parent_sources,
                 train, parent_report, parent_state, budgets, timeout, margin,
                 memory, max_evaluations=8, generation=0, search_plan=None,
                 classifier_timeout=300, classifier_memory_mb=8192, classifier_threads=1):
        self.gen_dir, self.policy_path, self.rules_path = gen_dir, policy_path, rules_path
        self.proposal_path = gen_dir / 'proposal.json'
        self.training_path = gen_dir / 'training.py'
        self.target_kind, self.parent_sources = target_kind, dict(parent_sources)
        self.train, self.parent_report, self.parent_state = train, parent_report, parent_state
        self.budgets, self.timeout, self.margin = budgets, timeout, margin
        self.memory, self.max_evaluations, self.generation = memory, max_evaluations, generation
        self.search_plan = search_plan or {'mode': 'explore', 'search_parent': {}}
        self.search_source = policy_path.read_text()
        self.classifier_timeout, self.classifier_memory_mb = classifier_timeout, classifier_memory_mb
        self.classifier_threads = classifier_threads
        self.attempts, self.results = [], {}
        self.evaluations_used, self.repair_granted = 0, False
        self.lock, self.trace = asyncio.Lock(), Trajectory(gen_dir)

    @property
    def remaining(self):
        return self.max_evaluations + int(self.repair_granted) - self.evaluations_used

    def _check_editable_files(self):
        for path in (self.policy_path, self.rules_path):
            if not path.is_file() or path.is_symlink() or path.parent.resolve() != self.gen_dir.resolve():
                raise ValueError('Candidate snapshot must be a regular file in this generation')

    def _read_candidate(self):
        self._check_editable_files()
        if self.policy_path.stat().st_size > MAX_MODEL_SOURCE_BYTES or self.rules_path.stat().st_size > 32_000:
            raise ValueError('Candidate/rules exceeds snapshot size limit')
        component = self.policy_path.read_text()
        if len(component.encode()) > 128_000 and digest(component) not in self.memory.fitted_sources:
            raise ValueError('Hand-written scorers are limited to 128 KB; larger sources must be frozen trained models')
        return component, self.rules_path.read_text()

    def _read_program(self):
        path = self.training_path
        if not path.is_file() or path.is_symlink() or path.stat().st_size > 128_000:
            raise ValueError('Write a regular training.py of at most 128 KB in this generation')
        program = path.read_text()
        validate_program(program)
        return program

    def _write_program(self, program):
        if self.training_path.is_symlink():
            raise ValueError('training.py cannot be a symlink')
        self.training_path.write_text(program)

    def _verify_model(self, component):
        # Syntax errors in ordinary formulas retain normal scoring-error accounting.
        try:
            model = frozen_model(component)
        except SyntaxError:
            return
        if model is not None and model['version'] == MODEL_VERSION:
            verify_artifacts(model, self.memory.artifact_store)

    def _check_mode(self, component):
        try:
            info, _ = formula_info(component)
        except SyntaxError:
            # A Python syntax failure goes through normal execution-error/repair accounting.
            return
        if info['candidate_type'] == 'classifier' and digest(component) not in self.memory.fitted_sources:
            raise ValueError('Measured models are immutable; edit training.py/parameters and call train_classifier()')
        if (self.search_plan['mode'] == 'tune'
                and info['formula_sha256'] != self.search_plan['search_parent']['formula_sha256']):
            raise ValueError('This is a tune generation: change only PARAMS or numeric training parameters; '
                             'restore_candidate("parent") recovers the assigned formula')

    def _merge_measurement(self, measured, measured_state, combined):
        """Reuse only this component; the other kind always comes from the accepted parent."""
        report, state = deepcopy(self.parent_report), dict(self.parent_state)
        report['policy_sha256'] = digest(combined)
        for cell in report['cells']:
            if cell.endswith('/' + self.target_kind):
                report['cells'][cell] = deepcopy(measured['cells'][cell])
                state[cell] = measured_state[cell]
        report.update(scoring.aggregate_precision(report['cells']))
        return report, state

    def evaluate(self):
        proposal = read_proposal(self.proposal_path)
        component, rules = self._read_candidate()
        self._check_mode(component)
        if 'classifier' in proposal:
            model = frozen_model(component)
            requested = normalize_config(proposal['classifier'])
            if (model is None or requested != model['config']
                    or model.get('program') != self._read_program()):
                raise ValueError('Training code/parameters have not been fitted: call train_classifier(); '
                                 'for a hand-written formula, remove proposal.json.classifier')
        return self._evaluate(component, rules, proposal)

    def _evaluate(self, component, rules, proposal, *, charge=True):
        self._verify_model(component)
        component_hash = digest(component)
        if component_hash in self.results:
            return {**self.results[component_hash]['public'], 'cached': True,
                    'evaluations_used': self.evaluations_used,
                    'evaluations_remaining': self.remaining}
        cached = self.memory.cache.get((self.target_kind, component_hash))
        if charge and not cached and self.remaining <= 0:
            return {'status': 'budget_exhausted', 'evaluations_used': self.evaluations_used,
                    'message': 'Use restore_candidate("best") or another measured candidate ID.'}
        number = len(self.attempts) + 1
        trial_dir = self.gen_dir / 'experiments' / f'attempt{number:03d}'
        trial_dir.mkdir(parents=True)
        (trial_dir / 'scorer.py').write_text(component)
        model = frozen_model(component) if digest(component) in self.memory.fitted_sources else None
        if model is not None and model.get('program') is not None:
            (trial_dir / 'training.py').write_text(model['program'])
            previous = frozen_model(self.search_source)
            before = previous.get('program', '') if previous else ''
            (trial_dir / 'training.py.diff').write_text(''.join(difflib.unified_diff(
                before.splitlines(keepends=True), model['program'].splitlines(keepends=True),
                fromfile='parent_training.py', tofile='training.py')))
        (trial_dir / 'rules.md').write_text(rules)
        combined = compose({**self.parent_sources, self.target_kind: component})
        (trial_dir / 'combined_scorer.py').write_text(combined)
        difference = ''.join(difflib.unified_diff(
            self.search_source.splitlines(keepends=True), component.splitlines(keepends=True),
            fromfile='parent_scorer.py', tofile='scorer.py'))
        (trial_dir / 'scorer.py.diff').write_text(difference)
        (trial_dir / 'proposal.json').write_text(json.dumps(proposal, indent=2))
        entry = {**proposal, 'target_kind': self.target_kind, 'sequence': len(self.memory.entries) + 1,
                 'experiment': f'{self.gen_dir.name}/{trial_dir.name}',
                 'search_mode': self.search_plan['mode'],
                 'search_parent': self.search_plan['search_parent'].get('experiment'),
                 'parent_sha256': digest(compose(self.parent_sources)),
                 'component_sha256': component_hash, 'candidate_sha256': digest(combined),
                 'snapshot': str(trial_dir), 'train': None, 'train_gate': None,
                 'ranking_delta': None, 'status': 'running', 'cached': bool(cached)}
        if component_hash in self.memory.fitted_sources:
            entry['training'] = deepcopy(self.memory.fitted_sources[component_hash])
            (trial_dir / 'training_summary.json').write_text(json.dumps(entry['training'], indent=2, allow_nan=False))
        started = time.monotonic()
        result = {'source': combined, 'component': component, 'rules': rules, 'report': None,
                  'state': None, 'entry': entry}
        self.trace.emit('train_experiment', f"{entry['experiment']}: {proposal['hypothesis'][:300]}",
                        target_kind=self.target_kind, search_mode=self.search_plan['mode'])
        stage = 'cache_load' if cached else 'preflight'
        if charge and not cached:
            self.evaluations_used += 1
        try:
            formula, _ = formula_info(component)
            entry.update(formula)
            (trial_dir / 'parameters.json').write_text(json.dumps(formula['parameters'], indent=2))
            self.trace.emit('candidate_formula', f"{entry['experiment']}: family={proposal['family']}; "
                            f"formula={formula['formula_sha256'][:12]}; PARAMS={formula['parameters']}",
                            **formula)
            if cached:
                cached_dir = Path(cached['snapshot'])
                measured = json.loads((cached_dir / 'train_evaluation.json').read_text())
                with np.load(cached_dir / 'train_state.npz', allow_pickle=False) as data:
                    measured_state = {cell: {key: data[f'{cell}:{key}'] for key in ('scores', 'chosen')}
                                      for cell in measured['cells'] if cell.endswith('/' + self.target_kind)}
            else:
                for bank in self.train.values():
                    preflight(component, bank.tables[self.target_kind].features, self.target_kind, self.timeout,
                              artifact_store=self.memory.artifact_store)
                stage = 'train_evaluation'
                measured_state = {}
                target_tables = {brain: SimpleNamespace(tables={self.target_kind: bank.tables[self.target_kind]})
                                 for brain, bank in self.train.items()}
                measured = scoring.evaluate(component, target_tables, self.budgets, self.timeout,
                                            state=measured_state, artifact_store=self.memory.artifact_store)
            stage = 'diagnostics'
            report, state = self._merge_measurement(measured, measured_state, combined)
            from .training_diagnostics import describe
            for brain, bank in self.train.items():
                for kind, table in bank.tables.items():
                    cell = f'{brain}/{kind}'
                    report['cells'][cell].update(describe(table, state[cell]['scores'], state[cell]['chosen'],
                                                         self.generation, cell, self.parent_state[cell]))
            passed, reason = scoring.acceptance(self.parent_report, report, self.margin)
            entry.update(status='evaluated', train=metrics_only(report),
                         train_gate={'passed': passed, 'reason': reason, 'required_for_promotion': False},
                         ranking_delta={k: v['ranking_delta'] for k, v in report['cells'].items()},
                         **candidate_metadata(component, self.target_kind, report, state))
            result.update(report=report, state=state)
            (trial_dir / 'train_evaluation.json').write_text(json.dumps(report, indent=2, allow_nan=False))
            save_state(trial_dir / 'train_state.npz', state)
            feedback_path = trial_dir / 'feedback.json'
            write_train_feedback(feedback_path, report, [], self.budgets, target_kind=self.target_kind,
                                 report_role='candidate')
            public = {**entry, 'feedback': json.loads(feedback_path.read_text())}
            log_metrics(self.trace, entry['experiment'], report, self.parent_report)
        except Exception as exc:
            execution_error = stage in {'preflight', 'train_evaluation'} and isinstance(
                exc, (ScorerExecutionError, SyntaxError, ValueError))
            entry.update(status='execution_error' if execution_error else 'evaluation_error',
                         failure_stage=stage, error=f'{type(exc).__name__}: {exc}')
            result.update(report=None, state=None)
            entry.update(train=None, train_gate=None, ranking_delta=None)
            if execution_error and not self.repair_granted:
                self.repair_granted = True
            public = {**entry, 'repair_allowance': int(self.repair_granted),
                      'message': ('Fix the execution error and evaluate again; no metric was assigned.'
                                  if execution_error else 'Evaluation infrastructure failed; no metric was assigned.')}
        entry['wall_seconds'] = time.monotonic() - started
        entry['evaluations_used'] = self.evaluations_used
        public.update(wall_seconds=entry['wall_seconds'], evaluations_used=self.evaluations_used,
                      evaluations_remaining=self.remaining)
        result['public'] = public
        (trial_dir / 'result.json').write_text(json.dumps(entry, indent=2, allow_nan=False))
        self.attempts.append(result)
        self.results[component_hash] = result
        self.memory.append(entry)
        gate = entry.get('train_gate') or {}
        changes = '; '.join(f"{cell}: +{d['gained_positives']}/-{d['lost_positives']} positives, "
                            f"overlap={d['top_k_overlap']}, same_top_k={d['same_top_k']}"
                            for cell, d in (entry['ranking_delta'] or {}).items())
        self.trace.emit('train_experiment_result', f"{entry['experiment']}: {entry['status']}; "
                        f"{gate.get('reason', entry.get('error', ''))}" + (f'; {changes}' if changes else ''),
                        ranking_delta=entry['ranking_delta'], evaluations_remaining=self.remaining)
        return public

    @staticmethod
    def _best(results):
        valid = [r for r in results if r['report'] is not None]
        if not valid:
            raise ValueError('No successfully measured candidate to restore')
        # TRAIN ranks exploration candidates; only the outer validation mean controls promotion.
        return max(valid, key=lambda r: (r['report']['macro_precision'], -len(r['component']),
                                        -r['entry']['sequence']))

    def _restore_result(self, result):
        self._check_editable_files()  # Restore even malformed/oversized code, but never redirected paths.
        if self.proposal_path.is_symlink():
            raise ValueError('proposal.json cannot be a symlink')
        self._verify_model(result['component'])
        self.policy_path.write_text(result['component'])
        model = frozen_model(result['component'])
        if model is not None and model.get('program') is not None:
            self._write_program(model['program'])
        self.rules_path.write_text(result['rules'])
        proposal = {k: result['entry'][k] for k in ('hypothesis', 'strategy', 'family', 'parameter_grid')}
        proposal['parameter_grid'] = {}  # A restored candidate represents one measured configuration.
        if model is not None and model['version'] == MODEL_VERSION:
            proposal['classifier'] = result['entry']['classifier']
        self.proposal_path.write_text(json.dumps(proposal, indent=2, allow_nan=False))
        self.trace.emit('candidate_restored', f"Restored {result['entry']['experiment']}; no new evaluation")
        return {**result['public'], 'restored': result['entry']['experiment'],
                'evaluations_used': self.evaluations_used,
                'evaluations_remaining': self.remaining}

    def restore(self, candidate_id):
        if candidate_id == 'best':
            return self._restore_result(self._best(self.attempts))
        if candidate_id == 'parent':
            candidate_id = self.search_plan['search_parent'].get('experiment')
        entry = self.memory.candidates.get(candidate_id)
        if not entry or entry['target_kind'] != self.target_kind:
            raise ValueError('Use a measured ID for this kind, e.g. gen002/attempt001, parent or best')
        directory = Path(entry['snapshot'])
        component, rules = (directory / 'scorer.py').read_text(), (directory / 'rules.md').read_text()
        if digest(component) != entry['component_sha256']:
            raise ValueError('Archived scorer snapshot has changed; refusing to restore mismatched measurements')
        self._check_mode(component)
        proposal = {k: entry[k] for k in ('hypothesis', 'strategy', 'family', 'parameter_grid')}
        model = frozen_model(component)
        if model is not None and model['version'] == MODEL_VERSION:
            proposal['classifier'] = entry['classifier']
        self._evaluate(component, rules, proposal)
        result = self.results[digest(component)]
        if result['report'] is None:
            raise ValueError('Cached candidate could not be loaded; inspect the tool error')
        return self._restore_result(result)

    def search_parameters(self):
        proposal = read_proposal(self.proposal_path)
        component, rules = self._read_candidate()
        self._check_mode(component)
        if frozen_model(component) is not None:
            raise ValueError('Use train_classifier() to refit classifier hyperparameters; fitted arrays cannot be edited')
        candidates = parameter_candidates(component, proposal)
        required = sum(digest(c) not in self.results and (self.target_kind, digest(c)) not in self.memory.cache
                       for c in candidates)
        if required > self.remaining:
            raise ValueError(f'Grid needs {required} new evaluations, but {self.remaining} remain; shrink parameter_grid')
        searches = self.gen_dir / 'parameter_searches'
        batch = searches / f'batch{len(list(searches.glob("batch*"))) + 1:03d}'
        batch.mkdir(parents=True)
        (batch / 'formula.py').write_text(component)
        (batch / 'proposal.json').write_text(json.dumps(proposal, indent=2))
        responses = [self._evaluate(candidate, rules, proposal) for candidate in candidates]
        rows = [{k: r.get(k) for k in ('experiment', 'status', 'parameters', 'target_precision',
                                       'train_gate', 'cached', 'error')} for r in responses]
        (batch / 'results.json').write_text(json.dumps(rows, indent=2, allow_nan=False))
        successful = [self.results[digest(c)] for c in candidates if self.results[digest(c)]['report'] is not None]
        restored = self._restore_result(self._best(successful)) if successful else None
        # The full proposal and metrics already live in snapshots; avoid echoing them for every grid row.
        best = ({key: restored[key] for key in ('restored', 'parameters', 'train_gate', 'feedback')}
                if restored else None)
        return {'status': 'searched' if restored else 'no_successful_candidate', 'candidates': rows,
                'best': best, 'evaluations_used': self.evaluations_used,
                'evaluations_remaining': self.remaining}

    def _training_failure(self, directory, proposal, config, program, exc, started):
        entry = {**proposal, 'classifier': config, **classifier_info(config, program),
                 'target_kind': self.target_kind, 'sequence': len(self.memory.entries) + 1,
                 'experiment': f'{self.gen_dir.name}/{directory.name}', 'snapshot': str(directory),
                 'component_sha256': digest('failed_fit:' + program + config_identity(config)),
                 'candidate_sha256': None, 'search_mode': self.search_plan['mode'],
                 'search_parent': self.search_plan['search_parent'].get('experiment'),
                 'status': 'training_error', 'failure_stage': 'classifier_fit', 'cached': False,
                 'error': f'{type(exc).__name__}: {exc}', 'train': None, 'train_gate': None, 'ranking_delta': None,
                 'evaluations_used': self.evaluations_used, 'wall_seconds': time.monotonic() - started}
        log = directory / 'worker.log'
        if log.is_file():
            with log.open('rb') as stream:
                stream.seek(max(0, log.stat().st_size - 8000))
                entry['training_output_tail'] = stream.read(8000).decode(errors='replace')
        (directory / 'result.json').write_text(json.dumps(entry, indent=2, allow_nan=False))
        result = {'entry': entry, 'report': None, 'state': None, 'source': None, 'component': None,
                  'public': {**entry, 'evaluations_remaining': self.remaining}}
        self.attempts.append(result)
        self.memory.append(entry)
        self.trace.emit('classifier_training_error', entry['error'], experiment=entry['experiment'],
                        evaluations_remaining=self.remaining)
        return result['public']

    def train_classifier(self):
        proposal = read_proposal(self.proposal_path)
        if 'classifier' not in proposal:
            raise ValueError('Write training.py and proposal.json.classifier.parameters first')
        _, rules = self._read_candidate()
        program = self._read_program()
        configurations = classifier_configs(proposal['classifier'], proposal['parameter_grid'])
        for config in configurations:
            if (self.search_plan['mode'] == 'tune' and classifier_info(config, program)['formula_sha256']
                    != self.search_plan['search_parent']['formula_sha256']):
                raise ValueError('Tune mode fixes training.py and nonnumeric parameters; '
                                 'switching formula/model family requires an explore generation')
        required = 0
        for config in configurations:
            cached = self.memory.fit_cache.get((self.target_kind, digest(program), config_identity(config)))
            if (cached is None or (cached['component_sha256'] not in self.results
                                  and (self.target_kind, cached['component_sha256']) not in self.memory.cache)):
                required += 1
        if required > self.remaining:
            raise ValueError(f'Classifier search needs {required} new fit/evaluations, but {self.remaining} remain')
        searches = self.gen_dir / 'classifier_searches'
        batch = searches / f'batch{len(list(searches.glob("batch*"))) + 1:03d}'
        batch.mkdir(parents=True)
        (batch / 'proposal.json').write_text(json.dumps(proposal, indent=2, allow_nan=False))
        (batch / 'training.py').write_text(program)
        responses, successful = [], []
        for config in configurations:
            key = (self.target_kind, digest(program), config_identity(config))
            cached = self.memory.fit_cache.get(key)
            charged = False
            if cached is None:
                fit_root = self.gen_dir / 'classifier_fits'
                directory = fit_root / f'fit{len(list(fit_root.glob("fit*"))) + 1:03d}'
                directory.mkdir(parents=True)
                actual_proposal = {**proposal, 'classifier': config}
                (directory / 'proposal.json').write_text(json.dumps(actual_proposal, indent=2, allow_nan=False))
                self.evaluations_used += 1
                charged, started = True, time.monotonic()
                self.trace.emit('classifier_fit', f"{self.target_kind}: fit agent program {digest(program)[:12]} on TRAIN only; "
                                f"parameters={config['parameters']}",
                                config=config, fit_snapshot=str(directory))
                try:
                    component, summary = fit_classifier(self.train, self.target_kind, config, directory,
                                                        program=program, artifact_store=self.memory.artifact_store,
                                                        timeout=self.classifier_timeout,
                                                        memory_mb=self.classifier_memory_mb,
                                                        threads=self.classifier_threads)
                    cached = {'snapshot': str(directory), 'component_sha256': digest(component)}
                    self.memory.fit_cache[key] = cached
                    self.memory.fitted_sources[digest(component)] = {**summary, 'fit_snapshot': str(directory)}
                    self.trace.emit('classifier_fitted', f"Program {digest(program)[:12]}: frozen artifacts ready; "
                                    f"rows={summary['training_rows']}; classes={summary['class_counts']}",
                                    training_summary=summary)
                except Exception as exc:
                    responses.append(self._training_failure(directory, actual_proposal, config, program, exc, started))
                    continue
            else:
                component = (Path(cached['snapshot']) / 'scorer.py').read_text()
                if digest(component) != cached['component_sha256']:
                    raise ValueError('Cached fitted model was modified; refusing to reuse it')
            response = self._evaluate(component, rules, {**proposal, 'classifier': config}, charge=not charged)
            responses.append(response)
            result = self.results.get(digest(component))
            if result is not None and result['report'] is not None:
                successful.append(result)
        rows = [{key: response.get(key) for key in ('experiment', 'status', 'classifier', 'target_precision',
                                                   'train_gate', 'cached', 'error', 'training_output_tail')} for response in responses]
        (batch / 'results.json').write_text(json.dumps(rows, indent=2, allow_nan=False))
        public_rows = [{**{key: value for key, value in row.items() if key != 'classifier'},
                        'program_sha256': digest(program),
                        'parameters': row['classifier']['parameters']} for row in rows]
        restored = self._restore_result(self._best(successful)) if successful else None
        best = ({key: restored[key] for key in ('restored', 'classifier', 'train_gate', 'feedback', 'training')}
                if restored else None)
        return {'status': 'trained' if restored else 'no_successful_candidate', 'candidates': public_rows, 'best': best,
                'training_metric_protocol': 'in_sample_resubstitution',
                'evaluations_used': self.evaluations_used, 'evaluations_remaining': self.remaining}

    def submitted(self):
        component, rules = self._read_candidate()
        result = self.results.get(digest(component))
        if result is None or result['report'] is None:
            raise ValueError('Final scorer.py must exactly match a successfully evaluated training snapshot')
        self._verify_model(component)
        model = frozen_model(component)
        if model is not None and model.get('program') is not None:
            if self._read_program() != model['program']:
                raise ValueError('Final training.py differs from the measured model; refit or restore a candidate')
            proposal = read_proposal(self.proposal_path)
            if normalize_config(proposal.get('classifier', {})) != model['config']:
                raise ValueError('Final training parameters differ from the measured model')
        return result, rules

    def mcp_server(self):
        from claude_agent_sdk import create_sdk_mcp_server, tool

        async def invoke(function, arguments):
            async with self.lock:
                try:
                    response = await asyncio.to_thread(function, **arguments)
                except Exception as exc:
                    response = {'status': 'error', 'error': f'{type(exc).__name__}: {exc}',
                                'evaluations_remaining': self.remaining}
            return {'content': [{'type': 'text', 'text': json.dumps(response, allow_nan=False)}]}

        @tool('evaluate_train', 'Evaluate scorer.py using hypothesis/strategy in proposal.json. '
              'Call with {}. TRAIN only; cached scores cost no evaluation budget.', {})
        async def evaluate_train(arguments):
            return await invoke(self.evaluate, arguments)

        @tool('search_parameters', 'Read proposal.json.parameter_grid and enumerate numeric PARAMS values '
              'with the formula fixed. Call with {}. Automatically restore the best successful grid candidate.', {})
        async def search_parameters(arguments):
            return await invoke(self.search_parameters, arguments)

        @tool('train_classifier', 'Run agent-written training.py with proposal.json.classifier.parameters and optional parameter_grid. '
              'Call with {}. Harness uses TRAIN only, freezes preprocessing/weights, evaluates and restores '
              'the best successful model. Shares the evaluation budget; validation is never supplied.', {})
        async def train_classifier(arguments):
            return await invoke(self.train_classifier, arguments)

        @tool('restore_candidate', 'Restore an immutable measured snapshot by short ID, e.g. '
              'gen002/attempt001, parent (assigned branch), or best (this generation). No evaluation charge.',
              {'candidate_id': str})
        async def restore_candidate(arguments):
            return await invoke(self.restore, arguments)

        @tool('search_memory', 'Search earlier TRAIN experiments for this kind; use returned experiment IDs '
              'with restore_candidate. No validation results.', {'query': str})
        async def search_memory(arguments):
            response = self.memory.search(self.target_kind, arguments['query'])
            for entry in response:
                directory = Path(entry['snapshot'])
                diff = directory / 'training.py.diff'
                if not diff.is_file():
                    diff = directory / 'scorer.py.diff'
                text = diff.read_text() if diff.is_file() else ''
                entry['diff_file'] = diff.name
                entry['code_diff'], entry['diff_truncated'] = text[:3000], len(text) > 3000
            return {'content': [{'type': 'text', 'text': json.dumps(response, allow_nan=False)}]}

        return create_sdk_mcp_server('training', tools=[evaluate_train, search_parameters,
                                                       train_classifier, restore_candidate, search_memory])
