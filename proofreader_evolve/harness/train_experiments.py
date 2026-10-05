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
from .failure_cases import matched_failure_cases
from .feature_ablation import (prepare_ablation, measure_ablation, ABLATION_VERSION,
                               CLASSIFIER_UNITS, FORMULA_UNITS)
from .hypothesis_memory import HypothesisMemory
from .research_evidence import ResearchEvidence, program_identity, has_measurement, scoring_source
from .isolated_scoring import preflight, ScorerExecutionError
from .selection_protocol import SelectionProtocol, load_selection, save_selection
from .descriptor_contract import descriptor_spec, referenced_columns, COLUMN_PREFIX
from .descriptor_runs import SCOPES as DESCRIPTOR_SCOPES
from .local_context import resolve_train_candidate
from .local_features import augmented_features
from .model_execution import ModelExecutionError
from .scorer_components import compose, components
from .search_proposals import read_proposal, formula_info, parameter_candidates
from .train_feedback import metrics_only, write_train_feedback
from .train_coverage import TrainingCoverage
from .trajectory import Trajectory, log_metrics
from .volume_analysis import read_analysis, execute_analysis, MAX_ANALYSES


def digest(source):
    return hashlib.sha256(source.encode()).hexdigest()


def save_state(path, state):
    np.savez(path, **{f'{cell}:{key}': values[key] for cell, values in state.items()
                     for key in ('scores', 'chosen')})


def in_sample_precision(report, kind):
    """Mean in-sample TRAIN precision of this kind's cells; a diagnostic, not a rank."""
    cells = [cell for name, cell in report['cells'].items() if name.endswith('/' + kind)]
    return sum(cell['precision'] for cell in cells) / len(cells)


class ExperimentMemory:
    """Host-owned archive: no validation values or validation decision reasons."""
    def __init__(self, run_dir, *, train=None, protocol=None, fit_limits=None, descriptor_runs=None):
        self.path = run_dir / 'train_experiments.jsonl'
        self.artifact_store = run_dir / 'model_artifacts'
        self.entries, self.cache, self.candidates = [], {}, {}
        self.fit_cache, self.fitted_sources = {}, {}
        # Official selection protocol; coverage/specialists live on its selection brains.
        self.protocol = protocol
        self.fit_limits = dict(fit_limits or {})
        self.oof_scores = {}
        # Agent descriptors: pool-scale computations over the cached contexts. The registry
        # maps registered columns to the code that produced them, for the whole run.
        self.descriptor_runs = descriptor_runs
        self.descriptors, self.descriptor_programs, self.descriptor_generation = {}, {}, 0
        if protocol is not None:
            self.coverage = TrainingCoverage(protocol.selection_train)
        else:
            self.coverage = TrainingCoverage(train) if train is not None else None
        self.train = train or {}
        self.hypotheses = HypothesisMemory(run_dir / 'hypothesis_memory.json')
        self.research = ResearchEvidence(run_dir / 'research_evidence.jsonl')
        # Ablation partitions for kinds without a protocol partition (in_sample mode only);
        # under grouped_oof ablations reuse the selection protocol's folds.
        self.internal_folds = {}

    def register(self, entry):
        self.cache[(entry['target_kind'], entry['component_sha256'])] = entry
        self.candidates[entry['experiment']] = entry

    def ensure_protocol(self, budgets, train=None):
        """Default to the historical in-sample protocol for sessions built without one."""
        if self.protocol is None:
            if not self.train and train:
                self.train = train
            self.protocol = SelectionProtocol(self.train, budgets, mode='in_sample')
            if self.coverage is None and self.train:
                self.coverage = TrainingCoverage(self.protocol.selection_train)
        return self.protocol

    def selection_measurement(self, kind, component, state, directory, *, model=None, budgets=None,
                              oof_scores=None):
        """Official selection report for a measured component.

        Formulas reuse their deterministic scores under the protocol's budgets.
        Fitted models need out-of-fold scores: recorded at fit time, loaded from a
        snapshot, or measured here by fold refits for a resumed seed model.
        """
        protocol = self.ensure_protocol(budgets or {})
        model = frozen_model(component) if model is None else model
        policy = digest(component)
        if model is None or not protocol.requires_fold_fits:
            return protocol.measure_formula(kind, state, policy_sha256=policy)
        scores = oof_scores if oof_scores is not None else self.oof_scores.get(policy)
        if scores is not None:
            self.oof_scores.setdefault(policy, scores)
            return protocol.score_report(kind, scores, policy_sha256=policy)
        if model.get('program') is None:
            raise ValueError('A legacy frozen model without its training program cannot be measured '
                             'under the grouped_oof selection protocol')
        report, selection_state = protocol.measure_classifier(
            kind, model['program'], model['config'], Path(directory) / 'selection_folds', **self.fit_limits)
        report['policy_sha256'] = policy
        self.oof_scores[policy] = {cell: values['scores'] for cell, values in selection_state.items()}
        return report, selection_state

    def seed(self, kind, component, rules, report, state):
        directory = self.path.parent / 'train_seed' / kind
        directory.mkdir(parents=True)
        (directory / 'scorer.py').write_text(component)
        (directory / 'rules.md').write_text(rules)
        (directory / 'train_evaluation.json').write_text(json.dumps(report, allow_nan=False))
        save_state(directory / 'train_state.npz', state)
        model = frozen_model(component)
        selection_report, selection_state = self.selection_measurement(
            kind, component, state, directory, model=model,
            budgets={name.rsplit('/', 1)[1]: cell['requested_k'] for name, cell in report['cells'].items()})
        save_selection(directory, selection_report, selection_state)
        entry = {'experiment': f'seed/{kind}', 'target_kind': kind, 'sequence': 0,
                 'component_sha256': digest(component), 'family': 'seed',
                 'hypothesis': 'Initial measured scorer', 'strategy': 'Starting component',
                 'parameter_grid': {}, 'snapshot': str(directory), 'status': 'evaluated',
                 'train': metrics_only(report), 'cached': True,
                 'in_sample_precision': in_sample_precision(report, kind),
                 'selection': metrics_only(selection_report),
                 **candidate_metadata(component, kind, selection_report, selection_state, coverage=self.coverage)}
        if model is not None:
            summary = {key: model[key] for key in ('config', 'training_brains', 'training_rows', 'training_fingerprint')}
            if model.get('program') is not None:
                verify_artifacts(model, self.artifact_store)
                (directory / 'training.py').write_text(model['program'])
            summary['resumed_frozen_model'] = True
            self.fitted_sources[digest(component)] = summary
            entry['training'] = summary
        self.register(entry)
        self.hypotheses.register_seed(entry)
        if self.train:
            cells = [cell for name, cell in report['cells'].items() if name.endswith('/' + kind)]
            self.research.record(kind=kind, generation=0, tool='evaluate_train', source=scoring_source(report, kind),
                category='measurement', identity={'program': program_identity((model or {}).get('program') or component),
                    'model_config': model['config'] if model else None,
                    'train': {brain: bank.tables[kind].meta for brain, bank in self.train.items()},
                    'k': cells[0]['requested_k']}, success=True, cached=True,
                artifact=directory / 'result.json')
        (directory / 'result.json').write_text(json.dumps(entry, indent=2, allow_nan=False))
        return entry

    def append(self, entry):
        self.entries.append(deepcopy(entry))
        with self.path.open('a') as stream:
            stream.write(json.dumps(entry, allow_nan=False) + '\n')
        if entry['status'] == 'evaluated':
            self.register(deepcopy(entry))
        self.hypotheses.observe(entry)

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
                 classifier_timeout=300, classifier_memory_mb=8192, classifier_threads=1,
                 parent_selection=None, descriptor_budget=None):
        self.gen_dir, self.policy_path, self.rules_path = gen_dir, policy_path, rules_path
        self.proposal_path = gen_dir / 'proposal.json'
        self.training_path = gen_dir / 'training.py'
        self.target_kind, self.parent_sources = target_kind, dict(parent_sources)
        self.train, self.parent_report, self.parent_state = train, parent_report, parent_state
        self.budgets, self.timeout, self.margin = budgets, timeout, margin
        self.memory, self.max_evaluations, self.generation = memory, max_evaluations, generation
        self.protocol = self.memory.ensure_protocol(budgets, train)
        self.protocol_info = self.protocol.describe(target_kind)
        if self.memory.coverage is None:
            self.memory.coverage = TrainingCoverage(self.protocol.selection_train)
        self.search_plan = search_plan or {'mode': 'explore', 'search_parent': {}}
        self.search_source = policy_path.read_text()
        # An observed workspace checkpoint, not an inference about the agent's
        # conceptual ancestry. Grid variants share this checkpoint until restore.
        self.edit_base = self.search_plan['search_parent'].get('experiment')
        self.edit_source = self.search_source
        self.edit_program = self.training_path.read_text() if self.training_path.exists() else ''
        self.restored_from = None
        self.classifier_timeout, self.classifier_memory_mb = classifier_timeout, classifier_memory_mb
        self.classifier_threads = classifier_threads
        self.attempts, self.results = [], {}
        self.evaluations_used, self.repair_granted = 0, False
        self.inspections_used = 0
        self.image_inspections_used = 0
        self.volume_analyses_used = 0
        self.lock, self.trace = asyncio.Lock(), Trajectory(gen_dir)
        self.feature_diagnostics = []
        self.frame_cache = (None, None)
        self.descriptor_budget = {'per_call_seconds': 1800., 'generation_seconds': 5400., **(descriptor_budget or {})}
        self.descriptor_seconds_used, self.descriptor_log = 0., []
        self.descriptor_path = gen_dir / 'descriptor.py'
        if parent_selection is None:
            parent_selection = self.memory.selection_measurement(
                target_kind, parent_sources[target_kind], parent_state, gen_dir / 'parent_selection',
                budgets=budgets)
        self.parent_selection, self.parent_selection_state = parent_selection
        # Matched failures come from the official selection measurement of the accepted parent.
        self.failure_cases = matched_failure_cases(self.protocol.selection_train, target_kind,
                                                   self.parent_selection_state)
        (gen_dir / 'failure_cases.json').write_text(json.dumps(self.failure_cases, indent=2, allow_nan=False))
        # The request starts with one matched pair when available. The agent may
        # replace it with any valid TRAIN handles, independent of scoring selection.
        request_path = gen_dir / 'analysis_request.json'
        if not request_path.exists():
            pair = self.failure_cases['pairs'][:1]
            requests = [{'candidate_ref': case['candidate_ref'], 'occurrence_index': 0}
                        for entry in pair for case in entry['cases']]
            request_path.write_text(json.dumps({'candidates': requests, 'radius_um': 40., 'level': 0,
                'channel': 0, 'timepoint': 0, 'max_nodes': 256}, indent=2))
        self._save_hypothesis_feedback()

    def _save_hypothesis_feedback(self):
        query = self.search_plan.get('search_parent', {}).get('hypothesis', '')
        snapshot = self.memory.hypotheses.snapshot(self.target_kind, query)
        (self.gen_dir / 'hypothesis_memory.json').write_text(json.dumps(snapshot, indent=2, allow_nan=False))
        self.research_status()

    def research_status(self):
        """Live, host-owned completion receipt, safe to expose to the reviser."""
        status = self.memory.research.status(self.target_kind, self.generation,
                                           self.search_plan, self.memory.candidates)
        (self.gen_dir / 'research_status.json').write_text(json.dumps(status, indent=2, allow_nan=False))
        return status

    def _record_research(self, tool, source, category, identity, success, artifact, **kwargs):
        # Lightweight inspection fixtures may omit the run archive.
        memory = getattr(self, 'memory', None)
        if memory is None or not hasattr(memory, 'research'):
            return
        self.memory.research.record(kind=self.target_kind, generation=self.generation,
            tool=tool, source=source, category=category, identity=identity,
            success=success, artifact=artifact, edit_base=self.edit_base, **kwargs)
        self.research_status()

    def _train_identity(self):
        return {brain: bank.tables[self.target_kind].meta for brain, bank in self.train.items()}

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

    def _selection_gate(self, selection_report):
        before, after = self.parent_selection['macro_precision'], selection_report['macro_precision']
        return {'metric': self.protocol_info['metric'], 'mode': self.protocol.mode,
                'parent': before, 'candidate': after, 'delta': after - before,
                'passed': after > before + 1e-12, 'required_for_promotion': False,
                'scope': 'Development selection signal on selection brains; the outer promotion gate decides acceptance'}

    def _cached_selection(self, cached, component, model, measured_state):
        directory = Path(cached['snapshot'])
        if (directory / 'selection_state.npz').exists():
            report, state = load_selection(directory)
            if model is not None:
                self.memory.oof_scores.setdefault(digest(component),
                                                  {cell: values['scores'] for cell, values in state.items()})
            return report, state
        return self.memory.selection_measurement(self.target_kind, component, measured_state, directory,
                                                 model=model, budgets=self.budgets)

    def _classifier_frames(self, program):
        """One feature extraction per program, shared by fold fits and the full fit."""
        key = digest(program)
        if self.frame_cache[0] != key:
            frames = {brain: augmented_features(program, bank.tables[self.target_kind],
                                                self.classifier_timeout, self.classifier_memory_mb)
                      for brain, bank in self.train.items()}
            self.frame_cache = (key, frames)
        return self.frame_cache[1]

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
        response = self._evaluate(component, rules, proposal)
        if response.get('experiment'):
            self._checkpoint(response['experiment'], component)
        return response

    def _lineage(self):
        return {'version': 1, 'basis': 'observed_workspace_checkpoint',
                'edit_base_experiment': self.edit_base, 'restored_from': self.restored_from}

    def _checkpoint(self, experiment, component):
        self.edit_base, self.edit_source = experiment, component
        self.edit_program = self.training_path.read_text() if self.training_path.exists() else ''

    def _edit_diffs(self, directory, component, program=None):
        for name, before, after in (('scorer.py', self.edit_source, component),
                                    ('training.py', self.edit_program, program)):
            if after is not None:
                difference = difflib.unified_diff(before.splitlines(keepends=True),
                    after.splitlines(keepends=True), fromfile='workspace_checkpoint/' + name, tofile=name)
                (directory / (name + '.from_edit_base.diff')).write_text(''.join(difference))

    def _evaluate(self, component, rules, proposal, *, charge=True, selection_scores=None):
        self._verify_model(component)
        component_hash = digest(component)
        if component_hash in self.results:
            self.trace.emit('candidate_cache_hit', 'Reusing this generation\'s measured snapshot',
                            experiment=self.results[component_hash]['entry']['experiment'])
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
        self._edit_diffs(trial_dir, component, (model or {}).get('program'))
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
                 'lineage': self._lineage(), 'cached_from': cached['experiment'] if cached else None,
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
                    features = augmented_features(component, bank.tables[self.target_kind], self.timeout)
                    preflight(component, features, self.target_kind, self.timeout,
                              artifact_store=self.memory.artifact_store)
                stage = 'train_evaluation'
                measured_state = {}
                target_tables = {brain: SimpleNamespace(tables={self.target_kind: bank.tables[self.target_kind]})
                                 for brain, bank in self.train.items()}
                measured = scoring.evaluate(component, target_tables, self.budgets, self.timeout,
                                            state=measured_state, artifact_store=self.memory.artifact_store)
            stage = 'selection'
            if cached:
                selection_report, selection_state = self._cached_selection(cached, component, model, measured_state)
            else:
                selection_report, selection_state = self.memory.selection_measurement(
                    self.target_kind, component, measured_state, trial_dir, model=model, budgets=self.budgets,
                    oof_scores=selection_scores)
            stage = 'diagnostics'
            report, state = self._merge_measurement(measured, measured_state, combined)
            from .training_diagnostics import describe
            features_by_brain = {brain: augmented_features(component, bank.tables[self.target_kind], self.timeout)
                                 for brain, bank in self.train.items()}
            for brain, bank in self.train.items():
                for kind, table in bank.tables.items():
                    cell = f'{brain}/{kind}'
                    features = features_by_brain[brain] if kind == self.target_kind else None
                    report['cells'][cell].update(describe(table, state[cell]['scores'], state[cell]['chosen'],
                                                         self.generation, cell, self.parent_state[cell], features=features))
            selection_feedback = deepcopy(selection_report)
            for cell, values in selection_state.items():
                brain = cell.rsplit('/', 1)[0]
                selection_feedback['cells'][cell].update(describe(
                    self.train[brain].tables[self.target_kind], values['scores'], values['chosen'],
                    self.generation, cell, self.parent_selection_state.get(cell), features=features_by_brain[brain]))
            passed, reason = scoring.acceptance(self.parent_report, report, self.margin)
            entry.update(status='evaluated', train=metrics_only(report),
                         train_gate={'passed': passed, 'reason': reason, 'required_for_promotion': False},
                         ranking_delta={k: v['ranking_delta'] for k, v in report['cells'].items()},
                         in_sample_precision=in_sample_precision(report, self.target_kind),
                         selection=metrics_only(selection_report),
                         selection_gate=self._selection_gate(selection_report),
                         selection_ranking_delta={k: v['ranking_delta'] for k, v in selection_feedback['cells'].items()},
                         **candidate_metadata(component, self.target_kind, selection_report, selection_state,
                                              coverage=self.memory.coverage))
            entry['feature_evidence'] = self.memory.hypotheses.candidate_evidence(entry)
            result.update(report=report, state=state, selection_report=selection_report,
                          selection_state=selection_state)
            (trial_dir / 'train_evaluation.json').write_text(json.dumps(report, indent=2, allow_nan=False))
            save_state(trial_dir / 'train_state.npz', state)
            save_selection(trial_dir, selection_report, selection_state)
            feedback_path = trial_dir / 'feedback.json'
            write_train_feedback(feedback_path, report, [], self.budgets, target_kind=self.target_kind,
                                 report_role='candidate', selection=selection_feedback, protocol=self.protocol_info)
            public = {**entry, 'feedback': json.loads(feedback_path.read_text())}
            log_metrics(self.trace, f"TRAIN {entry['experiment']}", report, self.parent_report)
            self.trace.emit('selection_measurement', f"{entry['experiment']}: {self.protocol.mode} selection "
                            f"precision={selection_report['macro_precision']:.6f} "
                            f"(parent {self.parent_selection['macro_precision']:.6f}); "
                            f"in-sample {entry['in_sample_precision']:.6f}",
                            selection=entry['selection'], selection_gate=entry['selection_gate'])
        except Exception as exc:
            execution_error = stage in {'preflight', 'train_evaluation'} and isinstance(
                exc, (ScorerExecutionError, ModelExecutionError, SyntaxError, ValueError))
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
        self._record_research('evaluate_train', scoring_source(entry.get('train'), self.target_kind), 'measurement',
            {'program': program_identity((model or {}).get('program') or component)
                if entry['status'] == 'evaluated' else component_hash,
             'model_config': model['config'] if model else None,
             'train': self._train_identity(), 'k': self.budgets[self.target_kind]},
            entry['status'] == 'evaluated', trial_dir / 'result.json', cached=bool(cached),
            outcome={'precision': entry.get('target_precision'), 'in_sample_precision': entry.get('in_sample_precision'),
                     'error': entry.get('error')})
        self._save_hypothesis_feedback()
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
        # The official selection score ranks exploration candidates; in-sample TRAIN only
        # breaks ties. Only the outer validation mean controls promotion.
        return max(valid, key=lambda r: (r['entry']['target_precision'], r['report']['macro_precision'],
                                        -len(r['component']), -r['entry']['sequence']))

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
        if 'research' in result['entry']:
            proposal['research'] = deepcopy(result['entry']['research'])
        proposal['parameter_grid'] = {}  # A restored candidate represents one measured configuration.
        if model is not None and model['version'] == MODEL_VERSION:
            proposal['classifier'] = result['entry']['classifier']
        self.proposal_path.write_text(json.dumps(proposal, indent=2, allow_nan=False))
        self.restored_from = result['entry']['experiment']
        self._checkpoint(self.restored_from, result['component'])
        self.trace.emit('candidate_restored', f"Restored {self.restored_from}; no new evaluation",
                        experiment=self.restored_from, cached_from=result['entry'].get('cached_from'))
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
        if 'research' in entry:
            proposal['research'] = deepcopy(entry['research'])
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
                 'lineage': self._lineage(),
                 'status': 'training_error', 'failure_stage': 'classifier_fit', 'cached': False,
                 'error': f'{type(exc).__name__}: {exc}', 'train': None, 'train_gate': None, 'ranking_delta': None,
                 'evaluations_used': self.evaluations_used, 'wall_seconds': time.monotonic() - started}
        log = directory / 'worker.log'
        if log.is_file():
            with log.open('rb') as stream:
                stream.seek(max(0, log.stat().st_size - 8000))
                entry['training_output_tail'] = stream.read(8000).decode(errors='replace')
        (directory / 'result.json').write_text(json.dumps(entry, indent=2, allow_nan=False))
        (directory / 'training.py').write_text(program)
        self._edit_diffs(directory, None, program)
        result = {'entry': entry, 'report': None, 'state': None, 'source': None, 'component': None,
                  'public': {**entry, 'evaluations_remaining': self.remaining}}
        self.attempts.append(result)
        self.memory.append(entry)
        self._record_research('train_classifier', 'scoring', 'measurement',
            {'program': digest(program), 'config': config, 'train': self._train_identity()},
            False, directory / 'result.json', outcome={'error': entry['error']})
        self._save_hypothesis_feedback()
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
                                f"parameters={config['parameters']}; selection={self.protocol.mode}",
                                config=config, fit_snapshot=str(directory))
                oof_scores = None
                try:
                    frames = self._classifier_frames(program)
                    if self.protocol.requires_fold_fits:
                        # Fold fits come first; the official score never sees the full-fit model.
                        selection_report, selection_state = self.protocol.measure_classifier(
                            self.target_kind, program, config, directory / 'selection_folds', frames=frames,
                            fit_timeout=self.classifier_timeout, score_timeout=self.timeout,
                            memory_mb=self.classifier_memory_mb, threads=self.classifier_threads)
                        oof_scores = {cell: values['scores'] for cell, values in selection_state.items()}
                        save_selection(directory, selection_report, selection_state)
                        self.trace.emit('classifier_fold_fits', f"Program {digest(program)[:12]}: "
                                        f"{len(selection_report['fold_fits'])} fold fits; grouped_cv_precision="
                                        f"{selection_report['macro_precision']:.6f}",
                                        fold_fits=selection_report['fold_fits'])
                    component, summary = fit_classifier(self.train, self.target_kind, config, directory,
                                                        program=program, artifact_store=self.memory.artifact_store,
                                                        timeout=self.classifier_timeout,
                                                        memory_mb=self.classifier_memory_mb,
                                                        threads=self.classifier_threads, feature_frames=frames)
                    summary['selection_protocol'] = self.protocol.mode
                    if oof_scores is not None:
                        self.memory.oof_scores[digest(component)] = oof_scores
                        summary['selection_fold_fits'] = selection_report['fold_fits']
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
                if self.protocol.requires_fold_fits and digest(component) not in self.memory.oof_scores:
                    _, cached_state = load_selection(cached['snapshot'])
                    self.memory.oof_scores[digest(component)] = {cell: v['scores'] for cell, v in cached_state.items()}
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
                'training_metric_protocol': ('grouped_cv_precision' if self.protocol.requires_fold_fits
                                             else 'in_sample_resubstitution'),
                'selection_protocol': self.protocol.mode,
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

    def inspect_candidate(self, candidate_ref, radius_um=50., occurrence_index=0):
        if self.inspections_used >= 8:
            raise ValueError('The eight TRAIN neighborhood inspections for this generation have been used')
        if type(radius_um) not in (int, float) or not 1 <= radius_um <= 200:
            raise ValueError('radius_um must be a finite number in [1, 200]')
        brain, table, index = resolve_train_candidate(self.train, candidate_ref)
        provider = getattr(table, 'local_context', None)
        if provider is None:
            raise ValueError('No local fragment provider is attached to this TRAIN table')
        geometry, locations = provider.context(index, radius_um=radius_um, max_nodes=128,
                                               occurrence_index=occurrence_index)
        self.inspections_used += 1
        response = {'status': 'ok', 'candidate_ref': candidate_ref, 'brain': brain,
                    'occurrence_index': occurrence_index, 'locations_xyz_um': locations,
                    'geometry': geometry, 'inspections_remaining': 8 - self.inspections_used,
                    'note': 'TRAIN fragments after cache preprocessing. Coordinates in geometry are relative xyz um; '
                            'node and segment indices are local. Edges form an induced, possibly truncated subgraph. '
                            'No GT graph, image or predicted segmentation volume is included.'}
        directory = self.gen_dir / 'context_inspections'
        directory.mkdir(exist_ok=True)
        path = directory / f'inspection{self.inspections_used:03d}.json'
        path.write_text(json.dumps(response, allow_nan=False))
        self._record_research('inspect_candidate', 'local_geometry', 'observation',
            {'candidate_ref': candidate_ref, 'radius_um': radius_um, 'occurrence': occurrence_index,
             'geometry': geometry, 'locations': locations}, True, path)
        self.trace.emit('candidate_inspected', 'Inspected TRAIN fragment neighborhood',
                        candidate_ref=candidate_ref, brain=brain, occurrence_index=occurrence_index,
                        radius_um=radius_um, snapshot=str(path),
                        nodes=sum(len(site['xyz_um']) for site in geometry['sites']),
                        truncated=any(site['truncated'] for site in geometry['sites']))
        return response

    def inspect_failure_cases(self):
        """Inspect the saved matched panel in one call using the shared query budget."""
        results = []
        for pair in self.failure_cases['pairs']:
            if self.inspections_used + 2 > 8:
                break
            cases = []
            for case in pair['cases']:
                try:
                    context = self.inspect_candidate(case['candidate_ref'])
                except Exception as exc:
                    context = {'status': 'error', 'error': f'{type(exc).__name__}: {exc}'}
                    self._record_research('inspect_candidate', 'local_geometry', 'observation',
                        {'candidate_ref': case['candidate_ref'], 'radius_um': 50., 'occurrence': 0},
                        False, self.gen_dir / 'trajectory.jsonl', outcome=context['error'])
                    self.trace.emit('candidate_inspection_error', context['error'],
                                    candidate_ref=case['candidate_ref'])
                cases.append({**case, 'context': context})
            results.append({**pair, 'cases': cases})
        return {'status': 'inspected' if results else 'no_pairs_or_budget', 'pairs': results,
                'inspections_remaining': 8 - self.inspections_used, 'scope': self.failure_cases['scope']}

    def run_volume_analysis(self):
        """Run editable analysis.py on selected TRAIN volumes without fitting a policy."""
        if self.volume_analyses_used >= MAX_ANALYSES:
            raise ValueError(f'The {MAX_ANALYSES} volume-analysis executions for this generation have been used')
        program, request, resolved = read_analysis(self.gen_dir, self.train)
        self.volume_analyses_used += 1  # IO and worker failures count; invalid files/handles do not.
        analysis_id = f'analysis{self.volume_analyses_used:03d}'
        directory = self.gen_dir / 'volume_analyses' / analysis_id
        directory.mkdir(parents=True)
        (directory / 'analysis.py').write_text(program)
        (directory / 'analysis_request.json').write_text(json.dumps(request, indent=2, allow_nan=False))
        self.trace.emit('volume_analysis_start', f'{analysis_id}: analyzing {len(resolved)} TRAIN 3D patches',
                        analysis=analysis_id, program_sha256=digest(program), request=request)
        started = time.monotonic()
        try:
            response = execute_analysis(program, request, resolved, directory,
                timeout=self.timeout, memory_mb=self.classifier_memory_mb, threads=self.classifier_threads)
        except Exception as exc:
            response = {'status': 'error', 'error': f'{type(exc).__name__}: {exc}',
                        'wall_seconds': time.monotonic() - started}
        response.update(analysis=f'{self.gen_dir.name}/{analysis_id}',
                        volume_analyses_remaining=MAX_ANALYSES - self.volume_analyses_used,
                        evaluations_remaining=self.remaining)
        (directory / 'result.json').write_text(json.dumps(response, indent=2, allow_nan=False))
        manifest_path = directory / 'input_manifest.json'
        identity = json.loads(manifest_path.read_text()) if manifest_path.exists() else {'request': request}
        identity['program_sha256'] = program_identity(program)
        cases = response.get('cases', [])
        compared = {case['candidate_ref'] for case in cases if has_measurement(case.get('result'))
                    and case.get('valid_fraction', 0) > 0}
        self._record_research('run_volume_analysis', 'raw_image',
            'measurement' if compared else 'observation', identity,
            response['status'] == 'analyzed', directory / 'result.json', comparison=len(compared) >= 2,
            outcome={'status': response['status'], 'numerical_candidates': len(compared),
                     'error': response.get('error')})
        self.trace.emit('volume_analysis_result', f"{analysis_id}: {response['status']}; "
                        f'{MAX_ANALYSES - self.volume_analyses_used} analyses remaining', **response)
        return response

    def plan_image_scoring(self):
        """Dry-run the active image contract on TRAIN predictors without image IO."""
        from .image_contract import image_spec
        from .image_selection import select_image_rows, selection_coverage
        proposal = json.loads(self.proposal_path.read_text())
        source = self._read_program() if proposal.get('classifier') is not None else self._read_candidate()[0]
        contract = image_spec(source)
        if contract is None:
            raise ValueError('Declare LOCAL_IMAGE in scorer.py or training.py before planning image scoring')
        spec = contract[1]
        cells = {}
        for brain, bank in self.train.items():
            table = bank.tables[self.target_kind]
            selected = select_image_rows(table.features, spec, table.keys)
            cells[f'{brain}/{self.target_kind}'] = selection_coverage(table.features, spec, selected, table.keys)
        k = self.budgets[self.target_kind]
        warnings = []
        if spec['selection_mode'] != 'top_k_boundary':
            warnings.append('Top-only sampling does not deliberately cover candidates just outside K')
        elif spec['selection_k'] != k:
            warnings.append('selection_k differs from this run\'s scoring K')
        if spec['max_candidates'] < 2 * k:
            warnings.append('Coverage is a pilot near K, not the complete Top-K plus K outside candidates')
        report = {'status': 'planned', 'scope': 'TRAIN predictors only; no images read, no labels used, no score gain measured',
            'target_kind': self.target_kind, 'scoring_k': k, 'contract': spec, 'cells': cells,
            'warnings': warnings, 'evaluations_used': 0,
            'resource_note': 'Numeric extraction batches share one preparation/extraction timeout. '
                'Actual byte and cache limits are enforced during IO; raw-patch transport still has a 1 GiB limit.',
            'next_step': 'Implement the proposed 3D measurement as extract_image_features or a raw-patch model. '
                'Measure a pilot, then expand max_candidates in explore mode. Run an image-only removal '
                'ablation at the final coverage before fitting/scoring the submitted candidate.'}
        (self.gen_dir / 'image_scoring_plan.json').write_text(json.dumps(report, indent=2, allow_nan=False))
        self.trace.emit('image_scoring_plan', 'Planned label-free TRAIN image coverage', **report)
        return report

    def inspect_candidate_image(self, candidate_ref, radius_um=40., level=0, occurrence_index=0):
        """Return an actual image content block for a TRAIN-only candidate."""
        import base64
        from .image_preview import fragment_patch_geometry, render_preview
        if self.image_inspections_used >= 8:
            raise ValueError('Image inspection budget exhausted (eight requests per generation)')
        if (type(radius_um) not in (int, float) or not np.isfinite(radius_um) or not 4 <= radius_um <= 100
                or type(level) is not int or not 0 <= level <= 6 or type(occurrence_index) is not int):
            raise ValueError('Use radius_um 4..100, integer level 0..6 and a valid occurrence_index')
        brain, table, index = resolve_train_candidate(self.train, candidate_ref)
        provider = getattr(table, 'image_context', None)
        if provider is None:
            raise ValueError('TRAIN image access is not configured')
        self.image_inspections_used += 1  # Failed IO also consumes a request slot.
        spec = {'radius_um': radius_um, 'level': level, 'channel': 0, 'timepoint': 0}
        path, metadata = provider.patch(table, index, spec, occurrence_index)
        with np.load(path, allow_pickle=False) as data:
            patch = {key: data[key] for key in data.files}
        nodes, edges, segments = fragment_patch_geometry(table, index, spec, metadata, occurrence_index)
        directory = self.gen_dir / 'image_inspections'
        directory.mkdir(exist_ok=True)
        preview = directory / f'inspection{self.image_inspections_used:03d}.png'
        render_preview(preview, patch, nodes, edges, segments, f'TRAIN {brain} / {table.kind} / occurrence {occurrence_index}')
        # The host audit retains source identity. The tool exposes just the
        # local preview, registration facts and existing TRAIN inspection handle.
        response = {'candidate_ref': candidate_ref, 'brain': brain, 'kind': table.kind,
                    'radius_um': radius_um, 'level': level, 'occurrence': occurrence_index,
                    'shape_zyx': metadata['shape_zyx'], 'spacing_xyz_um': metadata['spacing_xyz_um'],
                    'anchors_zyx': patch['anchors_zyx'].tolist(), 'valid_fraction': metadata['valid_fraction'],
                    'image_inspections_remaining': 8 - self.image_inspections_used,
                    'note': 'Raw MIPs, fragment overlays and thin slabs. No GT overlay. '
                            'Display intensity scaling is separate from model pixels. Sampled alignment review is not a whole-brain guarantee.'}
        preview.with_suffix('.json').write_text(json.dumps({'response': response, 'metadata': metadata}, indent=2))
        self._record_research('inspect_candidate_image', 'raw_image', 'observation',
            {'candidate_ref': candidate_ref, 'spec': spec, 'occurrence': occurrence_index,
             'metadata': metadata}, True, preview.with_suffix('.json'))
        self.trace.emit('candidate_image_inspected', f'TRAIN {brain}/{table.kind}: saved {preview.name}',
                        candidate_ref=candidate_ref, preview=str(preview), level=level,
                        image_inspections_remaining=8 - self.image_inspections_used)
        return {'content': [{'type': 'text', 'text': json.dumps(response, allow_nan=False)},
                            {'type': 'image', 'mimeType': 'image/png',
                             'data': base64.b64encode(preview.read_bytes()).decode('ascii')}]}

    def inspect_failure_images(self):
        """At most two matched pairs per call, within the shared image allowance."""
        content = []
        for pair in self.failure_cases['pairs'][:2]:
            if self.image_inspections_used + 2 > 8:
                break
            content.append({'type': 'text', 'text': json.dumps(pair, allow_nan=False)})
            for case in pair['cases']:
                try:
                    content.extend(self.inspect_candidate_image(case['candidate_ref'])['content'])
                except Exception as exc:
                    content.append({'type': 'text', 'text': f'Image unavailable: {type(exc).__name__}: {exc}'})
                    self._record_research('inspect_candidate_image', 'raw_image', 'observation',
                        {'candidate_ref': case['candidate_ref'], 'radius_um': 40., 'level': 0, 'occurrence': 0},
                        False, self.gen_dir / 'trajectory.jsonl', outcome=f'{type(exc).__name__}: {exc}')
                    self.trace.emit('candidate_image_error', f'{type(exc).__name__}: {exc}',
                                    candidate_ref=case['candidate_ref'])
        return {'content': content or [{'type': 'text', 'text': 'No paired cases or remaining image budget'}]}

    def evaluate_feature_ablation(self):
        """Diagnostic only: never register a fold model as a submission candidate."""
        proposal = read_proposal(self.proposal_path)
        if proposal['parameter_grid']:
            raise ValueError('Feature ablation holds parameters fixed; use an empty parameter_grid')
        classifier = 'classifier' in proposal
        source = self._read_program() if classifier else self._read_candidate()[0]
        if not classifier and frozen_model(source) is not None:
            raise ValueError('Classifier ablation requires training.py and proposal.json.classifier for fold refits')
        if classifier:
            config = proposal['classifier']
            if (self.search_plan['mode'] == 'tune' and classifier_info(config, source)['formula_sha256']
                    != self.search_plan['search_parent']['formula_sha256']):
                raise ValueError('Tune mode fixes the training program; add new features in explore mode')
        else:
            self._check_mode(source)
            config = {'parameters': formula_info(source)[0]['parameters']}
        required = CLASSIFIER_UNITS if classifier else FORMULA_UNITS
        # Resolve exact cache identity before charging; no fold fit starts unless
        # the entire paired comparison fits the remaining configuration budget.
        number = len(self.feature_diagnostics) + 1
        directory = self.gen_dir / 'feature_ablations' / f'ablation{number:03d}'
        directory.mkdir(parents=True)
        (directory / 'program.py').write_text(source)
        report = {'version': ABLATION_VERSION, 'experiment': f'{self.gen_dir.name}/ablation{number:03d}',
                  'generation': self.generation,
                  'target_kind': self.target_kind,
                  'program_sha256': digest(source), 'parameters_sha256': config_identity(config),
                  'status': 'unavailable', 'scope': 'TRAIN-only diagnostic, not a submission',
                  'feature_columns': proposal.get('research', {}).get('feature_columns', [])}
        started, before = time.monotonic(), self.evaluations_used
        preparation_seconds = None
        frames = {}
        identity = {'program_sha256': program_identity(source), 'config': config,
                    'feature_columns': report['feature_columns'], 'train': self._train_identity()}
        self.trace.emit('feature_ablation_start', 'Paired TRAIN feature diagnostic',
                        experiment=report['experiment'], classifier=classifier, required_evaluations=required)
        try:
            frames, columns, split, identity = prepare_ablation(
                self.train, self.target_kind, source, report['feature_columns'], classifier=classifier,
                config=config, timeout=self.classifier_timeout if classifier else self.timeout,
                memory_mb=self.classifier_memory_mb,
                partitions=(self.protocol.partitions.get(self.target_kind)
                            or self.memory.internal_folds.get(self.target_kind)),
                selection_brains=self.protocol.selection_brains)
            preparation_seconds = time.monotonic() - started
            identity.update(requested_k=self.budgets[self.target_kind],
                            worker_limits={'fit_timeout': self.classifier_timeout,
                                'score_timeout': self.timeout, 'memory_mb': self.classifier_memory_mb,
                                'threads': self.classifier_threads})
            if classifier:
                self.memory.internal_folds[self.target_kind] = split
            signature = config_identity(identity)
            (directory / 'identity.json').write_text(json.dumps(identity, indent=2, allow_nan=False))
            report.update(feature_columns=columns, signature=signature, split=split[1])
            cached = self.memory.hypotheses.ablation_cache.get(signature)
            if cached is not None:
                report.update(deepcopy(cached), experiment=report['experiment'], generation=self.generation, cached=True,
                              cached_from=cached['experiment'])
            else:
                if self.remaining < required:
                    raise ValueError(f'Feature ablation requires {required} evaluations; {self.remaining} remain. '
                                     'Reserve one additional evaluation to fit/score a final candidate.')
                def charge():
                    if self.remaining < 1:
                        raise ValueError('Feature diagnostic evaluation budget exhausted')
                    self.evaluations_used += 1
                report.update(measure_ablation(self.train, self.target_kind, source, config, frames,
                    columns, split, identity, self.budgets, directory, classifier=classifier, charge=charge,
                    fit_timeout=self.classifier_timeout, score_timeout=self.timeout,
                    memory_mb=self.classifier_memory_mb, threads=self.classifier_threads))
        except Exception as exc:
            report.update(status='unavailable', error=f'{type(exc).__name__}: {exc}',
                          conclusion='No feature conclusion; diagnostic failed or could not be executed')
        report.update(wall_seconds=time.monotonic() - started, preparation_seconds=preparation_seconds,
                      evaluations_used=self.evaluations_used - before,
                      evaluations_remaining=self.remaining)
        (directory / 'proposal.json').write_text(json.dumps(proposal, indent=2, allow_nan=False))
        (directory / 'result.json').write_text(json.dumps(report, indent=2, allow_nan=False))
        self.feature_diagnostics.append(report)
        evidence_identity = deepcopy(identity)
        evidence_identity['program_sha256'] = program_identity(source)
        # Extraction cache keys include the raw source hash. Use pixel identities
        # for evidence novelty so comment-only edits cannot manufacture new inputs.
        for brain, values in evidence_identity.get('tables', {}).items():
            image_inputs = frames[brain].attrs.get('image_context', {})
            if 'input_sha256' in image_inputs:
                values['image_inputs'] = image_inputs['input_sha256']
        image = report.get('image_evidence') or {}
        evidence_source = ('raw_image' if image.get('all_image_inputs_removed_in_control')
                           else 'feature_comparison')
        self._record_research('evaluate_feature_ablation', evidence_source, 'measurement', evidence_identity,
            report['status'] == 'measured', directory / 'result.json', cached=report.get('cached', False),
            comparison=report['status'] == 'measured',
            outcome={'delta_precision': report.get('delta_precision'), 'error': report.get('error')})
        if not report.get('cached'):
            self.memory.hypotheses.observe_ablation({**proposal, 'target_kind': self.target_kind}, report)
        self._save_hypothesis_feedback()
        delta = report.get('delta_precision')
        detail = f'; delta Precision@K={delta:+.6f}' if delta is not None else ''
        self.trace.emit('feature_ablation_result', f"{report['experiment']}: {report['status']}{detail}; "
                        f"evaluation units={report['evaluations_used']}; remaining={self.remaining}", **report)
        return report

    # ---- Agent descriptors over the cached band -------------------------------------------
    @property
    def descriptor_seconds_remaining(self):
        return max(0., self.descriptor_budget['generation_seconds'] - self.descriptor_seconds_used)

    def _read_descriptor(self):
        if self.memory.descriptor_runs is None:
            raise ValueError('Descriptor computation is unavailable: this run has no context cache (--context-cache)')
        path = self.descriptor_path
        if not path.is_file() or path.is_symlink():
            raise ValueError('Write a regular descriptor.py in this generation')
        program, spec = descriptor_spec(path.read_text())
        if spec['kind'] != self.target_kind:
            raise ValueError(f'DESCRIPTOR.kind must be {self.target_kind!r} in this generation')
        for column in spec['columns']:
            existing = self.memory.descriptors.get(column)
            if existing is not None and existing['code_sha256'] != spec['code_sha256']:
                raise ValueError(f'{column} is already registered by different code in this run; choose new names')
        return program, spec

    def _descriptor_tables(self):
        return {brain: bank.tables[self.target_kind] for brain, bank in self.train.items()}

    def _descriptor_source(self, spec):
        return 'raw_image' if spec['inputs'] != 'geometry' else 'local_geometry'

    def plan_descriptor_run(self):
        """Time describe on a small sample and extrapolate each scope; no evaluation charge."""
        program, spec = self._read_descriptor()
        runs = self.memory.descriptor_runs
        brain = self.protocol.selection_brains[0]
        table = self.train[brain].tables[self.target_kind]
        per_call = self.descriptor_budget['per_call_seconds']
        plan = runs.plan(table, program, spec, k=self.budgets[self.target_kind], timeout=min(600., per_call))
        k = self.budgets[self.target_kind]
        allowance = min(per_call, self.descriptor_seconds_remaining)
        response = {'status': 'planned', 'code_sha256': spec['code_sha256'], 'columns': spec['columns'],
                    'inputs': spec['inputs'], 'image_tier': spec['image_tier'] if spec['inputs'] != 'geometry' else None,
                    'timed_on_brain': brain, **plan,
                    'budget': {'per_call_seconds': per_call, 'generation_seconds_remaining': self.descriptor_seconds_remaining,
                               'evaluations_remaining': self.remaining},
                    'fits_budget': {scope: plan['estimates'][scope]['estimated_seconds'] <= allowance for scope in plan['estimates']},
                    'cached': {scope: all(runs.cached(t, spec, scope, k) for t in self._descriptor_tables().values())
                               for scope in DESCRIPTOR_SCOPES},
                    'note': 'Estimates assume linear scaling across workers; an uncached compute_descriptors call costs one evaluation unit.'}
        directory = self.gen_dir / 'descriptor_plans'
        directory.mkdir(exist_ok=True)
        path = directory / f'plan{len(list(directory.glob("plan*.json"))) + 1:03d}.json'
        path.write_text(json.dumps(response, indent=2, allow_nan=False))
        self._record_research('plan_descriptor_run', self._descriptor_source(spec), 'observation',
            {'code_sha256': spec['code_sha256'], 'sample_rows': plan['sample_rows']}, True, path)
        estimates = ', '.join('%s: %.0fs' % (scope, item['estimated_seconds']) for scope, item in plan['estimates'].items())
        self.trace.emit('descriptor_plan', f"{spec['columns']}: {plan['seconds_per_row_single_worker']:.3f}s/row single worker; "
                        f"estimates {{{estimates}}}", **{k: v for k, v in response.items() if k != 'note'})
        return response

    def compute_descriptors(self, scope='pilot'):
        """Run describe over the cached band of every TRAIN brain; register the columns."""
        if scope not in DESCRIPTOR_SCOPES:
            raise ValueError(f'scope must be one of {DESCRIPTOR_SCOPES}')
        program, spec = self._read_descriptor()
        runs = self.memory.descriptor_runs
        tables = self._descriptor_tables()
        k = self.budgets[self.target_kind]
        cached = all(runs.cached(table, spec, scope, k) for table in tables.values())
        per_call = self.descriptor_budget['per_call_seconds']
        if not cached:
            if self.descriptor_seconds_remaining <= 0:
                raise ValueError('The per-generation descriptor wall budget is exhausted; use cached results or submit')
            if self.remaining <= 0:
                return {'status': 'budget_exhausted', 'evaluations_used': self.evaluations_used,
                        'message': 'No evaluation units remain for an uncached descriptor computation.'}
            self.evaluations_used += 1
        number = len(self.descriptor_log) + 1
        directory = self.gen_dir / 'descriptor_runs' / f'run{number:03d}'
        directory.mkdir(parents=True)
        (directory / 'descriptor.py').write_text(program)
        (directory / 'spec.json').write_text(json.dumps(spec, indent=2))
        started = time.monotonic()
        entry = {'run': f'{self.gen_dir.name}/descriptor{number:03d}', 'code_sha256': spec['code_sha256'],
                 'columns': spec['columns'], 'inputs': spec['inputs'],
                 'image_tier': spec['image_tier'] if spec['inputs'] != 'geometry' else None, 'scope': scope,
                 'target_kind': self.target_kind, 'status': 'running', 'cached': cached, 'charged': not cached,
                 'brains': sorted(tables)}
        self.trace.emit('descriptor_compute_start', f"{entry['run']}: {spec['columns']} over scope {scope} "
                        f"on {', '.join(sorted(tables))}; cached={cached}", **{k: v for k, v in entry.items()})
        results = {}
        try:
            for brain, table in sorted(tables.items()):
                allowance = per_call if cached else max(1., min(per_call, self.descriptor_seconds_remaining)
                                                        - (time.monotonic() - started))
                results[brain] = runs.compute(table, program, spec, scope, k=k, wall_budget=allowance,
                                              log_dir=directory / brain)
            for brain, table in tables.items():
                runs.register(table, spec['columns'], results[brain]['rows'], results[brain]['values'])
            self.memory.descriptor_programs[spec['code_sha256']] = {'program': program, 'spec': spec}
            for column in spec['columns']:
                self.memory.descriptors[column] = {'code_sha256': spec['code_sha256'], 'scope': scope,
                    'kind': self.target_kind, 'inputs': spec['inputs'], 'image_tier': entry['image_tier'],
                    'registered_in': self.gen_dir.name}
            self.memory.descriptor_generation += 1
            entry.update(status='registered',
                         summaries={brain: runs.summarize(table, spec['columns'], results[brain]['rows'], results[brain]['values'])
                                    for brain, table in tables.items()},
                         coverage={brain: {'computed_rows': int(len(r['rows'])), 'pool_rows': int(len(tables[brain].features)),
                                           'cached': r['cached'], 'wall_seconds': r['wall_seconds'],
                                           'workers': r['workers'], 'batches': r['batches']}
                                   for brain, r in results.items()},
                         selection_brains=list(self.protocol.selection_brains),
                         auxiliary_brains=list(self.protocol.auxiliary_brains))
        except Exception as exc:
            entry.update(status='execution_error' if isinstance(exc, (ModelExecutionError, ValueError)) else 'error',
                         error=f'{type(exc).__name__}: {exc}')
        entry['wall_seconds'] = time.monotonic() - started
        if not cached:
            self.descriptor_seconds_used += entry['wall_seconds']
        entry['descriptor_seconds_remaining'] = self.descriptor_seconds_remaining
        (directory / 'result.json').write_text(json.dumps(entry, indent=2, allow_nan=False))
        self.descriptor_log.append(entry)
        registered = entry['status'] == 'registered'
        self._record_research('compute_descriptors', self._descriptor_source(spec), 'measurement',
            {'code_sha256': spec['code_sha256'], 'scope': scope, 'columns': spec['columns'],
             'train': self._train_identity()}, registered, directory / 'result.json', cached=cached,
            comparison=registered, outcome={'status': entry['status'], 'error': entry.get('error')})
        self._save_hypothesis_feedback()
        self.trace.emit('descriptor_compute_result', f"{entry['run']}: {entry['status']}; "
                        f"{entry['wall_seconds']:.0f}s; evaluations remaining={self.remaining}; "
                        f"descriptor seconds remaining={self.descriptor_seconds_remaining:.0f}",
                        **{k: v for k, v in entry.items() if k != 'summaries'})
        public = {**entry, 'evaluations_used': self.evaluations_used, 'evaluations_remaining': self.remaining}
        if registered:
            public['next'] = ('Use the columns in scorer.py or training.py (NaN outside the cached band; '
                              'bank_context_available marks cached rows), or list them in research.feature_columns '
                              'and call evaluate_feature_ablation for the out-of-fold increment.')
        return public

    def required_descriptors(self, source):
        """Registered descriptor columns a (possibly bundled) scorer references."""
        try:
            parts = list(components(source).values())
        except Exception:
            parts = [source]
        names = set()
        for part in parts:
            model = None
            try:
                model = frozen_model(part)
            except (SyntaxError, ValueError):
                model = None
            if model is not None and model.get('columns') is not None:
                names.update(c for c in model['columns'] if c.startswith(COLUMN_PREFIX))
            else:
                names.update(referenced_columns((model or {}).get('program') if model else part))
        unknown = sorted(n for n in names if n not in self.memory.descriptors)
        if unknown:
            raise ValueError(f'Unregistered descriptor columns referenced: {unknown[:5]}; call compute_descriptors first')
        return sorted(names)

    def ensure_descriptors(self, tables_by_brain, columns, *, log_dir=None):
        """Compute referenced descriptors on other brains for frozen inference; no summaries, no feedback."""
        if not columns:
            return []
        runs = self.memory.descriptor_runs
        groups = {}
        for column in columns:
            record = self.memory.descriptors[column]
            groups[record['code_sha256']] = record['scope']
        reports = []
        for code, scope in groups.items():
            stored = self.memory.descriptor_programs[code]
            program, spec = stored['program'], stored['spec']
            for brain, bank in sorted(tables_by_brain.items()):
                table = bank.tables[spec['kind']]
                if all(c in table.features.columns for c in spec['columns']):
                    continue
                result = runs.compute(table, program, spec, scope, k=self.budgets[spec['kind']],
                                      wall_budget=self.descriptor_budget['per_call_seconds'],
                                      log_dir=(log_dir or self.gen_dir / 'descriptor_runs' / 'inference') / brain / code[:12])
                runs.register(table, spec['columns'], result['rows'], result['values'])
                reports.append({'brain': brain, 'code_sha256': code, 'scope': scope, 'rows': int(len(result['rows'])),
                                'cached': result['cached'], 'wall_seconds': result['wall_seconds']})
        return reports

    def descriptor_registry(self, columns=None):
        """JSON-safe registry (code included) for resumable scorers."""
        selected = self.memory.descriptors if columns is None else {c: self.memory.descriptors[c] for c in columns}
        codes = {record['code_sha256'] for record in selected.values()}
        return {'version': 'descriptor-registry-v1',
                'columns': {column: dict(record) for column, record in selected.items()},
                'programs': {code: {'program': self.memory.descriptor_programs[code]['program'],
                                    'spec': self.memory.descriptor_programs[code]['spec']} for code in codes}}

    def mcp_server(self):
        from claude_agent_sdk import create_sdk_mcp_server, tool

        async def invoke(function, arguments):
            async with self.lock:
                try:
                    response = await asyncio.to_thread(function, **arguments)
                except Exception as exc:
                    response = {'status': 'error', 'error': f'{type(exc).__name__}: {exc}',
                                'evaluations_remaining': self.remaining}
                    self._record_research(function.__name__, 'execution', 'failure',
                        {'arguments': arguments, 'error': response['error']}, False,
                        self.gen_dir / 'trajectory.jsonl', outcome=response['error'])
                receipt = {k: v for k, v in self.research_status().items() if k in (
                    'status', 'complete', 'investigation_complete', 'followup_complete',
                    'new_measurements', 'reused_results', 'execution_failures', 'requirement')}
                if isinstance(response, dict) and 'content' not in response:
                    response['research_status'] = receipt
                elif isinstance(response, dict):
                    response['content'].append({'type': 'text', 'text': json.dumps({'research_status': receipt})})
            return response if isinstance(response, dict) and 'content' in response else {
                'content': [{'type': 'text', 'text': json.dumps(response, allow_nan=False)}]}

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
              'the best successful model. Under grouped_oof selection each configuration is first refit in '
              'three host-owned folds and ranked by out-of-fold precision on the selection brains; the full '
              'fit is the frozen candidate. One configuration costs one evaluation unit, folds included. '
              'Validation is never supplied.', {})
        async def train_classifier(arguments):
            return await invoke(self.train_classifier, arguments)

        @tool('inspect_candidate', 'Inspect one TRAIN candidate from feedback.candidate_refs. '
              'Supply radius_um (1..200; normally 50) and occurrence_index (normally 0). '
              'Returns absolute location plus relative fragment geometry, at most 128 nodes. '
              'Eight inspections per generation, no evaluation charge. No GT or heldout access.',
              {'candidate_ref': str, 'radius_um': float, 'occurrence_index': int})
        async def inspect_candidate(arguments):
            return await invoke(self.inspect_candidate, arguments)

        @tool('inspect_failure_cases', 'Inspect matched TRAIN missed-positive/selected-label0 pairs from '
              'failure_cases.json in one call. Shares the eight-neighborhood budget. Call with {}.', {})
        async def inspect_failure_cases(arguments):
            return await invoke(self.inspect_failure_cases, arguments)

        @tool('run_volume_analysis', 'Primary 3D exploration tool. Execute analysis.py analyze(context) on '
              '1..4 TRAIN candidates selected in analysis_request.json. Full original 3D pixels, spacing, '
              'candidate anchors and aligned local fragments; arbitrary installed CPU analysis. '
              'Returns compact JSON results, not projections. Eight executions per generation, separate '
              'from scoring; no GT or validation access. Edit the two files and call with {}.', {})
        async def run_volume_analysis(arguments):
            return await invoke(self.run_volume_analysis, arguments)

        @tool('plan_image_scoring', 'Preview LOCAL_IMAGE selection and Top-K boundary coverage on TRAIN '
              'without reading images or spending evaluations. Reads training.py when proposal.classifier '
              'is present, otherwise scorer.py. Edit the literal contract and call with {} before '
              'expanding a 3D hypothesis to image features or raw-patch scoring.', {})
        async def plan_image_scoring(arguments):
            return await invoke(self.plan_image_scoring, arguments)

        @tool('inspect_candidate_image', 'Optional 2D preview of a TRAIN candidate with fragment overlays. '
              'Use run_volume_analysis for direct 3D analysis. '
              'Eight requests per generation; defaults radius_um=40, level=0, occurrence_index=0. '
              'Actual image content, no GT or heldout access.',
              {'candidate_ref': str, 'radius_um': float, 'level': int, 'occurrence_index': int})
        async def inspect_candidate_image(arguments):
            return await invoke(self.inspect_candidate_image, arguments)

        @tool('inspect_failure_images', 'Optional 2D previews of up to two matched TRAIN failure pairs with image/fragment '
              'overlays in one call; shares the eight-image budget. Call with {}.', {})
        async def inspect_failure_images(arguments):
            return await invoke(self.inspect_failure_images, arguments)

        @tool('evaluate_feature_ablation', 'Measure the incremental value of research.feature_columns '
              '(default all LOCAL_CONTEXT/LOCAL_IMAGE columns; image_available also removes raw patches). '
              'Classifiers refit the same program/parameters in '
              'three TRAIN-only grouped folds for full and constant-masked inputs: two evaluation units (one per arm). '
              'Formulas use two descriptive full-TRAIN score evaluations. Empty parameter_grid; call with {}. '
              'Diagnostic only: still fit/evaluate a normal candidate before submission.', {})
        async def evaluate_feature_ablation(arguments):
            return await invoke(self.evaluate_feature_ablation, arguments)

        @tool('plan_descriptor_run', 'Time descriptor.py describe(context) on 64 cached rows with one worker and '
              'extrapolate pilot/boundary/all_cached to the configured worker count and remaining wall budget. '
              'No evaluation charge. Call with {}.', {})
        async def plan_descriptor_run(arguments):
            return await invoke(self.plan_descriptor_run, arguments)

        @tool('compute_descriptors', 'Run descriptor.py describe(context) over the cached candidate band of every '
              'TRAIN brain in parallel sandboxed workers and register the results as bank_agent_<name> predictor '
              'columns. scope: pilot (512 rows), boundary (2000 rows each side of K) or all_cached (whole band). '
              'Returns label-conditional TRAIN quartiles, finite fractions and coverage. One uncached call costs '
              'one evaluation unit plus wall time; results are cached by code hash across generations and runs.',
              {'scope': str})
        async def compute_descriptors(arguments):
            return await invoke(self.compute_descriptors, arguments)

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
                entry['hypothesis_evidence'] = self.memory.hypotheses.snapshot(
                    self.target_kind, entry.get('hypothesis', ''), limit=1)
            return {'content': [{'type': 'text', 'text': json.dumps(response, allow_nan=False)}]}

        return create_sdk_mcp_server('training', tools=[evaluate_train, search_parameters,
            train_classifier, inspect_candidate, inspect_failure_cases, evaluate_feature_ablation,
            inspect_candidate_image, inspect_failure_images,
            run_volume_analysis, plan_image_scoring,
            plan_descriptor_run, compute_descriptors,
            restore_candidate, search_memory])
