"""Synthetic regression cases for feature diagnostics; no external data or APIs."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from proofreader_evolve.harness.candidate_pool import CandidatePool
from proofreader_evolve.harness.classifier_contract import config_identity
from proofreader_evolve.harness.classifier_training import fit_classifier
from proofreader_evolve.harness.failure_cases import matched_failure_cases
from proofreader_evolve.harness.feature_ablation import measure_ablation
from proofreader_evolve.harness.hypothesis_memory import HypothesisMemory, identity
from proofreader_evolve.harness.internal_validation import grouped_folds
from proofreader_evolve.harness.local_context import resolve_train_candidate
from proofreader_evolve.harness.native_pool import NativeTable, NativeTables, pool_digest
from proofreader_evolve.harness.search_proposals import read_proposal
from proofreader_evolve.harness.train_experiments import ExperimentMemory, TrainingExperiments


PROGRAM = ('def fit(X_train, y_train, artifact_dir, params):\n    pass\n'
           'def predict(X, artifact_dir, params):\n    return X["local_signal"].to_numpy()\n')
FORMULA = 'def score_candidates(features, ctx):\n    return features["detector_score"].to_numpy()\n'
RESEARCH = {'hypothesis_id': 'continuity', 'failure_mode': 'Similar gaps yield different labels.',
            'information_source': 'local_geometry', 'prediction': 'Continuity recovers missed positives.',
            'feature_columns': ['local_signal']}


def training_fixture(kind='merge'):
    rows, labels, xyz = [], [], []
    for group in range(6):
        for offset in range(2 if kind == 'merge' else 3):
            index = len(rows)
            if kind == 'merge':
                rows.append({'segment_id': group + 1, 'node_id': index, 'degree': 3})
            else:
                rows.append({'segment_id_a': 999 if offset == 2 else 10 * group + offset + 1,
                             'segment_id_b': 2000 + index, 'node_id_a': 2 * index,
                             'node_id_b': 2 * index + 1, 'gap_um': 10.})
                xyz.extend([[1000. * group + 10, 0, 0], [1000. * group + 20, 0, 0]])
            labels.append(int(offset != 0))
    table = NativeTable(kind, pd.DataFrame({'detector_score': np.linspace(1., 0., len(rows)),
                                           'local_signal': np.asarray(labels, dtype=float)}),
                        rows, np.asarray(labels, dtype=np.int8),
                        {'pool_sha256': pool_digest(kind, rows), 'provenance': {'brain': '794495'}})
    if kind == 'split':
        store = SimpleNamespace(graph=Mock(return_value=(SimpleNamespace(node_xyz=np.asarray(xyz)), None)))
        table.local_context = SimpleNamespace(store=store)
    return {'794495': NativeTables('794495', {kind: table}, {})}


def entry(number, precision=.2, kind='merge', hypothesis='continuity'):
    return {'experiment': f'gen001/attempt{number:03d}', 'sequence': number, 'target_kind': kind,
            'component_sha256': f'component-{number}', 'top_k_sha256': f'ranking-{number}',
            'formula_sha256': 'formula', 'parameters': {'weight': 1.}, 'family': 'geometry',
            'status': 'evaluated', 'cached': False, 'target_precision': precision,
            'hypothesis': 'Continuity improves ranking.', 'research': {**RESEARCH, 'hypothesis_id': hypothesis}}


class FeatureDiscoveryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_optional_research_and_invalid_declarations(self):
        path = self.root / 'proposal.json'
        proposal = {'hypothesis': 'Geometry helps.', 'strategy': 'Measure geometry.'}
        path.write_text(json.dumps(proposal))
        self.assertNotIn('research', read_proposal(path))
        proposal['research'] = RESEARCH
        path.write_text(json.dumps(proposal))
        self.assertEqual(read_proposal(path)['research'], RESEARCH)
        # Declarations are bounded, never rejected (2026-10-08): duplicates and unknown keys are cleaned up.
        proposal['research'] = {**RESEARCH, 'feature_columns': ['same', 'same'], 'outcome': 'extra'}
        path.write_text(json.dumps(proposal))
        self.assertEqual(read_proposal(path)['research'], {**RESEARCH, 'feature_columns': ['same']})
        proposal['research'] = {'prediction': 'no hypothesis id'}
        path.write_text(json.dumps(proposal))
        self.assertNotIn('research', read_proposal(path))

    def test_failure_panel_contains_only_resolvable_train_roles(self):
        train = training_fixture()
        table = train['794495'].tables['merge']
        selected = np.array([0, 2, 4, 6])
        panel = matched_failure_cases(train, 'merge', {'794495/merge': {
            'scores': table.features.detector_score.to_numpy(), 'chosen': selected}})
        self.assertEqual(len(panel['pairs']), 4)
        references = set()
        for pair in panel['pairs']:
            for case in pair['cases']:
                brain, resolved, index = resolve_train_candidate(train, case['candidate_ref'])
                self.assertEqual(brain, '794495')
                self.assertIs(resolved, table)
                self.assertNotIn(case['candidate_ref'], references)
                references.add(case['candidate_ref'])
                if case['role'] == 'missed_positive':
                    self.assertEqual(table.truth[index], 1)
                    self.assertNotIn(index, selected)
                else:
                    self.assertEqual(table.truth[index], 0)
                    self.assertIn(index, selected)

    def test_grouped_folds_cover_pool_once_and_purge_shared_fragments(self):
        for kind in ('merge', 'split'):
            with self.subTest(kind=kind):
                train = training_fixture(kind)
                table = train['794495'].tables[kind]
                folds, description = grouped_folds(train, kind)
                coverage = np.zeros(len(table.truth), dtype=int)
                fields = ('segment_id',) if kind == 'merge' else ('segment_id_a', 'segment_id_b')
                for fitting, held in folds:
                    a, b = fitting['794495'], held['794495']
                    coverage[b] += 1
                    fit_fragments = {table.candidates[i][f] for i in a for f in fields}
                    held_fragments = {table.candidates[i][f] for i in b for f in fields}
                    self.assertFalse(fit_fragments & held_fragments)
                    self.assertEqual(set(table.truth[a]), {0, 1})
                np.testing.assert_array_equal(coverage, np.ones(len(coverage), dtype=int))
                if kind == 'split':
                    self.assertTrue(all(f['brains']['794495']['purged_rows'] > 0
                                        for f in description['fold_details']))
                original = description['partition_sha256']
                table.truth = 1 - table.truth
                self.assertEqual(grouped_folds(train, kind)[1]['partition_sha256'], original)

    def test_unusable_folds_do_not_fall_back_to_row_randomization(self):
        train = training_fixture()
        table = train['794495'].tables['merge']
        for row in table.candidates:
            row['segment_id'] = 1
        with self.assertRaisesRegex(ValueError, 'independent groups'):
            grouped_folds(train, 'merge')

    def test_fit_transport_excludes_internal_held_rows_and_labels(self):
        train = training_fixture()
        table = train['794495'].tables['merge']
        fitting = {'794495': np.array([0, 3, 6, 9])}
        def worker(root, request, **kwargs):
            np.testing.assert_array_equal(np.load(root / 'X.npy'), table.features.iloc[fitting['794495']])
            np.testing.assert_array_equal(np.load(root / 'y.npy'), table.truth[fitting['794495']])
            self.assertEqual(request['random_seed'], 2)
            (root / 'artifacts' / 'model.bin').write_bytes(b'synthetic opaque artifact')
            return {'wall_seconds': 0.}
        with patch('proofreader_evolve.harness.classifier_training.run_worker', side_effect=worker):
            _, summary = fit_classifier(train, 'merge', {'parameters': {}}, self.root / 'fit',
                program=PROGRAM, artifact_store=self.root / 'models', feature_frames={'794495': table.features},
                fit_rows=fitting, random_seed=2)
        self.assertEqual(summary['training_rows'], 4)

    def test_paired_classifier_refits_both_arms_with_same_folds_and_seeds(self):
        train = training_fixture()
        table = train['794495'].tables['merge']
        frames = {'794495': table.features.copy()}
        split = grouped_folds(train, 'merge')
        calls, units = [], []
        def fit(*args, **kwargs):
            calls.append({'rows': kwargs['fit_rows']['794495'].copy(), 'seed': kwargs['random_seed'],
                          'frame': kwargs['feature_frames']['794495'].copy()})
            self.assertIn('fold', str(kwargs['artifact_store']))
            return 'mock frozen model', {}
        def predict(model, store, frame, kind, **kwargs):
            self.assertEqual(list(frame.columns), list(table.features.columns))
            self.assertEqual(list(frame.index), list(range(len(frame))))
            return frame.local_signal.to_numpy()
        with patch('proofreader_evolve.harness.feature_ablation.fit_classifier', side_effect=fit), \
                patch('proofreader_evolve.harness.feature_ablation.frozen_model', return_value={}), \
                patch('proofreader_evolve.harness.feature_ablation.predict_model', side_effect=predict):
            report = measure_ablation(train, 'merge', PROGRAM, {'parameters': {}}, frames, ['local_signal'],
                split, {'program_sha256': 'program', 'parameters_sha256': 'params'}, {'merge': 3},
                self.root / 'comparison', classifier=True, charge=lambda: units.append(1))
        self.assertEqual(len(units), 2)  # one unit per arm; a fold triple is one measurement
        self.assertEqual(report['protocol'], 'TRAIN_grouped_out_of_fold_partitioned_k')
        self.assertEqual(report['scored_brains'], ['794495'])
        self.assertEqual(report['required_evaluations'], 2)
        self.assertGreater(report['delta_precision'], 0.)
        for full, removed in zip(calls[:3], calls[3:]):
            np.testing.assert_array_equal(full['rows'], removed['rows'])
            self.assertEqual(full['seed'], removed['seed'])
            np.testing.assert_array_equal(removed['frame'].local_signal, 0.)
            np.testing.assert_array_equal(full['frame'].detector_score, removed['frame'].detector_score)
        pd.testing.assert_frame_equal(frames['794495'], table.features)

    def test_insufficient_budget_never_starts_fold_fit_or_registers_candidate(self):
        train = training_fixture()
        table = train['794495'].tables['merge']
        gen = self.root / 'gen001'
        gen.mkdir()
        policy, rules = gen / 'scorer.py', gen / 'rules.md'
        policy.write_text(FORMULA)
        rules.write_text('Synthetic hypothesis.')
        (gen / 'training.py').write_text(PROGRAM)
        (gen / 'proposal.json').write_text(json.dumps({'hypothesis': 'Continuity helps.',
            'strategy': 'Compare features.', 'research': RESEARCH, 'classifier': {'parameters': {}}}))
        memory = ExperimentMemory(self.root)
        session = TrainingExperiments(gen, policy, rules, 'merge', {'merge': FORMULA}, train, {},
            {'794495/merge': {'scores': table.features.detector_score.to_numpy(), 'chosen': np.arange(3)}},
            {'merge': 3}, 120, 0, memory, max_evaluations=1)
        prepared = ({'794495': table.features}, ['local_signal'], grouped_folds(train, 'merge'), {'fixture': 1})
        with patch('proofreader_evolve.harness.train_experiments.prepare_ablation', return_value=prepared), \
                patch('proofreader_evolve.harness.train_experiments.measure_ablation') as measure:
            report = session.evaluate_feature_ablation()
        measure.assert_not_called()
        self.assertEqual(report['status'], 'unavailable')
        self.assertEqual(session.evaluations_used, 0)
        self.assertFalse(memory.candidates)
        self.assertEqual(policy.read_text(), FORMULA)

    def test_hypothesis_progress_ignores_cache_failure_and_counts_early_best(self):
        memory = HypothesisMemory(self.root / 'memory.json')
        first = entry(1)
        memory.observe(first)
        memory.finish_generation(1, [first])
        attempts = [entry(i, precision=.3 if i == 2 else .2) for i in range(2, 9)]
        for trial in attempts:
            memory.observe(trial)
        memory.finish_generation(2, attempts)
        record = memory.records[identity(first)]
        self.assertEqual(record['stalled_generations'], 0)
        self.assertEqual(record['best_train_precision'], .3)
        for generation in (3, 4):
            trial = entry(generation + 20)
            memory.observe(trial)
            memory.finish_generation(generation, [trial])
        self.assertEqual(memory.penalty(first), 2)
        cached, failed = {**entry(90), 'cached': True}, {**entry(91), 'status': 'training_error'}
        memory.observe(cached)
        memory.observe(failed)
        memory.finish_generation(5, [cached, failed])
        self.assertEqual(record['stalled_generations'], 2)
        # Identical source on a different kind is still a fresh measurement.
        other = entry(1, kind='split')
        memory.observe(other)
        self.assertEqual(memory.records[identity(other)]['fresh_measurements'], 1)

    def test_ablation_evidence_binds_code_and_parameters_and_updates_soft_penalty(self):
        memory = HypothesisMemory(self.root / 'memory.json')
        candidate = {**entry(1), 'program_sha256': hashlib.sha256(PROGRAM.encode()).hexdigest(),
                     'classifier': {'parameters': {'seed': 0}}}
        for number in range(3):
            report = {'experiment': f'ablation{number}', 'signature': str(number), 'status': 'measured',
                      'target_kind': 'merge',
                      'program_sha256': candidate['program_sha256'],
                      'parameters_sha256': config_identity(candidate['classifier']),
                      'protocol': 'TRAIN_grouped_out_of_fold', 'feature_columns': ['local_signal'],
                      'delta_precision': .01 if number == 2 else 0., 'conclusion': 'Synthetic evidence.'}
            memory.observe_ablation(candidate, report)
            if number == 1:
                self.assertEqual(memory.penalty(candidate), 2)
                self.assertIn('continuity', memory.advice('merge')['deprioritized_hypotheses'])
        self.assertEqual(memory.penalty(candidate), 0)
        self.assertEqual(len(memory.candidate_evidence(candidate)), 3)
        changed = deepcopy(candidate)
        changed['classifier']['parameters']['seed'] = 1
        self.assertFalse(memory.candidate_evidence(changed))
        self.assertFalse(memory.candidate_evidence({**candidate, 'program_sha256': 'different'}))
        self.assertFalse(memory.candidate_evidence({**candidate, 'target_kind': 'split'}))

    def test_stagnant_hypothesis_yields_to_alternative_without_displacing_reference(self):
        memory = HypothesisMemory(self.root / 'memory.json')
        champion, alternative = entry(1, .3), entry(2, .29, hypothesis='other-mechanism')
        memory.observe(champion)
        memory.observe(alternative)
        memory.records[identity(champion)]['stalled_generations'] = 2
        pool = CandidatePool(self.root / 'pool.json', ['merge'], hypotheses=memory, early_stop=False)
        pool.add(champion)
        pool.add(alternative)
        pool.set_reference(champion)
        plan = pool.next_plan()
        self.assertEqual(plan['reason'], 'hypothesis_plateau')
        self.assertEqual(plan['search_parent']['experiment'], alternative['experiment'])
        pool.next_plan()
        self.assertEqual(pool.next_plan()['branch_role'], 'reference')


if __name__ == '__main__':
    unittest.main()
