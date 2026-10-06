"""Grouped out-of-fold selection on detector-naive brains; synthetic fixtures only.

Covers fold roles (selection vs auxiliary), partitioned-K ranking, the official
selection score ranking a memorizing model below a generalizing one, one
evaluation unit per classifier configuration, feedback sourced from the
selection measurement, bootstrap logging and the driver's brain-role checks.
"""

import asyncio
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from proofreader_evolve.harness import fixed_pool_scoring as scoring
from proofreader_evolve.harness import selection_protocol as sp
from proofreader_evolve.harness.classifier_training import fit_classifier
from proofreader_evolve.harness.internal_validation import fold_assignments, grouped_folds
from proofreader_evolve.harness.native_pool import NativeTable, NativeTables, pool_digest
from proofreader_evolve.harness.scorer_components import components
from proofreader_evolve.harness.train_experiments import ExperimentMemory, TrainingExperiments
from proofreader_evolve.harness import train_feedback as feedback


FORMULA = 'def score_candidates(features, ctx):\n    return features["detector_score"].to_numpy(dtype=float)\n'
SIGNAL_FORMULA = 'def score_candidates(features, ctx):\n    return features["signal"].fillna(0).to_numpy(dtype=float)\n'
GENERALIZER = '''import numpy as np
def fit(X_train, y_train, artifact_dir, params):
    np.save(artifact_dir / 'scale.npy', np.array([params.get('scale', 1.0)]), allow_pickle=False)
def predict(X, artifact_dir, params):
    scale = np.load(artifact_dir / 'scale.npy', allow_pickle=False)[0]
    return scale * X['signal'].fillna(0).to_numpy(dtype=float)
'''
# Perfect on rows it has seen, systematically wrong on rows it has not.
MEMORIZER = '''import numpy as np
def fit(X_train, y_train, artifact_dir, params):
    np.save(artifact_dir / 'keys.npy', X_train['row_key'].to_numpy(dtype=float), allow_pickle=False)
    np.save(artifact_dir / 'labels.npy', np.asarray(y_train, dtype=float), allow_pickle=False)
def predict(X, artifact_dir, params):
    keys = np.load(artifact_dir / 'keys.npy', allow_pickle=False)
    labels = np.load(artifact_dir / 'labels.npy', allow_pickle=False)
    lookup = dict(zip(keys.tolist(), labels.tolist()))
    seen = np.array([lookup.get(float(k), np.nan) for k in X['row_key'].fillna(-1)])
    return np.where(np.isfinite(seen), seen + 2.0, -X['signal'].fillna(0).to_numpy(dtype=float))
'''
GROUPED = '''import numpy as np
def fit(X_train, y_train, artifact_dir, params, groups=None):
    names = sorted(set(groups.tolist())) if groups is not None else []
    (artifact_dir / 'groups.txt').write_text(','.join(names))
def predict(X, artifact_dir, params):
    return X['signal'].fillna(0).to_numpy(dtype=float)
'''


def brain_table(brain, *, groups=8, offset=0.):
    rows, labels, signal, key = [], [], [], []
    for group in range(groups):
        for position in range(2):
            index = len(rows)
            label = position  # one negative and one positive per fragment group
            rows.append({'segment_id': group + 1, 'node_id': index, 'degree': 3})
            labels.append(label)
            signal.append(label + .1 * ((index % 3) - 1))
            key.append(offset + index)
    frame = pd.DataFrame({'detector_score': np.linspace(1., 0., len(rows)),
                          'signal': np.asarray(signal, dtype=float),
                          'row_key': np.asarray(key, dtype=float)})
    table = NativeTable('merge', frame, rows, np.asarray(labels, dtype=np.int8),
                        {'pool_sha256': pool_digest('merge', rows), 'training_brain': '794495',
                         'provenance': {'brain': brain}})
    return NativeTables(brain, {'merge': table}, {})


def two_brains():
    return {'794495': brain_table('794495', offset=1000.), '802449': brain_table('802449')}


class FoldRoleTests(unittest.TestCase):
    def test_auxiliary_rows_join_every_fit_and_are_never_held(self):
        train = two_brains()
        folds, description = grouped_folds(train, 'merge', selection_brains=['802449'])
        self.assertEqual(description['selection_brains'], ['802449'])
        self.assertEqual(description['auxiliary_brains'], ['794495'])
        held = np.zeros(16, dtype=int)
        for fitting, held_rows in folds:
            np.testing.assert_array_equal(fitting['794495'], np.arange(16))
            self.assertEqual(len(held_rows['794495']), 0)
            held[held_rows['802449']] += 1
            self.assertFalse(set(fitting['802449']) & set(held_rows['802449']))
        np.testing.assert_array_equal(held, 1)
        for detail in description['fold_details']:
            self.assertEqual(detail['brains']['794495']['role'], 'auxiliary')
            self.assertEqual(detail['brains']['794495']['held_rows'], 0)
            self.assertEqual(detail['brains']['802449']['role'], 'selection')
        ids = fold_assignments(folds, '802449', 16)
        self.assertTrue((ids >= 0).all())
        np.testing.assert_array_equal(fold_assignments(folds, '794495', 16), -1)
        # Balancing counts only selection groups; roles change the partition identity.
        both = grouped_folds(train, 'merge')[1]
        self.assertNotEqual(both['partition_sha256'], description['partition_sha256'])
        with self.assertRaisesRegex(ValueError, 'subset'):
            grouped_folds(train, 'merge', selection_brains=['999'])


class PartitionedRankingTests(unittest.TestCase):
    def test_budgets_sum_to_k_and_respect_fold_sizes(self):
        self.assertEqual(sp.fold_budgets([6, 5, 5], 3), [1, 1, 1])
        self.assertEqual(sum(sp.fold_budgets([700, 650, 650], 2000)), 2000)
        self.assertEqual(sp.fold_budgets([1, 1, 50], 10), [0, 0, 10])  # proportional, tiny folds get none
        self.assertEqual(sp.fold_budgets([2, 2], 10), [2, 2])
        with self.assertRaises(ValueError):
            sp.fold_budgets([1], 0)

    def test_scores_are_ranked_inside_folds_not_pooled(self):
        # Fold 0 emits large scores, fold 1 small ones; pooling would pick only fold 0.
        scores = np.array([10., 9., 8., .3, .2, .1])
        truth = np.array([0, 0, 1, 1, 1, 0])
        keys = [f'k{i}' for i in range(6)]
        folds = np.array([0, 0, 0, 1, 1, 1])
        metrics, chosen = sp.partitioned_rank_metrics(scores, truth, keys, folds, 2)
        self.assertEqual(metrics['fold_budgets'], [1, 1])
        self.assertEqual(sorted(chosen.tolist()), [0, 3])
        self.assertEqual(metrics['tp'], 1)
        self.assertEqual(metrics['effective_k'], 2)
        pooled, _ = scoring.rank_metrics(scores, truth, keys, 2)
        self.assertEqual(pooled['tp'], 0)
        with self.assertRaisesRegex(ValueError, 'fold assignment'):
            sp.partitioned_rank_metrics(scores, truth, keys, folds - 1, 2)


class ProtocolMeasurementTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.train = two_brains()
        self.protocol = sp.SelectionProtocol(self.train, {'merge': 3}, selection_brains=['802449'])

    def test_formula_uses_selection_cells_and_same_partition(self):
        state = {}
        scoring.evaluate(SIGNAL_FORMULA, self.train, {'merge': 3}, state=state)
        report, selection_state = self.protocol.measure_formula('merge', state)
        self.assertEqual(list(report['cells']), ['802449/merge'])
        self.assertEqual(report['macro_precision'], 1.)
        self.assertEqual(report['cells']['802449/merge']['partition_sha256'], self.protocol.partition_sha256('merge'))
        self.assertEqual(len(selection_state['802449/merge']['chosen']), 3)
        legacy = sp.SelectionProtocol(self.train, {'merge': 3}, mode='in_sample')
        self.assertEqual(legacy.selection_brains, ['794495', '802449'])
        self.assertEqual(set(legacy.measure_formula('merge', state)[0]['cells']), {'794495/merge', '802449/merge'})
        with self.assertRaisesRegex(ValueError, 'grouped_oof'):
            legacy.measure_classifier('merge', GENERALIZER, {'parameters': {}}, self.root)
        info = self.protocol.describe('merge')
        self.assertEqual(info['partitions']['merge']['fold_budgets']['802449/merge'], [1, 1, 1])
        self.assertEqual(info['auxiliary_train_brains'], ['794495'])

    def test_fold_fits_exclude_held_rows_and_can_drop_auxiliary_rows(self):
        calls = []
        def fit(train, kind, config, directory, *, program, artifact_store, fit_rows, random_seed, **kwargs):
            calls.append((dict((b, r.copy()) for b, r in fit_rows.items()), random_seed, str(artifact_store)))
            return 'frozen', {'training_rows': sum(len(r) for r in fit_rows.values()), 'wall_seconds': 0.}
        def predict(model, store, frame, kind, **kwargs):
            return frame['signal'].to_numpy(dtype=float)
        with patch.object(sp, 'fit_classifier', side_effect=fit), patch.object(sp, 'frozen_model', return_value={}), \
                patch.object(sp, 'predict_model', side_effect=predict):
            report, state = self.protocol.measure_classifier('merge', GENERALIZER, {'parameters': {}}, self.root / 'a',
                                                             frames={b: bank.tables['merge'].features
                                                                     for b, bank in self.train.items()})
            reduced, _ = self.protocol.measure_classifier('merge', GENERALIZER, {'parameters': {}}, self.root / 'b',
                                                          frames={b: bank.tables['merge'].features
                                                                  for b, bank in self.train.items()},
                                                          include_auxiliary=False)
        self.assertEqual(report['macro_precision'], 1.)
        self.assertEqual(len(report['fold_fits']), 3)
        self.assertEqual(report['fitting_brains'], ['794495', '802449'])
        self.assertEqual(reduced['fitting_brains'], ['802449'])
        folds = self.protocol.partitions['merge'][0]
        for (rows, seed, store), (fitting, held) in zip(calls[:3], folds):
            np.testing.assert_array_equal(rows['794495'], np.arange(16))
            np.testing.assert_array_equal(rows['802449'], fitting['802449'])
            self.assertFalse(set(rows['802449']) & set(held['802449']))
            self.assertIn('selection_folds', store) if 'selection_folds' in store else None
        self.assertEqual([seed for _, seed, _ in calls], [0, 1, 2, 0, 1, 2])
        for rows, _, _ in calls[3:]:
            self.assertEqual(len(rows['794495']), 0)
        self.assertTrue(np.isfinite(state['802449/merge']['scores']).all())


class SessionSelectionTests(unittest.TestCase):
    """Real sandboxed fits on tiny data; Linux Landlock/seccomp required as elsewhere."""
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        redirect = redirect_stdout(io.StringIO())
        redirect.__enter__()
        self.addCleanup(redirect.__exit__, None, None, None)
        self.train = two_brains()
        self.state = {}
        self.parent = scoring.evaluate(FORMULA, self.train, {'merge': 3}, state=self.state)
        protocol = sp.SelectionProtocol(self.train, {'merge': 3}, selection_brains=['802449'])
        self.memory = ExperimentMemory(self.root, train=self.train, protocol=protocol)
        gen = self.root / 'gen001'
        gen.mkdir()
        policy, rules = gen / 'scorer.py', gen / 'rules.md'
        policy.write_text(FORMULA)
        rules.write_text('fixture')
        self.session = TrainingExperiments(gen, policy, rules, 'merge', {'merge': FORMULA}, self.train,
                                           self.parent, self.state, {'merge': 3}, 120, 0, self.memory,
                                           max_evaluations=4, generation=1)

    def proposal(self, program, scale=1.0):
        self.session.training_path.write_text(program)
        self.session.proposal_path.write_text(json.dumps({
            'hypothesis': 'Signal ranks positives', 'strategy': 'Fit on TRAIN', 'family': 'signal',
            'classifier': {'parameters': {'scale': scale}}, 'parameter_grid': {}}))

    def test_memorizer_loses_to_generalizer_under_out_of_fold_selection(self):
        self.assertEqual(set(self.memory.coverage._cells), {'802449/merge'})
        self.assertEqual(list(self.session.parent_selection['cells']), ['802449/merge'])
        self.proposal(MEMORIZER)
        memorized = self.session.train_classifier()
        self.assertEqual(memorized['status'], 'trained')
        self.assertEqual(memorized['training_metric_protocol'], 'grouped_cv_precision')
        self.assertEqual(self.session.evaluations_used, 1)  # folds plus full fit are one unit
        first = self.session.attempts[0]['entry']
        self.assertEqual(first['in_sample_precision'], 1.)
        self.assertLess(first['target_precision'], 1.)
        self.assertEqual(first['selection']['cells']['802449/merge']['fold_budgets'], [1, 1, 1])
        self.assertEqual(list(first['selection']['cells']), ['802449/merge'])
        self.assertTrue(all(name.startswith('802449/merge/') for name in first['specialist_profile']['slices']))
        fit_dir = Path(first['training']['fit_snapshot'])
        self.assertTrue(all((fit_dir / 'selection_folds' / f'fold{i}' / 'model.json').is_file() for i in range(3)))
        self.assertTrue((Path(first['snapshot']) / 'selection_state.npz').is_file())
        self.assertFalse((self.memory.artifact_store / 'selection_folds').exists())
        self.proposal(GENERALIZER)
        generalized = self.session.train_classifier()
        second = self.session.attempts[1]['entry']
        self.assertEqual(second['target_precision'], 1.)
        self.assertEqual(second['selection_gate']['metric'], 'grouped_cv_precision')
        self.assertTrue(second['selection_gate']['passed'])
        self.assertEqual(self.session.evaluations_used, 2)
        best = self.session.restore('best')
        self.assertEqual(best['restored'], second['experiment'])
        submitted, _ = self.session.submitted()
        self.assertEqual(list(submitted['selection_report']['cells']), ['802449/merge'])
        candidate_feedback = json.loads((Path(second['snapshot']) / 'feedback.json').read_text())
        self.assertEqual(candidate_feedback['format'], 'stratified-train-v4')
        self.assertEqual(list(candidate_feedback['examples']), ['802449/merge'])
        self.assertEqual(candidate_feedback['selection_protocol']['mode'], 'grouped_oof')
        self.assertIn('candidate_selection', candidate_feedback)
        archive = self.memory.path.read_text()
        self.assertNotIn('validation', archive.lower())
        # A cached repeat of the memorizer reuses its out-of-fold scores without refitting.
        self.proposal(MEMORIZER)
        with patch('proofreader_evolve.harness.train_experiments.fit_classifier') as fit, \
                patch.object(sp, 'fit_classifier') as fold_fit:
            repeated = self.session.train_classifier()
        fit.assert_not_called()
        fold_fit.assert_not_called()
        self.assertEqual(repeated['candidates'][0]['target_precision'], first['target_precision'])
        self.assertEqual(self.session.evaluations_used, 2)

    def test_failure_cases_come_from_selection_measurement(self):
        for pair in self.session.failure_cases['pairs']:
            self.assertEqual(pair['brain'], '802449')


class GroupTransportTests(unittest.TestCase):
    def test_fit_receives_label_free_brain_groups_only_when_declared(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root = Path(tmp)
            train = two_brains()
            source, summary = fit_classifier(train, 'merge', {'parameters': {}}, root / 'fit',
                                             program=GROUPED, artifact_store=root / 'models')
            self.assertEqual(summary['group_transport']['group_names'], ['794495', '802449'])
            self.assertEqual(summary['group_transport']['rows'], {'794495': 16, '802449': 16})
            from proofreader_evolve.harness.classifier_contract import frozen_model, verify_artifacts
            artifacts = verify_artifacts(frozen_model(source), root / 'models')
            self.assertEqual((artifacts / 'groups.txt').read_text(), '794495,802449')
            plain, _ = fit_classifier(train, 'merge', {'parameters': {}}, root / 'plain',
                                      program=GENERALIZER, artifact_store=root / 'models')
            self.assertIsNotNone(frozen_model(plain))


class FeedbackAndBootstrapTests(unittest.TestCase):
    def test_feedback_examples_follow_selection_report(self):
        train = {'1': {'macro_precision': .5, 'cells': {'1/merge': {'precision': .5, 'tp': 1, 'fp': 1,
                 'training_examples': [{'label': 1, 'score': .9, 'features': {'x': 1.}}]}}}}['1']
        selection = {'macro_precision': .25, 'mode': 'grouped_oof', 'cells': {'2/merge': {
            'precision': .25, 'tp': 1, 'fp': 3, 'fold_budgets': [2, 2], 'fold_tp': [1, 0],
            'training_examples': [{'label': 0, 'score': .4, 'features': {'x': 2.}, 'group': 'selected_label0'}],
            'ranking_delta': {'net_tp': 0}}}}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'feedback.json'
            feedback.write_train_feedback(path, train, [], {'merge': 4}, target_kind='merge',
                                          selection=selection, protocol={'mode': 'grouped_oof'})
            data = json.loads(path.read_text())
        self.assertEqual(data['format'], 'stratified-train-v4')
        self.assertEqual(list(data['examples']), ['2/merge'])
        self.assertEqual(data['parent_selection']['cells']['2/merge']['fold_budgets'], [2, 2])
        self.assertEqual(data['parent_train']['cells']['1/merge']['precision'], .5)
        self.assertEqual(data['ranking_delta'], {'2/merge': {'net_tp': 0}})
        self.assertIn('rank branches', data['reading_guide'])

    def test_paired_bootstrap_brackets_zero_for_identical_scorers_and_excludes_it_for_clear_gains(self):
        rng = np.random.default_rng(0)
        n = 4000
        rows = [{'segment_id': i, 'node_id': i} for i in range(n)]
        truth = (rng.random(n) < .2).astype(np.int8)
        good = truth + rng.normal(0, .3, n)
        table = NativeTable('merge', pd.DataFrame({'detector_score': rng.random(n)}), rows, truth,
                            {'pool_sha256': pool_digest('merge', rows)})
        banks = {'2': NativeTables('2', {'merge': table}, {})}
        same = {'2/merge': {'scores': table.features.detector_score.to_numpy()}}
        result = scoring.paired_bootstrap(same, same, banks, {'merge': 200}, draws=50)
        self.assertEqual(result['macro_delta_ci95'], [0., 0.])
        self.assertEqual(result['p_delta_le_0'], 1.)
        better = {'2/merge': {'scores': good}}
        result = scoring.paired_bootstrap(same, better, banks, {'merge': 200}, draws=50, seed=3)
        self.assertGreater(result['macro_delta_ci95'][0], 0.)
        self.assertEqual(result['p_delta_le_0'], 0.)
        self.assertEqual(result['draws'], 50)


class DriverRoleTests(unittest.TestCase):
    def banks(self):
        store = two_brains()
        validation = brain_table('2', offset=5000.)
        validation.tables['merge'].meta['training_brain'] = '794495'
        return {**store, '2': validation}

    def test_grouped_oof_run_records_roles_selection_and_bootstrap(self):
        from proofreader_evolve.cli import run_precision_evolution as driver
        banks = self.banks()
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            seed = Path(tmp) / 'seed.py'
            seed.write_text(FORMULA)
            args = driver.parse_args(['--train-brains', '794495,802449', '--validation-brains', '2',
                                      '--merge-k', '3', '--split-k', '1', '--generations', '1', '--runs-dir', tmp,
                                      '--start-from', str(seed), '--bootstrap-draws', '20', '--promotion-gate', 'margin'])
            seen = {}
            async def revise(run_dir, policy, rules, report, model, *, experiments):
                data = json.loads(report.read_text())
                seen['feedback'] = data
                self.assertEqual(experiments.protocol.mode, 'grouped_oof')
                self.assertEqual(experiments.protocol.selection_brains, ['802449'])
                self.assertNotIn('validation', report.read_text().lower())
                policy.write_text(SIGNAL_FORMULA)
                experiments.proposal_path.write_text(json.dumps({'hypothesis': 'Signal ranks positives',
                    'strategy': 'Use signal', 'family': 'signal', 'parameter_grid': {}}))
                experiments.evaluate()
                return {'summary': 'formula candidate', 'cost_usd': 0}
            with patch.object(driver.pc, 'resolve_detector_runs', return_value={}), \
                    patch.object(driver, 'ensure_native_tables', side_effect=lambda brain, *a, **k: banks[brain]), \
                    patch.object(driver, 'attach_local_context', return_value=None):
                path = asyncio.run(driver.run(args, revise_fn=revise))
            split = json.loads((path / 'dataset_split.json').read_text())
            self.assertEqual(split['selection_brains'], ['802449'])
            self.assertEqual(split['auxiliary_train_brains'], ['794495'])
            self.assertEqual(split['validation_brains'], ['2'])
            self.assertIn('auxiliary', split['roles']['794495'])
            manifest = json.loads((path / 'manifest.json').read_text())
            protocol = manifest['selection_protocol']
            self.assertEqual(protocol['mode'], 'grouped_oof')
            details = protocol['partitions']['merge']['fold_details']
            self.assertTrue(all(d['brains']['794495']['held_rows'] == 0 for d in details))
            self.assertEqual(manifest['feature_discovery']['classifier_evaluation_units'], 2)
            self.assertIn('grouped out-of-fold', manifest['classifier_training_protocol'])
            ablation = json.loads((path / 'auxiliary_ablation.json').read_text())['merge']
            self.assertEqual({arm['status'] for arm in ablation['arms'].values()}, {'measured'})
            self.assertEqual(ablation['arms']['selection_only']['cells'].keys(), {'802449/merge'})
            self.assertIsNotNone(ablation['delta_with_auxiliary'])
            self.assertEqual(list(seen['feedback']['examples']), ['802449/merge'])
            self.assertEqual(seen['feedback']['parent_selection']['mode'], 'grouped_oof')
            self.assertIn('794495/merge', seen['feedback']['parent_train']['cells'])
            row = json.loads((path / 'ledger.jsonl').read_text().splitlines()[0])
            self.assertEqual(row['selection_protocol'], 'grouped_oof')
            self.assertEqual(list(row['selection']['cells']), ['802449/merge'])
            self.assertEqual(row['selection']['macro_precision'], 1.)
            self.assertTrue(row['selection_gate']['passed'])
            self.assertFalse(row['selection_gate']['required_for_promotion'])
            self.assertIn('2/merge', row['validation']['cells'])
            bootstrap = row['validation_gate']['paired_bootstrap']
            self.assertEqual(bootstrap['draws'], 20)
            self.assertEqual(len(bootstrap['macro_delta_ci95']), 2)
            self.assertTrue(row['accepted'])
            pool = json.loads((path / 'candidate_pool.json').read_text())['kinds']['merge']
            self.assertEqual(pool['reference_branch']['selection']['macro_precision'], 1.)
            self.assertIn('in_sample_precision', pool['candidates'][0])
            seed_entry = json.loads((path / 'train_seed/merge/result.json').read_text())
            self.assertEqual(list(seed_entry['selection']['cells']), ['802449/merge'])
            self.assertTrue((path / 'train_seed/merge/selection_state.npz').is_file())

    def test_selection_brain_must_not_be_detector_fitted_and_must_exist(self):
        from proofreader_evolve.cli import run_precision_evolution as driver
        banks = self.banks()
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()), \
                patch.object(driver.pc, 'resolve_detector_runs', return_value={}), \
                patch.object(driver, 'ensure_native_tables', side_effect=lambda brain, *a, **k: banks[brain]):
            args = driver.parse_args(['--train-brains', '794495', '--validation-brains', '2',
                                      '--merge-k', '3', '--generations', '0', '--runs-dir', tmp])
            with self.assertRaisesRegex(ValueError, 'No eligible selection brain'):
                asyncio.run(driver.run(args))
            args = driver.parse_args(['--train-brains', '794495,802449', '--selection-brains', '794495',
                                      '--validation-brains', '2', '--merge-k', '3', '--generations', '0',
                                      '--runs-dir', tmp])
            with self.assertRaisesRegex(ValueError, 'detector training brains'):
                asyncio.run(driver.run(args))
            with self.assertRaises(SystemExit):
                driver.parse_args(['--train-brains', '794495', '--selection-brains', '802449'])
            legacy = driver.parse_args(['--train-brains', '794495', '--validation-brains', '2', '--merge-k', '3',
                                        '--generations', '0', '--runs-dir', tmp, '--selection-protocol', 'in_sample'])
            with patch.object(driver, 'attach_local_context', return_value=None):
                path = asyncio.run(driver.run(legacy))
            self.assertEqual(json.loads((path / 'manifest.json').read_text())['selection_protocol']['mode'], 'in_sample')
            self.assertFalse((path / 'auxiliary_ablation.json').exists())


if __name__ == '__main__':
    unittest.main()
