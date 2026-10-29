"""Regression tests for independent, consistent selection and promotion objectives."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from proofreader_evolve.harness import fixed_pool_scoring as scoring
from proofreader_evolve.harness import kind_schedule
from proofreader_evolve.harness.candidate_pool import candidate_metadata
from proofreader_evolve.harness.evolution_report import _metrics, write_report
from proofreader_evolve.harness.feature_ablation import measure_ablation, prepare_ablation
from proofreader_evolve.harness.train_experiments import ExperimentMemory, TrainingExperiments, in_sample_precision
from proofreader_evolve.harness.train_feedback import metrics_only
from proofreader_evolve.harness.trajectory import metric_summary
from proofreader_evolve.harness.selection_protocol import SelectionProtocol
from proofreader_evolve.harness.internal_validation import grouped_folds
from proofreader_evolve.tests.test_feature_discovery import training_fixture
from proofreader_evolve.tests.test_selection_protocol import brain_table


FORMULA = 'def score_candidates(features, ctx):\n    return features["signal"].to_numpy()\n'


def cell(tp, k, counted=True):
    return {'tp': tp, 'fp': k - tp, 'precision': tp / k, 'requested_k': k, 'effective_k': k,
            'positives': k, 'in_mean': counted, 'pool_sha256': f'pool-{k}',
            'labels_sha256': f'labels-{k}', 'pool_size': 2 * k, 'n_labeled': 2 * k,
            'recall': tp / k}


def report(cells, mode='positive-weighted'):
    return {'version': scoring.SCORING_VERSION, 'policy_sha256': 'fixture', 'cells': cells,
            **scoring.aggregate_precision(cells, mode)}


def ranking(k, hits):
    selected = np.r_[np.arange(hits), np.arange(k, 2 * k - hits)]
    order = np.r_[selected, np.setdiff1d(np.arange(2 * k), selected)]
    scores = np.empty(2 * k)
    scores[order] = np.arange(2 * k, 0, -1)
    return scores


class ObjectiveTests(unittest.TestCase):
    def test_one_remaining_hit_does_not_saturate_a_multi_brain_objective(self):
        cells = {'a/merge': cell(10, 10), 'b/merge': cell(100, 100),
                 'c/merge': cell(99, 100), 'excluded/merge': cell(0, 1, False)}
        for mode in scoring.AGGREGATIONS:
            with self.subTest(mode=mode):
                budgets = scoring.Budgets({'merge': None}, positive_ratio=1., aggregation=mode)
                result = kind_schedule.allocate(['merge'], cells=cells, budgets=budgets,
                                                records=[], paused={}, pending={})
                self.assertFalse(result['kinds']['merge']['saturated'])
                denominator = 210 if mode == 'positive-weighted' else 300
                self.assertAlmostEqual(result['kinds']['merge']['saturation_threshold'], .5 / denominator)

    def test_selection_description_matches_labeled_fold_budgets(self):
        train = {'802449': brain_table('802449'), '747807': brain_table('747807', offset=1000.)}
        protocol = SelectionProtocol(train, {'merge': 8}, selection_brains=list(train),
                                     holdout_brains=['747807'])
        table = train['802449'].tables['merge']
        ids = protocol.fold_ids('merge', '802449')
        table.truth = np.full(len(table.truth), np.nan)
        table.truth[np.flatnonzero(ids == 0)[:2]] = [1., 0.]
        scores = {f'{brain}/merge': bank.tables['merge'].features['signal'].to_numpy()
                  for brain, bank in train.items()}
        measured, _ = protocol.score_report('merge', scores)
        described = protocol.describe('merge')['partitions']['merge']['fold_budgets']
        for name, metrics in measured['cells'].items():
            self.assertEqual(described[name], metrics['fold_budgets'])
        self.assertEqual(described['802449/merge'], [2, 0, 0])

    def test_headroom_uses_effective_k_when_requested_k_exceeds_labeled_rows(self):
        cells = {'a/merge': {**cell(3, 10), 'requested_k': 100, 'positives': 5}}
        for mode in scoring.AGGREGATIONS:
            budgets = scoring.Budgets({'merge': 100}, aggregation=mode)
            self.assertAlmostEqual(kind_schedule.headroom(cells, budgets, ['merge'])['merge'], .2)
            self.assertAlmostEqual(kind_schedule.saturation_thresholds(cells, budgets, ['merge'])['merge'], .05)

    def test_candidate_and_best_selection_follow_weighting_and_floor(self):
        a = report({'large/merge': cell(50, 100), 'small/merge': cell(15, 20),
                    'tiny/merge': cell(9, 9, False)})
        b = report({'large/merge': cell(70, 100), 'small/merge': cell(5, 20),
                    'tiny/merge': cell(0, 9, False)})
        results = []
        for sequence, measurement in enumerate((a, b)):
            state = {name: {'chosen': np.arange(c['effective_k'])} for name, c in measurement['cells'].items()}
            entry = candidate_metadata(FORMULA, 'merge', measurement, state)
            entry['sequence'] = sequence
            self.assertEqual(entry['target_precision'], measurement['per_kind_precision']['merge'])
            self.assertEqual(in_sample_precision(measurement, 'merge'), entry['target_precision'])
            results.append({'entry': entry, 'report': measurement, 'component': FORMULA})
        self.assertAlmostEqual(results[0]['entry']['target_precision'], 65 / 120)
        self.assertAlmostEqual(results[1]['entry']['target_precision'], 75 / 120)
        self.assertIs(TrainingExperiments._best(results), results[1])
        self.assertGreater(scoring.precision_for_kind({**a, 'aggregation': 'equal-brain'}, 'merge'),
                           scoring.precision_for_kind({**b, 'aggregation': 'equal-brain'}, 'merge'))

    def test_restore_tie_does_not_use_in_sample_or_other_kind_performance(self):
        a = {'entry': {'target_precision': .5, 'sequence': 1}, 'component': 'short',
             'report': report({'a/merge': cell(0, 10), 'a/split': cell(0, 10)})}
        b = {'entry': {'target_precision': .5, 'sequence': 2}, 'component': 'much longer',
             'report': report({'a/merge': cell(10, 10), 'a/split': cell(10, 10)})}
        self.assertIs(TrainingExperiments._best([b, a]), a)

    def test_margin_is_in_target_kind_units_and_other_kind_cannot_compensate(self):
        before = report({'a/merge': cell(5, 20), 'a/split': cell(10, 20)})
        after = report({'a/merge': cell(7, 20), 'a/split': cell(0, 20)})
        self.assertTrue(scoring.acceptance(before, after, .075, target_kind='merge',
                                           require_no_cell_regression=False)[0])
        self.assertFalse(scoring.acceptance(before, after, target_kind='split',
                                            require_no_cell_regression=False)[0])
        with self.assertRaisesRegex(ValueError, 'No candidate pools'):
            scoring.acceptance(before, after, target_kind='unknown')

    def test_excluded_regression_does_not_veto_target_gain(self):
        before = report({'a/merge': cell(5, 20), 'tiny/merge': cell(9, 9, False)})
        after = report({'a/merge': cell(7, 20), 'tiny/merge': cell(0, 9, False)})
        self.assertTrue(scoring.acceptance(before, after, target_kind='merge')[0])

    def test_bootstrap_is_independent_of_other_kind_and_checks_its_contract(self):
        truth = np.r_[np.ones(100), np.zeros(100)]
        table = SimpleNamespace(truth=truth, keys=[str(i) for i in range(200)])
        banks = {'a': SimpleNamespace(tables={'merge': table})}
        before = {'a/merge': {'scores': ranking(100, 20)}}
        after = {'a/merge': {'scores': ranking(100, 90)}}
        budgets = scoring.Budgets({'merge': 100, 'split': 100}, aggregation='positive-weighted')
        interval = scoring.paired_bootstrap(before, after, banks, budgets, target_kind='merge', draws=60)
        banks['a'].tables['split'] = table
        # No split scores are supplied: this kind must not enter the merge bootstrap.
        other = scoring.paired_bootstrap(before, after, banks, budgets, target_kind='merge', draws=60)
        self.assertEqual(interval, other)
        self.assertGreater(interval['delta_ci'][0], 0.)
        parent = report({'a/merge': cell(20, 100), 'a/split': cell(100, 100)})
        candidate = report({'a/merge': cell(90, 100), 'a/split': cell(0, 100)})
        self.assertTrue(scoring.bootstrap_acceptance(parent, candidate, interval, target_kind='merge')[0])
        for key, wrong in (('target_kind', 'split'), ('aggregation', 'equal-brain')):
            with self.subTest(key=key), self.assertRaises(ValueError):
                scoring.bootstrap_acceptance(parent, candidate, {**interval, key: wrong}, target_kind='merge')

    def test_report_and_feedback_keep_separate_scores_and_exclusion_flags(self):
        measured = report({'a/merge': cell(10, 20), 'a/split': cell(18, 20),
                           'tiny/merge': cell(0, 9, False), 'tiny/split': cell(9, 9, False)})
        for summarized in (measured, metrics_only(measured), metric_summary(measured), _metrics(measured)):
            self.assertEqual(summarized['per_kind_precision'], {'merge': .5, 'split': .9})
            self.assertIs(summarized['cells']['tiny/merge']['in_mean'], False)
            self.assertNotIn('macro_precision', summarized)
            self.assertNotIn('per_brain_precision', summarized)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'manifest.json').write_text(json.dumps({'objective': scoring.SCORING_VERSION}))
            (root / 'seed.json').write_text(json.dumps({'train': measured, 'validation': measured}))
            html = write_report(root).read_text()
        self.assertNotIn('macro_precision', html)
        self.assertNotIn('Mean Precision@K', html)
        self.assertIn('per_kind_precision', html)


class AblationObjectiveTests(unittest.TestCase):
    def test_in_sample_formula_does_not_reuse_classifier_diagnostic_folds(self):
        train = training_fixture()
        table = train['794495'].tables['merge']
        source = FORMULA.replace('signal', 'local_signal')
        values = table.features.local_signal.to_numpy()
        metrics, chosen = scoring.rank_metrics(values, table.truth, table.keys, 3)
        parent = report({'794495/merge': metrics}, 'equal-brain')
        state = {'794495/merge': {'scores': values, 'chosen': chosen}}
        protocol = SelectionProtocol(train, {'merge': 3}, mode='in_sample')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gen = root / 'gen001'
            gen.mkdir()
            policy, rules = gen / 'scorer.py', gen / 'rules.md'
            policy.write_text(source)
            rules.write_text('Synthetic feature test.')
            (gen / 'proposal.json').write_text(json.dumps({'hypothesis': 'Signal helps.', 'strategy': 'Paired removal.',
                'research': {'hypothesis_id': 'signal', 'feature_columns': ['local_signal']}}))
            memory = ExperimentMemory(root, train=train, protocol=protocol)
            # An earlier classifier diagnostic populated the fold cache in this in-sample run.
            memory.internal_folds['merge'] = grouped_folds(train, 'merge')
            session = TrainingExperiments(gen, policy, rules, 'merge', {'merge': source}, train, parent, state,
                                           {'merge': 3}, 120, 0, memory)
            with patch('proofreader_evolve.harness.feature_ablation.score',
                       side_effect=lambda source, frame, kind, timeout: frame.local_signal.to_numpy()):
                measured = session.evaluate_feature_ablation()
        self.assertEqual(measured['status'], 'measured')
        self.assertEqual(measured['protocol'], 'TRAIN_formula_descriptive')
        self.assertNotIn('fold_budgets', measured['cells']['794495/merge']['full'])
        self.assertEqual(measured['precision_by_arm']['full'], parent['per_kind_precision']['merge'])

    def test_ablation_uses_weighted_delta_and_excludes_small_pools(self):
        train, frames = {}, {}
        for brain, k, full_hits, control_hits in [('large', 100, 70, 50), ('small', 20, 5, 15),
                                                 ('tiny', 9, 0, 9), ('auxiliary', 500, 0, 500)]:
            full, control = ranking(k, full_hits), ranking(k, control_hits)
            frames[brain] = pd.DataFrame({'baseline': control, 'signal': full - control})
            table = SimpleNamespace(truth=np.r_[np.ones(k), np.zeros(k)], keys=[str(i) for i in range(2 * k)])
            train[brain] = SimpleNamespace(tables={'merge': table})
        identity = {'program_sha256': 'program', 'parameters_sha256': 'params',
                    'selection_brains': ['large', 'small', 'tiny']}
        budgets = scoring.Budgets({'merge': None}, positive_ratio=1., min_positives=20,
                                  aggregation='positive-weighted')
        with tempfile.TemporaryDirectory() as tmp, patch('proofreader_evolve.harness.feature_ablation.score',
                side_effect=lambda source, frame, kind, timeout: (frame.baseline + frame.signal).to_numpy()):
            measured = measure_ablation(train, 'merge', FORMULA, {}, frames, ['signal'], (None, None), identity,
                                         budgets, tmp, classifier=False, charge=lambda: None)
        self.assertEqual(measured['scored_brains'], ['large', 'small', 'tiny'])
        self.assertEqual(measured['excluded_cells'], ['tiny/merge'])
        self.assertAlmostEqual(measured['precision_by_arm']['full'], 75 / 120)
        self.assertAlmostEqual(measured['precision_by_arm']['without_features'], 65 / 120)
        self.assertAlmostEqual(measured['delta_precision'], 10 / 120)
        # Unweighted averaging would reverse the feature conclusion on these same cells.
        self.assertLess(sum(c['delta_precision'] for c in measured['cells'].values()), 0.)

    def test_formula_ablation_uses_selection_partition_and_brains(self):
        train = {'794495': brain_table('794495'), '802449': brain_table('802449', offset=1000.)}
        protocol = SelectionProtocol(train, {'merge': 4}, selection_brains=['802449'])
        with tempfile.TemporaryDirectory() as tmp:
            frames, columns, split, identity = prepare_ablation(train, 'merge', FORMULA, ['signal'],
                classifier=False, config={}, timeout=120, memory_mb=1024,
                partitions=protocol.partitions['merge'], selection_brains=protocol.selection_brains)
            with patch('proofreader_evolve.harness.feature_ablation.score',
                       side_effect=lambda source, frame, kind, timeout: frame.signal.to_numpy()):
                measured = measure_ablation(train, 'merge', FORMULA, {}, frames, columns, split, identity,
                                             {'merge': 4}, tmp, classifier=False, charge=lambda: None)
        official, _ = protocol.score_report('merge', {'802449/merge': frames['802449'].signal.to_numpy()})
        self.assertEqual(measured['scored_brains'], ['802449'])
        self.assertEqual(measured['protocol'], 'TRAIN_formula_partitioned_k_descriptive')
        self.assertEqual(measured['cells']['802449/merge']['full'],
                         {k: v for k, v in official['cells']['802449/merge'].items()
                          if k not in ('partition_sha256', 'pool_sha256', 'labels_sha256')})


if __name__ == '__main__':
    unittest.main()
