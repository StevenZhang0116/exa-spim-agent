"""Rank statistics (v7): weighted, tie-aware definitions shared by point estimates and the
paired bootstrap; fold handling; the configurable objective in evaluation and selection
reports; scheduler, candidate-pool and driver behaviour under AUROC."""
from contextlib import redirect_stderr
import io
from pathlib import Path
import tempfile
import unittest

import numpy as np

from proofreader_evolve.harness import fixed_pool_scoring as scoring
from proofreader_evolve.harness import kind_schedule as ks
from proofreader_evolve.harness import rank_statistics as rs
from proofreader_evolve.harness import selection_protocol as sp
from proofreader_evolve.harness.candidate_pool import CandidatePool, POOL_TOLERANCES
from proofreader_evolve.harness.train_feedback import metrics_only
from proofreader_evolve.tests.test_bootstrap_gate import DETECTOR, K, SIGNAL, TWEAK, pool
from proofreader_evolve.tests.test_fixed_pool_scoring import BASELINE, IMPROVED, fixture
from proofreader_evolve.tests.test_selection_protocol import two_brains
from proofreader_evolve.tests.label_fixture import setUpModule, tearDownModule  # noqa: F401


def stats(scores, truth, k, folds=None, **kwargs):
    scores = np.asarray(scores, dtype=float)
    keys = [f'k{i}' for i in range(len(scores))]
    return rs.cell_statistics(scores, np.asarray(truth, dtype=float), keys, k, fold_ids=folds, **kwargs)


def ordered(scores, truth):
    keys = [f'k{i}' for i in range(len(scores))]
    _, labels, starts = rs.ranking(np.asarray(scores, dtype=float), np.asarray(truth, dtype=float), keys)
    return labels, starts


class StatisticDefinitionTests(unittest.TestCase):
    def test_unit_weights_reproduce_hand_values(self):
        labels, starts = ordered([.9, .8, .7, .6], [1, 0, 1, 0])
        ones = np.ones(4)
        self.assertAlmostEqual(rs.auroc(labels, ones, starts), .75)  # 3 of the 4 positive/negative pairs
        self.assertAlmostEqual(rs.average_precision(labels, ones, starts), (1. + 2 / 3) / 2)
        # ROC vertices (0,0) (0,.5) (.5,.5) (.5,1) (1,1): area up to FPR .5 is .25, standardised .5
        self.assertAlmostEqual(rs.pauc(labels, ones, starts, .5), .5)
        self.assertAlmostEqual(rs.pauc(labels, ones, starts, 1.), .75)  # full range equals AUROC
        out = stats([.9, .8, .7, .6], [1, 0, 1, 0], 2)
        self.assertEqual((out['auroc'], out['average_precision']), (.75, (1. + 2 / 3) / 2))
        self.assertAlmostEqual(out['precision_depth_mean'], (1. + .5 + .5) / 3)

    def test_ties_form_one_threshold(self):
        labels, starts = ordered([1., 1., 1., 1.], [1, 0, 1, 0])
        ones = np.ones(4)
        self.assertEqual(len(starts), 1)
        self.assertAlmostEqual(rs.auroc(labels, ones, starts), .5)
        self.assertAlmostEqual(rs.average_precision(labels, ones, starts), .5)  # the positive rate
        self.assertAlmostEqual(rs.pauc(labels, ones, starts, .5), .25)  # the diagonal
        out = stats([1., 1., 1., 1.], [1, 0, 1, 0], 2)
        self.assertEqual(out['precision_at_half_k'], 1.)  # precision keeps the candidate-id order
        self.assertAlmostEqual(out['auroc'], .5)

    def test_weights_equal_duplicated_rows(self):
        scores, truth = [.9, .8, .7, .6, .5], [1, 0, 1, 0, 1]
        labels, starts = ordered(scores, truth)
        weights = np.array([2., 1., 0., 3., 1.])
        dup_scores = [s for s, w in zip(scores, weights) for _ in range(int(w))]
        dup_truth = [t for t, w in zip(truth, weights) for _ in range(int(w))]
        dup_labels, dup_starts = ordered(dup_scores, dup_truth)
        ones = np.ones(len(dup_labels))
        for weighted, plain in ((rs.auroc, rs.auroc), (rs.average_precision, rs.average_precision)):
            self.assertAlmostEqual(weighted(labels, weights, starts), plain(dup_labels, ones, dup_starts))
        self.assertAlmostEqual(rs.pauc(labels, weights, starts, .4), rs.pauc(dup_labels, ones, dup_starts, .4))
        hits, depth = rs.prefix_hits(labels, weights, 3)
        self.assertEqual((hits, depth), (2., 3.))  # rows 1 (x2) and 2 (x1, weight 0 skipped) -> positives 2

    def test_zero_weight_tie_groups_do_not_undefine_ap(self):
        # A bootstrap draw that leaves out the top rows: those tie groups have zero weight and
        # are no threshold. The weighted AP equals the AP of the rows that remain.
        labels, starts = ordered([.9, .8, .7, .6], [0, 1, 0, 1])
        remaining_labels, remaining_starts = ordered([.8, .7, .6], [1, 0, 1])
        remaining = rs.average_precision(remaining_labels, np.ones(3), remaining_starts)
        self.assertAlmostEqual(remaining, (1. + 2 / 3) / 2)
        self.assertAlmostEqual(rs.average_precision(labels, np.array([0., 1., 1., 1.]), starts), remaining)
        self.assertAlmostEqual(rs.average_precision(labels, np.array([0., 0., 1., 1.]), starts), .5)
        self.assertAlmostEqual(rs.auroc(labels, np.array([0., 1., 1., 1.]), starts), .5)

    def test_undefined_without_both_classes(self):
        labels, starts = ordered([.3, .2], [0, 0])
        self.assertIsNone(rs.auroc(labels, np.ones(2), starts))
        self.assertIsNone(rs.average_precision(labels, np.ones(2), starts))
        self.assertIsNone(rs.pauc(labels, np.ones(2), starts))
        labels, starts = ordered([.3, .2], [1, 1])
        self.assertIsNone(rs.auroc(labels, np.ones(2), starts))
        self.assertEqual(rs.average_precision(labels, np.ones(2), starts), 1.)
        out = stats([.3, .2], [0, 0], 1)
        self.assertIsNone(out['auroc'])
        self.assertEqual(out['precision_at_half_k'], 0.)
        with self.assertRaises(ValueError):
            rs.check_pauc_fpr(0.)
        with self.assertRaises(ValueError):
            rs.check_objective('f1')

    def test_nan_labels_are_neither_ranked_nor_counted(self):
        truth = np.array([1, np.nan, 0, 1, np.nan, 0], dtype=float)
        out = stats([.9, .95, .8, .7, .99, .1], truth, 2)
        self.assertEqual((out['half_k'], out['double_k']), (1, 4))  # 4 labeled rows cap 2K
        self.assertEqual(out['double_k_tp'], 2)
        self.assertAlmostEqual(out['average_precision'], (1. + 2 / 3) / 2)
        self.assertAlmostEqual(out['auroc'], .75)

    def test_small_pool_caps_depths_and_half_k_rounds_up(self):
        out = stats([.5, .4, .3], [1, 1, 0], 3)
        self.assertEqual((out['half_k'], out['double_k']), (2, 3))
        self.assertEqual((out['precision_at_half_k'], out['precision_at_double_k']), (1., 2 / 3))

    def test_fold_statistics_are_not_pooled(self):
        scores = [100., 90., 80., .3, .2, .1]  # fold 0 is 100x larger; each fold ranked perfectly
        truth = [1, 1, 0, 1, 1, 0]
        out = stats(scores, truth, 4, folds=[0, 0, 0, 1, 1, 1])
        self.assertEqual((out['average_precision'], out['auroc'], out['pauc']), (1., 1., 1.))
        self.assertEqual(out['precision_at_half_k'], 1.)
        labels, starts = ordered(scores, truth)
        self.assertLess(rs.auroc(labels, np.ones(6), starts), 1.)  # pooled would be penalised

    def test_aggregation_follows_the_objective_rule(self):
        def cell(positives, in_mean, **values):
            base = {'in_mean': in_mean, 'positives': positives, 'tp': positives // 2, 'effective_k': positives,
                    'half_k': positives // 2, 'half_k_tp': positives // 2, 'double_k': 2 * positives,
                    'double_k_tp': positives}
            return {**base, **values}
        cells = {'a/merge': cell(10, True, auroc=.9, pauc=.5, average_precision=.8),
                 'b/merge': cell(30, True, auroc=.7, pauc=.3, average_precision=.4),
                 'tiny/merge': cell(2, False, auroc=0., pauc=0., average_precision=0.)}
        pooled = rs.aggregate_statistics(cells, 'positive-weighted')
        self.assertAlmostEqual(pooled['auroc'], (10 * .9 + 30 * .7) / 40)
        self.assertAlmostEqual(pooled['precision'], .5)
        self.assertAlmostEqual(pooled['precision_at_half_k'], 1.)
        self.assertAlmostEqual(pooled['precision_depth_mean'], (1. + .5 + .5) / 3)
        equal = rs.aggregate_statistics(cells, 'equal-brain')
        self.assertAlmostEqual(equal['auroc'], .8)
        self.assertAlmostEqual(equal['average_precision'], .6)


def _cell(precision, auroc, head, k=100, positives=100):
    tp = int(round(precision * k))
    return {'pool_sha256': 'p', 'labels_sha256': 'l', 'pool_size': 1000, 'n_labeled': 500, 'requested_k': k,
            'effective_k': k, 'positives': positives, 'in_mean': True, 'tp': tp, 'fp': k - tp,
            'precision': precision, 'auroc': auroc, 'pauc': auroc, 'average_precision': auroc,
            'half_k': k // 2, 'half_k_tp': int(round(head * k / 2)), 'precision_at_half_k': head,
            'double_k': 2 * k, 'double_k_tp': int(round(head * 2 * k)), 'precision_at_double_k': head,
            'precision_depth_mean': (2 * head + precision) / 3}


def _report(objective, **cells):
    report = {'version': scoring.SCORING_VERSION, 'objective': objective,
              'cells': {name: cell for name, cell in cells.items()}}
    report.update(scoring.aggregate_objective(report['cells'], 'positive-weighted', objective))
    return report


class ObjectiveTests(unittest.TestCase):
    def test_budgets_validate_the_objective(self):
        budgets = scoring.Budgets({'merge': 10}, objective='auroc')
        self.assertEqual(budgets.describe()['objective_label'], 'AUROC')
        self.assertNotIn('head_guard', budgets.describe())
        with self.assertRaises(TypeError):   # the point-estimate head guard was removed on 2026-11-12
            scoring.Budgets({'merge': 10}, head_guard={'statistic': 'precision_at_k', 'tolerance': .02})
        with self.assertRaises(ValueError):
            scoring.Budgets({'merge': 10}, objective='accuracy')
        with self.assertRaises(ValueError):
            scoring.Budgets({'merge': 10}, pauc_fpr=0.)

    def test_the_objective_decides_the_point_gate(self):
        parent = _report('auroc', **{'a/merge': _cell(.30, .80, .30)})
        broad = _report('auroc', **{'a/merge': _cell(.30, .85, .30)})       # AUROC up, head unchanged
        tail = _report('auroc', **{'a/merge': _cell(.25, .90, .25)})        # AUROC up, head down .05
        self.assertEqual(scoring.objective_for_kind(parent, 'merge'), .80)
        self.assertEqual(scoring.precision_for_kind(parent, 'merge'), .30)
        passed, reason = scoring.acceptance(parent, broad, target_kind='merge')
        self.assertTrue(passed, reason)
        self.assertIn('AUROC improved', reason)
        # The point gate reads the objective only; the head is protected by promotion_gate's guard rail.
        self.assertTrue(scoring.acceptance(parent, tail, target_kind='merge')[0])
        details = scoring.gate_details(parent, tail, 'merge')
        self.assertEqual((details['objective'], details['parent_objective'], details['candidate_objective']), ('auroc', .80, .90))
        self.assertNotIn('head_guard', details)
        self.assertAlmostEqual(details['candidate_precision'], .25)
        # Under precision at K the same tail candidate is simply a regression.
        parent_p = _report('precision_at_k', **{'a/merge': _cell(.30, .80, .30)})
        tail_p = _report('precision_at_k', **{'a/merge': _cell(.25, .90, .25)})
        passed, reason = scoring.acceptance(parent_p, tail_p, target_kind='merge', require_no_cell_regression=False)
        self.assertFalse(passed)
        self.assertIn('Precision@K', reason)
        with self.assertRaisesRegex(ValueError, 'different objectives'):
            scoring.acceptance(parent, tail_p, target_kind='merge')

    def test_reports_before_v7_are_precision(self):
        old = {'version': scoring.SCORING_VERSION, 'cells': {'a/merge': {k: v for k, v in _cell(.4, .9, .4).items()
                                                                        if k in ('tp', 'effective_k', 'precision', 'in_mean', 'positives')}}}
        old.update(scoring.aggregate_precision(old['cells'], 'positive-weighted'))
        self.assertEqual(scoring.report_objective(old), 'precision_at_k')
        self.assertEqual(scoring.objective_for_kind(old, 'merge'), .4)
        self.assertIsNone(scoring.statistic_for_kind(old, 'merge', 'auroc'))
        compact = metrics_only(old)
        self.assertEqual((compact['objective'], compact['per_kind_objective'], compact['per_kind_statistics']),
                         ('precision_at_k', {'merge': .4}, 'not recorded'))


class EvaluationUnderAurocTests(unittest.TestCase):
    def setUp(self):
        self.banks = {'2': pool('2')}
        self.budgets = scoring.Budgets({'merge': K}, objective='auroc')
        self.reports, self.states = {}, {}
        for name, source in (('detector', DETECTOR), ('signal', SIGNAL), ('tweak', TWEAK)):
            state = {}
            self.reports[name] = scoring.evaluate(source, self.banks, self.budgets, state=state)
            self.states[name] = state

    def test_report_names_the_objective_and_keeps_precision(self):
        report = self.reports['signal']
        cell = report['cells']['2/merge']
        self.assertEqual(report['objective'], 'auroc')
        self.assertEqual(report['per_kind_objective']['merge'], cell['auroc'])
        self.assertEqual(report['per_kind_precision']['merge'], cell['precision'])
        self.assertEqual(report['per_kind_statistics']['merge']['pauc'], cell['pauc'])
        self.assertGreater(cell['auroc'], .95)
        self.assertLess(self.reports['detector']['cells']['2/merge']['auroc'], .6)
        self.assertEqual(report['budgets']['objective'], 'auroc')

    def test_bootstrap_measures_the_objective(self):
        interval = scoring.paired_bootstrap(self.states['detector'], self.states['signal'], self.banks, self.budgets,
                                            target_kind='merge', draws=200, seed=1)
        point = (self.reports['signal']['per_kind_objective']['merge']
                 - self.reports['detector']['per_kind_objective']['merge'])
        self.assertEqual(interval['objective'], 'auroc')
        self.assertGreater(interval['delta_ci'][0], 0.)
        self.assertLess(interval['delta_ci'][0], point)
        self.assertGreater(interval['delta_ci'][1], point)
        self.assertEqual(interval['undefined_draws'], 0)
        same = scoring.paired_bootstrap(self.states['signal'], self.states['signal'], self.banks, self.budgets,
                                        target_kind='merge', draws=50, seed=2)
        self.assertEqual(same['delta_ci'], [0., 0.])
        precision_interval = scoring.paired_bootstrap(self.states['detector'], self.states['signal'], self.banks,
                                                      {'merge': K}, target_kind='merge', draws=50, seed=1)
        self.assertEqual(precision_interval['objective'], 'precision_at_k')

    def test_bootstrap_under_average_precision_defines_every_draw(self):
        budgets = scoring.Budgets({'merge': K}, objective='average_precision')
        interval = scoring.paired_bootstrap(self.states['detector'], self.states['signal'], self.banks, budgets,
                                            target_kind='merge', draws=100, seed=3)
        self.assertEqual((interval['objective'], interval['undefined_draws']), ('average_precision', 0))
        self.assertGreater(interval['delta_ci'][0], 0.)
        self.assertEqual(interval['p_delta_ge_0'], 1.)

    def test_bootstrap_probabilities_use_the_defined_draws(self):
        # One positive among three rows: a Poisson draw that leaves it out has no AUROC.
        banks, budgets = {'1': fixture()}, scoring.Budgets({'split': 1}, objective='auroc')
        states = {}
        for name, source in (('baseline', BASELINE), ('improved', IMPROVED)):
            states[name] = {}
            scoring.evaluate(source, banks, budgets, state=states[name])
        interval = scoring.paired_bootstrap(states['baseline'], states['improved'], banks, budgets,
                                            target_kind='split', draws=200, seed=0)
        self.assertGreater(interval['undefined_draws'], 0)
        self.assertLess(interval['undefined_draws'], 200)
        self.assertEqual((interval['p_delta_ge_0'], interval['p_delta_le_0']), (1., 0.))
        self.assertEqual((interval['bootstrap_mean_delta'], interval['delta_ci']), (1., [1., 1.]))

    def test_merged_train_measurement_carries_the_candidate_objective(self):
        from types import SimpleNamespace
        from proofreader_evolve.harness.train_experiments import TrainingExperiments, digest
        session = SimpleNamespace(parent_report=self.reports['detector'], parent_state=self.states['detector'],
                                  target_kind='merge', budgets=self.budgets)
        merged, state = TrainingExperiments._merge_measurement(session, self.reports['signal'],
                                                               self.states['signal'], SIGNAL)
        cell = self.reports['signal']['cells']['2/merge']
        self.assertEqual(merged['policy_sha256'], digest(SIGNAL))
        self.assertEqual(merged['objective'], 'auroc')
        self.assertEqual(merged['per_kind_objective']['merge'], cell['auroc'])
        self.assertEqual(merged['per_kind_statistics']['merge']['pauc'], cell['pauc'])
        self.assertEqual(merged['per_kind_precision']['merge'], cell['precision'])
        self.assertEqual(scoring.objective_for_kind(merged, 'merge'), merged['per_kind_objective']['merge'])
        self.assertIs(state['2/merge'], self.states['signal']['2/merge'])
        self.assertNotEqual(merged['per_kind_objective']['merge'], self.reports['detector']['per_kind_objective']['merge'])

    def test_one_swapped_hit_moves_auroc_by_less_than_precision(self):
        detector, tweak = self.reports['detector']['cells']['2/merge'], self.reports['tweak']['cells']['2/merge']
        self.assertAlmostEqual(tweak['precision'] - detector['precision'], 1 / K)
        self.assertGreater(tweak['auroc'], detector['auroc'])
        self.assertLess(tweak['auroc'] - detector['auroc'], 1 / K)


class SelectionProtocolTests(unittest.TestCase):
    def test_selection_report_carries_the_objective(self):
        train = two_brains()
        budgets = scoring.Budgets({'merge': 3}, objective='auroc')
        protocol = sp.SelectionProtocol(train, budgets, selection_brains=['802449'])
        self.assertEqual(protocol.describe()['metric'], 'grouped_cv_auroc')
        table = train['802449'].tables['merge']
        report, _ = protocol.score_report('merge', {'802449/merge': table.features['signal'].to_numpy(dtype=float)})
        cell = report['cells']['802449/merge']
        self.assertTrue(set(rs.CELL_KEYS) <= set(cell))
        self.assertEqual(cell['fold_budgets'], [1, 1, 1])
        self.assertEqual(report['objective'], 'auroc')
        self.assertEqual(report['per_kind_objective']['merge'], cell['auroc'])
        self.assertEqual(report['per_kind_precision']['merge'], cell['tp'] / cell['effective_k'])
        self.assertIsNone(report['holdout_objective'])
        compact = metrics_only(report)
        self.assertEqual(compact['per_kind_statistics']['merge']['auroc'], cell['auroc'])
        self.assertIn('auroc', compact['cells']['802449/merge'])
        plain = sp.SelectionProtocol(train, {'merge': 3}, selection_brains=['802449'])
        self.assertEqual(plain.describe()['metric'], 'grouped_cv_precision')
        self.assertEqual(sp.fold_budgets([5, 5, 5], 3), [1, 1, 1])


class SchedulerAndPoolTests(unittest.TestCase):
    def test_headroom_and_momentum_follow_the_objective(self):
        cells = {'a/merge': {'positives': 10, 'n_labeled': 110, 'requested_k': 10, 'tp': 5, 'precision': .5,
                             'auroc': .8, 'in_mean': True},
                 'b/merge': {'positives': 30, 'n_labeled': 330, 'requested_k': 30, 'tp': 15, 'precision': .5,
                             'auroc': .6, 'in_mean': True}}
        auroc = scoring.Budgets({'merge': 10}, aggregation='positive-weighted', objective='auroc')
        precision = scoring.Budgets({'merge': 10}, aggregation='positive-weighted')
        self.assertAlmostEqual(ks.headroom(cells, auroc, ['merge'])['merge'], 1. - (10 * .8 + 30 * .6) / 40)
        self.assertAlmostEqual(ks.headroom(cells, precision, ['merge'])['merge'], .5)
        self.assertAlmostEqual(ks.saturation_thresholds(cells, auroc, ['merge'])['merge'],
                               .5 / (10 * 100 + 30 * 300))
        rows = [{'target_kind': 'merge', 'accepted': False,
                 'validation_gate': {'target_kind': 'merge', 'parent_objective': .80, 'candidate_objective': .83,
                                     'parent_precision': .3, 'candidate_precision': .2}},
                {'target_kind': 'merge', 'accepted': False,
                 'validation_gate': {'target_kind': 'merge', 'parent_precision': .3, 'candidate_precision': .35}}]
        self.assertAlmostEqual(ks.momentum(rows, ['merge'])['merge'], (.03 + .05) / 2)

    def test_pool_tolerance_and_ranking_follow_the_objective(self):
        with tempfile.TemporaryDirectory() as tmp:
            pool_ = CandidatePool(Path(tmp) / 'pool.json', ['merge'], objective='auroc', size=2)
            self.assertEqual(pool_.score_tolerance, POOL_TOLERANCES['auroc'])
            self.assertEqual(CandidatePool(Path(tmp) / 'p2.json', ['merge']).score_tolerance, .02)

            def entry(number, precision, objective):
                return {'experiment': f'e{number}', 'target_kind': 'merge', 'sequence': number, 'status': 'evaluated',
                        'component_sha256': f'c{number}', 'formula_sha256': f'f{number}', 'top_k_sha256': f't{number}',
                        'target_precision': precision, 'target_objective': objective, 'parameters': {}}
            pool_._retain('merge', [entry(1, .5, .70), entry(2, .4, .80), entry(3, .45, .797)])
            self.assertEqual([e['experiment'] for e in pool_.members['merge']], ['e2', 'e3'])  # by AUROC, within .005
            legacy = {k: v for k, v in entry(4, .9, None).items()}
            self.assertEqual(CandidatePool._objective(legacy), .9)  # pre-v7 entries fall back to precision
            self.assertEqual(pool_.summary()['settings']['objective'], 'auroc')


class DriverDefaultsTests(unittest.TestCase):
    def test_cli_defaults_and_validation(self):
        from proofreader_evolve.cli.run_precision_evolution import parse_args
        args = parse_args([])
        self.assertEqual(args.image_previews_per_generation, 32)
        self.assertEqual(parse_args(['--image-previews-per-generation', '5']).image_previews_per_generation, 5)
        self.assertEqual(parse_args(['--image-previews-per-generation', '0']).image_previews_per_generation, 0)
        self.assertEqual((args.objective, args.pauc_fpr, args.promotion_gate), ('auroc', .1, 'rule'))
        self.assertIsNone(args.candidate_score_tolerance)
        self.assertEqual(parse_args(['--objective', 'precision_at_k']).objective, 'precision_at_k')
        for bad in (['--pauc-fpr', '0'], ['--head-guard-tolerance', '0'], ['--objective', 'f1'],
                    ['--image-previews-per-generation', '-1'], ['--image-previews-per-generation', '1.5']):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(bad)


if __name__ == '__main__':
    unittest.main()
