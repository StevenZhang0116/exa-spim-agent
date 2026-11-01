"""Rank diagnostics beside Precision@K: ties, NaN labels, small pools, no positives, fold
scales, and the guarantee that they never change the official selection result."""
import unittest

import numpy as np

from proofreader_evolve.harness import rank_diagnostics as rd
from proofreader_evolve.harness import selection_protocol as sp
from proofreader_evolve.harness.train_feedback import metrics_only
from proofreader_evolve.tests.test_selection_protocol import two_brains
from proofreader_evolve.tests.label_fixture import setUpModule, tearDownModule  # noqa: F401


def diagnostics(scores, truth, k, folds=None):
    scores = np.asarray(scores, dtype=float)
    folds = np.zeros(len(scores), dtype=np.int64) if folds is None else np.asarray(folds)
    keys = [f'k{i}' for i in range(len(scores))]
    return rd.cell_diagnostics(scores, truth, keys, folds, k, fold_budgets=sp.fold_budgets)


class CellDiagnosticsTests(unittest.TestCase):
    def test_average_precision_treats_ties_as_one_threshold(self):
        self.assertAlmostEqual(rd.average_precision([1, 0, 1, 0], [.9, .8, .7, .6]), (1. + 2 / 3) / 2)
        self.assertAlmostEqual(rd.average_precision([1, 0, 1, 0], [1., 1., 1., 1.]), .5)  # constant = positive rate
        self.assertIsNone(rd.average_precision([0, 0], [.2, .1]))
        out = diagnostics([1., 1., 1., 1.], [1, 0, 1, 0], 2)
        self.assertEqual((out['half_k'], out['double_k']), (1, 4))
        self.assertEqual(out['precision_at_half_k'], 1.)  # ties break by candidate id, like Precision@K
        self.assertAlmostEqual(out['average_precision'], .5)

    def test_nan_labels_are_neither_ranked_nor_counted(self):
        truth = np.array([1, np.nan, 0, 1, np.nan, 0], dtype=float)
        out = diagnostics([.9, .95, .8, .7, .99, .1], truth, 2)
        self.assertEqual((out['half_k'], out['double_k']), (1, 4))  # 4 labeled rows cap 2K
        self.assertEqual(out['precision_at_half_k'], 1.)
        self.assertEqual(out['double_k_tp'], 2)
        self.assertAlmostEqual(out['average_precision'], (1. + 2 / 3) / 2)

    def test_small_pool_caps_depths_and_half_k_rounds_up(self):
        out = diagnostics([.5, .4, .3], [1, 1, 0], 3)
        self.assertEqual((out['half_k'], out['double_k']), (2, 3))
        self.assertEqual((out['precision_at_half_k'], out['precision_at_double_k']), (1., 2 / 3))

    def test_no_positive_gives_no_average_precision(self):
        out = diagnostics([.3, .2], [0, 0], 1)
        self.assertIsNone(out['average_precision'])
        self.assertEqual(out['precision_at_half_k'], 0.)

    def test_fold_scores_are_not_pooled(self):
        scores = [100., 90., 80., .3, .2, .1]  # fold 0 is 100x larger; each fold ranked perfectly
        truth = [1, 1, 0, 1, 1, 0]
        out = diagnostics(scores, truth, 4, folds=[0, 0, 0, 1, 1, 1])
        self.assertEqual(out['average_precision'], 1.)
        self.assertEqual(out['precision_at_half_k'], 1.)
        self.assertEqual(out['double_k'], 6)
        self.assertLess(rd.average_precision(np.asarray(truth) == 1, scores), 1.)  # pooled would be penalised

    def test_aggregation_follows_the_objective_rule(self):
        cells = {'a/merge': {'in_mean': True, 'positives': 10, 'half_k': 5, 'half_k_tp': 5, 'double_k': 20,
                             'double_k_tp': 10, 'average_precision': .8, 'precision_at_half_k': 1., 'precision_at_double_k': .5},
                 'b/merge': {'in_mean': True, 'positives': 30, 'half_k': 15, 'half_k_tp': 3, 'double_k': 60,
                             'double_k_tp': 30, 'average_precision': .4, 'precision_at_half_k': .2, 'precision_at_double_k': .5},
                 'tiny/merge': {'in_mean': False, 'positives': 2, 'half_k': 1, 'half_k_tp': 0, 'double_k': 4,
                                'double_k_tp': 0, 'average_precision': 0., 'precision_at_half_k': 0., 'precision_at_double_k': 0.}}
        pooled = rd.aggregate_diagnostics(cells, 'positive-weighted')
        self.assertAlmostEqual(pooled['precision_at_half_k'], 8 / 20)
        self.assertAlmostEqual(pooled['average_precision'], (10 * .8 + 30 * .4) / 40)
        equal = rd.aggregate_diagnostics(cells, 'equal-brain')
        self.assertAlmostEqual(equal['precision_at_half_k'], .6)
        self.assertAlmostEqual(equal['average_precision'], .6)
        self.assertEqual(pooled['role'], 'diagnostic')
        self.assertIsNone(rd.aggregate_diagnostics({'x/merge': {'in_mean': True, 'precision': .1}}))


class SelectionUnchangedTests(unittest.TestCase):
    def test_higher_head_and_average_precision_do_not_outrank_higher_precision_at_k(self):
        # Ten labeled rows, five positives (rows 0-4), one fold, K = 5 (R-precision).
        truth = np.array([1] * 5 + [0] * 5, dtype=float)
        keys = [f'k{i}' for i in range(10)]
        folds = np.zeros(10, dtype=np.int64)

        def scores(positive_ranks, negative_ranks):
            ranks = np.array(positive_ranks + negative_ranks, dtype=float)
            return 1. - ranks / 20.  # rank 1 scores highest
        a = scores([3, 4, 5, 9, 10], [1, 2, 6, 7, 8])  # three positives inside K, two at the very end
        b = scores([1, 2, 6, 7, 8], [3, 4, 5, 9, 10])  # a cleaner list whose slots 3-5 hold negatives
        official_a, _ = sp.partitioned_rank_metrics(a, truth, keys, folds, 5)
        official_b, _ = sp.partitioned_rank_metrics(b, truth, keys, folds, 5)
        diag_a = rd.cell_diagnostics(a, truth, keys, folds, 5, fold_budgets=sp.fold_budgets)
        diag_b = rd.cell_diagnostics(b, truth, keys, folds, 5, fold_budgets=sp.fold_budgets)
        self.assertEqual((official_a['precision'], official_b['precision']), (.6, .4))
        self.assertAlmostEqual(diag_a['average_precision'], (1 / 3 + 2 / 4 + 3 / 5 + 4 / 9 + 5 / 10) / 5)
        self.assertAlmostEqual(diag_b['average_precision'], (1 + 1 + 3 / 6 + 4 / 7 + 5 / 8) / 5)
        self.assertGreater(diag_b['average_precision'], diag_a['average_precision'])
        self.assertEqual((diag_a['precision_at_half_k'], diag_b['precision_at_half_k']), (1 / 3, 2 / 3))
        self.assertEqual(diag_a['precision_at_double_k'], diag_b['precision_at_double_k'])  # both reach every positive

    def test_selection_report_carries_diagnostics_without_touching_the_score(self):
        train = two_brains()
        protocol = sp.SelectionProtocol(train, {'merge': 3}, selection_brains=['802449'])
        table = train['802449'].tables['merge']
        report, _ = protocol.score_report('merge', {'802449/merge': table.features['signal'].to_numpy(dtype=float)})
        cell = report['cells']['802449/merge']
        self.assertTrue(set(rd.CELL_KEYS) <= set(cell))
        self.assertEqual(cell['fold_budgets'], [1, 1, 1])
        self.assertEqual(cell['half_k'], 2)  # ceil(3 / 2) split over three folds
        self.assertEqual(cell['double_k'], 6)
        self.assertEqual(report['diagnostics']['role'], 'diagnostic')
        self.assertEqual(report['per_kind_precision']['merge'], cell['tp'] / cell['effective_k'])
        compact = metrics_only(report)
        self.assertEqual(compact['diagnostics']['average_precision'], report['diagnostics']['average_precision'])
        self.assertIn('precision_at_half_k', compact['cells']['802449/merge'])
        self.assertEqual(metrics_only({**report, 'diagnostics': None})['diagnostics'], 'not recorded')
        self.assertNotIn('diagnostics', metrics_only({'per_kind_precision': {}, 'cells': {}}))


if __name__ == '__main__':
    unittest.main()
