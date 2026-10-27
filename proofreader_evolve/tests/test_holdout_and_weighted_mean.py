"""Whole-brain holdout selection brains and the positive-weighted per-kind aggregation (2026-10-27).

Tiny fixtures from test_selection_protocol; no brain caches, no API calls.
"""
import io
import unittest
from contextlib import redirect_stderr

import numpy as np

from proofreader_evolve.harness import fixed_pool_scoring as scoring, kind_schedule
from proofreader_evolve.harness.internal_validation import FOLDS, fold_assignments, grouped_folds
from proofreader_evolve.harness.selection_protocol import SelectionProtocol
from proofreader_evolve.harness.train_feedback import metrics_only
from proofreader_evolve.tests.test_selection_protocol import brain_table


def three_brains():
    return {'794495': brain_table('794495', offset=1000.), '802449': brain_table('802449'),
            '747807': brain_table('747807', offset=2000.)}


def cell(tp, k, positives=None, in_mean=True):
    return {"tp": tp, "fp": k - tp, "precision": tp / k, "requested_k": k, "effective_k": k,
            "positives": k if positives is None else positives, "in_mean": in_mean}


class HoldoutFoldTests(unittest.TestCase):
    def test_holdout_brain_is_held_out_whole_in_one_fold_and_fitted_on_elsewhere(self):
        train = three_brains()
        folds, description = grouped_folds(train, 'merge', selection_brains=['802449', '747807'],
                                           holdout_brains=['747807'])
        self.assertEqual(description['holdout_brains'], ['747807'])
        self.assertIn('held out whole', description['holdout_rule'])
        ids = fold_assignments(folds, '747807', 16)
        self.assertEqual(len(set(ids.tolist())), 1)  # every row of the brain in the same fold
        self.assertGreaterEqual(int(ids[0]), 0)
        for fold, (fitting, held) in enumerate(folds):
            detail = description['fold_details'][fold]['brains']
            self.assertEqual(detail['747807']['role'], 'selection_holdout')
            self.assertEqual(detail['802449']['role'], 'selection')
            self.assertEqual(detail['794495']['role'], 'auxiliary')
            if fold == int(ids[0]):
                self.assertEqual(len(held['747807']), 16)
                self.assertEqual(len(fitting['747807']), 0)
            else:
                self.assertEqual(len(held['747807']), 0)
                np.testing.assert_array_equal(fitting['747807'], np.arange(16))
            np.testing.assert_array_equal(fitting['794495'], np.arange(16))  # auxiliary rows join every fit
        # The grouped brain keeps the balanced within-brain folds it would have on its own: holdout
        # groups stay outside the load balance (listing 747807 as a grouped brain would change them).
        grouped_ids = fold_assignments(folds, '802449', 16)
        self.assertEqual(len(set(grouped_ids.tolist())), FOLDS)
        alone = grouped_folds(train, 'merge', selection_brains=['802449'])
        np.testing.assert_array_equal(fold_assignments(alone[0], '802449', 16), grouped_ids)
        plain = grouped_folds(train, 'merge', selection_brains=['802449', '747807'])
        self.assertNotEqual(plain[1]['partition_sha256'], description['partition_sha256'])
        with self.assertRaisesRegex(ValueError, 'subset of the selection brains'):
            grouped_folds(train, 'merge', selection_brains=['802449'], holdout_brains=['747807'])

    def test_holdouts_alone_need_enough_brains(self):
        train = three_brains()
        with self.assertRaisesRegex(ValueError, 'at least'):
            grouped_folds(train, 'merge', selection_brains=['802449', '747807'],
                          holdout_brains=['802449', '747807'])

    def test_holdout_brains_round_robin_over_folds(self):
        train = {**three_brains(), '754610': brain_table('754610', offset=3000.),
                 '754612': brain_table('754612', offset=4000.)}
        folds, _ = grouped_folds(train, 'merge', selection_brains=['802449', '747807', '754610', '754612'],
                                 holdout_brains=['747807', '754610', '754612'])
        assigned = sorted(int(fold_assignments(folds, b, 16)[0]) for b in ('747807', '754610', '754612'))
        self.assertEqual(assigned, [0, 1, 2])


class HoldoutProtocolTests(unittest.TestCase):
    def test_protocol_reports_holdout_cells_and_pooled_holdout_precision(self):
        train = three_brains()
        protocol = SelectionProtocol(train, {'merge': 4}, selection_brains=['802449', '747807'],
                                     holdout_brains=['747807'])
        self.assertEqual(protocol.holdout_brains, ['747807'])
        info = protocol.describe('merge')
        self.assertEqual(info['holdout_brains'], ['747807'])
        self.assertIn('cross-brain', info['holdout_role'])
        state = {cell: {'scores': train[cell.split('/')[0]].tables['merge'].features['signal'].to_numpy()}
                 for cell in protocol.cells('merge')}
        report, _ = protocol.measure_formula('merge', state)
        self.assertEqual(report['holdout_cells'], ['747807/merge'])
        held = report['cells']['747807/merge']
        self.assertEqual(report['holdout_precision'], held['tp'] / held['effective_k'])
        self.assertEqual(sum(1 for b in held['fold_budgets'] if b), 1)  # one fold holds the whole brain
        compact = metrics_only(report)
        self.assertEqual(compact['holdout_cells'], ['747807/merge'])
        self.assertEqual(compact['holdout_precision'], report['holdout_precision'])
        self.assertEqual(compact['aggregation'], report['aggregation'])
        with self.assertRaisesRegex(ValueError, 'subset'):
            SelectionProtocol(train, {'merge': 4}, selection_brains=['802449'], holdout_brains=['747807'])
        with self.assertRaisesRegex(ValueError, 'grouped_oof'):
            SelectionProtocol(train, {'merge': 4}, mode='in_sample', holdout_brains=['747807'])


class WeightedAggregationTests(unittest.TestCase):
    cells = {"a/merge": cell(1, 4), "b/merge": cell(30, 40), "a/split": cell(2, 4), "b/split": cell(10, 40)}

    def test_positive_weighted_pools_hits_over_k_within_each_kind(self):
        equal = scoring.aggregate_precision(self.cells)
        pooled = scoring.aggregate_precision(self.cells, "positive-weighted")
        self.assertEqual(equal["aggregation"], scoring.AGGREGATION_LABELS["equal-brain"])
        self.assertEqual(pooled["aggregation"], scoring.AGGREGATION_LABELS["positive-weighted"])
        self.assertAlmostEqual(equal["per_kind_precision"]["merge"], (0.25 + 0.75) / 2)
        self.assertAlmostEqual(pooled["per_kind_precision"]["merge"], 31 / 44)
        self.assertAlmostEqual(pooled["per_kind_precision"]["split"], 12 / 44)
        self.assertAlmostEqual(pooled["macro_precision"], (31 / 44 + 12 / 44) / 2)
        # A label round-trips to its mode; unknown names are refused.
        self.assertEqual(scoring.aggregate_precision(self.cells, pooled["aggregation"])["macro_precision"],
                         pooled["macro_precision"])
        with self.assertRaises(ValueError):
            scoring.aggregate_precision(self.cells, "median")
        excluded = {**self.cells, "c/merge": cell(0, 2, in_mean=False), "c/split": cell(1, 2, in_mean=False)}
        self.assertEqual(scoring.aggregate_precision(excluded, "positive-weighted")["macro_precision"],
                         pooled["macro_precision"])

    def test_budgets_carry_the_aggregation_and_acceptance_follows_the_report(self):
        budgets = scoring.Budgets({"merge": None, "split": None}, positive_ratio=1., aggregation="positive-weighted")
        self.assertEqual(budgets.aggregation, "positive-weighted")
        self.assertEqual(budgets.describe()["aggregation_label"], scoring.AGGREGATION_LABELS["positive-weighted"])
        self.assertIn("pooled by GT positives", budgets.describe()["mean_rule"])
        self.assertEqual(scoring.Budgets({"merge": 4}).aggregation, "equal-brain")
        with self.assertRaises(ValueError):
            scoring.Budgets({"merge": 4}, aggregation="median")
        identity = {"pool_sha256": "p", "labels_sha256": "l", "pool_size": 50, "n_labeled": 44, "requested_k": 4,
                    "effective_k": 4, "positives": 4}
        before = {"version": scoring.SCORING_VERSION, "aggregation": scoring.AGGREGATION_LABELS["positive-weighted"],
                  "cells": {name: {**identity, **values} for name, values in self.cells.items()}}
        after = {**before, "cells": {name: dict(values) for name, values in before["cells"].items()}}
        after["cells"]["a/merge"].update(cell(2, 4))  # one more hit on the small pool
        passed, reason = scoring.acceptance(before, after, 0, require_no_cell_regression=False)
        self.assertTrue(passed)
        self.assertIn(f"{(32 / 44 + 12 / 44) / 2:.6f}", reason)
        mismatched = {**after, "aggregation": scoring.AGGREGATION_LABELS["equal-brain"]}
        with self.assertRaisesRegex(ValueError, "different aggregations"):
            scoring.acceptance(before, mismatched, 0)

    def test_headroom_pools_when_positive_weighted(self):
        cells = {"a/merge": cell(1, 4), "b/merge": cell(30, 40)}
        equal = kind_schedule.headroom(cells, {"merge": None}, ["merge"])["merge"]
        pooled = kind_schedule.headroom(cells, scoring.Budgets({"merge": None}, positive_ratio=1.,
                                                               aggregation="positive-weighted"), ["merge"])["merge"]
        self.assertAlmostEqual(equal, ((1 - 0.25) + (1 - 0.75)) / 2)
        self.assertAlmostEqual(pooled, (44 - 31) / 44)

    def test_paired_bootstrap_uses_the_pooled_delta(self):
        train = three_brains()
        tables = {b: train[b] for b in ('802449', '747807')}
        # detector_score ranks the alternating labels at half precision; signal puts every positive first.
        state = {f'{b}/merge': {'scores': tables[b].tables['merge'].features['detector_score'].to_numpy()}
                 for b in tables}
        budgets = scoring.Budgets({"merge": 4}, aggregation="positive-weighted")
        same = scoring.paired_bootstrap(state, state, tables, budgets, draws=40)
        self.assertEqual(same["aggregation"], scoring.AGGREGATION_LABELS["positive-weighted"])
        self.assertEqual(same["macro_delta_ci"], [0., 0.])
        better = {f'{b}/merge': {'scores': tables[b].tables['merge'].features['signal'].to_numpy()} for b in tables}
        gain = scoring.paired_bootstrap(state, better, tables, budgets, draws=40)
        self.assertGreater(gain["macro_delta_ci"][0], 0.)


class DriverFlagTests(unittest.TestCase):
    def test_flags_parse_with_the_new_defaults(self):
        from proofreader_evolve.cli import run_precision_evolution as driver
        args = driver.parse_args([])
        self.assertEqual(args.aggregation, "positive-weighted")
        self.assertIsNone(args.selection_holdout_brains)
        args = driver.parse_args(["--aggregation", "equal-brain", "--selection-holdout-brains", "747807,754610"])
        self.assertEqual(args.aggregation, "equal-brain")
        self.assertEqual(args.selection_holdout_brains, ["747807", "754610"])
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            driver.parse_args(["--aggregation", "median"])


if __name__ == "__main__":
    unittest.main()
