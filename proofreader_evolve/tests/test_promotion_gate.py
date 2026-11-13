"""--promotion-gate rule (default since 2026-11-12): the table reproduces the paired bootstrap's
interval, fragment groups are resampled, the guard rail and the rule on the objective decide and
fail closed, and the driver records the table and the rejection category.
Synthetic fixtures only; no brain caches, cloud reads or LLM calls."""

from contextlib import redirect_stdout
import io
import json
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from proofreader_evolve.harness import fixed_pool_scoring as scoring
from proofreader_evolve.harness import promotion_gate
from proofreader_evolve.harness.rank_statistics import aggregate_statistics
from proofreader_evolve.tests import label_fixture
from proofreader_evolve.tests.test_detector_prior import driver_banks, run_driver

setUpModule = label_fixture.setUpModule
tearDownModule = label_fixture.tearDownModule


def synthetic(seed=0, brains=("a", "b", "c"), rows=400):
    """Three pools with NaN labels, ties and two scorers of different quality."""
    rng = np.random.default_rng(seed)
    tables, parent, candidate = {}, {}, {}
    for index, brain in enumerate(brains):
        truth = (rng.random(rows) < .15).astype(float)
        truth[rng.random(rows) < .2] = np.nan
        signal = np.nan_to_num(truth) + rng.normal(0, 1.2, rows)
        table = SimpleNamespace(truth=truth, keys=[f"{brain}-{i:04d}" for i in range(rows)])
        tables[brain] = SimpleNamespace(tables={"merge": table})
        parent[f"{brain}/merge"] = {"scores": np.round(signal + rng.normal(0, 1.5, rows), 1)}
        candidate[f"{brain}/merge"] = {"scores": np.round(signal + rng.normal(0, .4, rows), 1)}
    return tables, parent, candidate


def budgets(objective, aggregation="positive-weighted"):
    return scoring.Budgets({"merge": None}, positive_ratio=1., min_positives=0, aggregation=aggregation,
                           objective=objective)


class StatisticsTests(unittest.TestCase):
    def test_the_objective_interval_equals_the_paired_bootstraps(self):
        tables, parent, candidate = synthetic()
        for aggregation in ("positive-weighted", "equal-brain"):
            for objective, name in (("auroc", "auroc"), ("average_precision", "average_precision"),
                                    ("pauc", "pauc"), ("precision_at_k", "precision")):
                with self.subTest(objective=objective, aggregation=aggregation):
                    rule = budgets(objective, aggregation)
                    reference = scoring.paired_bootstrap(parent, candidate, tables, rule, target_kind="merge",
                                                         draws=200, seed=7, alpha=.05)
                    statistics, pools = promotion_gate.joint_intervals(
                        parent, candidate, tables, rule, target_kind="merge", draws=200, seed=7, alpha=.05)
                    self.assertAlmostEqual(statistics[name]["low"], reference["delta_ci"][0], places=12)
                    self.assertAlmostEqual(statistics[name]["high"], reference["delta_ci"][1], places=12)
                    self.assertEqual(pools["pools"], 3)

    def test_point_statistics_equal_the_reports(self):
        tables, parent, candidate = synthetic(seed=3)
        rule = budgets("auroc")
        statistics, pools = promotion_gate.joint_intervals(parent, candidate, tables, rule, target_kind="merge",
                                                           draws=20, seed=0)
        for arm, state in (("parent", parent), ("candidate", candidate)):
            cells = {}
            for brain, bank in tables.items():
                table = bank.tables["merge"]
                k = rule.k_for("merge", table.truth)
                metrics, _ = scoring.rank_metrics(state[f"{brain}/merge"]["scores"], table.truth, table.keys, k)
                cells[f"{brain}/merge"] = metrics
            expected = aggregate_statistics(cells, "positive-weighted")
            for name in promotion_gate.STATISTICS:
                with self.subTest(arm=arm, statistic=name):
                    self.assertAlmostEqual(statistics[name][arm], expected[name], places=12)
        self.assertEqual(pools["positives"], int(sum(np.nansum(b.tables["merge"].truth) for b in tables.values())))

    def test_cells_below_the_floor_are_drawn_but_not_counted(self):
        tables, parent, candidate = synthetic()
        tables["c"].tables["merge"].truth[:] = np.where(np.arange(400) < 2, 1., 0.)  # two positives only
        rule = scoring.Budgets({"merge": None}, positive_ratio=1., min_positives=10,
                               aggregation="positive-weighted", objective="auroc")
        reference = scoring.paired_bootstrap(parent, candidate, tables, rule, target_kind="merge", draws=100, seed=1)
        statistics, pools = promotion_gate.joint_intervals(parent, candidate, tables, rule, target_kind="merge",
                                                           draws=100, seed=1)
        self.assertEqual(pools["pools"], 2)
        self.assertAlmostEqual(statistics["auroc"]["low"], reference["delta_ci"][0], places=12)


class ResamplingGroupTests(unittest.TestCase):
    def test_groups_are_fragment_segments(self):
        merge = SimpleNamespace(truth=np.zeros(4), candidates=[{"segment_id": s} for s in (7, 7, 3, 9)])
        self.assertEqual(scoring.resampling_groups("merge", merge).tolist(), [1, 1, 0, 2])
        split = SimpleNamespace(truth=np.zeros(3), candidates=[
            {"segment_id_a": 5, "segment_id_b": 8, "anchor_side": "a"},
            {"segment_id_a": 6, "segment_id_b": 5, "anchor_side": "b"},   # the anchor fragment is 5 again
            {"segment_id_a": 6, "segment_id_b": 5, "anchor_side": "a"}])
        self.assertEqual(scoring.resampling_groups("split", split).tolist(), [0, 0, 1])
        # No candidate rows: one group per row, which is row resampling.
        self.assertEqual(scoring.resampling_groups("merge", SimpleNamespace(truth=np.zeros(3))).tolist(), [0, 1, 2])
        with self.assertRaises(ValueError):
            scoring.resampling_groups("merge", SimpleNamespace(truth=np.zeros(2), candidates=[{"segment_id": 1}]))
        groups, count = scoring.labeled_groups("merge", merge, np.array([0, 1, 3]))
        self.assertEqual((groups.tolist(), count), ([0, 0, 1], 2))

    def test_rows_of_one_fragment_share_a_weight_and_the_interval_widens(self):
        tables, parent, candidate = synthetic(rows=600)
        rule = budgets("average_precision")
        by_row = scoring.paired_bootstrap(parent, candidate, tables, rule, target_kind="merge", draws=400, seed=3)
        for bank in tables.values():   # twenty rows per fragment
            table = bank.tables["merge"]
            table.candidates = [{"segment_id": index // 20} for index in range(len(table.truth))]
        grouped = scoring.paired_bootstrap(parent, candidate, tables, rule, target_kind="merge", draws=400, seed=3)
        self.assertEqual(grouped["version"], scoring.BOOTSTRAP_VERSION)
        self.assertTrue(all(count <= 30 for count in grouped["groups"].values()))
        self.assertTrue(all(count > 300 for count in by_row["groups"].values()))
        width = lambda result: result["delta_ci"][1] - result["delta_ci"][0]
        # Random labels inside a fragment carry no within-fragment correlation, so the grouped interval is
        # of the same order; it must not collapse and the two must differ (other weights were drawn).
        self.assertGreater(width(grouped), .5 * width(by_row))
        self.assertNotEqual(grouped["delta_ci"], by_row["delta_ci"])
        statistics, pools = promotion_gate.joint_intervals(parent, candidate, tables, rule, target_kind="merge",
                                                           draws=400, seed=3)
        self.assertAlmostEqual(statistics["average_precision"]["low"], grouped["delta_ci"][0], places=12)
        self.assertEqual(pools["groups"], sum(grouped["groups"].values()))


class DecisionTests(unittest.TestCase):
    common = dict(target_kind="merge", draws=100, seed=1)

    def table(self, low=None, high=None, delta=.1):
        return {name: {"parent": .5, "candidate": .5 + delta, "delta": delta, "low": (low or {}).get(name, .01),
                       "high": (high or {}).get(name, .2)} for name in promotion_gate.STATISTICS}

    def test_the_guard_rail(self):
        self.assertTrue(promotion_gate.guard_rail(self.table()))
        self.assertFalse(promotion_gate.guard_rail(self.table(high={"average_precision": -.001})))
        # A single-cut precision below zero is not guarded by itself.
        self.assertTrue(promotion_gate.guard_rail(self.table(high={"precision_at_half_k": -.01})))
        undefined = self.table()
        undefined["auroc"]["high"] = None
        self.assertFalse(promotion_gate.guard_rail(undefined))

    def test_the_rule_and_the_rejection_category_are_functions_of_the_table(self):
        passed = promotion_gate.rule_check(self.table(), "average_precision", .05)
        self.assertEqual((passed["statistic"], passed["significant"], passed["large_enough"], passed["passed"]),
                         ("average_precision", True, True, True))
        small = promotion_gate.rule_check(self.table(), "average_precision", .1)   # the gain must exceed the margin
        self.assertEqual((small["significant"], small["large_enough"], small["passed"]), (True, False, False))
        noise = promotion_gate.rule_check(self.table(low={"pauc": -.01}), "pauc")
        self.assertEqual((noise["significant"], noise["passed"]), (False, False))
        self.assertEqual(promotion_gate.rule_check(self.table(), "precision_at_k")["statistic"], "precision")
        category = promotion_gate.rejection_category
        self.assertEqual(category(self.table(), small), "gain_too_small")
        self.assertEqual(category(self.table(low={"pauc": -.01}), noise), "within_noise")
        # Worse on one guarded statistic wins over a gain in the objective.
        self.assertEqual(category(self.table(high={"auroc": -.001}), passed), "significantly_worse")
        self.assertEqual(promotion_gate.REJECTION_CATEGORIES, ("significantly_worse", "within_noise", "gain_too_small"))

    def test_the_gate_decides_on_the_objective_and_fails_closed(self):
        tables, parent, candidate = synthetic()
        rule = budgets("average_precision")
        accepted = promotion_gate.gate(parent, candidate, tables, rule, **self.common)
        self.assertEqual((accepted["passed"], accepted["status"], accepted["rejection_category"]), (True, "rule", None))
        self.assertEqual(accepted["rule"]["statistic"], "average_precision")
        self.assertIn("accepted by the promotion rule", accepted["reason"])
        self.assertEqual(accepted["resampling_unit"], scoring.RESAMPLING_UNIT)
        small = promotion_gate.gate(parent, candidate, tables, rule, margin=.9, **self.common)
        self.assertEqual((small["passed"], small["rejection_category"]), (False, "gain_too_small"))
        # The better scorer as the parent: significantly worse.
        worse = promotion_gate.gate(candidate, parent, tables, rule, **self.common)
        self.assertEqual((worse["passed"], worse["status"], worse["rejection_category"]),
                         (False, "guard_rail", "significantly_worse"))
        self.assertIn("rejected by the guard rail", worse["reason"])
        # Identical rankings: every delta is zero, the guard passes and nothing is beyond noise.
        same = promotion_gate.gate(parent, parent, tables, rule, **self.common)
        self.assertEqual((same["passed"], same["status"], same["rejection_category"]), (False, "rule", "within_noise"))
        broken = promotion_gate.gate({}, candidate, tables, rule, **self.common)
        self.assertEqual((broken["passed"], broken["status"], broken["rejection_category"]), (False, "unavailable", None))

    def test_the_rule_follows_the_runs_objective(self):
        tables, parent, candidate = synthetic()
        for objective, name in (("pauc", "pauc"), ("precision_at_k", "precision"), ("auroc", "auroc")):
            record = promotion_gate.gate(parent, candidate, tables, budgets(objective), **self.common)
            self.assertEqual(record["rule"]["statistic"], name)
            self.assertEqual(record["rule"]["low"], record["statistics"][name]["low"])

    def test_gate_modes_and_versions(self):
        self.assertEqual(promotion_gate.gate_modes(), ("rule", "margin"))
        self.assertEqual(promotion_gate.gate_version("rule"), promotion_gate.GATE_VERSION)
        self.assertEqual(promotion_gate.gate_version("margin"), scoring.gate_version("margin"))
        with self.assertRaises(ValueError):
            promotion_gate.gate_version("judge")


class DriverTests(unittest.TestCase):
    def run_once(self, *extra):
        from proofreader_evolve.cli import run_precision_evolution as driver
        banks = driver_banks()

        async def revise(run_dir, policy, rules, report, model, *, experiments):
            policy.write_text("def score_candidates(features, ctx):\n"
                              "    return features['signal'].fillna(0).to_numpy()\n")
            experiments.proposal_path.write_text(json.dumps({'hypothesis': 'signal ranks positives',
                'strategy': 'use signal', 'family': 'signal', 'parameter_grid': {}}))
            experiments.evaluate()
            return {'summary': 'signal', 'cost_usd': 0}

        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            args = driver.parse_args(['--min-positives', '0', '--train-brains', '794495,802449', '--validation-brains', '2',
                                      '--merge-k', '3', '--split-k', '1', '--generations', '1', '--runs-dir', tmp,
                                      '--skip-auxiliary-ablation', '--bootstrap-draws', '50',
                                      '--objective', 'average_precision', '--kind-schedule', 'alternate',
                                      '--detector-prior', 'no-score', *extra])
            path = run_driver(driver, args, banks, revise)
            manifest = json.loads((path / 'manifest.json').read_text())
            record = json.loads((path / 'ledger.jsonl').read_text().splitlines()[0])
            record['_pool'] = json.loads((path / 'candidate_pool.json').read_text())
        return manifest, record

    def test_the_driver_applies_and_records_the_rule(self):
        manifest, record = self.run_once()
        gate = record['validation_gate']
        self.assertEqual(manifest['promotion_gate']['mode'], 'rule')   # the default
        self.assertEqual(manifest['promotion_gate']['rule']['statistic'], 'average_precision')
        self.assertTrue(manifest['promotion_gate']['bootstrap']['affects_decision'])
        self.assertEqual(manifest['objective_statistic'], 'average_precision')
        self.assertEqual((gate['mode'], gate['version']), ('rule', promotion_gate.GATE_VERSION))
        table = gate['gate_table']
        self.assertEqual(table['status'], 'rule')
        self.assertEqual(record['accepted'], table['rule']['passed'])
        self.assertEqual(set(table['statistics']), set(promotion_gate.STATISTICS))
        self.assertIsNotNone(gate['candidate_objective'])  # kind scheduling reads the objective's gain
        for removed in ('judge', 'bootstrap_gate', 'paired_bootstrap', 'head_guard'):
            self.assertNotIn(removed, gate)
        if record['accepted']:
            self.assertIsNone(table['rejection_category'])
            self.assertNotIn('rejection', record['submission_validated'])

    def test_a_rejection_reaches_later_sessions_as_one_word(self):
        # A minimum effect no gain can reach: the candidate is rejected with a category.
        _, record = self.run_once('--precision-margin', '2')
        self.assertFalse(record['accepted'])
        category = record['validation_gate']['gate_table']['rejection_category']
        self.assertIn(category, ('gain_too_small', 'within_noise'))
        self.assertEqual(record['submission_validated']['rejection'], category)
        # What the next reviser session can read: the category on the measured candidate and in the
        # outcome history, and nothing else of the table.
        pool = json.dumps(record['_pool'])
        self.assertIn(f'"rejection": "{category}"', pool)
        for hidden in ('gate_table', 'guard_rail', 'resampling_unit', 'large_enough'):
            self.assertNotIn(hidden, pool)

    def test_gate_options(self):
        from proofreader_evolve.cli import run_precision_evolution as driver
        for bad in (['--bootstrap-draws', '0'], ['--promotion-gate', 'judge'], ['--promotion-gate', 'bootstrap'],
                    ['--judge-votes', '3'], ['--head-guard-statistic', 'none']):
            with self.subTest(bad=bad), redirect_stdout(io.StringIO()), patch('sys.stderr', io.StringIO()), \
                    self.assertRaises(SystemExit):
                driver.parse_args(bad)
        self.assertEqual(driver.parse_args(['--promotion-gate', 'margin', '--bootstrap-draws', '0']).promotion_gate, 'margin')
