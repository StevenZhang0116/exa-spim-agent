"""--promotion-gate judge (2026-11-10): the judge's table reproduces the bootstrap gate's interval,
the host decides inside the guard rail and fails closed, and the driver records every vote.
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
from proofreader_evolve.harness import promotion_judge
from proofreader_evolve.harness.rank_statistics import aggregate_statistics, cell_statistics
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


def ballots(*decisions):
    return [{"decision": d, "reason": "r"} if d in ("accept", "reject") else {"error": d} for d in decisions]


class StatisticsTests(unittest.TestCase):
    def test_the_objective_interval_equals_the_bootstrap_gates(self):
        tables, parent, candidate = synthetic()
        for aggregation in ("positive-weighted", "equal-brain"):
            for objective, name in (("auroc", "auroc"), ("average_precision", "average_precision"),
                                    ("pauc", "pauc"), ("precision_at_k", "precision")):
                with self.subTest(objective=objective, aggregation=aggregation):
                    rule = budgets(objective, aggregation)
                    reference = scoring.paired_bootstrap(parent, candidate, tables, rule, target_kind="merge",
                                                         draws=200, seed=7, alpha=.05)
                    statistics, pools = promotion_judge.joint_intervals(
                        parent, candidate, tables, rule, target_kind="merge", draws=200, seed=7, alpha=.05)
                    self.assertAlmostEqual(statistics[name]["low"], reference["delta_ci"][0], places=12)
                    self.assertAlmostEqual(statistics[name]["high"], reference["delta_ci"][1], places=12)
                    self.assertEqual(pools["pools"], 3)

    def test_point_statistics_equal_the_reports(self):
        tables, parent, candidate = synthetic(seed=3)
        rule = budgets("auroc")
        statistics, pools = promotion_judge.joint_intervals(parent, candidate, tables, rule, target_kind="merge",
                                                            draws=20, seed=0)
        for arm, state in (("parent", parent), ("candidate", candidate)):
            cells = {}
            for brain, bank in tables.items():
                table = bank.tables["merge"]
                k = rule.k_for("merge", table.truth)
                metrics, _ = scoring.rank_metrics(state[f"{brain}/merge"]["scores"], table.truth, table.keys, k)
                cells[f"{brain}/merge"] = metrics
            expected = aggregate_statistics(cells, "positive-weighted")
            for name in promotion_judge.STATISTICS:
                with self.subTest(arm=arm, statistic=name):
                    self.assertAlmostEqual(statistics[name][arm], expected[name], places=12)
        self.assertEqual(pools["positives"], int(sum(np.nansum(b.tables["merge"].truth) for b in tables.values())))
        self.assertIsNotNone(cell_statistics)  # the per-cell definitions the judge's table shares

    def test_cells_below_the_floor_are_drawn_but_not_counted(self):
        tables, parent, candidate = synthetic()
        tables["c"].tables["merge"].truth[:] = np.where(np.arange(400) < 2, 1., 0.)  # two positives only
        rule = scoring.Budgets({"merge": None}, positive_ratio=1., min_positives=10,
                               aggregation="positive-weighted", objective="auroc")
        reference = scoring.paired_bootstrap(parent, candidate, tables, rule, target_kind="merge", draws=100, seed=1)
        statistics, pools = promotion_judge.joint_intervals(parent, candidate, tables, rule, target_kind="merge",
                                                            draws=100, seed=1)
        self.assertEqual(pools["pools"], 2)
        self.assertAlmostEqual(statistics["auroc"]["low"], reference["delta_ci"][0], places=12)


class DecisionTests(unittest.TestCase):
    def table(self, **high):
        return {name: {"parent": .5, "candidate": .6, "delta": .1, "low": .01, "high": high.get(name, .2)}
                for name in promotion_judge.STATISTICS}

    def test_a_strict_majority_of_the_requested_votes_is_needed(self):
        self.assertEqual(promotion_judge.decide(self.table(), ballots("accept", "accept", "accept", "reject", "reject"), 5),
                         (True, 3, 5))
        self.assertEqual(promotion_judge.decide(self.table(), ballots("accept", "accept", "reject", "reject"), 4),
                         (False, 2, 4))
        # Failed calls count against the candidate: two accepts of five requested are no majority.
        self.assertEqual(promotion_judge.decide(self.table(), ballots("accept", "accept", "timeout", "timeout", "x"), 5),
                         (False, 2, 2))

    def test_the_guard_rail_overrides_unanimous_votes(self):
        table = self.table(average_precision=-.001)
        self.assertFalse(promotion_judge.guard_rail(table))
        self.assertEqual(promotion_judge.decide(table, ballots(*["accept"] * 5), 5), (False, 5, 5))
        # A single-cut precision below zero is not guarded by itself.
        self.assertTrue(promotion_judge.guard_rail(self.table(precision_at_half_k=-.01)))
        undefined = self.table()
        undefined["auroc"]["high"] = None
        self.assertFalse(promotion_judge.guard_rail(undefined))

    def test_the_prompt_carries_numbers_only(self):
        text = promotion_judge.prompt("merge", {"pools": 5, "labeled": 100, "positives": 9}, self.table())
        self.assertIn("95% interval", text)
        self.assertIn("| average precision | 0.5000 | 0.6000 | +0.1000 | [+0.0100, +0.2000] |", text)
        for forbidden in ("gen", "precision_2026", "attempt", "summary"):
            self.assertNotIn(forbidden, text)
        self.assertNotIn("interval", promotion_judge.prompt("merge", {"pools": 5, "labeled": 100, "positives": 9},
                                                              self.table(), with_intervals=False))

    def test_judge_fails_closed(self):
        tables, parent, candidate = synthetic()
        rule = budgets("average_precision")
        common = dict(target_kind="merge", draws=100, seed=1, votes=5)
        with patch.object(promotion_judge, "cast_votes", return_value=ballots(*["accept"] * 5)) as cast:
            accepted = promotion_judge.judge(parent, candidate, tables, rule, **common)
        self.assertTrue(accepted["passed"])
        self.assertEqual((accepted["accept_votes"], accepted["votes_cast"], accepted["status"]), (5, 5, "voted"))
        self.assertIn("accepted by the LLM judge", accepted["reason"])
        self.assertNotIn("precision_2026", cast.call_args.args[0])
        with patch.object(promotion_judge, "cast_votes", side_effect=RuntimeError("network")):
            failed = promotion_judge.judge(parent, candidate, tables, rule, **common)
        self.assertFalse(failed["passed"])
        self.assertEqual(failed["votes_cast"], 0)
        # The better scorer as the parent: significantly worse, so the model is never asked.
        with patch.object(promotion_judge, "cast_votes") as cast:
            worse = promotion_judge.judge(candidate, parent, tables, rule, **common)
        self.assertFalse(worse["passed"])
        self.assertEqual(worse["status"], "guard_rail")
        cast.assert_not_called()
        broken = promotion_judge.judge({}, candidate, tables, rule, **common)
        self.assertEqual((broken["passed"], broken["status"]), (False, "unavailable"))

    def test_rejection_category_is_a_function_of_the_intervals(self):
        def table(low=None, high=None):
            return {name: {"parent": .5, "candidate": .5, "delta": 0., "low": (low or {}).get(name, -.01),
                           "high": (high or {}).get(name, .01)} for name in promotion_judge.STATISTICS}
        category = promotion_judge.rejection_category
        self.assertEqual(category(table()), "within_noise")
        self.assertEqual(category(table(low={"auroc": .001})), "head_not_improved")
        self.assertEqual(category(table(low={"auroc": .001, "average_precision": .002})), "not_convincing")
        self.assertEqual(category(table(low={"precision": .01})), "not_convincing")
        # Worse on one guarded statistic wins over better on another.
        self.assertEqual(category(table(low={"auroc": .001}, high={"pauc": -.001})), "significantly_worse")
        # A single-cut precision that is neither guarded nor a head statistic changes nothing.
        self.assertEqual(category(table(low={"precision_at_half_k": .01})), "within_noise")
        for value in (category(table()), category(table(low={"auroc": .001}))):
            self.assertIn(value, promotion_judge.REJECTION_CATEGORIES)

    def test_judge_records_a_category_only_for_a_rejection_with_votes(self):
        tables, parent, candidate = synthetic()
        rule = budgets("average_precision")
        common = dict(target_kind="merge", draws=100, seed=1, votes=5)
        with patch.object(promotion_judge, "cast_votes", return_value=ballots(*["accept"] * 5)):
            self.assertIsNone(promotion_judge.judge(parent, candidate, tables, rule, **common)["rejection_category"])
        with patch.object(promotion_judge, "cast_votes", return_value=ballots(*["reject"] * 5)):
            rejected = promotion_judge.judge(parent, candidate, tables, rule, **common)
        self.assertEqual(rejected["rejection_category"], "not_convincing")  # the better scorer, voted down
        with patch.object(promotion_judge, "cast_votes", return_value=ballots(*["timeout"] * 5)):
            self.assertIsNone(promotion_judge.judge(parent, candidate, tables, rule, **common)["rejection_category"])
        with patch.object(promotion_judge, "cast_votes"):
            worse = promotion_judge.judge(candidate, parent, tables, rule, **common)
        self.assertEqual(worse["rejection_category"], "significantly_worse")
        self.assertIsNone(promotion_judge.judge({}, candidate, tables, rule, **common)["rejection_category"])

    def test_gate_modes_and_versions(self):
        self.assertEqual(promotion_judge.gate_modes(), ("bootstrap", "margin", "judge"))
        self.assertEqual(promotion_judge.gate_version("judge"), promotion_judge.GATE_VERSION)
        self.assertEqual(promotion_judge.gate_version("bootstrap"), scoring.gate_version("bootstrap"))


class DriverTests(unittest.TestCase):
    def arguments(self, driver, tmp, *extra):
        return driver.parse_args(['--min-positives', '0', '--train-brains', '794495,802449', '--validation-brains', '2',
                                  '--merge-k', '3', '--split-k', '1', '--generations', '1', '--runs-dir', tmp,
                                  '--skip-auxiliary-ablation', '--bootstrap-draws', '50', '--promotion-gate', 'judge',
                                  '--objective', 'average_precision', '--kind-schedule', 'alternate',
                                  '--detector-prior', 'no-score', *extra])

    def run_once(self, decisions):
        from proofreader_evolve.cli import run_precision_evolution as driver
        banks = driver_banks()

        async def revise(run_dir, policy, rules, report, model, *, experiments):
            policy.write_text("def score_candidates(features, ctx):\n"
                              "    return features['signal'].fillna(0).to_numpy()\n")
            experiments.proposal_path.write_text(json.dumps({'hypothesis': 'signal ranks positives',
                'strategy': 'use signal', 'family': 'signal', 'parameter_grid': {}}))
            experiments.evaluate()
            return {'summary': 'signal', 'cost_usd': 0}

        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()), \
                patch.object(promotion_judge, "cast_votes", return_value=ballots(*decisions)) as cast:
            path = run_driver(driver, self.arguments(driver, tmp), banks, revise)
            manifest = json.loads((path / 'manifest.json').read_text())
            record = json.loads((path / 'ledger.jsonl').read_text().splitlines()[0])
            record['_pool'] = json.loads((path / 'candidate_pool.json').read_text())
        return manifest, record, cast

    def test_the_driver_applies_and_records_the_judges_decision(self):
        manifest, record, cast = self.run_once(["accept"] * 5)
        gate = record['validation_gate']
        self.assertEqual(manifest['promotion_gate']['mode'], 'judge')
        self.assertEqual(manifest['promotion_gate']['judge']['votes'], 5)
        self.assertEqual(manifest['objective_statistic'], 'average_precision')
        self.assertTrue(record['accepted'])
        self.assertEqual((gate['mode'], gate['version']), ('judge', promotion_judge.GATE_VERSION))
        self.assertEqual(gate['judge']['accept_votes'], 5)
        self.assertEqual(len(gate['judge']['ballots']), 5)
        self.assertFalse(gate['bootstrap_gate']['binding'])
        self.assertIn('accepted by the LLM judge', record['reason'])
        self.assertEqual(cast.call_count, 1)
        self.assertIsNotNone(gate['candidate_objective'])  # kind scheduling still reads the objective's gain
        self.assertIsNone(gate['judge']['rejection_category'])
        self.assertNotIn('rejection', record['submission_validated'])

    def test_a_rejecting_judge_keeps_the_parent(self):
        _, record, _ = self.run_once(["reject", "reject", "reject", "accept", "accept"])
        self.assertFalse(record['accepted'])
        self.assertIn('rejected by the LLM judge', record['reason'])
        self.assertEqual(record['validation_gate']['judge']['accept_votes'], 2)
        category = record['validation_gate']['judge']['rejection_category']
        self.assertIn(category, promotion_judge.REJECTION_CATEGORIES)
        self.assertEqual(record['submission_validated']['rejection'], category)
        # What the next reviser session can read: the category on the measured candidate, and nothing else
        # of the judge (no ballot, reason or heldout statistic).
        pool = json.dumps(record['_pool'])
        self.assertIn(f'"rejection": "{category}"', pool)
        for hidden in ('ballots', 'accept_votes', 'votes_cast', 'guard_rail', 'prompt_sha256'):
            self.assertNotIn(hidden, pool)

    def test_the_judge_gate_needs_bootstrap_draws(self):
        from proofreader_evolve.cli import run_precision_evolution as driver
        with redirect_stdout(io.StringIO()), patch('sys.stderr', io.StringIO()), self.assertRaises(SystemExit):
            driver.parse_args(['--promotion-gate', 'judge', '--bootstrap-draws', '0'])
        self.assertEqual(driver.parse_args(['--promotion-gate', 'judge']).judge_votes, promotion_judge.DEFAULT_VOTES)


if __name__ == '__main__':
    unittest.main()
