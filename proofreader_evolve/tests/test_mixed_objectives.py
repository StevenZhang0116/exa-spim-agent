"""--objective merge=NAME,split=NAME (2026-11-13): one objective per kind for TRAIN selection, the
bootstrap, the gate, scheduling and the records; a single name keeps every record as it was.
Synthetic fixtures only; no brain caches, cloud reads or LLM calls."""

import asyncio
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from proofreader_evolve.harness import fixed_pool_scoring as scoring
from proofreader_evolve.harness import kind_schedule, promotion_gate, rank_statistics
from proofreader_evolve.harness.candidate_pool import CandidatePool, POOL_TOLERANCES
from proofreader_evolve.tests.label_fixture import setUpModule, tearDownModule  # noqa: F401
from proofreader_evolve.tests.test_kind_schedule import IMPROVED, two_kinds

MIXED = {"merge": "average_precision", "split": "pauc"}


def pools(seed=0, rows=300):
    """One brain with a merge and a split pool and two scorers of different quality."""
    rng = np.random.default_rng(seed)
    tables, parent, candidate = {}, {}, {}
    for kind, rate in (("merge", .1), ("split", .4)):
        truth = (rng.random(rows) < rate).astype(float)
        signal = truth + rng.normal(0, 1., rows)
        tables[kind] = SimpleNamespace(truth=truth, keys=[f"{kind}-{i:04d}" for i in range(rows)])
        parent[f"a/{kind}"] = {"scores": signal + rng.normal(0, 2., rows)}
        candidate[f"a/{kind}"] = {"scores": signal + rng.normal(0, .3, rows)}
    return {"a": SimpleNamespace(tables=tables)}, parent, candidate


def budgets(objective):
    return scoring.Budgets({"merge": None, "split": None}, positive_ratio=1., aggregation="positive-weighted",
                           objective=objective)


class ObjectiveArgumentTests(unittest.TestCase):
    def test_parsing_and_text(self):
        parse = rank_statistics.parse_objective
        self.assertEqual(parse("auroc"), "auroc")
        self.assertEqual(parse("merge=average_precision, split=pauc"), MIXED)
        for bad in ("f1", "merge=f1,split=pauc", "merge=auroc,merge=pauc", "=auroc"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                parse(bad)
        self.assertEqual(rank_statistics.objective_text(MIXED), "merge: average precision; split: partial AUROC")
        self.assertEqual(rank_statistics.objective_text(MIXED, labels=False), "merge=average_precision,split=pauc")
        self.assertEqual(rank_statistics.objective_text("auroc"), "AUROC")
        self.assertEqual(rank_statistics.objective_record({"merge": "auroc", "split": "auroc"}), "auroc")
        self.assertEqual(rank_statistics.objective_of(None, "merge"), "precision_at_k")   # reports before v7

    def test_driver_option(self):
        from proofreader_evolve.cli.run_precision_evolution import parse_args
        self.assertEqual(parse_args([]).objective, "auroc")
        self.assertEqual(parse_args(["--objective", "merge=average_precision,split=pauc"]).objective, MIXED)
        for bad in (["--objective", "merge=average_precision"], ["--objective", "f1"],
                    ["--objective", "merge=auroc,other=pauc"]):
            with self.subTest(bad=bad), redirect_stdout(io.StringIO()), patch("sys.stderr", io.StringIO()), \
                    self.assertRaises(SystemExit):
                parse_args(bad)


class BudgetsTests(unittest.TestCase):
    def test_one_name_keeps_the_old_records(self):
        single = budgets("average_precision")
        self.assertEqual((single.objective, single.objective_for("split"), single.objective_record()),
                         ("average_precision",) * 3)
        described = single.describe()
        self.assertEqual((described["objective"], described["objective_label"]), ("average_precision", "average precision"))
        self.assertIsInstance(described["objective_rule"], str)
        same = budgets({"merge": "auroc", "split": "auroc"})
        self.assertEqual((same.objective, same.objective_record()), ("auroc", "auroc"))

    def test_an_objective_per_kind(self):
        mixed = budgets(MIXED)
        self.assertEqual((mixed.objective_for("merge"), mixed.objective_for("split")), ("average_precision", "pauc"))
        self.assertEqual(mixed.objective_record(), MIXED)
        with self.assertRaisesRegex(ValueError, "differs by kind"):
            mixed.objective
        described = mixed.describe()
        self.assertEqual(described["objective"], MIXED)
        self.assertEqual(set(described["objective_rule"]), {"merge", "split"})
        with self.assertRaisesRegex(ValueError, "exactly the budgeted kinds"):
            budgets({"merge": "auroc"})
        with self.assertRaises(ValueError):
            mixed.objective_for("other")


class ScoringTests(unittest.TestCase):
    def setUp(self):
        self.tables, self.parent, self.candidate = pools()
        self.reports = {}
        for name, state in (("parent", self.parent), ("candidate", self.candidate)):
            cells = {}
            for kind, table in self.tables["a"].tables.items():
                cells[f"a/{kind}"], _ = scoring.rank_metrics(state[f"a/{kind}"]["scores"], table.truth, table.keys,
                                                              int(np.nansum(table.truth)))
                cells[f"a/{kind}"].update(pool_sha256=f"pool-{kind}", labels_sha256=f"labels-{kind}")
            report = {"version": scoring.SCORING_VERSION, "objective": MIXED, "cells": cells}
            report.update(scoring.aggregate_objective(cells, "positive-weighted", MIXED))
            self.reports[name] = report

    def test_each_kind_is_aggregated_under_its_own_objective(self):
        report = self.reports["candidate"]
        self.assertEqual(report["objective"], MIXED)
        self.assertEqual(report["objective_label"], "merge: average precision; split: partial AUROC")
        self.assertEqual(report["per_kind_objective"]["merge"], report["per_kind_statistics"]["merge"]["average_precision"])
        self.assertEqual(report["per_kind_objective"]["split"], report["per_kind_statistics"]["split"]["pauc"])
        self.assertEqual(scoring.objective_for_kind(report, "split"), report["per_kind_objective"]["split"])
        self.assertEqual(scoring.report_objective(report, "merge"), "average_precision")
        with self.assertRaisesRegex(ValueError, "name the kind"):
            scoring.report_objective(report)
        # One kind's cells under the mapping record that kind's name alone.
        merge_only = scoring.aggregate_objective({"a/merge": report["cells"]["a/merge"]}, "positive-weighted", MIXED)
        self.assertEqual(merge_only["objective"], "average_precision")

    def test_point_gate_bootstrap_and_rule_read_the_kinds_objective(self):
        for kind, name in MIXED.items():
            details = scoring.gate_details(self.reports["parent"], self.reports["candidate"], kind)
            self.assertEqual(details["objective"], name)
            passed, reason = scoring.acceptance(self.reports["parent"], self.reports["candidate"], target_kind=kind,
                                                require_no_cell_regression=False)
            self.assertTrue(passed, reason)
            self.assertIn(rank_statistics.OBJECTIVE_LABELS[name], reason)
            interval = scoring.paired_bootstrap(self.parent, self.candidate, self.tables, budgets(MIXED),
                                                target_kind=kind, draws=100, seed=2)
            self.assertEqual(interval["objective"], name)
            record = promotion_gate.gate(self.parent, self.candidate, self.tables, budgets(MIXED), target_kind=kind,
                                         draws=100, seed=2)
            self.assertEqual(record["rule"]["statistic"], name)
            self.assertEqual(record["rule"]["low"], interval["delta_ci"][0])
            self.assertIn(rank_statistics.OBJECTIVE_LABELS[name], record["reason"])

    def test_scheduling_reads_each_kinds_objective(self):
        cells = self.reports["candidate"]["cells"]
        headroom = kind_schedule.headroom(cells, budgets(MIXED), ["merge", "split"])
        self.assertAlmostEqual(headroom["merge"], 1. - cells["a/merge"]["average_precision"])
        self.assertAlmostEqual(headroom["split"], 1. - cells["a/split"]["pauc"])
        self.assertEqual(set(kind_schedule.saturation_thresholds(cells, budgets(MIXED), ["merge", "split"])),
                         {"merge", "split"})

    def test_the_branch_archive_keeps_a_tolerance_per_kind(self):
        with tempfile.TemporaryDirectory() as tmp:
            mixed = CandidatePool(Path(tmp) / "a.json", ["merge", "split"], objective={"merge": "auroc", "split": "pauc"})
            self.assertEqual(mixed.objective, {"merge": "auroc", "split": "pauc"})
            self.assertEqual(mixed.score_tolerance, {"merge": POOL_TOLERANCES["auroc"], "split": POOL_TOLERANCES["pauc"]})
            self.assertEqual(mixed.progress_tolerance, {"merge": .0005, "split": 0.})
            single = CandidatePool(Path(tmp) / "b.json", ["merge", "split"], objective="auroc")
            self.assertEqual((single.objective, single.score_tolerance, single.progress_tolerance), ("auroc", .005, .0005))
            fixed = CandidatePool(Path(tmp) / "c.json", ["merge", "split"], objective=MIXED, score_tolerance=.03)
            self.assertEqual(fixed.score_tolerance, .03)


class DriverTests(unittest.TestCase):
    def run_driver(self, objective):
        from proofreader_evolve.cli import run_precision_evolution as driver
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            args = driver.parse_args(['--min-positives', '0', '--selection-protocol', 'in_sample', '--promotion-gate', 'margin',
                                      '--train-brains', '1', '--validation-brains', '2', '--merge-k', '1',
                                      '--split-k', '2', '--generations', '2', '--runs-dir', tmp, '--objective', objective])

            async def revise(run_dir, policy, rules, report, model, *, experiments):
                if experiments.target_kind == 'split':
                    policy.write_text(IMPROVED)
                    experiments.proposal_path.write_text(json.dumps({'hypothesis': 'evidence', 'strategy': 'use it',
                                                                     'family': 'evidence', 'parameter_grid': {}}))
                    experiments.evaluate()
                else:
                    experiments.restore('parent')
                return {'summary': 'fixture', 'cost_usd': 0}

            with patch.object(driver.pc, 'resolve_detector_runs', return_value={}), \
                    patch.object(driver, 'ensure_native_tables', side_effect=lambda brain, *a, **k: two_kinds(brain)):
                path = asyncio.run(driver.run(args, revise_fn=revise))
            records = [json.loads(line) for line in (path / 'ledger.jsonl').read_text().splitlines()]
            manifest = json.loads((path / 'manifest.json').read_text())
            seed = json.loads((path / 'seed.json').read_text())
            feedback = {record['target_kind']: json.loads((path / f"gen{record['generation']:03d}/train_feedback.json").read_text())
                        for record in records}
        return records, manifest, seed, feedback

    def test_a_run_with_an_objective_per_kind(self):
        records, manifest, seed, feedback = self.run_driver('merge=average_precision,split=pauc')
        self.assertEqual(manifest['objective_statistic'], 'merge=average_precision,split=pauc')
        self.assertEqual(manifest['objective_statistics'], MIXED)
        self.assertEqual(manifest['objective_label'], 'merge: average precision; split: partial AUROC')
        self.assertEqual(manifest['budgets']['objective'], MIXED)
        self.assertEqual(seed['validation']['objective'], MIXED)
        by_kind = {record['target_kind']: record for record in records}
        self.assertEqual(set(by_kind), {'merge', 'split'})
        for kind, name in MIXED.items():
            record = by_kind[kind]
            self.assertNotIn('failure_stage', record, record.get('reason'))
            self.assertEqual(record['objective_statistic'], name)
            self.assertEqual(record['validation_gate']['objective'], name)
        split = by_kind['split']
        self.assertEqual(split['selection']['objective'], 'pauc')
        self.assertEqual(split['selection_gate']['objective'], 'pauc')
        self.assertEqual(split['validation']['per_kind_objective']['split'],
                         split['validation']['per_kind_statistics']['split']['pauc'])
        self.assertEqual(split['validation']['per_kind_objective']['merge'],
                         split['validation']['per_kind_statistics']['merge']['average_precision'])
        self.assertIn('partial AUROC', split['reason'])
        self.assertIn('pauc', json.dumps(feedback['split']))

    def test_one_name_records_what_it_always_did(self):
        records, manifest, seed, _ = self.run_driver('auroc')
        self.assertEqual((manifest['objective_statistic'], manifest['objective_label']), ('auroc', 'AUROC'))
        self.assertEqual(manifest['objective_statistics'], {'merge': 'auroc', 'split': 'auroc'})
        self.assertEqual(manifest['budgets']['objective'], 'auroc')
        self.assertEqual(seed['validation']['objective'], 'auroc')
        self.assertEqual({record['objective_statistic'] for record in records}, {'auroc'})


if __name__ == '__main__':
    unittest.main()
