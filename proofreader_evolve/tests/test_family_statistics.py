"""Per-family merge statistics (2026-11-11): reporting only. A pooled statistic can be high because
one candidate family ranks above the other; the within-family figures and the judge's second table
show whether errors are separated inside the family that holds them. Synthetic fixtures only."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from proofreader_evolve.harness import fixed_pool_scoring as scoring
from proofreader_evolve.harness import promotion_judge
from proofreader_evolve.harness.rank_statistics import (FAMILY_NOTE, aggregate_families, auroc, family_statistics,
                                                        ranking)
from proofreader_evolve.harness.train_feedback import metrics_only


def pool(brain, seed, rows=300, informative=0.):
    """Half junctions (20% positives), half tip_to_shaft rows (no positive, lower scores)."""
    rng = np.random.default_rng(seed)
    family = np.array(["junction"] * (rows // 2) + ["tip_to_shaft"] * (rows // 2))
    truth = np.where(family == "junction", (rng.random(rows) < .2).astype(float), 0.)
    truth[rng.random(rows) < .1] = np.nan
    scores = np.where(family == "junction", 5., 0.) + rng.normal(0, 1, rows) + informative * np.nan_to_num(truth)
    table = SimpleNamespace(truth=truth, keys=[f"{brain}-{i:04d}" for i in range(rows)],
                            candidates=[{"candidate_family": name} for name in family])
    return table, scores


def budgets():
    return scoring.Budgets({"merge": None}, positive_ratio=1., min_positives=0, aggregation="positive-weighted",
                           objective="average_precision")


class FamilyStatisticsTests(unittest.TestCase):
    def test_the_pooled_figure_hides_the_family(self):
        table, scores = pool("a", 0)
        k = budgets().k_for("merge", table.truth)
        metrics, chosen = scoring.rank_metrics(scores, table.truth, table.keys, k)
        by_family = family_statistics(scores, table.truth, table.keys, table.candidates, chosen)
        self.assertEqual(set(by_family), {"junction", "tip_to_shaft"})
        junction, tip = by_family["junction"], by_family["tip_to_shaft"]
        self.assertGreater(metrics["auroc"], .7)                       # junctions rank above tip_to_shaft rows
        self.assertLess(abs(junction["auroc"] - .5), .15)              # no signal inside the family
        _, labels, starts = ranking(scores, np.where(np.array([c["candidate_family"] for c in table.candidates])
                                                     == "junction", table.truth, np.nan), table.keys)
        self.assertAlmostEqual(junction["auroc"], auroc(labels.astype(float), np.ones(len(labels)), starts), places=12)
        self.assertEqual(junction["k"], junction["positives"])
        self.assertEqual(junction["positives"] + tip["positives"], metrics["positives"])
        self.assertEqual(junction["positives_in_pool_top_k"], metrics["tp"])
        self.assertEqual((tip["positives"], tip["auroc"], tip["k"]), (0, None, None))
        self.assertIsNone(family_statistics(scores, table.truth, table.keys, [{"node_id": 1}] * len(scores), chosen))
        folds = np.arange(len(scores)) % 2
        self.assertIsNone(family_statistics(scores, table.truth, table.keys, table.candidates, chosen, folds))
        self.assertIsNotNone(family_statistics(scores, table.truth, table.keys, table.candidates, chosen,
                                               np.zeros(len(scores), dtype=int)))
        self.assertIn("Reporting only", FAMILY_NOTE)

    def test_aggregation_and_feedback(self):
        cells = {}
        for index, brain in enumerate(("a", "b")):
            table, scores = pool(brain, index, informative=1.5)
            metrics, chosen = scoring.rank_metrics(scores, table.truth, table.keys, budgets().k_for("merge", table.truth))
            metrics["by_family"] = family_statistics(scores, table.truth, table.keys, table.candidates, chosen)
            cells[f"{brain}/merge"] = metrics
        total = aggregate_families(cells)
        parts = [cell["by_family"]["junction"] for cell in cells.values()]
        self.assertEqual(total["junction"]["positives"], sum(part["positives"] for part in parts))
        self.assertEqual(total["junction"]["top_k_hits"], sum(part["top_k_hits"] for part in parts))
        self.assertAlmostEqual(total["junction"]["auroc"], sum(part["auroc"] * part["positives"] for part in parts)
                               / sum(part["positives"] for part in parts), places=12)
        self.assertAlmostEqual(total["junction"]["precision_at_k"], total["junction"]["top_k_hits"] / total["junction"]["k"])
        self.assertIsNone(total["tip_to_shaft"]["auroc"])
        cells["a/merge"]["in_mean"] = False
        self.assertEqual(aggregate_families(cells)["junction"]["pools"], 1)
        self.assertIsNone(aggregate_families({"a/split": {"tp": 1}}))
        report = scoring.aggregate_objective({name: {**cell, "in_mean": True} for name, cell in cells.items()},
                                             "positive-weighted", "average_precision")
        self.assertEqual(set(report["per_kind_by_family"]), {"merge"})
        shown = metrics_only({**report, "cells": cells})
        self.assertEqual(shown["per_kind_by_family"], report["per_kind_by_family"])
        self.assertIn("by_family", shown["cells"]["a/merge"])

    def test_the_judge_gets_a_second_table_for_merge_junctions(self):
        tables, parent, candidate = {}, {}, {}
        for index, brain in enumerate(("a", "b", "c")):
            table, scores = pool(brain, index)
            tables[brain] = SimpleNamespace(tables={"merge": table})
            parent[f"{brain}/merge"] = {"scores": scores}
            candidate[f"{brain}/merge"] = {"scores": scores + 2. * np.nan_to_num(table.truth)}
        accept = [{"decision": "accept", "reason": "r"}] * 3
        with patch.object(promotion_judge, "cast_votes", return_value=accept) as cast:
            record = promotion_judge.judge(parent, candidate, tables, budgets(), target_kind="merge", draws=50, seed=1,
                                           votes=3)
        text = cast.call_args.args[0]
        self.assertIn("Second table: junction candidates only", text)
        self.assertEqual(record["family"], "junction")
        self.assertEqual(record["family_pools"]["positives"], record["pools"]["positives"])   # no tip_to_shaft positive
        self.assertLess(record["family_pools"]["labeled"], record["pools"]["labeled"])
        whole, inside = record["statistics"]["auroc"], record["family_statistics"]["auroc"]
        self.assertLess(inside["parent"], whole["parent"])              # the pooled figure flatters the parent
        self.assertGreater(inside["delta"], whole["delta"])
        self.assertTrue(record["passed"])
        self.assertEqual(text.count("| statistic |"), 2)
        for bank in tables.values():                                    # without families: one table, still a vote
            del bank.tables["merge"].candidates
        with patch.object(promotion_judge, "cast_votes", return_value=accept) as cast:
            record = promotion_judge.judge(parent, candidate, tables, budgets(), target_kind="merge", draws=50, seed=1,
                                           votes=3)
        self.assertNotIn("Second table", cast.call_args.args[0])
        self.assertIn("family_table_error", record)
        self.assertTrue(record["passed"])
        self.assertNotIn("split", promotion_judge.FAMILY_TABLES)


if __name__ == "__main__":
    unittest.main()
