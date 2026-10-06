"""Adaptive allocation of generations to kinds (v16): headroom, momentum, floor,
pending follow-up, saturation, and the driver wiring. Synthetic fixtures only.

The driver fixture makes merge saturated at the seed (its only positive is
already ranked first at K=1) while split has headroom, so adaptive scheduling
spends the first three generations on split and the floor brings merge in at
generation four; strict rotation on the same fixture alternates.
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

from proofreader_evolve.harness import kind_schedule as ks
from proofreader_evolve.harness.native_pool import NativeTable, NativeTables, pool_digest

BASELINE = "def score_candidates(features, ctx):\n    return features['detector_score'].to_numpy()\n"
IMPROVED = "def score_candidates(features, ctx):\n    return features['evidence'].fillna(0).to_numpy()\n"


def cell(positives, k, precision):
    return {"positives": positives, "requested_k": k, "precision": precision}


def row(kind, accepted, parent=None, candidate=None):
    gate = None if parent is None else {"parent_mean": parent, "candidate_mean": candidate}
    return {"target_kind": kind, "accepted": accepted, "validation_gate": gate}


def two_kinds(brain):
    """Split: seed precision 0 at K=2, IMPROVED reaches 0.5 (cap 1.0). Merge: seed already perfect at K=1."""
    frame = pd.DataFrame({"evidence": [.1, .9, .3, .2], "detector_score": [.9, .1, .2, .05]})
    split_rows = [{"segment_id_a": 1, "segment_id_b": b} for b in (2, 3, 4, 5)]
    split = NativeTable("split", frame.copy(), split_rows, np.array([0, 1, 0, 1]),
                        {"pool_sha256": pool_digest("split", split_rows), "training_brain": "3"})
    merge_rows = [{"segment_id": 1, "node_id": i} for i in range(4)]
    merge = NativeTable("merge", frame.copy(), merge_rows, np.array([1, 0, 0, 0]),
                        {"pool_sha256": pool_digest("merge", merge_rows), "training_brain": "3"})
    return NativeTables(brain, {"merge": merge, "split": split}, {})


class RuleTests(unittest.TestCase):
    KINDS = ["merge", "split"]
    BUDGETS = {"merge": 2000, "split": 2000}

    def test_headroom_is_the_equal_brain_mean_of_capped_remaining_precision(self):
        cells = {"a/merge": cell(62, 2000, .019), "b/merge": cell(88, 2000, .0115),
                 "a/split": cell(2720, 2000, .433), "b/split": cell(1813, 2000, .0885)}
        room = ks.headroom(cells, self.BUDGETS, self.KINDS)
        self.assertAlmostEqual(room["merge"], ((62 / 2000 - .019) + (88 / 2000 - .0115)) / 2)
        self.assertAlmostEqual(room["split"], ((1. - .433) + (1813 / 2000 - .0885)) / 2)
        self.assertEqual(ks.headroom({}, self.BUDGETS, self.KINDS), {"merge": 0., "split": 0.})

    def test_momentum_uses_gate_passing_gains_and_counts_rejections_as_zero(self):
        records = [row("split", True, .1, .13), row("merge", True, .13, .131), row("split", False, .131, .131),
                   row("merge", False), row("split", True, .131, .14)]
        gain = ks.momentum(records, self.KINDS)
        self.assertAlmostEqual(gain["split"], (0. + (.14 - .131)) / 2)  # last two split rows
        self.assertAlmostEqual(gain["merge"], ((.131 - .13) + 0.) / 2)
        self.assertEqual(ks.momentum([], self.KINDS), {"merge": 0., "split": 0.})
        self.assertEqual(ks.generations_since(records, self.KINDS), {"merge": 1, "split": 0})
        self.assertEqual(ks.generations_since([], self.KINDS), {"merge": 0, "split": 0})

    def allocate(self, cells, records, **overrides):
        options = {"paused": {}, "pending": {}, "floor_every": 4}
        options.update(overrides)
        return ks.allocate(self.KINDS, cells=cells, budgets=self.BUDGETS, records=records, **options)

    def test_headroom_decides_without_history_and_saturated_kinds_wait_for_the_floor(self):
        cells = {"a/merge": cell(62, 2000, .031), "a/split": cell(2720, 2000, .3)}  # merge at its cap
        result = self.allocate(cells, [])
        self.assertEqual((result["chosen"], result["rule"]), ("split", "headroom"))
        self.assertTrue(result["kinds"]["merge"]["saturated"])
        self.assertEqual(result["order"], ["split", "merge"])
        history = [row("split", True, .1, .2), row("split", False), row("split", False)]
        result = self.allocate(cells, history)
        self.assertEqual((result["chosen"], result["rule"]), ("merge", "floor"))
        self.assertEqual(result["kinds"]["merge"]["generations_since"], 3)

    def test_momentum_beats_headroom_until_it_fades(self):
        cells = {"a/merge": cell(2000, 2000, .1), "a/split": cell(2720, 2000, .3)}  # merge has more room (0.9 vs 0.7)
        history = [row("split", True, .1, .15), row("merge", False)]
        result = self.allocate(cells, history)
        self.assertEqual((result["chosen"], result["rule"]), ("split", "momentum"))
        faded = history + [row("split", False), row("split", False)]
        result = self.allocate(cells, faded)  # momentum gone, merge unscheduled for 2: headroom decides
        self.assertEqual((result["chosen"], result["rule"]), ("merge", "headroom"))
        result = self.allocate(cells, faded + [row("split", False)])  # merge unscheduled for 3: floor
        self.assertEqual((result["chosen"], result["rule"]), ("merge", "floor"))
        result = self.allocate(cells, [row("split", False), row("merge", False)])
        self.assertEqual((result["chosen"], result["rule"]), ("merge", "headroom"))

    def test_pending_followup_paused_and_validation(self):
        cells = {"a/merge": cell(1000, 2000, .1), "a/split": cell(2720, 2000, .3)}
        result = self.allocate(cells, [], pending={"split": True})
        self.assertEqual((result["chosen"], result["rule"]), ("split", "pending_followup"))
        result = self.allocate(cells, [], paused={"merge": True})
        self.assertEqual(result["order"], ["split"])
        result = self.allocate(cells, [], paused={"merge": True, "split": True})
        self.assertEqual((result["chosen"], result["rule"], result["order"]), (None, "all_paused", []))
        with self.assertRaises(ValueError):
            self.allocate(cells, [], floor_every=0)
        self.assertNotIn("validation", json.dumps(result).lower().replace("development-validation", ""))


class DriverTests(unittest.TestCase):
    def run_driver(self, extra, generations=4):
        from proofreader_evolve.cli import run_precision_evolution as driver
        targets = []
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            args = driver.parse_args(['--selection-protocol', 'in_sample', '--promotion-gate', 'margin',
                                      '--train-brains', '1', '--validation-brains', '2', '--merge-k', '1',
                                      '--split-k', '2', '--generations', str(generations), '--runs-dir', tmp, *extra])

            async def revise(run_dir, policy, rules, report, model, *, experiments):
                targets.append(experiments.target_kind)
                if experiments.target_kind == 'split':
                    # First split generation promotes IMPROVED; later ones measure a fresh variant (a real
                    # code change, since comment-only edits are not fresh measurements) with the same
                    # ranking, which completes the follow-up without a validation gain.
                    variants = targets.count('split') - 1
                    policy.write_text(IMPROVED.replace('.to_numpy()', '.to_numpy() * %d.0' % (variants + 1))
                                      if variants else IMPROVED)
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
            plans = [json.loads((path / f'gen{g:03d}/search_plan.json').read_text()) for g in range(1, len(records) + 1)]
            pools = [(path / f'gen{g:03d}/candidate_pool.json').read_text() for g in range(1, len(records) + 1)]
        return targets, records, manifest, plans, pools

    def test_adaptive_schedule_favours_headroom_and_momentum_with_a_floor(self):
        targets, records, manifest, plans, pools = self.run_driver(['--kind-floor-every', '4'])
        self.assertEqual(targets, ['split', 'split', 'split', 'merge'])
        self.assertEqual([r['kind_allocation']['rule'] for r in records],
                         ['headroom', 'pending_followup', 'momentum', 'floor'])
        first = records[0]['kind_allocation']['kinds']
        self.assertTrue(first['merge']['saturated'])
        self.assertAlmostEqual(first['split']['headroom'], 1.)
        self.assertTrue(records[0]['accepted'])
        self.assertGreater(records[2]['kind_allocation']['kinds']['split']['momentum'], 0.)
        self.assertEqual(records[3]['kind_allocation']['kinds']['merge']['generations_since'], 3)
        self.assertEqual(manifest['kind_schedule']['mode'], 'adaptive')
        self.assertEqual(manifest['search_version'], 'adaptive-kind-allocation-v16')
        self.assertEqual([plan['allocation_rule'] for plan in plans],
                         ['headroom', 'pending_followup', 'momentum', 'floor'])
        for text in [json.dumps(plan) for plan in plans] + pools:
            lowered = text.lower()
            self.assertNotIn('validation', lowered)
            # The rule name may appear; the numeric fields never reach agent-visible files.
            self.assertNotIn('"headroom":', lowered)
            self.assertNotIn('"momentum":', lowered)
            self.assertNotIn('generations_since', lowered)

    def test_alternate_schedule_rotates_and_records_no_allocation(self):
        targets, records, manifest, plans, _ = self.run_driver(['--kind-schedule', 'alternate'])
        self.assertEqual(targets, ['merge', 'split', 'merge', 'split'])
        self.assertTrue(all(r['kind_allocation'] is None for r in records))
        self.assertEqual(manifest['kind_schedule']['mode'], 'alternate')
        self.assertTrue(all('allocation_rule' not in plan for plan in plans))

    def test_defaults_and_validation(self):
        from proofreader_evolve.cli import run_precision_evolution as driver
        args = driver.parse_args([])
        self.assertEqual((args.kind_schedule, args.kind_floor_every, args.generations), ('adaptive', 4, 15))
        with redirect_stdout(io.StringIO()), self.assertRaises(SystemExit):
            driver.parse_args(['--kind-floor-every', '0'])


if __name__ == '__main__':
    unittest.main()
