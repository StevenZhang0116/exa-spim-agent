"""Adaptive allocation of generations to kinds: initialization, headroom, momentum, floor,
pending follow-up, saturation, and the driver wiring. Synthetic fixtures only.

The driver fixture makes merge saturated at the seed (its only positive is
already ranked first at K=1) while split has headroom, so adaptive scheduling
starts with split, gives merge its initial opportunity, then returns to split's
pending follow-up and momentum; strict rotation on the same fixture alternates.
"""

import asyncio
from contextlib import redirect_stdout
import io
import json
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from proofreader_evolve.harness import kind_schedule as ks
from proofreader_evolve.harness.native_pool import NativeTable, NativeTables, pool_digest
from proofreader_evolve.tests import label_fixture

setUpModule, tearDownModule = label_fixture.setUpModule, label_fixture.tearDownModule

BASELINE = "def score_candidates(features, ctx):\n    return features['detector_score'].to_numpy()\n"
IMPROVED = "def score_candidates(features, ctx):\n    return features['evidence'].fillna(0).to_numpy()\n"


def cell(positives, k, precision):
    return {"positives": positives, "requested_k": k, "precision": precision}


def row(kind, accepted, parent=None, candidate=None):
    gate = None if parent is None else {"target_kind": kind, "parent_precision": parent, "candidate_precision": candidate}
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

    def test_momentum_uses_point_gains_whether_or_not_the_gate_accepted(self):
        records = [row("split", True, .1, .13), row("merge", True, .13, .131), row("split", False, .131, .131),
                   row("merge", False, .131, .156), row("split", True, .131, .14)]
        gain = ks.momentum(records, self.KINDS)
        self.assertAlmostEqual(gain["split"], (0. + (.14 - .131)) / 2)  # last two split rows
        self.assertAlmostEqual(gain["merge"], ((.131 - .13) + (.156 - .131)) / 2)  # the rejected gain counts
        self.assertEqual(ks.momentum([], self.KINDS), {"merge": 0., "split": 0.})
        self.assertEqual(ks.generations_since(records, self.KINDS), {"merge": 1, "split": 0})
        self.assertEqual(ks.generations_since([], self.KINDS), {"merge": 0, "split": 0})
        # Failed generations carry no measurement; losses are clipped at zero.
        faded = records + [row("merge", False), row("split", False, .14, .139)]
        gain = ks.momentum(faded, self.KINDS)
        self.assertAlmostEqual(gain["merge"], ((.156 - .131) + 0.) / 2)
        self.assertAlmostEqual(gain["split"], ((.14 - .131) + 0.) / 2)

    def test_rejected_point_gain_outranks_a_smaller_accepted_gain(self):
        # The 2026-10-31 raw run: split's accepted +0.006 kept it scheduled while merge's
        # rejected +0.025 counted as nothing; point gains put merge first.
        cells = {"a/merge": cell(2000, 2000, .2), "a/split": cell(2720, 2000, .85)}
        history = [row("split", True, .843, .849), row("merge", False, .181, .206)]
        result = self.allocate(cells, history)
        self.assertEqual((result["chosen"], result["rule"]), ("merge", "momentum"))
        self.assertAlmostEqual(result["kinds"]["merge"]["momentum"], .206 - .181)
        self.assertEqual(result["version"], "adaptive-kind-allocation-v7-initial-exploration")

    def allocate(self, cells, records, **overrides):
        options = {"paused": {}, "pending": {}, "floor_every": 4}
        options.update(overrides)
        return ks.allocate(self.KINDS, cells=cells, budgets=self.BUDGETS, records=records, **options)

    def test_initial_exploration_includes_saturated_kinds_then_floor_applies(self):
        cells = {"a/merge": cell(62, 2000, .031), "a/split": cell(2720, 2000, .3)}  # merge at its cap
        result = self.allocate(cells, [])
        self.assertEqual((result["chosen"], result["rule"]), ("split", "initial_exploration"))
        self.assertTrue(result["kinds"]["merge"]["saturated"])
        self.assertEqual(result["order"], ["split", "merge"])
        result = self.allocate(cells, [row("split", True, .1, .2)], pending={"split": True})
        self.assertEqual((result["chosen"], result["rule"]), ("merge", "initial_exploration"))
        history = [row("merge", False), row("split", True, .1, .2), row("split", False), row("split", False)]
        result = self.allocate(cells, history)
        self.assertEqual((result["chosen"], result["rule"]), ("merge", "floor"))
        self.assertEqual(result["kinds"]["merge"]["generations_since"], 3)

    def test_initial_opportunity_precedes_followup_and_a_failure_still_counts(self):
        cells = {"a/merge": cell(2000, 2000, .1), "a/split": cell(2000, 2000, .5)}
        pending = {"merge": True}
        history = [row("merge", True, 0., .1)]
        result = self.allocate(cells, history, pending=pending)
        self.assertEqual((result["chosen"], result["rule"]), ("split", "initial_exploration"))
        self.assertEqual(pending, {"merge": True})
        result = self.allocate(cells, history + [row("split", False)], pending=pending)
        self.assertEqual((result["chosen"], result["rule"]), ("merge", "pending_followup"))

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
        result = self.allocate(cells, [row("merge", False), row("split", False)], pending={"split": True})
        self.assertEqual((result["chosen"], result["rule"]), ("split", "pending_followup"))
        result = self.allocate(cells, [], paused={"merge": True})
        self.assertEqual(result["order"], ["split"])
        result = self.allocate(cells, [], paused={"merge": True, "split": True})
        self.assertEqual((result["chosen"], result["rule"], result["order"]), (None, "all_paused", []))
        with self.assertRaises(ValueError):
            self.allocate(cells, [], floor_every=0)
        self.assertNotIn("validation", json.dumps(result).lower().replace("development-validation", ""))

    def test_repeated_promotions_cannot_starve_the_other_kind(self):
        cells = {"a/merge": cell(1000, 2000, .1), "a/split": cell(2720, 2000, .3)}
        history, pending = [], {"split": True}
        for _ in range(12):
            result = self.allocate(cells, history, pending=pending)
            history.append(row(result['chosen'], True, .1, .11))
        kinds = [r['target_kind'] for r in history]
        for start in range(len(kinds) - 3):
            self.assertEqual(set(kinds[start:start + 4]), {'merge', 'split'})
        self.assertEqual(pending, {'split': True})  # deferral never consumes the follow-up
        result = self.allocate(cells, [row('merge', False)] + [row('split', True)] * 3,
                               pending={'merge': True, 'split': True})
        self.assertEqual((result['chosen'], result['rule']), ('merge', 'floor'))


class DriverTests(unittest.TestCase):
    def run_driver(self, extra, generations=4):
        from proofreader_evolve.cli import run_precision_evolution as driver
        targets = []
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            args = driver.parse_args(['--min-positives', '0', '--selection-protocol', 'in_sample', '--promotion-gate', 'margin',
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

    def test_adaptive_schedule_initializes_both_kinds_and_preserves_followup(self):
        targets, records, manifest, plans, pools = self.run_driver(['--kind-floor-every', '4'])
        self.assertEqual(targets, ['split', 'merge', 'split', 'split'])
        self.assertEqual([r['kind_allocation']['rule'] for r in records],
                         ['initial_exploration', 'initial_exploration', 'pending_followup', 'momentum'])
        first = records[0]['kind_allocation']['kinds']
        self.assertTrue(first['merge']['saturated'])
        self.assertAlmostEqual(first['split']['headroom'], 1.)
        self.assertTrue(records[0]['accepted'])
        self.assertGreater(records[2]['kind_allocation']['kinds']['split']['momentum'], 0.)
        self.assertEqual(records[3]['kind_allocation']['kinds']['merge']['generations_since'], 1)
        self.assertEqual(plans[2]['reason'], 'promotion_followup')
        self.assertEqual(manifest['kind_schedule']['mode'], 'adaptive')
        self.assertEqual(manifest['search_version'], 'per-kind-selection-objective-v18')
        self.assertEqual([plan['allocation_rule'] for plan in plans],
                         ['initial_exploration', 'initial_exploration', 'pending_followup', 'momentum'])
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
            driver.parse_args(['--min-positives', '0', '--kind-floor-every', '0'])


if __name__ == '__main__':
    unittest.main()
