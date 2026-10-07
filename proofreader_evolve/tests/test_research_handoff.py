"""Branch-aligned feedback and the research handoff (2026-10-06).

The first generation of the driver fixture promotes IMPROVED and leaves a handoff in
proposal.json; the second generation (same kind, strict rotation disabled) must see
that handoff in hypothesis_memory.json next to host facts, and its feedback must
describe the assigned branch rather than the accepted parent.
"""

import asyncio
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from proofreader_evolve.harness import hypothesis_memory as hm
from proofreader_evolve.harness.search_proposals import read_proposal
from proofreader_evolve.tests.test_fixed_pool_scoring import fixture, BASELINE, IMPROVED

HANDOFF = {'open_questions': ['Do missed positives share a low evidence value?'],
           'evidence': [{'claim': 'evidence separates the labels on TRAIN', 'verdict': 'supports',
                         'conditions': 'three-row fixture'}],
           'next_experiment': {'what': 'add a gap-normalised evidence term', 'discriminates': 'gap vs evidence',
                               'units': 2}}


class HandoffValidationTests(unittest.TestCase):
    def test_accepts_bounded_handoff_and_rejects_malformed_ones(self):
        clean = hm.validate_handoff(HANDOFF)
        self.assertEqual(clean['next_experiment']['units'], 2)
        self.assertEqual(clean['evidence'][0]['verdict'], 'supports')
        for bad in ({}, {'unknown': 1}, {'open_questions': ['x'] * 4}, {'evidence': [{'claim': 'c', 'verdict': 'maybe'}]},
                    {'next_experiment': {'what': 'w', 'units': 99}}, {'open_questions': ['x' * 400]}):
            with self.assertRaises(ValueError):
                hm.validate_handoff(bad)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'proposal.json'
            path.write_text(json.dumps({'hypothesis': 'h', 'strategy': 's', 'handoff': HANDOFF}))
            self.assertEqual(read_proposal(path)['handoff']['open_questions'], HANDOFF['open_questions'])
            path.write_text(json.dumps({'hypothesis': 'h', 'strategy': 's', 'handoff': {'evidence': 'not a list'}}))
            with self.assertRaises(ValueError):
                read_proposal(path)

    def test_memory_keeps_handoffs_next_to_host_facts_and_ranks_the_assigned_branch_first(self):
        with tempfile.TemporaryDirectory() as tmp:
            memory = hm.HypothesisMemory(Path(tmp) / 'memory.json')
            entry_a = {'target_kind': 'split', 'hypothesis': 'A', 'research': {'hypothesis_id': 'gap-a',
                       'information_source': 'table', 'failure_mode': 'missed', 'prediction': 'p'}}
            entry_b = {'target_kind': 'split', 'hypothesis': 'B', 'research': {'hypothesis_id': 'gap-b',
                       'information_source': 'table', 'failure_mode': 'missed', 'prediction': 'p'}}
            for generation, entry in ((1, entry_a), (2, entry_b), (3, entry_a), (4, entry_a)):
                memory.record_handoff(entry, generation, HANDOFF, {'experiment': f'gen{generation:03d}/attempt001',
                                                                    'selection_precision': .5, 'promoted': False})
            memory.mark_promoted('gen002/attempt001')
            record_a = memory.records['split/gap-a']
            self.assertEqual([h['generation'] for h in record_a['handoffs']], [1, 3, 4])  # capped at three
            items = memory.handoffs('split', parent={'experiment': 'gen002/attempt001', 'hypothesis': 'B',
                                                     'research': entry_b['research']})
            self.assertTrue(items[0]['related_to_assigned_branch'])
            self.assertEqual(items[0]['generation'], 2)
            self.assertTrue(items[0]['host']['promoted'])
            self.assertEqual(items[0]['host']['source'], 'host measurements')
            self.assertEqual(len(items), 3)
            snapshot = memory.snapshot('split', parent={'experiment': 'gen002/attempt001', 'research': entry_b['research']})
            self.assertEqual(snapshot['handoff'][0]['hypothesis_id'], 'gap-b')
            self.assertNotIn('validation', json.dumps(snapshot).lower())


class DriverHandoffTests(unittest.TestCase):
    def test_handoff_reaches_the_next_session_with_host_outcome(self):
        from proofreader_evolve.cli import run_precision_evolution as driver
        seen = {}
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            args = driver.parse_args(['--selection-protocol', 'in_sample', '--promotion-gate', 'margin',
                                      '--kind-schedule', 'alternate', '--train-brains', '1', '--validation-brains', '2',
                                      '--split-k', '1', '--generations', '2', '--runs-dir', tmp, '--target-kind', 'split'])

            async def revise(run_dir, policy, rules, report, model, *, experiments):
                memory_file = json.loads((policy.parent / 'hypothesis_memory.json').read_text())
                if experiments.generation == 1:
                    self.assertEqual(memory_file['handoff'], [])
                    policy.write_text(IMPROVED)
                    experiments.proposal_path.write_text(json.dumps({
                        'hypothesis': 'evidence ranks positives first', 'strategy': 'use evidence',
                        'family': 'evidence', 'parameter_grid': {},
                        'research': {'hypothesis_id': 'evidence-first', 'information_source': 'table',
                                     'failure_mode': 'missed positive', 'prediction': 'evidence high for positives'},
                        'handoff': HANDOFF}))
                    experiments.evaluate()
                else:
                    seen['handoff'] = memory_file['handoff']
                    seen['feedback'] = json.loads(report.read_text())
                    experiments.restore('parent')
                return {'summary': 'fixture', 'cost_usd': 0}

            with patch.object(driver.pc, 'resolve_detector_runs', return_value={}), \
                    patch.object(driver, 'ensure_native_tables', side_effect=lambda brain, *a, **k: fixture(brain)):
                path = asyncio.run(driver.run(args, revise_fn=revise))
            records = [json.loads(line) for line in (path / 'ledger.jsonl').read_text().splitlines()]
        self.assertTrue(records[0]['accepted'])
        self.assertEqual(records[0]['research_handoff']['host']['promoted'], True)
        self.assertEqual(records[0]['research_handoff']['experiment'], 'gen001/attempt001')
        self.assertIsNone(records[1]['research_handoff'])  # the second session wrote no handoff
        items = seen['handoff']
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]['hypothesis_id'], 'evidence-first')
        self.assertEqual(items[0]['agent']['next_experiment']['what'], HANDOFF['next_experiment']['what'])
        self.assertTrue(items[0]['host']['promoted'])
        self.assertTrue(items[0]['related_to_assigned_branch'])  # gen2 starts from the promoted reference
        self.assertTrue(seen['feedback']['search_branch']['same_as_accepted'])
        self.assertEqual(seen['feedback']['search_branch']['experiment'], 'gen001/attempt001')
        self.assertNotIn('validation', json.dumps(items).lower())


if __name__ == '__main__':
    unittest.main()
