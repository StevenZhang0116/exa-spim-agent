"""Branch-aligned feedback and the research handoff (2026-10-06; lenient handoff bounds 2026-10-07).

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
from proofreader_evolve.tests.label_fixture import setUpModule, tearDownModule  # noqa: F401

HANDOFF = {'open_questions': ['Do missed positives share a low evidence value?'],
           'evidence': [{'claim': 'evidence separates the labels on TRAIN', 'verdict': 'supports',
                         'conditions': 'three-row fixture'}],
           'next_experiment': {'what': 'add a gap-normalised evidence term', 'discriminates': 'gap vs evidence',
                               'units': 2}}


MALFORMED = {'open_questions': HANDOFF['open_questions'] + ['second', 'third', 'fourth', 'fifth', 'sixth'],
             'do_not_repeat': ['the retired formula branch'],
             'evidence': [{'claim': 'band GBM beats the formula', 'verdict': 'Supports', 'measurement': 'OOF 0.55 vs 0.52'}],
             'next_experiment': {'what': 'tune nonband_weight', 'units': 99, 'why': 'extra'}}


class HandoffValidationTests(unittest.TestCase):
    def test_bounds_handoffs_without_rejecting_them(self):
        clean, notes = hm.sanitize_handoff(HANDOFF)
        self.assertEqual(notes, [])
        self.assertEqual(clean['next_experiment']['units'], 2)
        self.assertEqual(clean['evidence'][0]['verdict'], 'supports')
        # The shapes that failed run precision_20261007_001936 at the submission check are kept, bounded.
        clean, notes = hm.sanitize_handoff(MALFORMED)
        self.assertEqual(len(clean['open_questions']), hm.HANDOFF_LIMIT)
        self.assertEqual(clean['evidence'], [{'claim': 'band GBM beats the formula', 'verdict': 'supports',
                                              'conditions': 'measurement: OOF 0.55 vs 0.52'}])
        self.assertEqual(clean['next_experiment'], {'what': 'tune nonband_weight'})
        self.assertTrue(any('do_not_repeat' in n for n in notes) and any('kept the first 5 of 6' in n for n in notes))
        self.assertEqual(hm.sanitize_handoff({'evidence': [{'claim': 'c', 'verdict': 'maybe'}]})[0]['evidence'][0]['verdict'], 'mixed')
        self.assertEqual([e['verdict'] for e in hm.sanitize_handoff({'evidence': [{'claim': 'a', 'verdict': 'Supported'}, {'claim': 'b', 'verdict': 'refuted'}]})[0]['evidence']],
                         ['supports', 'contradicts'])
        self.assertEqual(len(hm.sanitize_handoff({'open_questions': ['x' * 900]})[0]['open_questions'][0]), hm.HANDOFF_TEXT)
        self.assertEqual(hm.sanitize_handoff('one bare question')[0]['open_questions'], ['one bare question'])
        self.assertEqual(hm.sanitize_handoff({'evidence': 'bare claim'})[0]['evidence'], [{'claim': 'bare claim', 'verdict': 'mixed'}])
        for unusable in ({}, {'unknown': 1}, 7, {'evidence': 42}, {'open_questions': [None, '']}):
            self.assertIsNone(hm.sanitize_handoff(unusable)[0])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'proposal.json'
            # The handoff is not part of the experiment: read_proposal accepts the key, never returns it
            # and never fails on it; the driver reads it from the file once.
            for handoff in (HANDOFF, MALFORMED, {'evidence': 42}, 'a bare string'):
                path.write_text(json.dumps({'hypothesis': 'h', 'strategy': 's', 'handoff': handoff}))
                self.assertNotIn('handoff', read_proposal(path))

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


class ResearchSanitizerTests(unittest.TestCase):
    """The research declaration is bounded like the handoff, never a proposal error (2026-10-08)."""

    GOOD = {'hypothesis_id': 'gap-a', 'information_source': 'table', 'failure_mode': 'missed',
            'prediction': 'p', 'feature_columns': ['x']}

    def test_research_is_bounded_not_rejected(self):
        self.assertEqual(hm.sanitize_research(self.GOOD), (self.GOOD, []))
        clean, notes = hm.sanitize_research({**self.GOOD, 'outcome': 'refuted', 'hypothesis_id': 'Gap A!',
                                             'feature_columns': ['x', 'x', 7, 'y' * 200], 'prediction': 'p' * 2000})
        self.assertEqual(clean['hypothesis_id'], 'Gap-A')
        self.assertNotIn('outcome', clean)
        self.assertEqual(clean['feature_columns'], ['x', 'y' * 128])
        self.assertEqual(len(clean['prediction']), hm.RESEARCH_TEXT)
        self.assertTrue(any('outcome' in n for n in notes))
        self.assertEqual(hm.sanitize_research({**self.GOOD, 'failure_mode': ''})[0]['failure_mode'], 'unspecified')
        for unusable in (7, {'prediction': 'p'}, {'hypothesis_id': '!!!'}):
            self.assertIsNone(hm.sanitize_research(unusable)[0])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'proposal.json'
            path.write_text(json.dumps({'hypothesis': 'h', 'strategy': 's', 'research': {**self.GOOD, 'outcome': 'refuted'}}))
            self.assertEqual(read_proposal(path)['research'], self.GOOD)  # the generation-16 shape now parses
            path.write_text(json.dumps({'hypothesis': 'h', 'strategy': 's', 'research': {'prediction': 'no id'}}))
            self.assertNotIn('research', read_proposal(path))


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
                        'handoff': MALFORMED}))  # the real-run shape; must not fail the generation
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
        self.assertTrue(any('do_not_repeat' in n for n in records[0]['research_handoff']['adjustments']))
        self.assertTrue(all('handoff' not in entry for entry in records[0]['experiments']))  # entries stay experiment-only
        self.assertIsNone(records[1]['research_handoff'])  # the second session wrote no handoff
        items = seen['handoff']
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]['hypothesis_id'], 'evidence-first')
        self.assertEqual(items[0]['agent']['next_experiment'], {'what': 'tune nonband_weight'})
        self.assertEqual(len(items[0]['agent']['open_questions']), hm.HANDOFF_LIMIT)
        self.assertEqual(items[0]['agent']['evidence'][0]['conditions'], 'measurement: OOF 0.55 vs 0.52')
        self.assertTrue(items[0]['host']['promoted'])
        self.assertTrue(items[0]['related_to_assigned_branch'])  # gen2 starts from the promoted reference
        self.assertTrue(seen['feedback']['search_branch']['same_as_accepted'])
        self.assertEqual(seen['feedback']['search_branch']['experiment'], 'gen001/attempt001')
        self.assertNotIn('validation', json.dumps(items).lower())


if __name__ == '__main__':
    unittest.main()
