"""Repeated submissions (2026-10-26).

A candidate the outer evaluator already measured, by bytes or by the exact scores of its kind
on every TRAIN and selection cell, ends its generation as `repeated` with no outer evaluation;
the agent is told beforehand through `outer_measurement` (generation and promotion status
only), the submission fallback prefers a not-yet-measured candidate, and earlier generations'
agent-authored files and TRAIN artefact trees are readable. Tiny fixtures, no API calls.
"""

import asyncio
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from proofreader_evolve.harness import reviser_access as access
from proofreader_evolve.harness.candidate_pool import CandidatePool, scores_digest
from proofreader_evolve.harness.evolution_report import generation_status
from proofreader_evolve.tests.test_fixed_pool_scoring import fixture, BASELINE, IMPROVED
from proofreader_evolve.tests.test_unchanged_submission import HANDOFF
from proofreader_evolve.tests.label_fixture import setUpModule, tearDownModule  # noqa: F401

# Same ranking as the seed with different scores: measured by the outer evaluator and rejected.
SCALED_SEED = BASELINE.replace('return ', 'return 2.0 * ')
# The accepted component with new code and the same scores everywhere (a comment-only edit would not
# count as a fresh measurement in a promotion follow-up round; multiplying by 1.0 keeps every score exact).
REWRITTEN = IMPROVED.replace('.to_numpy()', '.to_numpy() * 1.0')
# A fresh variant of the accepted component: same ranking, different scores.
HALVED = IMPROVED.replace('return ', 'return 0.5 * ')


def submit(policy, experiments, source, hypothesis='h', handoff=None):
    policy.write_text(source)
    proposal = {'hypothesis': hypothesis, 'strategy': 's', 'family': 'f', 'parameter_grid': {}}
    if handoff is not None:
        proposal['handoff'] = handoff
    experiments.proposal_path.write_text(json.dumps(proposal))
    return experiments.evaluate()


def run_driver(agents):
    """Run len(agents) split generations; agents[g](policy, experiments) acts as generation g+1's agent.

    Returns the ledger rows, the search plans and candidate pools the agent saw, each generation's
    trajectory, what the agents returned, and the outer evaluations (on brain 2) per generation.
    """
    from proofreader_evolve.cli import run_precision_evolution as driver
    with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
        args = driver.parse_args(['--min-positives', '0', '--selection-protocol', 'in_sample', '--promotion-gate', 'margin',
                                  '--kind-schedule', 'alternate', '--train-brains', '1', '--validation-brains', '2',
                                  '--split-k', '1', '--generations', str(len(agents)), '--runs-dir', tmp,
                                  '--target-kind', 'split'])
        state = {'phase': 0, 'outer': {}, 'returned': []}
        real_evaluate = driver.scoring.evaluate

        def counting_evaluate(source, tables, *a, **k):
            if sorted(tables) == ['2']:
                state['outer'][state['phase']] = state['outer'].get(state['phase'], 0) + 1
            return real_evaluate(source, tables, *a, **k)

        async def revise(run_dir, policy, rules, report, model, *, experiments):
            state['phase'] = experiments.generation
            state['returned'].append(agents[experiments.generation - 1](policy, experiments))
            return {'summary': 'fixture', 'cost_usd': 0}

        with patch.object(driver.pc, 'resolve_detector_runs', return_value={}), \
                patch.object(driver, 'ensure_native_tables', side_effect=lambda brain, *a, **k: fixture(brain)), \
                patch.object(driver.scoring, 'evaluate', side_effect=counting_evaluate):
            path = asyncio.run(driver.run(args, revise_fn=revise))
        records = [json.loads(line) for line in (path / 'ledger.jsonl').read_text().splitlines()]
        plans = [json.loads((path / f'gen{g:03d}' / 'search_plan.json').read_text()) for g in range(1, len(agents) + 1)]
        pools = [(path / f'gen{g:03d}' / 'candidate_pool.json').read_text() for g in range(1, len(agents) + 1)]
        trajectories = [(path / f'gen{g:03d}' / 'trajectory.txt').read_text() for g in range(1, len(agents) + 1)]
        memory = json.loads((path / 'hypothesis_memory.json').read_text())
    return records, plans, pools, trajectories, state['returned'], state['outer'], memory


class RepeatedSubmissionTests(unittest.TestCase):
    def test_a_cached_repeat_of_a_rejected_candidate_ends_as_repeated_without_an_outer_evaluation(self):
        records, plans, pools, trajectories, returned, outer, memory = run_driver([
            lambda policy, experiments: submit(policy, experiments, SCALED_SEED, 'scaled seed'),
            lambda policy, experiments: submit(policy, experiments, SCALED_SEED, 'scaled seed again', HANDOFF)])
        first, second = records
        self.assertIsNotNone(first['validation'])
        self.assertFalse(first['accepted'])
        self.assertEqual(first['submission_validated'],
                         {'generation': 1, 'experiment': 'gen001/attempt001', 'promoted': False,
                          'component_sha256': first['experiments'][0]['component_sha256'],
                          'scores_sha256': first['experiments'][0]['scores_sha256']})
        self.assertEqual(outer.get(1), 1)
        self.assertNotIn('outer_measurement', returned[0])
        # Generation 2: the training tool already says the outer evaluator saw this candidate.
        self.assertEqual(returned[1]['cached'], True)
        self.assertEqual({k: returned[1]['outer_measurement'][k] for k in ('generation', 'experiment', 'promoted', 'by')},
                         {'generation': 1, 'experiment': 'gen001/attempt001', 'promoted': False, 'by': 'component'})
        self.assertIn('ends the generation as repeated', returned[1]['outer_measurement']['note'])
        self.assertEqual(outer.get(2), None)  # no outer evaluation in generation 2
        self.assertIsNone(second['validation'])
        self.assertFalse(second['accepted'])
        self.assertEqual(second['submission_repeat'],
                         {'generation': 1, 'experiment': 'gen001/attempt001', 'promoted': False, 'by': 'component'})
        self.assertNotIn('submission_validated', second)
        self.assertTrue(second['reason'].startswith('VALIDATION: Repeated submission: gen002/attempt001 has the same component'))
        self.assertIn('validated in generation 1, not promoted', second['reason'])
        gate = second['validation_gate']
        self.assertEqual((gate['skipped'], gate['passed'], gate['candidate_precision']),
                         ('repeated_submission', False, None))
        self.assertEqual(second['descriptor_inference'], [])
        self.assertIsNone(second.get('failure_stage'))
        self.assertEqual(generation_status(second, set()), 'repeated')
        self.assertIn('repeated_submission', trajectories[1])
        self.assertIn('Generation 2: repeated;', trajectories[1])
        self.assertIn('outer_measurement_known', trajectories[1])
        self.assertEqual(second['research_handoff']['host']['status'], 'repeated_submission')
        host = [item['host']['status'] for record in memory['records'] for item in record.get('handoffs', [])]
        self.assertEqual(host, ['repeated_submission'])
        # The agent-visible plan and pool carry the status without any validation number or word.
        self.assertEqual(plans[0]['search_parent']['outer_measurement'],
                         {'generation': 0, 'experiment': 'seed/split', 'promoted': True, 'by': 'component'})
        for text in [json.dumps(plan) for plan in plans] + pools:
            self.assertNotIn('validation', text.lower())
            self.assertNotIn('macro_delta', text)
        pool = json.loads(pools[1])
        self.assertEqual([c['outer_measurement'] for c in pool['kinds']['split']['candidates']],
                         [{'generation': 0, 'experiment': 'seed/split', 'promoted': True, 'by': 'component'}])

    def test_a_rewritten_accepted_component_is_a_repeat_by_scores(self):
        records, plans, pools, trajectories, returned, outer, memory = run_driver([
            lambda policy, experiments: submit(policy, experiments, IMPROVED, 'evidence'),
            lambda policy, experiments: submit(policy, experiments, REWRITTEN, 'evidence rewritten')])
        first, second = records
        self.assertTrue(first['accepted'])
        self.assertEqual(first['submission_validated']['promoted'], True)
        self.assertEqual(outer.get(1), 1)
        self.assertNotEqual(second['experiments'][0]['component_sha256'], first['experiments'][0]['component_sha256'])
        self.assertEqual(second['experiments'][0]['scores_sha256'], first['experiments'][0]['scores_sha256'])
        self.assertFalse(returned[1].get('cached'))  # a fresh measurement, not a cache hit
        self.assertEqual(returned[1]['outer_measurement']['by'], 'scores')
        self.assertNotIn('submission_unchanged', second)  # bytes differ from the accepted component
        self.assertEqual(second['submission_repeat'],
                         {'generation': 1, 'experiment': 'gen001/attempt001', 'promoted': True, 'by': 'scores'})
        self.assertIn('has the same split scores as gen001/attempt001 (validated in generation 1, promoted)',
                      second['reason'])
        self.assertEqual(outer.get(2), None)
        self.assertEqual(generation_status(second, set()), 'repeated')
        self.assertEqual(plans[1]['search_parent']['outer_measurement'],
                         {'generation': 1, 'experiment': 'gen001/attempt001', 'promoted': True, 'by': 'component'})

    def test_a_fresh_variant_with_the_same_ranking_is_still_measured(self):
        records, *_ , outer, _ = run_driver([
            lambda policy, experiments: submit(policy, experiments, IMPROVED, 'evidence'),
            lambda policy, experiments: submit(policy, experiments, HALVED, 'halved evidence')])
        self.assertTrue(records[0]['accepted'])
        self.assertNotIn('submission_repeat', records[1])
        self.assertIsNotNone(records[1]['validation'])
        self.assertEqual(outer.get(2), 1)
        self.assertEqual(records[1]['submission_validated']['generation'], 2)

    def test_the_submission_fallback_prefers_a_candidate_the_outer_evaluator_has_not_seen(self):
        def second(policy, experiments):
            submit(policy, experiments, REWRITTEN, 'rewritten champion')  # known by scores, best selection score
            response = submit(policy, experiments, HALVED, 'halved')  # same selection score, not yet measured
            policy.write_text(HALVED.replace('return ', 'return 9.0 * '))  # untested edit left behind
            return response
        records, *_ , outer, _ = run_driver([
            lambda policy, experiments: submit(policy, experiments, IMPROVED, 'evidence'), second])
        row = records[1]
        self.assertEqual(row['submission_fallback']['restored'], 'gen002/attempt002')
        self.assertEqual(row['submitted_experiment'], 'gen002/attempt002')
        self.assertNotIn('submission_repeat', row)
        self.assertIsNotNone(row['validation'])
        self.assertEqual(outer.get(2), 1)


class PoolAndDigestTests(unittest.TestCase):
    def entry(self, number, scores='s', precision=.5):
        return {'experiment': f'gen001/attempt{number:03d}', 'sequence': number, 'target_kind': 'split',
                'component_sha256': f'c{number}', 'scores_sha256': scores, 'target_precision': precision,
                'top_k_sha256': f'r{number}', 'formula_sha256': 'f', 'family': 'f', 'parameters': {'w': 1},
                'status': 'evaluated', 'cached': False, 'train': {'per_kind_precision': {'split': precision}, 'cells': {}}}

    def test_scores_digest_is_exact_and_kind_scoped(self):
        import numpy as np
        train = {'1/split': {'scores': np.array([.1, .2, .3]), 'chosen': np.array([2])},
                 '1/merge': {'scores': np.array([.5]), 'chosen': np.array([0])}}
        selection = {'1/split': {'scores': np.array([.1, .2, .3]), 'chosen': np.array([2])}}
        same = scores_digest('split', ('train', train), ('selection', selection))
        self.assertEqual(same, scores_digest('split', ('train', {'1/split': train['1/split'], '1/merge': {'scores': np.array([.9]), 'chosen': np.array([0])}}),
                                             ('selection', selection)))
        scaled = {'1/split': {'scores': np.array([.2, .4, .6]), 'chosen': np.array([2])}}
        self.assertNotEqual(same, scores_digest('split', ('train', scaled), ('selection', selection)))
        self.assertNotEqual(same, scores_digest('split', ('train', train), ('selection', scaled)))
        self.assertNotEqual(same, scores_digest('split', ('selection', selection), ('train', train)))

    def test_pool_marks_and_annotates_measured_candidates_without_numbers(self):
        with tempfile.TemporaryDirectory() as tmp:
            pool = CandidatePool(Path(tmp) / 'pool.json', ['split'], size=3, score_tolerance=.02)
            champion, other = self.entry(1, scores='s1'), self.entry(2, scores='s2', precision=.49)
            pool.add(champion)
            pool.add(other)
            self.assertIsNone(pool.validation_of('split', champion))
            item = pool.mark_validated(champion, 3, False)
            self.assertEqual(item, {'generation': 3, 'experiment': 'gen001/attempt001', 'component_sha256': 'c1',
                                    'scores_sha256': 's1', 'promoted': False})
            self.assertEqual(pool.validation_of('split', champion),
                             {'generation': 3, 'experiment': 'gen001/attempt001', 'promoted': False, 'by': 'component'})
            rewritten = {**self.entry(9, scores='s1'), 'component_sha256': 'c9'}
            self.assertEqual(pool.validation_of('split', rewritten)['by'], 'scores')
            self.assertIsNone(pool.validation_of('split', other))
            self.assertIsNone(pool.validation_of('split', {'component_sha256': 'c9', 'scores_sha256': None}))
            summary = pool.summary()['kinds']['split']['candidates']
            self.assertEqual([c['outer_measurement'] for c in summary],
                             [{'generation': 3, 'experiment': 'gen001/attempt001', 'promoted': False, 'by': 'component'}, None])
            plan = pool.next_plan()
            self.assertEqual(plan['search_parent']['outer_measurement']['generation'], 3)
            self.assertNotIn('validation', json.dumps(plan).lower())
            self.assertNotIn('validation', json.dumps(pool.summary()).lower())
            with self.assertRaises(ValueError):
                pool.mark_validated({**champion, 'target_kind': 'merge'}, 4, True)

    def test_report_status_vocabulary(self):
        self.assertEqual(generation_status({'accepted': False, 'submission_repeat': {'generation': 1}, 'validation': None},
                                           set()), 'repeated')
        self.assertEqual(generation_status({'accepted': False, 'submission_unchanged': True, 'validation': None}, set()),
                         'unchanged')


class EarlierGenerationAccessTests(unittest.TestCase):
    def setUp(self):
        from proofreader_evolve.cli import run_precision_evolution as driver
        self.driver = driver
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.run_dir = Path(self.temp.name) / 'runs' / 'precision_x'
        for gen in ('gen001', 'gen002', 'gen003'):
            (self.run_dir / gen / 'experiments' / 'attempt001').mkdir(parents=True)
            (self.run_dir / gen / 'experiments' / 'attempt001' / 'training.py').write_text('# program\n')
            for name in ('training.py', 'descriptor.py', 'proposal.json', 'evaluation.json', 'reviser_result.json'):
                (self.run_dir / gen / name).write_text('{}\n')
        (self.run_dir / 'train_seed').mkdir()

    def decision(self, guard, tool, path):
        return asyncio.run(guard(tool, {'file_path': str(path)}, None)).behavior

    def test_lists_only_earlier_generations_and_only_train_material(self):
        paths, dirs = self.driver.earlier_generation_readables(self.run_dir, self.run_dir / 'gen001')
        self.assertEqual((paths, dirs), ([], []))
        paths, dirs = self.driver.earlier_generation_readables(self.run_dir, self.run_dir / 'gen003')
        self.assertEqual([p.parent.name for p in dirs], ['gen001'] * 4 + ['gen002'] * 4)
        self.assertEqual([d.name for d in dirs[:3]], ['experiments', 'classifier_fits', 'descriptor_runs'])
        names = {p.name for p in paths}
        self.assertTrue({'scorer.py', 'training.py', 'descriptor.py', 'proposal.json', 'search_plan.json',
                         'train_feedback.json', 'hypothesis_memory.json'} <= names)
        self.assertFalse({'evaluation.json', 'reviser_result.json', 'trajectory.txt', 'image_evidence.json'} & names)
        self.assertFalse({p.parent.name for p in paths} & {'gen003', 'train_seed'})

    def test_guard_allows_earlier_train_files_and_denies_validation_files(self):
        paths, dirs = self.driver.earlier_generation_readables(self.run_dir, self.run_dir / 'gen003')
        guard, state = access.make_guard(self.run_dir, self.run_dir / 'gen003', paths,
                                         self.run_dir / 'audit.jsonl', sdk_tool_results=False, readable_dirs=dirs)
        self.assertEqual(self.decision(guard, 'Read', self.run_dir / 'gen001' / 'experiments' / 'attempt001' / 'training.py'), 'allow')
        self.assertEqual(self.decision(guard, 'Read', self.run_dir / 'gen002' / 'training.py'), 'allow')
        self.assertEqual(self.decision(guard, 'Read', self.run_dir / 'gen002' / 'descriptor.py'), 'allow')
        self.assertEqual(self.decision(guard, 'Write', self.run_dir / 'gen002' / 'training.py'), 'deny')
        self.assertEqual(self.decision(guard, 'Edit', self.run_dir / 'gen001' / 'experiments' / 'attempt001' / 'training.py'), 'deny')
        self.assertEqual(self.decision(guard, 'Read', self.run_dir / 'gen001' / 'evaluation.json'), 'deny')
        self.assertEqual(self.decision(guard, 'Read', self.run_dir / 'gen002' / 'reviser_result.json'), 'deny')
        self.assertEqual(len(state['violations']), 4)

    def test_bound_session_and_hook_deny_validation_descriptor_logs_in_every_generation(self):
        from claude_agent_sdk import ClaudeAgentOptions
        from proofreader_evolve.harness.reviser_session import bind_session_options

        current = self.run_dir / 'gen003'
        paths, dirs = self.driver.earlier_generation_readables(self.run_dir, current)
        dirs.extend(current / name for name in self.driver.TRAIN_ARTEFACT_DIRS)
        bound = bind_session_options(ClaudeAgentOptions(cwd=str(self.run_dir)),
                                     current / 'scorer.py', current / 'rules.md', None,
                                     readable_paths=paths, readable_dirs=dirs)
        hook = bound.hooks['PreToolUse'][0].hooks[0]
        # Create logs after binding, as workers do during and after a live session.
        for generation in ('gen001', 'gen003'):
            for tree, expected in (('descriptor_runs/run001', 'allow'),
                                   ('descriptor_probes/probe001', 'allow'),
                                   ('descriptor_runs/inference', 'deny'),
                                   ('validation_descriptor_runs', 'deny')):
                log = self.run_dir / generation / tree / 'brain' / 'program' / 'batch0000.log'
                log.parent.mkdir(parents=True)
                log.write_text('SYNTHETIC WORKER OUTPUT\n')
                for tool in ('Read', 'Write', 'Edit'):
                    decision = expected if tool == 'Read' else 'deny'
                    with self.subTest(generation=generation, tree=tree, tool=tool):
                        self.assertEqual(self.decision(bound.can_use_tool, tool, log), decision)
                        result = asyncio.run(hook({'tool_name': tool, 'tool_input':
                                                   {'file_path': str(log)}}, None, None))
                        self.assertEqual(result['hookSpecificOutput']['permissionDecision'], decision)

    def test_explicit_allowlists_cannot_reopen_private_descriptor_logs(self):
        for tree in ('descriptor_runs/inference', 'validation_descriptor_runs'):
            log = self.run_dir / 'gen001' / tree / 'batch0000.log'
            log.parent.mkdir(parents=True)
            log.write_text('SYNTHETIC PRIVATE OUTPUT\n')
            for files, dirs in (([log], []), ([], [log.parent])):
                with self.subTest(tree=tree, explicit_file=bool(files)):
                    guard, _ = access.make_guard(self.run_dir, self.run_dir / 'gen003', files,
                                                 self.run_dir / 'audit.jsonl', readable_dirs=dirs)
                    self.assertEqual(self.decision(guard, 'Read', log), 'deny')

    def test_descriptor_log_aliases_do_not_bypass_private_paths(self):
        paths, dirs = self.driver.earlier_generation_readables(self.run_dir, self.run_dir / 'gen003')
        guard, _ = access.make_guard(self.run_dir, self.run_dir / 'gen003', paths,
                                     self.run_dir / 'audit.jsonl', readable_dirs=dirs)
        mount = self.run_dir.parent / 'mount_alias'
        mount.symlink_to(self.run_dir, target_is_directory=True)
        for index, tree in enumerate(('descriptor_runs/inference', 'validation_descriptor_runs')):
            private = self.run_dir / 'gen001' / tree
            private.mkdir(parents=True)
            log = private / 'batch0000.log'
            log.write_text('SYNTHETIC PRIVATE OUTPUT\n')
            alias = self.run_dir / 'gen001' / 'descriptor_runs' / f'run{index:03d}'
            alias.symlink_to(private, target_is_directory=True)
            for path in (alias / log.name, mount / log.relative_to(self.run_dir),
                         Path('gen001') / tree / log.name):
                with self.subTest(path=path):
                    self.assertEqual(self.decision(guard, 'Read', path), 'deny')


if __name__ == '__main__':
    unittest.main()
