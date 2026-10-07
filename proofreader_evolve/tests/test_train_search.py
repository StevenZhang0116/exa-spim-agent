"""Synthetic training search and worker boundaries. No datasets or API calls."""

import asyncio
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from proofreader_evolve.harness import fixed_pool_scoring as scoring
from proofreader_evolve.harness.isolated_scoring import score, preflight, ScorerExecutionError
from proofreader_evolve.harness.native_pool import NativeTable, pool_digest
from proofreader_evolve.harness.scorer_components import components, compose
from proofreader_evolve.harness.train_experiments import TrainingExperiments, ExperimentMemory
from proofreader_evolve.harness.search_proposals import formula_info, parameterize, parameter_candidates, read_proposal
from proofreader_evolve.harness.candidate_pool import CandidatePool
from proofreader_evolve.harness.training_diagnostics import describe, feature_statistics
from proofreader_evolve.tests.test_fixed_pool_scoring import fixture, BASELINE, IMPROVED


PARAMETERIZED = ("PARAMS = {'weight': 0.0}\n"
                 "def score_candidates(features, ctx):\n"
                 "    return (features['detector_score'].fillna(0) + "
                 "PARAMS['weight'] * features['evidence'].fillna(0)).to_numpy()\n")


def write_proposal(session, hypothesis='Evidence improves ranking', strategy='Blend evidence', grid=None):
    session.proposal_path.write_text(json.dumps({'hypothesis': hypothesis, 'strategy': strategy,
                                                'family': 'evidence', 'parameter_grid': grid or {}}))


def both_kinds(brain='1'):
    bank = fixture(brain)
    merge = deepcopy(bank.tables['split'])
    merge.kind = 'merge'
    merge.candidates = [{'segment_id': 1, 'node_id': i} for i in range(3)]
    merge.meta['pool_sha256'] = pool_digest('merge', merge.candidates)
    bank.tables['merge'] = merge
    return bank


class DiagnosticsTests(unittest.TestCase):
    def test_strata_rotation_and_parent_delta(self):
        n = 100
        rows = [{'segment_id_a': 1, 'segment_id_b': i + 2} for i in range(n)]
        table = NativeTable('split', pd.DataFrame({'x': np.arange(n)}), rows,
                            np.arange(n) % 2, {'pool_sha256': pool_digest('split', rows)})
        scores = np.arange(n, 0, -1, dtype=float)
        before, after = np.arange(10), np.arange(1, 11)
        result = describe(table, scores, after, generation=1, cell='1/split', parent={'chosen': before})
        self.assertEqual(result['ranking_delta']['net_tp'], 0)
        self.assertEqual(result['ranking_delta']['entered'], 1)
        self.assertFalse(result['ranking_delta']['same_top_k'])
        self.assertTrue(all(result['group_sizes'][g] > 0 for g in (
            'selected_positive', 'selected_label0', 'boundary_positive', 'boundary_label0', 'missed_positive')))
        self.assertEqual(result, describe(table, scores, after, 1, '1/split', {'chosen': before}))
        rotated = describe(table, scores, after, 2, '1/split', {'chosen': before})
        samples = lambda r: [e['features']['x'] for e in r['training_examples'] if e['group'] == 'missed_positive']
        self.assertEqual(samples(result)[0], samples(rotated)[0])
        self.assertNotEqual(samples(result)[1:], samples(rotated)[1:])
        self.assertTrue(all(set(e) == {'group', 'label', 'score', 'features', 'candidate_ref'}
                            for e in result['training_examples']))
        gain = describe(table, scores, np.array([1]), parent={'chosen': np.array([0])})
        self.assertEqual(gain['ranking_delta']['net_tp'], 1)
        self.assertEqual(gain['group_sizes']['gained_positive'], 1)

    def test_statistics_use_whole_pool_and_handle_missing_values(self):
        bank = fixture()
        bank.tables['split'].features.loc[0, 'evidence'] = np.nan
        cell = feature_statistics({'1': bank}, 'split')['cells']['1/split']['evidence']
        self.assertEqual(cell['0']['count'], 2)
        self.assertEqual(cell['0']['finite'], 1)
        self.assertEqual(cell['1']['q25_q50_q75'], [.9, .9, .9])


class WorkerTests(unittest.TestCase):
    def test_feature_math_and_missing_value_preflight(self):
        frame = fixture().tables['split'].features
        source = ('import numpy as np\nimport pandas as pd\n'
                  'def score_candidates(features, ctx):\n'
                  "    return np.log1p(features['evidence'].fillna(0).clip(lower=0).to_numpy())\n")
        np.testing.assert_allclose(score(source, frame, 'split'), np.log1p(frame.evidence))
        self.assertEqual(len(preflight(source, frame, 'split')), 4)
        with self.assertRaisesRegex(ScorerExecutionError, 'finite'):
            preflight(IMPROVED.replace('.fillna(0)', ''), frame, 'split')

    def test_files_network_and_child_processes_are_denied(self):
        frame = fixture().tables['split'].features
        with tempfile.TemporaryDirectory() as tmp:
            secret = Path(tmp) / 'secret.txt'
            secret.write_text('private-training-labels')
            for operation in (f'open({str(secret)!r}).read()', 'os.fork()',
                              "__import__('socket').socket()", f'os.unlink({str(secret)!r})'):
                source = ('import os\ndef score_candidates(features, ctx):\n'
                          f'    {operation}\n'
                          "    return features['detector_score'].to_numpy()\n")
                # A not-yet-loaded module can be blocked before its syscall is reached.
                errors = 'PermissionError|ModuleNotFoundError' if 'socket' in operation else 'PermissionError'
                with self.subTest(operation=operation), self.assertRaisesRegex(ScorerExecutionError, errors):
                    score(source, frame, 'split')
            self.assertEqual(secret.read_text(), 'private-training-labels')

    def test_socket_syscall_is_denied_even_without_a_socket_module_import(self):
        source = '''import ctypes
import errno
def score_candidates(features, ctx):
    libc = ctypes.CDLL(None, use_errno=True)
    result = libc.socket(2, 1, 0)
    if result != -1 or ctypes.get_errno() != errno.EPERM:
        raise RuntimeError('Socket syscall was not denied with EPERM')
    return features['detector_score'].to_numpy()
'''
        frame = fixture().tables['split'].features
        np.testing.assert_array_equal(score(source, frame, 'split'), frame.detector_score)

    def test_credentials_do_not_enter_worker(self):
        source = ("import os\nassert 'ANTHROPIC_API_KEY' not in os.environ\n"
                  "assert 'AWS_SECRET_ACCESS_KEY' not in os.environ\n" + BASELINE)
        with patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'fixture', 'AWS_SECRET_ACCESS_KEY': 'fixture'}):
            np.testing.assert_equal(score(source, fixture().tables['split'].features, 'split'), [.9, .1, .2])

    def test_nonterminating_worker_has_hard_timeout(self):
        with self.assertRaisesRegex(ScorerExecutionError, 'wall-clock budget'):
            score('while True: pass', fixture().tables['split'].features, 'split', timeout=3)

    def test_component_bundle_freezes_other_kind_and_round_trips(self):
        source = compose({'merge': BASELINE, 'split': IMPROVED})
        self.assertEqual(components(source), {'merge': BASELINE, 'split': IMPROVED})
        result = scoring.evaluate(source, {'1': both_kinds()}, {'merge': 1, 'split': 1})
        self.assertEqual(result['cells']['1/merge']['precision'], 0)
        self.assertEqual(result['cells']['1/split']['precision'], 1)
        with self.assertRaisesRegex(ValueError, 'Invalid harness component bundle'):
            components(source + '\nprint("tampered")\n')


class TrainingSearchTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        redirect = redirect_stdout(io.StringIO())
        redirect.__enter__()
        self.addCleanup(redirect.__exit__, None, None, None)
        self.train = {'1': fixture()}
        self.state = {}
        # One initial worker run per test; candidates exercise real isolated evaluation.
        self.parent = scoring.evaluate(BASELINE, self.train, {'split': 1}, state=self.state)
        self.memory = ExperimentMemory(self.root)
        self.session = self.make_session(1)

    def make_session(self, generation, max_evaluations=3):
        gen = self.root / f'gen{generation:03d}'
        gen.mkdir()
        policy, rules = gen / 'scorer.py', gen / 'rules.md'
        policy.write_text(BASELINE)
        rules.write_text('fixture notes')
        return TrainingExperiments(gen, policy, rules, 'split', components(BASELINE),
                                   self.train, self.parent, self.state, {'split': 1}, 120, 0,
                                   self.memory, max_evaluations, generation)

    def evaluate(self, source, hypothesis='Evidence improves ranking', strategy='Rank by evidence'):
        self.session.policy_path.write_text(source)
        write_proposal(self.session, hypothesis, strategy)
        return self.session.evaluate()

    def test_snapshots_deduplication_and_final_source_binding(self):
        result = self.evaluate(IMPROVED)
        self.assertEqual(result['status'], 'evaluated')
        self.assertTrue(result['train_gate']['passed'])
        self.assertEqual(result['ranking_delta']['1/split']['gained_positives'], 1)
        self.assertEqual((Path(result['snapshot']) / 'scorer.py').read_text(), IMPROVED)
        self.assertEqual(self.session.edit_base, result['experiment'])
        self.assertTrue((Path(result['snapshot']) / 'scorer.py.from_edit_base.diff').is_file())
        with patch('proofreader_evolve.harness.train_experiments.scoring.evaluate') as evaluate:
            repeated = self.session.evaluate()
        evaluate.assert_not_called()
        self.assertTrue(repeated['cached'])
        self.assertEqual(len(self.session.attempts), 1)
        self.session.policy_path.write_text(IMPROVED + '\n# untested final edit\n')
        with self.assertRaisesRegex(ValueError, 'exactly match'):
            self.session.submitted()
        self.session.policy_path.write_text(IMPROVED)
        self.assertEqual(self.session.submitted()[0]['component'], IMPROVED)
        self.assertNotIn('validation', self.memory.path.read_text().lower())
        self.assertEqual(self.session.parent_sources['merge'], BASELINE)

    def test_budget_one_extra_repair_and_no_fabricated_measurements(self):
        error = self.evaluate('bad syntax!')
        self.assertEqual(error['status'], 'execution_error')
        self.assertIsNone(error['train'])
        self.assertEqual(error['evaluations_remaining'], 3)
        self.assertEqual(self.evaluate(IMPROVED)['status'], 'evaluated')
        self.assertEqual(self.evaluate(BASELINE)['status'], 'evaluated')
        self.assertEqual(self.evaluate('also invalid!')['status'], 'execution_error')
        self.assertEqual(self.evaluate(IMPROVED + '\n')['status'], 'budget_exhausted')
        self.assertEqual(len(self.session.attempts), 4)
        self.assertEqual(len(self.memory.entries), 4)

    def test_no_repair_bonus_without_error(self):
        self.session.max_evaluations = 1
        self.assertEqual(self.evaluate(IMPROVED)['status'], 'evaluated')
        self.assertEqual(self.evaluate(BASELINE)['status'], 'budget_exhausted')
        self.assertFalse(self.session.repair_granted)

    def test_cross_generation_cache_and_search_recovers_older_distinct_failure(self):
        first = self.evaluate(IMPROVED, 'Rare eigenvalue feature', 'spectral ranking')
        self.evaluate('bad syntax!', 'Shape error', 'invalid syntax')
        # Later parents do not alter saved measurements; retrieval searches the whole archive.
        self.session = self.make_session(9)
        with patch('proofreader_evolve.harness.train_experiments.scoring.evaluate') as evaluate:
            again = self.evaluate(IMPROVED)
        evaluate.assert_not_called()
        self.assertTrue(again['cached'])
        self.assertEqual(self.session.evaluations_used, 0)
        self.assertEqual(again['train'], first['train'])
        self.assertEqual(again['cached_from'], first['experiment'])
        found = self.memory.search('split', 'rare eigenvalue', limit=1)
        self.assertEqual(found[0]['experiment'], 'gen001/attempt001')
        self.assertEqual(self.memory.search('merge'), [])

    def test_observed_workspace_lineage_tracks_restore_and_parameter_grid_siblings(self):
        self.session.max_evaluations = 4
        first = self.evaluate(IMPROVED)
        second = self.evaluate(BASELINE)
        self.assertEqual(second['lineage']['edit_base_experiment'], first['experiment'])
        restored = self.session.restore(first['experiment'])
        self.session.policy_path.write_text(PARAMETERIZED)
        write_proposal(self.session, grid={'weight': [0.0, 1.0]})
        batch = self.session.search_parameters()
        entries = [self.memory.candidates[row['experiment']] for row in batch['candidates']]
        self.assertTrue(all(entry['lineage']['edit_base_experiment'] == restored['restored'] for entry in entries))
        self.assertEqual(self.session.edit_base, batch['best']['restored'])

    def test_mcp_handlers_and_explicit_tool_permissions(self):
        import claude_agent_sdk as sdk
        from proofreader_evolve.harness.reviser_session import bind_session_options
        from proofreader_evolve.harness.reviser_access import pre_tool_hook
        self.session.policy_path.write_text(IMPROVED)
        write_proposal(self.session)
        with patch.object(sdk, 'create_sdk_mcp_server') as create:
            self.session.mcp_server()
        handlers = {t.name: t.handler for t in create.call_args.kwargs['tools']}
        result = asyncio.run(handlers['evaluate_train']({}))
        self.assertEqual(json.loads(result['content'][0]['text'])['status'], 'evaluated')
        used = self.session.evaluations_used
        result = asyncio.run(handlers['train_classifier']({}))
        self.assertIn('proposal.json.classifier', json.loads(result['content'][0]['text'])['error'])
        self.assertEqual(self.session.evaluations_used, used)
        result = asyncio.run(handlers['search_memory']({'query': 'Evidence'}))
        entry = json.loads(result['content'][0]['text'])[0]
        self.assertIn('code_diff', entry)
        options = sdk.ClaudeAgentOptions(cwd=str(self.root))
        bound = bind_session_options(options, self.session.policy_path, self.session.rules_path,
                                     self.session.gen_dir / 'train_feedback.json', training_server={'type': 'sdk'})
        guard = bound.can_use_tool
        for tool, expected in [('mcp__training__evaluate_train', 'allow'),
                               ('mcp__training__search_parameters', 'allow'),
                               ('mcp__training__train_classifier', 'allow'),
                               ('mcp__training__inspect_candidate', 'allow'),
                               ('mcp__training__restore_candidate', 'allow'),
                               ('mcp__training__search_memory', 'allow'),
                               ('mcp__training__evaluate_validation', 'deny'), ('Bash', 'deny')]:
            self.assertEqual(asyncio.run(guard(tool, {}, None)).behavior, expected)
            response = asyncio.run(pre_tool_hook(guard)({'tool_name': tool, 'tool_input': {}}, 'id', None))
            self.assertEqual(response['hookSpecificOutput']['permissionDecision'], expected)
        for tool in ('Read', 'Write'):
            denied = asyncio.run(guard(tool, {'file_path': str(self.root / 'final.json')}, None))
            self.assertEqual(denied.behavior, 'deny')
        self.assertEqual(asyncio.run(guard('Write', {'file_path': str(self.session.proposal_path)}, None)).behavior,
                         'allow')
        for tool in ('Read', 'Write', 'Edit'):
            self.assertEqual(asyncio.run(guard(tool, {'file_path': str(self.session.training_path)}, None)).behavior,
                             'allow')
        for name in ('candidate_pool.json', 'search_plan.json', 'classifier_guide.md', 'model_environment.json',
                     'research_status.json', '../research_evidence.jsonl',
                     'classifier_fits/fit001/model.json', 'experiments/attempt001/scorer.py'):
            self.assertEqual(asyncio.run(guard('Write', {'file_path': str(self.session.gen_dir / name)}, None)).behavior,
                             'deny')

    def test_bad_proposal_and_oversize_grid_do_not_spend_budget(self):
        self.session.proposal_path.write_text('{"hypothesis": "unfinished"}')
        with self.assertRaisesRegex(ValueError, 'strategy'):
            self.session.evaluate()
        self.session.policy_path.write_text(PARAMETERIZED)
        write_proposal(self.session, grid={'weight': [0, 1, 2, 3]})
        with self.assertRaisesRegex(ValueError, 'Grid needs 4'):
            self.session.search_parameters()
        self.assertEqual(self.session.evaluations_used, 0)
        self.assertEqual(self.session.attempts, [])

    def test_grid_restores_measured_winner_and_cache_is_free(self):
        self.session.policy_path.write_text(PARAMETERIZED)
        write_proposal(self.session, grid={'weight': [0, 2]})
        result = self.session.search_parameters()
        self.assertEqual(result['status'], 'searched')
        self.assertEqual(self.session.evaluations_used, 2)
        winner, _ = self.session.submitted()
        self.assertEqual(winner['entry']['parameters'], {'weight': 2})
        self.assertEqual(winner['report']['macro_precision'], 1)
        self.session.max_evaluations = 2
        with patch('proofreader_evolve.harness.train_experiments.scoring.evaluate') as evaluate:
            restored = self.session.restore('gen001/attempt001')
        evaluate.assert_not_called()
        self.assertEqual(restored['evaluations_remaining'], 0)
        self.assertEqual(self.session.submitted()[0]['entry']['parameters'], {'weight': 0})
        self.session.restore('best')
        self.assertEqual(self.session.submitted()[0]['entry']['parameters'], {'weight': 2})

    def test_tuning_locks_formula_but_accepts_numeric_changes(self):
        formula, _ = formula_info(PARAMETERIZED)
        self.session.search_plan = {'mode': 'tune', 'search_parent': formula}
        with self.assertRaisesRegex(ValueError, 'tune generation'):
            self.evaluate(IMPROVED)
        self.assertEqual(self.session.evaluations_used, 0)
        result = self.evaluate(parameterize(PARAMETERIZED, {'weight': 2}))
        self.assertEqual(result['status'], 'evaluated')

    def test_restore_reuses_target_cache_with_new_frozen_other_kind(self):
        first = self.evaluate(IMPROVED)
        self.session = self.make_session(2)
        self.session.train = {'1': both_kinds()}
        accepted = compose({'merge': IMPROVED, 'split': BASELINE})
        self.session.parent_sources = components(accepted)
        self.session.parent_state = {}
        self.session.parent_report = scoring.evaluate(accepted, self.session.train, {'merge': 1, 'split': 1},
                                                       state=self.session.parent_state)
        self.session.budgets = {'merge': 1, 'split': 1}
        with patch('proofreader_evolve.harness.train_experiments.scoring.evaluate') as evaluate:
            result = self.session.restore(first['experiment'])
        evaluate.assert_not_called()
        self.assertTrue(result['cached'])
        self.assertEqual(self.session.evaluations_used, 0)
        submitted, _ = self.session.submitted()
        self.assertEqual(components(submitted['source']), {'merge': IMPROVED, 'split': IMPROVED})
        self.assertEqual(submitted['report']['cells']['1/merge']['precision'], 1)


class SearchDriverTests(unittest.TestCase):
    def test_accepted_train_regression_is_protected_and_revisited(self):
        from proofreader_evolve.cli import run_precision_evolution as driver
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            seed = Path(tmp) / 'seed.py'
            seed.write_text(IMPROVED)
            args = driver.parse_args(['--selection-protocol', 'in_sample', '--promotion-gate', 'margin', '--kind-schedule', 'alternate', '--train-brains', '1', '--validation-brains', '2',
                                      '--split-k', '1', '--generations', '3', '--runs-dir', tmp,
                                      '--start-from', str(seed), '--candidate-pool-size', '1'])
            def bank(brain, *args, **kwargs):
                result = fixture(brain)
                if brain == '2':
                    result.tables['split'].truth = np.array([1, 0, 0])
                return result
            async def revise(run_dir, policy, rules, report, model, *, experiments):
                if experiments.generation == 1:
                    policy.write_text(BASELINE)  # Worse TRAIN, better outer evaluation.
                    write_proposal(experiments)
                    experiments.evaluate()
                else:
                    self.assertEqual(policy.read_text(), BASELINE)
                    self.assertEqual(experiments.parent_sources['split'], BASELINE)
                    if experiments.generation == 2:
                        # Follow the newly promoted, weaker-TRAIN branch immediately.
                        self.assertEqual(experiments.search_plan['reason'], 'promotion_followup')
                        policy.write_text(BASELINE.replace("features['detector_score']",
                                                           "(features['detector_score'] * 1.01)"))
                        write_proposal(experiments)
                        experiments.evaluate()
                    else:
                        experiments.restore('parent')
                    for name in ('search_plan.json', 'candidate_pool.json'):
                        text = (policy.parent / name).read_text()
                        self.assertNotIn('validation', text.lower())
                        self.assertNotIn('2/split', text)
                return {'summary': 'Reference branch regression fixture', 'cost_usd': 0}
            with patch.object(driver.pc, 'resolve_detector_runs', return_value={}), \
                    patch.object(driver, 'ensure_native_tables', side_effect=bank):
                path = asyncio.run(driver.run(args, revise_fn=revise))
            rows = [json.loads(line) for line in (path / 'ledger.jsonl').read_text().splitlines()]
            self.assertTrue(rows[0]['accepted'])
            self.assertFalse(rows[0]['train_gate']['passed'])
            self.assertEqual(rows[1]['search_parent'], 'gen001/attempt001')
            self.assertTrue(rows[1]['research_status']['followup_complete'])
            self.assertEqual(rows[2]['search_branch_role'], 'reference')
            self.assertEqual(rows[2]['search_parent'], 'gen001/attempt001')
            self.assertFalse(rows[1]['accepted'])
            summary = json.loads((path / 'candidate_pool.json').read_text())['kinds']['split']
            self.assertEqual(summary['candidates'][0]['experiment'], 'seed/split')
            self.assertEqual(summary['reference_branch']['experiment'], 'gen001/attempt001')
            self.assertEqual(components((path / 'best_scorer.py').read_text())['split'], BASELINE)

    def test_validation_rejected_branch_remains_searchable_without_becoming_accepted_parent(self):
        from proofreader_evolve.cli import run_precision_evolution as driver
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            seed = Path(tmp) / 'seed.py'
            seed.write_text(BASELINE)
            args = driver.parse_args(['--selection-protocol', 'in_sample', '--promotion-gate', 'margin', '--kind-schedule', 'alternate', '--train-brains', '1', '--validation-brains', '2',
                                      '--split-k', '1', '--generations', '2', '--runs-dir', tmp,
                                      '--start-from', str(seed)])
            def bank(brain, *args, **kwargs):
                result = fixture(brain)
                if brain == '2':
                    result.tables['split'].truth = np.array([1, 0, 0])
                return result
            async def revise(run_dir, policy, rules, report, model, *, experiments):
                if experiments.generation == 1:
                    policy.write_text(IMPROVED)
                    write_proposal(experiments)
                    experiments.evaluate()
                else:
                    self.assertEqual(policy.read_text(), IMPROVED)
                    self.assertEqual(experiments.parent_sources['split'], BASELINE)
                    self.assertEqual(experiments.search_plan['search_parent']['experiment'], 'gen001/attempt001')
                    # Feedback and failure cases describe the assigned branch (IMPROVED, not accepted),
                    # compared against the accepted scorer (BASELINE).
                    feedback = json.loads(report.read_text())
                    self.assertEqual(feedback['search_branch']['experiment'], 'gen001/attempt001')
                    self.assertFalse(feedback['search_branch']['same_as_accepted'])
                    self.assertEqual(feedback['search_branch']['selection']['macro_precision'], 1)
                    self.assertEqual(feedback['format'], 'stratified-train-v5')
                    self.assertEqual(feedback['parent_selection']['macro_precision'], 0)
                    self.assertIn('assigned search branch', feedback['reading_guide'])
                    cases = json.loads((policy.parent / 'failure_cases.json').read_text())
                    self.assertEqual(cases['branch'], {'experiment': 'gen001/attempt001', 'same_as_accepted': False})
                    comparison = cases['branch_vs_accepted']
                    self.assertEqual(comparison['cells']['1/split'],
                                     {'branch_tp': 1, 'accepted_tp': 0, 'branch_new_positives': 1,
                                      'branch_lost_positives': 0, 'shared_missed_positives': 0})
                    self.assertEqual(len(comparison['branch_new_positives']), 1)
                    self.assertEqual(cases['pairs'], [])  # the branch misses no positive on TRAIN
                    for text in (report.read_text(), json.dumps(cases)):
                        self.assertNotIn('validation', text.lower())
                    experiments.restore('parent')
                    self.assertEqual(experiments.evaluations_used, 0)
                return {'summary': 'TRAIN branch retained', 'cost_usd': 0}
            with patch.object(driver.pc, 'resolve_detector_runs', return_value={}), \
                    patch.object(driver, 'ensure_native_tables', side_effect=bank):
                path = asyncio.run(driver.run(args, revise_fn=revise))
            records = [json.loads(line) for line in (path / 'ledger.jsonl').read_text().splitlines()]
            self.assertTrue(all(not r['accepted'] and r['train_gate']['passed'] for r in records))
            self.assertEqual((path / 'best_scorer.py').read_text(), BASELINE)
            pool = json.loads((path / 'candidate_pool.json').read_text())
            self.assertEqual(pool['kinds']['split']['candidates'][0]['target_precision'], 1)

    def test_three_train_attempts_one_validation_and_alternating_components(self):
        from proofreader_evolve.cli import run_precision_evolution as driver
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            args = driver.parse_args(['--selection-protocol', 'in_sample', '--promotion-gate', 'margin', '--kind-schedule', 'alternate', '--train-brains', '1', '--validation-brains', '2',
                                      '--merge-k', '1', '--split-k', '1', '--generations', '2', '--runs-dir', tmp])
            targets = []
            async def revise(run_dir, policy, rules, report, model, *, experiments):
                targets.append(experiments.target_kind)
                feedback = json.loads(report.read_text())
                self.assertEqual(list(feedback['examples']), [f'1/{experiments.target_kind}'])
                self.assertNotIn('validation', report.read_text().lower())
                for source in (BASELINE, IMPROVED + '\n# intermediate\n', IMPROVED):
                    policy.write_text(source)
                    write_proposal(experiments, 'Fixture ranking', 'Evidence')
                    experiments.evaluate()
                return {'summary': 'three training attempts', 'cost_usd': 0}
            with patch.object(driver.pc, 'resolve_detector_runs', return_value={}), \
                    patch.object(driver, 'ensure_native_tables', side_effect=lambda brain, *a, **k: both_kinds(brain)), \
                    patch.object(driver.scoring, 'evaluate', wraps=scoring.evaluate) as evaluate:
                path = asyncio.run(driver.run(args, revise_fn=revise))
            self.assertEqual(targets, ['merge', 'split'])
            validation_calls = [c for c in evaluate.call_args_list if set(c.args[1]) == {'2'}]
            # Seed + frozen baseline + exactly one final candidate each generation.
            self.assertEqual(len(validation_calls), 4)
            records = [json.loads(s) for s in (path / 'ledger.jsonl').read_text().splitlines()]
            self.assertTrue(all(r['accepted'] for r in records))
            self.assertTrue(all(len(r['experiments']) == 3 for r in records))
            first = components((path / 'gen001/experiments/attempt003/combined_scorer.py').read_text())
            self.assertEqual(first['merge'], IMPROVED)
            self.assertEqual(first['split'], args.start_from.read_text())
            final = components((path / 'best_scorer.py').read_text())
            self.assertEqual(final, {'merge': IMPROVED, 'split': IMPROVED})
            saved_pool = json.loads((path / 'candidate_pool.json').read_text())
            self.assertEqual(saved_pool['selection'], 'train-champion-and-specialists-v1')
            for kind, summary in saved_pool['kinds'].items():
                for branch in summary['candidates']:
                    self.assertTrue(branch['specialist_profile']['slices'])
                    self.assertTrue(all(name.startswith(f'1/{kind}/')
                                        for name in branch['specialist_profile']['slices']))
                    self.assertIn('vs_champion', branch)
                    self.assertIn('vs_other_retained', branch)
            plan = json.loads((path / 'gen002/search_plan.json').read_text())
            self.assertIn('complementary_candidates', plan)


class ProposalTests(unittest.TestCase):
    def test_parameter_substitution_preserves_formula_and_unicode(self):
        source = '# Naive scorer: \u03b1-weighted geometry\n' + PARAMETERIZED
        changed = parameterize(source, {'weight': 2.5})
        self.assertIn('# Naive scorer: \u03b1-weighted geometry', changed)
        self.assertEqual(formula_info(source)[0]['formula_sha256'], formula_info(changed)[0]['formula_sha256'])
        self.assertNotEqual(formula_info(source)[0]['formula_sha256'], formula_info(IMPROVED)[0]['formula_sha256'])
        with self.assertRaisesRegex(ValueError, 'Every grid key'):
            parameterize(source, {'typo': 2})
        with self.assertRaises(ValueError):
            parameterize(source, {'weight': float('nan')})
        self.assertEqual(len(parameter_candidates(source, {'parameter_grid': {'weight': [1, 1, 2]}})), 2)

    def test_proposals_reject_duplicate_keys_and_nonfinite_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'proposal.json'
            path.write_text('{"hypothesis":"a", "hypothesis":"b", "strategy":"c"}')
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                read_proposal(path)
            path.write_text('{"hypothesis":"a", "strategy":"c", "parameter_grid":{"weight":[NaN]}}')
            with self.assertRaisesRegex(ValueError, 'finite numeric'):
                read_proposal(path)


class CandidatePoolTests(unittest.TestCase):
    def entry(self, number, precision=.5, ranking=None, formula='formula', cached=False):
        return {'experiment': f'gen001/attempt{number:03d}', 'sequence': number, 'target_kind': 'split',
                'component_sha256': str(number), 'target_precision': precision,
                'top_k_sha256': ranking or str(number), 'formula_sha256': formula,
                'family': formula, 'parameters': {'weight': 1}, 'status': 'evaluated',
                'cached': cached, 'train': {'macro_precision': precision, 'cells': {}}}

    def test_retains_diverse_near_best_branches_without_duplicate_top_k(self):
        with tempfile.TemporaryDirectory() as tmp:
            pool = CandidatePool(Path(tmp) / 'pool.json', ['split'], size=3, score_tolerance=.02)
            for entry in (self.entry(1), self.entry(2, .49, formula='other'),
                          self.entry(3, .5, ranking='1'), self.entry(4, .1)):
                pool.add(entry)
            self.assertEqual([e['sequence'] for e in pool.members['split']], [1, 2])
            self.assertEqual(pool.next_plan()['mode'], 'tune')

    def test_reference_cadence_is_per_kind_and_duplicate_roles_share_a_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            pool = CandidatePool(Path(tmp) / 'pool.json', ['split', 'merge'], size=1)
            split = self.entry(1)
            merge = {**self.entry(2), 'target_kind': 'merge'}
            for entry in (split, merge):
                pool.add(entry)
                pool.set_reference(entry)
            plans = [pool.next_plan() for _ in range(6)]
            self.assertEqual([p['target_kind'] for p in plans], ['split', 'merge'] * 3)
            self.assertEqual([p['scheduled_round'] for p in plans], [1, 1, 2, 2, 3, 3])
            self.assertEqual([p['branch_role'] for p in plans], ['train_champion'] * 4 + ['reference'] * 2)
            for kind in ('split', 'merge'):
                self.assertEqual(len(pool.members[kind]), 1)
                self.assertEqual(pool.members[kind][0]['component_sha256'], pool.references[kind]['component_sha256'])
            replacement = self.entry(3, .1)
            pool.set_reference(replacement)
            self.assertEqual(pool.references['merge']['experiment'], merge['experiment'])

    def test_reference_round_does_not_consume_periodic_train_alternative_slot(self):
        with tempfile.TemporaryDirectory() as tmp:
            pool = CandidatePool(Path(tmp) / 'pool.json', ['split'], size=2,
                                 explore_every=3, plateau_patience=99, early_stop=False)
            pool.add(self.entry(1))
            pool.add(self.entry(2, .49, formula='alternative'))
            pool.set_reference(self.entry(9, .1))
            for number in (3, 4):
                pool.finish(pool.next_plan(), [self.entry(number, ranking='1')])
            plan = pool.next_plan()
            self.assertEqual(plan['branch_role'], 'reference')
            pool.finish(plan, [self.entry(5, .1)])
            plan = pool.next_plan()
            self.assertEqual(plan['reason'], 'periodic_exploration')
            self.assertEqual(plan['search_parent']['sequence'], 2)

    def test_plateau_switches_branch_and_stops_only_after_measured_explorations(self):
        with tempfile.TemporaryDirectory() as tmp:
            pool = CandidatePool(Path(tmp) / 'pool.json', ['split'], explore_every=99)
            pool.add(self.entry(1))
            pool.add(self.entry(2, .49, formula='other'))
            for number in (3, 4):
                plan = pool.next_plan()
                pool.finish(plan, [self.entry(number, ranking='1')])
            plan = pool.next_plan()
            self.assertEqual(plan['mode'], 'explore')
            self.assertEqual(plan['reason'], 'plateau')
            self.assertEqual(plan['search_parent']['sequence'], 2)
            pool.finish(plan, [self.entry(5, cached=True)])
            self.assertEqual(pool.progress['split']['stalled_explorations'], 0)
            pool.finish(pool.next_plan(), [{'status': 'execution_error'}])
            self.assertEqual(pool.progress['split']['stalled_explorations'], 0)
            for number in (6, 7):
                pool.finish(pool.next_plan(), [self.entry(number, ranking='1')])
            self.assertIsNone(pool.next_plan())

    def test_improvement_resets_stagnation_and_other_kind_keeps_running(self):
        with tempfile.TemporaryDirectory() as tmp:
            pool = CandidatePool(Path(tmp) / 'pool.json', ['split', 'merge'])
            pool.add(self.entry(1))
            pool.add({**self.entry(2), 'target_kind': 'merge'})
            pool.progress['split']['stalled_rounds'] = 3
            pool.progress['split']['stalled_explorations'] = 1
            plan = pool.next_plan()
            pool.finish(plan, [self.entry(3, .6)])
            self.assertEqual(pool.progress['split']['stalled_rounds'], 0)
            self.assertEqual(pool.progress['split']['stalled_explorations'], 0)
            pool.progress['split']['paused'] = True
            self.assertEqual(pool.next_plan()['target_kind'], 'merge')

    def test_early_stop_can_be_disabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            pool = CandidatePool(Path(tmp) / 'pool.json', ['split'], early_stop=False,
                                 plateau_patience=1, exploration_patience=1)
            pool.add(self.entry(1))
            for number in range(2, 7):
                pool.finish(pool.next_plan(), [self.entry(number, ranking='1')])
            self.assertFalse(pool.progress['split']['paused'])
            self.assertEqual(pool.next_plan()['reason'], 'plateau')


if __name__ == '__main__':
    unittest.main()
