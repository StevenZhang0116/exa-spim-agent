"""Synthetic research completion and scheduling; no cloud data or LLM calls."""

import asyncio
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from proofreader_evolve.harness.candidate_pool import CandidatePool
from proofreader_evolve.harness.research_evidence import (
    ResearchEvidence, program_identity, observed_descendant, registered_image_available)
from proofreader_evolve.harness.train_experiments import TrainingExperiments, ExperimentMemory
from proofreader_evolve.harness.trajectory import Trajectory
from proofreader_evolve.tests.test_fixed_pool_scoring import fixture, BASELINE
from proofreader_evolve.tests.label_fixture import setUpModule, tearDownModule  # noqa: F401


def entry(number, kind='split', precision=.5):
    return {'experiment': f'gen{number:03d}/attempt001', 'sequence': number, 'target_kind': kind,
            'component_sha256': str(number), 'target_precision': precision, 'top_k_sha256': str(number),
            'formula_sha256': 'formula', 'family': 'fixture', 'parameters': {'weight': 1},
            'status': 'evaluated', 'cached': False, 'train': {'cells': {}}}


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.evidence = ResearchEvidence(self.root / 'research.jsonl')
        self.plan = {'search_parent': {'experiment': 'promoted'}, 'followup_required': True,
                     'investigation': {'required_source': 'raw_image'}}

    def record(self, generation=1, **kwargs):
        values = dict(kind='split', generation=generation, tool='run_volume_analysis',
                      source='raw_image', category='measurement', identity={'inputs': 'A', 'method': 'x'},
                      success=True, artifact=self.root / 'result.json', edit_base='promoted', comparison=True,
                      outcome={'delta': -.02})
        values.update(kwargs)
        return self.evidence.record(**values)

    def status(self, generation=1, candidates=None):
        return self.evidence.status('split', generation, self.plan, candidates or {})

    def test_negative_comparison_counts_but_repeating_it_does_not(self):
        self.record()
        self.assertTrue(self.status()['complete'])
        self.record(generation=2)
        status = self.status(2)
        self.assertFalse(status['complete'])
        self.assertEqual(status['reused_results'], 1)
        self.record(generation=2, identity={'inputs': 'B', 'method': 'x'})
        self.assertTrue(self.status(2)['complete'])

    def test_observations_and_wrong_source_do_not_complete_investigation(self):
        self.record(category='observation', comparison=False)
        self.record(tool='evaluate_feature_ablation', source='feature_comparison')
        self.assertFalse(self.status()['complete'])
        self.assertEqual(self.status()['new_observations'], 1)
        self.assertEqual(self.evidence.sources('split')['compared'], ['feature_comparison'])

    def test_execution_failure_is_retryable_not_a_negative_measurement(self):
        self.record(success=False, outcome={'error': 'IO unavailable'})
        self.assertEqual(self.status()['status'], 'execution_blocked')
        self.assertEqual(self.status()['new_measurements'], 0)
        self.record()
        self.assertTrue(self.status()['complete'])

    def test_cache_and_comment_or_docstring_edits_are_not_new_evidence(self):
        original = 'def analyze(context):\n    return {"mean": 0.0}\n'
        edited = '# changed comment\n"""new module docstring"""\n' + original.replace(
            '    return', '    """new function docstring"""\n    return')
        self.assertEqual(program_identity(original), program_identity(edited))
        self.record(identity={'program': program_identity(original)}, cached=True)
        self.record(identity={'program': program_identity(edited)})
        self.assertEqual(self.status()['new_measurements'], 0)

    def test_restoring_unrelated_candidate_cannot_fake_promoted_ancestry(self):
        candidates = {'restore': {'cached': True, 'cached_from': 'unrelated',
                                  'lineage': {'edit_base_experiment': 'promoted'}},
                      'child': {'lineage': {'edit_base_experiment': 'restore'}}}
        self.assertFalse(observed_descendant('child', 'promoted', candidates))
        self.record(edit_base='child')
        self.assertFalse(self.status(candidates=candidates)['complete'])
        candidates['restore']['cached_from'] = 'promoted'
        self.assertTrue(self.status(candidates=candidates)['complete'])

    def test_score_measurement_alone_does_not_fulfill_combined_assignment(self):
        self.record(tool='evaluate_train', source='scoring', comparison=False)
        self.record(edit_base='unrelated')
        status = self.status()
        self.assertTrue(status['followup_complete'])
        self.assertFalse(status['investigation_complete'])

    def test_image_requirement_needs_matching_registration_not_just_a_provider(self):
        table = SimpleNamespace(meta={'provenance': {'brain': 'TRAIN', 'source_cache': {'hash': 'a'}}},
                                local_context=object(), image_context=SimpleNamespace(reviews={}))
        self.assertFalse(registered_image_available(table))
        table.image_context.reviews['TRAIN'] = {'status': 'reviewed', 'source_cache': {'hash': 'b'}}
        self.assertFalse(registered_image_available(table))
        table.image_context.reviews['TRAIN']['source_cache']['hash'] = 'a'
        self.assertTrue(registered_image_available(table))


class SchedulingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.evidence = ResearchEvidence(self.root / 'research.jsonl')
        self.pool = CandidatePool(self.root / 'pool.json', ['split', 'merge'],
                                  research=self.evidence, image_kinds=['split'], explore_every=99)
        for candidate in (entry(0), entry(1, 'merge')):
            self.pool.add(candidate)
            self.pool.set_reference(candidate)

    def finish(self, plan, **receipt):
        return self.pool.finish(plan, [], research_status=receipt)

    def test_promotion_gets_next_same_kind_slot_and_only_one_bounded_retry(self):
        self.assertIsNone(self.pool.progress['split']['pending_followup'])
        self.assertEqual(self.pool.next_plan()['target_kind'], 'split')
        self.pool.set_reference(entry(8, precision=.1))
        self.assertEqual(self.pool.next_plan()['target_kind'], 'merge')
        for number in range(2):
            plan = self.pool.next_plan()
            self.assertEqual(plan['reason'], 'promotion_followup')
            self.assertEqual(plan['search_parent']['sequence'], 8)
            self.finish(plan, new_measurements=0, followup_complete=False)
            self.assertEqual(self.pool.next_plan()['target_kind'], 'merge')
        self.assertIsNone(self.pool.progress['split']['pending_followup'])
        self.assertEqual(self.pool.progress['split']['followup_outcome'], 'incomplete')
        self.pool.set_reference(entry(8, precision=.1))
        self.assertIsNone(self.pool.progress['split']['pending_followup'])

    def test_followup_completion_returns_to_regular_selection(self):
        self.pool.set_reference(entry(8, precision=.1))
        plan = self.pool.next_plan()
        self.finish(plan, new_measurements=1, followup_complete=True, complete=True)
        self.pool.next_plan()
        self.assertEqual(self.pool.next_plan()['search_parent']['sequence'], 0)

    def test_repeated_restores_force_image_investigation_and_incompletion_is_bounded(self):
        for _ in range(2):
            self.finish(self.pool.next_plan(), new_measurements=0)
            self.pool.next_plan()  # Other kind retains its independent counters.
        self.assertEqual(self.pool.progress['split']['stalled_rounds'], 0)
        plan = self.pool.next_plan()
        self.assertEqual(plan['investigation']['required_source'], 'raw_image')
        self.assertEqual(plan['investigation']['trigger'], 'no_new_evidence')
        for _ in range(2):
            progress = self.finish(plan, new_measurements=0, investigation_complete=False)
            self.pool.next_plan()
            plan = self.pool.next_plan()
        self.assertEqual(progress['pause_reason'], 'investigation_incomplete')
        self.assertTrue(progress['paused'])
        self.assertEqual(progress['stalled_explorations'], 0)

    def test_execution_errors_pause_separately_without_scientific_stall(self):
        for _ in range(2):
            result = self.finish(self.pool.next_plan(), new_measurements=0, execution_failures=1)
            self.pool.next_plan()
        self.assertEqual(result['pause_reason'], 'execution_blocked')
        self.assertEqual(result['no_evidence_rounds'], 0)
        self.assertEqual(result['stalled_rounds'], 0)

    def test_reexecuted_semantic_repeat_only_advances_no_evidence_counter(self):
        plan = self.pool.next_plan()
        progress = self.pool.finish(plan, [entry(2)], research_status={'new_measurements': 0})
        self.assertEqual(progress['no_evidence_rounds'], 1)
        self.assertEqual(progress['stalled_rounds'], 0)
        self.assertEqual(progress['rounds'], 0)

    def test_investigation_and_promoted_followup_are_combined(self):
        self.pool.progress['split']['no_evidence_rounds'] = 2
        self.pool.set_reference(entry(8, precision=.1))
        plan = self.pool.next_plan()
        self.assertTrue(plan['followup_required'])
        self.assertEqual(plan['investigation']['required_source'], 'raw_image')
        self.finish(plan, new_measurements=1, followup_complete=True,
                    investigation_complete=False, complete=False)
        self.assertIsNotNone(self.pool.progress['split']['pending_followup'])
        self.pool.next_plan()
        retry = self.pool.next_plan()
        self.assertEqual(retry['investigation'], plan['investigation'])
        self.finish(retry, new_measurements=1, followup_complete=True,
                    investigation_complete=True, complete=True)
        self.assertIsNone(self.pool.progress['split']['pending_followup'])
        self.assertIsNone(self.pool.progress['split']['pending_investigation'])

    def test_negative_comparison_resets_evidence_stall_but_not_performance_stall(self):
        self.pool.progress['split'].update(stalled_rounds=2, no_evidence_rounds=2)
        plan = self.pool.next_plan()
        progress = self.finish(plan, new_measurements=1, investigation_complete=True, complete=True)
        self.assertEqual(progress['no_evidence_rounds'], 0)
        self.assertEqual(progress['stalled_rounds'], 3)
        self.assertEqual(progress['stalled_explorations'], 1)

    def test_no_early_stop_keeps_incomplete_work_pending_without_unbounded_followup_slots(self):
        self.pool.early_stop = False
        self.pool.progress['split']['no_evidence_rounds'] = 2
        self.pool.set_reference(entry(8, precision=.1))
        for _ in range(3):
            result = self.finish(self.pool.next_plan(), new_measurements=0, investigation_complete=False)
            self.pool.next_plan()
        self.assertFalse(result['paused'])
        self.assertIsNone(result['pending_followup'])
        self.assertIsNotNone(result['pending_investigation'])


class DriverResearchTests(unittest.TestCase):
    def test_incomplete_investigation_skips_outer_evaluation_and_preserves_policy(self):
        from proofreader_evolve.cli import run_precision_evolution as driver
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            seed = Path(tmp) / 'seed.py'
            seed.write_text(BASELINE)
            args = driver.parse_args(['--selection-protocol', 'in_sample', '--promotion-gate', 'margin', '--kind-schedule', 'alternate', '--train-brains', '1', '--validation-brains', '2',
                '--split-k', '1', '--generations', '3', '--runs-dir', tmp, '--start-from', str(seed)])
            async def revise(run_dir, policy, rules, report, model, *, experiments):
                experiments.restore('parent')
                return {'summary': 'Repeated restore supplies no new evidence', 'cost_usd': 0}
            with patch.object(driver.pc, 'resolve_detector_runs', return_value={}), \
                    patch.object(driver, 'ensure_native_tables', side_effect=lambda brain, *a, **kw: fixture(brain)), \
                    patch.object(driver.scoring, 'evaluate', wraps=driver.scoring.evaluate) as evaluate:
                path = asyncio.run(driver.run(args, revise_fn=revise))
            rows = [json.loads(s) for s in (path / 'ledger.jsonl').read_text().splitlines()]
            self.assertEqual(rows[2]['failure_stage'], 'research_check')
            self.assertEqual(rows[2]['submitted_experiment'], 'gen003/attempt001')
            self.assertIsNotNone(rows[2]['train'])
            self.assertIsNone(rows[2].get('validation'))
            self.assertFalse(rows[2]['research_status']['complete'])
            self.assertEqual(len([c for c in evaluate.call_args_list if set(c.args[1]) == {'2'}]), 4)
            self.assertEqual((path / 'best_scorer.py').read_text(), BASELINE)
            plan = (path / 'gen003/search_plan.json').read_text()
            self.assertNotIn('validation', plan.lower())


class VolumeReceiptTests(unittest.TestCase):
    def test_volume_tool_records_numeric_comparison_and_deduplicates_across_generations(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            memory = ExperimentMemory(root)
            session = object.__new__(TrainingExperiments)
            plan = {'search_parent': {'experiment': 'promoted'}, 'followup_required': True,
                    'investigation': {'required_source': 'raw_image'}}
            for name, value in dict(gen_dir=root, train={}, volume_analyses_used=0, timeout=10,
                    classifier_memory_mb=256, classifier_threads=1, max_evaluations=8,
                    evaluations_used=0, repair_granted=False, memory=memory, generation=1,
                    search_plan=plan, target_kind='split', edit_base='promoted', trace=Trajectory(root)).items():
                setattr(session, name, value)
            program = 'def analyze(context):\n    return {"mean": 1.0}\n'
            response = {'status': 'analyzed', 'cases': [
                {'candidate_ref': ref, 'valid_fraction': 1., 'result': {'mean': val}}
                for ref, val in [('a', 1.), ('b', 2.)]]}
            with patch('proofreader_evolve.harness.train_experiments.read_analysis',
                       return_value=(program, {'candidates': ['a', 'b']}, [1, 2])), \
                    patch('proofreader_evolve.harness.train_experiments.execute_analysis', return_value=response):
                session.run_volume_analysis()
                self.assertTrue(session.research_status()['complete'])
                session.generation = 2
                session.run_volume_analysis()
                self.assertFalse(session.research_status()['complete'])
                self.assertEqual(session.research_status()['reused_results'], 1)


class AblationReceiptTests(unittest.TestCase):
    def test_negative_image_ablation_completes_task_and_comment_edit_is_reuse(self):
        from proofreader_evolve.tests.test_feature_discovery import training_fixture, FORMULA
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root = Path(tmp)
            gen = root / 'gen001'
            gen.mkdir()
            train = training_fixture()
            table = train['794495'].tables['merge']
            policy, rules = gen / 'scorer.py', gen / 'rules.md'
            policy.write_text(FORMULA)
            rules.write_text('Synthetic image comparison')
            (gen / 'proposal.json').write_text(json.dumps({'hypothesis': 'Image inputs help.',
                'strategy': 'Paired comparison.', 'family': 'image', 'parameter_grid': {}}))
            memory = ExperimentMemory(root)
            session = TrainingExperiments(gen, policy, rules, 'merge', {'merge': FORMULA}, train, {},
                {'794495/merge': {'scores': table.features.detector_score.to_numpy(), 'chosen': [0, 1, 2]}},
                {'merge': 3}, 30, 0, memory, generation=1, search_plan={
                    'mode': 'explore', 'search_parent': {'experiment': 'promoted'},
                    'followup_required': True, 'investigation': {'required_source': 'raw_image'}})
            table.features.attrs['image_context'] = {'input_sha256': 'same-actual-pixels'}
            def prepare(*args, **kwargs):
                source_hash = 'changed-cache' if policy.read_text().startswith('#') else 'initial-cache'
                return ({'794495': table.features}, ['image_available'], (None, {}),
                        {'program_sha256': source_hash, 'tables': {'794495': {'image_inputs': source_hash}}})
            def measure(*args, charge, **kwargs):
                charge()
                charge()
                return {'status': 'measured', 'delta_precision': -.1,
                        'image_evidence': {'all_image_inputs_removed_in_control': True}}
            with patch('proofreader_evolve.harness.train_experiments.prepare_ablation', side_effect=prepare), \
                    patch('proofreader_evolve.harness.train_experiments.measure_ablation', side_effect=measure):
                first = session.evaluate_feature_ablation()
                self.assertEqual(first['delta_precision'], -.1)
                self.assertTrue(session.research_status()['complete'])
                policy.write_text('# Same method, different extraction cache key\n' + FORMULA)
                session.generation = 2
                session.evaluate_feature_ablation()
                status = session.research_status()
                self.assertFalse(status['complete'])
                self.assertEqual(status['new_measurements'], 0)
                self.assertEqual(status['reused_results'], 1)
                self.assertEqual(memory.research.sources('merge')['compared'], ['raw_image'])


if __name__ == '__main__':
    unittest.main()
