"""Boundary coverage, bounded image extraction and evidence attribution on tiny fixtures."""
import asyncio
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from proofreader_evolve.harness.image_contract import image_spec
from proofreader_evolve.harness.image_selection import select_image_rows, selection_coverage, scoring_coverage
from proofreader_evolve.harness.image_features import augmented_image_features
from proofreader_evolve.harness.image_context import file_hash
from proofreader_evolve.harness.image_evidence import generation_image_evidence
from proofreader_evolve.harness.feature_ablation import measure_ablation
from proofreader_evolve.harness.fixed_pool_scoring import evaluate
from proofreader_evolve.harness.train_feedback import metrics_only
from proofreader_evolve.harness.train_experiments import TrainingExperiments
from proofreader_evolve.harness.evolution_report import build_report_data
from proofreader_evolve.tests.test_feature_discovery import training_fixture


def program(**settings):
    return f"LOCAL_IMAGE = {dict(feature_names=['image_mean'], **settings)!r}\n" + '''
def extract_image_features(context):
    patch = context['patches'][0]
    return {'image_mean': float(patch['image_zyx'][patch['valid_zyx']].mean())}
def score_candidates(features, ctx):
    return features['image_mean'].fillna(features['detector_score']).to_numpy()
'''


class ImageScoringTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.train = training_fixture()
        self.table = self.train['794495'].tables['merge']
        self.paths = {}
        for row in range(len(self.table.features)):
            path = self.root / f'pixels{row}.npz'
            np.savez(path, image_zyx=np.full((3, 4, 5), row + 1, dtype=np.uint16),
                     valid_zyx=np.ones((3, 4, 5), dtype=bool), anchors_zyx=np.array([[1., 2., 3.]]),
                     spacing_zyx_um=np.array([2., 3., 5.]))
            self.paths[row] = path
        self.provider = SimpleNamespace(directory=self.root / 'cache', trace=Mock(), patch=Mock(side_effect=self.get_patch))
        self.table.image_context = self.provider

    def get_patch(self, table, index, spec, occurrence):
        path = self.paths[index]
        return path, {'patch_sha256': file_hash(path), 'occurrences_total': 1}

    def test_boundary_pilot_expands_to_complete_top_k_and_outside(self):
        frame = pd.DataFrame({'detector_score': np.arange(6000, 0, -1)})
        spec = image_spec(program(selection_mode='top_k_boundary', selection_k=2000, max_candidates=256))[1]
        rows = select_image_rows(frame, spec)
        np.testing.assert_array_equal(rows, np.arange(1872, 2128))
        spec['max_candidates'] = 4000
        rows = select_image_rows(frame, spec)
        np.testing.assert_array_equal(rows, np.arange(4000))
        coverage = selection_coverage(frame, spec, rows)
        self.assertEqual((coverage['inside_top_k'], coverage['outside_top_k']), (2000, 2000))
        self.assertEqual(coverage['inside_top_k_fraction'], 1.)

    def test_boundary_backfills_small_pools_and_k_beyond_pool(self):
        frame = pd.DataFrame({'detector_score': np.arange(8, 0, -1)})
        for k in (1, 7, 20):
            spec = image_spec(program(selection_mode='top_k_boundary', selection_k=k, max_candidates=8))[1]
            np.testing.assert_array_equal(select_image_rows(frame, spec), np.arange(8))
        self.assertEqual(len(select_image_rows(frame.iloc[:0], spec)), 0)

    def test_frozen_rule_uses_each_brains_predictors_not_train_row_ids(self):
        train = pd.DataFrame({'detector_score': [9.,8.,7.,6.,5.,4.]})
        validation = train.iloc[::-1].reset_index(drop=True)
        spec = image_spec(program(selection_mode='top_k_boundary', selection_k=2, max_candidates=2))[1]
        a, b = select_image_rows(train, spec), select_image_rows(validation, spec)
        np.testing.assert_array_equal(a, [1,2])
        np.testing.assert_array_equal(b, [4,3])
        np.testing.assert_array_equal(train.detector_score.iloc[a], validation.detector_score.iloc[b])

    def test_legacy_top_ties_keep_original_row_order(self):
        frame=pd.DataFrame({'detector_score':[1.,1.,1.,1.]})
        spec=image_spec(program(max_candidates=2))[1]
        rows=select_image_rows(frame,spec,['z','y','b','a'])
        np.testing.assert_array_equal(rows,[0,1])
        coverage=selection_coverage(frame,spec,rows,['z','y','b','a'])
        self.assertEqual((coverage['rank_min'],coverage['rank_max']),(1,2))

    def test_keys_resolve_boundary_ties_and_nonfinite_values_rank_last(self):
        frame = pd.DataFrame({'detector_score': [1., 1., 1., np.nan, np.inf, 0.]})
        spec = image_spec(program(selection_mode='top_k_boundary', selection_k=2, max_candidates=2))[1]
        np.testing.assert_array_equal(select_image_rows(frame, spec, ['c','a','b','z','y','x']), [2, 0])
        spec['selection_largest'] = False
        np.testing.assert_array_equal(select_image_rows(frame, spec, ['c','a','b','z','y','x']), [1, 2])

    def test_invalid_selector_and_resource_limits_fail_before_io(self):
        for settings in ({'selection_mode':'other'}, {'selection_mode':'top_k_boundary'},
                         {'selection_mode':'top_k_boundary','selection_k':True}, {'selection_k':4},
                         {'batch_size':0}, {'batch_max_mb':1025}, {'max_total_mb':8193}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                image_spec(program(**settings))
        self.provider.patch.assert_not_called()

    def test_multiple_real_workers_preserve_row_alignment_and_numeric_scores(self):
        source = program(selection_mode='top_k_boundary', selection_k=6, max_candidates=6, batch_size=2)
        frame = augmented_image_features(source, self.table, self.table.features, timeout=60)
        self.assertEqual(frame.attrs['image_context']['extraction_batches'], 3)
        np.testing.assert_array_equal(frame.image_available, [0,0,0,1,1,1,1,1,1,0,0,0])
        np.testing.assert_array_equal(frame.image_mean.iloc[3:9], np.arange(4,10))
        # Reuse the very same frozen program on another brain: only host inputs differ.
        other = training_fixture(); other_table = other['794495'].tables['merge']
        other_table.image_context = self.provider
        other_table.truth = 1 - other_table.truth
        with patch('proofreader_evolve.harness.image_features.run_worker') as worker:
            again = augmented_image_features(source, other_table, other_table.features, timeout=60)
        worker.assert_not_called()
        np.testing.assert_array_equal(again.image_mean, frame.image_mean)
        self.assertTrue(again.attrs['image_context']['cached_features'])
        report = evaluate(source, self.train, {'merge':4}, timeout=60)
        coverage = report['cells']['794495/merge']['image_context']['scoring_coverage']
        self.assertEqual(coverage['top_k_with_images'], 4)
        self.assertEqual(coverage['top_k_hits_with_images'], 2)
        self.assertEqual(len(frame), len(self.table.truth))
        self.assertIn('image_context', metrics_only(report)['cells']['794495/merge'])

    def fake_worker(self, root, request, **kwargs):
        values = np.load(root / 'X.npy', allow_pickle=False)
        np.save(root / 'local_features.npy', values[:, :1])

    def test_numeric_batches_can_exceed_aggregate_worker_limit_but_raw_models_cannot(self):
        # Scale the transport cap down to avoid large test allocations.
        limit = self.paths[0].stat().st_size * 2
        with patch('proofreader_evolve.harness.image_features.MAX_STAGE_BYTES', limit), \
             patch('proofreader_evolve.harness.image_features.run_worker', side_effect=self.fake_worker):
            frame = augmented_image_features(program(max_candidates=8, batch_size=32), self.table, self.table.features)
            self.assertGreater(frame.attrs['image_context']['bytes'], limit)
            self.assertEqual(frame.attrs['image_context']['extraction_batches'], 4)
            raw = "LOCAL_IMAGE = {'raw_patches': True, 'max_candidates': 8}\ndef fit(*args): pass\ndef predict(*args): pass\n"
            with self.assertRaisesRegex(ValueError, 'raw-patch limit'):
                augmented_image_features(raw, self.table, self.table.features)

    def test_failed_batch_is_not_published_and_completed_batches_can_resume(self):
        source = program(max_candidates=6, batch_size=2)
        calls = []
        def flaky(root, request, **kwargs):
            calls.append(request)
            if len(calls) == 2:
                raise RuntimeError('synthetic worker failure')
            self.fake_worker(root, request)
        with patch('proofreader_evolve.harness.image_features.run_worker', side_effect=flaky):
            with self.assertRaisesRegex(RuntimeError, 'synthetic'):
                augmented_image_features(source, self.table, self.table.features)
        self.assertNotIn('image_available', self.table.features)
        with patch('proofreader_evolve.harness.image_features.run_worker', side_effect=self.fake_worker) as worker:
            frame = augmented_image_features(source, self.table, self.table.features)
        self.assertEqual(worker.call_count, 2)
        self.assertEqual(frame.attrs['image_context']['cached_batches'], 1)
        np.testing.assert_allclose(frame.image_mean.iloc[:6], frame.detector_score.iloc[:6])

    def test_last_patch_read_counts_against_shared_timeout(self):
        now = [0.]
        def slow(*args):
            now[0] += 2.
            return self.get_patch(*args)
        self.provider.patch.side_effect = slow
        with patch('proofreader_evolve.harness.image_features.time.monotonic', side_effect=lambda: now[0]), \
             patch('proofreader_evolve.harness.image_features.run_worker') as worker:
            with self.assertRaisesRegex(TimeoutError, 'time budget'):
                augmented_image_features(program(max_candidates=1), self.table, self.table.features, timeout=1)
        worker.assert_not_called()

    def test_total_bytes_and_single_row_batch_limits_fail_explicitly(self):
        large = self.root / 'large.npz'
        np.savez(large, image_zyx=np.ones((64,64,32),dtype=np.float32), valid_zyx=np.ones((64,64,32),dtype=bool))
        self.provider.patch.side_effect = None
        self.provider.patch.return_value = (large, {'patch_sha256':file_hash(large),'occurrences_total':1})
        with self.assertRaisesRegex(ValueError, 'max_total_mb'):
            augmented_image_features(program(max_candidates=2,max_total_mb=1), self.table, self.table.features)
        with patch('proofreader_evolve.harness.image_features.MAX_STAGE_BYTES', 100):
            with self.assertRaisesRegex(ValueError, 'One candidate'):
                augmented_image_features(program(max_candidates=1), self.table, self.table.features)

    def test_plan_reports_pilot_and_large_coverage_without_reading_images(self):
        exp = object.__new__(TrainingExperiments)
        exp.gen_dir, exp.train, exp.target_kind, exp.budgets = self.root, self.train, 'merge', {'merge':6}
        exp.trace = Mock(); exp.proposal_path = self.root/'proposal.json'
        exp.proposal_path.write_text('{}')
        exp._read_candidate = Mock(return_value=(program(selection_mode='top_k_boundary', selection_k=6, max_candidates=4), ''))
        result = exp.plan_image_scoring()
        self.assertEqual(result['cells']['794495/merge']['outside_top_k'], 2)
        self.assertTrue(result['warnings']); self.assertEqual(result['evaluations_used'], 0)
        self.provider.patch.assert_not_called()
        self.assertNotIn('positives', json.dumps(result))

    def test_plan_tool_is_registered_callable_and_explicitly_allowed(self):
        import claude_agent_sdk as sdk
        from proofreader_evolve.harness.reviser_session import bind_session_options
        exp = object.__new__(TrainingExperiments)
        exp.gen_dir=self.root/'gen001'; exp.gen_dir.mkdir()
        exp.train,exp.target_kind,exp.budgets=self.train,'merge',{'merge':6}
        exp.trace=Mock(); exp.lock=asyncio.Lock()
        exp.research_status=Mock(return_value={'status': 'complete', 'complete': True})
        exp.policy_path=exp.gen_dir/'scorer.py'; exp.rules_path=exp.gen_dir/'rules.md'
        exp.proposal_path=exp.gen_dir/'proposal.json'
        exp.policy_path.write_text(program(selection_mode='top_k_boundary',selection_k=6,max_candidates=4))
        exp.rules_path.write_text('Image hypothesis'); exp.proposal_path.write_text('{}')
        exp.max_evaluations,exp.evaluations_used,exp.repair_granted=8,0,False
        with patch.object(sdk,'create_sdk_mcp_server') as create:
            exp.mcp_server()
        handlers={tool.name:tool.handler for tool in create.call_args.kwargs['tools']}
        result=asyncio.run(handlers['plan_image_scoring']({}))
        self.assertEqual(json.loads(result['content'][0]['text'])['status'],'planned')
        self.assertTrue(json.loads(result['content'][0]['text'])['research_status']['complete'])
        plan=exp.gen_dir/'image_scoring_plan.json'
        options=bind_session_options(sdk.ClaudeAgentOptions(cwd=str(self.root)),
            exp.policy_path,exp.rules_path,exp.gen_dir/'feedback.json',
            readable_paths=[plan],training_server={'type':'sdk'})
        self.assertEqual(asyncio.run(options.can_use_tool('mcp__training__plan_image_scoring',{},None)).behavior,'allow')
        self.assertEqual(asyncio.run(options.can_use_tool('Read',{'file_path':str(plan)},None)).behavior,'allow')
        self.assertEqual(asyncio.run(options.can_use_tool('Write',{'file_path':str(plan)},None)).behavior,'deny')
        self.assertEqual(exp.remaining,8)
        self.provider.patch.assert_not_called()

    def test_ablation_distinguishes_image_only_from_joint_feature_removal(self):
        frame = self.table.features.copy()
        frame['image_available'] = 1.
        frame['image_mean'] = self.table.truth.astype(float)
        frame.attrs['image_context'] = {'selection_coverage': {'selected_rows':len(frame)}}
        def scorer(source, values, kind, timeout):
            return values.image_mean.to_numpy() + .01 * values.detector_score.to_numpy()
        with patch('proofreader_evolve.harness.feature_ablation.score', side_effect=scorer):
            for columns, isolated in [(['image_mean','image_available'],True),
                                      (['image_mean'],False),
                                      (['image_mean','image_available','local_signal'],False)]:
                result = measure_ablation(self.train, 'merge', program(), {}, {'794495':frame}, columns,
                    (None,None), {'program_sha256':'program','parameters_sha256':'params'}, {'merge':4},
                    self.root/f'ablation{len(columns)}', classifier=False, charge=lambda:None)
                self.assertEqual(result['image_evidence']['image_only_control'], isolated)
                self.assertEqual(result['image_evidence']['conclusion'],
                                 'positive_image_increment' if isolated else 'image_increment_not_isolated')
                self.assertEqual(result['cells']['794495/merge']['image_coverage']['full']['top_k_with_images'], 4)

    def test_generation_and_report_keep_access_separate_from_gain(self):
        directory = self.root/'gen001'; directory.mkdir()
        analysis = directory/'volume_analyses'/'analysis001'; analysis.mkdir(parents=True)
        (analysis/'result.json').write_text('{"status":"analyzed"}')
        frame=self.table.features.copy(); frame['image_available']=0.; frame.loc[[0,7],'image_available']=1.
        coverage=scoring_coverage(frame,np.array([0,1,2]),self.table.truth,3)
        self.assertEqual(coverage['top_k_with_images'],1)
        self.assertEqual(coverage['top_k_hits_with_images'],0)
        cell={'image_context':{'selected_rows':2,'scoring_coverage':coverage}}
        evidence=generation_image_evidence(directory, {'experiments':[{'experiment':'gen001/attempt001',
            'status':'evaluated','target_kind':'merge','train':{'cells':{'794495/merge':cell}}}],
            'feature_diagnostics':[]})
        self.assertEqual(evidence['volume_analyses_succeeded'],1)
        self.assertEqual(len(evidence['image_scored_attempts']),1)
        self.assertEqual(evidence['paired_diagnostics'],[])
        (self.root/'manifest.json').write_text('{"objective":"native-precision-at-k-v1"}')
        (directory/'evaluation.json').write_text(json.dumps({'generation':1,'accepted':False,
            'train':{'macro_precision':.1,'cells':{'794495/merge':cell}},'image_evidence':evidence}))
        report=build_report_data(self.root)
        saved=report['generations'][0]
        self.assertEqual(saved['image_evidence'],evidence)
        self.assertEqual(saved['train']['cells']['794495/merge']['image_context'],cell['image_context'])
        self.assertNotIn('events',saved)
