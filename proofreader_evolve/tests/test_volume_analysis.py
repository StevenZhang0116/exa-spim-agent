"""Direct 3D exploration tests; synthetic volumes, no fitting or live LLM."""
import asyncio
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from proofreader_evolve.harness.image_context import file_hash
from proofreader_evolve.harness.local_context import candidate_ref
from proofreader_evolve.harness.reviser_access import make_guard
from proofreader_evolve.harness.train_experiments import TrainingExperiments
from proofreader_evolve.harness.trajectory import Trajectory
from proofreader_evolve.harness.volume_analysis import read_analysis, MAX_ANALYSES
from proofreader_evolve.tests.test_feature_discovery import training_fixture


class VolumeAnalysisTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.gen = self.root / 'gen001'
        self.gen.mkdir()
        self.train = training_fixture('split')
        self.table = self.train['794495'].tables['split']
        self.ref = candidate_ref(self.table, 0)
        self.request = {'candidates': [{'candidate_ref': self.ref, 'occurrence_index': 0}],
                        'radius_um': 40., 'level': 0, 'max_nodes': 256}
        z, y, x = np.indices((7, 8, 9))
        self.volume = (100*z + 10*y + x).astype(np.uint16)
        self.anchors = np.array([[2., 8./3., 2.], [3., 11./3., 4.]])
        self.patch = self.root / 'pixels.npz'
        np.savez(self.patch, image_zyx=self.volume, valid_zyx=np.ones_like(self.volume, dtype=bool),
                 anchors_zyx=self.anchors, spacing_zyx_um=np.array([6.,3.,2.]))
        self.metadata = {'source_axes':['t','c','z','y','x'], 'source_shape':[1,1,100,100,100],
            'spacing_xyz_um':[2.,3.,6.], 'translation_xyz_um':[10.,20.,30.],
            'graph_to_world_um':np.eye(4).tolist(), 'dataset':'0', 'origin_zyx':[48,64,48],
            'shape_zyx':[7,8,9], 'valid_fraction':1., 'patch_sha256':file_hash(self.patch),
            'source_uri':'host-only-source', 'occurrences_total':1}
        site = {'xyz_um':[[-2.,-1.5,-3.],[2.,1.5,3.]], 'radius_um':[1.,None], 'degree':[1,1],
                'segment':[0,1], 'edges':[], 'anchor_nodes':[0,1], 'outside_radius':[False,False],
                'truncated':False, 'gt_private':'must not be transported'}
        self.table.local_context = SimpleNamespace(context=Mock(return_value=(
            {'sites':[site], 'occurrences_total':1}, [[112.,221.5,333.]])))
        self.table.image_context = SimpleNamespace(patch=Mock(return_value=(self.patch,self.metadata)))
        self.experiments = object.__new__(TrainingExperiments)
        for name, value in {'gen_dir':self.gen, 'train':self.train, 'volume_analyses_used':0,
                           'timeout':20., 'classifier_memory_mb':2048, 'classifier_threads':1,
                           'max_evaluations':8, 'evaluations_used':0, 'repair_granted':False,
                           'trace':Trajectory(self.gen)}.items():
            setattr(self.experiments, name, value)
        self.write('def analyze(context):\n    return {"voxel": int(context["image_zyx"][2,3,4])}\n')

    def write(self, program, request=None):
        (self.gen / 'analysis.py').write_text(program)
        (self.gen / 'analysis_request.json').write_text(json.dumps(self.request if request is None else request))

    def test_actual_3d_worker_maps_fragments_and_blocks_external_reads(self):
        forbidden = self.root / 'heldout.secret'
        forbidden.write_text('not a worker input')
        self.write(f'''import numpy as np
def analyze(context):
    assert context['image_zyx'].ndim == 3
    assert not context['image_zyx'].flags.writeable
    assert not context['fragment']['nodes_zyx'].flags.writeable
    assert context['fragment']['edges'].shape == (0, 2)
    np.testing.assert_allclose(context['fragment']['nodes_zyx'], context['anchors_zyx'])
    assert 'gt_private' not in context['fragment']
    assert not {{'brain', 'truth', 'label', 'image_uri', 'candidate_ref'}} & context.keys()
    try:
        open({str(forbidden)!r}).read()
    except PermissionError:
        blocked = True
    else:
        raise AssertionError('External file was exposed')
    return {{'voxel': np.int64(context['image_zyx'][2,3,4]),
             'spacing': context['spacing_zyx_um'], 'blocked': blocked,
             'anchors': context['fragment']['nodes_zyx']}}
''')
        result = self.experiments.run_volume_analysis()
        self.assertEqual(result['status'], 'analyzed', result)
        case = result['cases'][0]
        self.assertEqual(case['result']['voxel'], 234)
        self.assertEqual(case['result']['spacing'], [6.,3.,2.])
        self.assertTrue(case['result']['blocked'])
        np.testing.assert_allclose(case['result']['anchors'], self.anchors)
        self.assertNotIn('host-only-source', json.dumps(result))
        saved = json.loads((self.gen/'volume_analyses/analysis001/input_manifest.json').read_text())
        self.assertEqual(saved['inputs'][0]['image_metadata']['source_uri'], 'host-only-source')
        self.assertEqual(result['evaluations_remaining'],8)
        self.assertEqual(result['volume_analyses_remaining'],7)

    def test_batch_keeps_requested_order_and_reruns_current_code(self):
        self.request['candidates'] = [{'candidate_ref':candidate_ref(self.table,i)} for i in [4,1]]
        self.write('def analyze(context):\n    return {"base": context["features"]["detector_score"]}\n')
        first = self.experiments.run_volume_analysis()
        self.assertEqual(first['status'],'analyzed',first)
        self.assertEqual([c['candidate_ref'] for c in first['cases']],
                         [c['candidate_ref'] for c in self.request['candidates']])
        np.testing.assert_allclose([c['result']['base'] for c in first['cases']],
                                   self.table.features.detector_score.iloc[[4,1]])
        self.write('def analyze(context):\n    return {"changed": 123}\n')
        second = self.experiments.run_volume_analysis()
        self.assertEqual(second['status'],'analyzed',second)
        self.assertEqual(second['cases'][0]['result']['changed'],123)
        self.assertNotEqual(first['program_sha256'],second['program_sha256'])
        self.assertIn('"base"', (self.gen/'volume_analyses/analysis001/analysis.py').read_text())
        self.assertIn('"changed"', (self.gen/'volume_analyses/analysis002/analysis.py').read_text())

    def test_all_handles_resolved_before_io_including_valid_shaped_heldout(self):
        held = deepcopy(self.table)
        held.meta['provenance']['brain'] = '794491'
        self.request['candidates'].append({'candidate_ref':candidate_ref(held,0)})
        self.write('def analyze(context):\n    return {}\n')
        with self.assertRaisesRegex(ValueError,'absent'):
            self.experiments.run_volume_analysis()
        self.table.image_context.patch.assert_not_called()
        self.table.local_context.context.assert_not_called()
        self.assertEqual(self.experiments.volume_analyses_used,0)

    def test_request_limits_syntax_and_paths_fail_without_spending_budget(self):
        invalid = [dict(self.request, image_uri='s3://other'), dict(self.request, radius_um=101),
                   dict(self.request, level=True), dict(self.request, max_nodes=513),
                   {'candidates':self.request['candidates']*5},
                   {'candidates':[{'candidate_ref':self.ref,'occurrence_index':-1}]}]
        for request in invalid:
            self.write('def analyze(context):\n    return {}\n',request)
            with self.assertRaises(ValueError):
                self.experiments.run_volume_analysis()
        self.write('def analyze(:\n    pass\n')
        with self.assertRaises(SyntaxError):
            self.experiments.run_volume_analysis()
        self.assertEqual(self.experiments.volume_analyses_used,0)
        self.table.image_context.patch.assert_not_called()

    def test_misaligned_anchor_rejects_input_before_execution(self):
        self.metadata['origin_zyx'][0] += 1
        result = self.experiments.run_volume_analysis()
        self.assertEqual(result['status'],'error')
        self.assertIn('anchors disagree', result['error'])
        self.assertEqual(self.experiments.volume_analyses_used,1)
        self.assertEqual(self.experiments.evaluations_used,0)
        self.assertTrue((self.gen/'volume_analyses/analysis001/result.json').is_file())

    def test_nonfinite_and_oversized_outputs_fail_without_fabricated_results(self):
        for code in ('return {"bad": float("nan")}', 'return {"large": "x" * 30000}'):
            self.write('def analyze(context):\n    '+code+'\n')
            result = self.experiments.run_volume_analysis()
            self.assertEqual(result['status'],'error',result)
            self.assertNotIn('cases',result)
        self.assertEqual(self.experiments.volume_analyses_used,2)

    def test_worker_timeout_is_logged_and_counted(self):
        self.experiments.timeout = .25
        self.write('def analyze(context):\n    while True:\n        pass\n')
        result = self.experiments.run_volume_analysis()
        self.assertEqual(result['status'],'error')
        self.assertIn('exceeded',result['error'])
        self.assertEqual(result['volume_analyses_remaining'],7)

    def test_execution_budget_is_independent_and_no_scorer_is_modified(self):
        (self.gen/'scorer.py').write_text('unchanged policy')
        (self.gen/'training.py').write_text('unchanged model')
        self.experiments.image_inspections_used = 8
        with patch('proofreader_evolve.harness.train_experiments.execute_analysis',
                   return_value={'status':'analyzed','cases':[]}):
            for _ in range(MAX_ANALYSES):
                self.assertEqual(self.experiments.run_volume_analysis()['status'],'analyzed')
            with self.assertRaisesRegex(ValueError,'executions'):
                self.experiments.run_volume_analysis()
        self.assertEqual(self.experiments.remaining,8)
        self.assertEqual((self.gen/'scorer.py').read_text(),'unchanged policy')
        self.assertEqual((self.gen/'training.py').read_text(),'unchanged model')

    def test_analysis_files_and_tool_allowed_only_in_current_session_scope(self):
        guard,_ = make_guard(self.root,self.gen,[],self.root/'audit.jsonl',allow_proposal=True,
                            allowed_tools=['mcp__training__run_volume_analysis'])
        for name in ('analysis.py','analysis_request.json'):
            for tool in ('Read','Write','Edit'):
                decision=asyncio.run(guard(tool,{'file_path':str(self.gen/name)},None))
                self.assertEqual(decision.behavior,'allow')
            decision=asyncio.run(guard('Write',{'file_path':str(self.root/'gen002'/name)},None))
            self.assertEqual(decision.behavior,'deny')
        for name in ('image_alignment.json','volume_analyses/analysis001/input_manifest.json','validation.json'):
            decision=asyncio.run(guard('Read',{'file_path':str(self.gen/name)},None))
            self.assertEqual(decision.behavior,'deny')
        (self.gen/'analysis.py').unlink()
        (self.gen/'analysis.py').symlink_to(self.root/'external.py')
        self.assertEqual(asyncio.run(guard('Write',{'file_path':str(self.gen/'analysis.py')},None)).behavior,'deny')
        with self.assertRaises(ValueError):
            read_analysis(self.gen,self.train)

    def test_sdk_tool_returns_text_results_without_requiring_preview(self):
        import mcp.types as types
        self.experiments.lock=asyncio.Lock()
        self.experiments.research_status = Mock(return_value={'status': 'complete', 'complete': True})
        server=self.experiments.mcp_server()['instance']
        async def invoke():
            request=types.CallToolRequest(method='tools/call',params=types.CallToolRequestParams(
                name='run_volume_analysis',arguments={}))
            return await server.request_handlers[types.CallToolRequest](request)
        with patch('proofreader_evolve.harness.train_experiments.execute_analysis',
                   return_value={'status':'analyzed','cases':[{'result':{'value':3}}]}):
            result=asyncio.run(invoke()).root
        self.assertFalse(result.isError)
        self.assertEqual([block.type for block in result.content],['text'])
        self.assertEqual(json.loads(result.content[0].text)['cases'][0]['result']['value'],3)
        self.assertTrue(json.loads(result.content[0].text)['research_status']['complete'])


if __name__ == '__main__':
    unittest.main()
