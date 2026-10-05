"""Image coordinates, isolated pixel transport and SDK content; no model fitting."""
import asyncio
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from proofreader_evolve.harness.image_coordinates import ImageGeometry
from proofreader_evolve.harness.image_contract import image_spec
from proofreader_evolve.harness.image_features import stage_images, subset_features, augmented_image_features
from proofreader_evolve.harness.image_context import ImageSource, file_hash
from proofreader_evolve.harness.classifier_train_worker import ImageDataset
from proofreader_evolve.harness.classifier_training import fit_classifier
from proofreader_evolve.harness.feature_ablation import measure_ablation
from proofreader_evolve.harness.model_execution import run_worker, predict_model
from proofreader_evolve.harness.classifier_contract import frozen_model
from proofreader_evolve.harness.trajectory import image_safe_content
from proofreader_evolve.harness.train_experiments import TrainingExperiments
from proofreader_evolve.harness.local_context import candidate_ref
from proofreader_evolve.tests.test_feature_discovery import training_fixture

RAW_PROGRAM = """LOCAL_IMAGE = {'raw_patches': True}
def fit(X_train, y_train, artifact_dir, params, *, images):
    raise AssertionError('This test must not train a model')
def predict(X, artifact_dir, params, *, images):
    return [float(p[0]['image_zyx'].mean()) if p else 0. for p in images]
"""
EXTRACT_PROGRAM = """LOCAL_IMAGE = {'max_candidates': 2, 'feature_names': ['image_mean']}
def extract_image_features(context):
    p = context['patches'][0]
    return {'image_mean': float(p['image_zyx'][p['valid_zyx']].mean())}
"""


def ngff(axes='tczyx', units='micrometer'):
    scales = {'t': 1, 'c': 1, 'z': 2, 'y': 3, 'x': 5}
    offsets = {'t': 0, 'c': 0, 'z': 7, 'y': 11, 'x': 13}
    return {'multiscales': [{'axes': [{'name': a, 'type': 'space' if a in 'xyz' else a,
        **({'unit': units} if a in 'xyz' else {})} for a in axes], 'datasets': [
        {'path': '0', 'coordinateTransformations': [
            {'type': 'scale', 'scale': [scales[a] for a in axes]},
            {'type': 'translation', 'translation': [offsets[a] for a in axes]}]}]}]}


class ImageContextTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.pixels = self.root / 'patch.npz'
        np.savez(self.pixels, image_zyx=np.full((3, 4, 5), 17, dtype=np.uint16),
                 valid_zyx=np.ones((3, 4, 5), dtype=bool), anchors_zyx=np.array([[1., 2., 3.]]),
                 spacing_zyx_um=np.array([2., 3., 5.]))

    def test_units_transform_order_and_roundtrip(self):
        attrs = ngff(units='nanometer')
        attrs['multiscales'][0]['coordinateTransformations'] = [
            {'type': 'scale', 'scale': [1, 1, 2, 2, 2]},
            {'type': 'translation', 'translation': [0, 0, 100, 200, 300]}]
        geometry = ImageGeometry.from_ngff(attrs, {'shape': [1, 1, 8, 9, 10]}, '0')
        np.testing.assert_allclose(geometry.scale_xyz_um, [.010, .006, .004])
        np.testing.assert_allclose(geometry.translation_xyz_um, [.326, .222, .114])
        point = np.array([[.356, .234, .118]])
        np.testing.assert_allclose(geometry.graph_to_voxel(point), [[1, 2, 3]])
        np.testing.assert_allclose(geometry.voxel_to_graph([[1, 2, 3]]), point)
        matrix = np.eye(4); matrix[:3, 3] = [1, 2, 3]
        geometry = ImageGeometry.from_ngff(attrs, {'shape': [1, 1, 8, 9, 10]}, '0', matrix)
        np.testing.assert_allclose(geometry.voxel_to_graph(geometry.graph_to_voxel(point)), point)

    def test_axis_permutation_and_boundary_padding_preserve_phantom_pixels(self):
        # Deliberately asymmetric dimensions and a nonstandard storage order.
        shape = {'t': 1, 'c': 2, 'z': 8, 'y': 9, 'x': 10}
        axes = 'xctzy'
        geometry = ImageGeometry.from_ngff(ngff(axes), {'shape': [shape[a] for a in axes]}, '0')
        zz, yy, xx = np.indices((8, 9, 10))
        phantom = 100 * zz + 10 * yy + xx
        canonical = np.stack([phantom, phantom + 1000])[None]
        stored = canonical.transpose(['tczyx'.index(a) for a in axes])
        select, permutation, dest = geometry.selection(np.array([-1, 2, 3]), np.array([4, 3, 2]), 1, 0)
        block = stored[select].transpose(permutation)
        padded = np.zeros((4, 3, 2), dtype=int); valid = np.zeros_like(padded, dtype=bool)
        padded[dest], valid[dest] = block, True
        np.testing.assert_array_equal(padded[1:], phantom[:3, 2:5, 3:5] + 1000)
        self.assertFalse(valid[0].any()); self.assertTrue(valid[1:].all())
        # Exercise the actual reader's selection/transpose/padding with in-memory IO.
        source = object.__new__(ImageSource)
        source._arrays = {}
        class Array:
            shape = geometry.shape
            domain = SimpleNamespace(inclusive_min=[0]*5)
            def __getitem__(self, selection):
                return SimpleNamespace(read=lambda: SimpleNamespace(result=lambda **kw: stored[selection]))
        source._arrays[0] = Array(); source.timeout = 1
        source.level = lambda _: geometry
        anchors = geometry.voxel_to_graph([[.5, 3, 3.5]])
        with patch.object(geometry, 'crop', return_value=(np.array([-1,2,3]), np.array([4,3,2]), np.array([[1.5,1.,.5]]))):
            result, metadata = source.read(anchors, channel=1)
        np.testing.assert_array_equal(result['image_zyx'], padded)
        np.testing.assert_array_equal(result['valid_zyx'], valid)
        self.assertEqual(metadata['array_axes'], 'zyx')

    def test_crop_includes_both_anchors_and_preserves_physical_spacing(self):
        geometry = ImageGeometry.from_ngff(ngff('zyx'), {'shape': [1000]*3}, '0')
        origin, shape, anchors = geometry.crop(np.array([[100, 200, 300], [240, 200, 300]]), 40)
        self.assertTrue((anchors >= 0).all()); self.assertTrue((anchors < shape).all())
        self.assertTrue((shape * geometry.scale_xyz_um[::-1] >= 80).all())
        np.testing.assert_allclose(geometry.voxel_to_graph(anchors + origin), [[100,200,300], [240,200,300]])
        with self.assertRaisesRegex(ValueError, 'coarser'):
            geometry.crop(np.array([[0,0,0],[2000,0,0]]), 40)

    def test_ambiguous_axes_and_invalid_contracts_fail(self):
        attrs = ngff(); del attrs['multiscales'][0]['axes'][2]['unit']
        with self.assertRaisesRegex(ValueError, 'units'):
            ImageGeometry.from_ngff(attrs, {'shape': [1,1,8,9,10]}, '0')
        for declaration in ["{'image_uri': 's3://example'}", "{'raw_patches': True}",
                            "{'feature_names': ['image_available']}"]:
            with self.assertRaises(ValueError):
                image_spec('LOCAL_IMAGE = ' + declaration)
        self.assertTrue(image_spec(RAW_PROGRAM)[1]['raw_patches'])

    def test_row_subsets_and_missing_preflight_row_keep_images_aligned(self):
        frame = pd.DataFrame({'detector_score': [0,1,2,3], 'image_available': [1,0,1,1]})
        frame.attrs['_image_rows'] = {0: [str(self.pixels)], 2: [str(self.pixels)], 3: [str(self.pixels)]}
        selected = subset_features(frame, [3,1,0], missing_row=True)
        self.assertEqual(set(selected.attrs['_image_rows']), {0,2})
        self.assertEqual(selected.image_available.iloc[-1], 0)
        directory = self.root / 'stage'; directory.mkdir()
        info = stage_images(directory, [(selected, [2,1])])
        self.assertEqual(info['rows_with_images'], 1)
        dataset = ImageDataset(directory, np)
        self.assertEqual(len(dataset), 2); self.assertEqual(dataset[1], [])
        self.assertEqual(int(dataset[0][0]['image_zyx'][0,0,0]), 17)
        self.assertFalse(dataset[0][0]['image_zyx'].flags.writeable)
        self.assertEqual(sorted(p.name for p in (directory/'images').glob('*.npz')), ['0_0.npz'])
        self.assertEqual(set(frame.attrs['_image_rows']), {0,2,3})

    def test_fit_transports_only_fit_images_and_hashes_their_contents(self):
        train = training_fixture(); table = train['794495'].tables['merge']
        frame = table.features.copy(); frame.attrs['_image_rows'] = {i: [str(self.pixels)] for i in range(len(frame))}
        fitting = {'794495': np.array([2,7])}
        def worker(root, request, **kwargs):
            self.assertTrue(request['image_inputs'])
            np.testing.assert_array_equal(np.load(root/'y.npy'), table.truth[[2,7]])
            self.assertEqual(len(ImageDataset(root, np)), 2)
            self.assertEqual(len(list((root/'images').glob('*.npz'))), 2)
            (root/'artifacts'/'state.bin').write_bytes(b'no model training')
            return {'wall_seconds': 0.}
        fingerprints = []
        for iteration in range(2):
            with patch('proofreader_evolve.harness.classifier_training.run_worker', side_effect=worker):
                component, summary = fit_classifier(train, 'merge', {'parameters': {}}, self.root/f'fit{iteration}',
                    program=RAW_PROGRAM, artifact_store=self.root/'models', feature_frames={'794495': frame}, fit_rows=fitting)
            fingerprints.append(summary['training_fingerprint'])
            # Fit remains mocked; exercise real frozen prediction with pixel inputs.
            selected = subset_features(frame, [7,2], missing_row=True)
            predictions = predict_model(frozen_model(component), self.root/'models', selected, 'merge', timeout=30)
            expected = 17. if iteration == 0 else 99.
            np.testing.assert_array_equal(predictions, [expected, expected, 0.])
            np.savez(self.pixels, image_zyx=np.full((3,4,5), 99), valid_zyx=np.ones((3,4,5), dtype=bool))
        self.assertNotEqual(*fingerprints)

    def test_real_isolated_worker_reads_pixels_but_cannot_open_external_file(self):
        directory = self.root/'worker'; directory.mkdir(); (directory/'artifacts').mkdir()
        frame = pd.DataFrame({'detector_score': [.4,.7]}); frame.attrs['_image_rows'] = {0: [str(self.pixels)],1:[str(self.pixels)]}
        stage_images(directory, [(frame, [1,0])]); np.save(directory/'X.npy', frame.to_numpy())
        forbidden = self.root/'heldout.secret'; forbidden.write_text('never exposed')
        program = EXTRACT_PROGRAM + f"\ntry:\n    open({str(forbidden)!r}).read()\nexcept PermissionError:\n    pass\nelse:\n    raise AssertionError('External file became readable')\n"
        (directory/'training.py').write_text(program)
        request = {'mode':'extract_image','columns':list(frame.columns),'feature_names':['image_mean'],
                   'rows':2,'image_inputs':True,'config':{'parameters':{}},'timeout':30,'memory_mb':2048,'threads':1}
        run_worker(directory, request)
        np.testing.assert_array_equal(np.load(directory/'local_features.npy'), [[17],[17]])

    def test_image_feature_selection_keeps_full_pool_and_reuses_extraction(self):
        frame = pd.DataFrame({'detector_score': [.1,.9,.2,.8]})
        provider = SimpleNamespace(directory=self.root/'cache', trace=None,
            patch=Mock(return_value=(self.pixels, {'patch_sha256':file_hash(self.pixels),'occurrences_total':1})))
        table = SimpleNamespace(features=frame, image_context=provider, kind='merge',
                                meta={'pool_sha256':'pool','files':{'features.pkl':'hash'}})
        def worker(root, request, **kwargs):
            np.save(root/'local_features.npy', [[17.],[17.]])
        with patch('proofreader_evolve.harness.image_features.run_worker', side_effect=worker) as worker:
            first = augmented_image_features(EXTRACT_PROGRAM, table, frame)
            second = augmented_image_features(EXTRACT_PROGRAM, table, frame)
        self.assertEqual(worker.call_count,1); self.assertEqual(len(first),4)
        np.testing.assert_array_equal(first.image_available,[0,1,0,1])
        self.assertTrue(first.image_mean.iloc[[0,2]].isna().all())
        pd.testing.assert_frame_equal(first,second)

    def test_ablation_removes_raw_patches_only_in_control(self):
        train=training_fixture(); table=train['794495'].tables['merge']
        frame=table.features.copy(); frame['image_available']=1.
        frame.attrs['_image_rows']={0:[str(self.pixels)]}
        seen=[]
        def score(source, values, kind, timeout):
            seen.append(deepcopy(values.attrs['_image_rows']))
            return values.detector_score.to_numpy()
        with patch('proofreader_evolve.harness.feature_ablation.score',side_effect=score):
            measure_ablation(train,'merge','formula',{}, {'794495':frame}, ['image_available'], (None,None),
                {'program_sha256':'p','parameters_sha256':'q'}, {'merge':3}, self.root/'ablation',
                classifier=False,charge=lambda:None)
        self.assertTrue(seen[0]); self.assertFalse(seen[1]); self.assertTrue(frame.attrs['_image_rows'])

    def test_train_only_image_handle_and_budget(self):
        experiments=object.__new__(TrainingExperiments)
        experiments.image_inspections_used=0; experiments.train=training_fixture()
        with self.assertRaises(ValueError):
            experiments.inspect_candidate_image('heldout/merge/0')
        self.assertEqual(experiments.image_inspections_used,0)
        experiments.image_inspections_used=8
        with self.assertRaisesRegex(ValueError,'budget'):
            experiments.inspect_candidate_image('anything')

    def test_candidate_image_tool_returns_rendered_pixels_and_host_audit(self):
        experiments=object.__new__(TrainingExperiments)
        experiments.image_inspections_used=0; experiments.train=training_fixture()
        experiments.gen_dir=self.root; experiments.trace=Mock()
        table=experiments.train['794495'].tables['merge']
        table.image_context=SimpleNamespace(patch=Mock(return_value=(self.pixels, {
            'shape_zyx':[3,4,5], 'spacing_xyz_um':[5.,3.,2.], 'valid_fraction':1.,
            'source_path':'host-only-image-uri'})))
        with patch('proofreader_evolve.harness.image_preview.fragment_patch_geometry',
                   return_value=(np.array([[0.,0.,0.],[2.,3.,4.]]), [[0,1]], [0,0])):
            result=experiments.inspect_candidate_image(candidate_ref(table,0))
        self.assertEqual(result['content'][1]['type'],'image')
        self.assertGreater(len(result['content'][1]['data']),100)
        self.assertNotIn('host-only-image-uri',json.dumps(result))
        saved=json.loads((self.root/'image_inspections'/'inspection001.json').read_text())
        self.assertEqual(saved['metadata']['source_path'],'host-only-image-uri')
        self.assertEqual(experiments.image_inspections_used,1)

    def test_preflight_samples_images_even_if_not_on_regular_stride(self):
        from proofreader_evolve.harness.isolated_scoring import preflight
        frame=pd.DataFrame({'detector_score':np.arange(100), 'image_available':np.zeros(100)})
        frame.loc[3,'image_available']=1.; frame.attrs['_image_rows']={3:[str(self.pixels)]}
        def score(source, sample, kind, timeout, **kwargs):
            self.assertEqual(sample.detector_score.iloc[0],3)
            self.assertEqual(set(sample.attrs['_image_rows']),{0})
            self.assertEqual(sample.image_available.iloc[-1],0)
            return np.zeros(len(sample))
        with patch('proofreader_evolve.harness.isolated_scoring.score',side_effect=score):
            preflight('unused',frame,'merge')

    def test_mcp_sdk_preserves_image_blocks_and_trace_omits_base64(self):
        import mcp.types as types
        experiments=object.__new__(TrainingExperiments)
        experiments.lock=asyncio.Lock()
        experiments.research_status = Mock(return_value={'status': 'complete', 'complete': True})
        payload={'content':[{'type':'text','text':'TRAIN preview'},
                            {'type':'image','mimeType':'image/png','data':'cGl4ZWxz'}]}
        experiments.inspect_candidate_image=Mock(return_value=payload)
        server=experiments.mcp_server()['instance']
        async def invoke():
            request=types.CallToolRequest(method='tools/call',params=types.CallToolRequestParams(
                name='inspect_candidate_image',arguments={'candidate_ref':'train','radius_um':40.,'level':0,'occurrence_index':0}))
            return await server.request_handlers[types.CallToolRequest](request)
        result=asyncio.run(invoke()).root
        self.assertFalse(result.isError)
        images=[c for c in result.content if c.type=='image']
        self.assertEqual(len(images),1); self.assertEqual(images[0].data,'cGl4ZWxz')
        self.assertTrue(json.loads(result.content[-1].text)['research_status']['complete'])
        cleaned=image_safe_content(payload)
        self.assertNotIn('cGl4ZWxz',json.dumps(cleaned)); self.assertIn('encoded_sha256',cleaned['content'][1])


if __name__ == '__main__':
    unittest.main()
