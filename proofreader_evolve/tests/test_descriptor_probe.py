"""Exact-input probes and configurable descriptor images; synthetic CPU fixtures only."""
import base64
import asyncio
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from proofreader_evolve.harness import descriptor_inputs, descriptor_probe, descriptor_runs as dr
from proofreader_evolve.harness import detector_prior
from proofreader_evolve.harness.descriptor_contract import descriptor_spec
from proofreader_evolve.harness.image_context import CandidateImageStore
from proofreader_evolve.harness.image_coordinates import ImageGeometry
from proofreader_evolve.harness.image_evidence import descriptor_ablation_evidence, generation_image_evidence
from proofreader_evolve.harness.local_context import candidate_ref
from proofreader_evolve.tests import test_descriptor_compute as fixtures
from proofreader_evolve.tests.test_selection_protocol import brain_table
from proofreader_evolve.tests.label_fixture import setUpModule, tearDownModule  # noqa: F401


CUSTOM = fixtures.IMAGE_DESCRIPTOR.replace("'image_tier': 'level1'", "'image': {'level': 0, 'radius_um': 10}")


class SyntheticSource:
    identity = {'root': 'synthetic://registered-image'}

    def __init__(self):
        self._levels = {level: (ImageGeometry(('z', 'y', 'x'), (1000, 1000, 1000),
                          np.full(3, 2. ** level), np.zeros(3), np.eye(4), str(level)),
                          {'shape': [1000, 1000, 1000], 'level': level}) for level in (0, 1)}
        self.requests = []

    def level(self, level):
        return self._levels[level][0]

    def read(self, xyz, *, level, radius_um, channel, timepoint, timeout=None):
        self.requests.append((level, radius_um))
        geometry = self.level(level)
        origin, shape, anchors = geometry.crop(xyz, radius_um)
        data = {'image_zyx': (np.indices(shape).sum(axis=0) + xyz[0, 0]).astype(np.uint16),
                'valid_zyx': np.ones(shape, dtype=bool), 'anchors_zyx': anchors,
                'spacing_zyx_um': geometry.scale_xyz_um[::-1]}
        metadata = {**geometry.summary(), 'origin_zyx': origin.tolist(), 'shape_zyx': shape.tolist(),
                    'center_xyz_um': np.asarray(xyz).mean(axis=0).tolist(), 'level': level,
                    'channel': channel, 'timepoint': timepoint, 'valid_fraction': 1., 'dtype': 'uint16'}
        return data, metadata


def attach_synthetic_store(table, directory):
    store = CandidateImageStore(directory, directory / 'no-registration-file')
    source = SyntheticSource()
    graph = SimpleNamespace(node_xyz=np.asarray([fixtures.center_of(i) for i in range(len(table.features))]))
    store.source = lambda table, reviewed=True: (source, graph)
    table.meta['provenance'].setdefault('source_cache', None)
    table.image_context = store
    return store, source


class ProbeInputTests(unittest.TestCase):
    setUp = fixtures.DescriptorRunTests.setUp

    def test_contract_and_cached_explicit_alias(self):
        _, spec = descriptor_spec(CUSTOM)
        self.assertIsNone(spec['image_tier'])
        self.assertEqual(spec['image'], dict(level=0, radius_um=10., channel=0, timepoint=0))
        for replacement in ("{'level': True, 'radius_um': 40}", "{'level': 0, 'radius_um': 0}",
                            "{'level': 7, 'radius_um': 40}", "{'level': 0, 'radius_um': 40, 'center': 0}"):
            with self.assertRaises(ValueError):
                descriptor_spec(CUSTOM.replace("{'level': 0, 'radius_um': 10}", replacement))
        with self.assertRaisesRegex(ValueError, 'conflict'):
            descriptor_spec(CUSTOM.replace("'image':", "'image_tier': 'level1', 'image':"))
        with self.assertRaises(ValueError):
            descriptor_spec(CUSTOM.replace("'inputs': 'image'", "'inputs': 'geometry'"))
        legacy, old_spec = descriptor_spec(fixtures.IMAGE_DESCRIPTOR)
        explicit, new_spec = descriptor_spec(fixtures.IMAGE_DESCRIPTOR.replace(
            "'image_tier': 'level1'", "'image': {'level': 1, 'radius_um': 30}"))
        a = self.runs.probe(self.table, legacy, old_spec, [0, 7], log_dir=self.root / 'old')
        b = self.runs.probe(self.table, explicit, new_spec, [0, 7], log_dir=self.root / 'new')
        np.testing.assert_array_equal(a['values'], b['values'])
        self.assertIsNone(getattr(self.table, 'image_context', None))  # alias never needs cloud access
        identity = self.runs.identity(self.table, old_spec, 'all_cached', a['rows'])
        self.runs.cache.entry(self.table).manifest['image_tiers']['level1']['source'] = {'identity': 'changed-source'}
        self.assertNotEqual(identity, self.runs.identity(self.table, old_spec, 'all_cached', a['rows']))

    def test_custom_probe_full_parity_and_bounded_temporary_staging(self):
        store, source = attach_synthetic_store(self.table, self.root / 'patches')
        program, spec = descriptor_spec(CUSTOM)
        observed = {}
        probe = self.runs.probe(self.table, program, spec, [0, 3], log_dir=self.root / 'probe',
                               observe=lambda row, context, pixels: observed.update({row: (context, pixels)}))
        size = store.cache_bytes
        store.max_cache_bytes = size  # no further persistent patches can fit
        # Force multiple byte-bounded workers within the row batch.
        with patch.object(descriptor_inputs, 'MAX_BATCH_BYTES', 50000):
            full = self.runs.compute(self.table, program, spec, 'all_cached', k=3, wall_budget=120,
                                     log_dir=self.root / 'full')
        np.testing.assert_array_equal(probe['values'], full['values'][[0, 3]])
        self.assertEqual(store.cache_bytes, size)
        self.assertEqual(len(list(store.directory.glob('*/patch.npz'))), 2)
        self.assertGreater(len(list((self.root / 'full').glob('batch*.log'))), 1)
        context, pixels = observed[0]
        self.assertEqual(pixels['image_zyx'].shape, (20, 20, 20))
        self.assertEqual(context['radius_um'], 50.)
        self.assertNotIn('source_cache', context)
        np.testing.assert_allclose(np.asarray(context['fragment']['nodes_zyx'])[0], pixels['anchors_zyx'][0])
        _, larger = descriptor_spec(CUSTOM.replace("'radius_um': 10", "'radius_um': 12"))
        self.assertNotEqual(self.runs.identity(self.table, spec, 'all_cached', full['rows']),
                            self.runs.identity(self.table, larger, 'all_cached', full['rows']))
        with tempfile.TemporaryDirectory() as tmp:
            row, context, pixels = next(descriptor_inputs.iter_inputs(self.table, self.runs.cache.entry(self.table),
                                       larger, [0], Path(tmp)))
            self.assertEqual(pixels['image_zyx'].shape, (24, 24, 24))
            self.assertFalse(list(Path(tmp).rglob('patch.npz')))
        _, coarser = descriptor_spec(CUSTOM.replace("'level': 0", "'level': 1"))
        with tempfile.TemporaryDirectory() as tmp:
            _, context, pixels = next(descriptor_inputs.iter_inputs(self.table, self.runs.cache.entry(self.table),
                                     coarser, [0], Path(tmp)))
            self.assertEqual(pixels['image_zyx'].shape, (10, 10, 10))
            np.testing.assert_array_equal(pixels['spacing_zyx_um'], [2., 2., 2.])
        source.identity = {**source.identity, 'attributes_sha256': 'changed-source'}
        self.assertFalse(self.runs.cached(self.table, spec, 'all_cached', 3))

    def test_uniform_samples_and_raw_allowlists(self):
        initial = self.runs.sample_rows(self.table, 4, 7)
        self.table.truth = 1 - self.table.truth
        self.table.features['detector_score'] *= -1
        entry = self.runs.cache.entry(self.table)
        entry.rows = entry.rows[::-1]
        np.testing.assert_array_equal(initial, self.runs.sample_rows(self.table, 4, 7))
        self.assertFalse(np.array_equal(initial, np.arange(4)))
        bank = SimpleNamespace(tables={'merge': self.table})
        for row in self.table.candidates:
            row['candidate_family'] = 'junction'
        detector_prior.strip_to_raw({'802449': bank})
        entry.expose_band = False
        self.table.features['bank_agent_other'] = 123.
        attach_synthetic_store(self.table, self.root / 'patches')
        program = CUSTOM.replace("    if not context['image_available']:",
            "    assert 'row_in_band' not in context\n"
            "    assert all(k.startswith('raw_') for k in context['features'])\n"
            "    assert 'bank_agent_other' not in context['features']\n"
            "    if not context['image_available']:")
        program, spec = descriptor_spec(program)
        result = self.runs.probe(self.table, program, spec, initial, log_dir=self.root / 'raw')
        self.assertTrue(np.isfinite(result['values']).all())

    def test_sparse_diagnostics_do_not_confuse_zero_quartiles_with_constant(self):
        values = np.zeros((12, 1)); values[10, 0] = 2.; values[11, 0] = np.nan
        report = descriptor_probe.diagnostics(self.runs, self.table, ['sparse'], np.arange(12), values)['sparse']
        self.assertEqual(report['q25_q50_q75'], [0., 0., 0.])
        self.assertFalse(report['constant'])
        self.assertEqual((report['finite_count'], report['missing_count'], report['nonzero_count']), (11, 1, 1))
        self.assertEqual(report['examples']['nonzero'], candidate_ref(self.table, 10))
        self.assertEqual(report['examples']['missing'], candidate_ref(self.table, 11))
        self.assertNotIn('1', report)

    def test_empty_cache_plan_has_json_safe_missing_fractions(self):
        self.runs.cache.entry(self.table).rows = np.array([], dtype=int)
        program, spec = descriptor_spec(fixtures.GEOMETRY_DESCRIPTOR)
        report = self.runs.plan(self.table, program, spec, k=3)
        self.assertEqual(report['sample_rows'], 0)
        self.assertTrue(all(value is None for value in report['finite_fraction'].values()))
        self.assertEqual(report['estimates']['all_cached']['estimated_seconds'], 0.)
        json.dumps(report, allow_nan=False)


class ProbeSessionTests(unittest.TestCase):
    setUp = fixtures.SessionDescriptorTests.setUp

    def test_full_computation_does_not_grant_extra_time_to_later_brains(self):
        self.session.descriptor_path.write_text(fixtures.GEOMETRY_DESCRIPTOR)
        self.session.descriptor_budget['per_call_seconds'] = 10.
        clock = [0.]

        def overrun(*args, **kwargs):
            clock[0] = 11.
            return {}  # No registration may happen after this brain exceeded the shared deadline.

        runs = self.memory.descriptor_runs
        with patch.object(runs, 'cached', return_value=False), \
                patch.object(runs, 'compute', side_effect=overrun) as compute, \
                patch('proofreader_evolve.harness.train_experiments.time.monotonic', side_effect=lambda: clock[0]):
            result = self.session.compute_descriptors('all_cached')
        self.assertEqual(compute.call_count, 1)
        self.assertEqual(result['status'], 'execution_error')
        self.assertIn('before the next TRAIN brain', result['error'])
        self.assertEqual(self.session.descriptor_seconds_used, 11.)
        self.assertEqual(self.session.evaluations_used, 1)
        self.assertEqual(self.memory.descriptors, {})

    def test_preview_exact_inputs_draft_name_reuse_and_no_registration(self):
        table = self.train['802449'].tables['merge']
        self.session.descriptor_path.write_text(fixtures.IMAGE_DESCRIPTOR)
        before = table.features.copy(deep=True)
        self.memory.descriptors['bank_agent_mean_intensity'] = {'code_sha256': 'other-draft'}
        rendered = []
        from proofreader_evolve.harness.image_preview import render_preview

        def capture(path, pixels, nodes, edges, segments, title):
            rendered.append((pixels, nodes))
            return render_preview(path, pixels, nodes, edges, segments, title)

        with patch('proofreader_evolve.harness.image_preview.render_preview', side_effect=capture):
            result = self.session.probe_descriptor([candidate_ref(table, 0)], previews=True)
        report = json.loads(result['content'][0]['text'])
        self.assertEqual(report['status'], 'probed', report)
        self.assertTrue(base64.b64decode(result['content'][1]['data']).startswith(b'\x89PNG'))
        expected, _ = fixtures.fake_image_reader(0, {'level': 1})
        np.testing.assert_array_equal(rendered[0][0]['image_zyx'], expected['image_zyx'])
        np.testing.assert_allclose(rendered[0][1][0], expected['anchors_zyx'][0])
        self.assertTrue(before.equals(table.features))
        self.assertEqual(self.session.evaluations_used, 0)
        self.assertGreater(self.session.descriptor_seconds_used, 0)
        self.assertEqual(self.memory.descriptor_programs, {})
        self.assertEqual(self.session.image_inspections_used, 1)
        evidence = generation_image_evidence(self.session.gen_dir, {})
        self.assertEqual(evidence['descriptor_image_probes_succeeded'], 1)
        self.assertEqual(evidence['preview_records'], 1)
        self.assertEqual(evidence['paired_diagnostics'], [])
        with self.assertRaisesRegex(ValueError, 'already registered'):
            self.session.compute_descriptors('all_cached')

    def test_worker_failure_has_case_ref_and_import_failure_is_not_success(self):
        table = self.train['802449'].tables['merge']
        reference = candidate_ref(table, 2)
        for code, identified in ((fixtures.GEOMETRY_DESCRIPTOR.replace("fragment = context['fragment']",
                                  "raise RuntimeError('synthetic failure')"), True),
                                 (fixtures.GEOMETRY_DESCRIPTOR + '\nimport absent_descriptor_helper\n', False)):
            self.session.descriptor_path.write_text(code)
            before = self.session.descriptor_seconds_used
            result = self.session.probe_descriptor([reference])
            self.assertEqual(result['status'], 'execution_error', result)
            self.assertEqual('candidate_ref' in result, identified)
            if identified:
                self.assertEqual(result['candidate_ref'], reference)
            self.assertGreater(self.session.descriptor_seconds_used, before)
            self.assertEqual(self.session.evaluations_used, 0)
            self.assertEqual(self.memory.descriptors, {})

    def test_custom_config_reaches_frozen_inference_and_io_failure_is_identified(self):
        for bank in self.train.values():
            attach_synthetic_store(bank.tables['merge'], self.root / 'train_images')
        self.session.descriptor_path.write_text(CUSTOM)
        table = self.train['802449'].tables['merge']
        reference = candidate_ref(table, 0)
        with patch.object(table.image_context, 'patch', side_effect=OSError('synthetic missing image')):
            failed = self.session.probe_descriptor([reference])
        self.assertEqual(failed['status'], 'execution_error')
        self.assertEqual(failed['candidate_ref'], reference)
        result = self.session.compute_descriptors('all_cached')
        self.assertEqual(result['status'], 'registered', result)
        other = {'heldout': brain_table('heldout')}
        heldout = other['heldout'].tables['merge']
        fixtures.build_cache(self.root / 'cache', heldout)
        store, source = attach_synthetic_store(heldout, self.root / 'heldout_images')
        self.session.ensure_descriptors(other, ['bank_agent_mean_intensity'])
        np.testing.assert_array_equal(table.features['bank_agent_mean_intensity'], heldout.features['bank_agent_mean_intensity'])
        self.assertEqual(set(source.requests), {(0, 10.)})
        self.assertEqual(store.cache_bytes, 0)
        self.assertTrue(list((self.session.gen_dir / 'validation_descriptor_runs').rglob('batch*.log')))
        self.assertFalse((self.session.gen_dir / 'descriptor_runs' / 'inference').exists())

    def test_all_handles_validated_before_io_and_whole_train_estimate(self):
        self.session.descriptor_path.write_text(fixtures.GEOMETRY_DESCRIPTOR)
        heldout = brain_table('heldout').tables['merge']
        with patch.object(self.memory.descriptor_runs, 'probe', side_effect=AssertionError('must not execute')):
            with self.assertRaisesRegex(ValueError, 'TRAIN'):
                self.session.probe_descriptor([candidate_ref(self.train['802449'].tables['merge'], 0),
                                               candidate_ref(heldout, 0)])
        plan = {'sample_rows': 64, 'seconds_per_row_single_worker': 1.,
                'estimates': {'all_cached': {'rows': 100, 'batches': 2, 'estimated_seconds': 80.}}}
        runs = self.memory.descriptor_runs
        with patch.object(runs, 'plan', return_value=plan) as measured, patch.object(runs, 'cached', return_value=False):
            result = self.session.plan_descriptor_run()
        self.assertEqual(measured.call_count, 2)
        self.assertEqual(result['estimates']['all_cached']['estimated_seconds'], 160.)
        self.assertFalse(result['fits_budget']['all_cached'])
        self.assertIsNone(result['frozen_inference']['estimated_seconds'])
        evidence = generation_image_evidence(self.session.gen_dir, {})
        self.assertEqual(evidence['descriptor_plans_attempted'], 1)
        self.assertEqual(evidence['preview_records'], 0)
        self.session.descriptor_budget['frozen_inference_rows'] = [100, 200]
        with patch.object(runs, 'plan', return_value=plan), patch.object(runs, 'cached', return_value=False):
            estimated = self.session.plan_descriptor_run()['frozen_inference']
        self.assertEqual(estimated['estimated_seconds'], 300.)
        self.assertEqual(estimated['scope'], 'all_cached')
        with patch.object(runs, 'plan', return_value=plan), patch.object(runs, 'cached',
                side_effect=lambda table, *args: table.meta['provenance']['brain'] == '794495'):
            partial = self.session.plan_descriptor_run()
        self.assertEqual(partial['estimates']['all_cached']['estimated_seconds'], 160.)
        self.assertEqual(partial['estimates']['all_cached']['estimated_uncached_seconds'], 80.)
        self.assertEqual(partial['estimates']['all_cached']['cached_brains'], ['794495'])
        self.assertTrue(partial['fits_budget']['all_cached'])
        with patch.object(runs, 'plan', return_value=plan), patch.object(runs, 'cached', return_value=True):
            reused = self.session.plan_descriptor_run()
        self.assertEqual(reused['estimates']['all_cached']['estimated_uncached_seconds'], 0.)
        self.assertTrue(reused['fits_budget']['all_cached'])

    def test_sdk_accepts_optional_probe_arguments(self):
        import mcp.types as types
        self.session.descriptor_path.write_text(fixtures.GEOMETRY_DESCRIPTOR)
        self.session.evaluations_used = 2
        self.memory.descriptor_runs.allowed_scopes = ('all_cached',)
        server = self.session.mcp_server()['instance']

        async def invoke():
            listed = await server.request_handlers[types.ListToolsRequest](types.ListToolsRequest(method='tools/list'))
            compute = next(t for t in listed.root.tools if t.name == 'compute_descriptors')
            self.assertEqual(compute.inputSchema['properties']['scope']['enum'], ['all_cached'])
            self.assertNotIn('pilot', compute.description)
            request = types.CallToolRequest(method='tools/call', params=types.CallToolRequestParams(
                name='probe_descriptor', arguments={'sample_per_brain': 2, 'seed': 5}))
            return await server.request_handlers[types.CallToolRequest](request)

        result = asyncio.run(invoke()).root
        self.assertFalse(result.isError)
        report = json.loads(result.content[0].text)
        self.assertEqual(report['status'], 'probed', report)
        self.assertEqual({b: x['rows'] for b, x in report['brains'].items()}, {'794495': 2, '802449': 2})
        self.assertEqual(self.session.evaluations_used, 2)
        self.assertEqual(report['evaluations_used'], 2)
        self.assertEqual(report['evaluations_charged'], 0)
        self.assertEqual(report['evaluations_remaining'], 1)
        self.assertIsNone(report['image_tier'])

    def test_descriptor_ablation_is_visible_without_pixel_only_claim(self):
        report = {'feature_columns': ['bank_agent_x'], 'status': 'measured', 'delta_precision': .1,
                  'image_evidence': {'image_inputs_present': False}}
        registry = {'bank_agent_x': {'inputs': 'both', 'image': {'level': 0, 'radius_um': 40}}}
        report['image_evidence'] = descriptor_ablation_evidence(report, registry)
        evidence = generation_image_evidence(self.session.gen_dir, {'feature_diagnostics': [report]})
        self.assertEqual(len(evidence['paired_diagnostics']), 1)
        self.assertFalse(report['image_evidence']['image_only_control'])
        self.assertIn('bank_agent_x', report['image_evidence']['descriptor_columns'])
