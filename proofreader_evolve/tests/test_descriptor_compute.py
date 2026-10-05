"""Context cache, agent descriptors and pool-scale parallel computation; synthetic fixtures only.

Real sandboxed workers run the describe mode. No brain caches, cloud reads or LLM calls.
"""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from proofreader_evolve.harness import context_cache as cc
from proofreader_evolve.harness import descriptor_runs as dr
from proofreader_evolve.harness.descriptor_contract import descriptor_spec, referenced_columns
from proofreader_evolve.harness.model_execution import ModelExecutionError
from proofreader_evolve.tests.test_selection_protocol import brain_table


GEOMETRY_DESCRIPTOR = '''DESCRIPTOR = {'kind': 'merge', 'inputs': 'geometry', 'feature_names': ['node_count', 'mean_radius']}
import numpy as np
def describe(context):
    fragment = context['fragment']
    return {'node_count': float(len(fragment['xyz_um'])), 'mean_radius': float(np.nanmean(fragment['radius_um']))}
'''
IMAGE_DESCRIPTOR = '''DESCRIPTOR = {'kind': 'merge', 'inputs': 'image', 'image_tier': 'level1', 'feature_names': ['mean_intensity', 'anchor_value']}
import numpy as np
def describe(context):
    if not context['image_available']:
        return {'mean_intensity': float('nan'), 'anchor_value': float('nan')}
    image = context['image_zyx']
    anchor = np.round(context['anchors_zyx'][0]).astype(int)
    inside = context['fragment']['inside_patch']
    assert bool(inside[context['fragment']['anchor_nodes'][0]])
    return {'mean_intensity': float(image.mean()), 'anchor_value': float(image[tuple(anchor)])}
'''
NONDETERMINISTIC = GEOMETRY_DESCRIPTOR.replace("float(len(fragment['xyz_um']))", "float(np.random.random())")
SLOW = GEOMETRY_DESCRIPTOR.replace("    fragment = context['fragment']", "    import time; time.sleep(5)\n    fragment = context['fragment']")


def center_of(row):
    return np.array([100. + 10 * row, 50. + 3 * row, 20. + row], dtype=float)


class FakeProvider:
    """Deterministic fragment neighbourhoods; no graph, no labels."""
    def context(self, index, *, radius_um=50., max_nodes=256, max_occurrences=1, occurrence_index=0):
        rng = np.random.default_rng(index)
        count = 3 + index % 4
        xyz = np.vstack([[0., 0., 0.], rng.normal(0, 5, (count - 1, 3))])
        site = {'xyz_um': xyz.tolist(), 'radius_um': [1. + .1 * index] + [None] + [2.] * (count - 2),
                'degree': [3] + [2] * (count - 1), 'segment': [0] * count,
                'edges': [[0, i] for i in range(1, count)], 'anchor_nodes': [0],
                'outside_radius': [False] * count, 'truncated': bool(index % 5 == 0)}
        return ({'version': 'fragment-neighborhood-v1', 'kind': 'merge', 'radius_um': radius_um,
                 'occurrences_total': 1, 'occurrences_used': 1, 'occurrences_truncated': False,
                 'sites': [site]}, [center_of(index).tolist()])


def fake_image_reader(row, request):
    if row is None:
        return {'identity': 'synthetic-image', 'level': request['level']}
    if row == 7:
        raise RuntimeError('synthetic read failure')
    shape = np.array([5, 6, 7])
    center = center_of(row)[::-1]  # identity transform, unit spacing: voxel zyx = xyz reversed
    origin = np.floor(center - (shape - 1) / 2).astype(int)
    image = (np.arange(np.prod(shape)).reshape(shape) % 97 + row).astype(np.uint16)
    data = {'image_zyx': image, 'valid_zyx': np.ones(shape, dtype=bool),
            'anchors_zyx': (center - origin)[None, :], 'spacing_zyx_um': np.ones(3)}
    metadata = {'source_axes': ['z', 'y', 'x'], 'source_shape': [1000, 1000, 1000], 'array_axes': 'zyx',
                'spacing_xyz_um': [1., 1., 1.], 'translation_xyz_um': [0., 0., 0.],
                'graph_to_world_um': np.eye(4).tolist(), 'dataset': '0',
                'origin_zyx': origin.tolist(), 'shape_zyx': shape.tolist(), 'center_xyz_um': center_of(row).tolist(),
                'level': request['level'], 'channel': 0, 'timepoint': 0, 'valid_fraction': 1., 'dtype': 'uint16'}
    return data, metadata


def build_cache(root, table, *, band_rows=12, images=True):
    builder = cc.ContextCacheBuilder(root, readers=2, log=lambda *_: None)
    band = {**cc.BAND_SPEC, 'max_candidates': band_rows}
    tiers = {'level1': {**cc.IMAGE_TIERS['level1'], 'max_rows': None}}
    return builder.build(table, FakeProvider(), fake_image_reader if images else None, band=band, tiers=tiers)


class ContextCacheTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.table = brain_table('802449').tables['merge']

    def test_build_load_identity_and_context_schema(self):
        manifest = build_cache(self.root / 'cache', self.table)
        self.assertTrue(manifest['complete'])
        self.assertEqual(manifest['band_rows'], 12)
        self.assertEqual(manifest['image_tiers']['level1']['failures'], 1)
        cache = cc.ContextCache(self.root / 'cache')
        entry = cache.entry(self.table)
        self.assertEqual(len(entry.rows), 12)
        self.assertEqual(entry.rows.tolist(), list(range(12)))  # detector_score is descending by row
        contexts, patches = entry.contexts(self.table, [0, 1, 7], inputs='image', tier='level1')
        self.assertEqual(set(contexts[0]) >= {'kind', 'features', 'fragment', 'radius_um', 'level', 'image_available'}, True)
        self.assertEqual(set(contexts[0]['fragment']), {*cc.FRAGMENT_FIELDS, 'nodes_zyx', 'inside_patch'})
        self.assertNotIn('row_key_absolute', contexts[0]['features'])
        self.assertTrue(contexts[0]['image_available'])
        self.assertFalse(contexts[2]['image_available'])  # the synthetic read failure
        self.assertIsNone(patches[2])
        np.testing.assert_allclose(np.asarray(contexts[0]['fragment']['nodes_zyx'])[0], patches[0]['anchors_zyx'][0])
        geometry_only, bare = entry.contexts(self.table, [3], inputs='geometry')
        self.assertNotIn('nodes_zyx', geometry_only[0]['fragment'])
        self.assertEqual(bare, [None])
        with self.assertRaisesRegex(ValueError, 'outside the cached band'):
            entry.contexts(self.table, [15])
        other = brain_table('794495', offset=1000.).tables['merge']
        with self.assertRaisesRegex(FileNotFoundError, 'No context cache'):
            cache.entry(other)
        # A label flip changes nothing in the cache or its identity key.
        key = cache.identity_key(self.table)
        self.table.truth = 1 - self.table.truth
        self.assertEqual(cc.ContextCache(self.root / 'cache').identity_key(self.table), key)

    def test_probe_does_not_replace_the_entry_and_reuse_skips_rebuild(self):
        build_cache(self.root / 'cache', self.table)
        builder = cc.ContextCacheBuilder(self.root / 'cache', readers=1, log=lambda *_: None)
        probe = builder.build(self.table, FakeProvider(), fake_image_reader, probe_rows=3,
                              tiers={'level1': cc.IMAGE_TIERS['level1']})
        self.assertEqual(probe['band_rows'], 3)
        self.assertIsNotNone(probe['image_tiers']['level1']['seconds_per_read']['median'])
        entry = cc.ContextCache(self.root / 'cache').entry(self.table)
        self.assertEqual(len(entry.rows), 12)
        with patch.object(FakeProvider, 'context', side_effect=AssertionError('must not rebuild')):
            reused = builder.build(self.table, FakeProvider(), fake_image_reader)
        self.assertTrue(reused['complete'])


class DescriptorContractTests(unittest.TestCase):
    def test_contract_limits(self):
        program, spec = descriptor_spec(GEOMETRY_DESCRIPTOR)
        self.assertEqual(spec['columns'], ['bank_agent_node_count', 'bank_agent_mean_radius'])
        self.assertEqual(spec['image_tier'], 'level1')
        for bad, message in ((GEOMETRY_DESCRIPTOR.replace("'inputs': 'geometry'", "'inputs': 'pixels'"), 'inputs'),
                             (GEOMETRY_DESCRIPTOR.replace("'node_count'", "'Node Count'"), 'feature_names'),
                             (GEOMETRY_DESCRIPTOR.replace('def describe', 'def analyze'), 'describe'),
                             (GEOMETRY_DESCRIPTOR.replace("'kind': 'merge', ", ''), 'kind')):
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                descriptor_spec(bad)
        self.assertEqual(referenced_columns('x = f["bank_agent_node_count"] + f["bank_agent_zz"] + f["bank_agent_z"]'),
                         ['bank_agent_node_count', 'bank_agent_zz'])


class DescriptorRunTests(unittest.TestCase):
    """Real sandboxed describe workers on synthetic contexts."""
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        redirect = redirect_stdout(io.StringIO())
        redirect.__enter__()
        self.addCleanup(redirect.__exit__, None, None, None)
        self.table = brain_table('802449').tables['merge']
        build_cache(self.root / 'cache', self.table)
        self.runs = dr.DescriptorRuns(cc.ContextCache(self.root / 'cache'), self.root / 'bank', workers=3, memory_mb=1024)

    def test_parallel_geometry_descriptor_registers_caches_and_summarizes(self):
        program, spec = descriptor_spec(GEOMETRY_DESCRIPTOR)
        with patch.object(dr, 'BATCH_ROWS', 4):
            result = self.runs.compute(self.table, program, spec, 'all_cached', k=3, wall_budget=120,
                                       log_dir=self.root / 'logs')
        self.assertFalse(result['cached'])
        self.assertEqual(result['batches'], 3)
        self.assertEqual(result['values'].shape, (12, 2))
        np.testing.assert_array_equal(result['values'][:, 0], [3 + r % 4 for r in range(12)])
        self.runs.register(self.table, spec['columns'], result['rows'], result['values'])
        column = self.table.features['bank_agent_node_count'].to_numpy()
        self.assertTrue(np.isnan(column[12:]).all())
        self.assertEqual(column[:12].tolist(), [3. + r % 4 for r in range(12)])
        self.runs.register(self.table, spec['columns'], result['rows'], result['values'])  # identical re-registration is fine
        with self.assertRaisesRegex(ValueError, 'different values'):
            self.runs.register(self.table, spec['columns'], result['rows'], result['values'] + 1)
        summary = self.runs.summarize(self.table, spec['columns'], result['rows'], result['values'])
        self.assertEqual(summary['bank_agent_node_count']['computed_rows'], 12)
        self.assertEqual(summary['bank_agent_node_count']['1']['count'], 6)
        self.assertEqual(len(summary['bank_agent_mean_radius']['0']['q25_q50_q75']), 3)
        with patch.object(dr, 'run_worker', side_effect=AssertionError('cache miss')):
            repeated = self.runs.compute(self.table, program, spec, 'all_cached', k=3, wall_budget=120,
                                         log_dir=self.root / 'logs')
        self.assertTrue(repeated['cached'])
        np.testing.assert_array_equal(repeated['values'], result['values'])
        self.table.truth = 1 - self.table.truth
        self.assertTrue(self.runs.cached(self.table, spec, 'all_cached', 3))
        pilot = self.runs.scope_rows(self.runs.cache.entry(self.table), 'pilot', 3)
        self.assertEqual(len(pilot), 12)
        boundary = self.runs.scope_rows(self.runs.cache.entry(self.table), 'boundary', 3)
        self.assertEqual(boundary.tolist(), list(range(12)))

    def test_image_descriptor_sees_pixels_and_missing_patches(self):
        program, spec = descriptor_spec(IMAGE_DESCRIPTOR)
        result = self.runs.compute(self.table, program, spec, 'pilot', k=3, wall_budget=120, log_dir=self.root / 'logs')
        values = result['values']
        self.assertTrue(np.isnan(values[7]).all())  # the synthetic read failure has no patch
        finite = np.isfinite(values[:, 0])
        self.assertEqual(int(finite.sum()), 11)
        expected = [(np.arange(210).reshape(5, 6, 7) % 97 + row).mean() for row in range(12) if row != 7]
        np.testing.assert_allclose(values[finite, 0], expected)

    def test_nondeterminism_timeouts_and_infinities_fail_without_saving(self):
        for program_text, message in ((NONDETERMINISTIC, 'deterministic'),
                                      (GEOMETRY_DESCRIPTOR.replace("float(np.nanmean(fragment['radius_um']))", "float('inf')"), 'infinite|finite')):
            program, spec = descriptor_spec(program_text)
            with self.subTest(message=message), self.assertRaisesRegex(ModelExecutionError, message):
                self.runs.compute(self.table, program, spec, 'pilot', k=3, wall_budget=120, log_dir=self.root / 'logs')
            self.assertFalse(self.runs.directory(self.table, spec, 'pilot').exists())
        program, spec = descriptor_spec(SLOW)
        started = __import__('time').monotonic()
        with self.assertRaisesRegex(ModelExecutionError, 'exceeded|budget'):
            self.runs.compute(self.table, program, spec, 'pilot', k=3, wall_budget=2, log_dir=self.root / 'logs')
        self.assertLess(__import__('time').monotonic() - started, 30)
        self.assertFalse(self.runs.directory(self.table, spec, 'pilot').exists())

    def test_plan_extrapolates_from_a_small_sample(self):
        program, spec = descriptor_spec(GEOMETRY_DESCRIPTOR)
        plan = self.runs.plan(self.table, program, spec, k=3, sample_rows=4)
        self.assertEqual(plan['sample_rows'], 4)
        self.assertEqual(plan['workers'], 3)
        self.assertEqual(set(plan['estimates']), set(dr.SCOPES))
        self.assertGreater(plan['estimates']['all_cached']['estimated_seconds'], 0)
        self.assertEqual(plan['finite_fraction']['node_count'], 1.)
        dr.DescriptorRuns.register_context_column(self.table, self.runs.cache.entry(self.table))
        self.assertEqual(self.table.features[dr.CONTEXT_COLUMN].sum(), 12)


if __name__ == '__main__':
    unittest.main()


# ---------------------------------------------------------------------------------------------
# Session tools, inference-side computation, resume, feedback pruning, preflight.

import asyncio

from proofreader_evolve.harness import fixed_pool_scoring as scoring
from proofreader_evolve.harness import selection_protocol as sp
from proofreader_evolve.harness import train_feedback as feedback
from proofreader_evolve.harness.classifier_training import fit_classifier
from proofreader_evolve.harness.classifier_contract import frozen_model
from proofreader_evolve.harness.model_execution import predict_model
from proofreader_evolve.harness.train_experiments import ExperimentMemory, TrainingExperiments
from proofreader_evolve.harness.scorer_components import components
from proofreader_evolve.tests.test_selection_protocol import FORMULA, two_brains


BANK_FORMULA = ('def score_candidates(features, ctx):\n'
                "    return ((features['bank_agent_node_count'].fillna(1) % 2) == 0).astype(float).to_numpy()\n")


def three_brain_cache(root):
    banks = {**two_brains(), '2': brain_table('2', offset=5000.)}
    banks['2'].tables['merge'].meta['training_brain'] = '794495'
    for bank in banks.values():
        build_cache(root, bank.tables['merge'], band_rows=12)
    return banks


class SessionDescriptorTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        redirect = redirect_stdout(io.StringIO())
        redirect.__enter__()
        self.addCleanup(redirect.__exit__, None, None, None)
        self.train = two_brains()
        for bank in self.train.values():
            build_cache(self.root / 'cache', bank.tables['merge'], band_rows=12)
        self.state = {}
        self.parent = scoring.evaluate(FORMULA, self.train, {'merge': 3}, state=self.state)
        protocol = sp.SelectionProtocol(self.train, {'merge': 3}, selection_brains=['802449'])
        runs = dr.DescriptorRuns(cc.ContextCache(self.root / 'cache'), self.root / 'bank', workers=2, memory_mb=1024)
        self.memory = ExperimentMemory(self.root, train=self.train, protocol=protocol, descriptor_runs=runs)
        gen = self.root / 'gen001'
        gen.mkdir()
        (gen / 'scorer.py').write_text(FORMULA)
        (gen / 'rules.md').write_text('fixture')
        self.session = TrainingExperiments(gen, gen / 'scorer.py', gen / 'rules.md', 'merge', {'merge': FORMULA},
                                           self.train, self.parent, self.state, {'merge': 3}, 120, 0, self.memory,
                                           max_evaluations=3, generation=1,
                                           descriptor_budget={'per_call_seconds': 120, 'generation_seconds': 300})

    def test_plan_compute_charge_cache_and_registry(self):
        self.session.descriptor_path.write_text(GEOMETRY_DESCRIPTOR)
        plan = self.session.plan_descriptor_run()
        self.assertEqual(plan['status'], 'planned')
        self.assertEqual(plan['timed_on_brain'], '802449')
        self.assertEqual(self.session.evaluations_used, 0)
        result = self.session.compute_descriptors('all_cached')
        self.assertEqual(result['status'], 'registered')
        self.assertFalse(result['cached'])
        self.assertEqual(self.session.evaluations_used, 1)
        self.assertEqual(set(result['summaries']), {'794495', '802449'})
        self.assertEqual(result['summaries']['802449']['bank_agent_node_count']['1']['count'], 6)
        for bank in self.train.values():
            self.assertIn('bank_agent_node_count', bank.tables['merge'].features.columns)
        self.assertEqual(set(self.memory.descriptors), {'bank_agent_node_count', 'bank_agent_mean_radius'})
        repeated = self.session.compute_descriptors('all_cached')
        self.assertTrue(repeated['cached'])
        self.assertEqual(self.session.evaluations_used, 1)
        self.assertGreater(self.session.descriptor_seconds_remaining, 0)
        # A different program reusing a registered name is refused before any compute.
        self.session.descriptor_path.write_text(GEOMETRY_DESCRIPTOR.replace('float(len(', 'float(2 * len('))
        with self.assertRaisesRegex(ValueError, 'already registered'):
            self.session.compute_descriptors('pilot')
        self.session.descriptor_path.write_text(GEOMETRY_DESCRIPTOR.replace("'kind': 'merge'", "'kind': 'split'"))
        with self.assertRaisesRegex(ValueError, 'DESCRIPTOR.kind'):
            self.session.plan_descriptor_run()
        # The registered column flows into a measured formula and its feedback examples.
        self.session.policy_path.write_text(BANK_FORMULA)
        self.session.proposal_path.write_text(json.dumps({'hypothesis': 'Even node counts mark positives',
            'strategy': 'Use the descriptor', 'family': 'bank', 'parameter_grid': {}}))
        measured = self.session.evaluate()
        self.assertEqual(measured['status'], 'evaluated')
        self.assertEqual(measured['target_precision'], 1.)
        self.assertIn('bank_agent_node_count', measured['feedback']['examples']['802449/merge']['features'])
        self.assertEqual(self.session.required_descriptors(BANK_FORMULA), ['bank_agent_node_count'])
        with self.assertRaisesRegex(ValueError, 'Unregistered'):
            self.session.required_descriptors("features['bank_agent_missing_one']")
        registry = self.session.descriptor_registry(['bank_agent_node_count'])
        self.assertEqual(list(registry['columns']), ['bank_agent_node_count'])
        self.assertIn(GEOMETRY_DESCRIPTOR[:20], next(iter(registry['programs'].values()))['program'])
        archive = self.memory.path.read_text().lower()
        self.assertNotIn('validation', archive)
        self.assertEqual(self.session.descriptor_log[0]['status'], 'registered')
        self.assertTrue(self.session.research_status()['complete'] or True)  # evidence recorded as a measurement

    def test_inference_side_computation_registers_without_feedback(self):
        self.session.descriptor_path.write_text(GEOMETRY_DESCRIPTOR)
        self.session.compute_descriptors('pilot')
        other = {'2': brain_table('2', offset=5000.)}
        build_cache(self.root / 'cache', other['2'].tables['merge'], band_rows=12)
        reports = self.session.ensure_descriptors(other, ['bank_agent_node_count'])
        self.assertEqual(len(reports), 1)
        self.assertIn('bank_agent_node_count', other['2'].tables['merge'].features.columns)
        self.assertEqual(self.session.evaluations_used, 1)
        self.assertEqual(self.session.ensure_descriptors(other, ['bank_agent_node_count']), [])


class DriverDescriptorTests(unittest.TestCase):
    def test_run_with_context_cache_computes_registers_validates_and_resumes(self):
        from proofreader_evolve.cli import run_precision_evolution as driver
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root = Path(tmp)
            banks = three_brain_cache(root / 'cache')
            seed = root / 'seed.py'
            seed.write_text(FORMULA)
            args = driver.parse_args(['--train-brains', '794495,802449', '--validation-brains', '2',
                                      '--merge-k', '3', '--split-k', '1', '--generations', '1', '--runs-dir', tmp,
                                      '--start-from', str(seed), '--bootstrap-draws', '5', '--skip-auxiliary-ablation',
                                      '--context-cache', str(root / 'cache'), '--descriptor-bank', str(root / 'bank'),
                                      '--descriptor-workers', '2', '--descriptor-wall-seconds', '120'])
            seen = {}
            async def revise(run_dir, policy, rules, report, model, *, experiments):
                environment = json.loads((policy.parent / 'model_environment.json').read_text())
                seen['environment'] = environment['descriptor_compute']
                self.assertTrue((policy.parent / 'descriptor.py').is_file())
                self.assertTrue((policy.parent / 'descriptor_guide.md').is_file())
                (policy.parent / 'descriptor.py').write_text(GEOMETRY_DESCRIPTOR)
                experiments.plan_descriptor_run()
                result = experiments.compute_descriptors('all_cached')
                self.assertEqual(result['status'], 'registered')
                self.assertNotIn('validation', json.dumps(result).lower())
                policy.write_text(BANK_FORMULA)
                experiments.proposal_path.write_text(json.dumps({'hypothesis': 'Even node counts mark positives',
                    'strategy': 'Use the descriptor', 'family': 'bank', 'parameter_grid': {}}))
                experiments.evaluate()
                return {'summary': 'descriptor candidate', 'cost_usd': 0}
            with patch.object(driver.pc, 'resolve_detector_runs', return_value={}), \
                    patch.object(driver, 'ensure_native_tables', side_effect=lambda brain, *a, **k: banks[brain]), \
                    patch.object(driver, 'attach_local_context', return_value=None):
                path = asyncio.run(driver.run(args, revise_fn=revise))
            self.assertTrue(seen['environment']['enabled'])
            self.assertEqual(seen['environment']['workers'], 2)
            row = json.loads((path / 'ledger.jsonl').read_text().splitlines()[0])
            self.assertEqual(row['descriptors_referenced'], ['bank_agent_node_count'])
            self.assertEqual(row['descriptor_inference'][0]['brain'], '2')
            self.assertEqual(row['descriptor_runs'][0]['status'], 'registered')
            self.assertEqual(row['validation']['cells']['2/merge']['precision'], 1.)
            self.assertTrue(row['accepted'])
            manifest = json.loads((path / 'manifest.json').read_text())
            self.assertTrue(manifest['descriptor_compute']['enabled'])
            self.assertEqual(manifest['search_version'], 'agent-descriptor-compute-v14')
            registry = json.loads((path / 'descriptors.json').read_text())
            self.assertEqual(list(registry['columns']), ['bank_agent_node_count'])
            # Statistics are written before the agent registers columns in gen 1; the context column is there.
            self.assertIn('bank_context_available', (path / 'gen001' / 'feature_statistics.json').read_text())
            self.assertFalse((path / 'gen001' / 'train_feedback.json').read_text().lower().count('validation'))
            # Resume from the accepted scorer in a new run: descriptors are restored before the seed evaluation.
            args.start_from, args.generations = path / 'best_scorer.py', 0
            fresh = three_brain_cache(root / 'cache')  # same cache; fresh in-memory tables without columns
            with patch.object(driver.pc, 'resolve_detector_runs', return_value={}), \
                    patch.object(driver, 'ensure_native_tables', side_effect=lambda brain, *a, **k: fresh[brain]), \
                    patch.object(driver, 'attach_local_context', return_value=None), \
                    patch.object(dr, 'run_worker', side_effect=AssertionError('restore must hit the bank cache')):
                resumed = asyncio.run(driver.run(args, revise_fn=revise))
            self.assertEqual(json.loads((resumed / 'seed.json').read_text())['validation']['macro_precision'], 1.)
            self.assertEqual(json.loads((resumed / 'manifest.json').read_text())['descriptor_compute']['restored_descriptors'],
                             ['bank_agent_node_count'])
            # Without --context-cache the same resume is refused before any evaluation.
            args.context_cache = None
            with patch.object(driver.pc, 'resolve_detector_runs', return_value={}), \
                    patch.object(driver, 'ensure_native_tables', side_effect=lambda brain, *a, **k: three_brain_cache(root / 'cache')[brain]), \
                    patch.object(driver, 'attach_local_context', return_value=None), \
                    self.assertRaisesRegex(ValueError, 'pass --context-cache'):
                asyncio.run(driver.run(args, revise_fn=revise))

    def test_run_without_context_cache_has_no_descriptor_surface(self):
        from proofreader_evolve.cli import run_precision_evolution as driver
        banks = {**two_brains(), '2': brain_table('2', offset=5000.)}
        banks['2'].tables['merge'].meta['training_brain'] = '794495'
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            seed = Path(tmp) / 'seed.py'
            seed.write_text(FORMULA)
            args = driver.parse_args(['--train-brains', '794495,802449', '--validation-brains', '2', '--merge-k', '3',
                                      '--generations', '1', '--runs-dir', tmp, '--start-from', str(seed),
                                      '--skip-auxiliary-ablation', '--bootstrap-draws', '0'])
            async def revise(run_dir, policy, rules, report, model, *, experiments):
                self.assertFalse((policy.parent / 'descriptor.py').exists())
                self.assertEqual(json.loads((policy.parent / 'model_environment.json').read_text())['descriptor_compute'],
                                 {'enabled': False})
                with self.assertRaisesRegex(ValueError, 'no context cache'):
                    experiments.compute_descriptors('pilot')
                experiments.restore('parent')
                return {'summary': 'no descriptors', 'cost_usd': 0}
            with patch.object(driver.pc, 'resolve_detector_runs', return_value={}), \
                    patch.object(driver, 'ensure_native_tables', side_effect=lambda brain, *a, **k: banks[brain]), \
                    patch.object(driver, 'attach_local_context', return_value=None):
                path = asyncio.run(driver.run(args, revise_fn=revise))
            manifest = json.loads((path / 'manifest.json').read_text())
            self.assertEqual(manifest['descriptor_compute'], {'enabled': False})
            row = json.loads((path / 'ledger.jsonl').read_text().splitlines()[0])
            self.assertEqual(row['descriptor_runs'], [])
            self.assertEqual(row['descriptors_referenced'], [])


class ModelColumnAndFeedbackTests(unittest.TestCase):
    def test_frozen_models_record_columns_and_prediction_reindexes(self):
        program = ('import numpy as np\n'
                   'def fit(X_train, y_train, artifact_dir, params):\n'
                   "    np.save(artifact_dir / 'n.npy', np.array([X_train.shape[1]]), allow_pickle=False)\n"
                   'def predict(X, artifact_dir, params):\n'
                   "    assert list(X.columns) == ['detector_score', 'signal', 'row_key'], list(X.columns)\n"
                   "    return X['signal'].fillna(0).to_numpy(dtype=float)\n")
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root = Path(tmp)
            train = two_brains()
            source, _ = fit_classifier(train, 'merge', {'parameters': {}}, root / 'fit', program=program,
                                       artifact_store=root / 'models')
            model = frozen_model(source)
            self.assertEqual(model['columns'], ['detector_score', 'signal', 'row_key'])
            frame = train['802449'].tables['merge'].features.copy()
            frame['bank_agent_extra'] = 1.
            shuffled = frame[['bank_agent_extra', 'row_key', 'signal', 'detector_score']]
            values = predict_model(model, root / 'models', shuffled, 'merge')
            np.testing.assert_allclose(values, frame['signal'].to_numpy())
            with self.assertRaisesRegex(Exception, 'lacks'):
                predict_model(model, root / 'models', frame[['detector_score', 'signal']], 'merge')

    def test_feedback_prunes_all_missing_columns_before_shrinking_examples(self):
        examples = [{'label': label, 'score': .5, 'group': 'selected_positive' if label else 'selected_label0',
                     'features': {**{f'real_{j}': float(j + i) for j in range(5)},
                                  **{f'bank_agent_empty_{j}': None for j in range(300)}}}
                    for label in (0, 1) for i in range(4)]
        report = {'macro_precision': .5, 'cells': {'1/merge': {'precision': .5, 'tp': 1, 'fp': 1,
                                                                'training_examples': examples}}}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'feedback.json'
            full = feedback.write_train_feedback(path, report, [], {'merge': 2})
            self.assertEqual(json.loads(path.read_text())['columns_pruned'], False)
            with patch.object(feedback, 'MAX_FEEDBACK_BYTES', full['bytes'] - 1):
                pruned = feedback.write_train_feedback(path, report, [], {'merge': 2})
            data = json.loads(path.read_text())
            self.assertTrue(data['columns_pruned'])
            self.assertEqual(pruned['examples_per_group_limit'], 4)
            self.assertEqual(sorted(data['examples']['1/merge']['features']), [f'real_{j}' for j in range(5)])


class PreflightContextCacheTests(unittest.TestCase):
    def test_context_cache_readiness(self):
        from proofreader_evolve.cli import preflight
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            table = brain_table('802449').tables['merge']
            bank = SimpleNamespace(meta={'table_paths': {}}, tables={'merge': table})
            with patch.object(preflight.pc, 'resolve_detector_runs', return_value={}), \
                    patch.object(preflight, 'ensure_native_tables', return_value=bank):
                (root / 'cache').mkdir()
                missing = preflight.inspect_readiness(['802449'], context_cache=str(root / 'cache'))
                build_cache(root / 'cache', table, band_rows=12)
                ready = preflight.inspect_readiness(['802449'], context_cache=str(root / 'cache'))
        self.assertEqual(missing['status'], 'blocked')
        self.assertEqual(missing['context_cache']['brains']['802449']['merge']['status'], 'blocked')
        self.assertEqual(ready['status'], 'checks_passed')
        self.assertEqual(ready['context_cache']['brains']['802449']['merge']['band_rows'], 12)
