"""Local geometry boundaries, fixed-pool alignment and frozen extraction regressions."""
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from proofreader_evolve.harness import local_features as features
from proofreader_evolve.harness import fixed_pool_scoring as scoring
from proofreader_evolve.harness.classifier_contract import MODEL_VERSION, config_identity, render_model
from proofreader_evolve.harness.classifier_train_worker import extract
from proofreader_evolve.harness.classifier_training import fit_classifier
from proofreader_evolve.harness.local_context import (
    FragmentStore, LocalContextProvider, attach_local_context, cache_identity,
    candidate_ref, neighborhood, resolve_train_candidate,
)
from proofreader_evolve.harness.local_feature_contract import local_feature_spec
from proofreader_evolve.harness.model_execution import run_worker
from proofreader_evolve.harness.native_pool import NativeTable, NativeTables, pool_digest
from proofreader_evolve.harness.train_experiments import TrainingExperiments
from proofreader_evolve.harness.training_diagnostics import describe


PROGRAM = '''LOCAL_CONTEXT = {"feature_names": ["local_size"], "max_candidates": 2}
def extract_local_features(context):
    return {"local_size": float(len(context["sites"][0]["xyz_um"]))}
def score_candidates(features, ctx):
    return features["local_size"].fillna(0).to_numpy()
def fit(X_train, y_train, artifact_dir, params):
    pass
def predict(X, artifact_dir, params):
    return X["local_size"].fillna(0).to_numpy()
'''


def graph_fixture():
    graph = nx.Graph([(0, 1), (1, 2), (3, 4)])
    graph.node_xyz = np.array([[0., 0., 0.], [2., 0., 0.], [4., 1., 0.],
                               [7., 0., 0.], [9., 0., 0.]])
    graph.node_radius = np.ones(5)
    graph.node_segment_id = lambda node: 'private_segment_a' if node < 3 else 'private_segment_b'
    nx.set_node_attributes(graph, 'private_gt_annotation', 'gt_label')
    graph.graph['gt'] = 'private_gt_graph'
    return graph


def table_fixture(root, brain='1'):
    path = root / f'{brain}_cache.pkl'
    path.write_bytes(b'Trusted fixture placeholder, never unpickled')
    rows = [{'segment_id_a': i + 1, 'segment_id_b': i + 10, 'node_id_a': 2, 'node_id_b': 3,
             'occurrences': [{'node_id_a': 2, 'node_id_b': 3}, {'node_id_a': 1, 'node_id_b': 4}]}
            for i in range(3)]
    table = NativeTable('split', pd.DataFrame({'detector_score': [.3, .9, .5]}), rows,
                        np.array([0, 1, 0]), {'pool_sha256': pool_digest('split', rows),
                        'files': {'features.pkl': 'fixture-feature-digest'},
                        'provenance': {'source_cache': cache_identity(path), 'brain': brain, 'mcl': 100}})
    graph = graph_fixture()
    store = FragmentStore(root / 'feature_cache')
    store.graph = Mock(return_value=(graph, cKDTree(graph.node_xyz)))
    table.local_context = LocalContextProvider(store, table)
    return table


def manifest(program):
    return render_model({'version': MODEL_VERSION, 'kind': 'split', 'program': program,
                         'config': {'parameters': {}}, 'training_brains': ['1'], 'training_rows': 3,
                         'training_fingerprint': '0' * 64, 'files': {},
                         'artifact_sha256': config_identity({}), 'threads': 1})


class GeometryTests(unittest.TestCase):
    def test_relative_coordinates_and_whitelisted_fields(self):
        graph = graph_fixture()
        data, center = neighborhood(graph, cKDTree(graph.node_xyz), [2, 3], 50, 128)
        self.assertEqual(data['anchor_nodes'], [0, 1])
        self.assertEqual(data['segment'][:2], [0, 1])
        self.assertNotIn('private', json.dumps(data))
        self.assertNotIn('gt_label', json.dumps(data))
        self.assertEqual(set(data), {'xyz_um', 'radius_um', 'degree', 'segment', 'edges',
                                    'anchor_nodes', 'outside_radius', 'truncated'})
        graph.node_xyz += [100., -50., 20.]
        moved, shifted = neighborhood(graph, cKDTree(graph.node_xyz), [2, 3], 50, 128)
        self.assertEqual(data, moved)
        np.testing.assert_allclose(np.array(shifted) - center, [100, -50, 20])

    def test_anchors_survive_truncation_and_small_radius(self):
        graph = graph_fixture()
        data, _ = neighborhood(graph, cKDTree(graph.node_xyz), [0, 4], 50, 2)
        self.assertTrue(data['truncated'])
        self.assertEqual(data['anchor_nodes'], [0, 1])
        self.assertEqual(len(data['xyz_um']), 2)
        self.assertEqual(data['edges'], [])
        narrow, _ = neighborhood(graph, cKDTree(graph.node_xyz), [0, 4], 1, 128)
        self.assertTrue(all(narrow['outside_radius'][:2]))

    def test_split_occurrences_and_merge_location(self):
        with tempfile.TemporaryDirectory() as tmp:
            table = table_fixture(Path(tmp))
            geometry, _ = table.local_context.context(0, max_occurrences=2)
            self.assertEqual(geometry['occurrences_used'], 2)
            self.assertFalse(geometry['occurrences_truncated'])
            second, _ = table.local_context.context(0, occurrence_index=1)
            self.assertEqual(geometry['sites'][1], second['sites'][0])
            with self.assertRaises(ValueError):
                table.local_context.context(0, occurrence_index=-1)
            table.kind, table.candidates = 'merge', [{'node_id': 1}]
            merged, centers = table.local_context.context(0)
            self.assertEqual(centers, [[2., 0., 0.]])
            self.assertEqual(merged['sites'][0]['anchor_nodes'], [0])

    def test_train_handles_and_inspection_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            table = table_fixture(root)
            bank = NativeTables('1', {'split': table}, {})
            train = {'1': bank}
            ref = candidate_ref(table, 1)
            self.assertEqual(resolve_train_candidate(train, ref)[2], 1)
            example = describe(table, np.array([.3, .9, .5]), np.array([1]))
            self.assertIn(ref, [e['candidate_ref'] for e in example['training_examples']])
            with self.assertRaises(ValueError):
                resolve_train_candidate(train, 'ctx:heldout:split:1')
            heldout = table_fixture(root, '2')
            with self.assertRaises(ValueError):
                resolve_train_candidate(train, candidate_ref(heldout, 1))
            session = TrainingExperiments.__new__(TrainingExperiments)
            session.train, session.gen_dir, session.inspections_used = train, root, 0
            session.trace = Mock()
            for i in range(8):
                result = session.inspect_candidate(ref, 50., 1)
                self.assertEqual(result['inspections_remaining'], 7 - i)
                self.assertNotIn('private', json.dumps(result))
            self.assertTrue((root / 'context_inspections/inspection008.json').is_file())
            with self.assertRaises(ValueError):
                session.inspect_candidate(ref)

    def test_context_attachment_is_lazy_and_stale_cache_fails_before_loading(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            table = table_fixture(root)
            with patch('proofreader_evolve.harness.local_context.load_cached_graphs') as load:
                store = attach_local_context({'1': NativeTables('1', {'split': table}, {})}, root / 'cache')
                load.assert_not_called()
                Path(table.meta['provenance']['source_cache']['path']).write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError, 'changed'):
                    store.graph(table.meta['provenance'])
                load.assert_not_called()


class FeatureTransportTests(unittest.TestCase):
    def test_static_contract_is_shared_by_formula_and_frozen_model(self):
        self.assertEqual(local_feature_spec(PROGRAM), local_feature_spec(manifest(PROGRAM)))
        with self.assertRaises(ValueError):
            local_feature_spec(PROGRAM.replace('"max_candidates": 2', '"max_candidates": 20001'))
        with self.assertRaises(ValueError):
            local_feature_spec(PROGRAM.replace('local_size', 'detector_score'))
        self.assertIsNone(local_feature_spec('raise RuntimeError("must not execute")'))

    def test_cached_transport_has_no_gt_and_preserves_all_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            table = table_fixture(root)
            original = table.features.copy(deep=True)
            calls = []
            def worker(folder, request, **kwargs):
                self.assertEqual(request['mode'], 'extract')
                self.assertFalse((folder / 'y.npy').exists())
                self.assertFalse((folder / 'X.npy').exists())
                contexts = [json.loads(line) for line in (folder / 'contexts.jsonl').read_text().splitlines()]
                self.assertNotIn('private', json.dumps(contexts))
                self.assertTrue(all(set(c) == {'version', 'kind', 'radius_um', 'occurrences_total',
                    'occurrences_used', 'occurrences_truncated', 'sites', 'features'} for c in contexts))
                calls.append(contexts)
                np.save(folder / 'local_features.npy', np.array([[2.], [4.]]), allow_pickle=False)
                return {'wall_seconds': .01}
            with patch.object(features, 'run_worker', side_effect=worker):
                enriched = features.augmented_features(PROGRAM, table)
                table.truth[:] = 1  # GT must not influence context selection or the feature cache.
                cached = features.augmented_features(manifest(PROGRAM), table)
                self.assertEqual(len(calls), 1)
                self.assertTrue(cached.attrs['local_context']['cached'])
                np.testing.assert_array_equal(enriched['local_size'], [np.nan, 2., 4.])
                np.testing.assert_array_equal(enriched['local_context_available'], [0, 1, 1])
                pd.testing.assert_frame_equal(table.features, original)
                self.assertEqual(len(enriched), len(table.truth))
                changed = PROGRAM.replace('return {"local_size": float', 'return {"local_size": 2 * float')
                features.augmented_features(changed, table)
                self.assertEqual(len(calls), 2)
            Path(table.meta['provenance']['source_cache']['path']).write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'changed'):
                features.augmented_features(PROGRAM, table)

    def test_selection_is_stable_for_ties_and_missing_values(self):
        spec = local_feature_spec(PROGRAM)[1]
        frame = pd.DataFrame({'detector_score': [np.nan, 2., 2., 1.]})
        np.testing.assert_array_equal(features.select_rows(frame, spec), [1, 2])
        spec['selection_largest'] = False
        np.testing.assert_array_equal(features.select_rows(frame, spec), [3, 1])

    def test_worker_rejects_nondeterministic_and_malformed_features(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'contexts.jsonl').write_text('{}\n')
            request = {'feature_names': ['local_size'], 'rows': 1}
            for function in (Mock(side_effect=[{'local_size': 1}, {'local_size': 2}]),
                             lambda c: {'wrong': 2}, lambda c: {'local_size': [1]},
                             lambda c: {'local_size': float('inf')}):
                with self.subTest(function=function), self.assertRaises(ValueError):
                    extract(SimpleNamespace(extract_local_features=function), root, request, io.BytesIO(), np)

    def test_extraction_sandbox_denies_labels_model_artifacts_and_full_feature_matrix(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'artifacts').mkdir()
            for name in ('y.npy', 'X.npy', 'artifacts/private.txt'):
                (root / name).write_bytes(b'Private data that extraction must not read')
            (root / 'contexts.jsonl').write_text('{}\n')
            (root / 'training.py').write_text('''def extract_local_features(context):
    from pathlib import Path
    for name in ("y.npy", "X.npy", "artifacts/private.txt"):
        try:
            Path(name).read_bytes()
        except PermissionError:
            pass
        else:
            raise RuntimeError("Extraction accessed private input: " + name)
    return {"local_size": 1.0}
''')
            run_worker(root, {'mode': 'extract', 'feature_names': ['local_size'], 'rows': 1,
                              'timeout': 20, 'memory_mb': 8192, 'threads': 1})
            np.testing.assert_array_equal(np.load(root / 'local_features.npy'), [[1.]])

    def test_outer_scoring_augments_train_and_validation_without_changing_k(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            banks = {brain: NativeTables(brain, {'split': table_fixture(root, brain)}, {}) for brain in ('1', '2')}
            seen = []
            def augment(source, table, timeout):
                seen.append((source, table.meta['provenance']['brain']))
                return table.features.assign(local_size=[0., 1., 0.])
            def scorer(source, frame, kind, timeout):
                self.assertNotIn('label', frame)
                return frame['local_size'].to_numpy()
            with patch.object(scoring, 'augmented_features', side_effect=augment), patch.object(scoring, 'score', side_effect=scorer):
                report = scoring.evaluate(PROGRAM, banks, {'split': 1})
            self.assertEqual(seen, [(PROGRAM, '1'), (PROGRAM, '2')])
            for cell in report['cells'].values():
                self.assertEqual((cell['effective_k'], cell['pool_size'], cell['tp']), (1, 3, 1))

    def test_fit_receives_augmented_train_only_and_freezes_extractor(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            table = table_fixture(root)
            bank = NativeTables('1', {'split': table}, {})
            augmented = table.features.assign(local_size=[np.nan, 2., 4.], local_context_available=[0., 1., 1.])
            def worker(folder, request, **kwargs):
                self.assertEqual(request['mode'], 'fit')
                self.assertIn('local_size', request['columns'])
                np.testing.assert_array_equal(np.load(folder / 'y.npy'), table.truth)
                self.assertEqual(np.load(folder / 'X.npy').shape, (3, 3))
                return {'wall_seconds': .01}
            with patch('proofreader_evolve.harness.classifier_training.augmented_features', return_value=augmented), \
                 patch('proofreader_evolve.harness.classifier_training.run_worker', side_effect=worker):
                source, summary = fit_classifier({'1': bank}, 'split', {'parameters': {}}, root / 'fit',
                    program=PROGRAM, artifact_store=root / 'models')
            self.assertEqual(local_feature_spec(source), local_feature_spec(PROGRAM))
            self.assertEqual(summary['training_brains'], ['1'])
            self.assertIn('local_size', summary['input_columns'])


if __name__ == '__main__':
    unittest.main()
