"""On-demand contexts: whole-pool coverage, no saved patches, and frozen-input parity."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from proofreader_evolve.harness.descriptor_context import DescriptorContexts
from proofreader_evolve.harness.descriptor_contract import descriptor_spec
from proofreader_evolve.harness.descriptor_runs import DescriptorRuns
from proofreader_evolve.harness import descriptor_inputs, detector_prior
from proofreader_evolve.tests.test_descriptor_compute import FakeProvider, GEOMETRY_DESCRIPTOR, IMAGE_DESCRIPTOR
from proofreader_evolve.tests.test_descriptor_probe import attach_synthetic_store, CUSTOM
from proofreader_evolve.tests.test_selection_protocol import brain_table
from proofreader_evolve.tests.label_fixture import setUpModule, tearDownModule  # noqa: F401


class OnDemandContextTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.table = brain_table('802449').tables['merge']
        self.table.local_context = FakeProvider()
        self.contexts = DescriptorContexts()
        self.runs = DescriptorRuns(self.contexts, self.root / 'bank', workers=2, memory_mb=1024)
        self.store, self.source = attach_synthetic_store(self.table, self.root / 'images')

    def test_full_pool_includes_rows_outside_old_band_without_context_files(self):
        program, spec = descriptor_spec(GEOMETRY_DESCRIPTOR)
        with patch('proofreader_evolve.harness.context_cache.ContextCache.entry', side_effect=AssertionError('offline cache used')):
            result = self.runs.compute(self.table, program, spec, 'all_candidates', k=3, wall_budget=120,
                                       log_dir=self.root / 'logs')
            probe = self.runs.probe(self.table, program, spec, [15], log_dir=self.root / 'probe')
        self.assertEqual(result['rows'].tolist(), list(range(16)))
        np.testing.assert_array_equal(probe['values'], result['values'][[15]])
        self.runs.register(self.table, spec['columns'], result['rows'], result['values'])
        self.assertTrue(np.isfinite(self.table.features['bank_agent_node_count']).all())
        self.assertNotIn('bank_context_available', self.table.features)
        # Historical scope spelling resolves to the same numeric result, independent of any context directory.
        self.assertTrue(self.runs.cached(self.table, spec, 'all_cached', 3))
        self.assertEqual(self.runs.directory(self.table, spec, 'all_cached'),
                         self.runs.directory(self.table, spec, 'all_candidates'))
        self.assertFalse((self.root / 'images').exists())

    def test_uniform_sampling_ignores_rank_and_labels_and_covers_tail(self):
        before = self.runs.sample_rows(self.table, 16, 8)
        self.table.truth = 1 - self.table.truth
        self.table.features['detector_score'] *= -1
        np.testing.assert_array_equal(before, self.runs.sample_rows(self.table, 16, 8))
        self.assertEqual(before.tolist(), list(range(16)))
        self.assertTrue(self.contexts.entry(self.table).has_rows([15]).all())
        self.assertFalse(self.contexts.entry(self.table).has_rows([-1, 16]).any())

    def test_legacy_image_tier_and_explicit_request_both_read_source(self):
        _, legacy = descriptor_spec(IMAGE_DESCRIPTOR)
        _, explicit = descriptor_spec(IMAGE_DESCRIPTOR.replace("'image_tier': 'level1'",
            "'image': {'level': 1, 'radius_um': 30}"))
        entry = self.contexts.entry(self.table)
        with tempfile.TemporaryDirectory() as tmp:
            a = next(descriptor_inputs.iter_inputs(self.table, entry, legacy, [15], Path(tmp)))
            b = next(descriptor_inputs.iter_inputs(self.table, entry, explicit, [15], Path(tmp)))
            np.testing.assert_array_equal(a[2]['image_zyx'], b[2]['image_zyx'])
            self.assertFalse(list(Path(tmp).rglob('patch.npz')))
        self.assertEqual(self.source.requests, [(1, 30.), (1, 30.)])
        self.assertEqual(self.store.cache_bytes, 0)
        self.assertNotIn('row_in_band', a[1])
        self.assertNotIn('source_cache', json.dumps(a[1]))
        np.testing.assert_allclose(np.asarray(a[1]['fragment']['nodes_zyx'])[0], a[2]['anchors_zyx'][0])

    def test_custom_probe_compute_and_frozen_table_parity_no_persistent_pixels(self):
        program, spec = descriptor_spec(CUSTOM)
        measured = self.runs.probe(self.table, program, spec, [0, 15], log_dir=self.root / 'probe')
        full = self.runs.compute(self.table, program, spec, 'all_candidates', k=3, wall_budget=120,
                                 log_dir=self.root / 'full')
        heldout = brain_table('heldout').tables['merge']
        heldout.local_context = FakeProvider()
        heldout_store, _ = attach_synthetic_store(heldout, self.root / 'heldout_images')
        frozen = self.runs.compute(heldout, program, spec, 'all_candidates', k=3, wall_budget=120,
                                   log_dir=self.root / 'validation_descriptor_runs')
        np.testing.assert_array_equal(measured['values'], full['values'][[0, 15]])
        np.testing.assert_array_equal(full['values'], frozen['values'])
        self.assertEqual((self.store.cache_bytes, heldout_store.cache_bytes), (0, 0))
        self.assertFalse(list(self.root.rglob('patch.npz')))
        self.source.identity = {'changed': True}
        self.assertFalse(self.runs.cached(self.table, spec, 'all_candidates', 3))

    def test_raw_features_only_and_no_descriptor_recursion(self):
        from types import SimpleNamespace
        for row in self.table.candidates:
            row['candidate_family'] = 'junction'
        detector_prior.strip_to_raw({'802449': SimpleNamespace(tables={'merge': self.table})})
        self.table.features['bank_agent_previous'] = 99.
        entry = self.contexts.entry(self.table)
        contexts, _ = entry.contexts(self.table, [15])
        self.assertTrue(all(c.startswith('raw_') for c in contexts[0]['features']))
        self.assertNotIn('row_in_band', contexts[0])
        self.assertNotIn('bank_agent_previous', contexts[0]['features'])

    def test_context_rows_are_positions_even_with_nondefault_frame_index(self):
        self.table.features.index = np.arange(100, 116)
        expected = float(self.table.features.iloc[15]['signal'])
        contexts, _ = self.contexts.entry(self.table).contexts(self.table, [15])
        self.assertEqual(contexts[0]['features']['signal'], expected)

    def test_source_failure_is_explicit_and_does_not_persist_results(self):
        program, spec = descriptor_spec(CUSTOM)
        with patch.object(self.store, 'patch', side_effect=OSError('source unavailable')):
            with self.assertRaisesRegex(Exception, 'source unavailable'):
                self.runs.compute(self.table, program, spec, 'all_candidates', k=3, wall_budget=120,
                                  log_dir=self.root / 'failed')
        self.assertFalse(self.runs.directory(self.table, spec, 'all_candidates').exists())

    def test_human_plot_uses_live_inputs_for_a_tail_candidate(self):
        from proofreader_evolve.plotting.plot_error_contexts import live_patch, plot_kind
        import matplotlib.pyplot as plt
        entry = self.contexts.entry(self.table)
        self.table.candidates[15]['segment_id'] = 1
        context, pixels, origin, shape = live_patch(self.table, entry, 15, 'level0')
        self.assertEqual(context['level'], 0)
        np.testing.assert_array_equal(shape, pixels['image_zyx'].shape)
        self.assertTrue((shape > 0).all())
        graph = self.store.source(self.table)[1]
        with patch('proofreader_evolve.plotting.plot_error_contexts.draw_ground_truth_panel'):
            figure, picks = plot_kind('merge', self.table, entry, (graph, object(), np.ones(3)), None,
                                     '802449', examples=1, rows=[15], out=self.root / 'plot.png')
        plt.close(figure)
        self.assertEqual(picks, [15])
        self.assertTrue((self.root / 'plot.png').is_file())
        self.assertEqual(self.store.cache_bytes, 0)

    def test_offset_diagnostics_use_live_nodes_and_uniform_full_pool(self):
        from scripts.check_fragment_image_offset import iter_sites
        with tempfile.TemporaryDirectory() as tmp:
            samples = list(iter_sites(self.table, 'level0', 16, Path(tmp)))
            self.assertFalse(list(Path(tmp).rglob('patch.npz')))
        self.assertEqual([site['row'] for site in samples], list(range(16)))
        self.assertEqual(len(self.source.requests), 16)
        entry = self.contexts.entry(self.table)
        with tempfile.TemporaryDirectory() as tmp:
            _, context, pixels = next(descriptor_inputs.iter_inputs(
                self.table, entry, {'inputs': 'both', 'image_tier': 'level0'}, [15], Path(tmp)))
        inside = ~np.asarray(context['fragment']['outside_radius'], dtype=bool)
        np.testing.assert_allclose(samples[-1]['nodes_zyx'], np.asarray(context['fragment']['nodes_zyx'])[inside])
        np.testing.assert_array_equal(samples[-1]['image'], pixels['image_zyx'])
        before = [site['row'] for site in iter_sites(self.table, 'level0', 5, self.root, seed=9)]
        self.table.features['detector_score'] *= -1
        self.table.truth = 1 - self.table.truth
        after = [site['row'] for site in iter_sites(self.table, 'level0', 5, self.root, seed=9)]
        self.assertEqual(before, after)
        self.assertEqual(self.store.cache_bytes, 0)

    def test_judgeable_scope_is_the_labeled_rows_and_measures_like_the_full_pool(self):
        from proofreader_evolve.harness.fixed_pool_scoring import rank_metrics
        from proofreader_evolve.harness.label_availability import labeled_rows
        truth = np.asarray(self.table.truth, dtype=float)
        truth[[0, 5, 10]] = np.nan  # rows GT cannot judge
        self.table.truth = truth
        labeled = labeled_rows(truth)
        unlabeled = np.setdiff1d(np.arange(len(truth)), labeled)
        program, spec = descriptor_spec(GEOMETRY_DESCRIPTOR)
        entry = self.runs.contexts.entry(self.table)
        np.testing.assert_array_equal(self.runs.scope_rows(entry, 'judgeable', 3, truth=truth), labeled)
        with self.assertRaisesRegex(ValueError, 'labels'):
            self.runs.scope_rows(entry, 'judgeable', 3)
        partial = self.runs.compute(self.table, program, spec, 'judgeable', k=3, wall_budget=120,
                                    log_dir=self.root / 'judgeable')
        full = self.runs.compute(self.table, program, spec, 'all_candidates', k=3, wall_budget=120,
                                 log_dir=self.root / 'full')
        np.testing.assert_array_equal(partial['rows'], labeled)
        np.testing.assert_array_equal(partial['values'], full['values'][labeled])
        probe = self.runs.probe(self.table, program, spec, labeled[:2], log_dir=self.root / 'probe')
        np.testing.assert_array_equal(probe['values'], partial['values'][:2])
        # Registered on the judgeable rows only: NaN elsewhere, and a row-wise ranking cannot tell the
        # difference because rank_metrics drops unlabeled rows before ordering (the next test shows the
        # two paths that can).
        self.runs.register(self.table, spec['columns'], partial['rows'], partial['values'])
        column = self.table.features['bank_agent_node_count'].to_numpy()
        self.assertTrue(np.isfinite(column[labeled]).all())
        self.assertTrue(np.isnan(column[unlabeled]).all())
        from_partial, chosen_partial = rank_metrics(np.nan_to_num(column, nan=-1e9), truth, self.table.keys, 3)
        from_full, chosen_full = rank_metrics(full['values'][:, 0], truth, self.table.keys, 3)
        self.assertEqual(from_partial, from_full)
        np.testing.assert_array_equal(chosen_partial, chosen_full)
        # The bank entry is keyed by the labeled row set: one more judgeable row invalidates it.
        self.assertTrue(self.runs.cached(self.table, spec, 'judgeable', 3))
        truth[unlabeled[0]] = 0.
        self.table.truth = truth
        self.assertFalse(self.runs.cached(self.table, spec, 'judgeable', 3))
        # Frozen inference on another brain computes that brain's own labeled rows.
        heldout = brain_table('heldout').tables['merge']
        heldout.local_context = FakeProvider()
        other = np.asarray(heldout.truth, dtype=float)
        other[[1, 2]] = np.nan
        heldout.truth = other
        frozen = self.runs.compute(heldout, program, spec, 'judgeable', k=3, wall_budget=120,
                                   log_dir=self.root / 'validation_descriptor_runs')
        np.testing.assert_array_equal(frozen['rows'], labeled_rows(other))

    def test_judgeable_rows_are_visible_to_whole_column_statistics_and_unlabeled_fits(self):
        """What the judgeable scope does not guarantee: a formula's whole-column statistics and a fit's
        X_unlabeled rows see NaN outside the labeled rows, so they differ from an all_candidates run."""
        from proofreader_evolve.harness.classifier_contract import frozen_model
        from proofreader_evolve.harness.classifier_training import fit_classifier, verify_artifacts
        from proofreader_evolve.harness.isolated_scoring import score
        from proofreader_evolve.harness.label_availability import labeled_rows
        from proofreader_evolve.harness.native_pool import NativeTables
        truth = np.asarray(self.table.truth, dtype=float)
        truth[[0, 5, 10]] = np.nan
        self.table.truth = truth
        labeled = labeled_rows(truth)
        program, spec = descriptor_spec(GEOMETRY_DESCRIPTOR)
        partial = self.runs.compute(self.table, program, spec, 'judgeable', k=3, wall_budget=120,
                                    log_dir=self.root / 'judgeable')
        full = self.runs.compute(self.table, program, spec, 'all_candidates', k=3, wall_budget=120,
                                 log_dir=self.root / 'full')
        self.runs.register(self.table, spec['columns'], partial['rows'], partial['values'])
        judgeable_frame = self.table.features
        full_frame = judgeable_frame.copy()
        for position, column in enumerate(spec['columns']):
            full_frame[column] = full['values'][:, position]
        # 1. The scorer receives the whole table: a z-score over the column uses the computed rows only.
        formula = """import numpy as np

def score_candidates(features, ctx):
    column = features['bank_agent_node_count'].to_numpy(dtype=float)
    centred = (column - np.nanmean(column)) / np.nanstd(column)
    return features['signal'].to_numpy(dtype=float) + np.nan_to_num(centred, nan=0.)
"""
        from_judgeable = score(formula, judgeable_frame, 'merge', 60)
        from_full = score(formula, full_frame, 'merge', 60)
        self.assertTrue(np.isfinite(from_judgeable).all() and np.isfinite(from_full).all())
        self.assertFalse(np.allclose(from_judgeable[labeled], from_full[labeled]))
        # 2. A fit that declares X_unlabeled gets finite descriptor values on X_train and NaN on every
        # unlabeled row it is handed.
        training = """import numpy as np

def fit(X_train, y_train, artifact_dir, params, X_unlabeled=None):
    np.save(artifact_dir / 'seen.npy', np.array([
        np.isfinite(X_train['bank_agent_node_count']).all(), len(X_unlabeled),
        np.isnan(X_unlabeled['bank_agent_node_count']).all()]), allow_pickle=False)


def predict(X, artifact_dir, params):
    return X['signal'].fillna(0).to_numpy()
"""
        bank = NativeTables('802449', {'merge': self.table}, {})
        source, summary = fit_classifier({'802449': bank}, 'merge', {'parameters': {}}, self.root / 'fit',
                                         program=training, artifact_store=self.root / 'store')
        seen = np.load(verify_artifacts(frozen_model(source), self.root / 'store') / 'seen.npy')
        self.assertEqual(seen.tolist(), [1., 3., 1.])
        self.assertEqual(summary['unlabeled_rows'], 3)

    def test_fragment_source_is_loaded_once_under_concurrency(self):
        from types import SimpleNamespace
        import threading
        import time
        from proofreader_evolve.harness.local_context import FragmentStore, cache_identity
        path = self.root / 'source.pkl'
        path.write_bytes(b'synthetic source identity')
        graph = SimpleNamespace(node_xyz=np.zeros((1, 3)), kdtree=object())
        count = []
        barrier = threading.Barrier(4)
        def load(*args, **kwargs):
            count.append(1)
            time.sleep(.02)
            return graph, None, {}
        store = FragmentStore(self.root)
        provenance = {'source_cache': cache_identity(path), 'brain': '1', 'mcl': 100}
        def read(_):
            barrier.wait()
            return store.graph(provenance)[0]
        with patch('proofreader_evolve.harness.local_context.load_cached_graphs', side_effect=load):
            with ThreadPoolExecutor(max_workers=4) as pool:
                results = list(pool.map(read, range(4)))
        self.assertEqual(len(count), 1)
        self.assertTrue(all(item is graph for item in results))
