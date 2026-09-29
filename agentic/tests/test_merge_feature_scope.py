"""Scope contracts through semantic compilation, assembly, and real extraction."""
import importlib.util
import json
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

import numpy as np

from agentic.detector_build.assembly import assemble_detector
from agentic.detector_build.contracts import DetectorTarget
from agentic.detector_build.feature_scope import inventory_feature_scopes
from agentic.detector_build.inventory import compile_feature_inventory, validate_semantic_draft
from agentic.tests.test_merge_site_target import (
    MERGE_SITE_POLICY, _merge_site_payload, workflow,
)


def definitions():
    common = dict(constants={}, traversal_phase="junction_site",
                  measurable_condition="finite node radius", historical_undefined_sentinel=None,
                  source_feature="max_radius", source_aggregation="max over segment nodes")
    return [dict(common, name="candidate_radius", quantity="radius at candidate node",
                 aggregation="none", reduction="identity at candidate",
                 scope="candidate", adaptation="candidate_local"),
            dict(common, name="max_radius", quantity="maximum segment node radius",
                 aggregation="segment max", reduction="max",
                 scope="segment", adaptation="identity")]


FRAGMENT = '''
FEATURE_REGISTRY = [('candidate_radius',), ('max_radius',)]
ANALYSIS_TIMING_GROUPS = [
    {'key': 'local', 'hypothesis_ids': [1], 'phase': 'junction_site',
     'feature_names': ['candidate_radius']},
    {'key': 'context', 'hypothesis_ids': [2], 'phase': 'segment',
     'feature_names': ['max_radius']},
]
COMPUTATION_PLAN = [
    {'primitive': 'node_radius_lookup', 'cost_class': 'constant', 'bound': '',
     'amortization': '', 'consumers': ['local']},
    {'primitive': 'segment_radius_max', 'cost_class': 'global_scan', 'bound': '',
     'amortization': 'one-time pre-pass before candidate iteration', 'consumers': ['context']},
]
def extract_features(payload, verbose=True, timing=None, enabled_analysis_keys=None,
                     profile_segment_limit=None, image_workers=1):
    def _analysis_enabled(key):
        return enabled_analysis_keys is None or key in enabled_analysis_keys
    samples, labels = build_sample_universe(payload)
    if profile_segment_limit is not None:
        segments = list(dict.fromkeys(s['segment_id'] for s in samples))[:profile_segment_limit]
        samples = [s for s in samples if s['segment_id'] in segments]
        labels = labels[:len(samples)]
    acc = FeatureAccumulator(samples)
    graph = payload['fragments_graph']
    if _analysis_enabled('context'):
        if timing is not None:
            timing.start_phase('segment')
            timing.start_analysis('context')
        comp_to_seg = build_comp_to_seg(graph)
        maxima = {}
        for node in graph:
            seg = comp_to_seg[int(graph.node_component_id[node])]
            value = float(graph.node_radius[node])
            if np.isfinite(value):
                maxima[seg] = max(maxima.get(seg, value), value)
        for seg in {s['segment_id'] for s in samples}:
            if seg in maxima:
                acc.set_segment('max_radius', seg, maxima[seg])
    if _analysis_enabled('local'):
        if timing is not None:
            timing.start_phase('junction_site')
            timing.start_analysis('local')
        for sample in samples:
            value = float(graph.node_radius[sample['node_id']])
            if np.isfinite(value):
                acc.set_candidate('candidate_radius', sample['candidate_id'], value)
    return samples, labels, acc
'''


class MergeFeatureScopeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.inventory = self.root / 'feature_inventory.json'
        self.data = {'schema_version': 4, 'hypotheses': [
            {'included': True, 'features': definitions()}]}
        self.inventory.write_text(json.dumps(self.data))
        self.policy = self.root / 'merge_candidate_policy.json'
        self.policy.write_text(json.dumps({
            'schema_version': 1, 'artifact_type': 'merge_site_candidate_policy_selection',
            'application_status': 'selected_for_detector_runtime',
            'selected_policy': MERGE_SITE_POLICY,
        }))
        self.feature = self.root / 'features.py'
        self.feature.write_text(FRAGMENT)
        self.output = self.root / 'detector.py'

    def assemble(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            assemble_detector(workflow.RUNTIME_TEMPLATE_PATH, self.feature, self.output,
                              target=DetectorTarget.MERGE_SITE,
                              candidate_policy_path=self.policy, inventory_path=self.inventory)

    def module(self):
        self.assemble()
        spec = importlib.util.spec_from_file_location('scope_test_detector', self.output)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def payload(self):
        payload = _merge_site_payload()
        graph = payload['fragments_graph']
        # Two distant junctions, from two fragments belonging to the SAME segment.
        graph.component_id_to_swc_id[20] = '111.1.swc'
        graph.node_radius = np.ones(len(graph))
        graph.node_radius[2] = 2.
        graph.node_radius[8] = 7.
        return payload

    def test_semantics_compile_and_validate_v4(self):
        rerun = self.root / 'run.rerun'
        rerun.mkdir()
        (rerun / 'hypo_1.py').write_text('MAX_RADIUS = 1\n')
        summary = self.root / 'run.summary.md'
        summary.write_text('### 1. Radius\n- **Run:** run · **ID:** 1 · **Direction:** Positive\n'
                           '- **Reproduction:** REPRODUCED\n- **Verdict:** SOUND\n')
        draft = self.root / 'semantic.json'
        draft.write_text(json.dumps({'schema_version': 1, 'hypotheses': [{
            'id': 1, 'included': True, 'exclusion_reason': None, 'correction_scope': 'none',
            'feature_source': 'rerun', 'source_reason': 'Source local statistic is available.',
            'features': definitions(),
        }]}))
        compile_feature_inventory(draft, self.inventory, selected_ids=[1], selection_manifest=None,
                                  summary_path=summary, corrected_results_path=None, rerun_dir=rerun,
                                  fixed_dir=None, project_root=self.root, merge_site=True)
        with patch.object(workflow, 'PROJECT_ROOT', self.root):
            workflow.validate_inventory(self.inventory, [1], None, rerun, None, summary,
                                        merge_site=True)
        compiled = json.loads(self.inventory.read_text())
        self.assertEqual(compiled['schema_version'], 4)
        self.assertEqual(compiled['hypotheses'][0]['features'], definitions())
        # Base schemas must not silently accept scope declarations for another target.
        with self.assertRaisesRegex(SystemExit, 'exactly'):
            validate_semantic_draft(draft, selected_ids=[1], project_root=self.root)

    def test_rejects_legacy_context_only_and_unnamed_adaptations(self):
        for mutate in (
            lambda d: d.update(schema_version=2),
            lambda d: d['hypotheses'][0].update(features=[definitions()[1]]),
            lambda d: d['hypotheses'][0]['features'][0].update(scope='segment'),
            lambda d: d['hypotheses'][0]['features'][0].update(name='max_radius'),
            lambda d: d['hypotheses'][0]['features'][0].pop('source_aggregation'),
        ):
            data = json.loads(json.dumps(self.data))
            mutate(data)
            with self.subTest(data=data), self.assertRaises(SystemExit):
                inventory_feature_scopes(data)

    def test_assembly_rejects_custom_storage_and_wrong_scope(self):
        for fragment, error in (
            (FRAGMENT + '\nclass FeatureAccumulator: pass\n', 'runtime-owned'),
            (FRAGMENT.replace("set_candidate('candidate_radius'", "set_segment('candidate_radius'"),
             'cannot use set_segment'),
            (FRAGMENT.replace('candidate_radius', 'other_radius'), 'FEATURE_REGISTRY'),
        ):
            with self.subTest(error=error), self.assertRaisesRegex(SystemExit, error):
                self.feature.write_text(fragment)
                self.assemble()

    def test_same_segment_retains_local_geometry_and_explicit_context(self):
        module = self.module()
        rows, _, acc = module._extract_features_runtime(self.payload(), verbose=False)
        self.assertEqual([s['node_id'] for s in rows], [2, 8])
        self.assertEqual([s['segment_id'] for s in rows], [111, 111])
        frame = acc.to_frame()
        self.assertEqual(frame.candidate_radius.tolist(), [2., 7.])
        self.assertEqual(frame.max_radius.tolist(), [7., 7.])
        self.assertEqual(frame.candidate_radius_is_defined.tolist(), [1, 1])

    def test_subset_selection_skips_unused_context_and_preserves_missingness(self):
        module = self.module()
        payload = self.payload()
        payload['fragments_graph'].node_radius[8] = np.nan
        # The context pass needs this helper, whereas the local pass does not.
        # Populate the runtime candidate cache before installing the helper guard.
        module.build_sample_universe(payload)
        with patch.object(module, 'build_comp_to_seg', side_effect=AssertionError('unused context ran')):
            _, _, acc = module._extract_features_runtime(payload, enabled_analysis_keys={'local'})
        frame = acc.to_frame()
        self.assertEqual(frame.candidate_radius_is_defined.tolist(), [1, 0])
        self.assertTrue(np.isnan(frame.candidate_radius.iloc[1]))
        self.assertEqual(frame.max_radius_is_defined.tolist(), [0, 0])

    def test_storage_permutation_undefined_and_identity_checks(self):
        module = self.module()
        rows, _ = module.build_sample_universe(self.payload())
        acc = module.FeatureAccumulator(rows[::-1])
        acc.set_candidate('candidate_radius', rows[0]['candidate_id'], 2.)
        acc.set_segment('max_radius', 111, 7.)
        frame = acc.to_frame()
        self.assertEqual(frame.candidate_radius_is_defined.tolist(), [0, 1])
        self.assertEqual(frame.max_radius.tolist(), [7., 7.])
        with self.assertRaisesRegex(ValueError, 'rows differ'):
            acc.validate_rows(rows)
        with self.assertRaisesRegex(ValueError, 'duplicate candidate'):
            module.FeatureAccumulator(rows + rows[:1])
        for fn, args, error in (
            (acc.set_segment, ('candidate_radius', 111, 2.), ValueError),
            (acc.set_candidate, ('max_radius', rows[0]['candidate_id'], 2.), ValueError),
            (acc.set_candidate, ('candidate_radius', 'unknown', 2.), KeyError),
            (acc.set_segment, ('max_radius', 111, 9.), ValueError),
            (acc.set_candidate, ('candidate_radius', rows[1]['candidate_id'], np.inf), ValueError),
        ):
            with self.subTest(args=args), self.assertRaises(error):
                fn(*args)

    def test_gt_changes_do_not_change_features_and_fake_accumulators_fail(self):
        module = self.module()
        first = self.payload()
        _, labels, acc = module._extract_features_runtime(first)
        second = self.payload()
        second['gt_merge_sites'] = []
        _, labels2, acc2 = module._extract_features_runtime(second)
        self.assertFalse(np.array_equal(labels, labels2))
        np.testing.assert_array_equal(acc.to_frame().values, acc2.to_frame().values)
        rows, labels = module.build_sample_universe(second)
        with patch.object(module, 'extract_features', return_value=(rows, labels, object())):
            with self.assertRaisesRegex(TypeError, 'runtime-owned'):
                module._extract_features_runtime(second)


if __name__ == '__main__':
    unittest.main()
