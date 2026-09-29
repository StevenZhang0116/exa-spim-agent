"""Split anchoring/reduction contracts tested through assembled extraction."""
import ast
import importlib.util
import json
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

import networkx as nx
import numpy as np

from agentic.detector_build.assembly import assemble_detector
from agentic.detector_build.contracts import DetectorTarget
from agentic.detector_build.cost_validation import analyze_row_cost
from agentic.detector_build.split_feature_scope import split_inventory_specs
from agentic.tests.test_detector_build_workflow import workflow, _write_split_candidate_policy


def feature(name, anchor, reduction='first_compatible', requirement='no_tip_requirement'):
    return dict(name=name, scope='candidate', anchor=anchor, occurrence_reduction=reduction,
                source_feature=name, source_formula='source scalar formula',
                source_aggregation=reduction, adaptation='identity',
                node_role_requirement=requirement)


def features():
    return [feature('tip_radius', 'tip_anchor', requirement='requires_tip_anchor'),
            feature('midpoint_z', 'gap_midpoint'),
            feature('mean_gap', 'endpoint_pair', 'mean'),
            dict(feature('length_ratio', 'component_pair'), source_formula='max(L_a,L_b)/min(L_a,L_b)')]


FRAGMENT = '''
FEATURE_REGISTRY = [('tip_radius',), ('midpoint_z',), ('mean_gap',), ('length_ratio',)]
ANALYSIS_TIMING_GROUPS = [
    {'key': 'radius', 'hypothesis_ids': [1], 'phase': 'candidate_pair', 'feature_names': ['tip_radius']},
    {'key': 'midpoint', 'hypothesis_ids': [2], 'phase': 'candidate_pair', 'feature_names': ['midpoint_z']},
    {'key': 'gap', 'hypothesis_ids': [3], 'phase': 'candidate_pair', 'feature_names': ['mean_gap']},
    {'key': 'context', 'hypothesis_ids': [4], 'phase': 'candidate_pair', 'feature_names': ['length_ratio']},
]
COMPUTATION_PLAN = [{'primitive': 'lookups', 'cost_class': 'constant', 'bound': '',
                     'amortization': '', 'consumers': ['radius', 'midpoint', 'gap', 'context']}]
def extract_features(payload, verbose=True, timing=None, enabled_analysis_keys=None,
                     profile_segment_limit=None, image_workers=1):
    def _analysis_enabled(key):
        return enabled_analysis_keys is None or key in enabled_analysis_keys
    samples, labels = build_sample_universe(payload)
    if profile_segment_limit is not None:
        chosen = list(dict.fromkeys(s[k] for s in samples
                                   for k in ('segment_id_a', 'segment_id_b')))[:profile_segment_limit]
        positions = [i for i,s in enumerate(samples)
                     if s['segment_id_a'] in chosen or s['segment_id_b'] in chosen]
        samples = [samples[i] for i in positions]
        labels = labels[positions]
    graph = payload['fragments_graph']
    acc = FeatureAccumulator(samples, graph)
    if timing is not None:
        timing.start_phase('candidate_pair')
        timing.start_analysis('radius')
        timing.start_analysis('midpoint')
        timing.start_analysis('gap')
        timing.start_analysis('context')
    def radius(context):
        return float(graph.node_radius[context['anchor_node_id']])
    def ratio(context):
        a,b = context['component_ids']
        x,y = graph.component_lengths[a], graph.component_lengths[b]
        return max(x,y)/min(x,y) if min(x,y) > 0 else None
    for sample in samples:
        cid = sample['candidate_id']
        if _analysis_enabled('radius'):
            acc.compute('tip_radius', cid, radius)
        if _analysis_enabled('midpoint'):
            acc.compute('midpoint_z', cid, lambda c: c['center_um'][2])
        if _analysis_enabled('gap'):
            acc.compute('mean_gap', cid, lambda c: c['gap_um'])
        if _analysis_enabled('context'):
            acc.compute('length_ratio', cid, ratio)
    return samples, labels, acc
'''


def payload():
    graph = nx.Graph([(0, 1), (0, 2), (3, 4), (5, 6)])
    graph.node_xyz = np.array([[0, 0, 0], [-100, 0, 0], [100, 0, 0],
                               [5, 0, 0], [5, 100, 0], [12, 0, 4], [12, 100, 4]], dtype=float)
    graph.node_component_id = np.array([10, 10, 10, 20, 20, 30, 30])
    graph.component_id_to_swc_id = {10: '111.0.swc', 20: '222.0.swc', 30: '333.0.swc'}
    graph.component_lengths = {10: 200., 20: 100., 30: 100.}
    graph.node_radius = np.array([99., 1., 1., 2., 9., 7., 11.])
    gt = nx.Graph([(0, 1)])
    return dict(fragments_graph=graph, gt_graph=gt,
                gt_node_canonical_label=np.array([111, 222]), gt_edge_error=np.array([1]),
                gt_merge_labels=np.array([], dtype=int))


class SplitFeatureScopeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.inventory = self.root / 'feature_inventory.json'
        self.data = dict(schema_version=5, hypotheses=[dict(included=True, features=features())])
        self.inventory.write_text(json.dumps(self.data))
        self.policy = _write_split_candidate_policy(self.root)
        self.fragment = self.root / 'feature.py'
        self.fragment.write_text(FRAGMENT)
        self.output = self.root / 'detector.py'

    def assemble(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            assemble_detector(workflow.RUNTIME_TEMPLATE_PATH, self.fragment, self.output,
                              target=DetectorTarget.SPLIT, candidate_policy_path=self.policy,
                              inventory_path=self.inventory)

    def module(self):
        self.assemble()
        spec = importlib.util.spec_from_file_location('split_scope_detector', self.output)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_actual_universe_uses_b_tip_and_distinguishes_pairs_sharing_a(self):
        module = self.module()
        rows, labels, acc = module._extract_features_runtime(payload(), verbose=False)
        frame = acc.to_frame()
        by_pair = {(s['segment_id_a'], s['segment_id_b']): i for i,s in enumerate(rows)}
        ab, ac = by_pair[(111, 222)], by_pair[(111, 333)]
        self.assertEqual(rows[ab]['anchor_side'], 'b')
        self.assertEqual(rows[ab]['node_role_a'], 'non_tip')
        self.assertEqual(frame.tip_radius.iloc[ab], 2.)
        self.assertEqual(frame.tip_radius.iloc[ac], 7.)
        self.assertEqual(frame.midpoint_z.iloc[ab], 0.)
        self.assertEqual(frame.midpoint_z.iloc[ac], 2.)
        self.assertEqual(frame.length_ratio.iloc[ab], 2.)  # source max/min, not its inverse
        self.assertEqual(frame.length_ratio.iloc[ac], 2.)
        self.assertEqual(labels[ab], 1)

    def test_label_changes_disabled_features_and_profile_subset(self):
        module = self.module()
        one, two = payload(), payload()
        two['gt_edge_error'][:] = 0
        rows, labels, first = module._extract_features_runtime(one)
        _, changed_labels, second = module._extract_features_runtime(two)
        self.assertFalse(np.array_equal(labels, changed_labels))
        np.testing.assert_equal(first.to_frame().values, second.to_frame().values)
        del one['fragments_graph'].component_lengths  # would fail if context evaluator ran
        subset, _, acc = module._extract_features_runtime(
            one, enabled_analysis_keys={'radius'}, profile_segment_limit=1)
        self.assertEqual(len(subset), 2)  # A-B and A-C, not first one row or BOTH endpoints
        self.assertTrue(acc.to_frame().length_ratio.isna().all())
        self.assertTrue(acc.to_frame().tip_radius.notna().all())

    def test_relabeling_segments_swaps_canonical_sides_without_changing_features(self):
        module = self.module()
        original = payload()
        swapped = payload()
        swapped['fragments_graph'].component_id_to_swc_id[10] = '999.0.swc'
        swapped['gt_node_canonical_label'][0] = 999
        first_rows, _, first = module._extract_features_runtime(original)
        second_rows, _, second = module._extract_features_runtime(swapped)
        first_index = next(i for i,s in enumerate(first_rows)
                           if (s['segment_id_a'], s['segment_id_b']) == (111, 222))
        second_index = next(i for i,s in enumerate(second_rows)
                            if (s['segment_id_a'], s['segment_id_b']) == (222, 999))
        self.assertEqual(first_rows[first_index]['anchor_side'], 'b')
        self.assertEqual(second_rows[second_index]['anchor_side'], 'a')
        np.testing.assert_equal(first.to_frame().iloc[first_index].values,
                                second.to_frame().iloc[second_index].values)

    def test_runtime_reduces_multiple_occurrences_and_does_not_fallback(self):
        module = self.module()
        data = payload()
        rows, _ = module.build_sample_universe(data)
        row = dict(rows[0])
        first = dict(row['occurrences'][0], gap_um=5.)
        later = dict(first, node_id_b=4, gap_um=10., partner_rank=2)
        row['occurrences'] = (first, later)
        for reduction, expected in [('first_compatible', 2.), ('min', 2.), ('max', 9.),
                                    ('mean', 5.5), ('sum', 11.), ('all_compatible', 7.)]:
            with self.subTest(reduction=reduction):
                spec = dict(module.SPLIT_FEATURE_SPECS['tip_radius'], occurrence_reduction=reduction)
                with patch.dict(module.SPLIT_FEATURE_SPECS, {'tip_radius': spec}):
                    acc = module.FeatureAccumulator([row], data['fragments_graph'])
                radii = data['fragments_graph'].node_radius
                evaluator = (lambda cs: radii[cs[1]['anchor_node_id']] - radii[cs[0]['anchor_node_id']]) \
                    if reduction == 'all_compatible' else (lambda c: radii[c['anchor_node_id']])
                acc.compute('tip_radius', row['candidate_id'], evaluator)
                self.assertEqual(acc.to_frame().tip_radius.iloc[0], expected)
        calls = []
        acc = module.FeatureAccumulator([row], data['fragments_graph'])
        def undefined(context):
            calls.append(context['anchor_node_id'])
            return None
        acc.compute('tip_radius', row['candidate_id'], undefined)
        self.assertEqual(calls, [3])
        self.assertEqual(acc.to_frame().tip_radius_is_defined.tolist(), [0])

    def test_absent_applicability_never_calls_formula_or_drops_row(self):
        module = self.module()
        data = payload()
        rows, _ = module.build_sample_universe(data)
        spec = dict(module.SPLIT_FEATURE_SPECS['tip_radius'], node_role_requirement='requires_both_tips')
        with patch.dict(module.SPLIT_FEATURE_SPECS, {'tip_radius': spec}):
            acc = module.FeatureAccumulator(rows[:1], data['fragments_graph'])
        def fail(_):
            self.fail('Formula called on an incompatible occurrence')
        acc.compute('tip_radius', rows[0]['candidate_id'], fail)
        self.assertEqual(len(acc.to_frame()), 1)
        self.assertTrue(acc.to_frame().tip_radius.isna().all())

    def test_context_orientation_and_segment_pair_context(self):
        module = self.module()
        data = payload()
        rows, _ = module.build_sample_universe(data)
        row = rows[0]
        contexts = module._split_feature_inputs(row, module.SPLIT_FEATURE_SPECS['tip_radius'],
                                               data['fragments_graph'])
        self.assertEqual((contexts[0]['anchor_node_id'], contexts[0]['anchor_segment_id']), (3, 222))
        self.assertEqual((contexts[0]['partner_node_id'], contexts[0]['partner_segment_id']), (0, 111))
        with self.assertRaises(TypeError):
            contexts[0]['anchor_node_id'] = 0
        spec = dict(anchor='segment_pair', occurrence_reduction='pair_once',
                    node_role_requirement='no_tip_requirement')
        with patch.dict(module.SPLIT_FEATURE_SPECS, {'length_ratio': spec}):
            acc = module.FeatureAccumulator(rows, data['fragments_graph'])
        for sample in rows:
            acc.compute('length_ratio', sample['candidate_id'], lambda c: sum(c['segment_ids']))
        self.assertEqual(acc.to_frame().length_ratio.tolist(), [333., 444., 555.])

    def test_identity_order_missing_values_and_invalid_writes(self):
        module = self.module()
        data = payload()
        rows, labels = module.build_sample_universe(data)
        acc = module.FeatureAccumulator(rows[::-1], data['fragments_graph'])
        acc.compute('tip_radius', rows[0]['candidate_id'], lambda c: 2.)
        self.assertEqual(acc.to_frame().tip_radius_is_defined.tolist(), [0, 0, 1])
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            acc.compute('tip_radius', rows[0]['candidate_id'], lambda c: 3.)
        with self.assertRaisesRegex(ValueError, 'finite'):
            acc.compute('tip_radius', rows[1]['candidate_id'], lambda c: np.nan)
        with self.assertRaises(KeyError):
            acc.compute('tip_radius', -999, lambda c: 1.)
        with self.assertRaises(ValueError):
            module.FeatureAccumulator(rows + rows[:1], data['fragments_graph'])
        with patch.object(module, 'extract_features', return_value=(rows[::-1], labels[::-1], acc)):
            with self.assertRaisesRegex(ValueError, 'candidate order'):
                module._extract_features_runtime(data)
        with patch.object(module, 'extract_features', return_value=(rows, labels, object())):
            with self.assertRaisesRegex(TypeError, 'runtime-owned'):
                module._extract_features_runtime(data)
        modified = [dict(s) for s in rows]
        modified[0]['occurrences'] = ()
        fake = module.FeatureAccumulator(modified, data['fragments_graph'])
        with patch.object(module, 'extract_features', return_value=(modified, labels, fake)):
            with self.assertRaisesRegex(ValueError, 'occurrences differ'):
                module._extract_features_runtime(data)

    def test_inventory_rejects_legacy_and_inconsistent_contracts(self):
        mutations = [lambda d: d.update(schema_version=3),
                     lambda d: d['hypotheses'][0]['features'][0].update(anchor='side_a'),
                     lambda d: d['hypotheses'][0]['features'][0].update(occurrence_reduction='pair_once'),
                     lambda d: d['hypotheses'][0]['features'][0].update(node_role_requirement='no_tip_requirement'),
                     lambda d: d['hypotheses'][0]['features'][0].update(adaptation='candidate_pair'),
                     lambda d: d['hypotheses'][0]['features'][0].update(source_formula='')]
        for mutate in mutations:
            data = json.loads(json.dumps(self.data))
            mutate(data)
            with self.subTest(data=data), self.assertRaises(SystemExit):
                split_inventory_specs(data)

    def test_assembly_rejects_custom_storage_and_bypassed_anchor_api(self):
        for source, message in [
            (FRAGMENT + '\nclass FeatureAccumulator: pass\n', 'runtime-owned'),
            (FRAGMENT.replace("acc.compute('tip_radius'", "acc.set_candidate('tip_radius'"), 'acc.compute'),
            (FRAGMENT.replace('tip_radius', 'undeclared_radius'), 'FEATURE_REGISTRY'),
        ]:
            self.fragment.write_text(source)
            with self.subTest(message=message), self.assertRaisesRegex(SystemExit, message):
                self.assemble()

    def test_global_scan_hidden_in_evaluator_is_still_rejected(self):
        source = FRAGMENT.replace("float(graph.node_radius[context['anchor_node_id']])",
                                  "float(np.min(graph.node_xyz[:, 2]))")
        report = analyze_row_cost(ast.parse(source))
        self.assertEqual(report.status, 'rejected', report.unverified)
        self.assertTrue(any('min' in item for item in report.violations))
        self.fragment.write_text(source)
        with self.assertRaisesRegex(SystemExit, 'global|whole|row-path'):
            self.assemble()

    def test_bound_method_evaluator_is_analyzed_and_unknown_callback_is_partial(self):
        source = '''
class Evaluator:
    def __init__(self, graph):
        self.graph = graph
    def evaluate(self, context):
        return np.min(self.graph.node_xyz[:, 2])
def extract_features(payload):
    rows, labels = build_sample_universe(payload)
    graph = payload['fragments_graph']
    acc = FeatureAccumulator(rows, graph)
    evaluator = Evaluator(graph)
    for row in rows:
        acc.compute('radius', row['candidate_id'], evaluator.evaluate)
    return rows, labels, acc
'''
        self.assertEqual(analyze_row_cost(ast.parse(source)).status, 'rejected')
        unresolved = source.replace('evaluator.evaluate)', "payload['dynamic_evaluator'])")
        report = analyze_row_cost(ast.parse(unresolved))
        self.assertEqual(report.status, 'partial')
        self.assertTrue(any('evaluator cannot be resolved' in x for x in report.unverified))


if __name__ == '__main__':
    unittest.main()
