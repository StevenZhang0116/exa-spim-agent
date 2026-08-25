from __future__ import annotations

from collections import defaultdict
import unittest

import networkx as nx
import numpy as np

from agentic.detector_build.contracts import DetectorTarget, target_spec
from agentic.detector_build.target_runtime import target_adapter_source


class _Graph(nx.Graph):
    pass


SPLIT_POLICY = {
    "config_id": "tip_to_any_node|r=50|k=2",
    "mode": "tip_to_any_node",
    "radius_um": 50,
    "per_anchor_k": 2,
    "global_cap": None,
}
SPLIT_POLICY_SHA256 = "a" * 64


def _adapter_namespace(target: DetectorTarget) -> dict:
    namespace = {"np": np, "defaultdict": defaultdict}

    def build_comp_to_seg(graph):
        return {
            int(component): int(str(swc).split(".")[0])
            for component, swc in graph.component_id_to_swc_id.items()
        }

    namespace["build_comp_to_seg"] = build_comp_to_seg
    kwargs = (
        {
            "candidate_policy": SPLIT_POLICY,
            "candidate_policy_sha256": SPLIT_POLICY_SHA256,
        }
        if target is DetectorTarget.SPLIT else {}
    )
    exec(target_adapter_source(target, **kwargs), namespace)
    return namespace


def _fragment_graph(second_segment: int = 2) -> _Graph:
    graph = _Graph()
    graph.add_edges_from([(0, 1), (2, 3)])
    graph.node_xyz = np.asarray([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0], [3.0, 0.0, 0.0],
    ])
    graph.node_component_id = np.asarray([10, 10, 20, 20])
    graph.component_id_to_swc_id = {
        10: "1.0.swc", 20: f"{second_segment}.0.swc"
    }
    return graph


class DetectorTargetRuntimeTests(unittest.TestCase):
    def test_split_feature_applicability_filters_occurrences_not_rows(self) -> None:
        namespace = _adapter_namespace(DetectorTarget.SPLIT)
        sample = {
            "candidate_id": 7,
            "occurrences": (
                {
                    "anchor_side": "a",
                    "node_role_a": "tip",
                    "node_role_b": "non_tip",
                },
                {
                    "anchor_side": "b",
                    "node_role_a": "tip",
                    "node_role_b": "tip",
                },
            ),
        }
        compatible = namespace["compatible_occurrences"]

        self.assertEqual(len(compatible(sample, "no_tip_requirement")), 2)
        self.assertEqual(len(compatible(sample, "requires_tip_anchor")), 2)
        self.assertEqual(len(compatible(sample, "requires_both_tips")), 1)
        self.assertEqual(sample["candidate_id"], 7)

    def test_target_specs_keep_public_names_separate(self) -> None:
        merge = target_spec(DetectorTarget.MERGE)
        split = target_spec(DetectorTarget.SPLIT)
        self.assertEqual(merge.detector_name, "merge_site_detector.py")
        self.assertEqual(split.detector_name, "split_site_detector.py")
        self.assertEqual(split.label_name, "is_split")
        self.assertFalse(hasattr(split, "candidate_radius_um"))

    def test_split_adapter_builds_one_direct_candidate_pair(self) -> None:
        gt = _Graph()
        gt.add_edge(0, 1)
        gt.node_xyz = np.asarray([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        payload = {
            "fragments_graph": _fragment_graph(),
            "gt_graph": gt,
            "gt_node_canonical_label": np.asarray([1, 2]),
            "gt_edge_error": np.asarray([1]),
        }
        namespace = _adapter_namespace(DetectorTarget.SPLIT)

        samples, labels = namespace["build_sample_universe"](payload)
        audit = namespace["sample_universe_audit"](payload, samples, labels)

        self.assertEqual(len(samples), 1)
        self.assertEqual((samples[0]["segment_id_a"], samples[0]["segment_id_b"]), (1, 2))
        self.assertEqual(
            (samples[0]["node_id_a"], samples[0]["node_id_b"]),
            (1, 2),
        )
        self.assertEqual(labels.tolist(), [1])
        self.assertEqual(audit["n_candidate_truth_pairs"], 1)
        self.assertEqual(audit["candidate_recall_of_reachable"], 1.0)
        self.assertEqual(audit["candidate_policy_config_id"], SPLIT_POLICY["config_id"])
        self.assertEqual(audit["candidate_policy_sha256"], SPLIT_POLICY_SHA256)
        self.assertEqual(audit["candidate_pairing_rule"], "tip_to_any_node")
        self.assertEqual(audit["candidate_max_distance_um"], 50.0)
        self.assertEqual(audit["candidate_per_anchor_k"], 2)

    def test_split_adapter_labels_gap_split(self) -> None:
        gt = _Graph()
        gt.add_edges_from([(0, 1), (1, 2)])
        gt.node_xyz = np.asarray([
            [1.0, 0.0, 0.0], [1.5, 0.0, 0.0], [2.0, 0.0, 0.0]
        ])
        payload = {
            "fragments_graph": _fragment_graph(),
            "gt_graph": gt,
            "gt_node_canonical_label": np.asarray([1, 0, 2]),
            "gt_edge_error": np.asarray([2, 2]),
        }
        namespace = _adapter_namespace(DetectorTarget.SPLIT)

        samples, labels = namespace["build_sample_universe"](payload)
        audit = namespace["sample_universe_audit"](payload, samples, labels)

        self.assertEqual(labels.tolist(), [1])
        self.assertEqual(audit["truth_kind_counts"], {"gap": 1})

    def test_split_adapter_excludes_same_segment_pairs(self) -> None:
        gt = _Graph()
        gt.add_edge(0, 1)
        gt.node_xyz = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        payload = {
            "fragments_graph": _fragment_graph(second_segment=1),
            "gt_graph": gt,
            "gt_node_canonical_label": np.asarray([1, 2]),
            "gt_edge_error": np.asarray([1]),
        }
        namespace = _adapter_namespace(DetectorTarget.SPLIT)

        samples, labels = namespace["build_sample_universe"](payload)

        self.assertEqual(samples, [])
        self.assertEqual(labels.tolist(), [])

    def test_tip_to_any_node_uses_two_distinct_partner_segments_per_tip(self) -> None:
        frag = _Graph()
        # Segment 10 contributes the only nearby tip anchor (node 1). Segments
        # 20/30/40 expose nearby interior nodes while their tips stay far away.
        frag.add_edges_from([
            (0, 1), (2, 3), (3, 4), (5, 6), (6, 7), (8, 9), (9, 10),
        ])
        frag.node_xyz = np.asarray([
            [-100.0, 0.0, 0.0], [0.0, 0.0, 0.0],
            [1000.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1100.0, 0.0, 0.0],
            [0.0, 1000.0, 0.0], [3.0, 0.0, 0.0], [0.0, 1100.0, 0.0],
            [-1000.0, 0.0, 0.0], [4.0, 0.0, 0.0], [-1100.0, 0.0, 0.0],
        ])
        frag.node_component_id = np.asarray(
            [10, 10, 20, 20, 20, 30, 30, 30, 40, 40, 40]
        )
        frag.component_id_to_swc_id = {
            10: "10.0.swc", 20: "20.0.swc", 30: "30.0.swc", 40: "40.0.swc"
        }
        gt = _Graph()
        gt.add_edges_from([(0, 1), (0, 2), (0, 3)])
        gt.node_xyz = np.zeros((4, 3), dtype=float)
        payload = {
            "fragments_graph": frag,
            "gt_graph": gt,
            "gt_node_canonical_label": np.asarray([10, 20, 30, 40]),
            "gt_edge_error": np.asarray([1, 1, 1]),
        }

        samples, labels = _adapter_namespace(DetectorTarget.SPLIT)[
            "build_sample_universe"
        ](payload)
        pairs = {
            (sample["segment_id_a"], sample["segment_id_b"]): sample
            for sample in samples
        }

        self.assertEqual(set(pairs), {(10, 20), (10, 30)})
        self.assertNotIn((10, 40), pairs)
        self.assertEqual(labels.tolist(), [1, 1])
        self.assertEqual(pairs[(10, 20)]["node_role_a"], "tip")
        self.assertEqual(pairs[(10, 20)]["node_role_b"], "non_tip")
        self.assertEqual(pairs[(10, 20)]["partner_rank"], 1)
        self.assertEqual(pairs[(10, 30)]["partner_rank"], 2)


if __name__ == "__main__":
    unittest.main()
