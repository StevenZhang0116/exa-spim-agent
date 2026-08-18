from __future__ import annotations

from collections import defaultdict
import unittest

import networkx as nx
import numpy as np

from agentic.detector_build.contracts import DetectorTarget, target_spec
from agentic.detector_build.target_runtime import target_adapter_source


class _Graph(nx.Graph):
    pass


def _adapter_namespace(target: DetectorTarget) -> dict:
    namespace = {"np": np, "defaultdict": defaultdict}

    def build_comp_to_seg(graph):
        return {
            int(component): int(str(swc).split(".")[0])
            for component, swc in graph.component_id_to_swc_id.items()
        }

    namespace["build_comp_to_seg"] = build_comp_to_seg
    exec(target_adapter_source(target), namespace)
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
    def test_target_specs_keep_public_names_separate(self) -> None:
        merge = target_spec(DetectorTarget.MERGE)
        split = target_spec(DetectorTarget.SPLIT)
        self.assertEqual(merge.detector_name, "merge_site_detector.py")
        self.assertEqual(split.detector_name, "split_site_detector.py")
        self.assertEqual(split.label_name, "is_split")
        self.assertEqual(split.candidate_radius_um, 30.0)

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
            (samples[0]["endpoint_node_id_a"], samples[0]["endpoint_node_id_b"]),
            (1, 2),
        )
        self.assertEqual(labels.tolist(), [1])
        self.assertEqual(audit["n_candidate_truth_pairs"], 1)
        self.assertEqual(audit["candidate_recall_of_reachable"], 1.0)

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


if __name__ == "__main__":
    unittest.main()
