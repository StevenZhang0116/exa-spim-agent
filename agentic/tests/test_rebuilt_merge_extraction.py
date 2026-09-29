import ast
import importlib.util
import unittest
from pathlib import Path

import networkx as nx
import numpy as np

from agentic.detector_build.assembly import _validate_accumulator_set_calls


DETECTOR = (Path(__file__).resolve().parents[2] / "autodiscovery-application"
            / "merge-error-794495-mcl100_2026-08-04-rebuild-20260923-160112"
            / "merge_junction_detector.py")


def synthetic_payload():
    graph = nx.Graph()
    coordinates, components, hubs = [], [], []
    for component in range(2):
        offset = np.array([component * 500., 0., 0.])
        hub = len(coordinates)
        hubs.append(hub)
        coordinates.append(offset)
        components.append(component)
        for arm in range(3):
            previous = hub
            for step in range(1, 21):
                coordinate = np.roll(np.array([step * 5., 4. * np.sin(step * .7),
                                                3. * np.cos(step * .5) - 3.]), arm) + offset
                node = len(coordinates)
                coordinates.append(coordinate)
                components.append(component)
                graph.add_edge(previous, node)
                previous = node
    graph.node_xyz = np.array(coordinates)
    graph.node_component_id = np.array(components)
    graph.node_radius = 1. + .3 * np.sin(np.arange(len(coordinates)) * .2)
    graph.component_id_to_swc_id = {0: "32323215387.0.swc", 1: "111.0.swc"}
    return {
        "fragments_graph": graph, "min_cable_length": 100, "anisotropy": (1., 1., 1.),
        "gt_merge_sites": [{"segment_id": 32323215387, "xyz": graph.node_xyz[hubs[0]].tolist(),
                            "source": "two_gt_junction"}],
        "gt_merge_labels": np.array([32323215387]),
    }, hubs


@unittest.skipUnless(DETECTOR.is_file(), "Published rebuild instance is not present")
class RebuiltMergeExtractionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("rebuilt_merge_extraction", DETECTOR)
        cls.detector = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.detector)

    def test_original_accumulator_failure_and_missingness(self):
        accumulator = self.detector.FeatureAccumulator(["max_windowed_path_tortuosity", "unused"])
        accumulator.set("max_windowed_path_tortuosity", 32323215387, 1.5)
        accumulator.bind_rows([32323215387, 7, 32323215387])
        frame = accumulator.to_frame()
        self.assertEqual(accumulator.value_for("max_windowed_path_tortuosity", 32323215387), 1.5)
        self.assertEqual(frame["max_windowed_path_tortuosity_is_defined"].tolist(), [True, False, True])
        np.testing.assert_allclose(frame["max_windowed_path_tortuosity"], [1.5, np.nan, 1.5])
        self.assertFalse(frame["unused_is_defined"].any())

    def test_static_gate_rejects_original_signature_but_accepts_repair(self):
        tree = ast.parse(DETECTOR.read_text())
        _validate_accumulator_set_calls(tree, DETECTOR)
        accumulator = next(node for node in tree.body
                           if isinstance(node, ast.ClassDef) and node.name == "FeatureAccumulator")
        setter = next(node for node in accumulator.body if isinstance(node, ast.FunctionDef) and node.name == "set")
        setter.args.args[1], setter.args.args[2] = setter.args.args[2], setter.args.args[1]
        with self.assertRaisesRegex(SystemExit, "argument order mismatch"):
            _validate_accumulator_set_calls(tree, DETECTOR)

    def test_actual_generated_extraction_all_single_and_disabled_groups(self):
        data, hubs = synthetic_payload()
        columns = [column for name in self.detector.FEATURE_NAMES for column in (name, name + "_is_defined")]
        for enabled in (None, {"windowed_path_tortuosity"}, set()):
            with self.subTest(enabled=enabled):
                rows, labels, accumulator = self.detector._extract_features_runtime(
                    data, verbose=False, enabled_analysis_keys=enabled, image_workers=8)
                frame = accumulator.to_frame()
                self.assertEqual([row["node_id"] for row in rows], hubs)
                self.assertEqual(labels.tolist(), [1, 0])
                self.assertEqual(frame.shape, (2, len(columns)))
                self.assertEqual(list(frame.columns), columns)
                if enabled == set():
                    self.assertFalse(frame.filter(like="_is_defined").any().any())
                    self.assertTrue(frame[self.detector.FEATURE_NAMES].isna().all().all())
                else:
                    self.assertTrue(frame["max_windowed_path_tortuosity_is_defined"].all())
                    if enabled is not None:
                        self.assertEqual(frame.filter(like="_is_defined").to_numpy().sum(), 2)

    def test_actual_generated_profile_subset_keeps_aligned_rows(self):
        data, _ = synthetic_payload()
        rows, labels, accumulator = self.detector._extract_features_runtime(
            data, verbose=False, enabled_analysis_keys={"windowed_path_tortuosity"}, profile_segment_limit=1)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["segment_id"], 111)
        self.assertEqual(labels.tolist(), [0])
        self.assertEqual(len(accumulator.to_frame()), 1)


if __name__ == "__main__":
    unittest.main()