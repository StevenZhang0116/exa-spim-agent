import importlib.util
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from matplotlib.colors import to_rgba


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "plot_top_split_predictions.py"
SPEC = importlib.util.spec_from_file_location("plot_top_split_predictions", SCRIPT)
plotter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(plotter)

CSV_HEADER = "candidate_id,segment_id_a,segment_id_b,node_id_a,node_id_b,gap_um,is_split,score\n"


class ArrayImage:
    def __init__(self):
        self.volume = np.arange(12 * 16 * 20).reshape(12, 16, 20)

    def shape(self):
        return self.volume.shape

    def read(self, center, shape):
        start = np.asarray(center) - np.asarray(shape) // 2
        return self.volume[tuple(slice(int(begin), int(begin + size)) for begin, size in zip(start, shape))]


class FakeGraph:
    nodes = [1, 0]
    node_xyz = np.array([[10., 12., 4.], [12., 12., 4.]])
    component_id_to_swc_id = {0: "111.0", 1: "222.0"}
    segment_by_node = {0: "111", 1: "222"}

    def node_segment_id(self, node):
        return self.segment_by_node[node]

    def nodes_in_patch(self, origin, shape, return_components=False):
        coords = (self.node_xyz / [2., 3., 1.])[:, ::-1] - origin
        return coords, np.array([0, 1])

    def edges_in_patch(self, origin, shape, return_components=False):
        coords, _ = self.nodes_in_patch(origin, shape)
        return np.stack([coords, coords], axis=1), np.array([0, 1])


def make_prediction(is_split=0):
    return {
        "candidate_id": 7, "segment_id_a": 111, "segment_id_b": 222,
        "node_id_a": 0, "node_id_b": 1, "gap_um": 2.0,
        "is_split": is_split, "score": 0.95,
    }


class PlotTopSplitPredictionsTests(unittest.TestCase):
    def test_output_directories_are_run_specific_and_overridable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, override in (("run_a", None), ("run_b", None), ("run_c", root / "custom")):
                with self.subTest(run=name):
                    result_dir = root / name
                    result_dir.mkdir()
                    csv_path = result_dir / "split_detector_794495.csv"
                    csv_path.write_text(
                        f"candidate_id,segment_id_a,segment_id_b,node_id_a,node_id_b,"
                        f"gap_um,is_split,{plotter.DEFAULT_SCORE}\n7,111,222,0,1,2.0,0,0.95\n")
                    argv = [str(result_dir), "--pkl", str(root / "missing.pkl")]
                    if override is not None:
                        argv.extend(["--output-dir", str(override)])
                    with patch("builtins.print"), self.assertRaises(FileNotFoundError):
                        plotter.main(argv)
                    expected = override or result_dir / "top_split_predictions"
                    self.assertTrue(expected.is_dir())
                    self.assertFalse((result_dir / "figures").exists())

    def test_ranking_skips_blank_and_nonfinite_scores(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scores.csv"
            path.write_text(CSV_HEADER + (
                "9007199254740993,1,2,10,11,2.5,0,0.99\n"
                "2,3,4,12,13,1.0,1,0.8\n"
                "3,5,6,14,15,1.0,1,\n"
                "4,5,6,16,17,1.0,0,nan\n"
                "5,7,8,18,19,3.0,0,inf\n"
                "6,9,10,20,21,,1,0.5\n"
            ))
            rows = plotter.rank_predictions(path, 5, "score")
            self.assertEqual([row["candidate_id"] for row in rows], [9007199254740993, 2, 6])
            self.assertEqual(rows[0]["segment_id_a"], 1)
            self.assertEqual(rows[0]["node_id_b"], 11)
            self.assertEqual(rows[1]["is_split"], 1)
            self.assertTrue(math.isnan(rows[2]["gap_um"]))
            self.assertEqual(len(plotter.rank_predictions(path, 5, "score", 0.9)), 1)
            self.assertEqual(len(plotter.rank_predictions(path, 1, "score")), 1)
            with self.assertRaises(ValueError):
                plotter.rank_predictions(path, 0, "score")

    def test_duplicate_candidate_id_and_missing_columns_raise(self):
        with tempfile.TemporaryDirectory() as directory:
            duplicated = Path(directory) / "dup.csv"
            duplicated.write_text(CSV_HEADER + "1,1,2,0,1,1.0,0,0.5\n1,3,4,2,3,1.0,0,0.6\n")
            with self.assertRaisesRegex(ValueError, "Duplicate candidate_id"):
                plotter.rank_predictions(duplicated, 5, "score")
            partial = Path(directory) / "partial.csv"
            partial.write_text("candidate_id,segment_id_a,score\n1,1,0.5\n")
            with self.assertRaisesRegex(ValueError, "CSV must contain"):
                plotter.rank_predictions(partial, 5, "score")

    def test_candidate_center_is_gap_midpoint_and_validates_segments(self):
        center = plotter.candidate_center(FakeGraph(), make_prediction())
        np.testing.assert_array_equal(center["xyz"], [11., 12., 4.])
        np.testing.assert_array_equal(center["endpoint_a"], [10., 12., 4.])
        np.testing.assert_array_equal(center["endpoint_b"], [12., 12., 4.])
        mismatched = {**make_prediction(), "segment_id_b": 999}
        with self.assertRaisesRegex(ValueError, "wrong --pkl"):
            plotter.candidate_center(FakeGraph(), mismatched)

    def test_patch_clipping_preserves_origin(self):
        image = ArrayImage()
        patch, origin = plotter.read_patch(image, [0., 0., 0.], [2., 3., 1.], [12., 18., 6.])
        np.testing.assert_array_equal(origin, [0, 0, 0])
        np.testing.assert_array_equal(patch, image.volume[:3, :3, :3])
        with self.assertRaises(ValueError):
            plotter.read_patch(image, [-1., 0., 0.], [2., 3., 1.], [12., 18., 6.])

    def test_render_nine_matched_panels_and_negative_label(self):
        graph = FakeGraph()
        payload = {"fragments_graph": graph, "gt_graph": graph, "anisotropy": [2., 3., 1.],
                   "gt_node_canonical_label": np.array([0, 111, 222])}
        figure, record = plotter.render_prediction(
            payload, ArrayImage(), make_prediction(), 1, [12., 18., 6.], "score",
        )
        self.assertEqual(len(figure.axes), 9)
        self.assertEqual(record["cache_is_split"], 0)
        self.assertEqual(record["rank"], 1)
        self.assertEqual(record["candidate_id"], 7)
        np.testing.assert_array_equal(
            [record["x_um"], record["y_um"], record["z_um"]], [11., 12., 4.])
        np.testing.assert_array_equal(
            [record["endpoint_a_x_um"], record["endpoint_b_x_um"]], [10., 12.])
        self.assertEqual([record["z_start_um"], record["z_stop_um"]], [1., 7.])
        self.assertIn("Pool GT: NOT SPLIT-LABELLED", figure._suptitle.get_text())
        self.assertIn("GT coverage: both sides traced", figure._suptitle.get_text())
        self.assertEqual(record["gt_covered_a"], 1)
        self.assertEqual(record["gt_covered_b"], 1)
        self.assertIn("candidate gap midpoint", figure._suptitle.get_text())
        self.assertIn("not a GT site", figure._supxlabel.get_text())
        self.assertIn("does not prove", figure._supxlabel.get_text())
        axes = np.asarray(figure.axes).reshape(3, 3)
        volume = ArrayImage().volume[1:7, 1:7, 2:8]
        anchors = ([7., 9.], [7., 3.], [9., 3.])
        endpoints_a = ([6., 9.], [6., 3.], [9., 3.])
        endpoints_b = ([8., 9.], [8., 3.], [9., 3.])
        extents = ([-1., 11., 16.5, -1.5], [-1., 11., 5.5, -0.5], [-1.5, 16.5, 5.5, -0.5])
        for column, view_name in enumerate(("XY", "XZ", "YZ")):
            self.assertEqual(axes[0, column].get_title(), f"Image (MIP) -- {view_name}")
            self.assertEqual(axes[0, column].images[0].origin, "upper")
            self.assertEqual(axes[0, column].images[0].get_cmap().name, "viridis")
            np.testing.assert_array_equal(axes[0, column].images[0].get_array(), volume.max(axis=column))
            np.testing.assert_allclose(axes[0, column].images[0].get_extent(), extents[column])
            for row in range(3):
                axis = axes[row, column]
                self.assertEqual(len(axis.get_xticks()), 0)
                self.assertEqual(len(axis.get_yticks()), 0)
                np.testing.assert_allclose(axis.collections[0].get_offsets(), [anchors[column]])
                np.testing.assert_allclose(axis.collections[1].get_offsets(), [endpoints_a[column]])
                np.testing.assert_allclose(axis.collections[2].get_offsets(), [endpoints_b[column]])
                np.testing.assert_allclose(axis.get_xlim(), extents[column][:2])
                np.testing.assert_allclose(axis.get_ylim(), extents[column][2:])
                if row > 0:
                    self.assertEqual(len(axis.images), 0)
                    self.assertEqual(len(axis.collections), 7)
        figure.canvas.draw()
        self.assertGreater(np.std(np.asarray(figure.canvas.buffer_rgba())), 0)
        figure.clear()

    def test_both_pair_segments_are_highlighted_only_in_fragments_row(self):
        graph = FakeGraph()
        payload = {"fragments_graph": graph, "gt_graph": graph, "anisotropy": [2., 3., 1.]}
        figure, _ = plotter.render_prediction(
            payload, ArrayImage(), make_prediction(), 1, [12., 18., 6.], "score",
        )
        axes = np.asarray(figure.axes).reshape(3, 3)
        side_colors = (to_rgba(plotter.SIDE_A_COLOR)[:3], to_rgba(plotter.SIDE_B_COLOR)[:3])
        for column in range(3):
            fragment_lines = [axes[2, column].collections[index] for index in (3, 5)]
            for line, expected in zip(fragment_lines, side_colors):
                np.testing.assert_allclose(line.get_colors()[0][:3], expected)
                self.assertEqual(line.get_linewidths()[0], 3.0)
            for index in (3, 5):
                gt_line = axes[1, column].collections[index]
                self.assertEqual(gt_line.get_linewidths()[0], 2.0)
                for expected in side_colors:
                    self.assertFalse(np.allclose(gt_line.get_colors()[0][:3], expected))
        figure.clear()

    def test_positive_pool_label_has_split_title(self):
        graph = FakeGraph()
        payload = {"fragments_graph": graph, "gt_graph": graph, "anisotropy": [2., 3., 1.],
                   "gt_node_canonical_label": np.array([0, 111, 222])}
        figure, record = plotter.render_prediction(
            payload, ArrayImage(), make_prediction(is_split=1), 1, [12., 18., 6.], "score",
        )
        self.assertEqual(
            figure._suptitle.get_text(),
            "Pool GT: SPLIT-LABELLED PAIR | GT coverage: both sides traced\n"
            "Center: candidate gap midpoint")
        self.assertEqual(figure._suptitle.get_fontweight(), "bold")
        self.assertEqual(record["cache_is_split"], 1)
        self.assertNotIn("does not prove", figure._supxlabel.get_text())
        figure.clear()

    def test_gt_coverage_indicator_tracks_each_side(self):
        graph = FakeGraph()
        cases = (
            (np.array([0, 111, 222]), "both sides traced", 1, 1, "does not prove"),
            (np.array([0, 111]), "side A only", 1, 0, "does not prove"),
            (np.array([222, 0]), "side B only", 0, 1, "does not prove"),
            (np.array([0, 0, 5]), "neither side traced", 0, 0, "is_split=0 is vacuous"),
        )
        for labels, expected, covered_a, covered_b, caveat in cases:
            with self.subTest(expected=expected):
                payload = {"fragments_graph": graph, "gt_graph": graph,
                           "anisotropy": [2., 3., 1.], "gt_node_canonical_label": labels}
                figure, record = plotter.render_prediction(
                    payload, ArrayImage(), make_prediction(), 1, [12., 18., 6.], "score",
                )
                self.assertIn(f"GT coverage: {expected}", figure._suptitle.get_text())
                self.assertEqual(record["gt_covered_a"], covered_a)
                self.assertEqual(record["gt_covered_b"], covered_b)
                self.assertIn(caveat, figure._supxlabel.get_text())
                figure.clear()

    def test_missing_canonical_labels_reads_unknown(self):
        graph = FakeGraph()
        payload = {"fragments_graph": graph, "gt_graph": graph, "anisotropy": [2., 3., 1.]}
        figure, record = plotter.render_prediction(
            payload, ArrayImage(), make_prediction(), 1, [12., 18., 6.], "score",
        )
        self.assertIn("GT coverage: unknown", figure._suptitle.get_text())
        self.assertEqual(record["gt_covered_a"], "")
        self.assertEqual(record["gt_covered_b"], "")
        self.assertIn("does not prove", figure._supxlabel.get_text())
        figure.clear()
        self.assertIsNone(plotter.covered_segment_ids(payload))
        self.assertEqual(
            plotter.covered_segment_ids({"gt_node_canonical_label": np.array([0, 7, 7, 9])}),
            frozenset({7, 9}))


if __name__ == "__main__":
    unittest.main()
