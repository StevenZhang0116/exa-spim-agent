import csv
import importlib.util
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from matplotlib.colors import to_rgba


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "plot_top_merge_predictions.py"
SPEC = importlib.util.spec_from_file_location("plot_top_merge_predictions", SCRIPT)
plotter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(plotter)


class ArrayImage:
    def __init__(self):
        self.volume = np.arange(12 * 16 * 20).reshape(12, 16, 20)

    def shape(self):
        return self.volume.shape

    def read(self, center, shape):
        start = np.asarray(center) - np.asarray(shape) // 2
        return self.volume[tuple(slice(int(begin), int(begin + size)) for begin, size in zip(start, shape))]


class FakeGraph:
    node_xyz = np.array([[10., 12., 4.], [12., 12., 4.]])
    node_component_id = np.array([0, 1])
    component_id_to_swc_id = {0: "111.0", 1: "222.0"}

    def edges(self):
        return [(0, 0), (1, 1)]

    def node_segment_id(self, node):
        return ("111", "222")[node]

    def degree(self, node):
        return 3

    def nodes_in_patch(self, origin, shape, return_components=False):
        coords = (self.node_xyz / [2., 3., 1.])[:, ::-1] - origin
        return coords, np.array([0, 1])

    def edges_in_patch(self, origin, shape, return_components=False):
        coords, _ = self.nodes_in_patch(origin, shape)
        return np.stack([coords, coords], axis=1), np.array([0, 1])


class SpatialGT:
    def __init__(self, voxels):
        self.voxels = np.asarray(voxels, dtype=float).reshape(-1, 3)
        self.queries = []

    def nodes_in_patch(self, origin, shape, return_components=False):
        self.queries.append((np.array(origin), np.array(shape)))
        inside = np.all((self.voxels >= origin) & (self.voxels < origin + shape), axis=1)
        return self.voxels[inside] - origin, np.zeros(int(inside.sum()), dtype=int)


def prediction(**changes):
    return {"candidate_id": 7, "node_id": 0, "segment_id": 111, "degree": 3,
            "x_um": 10., "y_um": 12., "z_um": 4., "is_merge_site": 0,
            "in_ambiguous_ring": 0, "distance_to_nearest_gt_site_um": float("nan"),
            "score": 0.95, **changes}


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class PlotTopMergePredictionsTests(unittest.TestCase):
    def test_gt_filter_backfills_top_k_without_reading_images_or_using_merge_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scores.csv"
            write_csv(path, [prediction(candidate_id=1, score=0.99, is_merge_site=1),
                             prediction(candidate_id=3, node_id=1, segment_id=222, x_um=12., score=0.8),
                             prediction(candidate_id=2, node_id=1, segment_id=222, x_um=12., score=0.9)])
            ranked = plotter.rank_predictions(path, None, "score")
            gt = SpatialGT([[4., 4., 6.]])
            image = ArrayImage()
            payload = {"fragments_graph": FakeGraph(), "gt_graph": gt, "anisotropy": [2., 3., 1.],
                       "gt_node_canonical_label": np.array([111])}
            with patch.object(image, "read", side_effect=AssertionError("Selection must not fetch pixels")):
                selected = plotter.select_gt_visible_predictions(payload, image, ranked, 2, [4., 6., 2.])
            self.assertEqual([row["candidate_id"] for row in selected], [2, 3])
            self.assertEqual([row["score_rank"] for row in selected], [2, 3])
            self.assertEqual([row["is_merge_site"] for row in selected], [0, 0])
            self.assertEqual(selected[0]["gt_nodes_in_patch"], 1)
            self.assertEqual(selected[0]["gt_components_in_patch"], 1)
            self.assertNotIn("score_rank", ranked[1])

    def test_gt_filter_uses_clipped_volume_and_handles_shortfall_or_no_gt(self):
        graph = FakeGraph()
        graph.node_xyz = np.array([[0., 0., 0.]])
        candidate = prediction(x_um=0., y_um=0., z_um=0.)
        image = ArrayImage()
        gt = SpatialGT([[-1., 0., 0.], [1., 1., 1.], [3., 0., 0.]])
        payload = {"fragments_graph": graph, "gt_graph": gt, "anisotropy": [2., 3., 1.]}
        selected = plotter.select_gt_visible_predictions(payload, image, [candidate], 5, [12., 18., 6.])
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["gt_nodes_in_patch"], 1)
        pixels, origin = plotter.read_patch(image, [0., 0., 0.], [2., 3., 1.], [12., 18., 6.])
        np.testing.assert_array_equal(gt.queries[0][0], origin)
        np.testing.assert_array_equal(gt.queries[0][1], pixels.shape)
        payload["gt_graph"] = SpatialGT([])
        self.assertEqual(plotter.select_gt_visible_predictions(payload, image, [candidate], 2,
                                                               [12., 18., 6.]), [])

    def test_render_discloses_gt_selection_and_original_score_rank(self):
        graph = FakeGraph()
        payload = {"fragments_graph": graph, "gt_graph": graph, "anisotropy": [2., 3., 1.]}
        candidate = prediction(score_rank=12, gt_nodes_in_patch=2, gt_components_in_patch=2)
        figure, record = plotter.render_prediction(payload, ArrayImage(), candidate, 1,
                                                   [12., 18., 6.], "score")
        self.assertEqual(record["rank"], 1)
        self.assertEqual(record["score_rank"], 12)
        self.assertEqual(record["gt_nodes_in_patch"], 2)
        self.assertIn("score rank 12", figure._supxlabel.get_text())
        self.assertIn("visual inspection only", figure._supxlabel.get_text())
        figure.clear()

    def test_score_rank_does_not_imply_gt_selection_metadata(self):
        graph = FakeGraph()
        payload = {"fragments_graph": graph, "gt_graph": graph, "anisotropy": [2., 3., 1.]}
        figure, record = plotter.render_prediction(
            payload, ArrayImage(), prediction(score_rank=7), 4, [12., 18., 6.], plotter.DEFAULT_SCORE)
        self.assertIn("score rank 7", figure._supxlabel.get_text())
        self.assertNotIn("GT-visible selection:", figure._supxlabel.get_text())
        self.assertEqual(record["score_rank"], 7)
        figure.clear()

    def test_ranking_ignores_gt_and_skips_nonfinite_with_stable_ties(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scores.csv"
            rows = [prediction(candidate_id=9007199254740993, score=0.99),
                    prediction(candidate_id=2, score=0.8, is_merge_site=1),
                    prediction(candidate_id=1, score=0.8, in_ambiguous_ring=1),
                    prediction(candidate_id=3, score=""),
                    prediction(candidate_id=4, score="nan"),
                    prediction(candidate_id=5, score="inf")]
            write_csv(path, rows)
            ranked = plotter.rank_predictions(path, 10, "score")
            self.assertEqual([row["candidate_id"] for row in ranked], [9007199254740993, 1, 2])
            self.assertEqual(ranked[0]["is_merge_site"], 0)
            self.assertTrue(math.isnan(ranked[0]["distance_to_nearest_gt_site_um"]))
            self.assertEqual(len(plotter.rank_predictions(path, 10, "score", 0.9)), 1)
            with self.assertRaises(ValueError):
                plotter.rank_predictions(path, 0, "score")
            with self.assertRaises(ValueError):
                plotter.rank_predictions(path, 1, "score", float("nan"))

    def test_winner_oof_default_and_explicit_selector_override(self):
        winner = "merge_site_probability_oof"
        selector = "merge_site_probability_oof_selector"
        self.assertEqual(plotter.DEFAULT_SCORE, winner)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "scores.csv"
            rows = [prediction(candidate_id=1), prediction(candidate_id=2, is_merge_site=1),
                    prediction(candidate_id=3)]
            for row, winner_score, selector_score in zip(rows, [0.01, 0.02, ""], [0.9, 0.1, 1.0]):
                del row["score"]
                row.update({winner: winner_score, selector: selector_score})
            write_csv(path, rows)
            self.assertEqual([row["candidate_id"] for row in plotter.rank_predictions(path, 3)], [2, 1])
            self.assertEqual([row["candidate_id"] for row in plotter.rank_predictions(path, 3, selector)],
                             [3, 1, 2])
            for options, column in (([], winner), (["--score-column", selector], selector)):
                with self.subTest(column=column):
                    argv = ["--csv", str(path), "--pkl", str(root / "missing.pkl"),
                            "--output-dir", str(root / "plots"), *options]
                    with patch.object(plotter, "rank_predictions", wraps=plotter.rank_predictions) as rank:
                        with patch("builtins.print"), self.assertRaises(FileNotFoundError):
                            plotter.main(argv)
                        self.assertEqual(rank.call_args.args[2], column)

    def test_duplicate_and_legacy_csv_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scores.csv"
            write_csv(path, [prediction(), prediction()])
            with self.assertRaisesRegex(ValueError, "Duplicate candidate_id"):
                plotter.rank_predictions(path, 2, "score")
            path.write_text("segment_id,is_merge,score\n111,1,0.9\n")
            with self.assertRaisesRegex(ValueError, "legacy segment-level"):
                plotter.rank_predictions(path, 2, "score")

    def test_missing_audit_columns_are_unknown_not_negative(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scores.csv"
            row = prediction()
            del row["in_ambiguous_ring"]
            del row["distance_to_nearest_gt_site_um"]
            write_csv(path, [row])
            ranked = plotter.rank_predictions(path, 1, "score")
            self.assertIsNone(ranked[0]["in_ambiguous_ring"])
            self.assertTrue(math.isnan(ranked[0]["distance_to_nearest_gt_site_um"]))

    def test_center_is_junction_and_wrong_cache_rejected(self):
        graph = FakeGraph()
        np.testing.assert_array_equal(plotter.candidate_center(graph, prediction()), [10., 12., 4.])
        for changes in ({"node_id": -1}, {"node_id": 99}, {"segment_id": 222},
                        {"x_um": 11.}, {"degree": 4}):
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, "wrong --pkl"):
                plotter.candidate_center(graph, prediction(**changes))

    def test_patch_clips_boundaries_without_shifting_origin(self):
        image = ArrayImage()
        data, origin = plotter.read_patch(image, [0., 0., 0.], [2., 3., 1.], [12., 18., 6.])
        np.testing.assert_array_equal(origin, [0, 0, 0])
        np.testing.assert_array_equal(data, image.volume[:3, :3, :3])
        with self.assertRaises(ValueError):
            plotter.read_patch(image, [-1., 0., 0.], [2., 3., 1.], [12., 18., 6.])

    def test_nine_panels_share_geometry_and_highlight_only_target_fragment(self):
        graph = FakeGraph()
        payload = {"fragments_graph": graph, "gt_graph": graph, "anisotropy": [2., 3., 1.],
                   "gt_node_canonical_label": np.array([0, 111])}
        figure, record = plotter.render_prediction(payload, ArrayImage(), prediction(), 1,
                                                   [12., 18., 6.], "score")
        self.assertEqual(len(figure.axes), 9)
        self.assertEqual(record["gt_covered_segment"], 1)
        self.assertEqual(record["x_um"], 10.)
        self.assertEqual([record["z_start_um"], record["z_stop_um"]], [1., 7.])
        self.assertIn("candidate junction node", figure._suptitle.get_text())
        self.assertIn("does not prove", figure._supxlabel.get_text())
        axes = np.asarray(figure.axes).reshape(3, 3)
        volume = ArrayImage().volume[1:7, 1:7, 2:8]
        anchors = ([6., 9.], [6., 3.], [9., 3.])
        extents = ([-1., 11., 16.5, -1.5], [-1., 11., 5.5, -0.5], [-1.5, 16.5, 5.5, -0.5])
        for column, view in enumerate(("XY", "XZ", "YZ")):
            self.assertEqual(axes[0, column].get_title(), f"Image (MIP) -- {view}")
            np.testing.assert_array_equal(axes[0, column].images[0].get_array(), volume.max(axis=column))
            np.testing.assert_allclose(axes[0, column].images[0].get_extent(), extents[column])
            for row in range(3):
                axis = axes[row, column]
                np.testing.assert_allclose(axis.collections[0].get_offsets(), [anchors[column]])
                np.testing.assert_allclose(axis.get_xlim(), extents[column][:2])
                np.testing.assert_allclose(axis.get_ylim(), extents[column][2:])
                if row > 0:
                    self.assertEqual(len(axis.images), 0)
            target = axes[2, column].collections[1]
            np.testing.assert_allclose(target.get_colors()[0][:3], to_rgba(plotter.SEGMENT_COLOR)[:3])
            self.assertEqual(target.get_linewidths()[0], 3.0)
            self.assertEqual(axes[2, column].collections[3].get_linewidths()[0], 2.0)
            self.assertEqual(axes[1, column].collections[1].get_linewidths()[0], 2.0)
        figure.canvas.draw()
        self.assertGreater(np.std(np.asarray(figure.canvas.buffer_rgba())), 0)
        figure.clear()

    def test_positive_ambiguous_and_untraced_annotations(self):
        graph = FakeGraph()
        cases = ((prediction(is_merge_site=1), "MERGE-SITE LABELLED", "not an exact cut"),
                 (prediction(in_ambiguous_ring=1, distance_to_nearest_gt_site_um=30.),
                  "AMBIGUOUS RING", "claim radius"),
                 (prediction(), "NOT SITE-LABELLED", "does not prove"))
        for candidate, status, caveat in cases:
            with self.subTest(status=status):
                payload = {"fragments_graph": graph, "gt_graph": graph, "anisotropy": [2., 3., 1.],
                           "gt_node_canonical_label": np.array([0, 222])}
                figure, record = plotter.render_prediction(payload, ArrayImage(), candidate, 1,
                                                           [12., 18., 6.], "merge_site_probability_in_sample")
                self.assertIn(status, figure._suptitle.get_text())
                self.assertIn(caveat, figure._supxlabel.get_text())
                self.assertIn("IN-SAMPLE", figure._supxlabel.get_text())
                self.assertEqual(record["gt_covered_segment"], 0)
                figure.clear()
        self.assertEqual(plotter.coverage_status(prediction(), None), (None, "Segment GT coverage: unknown"))

    def test_segment_merge_and_junction_labels_are_independent(self):
        graph = FakeGraph()
        for labels, expected, value in ((np.array([111]), "MERGE", 1),
                                        (np.array([], dtype=int), "NOT MERGE-LABELLED", 0),
                                        (None, "UNKNOWN", "")):
            with self.subTest(segment_status=expected):
                payload = {"fragments_graph": graph, "gt_graph": graph,
                           "anisotropy": [2., 3., 1.], "gt_merge_labels": labels}
                candidate = prediction()
                figure, record = plotter.render_prediction(
                    payload, ArrayImage(), candidate, 1, [12., 18., 6.], plotter.DEFAULT_SCORE)
                self.assertIn(f"Segment GT: {expected} |", figure._suptitle.get_text())
                self.assertIn("Junction GT: NOT SITE-LABELLED", figure._suptitle.get_text())
                self.assertEqual(record["gt_segment_is_merge"], value)
                self.assertEqual(record["junction_gt_status"], "NOT SITE-LABELLED")
                self.assertEqual(record["is_merge_site"], 0)
                self.assertEqual(record["score"], candidate["score"])
                if value == 1:
                    self.assertIn("Known merge segment", figure._supxlabel.get_text())
                    figure.canvas.draw()
                    renderer = figure.canvas.get_renderer()
                    title_box = figure._suptitle.get_bbox_patch().get_window_extent(renderer)
                    for axis in figure.axes[:3]:
                        self.assertFalse(title_box.overlaps(axis.title.get_window_extent(renderer)))
                figure.clear()

    def test_output_paths_are_separate_from_split_and_run_specific(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, override in (("run_a", None), ("run_b", root / "custom")):
                result_dir = root / name
                result_dir.mkdir()
                candidate = prediction()
                candidate[plotter.DEFAULT_SCORE] = candidate.pop("score")
                write_csv(result_dir / "merge_junction_detector_794495.csv", [candidate])
                argv = [str(result_dir), "--pkl", str(root / "missing.pkl")]
                if override:
                    argv.extend(["--output-dir", str(override)])
                with patch("builtins.print"), self.assertRaises(FileNotFoundError):
                    plotter.main(argv)
                self.assertTrue((override or result_dir / "top_merge_predictions").is_dir())
                self.assertFalse((result_dir / "top_split_predictions").exists())

    def test_explicit_csv_requires_matching_cache(self):
        with patch("sys.stderr"), self.assertRaises(SystemExit) as caught:
            plotter.main(["--csv", "heldout.csv"])
        self.assertEqual(caught.exception.code, 2)


if __name__ == "__main__":
    unittest.main()