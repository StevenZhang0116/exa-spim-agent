import contextlib
import csv
import hashlib
import io
import json
import pickle
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import matplotlib.image as mpimg
from matplotlib.colors import to_rgba
import numpy as np

from notebooks import plot_positive_merge_sites as plotter
from agentic.tests.test_plot_top_merge_predictions import prediction, write_csv


def write_cache(root, sites):
    path = root / "dataset_cache_794495_mcl100_add.pkl"
    path.write_bytes(pickle.dumps({"min_cable_length": 100, "gt_merge_sites": sites}))
    return path


class PositiveMergeSitesTests(unittest.TestCase):
    def test_all_positives_independent_of_scores_rings_or_shared_location(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.csv"
            rows = [prediction(candidate_id=9007199254740993, is_merge_site=1, score="nan"),
                    prediction(candidate_id=2, is_merge_site=1, score=""),
                    prediction(candidate_id=3, is_merge_site=0, score=1., in_ambiguous_ring=1),
                    prediction(candidate_id=4, is_merge_site=0, score=.99)]
            write_csv(path, rows)
            selected = plotter.load_positive_candidates(path)
            self.assertEqual([row["candidate_id"] for row in selected], [2, 9007199254740993])
            self.assertNotIn("score", selected[0])
            self.assertEqual(selected[0]["x_um"], selected[1]["x_um"])
            for row in rows:
                del row["score"]
            write_csv(path, rows)
            self.assertEqual(plotter.load_positive_candidates(path), selected)

    def test_invalid_or_duplicate_csv_rows_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.csv"
            for rows in ([prediction(), prediction()], [prediction(is_merge_site=2)],
                         [prediction(is_merge_site=1, x_um=float("nan"))],
                         [prediction(is_merge_site=1, degree=2)]):
                with self.subTest(rows=rows), self.assertRaises(ValueError):
                    write_csv(path, rows)
                    plotter.load_positive_candidates(path)
            path.write_text("segment_id,is_merge\n111,1\n")
            with self.assertRaisesRegex(ValueError, "Junction CSV"):
                plotter.load_positive_candidates(path)

    def test_one_figure_all_positive_points_and_world_coordinate_projections(self):
        rows = [prediction(candidate_id=1, is_merge_site=1, x_um=1000., y_um=2000., z_um=3000., score=""),
                prediction(candidate_id=2, is_merge_site=1, x_um=4000., y_um=5000., z_um=6000., score="nan"),
                prediction(candidate_id=3, is_merge_site=0, x_um=10000., y_um=12000., z_um=15000.)]
        sites = [{"xyz": [1000., 2000., 3000.]}, {"xyz": [20000., -5000., 30000.]},
                 {"xyz": [1000., 2000., 3000.]}]
        figure = plotter.render_distribution(rows, "794495", 100, sites)
        self.assertEqual(len(figure.axes), 2)
        self.assertIn("Brain 794495 | mcl100", figure._suptitle.get_text())
        self.assertIn("All GT site records and positive rows retained", figure._supxlabel.get_text())
        expected = ([[1., 2.], [4., 5.]], [[1., 3.], [4., 6.]])
        expected_gt = ([[1., 2.], [20., -5.], [1., 2.]], [[1., 3.], [20., 30.], [1., 3.]])
        self.assertEqual([axis.get_title() for axis in figure.axes], ["XY", "XZ"])
        for axis, coordinates, gt_coordinates, (view, horizontal, vertical) in zip(
                figure.axes, expected, expected_gt, plotter.VIEWS):
            self.assertEqual(axis.get_title(), view)
            self.assertEqual(axis.get_xlabel(), f"{'XYZ'[horizontal]} (mm)")
            self.assertEqual(axis.get_ylabel(), f"{'XYZ'[vertical]} (mm)")
            self.assertEqual(len(axis.collections[0].get_offsets()), 3)
            np.testing.assert_allclose(axis.collections[1].get_offsets(), coordinates)
            np.testing.assert_allclose(axis.collections[2].get_offsets(), gt_coordinates)
            self.assertEqual(len(axis.collections[2].get_facecolors()), 0)
            np.testing.assert_allclose(axis.collections[2].get_edgecolors()[0], to_rgba(plotter.GT_COLOR))
            np.testing.assert_allclose(axis.collections[1].get_facecolors()[0][:3], to_rgba(plotter.POSITIVE_COLOR)[:3])
            self.assertGreater(axis.collections[2].get_sizes()[0], axis.collections[1].get_sizes()[0])
            self.assertEqual(axis.get_aspect(), 1.)
        np.testing.assert_allclose(figure.axes[0].get_xlim(), figure.axes[1].get_xlim())
        self.assertGreater(figure.axes[0].get_xlim()[1], 20.)
        self.assertLess(figure.axes[0].get_ylim()[0], -5.)
        self.assertGreater(figure.axes[1].get_ylim()[1], 30.)
        labels = [text.get_text() for text in figure.legends[0].texts]
        self.assertEqual(labels, ["All candidate positions (3)", "gt_merge_sites (3)", "is_merge_site = 1 (2)"])
        figure.canvas.draw()
        self.assertGreater(np.std(np.asarray(figure.canvas.buffer_rgba())[:, :, :3]), 1.)
        renderer = figure.canvas.get_renderer()
        legend_box = figure.legends[0].get_window_extent(renderer)
        self.assertFalse(legend_box.overlaps(figure._suptitle.get_window_extent(renderer)))
        for axis in figure.axes:
            self.assertFalse(legend_box.overlaps(axis.title.get_window_extent(renderer)))
            self.assertFalse(figure._supxlabel.get_window_extent(renderer).overlaps(
                axis.xaxis.label.get_window_extent(renderer)))
        figure.clear()

    def test_coincident_positive_rows_are_retained_with_valid_limits(self):
        rows = [prediction(candidate_id=1, is_merge_site=1), prediction(candidate_id=2, is_merge_site=1)]
        sites = [{"xyz": [rows[0][key] for key in ("x_um", "y_um", "z_um")]}] * 2
        figure = plotter.render_distribution(rows, "794495", 100, sites)
        for axis in figure.axes:
            self.assertEqual(len(axis.collections[1].get_offsets()), 2)
            self.assertEqual(len(axis.collections[2].get_offsets()), 2)
            self.assertLess(*axis.get_xlim())
            self.assertLess(*axis.get_ylim())
        figure.clear()

    def test_cli_saves_every_positive_and_leaves_inputs_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "predictions.csv"
            write_csv(path, [prediction(candidate_id=1, is_merge_site=1, score=""),
                             prediction(candidate_id=2, is_merge_site=1, node_id=1, segment_id=222, x_um=12.),
                             prediction(candidate_id=3, is_merge_site=0, in_ambiguous_ring=1)])
            cache = write_cache(root, [{"segment_id": 111, "xyz": [10., 20., 30.]},
                                      {"segment_id": 222, "xyz": [15., 25., 35.]},
                                      {"segment_id": 333, "xyz": [10., 20., 30.]}])
            before = {file: hashlib.sha256(file.read_bytes()).digest() for file in (path, cache)}
            output = root / "plots"
            argv = ["--brain-id", "794495", "--csv", str(path), "--cache-dir", str(root),
                    "--output-dir", str(output), "--dpi", "60"]
            with patch.object(pickle, "load", wraps=pickle.load) as load, \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(plotter.main(argv), 0)
            load.assert_called_once()
            with (output / "positive_merge_sites.csv").open() as handle:
                records = list(csv.DictReader(handle))
            self.assertEqual([row["candidate_id"] for row in records], ["1", "2"])
            self.assertEqual(len(list(output.glob("*.png"))), 1)
            self.assertEqual(len({row["figure"] for row in records}), 1)
            self.assertGreater(np.std(mpimg.imread(output / records[0]["figure"])[:, :, :3]), .01)
            selection = json.loads((output / "selection.json").read_text())
            self.assertEqual(selection["candidate_count"], 2)
            self.assertEqual(selection["total_candidate_count"], 3)
            self.assertEqual(selection["plot_units"], "mm")
            self.assertEqual(selection["views"], ["XY", "XZ"])
            self.assertEqual(selection["gt_site_count"], 3)
            self.assertEqual(selection["unique_gt_xyz_count"], 2)
            self.assertEqual(selection["cache"], str(cache.resolve()))
            self.assertEqual(selection["cache_size_bytes"], cache.stat().st_size)
            self.assertEqual(selection["unique_positive_xyz_count"], 2)
            self.assertEqual(selection["csv_sha256"], before[path].hex())
            self.assertEqual(records[0]["x_um"], "10.0")
            with (output / "gt_merge_sites.csv").open() as handle:
                gt_records = list(csv.DictReader(handle))
            self.assertEqual([row["site_index"] for row in gt_records], ["0", "1", "2"])
            self.assertEqual([row["segment_id"] for row in gt_records], ["111", "222", "333"])
            self.assertEqual([float(row["x_um"]) for row in gt_records], [10., 15., 10.])
            self.assertEqual({row["figure"] for row in gt_records}, {selection["figure"]})
            for file, digest in before.items():
                self.assertEqual(hashlib.sha256(file.read_bytes()).digest(), digest)
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                plotter.main(argv)

    def test_no_positives_still_produces_one_explicit_empty_distribution(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.csv"
            write_csv(path, [prediction(in_ambiguous_ring=1)])
            sites = [{"segment_id": 111, "xyz": [10., 20., 30.]}]
            write_cache(Path(directory), sites)
            output = Path(directory) / "plots"
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(plotter.main(["--brain-id", "794495", "--csv", str(path),
                                               "--cache-dir", directory,
                                               "--output-dir", str(output)]), 0)
            self.assertEqual(len(list(output.glob("*.png"))), 1)
            self.assertEqual(json.loads((output / "selection.json").read_text())["candidate_count"], 0)
            figure = plotter.render_distribution(plotter.load_candidates(path), "794495", 100, sites)
            for axis in figure.axes:
                self.assertEqual(len(axis.collections[1].get_offsets()), 0)
                self.assertEqual(len(axis.collections[2].get_offsets()), 1)
                self.assertIn("No is_merge_site=1", axis.texts[0].get_text())
            figure.clear()

    def test_empty_csv_and_invalid_cli(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "predictions.csv"
            path.write_text(",".join(plotter.IDENTITY_COLUMNS) + "\n")
            write_cache(root, [])
            output = root / "plots"
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(plotter.main(["--brain-id", "794495", "--csv", str(path),
                                               "--cache-dir", str(root),
                                               "--output-dir", str(output)]), 0)
            self.assertFalse(output.exists())
            write_cache(root, [{"segment_id": 111, "xyz": [1000., 2000., 3000.]}])
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(plotter.main(["--brain-id", "794495", "--csv", str(path),
                                               "--cache-dir", str(root), "--output-dir", str(output)]), 0)
            self.assertEqual(len(list(output.glob("*.png"))), 1)
            self.assertEqual(json.loads((output / "selection.json").read_text())["gt_site_count"], 1)
            for extra in (["--dpi", "0"], ["--mcl", "-1"], ["--brain-id", "../brain"]):
                with self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()), \
                        self.assertRaises(SystemExit):
                    plotter.main(["--brain-id", "794495", "--csv", str(path), *extra])

    def test_empty_sets_render_independently(self):
        for rows, sites in (([], [{"xyz": [1000., 2000., 3000.]}]),
                            ([prediction(is_merge_site=1)], [])):
            figure = plotter.render_distribution(rows, "794495", 100, sites)
            for axis in figure.axes:
                self.assertEqual([len(collection.get_offsets()) for collection in axis.collections],
                                 [len(rows), len(rows), len(sites)])
                self.assertTrue(axis.texts)
            figure.canvas.draw()
            figure.clear()
        with self.assertRaises(ValueError):
            plotter.render_distribution([], "794495", 100, [])

    def test_gt_loader_validation_and_legacy_fallback(self):
        site = {"segment_id": 111, "xyz": [1000., 2000., 3000.]}
        with tempfile.TemporaryDirectory() as directory:
            path = write_cache(Path(directory), [site, site])
            loaded = plotter.load_gt_sites(path, 100)
            self.assertEqual([row["site_index"] for row in loaded], [0, 1])
            self.assertEqual(loaded[0]["xyz"], site["xyz"])
            for payload in ({}, {"gt_merge_sites": None},
                            {"gt_merge_sites": [site], "min_cable_length": 10},
                            {"gt_merge_sites": [{**site, "xyz": [1., 2.]}]},
                            {"gt_merge_sites": [{**site, "xyz": [1., 2., float("nan")]}]},
                            {"gt_merge_sites": [{**site, "segment_id": 0}]}):
                with self.subTest(payload=payload), self.assertRaises(ValueError):
                    path.write_bytes(pickle.dumps(payload))
                    plotter.load_gt_sites(path, 100)
            path.write_bytes(pickle.dumps({"gt_graph": SimpleNamespace(merge_sites=[site])}))
            self.assertEqual(plotter.load_gt_sites(path, 100), loaded[:1])
            path.write_bytes(pickle.dumps({"gt_merge_sites": [], "gt_graph": SimpleNamespace(merge_sites=[site])}))
            self.assertEqual(plotter.load_gt_sites(path, 100), [])


if __name__ == "__main__":
    unittest.main()