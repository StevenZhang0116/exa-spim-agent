import contextlib
import csv
import hashlib
import io
import json
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib.image as mpimg
import numpy as np
from matplotlib.colors import to_rgba

from notebooks import plot_gt_merge_sites as plotter
from agentic.tests.test_plot_top_merge_predictions import ArrayImage, FakeGraph


def site(xyz=(10., 12., 4.), **changes):
    return {"segment_id": 111, "gt_neuron": "neuron-a", "xyz": xyz, **changes}


class PlotGtMergeSitesTests(unittest.TestCase):
    def test_unique_positions_reproducible_sampling_and_sources(self):
        records = [site(), site(source="two_gt_junction", gt_neurons=["neuron-a", "neuron-b"]),
                   site((12., 12., 4.), segment_id=222), site((14., 12., 4.))]
        selected, count = plotter.select_sites(records, 10, seed=3)
        self.assertEqual(count, 3)
        self.assertEqual(len({tuple(row["xyz"]) for row in selected}), 3)
        duplicate = next(row for row in selected if row["site_index"] == 0)
        self.assertEqual(duplicate["site_indices"], [0, 1])
        self.assertEqual(duplicate["sources"], ["geometric_walk", "two_gt_junction"])
        self.assertEqual(duplicate["gt_neurons"], ["neuron-a", "neuron-b"])
        self.assertEqual(plotter.select_sites(records, 2, seed=3), plotter.select_sites(records, 2, seed=3))
        filtered, count = plotter.select_sites(records, 2, source="two_gt_junction")
        self.assertEqual(count, 1)
        self.assertEqual(filtered[0]["site_index"], 1)
        self.assertNotIn("site_index", records[0])

    def test_shared_evidence_and_coincident_segments_keep_metadata(self):
        records = [site(source="geometric_walk", sources=["geometric_walk", "two_gt_junction"]),
                   site(segment_id=222)]
        selected, count = plotter.select_sites(records, 2)
        self.assertEqual(count, 1)
        self.assertEqual(selected[0]["segment_ids"], [111, 222])
        self.assertEqual(selected[0]["segment_id"], 111)
        self.assertEqual(len(plotter.select_sites(records, 1, source="two_gt_junction")[0]), 1)

    def test_invalid_sites_and_empty_selection(self):
        for record in (site((1., 2.)), site((np.nan, 0., 0.)), site(segment_id=0),
                       site(source="unexpected"), site(sources="geometric_walk")):
            with self.subTest(record=record), self.assertRaises(ValueError):
                plotter.select_sites([record], 1)
        with self.assertRaises(ValueError):
            plotter.select_sites([site()], 0)
        self.assertEqual(plotter.select_sites([], 2), ([], 0))

    def test_exact_site_center_nine_panels_no_candidate_or_score(self):
        graph = FakeGraph()
        data = {"fragments_graph": graph, "gt_graph": graph, "anisotropy": [2., 3., 1.]}
        selected = plotter.select_sites([site((11., 12., 4.))], 1)[0][0]
        figure, record = plotter.render_site(data, ArrayImage(), selected, "794495", 1, [12., 18., 6.])
        self.assertEqual(len(figure.axes), 9)
        self.assertEqual(record["x_um"], 11.)
        self.assertNotIn("score", record)
        self.assertNotIn("candidate_id", record)
        self.assertIn("GT merge site", figure._suptitle.get_text())
        self.assertNotIn("score", figure._supxlabel.get_text())
        self.assertIn("neuron-a", figure._supxlabel.get_text())
        anchors = ([7., 9.], [7., 3.], [9., 3.])
        for index, axis in enumerate(figure.axes):
            np.testing.assert_allclose(axis.collections[0].get_offsets()[0], anchors[index % 3])
        np.testing.assert_allclose(figure.axes[6].collections[1].get_colors()[0, :3], to_rgba("#00e5ff")[:3])
        figure.canvas.draw()
        self.assertGreater(np.std(np.asarray(figure.canvas.buffer_rgba())[:, :, :3]), 1.)
        renderer = figure.canvas.get_renderer()
        for axis in figure.axes[:3]:
            self.assertFalse(figure._suptitle.get_window_extent(renderer).overlaps(axis.title.get_window_extent(renderer)))
        figure.clear()

    def test_cli_saves_k_distinct_figures_without_changing_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = root / "dataset_cache_794495_mcl100_add.pkl"
            graph = FakeGraph()
            data = {"fragments_graph": graph, "gt_graph": graph, "anisotropy": [2., 3., 1.],
                    "gt_merge_sites": [site(), site(), site((12., 12., 4.), segment_id=222)],
                    "min_cable_length": 100, "img_path": "test-image"}
            cache.write_bytes(pickle.dumps(data))
            before = hashlib.sha256(cache.read_bytes()).digest()
            output = root / "plots"
            argv = ["--brain-id", "794495", "--k", "5", "--cache-dir", str(root),
                    "--output-dir", str(output), "--dpi", "50", "--patch-um", "12", "18", "6"]
            log = io.StringIO()
            with patch.object(plotter, "open_image", return_value=ArrayImage()) as image, \
                    contextlib.redirect_stdout(log):
                self.assertEqual(plotter.main(argv), 0)
            image.assert_called_once_with("test-image")
            self.assertIn("only 2 distinct positions", log.getvalue())
            self.assertEqual(len(list(output.glob("*.png"))), 2)
            with (output / "gt_merge_sites.csv").open() as handle:
                manifest = list(csv.DictReader(handle))
            self.assertEqual(len(manifest), 2)
            self.assertEqual(len({row["x_um"] for row in manifest}), 2)
            self.assertNotIn("score", manifest[0])
            for row in manifest:
                pixels = mpimg.imread(output / row["figure"])
                self.assertGreater(np.std(pixels[:, :, :3]), 0.01)
            selection = json.loads((output / "selection.json").read_text())
            self.assertEqual(selection["requested_k"], 5)
            self.assertEqual(selection["available_positions"], 2)
            self.assertEqual(hashlib.sha256(cache.read_bytes()).digest(), before)
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                plotter.main(argv)

    def test_site_at_volume_boundary_uses_clipped_patch_bounds(self):
        graph = FakeGraph()
        data = {"fragments_graph": graph, "gt_graph": graph, "anisotropy": [2., 3., 1.]}
        selected = plotter.select_sites([site((0., 0., 0.))], 1)[0][0]
        figure, record = plotter.render_site(data, ArrayImage(), selected, "794495", 1, [12., 18., 6.])
        self.assertEqual(json.loads(record["origin_zyx"]), [0, 0, 0])
        self.assertEqual(json.loads(record["shape_zyx"]), [3, 3, 3])
        for axis in figure.axes:
            np.testing.assert_allclose(axis.collections[0].get_offsets()[0], [0., 0.])
        figure.clear()

    def test_mcl_mismatch_and_invalid_cli_do_not_open_image(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = root / "dataset_cache_794495_mcl100_add.pkl"
            cache.write_bytes(pickle.dumps({"gt_graph": FakeGraph(), "min_cable_length": 10}))
            for extra in ([], ["--k", "0"], ["--patch-um", "-1", "10", "10"]):
                with self.subTest(extra=extra), patch.object(plotter, "open_image") as image, \
                        contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()), \
                        self.assertRaises(SystemExit):
                    plotter.main(["--brain-id", "794495", "--cache-dir", str(root), *extra])
                image.assert_not_called()

    def test_missing_sites_fail_before_image_access(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = root / "dataset_cache_794495_mcl100_add.pkl"
            graph = FakeGraph()
            graph.merge_sites = [site()]
            cases = [({}, "Cache has no gt_merge_sites"),
                     ({"gt_graph": FakeGraph()}, "Cache has no gt_merge_sites"),
                     ({"gt_graph": graph, "gt_merge_sites": None}, "Cache has no gt_merge_sites"),
                     ({"gt_graph": graph, "gt_merge_sites": []}, "No GT merge-site positions")]
            for payload, message in cases:
                with self.subTest(message=message, keys=list(payload)):
                    cache.write_bytes(pickle.dumps(payload))
                    error = io.StringIO()
                    with patch.object(plotter, "open_image") as image, \
                            contextlib.redirect_stderr(error), contextlib.redirect_stdout(io.StringIO()), \
                            self.assertRaises(SystemExit):
                        plotter.main(["--brain-id", "794495", "--cache-dir", str(root)])
                    self.assertIn(message, error.getvalue())
                    image.assert_not_called()

    def test_legacy_sites_used_only_when_primary_list_is_absent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = root / "dataset_cache_794495_mcl100_add.pkl"
            graph = FakeGraph()
            graph.merge_sites = [site((11., 12., 4.))]
            payload = {"fragments_graph": graph, "gt_graph": graph, "anisotropy": [2., 3., 1.],
                       "min_cable_length": 100, "img_path": "test-image"}
            for name, primary, expected_x in (("legacy", None, 11.), ("primary", [site()], 10.)):
                with self.subTest(name=name):
                    data = dict(payload)
                    if primary is not None:
                        data["gt_merge_sites"] = primary
                    cache.write_bytes(pickle.dumps(data))
                    output = root / name
                    with patch.object(plotter, "open_image", return_value=ArrayImage()), \
                            contextlib.redirect_stdout(io.StringIO()):
                        self.assertEqual(plotter.main([
                            "--brain-id", "794495", "--k", "1", "--cache-dir", str(root),
                            "--output-dir", str(output), "--dpi", "40", "--patch-um", "12", "18", "6"]), 0)
                    selection = json.loads((output / "selection.json").read_text())
                    self.assertEqual(selection["selected_positions"][0]["xyz"][0], expected_x)


if __name__ == "__main__":
    unittest.main()