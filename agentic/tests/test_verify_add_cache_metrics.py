import contextlib
import hashlib
import io
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import networkx as nx
import numpy as np
import pandas as pd

from notebooks import verify_add_cache_metrics as verify


class CacheGraph(nx.Graph):
    def node_segment_id(self, node):
        return ("A", "B", "C")[node // 2]

    def cable_length(self, root):
        return float(root + 1)


def payload(sites=None):
    graph = CacheGraph([(0, 1), (2, 3), (4, 5)])
    graph.node_xyz = np.zeros((6, 3))
    return {
        "gt_graph": graph, "gt_node_canonical_label": np.array([111, 111, 111, 111, 222, 0]),
        "gt_edge_error": np.array([0, 1, 2]), "gt_merge_labels": np.array([111]),
        "gt_merge_sites": sites, "min_cable_length": 100, "segmentation_path": "test-segmentation",
        "img_path": "unreachable-image-that-must-not-be-opened",
    }


class VerifyCacheMetricsTests(unittest.TestCase):
    def setUp(self):
        self.quiet = contextlib.redirect_stdout(io.StringIO())
        self.quiet.__enter__()

    def tearDown(self):
        self.quiet.__exit__(None, None, None)

    def test_legacy_and_combined_sources_preserve_comparison_metrics(self):
        legacy = [{"gt_neuron": "A"}, {"gt_neuron": "B", "source": "geometric_walk"}]
        baseline, _, _ = verify.summarize_payload(payload(legacy))
        extra = {"gt_neuron": "C", "source": "two_gt_junction", "gt_neurons": ["B", "C"]}
        combined, info, _ = verify.summarize_payload(payload([*legacy, extra]))
        pd.testing.assert_frame_equal(baseline[verify.COMPARE_COLS], combined[verify.COMPARE_COLS])
        self.assertEqual(combined["# Merges"].sum(), 2)
        self.assertEqual(combined["# Combined Sites"].sum(), 3)
        self.assertEqual(combined.loc["C", "# Two-GT-only Sites"], 1)
        self.assertEqual(combined.loc["B", "# Two-GT-only Sites"], 0)
        self.assertEqual(info["counts"]["# Combined Sites"], 3)
        self.assertNotIn("source", legacy[0])

    def test_shared_evidence_is_not_double_counted(self):
        shared = {"gt_neuron": "A", "source": "geometric_walk",
                  "sources": ["geometric_walk", "two_gt_junction"]}
        _, info, _ = verify.summarize_payload(payload([
            shared, {"gt_neuron": "B", "source": "two_gt_junction"}]))
        self.assertEqual(info["counts"], {"# Geometric Sites": 1, "# Two-GT-only Sites": 1,
                                          "# Shared-evidence Sites": 1, "# Combined Sites": 2})

    def test_unknown_sources_are_rejected(self):
        for site in ({"source": "other"}, {"sources": "two_gt_junction"}):
            with self.subTest(site=site), self.assertRaises(ValueError):
                verify.partition_merge_sites([site])

    def test_missing_sites_are_nan_but_empty_or_two_gt_only_are_zero(self):
        frame, info, _ = verify.summarize_payload(payload(None))
        self.assertTrue(frame["# Merges"].isna().all())
        comparison = verify.compare_metrics(frame, frame)
        checks = verify.consistency_checks(frame, frame, comparison, info)
        self.assertFalse(any("geometric only" in check["name"] for check in checks))
        figure = verify.make_scatter(comparison, "test", 100)
        self.assertEqual(figure.axes[1].texts[0].get_text(), "No comparable data")
        figure.clear()
        for sites in ([], [{"gt_neuron": "A", "source": "two_gt_junction"}]):
            frame, info, _ = verify.summarize_payload(payload(sites))
            self.assertEqual(frame["# Merges"].sum(), 0)
            self.assertFalse(frame["# Merges"].isna().any())
            self.assertTrue(info["available"])

    def test_top_level_fields_and_graph_only_legacy_payload(self):
        data = payload([])
        graph = data["gt_graph"]
        for key, attribute in (("gt_node_canonical_label", "node_label"),
                               ("gt_edge_error", "edge_error"), ("gt_merge_labels", "merge_labels"),
                               ("gt_merge_sites", "merge_sites")):
            setattr(graph, attribute, data[key])
        legacy, _, _ = verify.summarize_payload({"gt_graph": graph})
        frame, _, _ = verify.summarize_payload(data)
        pd.testing.assert_frame_equal(legacy, frame)
        graph.merge_sites = [{"gt_neuron": "A"}]
        _, info, _ = verify.summarize_payload(data)
        self.assertEqual(info["counts"]["# Combined Sites"], 0)

    def test_invalid_label_arrays_rejected(self):
        for changes in ({"gt_node_canonical_label": None}, {"gt_edge_error": np.array([0])},
                        {"gt_node_canonical_label": np.array([111])}, {"gt_edge_error": np.array([0, 1, 99])}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                verify.summarize_payload({**payload([]), **changes})

    def test_combined_counts_do_not_trigger_canonical_failure(self):
        frame, info, _ = verify.summarize_payload(payload([
            {"gt_neuron": "A"}, *[{"gt_neuron": "B", "source": "two_gt_junction"} for _ in range(10)]]))
        canonical = frame.copy()
        checks = verify.consistency_checks(frame, canonical, verify.compare_metrics(frame, canonical), info)
        self.assertFalse(any(check["status"] == "FAIL" for check in checks))
        canonical.loc["A", "# Merges"] = 0
        checks = verify.consistency_checks(frame, canonical, verify.compare_metrics(frame, canonical), info)
        self.assertIn({"name": "# Merges (geometric only): cache <= canonical", "status": "WARN"}, checks)

    def test_no_common_or_duplicate_neurons_rejected(self):
        frame, _, _ = verify.summarize_payload(payload([]))
        other = frame.copy()
        other.index = ["D", "E", "F"]
        with self.assertRaisesRegex(ValueError, "No shared"):
            verify.compare_metrics(frame, other)
        other.index = ["A", "A", "C"]
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            verify.compare_metrics(frame, other)

    def test_exports_preserve_schema_and_whole_cache_totals(self):
        frame, info, _ = verify.summarize_payload(payload([
            {"gt_neuron": "A"}, {"gt_neuron": "B", "source": "two_gt_junction"},
            {"gt_neuron": "outside", "source": "two_gt_junction"}]))
        comparison = verify.compare_metrics(frame, frame)
        figure = verify.make_scatter(comparison, "test", 100)
        self.assertEqual(figure.axes[1].get_title(), "# Merges (geometric only)")
        figure.clear()
        with tempfile.TemporaryDirectory() as directory:
            verify.write_results(frame, frame, comparison, info, "123", 100, "test", directory)
            root = Path(directory)
            per_neuron = pd.read_csv(root / "123_mcl100_per_neuron.csv")
            summary = pd.read_csv(root / "123_mcl100_summary.csv").iloc[0]
            for column in verify.COMPARE_COLS:
                for kind in ("cache", "canon", "diff"):
                    self.assertIn(f"{column}_{kind}", per_neuron)
            self.assertEqual(per_neuron["# Merges_cache"].sum(), 1)
            self.assertEqual(per_neuron["# Combined Sites_cache"].sum(), 2)
            self.assertEqual(summary["n_combined_sites"], 3)
            self.assertEqual(summary["n_sites_outside_common_neurons"], 1)
            self.assertEqual(summary["merge_count_definition"], "geometric_walk_sites")
            self.assertEqual(summary["site_total_scope"], "whole_cache")
            self.assertTrue((root / "123_mcl100_scatter.png").exists())

    def test_discovery_filters_mcl_brains_and_exact_add_filenames(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("dataset_cache_111_mcl100_add.pkl", "dataset_cache_222_mcl100_add.pkl",
                         "dataset_cache_333_mcl10_add.pkl", "dataset_cache_444_mcl100.pkl",
                         "dataset_cache_555_mcl100_add.pkl.backup"):
                (root / name).touch()
            self.assertEqual([brain for brain, _ in verify.discover_caches(root, 100)], ["111", "222"])
            self.assertEqual([brain for brain, _ in verify.discover_caches(root, 100, ["222"])], ["222"])
            with self.assertRaisesRegex(ValueError, "requested brains"):
                verify.discover_caches(root, 100, ["333"])
            with self.assertRaisesRegex(ValueError, "No mcl"):
                verify.discover_caches(root, 20)

    def write_inputs(self, root, brain, mcl=100, with_metrics=True):
        cache_dir = root / "cache"
        cache_dir.mkdir(exist_ok=True)
        data = {**payload([{"gt_neuron": "A"}]), "min_cable_length": mcl}
        cache = cache_dir / f"dataset_cache_{brain}_mcl{mcl}_add.pkl"
        cache.write_bytes(pickle.dumps(data))
        if with_metrics:
            metrics = root / "metrics" / brain / "segmentation"
            metrics.mkdir(parents=True)
            verify.summarize_payload(data)[0].to_csv(metrics / "results.csv")
        return cache

    def arguments(self, root):
        return ["--cache-dir", str(root / "cache"), "--metrics-dir", str(root / "metrics"),
                "--output-dir", str(root / "output"), "--mcl", "100"]

    def test_cli_processes_all_same_mcl_without_modifying_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_inputs(root, "111")
            self.write_inputs(root, "222")
            self.write_inputs(root, "333", with_metrics=False)
            self.write_inputs(root, "444", mcl=10)
            inputs = {path: hashlib.sha256(path.read_bytes()).digest() for path in root.rglob("*") if path.is_file()}
            with patch.object(verify, "verify_brain", wraps=verify.verify_brain) as run:
                self.assertEqual(verify.main(self.arguments(root)), 0)
            self.assertEqual([call.args[2] for call in run.call_args_list], ["111", "222"])
            self.assertEqual(len(list((root / "output").iterdir())), 6)
            for path, digest in inputs.items():
                self.assertEqual(hashlib.sha256(path.read_bytes()).digest(), digest)

    def test_cli_continues_after_failed_brain_and_returns_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bad = self.write_inputs(root, "111")
            bad.write_bytes(b"not a pickle")
            self.write_inputs(root, "222")
            self.assertEqual(verify.main(self.arguments(root)), 1)
            self.assertTrue((root / "output" / "222_mcl100_summary.csv").exists())
            self.assertFalse((root / "output" / "111_mcl100_summary.csv").exists())

    def test_cli_brain_filter_and_mcl_metadata_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_inputs(root, "111")
            selected = self.write_inputs(root, "222")
            self.assertEqual(verify.main(self.arguments(root) + ["--brain", "222"]), 0)
            self.assertFalse((root / "output" / "111_mcl100_summary.csv").exists())
            data = payload([])
            data["min_cable_length"] = 10
            selected.write_bytes(pickle.dumps(data))
            with self.assertRaisesRegex(ValueError, "disagrees"):
                verify.verify_brain(selected, root / "metrics" / "222" / "segmentation" / "results.csv",
                                    "222", 100, root / "output")

    def test_missing_or_ambiguous_canonical_never_loads_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_inputs(root, "111", with_metrics=False)
            with patch.object(verify, "verify_brain") as run:
                self.assertEqual(verify.main(self.arguments(root)), 1)
                run.assert_not_called()
            for run_name in ("first", "second"):
                location = root / "metrics" / "111" / run_name
                location.mkdir(parents=True)
                (location / "results.csv").touch()
            with patch.object(verify, "verify_brain") as run:
                self.assertEqual(verify.main(self.arguments(root)), 1)
                run.assert_not_called()


if __name__ == "__main__":
    unittest.main()