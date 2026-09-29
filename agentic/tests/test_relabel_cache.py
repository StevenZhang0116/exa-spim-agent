import importlib.util
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scripts import relabel_cache


UPSTREAM_TEST = (Path(__file__).resolve().parents[3] / "agentic-neuron-proofreader"
                 / "tests" / "test_junction_gt_audit.py")
SPEC = importlib.util.spec_from_file_location("junction_fixture", UPSTREAM_TEST)
fixture = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fixture)


class RelabelCacheTests(unittest.TestCase):
    def test_refresh_unifies_sites_and_discards_stale_candidate_labels(self):
        fragment, gt, labels = fixture.crossing_graphs()
        payload = {"fragments_graph": fragment, "gt_graph": gt,
                   "gt_node_canonical_label": labels, "gt_merge_labels": np.array([111, 222]),
                   "gt_merge_sites": [{"segment_id": 222, "gt_neuron": "neuron-a", "xyz": [999., 0., 0.]}],
                   "img_path": "preserved-image", "custom_key": "preserved",
                   "__detector_sample_universe_cache__": "stale",
                   "__detector_universe_site_audit__": "stale"}
        with patch.object(relabel_cache, "segmentation_path_for", side_effect=AssertionError("No cloud")):
            summary = relabel_cache.refresh_payload_merge_sites(payload, verbose=False)
        self.assertEqual(summary["previous_site_count"], 1)
        self.assertEqual(summary["n_combined_sites"], 2)
        self.assertEqual(payload["gt_merge_sites"][1]["node_id"], 0)
        self.assertIs(payload["gt_merge_sites"], gt.merge_sites)
        np.testing.assert_array_equal(payload["gt_node_canonical_label"], labels)
        np.testing.assert_array_equal(payload["gt_merge_labels"], [111, 222])
        self.assertNotIn("__detector_sample_universe_cache__", payload)
        self.assertNotIn("__detector_universe_site_audit__", payload)
        self.assertEqual(payload["custom_key"], "preserved")
        self.assertEqual(payload["img_path"], "preserved-image")
        repeated = relabel_cache.refresh_payload_merge_sites(payload, verbose=False)
        self.assertEqual(summary["site_identity_sha256"], repeated["site_identity_sha256"])

    def test_atomic_replacement_and_failure_preserve_original(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.pkl"
            path.write_bytes(pickle.dumps({"old": True}))
            before = path.read_bytes()

            def fail(temporary):
                temporary.write_bytes(b"partial")
                raise RuntimeError("write failed")

            with self.assertRaisesRegex(RuntimeError, "write failed"):
                relabel_cache.atomic_cache_write(path, fail)
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(list(Path(directory).iterdir()), [path])
            signature = relabel_cache.cache_signature(path)
            relabel_cache.atomic_cache_write(path, lambda temporary: temporary.write_bytes(
                pickle.dumps({"new": True})), signature)
            self.assertEqual(pickle.loads(path.read_bytes()), {"new": True})
            with self.assertRaisesRegex(RuntimeError, "changed"):
                relabel_cache.atomic_cache_write(path, lambda temporary: temporary.write_bytes(before), signature)
            self.assertEqual(pickle.loads(path.read_bytes()), {"new": True})

    def test_refresh_dry_run_selects_only_requested_add_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("dataset_cache_794495_mcl100.pkl", "dataset_cache_794495_mcl100_add.pkl",
                         "dataset_cache_794495_mcl10_add.pkl", "dataset_cache_794493_mcl100_add.pkl"):
                (root / name).touch()
            with patch.object(relabel_cache, "segmentation_path_for", side_effect=AssertionError("No cloud")), \
                    patch("builtins.print") as output:
                relabel_cache.main(["--cache-dir", directory, "--brain", "794495", "--mcl", "100",
                                    "--refresh-merge-sites", "--dry-run"])
            text = " ".join(str(call) for call in output.call_args_list)
            self.assertIn("dataset_cache_794495_mcl100_add.pkl", text)
            self.assertNotIn("794493", text)
            self.assertNotIn("mcl10_add", text)


if __name__ == "__main__":
    unittest.main()