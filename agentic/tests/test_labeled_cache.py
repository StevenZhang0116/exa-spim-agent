"""Tests of the labeled-cache builder (``labeled_cache``): policies, stage planning and the
in-place policy stage on a synthetic cache. No brain is loaded and nothing is read from GCS."""

from __future__ import annotations

import contextlib
import io as stringio
import pickle
import sys
import tempfile
import unittest
from pathlib import Path

import networkx as nx
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from labeled_cache import __main__ as cli  # noqa: E402
from labeled_cache import build, io, policy  # noqa: E402
from labeled_cache.build import BuildRequest, ReproductionMismatch, apply_policy, plan  # noqa: E402


class GtGraph(nx.Graph):
    """Two GT neurons, straight chains with a fixed node spacing (picklable, top level)."""

    def __init__(self, nodes_per_neuron=30, spacing=4.0):
        super().__init__()
        n = nodes_per_neuron
        a = np.column_stack([np.arange(n) * spacing, np.zeros(n), np.zeros(n)])
        b = np.column_stack([np.zeros(n), np.arange(n) * spacing + 500.0, np.zeros(n)])
        self.node_xyz = np.concatenate([a, b])
        self.node_component_id = np.repeat([0, 1], n)
        self.component_id_to_swc_id = {0: "neuron-a", 1: "neuron-b"}
        self.add_nodes_from(range(2 * n))
        self.add_edges_from((i, i + 1) for i in range(n - 1))
        self.add_edges_from((n + i, n + i + 1) for i in range(n - 1))

    def node_segment_id(self, node):
        return self.component_id_to_swc_id[int(self.node_component_id[node])]


def synthetic_payload(merge_labels=(), extra=True):
    """A labeled cache whose one segment (id 5) covers 30 nodes of each neuron."""
    gt = GtGraph()
    labels = np.full(60, 5, dtype=np.int64)
    payload = {
        "fragments_path": "gs://x/fragments", "gt_path": "gs://x/gt", "img_path": "s3://x/img",
        "segmentation_path": "gs://x/seg", "anisotropy": (0.748, 0.748, 1.0), "min_cable_length": 100,
        "node_spacing": 5.0, "fragments_graph": None, "gt_graph": gt,
        "gt_node_canonical_label": labels, "gt_edge_error": np.zeros(gt.number_of_edges(), dtype=np.uint8),
        "gt_merge_labels": np.array(sorted(merge_labels), dtype=np.int64), "gt_merge_sites": [],
        "gt_merge_site_metadata": None, "gt_junction_audit": None,
    }
    if extra:
        payload["__detector_sample_universe_cache__"] = {"stale": True}
    return payload


def write_payload(directory, payload, brain="111", mcl=100):
    path = io.labeled_cache_path(directory, brain, mcl)
    with open(path, "wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def request(directory, **overrides):
    options = dict(brain="111", mcl=100, cache_dir=Path(directory), verify_gt=False, verbose=False)
    options.update(overrides)
    return BuildRequest(**options)


class PolicyTests(unittest.TestCase):
    def test_policy_matching_rules(self):
        self.assertTrue(policy.policy_matches(None, policy.POLICY_ORIGINAL))
        self.assertTrue(policy.policy_matches({"policy": "nodes", "min_nodes": 50, "passthru_min_cc": 50}, policy.POLICY_ORIGINAL))
        self.assertFalse(policy.policy_matches({"policy": "nodes", "min_nodes": 12, "passthru_min_cc": 12}, policy.POLICY_ORIGINAL))
        self.assertFalse(policy.policy_matches(None, policy.POLICY_LENGTH_V1))
        length = {"policy": "length", "length_um": 50.0, "min_nodes": 12, "passthru_min_cc": 12, "reused_geometric_sites": False}
        self.assertTrue(policy.policy_matches(length, policy.POLICY_LENGTH_V1, 50.0))
        self.assertFalse(policy.policy_matches(length, policy.POLICY_LENGTH_V1, 40.0))
        self.assertFalse(policy.policy_matches({**length, "reused_geometric_sites": True}, policy.POLICY_LENGTH_V1, 50.0))
        with self.assertRaises(ValueError):
            policy.policy_matches(None, "bogus")

    def test_thresholds_for_each_policy(self):
        original = policy.thresholds_for(policy.POLICY_ORIGINAL)
        self.assertEqual((original.min_nodes, original.passthru_min_cc, original.policy), (50, 50, "nodes"))
        scaled = policy.thresholds_for(policy.POLICY_LENGTH_V1, GtGraph(spacing=4.0), 50.0)
        self.assertEqual((scaled.min_nodes, scaled.passthru_min_cc, scaled.policy, scaled.length_um), (12, 12, "length", 50.0))
        with self.assertRaises(ValueError):
            policy.thresholds_for(policy.POLICY_LENGTH_V1)


class AtomicWriteTests(unittest.TestCase):
    def test_atomic_replacement_and_failure_preserve_original(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.pkl"
            path.write_bytes(pickle.dumps({"old": True}))
            before = path.read_bytes()

            def fail(temporary):
                temporary.write_bytes(b"partial")
                raise RuntimeError("write failed")

            with self.assertRaisesRegex(RuntimeError, "write failed"):
                io.atomic_cache_write(path, fail)
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(list(Path(directory).iterdir()), [path])
            signature = io.cache_signature(path)
            io.atomic_cache_write(path, lambda temporary: temporary.write_bytes(pickle.dumps({"new": True})), signature)
            self.assertEqual(pickle.loads(path.read_bytes()), {"new": True})
            with self.assertRaisesRegex(RuntimeError, "changed"):
                io.atomic_cache_write(path, lambda temporary: temporary.write_bytes(before), signature)
            self.assertEqual(pickle.loads(path.read_bytes()), {"new": True})


class PlanTests(unittest.TestCase):
    def test_stages_follow_the_files_present(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(plan(request(directory)), ["base", "label"])
            io.base_cache_path(directory, "111", 100).touch()
            self.assertEqual(plan(request(directory)), ["label"])
            io.labeled_cache_path(directory, "111", 100).touch()
            self.assertEqual(plan(request(directory)), ["policy"])
            self.assertEqual(plan(request(directory, rebuild_base=True)), ["base", "label"])
        self.assertEqual(io.parse_cache_name("cache/dataset_cache_794495_mcl100_add.pkl"), ("794495", 100, True))
        self.assertEqual(io.parse_cache_name("cache/dataset_cache_794495_mcl10.pkl"), ("794495", 10, False))
        self.assertIsNone(io.parse_cache_name("cache/other.pkl"))


class PolicyStageTests(unittest.TestCase):
    def test_relabel_in_place_then_skip_then_force(self):
        with tempfile.TemporaryDirectory() as directory:
            path = write_payload(directory, synthetic_payload())
            result = apply_policy(request(directory, check_reproduce=True), log=lambda *_: None)
            self.assertEqual(result["action"], "relabeled")
            self.assertTrue(result["reproduction"]["labels_equal"] and result["reproduction"]["sites_equal"])
            self.assertEqual(result["previous"]["merge_labels"], 0)
            self.assertEqual(result["current"]["merge_labels"], 1)        # 30 > 12 nodes on both neurons
            with open(path, "rb") as handle:
                payload = pickle.load(handle)
            np.testing.assert_array_equal(payload["gt_merge_labels"], [5])
            self.assertEqual(payload["gt_label_policy"]["policy"], "length")
            self.assertEqual(payload["gt_label_policy"]["min_nodes"], 12)
            self.assertEqual(payload["gt_merge_site_metadata"]["thresholds"]["min_nodes"], 12)
            self.assertNotIn("__detector_sample_universe_cache__", payload)
            self.assertEqual(int((payload["gt_edge_error"] == 3).sum()), payload["gt_graph"].number_of_edges())
            self.assertEqual(payload["gt_graph"].label_policy, payload["gt_label_policy"])
            # same policy again: skipped, file untouched
            signature = io.cache_signature(path)
            self.assertEqual(apply_policy(request(directory), log=lambda *_: None)["action"], "skipped")
            self.assertEqual(io.cache_signature(path), signature)
            # forced: relabeled again; back to the original policy drops the merge label
            self.assertEqual(apply_policy(request(directory, force_relabel=True), log=lambda *_: None)["action"], "relabeled")
            back = apply_policy(request(directory, policy=policy.POLICY_ORIGINAL), log=lambda *_: None)
            self.assertEqual(back["action"], "relabeled")
            self.assertEqual(back["dropped_merge_labels"], [5])
            with open(path, "rb") as handle:
                self.assertEqual(len(pickle.load(handle)["gt_merge_labels"]), 0)

    def test_check_reproduce_refuses_to_write_when_stored_labels_differ(self):
        with tempfile.TemporaryDirectory() as directory:
            path = write_payload(directory, synthetic_payload(merge_labels=(7,)))
            signature = io.cache_signature(path)
            with self.assertRaises(ReproductionMismatch):
                apply_policy(request(directory, check_reproduce=True), log=lambda *_: None)
            self.assertEqual(io.cache_signature(path), signature)
            with open(path, "rb") as handle:
                payload = pickle.load(handle)
            np.testing.assert_array_equal(payload["gt_merge_labels"], [7])
            self.assertIn("__detector_sample_universe_cache__", payload)

    def test_unlabeled_cache_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            payload = synthetic_payload()
            payload["gt_node_canonical_label"] = None
            write_payload(directory, payload)
            with self.assertRaises(ValueError):
                apply_policy(request(directory), log=lambda *_: None)

    def test_run_reports_the_policy_stage(self):
        with tempfile.TemporaryDirectory() as directory:
            write_payload(directory, synthetic_payload())
            summary = build.run(request(directory), log=lambda *_: None)
            self.assertEqual(summary["stages"], ["policy"])
            self.assertEqual(summary["results"]["policy"]["action"], "relabeled")


class CliTests(unittest.TestCase):
    def test_dry_run_prints_the_plan_without_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            io.labeled_cache_path(directory, "111", 100).write_bytes(b"not a pickle")
            out = stringio.StringIO()
            with contextlib.redirect_stdout(out):
                code = cli.main(["--brains", "111", "222", "--cache-dir", directory, "--dry-run", "--no-verify-gt"])
            self.assertEqual(code, 0)
            self.assertIn("111 mcl100: policy", out.getvalue())
            self.assertIn("222 mcl100: base -> label", out.getvalue())

    def test_argument_validation(self):
        args = cli.parse_args(["--brains", "1", "--policy", "original", "--threshold-um", "40"])
        self.assertEqual((args.policy, args.threshold_um, args.mcl), ("original", 40.0, 100))
        for bad in (["--brains", "1", "--threshold-um", "0"], ["--brains", "1", "--gt-workers", "0"],
                    ["--brains", "1", "--policy", "bogus"]):
            with self.assertRaises(SystemExit), contextlib.redirect_stderr(stringio.StringIO()):
                cli.parse_args(bad)


if __name__ == "__main__":
    unittest.main()
