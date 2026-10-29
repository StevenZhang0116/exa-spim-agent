"""Tests of the labeled-cache builder (``labeled_cache``): policies, stage planning, the
in-place policy stage and the image-path refresh on synthetic caches. No brain is loaded and
nothing is read from GCS or S3."""

from __future__ import annotations

import contextlib
import io as stringio
import pickle
import random
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import networkx as nx
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from labeled_cache import __main__ as cli  # noqa: E402
from labeled_cache import build, io, policy, source_check  # noqa: E402
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


class ImagePathStageTests(unittest.TestCase):
    NEW = "s3://y/img/0"

    def caches(self, directory):
        labeled = write_payload(directory, synthetic_payload(extra=False))
        base = io.base_cache_path(directory, "111", 100)
        with open(base, "wb") as handle:
            pickle.dump({"img_path": "s3://x/img", "fragments_path": "gs://x/fragments"}, handle)
        return base, labeled

    def test_refresh_replaces_only_a_stale_img_path(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(io, "image_path", return_value=self.NEW):
            base, labeled = self.caches(directory)
            opened = []
            refresh = lambda: build.refresh_image_path(  # noqa: E731
                request(directory, refresh_image_path=True), log=lambda *_: None, open_image=opened.append)
            result = refresh()
            self.assertEqual({r["action"] for r in result.values()}, {"replaced"})
            self.assertEqual(result[labeled.name]["previous"], "s3://x/img")
            self.assertEqual(opened, [self.NEW])                  # opened once, before the first write
            expected = synthetic_payload(extra=False)
            with open(labeled, "rb") as handle:
                payload = pickle.load(handle)
            self.assertEqual(payload["img_path"], self.NEW)
            self.assertEqual(set(payload), set(expected))         # no other key added or dropped
            np.testing.assert_array_equal(payload["gt_node_canonical_label"], expected["gt_node_canonical_label"])
            self.assertEqual(payload["fragments_path"], expected["fragments_path"])
            with open(base, "rb") as handle:
                self.assertEqual(pickle.load(handle), {"img_path": self.NEW, "fragments_path": "gs://x/fragments"})
            # already current: nothing opened, nothing rewritten
            signatures = [io.cache_signature(p) for p in (base, labeled)]
            self.assertEqual({r["action"] for r in refresh().values()}, {"unchanged"})
            self.assertEqual(opened, [self.NEW])
            self.assertEqual([io.cache_signature(p) for p in (base, labeled)], signatures)

    def test_an_image_that_does_not_open_is_never_written(self):
        def missing(path):
            raise ValueError(f"NOT_FOUND: {path}")

        with tempfile.TemporaryDirectory() as directory, patch.object(io, "image_path", return_value=self.NEW):
            base, labeled = self.caches(directory)
            signatures = [io.cache_signature(p) for p in (base, labeled)]
            with self.assertRaises(ValueError):
                build.refresh_image_path(request(directory, refresh_image_path=True), log=lambda *_: None,
                                         open_image=missing)
            self.assertEqual([io.cache_signature(p) for p in (base, labeled)], signatures)
            with self.assertRaises(FileNotFoundError):
                build.refresh_image_path(request(directory, brain="222", refresh_image_path=True),
                                         log=lambda *_: None, open_image=missing)

    def test_plan_and_run_report_only_the_image_path_stage(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(io, "image_path", return_value=self.NEW), \
                patch.object(build, "_open_image"):
            self.caches(directory)
            self.assertEqual(plan(request(directory, refresh_image_path=True)), ["image_path"])
            summary = build.run(request(directory, refresh_image_path=True), log=lambda *_: None)
            self.assertEqual(summary["stages"], ["image_path"])
            self.assertEqual(len(summary["results"]["image_path"]), 2)


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
        self.assertTrue(cli.parse_args(["--brains", "1", "--refresh-image-path"]).refresh_image_path)
        for bad in (["--brains", "1", "--threshold-um", "0"], ["--brains", "1", "--gt-workers", "0"],
                    ["--brains", "1", "--policy", "bogus"],
                    ["--brains", "1", "--refresh-image-path", "--force-relabel"]):
            with self.assertRaises(SystemExit), contextlib.redirect_stderr(stringio.StringIO()):
                cli.parse_args(bad)


def _fragment(name, n, start=(100.0, 200.0, 300.0), step=(2.0, 0.0, 0.0)):
    """A parsed-SWC-like dict: a straight chain of ``n`` nodes in microns."""
    xyz = np.asarray(start, dtype=np.float64) + np.arange(n)[:, None] * np.asarray(step, dtype=np.float64)
    return {"swc_name": name, "xyz": xyz}


class SourceCheckTests(unittest.TestCase):
    """``labeled_cache.source_check``: sampling, tallying and reporting on synthetic data; no cloud."""

    aniso = (0.5, 0.5, 1.0)

    def test_fragment_id_accepts_swc_name_variants(self):
        self.assertEqual([source_check.fragment_id(n) for n in ("123", "123.swc", "123.1", "x7")], [123, 123, 123, None])

    def test_sampling_uses_interior_nodes_and_the_builder_voxel_convention(self):
        rng = random.Random(1)
        frag = _fragment("42", 10)
        samples = source_check.sample_fragment_nodes([frag, _fragment("7", 4), _fragment("bad", 10)], 3, 5, self.aniso, rng)
        self.assertEqual(len(samples), 3)
        for segment, voxel in samples:
            self.assertEqual(segment, 42)
            index = (voxel[2] * self.aniso[0] - 100.0) / 2.0  # x back to the chain index
            self.assertTrue(2 <= index <= 7, voxel)
            self.assertEqual(voxel, (300, 400, int((100.0 + 2.0 * index) / 0.5)))  # (z, y, x) = xyz / anisotropy reversed
        again = source_check.sample_fragment_nodes([frag, _fragment("7", 4), _fragment("bad", 10)], 3, 5, self.aniso, random.Random(1))
        self.assertEqual(samples, again)

    def test_tally_classifies_every_outcome(self):
        patches = {
            (1, 1, 1): np.full((3, 3, 3), 5),                   # exact
            (2, 2, 2): np.pad(np.full((1, 1, 1), 5), 1),        # 5 only at the centre -> exact
            (3, 3, 3): np.where(np.indices((3, 3, 3)).sum(0) == 0, 5, 9),  # 5 at a corner, 9 at the centre -> within 1 vox
            (4, 4, 4): np.zeros((3, 3, 3), dtype=int),          # background
            (5, 5, 5): np.full((3, 3, 3), 8),                   # other id
        }

        def read(voxel):
            if voxel == (6, 6, 6):
                raise IndexError("outside the volume")
            return patches[voxel]

        samples = [(5, v) for v in sorted(patches)] + [(5, (6, 6, 6))]
        counts = source_check.tally(samples, read)
        self.assertEqual((counts["nodes"], counts["exact_match"], counts["match_within_1vox"], counts["background"],
                          counts["other_id"], counts["read_failed"]), (6, 2, 1, 1, 1, 1))
        self.assertEqual(counts["agreement_pct"], 60.0)
        self.assertEqual(counts["top_other_ids"], [(8, 1)])

    def test_verdicts_and_table(self):
        good = {"segmentation_id": "seg-a", "fragments": 2, "nodes": 200, "exact_match": 190, "match_within_1vox": 8,
                "background": 2, "other_id": 0, "read_failed": 0, "agreement_pct": 99.0}
        bad = dict(good, agreement_pct=3.0, exact_match=6, match_within_1vox=0, other_id=194)
        few = dict(good, nodes=20, read_failed=0)
        self.assertEqual(source_check.verdict(good, 95, 100), "matched")
        self.assertEqual(source_check.verdict(bad, 95, 100), "MISMATCH")
        self.assertEqual(source_check.verdict(few, 95, 100), "insufficient")
        self.assertEqual(source_check.verdict({"error": "x"}, 95, 100), "error")
        reports = {"1": dict(good, verdict="matched"), "2": {"error": "boom"}}
        table = source_check.markdown_table(reports)
        self.assertIn("| 1 | `seg-a` | 2 | 200 | 190 | 8 | 2 | 0 | 0 | 99.0% | matched |", table)
        self.assertIn("| 2 |", table)
        self.assertIn("error: boom", table)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "receipts.md"
            source_check.record(reports, path, "cmd one")
            source_check.record(reports, path, "cmd two")
            text = path.read_text()
            self.assertEqual(text.count("# Fragment / segmentation source agreement"), 1)
            self.assertEqual(text.count("`cmd one`"), 1)
            self.assertEqual(text.count("`cmd two`"), 1)
            self.assertEqual(text.count("| 1 | `seg-a` |"), 2)

    def test_cli_exit_code_follows_the_verdicts(self):
        def fake_check(brain, **kwargs):
            if brain == "9":
                return {"segmentation_id": "s", "error": "no such brain", "verdict": "error"}
            return {"segmentation_id": "s", "fragments_layout": "l", "fragments": 1, "nodes": 120, "exact_match": 118,
                    "match_within_1vox": 2, "background": 0, "other_id": 0, "read_failed": 0, "agreement_pct": 100.0,
                    "seconds": 0.1, "verdict": "matched"}

        with patch.object(source_check, "check_brain", side_effect=fake_check):
            out = stringio.StringIO()
            with contextlib.redirect_stdout(out):
                self.assertEqual(source_check.main(["--brains", "1", "1"]), 0)
            self.assertEqual(out.getvalue().count("1: matched"), 1)
            with contextlib.redirect_stdout(stringio.StringIO()):
                self.assertEqual(source_check.main(["--brains", "1", "9"]), 1)
        for bad in (["--brains", "1", "--fragments", "0"], ["--brains", "1", "--min-agreement", "120"],
                    ["--brains", "1", "--anisotropy", "0", "1", "1"],
                    ["--brains", "1", "--fragments", "10", "--nodes-per-fragment", "4"]):
            with self.assertRaises(SystemExit), contextlib.redirect_stderr(stringio.StringIO()):
                source_check.parse_args(bad)


if __name__ == "__main__":
    unittest.main()
