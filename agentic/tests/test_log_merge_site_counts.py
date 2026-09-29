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

from agentic.tests.test_merge_site_target import _merge_site_payload
from notebooks import log_merge_site_counts as logger


class MergeSiteCountsTests(unittest.TestCase):
    def test_counts_use_shared_runtime_and_ignore_stale_universe(self):
        payload = _merge_site_payload()
        stale = ([], [])
        payload["__detector_sample_universe_cache__"] = stale
        payload["__detector_universe_site_audit__"] = {"n_sites": 999}
        runtime, _ = logger.make_runtime(10., 4., 15.)
        summary, detail = logger.summarize_payload(payload, 100, runtime)
        self.assertEqual(summary["n_gt_merge_sites"], 4)
        self.assertEqual(summary["n_candidates"], 2)
        self.assertEqual(summary["n_is_merge_site_1"], 1)
        self.assertEqual(summary["n_in_ambiguous_ring_1"], 1)
        self.assertEqual(summary["n_gt_sites_covered_at_positive_radius"], 1)
        self.assertEqual(summary["n_gt_sites_only_within_claim_radius"], 2)
        self.assertEqual(summary["n_gt_sites_not_covered_at_claim_radius"], 1)
        self.assertEqual(summary["n_gt_sites_unsnappable"], 1)
        self.assertEqual([site["distance_to_nearest_candidate_um"] for site in detail["site_distances"]],
                         [0., 10., 5., None])
        self.assertIs(payload["__detector_sample_universe_cache__"], stale)
        self.assertEqual(payload["__detector_universe_site_audit__"], {"n_sites": 999})

    def test_gt_records_and_positive_candidates_are_not_interchangeable(self):
        payload = _merge_site_payload()
        payload["gt_merge_sites"] = [payload["gt_merge_sites"][0]]
        runtime, _ = logger.make_runtime(0., 5., 15.)
        summary, _ = logger.summarize_payload(payload, 100, runtime)
        self.assertEqual(summary["n_gt_merge_sites"], 1)
        self.assertEqual(summary["n_is_merge_site_1"], 2)
        self.assertEqual(summary["n_gt_sites_covered_at_positive_radius"], 1)
        payload["gt_merge_sites"] *= 2
        summary, _ = logger.summarize_payload(payload, 100, runtime)
        self.assertEqual(summary["n_gt_merge_sites"], 2)
        self.assertEqual(summary["n_is_merge_site_1"], 2)
        self.assertEqual(summary["n_gt_sites_covered_at_positive_radius"], 2)

    def test_missing_empty_and_mcl_mismatch(self):
        runtime, _ = logger.make_runtime()
        payload = _merge_site_payload()
        payload["gt_merge_sites"] = None
        with self.assertRaisesRegex(ValueError, "missing labels"):
            logger.summarize_payload(payload, 100, runtime)
        del payload["gt_merge_sites"]
        with self.assertRaisesRegex(ValueError, "missing labels"):
            logger.summarize_payload(payload, 100, runtime)
        payload["gt_merge_sites"] = []
        summary, _ = logger.summarize_payload(payload, 100, runtime)
        self.assertEqual(summary["n_gt_merge_sites"], 0)
        self.assertEqual(summary["n_is_merge_site_1"], 0)
        self.assertIsNone(summary["gt_site_recall_at_positive_radius"])
        payload["min_cable_length"] = 10
        with self.assertRaisesRegex(ValueError, "MCL"):
            logger.summarize_payload(payload, 100, runtime)

    def test_legacy_gt_sites_and_primary_field_precedence(self):
        payload = _merge_site_payload()
        payload["gt_graph"] = SimpleNamespace(merge_sites=payload.pop("gt_merge_sites"))
        runtime, _ = logger.make_runtime()
        summary, detail = logger.summarize_payload(payload, 100, runtime)
        self.assertEqual(summary["n_gt_merge_sites"], 4)
        self.assertEqual(detail["gt_sites_storage"], "gt_graph.merge_sites")
        payload["gt_merge_sites"] = []
        summary, _ = logger.summarize_payload(payload, 100, runtime)
        self.assertEqual(summary["n_gt_merge_sites"], 0)
        payload["gt_merge_sites"] = None
        with self.assertRaisesRegex(ValueError, "missing labels"):
            logger.summarize_payload(payload, 100, runtime)

    def test_changed_cache_rejected(self):
        runtime, _ = logger.make_runtime()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dataset_cache_1_mcl100_add.pkl"
            path.write_bytes(pickle.dumps(_merge_site_payload()))
            identity = logger.cache_identity(path)
            changed = {**identity, "mtime_ns": identity["mtime_ns"] + 1}
            with patch.object(logger, "cache_identity", side_effect=[identity, changed]), \
                    contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "changed"):
                logger.inspect_cache(path, 100, runtime)

    def test_invalid_policy(self):
        for settings in ((-1., 20., 150.), (0., 0., 150.), (20., 160., 150.),
                         (float("nan"), 20., 150.), (20., 20., float("inf"))):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                logger.make_runtime(*settings)

    def test_cli_all_datasets_one_mcl_read_only_and_partial_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for brain, mcl in (("1", 100), ("2", 100), ("4", 100), ("1", 10)):
                payload = _merge_site_payload()
                payload["min_cable_length"] = mcl
                (root / f"dataset_cache_{brain}_mcl{mcl}_add.pkl").write_bytes(pickle.dumps(payload))
            (root / "dataset_cache_3_mcl100_add.pkl").write_bytes(pickle.dumps({}))
            before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in root.glob("*.pkl")}
            output = root / "counts"
            argv = ["--mcl", "100", "--cache-dir", str(root), "--output-dir", str(output),
                    "--nms-um", "10", "--positive-radius-um", "4", "--claim-radius-um", "15"]
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(logger.main(argv), 1)
            with (output / "summary.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([row["brain_id"] for row in rows], ["1", "2", "3", "4"])
            self.assertEqual([row["status"] for row in rows], ["ok", "ok", "error", "ok"])
            self.assertEqual(rows[0]["n_gt_merge_sites"], "4")
            self.assertEqual(rows[0]["n_is_merge_site_1"], "1")
            self.assertEqual(rows[2]["n_gt_merge_sites"], "")
            self.assertEqual(len(list((output / "per_brain").glob("*.json"))), 3)
            run = json.loads((output / "run.json").read_text())
            self.assertEqual(run["policy"]["settings"]["positive_label_radius_um"], 4.)
            for path, digest in before.items():
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                logger.main(argv)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(logger.main(["--mcl", "100", "--cache-dir", str(root),
                                              "--output-dir", str(root / "selected"), "--brains", "2"]), 0)
            with (root / "selected" / "summary.csv").open() as handle:
                self.assertEqual([row["brain_id"] for row in csv.DictReader(handle)], ["2"])

    def test_invalid_selection_fails_without_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "counts"
            for extra in ([], ["--brains", "1"], ["--mcl", "-1"], ["--brains", "../brain"]):
                with self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()), \
                        self.assertRaises(SystemExit):
                    logger.main(["--mcl", "100", "--cache-dir", directory,
                                 "--output-dir", str(output), *extra])
                self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()