import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scripts import audit_merge_junction_gt as audit_script
from agentic.tests.test_plot_top_merge_predictions import FakeGraph, prediction, write_csv


class JunctionAuditCommandTests(unittest.TestCase):
    def test_preserves_original_label_and_segment_evidence(self):
        candidate = prediction(score_rank=7)
        payload = {"fragments_graph": FakeGraph(), "gt_graph": object(),
                   "gt_node_canonical_label": np.array([111]), "gt_merge_labels": np.array([111]),
                   "gt_merge_sites": [{"segment_id": 111, "xyz": [1010., 12., 4.]}]}
        evidence = {"audit_only": True, "schema_version": 1, "junctions": [
            {"node_id": 0, "xyz": [10., 12., 4.], "status": "merge_supported"}]}
        with patch("agentic_neuron_proofreader.data_modules.canonical_labeling.audit_junction_gt_connections",
                   return_value=evidence):
            report = audit_script.audit_predictions(payload, [candidate])
        record = report["junctions"][0]
        self.assertEqual(record["original_is_merge_site"], 0)
        self.assertTrue(record["segment_gt_is_merge"])
        self.assertIsNone(record["original_gt_site_distance_um"])
        self.assertEqual(record["nearest_recorded_segment_site_euclidean_um"], 1000.)
        self.assertEqual(record["score_rank"], 7)
        self.assertNotIn("candidate_id", evidence["junctions"][0])
        json.dumps(report, allow_nan=False)

    def test_refuses_overwrite_and_missing_candidates_before_loading_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "report.json"
            output.write_text("preserve me")
            argv = ["--csv", str(root / "scores.csv"), "--pkl", str(root / "missing.pkl"),
                    "--output", str(output), "--candidate-id", "7"]
            with patch("sys.stderr"), self.assertRaises(SystemExit):
                audit_script.main(argv)
            self.assertEqual(output.read_text(), "preserve me")
            output.unlink()
            row = prediction(candidate_id=8)
            row[audit_script.DEFAULT_SCORE] = row.pop("score")
            write_csv(root / "scores.csv", [row])
            with patch("sys.stderr"), self.assertRaises(SystemExit):
                audit_script.main(argv)
            self.assertFalse(output.exists())

    def test_validates_cache_identity_and_requires_canonical_labels(self):
        payload = {"fragments_graph": FakeGraph(), "gt_graph": object()}
        with self.assertRaisesRegex(ValueError, "canonical GT"):
            audit_script.audit_predictions(payload, [prediction()])
        payload["gt_node_canonical_label"] = np.array([111])
        with self.assertRaisesRegex(ValueError, "belongs to segment"):
            audit_script.audit_predictions(payload, [prediction(segment_id=222)])


if __name__ == "__main__":
    unittest.main()