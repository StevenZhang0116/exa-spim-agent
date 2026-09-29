"""Remaining preparation contracts and rejection of removed CLI interfaces."""

from contextlib import redirect_stderr, redirect_stdout
import io
from pathlib import Path
import pickle
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pandas as pd

from proofreader_evolve.cli import precompute_error_scores as pc, preflight
from proofreader_evolve.harness import dataset
from proofreader_evolve.harness.detector_metadata import inference_metadata


class PreparationTests(unittest.TestCase):
    def test_model_group_selection_preserves_shared_groups_and_definedness(self):
        module = SimpleNamespace(ANALYSIS_TIMING_GROUPS=[
            {"key": "shared", "feature_names": ["a", "b"]},
            {"key": "unused", "feature_names": ["costly"]},
            {"key": "defined", "feature_names": ["c"]},
        ])
        self.assertEqual(pc._model_analysis_keys(module, ["b", "c_is_defined"]),
                         {"shared", "defined"})

    def test_invalid_model_group_mapping_fails_before_extraction(self):
        for groups, order, message in (
                (None, ["a"], "lacks ANALYSIS_TIMING_GROUPS"),
                ([{"key": "a", "feature_names": ["a"]}], ["missing"], "no analysis group"),
                ([{"key": "a", "feature_names": ["a"]}], ["a", "a"], "duplicate features"),
                ([{"key": "a", "feature_names": ["a"]},
                  {"key": "b", "feature_names": ["a"]}], ["a"], "Ambiguous"),
                ([{"key": "a", "feature_names": ["a"]},
                  {"key": "a", "feature_names": ["b"]}], ["a"], "Duplicate analysis")):
            with self.subTest(message=message):
                module = SimpleNamespace(ANALYSIS_TIMING_GROUPS=groups)
                with patch.object(module, "_extract_features_runtime", create=True) as extract:
                    with self.assertRaisesRegex(ValueError, message):
                        pc._extract_candidate_features({}, module, [], order)
                extract.assert_not_called()

    def test_model_group_selection_reaches_runtime_and_direct_extractors(self):
        for runtime in (True, False):
            with self.subTest(runtime=runtime):
                rows = [{"candidate_id": "one"}]
                seen = []

                def extract(working, verbose=True, enabled_analysis_keys=None):
                    self.assertEqual(enabled_analysis_keys, {"needed"})
                    self.assertNotIn("gt_graph", working)
                    seen.append(enabled_analysis_keys)
                    frame = pd.DataFrame({"a": [1.]})
                    return rows, [0], SimpleNamespace(to_frame=lambda: frame)

                module = SimpleNamespace(
                    ANALYSIS_TIMING_GROUPS=[
                        {"key": "needed", "feature_names": ["a"]},
                        {"key": "costly", "feature_names": ["b"]}],
                    sample_display=lambda row: row["candidate_id"])
                setattr(module, "_extract_features_runtime" if runtime else "extract_features", extract)
                with redirect_stdout(io.StringIO()):
                    actual, _ = pc._extract_candidate_features({"gt_graph": object()}, module, rows, ["a"])
                self.assertEqual(actual, rows)
                self.assertEqual(len(seen), 1)

    def test_removed_preparation_flags_rejected_before_loading(self):
        for main, prefix in (
                (pc.main, ["--brain", "1", "--pkl", "unused"]),
                (preflight.main, ["--brains", "1"])):
            with patch.object(pc, "resolve_detector_runs") as resolve:
                with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                    main(prefix + ["--candidate-mode", "legacy"])
                self.assertEqual(error.exception.code, 2)
                resolve.assert_not_called()
        with patch.object(pc, "resolve_detector_runs") as resolve:
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                pc.main(["--brain", "1", "--pkl", "unused", "--max-gap-um", "40"])
            resolve.assert_not_called()

    def test_dataset_exposes_loading_not_repair_enumeration(self):
        self.assertTrue(dataset.default_cache_path("1").endswith("dataset_cache_1_mcl100_add.pkl"))
        for name in ("candidate_split_sites", "candidate_merge_sites", "resolve_enum_params",
                     "train_heldout_split", "SplitSite", "MergeSite"):
            self.assertFalse(hasattr(dataset, name))
        self.assertFalse(hasattr(pc, "ensure_feature_banks"))

    def test_cache_loading_preserves_content_checks(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "dataset_cache_1_mcl100_add.pkl"
            payload = {"fragments_graph": "fragments", "gt_graph": "gt",
                       "min_cable_length": 100, "fragments_path": "/data/1/fragments"}
            source.write_bytes(pickle.dumps(payload))
            self.assertEqual(dataset.load_cached_graphs(source, "1", 100), ("fragments", "gt", payload))
            for brain, mcl in (("2", 100), ("1", 200)):
                with self.assertRaises(SystemExit):
                    dataset.load_cached_graphs(source, brain, mcl)

    def test_metadata_whitelist_and_invalid_mcl(self):
        self.assertEqual(inference_metadata({"min_cable_length": 100, "gt_graph": object()}, 100),
                         {"min_cable_length": 100})
        self.assertEqual(inference_metadata({}, 100), {"min_cable_length": 100})
        for payload, expected in (({}, None), ({"min_cable_length": 100}, 200),
                                  ({}, True), ({}, -1), ({}, 1.5)):
            with self.subTest(expected=expected), self.assertRaises(ValueError):
                inference_metadata(payload, expected)

    def test_exactly_one_detector_per_kind(self):
        for directories in ([], ["a", "b"]):
            with patch.object(pc, "_resolve_detector_dir") as resolve:
                with self.assertRaisesRegex(ValueError, "Exactly one"):
                    pc.resolve_detector_runs(directories, ["split"])
                resolve.assert_not_called()

    def test_detector_resolution_and_artifact_hash_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            merge, split = root / "merge", root / "split"
            merge.mkdir()
            split.mkdir()
            (merge / "merge_junction_detector.py").write_text("# trusted fixture")
            (merge / "merge_junction_detector_1.joblib").write_bytes(b"fixture")
            (split / "split_site_detector.py").write_text("# trusted fixture")
            (split / "split_detector_1.joblib").write_bytes(b"fixture")
            selected = pc.resolve_detector_runs([merge], [split])
            record = selected["merge"][0]
            pc._assert_detector_unchanged(record)
            (merge / "merge_junction_detector.py").write_text("# changed")
            with self.assertRaisesRegex(RuntimeError, "changed"):
                pc._assert_detector_unchanged(record)
            (merge / "merge_junction_detector_2.joblib").write_bytes(b"another")
            with self.assertRaises(SystemExit):
                pc._resolve_detector_dir(merge, "merge")

    def test_removed_modules_have_no_source_files(self):
        root = Path(__file__).resolve().parents[1]
        for path in ("cli/run_structural_evolution.py", "harness/candidate.py",
                     "harness/incremental_scoring.py", "harness/edit_handler.py",
                     "artifacts/heuristics.py", "harness/feature_bank.py",
                     "harness/detector_queries.py"):
            self.assertFalse((root / path).exists(), path)


if __name__ == "__main__":
    unittest.main()
