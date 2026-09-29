import ast
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

from agentic.tests.test_merge_site_target import _runtime_namespace, _merge_site_payload


TEMPLATE = Path(__file__).resolve().parents[1] / "detector_build/templates/detector_runtime.py.tmpl"


def runtime():
    namespace = _runtime_namespace()
    namespace.update(os=os, json=json, hashlib=hashlib, __file__=str(TEMPLATE))
    names = {"_cache_file_identity", "_input_provenance", "run_detector", "_run_heldout",
             "_apply_scope", "_drop_all_undefined", "_json_default"}
    tree = ast.parse(TEMPLATE.read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(TEMPLATE), "exec"), namespace)
    return namespace


class DetectorLabelProvenanceTests(unittest.TestCase):
    def test_filtered_scope_digest_covers_only_retained_rows(self):
        namespace = runtime()
        samples = [{"node_id": index, "segment_id": 111, "degree": 3} for index in (1, 2)]
        frame = pd.DataFrame({"x": [0., np.nan], "x_is_defined": [True, False]})
        _, retained, labels = namespace["_apply_scope"](frame, samples, np.array([0, 1]), True, lambda _: None)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.pkl"
            path.write_bytes(b"cache")
            identity = namespace["_cache_file_identity"](path)
            evidence = namespace["_input_provenance"](path, identity, retained, labels,
                                                     {"n_rows": 2}, "excludeempty")
            self.assertEqual(evidence["scope"], "excludeempty")
            self.assertEqual(evidence["n_rows"], 1)
            self.assertEqual(evidence["n_positive"], 0)
            self.assertEqual(evidence["sample_universe_audit"]["n_rows"], 2)

    def test_hash_records_actual_ordered_labels_and_detects_replacement(self):
        namespace = runtime()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.pkl"
            path.write_bytes(b"cache")
            identity = namespace["_cache_file_identity"](path)
            samples = [{"node_id": 1, "segment_id": 111, "degree": 3}]
            audit = {"site_provenance": {"n_sites": 5}}
            first = namespace["_input_provenance"](path, identity, samples, [0], audit, "full")
            second = namespace["_input_provenance"](path, identity, samples, [1], audit, "full")
            self.assertNotEqual(first["ordered_row_labels_sha256"], second["ordered_row_labels_sha256"])
            audit["site_provenance"]["n_sites"] = 99
            self.assertEqual(first["sample_universe_audit"]["site_provenance"]["n_sites"], 5)
            replacement = path.with_suffix(".tmp")
            replacement.write_bytes(b"cache")
            replacement.replace(path)
            with self.assertRaisesRegex(RuntimeError, "changed during"):
                namespace["_input_provenance"](path, identity, samples, [1], audit, "full")

    def test_model_csv_and_heldout_share_provenance_without_real_training(self):
        namespace = runtime()
        train_payload = _merge_site_payload()
        train_payload["gt_merge_sites"].append({"segment_id": 222, "xyz": [105., 0., 0.],
                                                "source": "two_gt_junction"})
        train_payload["gt_merge_site_metadata"] = {"method": "two-gt", "parameters": {"radius": 6}}
        test_payload = _merge_site_payload()
        seen_labels = []

        def extract(blind, **kwargs):
            rows, labels = namespace["build_sample_universe"](blind)
            self.assertFalse(np.any(labels))
            for key in ("gt_merge_sites", "gt_junction_audit", "gt_merge_site_metadata"):
                with self.assertRaises(namespace["BlindnessViolation"]):
                    blind[key]
            accumulator = namespace["FeatureAccumulator"](rows)
            for row, value in zip(rows, (0., 1.)):
                accumulator.set_candidate("x", row["candidate_id"], value)
            return rows, labels, accumulator

        def nested(features, labels, *args, **kwargs):
            seen_labels.append(labels.tolist())
            return {"oof": {"test": np.array([.2, .8])}, "oof_selector": np.array([.2, .8]),
                    "family_metrics": {}, "selector_metrics": {"oof_average_precision": 1.,
                        "oof_roc_auc": None, "n_scored": 2}, "selection_frequency": {"test": 1}, "per_fold": []}

        policy = {"selection": {"outer_folds": 2, "inner_folds": 2, "simplicity_order": ["test"]}}
        namespace.update(
            extract_features=extract, _inventory_sha=lambda path: "inventory-sha",
            load_and_validate_config=lambda *args: {}, _sha256_file=lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            assemble_configured=lambda *args: ({"test": {"grid": {}}}, {}), EMBEDDED_MODEL_POLICY=policy,
            EMBEDDED_MODEL_POLICY_SHA256="policy-sha", RANDOM_SEED=0,
            load_hypothesis_selection=lambda *args, **kwargs: {
                "source_timing_feature_inventory_sha256": None, "enabled_analysis_keys": ["hypo"],
                "excluded_feature_names": [], "excluded_hypothesis_ids": []},
            ANALYSIS_TIMING_GROUPS=[], FEATURE_NAMES=["x"],
            brain_id_from_path=lambda path: "794495" if "train" in path else "794493",
            mcl_from_payload=lambda *args: 100, load_payload=lambda path: train_payload if "train" in path else test_payload,
            ascertainment_covariates=lambda *args: {}, gc=SimpleNamespace(collect=lambda: None),
            undefined_audit=lambda *args: {}, run_nested_selection=nested,
            fit_final_winner=lambda *args, **kwargs: ("test", {}, "fitted-placeholder", {}),
            _score_matrix=lambda estimator, features: np.array([.2, .8]),
            _report_thresholds=lambda *args, **kwargs: None, _print_queue=lambda *args, **kwargs: None,
            ascertainment_audit=lambda *args: {}, _collect_versions=lambda: {},
            _fmt=lambda value: str(value),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("train.pkl", "test.pkl", "config.json"):
                (root / name).write_text("placeholder")
            args = SimpleNamespace(inventory="unused", model_config=str(root / "config.json"),
                hypothesis_selection=None, exclude_hypotheses=None, pkl=str(root / "train.pkl"),
                test_pkl=str(root / "test.pkl"), exclude_empty=False, n_jobs=1, review_budget=1,
                out_csv=str(root / "scores.csv"), out_dir=str(root), no_figures=True)
            with patch("builtins.print"):
                self.assertEqual(namespace["run_detector"](args, lambda message: None), 0)
            model = json.loads((root / "model_selection_794495.json").read_text())
            saved = joblib.load(root / "merge_junction_detector_794495.joblib")
            sidecar = json.loads((root / "scores.csv.provenance.json").read_text())
            self.assertEqual(saved["data_provenance"], model["data_provenance"])
            self.assertEqual(sidecar["data_provenance"], model["data_provenance"])
            self.assertEqual(sidecar["csv_sha256"], hashlib.sha256((root / "scores.csv").read_bytes()).hexdigest())
            training = model["data_provenance"]["training"]
            heldout = model["data_provenance"]["heldout"]
            self.assertEqual(seen_labels, [[1, 1]])
            self.assertEqual(training["n_positive"], 2)
            self.assertEqual(heldout["n_positive"], 1)
            self.assertEqual(training["sample_universe_audit"]["site_provenance"]["source_counts"]["two_gt_only"], 1)
            self.assertEqual(training["sample_universe_audit"]["site_provenance"]["generation_metadata"]["method"], "two-gt")
            self.assertNotEqual(training["ordered_row_labels_sha256"], heldout["ordered_row_labels_sha256"])
            self.assertEqual(pd.read_csv(root / "scores.csv")["is_merge_site"].tolist(), [1, 1])
            self.assertEqual(model["heldout"]["data_provenance"], heldout)


if __name__ == "__main__":
    unittest.main()
