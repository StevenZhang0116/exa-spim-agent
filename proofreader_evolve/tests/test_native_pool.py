"""Native tables on tiny synthetic graphs with stand-in detectors; no real-brain or API calls.

The 2026-08 frozen detectors were deleted on 2026-10-17. The rows never come from a detector
(candidate_rules.py), so these tests use minimal stand-in detector folders; the one parity test
against real frozen detectors runs only when default detectors are configured.
"""

from contextlib import redirect_stdout
from copy import deepcopy
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import networkx as nx
import numpy as np
import pandas as pd

from agentic.tests.test_merge_site_target import _merge_site_payload
from proofreader_evolve.cli import precompute_error_scores as pc
from proofreader_evolve.harness import candidate_pools as cp
from proofreader_evolve.harness import native_pool as native


STAND_IN_DETECTOR = """
DETECTOR_TARGET = {target!r}
CANDIDATE_POLICY_CONFIG_ID = {config!r}
CANDIDATE_POLICY_SHA256 = "0" * 64
ANALYSIS_TIMING_GROUPS = [{{"key": "g0", "feature_names": ["f0"]}}]


def sample_display(row):
    return repr(row)


def _extract_features_runtime(payload, verbose=True, enabled_analysis_keys=None):
    raise NotImplementedError("stand-in detector: patch this in the test")


def build_sample_universe(payload):
    raise NotImplementedError("stand-in detector: rows come from candidate_rules.py")
"""


def stand_in_detectors(root):
    """Two minimal detector folders that ``resolve_detector_runs`` accepts (brain 1, mcl 100)."""
    import joblib
    import json
    directories = {}
    for kind, script, model, target, config in (
            ("merge", "merge_junction_detector.py", "merge_junction_detector_1.joblib",
             "merge_site_detection", "junction|nms=20|r=150"),
            ("split", "split_site_detector.py", "split_detector_1.joblib",
             "split_detection", "tip_to_any_node|r=50|k=2")):
        directory = Path(root) / f"{kind}-stand-in"
        directory.mkdir()
        (directory / script).write_text(STAND_IN_DETECTOR.format(target=target, config=config))
        joblib.dump({"pipeline": None, "feature_order": ["f0"]}, directory / model)
        (directory / "model_selection_1.json").write_text(json.dumps(
            {"train_brain": "1", "train_mcl": 100, "feature_order": ["f0"]}))
        directories[kind] = directory
    return pc.resolve_detector_runs([directories["merge"]], [directories["split"]])


def payload():
    data = _merge_site_payload()
    graph = data["fragments_graph"]
    # Bring the second component inside the native split search radius.
    graph.node_xyz[7:, 0] -= 80
    graph.node_radius = np.ones(len(graph))
    graph.cable_length = lambda root: sum(
        float(np.linalg.norm(graph.node_xyz[a] - graph.node_xyz[b]))
        for a, b in graph.subgraph(nx.node_connected_component(graph, root)).edges())
    data.update(gt_graph=nx.path_graph(3), gt_edge_error=np.array([2, 2]),
                gt_node_canonical_label=np.array([111, 0, 222]), min_cable_length=100)
    return data


class NativePoolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._detectors = tempfile.TemporaryDirectory()
        cls.selected = stand_in_detectors(cls._detectors.name)
        cls.modules = {kind: pc._import_module(Path(records[0]["dir"]) / records[0]["script"],
                                               "test_native_" + kind)
                       for kind, records in cls.selected.items()}

    @classmethod
    def tearDownClass(cls):
        cls._detectors.cleanup()

    def test_exact_declared_rows_occurrences_and_labels_with_no_gt_in_extraction(self):
        for kind, module in self.modules.items():
            with self.subTest(kind=kind):
                data = payload()
                expected, labels = cp.build_universe(kind, dict(data), cp.load_pool_policies()[kind])
                safe = native.blind_rows(kind, expected)
                seen = []
                group = module.ANALYSIS_TIMING_GROUPS[0]
                feature = group["feature_names"][0]

                def extract(working, verbose=True, enabled_analysis_keys=None):
                    self.assertEqual(enabled_analysis_keys, {group["key"]})
                    self.assertFalse(any(k.startswith("gt_") for k in working))
                    rows, synthetic_labels = working["__detector_sample_universe_cache__"]
                    self.assertTrue((synthetic_labels == 0).all())
                    self.assertEqual(rows, safe)
                    self.assertFalse(any("gt" in key or key.startswith("is_") for r in rows for key in r))
                    seen.extend(rows)
                    frame = pd.DataFrame({feature: [len(r.get("occurrences", ())) for r in rows]})
                    return rows, synthetic_labels, SimpleNamespace(to_frame=lambda: frame)

                model = SimpleNamespace(predict_proba=lambda x: np.tile([.2, .8], (len(x), 1)))
                with patch.object(module, "_extract_features_runtime", side_effect=extract), redirect_stdout(io.StringIO()):
                    table = native.build_native_table(data, kind, module,
                                                      {"pipeline": model, "feature_order": [feature]})
                self.assertEqual(table.candidates, safe)
                np.testing.assert_array_equal(table.truth, labels)
                self.assertEqual(len(table.features), len(safe))
                self.assertTrue(seen)
                if kind == "split":
                    self.assertGreater(len(table.candidates[0]["occurrences"]), 1)
                else:
                    self.assertEqual([r["node_id"] for r in safe], [2, 8])

    def test_rows_follow_the_given_rule_never_the_detector_builder(self):
        module = self.modules["split"]
        wide = {"config_id": "tip_to_any_node|r=200|k=4", "mode": "tip_to_any_node", "radius_um": 200.0,
                "per_anchor_k": 4, "global_cap": None}
        group = module.ANALYSIS_TIMING_GROUPS[0]
        feature = group["feature_names"][0]

        def extract(working, verbose=True, enabled_analysis_keys=None):
            rows, synthetic_labels = working["__detector_sample_universe_cache__"]
            frame = pd.DataFrame({feature: [1.0] * len(rows)})
            return rows, synthetic_labels, SimpleNamespace(to_frame=lambda: frame)

        def forbidden(data):
            raise AssertionError("the detector's own universe builder must not enumerate the pool")

        model = SimpleNamespace(predict_proba=lambda x: np.tile([.2, .8], (len(x), 1)))
        bundle = {"pipeline": model, "feature_order": [feature]}
        tables = {}
        with patch.object(module, "_extract_features_runtime", side_effect=extract), \
                patch.object(module, "build_sample_universe", side_effect=forbidden), redirect_stdout(io.StringIO()):
            tables["declared"] = native.build_native_table(payload(), "split", module, bundle)
            tables["wide"] = native.build_native_table(payload(), "split", module, bundle, pool_policy=wide)
        declared = cp.load_pool_policies()["split"]
        self.assertEqual(tables["declared"].meta["candidate_policy"], declared["config_id"])
        self.assertEqual(tables["declared"].meta["candidate_rule"], declared)
        self.assertEqual(tables["wide"].meta["candidate_policy"], "tip_to_any_node|r=200|k=4")
        self.assertEqual(tables["wide"].meta["candidate_policy_sha256"], cp.policy_digest("split", wide))
        for table in tables.values():
            self.assertEqual(table.meta["candidate_policy_source"], "candidate_pools.json")
            self.assertEqual(table.meta["candidate_rules_sha256"], cp.rules_sha256())
            self.assertEqual(table.meta["detector_candidate_policy"], module.CANDIDATE_POLICY_CONFIG_ID)
            self.assertEqual(table.meta["detector_candidate_policy_sha256"], module.CANDIDATE_POLICY_SHA256)
            self.assertIsNone(table.meta["detector_candidate_rules_sha256"])
            self.assertFalse(table.meta["detector_rule_matches"])
            self.assertGreater(table.meta["rows"], 0)

    def test_table_provenance_carries_the_declared_pool_policy(self):
        from proofreader_evolve.harness import candidate_pools as cp
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "dataset_cache_1_mcl100_add.pkl"
            source.write_bytes(b"synthetic labeled cache placeholder")
            seen = {}

            def build(data, kind, module, bundle, pool_policy=None, image_workers=0):
                seen[kind] = pool_policy
                self.assertEqual(image_workers, 0)
                rows = [{"segment_id": 1, "node_id": 0}] if kind == "merge" else [{"segment_id_a": 1, "segment_id_b": 2}]
                return native.NativeTable(kind, pd.DataFrame({"detector_score": [.5]}), rows, np.array([1]),
                                          {"pool_sha256": native.pool_digest(kind, rows), "rows": 1})
            with patch.object(pc.ds, "load_cached_graphs", return_value=(None, None, payload())), \
                    patch.object(native, "build_native_table", side_effect=build), redirect_stdout(io.StringIO()):
                tables = native.ensure_native_tables("1", source, self.selected, root / "tables", True, 100)
            declared = cp.load_pool_policies()
            self.assertEqual(seen, declared)
            for kind in ("merge", "split"):
                self.assertEqual(tables.tables[kind].meta["provenance"]["pool_policy"], declared[kind])
                self.assertEqual(tables.tables[kind].meta["provenance"]["version"], "detector-native-v3")
                self.assertIn("candidate_rules.py", tables.tables[kind].meta["provenance"]["implementation"])

    def test_universe_and_table_share_one_enumeration(self):
        from proofreader_evolve.harness import candidate_pools as cp
        policy = cp.load_pool_policies()["split"]
        for kind, pool_policy in (("merge", None), ("split", policy)):
            with self.subTest(kind=kind):
                module = self.modules[kind]
                rows, truth, meta = native.build_native_universe(payload(), kind, module, pool_policy)
                group = module.ANALYSIS_TIMING_GROUPS[0]
                feature = group["feature_names"][0]

                def extract(working, verbose=True, enabled_analysis_keys=None):
                    found, synthetic_labels = working["__detector_sample_universe_cache__"]
                    frame = pd.DataFrame({feature: [1.0] * len(found)})
                    return found, synthetic_labels, SimpleNamespace(to_frame=lambda: frame)

                model = SimpleNamespace(predict_proba=lambda x: np.tile([.2, .8], (len(x), 1)))
                with patch.object(module, "_extract_features_runtime", side_effect=extract), redirect_stdout(io.StringIO()):
                    table = native.build_native_table(payload(), kind, module,
                                                      {"pipeline": model, "feature_order": [feature]},
                                                      pool_policy=pool_policy)
                self.assertEqual(rows, table.candidates)
                np.testing.assert_array_equal(truth, table.truth)
                self.assertEqual(truth.dtype, np.int8)
                for key, value in meta.items():
                    self.assertEqual(table.meta[key], value)
                self.assertEqual(meta["candidate_policy_source"], "candidate_pools.json")

    def test_gt_changes_labels_not_pool_identity(self):
        for kind, module in self.modules.items():
            with self.subTest(kind=kind):
                before = payload()
                after = deepcopy(before)
                after["gt_merge_sites"] = []
                after["gt_node_canonical_label"][:] = 0
                rows_a, truth_a, _ = native.build_native_universe(before, kind, module)
                rows_b, truth_b, _ = native.build_native_universe(after, kind, module)
                self.assertEqual(rows_a, rows_b)
                self.assertTrue(np.any(truth_a != truth_b))

    @unittest.skipUnless(pc.DEFAULT_MERGE_DIR and pc.DEFAULT_SPLIT_DIR,
                         "no default frozen detectors configured (the 2026-08 ones were deleted)")
    def test_real_frozen_feature_extraction_and_model_parity_on_tiny_graph(self):
        selected = pc.resolve_detector_runs()
        for kind, records in selected.items():
            module = pc._import_module(Path(records[0]["dir"]) / records[0]["script"], "test_real_" + kind)
            with self.subTest(kind=kind), redirect_stdout(io.StringIO()):
                record = records[0]
                bundle = pc._load_joblib(Path(record["dir"]) / record["file"])
                data = payload()
                table = native.build_native_table(data, kind, module, bundle)
                # Native runtime directly, with its own original candidate builder: on this graph
                # it enumerates the same candidates as the declared rule.
                rows, _, accumulator = module._extract_features_runtime(dict(data), verbose=False)
                frame = accumulator.to_frame().reset_index(drop=True)[bundle["feature_order"]]
                self.assertEqual([module.sample_display(r) for r in table.candidates],
                                 [module.sample_display(r) for r in rows])
                np.testing.assert_allclose(table.features[bundle["feature_order"]], frame, equal_nan=True)
                np.testing.assert_allclose(table.features["detector_score"],
                                           pc._score(bundle["pipeline"], frame.to_numpy(dtype=float)))

    def test_native_storage_reuse_no_raw_reload_and_corruption_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "dataset_cache_1_mcl100_add.pkl"
            source.write_bytes(b"synthetic labeled cache placeholder")
            built = []
            def build(data, kind, module, bundle, pool_policy=None, image_workers=0):
                rows = [{"segment_id": 1, "node_id": 0}] if kind == "merge" else [{"segment_id_a": 1, "segment_id_b": 2}]
                built.append(kind)
                return native.NativeTable(kind, pd.DataFrame({"detector_score": [.5]}), rows, np.array([1]),
                                          {"pool_sha256": native.pool_digest(kind, rows), "rows": 1})
            with patch.object(pc.ds, "load_cached_graphs", return_value=(None, None, payload())) as load, \
                    patch.object(native, "build_native_table", side_effect=build), redirect_stdout(io.StringIO()):
                first = native.ensure_native_tables("1", source, self.selected, root / "tables", True, 100)
                second = native.ensure_native_tables("1", source, self.selected, root / "tables", False, 100)
            self.assertEqual(load.call_count, 1)
            self.assertEqual(built, ["merge", "split"])
            self.assertEqual(first.meta, second.meta)
            directory = Path(first.meta["table_paths"]["split"])
            (directory / "evaluator_labels.npy").write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "Corrupt"):
                native.ensure_native_tables("1", source, self.selected, root / "tables", False, 100)

    def test_missing_native_tables_never_load_or_fall_back_to_legacy(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.pkl"
            source.write_bytes(b"source")
            with patch.object(pc.ds, "load_cached_graphs") as load, self.assertRaises(FileNotFoundError):
                native.ensure_native_tables("1", source, self.selected, Path(tmp) / "tables", False, 100)
            load.assert_not_called()
