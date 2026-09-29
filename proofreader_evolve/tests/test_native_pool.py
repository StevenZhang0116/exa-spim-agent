"""Frozen-detector parity on tiny synthetic graphs; no real-brain or API calls."""

from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
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
from proofreader_evolve.harness import native_pool as native


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
        cls.selected = pc.resolve_detector_runs()
        cls.modules = {kind: pc._import_module(Path(records[0]["dir"]) / records[0]["script"],
                                               "test_native_" + kind)
                       for kind, records in cls.selected.items()}

    def test_exact_frozen_rows_occurrences_and_labels_with_no_gt_in_extraction(self):
        for kind, module in self.modules.items():
            with self.subTest(kind=kind):
                data = payload()
                expected, labels = module.build_sample_universe(dict(data))
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

    def test_gt_changes_labels_not_pool_identity(self):
        for kind, module in self.modules.items():
            with self.subTest(kind=kind):
                before = payload()
                after = deepcopy(before)
                after["gt_merge_sites"] = []
                after["gt_node_canonical_label"][:] = 0
                rows_a, truth_a = module.build_sample_universe(before)
                rows_b, truth_b = module.build_sample_universe(after)
                self.assertEqual(native.blind_rows(kind, rows_a), native.blind_rows(kind, rows_b))
                self.assertTrue(np.any(truth_a != truth_b))

    def test_real_frozen_feature_extraction_and_model_parity_on_tiny_graph(self):
        for kind, module in self.modules.items():
            with self.subTest(kind=kind), redirect_stdout(io.StringIO()):
                record = self.selected[kind][0]
                bundle = pc._load_joblib(Path(record["dir"]) / record["file"])
                data = payload()
                table = native.build_native_table(data, kind, module, bundle)
                # Native runtime directly, with its own original candidate builder.
                rows, _, accumulator = module._extract_features_runtime(dict(data), verbose=False)
                frame = accumulator.to_frame().reset_index(drop=True)[bundle["feature_order"]]
                self.assertEqual(table.candidates, native.blind_rows(kind, rows))
                np.testing.assert_allclose(table.features[bundle["feature_order"]], frame, equal_nan=True)
                np.testing.assert_allclose(table.features["detector_score"],
                                           pc._score(bundle["pipeline"], frame.to_numpy(dtype=float)))

    def test_frozen_split_model_never_computes_excluded_h33(self):
        module = self.modules["split"]
        record = self.selected["split"][0]
        bundle = pc._load_joblib(Path(record["dir"]) / record["file"])
        computed = set()
        original = module._compute_group

        def compute(key, *args, **kwargs):
            if key == "h33_positional":
                self.fail("Precomputation executed the unused whole-brain scan")
            computed.add(key)
            return original(key, *args, **kwargs)

        with patch.object(module, "_compute_group", side_effect=compute), redirect_stdout(io.StringIO()):
            table = native.build_native_table(payload(), "split", module, bundle)
        self.assertTrue(computed)
        self.assertIn("h30_positional", computed)
        self.assertIn("h38_positional", computed)
        self.assertEqual(list(table.features), [*bundle["feature_order"], "detector_score"])
        enabled = pc._model_analysis_keys(module, bundle["feature_order"])
        self.assertEqual({g["key"] for g in module.ANALYSIS_TIMING_GROUPS} - enabled,
                         {"h33_positional"})

    def test_native_storage_reuse_no_raw_reload_and_corruption_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "dataset_cache_1_mcl100_add.pkl"
            source.write_bytes(b"synthetic labeled cache placeholder")
            built = []
            def build(data, kind, module, bundle):
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
