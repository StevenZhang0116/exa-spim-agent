"""Pool-only companion export: same rows, labels and pool hash as the native tables, no tables needed."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from proofreader_evolve.cli import export_rerank_companion as ex
from proofreader_evolve.cli import precompute_error_scores as pc
from proofreader_evolve.harness import candidate_pools as cp
from proofreader_evolve.harness import label_availability as la
from proofreader_evolve.harness import native_pool as native
from proofreader_evolve.tests.test_native_pool import payload


def split_payload_with_negatives():
    """The tiny native-pool payload plus a third segment (333) on its own GT neuron.

    Pair (111, 222) is a GT split (two flanks of an omitted GT stretch); pairs with 333
    are GT-judgeable non-splits, so the split labels carry both classes.
    """
    import networkx as nx
    data = payload()
    graph = data["fragments_graph"]
    start = len(graph.node_xyz)
    graph.add_edges_from([(start, start + 1), (start + 1, start + 2)])
    graph.node_xyz = np.vstack([graph.node_xyz, [[50.0, 20.0, 0.0], [55.0, 20.0, 0.0], [60.0, 20.0, 0.0]]])
    graph.node_component_id = np.concatenate([graph.node_component_id, [30, 30, 30]])
    graph.component_id_to_swc_id = {**graph.component_id_to_swc_id, 30: "333.0.swc"}
    graph.node_radius = np.ones(len(graph.node_xyz))
    gt = nx.Graph()
    gt.add_edges_from([(0, 1), (1, 2), (3, 4)])
    data.update(gt_graph=gt, gt_edge_error=np.array([2, 2, 0]),
                gt_node_canonical_label=np.array([111, 0, 222, 333, 333]))
    return data


class PoolOnlyCompanionTests(unittest.TestCase):
    def test_pool_only_bank_matches_the_enumeration_and_availability_rules(self):
        data = split_payload_with_negatives()
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "dataset_cache_1_mcl100_add.pkl"
            source.write_bytes(b"placeholder")
            with patch.object(ex, "cache_path", return_value=source), \
                    patch.object(pc.ds, "load_cached_graphs", return_value=(data["fragments_graph"], None, data)), \
                    redirect_stdout(io.StringIO()):
                bank, graph = ex.pool_only_bank("1", 100, ["split"])
        table = bank.tables["split"]
        policy = cp.load_pool_policies()["split"]
        module = pc._import_module(Path(pc.resolve_detector_runs()["split"][0]["dir"]) /
                                   pc.resolve_detector_runs()["split"][0]["script"], "test_pool_only_split")
        rows, truth, meta = native.build_native_universe(split_payload_with_negatives(), "split", module, policy)
        self.assertEqual(table.candidates, rows)
        self.assertEqual(table.meta["pool_sha256"], meta["pool_sha256"])
        self.assertEqual(table.meta["candidate_policy"], "tip_to_any_node|r=80|k=2")
        self.assertEqual(table.meta["provenance"]["pool_policy"], policy)
        available = la.compute("split", rows, data["gt_node_canonical_label"])["segment_on_gt"]
        np.testing.assert_array_equal(table.truth, la.three_valued(truth, available))
        self.assertEqual(len(table.features), len(rows))
        self.assertIs(graph, data["fragments_graph"])
        self.assertEqual(table.availability["counts"], la.label_counts(table.truth))
        self.assertGreater(table.availability["counts"]["0"], 0)
        self.assertGreater(table.availability["counts"]["1"], 0)
        keys = {(r["segment_id_a"], r["segment_id_b"]) for r in rows}
        self.assertIn((111, 222), keys)

    def test_a_pool_without_negatives_is_refused(self):
        data = payload()  # one split pair, positive: no GT-judgeable negative
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "dataset_cache_1_mcl100_add.pkl"
            source.write_bytes(b"placeholder")
            with patch.object(ex, "cache_path", return_value=source), \
                    patch.object(pc.ds, "load_cached_graphs", return_value=(data["fragments_graph"], None, data)), \
                    redirect_stdout(io.StringIO()), self.assertRaisesRegex(la.AvailabilityMismatch, "lack one"):
                ex.pool_only_bank("1", 100, ["split"])

    def test_split_rule_text_in_the_manifest_follows_the_table(self):
        table = native.NativeTable("split", None, [], np.array([]), {"candidate_policy": "tip_to_any_node|r=80|k=2"})
        self.assertIn("within 80 um; the 2 nearest partner segments are kept", ex.candidate_rule("split", table))
        self.assertEqual(ex.candidate_rule("merge", table), ex.RULES["merge"])


if __name__ == "__main__":
    unittest.main()
