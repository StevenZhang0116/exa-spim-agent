"""
Smoke tests for the detector-seeded evolution stack (no real brain data):

  * harness.feature_bank.FeatureBank — lookups, pseudo-label stripping,
    order-free pair keys, quantiles, site annotation, disk roundtrip;
  * artifacts/heuristics.py (the detector-prior two-phase seed) — all three
    ctx["phase"] dispatches, score gates, geometric vetoes, nan fallbacks,
    and the bank-less degradation path;
  * cli.precompute_error_scores._build_split_universe — detector-schema rows
    from loop SplitSites (pair grouping, occurrence orientation, node roles).

Run (compute node, panda env):
    python -m unittest discover -s proofreader_evolve/tests -t .
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from proofreader_evolve.harness.feature_bank import (
    FeatureBank, base_label, pair_key)

REPO = Path(__file__).resolve().parent.parent.parent
SEED_PY = REPO / "proofreader_evolve" / "artifacts" / "heuristics.py"
GEOM_SEED_PY = REPO / "proofreader_evolve" / "artifacts" / "seeds" / "geometric.py"


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #

def _make_bank():
    import pandas as pd
    merge_df = pd.DataFrame(
        {"merge_score": [0.9, 0.2],
         "f1": [1.5, float("nan")], "f1_is_defined": [True, False]},
        index=pd.Index(["100", "200"], name="label"))
    split_df = pd.DataFrame({
        "label_a": ["100", "300", "400"],
        "label_b": ["300", "100", "500"],
        "node_a": [1, 5, 9],
        "node_b": [2, 6, 10],
        "gap_um": [3.0, 2.0, 1.0],
        "split_score": [0.8, 0.5, 0.1],
        "g1": [7.0, 8.0, 9.0],
        "g1_is_defined": [True, True, True],
    })
    return FeatureBank("test", merge_df, split_df, {"n_graph_nodes": 4})


class _FakeChainGraph:
    """Two colinear 2-node fragments along +x: A = nodes 0-1, B = nodes 2-3.

    Node 1 (tip of A) and node 2 (tip of B) face each other across a 2 µm gap
    on a straight line, so the seed's colinearity test passes; node 4/5 form a
    perpendicular fragment C used for the non-colinear case.
    """

    def __init__(self):
        self.node_xyz = np.array([
            [0.0, 0.0, 0.0], [6.0, 0.0, 0.0],     # A
            [8.0, 0.0, 0.0], [14.0, 0.0, 0.0],    # B
            [8.0, 2.0, 0.0], [8.0, 8.0, 0.0],     # C (perpendicular)
        ])
        self._seg = ["A", "A", "B", "B", "C", "C"]
        self._nbrs = {0: [1], 1: [0], 2: [3], 3: [2], 4: [5], 5: [4]}

    def node_segment_id(self, n):
        return self._seg[n]

    def neighbors(self, n):
        return list(self._nbrs[n])

    def degree(self, n):
        return len(self._nbrs[n])


def _split_site(label_a="A", label_b="B", node_a=1, node_b=2, gap_um=2.0,
                score=float("nan"), mutual=False):
    return SimpleNamespace(
        kind="split", label_a=label_a, label_b=label_b, node_a=node_a,
        node_b=node_b, gap_um=gap_um, xyz_a=None, xyz_b=None, alt_gaps=[],
        recip_rank_a=1, recip_rank_b=1, mutual_nearest=mutual,
        split_score=score, as_edit=lambda a=label_a, b=label_b: (a, b))


def _merge_site(label="100", score=float("nan"), reconverge=None,
                cable=(30.0, 30.0), angle=60.0, detector="branch"):
    return SimpleNamespace(
        kind="merge", label=label, cut_node=0, cut_xyz=(0, 0, 0),
        seed_a_node=0, seed_b_node=1, seed_a_xyz=(0, 0, 0), seed_b_xyz=(1, 0, 0),
        branch_degree=3, angle_deg=angle, radius_ratio=1.5,
        cable_a_um=cable[0], cable_b_um=cable[1], detector=detector,
        arms_reconverge=reconverge, extra_seeds=[], seed_groups=[],
        label_merge_score=score,
        as_edit=lambda lab=label: {"kind": "split_label", "label": lab,
                                   "seed_a_xyz": (0, 0, 0),
                                   "seed_b_xyz": (1, 0, 0)})


def _load_seed(path=SEED_PY):
    spec = importlib.util.spec_from_file_location(f"_seed_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------- #
# FeatureBank
# --------------------------------------------------------------------------- #

class TestFeatureBank(unittest.TestCase):

    def test_label_helpers(self):
        self.assertEqual(base_label("123#a"), "123")
        self.assertEqual(base_label("123"), "123")
        self.assertEqual(pair_key("2", "10"), ("10", "2"))       # lexicographic
        self.assertEqual(pair_key("10#a", "2"), pair_key("2", "10"))

    def test_merge_score_and_pseudo_labels(self):
        bank = _make_bank()
        self.assertEqual(bank.merge_score("100"), 0.9)
        self.assertEqual(bank.merge_score("200#b"), 0.2)   # pseudo-label strips
        self.assertTrue(math.isnan(bank.merge_score("999")))

    def test_split_score_orderfree_best_gap(self):
        bank = _make_bank()
        # pair (100,300) has two gaps (0.8 and 0.5) -> best wins, order-free
        self.assertEqual(bank.split_score("100", "300"), 0.8)
        self.assertEqual(bank.split_score("300", "100"), 0.8)
        self.assertEqual(bank.split_score("300#a", "100"), 0.8)  # pseudo strips
        self.assertTrue(math.isnan(bank.split_score("100", "999")))

    def test_feature_rows_exclude_key_columns(self):
        bank = _make_bank()
        mf = bank.merge_features("100")
        self.assertIn("f1", mf)
        self.assertNotIn("merge_score", mf)
        sf = bank.split_features("300", "100")
        self.assertIn("g1", sf)
        self.assertEqual(sf["g1"], 7.0)   # the BEST gap's row (score 0.8)
        for key in ("label_a", "label_b", "node_a", "node_b", "split_score"):
            self.assertNotIn(key, sf)
        self.assertEqual(bank.merge_features("999"), {})
        self.assertEqual(bank.split_features("1", "2"), {})

    def test_annotate_sites(self):
        bank = _make_bank()
        splits = [_split_site("100", "300"), _split_site("400", "999")]
        merges = [_merge_site("100"), _merge_site("777")]
        self.assertEqual(bank.annotate_split_sites(splits), 1)
        self.assertEqual(splits[0].split_score, 0.8)
        self.assertTrue(math.isnan(splits[1].split_score))
        self.assertEqual(bank.annotate_merge_sites(merges), 1)
        self.assertEqual(merges[0].label_merge_score, 0.9)
        self.assertTrue(math.isnan(merges[1].label_merge_score))

    def test_quantiles(self):
        bank = _make_bank()
        self.assertAlmostEqual(bank.score_quantile("merge", 0.0), 0.2)
        self.assertAlmostEqual(bank.score_quantile("merge", 1.0), 0.9)
        # split pair-level scores are {0.8, 0.1} (best-per-pair)
        self.assertAlmostEqual(bank.score_quantile("split", 1.0), 0.8)
        with self.assertRaises(ValueError):
            bank.score_quantile("bogus", 0.5)

    def test_disk_roundtrip_and_missing(self):
        bank = _make_bank()
        with tempfile.TemporaryDirectory() as td:
            d = Path(td) / "test"
            d.mkdir()
            bank._merge_df.to_pickle(d / "merge_scores.pkl")
            bank._split_df.to_pickle(d / "split_scores.pkl")
            (d / "meta.json").write_text(json.dumps({"n_graph_nodes": 4}))
            loaded = FeatureBank.load("test", tables_dir=td)
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded.split_score("100", "300"), 0.8)
            self.assertEqual(loaded.merge_score("100"), 0.9)
            # graph validation: matching count -> no warning; mismatch -> warning
            ok_graph = SimpleNamespace(number_of_nodes=lambda: 4)
            self.assertEqual(loaded.validate_graph(ok_graph), [])
            bad_graph = SimpleNamespace(number_of_nodes=lambda: 5)
            self.assertEqual(len(loaded.validate_graph(bad_graph)), 1)
            self.assertIsNone(FeatureBank.load("no_such_brain", tables_dir=td))


# --------------------------------------------------------------------------- #
# the detector-prior seed policy
# --------------------------------------------------------------------------- #

class TestSeedPolicy(unittest.TestCase):

    def setUp(self):
        self.seed = _load_seed()
        self.graph = _FakeChainGraph()
        self.bank = _make_bank()

    def _ctx(self, phase, bank="default"):
        return {
            "phase": phase,
            "feature_bank": self.bank if bank == "default" else bank,
            "fragments_graph": self.graph,
            "max_gap_um": 15.0, "enum_params": {"max_gap_um": 15.0},
            "n_split_sites": 0, "n_merge_sites": 0,
            "node_radius": None, "read_image_patch": None,
            "image_patch_shape": (16, 16, 16),
        }

    def test_merge_repair_accepts_top_scored_guarded_site(self):
        # quantile(0.99) over {0.9, 0.2} sits just below 0.9 -> only 0.9 clears
        sites = [_merge_site("100", score=0.9),
                 _merge_site("200", score=0.2)]
        edits = self.seed.propose_edits(sites, self._ctx("merge_repair"))
        self.assertEqual(len(edits), 1)
        self.assertEqual(edits[0]["kind"], "split_label")
        self.assertEqual(edits[0]["label"], "100")

    def test_merge_repair_geometric_vetoes(self):
        vetoed = [
            _merge_site("100", score=0.9, reconverge=True),        # loop/branch
            _merge_site("100", score=0.9, cable=(5.0, 30.0)),      # short arm
            _merge_site("100", score=0.9, angle=175.0),            # straight
        ]
        for site in vetoed:
            edits = self.seed.propose_edits([site], self._ctx("merge_repair"))
            self.assertEqual(edits, [], f"site should be vetoed: {site}")
        # nan angle (bridge/component detectors) must NOT veto by itself
        ok = _merge_site("100", score=0.9, angle=float("nan"),
                         detector="component")
        edits = self.seed.propose_edits([ok], self._ctx("merge_repair"))
        self.assertEqual(len(edits), 1)

    def test_merge_repair_without_bank_or_scores(self):
        site = _merge_site("100", score=float("nan"))
        self.assertEqual(
            self.seed.propose_edits([site], self._ctx("merge_repair")), [])
        self.assertEqual(
            self.seed.propose_edits([_merge_site("100", score=0.9)],
                                    self._ctx("merge_repair", bank=None)), [])

    def test_merge_repair_topk_budget(self):
        sites = [_merge_site(str(i), score=0.9) for i in range(1000, 1030)]
        edits = self.seed.propose_edits(sites, self._ctx("merge_repair"))
        self.assertLessEqual(len(edits), self.seed.MERGE_TOPK_CUTS)

    def test_split_repair_score_path_needs_geometric_agreement(self):
        # colinear A-B pair, high score -> accepted even without mutual_nearest
        colinear = _split_site("A", "B", 1, 2, score=0.9)
        edits = self.seed.propose_edits([colinear], self._ctx("split_repair"))
        self.assertEqual(edits, [("A", "B")])
        # perpendicular A-C pair, high score, not mutual -> refused
        perp = _split_site("A", "C", 1, 4, score=0.9)
        self.assertEqual(
            self.seed.propose_edits([perp], self._ctx("split_repair")), [])
        # perpendicular but mutual_nearest -> the guard is satisfied
        perp_mutual = _split_site("A", "C", 1, 4, score=0.9, mutual=True)
        self.assertEqual(
            self.seed.propose_edits([perp_mutual], self._ctx("split_repair")),
            [("A", "C")])

    def test_split_repair_nan_score_falls_back_to_geometry(self):
        near_colinear = _split_site("A", "B", 1, 2, gap_um=2.0)   # nan score
        edits = self.seed.propose_edits([near_colinear],
                                        self._ctx("split_repair", bank=None))
        self.assertEqual(edits, [("A", "B")])
        far = _split_site("A", "B", 1, 2, gap_um=10.0)            # gap too wide
        self.assertEqual(
            self.seed.propose_edits([far], self._ctx("split_repair", bank=None)),
            [])

    def test_single_phase_emits_both_kinds(self):
        sites = [_merge_site("100", score=0.9), _split_site("A", "B", 1, 2, score=0.9)]
        edits = self.seed.propose_edits(sites, self._ctx("single"))
        kinds = {e["kind"] if isinstance(e, dict) else "merge_tuple" for e in edits}
        self.assertEqual(kinds, {"split_label", "merge_tuple"})

    def test_geometric_control_seed_still_works(self):
        geom = _load_seed(GEOM_SEED_PY)
        site = _split_site("A", "B", 1, 2, gap_um=2.0)
        edits = geom.propose_edits([site], self._ctx("single", bank=None))
        self.assertEqual(edits, [("A", "B")])


# --------------------------------------------------------------------------- #
# precompute: loop sites -> detector-schema universe rows
# --------------------------------------------------------------------------- #

class TestBuildSplitUniverse(unittest.TestCase):

    def test_rows_group_orient_and_role(self):
        from proofreader_evolve.cli.precompute_error_scores import (
            _build_split_universe)
        graph = _FakeChainGraph()
        # numeric labels (the real loop labels are digit strings); two sites for
        # the SAME pair (both directions) must fold into ONE row with 2 occurrences
        s1 = _split_site("20", "10", node_a=1, node_b=2, gap_um=2.0)
        s2 = _split_site("10", "20", node_a=2, node_b=1, gap_um=2.5)
        samples, labels, meta = _build_split_universe(graph, [s1, s2])
        self.assertEqual(len(samples), 1)
        self.assertTrue((labels == 0).all())
        row = samples[0]
        self.assertEqual((row["segment_id_a"], row["segment_id_b"]), (10, 20))
        self.assertEqual(len(row["occurrences"]), 2)
        # representative = closest gap; node_id_a must lie on segment 10
        self.assertEqual(row["gap_um"], 2.0)
        self.assertEqual(row["node_id_a"], 2)   # node 2 has label "10" (site s1: b)
        self.assertEqual(row["node_role_a"], "tip")
        # loop-facing identity keeps the original label strings
        la, lb, _na, _nb, gap = meta[0]
        self.assertEqual({la, lb}, {"10", "20"})
        self.assertEqual(gap, 2.0)

    def test_same_segment_pairs_dropped(self):
        from proofreader_evolve.cli.precompute_error_scores import (
            _build_split_universe)
        graph = _FakeChainGraph()
        s = _split_site("10", "10#a", node_a=1, node_b=2)   # same base label
        samples, labels, meta = _build_split_universe(graph, [s])
        self.assertEqual(samples, [])


if __name__ == "__main__":
    unittest.main()
