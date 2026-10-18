"""candidate_rules.py enumerates and labels exactly as the code it replaced.

References: the split and merge-site adapters frozen in ``legacy_candidate_adapters.py``
(what every detector embedded until 2026-10-16) and, for tip-to-shaft candidates, the
measurement in ``notebooks/merge_shaft_candidate_reach.py`` (shaft nodes from the split
``tip_to_any_node|r=80|k=2`` occurrences, then ``shaft_keep_mask``), whose per-brain counts
are the evidence for the declared merge rule.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys
import unittest

import networkx as nx
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import candidate_rules as cr  # noqa: E402
from agentic.tests.legacy_candidate_adapters import (  # noqa: E402
    LEGACY_MERGE_SITE_ADAPTER, LEGACY_SPLIT_ADAPTER)


class _Graph(nx.Graph):
    pass


def random_payload(seed, n_segments=7, box_um=70.0):
    """Branching fragments of several segments in a small box (tips land near other
    segments' shafts), one invalid component, a labeled GT graph and merge sites."""
    rng = np.random.default_rng(seed)
    graph, xyz, components, mapping = _Graph(), [], [], {}

    def add_tree(component, size):
        ids = []
        for i in range(size):
            node = len(xyz)
            if i == 0:
                position = rng.uniform(0, box_um, 3)
                parent = None
            else:
                # mostly unbranched paths with occasional branches, so shafts exist away from junctions
                parent = ids[-1] if rng.random() > 0.08 else ids[int(rng.integers(0, i))]
                position = xyz[parent] + rng.normal(0, 3.0, 3)
            xyz.append(position)
            components.append(component)
            graph.add_node(node)
            if parent is not None:
                graph.add_edge(parent, node)
            ids.append(node)

    for segment in range(1, n_segments + 1):
        for part in range(int(rng.integers(1, 3))):
            component = segment * 10 + part
            mapping[component] = f"{segment}.{part}.swc"
            add_tree(component, int(rng.integers(4, 60)))
    add_tree(999, 6)  # a component without a segment: never a candidate or partner
    graph.node_xyz = np.asarray(xyz, dtype=np.float32)
    graph.node_component_id = np.asarray(components)
    graph.component_id_to_swc_id = mapping

    gt = _Graph()
    n_gt = 60
    for node in range(n_gt):
        gt.add_node(node)
        if node and rng.random() > 0.1:
            gt.add_edge(int(rng.integers(max(0, node - 4), node)), node)
    labels = rng.choice(np.arange(0, n_segments + 1), size=n_gt, p=None)
    labels[rng.random(n_gt) < 0.25] = 0
    edge_error = (rng.random(gt.number_of_edges()) < 0.2).astype(np.int64)
    segments = cr.node_arrays(graph)[1]
    valid = np.flatnonzero(segments > 0)
    sites = []
    for node in rng.choice(valid, size=8, replace=False):
        sites.append({"segment_id": int(segments[node]),
                      "xyz": (np.asarray(xyz[node]) + rng.normal(0, 2.0, 3)).tolist(),
                      "gt_neuron": f"N{int(rng.integers(0, 3))}",
                      "sources": [cr.SITE_SOURCES[int(rng.integers(0, 2))]]})
    return {"fragments_graph": graph, "gt_graph": gt, "gt_node_canonical_label": labels,
            "gt_edge_error": edge_error, "gt_merge_sites": sites,
            "gt_merge_labels": sorted({s["segment_id"] for s in sites}),
            "gt_merge_site_metadata": {"generator": "test"}}


def _comp_to_seg(graph):
    return {int(c): int(str(s).split(".")[0]) for c, s in graph.component_id_to_swc_id.items()}


def legacy_split(rule):
    namespace = {"np": np, "defaultdict": defaultdict, "build_comp_to_seg": _comp_to_seg,
                 "ROW_UNIT": "candidate segment pair",
                 "CANDIDATE_POLICY_CONFIG_ID": rule["config_id"], "CANDIDATE_PAIRING_RULE": rule["mode"],
                 "CANDIDATE_MAX_DISTANCE_UM": float(rule["radius_um"]),
                 "CANDIDATE_PER_ANCHOR_K": rule["per_anchor_k"], "CANDIDATE_GLOBAL_CAP": None,
                 "CANDIDATE_POLICY_SHA256": "0" * 64}
    exec(LEGACY_SPLIT_ADAPTER, namespace)
    return namespace


def legacy_merge(nms_um, claim_um, label_um):
    namespace = {"np": np, "defaultdict": defaultdict, "build_comp_to_seg": _comp_to_seg,
                 "ROW_UNIT": "candidate junction site",
                 "CANDIDATE_POLICY_CONFIG_ID": f"junction|nms={nms_um:g}|r={claim_um:g}",
                 "CANDIDATE_MODE": "junction", "CANDIDATE_NMS_UM": float(nms_um),
                 "CANDIDATE_CLAIM_RADIUS_UM": float(claim_um),
                 "CANDIDATE_POSITIVE_LABEL_RADIUS_UM": float(label_um),
                 "CANDIDATE_SITE_SNAP_MAX_UM": 50.0, "CANDIDATE_POLICY_SHA256": "0" * 64}
    exec(LEGACY_MERGE_SITE_ADAPTER, namespace)
    return namespace


def split_rule(mode, radius, k):
    return {"config_id": cr.split_config_id(mode, radius, k), "mode": mode, "radius_um": radius,
            "per_anchor_k": k, "global_cap": None}


def merge_rule(nms=20.0, claim=150.0, label=20.0, gap=None, k=None):
    mode = "junction" if gap is None else "junction+tip_to_shaft"
    return {"config_id": cr.merge_config_id(mode, nms, claim, gap, k), "mode": mode, "nms_um": nms,
            "tip_gap_um": gap, "tip_partner_k": k, "claim_radius_um": claim, "positive_label_radius_um": label}


def same(a, b):
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return a == b


SEEDS = range(12)


class SplitParityTests(unittest.TestCase):
    def test_rows_labels_and_audit_match_the_legacy_adapter(self):
        rules = [split_rule("tip_to_any_node", 80.0, 2), split_rule("tip_to_any_node", 20.0, None),
                 split_rule("tip_to_tip", 50.0, 1), split_rule("any_node_to_any_node", 15.0, 3)]
        for seed in SEEDS:
            for rule in rules:
                payload = random_payload(seed)
                legacy = legacy_split(rule)
                old_samples, old_labels = legacy["build_sample_universe"](dict(payload))
                old_audit = legacy["sample_universe_audit"](dict(payload), old_samples, old_labels)
                samples, labels = cr.build_split_universe(dict(payload), rule)
                with self.subTest(seed=seed, rule=rule["config_id"]):
                    self.assertTrue(len(samples) > 0)
                    self.assertTrue(same(samples, old_samples))
                    self.assertEqual(labels.tolist(), old_labels.tolist())
                    audit = cr.split_audit(payload, samples, labels)
                    self.assertTrue(same(audit, {k: v for k, v in old_audit.items()
                                                 if k in audit}))
                    self.assertEqual(set(old_audit) - set(audit), {
                        "row_unit", "candidate_policy_config_id", "candidate_policy_sha256",
                        "candidate_pairing_rule", "candidate_max_distance_um", "candidate_per_anchor_k",
                        "candidate_global_cap"})


class MergeParityTests(unittest.TestCase):
    def test_junction_rule_matches_the_legacy_merge_site_adapter(self):
        for seed in SEEDS:
            for nms, claim, label in ((20.0, 150.0, 20.0), (0.0, 30.0, 10.0), (10.0, 60.0, 60.0)):
                payload = random_payload(seed)
                legacy = legacy_merge(nms, claim, label)
                old_payload = dict(payload)
                old_samples, old_labels = legacy["build_sample_universe"](old_payload)
                old_audit = legacy["sample_universe_audit"](old_payload, old_samples, old_labels)
                rule = merge_rule(nms, claim, label)
                samples, labels, raw = cr.build_merge_universe(dict(payload), rule)
                audit = cr.merge_audit(raw, samples, labels, rule)
                with self.subTest(seed=seed, rule=rule["config_id"]):
                    self.assertTrue(len(samples) > 0)
                    self.assertEqual({s["candidate_family"] for s in samples}, {"junction"})
                    self.assertTrue(all(s["tip_gap_um"] is None for s in samples))
                    stripped = [{k: v for k, v in s.items() if k not in ("candidate_family", "tip_gap_um")}
                                for s in samples]
                    self.assertTrue(same(stripped, old_samples))
                    self.assertEqual(labels.tolist(), old_labels.tolist())
                    for key in ("n_rows", "n_positive", "n_negative", "n_ambiguous_ring", "n_junctions_raw",
                                "n_sites", "n_sites_snapped", "n_sites_covered_at_claim_radius",
                                "site_recall_at_claim_radius", "site_provenance"):
                        self.assertTrue(same(audit[key], old_audit[key]), key)

    def test_tip_to_shaft_matches_the_reach_measurement(self):
        try:
            sys.path.insert(0, str(REPO / "notebooks"))
            from merge_shaft_candidate_reach import shaft_keep_mask, NMS_UM
        except Exception as exc:  # pragma: no cover - depends on notebook imports
            self.skipTest(f"reach script not importable: {exc}")
        self.assertEqual(NMS_UM, 20.0)
        found = defaultdict(int)
        for seed in SEEDS:
            payload = random_payload(seed)
            xyz, node_segments, degrees = cr.node_arrays(payload["fragments_graph"])
            # the measurement: non-tip partners of the split tip_to_any_node|r=80|k=2 occurrences
            old_samples, _ = legacy_split(split_rule("tip_to_any_node", 80.0, 2))["build_sample_universe"](
                dict(payload))
            best = {}
            for row in old_samples:
                for occ in row["occurrences"]:
                    partner = "b" if occ["anchor_side"] == "a" else "a"
                    if occ[f"node_role_{partner}"] != "non_tip":
                        continue
                    node, gap = int(occ[f"node_id_{partner}"]), float(occ["gap_um"])
                    if gap < best.get(node, math.inf):
                        best[node] = gap
            nodes = np.fromiter(best.keys(), dtype=np.int64, count=len(best))
            gaps = np.fromiter(best.values(), dtype=float, count=len(best))
            _, kept_j = cr.kept_junctions(xyz, node_segments, degrees, NMS_UM)
            index = {"n_nodes": len(xyz), "xyz": xyz, "node_segments": node_segments}
            for gap in (10.0, 20.0, 40.0):
                reference = shaft_keep_mask(index, nodes, gaps, kept_j, gap)
                kept, tip_gap, counts = cr.merge_candidate_nodes(
                    xyz, node_segments, degrees, merge_rule(gap=gap, k=2))
                found[gap] += int(reference.sum())
                with self.subTest(seed=seed, gap=gap):
                    self.assertEqual(np.flatnonzero(kept & ~kept_j).tolist(), np.flatnonzero(reference).tolist())
                    self.assertEqual(sorted(tip_gap), np.flatnonzero(reference).tolist())
                    self.assertTrue(all(best[n] == g for n, g in tip_gap.items()))
                    self.assertEqual(counts["n_tip_to_shaft_candidates"], int(reference.sum()))
        self.assertTrue(all(found[gap] >= 10 for gap in (10.0, 20.0, 40.0)), dict(found))

    def test_shaft_rows_carry_family_gap_and_labels(self):
        payload = random_payload(3)
        rule = merge_rule(gap=20.0, k=2)
        samples, labels, raw = cr.build_merge_universe(payload, rule)
        families = {s["candidate_family"] for s in samples}
        self.assertEqual(families, {"junction", "tip_to_shaft"})
        for s in samples:
            if s["candidate_family"] == "tip_to_shaft":
                self.assertEqual(s["degree"], 2)
                self.assertTrue(0 <= s["tip_gap_um"] <= 20.0)
            else:
                self.assertGreaterEqual(s["degree"], 3)
        self.assertEqual([s["node_id"] for s in samples], sorted(s["node_id"] for s in samples))
        self.assertEqual([s["candidate_id"] for s in samples], list(range(len(samples))))
        junction_only, _, _ = cr.build_merge_universe(random_payload(3), merge_rule())
        self.assertEqual([s["node_id"] for s in junction_only],
                         [s["node_id"] for s in samples if s["candidate_family"] == "junction"])
        audit = cr.merge_audit(raw, samples, labels, rule)
        self.assertEqual(audit["n_rows"], len(samples))
        self.assertEqual(audit["n_junction_candidates"] + audit["n_tip_to_shaft_candidates"], len(samples))
        json.dumps([{k: s[k] for k in cr.GEOMETRY_FIELDS["merge"]} for s in samples], allow_nan=False)


class RuleTests(unittest.TestCase):
    def test_config_ids(self):
        self.assertEqual(cr.split_config_id("tip_to_any_node", 80, 2), "tip_to_any_node|r=80|k=2")
        self.assertEqual(cr.split_config_id("tip_to_tip", 50.5, None), "tip_to_tip|r=50.5|k=all")
        self.assertEqual(cr.merge_config_id("junction", 20, 150), "junction|nms=20|r=150")
        self.assertEqual(cr.merge_config_id("junction+tip_to_shaft", 20, 150, 20, 2),
                         "junction+tip_to_shaft|nms=20|gap=20|k=2|r=150")

    def test_split_digest_is_unchanged_for_existing_declarations(self):
        rule = split_rule("tip_to_any_node", 80.0, 2)
        expected = hashlib.sha256(json.dumps(
            {"config_id": "tip_to_any_node|r=80|k=2", "mode": "tip_to_any_node", "radius_um": 80.0,
             "per_anchor_k": 2, "global_cap": None}, sort_keys=True).encode()).hexdigest()
        self.assertEqual(cr.rule_digest("split", rule), expected)

    def test_invalid_rules_are_rejected(self):
        bad = [
            ("split", {**split_rule("tip_to_any_node", 80.0, 2), "config_id": "tip_to_any_node|r=50|k=2"}),
            ("split", {**split_rule("tip_to_any_node", 80.0, 2), "global_cap": 10}),
            ("split", {**split_rule("tip_to_any_node", 80.0, 2), "mode": "nearest"}),
            ("merge", {**merge_rule(), "tip_gap_um": 20.0}),
            ("merge", merge_rule(gap=20.0, k=0)),
            ("merge", merge_rule(label=200.0)),
            ("merge", {**merge_rule(gap=20.0, k=2), "extra": 1}),
        ]
        for kind, rule in bad:
            with self.subTest(rule=rule), self.assertRaises(ValueError):
                cr.validate_rule(kind, rule)

    def test_normalisation(self):
        rule = cr.validate_rule("merge", {**merge_rule(gap=20.0, k=2), "nms_um": 20, "tip_partner_k": 2.0})
        self.assertEqual((rule["nms_um"], rule["tip_partner_k"]), (20.0, 2))
        self.assertIsInstance(rule["nms_um"], float)

    def test_descriptions(self):
        split_text = cr.describe_rule("split", split_rule("tip_to_any_node", 80.0, 2))
        self.assertIn("the 2 nearest partner segments are kept (tip_to_any_node, r=80, k=2)", split_text)
        merge_text = cr.describe_rule("merge", merge_rule(gap=20.0, k=2))
        self.assertIn("within 20 um of the tip", merge_text)
        self.assertIn("each of its 2 nearest other segments", merge_text)
        self.assertNotIn("tip-to-shaft", cr.describe_rule("merge", merge_rule()))


if __name__ == "__main__":
    unittest.main()
