"""The benchmark pool declaration (candidate_pools.json) and the shared rule module it drives."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from proofreader_evolve.harness import candidate_pools as cp
from proofreader_evolve.harness import label_availability as la

MERGE = {"config_id": "junction+tip_to_shaft|nms=20|gap=20|k=2|r=150", "mode": "junction+tip_to_shaft",
         "nms_um": 20.0, "tip_gap_um": 20.0, "tip_partner_k": 2, "claim_radius_um": 150.0,
         "positive_label_radius_um": 20.0}
SPLIT = {"config_id": "tip_to_any_node|r=80|k=2", "mode": "tip_to_any_node", "radius_um": 80.0,
         "per_anchor_k": 2, "global_cap": None}


class CandidatePoolDeclarationTests(unittest.TestCase):
    def test_shipped_declaration_names_both_rules(self):
        pools = cp.load_pool_policies()
        self.assertEqual(pools, {"merge": MERGE, "split": SPLIT})
        raw = json.loads(cp.POLICY_PATH.read_text())
        self.assertEqual(raw["schema_version"], cp.SCHEMA_VERSION)
        for kind in cp.KINDS:
            self.assertEqual(raw["pools"][kind]["evidence"]["datasets"], 11)

    def test_merge_claim_radius_is_the_availability_radius(self):
        self.assertEqual(cp.load_pool_policies()["merge"]["claim_radius_um"], la.NEAR_GT_UM)

    def test_validation_rejects_inconsistent_or_unsupported_entries(self):
        good = {"config_id": "tip_to_tip|r=35|k=all", "mode": "tip_to_tip", "radius_um": 35, "per_anchor_k": None,
                "global_cap": None, "evidence": {"note": "ignored"}}
        self.assertEqual(cp.validate_policy("split", good)["config_id"], "tip_to_tip|r=35|k=all")
        for bad in ({**good, "config_id": "tip_to_tip|r=36|k=all"}, {**good, "mode": "junction"},
                    {**good, "radius_um": 0}, {**good, "per_anchor_k": 0}, {**good, "global_cap": 10}):
            with self.assertRaises(ValueError):
                cp.validate_policy("split", bad)
        with self.assertRaises(ValueError):
            cp.validate_policy("merge", good)
        for bad in ({**MERGE, "tip_gap_um": None}, {**MERGE, "config_id": "junction|nms=20|r=150"},
                    {**MERGE, "positive_label_radius_um": 200.0}):
            with self.assertRaises(ValueError):
                cp.validate_policy("merge", bad)
        junction = {**MERGE, "config_id": "junction|nms=20|r=150", "mode": "junction", "tip_gap_um": None,
                    "tip_partner_k": None}
        self.assertEqual(cp.validate_policy("merge", junction)["mode"], "junction")

    def test_declaration_file_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pools.json"
            path.write_text(json.dumps({"schema_version": 2, "pools": {"split": SPLIT}}))
            with self.assertRaisesRegex(ValueError, "exactly"):
                cp.load_pool_policies(path)
            path.write_text(json.dumps({"schema_version": 2, "pools": {"merge": None, "split": SPLIT}}))
            with self.assertRaisesRegex(ValueError, "must be declared"):
                cp.load_pool_policies(path)
            path.write_text(json.dumps({"schema_version": 1, "pools": {"merge": MERGE, "split": SPLIT}}))
            with self.assertRaisesRegex(ValueError, "schema_version"):
                cp.load_pool_policies(path)
            path.write_text(json.dumps({"schema_version": 2, "pools": {"merge": MERGE, "split": SPLIT}}))
            self.assertEqual(cp.load_pool_policies(path), {"merge": MERGE, "split": SPLIT})

    def test_digest_ignores_evidence_and_keeps_the_split_value(self):
        policy = cp.load_pool_policies()["split"]
        self.assertEqual(cp.policy_digest("split", policy), cp.policy_digest("split", {**policy, "evidence": {"x": 1}}))
        # the 2026-10-15 tables recorded sha256(json.dumps(fields, sort_keys=True)); the split value is unchanged
        legacy = hashlib.sha256(json.dumps(SPLIT, sort_keys=True).encode()).hexdigest()
        self.assertEqual(cp.policy_digest("split", policy), legacy)
        self.assertNotEqual(cp.policy_digest("split", policy), cp.policy_digest(
            "split", {**policy, "radius_um": 50.0, "config_id": "tip_to_any_node|r=50|k=2"}))
        self.assertNotEqual(cp.policy_digest("merge", MERGE), cp.policy_digest(
            "merge", {**MERGE, "tip_gap_um": 10.0, "config_id": "junction+tip_to_shaft|nms=20|gap=10|k=2|r=150"}))

    def test_rule_text_follows_the_policy(self):
        text = cp.describe("split", SPLIT)
        self.assertIn("within 80 um; the 2 nearest partner segments are kept (tip_to_any_node, r=80, k=2)", text)
        self.assertIn("every partner segment is kept", cp.describe(
            "split", {**SPLIT, "mode": "tip_to_tip", "radius_um": 35.0, "per_anchor_k": None,
                      "config_id": "tip_to_tip|r=35|k=all"}))
        merge = cp.describe("merge", MERGE)
        self.assertIn("tip-to-shaft candidates", merge)
        self.assertIn("within 20 um of the tip", merge)

    def test_the_rule_module_is_the_repository_file(self):
        module = cp.rules()
        self.assertEqual(Path(module.__file__).resolve(), cp.RULES_PATH.resolve())
        self.assertIs(cp.rules(), module)
        self.assertEqual(cp.rules_sha256(), hashlib.sha256(cp.RULES_PATH.read_bytes()).hexdigest())


if __name__ == "__main__":
    unittest.main()
