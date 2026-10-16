"""The benchmark pool declaration (candidate_pools.json) and its application to a detector runtime."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from proofreader_evolve.harness import candidate_pools as cp


def fake_runtime(**overrides):
    module = SimpleNamespace(CANDIDATE_PAIRING_RULE="tip_to_any_node", CANDIDATE_MAX_DISTANCE_UM=50.0,
                             CANDIDATE_PER_ANCHOR_K=2, CANDIDATE_GLOBAL_CAP=None,
                             CANDIDATE_POLICY_CONFIG_ID="tip_to_any_node|r=50|k=2",
                             CANDIDATE_POLICY_SHA256="c74f" + "0" * 60)
    module.build_sample_universe = lambda payload: ([{"radius": module.CANDIDATE_MAX_DISTANCE_UM,
                                                      "k": module.CANDIDATE_PER_ANCHOR_K,
                                                      "id": module.CANDIDATE_POLICY_CONFIG_ID}], [0])
    for key, value in overrides.items():
        setattr(module, key, value)
    return module


class CandidatePoolDeclarationTests(unittest.TestCase):
    def test_shipped_declaration_is_the_eleven_brain_split_rule(self):
        pools = cp.load_pool_policies()
        self.assertIsNone(pools["merge"])
        self.assertEqual(pools["split"], {"config_id": "tip_to_any_node|r=80|k=2", "mode": "tip_to_any_node",
                                          "radius_um": 80.0, "per_anchor_k": 2, "global_cap": None})
        raw = json.loads(cp.POLICY_PATH.read_text())
        self.assertEqual(raw["pools"]["split"]["evidence"]["datasets"], 11)

    def test_validation_rejects_inconsistent_or_unsupported_entries(self):
        good = {"config_id": "tip_to_tip|r=35|k=all", "mode": "tip_to_tip", "radius_um": 35, "per_anchor_k": None, "global_cap": None}
        self.assertEqual(cp.validate_policy("split", good)["config_id"], "tip_to_tip|r=35|k=all")
        for bad in ({**good, "config_id": "tip_to_tip|r=36|k=all"}, {**good, "mode": "junction"},
                    {**good, "radius_um": 0}, {**good, "per_anchor_k": 0}, {**good, "global_cap": 10}):
            with self.assertRaises(ValueError):
                cp.validate_policy("split", bad)
        with self.assertRaises(ValueError):
            cp.validate_policy("merge", good)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pools.json"
            path.write_text(json.dumps({"schema_version": 1, "pools": {"split": None}}))
            with self.assertRaisesRegex(ValueError, "exactly"):
                cp.load_pool_policies(path)
            path.write_text(json.dumps({"schema_version": 1, "pools": {"merge": None, "split": None}}))
            self.assertEqual(cp.load_pool_policies(path), {"merge": None, "split": None})

    def test_config_id_round_trip_and_digest_ignore_evidence(self):
        self.assertEqual(cp.parse_config_id("tip_to_any_node|r=80|k=2"), {"mode": "tip_to_any_node", "radius_um": 80.0, "per_anchor_k": 2})
        self.assertEqual(cp.parse_config_id("tip_to_tip|r=12.5|k=all")["per_anchor_k"], None)
        with self.assertRaises(ValueError):
            cp.parse_config_id("junction|nms=20|r=150")
        policy = cp.load_pool_policies()["split"]
        self.assertEqual(cp.policy_digest(policy), cp.policy_digest({**policy, "evidence": {"x": 1}}))
        self.assertNotEqual(cp.policy_digest(policy), cp.policy_digest({**policy, "radius_um": 50.0, "config_id": "tip_to_any_node|r=50|k=2"}))

    def test_apply_pool_policy_sets_the_runtime_rule_and_restores_it(self):
        module = fake_runtime()
        policy = cp.load_pool_policies()["split"]
        with cp.apply_pool_policy(module, policy):
            rows, _ = module.build_sample_universe({})
        self.assertEqual(rows, [{"radius": 80.0, "k": 2, "id": "tip_to_any_node|r=80|k=2"}])
        self.assertEqual((module.CANDIDATE_MAX_DISTANCE_UM, module.CANDIDATE_POLICY_CONFIG_ID), (50.0, "tip_to_any_node|r=50|k=2"))
        self.assertTrue(module.CANDIDATE_POLICY_SHA256.startswith("c74f"))  # the detector's own hash is never rewritten
        with self.assertRaises(ValueError):
            with cp.apply_pool_policy(SimpleNamespace(build_sample_universe=lambda p: None), policy):
                pass

    def test_split_rule_text_follows_the_policy(self):
        text = cp.describe_split_rule("tip_to_any_node|r=80|k=2")
        self.assertIn("within 80 um; the 2 nearest partner segments are kept (tip_to_any_node, r=80, k=2)", text)
        self.assertIn("every partner segment is kept", cp.describe_split_rule("tip_to_tip|r=35|k=all"))


if __name__ == "__main__":
    unittest.main()
