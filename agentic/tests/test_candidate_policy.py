"""The build freezes the declared candidate rule (candidate_pools.json + candidate_rules.py).

Since 2026-10-16 a detector build no longer selects its candidate rule (the split
advice stage and the merge sweep floor are gone); these tests cover freezing,
staleness, the runtime loader and copying candidate_rules.py beside a detector.
Merge-specific cases live in test_merge_site_target.py.
"""
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from agentic.detector_build.candidate_policy import (
    ARTIFACT_TYPE,
    DEFAULT_DECLARATION_PATH,
    compile_declared_candidate_policy,
    declared_rule,
    kind_of_target,
    load_runtime_candidate_policy,
    load_runtime_policy,
    validate_frozen_declared_candidate_policy,
    vendor_candidate_rules,
)
from agentic.detector_build.contracts import CANDIDATE_POLICY_NAME, DetectorTarget
from agentic.tests.candidate_rule_fixtures import (
    RULES_PATH,
    rules_sha256,
    split_rule,
    write_declaration,
)


class DeclaredCandidatePolicyTests(unittest.TestCase):
    def test_repository_declaration_gives_both_rules(self) -> None:
        split = declared_rule("split", DEFAULT_DECLARATION_PATH, RULES_PATH)["rule"]
        merge = declared_rule("merge", DEFAULT_DECLARATION_PATH, RULES_PATH)["rule"]
        self.assertEqual(split["config_id"], "tip_to_any_node|r=80|k=2")
        self.assertEqual(merge["config_id"], "junction+tip_to_shaft|nms=20|gap=20|k=2|r=150")
        self.assertEqual(kind_of_target(DetectorTarget.SPLIT), "split")
        self.assertEqual(kind_of_target(DetectorTarget.MERGE_SITE), "merge")
        with self.assertRaises(SystemExit):
            kind_of_target(DetectorTarget.MERGE)

    def test_split_rule_is_frozen_validated_and_loaded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rule = split_rule("tip_to_any_node", 80.0, 2)
            declaration = write_declaration(root / "candidate_pools.json", split=rule)
            output = root / "app" / CANDIDATE_POLICY_NAME
            artifact = compile_declared_candidate_policy(
                "split", output, project_root=root, declaration_path=declaration, rules_path=RULES_PATH)
            self.assertEqual(artifact["artifact_type"], ARTIFACT_TYPE)
            self.assertEqual(artifact["selected_policy"], rule)
            self.assertEqual(artifact["declaration"]["sha256"],
                             hashlib.sha256(declaration.read_bytes()).hexdigest())
            self.assertEqual(json.loads(output.read_text()), artifact)
            self.assertEqual(validate_frozen_declared_candidate_policy(
                output, "split", project_root=root, declaration_path=declaration, rules_path=RULES_PATH),
                artifact)
            loaded, sha = load_runtime_candidate_policy(output)
            self.assertEqual(loaded, rule)
            self.assertEqual(sha, hashlib.sha256(output.read_bytes()).hexdigest())
            self.assertEqual(load_runtime_policy(output, "split")[2], rules_sha256())

            # a later edit of the declaration makes the frozen copy stale
            write_declaration(declaration, split=split_rule("tip_to_any_node", 50.0, 2))
            with self.assertRaisesRegex(SystemExit, "stale"):
                validate_frozen_declared_candidate_policy(
                    output, "split", project_root=root, declaration_path=declaration, rules_path=RULES_PATH)

    def test_invalid_declarations_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            declaration = write_declaration(root / "candidate_pools.json")
            for change in ("schema", "missing", "inconsistent", "null"):
                data = json.loads(declaration.read_text())
                if change == "schema":
                    data["schema_version"] = 1
                elif change == "missing":
                    del data["pools"]["split"]["per_anchor_k"]
                elif change == "inconsistent":
                    data["pools"]["split"]["radius_um"] = 50.0
                else:
                    data["pools"]["split"] = None
                broken = root / f"{change}.json"
                broken.write_text(json.dumps(data))
                with self.subTest(change=change), self.assertRaises(SystemExit):
                    declared_rule("split", broken, RULES_PATH)

    def test_tampered_artifact_is_rejected_by_the_runtime_loader(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            declaration = write_declaration(root / "candidate_pools.json")
            output = root / CANDIDATE_POLICY_NAME
            compile_declared_candidate_policy(
                "split", output, project_root=root, declaration_path=declaration, rules_path=RULES_PATH)
            artifact = json.loads(output.read_text())
            artifact["selected_policy"]["radius_um"] = 20.0
            output.write_text(json.dumps(artifact))
            with self.assertRaisesRegex(SystemExit, "invalid rule"):
                load_runtime_candidate_policy(output)
            artifact["selected_policy"]["radius_um"] = 80.0
            artifact["kind"] = "merge"
            output.write_text(json.dumps(artifact))
            with self.assertRaisesRegex(SystemExit, "'merge' rule, not 'split'"):
                load_runtime_candidate_policy(output)
            artifact["kind"] = "split"
            artifact["candidate_rules"]["sha256"] = "not-a-hash"
            output.write_text(json.dumps(artifact))
            with self.assertRaisesRegex(SystemExit, "lacks the SHA-256"):
                load_runtime_candidate_policy(output)

    def test_vendored_module_is_byte_identical_and_checked(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            declaration = write_declaration(root / "candidate_pools.json")
            output = root / "app" / CANDIDATE_POLICY_NAME
            compile_declared_candidate_policy(
                "split", output, project_root=root, declaration_path=declaration, rules_path=RULES_PATH)
            copy = vendor_candidate_rules(output, output.parent, rules_path=RULES_PATH)
            self.assertEqual(copy.read_bytes(), RULES_PATH.read_bytes())
            self.assertEqual(vendor_candidate_rules(output, output.parent, rules_path=RULES_PATH), copy)
            edited = root / "candidate_rules.py"
            edited.write_text(RULES_PATH.read_text() + "\n# edited\n")
            with self.assertRaisesRegex(SystemExit, "changed since"):
                vendor_candidate_rules(output, root / "other", rules_path=edited)


if __name__ == "__main__":
    unittest.main()
