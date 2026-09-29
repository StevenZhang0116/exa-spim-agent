from __future__ import annotations

import ast
import hashlib
import inspect
import json
import sys
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import networkx as nx
import numpy as np

from agentic.detector_build.candidate_policy import (
    compile_merge_candidate_policy,
    load_runtime_merge_candidate_policy,
    validate_frozen_merge_candidate_policy,
    validate_merge_candidate_policy_sources,
)
from agentic.detector_build.contracts import (
    MERGE_CANDIDATE_POLICY_NAME,
    DetectorTarget,
    build_artifact_names,
    target_spec,
)
from agentic.detector_build.target_runtime import target_adapter_source
from agentic import run_detector_build_workflow as workflow


MERGE_SITE_POLICY = {
    "config_id": "junction|nms=10|r=15",
    "mode": "junction",
    "nms_um": 10,
    "claim_radius_um": 15,
    "positive_label_radius_um": 4,
}
MERGE_SITE_POLICY_SHA256 = "b" * 64

AGGREGATE_HEADER = (
    "config_id,mode,nms_um,claim_radius_um,total_candidates,"
    "min_site_recall,mean_site_recall,min_label_recall,mean_label_recall,"
    "n_brains,micro_site_recall\n"
)
AGGREGATE_ROWS = (
    "junction|nms=0|r=15,junction,0.0,15.0,100,0.85,0.9,0.8,0.85,2,0.9\n"
    "junction|nms=20|r=15,junction,20.0,15.0,60,0.80,0.86,0.75,0.8,2,0.86\n"
    "junction|nms=20|r=10,junction,20.0,10.0,60,0.70,0.78,0.7,0.75,2,0.79\n"
)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_sweep_bundle(root: Path) -> Path:
    tables = root / "tables"
    tables.mkdir(parents=True)
    aggregate = tables / "cross_dataset_aggregate.csv"
    aggregate.write_text(AGGREGATE_HEADER + AGGREGATE_ROWS, encoding="utf-8")
    source_policy = root / "recommended_policy.json"
    source_policy.write_text(json.dumps({
        "artifact_type": "merge_candidate_policy_selection",
        "evidence_scope": {
            "mcl": 100,
            "config_signature": "deadbeefdeadbeef",
            "brains": ["794495", "794493"],
            "segment_detector_gating_evaluated": False,
            "bridge_candidate_family_evaluated": False,
        },
        "structural_ceilings": {"note": "test", "per_brain": []},
        "provenance": {
            "aggregate_sha256": _sha256_bytes(aggregate.read_bytes()),
        },
    }, indent=2) + "\n", encoding="utf-8")
    manifest = root / "artifact_manifest.json"
    manifest.write_text(json.dumps({"artifacts": [
        {
            "path": "recommended_policy.json",
            "sha256": _sha256_bytes(source_policy.read_bytes()),
            "bytes": source_policy.stat().st_size,
        },
        {
            "path": "tables/cross_dataset_aggregate.csv",
            "sha256": _sha256_bytes(aggregate.read_bytes()),
            "bytes": aggregate.stat().st_size,
        },
    ]}, indent=2) + "\n", encoding="utf-8")
    entries = json.loads(manifest.read_text())["artifacts"]
    (root / "per_brain").mkdir()
    (root / "cache").mkdir()
    for brain in ("794495", "794493"):
        cache = root / "cache" / f"dataset_cache_{brain}_mcl100_add.pkl"
        cache.write_bytes(b"synthetic cache identity")
        stat = cache.stat()
        metadata = root / "per_brain" / f"{brain}_meta.json"
        metadata.write_text(json.dumps({
            "brain": brain, "config_signature": "deadbeefdeadbeef", "cache": str(cache.resolve()),
            "cache_identity": {"path": str(cache.resolve()), "size": stat.st_size,
                               "mtime_ns": stat.st_mtime_ns, "inode": stat.st_ino},
        }))
        entries.append({"path": f"per_brain/{brain}_meta.json", "sha256": _sha256_bytes(metadata.read_bytes()),
                        "bytes": metadata.stat().st_size})
    manifest.write_text(json.dumps({"artifacts": entries}))
    return source_policy


class _Graph(nx.Graph):
    pass


def _merge_site_payload() -> dict:
    graph = _Graph()
    graph.add_edges_from([
        (0, 1), (1, 2), (2, 3), (2, 4), (3, 5), (3, 6),
        (7, 8), (8, 9), (8, 10),
    ])
    graph.node_xyz = np.asarray([
        [0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [10.0, 0.0, 0.0],
        [15.0, 0.0, 0.0], [10.0, 5.0, 0.0], [20.0, 0.0, 0.0],
        [15.0, 5.0, 0.0],
        [100.0, 0.0, 0.0], [105.0, 0.0, 0.0], [110.0, 0.0, 0.0],
        [105.0, 5.0, 0.0],
    ])
    graph.node_component_id = np.asarray([10] * 7 + [20] * 4)
    graph.component_id_to_swc_id = {10: "111.0.swc", 20: "222.0.swc"}
    return {
        "fragments_graph": graph,
        "gt_merge_sites": [
            {"segment_id": 111, "xyz": (10.0, 0.0, 0.0)},   # on junction 2
            {"segment_id": 111, "xyz": (20.0, 0.0, 0.0)},   # 10 um to node 2
            {"segment_id": 222, "xyz": (110.0, 0.0, 0.0)},  # 5 um to node 8
            {"segment_id": 111, "xyz": (999.0, 999.0, 999.0)},  # unsnappable
        ],
        "gt_merge_labels": np.asarray([111]),
        "gt_node_canonical_label": np.asarray([111, 222, 0]),
    }


def _adapter_namespace() -> dict:
    namespace = {"np": np, "defaultdict": defaultdict}

    def build_comp_to_seg(graph):
        return {
            int(component): int(str(swc).split(".")[0])
            for component, swc in graph.component_id_to_swc_id.items()
        }

    namespace["build_comp_to_seg"] = build_comp_to_seg
    exec(target_adapter_source(
        DetectorTarget.MERGE_SITE,
        candidate_policy=MERGE_SITE_POLICY,
        candidate_policy_sha256=MERGE_SITE_POLICY_SHA256,
    ), namespace)
    return namespace


def _runtime_namespace() -> dict:
    namespace = _adapter_namespace()
    namespace["inspect"] = inspect
    namespace["MERGE_FEATURE_SCOPES"] = {"x": "candidate"}
    storage = (Path(__file__).resolve().parents[1] / "detector_build"
               / "templates" / "merge_feature_accumulator.py.tmpl")
    exec(storage.read_text(), namespace)
    template = (Path(__file__).resolve().parents[1] / "detector_build"
                / "templates" / "detector_runtime.py.tmpl").read_text()
    start = template.index("# --- blindness guard")
    end = template.index("# --- end blindness guard ---")
    exec(template[start:end], namespace)
    wrapper = next(
        node for node in ast.parse(template).body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_extract_features_runtime"
    )
    exec(compile(ast.Module(body=[wrapper], type_ignores=[]),
                 "detector_runtime.py.tmpl", "exec"), namespace)
    return namespace


class MergeSiteContractTests(unittest.TestCase):
    def test_junction_audit_is_not_a_feature_or_automatic_label(self):
        from agentic.detector_build.assembly import FEATURE_FORBIDDEN_PAYLOAD_KEYS

        namespace = _runtime_namespace()
        payload = _merge_site_payload()
        samples, expected_labels = namespace["build_sample_universe"](payload)
        payload["gt_junction_audit"] = {
            "schema_version": 1, "audit_only": True,
            "junctions": [{"node_id": 8, "status": "merge_supported"}],
        }
        del payload["__detector_sample_universe_cache__"]
        _, labels = namespace["build_sample_universe"](payload)
        np.testing.assert_array_equal(labels, expected_labels)
        blind = namespace["_BlindPayloadView"](payload, samples)
        for key in ("gt_junction_audit", "gt_merge_site_metadata"):
            self.assertIn(key, FEATURE_FORBIDDEN_PAYLOAD_KEYS)
            self.assertNotIn(key, blind)
            with self.assertRaises(namespace["BlindnessViolation"]):
                blind[key]

    def test_spec_names_do_not_collide_with_segment_merge(self) -> None:
        site = target_spec(DetectorTarget.MERGE_SITE)
        merge = target_spec(DetectorTarget.MERGE)
        self.assertEqual(site.detector_name, "merge_junction_detector.py")
        self.assertEqual(merge.detector_name, "merge_site_detector.py")
        self.assertEqual(site.label_name, "is_merge_site")
        self.assertEqual(site.score_prefix, "merge_site_probability")
        self.assertIn(
            MERGE_CANDIDATE_POLICY_NAME,
            build_artifact_names(DetectorTarget.MERGE_SITE),
        )
        self.assertNotIn(
            MERGE_CANDIDATE_POLICY_NAME,
            build_artifact_names(DetectorTarget.MERGE),
        )

    def test_adapter_requires_policy_and_merge_rejects_it(self) -> None:
        with self.assertRaises(ValueError):
            target_adapter_source(DetectorTarget.MERGE_SITE)
        with self.assertRaises(ValueError):
            target_adapter_source(
                DetectorTarget.MERGE,
                candidate_policy=MERGE_SITE_POLICY,
                candidate_policy_sha256=MERGE_SITE_POLICY_SHA256,
            )


class MergeCandidatePolicyTests(unittest.TestCase):
    def test_stale_cache_aborts_workflow_before_output_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = _write_sweep_bundle(root / "bundle")
            (source.parent / "cache/dataset_cache_794495_mcl100_add.pkl").write_bytes(b"refreshed")
            run = root / "merge-error-794495-mcl100_2026-08-04.json"
            run.write_text("{}")
            output = root / "existing-experiment"
            output.mkdir()
            baseline = output / "baseline.txt"
            baseline.write_text("preserve")
            argv = ["run_detector_build_workflow.py", str(run), "--out-dir", str(output),
                    "--merge-candidate-policy", str(source), "--yes"]
            with patch.object(workflow, "PROJECT_ROOT", root), patch.object(sys, "argv", argv), \
                    patch.object(workflow, "resolve_run_context", return_value=SimpleNamespace(target=DetectorTarget.MERGE)), \
                    patch.object(workflow, "confirm_and_clean") as cleanup, \
                    self.assertRaisesRegex(SystemExit, "cache identity is stale"):
                workflow.main()
            cleanup.assert_not_called()
            self.assertEqual(baseline.read_text(), "preserve")

    def test_cache_freshness_and_metadata_are_validated(self):
        for change in ("replaced_cache", "missing_cache", "missing_meta", "tampered_meta", "legacy_meta"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source = _write_sweep_bundle(root / "bundle")
                cache = source.parent / "cache" / "dataset_cache_794495_mcl100_add.pkl"
                meta = source.parent / "per_brain" / "794495_meta.json"
                if change == "replaced_cache":
                    replacement = cache.with_suffix(".tmp")
                    replacement.write_bytes(cache.read_bytes())
                    replacement.replace(cache)
                elif change == "missing_cache":
                    cache.unlink()
                elif change == "missing_meta":
                    meta.unlink()
                else:
                    data = json.loads(meta.read_text())
                    data.pop("cache_identity")
                    meta.write_text(json.dumps(data))
                    if change == "legacy_meta":
                        manifest = source.parent / "artifact_manifest.json"
                        data = json.loads(manifest.read_text())
                        for entry in data["artifacts"]:
                            if entry["path"] == "per_brain/794495_meta.json":
                                entry["sha256"] = _sha256_bytes(meta.read_bytes())
                        manifest.write_text(json.dumps(data))
                with self.assertRaises(SystemExit):
                    validate_merge_candidate_policy_sources(source, minimum_site_recall=.8,
                                                            expected_mcl=100, project_root=root)

    def test_compile_validate_and_tamper_detection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_policy = _write_sweep_bundle(root / "bundle")
            output = root / "out" / MERGE_CANDIDATE_POLICY_NAME

            report = validate_merge_candidate_policy_sources(
                source_policy, minimum_site_recall=0.80,
                expected_mcl=100, project_root=root,
            )
            self.assertEqual(
                report["selected"]["config_id"], "junction|nms=20|r=15")

            policy = compile_merge_candidate_policy(
                output, source_policy,
                minimum_site_recall=0.80, positive_label_radius_um=10.0,
                expected_mcl=100, project_root=root,
            )
            self.assertEqual(
                policy["selected_policy"]["config_id"], "junction|nms=20|r=15")
            self.assertEqual(
                policy["selected_policy"]["positive_label_radius_um"], 10.0)
            self.assertEqual(policy["evidence_metrics"]["total_candidates"], 60)
            self.assertEqual(policy["evidence_scope"]["mcl"], 100)
            self.assertEqual(len(policy["provenance"]["cache_evidence"]), 2)
            self.assertFalse(
                policy["evidence_scope"]["bridge_candidate_family_evaluated"])

            validate_frozen_merge_candidate_policy(
                output, source_policy,
                minimum_site_recall=0.80, positive_label_radius_um=10.0,
                expected_mcl=100, project_root=root,
            )
            runtime, sha = load_runtime_merge_candidate_policy(output)
            self.assertEqual(runtime["nms_um"], 20)
            self.assertEqual(runtime["claim_radius_um"], 15)
            self.assertEqual(len(sha), 64)

            with self.assertRaises(SystemExit):
                validate_frozen_merge_candidate_policy(
                    output, source_policy,
                    minimum_site_recall=0.80, positive_label_radius_um=12.0,
                    expected_mcl=100, project_root=root,
                )
            with self.assertRaises(SystemExit):
                compile_merge_candidate_policy(
                    output, source_policy,
                    minimum_site_recall=0.80, positive_label_radius_um=20.0,
                    expected_mcl=100, project_root=root,
                )
            with self.assertRaises(SystemExit):
                compile_merge_candidate_policy(
                    output, source_policy,
                    minimum_site_recall=0.80, positive_label_radius_um=10.0,
                    expected_mcl=10, project_root=root,
                )
            with self.assertRaises(SystemExit):
                compile_merge_candidate_policy(
                    output, source_policy,
                    minimum_site_recall=0.99, positive_label_radius_um=10.0,
                    expected_mcl=100, project_root=root,
                )
            aggregate = source_policy.parent / "tables" / "cross_dataset_aggregate.csv"
            aggregate.write_text(
                AGGREGATE_HEADER + AGGREGATE_ROWS.replace("60", "10"),
                encoding="utf-8")
            with self.assertRaises(SystemExit):
                compile_merge_candidate_policy(
                    output, source_policy,
                    minimum_site_recall=0.80, positive_label_radius_um=10.0,
                    expected_mcl=100, project_root=root,
                )


class MergeSiteAdapterTests(unittest.TestCase):
    def test_combined_sites_add_positive_without_changing_candidates_or_legacy_positive(self):
        namespace = _runtime_namespace()
        payload = _merge_site_payload()
        old_samples, old_labels = namespace["build_sample_universe"](payload)
        payload.pop("__detector_sample_universe_cache__")
        payload["gt_merge_sites"].append({
            "segment_id": 222, "xyz": [105., 0., 0.], "source": "two_gt_junction",
            "gt_neuron": "A", "gt_neurons": ["A", "B"],
        })
        payload["gt_merge_site_metadata"] = {"method": "junction_gt_branch_support_v1",
                                             "parameters": {"match_radius_um": 6.}}
        samples, labels = namespace["build_sample_universe"](payload)
        self.assertEqual([row["node_id"] for row in samples], [row["node_id"] for row in old_samples])
        self.assertEqual(old_labels.tolist(), [1, 0])
        self.assertEqual(labels.tolist(), [1, 1])
        provenance = namespace["sample_universe_audit"](payload, samples, labels)["site_provenance"]
        self.assertEqual(provenance["source_counts"], {
            "geometric_only": 4, "two_gt_only": 1, "shared": 0, "untagged_legacy": 4})
        self.assertEqual(provenance["n_sites"], 5)
        self.assertEqual(provenance["generation_metadata"]["parameters"]["match_radius_um"], 6.)
        blind = namespace["_BlindPayloadView"](payload, samples)
        with self.assertRaises(namespace["BlindnessViolation"]):
            namespace["_merge_site_provenance"](blind)
        payload["gt_merge_site_metadata"]["parameters"]["match_radius_um"] = 99.
        self.assertEqual(provenance["generation_metadata"]["parameters"]["match_radius_um"], 6.)

    def test_site_provenance_hashes_current_content_and_handles_shared_and_missing(self):
        namespace = _adapter_namespace()
        payload = _merge_site_payload()
        payload["gt_merge_sites"][0].update(source="geometric_walk",
            sources=["geometric_walk", "two_gt_junction"])
        first = namespace["_merge_site_provenance"](payload)
        self.assertEqual(first["source_counts"]["shared"], 1)
        payload["gt_merge_sites"].reverse()
        self.assertEqual(namespace["_merge_site_provenance"](payload), first)
        payload["gt_merge_sites"][0]["xyz"] = [998., 999., 999.]
        self.assertNotEqual(namespace["_merge_site_provenance"](payload)["site_records_sha256"],
                            first["site_records_sha256"])
        empty = namespace["_merge_site_provenance"]({"gt_merge_sites": []})
        missing = namespace["_merge_site_provenance"]({})
        self.assertTrue(empty["sites_available"])
        self.assertFalse(missing["sites_available"])
        self.assertIsNone(missing["site_records_sha256"])
        self.assertIsNotNone(empty["site_records_sha256"])

    def test_extraction_restores_audit_rows_without_exposing_gt(self) -> None:
        namespace = _runtime_namespace()
        payload = _merge_site_payload()
        expected_samples, expected_labels = namespace["build_sample_universe"](payload)
        accumulator = namespace["FeatureAccumulator"](expected_samples)

        def extract_features(blind, **kwargs):
            samples, labels = namespace["build_sample_universe"](blind)
            self.assertFalse(np.any(labels))
            for sample in samples:
                self.assertNotIn("in_ambiguous_ring", sample)
                self.assertNotIn("distance_to_nearest_gt_site_um", sample)
            with self.assertRaises(namespace["BlindnessViolation"]):
                blind["gt_merge_sites"]
            return samples, labels, accumulator

        namespace["extract_features"] = extract_features
        samples, labels, returned = namespace["_extract_features_runtime"](payload)

        self.assertEqual(samples, expected_samples)
        np.testing.assert_array_equal(labels, expected_labels)
        self.assertIs(returned, accumulator)
        audit = namespace["sample_universe_audit"](payload, samples, labels)
        self.assertEqual(audit["n_ambiguous_ring"], 1)
        frame = namespace["sample_output_frame"](samples, labels)
        self.assertEqual(frame["in_ambiguous_ring"].tolist(), [0, 1])

    def test_profile_extraction_restores_subset_audit_by_identity(self) -> None:
        namespace = _runtime_namespace()
        payload = _merge_site_payload()
        expected_samples, expected_labels = namespace["build_sample_universe"](payload)
        accumulator = namespace["FeatureAccumulator"](expected_samples[1:])

        def extract_features(blind, **kwargs):
            self.assertEqual(kwargs["profile_segment_limit"], 1)
            samples, labels = namespace["build_sample_universe"](blind)
            return samples[1:], labels[1:], accumulator

        namespace["extract_features"] = extract_features
        samples, labels, _ = namespace["_extract_features_runtime"](
            payload, profile_segment_limit=1)

        self.assertEqual(samples, expected_samples[1:])
        np.testing.assert_array_equal(labels, expected_labels[1:])
        audit = namespace["sample_universe_audit"](payload, samples, labels)
        self.assertEqual(audit["n_rows"], 1)
        self.assertEqual(audit["n_ambiguous_ring"], 1)

    def test_universe_nms_labels_and_audit(self) -> None:
        namespace = _adapter_namespace()
        payload = _merge_site_payload()

        samples, labels = namespace["build_sample_universe"](payload)

        # Junctions are nodes 2, 3, 8; segment-scoped NMS at 10 um keeps 2
        # (suppressing same-segment 3 at 5 um) and 8 (other segment).
        self.assertEqual([s["node_id"] for s in samples], [2, 8])
        self.assertEqual([s["segment_id"] for s in samples], [111, 222])
        self.assertEqual([s["candidate_id"] for s in samples], [0, 1])
        self.assertEqual([s["degree"] for s in samples], [3, 3])
        # Node 2 sits ON a GT site (distance 0 <= label radius 4) -> positive.
        # Node 8 is 5 um from its site: 4 < 5 <= 15 -> ambiguous ring, negative.
        self.assertEqual(labels.tolist(), [1, 0])
        self.assertEqual(samples[0]["distance_to_nearest_gt_site_um"], 0.0)
        self.assertEqual(samples[0]["in_ambiguous_ring"], 0)
        self.assertEqual(samples[1]["distance_to_nearest_gt_site_um"], 5.0)
        self.assertEqual(samples[1]["in_ambiguous_ring"], 1)
        self.assertEqual(
            (samples[0]["x_um"], samples[0]["y_um"], samples[0]["z_um"]),
            (10.0, 0.0, 0.0),
        )

        audit = namespace["sample_universe_audit"](payload, samples, labels)
        self.assertEqual(audit["candidate_policy_config_id"], "junction|nms=10|r=15")
        self.assertEqual(audit["candidate_policy_sha256"], MERGE_SITE_POLICY_SHA256)
        self.assertEqual(audit["candidate_nms_um"], 10.0)
        self.assertEqual(audit["candidate_claim_radius_um"], 15.0)
        self.assertEqual(audit["candidate_positive_label_radius_um"], 4.0)
        self.assertEqual(audit["n_junctions_raw"], 3)
        self.assertEqual(audit["n_rows"], 2)
        self.assertEqual(audit["n_positive"], 1)
        self.assertEqual(audit["n_negative"], 1)
        self.assertEqual(audit["n_ambiguous_ring"], 1)
        self.assertEqual(audit["n_sites"], 4)
        self.assertEqual(audit["n_sites_snapped"], 3)
        # Site min distances: 0 (on node 2), 10 (node 5 -> node 2), 5
        # (node 9 -> node 8), inf (unsnappable) -> 3 of 4 covered at claim 15.
        self.assertEqual(audit["n_sites_covered_at_claim_radius"], 3)
        self.assertAlmostEqual(audit["site_recall_at_claim_radius"], 0.75)

        frame = namespace["sample_output_frame"](samples, labels)
        self.assertEqual(list(frame.columns), [
            "candidate_id", "node_id", "segment_id", "degree",
            "x_um", "y_um", "z_um",
            "distance_to_nearest_gt_site_um", "in_ambiguous_ring",
            "is_merge_site",
        ])
        self.assertEqual(frame["is_merge_site"].tolist(), [1, 0])

        display = namespace["sample_display"](samples[0])
        self.assertIn("node 2", display)
        self.assertIn("segment 111", display)

        covariates = namespace["ascertainment_covariates"](payload, samples)
        self.assertEqual(covariates["segment_fragment_node_count"].tolist(),
                         [7.0, 4.0])

    def test_enumeration_matches_sweep_implementation(self) -> None:
        """Parity: the runtime adapter and the candidate-pool sweep must keep
        the same junction set under the same NMS parameters."""
        import importlib.util
        import sys

        notebooks = Path(__file__).resolve().parents[2] / "notebooks"
        census_spec = importlib.util.spec_from_file_location(
            "merge_topology_diagnosis", notebooks / "merge_topology_diagnosis.py")
        census = importlib.util.module_from_spec(census_spec)
        sys.modules["merge_topology_diagnosis"] = census
        census_spec.loader.exec_module(census)
        sweep_spec = importlib.util.spec_from_file_location(
            "merge_candidate_pool_sweep", notebooks / "merge_candidate_pool_sweep.py")
        sweep = importlib.util.module_from_spec(sweep_spec)
        sys.modules["merge_candidate_pool_sweep"] = sweep
        sweep_spec.loader.exec_module(sweep)

        payload = _merge_site_payload()
        index = census.build_fragment_index(payload["fragments_graph"])
        junctions = sweep.enumerate_junctions(index)
        kept = sweep.nms_keep_mask(index, junctions, 10.0)

        namespace = _adapter_namespace()
        samples, _ = namespace["build_sample_universe"](payload)
        self.assertEqual(
            sorted(int(n) for n in np.flatnonzero(kept)),
            [s["node_id"] for s in samples],
        )

    def test_detector_assembly_injects_merge_site_adapter(self) -> None:
        from agentic.detector_build.assembly import assemble_detector

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            template = root / "runtime.py.tmpl"
            feature = root / "feature.py"
            output = root / "merge_junction_detector.py"
            policy_path = root / MERGE_CANDIDATE_POLICY_NAME
            inventory_path = root / "feature_inventory.json"
            inventory_path.write_text(json.dumps({
                "schema_version": 4,
                "hypotheses": [{"included": True, "features": [{
                    "name": "degree_feature", "scope": "candidate",
                    "source_feature": "degree_feature", "source_aggregation": "node degree",
                    "adaptation": "identity",
                }]}],
            }))
            policy_path.write_text(json.dumps({
                "schema_version": 1,
                "artifact_type": "merge_site_candidate_policy_selection",
                "application_status": "selected_for_detector_runtime",
                "selected_policy": dict(MERGE_SITE_POLICY),
            }) + "\n", encoding="utf-8")
            template.write_text(
                "import numpy as np\n"
                "from collections import defaultdict\n"
                "def build_comp_to_seg(graph):\n    return {}\n"
                "# __DETECTOR_TARGET_ADAPTER__\n"
                "# __DETECTOR_FEATURE_IMPLEMENTATION__\n"
            )
            feature.write_text(
                "FEATURE_REGISTRY = [('degree_feature', 'junction_site')]\n"
                "ANALYSIS_TIMING_GROUPS = [{\n"
                "    'key': 'hypo_1', 'hypothesis_ids': [1],\n"
                "    'phase': 'junction_site',\n"
                "    'feature_names': ['degree_feature'],\n"
                "}]\n"
                "COMPUTATION_PLAN = [{\n"
                "    'primitive': 'row_math', 'cost_class': 'constant',\n"
                "    'bound': '', 'amortization': '', 'consumers': ['hypo_1'],\n"
                "}]\n"
                "def extract_features(payload, verbose=True, timing=None, "
                "enabled_analysis_keys=None, profile_segment_limit=None):\n"
                "    _ = profile_segment_limit\n"
                "    def _analysis_enabled(key):\n"
                "        return enabled_analysis_keys is None or key in enabled_analysis_keys\n"
                "    _analysis_enabled('hypo_1')\n"
                "    if timing is not None:\n"
                "        timing.start_phase('junction_site')\n"
                "        timing.start_analysis('hypo_1')\n"
                "    samples, labels = build_sample_universe(payload)\n"
                "    return samples, labels, FeatureAccumulator(samples)\n"
            )

            assemble_detector(
                template, feature, output, target=DetectorTarget.MERGE_SITE,
                candidate_policy_path=policy_path, inventory_path=inventory_path)

            text = output.read_text()
            self.assertIn("DETECTOR_TARGET = 'merge_site_detection'", text)
            self.assertIn("CANDIDATE_MODE = 'junction'", text)
            self.assertIn("CANDIDATE_NMS_UM = 10.0", text)
            self.assertIn("CANDIDATE_CLAIM_RADIUS_UM = 15.0", text)
            self.assertIn("CANDIDATE_POSITIVE_LABEL_RADIUS_UM = 4.0", text)
            self.assertIn(
                hashlib.sha256(policy_path.read_bytes()).hexdigest(), text)
            self.assertIn("def _nms_kept_junctions", text)
            self.assertIn("class FeatureAccumulator", text)

            with self.assertRaisesRegex(
                    SystemExit, "requires merge_candidate_policy"):
                assemble_detector(
                    template, feature, output,
                    target=DetectorTarget.MERGE_SITE)

            feature.write_text(
                feature.read_text().replace(
                    "samples, labels = build_sample_universe(payload)",
                    "_ = samples_row['distance_to_nearest_gt_site_um']\n"
                    "    samples, labels = build_sample_universe(payload)",
                )
            )
            with self.assertRaisesRegex(
                    SystemExit, "GT-only.*distance_to_nearest_gt_site_um"):
                assemble_detector(
                    template, feature, output,
                    target=DetectorTarget.MERGE_SITE,
                    candidate_policy_path=policy_path, inventory_path=inventory_path)

    def test_assembly_rejects_per_edge_graph_rebuilds(self) -> None:
        """Regression: the 2026-09-09 merge-site fragment's 'tree fast path'
        copied the whole component graph and re-ran connected_components for
        every edge — O(E x (V+E)) — turning a bounded per-component feature
        into hours on large tree components."""
        from agentic.detector_build.assembly import (
            _validate_no_per_element_graph_rebuilds,
        )
        import ast as ast_module

        bad = ast_module.parse(
            "def _get_max_ebc(sub):\n"
            "    best = 0.0\n"
            "    for edge in sub.edges():\n"
            "        g2 = nx.Graph(sub)\n"
            "        g2.remove_edge(*edge)\n"
            "        comps = list(nx.connected_components(g2))\n"
            "    return best\n"
        )
        with self.assertRaisesRegex(SystemExit, "per-edge loop"):
            _validate_no_per_element_graph_rebuilds(bad, Path("bad.py"))

        also_bad = ast_module.parse(
            "def f(sub):\n"
            "    for node in sub.nodes():\n"
            "        h = sub.subgraph([node])\n"
        )
        with self.assertRaisesRegex(SystemExit, "per-node loop"):
            _validate_no_per_element_graph_rebuilds(also_bad, Path("bad.py"))

        good = ast_module.parse(
            "def tree_edge_betweenness(sub, root):\n"
            "    subtree = {}\n"
            "    for node in reversed(list(nx.dfs_postorder_nodes(sub, root))):\n"
            "        subtree[node] = 1 + sum(\n"
            "            subtree.get(c, 0) for c in sub.neighbors(node))\n"
            "    best = 0.0\n"
            "    for a, b in sub.edges():\n"
            "        s1 = min(subtree.get(a, 1), subtree.get(b, 1))\n"
            "        best = max(best, s1 * (sub.number_of_nodes() - s1))\n"
            "    return best\n"
        )
        _validate_no_per_element_graph_rebuilds(good, Path("good.py"))

    def test_workflow_builds_merge_site_steps_without_policy_agent_turn(self) -> None:
        steps = workflow.build_steps(
            run_rel="autodiscovery/merge-error-test.json",
            summary_rel="autodiscovery/merge-error-test.summary.md",
            rerun_rel="autodiscovery/merge-error-test.rerun",
            fixed_rel=None,
            selection_rel="autodiscovery/merge-error-test.predictive-selection.json",
            selected_ids=[1],
            out_dir_rel="autodiscovery-application/merge-error-test",
            target=DetectorTarget.MERGE_SITE,
        )
        names = [step["name"] for step in steps]
        self.assertNotIn("select-candidate-policy", names)
        self.assertEqual(names, [
            "inventory-features", "configure-models",
            "generate-detector", "verify-and-document",
        ])
        inventory = steps[0]["instruction"]
        self.assertIn(MERGE_CANDIDATE_POLICY_NAME, inventory)
        self.assertIn("JUNCTION-LOCAL", inventory)
        self.assertIn("SEGMENT-CONTEXT", inventory)
        self.assertIn("two_gt_junction", inventory)
        self.assertIn("not a verified non-merge", inventory)
        self.assertIn("geometric_walk", inventory)
        generate = steps[2]["instruction"]
        self.assertIn("--target merge_site_detection", generate)
        self.assertIn(
            f"--candidate-policy autodiscovery-application/merge-error-test/"
            f"{MERGE_CANDIDATE_POLICY_NAME}", generate)
        self.assertIn("gt_merge_sites", generate)
        self.assertIn("gt_junction_audit", generate)
        self.assertIn("gt_merge_site_metadata", generate)
        verify = steps[3]["instruction"]
        self.assertIn("site_records_sha256", verify)
        self.assertIn("ordered_row_labels_sha256", verify)
        self.assertIn("not an untouched", verify)


if __name__ == "__main__":
    unittest.main()
