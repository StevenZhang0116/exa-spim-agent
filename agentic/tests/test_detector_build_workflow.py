from __future__ import annotations

import asyncio
import importlib.util
import hashlib
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

from agentic.detector_build.assembly import assemble_detector
from agentic.detector_build.contracts import DetectorTarget


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_module():
    sdk = types.ModuleType("claude_agent_sdk")
    for name in (
        "AssistantMessage",
        "ClaudeAgentOptions",
        "ClaudeSDKClient",
        "ResultMessage",
        "TextBlock",
        "ToolUseBlock",
    ):
        setattr(sdk, name, type(name, (), {}))
    sys.modules["claude_agent_sdk"] = sdk
    path = PROJECT_ROOT / "agentic" / "run_detector_build_workflow.py"
    spec = importlib.util.spec_from_file_location("test_detector_workflow", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


workflow = _load_module()


class DetectorBuildWorkflowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.old_root = workflow.PROJECT_ROOT

    def tearDown(self) -> None:
        workflow.PROJECT_ROOT = self.old_root

    def test_detector_check_timeout_has_long_default(self) -> None:
        self.assertEqual(workflow.DEFAULT_DETECTOR_CHECK_TIMEOUT_S, 30_000)

    def test_workflow_cost_summary_totals_session_cumulative_costs(self) -> None:
        costs = workflow.WorkflowCostSummary()
        for _ in range(6):
            costs.start_turn()
        # total_cost_usd reports are session-cumulative: later turns repeat
        # earlier spend, so only the per-session latest value may be totalled.
        self.assertEqual(costs.add(0.125, "session-a"), 0.125)
        self.assertEqual(costs.add("0.375", "session-a"), 0.25)
        self.assertIsNone(costs.add(None, "session-a"))
        self.assertIsNone(costs.add(float("nan"), "session-a"))
        self.assertEqual(costs.add(0.125, "session-b"), 0.125)

        self.assertEqual(costs.total_usd, 0.5)
        self.assertEqual(costs.reported_turns, 3)
        self.assertEqual(costs.unreported_turns, 2)
        self.assertEqual(
            costs.describe(),
            "$0.5000 (3 reported turn(s), 2 completed turn(s) without a cost, "
            "1 started turn(s) without a ResultMessage)",
        )

    def test_workflow_cost_summary_marks_missing_total_unavailable(self) -> None:
        costs = workflow.WorkflowCostSummary()
        costs.start_turn()
        costs.add(None)

        self.assertEqual(
            costs.describe(),
            "unavailable (0 reported turn(s), 1 completed turn(s) without a cost, "
            "0 started turn(s) without a ResultMessage)",
        )

    def test_agent_session_connect_timeout_is_bounded(self) -> None:
        class NeverConnects:
            async def __aenter__(self):
                await asyncio.Event().wait()

            async def __aexit__(self, *_args):
                return None

        async def exercise() -> None:
            with self.assertRaisesRegex(
                SystemExit, "session did not open.*no workflow stage was started"
            ):
                async with workflow.open_agent_session(
                    lambda **_kwargs: NeverConnects(),
                    options=object(),
                    connect_timeout_s=0.01,
                ):
                    self.fail("an unavailable SDK session was yielded")

        asyncio.run(exercise())

    def test_agent_session_closes_after_success(self) -> None:
        events = []

        class Client:
            async def __aenter__(self):
                events.append("open")
                return self

            async def __aexit__(self, *_args):
                events.append("close")

        async def exercise() -> None:
            async with workflow.open_agent_session(
                lambda **_kwargs: Client(),
                options=object(),
                connect_timeout_s=1,
            ) as client:
                self.assertIsInstance(client, Client)
                events.append("use")

        asyncio.run(exercise())
        self.assertEqual(events, ["open", "use", "close"])

    def test_detector_assembly_keeps_feature_and_runtime_ownership_separate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            template = root / "runtime.py.tmpl"
            feature = root / "feature.py"
            output = root / "detector.py"
            template.write_text(
                "# __DETECTOR_FEATURE_IMPLEMENTATION__\n"
                "def main():\n    return extract_features({})\n"
            )
            feature.write_text(
                "FEATURE_REGISTRY = [('x', 'segment', None, 1)]\n"
                "ANALYSIS_TIMING_GROUPS = [{\n"
                "    'key': 'hypo_1', 'hypothesis_ids': [1],\n"
                "    'phase': 'segment', 'feature_names': ['x'],\n"
                "}]\n"
                "class SegmentAccumulator:\n    pass\n"
                "def extract_features(payload, verbose=True, timing=None, "
                "enabled_analysis_keys=None, profile_segment_limit=None):\n"
                "    _ = profile_segment_limit\n"
                "    def _analysis_enabled(key):\n"
                "        return enabled_analysis_keys is None or key in enabled_analysis_keys\n"
                "    selected = _analysis_enabled('hypo_1')\n"
                "    if timing is not None:\n"
                "        timing.start_phase('segment')\n"
                "        timing.start_analysis('hypo_1')\n"
                "    return FEATURE_REGISTRY\n"
            )

            assemble_detector(template, feature, output)

            text = output.read_text()
            self.assertIn("FEATURE_REGISTRY", text)
            self.assertIn("def main", text)
            self.assertNotIn("__DETECTOR_FEATURE_IMPLEMENTATION__", text)

            feature.write_text(
                feature.read_text() + "def run_detector():\n    pass\n"
            )
            with self.assertRaisesRegex(SystemExit, "runtime-owned.*run_detector"):
                assemble_detector(template, feature, output)

    def test_detector_assembly_injects_reviewed_split_adapter(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            template = root / "runtime.py.tmpl"
            feature = root / "feature.py"
            output = root / "split_detector.py"
            template.write_text(
                "import numpy as np\n"
                "from collections import defaultdict\n"
                "def build_comp_to_seg(graph):\n    return {}\n"
                "# __DETECTOR_TARGET_ADAPTER__\n"
                "# __DETECTOR_FEATURE_IMPLEMENTATION__\n"
            )
            feature.write_text(
                "FEATURE_REGISTRY = [('gap_um_feature', 'candidate_pair')]\n"
                "ANALYSIS_TIMING_GROUPS = [{\n"
                "    'key': 'hypo_1', 'hypothesis_ids': [1],\n"
                "    'phase': 'candidate_pair',\n"
                "    'feature_names': ['gap_um_feature'],\n"
                "}]\n"
                "class FeatureAccumulator:\n    pass\n"
                "def extract_features(payload, verbose=True, timing=None, "
                "enabled_analysis_keys=None, profile_segment_limit=None):\n"
                "    _ = profile_segment_limit\n"
                "    def _analysis_enabled(key):\n"
                "        return enabled_analysis_keys is None or key in enabled_analysis_keys\n"
                "    _analysis_enabled('hypo_1')\n"
                "    if timing is not None:\n"
                "        timing.start_phase('candidate_pair')\n"
                "        timing.start_analysis('hypo_1')\n"
                "    return build_sample_universe(payload)\n"
            )

            assemble_detector(
                template, feature, output, target=DetectorTarget.SPLIT)

            text = output.read_text()
            self.assertIn("DETECTOR_TARGET = 'split_detection'", text)
            self.assertIn("def _derive_split_truth", text)
            self.assertIn("class FeatureAccumulator", text)

            feature.write_text(
                feature.read_text().replace(
                    "return build_sample_universe(payload)",
                    "_ = payload['gt_edge_error']\n"
                    "    return build_sample_universe(payload)",
                )
            )
            with self.assertRaisesRegex(SystemExit, "GT-only.*gt_edge_error"):
                assemble_detector(
                    template, feature, output, target=DetectorTarget.SPLIT)

    def test_detector_assembly_rejects_every_template_owned_symbol(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            template = root / "runtime.py.tmpl"
            feature = root / "feature.py"
            output = root / "detector.py"
            template.write_text(
                "def load_payload(path):\n    return path\n"
                "# __DETECTOR_FEATURE_IMPLEMENTATION__\n"
            )
            feature.write_text(
                "FEATURE_REGISTRY = [('x', 'segment', None, 1)]\n"
                "ANALYSIS_TIMING_GROUPS = [{\n"
                "    'key': 'hypo_1', 'hypothesis_ids': [1],\n"
                "    'phase': 'segment', 'feature_names': ['x'],\n"
                "}]\n"
                "class SegmentAccumulator:\n    pass\n"
                "def load_payload(path):\n    return {'shadowed': path}\n"
                "def extract_features(payload, verbose=True, timing=None, "
                "enabled_analysis_keys=None, profile_segment_limit=None):\n"
                "    _ = profile_segment_limit\n"
                "    def _analysis_enabled(key):\n"
                "        return enabled_analysis_keys is None or key in enabled_analysis_keys\n"
                "    _analysis_enabled('hypo_1')\n"
                "    if timing is not None:\n"
                "        timing.start_phase('segment')\n"
                "        timing.start_analysis('hypo_1')\n"
                "    return FEATURE_REGISTRY\n"
            )

            with self.assertRaisesRegex(SystemExit, "runtime-owned.*load_payload"):
                assemble_detector(template, feature, output)

    def test_detector_assembly_rejects_incomplete_timing_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            template = root / "runtime.py.tmpl"
            feature = root / "feature.py"
            output = root / "detector.py"
            template.write_text("# __DETECTOR_FEATURE_IMPLEMENTATION__\n")
            feature.write_text(
                "FEATURE_REGISTRY = [('x', 'segment'), ('y', 'segment')]\n"
                "ANALYSIS_TIMING_GROUPS = [{\n"
                "    'key': 'hypo_1', 'hypothesis_ids': [1],\n"
                "    'phase': 'segment', 'feature_names': ['x'],\n"
                "}]\n"
                "class SegmentAccumulator:\n    pass\n"
                "def extract_features(payload, verbose=True, timing=None, "
                "enabled_analysis_keys=None, profile_segment_limit=None):\n"
                "    _ = profile_segment_limit\n"
                "    def _analysis_enabled(key):\n"
                "        return enabled_analysis_keys is None or key in enabled_analysis_keys\n"
                "    selected = _analysis_enabled('hypo_1')\n"
                "    if timing is not None:\n"
                "        timing.start_phase('segment')\n"
                "        timing.start_analysis('hypo_1')\n"
                "    return None\n"
            )

            with self.assertRaisesRegex(SystemExit, "timing coverage mismatch"):
                assemble_detector(template, feature, output)

    def test_measuretime_writes_cost_and_selection_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            feature = root / "feature.py"
            detector = root / "detector.py"
            inventory = root / "feature_inventory.json"
            input_path = root / "dataset_cache_123456_mcl100_add.pkl"
            inventory.write_text('{"schema_version": 2}\n')
            input_path.write_bytes(b"placeholder")
            feature.write_text(
                "class FeatureSpec:\n"
                "    def __init__(self, name):\n"
                "        self.name = name\n"
                "FEATURE_REGISTRY = [FeatureSpec('x')]\n"
                "FEATURE_NAMES = [f.name for f in FEATURE_REGISTRY]\n"
                "ANALYSIS_TIMING_GROUPS = [{\n"
                "    'key': 'hypo_1', 'hypothesis_ids': [1],\n"
                "    'phase': 'segment', 'feature_names': ['x'],\n"
                "}]\n"
                "class SegmentAccumulator:\n    pass\n"
                "def extract_features(payload, verbose=True, timing=None, "
                "enabled_analysis_keys=None, profile_segment_limit=None):\n"
                "    _ = profile_segment_limit\n"
                "    def _analysis_enabled(key):\n"
                "        return enabled_analysis_keys is None or key in enabled_analysis_keys\n"
                "    _analysis_enabled('hypo_1')\n"
                "    phase = timing.start_phase('segment') if timing else None\n"
                "    token = timing.start_analysis(\n"
                "        'hypo_1', context={'segment_id': 1, 'nodes': 3}\n"
                "    ) if timing else None\n"
                "    if timing:\n"
                "        timing.stop_analysis(token)\n"
                "        timing.stop_phase(phase)\n"
                "    return [1], np.array([0], dtype=np.int64), SegmentAccumulator()\n"
            )
            assemble_detector(workflow.RUNTIME_TEMPLATE_PATH, feature, detector)
            spec = importlib.util.spec_from_file_location("measuretime_detector", detector)
            module = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(module)
            module.load_payload = lambda _path: {
                "min_cable_length": 100,
                "gt_node_canonical_label": module.np.array(
                    [1], dtype=module.np.int64),
                "gt_merge_labels": [],
            }
            module._collect_measuretime_versions = lambda: {"python": "test"}
            args = types.SimpleNamespace(
                pkl=str(input_path), test_pkl=None, out_csv=None,
                exclude_empty=False, out_dir=str(root), inventory=str(inventory),
                hypothesis_selection=None,
                exclude_hypotheses=None,
                measuretime_occurrences=3,
            )

            rc = module.run_measuretime(args, lambda _msg: None)

            artifact = root / "analysis_timing_123456.json"
            payload = json.loads(artifact.read_text())
            self.assertEqual(rc, 0)
            self.assertEqual(payload["status"], "complete")
            self.assertEqual(payload["mode"], "measuretime")
            self.assertEqual(payload["schema_version"], 2)
            self.assertEqual(payload["analyses"][0]["eligible_calls"], 1)
            self.assertTrue(payload["metadata"]["sampled"])
            self.assertEqual(payload["metadata"]["profile_occurrence_limit"], 3)
            self.assertEqual(payload["metadata"]["n_profiled_segments"], 1)
            self.assertEqual(payload["metadata"]["profiled_segment_ids"], [1])
            self.assertEqual(
                payload["hypothesis_costs"][0][
                    "estimated_removable_seconds_if_excluded_alone"
                ],
                payload["analyses"][0]["total_seconds"],
            )
            self.assertIn("feature_extraction", payload["wall_seconds"])
            self.assertTrue((root / "hypothesis_cost_report_123456.md").exists())
            selection_template = json.loads(
                (root / "hypothesis_selection_template_123456.json").read_text()
            )
            self.assertEqual(selection_template["excluded_hypothesis_ids"], [])
            self.assertFalse(any(root.glob("merge_detector_*.csv")))
            self.assertFalse(any(root.glob("merge_detector_*.joblib")))
            self.assertFalse(any(root.glob("model_selection_*.json")))

            failed_dir = root / "failed"
            failed_dir.mkdir()
            args.out_dir = str(failed_dir)

            def fail_extraction(*_args, **_kwargs):
                raise RuntimeError("synthetic extraction failure")

            module.extract_features = fail_extraction
            with self.assertRaisesRegex(RuntimeError, "synthetic extraction failure"):
                module.run_measuretime(args, lambda _msg: None)
            failed_payload = json.loads(
                (failed_dir / "analysis_timing_123456.json").read_text()
            )
            self.assertEqual(failed_payload["status"], "failed")
            self.assertIn("synthetic extraction failure", failed_payload["error"])

    def test_hypothesis_selection_rejects_partial_shared_unit(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            selection = root / "selection.json"
            groups = [
                {"key": "shared", "hypothesis_ids": [1, 2],
                 "phase": "chain", "feature_names": ["a", "b"]},
                {"key": "hypo_3", "hypothesis_ids": [3],
                 "phase": "segment", "feature_names": ["c"]},
            ]
            # The runtime loader is exercised from a minimally assembled module
            # in the preceding test; here use the reviewed template's semantics
            # through the current example module.
            detector_path = (
                Path(__file__).resolve().parents[2] / "autodiscovery-application" /
                "merge-error-794495-mcl100_2026-08-04" /
                "merge_site_detector.py"
            )
            spec = importlib.util.spec_from_file_location(
                "selection_detector", detector_path)
            module = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(module)
            module.FEATURE_NAMES = ["a", "b", "c"]
            selection.write_text(json.dumps({
                "schema_version": 1,
                "excluded_hypothesis_ids": [1],
            }))
            with self.assertRaisesRegex(ValueError, "partial exclusion"):
                module.load_hypothesis_selection(str(selection), groups)
            selection.write_text(json.dumps({
                "schema_version": 1,
                "excluded_hypothesis_ids": [1, 2],
            }))
            resolved = module.load_hypothesis_selection(str(selection), groups)
            self.assertEqual(resolved["enabled_analysis_keys"], ["hypo_3"])
            self.assertEqual(resolved["excluded_feature_names"], ["a", "b"])

            direct = module.load_hypothesis_selection(
                None, groups, excluded_hypothesis_ids=[1, 2])
            self.assertEqual(direct["selection_source"], "manual_cli")
            self.assertEqual(direct["excluded_hypothesis_ids"], [1, 2])
            self.assertEqual(direct["enabled_analysis_keys"], ["hypo_3"])
            self.assertIsNone(direct["path"])
            with self.assertRaisesRegex(ValueError, "mutually exclusive"):
                module.load_hypothesis_selection(
                    str(selection), groups, excluded_hypothesis_ids=[1, 2])
            with self.assertRaisesRegex(ValueError, "duplicates"):
                module.load_hypothesis_selection(
                    None, groups, excluded_hypothesis_ids=[3, 3])

    def test_hypothesis_selection_validates_timing_provenance_and_reasons(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            detector_path = (
                Path(__file__).resolve().parents[2] / "autodiscovery-application" /
                "merge-error-794495-mcl100_2026-08-04" /
                "merge_site_detector.py"
            )
            spec = importlib.util.spec_from_file_location(
                "selection_provenance_detector", detector_path)
            module = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(module)
            module.FEATURE_NAMES = ["a", "b"]
            groups = [
                {"key": "hypo_1", "hypothesis_ids": [1],
                 "phase": "edge", "feature_names": ["a"]},
                {"key": "hypo_2", "hypothesis_ids": [2],
                 "phase": "segment", "feature_names": ["b"]},
            ]
            timing = root / "analysis_timing.json"
            timing.write_text(json.dumps({
                "schema_version": 2,
                "mode": "measuretime",
                "status": "complete",
                "metadata": {
                    "detector_sha256": module._sha256_file(module.__file__),
                    "feature_inventory_sha256": "a" * 64,
                },
            }))
            selection = root / "selection.json"
            selection.write_text(json.dumps({
                "schema_version": 1,
                "source_timing_artifact": str(timing),
                "source_timing_sha256": module._sha256_file(str(timing)),
                "excluded_hypothesis_ids": [1],
                "reasons": {"1": "Too expensive after scientific review."},
            }))

            resolved = module.load_hypothesis_selection(str(selection), groups)
            self.assertEqual(
                resolved["source_timing_feature_inventory_sha256"], "a" * 64)
            self.assertEqual(
                resolved["reasons"], {"1": "Too expensive after scientific review."})

            bad = json.loads(selection.read_text())
            bad["source_timing_sha256"] = "0" * 64
            selection.write_text(json.dumps(bad))
            with self.assertRaisesRegex(ValueError, "source timing SHA-256"):
                module.load_hypothesis_selection(str(selection), groups)

    def test_current_detector_selection_skips_unselected_computation(self) -> None:
        import networkx as nx
        import numpy as np

        detector_path = (
            Path(__file__).resolve().parents[2] / "autodiscovery-application" /
            "merge-error-794495-mcl100_2026-08-04" /
            "merge_site_detector.py"
        )
        spec = importlib.util.spec_from_file_location(
            "current_selection_detector", detector_path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

        parsed = module.build_arg_parser().parse_args(
            ["dataset.pkl", "--exclude-hypotheses", "39", "6"])
        self.assertEqual(parsed.exclude_hypotheses, [39, 6])
        direct = module.load_hypothesis_selection(
            None, module.ANALYSIS_TIMING_GROUPS,
            excluded_hypothesis_ids=[39, 6])
        self.assertEqual(direct["selection_source"], "manual_cli")
        self.assertNotIn("hypo_39", direct["enabled_analysis_keys"])
        self.assertNotIn("hypo_6", direct["enabled_analysis_keys"])
        self.assertIn("max_bridge_radius_variance_ratio",
                      direct["excluded_feature_names"])
        self.assertIn("max_normalized_edge_betweenness_bridge",
                      direct["excluded_feature_names"])

        graph = nx.Graph()
        graph.add_edges_from([(0, 1), (1, 2), (3, 4)])
        graph.node_xyz = np.array([
            [0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [5.0, 0.0, 0.0],
            [0.0, 10.0, 0.0], [1.0, 10.0, 0.0],
        ])
        graph.node_radius = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        graph.node_component_id = np.array([1, 1, 1, 2, 2])
        graph.component_id_to_swc_id = {1: "1.0", 2: "2.0"}
        payload = {
            "fragments_graph": graph,
            "gt_node_canonical_label": np.array([1, 1, 1, 2, 2]),
            "gt_merge_labels": [],
            "min_cable_length": 100,
        }

        _, _, default_acc = module.extract_features(payload, verbose=False)
        all_keys = {group["key"] for group in module.ANALYSIS_TIMING_GROUPS}
        _, _, explicit_all_acc = module.extract_features(
            payload, verbose=False, enabled_analysis_keys=all_keys
        )
        self.assertTrue(default_acc.to_frame().equals(
            explicit_all_acc.to_frame()
        ))
        sampled_ids, _, sampled_acc = module.extract_features(
            payload, verbose=False, profile_segment_limit=1
        )
        self.assertEqual(sampled_ids, [1])
        self.assertEqual(len(sampled_acc.to_frame()), 1)

        original = module._get_max_ebc_contracted
        module._get_max_ebc_contracted = lambda *_args: (_ for _ in ()).throw(
            AssertionError("excluded component hypothesis executed")
        )
        try:
            _, _, acc = module.extract_features(
                payload, verbose=False, enabled_analysis_keys={"hypo_8"}
            )
        finally:
            module._get_max_ebc_contracted = original
        frame = acc.to_frame()
        self.assertTrue(frame["max_euclidean_edge_jump_is_defined"].all())
        for name in module.FEATURE_NAMES:
            if name != "max_euclidean_edge_jump":
                self.assertFalse(frame[name + "_is_defined"].any(), name)

    def test_reviewed_runtime_template_owns_model_and_cli_contracts(self) -> None:
        template = workflow.RUNTIME_TEMPLATE_PATH.read_text()
        self.assertEqual(template.count("# __DETECTOR_FEATURE_IMPLEMENTATION__"), 1)
        self.assertEqual(template.count("# __DETECTOR_TARGET_ADAPTER__"), 1)
        for symbol in (
            "def validate_model_config", "def build_estimator",
            "def run_nested_selection", "def run_smoke_test",
            "def build_arg_parser", "def run_detector", "def run_measuretime",
            "def load_hypothesis_selection", "class AnalysisTimingRecorder",
            "def write_hypothesis_cost_artifacts", "def main",
        ):
            self.assertIn(symbol, template)
        self.assertIn('ap.add_argument("--measuretime"', template)
        self.assertIn('ap.add_argument("--measuretime-occurrences"', template)
        self.assertIn('ap.add_argument("--exclude-hypotheses"', template)
        self.assertIn('ap.add_argument("--hypothesis-selection"', template)
        self.assertIn("--hypothesis-selection or\n    --exclude-hypotheses", template)
        self.assertNotIn("--exclude-hypotheses 39 6", template)
        self.assertIn('save(fig, 6, "winner_model_explanation")', template)
        self.assertIn('winner == "explainable_boosting"', template)
        self.assertIn("clf.term_importances()", template)
        self.assertIn("clf.explain_global()", template)
        self.assertIn("native model-specific importance", template)
        self.assertIn("def validation_permutation():", template)
        self.assertNotIn('save(fig, 6, "linear_coefficients")', template)
        # Nested-selection progress logging: every grid point, every family
        # (per outer fold and in the final all-rows search) must announce
        # itself with a timing, and a failing fit must surface its exception —
        # a silent `except: pass` here once hid a KeyError as 0/240 OOF rows.
        self.assertIn("grid %d/%d mean_ap=%.4f (%.1fs)", template)
        self.assertIn("grid %d/%d FAILED after %.1fs", template)
        self.assertIn("outer refit FAILED after %.1fs", template)
        self.assertIn("final search %s:", template)

    def test_split_runtime_passes_driver_owned_no_data_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            feature = root / "feature.py"
            detector = root / "split_site_detector.py"
            inventory = root / "feature_inventory.json"
            model_config = root / "model_candidates.json"
            inventory.write_text("{}\n")
            model_config.write_text(json.dumps(self._valid_model_config(inventory)))
            feature.write_text(
                "FEATURE_REGISTRY = [('gap_um_feature', 'candidate_pair')]\n"
                "ANALYSIS_TIMING_GROUPS = [{\n"
                "    'key': 'hypo_1', 'hypothesis_ids': [1],\n"
                "    'phase': 'candidate_pair',\n"
                "    'feature_names': ['gap_um_feature'],\n"
                "}]\n"
                "class FeatureAccumulator:\n    pass\n"
                "def extract_features(payload, verbose=True, timing=None, "
                "enabled_analysis_keys=None, profile_segment_limit=None):\n"
                "    _ = profile_segment_limit\n"
                "    def _analysis_enabled(key):\n"
                "        return enabled_analysis_keys is None or key in enabled_analysis_keys\n"
                "    _analysis_enabled('hypo_1')\n"
                "    if timing is not None:\n"
                "        timing.start_phase('candidate_pair')\n"
                "        timing.start_analysis('hypo_1')\n"
                "    samples, labels = build_sample_universe(payload)\n"
                "    return samples, labels, FeatureAccumulator()\n"
            )
            assemble_detector(
                workflow.RUNTIME_TEMPLATE_PATH,
                feature,
                detector,
                target=DetectorTarget.SPLIT,
            )

            workflow.validate_detector_executable(detector, model_config)

    def test_current_detector_generates_ebm_winner_explanation(self) -> None:
        try:
            import interpret  # noqa: F401
        except ImportError:
            self.skipTest("optional interpret package is unavailable")
        import numpy as np
        import pandas as pd

        detector_path = (
            Path(__file__).resolve().parents[2] / "autodiscovery-application" /
            "merge-error-794495-mcl100_2026-08-04" /
            "merge_site_detector.py"
        )
        spec = importlib.util.spec_from_file_location(
            "ebm_figure_detector", detector_path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

        rng = np.random.default_rng(42)
        feature_order = ["feature_a", "feature_b", "feature_c"]
        values = rng.normal(size=(120, len(feature_order)))
        y = ((values[:, 0] + values[:, 1] ** 2) > 0.8).astype(int)
        frame = pd.DataFrame(values, columns=feature_order)
        for name in feature_order:
            frame[name + "_is_defined"] = True
        params = {
            "max_bins": 16,
            "interactions": 1,
            "learning_rate": 0.05,
            "max_rounds": 20,
            "min_samples_leaf": 2,
        }
        estimator = module.build_estimator("explainable_boosting", params)
        estimator.fit(values, y)
        scores = module._score_matrix(estimator, values)
        nested = {
            "oof": {"explainable_boosting": scores},
            "per_fold": [{
                "family_choice": {
                    "explainable_boosting": {"params": params}
                }
            }],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = module.make_figures(
                tmpdir, "synthetic_full", frame, y, nested, scores, scores,
                "explainable_boosting", estimator, feature_order,
                winner_params=params,
            )
            names = {Path(path).name for path in paths}
            self.assertIn(
                "synthetic_full_06_winner_model_explanation.png", names
            )
            self.assertIn(
                "synthetic_full_07_permutation_importance.png", names
            )

    def test_public_build_artifact_names_remain_compatible(self) -> None:
        self.assertEqual(workflow.BUILD_ARTIFACT_NAMES, (
            "feature_inventory.json",
            "model_candidates.json",
            "merge_site_detector.py",
            "README.md",
            "RUN_COMMANDS.md",
            "detector_build_workflow.log.txt",
        ))

    def test_driver_readme_skeleton_preserves_public_document_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            inventory = root / "feature_inventory.json"
            inventory.write_text(json.dumps({
                "selection_manifest": None,
                "hypotheses": [{
                    "id": 1,
                    "included": False,
                    "feature_source_path": None,
                    "features": [],
                    "exclusion_reason": "Overturned.",
                }],
            }))
            config = root / "model_candidates.json"
            config.write_text(json.dumps({
                "selection_basis": "Conservative baselines.",
                "candidates": [{
                    "name": "logistic_l2",
                    "role": "baseline",
                    "native_nan": False,
                    "requires_package": None,
                    "reason": "Linear baseline.",
                }],
            }))
            readme = root / "README.md"

            workflow.write_readme_skeleton(
                readme,
                run_rel="autodiscovery/run.json",
                summary_rel="autodiscovery/run.summary.md",
                inventory_path=inventory,
                inventory_rel="app/feature_inventory.json",
                model_config_path=config,
                model_config_rel="app/model_candidates.json",
                detector_rel="app/merge_site_detector.py",
                run_commands_rel="app/RUN_COMMANDS.md",
                cache_hint=None,
                primary_metric="average_precision",
            )

            text = readme.read_text()
            self.assertIn("**Build-time status:**", text)
            self.assertIn("not\n> updated by later detector runs", text)
            self.assertIn("average_precision", text)
            self.assertIn("merge_detector_<brain>.csv", text)
            self.assertIn("app/RUN_COMMANDS.md", text)
            block = workflow.read_driver_generated_block(readme)
            self.assertIn("Feature and source mapping", block)
            skeleton_sha = hashlib.sha256(readme.read_bytes()).hexdigest()
            with self.assertRaisesRegex(SystemExit, "skeleton unchanged"):
                workflow.validate_readme_enrichment(
                    readme,
                    expected_driver_block=block,
                    skeleton_sha256=skeleton_sha,
                )
            readme.write_text(text + "\nReviewed feature semantics.\n")
            workflow.validate_readme_enrichment(
                readme,
                expected_driver_block=block,
                skeleton_sha256=skeleton_sha,
            )

    def test_driver_run_commands_are_bound_to_exact_artifact_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            app = root / "app"
            app.mkdir()
            detector = app / "merge_site_detector.py"
            inventory = app / "feature_inventory.json"
            config = app / "model_candidates.json"
            policy = root / "detector_model_policy.json"
            template = root / "detector_runtime.py.tmpl"
            detector.write_text("print('detector')\n")
            inventory.write_text('{"schema_version": 2}\n')
            config.write_text('{"schema_version": 2}\n')
            policy.write_text('{"schema_version": 1}\n')
            template.write_text("# runtime\n")
            commands = app / "RUN_COMMANDS.md"

            workflow.write_run_commands(
                commands,
                project_root=root,
                run_rel="autodiscovery/merge-error-794495-mcl100_run.json",
                detector_path=detector,
                detector_rel="app/merge_site_detector.py",
                inventory_path=inventory,
                inventory_rel="app/feature_inventory.json",
                model_config_path=config,
                model_config_rel="app/model_candidates.json",
                model_policy_path=policy,
                model_policy_rel="detector_model_policy.json",
                runtime_template_path=template,
                runtime_template_rel="detector_runtime.py.tmpl",
                cache_hint="cache/dataset_cache_794495_mcl100_add.pkl",
                agent_model="claude-test",
                agent_effort="high",
            )

            text = commands.read_text()
            self.assertIn(hashlib.sha256(detector.read_bytes()).hexdigest(), text)
            self.assertIn(hashlib.sha256(inventory.read_bytes()).hexdigest(), text)
            self.assertIn(hashlib.sha256(config.read_bytes()).hexdigest(), text)
            self.assertIn("not\npurely deterministic", text)
            self.assertIn("--synthetic-smoke-test", text)
            self.assertIn("--measuretime-occurrences 3", text)
            self.assertIn("observed sample costs", text)
            self.assertIn("--nodelist=n287", text)
            self.assertIn("dataset_cache_794495_mcl100_add.pkl", text)
            self.assertIn("merge_detector_794495.csv", text)
            self.assertIn("Monitor component extraction", text)
            self.assertIn("feature:start", text)
            self.assertIn("calls` and `total_s", text)
            self.assertIn("Profile hypothesis computational cost", text)
            self.assertIn("--measuretime", text)
            self.assertIn("analysis_timing_${BRAIN}.json", text)
            self.assertIn("hypothesis_cost_report_${BRAIN}.md", text)
            self.assertIn("hypothesis_selection_template_${BRAIN}.json", text)
            self.assertIn("--hypothesis-selection", text)
            self.assertIn("EXCLUDE_IDS=()", text)
            self.assertIn("Set EXCLUDE_IDS from the cost report", text)
            self.assertNotIn("EXCLUDE_IDS=(<ID1> <ID2>)", text)
            self.assertNotIn("39 6", text)
            self.assertIn('runs/${BRAIN}-manual-exclusions', text)
            self.assertIn(
                'python "$DETECTOR" "$DATA_PKL" \\\n    --measuretime', text
            )
            self.assertIn(
                '--exclude-hypotheses "${EXCLUDE_IDS[@]}" \\\n    --model-config', text
            )
            self.assertNotIn("hand-authored selection. Omitting\nOmitting", text)
            self.assertIn("--mem=80G", text)
            self.assertIn("conda list --explicit", text)
            self.assertIn("Generate the bilingual result-analysis report", text)
            self.assertIn('run_detector_result_analysis.py "$RUN_DIR"', text)
            self.assertIn("result_analysis_evidence.json", text)
            self.assertIn("complete English", text)
            self.assertIn("complete Chinese translation", text)

    def test_split_documentation_uses_split_public_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            app = root / "app"
            app.mkdir()
            detector = app / "split_site_detector.py"
            inventory = app / "feature_inventory.json"
            config = app / "model_candidates.json"
            policy = root / "detector_model_policy.json"
            template = root / "detector_runtime.py.tmpl"
            detector.write_text("# split detector\n")
            inventory.write_text(json.dumps({
                "selection_manifest": None,
                "hypotheses": [{
                    "id": 1, "included": False,
                    "feature_source_path": None, "features": [],
                    "exclusion_reason": "Excluded in test.",
                }],
            }))
            config.write_text(json.dumps({
                "selection_basis": "Conservative baseline.",
                "candidates": [{
                    "name": "logistic_l2", "role": "baseline",
                    "native_nan": False, "requires_package": None,
                    "reason": "Stable baseline.",
                }],
            }))
            policy.write_text("{}\n")
            template.write_text("# runtime\n")
            readme = app / "README.md"
            commands = app / "RUN_COMMANDS.md"

            workflow.write_readme_skeleton(
                readme,
                run_rel="autodiscovery/split-error.json",
                summary_rel="autodiscovery/split-error.summary.md",
                inventory_path=inventory,
                inventory_rel="app/feature_inventory.json",
                model_config_path=config,
                model_config_rel="app/model_candidates.json",
                detector_rel="app/split_site_detector.py",
                run_commands_rel="app/RUN_COMMANDS.md",
                cache_hint=None,
                primary_metric="average_precision",
                target=DetectorTarget.SPLIT,
            )
            workflow.write_run_commands(
                commands,
                project_root=root,
                run_rel="autodiscovery/split-error.json",
                detector_path=detector,
                detector_rel="app/split_site_detector.py",
                inventory_path=inventory,
                inventory_rel="app/feature_inventory.json",
                model_config_path=config,
                model_config_rel="app/model_candidates.json",
                model_policy_path=policy,
                model_policy_rel="detector_model_policy.json",
                runtime_template_path=template,
                runtime_template_rel="detector_runtime.py.tmpl",
                cache_hint=None,
                agent_model="test",
                agent_effort="test",
                target=DetectorTarget.SPLIT,
            )

            readme_text = readme.read_text()
            commands_text = commands.read_text()
            self.assertIn("# Split detector", readme_text)
            self.assertIn("split_detector_<brain>.csv", readme_text)
            self.assertIn('DETECTOR="$APP_DIR/split_site_detector.py"', commands_text)
            self.assertIn("split_detector_", commands_text)
            self.assertNotIn("merge_detector_", commands_text)
            self.assertIn("--job-name=split-detector", commands_text)
            self.assertIn("--nodelist=n287", commands_text)

    @staticmethod
    def _valid_model_config(inventory_path: Path) -> dict:
        def candidate(name: str, role: str, grid: dict, *, native_nan=False,
                      requires_package=None) -> dict:
            return {
                "name": name,
                "role": role,
                "reason": f"Exercise the {name} inductive bias.",
                "grid": grid,
                "native_nan": native_nan,
                "requires_package": requires_package,
            }

        return {
            "schema_version": workflow.MODEL_CONFIG_SCHEMA_VERSION,
            "feature_inventory_sha256": hashlib.sha256(
                inventory_path.read_bytes()
            ).hexdigest(),
            "model_policy_sha256": hashlib.sha256(
                workflow.MODEL_POLICY_PATH.read_bytes()
            ).hexdigest(),
            "selection_basis": "Three conservative baselines plus one smooth model.",
            "candidates": [
                candidate("logistic_l2", "baseline", {"C": [0.1, 1.0]}),
                candidate(
                    "logistic_elasticnet", "baseline",
                    {"C": [0.1, 1.0], "l1_ratio": [0.25, 0.75]},
                ),
                candidate(
                    "hist_gradient_boosting", "baseline",
                    {
                        "max_leaf_nodes": [7, 15], "max_depth": [None, 3],
                        "learning_rate": [0.03],
                    },
                    native_nan=True,
                ),
                candidate(
                    "spline_logistic", "optional",
                    {"n_knots": [3, 4], "degree": [2], "C": [0.1, 1.0]},
                ),
            ],
        }

    @staticmethod
    def _touch_inputs(root: Path, *, predictive: bool, fixed: bool) -> Path:
        run_json = root / "run.json"
        run_json.write_text("[]")
        (root / "run.summary.md").write_text("## Header\n")
        scope = ".predictive" if predictive else ""
        rerun = root / f"run.json{scope}.rerun"
        rerun.mkdir()
        (rerun / "hypo_1.py").write_text("print('rerun')\n")
        (rerun / "MANIFEST.json").write_text(json.dumps({
            "records": [{"id": 1, "file": "hypo_1.py"}],
        }))
        if predictive:
            source_hash = hashlib.sha256(run_json.read_bytes()).hexdigest()
            (root / "run.predictive-selection.json").write_text(json.dumps({
                "source_file": "run.json",
                "criterion": "predictive",
                "selected_ids": [1],
                "source_sha256": source_hash,
                "policy_version": workflow.PREDICTIVE_POLICY_VERSION,
            }))
        if fixed:
            fixed_dir = root / f"run.json{scope}.fixed"
            fixed_dir.mkdir()
            (fixed_dir / "hypo_1.py").write_text("print('fixed')\n")
        return run_json

    def test_resolve_inputs_returns_matching_predictive_fixed_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            workflow.PROJECT_ROOT = root
            run_json = self._touch_inputs(root, predictive=True, fixed=True)
            # A legacy fixed directory must never be paired with predictive rerun.
            legacy_fixed = root / "run.json.fixed"
            legacy_fixed.mkdir()
            (legacy_fixed / "hypo_9.py").write_text("print('legacy')\n")

            summary, rerun, fixed, selection, selected_ids = workflow.resolve_inputs(
                run_json
            )

            self.assertEqual(summary, root / "run.summary.md")
            self.assertEqual(rerun, root / "run.json.predictive.rerun")
            self.assertEqual(fixed, root / "run.json.predictive.fixed")
            self.assertEqual(selection, root / "run.predictive-selection.json")
            self.assertEqual(selected_ids, [1])

            context = workflow.resolve_run_context(run_json)
            self.assertTrue(context.predictive)
            self.assertEqual(context.selected_ids, (1,))
            self.assertEqual(context.target.value, "legacy_unspecified")
            protected = workflow.protected_source_paths(context)
            self.assertIn(run_json, protected)
            self.assertIn(rerun / "MANIFEST.json", protected)
            self.assertIn(rerun / "hypo_1.py", protected)
            self.assertIn(fixed / "hypo_1.py", protected)

    def test_resolve_inputs_allows_no_fixed_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            workflow.PROJECT_ROOT = root
            run_json = self._touch_inputs(root, predictive=False, fixed=False)

            _, rerun, fixed, selection, selected_ids = workflow.resolve_inputs(run_json)

            self.assertEqual(rerun, root / "run.json.rerun")
            self.assertIsNone(fixed)
            self.assertIsNone(selection)
            self.assertEqual(selected_ids, [1])

    def test_resolve_inputs_accepts_named_split_run_with_split_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            workflow.PROJECT_ROOT = root
            generic_run = self._touch_inputs(root, predictive=False, fixed=False)
            split_run = root / "split-error-794495-mcl100-run-2.json"
            generic_run.rename(split_run)
            (root / "run.summary.md").rename(
                root / "split-error-794495-mcl100-run-2.summary.md"
            )
            (root / "run.json.rerun").rename(
                root / "split-error-794495-mcl100-run-2.json.rerun"
            )

            context = workflow.resolve_run_context(split_run)

            self.assertEqual(context.target.value, "split_detection")
            self.assertEqual(context.selected_ids, (1,))

    def test_resolve_inputs_accepts_named_merge_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            workflow.PROJECT_ROOT = root
            generic_run = self._touch_inputs(root, predictive=False, fixed=False)
            merge_run = root / "merge-error-794495-mcl100.json"
            generic_run.rename(merge_run)
            (root / "run.summary.md").rename(
                root / "merge-error-794495-mcl100.summary.md"
            )
            (root / "run.json.rerun").rename(
                root / "merge-error-794495-mcl100.json.rerun"
            )

            _, _, _, _, selected_ids = workflow.resolve_inputs(merge_run)

            self.assertEqual(selected_ids, [1])

    def test_resolve_inputs_rejects_stale_predictive_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            workflow.PROJECT_ROOT = root
            run_json = self._touch_inputs(root, predictive=True, fixed=False)
            selection_path = root / "run.predictive-selection.json"
            selection = json.loads(selection_path.read_text())
            selection["selected_ids"] = [2]
            selection_path.write_text(json.dumps(selection))

            with self.assertRaisesRegex(SystemExit, "MANIFEST disagree"):
                workflow.resolve_inputs(run_json)

    def test_resolve_inputs_rejects_missing_corrected_results_before_build(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            workflow.PROJECT_ROOT = root
            run_json = self._touch_inputs(root, predictive=True, fixed=True)
            (root / "run.summary.md").write_text(
                "### 1. Corrected finding\n"
                "- **ID:** 1\n"
                "- **Reproduction:** REPRODUCED\n"
                "- **Verdict:** SOUND\n"
                "- **Post-correction verdict:** UPHELD\n"
            )

            with self.assertRaisesRegex(
                SystemExit, "post-correction verdict.*corrected.json.*missing"
            ):
                workflow.resolve_inputs(run_json)

            context = workflow.resolve_run_context(
                run_json, reconcile_unbacked_verdicts=True
            )
            self.assertEqual(context.unbacked_verdict_ids, (1,))

    def test_resolve_inputs_rejects_unbacked_post_correction_verdict(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            workflow.PROJECT_ROOT = root
            run_json = self._touch_inputs(root, predictive=True, fixed=True)
            (root / "run.summary.md").write_text(
                "### 1. Corrected finding\n"
                "- **ID:** 1\n"
                "- **Reproduction:** DIVERGED\n"
                "- **Verdict:** MAJOR\n"
                "- **Post-correction verdict:** INCONCLUSIVE\n"
            )
            (root / "run.json.predictive.corrected.json").write_text(json.dumps({
                "code_dir": "run.json.predictive.rerun",
                "corrected_dir": "run.json.predictive.fixed",
                "results": [{"id": 1, "result_status": "UNUSABLE"}],
            }))

            # Default: hard abort naming the id, both tokens, and the remedy flag.
            with self.assertRaisesRegex(
                SystemExit,
                "id 1.*INCONCLUSIVE.*UNUSABLE.*--reconcile-unbacked-verdicts",
            ):
                workflow.resolve_inputs(run_json)

            # Opt-in: the verdict is treated as null and the id is recorded.
            context = workflow.resolve_run_context(
                run_json, reconcile_unbacked_verdicts=True
            )
            self.assertEqual(context.unbacked_verdict_ids, (1,))

    def test_build_steps_require_source_audit_and_safe_exclusions(self) -> None:
        steps = workflow.build_steps(
            run_rel="autodiscovery/run.json",
            summary_rel="autodiscovery/run.summary.md",
            rerun_rel="autodiscovery/run.json.predictive.rerun",
            fixed_rel="autodiscovery/run.json.predictive.fixed",
            selection_rel="autodiscovery/run.predictive-selection.json",
            selected_ids=[1],
            out_dir_rel="autodiscovery-application/run",
        )
        by_name = {step["name"]: step["instruction"] for step in steps}
        inventory = by_name["inventory-features"]
        configure = by_name["configure-models"]
        generate = by_name["generate-detector"]
        verify = by_name["verify-and-document"]

        self.assertEqual([step["name"] for step in steps], [
            "inventory-features", "configure-models", "generate-detector",
            "verify-and-document",
        ])
        generate_step = next(
            step for step in steps if step["name"] == "generate-detector"
        )
        self.assertTrue(
            generate_step["expects_file"].endswith("/.feature_implementation.py")
        )
        self.assertIn("FAILED/UNUSABLE", inventory)
        self.assertIn("uncorrected CRITICAL", inventory)
        self.assertIn("feature_semantics_changed", inventory)
        self.assertIn("stale_fixed_ignored", inventory)
        self.assertIn("schema_version=1", inventory)
        self.assertIn("authoritative hypothesis set", inventory)
        self.assertIn("Evidence transcription, paths, hashes", inventory)
        self.assertIn("DRIVER responsibilities", inventory)
        self.assertIn("Do not copy reproduction_status", inventory)
        self.assertIn(".feature_semantics.json", inventory)
        self.assertIn("strict inventory-v2 validator", inventory)
        self.assertIn("MUST be JSON null (not an empty string)", inventory)
        self.assertIn("draft_validation semantic", inventory)
        self.assertIn("draft_validation model-advice", configure)
        self.assertIn("draft_validation feature", generate)
        self.assertEqual(
            inventory.count("DRAFT_VALIDATION_OK"), 1,
        )
        self.assertIn("display rank, NOT the hypothesis ID", inventory)
        self.assertNotIn("all 11 executed corrected", inventory)
        self.assertIn("ONLY the inventory-features semantic stage", inventory)
        self.assertIn("ONLY the configure-models advice stage", configure)
        self.assertIn(".model_advice.json", configure)
        self.assertIn("driver will add role, native_nan", configure)
        self.assertIn("ONLY the feature-implementation stage", generate)
        self.assertIn("ONLY the verify-and-document stage", verify)
        self.assertNotIn("matching SHA-256", inventory)
        self.assertIn("zero through 2", configure)
        self.assertIn("Never write", configure)
        self.assertIn("inventory hash", configure)
        self.assertIn("policy hash", configure)
        self.assertIn("detector_model_policy.json", configure)
        self.assertIn("owns native-NaN declarations", configure)
        # Cost discipline for the most expensive family: the advice stage must
        # cap the explainable_boosting grid instead of multiplying wall-clock.
        self.assertIn("AT MOST 4 combinations", configure)
        self.assertIn("feature_source_path", generate)
        self.assertIn("Never fall back", generate)
        self.assertIn(".feature_implementation.py", generate)
        self.assertIn("driver owns the reviewed runtime template", generate)
        self.assertIn("Do not define validate_model_config", generate)
        self.assertIn("weighted or unweighted", generate)
        self.assertIn("does not retain custom SkeletonGraph", generate)
        self.assertIn("ANALYSIS_TIMING_GROUPS", generate)
        self.assertIn("cover every\nFEATURE_REGISTRY name exactly once", generate)
        self.assertIn("runtime-provided\ntiming recorder around every analysis group", generate)
        self.assertIn("enabled_analysis_keys=None", generate)
        self.assertIn("profile_segment_limit=None", generate)
        self.assertIn("default limit of\n3", generate)
        self.assertIn("final-run selection unit", generate)
        self.assertIn("does not make the final\ndecision automatically", generate)
        self.assertIn("must not change feature math", generate)
        self.assertIn("component traversal has flushed segment/component", verify)
        self.assertIn("final per-feature cumulative timing/call-count", verify)
        self.assertIn("schema-v2 timing JSON", verify)
        self.assertIn("without audit, CV, fitting", verify)
        self.assertIn("--hypothesis-selection", verify)
        self.assertIn("does not choose exclusions automatically", verify)
        self.assertIn("RUN_COMMANDS.md", generate)
        self.assertIn("read-only", verify)
        self.assertIn("driver-assembled", verify)
        self.assertIn("rerun-vs-fixed choice", verify)
        self.assertIn("instance-bound", verify)
        self.assertIn("without duplicating them", verify)

    def test_build_steps_define_split_candidate_contract(self) -> None:
        steps = workflow.build_steps(
            run_rel="autodiscovery/split-error-run.json",
            summary_rel="autodiscovery/split-error-run.summary.md",
            rerun_rel="autodiscovery/split-error-run.json.predictive.rerun",
            fixed_rel=None,
            selection_rel="autodiscovery/split-error-run.predictive-selection.json",
            selected_ids=[6, 39],
            out_dir_rel="autodiscovery-application/split-error-run",
            target=DetectorTarget.SPLIT,
        )
        by_name = {step["name"]: step for step in steps}
        inventory = by_name["inventory-features"]["instruction"]
        generate = by_name["generate-detector"]["instruction"]
        verify = by_name["verify-and-document"]["instruction"]

        self.assertTrue(
            by_name["generate-detector"]["expects_file"].endswith(
                "/.feature_implementation.py"))
        self.assertIn("canonical unordered\ncandidate segment pair", inventory)
        self.assertIn("within 30 um", inventory)
        self.assertIn("FeatureAccumulator", generate)
        self.assertIn("build_sample_universe(payload)", generate)
        self.assertIn("must be ignored by feature extraction", generate)
        self.assertIn("For split measuretime", generate)
        self.assertIn("EITHER segment of the pair is sampled", generate)
        self.assertIn("Do NOT require BOTH", generate)
        self.assertIn("number of GT edges", generate)
        self.assertIn("candidate-pair-level", verify)
        self.assertIn("remain explicitly unscored (NaN)", verify)
        self.assertIn("conditional on\nthe OOF-scored subset", verify)
        self.assertIn("split_site_detector.py", verify)

    def test_validate_model_config_accepts_bounded_allowlisted_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            workflow.PROJECT_ROOT = root
            inventory_path = root / "feature_inventory.json"
            inventory_path.write_text('{"schema_version": 2}')
            config_path = root / workflow.MODEL_CONFIG_NAME
            config_path.write_text(json.dumps(self._valid_model_config(inventory_path)))

            workflow.validate_model_config(config_path, inventory_path)

    def test_model_advice_compiles_mechanical_policy_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            inventory_path = root / "feature_inventory.json"
            inventory_path.write_text('{"schema_version": 2}')
            advice_path = root / ".model_advice.json"
            advice_path.write_text(json.dumps({
                "schema_version": 1,
                "selection_basis": "Use conservative baselines only.",
                "candidates": [
                    {
                        "name": "logistic_l2",
                        "reason": "Stable linear baseline.",
                        "grid": {"C": [0.1, 1.0]},
                    },
                    {
                        "name": "logistic_elasticnet",
                        "reason": "Sparse correlated-feature baseline.",
                        "grid": {"C": [0.1], "l1_ratio": [0.5]},
                    },
                    {
                        "name": "hist_gradient_boosting",
                        "reason": "Native missingness and interactions baseline.",
                        "grid": {"max_leaf_nodes": [7], "learning_rate": [0.03]},
                    },
                ],
            }))
            config_path = root / workflow.MODEL_CONFIG_NAME

            workflow.compile_model_config(
                advice_path,
                config_path,
                inventory_path,
                workflow.MODEL_POLICY_PATH,
                workflow.MODEL_POLICY_CONTRACT,
                workflow.MODEL_CONFIG_SCHEMA_VERSION,
                workflow.PROJECT_ROOT,
            )

            config = json.loads(config_path.read_text())
            self.assertEqual(config["schema_version"], 2)
            self.assertEqual(config["candidates"][0]["role"], "baseline")
            self.assertFalse(config["candidates"][0]["native_nan"])
            self.assertTrue(config["candidates"][2]["native_nan"])
            self.assertIsNone(config["candidates"][2]["requires_package"])
            workflow.validate_model_config(config_path, inventory_path)

    def test_validate_model_config_rejects_missing_baseline_and_stale_hash(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            workflow.PROJECT_ROOT = root
            inventory_path = root / "feature_inventory.json"
            inventory_path.write_text('{"schema_version": 2}')
            config_path = root / workflow.MODEL_CONFIG_NAME
            config = self._valid_model_config(inventory_path)
            config["candidates"] = [
                row for row in config["candidates"]
                if row["name"] != "logistic_l2"
            ]
            config_path.write_text(json.dumps(config))
            with self.assertRaisesRegex(SystemExit, "missing required baselines"):
                workflow.validate_model_config(config_path, inventory_path)

            config = self._valid_model_config(inventory_path)
            config["feature_inventory_sha256"] = "0" * 64
            config_path.write_text(json.dumps(config))
            with self.assertRaisesRegex(SystemExit, "is stale"):
                workflow.validate_model_config(config_path, inventory_path)

    def test_validate_model_config_rejects_unsafe_or_excessive_extensions(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            workflow.PROJECT_ROOT = root
            inventory_path = root / "feature_inventory.json"
            inventory_path.write_text('{"schema_version": 2}')
            config_path = root / workflow.MODEL_CONFIG_NAME

            config = self._valid_model_config(inventory_path)
            config["candidates"][0]["grid"]["class_path"] = ["evil.Model"]
            config_path.write_text(json.dumps(config))
            with self.assertRaisesRegex(SystemExit, "disallowed parameters"):
                workflow.validate_model_config(config_path, inventory_path)

            config = self._valid_model_config(inventory_path)
            for name in ("extra_trees", "random_forest"):
                config["candidates"].append({
                    "name": name,
                    "role": "optional",
                    "reason": "Test bounded extension count.",
                    "grid": {"n_estimators": [100], "min_samples_leaf": [5]},
                    "native_nan": False,
                    "requires_package": None,
                })
            config_path.write_text(json.dumps(config))
            with self.assertRaisesRegex(SystemExit, "maximum is 2"):
                workflow.validate_model_config(config_path, inventory_path)

    def test_validate_model_config_rejects_false_metadata_and_bad_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            workflow.PROJECT_ROOT = root
            inventory_path = root / "feature_inventory.json"
            inventory_path.write_text('{"schema_version": 2}')
            config_path = root / workflow.MODEL_CONFIG_NAME

            self.assertEqual(
                set(workflow.MODEL_PARAMETER_ALLOWLIST),
                workflow.ALLOWED_MODEL_FAMILIES,
            )
            self.assertEqual(
                set(workflow.MODEL_NATIVE_NAN),
                workflow.ALLOWED_MODEL_FAMILIES,
            )

            config = self._valid_model_config(inventory_path)
            config["candidates"][0]["native_nan"] = True
            config_path.write_text(json.dumps(config))
            with self.assertRaisesRegex(SystemExit, "native_nan must be False"):
                workflow.validate_model_config(config_path, inventory_path)

            config = self._valid_model_config(inventory_path)
            config["candidates"][0]["grid"]["C"] = [None, "large"]
            config_path.write_text(json.dumps(config))
            with self.assertRaisesRegex(SystemExit, "has invalid values"):
                workflow.validate_model_config(config_path, inventory_path)

            config = self._valid_model_config(inventory_path)
            config["candidates"][0]["name"] = "arbitrary.Estimator"
            config_path.write_text(json.dumps(config))
            with self.assertRaisesRegex(SystemExit, "disallowed family"):
                workflow.validate_model_config(config_path, inventory_path)

    def test_model_contract_accepts_hgb_depth_and_integer_ebm_interactions(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            workflow.PROJECT_ROOT = root
            inventory_path = root / "feature_inventory.json"
            inventory_path.write_text('{"schema_version": 2}')
            config_path = root / workflow.MODEL_CONFIG_NAME
            config = self._valid_model_config(inventory_path)
            config["candidates"].append({
                "name": "explainable_boosting",
                "role": "optional",
                "reason": "Model a bounded number of pairwise interactions.",
                "grid": {
                    "max_bins": [128],
                    "interactions": [0, 5],
                    "learning_rate": [0.01],
                    "max_rounds": [2000],
                    "min_samples_leaf": [10],
                },
                "native_nan": True,
                "requires_package": "interpret",
            })
            config_path.write_text(json.dumps(config))
            workflow.validate_model_config(config_path, inventory_path)

            config["candidates"][-1]["grid"]["interactions"] = [-1, 0.5]
            config_path.write_text(json.dumps(config))
            with self.assertRaisesRegex(SystemExit, "interactions has invalid values"):
                workflow.validate_model_config(config_path, inventory_path)

    def test_validate_model_config_rejects_grid_missing_runtime_parameters(
        self,
    ) -> None:
        # Regression: the split-error-794495 run shipped an EBM grid without
        # max_rounds; build_estimator KeyErrored on every fold and the family
        # silently filled 0 OOF rows. The allowlist alone cannot catch this.
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            workflow.PROJECT_ROOT = root
            inventory_path = root / "feature_inventory.json"
            inventory_path.write_text('{"schema_version": 2}')
            config_path = root / workflow.MODEL_CONFIG_NAME
            config = self._valid_model_config(inventory_path)
            config["candidates"].append({
                "name": "explainable_boosting",
                "role": "optional",
                "reason": "Exercise the missing-required-parameter guard.",
                "grid": {
                    "max_bins": [128],
                    "interactions": [0, 5],
                    "learning_rate": [0.01],
                    "min_samples_leaf": [10],
                },
                "native_nan": True,
                "requires_package": "interpret",
            })
            config_path.write_text(json.dumps(config))
            with self.assertRaisesRegex(
                SystemExit,
                "missing parameters the assembled runtime requires: max_rounds",
            ):
                workflow.validate_model_config(config_path, inventory_path)

    def test_required_grid_parameters_match_the_runtime_template(self) -> None:
        import ast

        template_tree = ast.parse(workflow.RUNTIME_TEMPLATE_PATH.read_text())
        embedded = None
        for node in template_tree.body:
            if (isinstance(node, ast.Assign)
                    and any(getattr(t, "id", None) == "EMBEDDED_MODEL_POLICY"
                            for t in node.targets)):
                embedded = ast.literal_eval(node.value)
        self.assertIsNotNone(embedded, "template lost EMBEDDED_MODEL_POLICY")

        template_required = {
            name: frozenset(spec["required_parameters"])
            for name, spec in embedded["families"].items()
        }
        self.assertEqual(template_required, dict(workflow.REQUIRED_GRID_PARAMETERS))
        for name, required in workflow.REQUIRED_GRID_PARAMETERS.items():
            self.assertLessEqual(
                required, workflow.MODEL_PARAMETER_ALLOWLIST[name], name)

    def test_model_policy_cannot_introduce_an_unreviewed_adapter(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            policy_path = Path(tmpdir) / "policy.json"
            policy = json.loads(workflow.MODEL_POLICY_PATH.read_text())
            policy["families"]["arbitrary_estimator"] = {
                "required": False,
                "native_nan": False,
                "requires_package": None,
                "parameters": {"C": "positive_number"},
            }
            policy["selection"]["simplicity_order"].append("arbitrary_estimator")
            policy_path.write_text(json.dumps(policy))
            with self.assertRaisesRegex(RuntimeError, "without reviewed safe adapters"):
                workflow._load_model_policy(policy_path)

    def test_validate_inventory_checks_selected_source_hash(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            workflow.PROJECT_ROOT = root
            rerun = root / "run.json.predictive.rerun"
            rerun.mkdir()
            source = rerun / "hypo_1.py"
            source.write_text("FEATURE = 1\n")
            source_rel = workflow._rel_to_root(source)
            source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
            inventory_path = root / "feature_inventory.json"
            inventory = {
                "schema_version": 2,
                "selection_manifest": "run.predictive-selection.json",
                "selected_ids": [1],
                "hypotheses": [{
                    "id": 1,
                    "included": True,
                    "exclusion_reason": None,
                    "reproduction_status": "REPRODUCED",
                    "statistical_verdict": "SOUND",
                    "corrected_result_status": None,
                    "post_correction_verdict": None,
                    "correction_scope": "none",
                    "rerun_path": source_rel,
                    "rerun_sha256": source_hash,
                    "fixed_path": None,
                    "fixed_sha256": None,
                    "feature_source": "rerun",
                    "feature_source_path": source_rel,
                    "feature_source_sha256": source_hash,
                    "source_reason": "No correction was required.",
                    "features": [{
                        "name": "feature",
                        "quantity": "one",
                        "constants": {},
                        "aggregation": "segment",
                        "reduction": "identity",
                        "traversal_phase": "segment",
                        "measurable_condition": "always",
                        "historical_undefined_sentinel": None,
                    }],
                }],
            }
            inventory_path.write_text(json.dumps(inventory))

            workflow.validate_inventory(
                inventory_path,
                [1],
                "run.predictive-selection.json",
                rerun,
                None,
            )
            inventory["selection_manifest"] = {
                "path": "run.predictive-selection.json",
                "sha256": "0" * 64,
            }
            inventory_path.write_text(json.dumps(inventory))
            with self.assertRaisesRegex(
                SystemExit,
                r"expected 'run\.predictive-selection\.json'.*got .*dict",
            ):
                workflow.validate_inventory(
                    inventory_path,
                    [1],
                    "run.predictive-selection.json",
                    rerun,
                    None,
                )
            inventory["selection_manifest"] = "run.predictive-selection.json"
            inventory["hypotheses"][0]["corrected_result_status"] = "UPHELD"
            inventory_path.write_text(json.dumps(inventory))
            with self.assertRaisesRegex(
                SystemExit,
                "corrected_result_status describes execution/measurement",
            ):
                workflow.validate_inventory(
                    inventory_path,
                    [1],
                    "run.predictive-selection.json",
                    rerun,
                    None,
                )
            inventory["hypotheses"][0]["corrected_result_status"] = None
            inventory["hypotheses"][0]["correction_scope"] = "unclear"
            inventory_path.write_text(json.dumps(inventory))
            with self.assertRaisesRegex(SystemExit, "cannot be included"):
                workflow.validate_inventory(
                    inventory_path,
                    [1],
                    "run.predictive-selection.json",
                    rerun,
                    None,
                )
            inventory["hypotheses"][0]["correction_scope"] = "none"
            inventory["hypotheses"][0]["feature_source_sha256"] = "0" * 64
            inventory_path.write_text(json.dumps(inventory))
            with self.assertRaisesRegex(SystemExit, "source SHA-256 is stale"):
                workflow.validate_inventory(
                    inventory_path,
                    [1],
                    "run.predictive-selection.json",
                    rerun,
                    None,
                )

    def test_validate_inventory_cross_checks_explicit_report_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            workflow.PROJECT_ROOT = root
            rerun = root / "run.json.predictive.rerun"
            rerun.mkdir()
            source = rerun / "hypo_23.py"
            source.write_text("FEATURE = 1\n")
            source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
            summary = root / "run.summary.md"
            summary.write_text(
                "### 2. Display rank is not the ID\n"
                "- **Run:** run · **ID:** 23 · **Direction:** Negative\n"
                "- **Reproduction:** REPRODUCED\n"
                "- **Verdict:** MINOR\n"
                "- **Post-correction verdict:** UPHELD — survives\n"
            )
            inventory_path = root / "feature_inventory.json"
            inventory = {
                "schema_version": 2,
                "selection_manifest": None,
                "selected_ids": [23],
                "hypotheses": [{
                    "id": 23,
                    "included": False,
                    "exclusion_reason": "Test fixture exclusion.",
                    "reproduction_status": "REPRODUCED",
                    "statistical_verdict": "MAJOR",
                    "corrected_result_status": "USABLE",
                    "post_correction_verdict": "UPHELD",
                    "correction_scope": "unclear",
                    "rerun_path": workflow._rel_to_root(source),
                    "rerun_sha256": source_hash,
                    "fixed_path": None,
                    "fixed_sha256": None,
                    "feature_source": None,
                    "feature_source_path": None,
                    "feature_source_sha256": None,
                    "source_reason": "Correction source is unavailable.",
                    "features": [],
                }],
            }
            inventory_path.write_text(json.dumps(inventory))

            with self.assertRaisesRegex(
                SystemExit, "statistical_verdict.*same-ID report row"
            ):
                workflow.validate_inventory(
                    inventory_path, [23], None, rerun, None, summary
                )

            inventory["hypotheses"][0]["statistical_verdict"] = "MINOR"
            inventory_path.write_text(json.dumps(inventory))
            workflow.validate_inventory(
                inventory_path, [23], None, rerun, None, summary
            )

            corrected = root / "run.json.corrected.json"
            corrected.write_text(json.dumps({
                "code_dir": workflow._rel_to_root(rerun),
                "corrected_dir": None,
                "results": [{"id": 23, "result_status": "FAILED"}],
            }))
            with self.assertRaisesRegex(
                SystemExit, "corrected_result_status.*corrected results"
            ):
                workflow.validate_inventory(
                    inventory_path, [23], None, rerun, None, summary, corrected
                )

    def test_semantic_draft_compiles_to_compatible_inventory_v2(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            workflow.PROJECT_ROOT = root
            rerun = root / "run.json.rerun"
            rerun.mkdir()
            source = rerun / "hypo_1.py"
            source.write_text("FEATURE = 1\n")
            summary = root / "run.summary.md"
            summary.write_text(
                "### 8. Display rank\n"
                "- **Run:** run · **ID:** 1 · **Direction:** Positive\n"
                "- **Reproduction:** REPRODUCED\n"
                "- **Verdict:** SOUND\n"
            )
            semantics = root / ".feature_semantics.json"
            semantics.write_text(json.dumps({
                "schema_version": 1,
                "hypotheses": [{
                    "id": 1,
                    "included": True,
                    "exclusion_reason": None,
                    "correction_scope": "none",
                    "feature_source": "rerun",
                    "source_reason": "Original feature semantics are usable.",
                    "features": [{
                        "name": "feature",
                        "quantity": "one",
                        "constants": {},
                        "aggregation": "segment",
                        "reduction": "identity",
                        "traversal_phase": "segment",
                        "measurable_condition": "always",
                        "historical_undefined_sentinel": None,
                    }],
                }],
            }))
            inventory_path = root / "feature_inventory.json"

            workflow.compile_feature_inventory(
                semantics,
                inventory_path,
                selected_ids=[1],
                selection_manifest=None,
                summary_path=summary,
                corrected_results_path=None,
                rerun_dir=rerun,
                fixed_dir=None,
                project_root=root,
            )

            inventory = json.loads(inventory_path.read_text())
            self.assertEqual(set(inventory), {
                "schema_version", "selection_manifest", "selected_ids",
                "hypotheses",
            })
            row = inventory["hypotheses"][0]
            self.assertEqual(row["reproduction_status"], "REPRODUCED")
            self.assertEqual(row["statistical_verdict"], "SOUND")
            self.assertEqual(row["feature_source_path"], "run.json.rerun/hypo_1.py")
            self.assertEqual(row["feature_source_sha256"], hashlib.sha256(
                source.read_bytes()
            ).hexdigest())
            workflow.validate_inventory(
                inventory_path, [1], None, rerun, None, summary
            )

    def test_reconciled_unbacked_verdict_compiles_null_and_validates(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            workflow.PROJECT_ROOT = root
            rerun = root / "run.json.rerun"
            rerun.mkdir()
            (rerun / "hypo_1.py").write_text("FEATURE = 1\n")
            (rerun / "hypo_2.py").write_text("FEATURE = 2\n")
            fixed = root / "run.json.fixed"
            fixed.mkdir()
            (fixed / "hypo_1.py").write_text("FEATURE = 1  # corrected\n")
            summary = root / "run.summary.md"
            summary.write_text(
                "### 1. Unbacked correction\n"
                "- **Run:** run · **ID:** 1 · **Direction:** Positive\n"
                "- **Reproduction:** DIVERGED\n"
                "- **Verdict:** MAJOR\n"
                "- **Post-correction verdict:** INCONCLUSIVE\n"
                "### 2. Clean finding\n"
                "- **Run:** run · **ID:** 2 · **Direction:** Positive\n"
                "- **Reproduction:** REPRODUCED\n"
                "- **Verdict:** SOUND\n"
            )
            corrected = root / "run.json.corrected.json"
            corrected.write_text(json.dumps({
                "code_dir": "run.json.rerun",
                "corrected_dir": "run.json.fixed",
                "results": [{"id": 1, "result_status": "UNUSABLE"}],
            }))
            semantics = root / ".feature_semantics.json"
            semantics.write_text(json.dumps({
                "schema_version": 1,
                "hypotheses": [
                    {
                        "id": 1,
                        "included": False,
                        "exclusion_reason": (
                            "Corrected measurement unusable (timeout); the "
                            "corrected feature definition cannot be validated."
                        ),
                        "correction_scope": "unclear",
                        "feature_source": None,
                        "source_reason": None,
                        "features": [],
                    },
                    {
                        "id": 2,
                        "included": True,
                        "exclusion_reason": None,
                        "correction_scope": "none",
                        "feature_source": "rerun",
                        "source_reason": "Original feature semantics are usable.",
                        "features": [{
                            "name": "feature",
                            "quantity": "one",
                            "constants": {},
                            "aggregation": "segment",
                            "reduction": "identity",
                            "traversal_phase": "segment",
                            "measurable_condition": "always",
                            "historical_undefined_sentinel": None,
                        }],
                    },
                ],
            }))
            inventory_path = root / "feature_inventory.json"

            workflow.compile_feature_inventory(
                semantics,
                inventory_path,
                selected_ids=[1, 2],
                selection_manifest=None,
                summary_path=summary,
                corrected_results_path=corrected,
                rerun_dir=rerun,
                fixed_dir=fixed,
                project_root=root,
                reconcile_unbacked_verdicts=True,
            )

            rows = json.loads(inventory_path.read_text())["hypotheses"]
            self.assertIsNone(rows[0]["post_correction_verdict"])
            self.assertEqual(rows[0]["corrected_result_status"], "UNUSABLE")

            # The raw report still says INCONCLUSIVE, so validation only passes
            # when it applies the same reconciliation as the compiler.
            with self.assertRaisesRegex(SystemExit, "does not match"):
                workflow.validate_inventory(
                    inventory_path, [1, 2], None, rerun, fixed, summary, corrected
                )
            workflow.validate_inventory(
                inventory_path, [1, 2], None, rerun, fixed, summary, corrected,
                reconcile_unbacked_verdicts=True,
            )

    def test_included_empty_exclusion_reason_is_canonicalized_to_null(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            workflow.PROJECT_ROOT = root
            rerun = root / "run.json.rerun"
            rerun.mkdir()
            (rerun / "hypo_1.py").write_text("FEATURE = 1\n")
            summary = root / "run.summary.md"
            summary.write_text(
                "### 1. Finding\n"
                "- **Run:** run · **ID:** 1 · **Direction:** Positive\n"
                "- **Reproduction:** REPRODUCED\n"
                "- **Verdict:** SOUND\n"
            )
            semantics = root / ".feature_semantics.json"
            semantics.write_text(json.dumps({
                "schema_version": 1,
                "hypotheses": [{
                    "id": 1,
                    "included": True,
                    "exclusion_reason": "",
                    "correction_scope": "none",
                    "feature_source": "rerun",
                    "source_reason": "Usable source.",
                    "features": [{
                        "name": "feature",
                        "quantity": "one",
                        "constants": {},
                        "aggregation": "segment",
                        "reduction": "identity",
                        "traversal_phase": "segment",
                        "measurable_condition": "always",
                        "historical_undefined_sentinel": None,
                    }],
                }],
            }))
            inventory_path = root / "feature_inventory.json"

            changes = workflow.compile_feature_inventory(
                semantics,
                inventory_path,
                selected_ids=[1],
                selection_manifest=None,
                summary_path=summary,
                corrected_results_path=None,
                rerun_dir=rerun,
                fixed_dir=None,
                project_root=root,
            )

            self.assertEqual(len(changes), 1)
            self.assertIsNone(json.loads(
                semantics.read_text())["hypotheses"][0]["exclusion_reason"])
            self.assertIsNone(json.loads(
                inventory_path.read_text())["hypotheses"][0]["exclusion_reason"])

    def test_finalize_step_repairs_rejected_draft_in_same_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            workflow.PROJECT_ROOT = root
            draft = root / "draft.json"
            draft.write_text("bad")
            original_run_step = workflow.run_step
            calls = []

            async def fake_run_step(_client, repair_step, _verbose, _costs):
                calls.append(repair_step)
                draft.write_text("good")
                return "repaired"

            def finalize() -> None:
                if draft.read_text() != "good":
                    raise SystemExit("draft must say good")

            try:
                workflow.run_step = fake_run_step
                asyncio.run(workflow.finalize_step_with_repairs(
                    object(),
                    {"name": "stage", "expects_file": "draft.json"},
                    finalize,
                    lambda: None,
                    False,
                    workflow.WorkflowCostSummary(),
                ))
            finally:
                workflow.run_step = original_run_step

            self.assertEqual(len(calls), 1)
            self.assertIn("draft must say good", calls[0]["instruction"])
            self.assertEqual(calls[0]["name"], "stage-repair-1")

    def test_synthetic_smoke_uses_one_grid_point_per_family(self) -> None:
        template = (PROJECT_ROOT / "agentic" / "detector_build" / "templates"
                    / "detector_runtime.py.tmpl").read_text()
        self.assertIn("smoke_configured", template)
        self.assertIn("values[0]", template)
        self.assertIn("one grid point per available family", template)

    def test_excluded_semantics_are_canonicalized_without_agent_retry(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            workflow.PROJECT_ROOT = root
            rerun = root / "run.json.rerun"
            fixed = root / "run.json.fixed"
            rerun.mkdir()
            fixed.mkdir()
            (rerun / "hypo_5.py").write_text("FEATURE = 1\n")
            (fixed / "hypo_5.py").write_text("FEATURE = 2\n")
            summary = root / "run.summary.md"
            summary.write_text(
                "### 1. Excluded\n"
                "- **Run:** run · **ID:** 5 · **Direction:** Negative\n"
                "- **Reproduction:** REPRODUCED\n"
                "- **Verdict:** CRITICAL\n"
                "- **Post-correction verdict:** OVERTURNED — effect vanished\n"
            )
            corrected = root / "run.json.corrected.json"
            corrected.write_text(json.dumps({
                "code_dir": workflow._rel_to_root(rerun),
                "corrected_dir": workflow._rel_to_root(fixed),
                "results": [{"id": 5, "result_status": "USABLE"}],
            }))
            semantics = root / ".feature_semantics.json"
            semantics.write_text(json.dumps({
                "schema_version": 1,
                "hypotheses": [{
                    "id": 5,
                    "included": False,
                    "exclusion_reason": "Corrected result overturned the claim.",
                    "correction_scope": "none",
                    "feature_source": None,
                    "source_reason": None,
                    "features": [],
                }],
            }))
            inventory_path = root / "feature_inventory.json"

            workflow.compile_feature_inventory(
                semantics,
                inventory_path,
                selected_ids=[5],
                selection_manifest=None,
                summary_path=summary,
                corrected_results_path=corrected,
                rerun_dir=rerun,
                fixed_dir=fixed,
                project_root=root,
            )

            row = json.loads(inventory_path.read_text())["hypotheses"][0]
            self.assertEqual(row["correction_scope"], "unclear")
            self.assertEqual(
                row["source_reason"],
                "Excluded: Corrected result overturned the claim.",
            )
            workflow.validate_inventory(
                inventory_path, [5], None, rerun, fixed, summary, corrected
            )

    def test_manifest_rejects_id_filename_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "MANIFEST.json"
            path.write_text(json.dumps({
                "records": [{"id": 23, "file": "hypo_2.py"}],
            }))
            with self.assertRaisesRegex(SystemExit, "maps id 23"):
                workflow._manifest_ids(path)

    def test_validate_detector_source_and_immutable_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            workflow.PROJECT_ROOT = root
            detector = root / "detector.py"
            detector.write_text("def main():\n    return 0\n")
            workflow.validate_detector_source(detector)
            original_hash = hashlib.sha256(detector.read_bytes()).hexdigest()
            workflow.require_unchanged(detector, original_hash, "generate")
            detector.write_text("def broken(:\n")
            with self.assertRaisesRegex(SystemExit, "does not parse"):
                workflow.validate_detector_source(detector)
            with self.assertRaisesRegex(SystemExit, "modified validated"):
                workflow.require_unchanged(detector, original_hash, "generate")

    def test_driver_executes_help_and_synthetic_smoke_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            detector = root / "detector.py"
            model_config = root / "model_candidates.json"
            model_config.write_text(json.dumps({
                "candidates": [{"name": "logistic_l2"}],
            }))
            detector.write_text(
                "import argparse, json\n"
                "p = argparse.ArgumentParser()\n"
                "p.add_argument('pkl', nargs='?')\n"
                "p.add_argument('--model-config')\n"
                "p.add_argument('--measuretime', action='store_true')\n"
                "p.add_argument('--measuretime-occurrences', type=int, default=3)\n"
                "p.add_argument('--hypothesis-selection')\n"
                "p.add_argument('--exclude-hypotheses', nargs='+', type=int)\n"
                "p.add_argument('--synthetic-smoke-test', action='store_true')\n"
                "args = p.parse_args()\n"
                "if args.synthetic_smoke_test:\n"
                "    assert args.pkl is None\n"
                "    cfg = json.load(open(args.model_config))\n"
                "    if cfg['candidates'][0]['name'].startswith('__'):\n"
                "        p.error('unreviewed family')\n"
                "    print('DETECTOR_SMOKE_OK')\n"
            )

            workflow.validate_detector_executable(detector, model_config)

    def test_driver_rejects_smoke_without_success_marker(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            detector = root / "detector.py"
            model_config = root / "model_candidates.json"
            model_config.write_text(json.dumps({
                "candidates": [{"name": "logistic_l2"}],
            }))
            detector.write_text(
                "import argparse\n"
                "p = argparse.ArgumentParser()\n"
                "p.add_argument('--model-config')\n"
                "p.add_argument('--measuretime', action='store_true')\n"
                "p.add_argument('--measuretime-occurrences', type=int, default=3)\n"
                "p.add_argument('--hypothesis-selection')\n"
                "p.add_argument('--exclude-hypotheses', nargs='+', type=int)\n"
                "p.add_argument('--synthetic-smoke-test', action='store_true')\n"
                "p.parse_args()\n"
            )

            with self.assertRaisesRegex(SystemExit, "required.*marker"):
                workflow.validate_detector_executable(detector, model_config)


if __name__ == "__main__":
    unittest.main()
