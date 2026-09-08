"""
Blindness-gate tests: the layers that keep GT-referencing quantities out of
detector feature math.

  * layer 2 (static): assembly._validate_feature_no_gt_access now runs for BOTH
    targets (the merge image-only-v2 build shipped fragment_to_gt_distance
    precisely because this lint used to be split-only);
  * layer 1 (runtime): the detector runtime template's _BlindPayloadView makes a
    GT payload read raise at extraction time, hands feature math audit-stripped
    samples and zeroed labels;
  * layer 3 (workflow): the verifier's Deployability token is parsed from report
    rows (detector-build hard-excludes GT-REFERENCING) and scored by the heatmap;
  * layer 4 (audit): the target adapters expose ascertainment covariates
    (GT/size; audit-only) for the confound audit.
"""

from __future__ import annotations

import ast
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path

import networkx as nx
import numpy as np

from agentic.detector_build.assembly import _validate_feature_no_gt_access
from agentic.detector_build.contracts import DetectorTarget
from agentic.detector_build.inputs import report_evidence_by_id
from agentic.detector_build.target_runtime import target_adapter_source

REPO = Path(__file__).resolve().parent.parent.parent
TEMPLATE = (REPO / "agentic" / "detector_build" / "templates"
            / "detector_runtime.py.tmpl")

SPLIT_POLICY = {
    "config_id": "tip_to_any_node|r=50|k=2",
    "mode": "tip_to_any_node",
    "radius_um": 50,
    "per_anchor_k": 2,
    "global_cap": None,
}


def _lint(source: str, target: DetectorTarget) -> None:
    _validate_feature_no_gt_access(
        ast.parse(source), Path("fragment.py"), target)


class GtLintTests(unittest.TestCase):
    """Layer 2: the static gate rejects GT access for BOTH targets."""

    def test_rejects_payload_gt_subscript_for_both_targets(self) -> None:
        src = "def f(payload):\n    return payload['gt_graph']\n"
        for target in (DetectorTarget.MERGE, DetectorTarget.SPLIT):
            with self.assertRaises(SystemExit):
                _lint(src, target)

    def test_rejects_aliased_owner_and_get_calls(self) -> None:
        src = ("def f(payload):\n"
               "    p = payload\n"
               "    return p.get('gt_merge_labels', [])\n")
        with self.assertRaises(SystemExit):
            _lint(src, DetectorTarget.MERGE)

    def test_rejects_forbidden_runtime_names(self) -> None:
        for name in ("_derive_split_truth", "_gt_neuron_membership",
                     "ascertainment_covariates"):
            with self.assertRaises(SystemExit):
                _lint(f"def f(payload):\n    return {name}(payload)\n",
                      DetectorTarget.MERGE)

    def test_rejects_merge_label_sample_key(self) -> None:
        with self.assertRaises(SystemExit):
            _lint("def f(row):\n    return row['is_merge']\n",
                  DetectorTarget.MERGE)
        # split keeps its own audit-key set
        with self.assertRaises(SystemExit):
            _lint("def f(row):\n    return row['is_merge_creating']\n",
                  DetectorTarget.SPLIT)

    def test_accepts_clean_fragment(self) -> None:
        src = ("def extract_features(payload):\n"
               "    frag = payload['fragments_graph']\n"
               "    samples, labels = build_sample_universe(payload)\n"
               "    return samples, labels, frag\n")
        for target in (DetectorTarget.MERGE, DetectorTarget.SPLIT):
            _lint(src, target)  # must not raise


class BlindPayloadViewTests(unittest.TestCase):
    """Layer 1: the runtime guard extracted from the detector template."""

    @classmethod
    def setUpClass(cls) -> None:
        text = TEMPLATE.read_text(encoding="utf-8")
        start = text.index("# --- blindness guard")
        end = text.index("# --- end blindness guard ---")
        cls.ns: dict = {"np": np}
        exec(text[start:end], cls.ns)  # noqa: S102 — template's own code

    def _view(self):
        payload = {"fragments_graph": "FRAG", "gt_graph": object(),
                   "gt_merge_labels": [7], "min_cable_length": 100}
        samples = [{"segment_id_a": 1, "segment_id_b": 2, "gap_um": 1.5,
                    "is_merge_creating": 1, "contains_known_merge_segment": 0,
                    "occurrences": ()}]
        return self.ns["_BlindPayloadView"](payload, samples), payload

    def test_gt_reads_raise_and_membership_hides(self) -> None:
        view, _ = self._view()
        violation = self.ns["BlindnessViolation"]
        for key in ("gt_graph", "gt_merge_labels", "gt_node_canonical_label",
                    "gt_edge_error"):
            self.assertNotIn(key, view)
            with self.assertRaises(violation):
                view[key]
            with self.assertRaises(violation):
                view.get(key)

    def test_non_gt_keys_pass_through(self) -> None:
        view, _ = self._view()
        self.assertEqual(view["fragments_graph"], "FRAG")
        self.assertEqual(view.get("min_cable_length", 0), 100)
        self.assertIn("fragments_graph", view)
        self.assertNotIn("gt_graph", list(view))

    def test_universe_cache_is_stripped_and_zero_labeled(self) -> None:
        view, _ = self._view()
        samples, labels = view["__detector_sample_universe_cache__"]
        self.assertFalse(np.any(labels))
        self.assertNotIn("is_merge_creating", samples[0])
        self.assertNotIn("contains_known_merge_segment", samples[0])
        self.assertEqual(samples[0]["gap_um"], 1.5)  # geometry retained

    def test_gt_writes_raise(self) -> None:
        view, payload = self._view()
        with self.assertRaises(self.ns["BlindnessViolation"]):
            view["gt_merge_labels"] = []
        view["scratch"] = 1                    # non-GT writes pass through
        self.assertEqual(payload["scratch"], 1)

    def test_template_parses_and_carries_new_runtime_pieces(self) -> None:
        tree = ast.parse(TEMPLATE.read_text(encoding="utf-8"))
        names = {n.name for n in ast.walk(tree)
                 if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
        self.assertIn("_BlindPayloadView", names)
        self.assertIn("BlindnessViolation", names)
        self.assertIn("ascertainment_audit", names)


class DeployabilityTokenTests(unittest.TestCase):
    """Layer 3: report parsing and heatmap vocabulary."""

    def test_report_evidence_parses_deployability(self) -> None:
        report = "\n".join([
            "## Ranked Conclusions (highest priority first)",
            "### 1. (Priority 0.3 · Surprise 0.1) gt distance",
            "- **Run:** r · **ID:** 55 · x",
            "- **Verdict:** SOUND",
            "- **Deployability:** GT-REFERENCING — distance to gt_graph nodes",
            "### 2. (Priority 0.2 · Surprise 0.1) tubularity",
            "- **Run:** r · **ID:** 7 · x",
            "- **Verdict:** WEAK",
            "- **Deployability:** BLIND-COMPUTABLE — image intensity only",
            "### 3. (Priority 0.1 · Surprise 0.1) legacy entry",
            "- **Run:** r · **ID:** 9 · x",
            "- **Verdict:** MINOR",
            "",
        ])
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "run.summary.md"
            path.write_text(report, encoding="utf-8")
            evidence = report_evidence_by_id(path, Path(td))
        self.assertEqual(evidence[55]["deployability"], "GT-REFERENCING")
        self.assertEqual(evidence[7]["deployability"], "BLIND-COMPUTABLE")
        self.assertIsNone(evidence[9]["deployability"])
        # the hyphen-widened regex must not distort the old verdict fields
        self.assertEqual(evidence[55]["statistical_verdict"], "SOUND")

    def test_heatmap_scores_deployability(self) -> None:
        from agentic.summary_heatmap import DEPLOYABILITY, ROWS
        self.assertEqual(DEPLOYABILITY["blind-computable"], 1.0)
        self.assertEqual(DEPLOYABILITY["gt-referencing"], 0.0)
        self.assertIn("Deployability", [row[0] for row in ROWS])


class _Graph(nx.Graph):
    pass


def _adapter_namespace(target: DetectorTarget) -> dict:
    namespace = {"np": np, "defaultdict": defaultdict}

    def build_comp_to_seg(graph):
        return {
            int(component): int(str(swc).split(".")[0])
            for component, swc in graph.component_id_to_swc_id.items()
        }

    namespace["build_comp_to_seg"] = build_comp_to_seg
    kwargs = ({"candidate_policy": SPLIT_POLICY,
               "candidate_policy_sha256": "a" * 64}
              if target is DetectorTarget.SPLIT else {})
    exec(target_adapter_source(target, **kwargs), namespace)  # noqa: S102
    return namespace


def _payload() -> dict:
    graph = _Graph()
    graph.add_edges_from([(0, 1), (2, 3)])
    graph.node_xyz = np.asarray([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0], [3.0, 0.0, 0.0],
    ])
    graph.node_component_id = np.asarray([10, 10, 20, 20])
    graph.component_id_to_swc_id = {10: "1.0.swc", 20: "2.0.swc"}
    return {
        "fragments_graph": graph,
        # nodes 0,1 belong to GT segment 1; node 2 to segment 2; node 3 untraced
        "gt_node_canonical_label": np.asarray([1, 1, 2, 0]),
    }


class AscertainmentCovariateTests(unittest.TestCase):
    """Layer 4: audit-only GT/size covariates from the target adapters."""

    def test_merge_covariates_count_fragment_and_gt_nodes(self) -> None:
        ns = _adapter_namespace(DetectorTarget.MERGE)
        covs = ns["ascertainment_covariates"](_payload(), [1, 2])
        np.testing.assert_array_equal(
            covs["segment_fragment_node_count"], [2.0, 2.0])
        np.testing.assert_array_equal(
            covs["segment_gt_node_count"], [2.0, 1.0])

    def test_split_covariates_take_pair_minima(self) -> None:
        ns = _adapter_namespace(DetectorTarget.SPLIT)
        samples = [{"segment_id_a": 1, "segment_id_b": 2, "gap_um": 2.5}]
        covs = ns["ascertainment_covariates"](_payload(), samples)
        np.testing.assert_array_equal(covs["gap_um"], [2.5])
        np.testing.assert_array_equal(
            covs["min_segment_fragment_node_count"], [2.0])
        np.testing.assert_array_equal(covs["min_segment_gt_node_count"], [1.0])


if __name__ == "__main__":
    unittest.main()
