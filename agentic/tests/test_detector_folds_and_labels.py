"""The runtime fits on GT-judgeable rows with purged group folds and scores every row.

The template's selection functions run on a synthetic split universe: labels are
1 / 0 / NaN, folds come from candidate_rules.fold_partitions through the split adapter,
and no real cache, detector or API is touched.
"""

import ast
import time
from collections import defaultdict
from pathlib import Path
import unittest

import numpy as np

from agentic.detector_build.contracts import DetectorTarget
from agentic.tests.candidate_rule_fixtures import adapter_namespace, split_rule, RULES_PATH


TEMPLATE = Path(__file__).resolve().parents[1] / "detector_build/templates/detector_runtime.py.tmpl"
FUNCTIONS = {
    "_judged", "_cv_splits", "inner_cv_family", "best_params_1se", "select_family_1se",
    "run_nested_selection", "fit_final_winner", "_score_matrix", "iter_param_grid",
    "_grid_size", "build_estimator", "_logistic_regularization_kwargs", "_package_available",
    "undefined_audit", "_single_feature_auc", "_fmt", "_report_thresholds",
}
POLICY = {"selection": {"outer_folds": 3, "inner_folds": 2, "random_seed": 0,
                        "simplicity_order": ["logistic_l2"]}}


def runtime():
    namespace = adapter_namespace(DetectorTarget.SPLIT, split_rule(radius_um=80.0))
    namespace.update(np=np, time=time, defaultdict=defaultdict, MODEL_FIT_JOBS=1,
                     RANDOM_SEED=0, FEATURE_NAMES=["f0", "f1"], ROW_UNIT="candidate segment pair",
                     ROW_UNIT_PLURAL="candidate segment pairs", POSITIVE_NAME="split",
                     NEGATIVE_NAME="non-split candidate", __file__=str(RULES_PATH))
    tree = ast.parse(TEMPLATE.read_text())
    functions = [node for node in tree.body
                 if isinstance(node, ast.FunctionDef) and node.name in FUNCTIONS]
    assert {node.name for node in functions} == FUNCTIONS
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(TEMPLATE), "exec"), namespace)
    return namespace


def synthetic_universe(n_blocks=9, per_block=8, seed=0):
    """Split rows in ``n_blocks`` 500 um blocks, two classes per block, unique segments."""
    rng = np.random.RandomState(seed)
    rows, labels, xyz = [], [], []
    for block in range(n_blocks):
        for i in range(per_block):
            node_a, node_b = len(xyz), len(xyz) + 1
            base = np.array([block * 500.0 + 100.0, 100.0, 100.0])
            xyz.append(base + rng.uniform(-5, 5, 3))
            xyz.append(base + rng.uniform(-5, 5, 3))
            segment_a, segment_b = 1000 + 2 * len(rows), 1001 + 2 * len(rows)
            rows.append({"candidate_id": len(rows), "segment_id_a": segment_a,
                         "segment_id_b": segment_b, "node_id_a": node_a, "node_id_b": node_b,
                         "gap_um": 3.0, "anchor_side": "a", "partner_rank": 1,
                         "node_role_a": "tip", "node_role_b": "tip", "occurrences": ()})
            labels.append(float(i % 2))
    labels = np.asarray(labels)
    labels[rng.rand(len(labels)) < 0.15] = np.nan  # rows GT cannot judge
    X = rng.randn(len(rows), 2)
    X[:, 0] += 2.0 * np.nan_to_num(labels)
    return rows, labels, np.asarray(xyz), X


class PurgedFoldsAndThreeValuedLabelsTests(unittest.TestCase):
    def setUp(self):
        self.ns = runtime()
        self.rows, self.y, self.xyz, self.X = synthetic_universe()
        self.groups = self.ns["_candidate_rules"]().fold_groups(
            "split", self.rows, self.xyz, prefix="794495")

    def test_cv_splits_hold_every_judgeable_row_once_without_segment_overlap(self):
        judged = np.flatnonzero(self.ns["_judged"](self.y))
        rows = [self.rows[i] for i in judged]
        groups = [self.groups[i] for i in judged]
        splits = self.ns["_cv_splits"](self.X[judged], self.y[judged].astype(int), 3, 0,
                                       samples=rows, groups=groups)
        held = np.concatenate([test for _, test in splits])
        self.assertEqual(sorted(held.tolist()), list(range(len(judged))))
        for train, test in splits:
            held_segments = {rows[i]["segment_id_a"] for i in test} | {rows[i]["segment_id_b"] for i in test}
            for i in train:
                self.assertNotIn(rows[i]["segment_id_a"], held_segments)
                self.assertNotIn(rows[i]["segment_id_b"], held_segments)
        # without groups the runtime falls back to stratified folds
        plain = self.ns["_cv_splits"](self.X[judged], self.y[judged].astype(int), 3, 0)
        self.assertEqual(len(plain), 3)

    def test_nested_selection_scores_only_judgeable_rows_and_final_model_scores_all(self):
        configured = {"logistic_l2": {"grid": {"C": [1.0]}, "role": "baseline"}}
        logs = []
        nested = self.ns["run_nested_selection"](
            self.X, self.y, configured, POLICY, logs.append, samples=self.rows, groups=self.groups)
        judged = self.ns["_judged"](self.y)
        oof = nested["oof"]["logistic_l2"]
        self.assertEqual(oof.shape, (len(self.y),))
        self.assertTrue(np.all(np.isfinite(oof[judged])))
        self.assertTrue(np.all(np.isnan(oof[~judged])))
        self.assertTrue(np.all(np.isnan(nested["oof_selector"][~judged])))
        self.assertEqual(nested["n_judgeable"], int(judged.sum()))
        self.assertEqual(nested["n_rows"], len(self.y))
        self.assertEqual(nested["family_metrics"]["logistic_l2"]["n_scored"], int(judged.sum()))
        self.assertTrue(any("GT-judgeable" in line for line in logs))
        winner, params, estimator, _ = self.ns["fit_final_winner"](
            self.X, self.y, configured, POLICY, logs.append, samples=self.rows, groups=self.groups)
        self.assertEqual(winner, "logistic_l2")
        scores = self.ns["_score_matrix"](estimator, self.X)
        self.assertEqual(scores.shape, (len(self.y),))
        self.assertTrue(np.all(np.isfinite(scores)))

    def test_audits_count_judgeable_rows(self):
        import io
        import pandas as pd
        from contextlib import redirect_stdout
        frame = pd.DataFrame({"f0": self.X[:, 0], "f0_is_defined": 1,
                              "f1": self.X[:, 1], "f1_is_defined": 1})
        with redirect_stdout(io.StringIO()) as out:
            audit = self.ns["undefined_audit"](frame, self.y, lambda _: None)
        judged = self.ns["_judged"](self.y)
        self.assertEqual(audit["n_judgeable"], int(judged.sum()))
        self.assertEqual(audit["n_unjudgeable"], int((~judged).sum()))
        self.assertEqual(audit["n_positive"], int(np.sum(self.y == 1)))
        self.assertAlmostEqual(audit["prevalence"], np.sum(self.y == 1) / judged.sum())
        self.assertIn("GT-judgeable", out.getvalue())
        oof = np.where(judged, self.X[:, 0], np.nan)
        with redirect_stdout(io.StringIO()) as out:
            self.ns["_report_thresholds"](self.y, oof, 10, samples=self.rows)
        self.assertIn("over the %d GT-judgeable rows" % int(judged.sum()), out.getvalue())


if __name__ == "__main__":
    unittest.main()
