"""Label-availability sidecars and three-valued (1 / 0 / NaN) labels in scoring."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from proofreader_evolve.harness import fixed_pool_scoring as scoring
from proofreader_evolve.harness import label_availability as la
from proofreader_evolve.harness.native_pool import NativeTable, NativeTables, pool_digest

SOURCE = {"path": "/x/cache.pkl", "size": 1, "mtime_ns": 2, "inode": 3}
BASELINE = "def score_candidates(features, ctx):\n    return features['detector_score'].to_numpy()\n"


def split_table(truth=(0, 1, 0, 0)):
    rows = [{"segment_id_a": 1, "segment_id_b": b} for b in (2, 3, 4, 5)]
    frame = pd.DataFrame({"detector_score": [.9, .8, .7, .1]})
    return NativeTable("split", frame, rows, np.array(truth),
                       {"pool_sha256": pool_digest("split", rows), "provenance": {"source_cache": SOURCE}})


def merge_table():
    rows = [{"segment_id": s, "node_id": i, "x_um": x, "y_um": 0., "z_um": 0.}
            for i, (s, x) in enumerate([(10, 0.), (10, 500.), (11, 0.), (12, 0.)])]
    frame = pd.DataFrame({"detector_score": [.4, .3, .2, .1]})
    return NativeTable("merge", frame, rows, np.array([1, 0, 0, 0]),
                       {"pool_sha256": pool_digest("merge", rows), "provenance": {"source_cache": SOURCE}})


# GT nodes carry segments 1, 2, 3 and 10, 11; GT nodes sit at x = 0 and x = 100.
GT_LABEL = np.array([0, 1, 2, 3, 10, 11])
GT_XYZ = np.array([[0., 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [100, 0, 0]])


class DefinitionTests(unittest.TestCase):
    def test_split_requires_both_segments_on_gt(self):
        arrays = la.compute("split", split_table().candidates, GT_LABEL)
        self.assertEqual(arrays["segment_on_gt"].tolist(), [True, True, False, False])

    def test_merge_segment_and_distance_definitions(self):
        arrays = la.compute("merge", merge_table().candidates, GT_LABEL, GT_XYZ)
        self.assertEqual(arrays["segment_on_gt"].tolist(), [True, True, True, False])
        # node 1 is on a GT segment but 400 um from the nearest GT node (> 150 um)
        self.assertEqual(arrays["near_gt_150"].tolist(), [True, False, True, False])


class ThreeValuedTests(unittest.TestCase):
    def test_one_zero_and_nan(self):
        labels = la.three_valued(np.array([1, 0, 0, 0]), np.array([True, True, False, False]))
        self.assertEqual(labels[:2].tolist(), [1., 0.])
        self.assertTrue(np.isnan(labels[2:]).all())

    def test_contradiction_and_misalignment_raise(self):
        for truth, available in (([1, 0], [False, True]), ([1, 0], [True]), ([2, 0], [True, True]),
                                 ([1, 0], [1, 1])):
            with self.subTest(truth=truth, available=available), self.assertRaises(la.AvailabilityMismatch):
                la.three_valued(np.array(truth), np.array(available))


class SidecarBindingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.table_dir = Path(self.tmp.name) / "brain" / ("a" * 64)
        self.table_dir.mkdir(parents=True)

    def tearDown(self):
        self.tmp.cleanup()

    def _write(self, table, arrays=None):
        arrays = arrays or la.compute(table.kind, table.candidates, GT_LABEL, GT_XYZ)
        return la.write(self.table_dir, table.kind, arrays, truth=table.truth, keys=table.keys,
                        pool_sha256=table.meta["pool_sha256"], source_cache=SOURCE)

    def test_written_beside_table_and_loaded(self):
        table = split_table()
        target = self._write(table)
        self.assertEqual(target, self.table_dir.parent / (self.table_dir.name + ".availability"))
        self.assertEqual(sorted(p.name for p in self.table_dir.iterdir()), [])  # table dir untouched
        record = la.load(table, self.table_dir)
        self.assertEqual(record["mask"].tolist(), [True, True, False, False])
        self.assertEqual(record["definition"], "segment_on_gt")
        meta = json.loads((target / "meta.json").read_text())
        self.assertEqual(meta["definitions"]["segment_on_gt"]["positives_available"], 1)
        self.assertEqual(meta["definitions"]["segment_on_gt"]["three_valued_counts"], {"1": 1, "0": 1, "nan": 2})

    def test_missing_sidecar_is_absent_not_an_error(self):
        self.assertIsNone(la.load(split_table(), self.table_dir))

    def test_positive_marked_unavailable_is_rejected_on_write(self):
        table = split_table(truth=(0, 0, 1, 0))  # row 2 positive, but segment 4 is not on GT
        with self.assertRaises(la.AvailabilityMismatch):
            self._write(table)
        self.assertFalse(la.sidecar_dir(self.table_dir).exists())

    def test_changed_labels_are_rejected_on_load(self):
        self._write(split_table())
        with self.assertRaises(la.AvailabilityMismatch):
            la.load(split_table(truth=(1, 1, 0, 0)), self.table_dir)

    def test_reordered_candidates_with_same_length_are_rejected(self):
        self._write(split_table())
        table = split_table()
        table.candidates = list(reversed(table.candidates))
        table.meta["pool_sha256"] = pool_digest("split", table.candidates)
        with self.assertRaises(la.AvailabilityMismatch):
            la.load(table, self.table_dir)

    def test_other_source_cache_is_rejected(self):
        self._write(split_table())
        table = split_table()
        table.meta["provenance"]["source_cache"] = {**SOURCE, "inode": 99}
        with self.assertRaises(la.AvailabilityMismatch):
            la.load(table, self.table_dir)

    def test_tampered_array_is_rejected(self):
        target = self._write(split_table())
        np.savez(target / "availability.npz", segment_on_gt=np.array([True, False, False, False]))
        with self.assertRaises(la.AvailabilityMismatch):
            la.load(split_table(), self.table_dir)

    def _bank(self, table, directory):
        return NativeTables("b", {table.kind: table}, {"table_paths": {table.kind: str(directory)}})

    def test_apply_replaces_truth_with_three_valued_labels(self):
        table = split_table()
        self._write(table)
        summary = la.apply({"b": self._bank(table, self.table_dir)})
        self.assertEqual(table.truth[:2].tolist(), [0., 1.])
        self.assertTrue(np.isnan(table.truth[2:]).all())
        self.assertEqual(summary["b/split"]["counts"], {"1": 1, "0": 1, "nan": 2})
        self.assertFalse(hasattr(table, "available"))  # only NaN in truth carries availability
        # A table shared by several bank dicts is converted once (no reload of a float array).
        la.apply({"b": self._bank(table, self.table_dir)})
        self.assertEqual(la.label_counts(table.truth), {"1": 1, "0": 1, "nan": 2})

    def test_apply_requires_a_sidecar(self):
        with self.assertRaises(la.AvailabilityMismatch):
            la.apply({"b": self._bank(split_table(), Path(self.tmp.name) / "none" / "x")})

    def test_apply_rejects_a_mismatched_sidecar(self):
        self._write(split_table())
        with self.assertRaises(la.AvailabilityMismatch):
            la.apply({"b": self._bank(split_table(truth=(1, 1, 0, 0)), self.table_dir)})

    def test_labels_digest_is_nan_safe_and_binary_compatible(self):
        binary = np.array([0, 1, 0, 0], dtype=np.int8)
        self.assertEqual(la.labels_digest(binary), la.labels_digest(binary.astype(float)))
        self.assertNotEqual(la.labels_digest(np.array([0., 1., np.nan, np.nan])), la.labels_digest(binary))


def three_valued_split(truth=(0., 1., np.nan, np.nan)):
    table = split_table()
    table.truth = np.array(truth, dtype=float)
    table.availability = {"definition": "segment_on_gt"}
    return table


class ThreeValuedScoringTests(unittest.TestCase):
    def test_nan_rows_are_never_ranked(self):
        truth = np.array([np.nan, np.nan, 1., 0.])
        metrics, chosen = scoring.rank_metrics([.9, .8, .7, .1], truth, list("abcd"), 1)
        self.assertEqual(chosen.tolist(), [2])
        self.assertEqual((metrics["precision"], metrics["n_labeled"], metrics["n_unlabeled"]), (1., 2, 2))
        self.assertEqual(metrics["positives"], 1)

    def test_nan_rows_do_not_change_metrics(self):
        labeled = scoring.rank_metrics([.7, .1], np.array([1., 0.]), ["c", "d"], 2)[0]
        mixed = scoring.rank_metrics([.9, .8, .7, .1], np.array([np.nan, np.nan, 1., 0.]), list("abcd"), 2)[0]
        for key in ("precision", "tp", "fp", "effective_k", "recall", "positives"):
            self.assertEqual(labeled[key], mixed[key], key)

    def test_effective_k_is_capped_by_labeled_rows(self):
        metrics, _ = scoring.rank_metrics([.5, .4, .3], np.array([1., np.nan, np.nan]), list("abc"), 5)
        self.assertEqual((metrics["effective_k"], metrics["precision"]), (1, 1.))

    def test_invalid_labels_and_empty_labeled_set_raise(self):
        for truth in ([2., 0.], [np.nan, np.nan]):
            with self.subTest(truth=truth), self.assertRaises(ValueError):
                scoring.rank_metrics([.5, .4], np.array(truth), ["a", "b"], 1)

    def test_evaluate_reports_labeled_counts_and_nan_safe_hash(self):
        report = scoring.evaluate(BASELINE, {"1": NativeTables("1", {"split": three_valued_split()}, {})},
                                  {"split": 2})
        cell = report["cells"]["1/split"]
        self.assertEqual((cell["precision"], cell["n_labeled"], cell["pool_size"]), (.5, 2, 4))
        self.assertEqual(cell["labels_sha256"], la.labels_digest(three_valued_split().truth))
        self.assertNotIn("macro_precision_available", report)

    def test_bootstrap_resamples_labeled_rows_only(self):
        table = three_valued_split((np.nan, 1., 0., np.nan))
        banks = {"1": NativeTables("1", {"split": table}, {})}
        # The candidate only reorders unlabeled rows: the labeled ranking is unchanged.
        parent = {"1/split": {"scores": np.array([.9, .8, .7, .1])}}
        candidate = {"1/split": {"scores": np.array([.1, .8, .7, .95])}}
        result = scoring.paired_bootstrap(parent, candidate, banks, {"split": 1}, draws=50)
        self.assertEqual(result["macro_delta_ci"], [0., 0.])

    def test_partitioned_metrics_rank_labeled_rows_per_fold(self):
        from proofreader_evolve.harness.selection_protocol import partitioned_rank_metrics
        truth = np.array([np.nan, 1., 0., np.nan, 0., 1.])
        folds = np.array([0, 0, 0, 1, 1, 1])
        metrics, chosen = partitioned_rank_metrics([.9, .5, .4, .9, .6, .3], truth, list("abcdef"), folds, 2)
        self.assertEqual(metrics["fold_held_rows"], [2, 2])  # labeled rows per fold
        self.assertEqual(sorted(chosen.tolist()), [1, 4])
        self.assertEqual((metrics["tp"], metrics["n_labeled"]), (1, 4))

    def test_scorer_never_sees_labels_or_availability(self):
        source = ("def score_candidates(features, ctx):\n"
                  "    assert list(features.columns) == ['detector_score'], list(features.columns)\n"
                  "    assert set(ctx) == {'kind'}\n"
                  "    return features['detector_score'].to_numpy()\n")
        scoring.evaluate(source, {"1": NativeTables("1", {"split": three_valued_split()}, {})}, {"split": 2})

    def test_diagnostic_examples_are_labeled_rows(self):
        from proofreader_evolve.harness.training_diagnostics import describe
        table = three_valued_split((np.nan, 1., np.nan, 0.))
        result = describe(table, np.array([.9, .8, .7, .1]), np.array([1]))
        self.assertTrue(all(e["label"] in (0, 1) for e in result["training_examples"]))
        self.assertEqual(result["group_sizes"]["boundary_label0"], 1)


if __name__ == "__main__":
    unittest.main()
