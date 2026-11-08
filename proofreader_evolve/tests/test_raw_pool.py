"""Detector-free raw pools (2026-11-06): built from the cache without resolving, hashing or
running a frozen detector; the same rows and pool_sha256 as the native enumeration; the raw
columns as the only feature frame; loaded by raw evolution, preflight and the sidecar writer.
Synthetic fixtures only; no brain caches, cloud reads or LLM calls."""

import asyncio
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from proofreader_evolve import prepare_feature_tables as launcher
from proofreader_evolve.cli import precompute_availability
from proofreader_evolve.cli import precompute_error_scores as pc
from proofreader_evolve.cli import preflight
from proofreader_evolve.harness import candidate_pools as cp
from proofreader_evolve.harness import detector_prior
from proofreader_evolve.harness import label_availability as la
from proofreader_evolve.harness import raw_pool
from proofreader_evolve.harness.native_pool import build_native_universe
from proofreader_evolve.tests.test_companion_pool_only import split_payload_with_negatives
from proofreader_evolve.tests.test_detector_prior import parse
from proofreader_evolve.tests.test_raw_prior import raw_merge_table
from proofreader_evolve.tests.label_fixture import setUpModule, tearDownModule  # noqa: F401

HERE = Path(__file__).resolve().parents[1]


def build(tmp, data):
    """Raw pools of brain 1 from the synthetic payload; the placeholder file is the cache identity."""
    source = Path(tmp) / "dataset_cache_1_mcl100_add.pkl"
    source.write_bytes(b"placeholder")
    tables = Path(tmp) / "tables"
    with patch.object(pc.ds, "load_cached_graphs", return_value=(data["fragments_graph"], None, data)), \
            redirect_stdout(io.StringIO()):
        bank = raw_pool.ensure_raw_tables("1", source, tables, prepare=True)
    return source, tables, bank


class RawPoolTests(unittest.TestCase):
    def test_pools_are_built_and_reloaded_without_any_detector(self):
        data = split_payload_with_negatives()
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(pc, "resolve_detector_runs", side_effect=AssertionError("no detector may be resolved")), \
                patch.object(pc, "_import_module", side_effect=AssertionError("no detector may be imported")):
            source, tables, bank = build(tmp, data)
            self.assertEqual(sorted(bank.tables), ["merge", "split"])
            self.assertEqual((bank.meta["version"], bank.meta["detector_prior"]), (raw_pool.RAW_POOL_VERSION, "raw"))
            for kind, table in bank.tables.items():
                with self.subTest(kind=kind):
                    rows, truth, meta = build_native_universe(split_payload_with_negatives(), kind, None,
                                                              cp.load_pool_policies()[kind])
                    self.assertEqual(table.candidates, rows)
                    np.testing.assert_array_equal(table.truth, truth)
                    self.assertEqual(table.meta["pool_sha256"], meta["pool_sha256"])
                    self.assertEqual(list(table.features.columns), list(detector_prior.RAW_COLUMNS[kind]))
                    self.assertEqual(len(table.features), len(rows))
                    pd.testing.assert_frame_equal(table.features, detector_prior.raw_columns(kind, rows))
                    self.assertIsNone(table.meta["training_brain"])
                    self.assertIsNone(table.meta["detector_rule_matches"])
                    provenance = table.meta["provenance"]
                    self.assertEqual((provenance["detector"], provenance["training"], provenance["detector_prior"]),
                                     (None, None, "raw"))
                    self.assertEqual(provenance["raw_columns"]["version"], detector_prior.RAW_COLUMNS_VERSION)
                    self.assertEqual(sorted(provenance["implementation"]),
                                     ["candidate_rules.py", "detector_metadata.py", "native_pool.py"])
                    directory = Path(bank.meta["table_paths"][kind])
                    self.assertEqual(sorted(p.name for p in directory.iterdir()),
                                     ["candidates.pkl", "evaluator_labels.npy", "features.pkl", "meta.json"])
            # A second call reloads the saved pools without the cache and leaves them untouched.
            with patch.object(pc.ds, "load_cached_graphs", side_effect=AssertionError("must not reload the cache")):
                again = raw_pool.ensure_raw_tables("1", source, tables, prepare=False)
            for kind in again.tables:
                self.assertEqual(again.tables[kind].candidates, bank.tables[kind].candidates)
                pd.testing.assert_frame_equal(again.tables[kind].features, bank.tables[kind].features)
            # Raw frames are left alone by the raw stripping: no host copies, nothing to withhold.
            self.assertEqual(detector_prior.strip_to_raw({"1": again}), [])
            self.assertFalse(hasattr(again.tables["merge"], "host_detector_score"))

    def test_missing_pools_are_reported_not_built_and_mismatches_are_refused(self):
        data = split_payload_with_negatives()
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "dataset_cache_1_mcl100_add.pkl"
            source.write_bytes(b"placeholder")
            with self.assertRaisesRegex(FileNotFoundError, "prepare_feature_tables.py --detector-prior raw"):
                raw_pool.ensure_raw_tables("1", source, Path(tmp) / "tables", prepare=False)
            source, tables, bank = build(tmp, data)
            directory = Path(bank.meta["table_paths"]["split"])
            labels = np.load(directory / "evaluator_labels.npy")
            np.save(directory / "evaluator_labels.npy", 1 - labels, allow_pickle=False)
            with self.assertRaisesRegex(ValueError, "Corrupt raw pool"):
                raw_pool.ensure_raw_tables("1", source, tables, prepare=False)
            # A changed cache identity is a different digest: the old pools are not matched.
            source.write_bytes(b"placeholder-v2")
            with self.assertRaises(FileNotFoundError):
                raw_pool.ensure_raw_tables("1", source, tables, prepare=False)

    def test_sidecars_are_written_beside_raw_pools_without_a_detector(self):
        data = split_payload_with_negatives()
        # The sidecar writer reads GT node coordinates; the synthetic GT graph has none.
        data["gt_graph"].node_xyz = np.array([[0., 0., 0.], [5., 0., 0.], [10., 0., 0.],
                                              [50., 20., 0.], [55., 20., 0.]])

        def available(kind, candidates, *args, **kwargs):
            return {name: np.ones(len(candidates), dtype=bool) for name in la.DEFINITIONS[kind]}

        with tempfile.TemporaryDirectory() as tmp:
            source, tables, bank = build(tmp, data)
            args = precompute_availability.parse_args(["--brain", "1", "--feature-tables-dir", str(tables),
                                                       "--detector-prior", "raw"])
            with patch.object(precompute_availability, "cache_path", return_value=source), \
                    patch.object(precompute_availability, "pickle", SimpleNamespace(load=lambda stream: data)), \
                    patch.object(precompute_availability.pc, "resolve_detector_runs",
                                 side_effect=AssertionError("no detector may be resolved")), \
                    patch.object(precompute_availability.la, "compute", side_effect=available), \
                    redirect_stdout(io.StringIO()):
                self.assertEqual(precompute_availability.single(args), 0)
            for kind, table in bank.tables.items():
                with self.subTest(kind=kind):
                    record = la.load(table, bank.meta["table_paths"][kind])
                    self.assertIsNotNone(record)
                    meta = json.loads((Path(record["path"]) / "meta.json").read_text())
                    self.assertIsNone(meta["provenance"]["detector"])
                    self.assertEqual(meta["provenance"]["pools"], raw_pool.RAW_POOL_VERSION)
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(precompute_availability.main(["--brains", "1", "--detector-prior", "raw", "--dry-run"]), 0)
            self.assertIn("--detector-prior raw", output.getvalue())


class PreparationTests(unittest.TestCase):
    def test_raw_prior_plans_raw_pools_and_forwards_the_prior(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(launcher, "PROJECT_ROOT", Path(tmp)):
            planned = launcher.commands(launcher.parse_args(["--brains", "1", "2", "--detector-prior", "raw",
                                                             "--image-workers", "-1"]))
            frozen = launcher.commands(launcher.parse_args(["--brains", "1"]))
        self.assertEqual([command[3] for command in planned],
                         ["proofreader_evolve.cli.precompute_raw_pools"] * 2
                         + ["proofreader_evolve.cli.precompute_availability", "proofreader_evolve.cli.preflight"])
        for command in planned[:2]:
            self.assertNotIn("--image-workers", command)
            self.assertIn("dataset_cache_", command[command.index("--pkl") + 1])
        for command in planned[2:]:
            self.assertEqual(command[command.index("--detector-prior") + 1], "raw")
        self.assertEqual(frozen[0][3], "proofreader_evolve.cli.precompute_error_scores")
        self.assertEqual(frozen[-1][frozen[-1].index("--detector-prior") + 1], "frozen")

    def test_preflight_under_raw_resolves_no_detector(self):
        def bank(brain, *args, **kwargs):
            table = SimpleNamespace(meta={"training_brain": None}, truth=[0, 1])
            return SimpleNamespace(meta={"table_paths": {}}, tables={"merge": table, "split": table})
        with patch.object(preflight.pc, "resolve_detector_runs",
                          side_effect=SystemExit("--merge-dir is not a directory")), \
                patch.object(preflight, "ensure_raw_tables", side_effect=bank) as raw, \
                patch.object(preflight, "ensure_native_tables",
                             side_effect=AssertionError("native tables must not be loaded under raw")):
            # One positive per fixture pool: opt out of the GT-positive floor (min_positives=0) so only the
            # detector resolution and table loading are under test here.
            report = preflight.inspect_readiness(["1", "2"], train_brains=["1", "2"], selection_brains=["all"],
                                                 detector_prior="raw", min_positives=0)
            frozen = preflight.inspect_readiness(["1"], train_brains=["1"], min_positives=0)
        self.assertEqual(report["status"], "checks_passed")
        self.assertIsNone(report["detectors"])
        self.assertIn("raw", report["candidate_pools"])
        self.assertEqual(raw.call_count, 2)
        self.assertTrue(all(call.kwargs["prepare"] is False for call in raw.call_args_list))
        self.assertEqual(report["selection"]["detector_training_brains"], [])
        self.assertEqual(report["selection"]["selection_brains"], ["1", "2"])
        self.assertEqual(frozen["status"], "blocked")
        self.assertIn("not a directory", frozen["detector_error"])


def raw_pool_banks():
    """Driver fixture: merge pools whose frame already holds only the raw columns and no detector."""
    banks = {}
    for brain, offset in (("794495", 1000.), ("802449", 0.), ("2", 5000.)):
        bank = raw_merge_table(brain, offset=offset)
        table = bank.tables["merge"]
        table.features = detector_prior.raw_columns("merge", table.candidates)
        table.meta.update(training_brain=None, provenance={"brain": brain, "detector": None})
        banks[brain] = bank
    return banks


class DriverTests(unittest.TestCase):
    def test_detector_directories_are_refused_under_raw(self):
        from proofreader_evolve.cli import run_precision_evolution as driver
        for flag in ("--merge-dir", "--split-dir"):
            with self.subTest(flag=flag), redirect_stdout(io.StringIO()), patch("sys.stderr", io.StringIO()), \
                    self.assertRaises(SystemExit):
                driver.parse_args(["--detector-prior", "raw", flag, "some-detector"])
        self.assertEqual(driver.parse_args(["--detector-prior", "raw"]).merge_dir, None)

    def test_raw_run_on_raw_pools_resolves_no_detector_and_has_no_frozen_baseline(self):
        from proofreader_evolve.cli import run_precision_evolution as driver
        banks = raw_pool_banks()
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            args = parse(driver, tmp, "--detector-prior", "raw")

            async def revise(run_dir, policy, rules, report, model, *, experiments):
                policy.write_text("def score_candidates(features, ctx):\n"
                                  "    assert list(features.columns) == "
                                  "['raw_degree', 'raw_candidate_family', 'raw_tip_gap_um']\n"
                                  "    return -features['raw_tip_gap_um'].fillna(1e6).to_numpy()\n")
                experiments.proposal_path.write_text(json.dumps({"hypothesis": "small tip gaps are merges",
                    "strategy": "rank by the raw tip gap", "family": "raw-gap", "parameter_grid": {}}))
                experiments.evaluate()
                return {"summary": "raw gap", "cost_usd": 0}

            with patch.object(driver.pc, "resolve_detector_runs",
                              side_effect=AssertionError("raw must not resolve detectors")), \
                    patch.object(driver, "ensure_native_tables",
                                 side_effect=AssertionError("raw must load raw pools")), \
                    patch.object(driver, "ensure_raw_tables", side_effect=lambda brain, *a, **k: banks[brain]), \
                    patch.object(driver, "attach_local_context", return_value=None):
                path = asyncio.run(driver.run(args, revise_fn=revise))
            manifest = json.loads((path / "manifest.json").read_text())
            baseline = json.loads((path / "baseline.json").read_text())
            record = json.loads((path / "ledger.jsonl").read_text().splitlines()[0])
            trajectory = (path / "trajectory.jsonl").read_text() if (path / "trajectory.jsonl").exists() else ""
        self.assertIsNone(manifest["selected_detectors"])
        self.assertEqual(manifest["candidate_pools"], "raw (detector-free)")
        self.assertEqual(manifest["detector_prior"]["mode"], "raw")
        self.assertEqual(manifest["detector_prior"]["stripped_cells"], [])
        self.assertEqual(manifest["dataset_selection"]["detector_training_brains"], [])
        self.assertIsNone(baseline["validation"])
        self.assertIn("no detector score", baseline["note"])
        self.assertEqual(baseline["raw_heuristics"]["validation"]["cells"]["2/merge"]["precision"], 1.)
        self.assertEqual(record["validation"]["cells"]["2/merge"]["precision"], 1.)
        self.assertTrue(record["accepted"])
        self.assertNotIn("Evaluating frozen detector-score baseline", trajectory)


if __name__ == "__main__":
    unittest.main()
