"""Readiness checks never implicitly build tables or call the LLM."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

from proofreader_evolve.cli import preflight
from proofreader_evolve.tests.label_fixture import setUpModule, tearDownModule  # noqa: F401


class PreflightTests(unittest.TestCase):
    def test_reports_all_missing_brains_without_build_or_api(self):
        with patch.object(preflight.pc, "resolve_detector_runs", return_value={}), \
                patch.object(preflight, "ensure_native_tables", side_effect=FileNotFoundError("missing")) as ensure, \
                patch.object(preflight, "api_probe") as api:
            result = preflight.inspect_readiness(["789202", "794491"])
        self.assertEqual(result["status"], "blocked")
        self.assertEqual(len(result["brains"]), 2)
        self.assertEqual(ensure.call_count, 2)
        self.assertTrue(all(call.kwargs["prepare"] is False for call in ensure.call_args_list))
        api.assert_not_called()

    def test_api_requires_explicit_opt_in_and_failure_blocks(self):
        with patch.object(preflight.pc, "resolve_detector_runs", return_value={}), \
                patch.object(preflight, "ensure_native_tables", return_value=SimpleNamespace(meta={"table_paths": {}})), \
                patch.object(preflight, "api_probe", return_value={"status": "failed"}) as api:
            result = preflight.inspect_readiness(["789202"], check_api=True, model="chosen")
        api.assert_called_once_with("chosen")
        self.assertEqual(result["status"], "blocked")

    def test_missing_key_never_calls_provider(self):
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(preflight.api_probe("chosen")["status"], "missing_key")

    def test_missing_label_sidecar_blocks(self):
        from proofreader_evolve.harness import label_availability
        table = SimpleNamespace(meta={}, truth=[0, 1])
        bank = SimpleNamespace(meta={"table_paths": {}}, tables={"split": table})
        with patch.object(preflight.pc, "resolve_detector_runs", return_value={}), \
                patch.object(preflight, "ensure_native_tables", return_value=bank), \
                patch.object(label_availability, "apply",
                             side_effect=label_availability.AvailabilityMismatch("no sidecar")):
            result = preflight.inspect_readiness(["802449"], selection_protocol="in_sample")
        self.assertEqual(result["status"], "blocked")
        self.assertIn("no sidecar", result["brains"]["802449"]["reason"])

    def test_raw_prior_allows_detector_fitted_selection_brains(self):
        def bank(brain, *args, **kwargs):
            table = SimpleNamespace(meta={"training_brain": "794495"}, truth=[0, 1])
            return SimpleNamespace(meta={"table_paths": {}}, tables={"merge": table, "split": table})
        with patch.object(preflight.pc, "resolve_detector_runs", return_value={}), \
                patch.object(preflight, "ensure_native_tables", side_effect=bank), \
                patch.object(preflight, "ensure_raw_tables", side_effect=bank):
            # One positive per fixture pool: opt out of the GT-positive floor (min_positives=0), as the
            # driver tests do, so only the role check decides here.
            raw = preflight.inspect_readiness(["794495", "802449"], train_brains=["794495", "802449"],
                                              selection_brains=["all"], detector_prior="raw", min_positives=0)
            default = preflight.inspect_readiness(["794495", "802449"], train_brains=["794495", "802449"],
                                                  detector_prior="raw", min_positives=0)
            frozen = preflight.inspect_readiness(["794495", "802449"], train_brains=["794495", "802449"],
                                                 selection_brains=["all"], min_positives=0)
        for report in (raw, default):
            self.assertEqual(report["status"], "checks_passed")
            self.assertEqual(report["selection"]["selection_brains"], ["794495", "802449"])
            self.assertEqual((report["selection"]["conflicts"], report["selection"]["auxiliary_train_brains"]), ([], []))
            self.assertIn("withheld", report["selection"]["note"])
        self.assertEqual(frozen["status"], "blocked")
        self.assertEqual(frozen["selection"]["conflicts"], ["794495"])

    def test_selection_roles_block_detector_fitted_selection_brains(self):
        def bank(brain, *args, **kwargs):
            table = SimpleNamespace(meta={"training_brain": "794495"}, truth=[0, 1])
            return SimpleNamespace(meta={"table_paths": {}}, tables={"merge": table, "split": table})
        with patch.object(preflight.pc, "resolve_detector_runs", return_value={}), \
                patch.object(preflight, "ensure_native_tables", side_effect=bank):
            good = preflight.inspect_readiness(["794495", "802449"], train_brains=["794495", "802449"],
                                               min_positives=0)
            bad = preflight.inspect_readiness(["794495"], train_brains=["794495"], min_positives=0)
            forced = preflight.inspect_readiness(["794495", "802449"], train_brains=["794495", "802449"],
                                                 selection_brains=["794495"], min_positives=0)
            legacy = preflight.inspect_readiness(["794495"], train_brains=["794495"], selection_protocol="in_sample",
                                                 min_positives=0)
        self.assertEqual(good["status"], "checks_passed")
        self.assertEqual(good["selection"]["selection_brains"], ["802449"])
        self.assertEqual(good["selection"]["auxiliary_train_brains"], ["794495"])
        self.assertEqual(bad["status"], "blocked")
        self.assertEqual(bad["selection"]["selection_brains"], [])
        self.assertEqual(forced["selection"]["conflicts"], ["794495"])
        self.assertEqual(forced["status"], "blocked")
        self.assertEqual(legacy["status"], "checks_passed")
        self.assertEqual(legacy["selection"]["selection_brains"], ["794495"])

    def test_positive_floor_lists_small_pools_and_blocks_an_uncounted_train_kind(self):
        from proofreader_evolve.harness import label_availability
        counts = {"802449": {"merge": 9, "split": 50}, "747807": {"merge": 30, "split": 50}}

        def bank(brain, *args, **kwargs):
            table = SimpleNamespace(meta={}, truth=[0, 1])
            return SimpleNamespace(meta={"table_paths": {}, "brain": brain},
                                   tables={"merge": table, "split": table})

        def labels(banks):
            (brain, _), = banks.items()
            return {f"{brain}/{kind}": {"counts": {"1": n, "0": 100, "nan": 5}}
                    for kind, n in counts[brain].items()}

        with patch.object(preflight.pc, "resolve_detector_runs", return_value={}), \
                patch.object(preflight, "ensure_raw_tables", side_effect=bank), \
                patch.object(label_availability, "apply", side_effect=labels):
            default = preflight.inspect_readiness(["802449", "747807"], train_brains=["802449", "747807"],
                                                  selection_brains=["all"], detector_prior="raw")
            alone = preflight.inspect_readiness(["802449"], train_brains=["802449"],
                                                selection_brains=["all"], detector_prior="raw")
            counted = preflight.inspect_readiness(["802449"], train_brains=["802449"], selection_brains=["all"],
                                                  detector_prior="raw", min_positives=0)
        self.assertEqual(preflight.DEFAULT_MIN_POSITIVES, 10)
        floor = default["positive_floor"]
        self.assertEqual((floor["min_positives"], floor["below_floor"], floor["status"]), (10, {"802449/merge": 9}, "passed"))
        self.assertEqual(floor["positives"]["747807/merge"], 30)
        self.assertEqual(default["status"], "checks_passed")
        self.assertTrue(any("802449/merge (9)" in w for w in default["warnings"]))
        self.assertEqual(alone["positive_floor"]["train_kinds_without_counted_pool"], ["merge"])
        self.assertEqual((alone["positive_floor"]["status"], alone["status"]), ("blocked", "blocked"))
        self.assertEqual((counted["positive_floor"]["below_floor"], counted["status"]), ({}, "checks_passed"))

    def test_valid_offline_checks_do_not_claim_api_success(self):
        with patch.object(preflight.pc, "resolve_detector_runs", return_value={}), \
                patch.object(preflight, "ensure_native_tables", return_value=SimpleNamespace(meta={"table_paths": {}})):
            result = preflight.inspect_readiness(["789202"])
        self.assertEqual(result["status"], "checks_passed")
        self.assertEqual(result["api"]["status"], "not_checked")
