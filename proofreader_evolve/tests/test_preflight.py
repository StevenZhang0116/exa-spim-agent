"""Readiness checks never implicitly build tables or call the LLM."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

from proofreader_evolve.cli import preflight


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

    def test_valid_offline_checks_do_not_claim_api_success(self):
        with patch.object(preflight.pc, "resolve_detector_runs", return_value={}), \
                patch.object(preflight, "ensure_native_tables", return_value=SimpleNamespace(meta={"table_paths": {}})):
            result = preflight.inspect_readiness(["789202"])
        self.assertEqual(result["status"], "checks_passed")
        self.assertEqual(result["api"]["status"], "not_checked")
