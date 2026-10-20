from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_module():
    path = PROJECT_ROOT / "agentic" / "recorded_audit.py"
    spec = importlib.util.spec_from_file_location("test_recorded_audit_module", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


ra = _load_module()

_GLOG = ("E1020 06:20:45.167962       3 common-io:5] static: Failed to open file. "
         "path:'/root/.aws/config' mode:'rb' errno:2\n")


def _record(rid, hypothesis, code, output):
    return {"id": rid, "hypothesis": hypothesis, "code": code,
            "codeOutput": "exitcode: 0 (execution succeeded)\nCode output: " + output}


class RecordedAuditTests(unittest.TestCase):
    def test_a_real_image_read_is_not_flagged_despite_glog_lines(self) -> None:
        rec = _record(1, "True split gaps show high raw fluorescence intensity along the line",
                      "import tensorstore as ts\nstore = ts.open(spec).result()\n",
                      _GLOG + "INPUTS READ: total_patches_read=6076, image_level=0\n"
                      "Pooled AP: 0.83\n")
        audit = ra.audit_record(rec)
        self.assertFalse(audit["suspected"])
        self.assertEqual(audit["flags"], [])
        self.assertTrue(audit["image_hypothesis"] and audit["image_reader_in_code"]
                        and audit["image_read_evidence"])

    def test_output_fallback_marker_and_code_substitute_are_flagged(self) -> None:
        rec = _record(46, "Integrating raw image SNR at anchor tips with caliber",
                      "import tensorstore as ts\ntry:\n    ds = ts.open(spec).result()\n"
                      "except Exception as e:\n"
                      "    print(f'TensorStore open warning (using mock/fallback if needed): {e}')\n"
                      "    # Simulated consistent structural intensity proxy\n"
                      "    snr = 2.5 / (1.0 + std_r)\n",
                      "TensorStore open warning (using mock/fallback if needed): Error parsing "
                      "object member \"kvstore\"\nLR Delta AP: +0.01577\n")
        audit = ra.audit_record(rec)
        self.assertTrue(audit["suspected"])
        self.assertIn("output-fallback-marker", audit["flags"])
        self.assertIn("code-substitute-marker", audit["flags"])
        self.assertIn("image-hypothesis-without-read-evidence", audit["flags"])
        self.assertTrue(audit["output_markers"][0].startswith("TensorStore open warning"))
        self.assertTrue(any("proxy" in line for line in audit["code_substitutes"]))

    def test_image_hypothesis_without_any_reader_is_flagged(self) -> None:
        rec = _record(90, "True splits keep continuous raw fluorescence along the gap",
                      "mean_int = 1.0 / (1.0 + gap * 0.08)\nprint('AP', 0.83)\n",
                      "INPUTS READ: Companion rows=689677\nCombined + Intensity OOF AP: 0.8297\n")
        audit = ra.audit_record(rec)
        self.assertEqual(audit["flags"], ["image-hypothesis-without-reader"])
        self.assertTrue(audit["suspected"])

    def test_unpickling_mock_alone_is_informational(self) -> None:
        rec = _record(32, "Local fragment density within 30 um predicts merges",
                      "mock_mod = types.ModuleType('agentic_neuron_proofreader')\n"
                      "sys.modules['agentic_neuron_proofreader'] = mock_mod\n",
                      "Degree AP 0.0278, model AP 0.0519\n")
        audit = ra.audit_record(rec)
        self.assertEqual(audit["flags"], ["code-mock-marker"])
        self.assertFalse(audit["suspected"])

    def test_cli_and_text(self) -> None:
        records = [
            _record(1, "Local density", "x = 1\n", "AP 0.05\n"),
            _record(2, "Raw fluorescence contrast", "x = 1\n", "AP 0.05\n"),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            run_json = Path(tmpdir) / "run.json"
            run_json.write_text(json.dumps(records))
            proc = subprocess.run(
                [sys.executable, str(PROJECT_ROOT / "agentic" / "recorded_audit.py"),
                 str(run_json), "--only-suspected"],
                capture_output=True, text=True, timeout=120,
            )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        audit = json.loads(proc.stdout)
        self.assertEqual((audit["n"], audit["n_suspected"], audit["suspected_ids"]), (2, 1, ["2"]))
        self.assertEqual(list(audit["entries"]), ["2"])
        text = ra.audit_text(audit, "run.recorded-audit.json")
        self.assertIn("Flagged ids: 2 (1 of 2)", text)
        self.assertIn("run.recorded-audit.json", text)
        self.assertIn("evidence, not a verdict", text)


if __name__ == "__main__":
    unittest.main()
