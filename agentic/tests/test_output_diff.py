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
    path = PROJECT_ROOT / "agentic" / "output_diff.py"
    spec = importlib.util.spec_from_file_location("test_output_diff_module", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


od = _load_module()

_RECORDED = (
    "exitcode: 0 (execution succeeded)\n"
    "Code output: === EXPERIMENT ===\n"
    "Loading dataset from: ../data/dataset_cache_794495_mcl100_add.pkl\n"
    "Loaded 192177 candidates in 54.2 s\n"
    "Fold 0: Baseline AP = 0.02041 | Model AP = 0.05190\n"
    "Fold 1: Baseline AP = 0.02440 | Model AP = 0.04880\n"
    "Pooled AP: baseline 0.0278, model 0.0519 (delta +0.0241)\n"
    "Bootstrap 95% CI: [+0.0120, +0.0360], p = 0.0010\n"
    "Processed 8,000/11,245 patches in 35.7s (224.0 patches/s)\n"
    "Top 1% precision: 7/97\n"
    "\nSTDERR:\n\n[notice] A new release of pip is available: 25.1.1 -> 26.2.1\n"
)
_RERUN_SAME = (
    "=== EXPERIMENT ===\n"
    "Loading dataset from: /tmp/rerun_x/run/dataset_cache_794495_mcl100_add.pkl\n"
    "Loaded 192177 candidates in 3.1 s\n"
    "Fold 0: Baseline AP = 0.02041 | Model AP = 0.05190\n"
    "Fold 1: Baseline AP = 0.02440 | Model AP = 0.04881\n"
    "Pooled AP: baseline 0.0278, model 0.0519 (delta +0.0241)\n"
    "Bootstrap 95% CI: [+0.0120, +0.0360], p = 0.0010\n"
    "Processed 8,000/11,245 patches in 70.2s (114.0 patches/s)\n"
    "Top 1% precision: 7/97\n"
)


class OutputDiffTests(unittest.TestCase):
    def test_recorded_stdout_strips_the_autodiscovery_wrapper(self) -> None:
        text = od.recorded_stdout(_RECORDED)
        self.assertTrue(text.startswith("=== EXPERIMENT ==="))
        self.assertNotIn("STDERR", text)
        self.assertNotIn("exitcode", text)
        self.assertEqual(od.recorded_stdout("plain\n"), "plain")

    def test_identical_outputs_ignore_noise_and_fit_noise(self) -> None:
        result = od.diff_outputs(od.recorded_stdout(_RECORDED), _RERUN_SAME)
        self.assertEqual(result["hint"], "identical")
        self.assertEqual(result["counts"]["material"], 0)
        self.assertEqual(result["counts"]["tiny"], 1)      # 0.04880 vs 0.04881
        self.assertGreater(result["counts"]["exact"], 8)
        self.assertEqual(result["only_recorded"], [])
        self.assertEqual(result["only_rerun"], [])
        # the load path, timing and throughput lines never count
        templates = [t for t, _, _ in od.numeric_lines(_RERUN_SAME)]
        self.assertFalse(any("Loading" in t or "patches/s" in t for t in templates))

    def test_a_changed_headline_number_is_material_and_paired_in_order(self) -> None:
        rerun = _RERUN_SAME.replace("Fold 1: Baseline AP = 0.02440", "Fold 1: Baseline AP = 0.04100")
        result = od.diff_outputs(od.recorded_stdout(_RECORDED), rerun)
        self.assertEqual(result["hint"], "numeric-differences")
        self.assertEqual(result["counts"]["material"], 1)
        (mismatch,) = result["mismatches"]
        self.assertEqual((mismatch["recorded_value"], mismatch["rerun_value"]), (0.0244, 0.041))
        self.assertIn("Fold 1", mismatch["recorded"])  # fold 1 paired with fold 1
        self.assertGreater(mismatch["rel_diff"], 0.3)

    def test_lines_on_one_side_only_are_structural(self) -> None:
        rerun = _RERUN_SAME + "Extra diagnostic: 42 rows\n"
        result = od.diff_outputs(od.recorded_stdout(_RECORDED), rerun)
        self.assertEqual(result["hint"], "structural-differences")
        self.assertEqual(result["only_rerun"], ["Extra diagnostic: 42 rows"])
        shorter = "\n".join(_RERUN_SAME.splitlines()[:-1])
        result = od.diff_outputs(od.recorded_stdout(_RECORDED), shorter)
        self.assertEqual(result["only_recorded"], ["Top 1% precision: 7/97"])

    def test_value_count_change_is_material(self) -> None:
        rerun = _RERUN_SAME.replace("Top 1% precision: 7/97", "Top 1% precision: 7/97 (p=0.5)")
        result = od.diff_outputs(od.recorded_stdout(_RECORDED), rerun)
        # the template differs, so the line is unpaired on both sides
        self.assertEqual(result["hint"], "structural-differences")
        self.assertEqual(result["n_only_recorded"], 1)
        self.assertEqual(result["n_only_rerun"], 1)

    def test_cli_reads_full_outputs_from_the_outputs_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_json = root / "run.json"
            run_json.write_text(json.dumps([
                {"id": 1, "codeOutput": _RECORDED},
                {"id": 2, "codeOutput": _RECORDED},
                {"id": 3, "codeOutput": _RECORDED},
            ]))
            outputs = root / "res.outputs"
            outputs.mkdir()
            (outputs / "hypo_1.stdout.txt").write_text(_RERUN_SAME)
            results = root / "res.json"
            results.write_text(json.dumps({"results": [
                {"id": 1, "result_status": "USABLE", "rerun_stdout": "truncated…",
                 "rerun_stdout_path": str(outputs / "hypo_1.stdout.txt")},
                {"id": 2, "result_status": "USABLE",
                 "rerun_stdout": _RERUN_SAME.replace("0.0519", "0.0300")},
                {"id": 3, "result_status": "UNUSABLE",
                 "result_failure_reason": "swallowed-input-error"},
            ]}))
            proc = subprocess.run(
                [sys.executable, str(PROJECT_ROOT / "agentic" / "output_diff.py"),
                 str(run_json), str(results), "--outputs-dir", str(outputs)],
                capture_output=True, text=True, timeout=120,
            )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        payload = json.loads(proc.stdout)
        self.assertEqual(payload["summary"], {"identical": 1, "numeric-differences": 1,
                                              "structural-differences": 0, "not-usable": 1})
        self.assertEqual(payload["entries"]["1"]["rerun_source"], "file")
        self.assertEqual(payload["entries"]["2"]["rerun_source"], "payload")
        self.assertEqual(payload["entries"]["2"]["counts"]["material"], 2)
        self.assertEqual(payload["entries"]["3"], {"hint": "not-usable",
                                                   "reason": "swallowed-input-error"})


if __name__ == "__main__":
    unittest.main()
