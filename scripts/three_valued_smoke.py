"""Real-table smoke check of the three-valued labels: load, apply sidecars, score the frozen baseline.

Usage (compute node, panda): python scripts/three_valued_smoke.py --brains 789202 --k 2000 500
"""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from proofreader_evolve.cli import precompute_error_scores as pc  # noqa: E402
from proofreader_evolve.harness import fixed_pool_scoring as scoring, label_availability as la  # noqa: E402
from proofreader_evolve.harness.native_pool import cache_path, ensure_native_tables  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--brains", nargs="+", default=["789202"])
parser.add_argument("--k", nargs="+", type=int, default=[2000, 500])
args = parser.parse_args()
baseline = (Path(__file__).resolve().parents[1] / "proofreader_evolve/artifacts/scorer.py").read_text()
selected = pc.resolve_detector_runs(None, None)
for brain in args.brains:
    bank = ensure_native_tables(brain, cache_path(brain, 100), selected, pc.DEFAULT_OUT, prepare=False, mcl=100)
    for cell, info in la.apply({brain: bank}).items():
        print(cell, info["definition"], info["counts"])
    for k in args.k:
        report = scoring.evaluate(baseline, {brain: bank}, {"merge": k, "split": k})
        for cell, m in report["cells"].items():
            print(f"K={k} {cell}: P@K={m['precision']:.4f} tp={m['tp']} eff_k={m['effective_k']} "
                  f"labeled={m['n_labeled']} unlabeled={m['n_unlabeled']} positives={m['positives']}")
