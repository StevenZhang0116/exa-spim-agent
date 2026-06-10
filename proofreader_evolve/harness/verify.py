"""
Faithfulness check: incremental scorer == the real evaluate().

Proves the fast path produces the SAME metrics as the runnable
``segmentation_skeleton_metrics.evaluate`` before the loop relies on it. Run
once after any change to incremental_scoring.py.

    python proofreader_evolve/harness/verify.py --brain 789202

What it does:
  1. Builds (or loads) the PreparedBrain once.
  2. Scores the BASELINE (no edits) incrementally.
  3. Runs the real evaluate() baseline via scoring.score (the ~30 min path).
  4. Asserts the per-skeleton metrics agree within tolerance.
  5. Scores ONE non-trivial candidate (a few real label-pairs) incrementally and
     via evaluate(), and asserts those agree too — so we know the equivalence
     holds when edits are applied, not just at baseline.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from proofreader_evolve.harness import scoring, incremental_scoring as inc, dataset as ds

HERE = Path(__file__).resolve().parent
RUNS = HERE.parent / "runs"

# Metrics to compare and absolute tolerances. ERL is in microns (large), so a
# small absolute slack; percentages/counts should match very tightly.
TOL = {
    "Edge Accuracy": 0.05,
    "# Splits": 0.5,
    "# Merges": 0.5,
    "% Split Edges": 0.05,
    "% Omit Edges": 0.05,
    "% Merged Edges": 0.05,
    "ERL": 5.0,
    "Normalized ERL": 0.001,
}


def _compare(label, inc_res, ref_res) -> bool:
    """Compare two ScoreResults per-skeleton; print and return pass/fail."""
    print(f"\n--- {label} ---")
    a = inc_res.per_swc.sort_index()
    b = ref_res.per_swc.sort_index()
    common = a.index.intersection(b.index)
    ok = True
    for metric, tol in TOL.items():
        if metric not in a.columns or metric not in b.columns:
            continue
        diff = (a.loc[common, metric] - b.loc[common, metric]).abs()
        worst = float(np.nanmax(diff.values)) if len(diff) else 0.0
        status = "OK " if worst <= tol else "FAIL"
        if worst > tol:
            ok = False
        print(f"  [{status}] {metric:<16} max|Δ|={worst:.4g} (tol {tol})")
    print(f"  incremental: {inc_res.seconds:.2f}s   evaluate(): {ref_res.seconds:.1f}s")
    return ok


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--brain", default="789202")
    p.add_argument("--n-edits", type=int, default=10,
                   help="how many real label-pairs to use for the candidate check")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    paths = scoring.BrainPaths(args.brain)
    print(f"brain_id={args.brain} -> segmentation_id={paths.segmentation_id}")

    prepared_cache = str(RUNS / f"prepared_{args.brain}.pkl")
    prepared = inc.get_or_build(paths, prepared_cache, verbose=True)

    # --- 1) BASELINE: incremental vs evaluate() ------------------------------
    inc_base = inc.score_incremental(prepared, label_pairs=None, verbose=args.verbose)
    ref_base = scoring.score(
        paths, str(RUNS / f"_verify_baseline_{args.brain}"),
        label_pairs=None, verbose=args.verbose,
    )
    ok1 = _compare("BASELINE (no edits)", inc_base, ref_base)

    # --- 2) CANDIDATE: a handful of real proximity edits ----------------------
    # Derive a few label-pairs from the fragment geometry so the test exercises
    # the equivalence-collapse path, not just the identity case.
    frags, _, _ = ds.load_cached_graphs(ds.default_cache_path(args.brain))
    sites = ds.candidate_split_sites(frags, max_gap_um=5.0)
    edits = []
    seen = set()
    for s in sites:
        k = frozenset(s.as_edit())
        if k not in seen:
            seen.add(k)
            edits.append(s.as_edit())
        if len(edits) >= args.n_edits:
            break
    print(f"\nCandidate uses {len(edits)} label-pair edits.")

    inc_cand = inc.score_incremental(prepared, label_pairs=edits, verbose=args.verbose)
    ref_cand = scoring.score(
        paths, str(RUNS / f"_verify_candidate_{args.brain}"),
        label_pairs=edits, all_fragment_labels=ds.list_fragment_labels(frags),
        verbose=args.verbose,
    )
    ok2 = _compare("CANDIDATE (with edits)", inc_cand, ref_cand)

    # --- 3) SUBSET: scoring a GT subset must equal evaluate()-then-filter -------
    # Guards the cross-skeleton merge bug: % Merged Edges depends on BOTH
    # skeletons sharing a label being present, so a subset must be scored on the
    # FULL set and filtered, NOT by handing the metric only the subset. We score
    # the same baseline on a held-out subset two ways and require agreement.
    half = sorted(prepared.gt_names)[: max(1, len(prepared.gt_names) // 2)]
    inc_subset = inc.score_incremental(
        prepared, label_pairs=None, gt_swc_names=half, verbose=args.verbose
    )
    ref_subset = ref_base.per_swc.loc[ref_base.per_swc.index.isin(half)]
    ok3 = _compare(
        f"SUBSET (baseline, {len(half)}/{len(prepared.gt_names)} GT)",
        inc_subset,
        scoring.ScoreResult(
            primary=0.0, metrics={}, per_swc=ref_subset, output_dir="", seconds=0.0
        ),
    )

    print("\n==================== RESULT ====================")
    if ok1 and ok2 and ok3:
        print("PASS — incremental scorer matches evaluate() on baseline, "
              "candidate, AND GT subset.")
        return 0
    print("FAIL — discrepancy exceeds tolerance; do NOT trust the fast path yet.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
