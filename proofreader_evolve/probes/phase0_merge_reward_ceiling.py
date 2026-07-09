"""
Phase 0 — merge-repair reward CEILING probe (go/no-go before building the gate).

Before writing ``classify_split_edits`` and folding a merge-repair reward into the
gate (see the staged plan), this answers ONE question cheaply: on the brain the gate
actually scores, how many merge errors could a ``split_label`` reward EVER sum over?

The decision variable is a COUNT, not Edge Accuracy. Edge Accuracy / %Merged Edges
are sparse edge metrics that already flat-lined the split side, so the oracle's
headline delta understates a real signal; the dense reward will instead sum over
individual merge-causing labels, so the count of those labels IS the ceiling.

There are TWO stages that "identify a merge error", and they cap the reward jointly:

  * n_mergeable  (GT-LABEL way / SCORING) — raw segment labels that land (>= min_nodes
    nodes) on >= 2 DISTINCT held-out GT skeletons. Same rule as probe_split_oracle /
    MergedEdgePercentMetric. This is what the GT-based reward can score.

  * n_detectable (GEOMETRIC-WALK way / DETECTION) — of those mergeable labels, how many
    the DEPLOYABLE, GT-free ``candidate_merge_sites`` scan actually surfaces a MergeSite
    for. A ``split_label`` edit can only ever act on a detected MergeSite, so an
    undetected merge error is unreachable no matter how good the policy is.

The REWARD CEILING is the INTERSECTION (mergeable AND detectable). n_mergeable alone
overstates it (some are unreachable — the "detector recall gap"); n_detectable over
ALL labels overstates the useful part (some detected sites are not real merges).

Held-out scoping matches the gate:
  * DEFAULT (cross-brain gate): the whole ``--brain`` is the held-out test brain
    (role="heldout" in run_evolution), so every GT neuron is in scope.
  * ``--heldout-fraction F`` (per-brain split): reserve a within-brain fraction, to
    estimate the ceiling for the legacy per-brain regime. ``--split-seed`` pins it.

Run from the project root (exa-spim-agent/), in the panda env:
    python proofreader_evolve/probes/phase0_merge_reward_ceiling.py --brain 794495 --mcl 10
    python proofreader_evolve/probes/phase0_merge_reward_ceiling.py --brain 789202 --mcl 10 --heldout-fraction 0.5 --split-seed 0
    # add --with-oracle to ALSO run the (slow) oracle rescore for held-out sparse-metric deltas
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Make ``proofreader_evolve`` importable when run as a script from the project root.
# This file lives in proofreader_evolve/probes/, so PROJECT_ROOT (exa-spim-agent/) is
# three parents up: probes/ -> proofreader_evolve/ -> exa-spim-agent/.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from proofreader_evolve.harness import (
    scoring,
    dataset as ds,
    incremental_scoring as inc,
)


def _heldout_names(prepared, heldout_fraction: float | None, split_seed: int) -> list[str]:
    """Which GT skeletons the gate scores on.

    None/>=1.0 fraction => cross-brain gate: the WHOLE brain is held out (every GT
    neuron in scope), matching run_evolution's role="heldout". A fraction in (0,1)
    reproduces the legacy per-brain split via the same ``train_heldout_split``.
    """
    all_names = prepared.gt_names
    if not heldout_fraction or heldout_fraction >= 1.0:
        return list(all_names)
    _, heldout = ds.train_heldout_split(
        all_names, heldout_fraction=heldout_fraction, seed=split_seed)
    return heldout


def mergeable_labels(prepared, heldout_names: list[str], min_nodes: int) -> set[str]:
    """GT-LABEL way: raw labels spanning >= 2 held-out GT neurons (>= min_nodes each).

    Restricted to ``heldout_names`` so this is the LEAK-SAFE held-out scope the gate's
    reward would use (``label_gt_counts(prepared, gt_names=heldout_names)``). A label
    that only reaches held-out GT on one neuron (or <min_nodes) is NOT a scorable merge
    error in this scope. Same >=min_nodes-on->=2 rule as probe_split_oracle (line 934).
    """
    heldout_map = inc.label_gt_counts(prepared, gt_names=heldout_names)
    return {
        lab
        for lab, counts in heldout_map.items()
        if sum(1 for c in counts.values() if c >= min_nodes) >= 2
    }


def detected_labels(fragments_graph, verbose: bool) -> set[str]:
    """GEOMETRIC-WALK way: raw labels the deployable, GT-free MergeSite scan surfaces.

    Uses ``candidate_merge_sites`` DEFAULTS (min_arm_cable_um=10, seed_depth_um=8,
    max_sites=5000, max_per_label=8) — the same values run_evolution's ENUM_PARAMS
    resolve to — so the detected set matches what a real run's policy would see. The
    count is sensitive to these params; a run that widens them would detect more.
    """
    sites = ds.candidate_merge_sites(fragments_graph)
    if verbose:
        print(f"[detect] candidate_merge_sites enumerated {len(sites)} MergeSite(s)")
    return {str(s.label) for s in sites}


def _heldout_weighted_delta(probe: dict, heldout_names: list[str]) -> dict:
    """Held-out-scoped oracle metric deltas (only with --with-oracle).

    probe_split_oracle rescores the WHOLE brain; slice its per_swc frames to the
    held-out skeletons and run-length-weight, so the delta reflects the gate's scope
    rather than the whole brain. Secondary context only — the DECISION is the counts.
    """
    base_sw = probe["baseline"].per_swc
    orc_sw = probe["oracle"].per_swc
    hn = set(heldout_names)
    base_h = base_sw.loc[base_sw.index.isin(hn)]
    orc_h = orc_sw.loc[orc_sw.index.isin(hn)]
    out = {}
    for m in ("Edge Accuracy", "% Merged Edges", "% Split Edges"):
        try:
            out[m] = (scoring._weighted_avg(orc_h, m) - scoring._weighted_avg(base_h, m))
        except Exception:
            out[m] = float("nan")
    return out


def main() -> int:
    p = argparse.ArgumentParser(
        description="Phase 0: merge-repair reward ceiling (n_mergeable ∩ n_detectable).")
    p.add_argument("--brain", required=True,
                   help="the GATE/test brain (cross-brain: the whole held-out brain)")
    p.add_argument("--mcl", type=int, default=10,
                   help="min_cable_length of the fragment cache used for DETECTION "
                        "(candidate_merge_sites); must match the run's --mcl. Default 10.")
    p.add_argument("--min-nodes", type=int, default=50,
                   help="a label is a merge error iff it lands on >= this many nodes on "
                        ">= 2 held-out GT neurons (=_MERGE_MIN_NODES; matches the oracle "
                        "and %Merged Edges). Default 50.")
    p.add_argument("--heldout-fraction", type=float, default=None,
                   help="per-brain regime: reserve this within-brain fraction as held-out "
                        "(else the WHOLE brain is held-out, i.e. cross-brain gate scope)")
    p.add_argument("--split-seed", type=int, default=0,
                   help="seed for --heldout-fraction split (ignored for whole-brain)")
    p.add_argument("--with-oracle", action="store_true",
                   help="ALSO run the (slow) oracle rescore for held-out sparse-metric "
                        "deltas — secondary context; the go/no-go is the counts.")
    args = p.parse_args()

    brain = args.brain
    paths = scoring.BrainPaths(brain)

    # DETECTION graph: the mcl-filtered fragment cache (same as run_evolution).
    frag_cache = ds.default_cache_path(brain, min_cable_length=args.mcl)
    if not os.path.exists(frag_cache):
        raise SystemExit(
            f"[{brain}] fragment cache not found: {frag_cache}\n"
            f"  build it (notebooks/load_skeletons.ipynb, min_cable_length={args.mcl}) "
            f"or check --brain/--mcl.")
    print(f"[{brain}] loading fragment graph (mcl={args.mcl}): {frag_cache}")
    fragments_graph, _gt, _ = ds.load_cached_graphs(
        frag_cache, expect_brain=brain, expect_mcl=args.mcl)

    # SCORING state: the prepared brain (GT node snapshots). Reuses the shared cross-run
    # prepared cache if present (seconds); cold-builds from GCS (~30 min) only once ever.
    prepared_cache = str(PROJECT_ROOT / "proofreader_evolve" / "runs"
                         / f"prepared_{brain}.pkl")
    print(f"[{brain}] loading/preparing brain for GT scoring "
          f"(shared cache: {inc.shared_prepared_cache_path(brain)})")
    prepared = inc.get_or_build(paths, prepared_cache, verbose=True)

    heldout_names = _heldout_names(prepared, args.heldout_fraction, args.split_seed)
    scope = ("WHOLE BRAIN (cross-brain gate)" if not args.heldout_fraction
             or args.heldout_fraction >= 1.0
             else f"within-brain held-out (fraction={args.heldout_fraction}, "
                  f"seed={args.split_seed})")
    print(f"[{brain}] held-out scope: {scope} — {len(heldout_names)}/"
          f"{len(prepared.gt_names)} GT neurons")

    # The two identifications + their intersection.
    mergeable = mergeable_labels(prepared, heldout_names, args.min_nodes)
    detected = detected_labels(fragments_graph, verbose=True)
    detectable = mergeable & detected
    recall_gap = mergeable - detected          # mergeable but NOT reachable (ceiling loss)

    ceiling = len(detectable)
    n_merge = len(mergeable)

    print("\n" + "=" * 64)
    print(f"PHASE 0 — merge-repair reward ceiling (brain {brain})")
    print("=" * 64)
    print(f"  n_mergeable  (GT-label, held-out scope)   : {n_merge}")
    print(f"  n_detectable (∩ geometric-walk MergeSites) : {ceiling}   <-- REWARD CEILING")
    print(f"  detector recall gap (mergeable, unreachable): {len(recall_gap)}")
    print(f"  detected labels total (all, incl. non-merge): {len(detected)}")
    if n_merge:
        print(f"  detectable fraction of mergeable           : {ceiling / n_merge:.0%}")

    if args.with_oracle:
        print("\n[oracle] running full baseline+oracle rescore (slow) for held-out "
              "sparse-metric deltas (SECONDARY context; not the decision)...")
        probe = inc.probe_split_oracle(prepared, min_nodes=args.min_nodes, verbose=True)
        d = _heldout_weighted_delta(probe, heldout_names)
        print("  held-out oracle deltas (perfect split - baseline):")
        for m, v in d.items():
            print(f"    {m:<16} {v:+.4f}")
        print("  (small deltas here do NOT veto: these are the sparse metrics that "
              "flat-lined the split side; trust the COUNTS above.)")

    # Go/no-go verdict on the COUNT (the ceiling), per the plan's kill-criterion.
    print("\n" + "-" * 64)
    if ceiling == 0:
        verdict = ("NO-GO: zero reachable merge errors on held-out — a split_label "
                   "reward would have nothing to score. Do NOT build the gate.")
    elif ceiling < 5:
        verdict = (f"MARGINAL: only {ceiling} reachable merge error(s) on held-out. "
                   f"The dense reward would be very sparse; likely not worth building "
                   f"unless this brain is unusually undertraced. Consider a denser gate "
                   f"brain first.")
    else:
        verdict = (f"GO: {ceiling} reachable merge errors on held-out is a scorable "
                   f"signal. Proceed to Phase 1 (classify_split_edits). Record this as "
                   f"the ceiling — later runs should report 'captured K/{ceiling}'.")
    print(verdict)
    if recall_gap:
        print(f"note: {len(recall_gap)} mergeable label(s) are UNREACHABLE (no MergeSite) "
              f"— pure detector recall gap; widening candidate_merge_sites params (or a "
              f"new detector) is the only lever for those, not the policy.")
    print("-" * 64)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
