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
  6. SUBSET: scoring a GT subset == evaluate()-then-filter (cross-skeleton merge).
  7. SPLIT coordinate contract: a split_label's effect on % Merged Edges / Edge
     Accuracy is NOT directly verifiable against evaluate() (the union-only
     LabelHandler can't represent a split), so instead we verify the thing that
     actually makes split correct — the coordinate frames line up:
       (a) the agentic seed frame (fragment/GT node_xyz, microns x,y,z) matches
           the metrics frame (LabeledGraph.node_xyz(i)) the split is compared in,
       (b) apply_split assigns each node to its nearer seed in that frame, and
       (c) a degenerate split (both seeds coincident) is a metric no-op.
     A swapped axis or wrong anisotropy anywhere in the chain trips (a).
  8. FRAGMENT-SPLIT # MERGES: a synthetic merge (one fragment label spanning two
     GT neurons) whose official ``# Merges`` count, computed by the package's own
     MergeCountMetric, drops after a split_label virtually partitions the fused
     fragment graph. This is what P0-2 fixed: before the fragment-graph split,
     ``# Merges`` keyed on ``fragment.label in gt.node_labels()`` and was blind to
     a split (the fragment kept label ``L`` while GT became ``L#a``/``L#b``).
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
    "Split Accuracy": 0.05,  # = 100 - % Split Edges (auxiliary over-split watchdog)
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


def _report_recall_gap(prepared, frags, candidate_min_cable_um: float = 100.0) -> bool:
    """Quantify the candidate-vs-scoring label-universe gap (INFORMATIONAL).

    Candidate sites are enumerated over the agentic cache (filtered at
    ``min_cable_length`` µm), but scoring loads the metrics fragment graph with NO
    length filter. So the scorer knows segment labels the candidate stream can
    never propose — a silent recall ceiling (not a correctness bug). This reports:

      * candidate universe size vs scoring universe size and their overlap,
      * how many labels are scoring-only (proposable=no, scorable=yes),
      * the cable length of those scoring-only segments, to confirm the gap is
        the expected "short fragments dropped by the filter" and NOT a deeper
        id-derivation mismatch. Any scoring-only segment LONGER than the filter
        threshold is flagged — that would mean labels are being lost for a reason
        other than length (the thing to actually worry about).

    Always returns True (informational); it never fails the run. The number it
    prints is the recall headroom you'd recover by lowering the candidate cache's
    min_cable_length.
    """
    print("\n--- Candidate vs scoring label universe (recall gap) ---")
    candidate = set(ds.list_fragment_labels(frags))
    scoring = set(map(str, prepared.all_fragment_labels))
    scoring_only = scoring - candidate
    candidate_only = candidate - scoring  # expected empty: scoring is a superset

    overlap = len(candidate & scoring)
    print(f"  candidate universe : {len(candidate):>8} labels (mcl-filtered cache)")
    print(f"  scoring universe   : {len(scoring):>8} labels (unfiltered metrics load)")
    print(f"  overlap            : {overlap:>8} "
          f"({100 * overlap / max(len(candidate), 1):.1f}% of candidates scorable)")
    print(f"  scoring-only       : {len(scoring_only):>8} "
          f"(scorable but NOT proposable — the recall ceiling)")
    if candidate_only:
        print(f"  [WARN] candidate-only: {len(candidate_only)} labels are in the "
              f"candidate cache but NOT in the scoring universe — these merge edits "
              f"silently no-op. Examples: {sorted(candidate_only)[:5]}")

    # Total cable length per scoring-only SEGMENT (sum over its fragment
    # components), to test the "only short fragments are missing" hypothesis.
    if scoring_only:
        seg_len: dict[str, float] = {}
        for g in prepared.fragment_graphs.values():
            seg = str(getattr(g, "segment_id", None) or g.label)
            if seg in scoring_only and g.number_of_nodes() > 0:
                root = next(iter(g.nodes))
                seg_len[seg] = seg_len.get(seg, 0.0) + g.run_length_from(root)
        if seg_len:
            lengths = np.array(list(seg_len.values()))
            longer = {s: L for s, L in seg_len.items() if L >= candidate_min_cable_um}
            print(f"  scoring-only cable length (µm): "
                  f"min={lengths.min():.1f} median={np.median(lengths):.1f} "
                  f"max={lengths.max():.1f}")
            if longer:
                print(f"  [WARN] {len(longer)} scoring-only segment(s) are >= "
                      f"{candidate_min_cable_um} µm — NOT explained by the length "
                      f"filter; possible id-derivation mismatch. "
                      f"Examples: {sorted(longer)[:5]}")
            else:
                print(f"  [OK ] all scoring-only segments are < "
                      f"{candidate_min_cable_um} µm — the gap is the benign length "
                      f"filter, exactly as expected (most pairs match; only the "
                      f"short fragments are missing).")
    print("  (informational — lowering the candidate cache min_cable_length "
          "recovers this recall.)")
    return True


def _verify_split_contract(prepared, agentic_gt, tol_um: float = 1.0) -> bool:
    """Verify the split_label coordinate contract (no evaluate() oracle exists).

    split_label compares each GT node's coordinate to two seed points and assigns
    it to the nearer one. Seeds are read by the policy from the AGENTIC graph's
    node_xyz; the assignment happens at score time on the METRICS graph's
    node_xyz(i). Those two frames must be the same physical (microns, x,y,z) space
    or the bisecting plane silently shifts. We check three things, all from data
    already in memory (no extra cloud read):

      (a) FRAME MATCH: for shared GT skeletons, the metrics-graph node coordinates
          are a permutation-free match to the agentic-graph coordinates (same
          point cloud, same axis order, same anisotropy). Catches swap_axes /
          anisotropy / use_anisotropy misconfig.
      (b) NEAREST-SEED: EditHandler.apply_split assigns a node to '#a' iff it is
          closer to seed_a than seed_b in the metrics frame.
      (c) NO-OP SPLIT: coincident seeds => every node maps to one pseudo-label
          (a split that partitions nothing must not change which nodes share L).
    """
    from proofreader_evolve.harness.edit_handler import EditHandler, PSEUDO_SEP

    print("\n--- SPLIT coordinate contract ---")
    ok = True

    # Pick a GT skeleton present in BOTH graphs.
    metrics_names = set(prepared.gt_names)
    agentic_names = {
        agentic_gt.node_segment_id(n) for n in agentic_gt.nodes
    } if hasattr(agentic_gt, "node_segment_id") else set()
    shared = sorted(metrics_names & agentic_names)
    if not shared:
        print("  [SKIP] no GT skeleton name shared between agentic and metrics "
              f"graphs (agentic={len(agentic_names)}, metrics={len(metrics_names)})")
        return True
    name = shared[0]

    # (a) FRAME MATCH — compare the two coordinate clouds for this skeleton.
    mg = prepared.gt_graphs[name]
    metrics_xyz = np.array([mg.node_xyz(i) for i in mg.nodes])
    agentic_idx = [n for n in agentic_gt.nodes if agentic_gt.node_segment_id(n) == name]
    agentic_xyz = np.asarray(agentic_gt.node_xyz)[agentic_idx]
    # Match without assuming node-id alignment: nearest-neighbour each metrics
    # point to the agentic cloud; the frames agree iff every match is ~0 µm.
    from scipy.spatial import cKDTree
    tree = cKDTree(agentic_xyz)
    d, _ = tree.query(metrics_xyz, k=1)
    worst = float(np.max(d)) if len(d) else float("nan")
    frame_ok = worst <= tol_um
    ok = ok and frame_ok
    print(f"  [{'OK ' if frame_ok else 'FAIL'}] (a) frame match on {name}: "
          f"max nearest-point gap = {worst:.4g} µm (tol {tol_um}); "
          f"agentic axes/anisotropy == metrics' — split seeds land in the right place")

    # (b) NEAREST-SEED — seeds straddling the skeleton's x-extent.
    xs = metrics_xyz[:, 0]
    lo_pt = metrics_xyz[int(np.argmin(xs))]
    hi_pt = metrics_xyz[int(np.argmax(xs))]
    eh = EditHandler(
        [{"kind": "split_label", "label": name,
          "seed_a_xyz": tuple(lo_pt), "seed_b_xyz": tuple(hi_pt)}],
        all_labels=[name],
    )
    mism = 0
    for i in mg.nodes:
        xyz = mg.node_xyz(i)
        expect = "a" if np.linalg.norm(lo_pt - xyz) <= np.linalg.norm(hi_pt - xyz) else "b"
        got = eh.get(name, xyz).split(PSEUDO_SEP)[-1]
        mism += (got != expect)
    seed_ok = mism == 0
    ok = ok and seed_ok
    print(f"  [{'OK ' if seed_ok else 'FAIL'}] (b) nearest-seed assignment: "
          f"{mism} mismatches over {len(metrics_xyz)} nodes")

    # (c) NO-OP SPLIT — coincident seeds => one pseudo-label for all nodes.
    eh0 = EditHandler(
        [{"kind": "split_label", "label": name,
          "seed_a_xyz": tuple(lo_pt), "seed_b_xyz": tuple(lo_pt)}],
        all_labels=[name],
    )
    pseudo = {eh0.get(name, mg.node_xyz(i)) for i in mg.nodes}
    noop_ok = len(pseudo) == 1
    ok = ok and noop_ok
    print(f"  [{'OK ' if noop_ok else 'FAIL'}] (c) coincident-seed no-op: "
          f"{len(pseudo)} distinct pseudo-label(s) (expected 1)")

    return ok


def _verify_handler_parity(prepared, verbose: bool = False) -> bool:
    """Verify the two no-edit scoring paths agree PER SKELETON (handler parity).

    The evolution loop scores the no-edit BASELINE via the LabelHandler path
    (``score_incremental(label_pairs=None)``) but scores every CANDIDATE via the
    EditHandler path (``score_incremental(edits=...)``). ``write_failure_report``
    then prints candidate-minus-baseline deltas — which is only meaningful if the
    two handlers label an UNEDITED graph identically. If they diverge (e.g. one
    reduces a raw label to its segment-id prefix and the other keeps the raw
    string), the per-skeleton deltas — most visibly the ``# Splits`` count — are
    apples-to-oranges, NOT a real effect of the candidate's edits.

    We score the same graph two ways with NO real edits:
      - LabelHandler path: label_pairs=None
      - EditHandler path : a single inert ``flag_review`` edit (forces the
        ``if edits:`` branch without changing any label)
    and require every metric to match per skeleton.

    If they MATCH (expected): the failure-report deltas are trustworthy — a
    surprising ``d#Splits`` is then a REAL effect (e.g. merges let
    ``fix_label_misalignments`` fill background gaps, adding distinct labels and
    coverage, which raises both ``# Splits`` and Edge Accuracy together), not a
    path artifact. If they MISMATCH: the baseline and candidate are scored on
    different label spaces and the deltas are not comparable — fix the report to
    score its baseline through the same EditHandler path as the candidate.
    """
    print("\n--- Handler parity (LabelHandler vs EditHandler, no edits) ---")
    label_path = inc.score_incremental(prepared, label_pairs=None, verbose=verbose)
    edit_path = inc.score_incremental(
        prepared, edits=[{"kind": "flag_review", "reason": "parity-probe"}],
        verbose=verbose,
    )
    ok = _compare("no-edit: LabelHandler vs EditHandler", edit_path, label_path)
    if not ok:
        print("  -> The baseline (LabelHandler) and candidate (EditHandler) paths "
              "label unedited graphs DIFFERENTLY. Failure-report deltas (esp. "
              "#Splits) are path-mismatched, not real. Fix: score the report's "
              "baseline through the SAME EditHandler path as the candidate.")
    return ok


def _verify_fragment_split_merges() -> bool:
    """Synthetic check: a split_label edit reduces the OFFICIAL ``# Merges``.

    Self-contained (no brain build, no cloud): builds two ground-truth neurons and
    ONE fragment label ``L`` that physically spans both — a real merge. Runs the
    package's own ``MergeCountMetric`` on (a) the unsplit fragment and (b) the
    virtually-split fragments from ``incremental_scoring._edited_fragment_graphs``,
    and asserts the count strictly drops. This is the guarantee P0-2 adds: before
    the fragment-graph split, ``# Merges`` was insensitive to ``split_label`` (the
    fragment kept label ``L`` while GT nodes became ``L#a``/``L#b``, so the metric's
    ``fragment.label in gt.node_labels()`` join missed it). Now the split yields
    two fragment graphs the metric re-counts geometrically.
    """
    from segmentation_skeleton_metrics.data_handling.graph_classes import (
        FragmentGraph, LabeledGraph,
    )
    from segmentation_skeleton_metrics.skeleton_metrics import MergeCountMetric
    from proofreader_evolve.harness.edit_handler import EditHandler
    from proofreader_evolve.harness import incremental_scoring as _inc

    print("\n--- Fragment-split # Merges contract (synthetic) ---")

    def _line(n, start, step):
        return [list(np.array(start) + i * np.array(step)) for i in range(n)]

    def _make_gt(name, start, n=80):
        g = LabeledGraph(anisotropy=(1, 1, 1), name=name)
        g.set_voxels(_line(n, start, (0, 0, 1)))
        g.add_edges_from([(i, i + 1) for i in range(n - 1)])
        g.init_node_labels()
        for i in g.nodes:
            g.node_label[i] = "L"  # both GTs carry L -> a cross-skeleton merge
        g.set_kdtree()
        return g

    gt_graphs = {"gtA": _make_gt("gtA", (0, 0, 0)), "gtB": _make_gt("gtB", (0, 200, 0))}

    fragL = FragmentGraph(anisotropy=(1, 1, 1), name="fragL", label="L", segment_id="L")
    fragL.set_voxels(_line(80, (0, 0, 0), (0, 0, 1)) + _line(80, (0, 200, 0), (0, 0, 1)))
    fragL.add_edges_from([(i, i + 1) for i in range(159)])

    class _P:
        pass
    prep = _P()
    prep.fragment_graphs = {"L": fragL}
    prep._frag_raw_label = {"L": "L"}

    def _count(handler):
        frags = _inc._edited_fragment_graphs(prep, handler)
        for g in gt_graphs.values():
            g.labels_with_merge = set()
            g.set_kdtree()
        res = MergeCountMetric(verbose=False)(gt_graphs, frags)
        return int(sum(res["# Merges"])), len(frags)

    base, n_base = _count(EditHandler([{"kind": "flag_review", "reason": "x"}],
                                      all_labels=["L"]))
    split, n_split = _count(EditHandler(
        [{"kind": "split_label", "label": "L",
          "seed_a_xyz": (0, 0, 0), "seed_b_xyz": (0, 200, 0)}], all_labels=["L"]))

    base_ok = base >= 1                      # the synthetic merge must be detected
    split_ok = split < base and n_split == 2  # split fixes it and yields 2 fragments
    ok = base_ok and split_ok
    print(f"  [{'OK ' if base_ok else 'FAIL'}] baseline detects the merge: "
          f"# Merges={base} ({n_base} fragment graph)")
    print(f"  [{'OK ' if split_ok else 'FAIL'}] split_label reduces it: "
          f"# Merges={split} ({n_split} fragment graphs, labels L#a/L#b)")
    if not ok:
        print("  -> _edited_fragment_graphs is NOT partitioning the fused fragment; "
              "split_label cannot move the official # Merges.")
    return ok


def _verify_graph_aware_split() -> bool:
    """Synthetic check: graph-aware split CORRECTS a point Euclidean misassigns.

    Self-contained. Builds a fasciculation — two neurites running parallel and
    joined only at their far-right ends (a real merge) — and places the two seeds
    so that a node near the LEFT end of arm A is spatially closest to arm B's seed
    but graph-closest to arm A's seed. Asserts:
      * Euclidean nearest-seed (no partition built) MISassigns the node to B, and
      * the graph-aware partition (multi-source Dijkstra on the fused subgraph)
        correctly assigns it to A.
    This is the crossing/fasciculation failure mode P0-3 targets. Also checks
    multi-seed composition (two split edits on one label do NOT clobber — they
    compose into a 3-way partition), guarding the old last-one-wins bug.
    """
    from segmentation_skeleton_metrics.data_handling.graph_classes import FragmentGraph
    from proofreader_evolve.harness.edit_handler import EditHandler, PSEUDO_SEP

    print("\n--- Graph-aware split contract (synthetic) ---")
    ok = True

    # Two parallel arms (y=0 and y=1), x=0..20, joined at the right (x=20).
    g = FragmentGraph(anisotropy=(1, 1, 1), name="L", label="L", segment_id="L")
    arm_a = [[0, 0, x] for x in range(0, 21)]
    arm_b = [[0, 1, x] for x in range(0, 21)]
    g.set_voxels(arm_a + arm_b)
    n = len(arm_a)
    g.add_edges_from([(i, i + 1) for i in range(n - 1)])
    g.add_edges_from([(n + i, n + i + 1) for i in range(len(arm_b) - 1)])
    g.add_edge(n - 1, 2 * n - 1)  # fuse the two arms at their far-right ends

    # seed_a at A's far-right end (20,0,0); seed_b at B's near-left end (0,1,0).
    h = EditHandler([{"kind": "split_label", "label": "L",
                      "seed_a_xyz": (20, 0, 0), "seed_b_xyz": (0, 1, 0)}],
                    all_labels=["L"])
    query = (1, 0, 0)  # a node near A's left end
    euclid = h.apply_split("L", query).split(PSEUDO_SEP)[-1]
    h.build_graph_partitions({"L": g})
    graph_aware = h.apply_split("L", query).split(PSEUDO_SEP)[-1]
    diverge_ok = euclid == "b" and graph_aware == "a"
    ok = ok and diverge_ok
    print(f"  [{'OK ' if diverge_ok else 'FAIL'}] divergence: euclidean={euclid} "
          f"(misassigns), graph-aware={graph_aware} (correct on arm A)")

    # Multi-seed composition: two edits on L compose to a 3-way partition.
    h3 = EditHandler([
        {"kind": "split_label", "label": "L", "seed_a_xyz": (0, 0, 0), "seed_b_xyz": (10, 0, 0)},
        {"kind": "split_label", "label": "L", "seeds": [{"xyz": (0, 20, 0)}]},
    ], all_labels=["L"])
    suffixes = sorted(s for s, _ in h3._split["L"])
    compose_ok = suffixes == ["a", "b", "c"]
    ok = ok and compose_ok
    print(f"  [{'OK ' if compose_ok else 'FAIL'}] multi-seed compose (no "
          f"last-one-wins): suffixes {suffixes} (expected a,b,c)")
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
    frags, agentic_gt, _ = ds.load_cached_graphs(ds.default_cache_path(args.brain))
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

    # --- 4) SPLIT coordinate contract (no evaluate() oracle; see docstring) ----
    ok4 = _verify_split_contract(prepared, agentic_gt)

    # --- 5) Handler parity: no-edit LabelHandler == no-edit EditHandler ---------
    # Guards the failure-report deltas (baseline=LabelHandler vs candidate=EditHandler).
    ok5 = _verify_handler_parity(prepared, verbose=args.verbose)

    # --- 6) Fragment-split # Merges contract (synthetic; no brain needed) -------
    # Proves split_label now moves the OFFICIAL # Merges (P0-2), not just %Merged.
    ok6 = _verify_fragment_split_merges()

    # --- 7) Graph-aware split contract (synthetic; no brain needed) -------------
    # Proves graph-aware partitioning corrects a Euclidean misassignment and that
    # multi-seed splits compose (P0-3).
    ok7 = _verify_graph_aware_split()

    # --- 8) Recall gap (informational — never fails the run) -------------------
    _report_recall_gap(prepared, frags)

    print("\n==================== RESULT ====================")
    if ok1 and ok2 and ok3 and ok4 and ok5 and ok6 and ok7:
        print("PASS — incremental scorer matches evaluate() on baseline, "
              "candidate, and GT subset; split coordinate contract holds; "
              "handler paths agree on unedited graphs; split_label reduces the "
              "official # Merges; graph-aware split corrects Euclidean errors.")
        return 0
    failed = [n for n, ok in
              [("baseline", ok1), ("candidate", ok2), ("subset", ok3),
               ("split-contract", ok4), ("handler-parity", ok5),
               ("fragment-split-merges", ok6), ("graph-aware-split", ok7)] if not ok]
    print(f"FAIL — checks failed: {', '.join(failed)}; do NOT trust the fast path "
          "(and do NOT use split_label if split-contract failed; failure-report "
          "deltas are unreliable if handler-parity failed).")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
