"""
Incremental scorer — pays the expensive whole-brain load ONCE, scores each
candidate in seconds.

Why this exists
---------------
``scoring.score`` calls the metric package's ``evaluate(...)``, which every time
re-instantiates the TensorStore reader, re-reads ~10k fragment SWCs from GCS,
rebuilds graphs, and re-labels all GT skeletons against the image — ~25-30 min
(see notebooks/evaluate_skeleton_metrics.ipynb timings). But the only thing a
candidate proofreader changes is the ``LabelHandler`` equivalence collapse
(which fragment labels are unified). The raw segment ID under each GT node, and
the fragment graphs themselves, are invariant across candidates.

Verified against the source:
  - graph_loading.GraphLoader._label_graph -> get_patch_labels -> get_label:
      node_label[i] = label_handler.get( patch[voxel] )
    The image read `patch[voxel]` (the raw segment id) does NOT depend on the
    handler. The handler only maps raw -> class id.
  - graph_classes.LabeledGraph.relabel_nodes(handler): re-derives node_label
    from a handler, then fix_label_misalignments(). This is the entire
    per-candidate delta and is pure CPU.

So we:
  1. BUILD (once, ~30 min, then pickled to disk): load GT + fragment graphs with
     an *identity* handler and misalignment-fix DEFERRED, and snapshot each GT
     node's raw segment id and each fragment's raw label.
  2. SCORE (per candidate, seconds): rebuild a LabelHandler from the candidate's
     label-pairs, reset every graph from its raw snapshot, relabel + fix, then
     run the REAL metric classes (Evaluator) over ALL GT and filter the reported
     rows to the requested subset. Scoring the full set is required for the
     cross-skeleton merge metric (see score_incremental); per-skeleton metrics
     are unaffected by the filter.

Because step 2 reuses the package's own metric classes on relabeled graphs, the
numbers are identical to ``evaluate()`` — not an approximation. ``verify.py``
asserts this on the baseline and a sample candidate.
"""

from __future__ import annotations

import os
import pickle
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from segmentation_skeleton_metrics.data_handling.graph_loading import (
    GraphLoader,
    LabelHandler,
)
from segmentation_skeleton_metrics.utils.img_util import TensorStoreImage
from segmentation_skeleton_metrics.skeleton_metrics import (
    SplitCountMetric,
    MergeCountMetric,
    SplitEdgePercentMetric,
    OmitEdgePercentMetric,
    MergedEdgePercentMetric,
    ERLMetric,
    NormalizedERLMetric,
    EdgeAccuracyMetric,
    SplitRateMetric,
    MergeRateMetric,
)

from proofreader_evolve.harness import scoring  # ScoreResult, BrainPaths, helpers


class _SerialExecutor:
    """A drop-in, single-process stand-in for ProcessPoolExecutor.

    Mimics the submit()/as_completed() + context-manager API the library uses,
    but runs each task inline in the calling process. Returned futures are
    already-completed, so concurrent.futures.as_completed yields them immediately.
    """

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def submit(self, fn, *args, **kwargs):
        from concurrent.futures import Future

        fut = Future()
        try:
            fut.set_result(fn(*args, **kwargs))
        except BaseException as e:  # propagate exactly like a real pool would
            fut.set_exception(e)
        return fut


def _cap_graph_loading_workers(max_workers: int) -> None:
    """Force graph loading to run serially, in-process.

    Two independent failure modes hit the library's parallel graph loaders on a
    constrained interactive node, and the serial shim kills both:

      1. OOM. ``_build_graphs_from_swcs`` uses a bare ``ProcessPoolExecutor()``
         (graph_loading.py:264) = os.cpu_count() workers, each building and
         pickling a hundreds-of-thousands-of-nodes graph. On a ~9 GB memory
         cgroup that gets OOM-killed -> BrokenProcessPool.
      2. Fork-after-threads deadlock. The build opens TensorStore (gRPC bg
         threads) in the MAIN process for the segmentation; the pool then forks,
         copying a locked thread state -> children wedge on the comm pipe
         (observed: Threads=1, blocked on pipe_write, ~0% CPU, no progress).

    Capping worker count doesn't fix (2). Running serially does, and since the
    build is a ONE-TIME ~30-40 min step that we pickle, the lost parallelism is
    a non-issue. ``max_workers`` is accepted for API symmetry; >1 still runs
    serially here (correctness over speed for the one-time build).

    We replace ONLY ``ProcessPoolExecutor`` (the CPU-bound graph builder that
    forks and deadlocks/OOMs). The ``ThreadPoolExecutor`` used for labeling
    (line 361) is left intact but capped: it's I/O-bound GCS patch reads where
    threads genuinely help and there is no fork, so serializing it would only
    make the build slower. The cap keeps its memory bounded.
    ``ProcessPoolExecutor`` is a module-level name, so we rebind it. Idempotent.
    """
    import functools
    from concurrent.futures import ThreadPoolExecutor
    from segmentation_skeleton_metrics.data_handling import graph_loading as _gl

    if not getattr(_gl.ProcessPoolExecutor, "_serial_shim", False):
        _SerialExecutor._serial_shim = True
        _gl.ProcessPoolExecutor = _SerialExecutor
    if not getattr(_gl.ThreadPoolExecutor, "_te_capped", False):
        capped_te = functools.partial(ThreadPoolExecutor, max_workers=max_workers * 4)
        capped_te._te_capped = True
        _gl.ThreadPoolExecutor = capped_te


class PreparedBrain:
    """The once-loaded, candidate-invariant state for one brain.

    Holds the built GT + fragment graphs plus a snapshot of every GT node's raw
    segment id and every fragment's raw label, so any candidate can be scored by
    resetting from these snapshots. Picklable, so the ~30 min build is paid once
    ever per (brain, segmentation).
    """

    def __init__(self, brain_id, anisotropy, gt_graphs, fragment_graphs):
        self.brain_id = brain_id
        self.anisotropy = anisotropy
        self.gt_graphs = gt_graphs
        self.fragment_graphs = fragment_graphs

        # Snapshot raw labels (the candidate-invariant truth).
        self._gt_raw = {
            name: np.array(g.node_label, dtype=object) for name, g in gt_graphs.items()
        }
        self._frag_raw_label = {
            key: g.label for key, g in fragment_graphs.items()
        }
        # The LabelHandler universe: every fragment label.
        self.all_fragment_labels = sorted(
            {str(v) for v in self._frag_raw_label.values()} - {"0"}
        )

    # --- persistence ---
    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path) -> "PreparedBrain":
        with open(path, "rb") as f:
            return pickle.load(f)

    @property
    def gt_names(self):
        return sorted(self.gt_graphs.keys())

    # --- per-candidate reset ---
    def _apply_handler(self, handler, coordinate_aware: bool = False):
        """Reset every graph from its raw snapshot and apply the candidate handler.

        Mirrors the real pipeline order exactly: derive class labels from raw via
        the handler, then fix_label_misalignments() on the class labels.

        ``handler`` may be a metric-package ``LabelHandler`` (``get(raw)``) or our
        ``EditHandler`` (``get(raw, xyz)``). When ``coordinate_aware`` is True the
        GT-node lookup passes the node's physical coordinate, so a coordinate-
        dependent ``split_label`` edit can map one raw segment id to different
        pseudo-labels by location. Fragment labels are always looked up by id only
        (no per-node coordinate); virtual splits act on the GT-node labels that
        the cross-skeleton merge metric reads, which is what lets a merged segment
        register as split without materializing a new dense volume.
        """
        for name, g in self.gt_graphs.items():
            raw = self._gt_raw[name]
            if coordinate_aware:
                g.node_label = np.array(
                    [handler.get(raw[i], g.node_xyz(i)) for i in range(len(raw))],
                    dtype=object,
                )
            else:
                # relabel from raw -> class (same as LabeledGraph.relabel_nodes body)
                g.node_label = np.array(
                    [handler.get(raw[i]) for i in range(len(raw))], dtype=object
                )
            g.fix_label_misalignments()
            # reset per-run accumulators the metric classes mutate
            g.labels_with_merge = set()
            g.labeled_run_length = 0
            g.set_kdtree()
        for key, g in self.fragment_graphs.items():
            g.label = handler.get(self._frag_raw_label[key])


def build_prepared_brain(
    paths: scoring.BrainPaths, verbose: bool = True, max_workers: int = 2
) -> PreparedBrain:
    """Run the expensive load ONCE (identity handler, misalignment-fix deferred).

    This is the ~30 min step; pickle the result with PreparedBrain.save and the
    cost is never paid again for this brain.

    max_workers caps the graph-loading pools so the build fits a memory-limited
    cgroup. Default 2 is safe for a ~9 GB session; raise it if you have more RAM.
    """
    _cap_graph_loading_workers(max_workers)
    segmentation = TensorStoreImage(paths.segmentation_path, swap_axes=True)
    identity = LabelHandler()  # LazyMapping: get(raw) -> raw segment id

    # GT graphs: labeled against the image, but DEFER misalignment fix so the raw
    # snapshot is truly raw; per-candidate _apply_handler does the fix on class
    # labels (faithful to evaluate()'s read-then-fix order).
    if verbose:
        print("\n(1) Load Ground Truth [incremental build]")
    gt_loader = GraphLoader(
        anisotropy=paths.anisotropy,
        is_groundtruth=True,
        label_handler=identity,
        segmentation=segmentation,
        use_anisotropy=False,
        fix_label_misalignments=False,
        verbose=verbose,
    )
    gt_graphs = gt_loader(paths.gt_path)

    if verbose:
        print("\n(2) Load Fragments [incremental build]")
    frag_loader = GraphLoader(
        anisotropy=paths.anisotropy,
        is_groundtruth=False,
        label_handler=identity,
        use_anisotropy=False,
        verbose=verbose,
    )
    fragment_graphs = frag_loader(paths.fragments_path)

    return PreparedBrain(paths.brain_id, paths.anisotropy, gt_graphs, fragment_graphs)


# The metric set, instantiated quietly. Mirrors evaluate.Evaluator's wiring.
def _core_metrics(verbose):
    return {
        "# Splits": SplitCountMetric(verbose=verbose),
        "# Merges": MergeCountMetric(verbose=verbose),
        "% Split Edges": SplitEdgePercentMetric(verbose=verbose),
        "% Omit Edges": OmitEdgePercentMetric(verbose=verbose),
        "% Merged Edges": MergedEdgePercentMetric(verbose=verbose),
        "ERL": ERLMetric(verbose=verbose),
    }


def _derived_metrics(verbose):
    return {
        "Normalized ERL": NormalizedERLMetric(verbose=verbose),
        "Edge Accuracy": EdgeAccuracyMetric(verbose=verbose),
        "Split Rate": SplitRateMetric(verbose=verbose),
        "Merge Rate": MergeRateMetric(verbose=verbose),
    }


def score_incremental(
    prepared: PreparedBrain,
    label_pairs=None,
    gt_swc_names=None,
    edits=None,
    verbose: bool = False,
) -> scoring.ScoreResult:
    """Score a candidate (set of edits) in seconds.

    Reuses the real metric classes on relabeled graphs, so results equal
    evaluate()'s. Returns a scoring.ScoreResult (same shape as scoring.score),
    so candidate.py / run_evolution.py are drop-in compatible.

    Parameters
    ----------
    prepared : PreparedBrain
    label_pairs : sequence of (label, label), optional
        Legacy split-correction edits (merge-only). None/empty == baseline. Routed
        through the metric package's LabelHandler exactly as before, so existing
        callers get byte-identical results.
    gt_swc_names : iterable of str, optional
        Restrict scoring to this GT subset (train or held-out).
    edits : list of typed edits, optional
        Typed proofreading edits (see harness.edit_handler): ``merge_labels``,
        ``split_label``, ``flag_review``, ``reject_candidate``. When provided
        (and non-empty), scoring uses the coordinate-aware EditHandler so a
        ``split_label`` can virtually partition a merged segment by location.
        Mutually exclusive with ``label_pairs``; if both are given, ``edits``
        wins (and a pure-merge ``edits`` list is equivalent to ``label_pairs``).
    """
    t0 = time.monotonic()
    if edits:
        from proofreader_evolve.harness.edit_handler import EditHandler
        handler = EditHandler(edits, all_labels=prepared.all_fragment_labels)
        prepared._apply_handler(handler, coordinate_aware=True)
    else:
        pairs = [(str(a), str(b)) for a, b in (label_pairs or [])]
        handler = LabelHandler(labels=set(prepared.all_fragment_labels), label_pairs=pairs)
        prepared._apply_handler(handler)

    # IMPORTANT: score on the FULL GT set, then filter rows to the requested
    # subset. Some metrics are cross-skeleton — MergedEdgePercentMetric calls
    # detect_label_intersections(), which flags a merge only when BOTH skeletons
    # sharing a label are present. If we scored a train/held-out subset directly,
    # a merge spanning the train<->held-out boundary would be invisible in either
    # subset (its partner is missing), so % Merged Edges — and therefore the
    # gate's Edge Accuracy = 100 - %Split - %Omit - %Merged — would be silently
    # inflated. Full-set-then-filter reproduces evaluate()'s semantics exactly
    # (this is what scoring.score does after evaluate()). The per-skeleton metrics
    # are independent, so filtering afterward changes nothing for them.
    gt_full = prepared.gt_graphs

    metrics = _core_metrics(verbose)
    derived = _derived_metrics(verbose)

    # Build results frame exactly like Evaluator.init_results (over ALL GT).
    cols = ["SWC Run Length"] + list(metrics) + list(derived)
    results = pd.DataFrame(np.nan, index=sorted(gt_full), columns=cols)
    for key, g in gt_full.items():
        results.loc[key, "SWC Run Length"] = g.run_length

    # Core metrics (mirror Evaluator.__call__).
    for name, metric in metrics.items():
        if name == "# Merges":
            results[name] = metric(gt_full, prepared.fragment_graphs)
        else:
            results.update(metric(gt_full))
    # Derived metrics.
    for name, metric in derived.items():
        results[name] = metric(gt_full, results)

    # Attribute each detected merge back to the candidate edit that caused it.
    # MergeCountMetric.merge_sites is a per-merge DataFrame with GroundTruth_ID,
    # Label (the post-collapse class id) and World location. The handler's
    # inverse_mapping turns that class id back into the raw fragment labels fused
    # into it; if a class holds >=2 raw labels, the candidate's edits created it,
    # so we tag the site with those raw labels. This is the "which edit caused
    # which merge" signal the reviser needs (per-skeleton deltas can't show it).
    merge_sites = _attribute_merge_sites(metrics["# Merges"], handler)

    # Now restrict the *reported* rows to the requested subset.
    if gt_swc_names is not None:
        names = set(gt_swc_names)
        results = results[results.index.isin(names)]
        if merge_sites is not None and len(merge_sites):
            merge_sites = merge_sites[merge_sites["GroundTruth_ID"].isin(names)]

    elapsed = time.monotonic() - t0
    tracked = {m: scoring._weighted_avg(results, m) for m in scoring.TRACKED_METRICS}
    return scoring.ScoreResult(
        primary=tracked[scoring.PRIMARY_METRIC],
        metrics=tracked,
        per_swc=results,
        output_dir="",  # in-memory; nothing written
        seconds=elapsed,
        merge_sites=merge_sites,
    )


def _attribute_merge_sites(merge_metric, handler):
    """Tag each merge site with the raw fragment labels fused into its class.

    Returns a copy of MergeCountMetric.merge_sites with an extra
    ``Fused_Labels`` column (the raw labels the handler unified for that site's
    class id) and ``Caused_By_Edit`` (True when >=2 raw labels were fused, i.e.
    a candidate edit — not the original segmentation — produced the class).
    Returns None when no merges were detected.
    """
    sites = getattr(merge_metric, "merge_sites", None)
    if sites is None or len(sites) == 0:
        return None
    sites = sites.copy()
    inv = handler.inverse_mapping

    def raw_labels(class_id):
        raws = inv.get(str(class_id))
        return sorted(raws) if raws else [str(class_id)]

    fused = [raw_labels(c) for c in sites["Label"]]
    sites["Fused_Labels"] = fused
    sites["Caused_By_Edit"] = [len(f) >= 2 for f in fused]
    return sites


def probe_split_oracle(prepared: PreparedBrain, min_nodes: int = 50, verbose: bool = False):
    """Diagnostic: the *ceiling* gain from a perfect ``split_label`` correction.

    DE-RISK MEASUREMENT (uses ground-truth knowledge — NOT a deployable policy).
    Before building a candidate generator and policy for merge repair, this
    answers: "if we could split every merge-causing segment perfectly, how much
    would the metrics actually improve?" If even the perfect split barely moves
    %Merged Edges / Edge Accuracy, the whole split_label track is not worth it.

    Method:
      1. Score the baseline (no edits).
      2. Find each raw segment label that lands (>= ``min_nodes`` nodes) on two or
         more *distinct* GT skeletons — i.e. a label causing a cross-skeleton
         merge (same rule as MergedEdgePercentMetric.detect_label_intersections).
      3. Oracle relabel: every GT node carrying such a label ``L`` gets
         ``L#<gt_skeleton_name>`` — a perfect split that removes the cross-skeleton
         intersection entirely. (A real seed-based split can only approximate this.)
      4. Re-score and report the deltas.

    Returns
    -------
    dict with ``baseline`` and ``oracle`` ScoreResults, the ``merged_labels``
    found, and a ``delta`` dict of tracked-metric changes (oracle - baseline).
    """
    # 1) Baseline.
    base = score_incremental(prepared, label_pairs=None, verbose=verbose)

    # 2) Which raw labels touch >=2 GT skeletons with >= min_nodes each?
    #    Build {raw_label -> {gt_name -> count}} from the raw snapshots.
    label_to_gt_counts: dict[str, dict[str, int]] = {}
    for name, raw in prepared._gt_raw.items():
        for lab in raw:
            lab = str(lab)
            if lab == "0":
                continue
            label_to_gt_counts.setdefault(lab, {}).setdefault(name, 0)
            label_to_gt_counts[lab][name] += 1
    merged_labels = {
        lab
        for lab, counts in label_to_gt_counts.items()
        if sum(1 for c in counts.values() if c >= min_nodes) >= 2
    }
    if verbose:
        print(f"[oracle] {len(merged_labels)} merge-causing labels "
              f"(>= {min_nodes} nodes on >=2 GT skeletons)")

    # 3) Oracle relabel directly on GT node labels: L -> L#<gt_name> for merged L.
    #    (Reset fragment labels from raw too, for a clean baseline comparison.)
    for name, g in prepared.gt_graphs.items():
        raw = prepared._gt_raw[name]
        g.node_label = np.array(
            [f"{raw[i]}#{name}" if str(raw[i]) in merged_labels else raw[i]
             for i in range(len(raw))],
            dtype=object,
        )
        g.fix_label_misalignments()
        g.labels_with_merge = set()
        g.labeled_run_length = 0
        g.set_kdtree()
    for key, g in prepared.fragment_graphs.items():
        g.label = prepared._frag_raw_label[key]

    # 4) Re-score with the oracle labels in place (mirror score_incremental's body
    #    WITHOUT re-applying a handler, since we set node_label directly above).
    gt_full = prepared.gt_graphs
    metrics = _core_metrics(verbose)
    derived = _derived_metrics(verbose)
    cols = ["SWC Run Length"] + list(metrics) + list(derived)
    results = pd.DataFrame(np.nan, index=sorted(gt_full), columns=cols)
    for key, g in gt_full.items():
        results.loc[key, "SWC Run Length"] = g.run_length
    for nm, metric in metrics.items():
        if nm == "# Merges":
            results[nm] = metric(gt_full, prepared.fragment_graphs)
        else:
            results.update(metric(gt_full))
    for nm, metric in derived.items():
        results[nm] = metric(gt_full, results)
    tracked = {m: scoring._weighted_avg(results, m) for m in scoring.TRACKED_METRICS}
    oracle = scoring.ScoreResult(
        primary=tracked[scoring.PRIMARY_METRIC],
        metrics=tracked, per_swc=results, output_dir="", seconds=0.0,
    )

    delta = {m: oracle.metrics.get(m, float("nan")) - base.metrics.get(m, float("nan"))
             for m in scoring.TRACKED_METRICS}
    return {
        "baseline": base,
        "oracle": oracle,
        "merged_labels": sorted(merged_labels),
        "delta": delta,
    }


def get_or_build(
    paths: scoring.BrainPaths, cache_path: str, verbose=True, max_workers: int = 2
) -> PreparedBrain:
    """Load the PreparedBrain pickle if present, else build it once and save."""
    if os.path.exists(cache_path):
        if verbose:
            print(f"Loading prepared brain from {cache_path}")
        return PreparedBrain.load(cache_path)
    if verbose:
        print(f"No prepared cache at {cache_path}; building (one-time ~30 min)...")
    prepared = build_prepared_brain(paths, verbose=verbose, max_workers=max_workers)
    prepared.save(cache_path)
    if verbose:
        print(f"Saved prepared brain -> {cache_path}")
    return prepared


if __name__ == "__main__":
    # De-risk probe runner: measures the ceiling gain from perfect merge-splitting.
    #   python proofreader_evolve/harness/incremental_scoring.py --brain 789202
    import argparse
    from pathlib import Path

    p = argparse.ArgumentParser(description="Oracle split_label de-risk probe.")
    p.add_argument("--brain", default="789202")
    p.add_argument("--min-nodes", type=int, default=50)
    args = p.parse_args()

    _paths = scoring.BrainPaths(args.brain)
    _cache = str(Path(__file__).resolve().parent.parent / "runs" / f"prepared_{args.brain}.pkl")
    _prepared = get_or_build(_paths, _cache, verbose=True)
    out = probe_split_oracle(_prepared, min_nodes=args.min_nodes, verbose=True)

    print(f"\n=== Oracle split_label ceiling (brain {args.brain}) ===")
    print(f"merge-causing labels: {len(out['merged_labels'])}")
    b, o = out["baseline"].metrics, out["oracle"].metrics
    for m in scoring.TRACKED_METRICS:
        print(f"  {m:<16} {b.get(m, float('nan')):>10.4f} -> {o.get(m, float('nan')):>10.4f}"
              f"   (Δ {out['delta'][m]:+.4f})")
    print("\nIf %Merged Edges / Edge Accuracy barely move, the split_label track")
    print("is not worth building; if they move a lot, proceed to a candidate")
    print("generator + policy. (#Merges may not move: it keys on fragment labels,")
    print("not GT-node labels — see the diagnosis.)")
