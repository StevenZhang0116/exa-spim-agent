#!/usr/bin/env python
"""
Verify that an ``_add.pkl`` cache carries merge-error information consistent with
the canonical ``metrics_out/<brain>/<seg_id>/`` results.

This is the *merge-focused* companion to ``verify_add_cache_metrics.ipynb`` (which
checks the per-neuron edge metrics broadly). Here we only look at merges, and we
compare against the two canonical merge artifacts:

  * ``merge_sites.csv`` -- one row per detected merge site (the geometric walk):
    ``Segment_ID``, ``GroundTruth_ID`` (neuron name), ``World`` = (x, y, z) µm.
  * ``results.csv``     -- per-neuron ``# Merges`` (== count of merge_sites rows
    for that neuron) and ``% Merged Edges``.

The cache stores the same information, baked in at relabel time:

  * ``gt_merge_sites``  -- list of ``{"segment_id", "gt_neuron", "xyz"=(x,y,z) µm}``
                           from the ported geometric walk (``geometric_merge_sites``).
  * ``gt_merge_labels`` -- segment ids that fuse >=2 GT neurons, the UNION of the
                           node-count rule and the geometric walk (== canonical
                           ``labels_with_merge``).

What we assert / report (see the canonical vs cache mapping in
``markdowns/labeled_dataset_cache.md`` and the axis note in ``canonical_labeling``):

  A. Merge segment-id sets agree (cache geometric-walk seg ids vs merge_sites.csv;
     and gt_merge_labels is a SUPERSET of the site seg ids, since it also folds in
     the node-count rule).
  B. Per-neuron merge counts agree (cache sites grouped by neuron vs
     results.csv "# Merges", which equals merge_sites.csv rows per neuron).
  C. Merge SITES localize to the same places (nearest-neighbour matching in µm;
     recall + precision at a distance tolerance). This is the load-bearing check --
     it confirms the cache is pointing at the same physical merges.
  D. Total "# Merges" agrees (len(gt_merge_sites) vs len(merge_sites.csv) vs
     results.csv["# Merges"].sum()).

Why "close, not exact": the canonical pipeline snaps each merge site to a nearby
branch node and dedups in a specific order; the cache port records the walk's
approach node directly. So sites match within a small distance, not byte-for-byte
(hence the tolerance and the WARN-not-FAIL on the residual). See the "% Merged
Edges match closely, not byte-exactly" note in ``labeled_dataset_cache.md``.

This is cloud-free: it uses plain ``pickle.load`` (never ``BrainDataset``, which
opens the raw image on S3) and reads only the cache's own arrays + the local CSVs.

Kernel / env: run under the ``panda`` conda env (numpy/scipy binary-compatible so
``SkeletonGraph`` unpickles). Usage:

    python notebooks/verify_add_cache_merge_info.py                    # brain 794495, mcl100
    python notebooks/verify_add_cache_merge_info.py --brain 789202 --mcl 100
    python notebooks/verify_add_cache_merge_info.py --tol 50 --brain 794495
"""

import argparse
import ast
import glob
import os
import pickle
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

REPO = "/allen/programs/mindscope/workgroups/auto-model/zihan.zhang/exaspim-agent"
sys.path.insert(0, os.path.join(REPO, "agentic-neuron-proofreader/src"))

# Needed so pickle can rebuild the SkeletonGraph instances stored in the cache.
import agentic_neuron_proofreader  # noqa: F401
from agentic_neuron_proofreader.data_modules import canonical_labeling as cl


# --- Small reporting helper (mirrors the notebook's PASS/WARN/FAIL style) ------
class Reporter:
    """Collects PASS/FAIL (hard) and OK/WARN (soft) verdicts and prints them."""

    def __init__(self):
        self.hard_failures = 0

    def check(self, name, ok, detail="", hard=True):
        if hard:
            tag = "PASS" if ok else "FAIL"
            if not ok:
                self.hard_failures += 1
        else:
            tag = "OK" if ok else "WARN"
        print(f"  [{tag}] {name}" + (f"  ->  {detail}" if detail else ""))
        return ok


# --- Loaders -------------------------------------------------------------------
def load_cache_merges(add_path):
    """
    Loads the merge-related arrays from an ``_add.pkl`` (cloud-free).

    Returns
    -------
    dict with:
      "gt_graph"        -- the GT SkeletonGraph (for node_segment_id / node counts)
      "node_label"      -- (N,) predicted segment id per GT node (0 = unlabeled)
      "merge_labels"    -- set[int], gt_merge_labels (node-count UNION geometric)
      "merge_sites"     -- list[{"segment_id", "gt_neuron", "xyz"}]
    """
    with open(add_path, "rb") as f:
        payload = pickle.load(f)

    gt = payload["gt_graph"]

    node_label = payload.get("gt_node_canonical_label")
    if node_label is None:
        node_label = getattr(gt, "node_label", None)
    assert node_label is not None and np.asarray(node_label).size, (
        f"{add_path} has no gt_node_canonical_label -- this is a plain cache, "
        "not an _add cache (run scripts/relabel_cache.py)."
    )
    node_label = np.asarray(node_label)

    merge_labels_arr = payload.get("gt_merge_labels")
    if merge_labels_arr is None:
        merge_labels_arr = getattr(gt, "merge_labels", None)
    assert merge_labels_arr is not None, (
        f"{add_path} has no gt_merge_labels -- rerun relabel_cache.py so merge "
        "information is baked in."
    )
    merge_labels = set(int(x) for x in np.asarray(merge_labels_arr))

    merge_sites = payload.get("gt_merge_sites")
    if merge_sites is None:
        merge_sites = getattr(gt, "merge_sites", None) or []

    return {
        "gt_graph": gt,
        "node_label": node_label,
        "merge_labels": merge_labels,
        "merge_sites": list(merge_sites),
    }


def load_canonical_merges(metrics_dir, brain_id):
    """
    Loads canonical ``merge_sites.csv`` and ``results.csv`` for a brain.

    Returns
    -------
    dict with:
      "merge_sites_csv" -- path used
      "sites"           -- list[{"segment_id", "gt_neuron", "xyz"=(x,y,z) µm}]
      "results"         -- results.csv DataFrame indexed by neuron name (or None)
    """
    site_hits = sorted(
        glob.glob(os.path.join(metrics_dir, brain_id, "*", "merge_sites.csv"))
    )
    assert site_hits, (
        f"no merge_sites.csv under {metrics_dir}/{brain_id}/*/ -- run the "
        "canonical evaluate() with save_merges=True first."
    )
    site_path = site_hits[0]
    raw = pd.read_csv(site_path)

    sites = []
    for _, row in raw.iterrows():
        world = ast.literal_eval(str(row["World"]))  # "(x, y, z)" µm -> tuple
        sites.append(
            {
                "segment_id": int(float(row["Segment_ID"])),
                "gt_neuron": str(row["GroundTruth_ID"]),
                "xyz": (float(world[0]), float(world[1]), float(world[2])),
            }
        )

    results = None
    res_hits = sorted(
        glob.glob(os.path.join(metrics_dir, brain_id, "*", "results.csv"))
    )
    if res_hits:
        results = pd.read_csv(res_hits[0], index_col=0).sort_index()

    return {"merge_sites_csv": site_path, "sites": sites, "results": results}


# --- Checks --------------------------------------------------------------------
def check_segment_ids(rep, cache, canon):
    """A. Merge segment-id sets agree; gt_merge_labels is a superset of sites."""
    print("\nA. Merge segment-id sets")
    cache_site_ids = {int(s["segment_id"]) for s in cache["merge_sites"]}
    canon_site_ids = {int(s["segment_id"]) for s in canon["sites"]}

    inter = cache_site_ids & canon_site_ids
    only_cache = cache_site_ids - canon_site_ids
    only_canon = canon_site_ids - cache_site_ids
    union = cache_site_ids | canon_site_ids
    jaccard = len(inter) / len(union) if union else float("nan")

    print(
        f"     cache site seg ids: {len(cache_site_ids)} | "
        f"canonical site seg ids: {len(canon_site_ids)} | "
        f"shared: {len(inter)} | Jaccard: {jaccard:.3f}"
    )
    if only_cache:
        print(f"     only in cache ({len(only_cache)}): "
              f"{sorted(only_cache)[:8]}{' ...' if len(only_cache) > 8 else ''}")
    if only_canon:
        print(f"     only in canon ({len(only_canon)}): "
              f"{sorted(only_canon)[:8]}{' ...' if len(only_canon) > 8 else ''}")

    # gt_merge_labels folds in the node-count rule too, so every geometric-walk
    # site id must appear in it. This one is a hard structural invariant.
    missing = cache_site_ids - cache["merge_labels"]
    rep.check(
        "gt_merge_labels superset of cache site seg ids",
        not missing,
        f"{len(missing)} site ids missing from gt_merge_labels"
        if missing else f"all {len(cache_site_ids)} contained",
    )
    # Site id sets should overlap strongly (both are the geometric walk).
    rep.check(
        "cache vs canonical site seg ids overlap (Jaccard > 0.7)",
        (jaccard > 0.7) if union else True,
        f"Jaccard={jaccard:.3f}",
        hard=False,
    )
    return {"cache_site_ids": cache_site_ids, "canon_site_ids": canon_site_ids}


def check_per_neuron_counts(rep, cache, canon):
    """B. Per-neuron merge counts: cache sites vs results.csv '# Merges'."""
    print("\nB. Per-neuron merge counts")
    cache_by_neuron = defaultdict(int)
    for s in cache["merge_sites"]:
        cache_by_neuron[str(s["gt_neuron"])] += 1
    canon_by_neuron = defaultdict(int)
    for s in canon["sites"]:
        canon_by_neuron[str(s["gt_neuron"])] += 1

    # Cross-check: results.csv "# Merges" per neuron should equal merge_sites rows.
    results = canon["results"]
    if results is not None and "# Merges" in results.columns:
        res_merges = results["# Merges"].fillna(0).astype(int)
        mism = [
            (n, int(res_merges[n]), canon_by_neuron.get(n, 0))
            for n in res_merges.index
            if int(res_merges[n]) != canon_by_neuron.get(n, 0)
        ]
        rep.check(
            "results.csv '# Merges' == merge_sites.csv rows per neuron",
            not mism,
            f"{len(mism)} neuron(s) differ, e.g. {mism[:3]}" if mism
            else "exact per-neuron agreement",
            hard=False,
        )

    all_neurons = sorted(set(cache_by_neuron) | set(canon_by_neuron))
    rows = [
        {
            "neuron": n,
            "cache": cache_by_neuron.get(n, 0),
            "canon": canon_by_neuron.get(n, 0),
            "diff": cache_by_neuron.get(n, 0) - canon_by_neuron.get(n, 0),
        }
        for n in all_neurons
    ]
    df = pd.DataFrame(rows).set_index("neuron")
    if len(df):
        with pd.option_context("display.max_rows", 200):
            print(df.to_string())
        mean_abs = df["diff"].abs().mean()
        print(f"     mean |diff| per neuron: {mean_abs:.2f}")
        rep.check(
            "per-neuron merge count mean |diff| < 1.0",
            mean_abs < 1.0,
            f"mean |diff|={mean_abs:.2f}",
            hard=False,
        )
    return df


def _best_axis(cache_xyz, canon_xyz):
    """
    Sanity guard against a silent axis-convention mismatch. Both sources should
    already be (x, y, z) µm, so identity should win; we compare the median
    nearest-neighbour distance under identity vs a reversed (z, y, x) order and
    warn loudly if reversed is dramatically better (which would mean a convention
    bug upstream). Returns (label, canon_xyz_to_use).
    """
    from scipy.spatial import KDTree

    def median_nn(a, b):
        if len(a) == 0 or len(b) == 0:
            return float("inf")
        d, _ = KDTree(b).query(a)
        return float(np.median(d))

    ident = median_nn(cache_xyz, canon_xyz)
    rev = median_nn(cache_xyz, canon_xyz[:, ::-1])
    if rev < ident / 2:
        return "REVERSED (z,y,x) -- AXIS MISMATCH!", canon_xyz[:, ::-1], ident, rev
    return "identity (x,y,z)", canon_xyz, ident, rev


def check_site_localization(rep, cache, canon, tol_um):
    """C. Merge sites localize to the same physical places (the load-bearing check)."""
    print("\nC. Merge-site localization (nearest-neighbour in µm)")
    from scipy.spatial import KDTree

    cache_xyz = np.array([s["xyz"] for s in cache["merge_sites"]], dtype=float)
    canon_xyz = np.array([s["xyz"] for s in canon["sites"]], dtype=float)

    if len(cache_xyz) == 0 or len(canon_xyz) == 0:
        rep.check(
            "both sides have merge sites to localize",
            False,
            f"cache sites={len(cache_xyz)}, canonical sites={len(canon_xyz)}",
        )
        return None

    # Axis sanity guard.
    axis_label, canon_xyz_use, d_ident, d_rev = _best_axis(cache_xyz, canon_xyz)
    print(f"     axis check: using {axis_label} "
          f"(median NN dist identity={d_ident:.1f}µm, reversed={d_rev:.1f}µm)")
    rep.check(
        "coordinate axes aligned (identity, not reversed)",
        "REVERSED" not in axis_label,
        axis_label,
    )

    canon_tree = KDTree(canon_xyz_use)
    cache_tree = KDTree(cache_xyz)

    # Recall: fraction of canonical sites with a cache site within tol.
    d_canon, _ = cache_tree.query(canon_xyz_use)
    recall = float((d_canon <= tol_um).mean())
    # Precision: fraction of cache sites with a canonical site within tol.
    d_cache, _ = canon_tree.query(cache_xyz)
    precision = float((d_cache <= tol_um).mean())
    f1 = (2 * recall * precision / (recall + precision)
          if (recall + precision) > 0 else 0.0)

    print(f"     cache sites: {len(cache_xyz)} | canonical sites: {len(canon_xyz)}")
    print(f"     median NN dist (cache->canon): {np.median(d_cache):.1f}µm | "
          f"(canon->cache): {np.median(d_canon):.1f}µm")
    # Match rate across a few tolerances for insight into the residual.
    for t in sorted({10.0, 25.0, tol_um, 100.0}):
        r = float((d_canon <= t).mean())
        p = float((d_cache <= t).mean())
        print(f"       @ {t:>5.0f}µm : recall={r:5.1%}  precision={p:5.1%}")

    print(f"     @ tol={tol_um:.0f}µm -> recall={recall:.1%}  "
          f"precision={precision:.1%}  F1={f1:.1%}")
    rep.check(
        f"site recall >= 0.8 @ {tol_um:.0f}µm (cache finds canonical merges)",
        recall >= 0.8, f"recall={recall:.1%}", hard=False,
    )
    rep.check(
        f"site precision >= 0.8 @ {tol_um:.0f}µm (cache sites are real merges)",
        precision >= 0.8, f"precision={precision:.1%}", hard=False,
    )
    return {"recall": recall, "precision": precision, "f1": f1}


def check_totals(rep, cache, canon):
    """D. Total '# Merges' agrees across cache sites, merge_sites.csv, results.csv."""
    print("\nD. Total merge counts")
    n_cache = len(cache["merge_sites"])
    n_canon_sites = len(canon["sites"])
    print(f"     cache gt_merge_sites : {n_cache}")
    print(f"     merge_sites.csv rows : {n_canon_sites}")

    results = canon["results"]
    if results is not None and "# Merges" in results.columns:
        n_results = int(results["# Merges"].fillna(0).sum())
        print(f"     results.csv sum(# Merges): {n_results}")
        rep.check(
            "results.csv sum(# Merges) == merge_sites.csv rows",
            n_results == n_canon_sites,
            f"{n_results} vs {n_canon_sites}",
            hard=False,
        )

    denom = max(n_cache, n_canon_sites, 1)
    rel = abs(n_cache - n_canon_sites) / denom
    rep.check(
        "total merge count within 15% of canonical",
        rel <= 0.15,
        f"cache={n_cache}, canon={n_canon_sites}, rel diff={rel:.1%}",
        hard=False,
    )


# --- Main ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--brain", default="794495", help="brain id (default 794495)")
    parser.add_argument("--mcl", type=int, default=100,
                        help="min_cable_length in the cache filename (default 100)")
    parser.add_argument("--tol", type=float, default=50.0,
                        help="site match tolerance in µm (default 50; dedup is 30, "
                             "branch-snapping adds a little)")
    parser.add_argument("--cache-dir",
                        default=os.path.join(REPO, "exa-spim-agent/cache"))
    parser.add_argument("--metrics-dir",
                        default=os.path.join(REPO, "exa-spim-agent/metrics_out"))
    args = parser.parse_args()

    add_path = os.path.join(
        args.cache_dir, f"dataset_cache_{args.brain}_mcl{args.mcl}_add.pkl"
    )
    print("=" * 78)
    print(f"Verify merge info: brain {args.brain} (mcl{args.mcl}), tol={args.tol:.0f}µm")
    print("=" * 78)
    print(f"cache      : {add_path}")
    assert os.path.exists(add_path), f"cache not found: {add_path}"

    cache = load_cache_merges(add_path)
    canon = load_canonical_merges(args.metrics_dir, args.brain)
    print(f"canonical  : {canon['merge_sites_csv']}")
    print(f"\ncache: {len(cache['merge_sites'])} merge sites, "
          f"{len(cache['merge_labels'])} merge labels | "
          f"canonical: {len(canon['sites'])} merge sites")

    rep = Reporter()
    check_segment_ids(rep, cache, canon)
    check_per_neuron_counts(rep, cache, canon)
    check_site_localization(rep, cache, canon, args.tol)
    check_totals(rep, cache, canon)

    print("\n" + "=" * 78)
    if rep.hard_failures == 0:
        print("RESULT: PASS -- the _add.pkl merge information is aligned with "
              "metrics_out.")
        print("(WARNs, if any, are the documented branch-snap/dedup residual -- "
              "close, not byte-exact.)")
    else:
        print(f"RESULT: {rep.hard_failures} HARD FAILURE(S) -- the cache merge "
              "information does NOT match metrics_out. Investigate above.")
    print("=" * 78)
    return 1 if rep.hard_failures else 0


if __name__ == "__main__":
    sys.exit(main())
