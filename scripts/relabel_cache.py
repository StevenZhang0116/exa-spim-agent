#!/usr/bin/env python
"""
Bake canonical GT error labels into existing dataset caches.

For each ``dataset_cache_<brain_id>_mcl<N>.pkl`` this:

  1. loads the cache (reusing the graphs already in it -- no SWC rebuild),
  2. resolves the brain's dense segmentation path from ``brain_id``
     (via ``dataset_config.get_segmentation_id`` + the standard GCS template),
  3. reads the segmentation at every GT node's voxel, stores the canonical
     per-node label and per-edge error class on ``gt_graph``,
  4. writes a NEW cache alongside the original with the two added keys
     (``gt_node_canonical_label``, ``gt_edge_error``). The output is named
     ``dataset_cache_<brain>_mcl<N>_add.pkl`` -- the original ``.pkl`` is left
     untouched, so this step is safe to re-run and easy to roll back.

This is the ONE cloud-dependent step (needs GOOGLE_APPLICATION_CREDENTIALS and
read access to gs://allen-nd-goog/...). Run it once per brain; every later load
is cache-only.

Verification: pass ``--results-dir metrics_out`` to compare the freshly computed
cache-only summary against the canonical ``results.csv`` for that brain. The
cache-only numbers will differ from canonical wherever fragment filtering or the
voxel/axis convention diverge -- the comparison is the gate for getting the
labeling right before trusting the stored numbers.

Usage
-----
    python relabel_cache.py --cache-dir ../cache                 # all caches
    python relabel_cache.py --cache-dir ../cache --brain 794495  # one brain
    python relabel_cache.py --cache-dir ../cache --dry-run       # resolve paths only
    python relabel_cache.py --cache-dir ../cache --results-dir ../metrics_out
"""

import argparse
import glob
import os
import re
import sys

import pandas as pd

# --- Config / credentials (mirror proofreader_evolve/harness/scoring.py) ------
_HERE = os.path.dirname(os.path.abspath(__file__))
_CONFIG_DIR = os.path.abspath(os.path.join(_HERE, "..", "configs"))
os.environ.setdefault(
    "GOOGLE_APPLICATION_CREDENTIALS",
    os.path.join(_CONFIG_DIR, "zihan_gcs_token.json"),
)
os.environ.setdefault("AWS_EC2_METADATA_DISABLED", "true")

sys.path.insert(0, _HERE)  # for dataset_config
from dataset_config import get_segmentation_id  # noqa: E402

_CONFIG_RTF = os.path.join(_CONFIG_DIR, "segmentation_datasets.rtf")

from agentic_neuron_proofreader.data_modules import canonical_labeling  # noqa: E402
from agentic_neuron_proofreader.data_modules.datasets import BrainDataset  # noqa: E402


def brain_id_from_path(path):
    m = re.search(r"dataset_cache_(\d+)_mcl(\d+)\.pkl$", os.path.basename(path))
    return m.group(1) if m else None


def segmentation_path_for(brain_id):
    """brain_id -> dense segmentation volume path (standard GCS template)."""
    segmentation_id = get_segmentation_id(brain_id, rtf_path=_CONFIG_RTF)
    return (
        f"gs://allen-nd-goog/from_google/{brain_id}"
        f"/whole_brain/{segmentation_id}/"
    )


def load_canonical_results(results_dir, brain_id):
    """Best-effort load of metrics_out/<brain_id>/<seg_id>/results.csv."""
    base = os.path.join(results_dir, str(brain_id))
    hits = glob.glob(os.path.join(base, "*", "results.csv"))
    if not hits:
        return None
    return pd.read_csv(sorted(hits)[0], index_col=0)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir", required=True,
                    help="Directory of dataset_cache_*.pkl files.")
    ap.add_argument("--brain", default=None,
                    help="Relabel only this brain_id (default: all in cache-dir).")
    ap.add_argument("--results-dir", default=None,
                    help="metrics_out/ root; if given, verify against results.csv.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Resolve segmentation paths and exit (no cloud read).")
    ap.add_argument("--quiet", action="store_true",
                    help="Suppress the per-batch progress bar.")
    args = ap.parse_args()

    pattern = os.path.join(args.cache_dir, "dataset_cache_*.pkl")
    paths = sorted(glob.glob(pattern))
    if args.brain:
        paths = [p for p in paths if brain_id_from_path(p) == str(args.brain)]
    if not paths:
        raise SystemExit(f"No caches matched {pattern}"
                         + (f" for brain {args.brain}" if args.brain else ""))

    for path in paths:
        brain_id = brain_id_from_path(path)
        if brain_id is None:
            print(f"[skip] cannot parse brain_id from {os.path.basename(path)}")
            continue
        seg_path = segmentation_path_for(brain_id)
        print(f"\n=== brain {brain_id} ===")
        print(f"  cache:        {path}")
        print(f"  segmentation: {seg_path}")
        if args.dry_run:
            continue

        ds = BrainDataset.load_from_cache(path)
        ds.label_gt_from_segmentation(seg_path, verbose=not args.quiet)

        summary = canonical_labeling.summarize(
            ds.gt_graph, ds.gt_graph.node_label, ds.gt_graph.edge_error
        )
        print(f"  -> splits={summary['total_splits']} "
              f"merges={summary['total_merges']} "
              f"%split={summary['pct_split_edges']:.2f} "
              f"%omit={summary['pct_omit_edges']:.2f} "
              f"%merged={summary['pct_merged_edges']:.2f}")

        if args.results_dir:
            canon = load_canonical_results(args.results_dir, brain_id)
            if canon is not None:
                tot_splits = canon["# Splits"].sum()
                tot_merges = canon["# Merges"].sum()
                print(f"  canonical results.csv: splits={tot_splits:.0f} "
                      f"merges={tot_merges:.0f}  (per-neuron rows: {len(canon)})")
            else:
                print(f"  [warn] no results.csv under {args.results_dir}/{brain_id}")

        out_path = path[:-4] + "_add.pkl" if path.endswith(".pkl") else path + "_add"
        ds.save(out_path)
        print(f"  saved canonical labels into {os.path.basename(out_path)} "
              f"(original left untouched)")


if __name__ == "__main__":
    main()
