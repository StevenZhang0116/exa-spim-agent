#!/usr/bin/env python
"""
Verify canonical GT labeling against the ACTUAL segmentation volume.

Unlike a proxy check, this reads the real dense segmentation from the cloud
(exactly as load_skeletons / load_skeletons_from_cache do), labels every GT node
with ``BrainDataset.label_gt_from_segmentation``, and checks the result two ways:

  1. Patch cross-check (cheap, fast): read one segmentation patch the way the
     notebook does (``segmentation.read(center_voxel, patch_shape)``) and confirm
     the per-node labels from the full labeling pass match the labels indexed
     straight out of that patch. This pins down the voxel/axis convention without
     scanning the whole brain.

  2. Whole-brain canonical check: run the full labeling pass, derive the
     split / merge / edge-error summary, and compare to the canonical
     ``metrics_out/<brain_id>/<seg_id>/results.csv``. Because the cache dropped
     fragments shorter than ``min_cable_length`` and the canonical pipeline reads
     its own (unfiltered) skeletons, these will not match to the digit -- the
     comparison is a sanity band (same order of magnitude, omits an upper bound),
     not an equality assert.

Requires GCS credentials (configs/allen-nd-goog-f5d46dbfa2cd.json) and read access to
gs://allen-nd-goog/... . Run in the `panda` env.

Usage
-----
    conda run -n panda python test_canonical_labeling.py                 # 794495, mcl100
    conda run -n panda python test_canonical_labeling.py --brain 789202
    conda run -n panda python test_canonical_labeling.py --brain 789202 --mcl 10
    conda run -n panda python test_canonical_labeling.py --patch-only    # skip full scan
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

# --- Paths / credentials (mirror the notebooks) ------------------------------
REPO = "/allen/programs/mindscope/workgroups/auto-model/zihan.zhang/exaspim-agent"
sys.path.insert(0, os.path.join(REPO, "agentic-neuron-proofreader/src"))
_SCRIPTS = os.path.join(REPO, "exa-spim-agent/scripts")
_CONFIG = os.path.join(REPO, "exa-spim-agent/configs")
sys.path.insert(0, _SCRIPTS)

os.environ.setdefault(
    "GOOGLE_APPLICATION_CREDENTIALS", os.path.join(_CONFIG, "allen-nd-goog-f5d46dbfa2cd.json")
)
os.environ.setdefault("AWS_EC2_METADATA_DISABLED", "true")

from dataset_config import get_segmentation_path  # noqa: E402

from agentic_neuron_proofreader.data_modules import canonical_labeling as cl  # noqa: E402
from agentic_neuron_proofreader.data_modules.datasets import BrainDataset  # noqa: E402
from agentic_neuron_proofreader.utils import img_util  # noqa: E402

_CONFIG_RTF = os.path.join(_CONFIG, "segmentation_datasets.rtf")


def segmentation_path_for(brain_id):
    return get_segmentation_path(brain_id, rtf_path=_CONFIG_RTF)


def patch_cross_check(dataset, segmentation, patch_shape=(256, 256, 256), n_patches=4):
    """
    Read a few segmentation patches the notebook way and confirm the full-pass
    per-node labels match labels indexed straight from the patch. Returns
    (matched, total) over GT nodes that land inside the sampled patches.
    """
    gt = dataset.gt_graph
    node_label = gt.node_label  # already computed by the full labeling pass
    rng_nodes = list(gt.nodes)
    step = max(1, len(rng_nodes) // (n_patches + 1))
    centers = [gt.node_voxel(rng_nodes[(k + 1) * step]) for k in range(n_patches)]

    matched = total = 0
    for center_voxel in centers:
        seg_patch = np.asarray(segmentation.read(center_voxel, patch_shape))
        seg_patch = seg_patch.reshape(patch_shape)  # drop leading singleton axes
        offset = tuple(c - s // 2 for c, s in zip(center_voxel, patch_shape))
        # Use nodes_in_patch only to find WHICH nodes fall in the patch; index
        # each by gt.node_voxel(i) -- the SAME (truncated int) voxelization the
        # labeling pass uses. (nodes_in_patch returns rounded float coords, which
        # can land a boundary node one voxel over and spuriously disagree.)
        _, ids = gt.nodes_in_patch(offset, patch_shape, return_ids=True)
        for nid in ids:
            z, y, x = gt.node_voxel(int(nid))
            a, b, c = z - offset[0], y - offset[1], x - offset[2]
            if 0 <= a < patch_shape[0] and 0 <= b < patch_shape[1] and 0 <= c < patch_shape[2]:
                patch_lab = int(seg_patch[a, b, c])
                full_lab = int(node_label[int(nid)])
                # The full pass applied fix_label_misalignments, which can turn a
                # patch '0' into a neighbor's id; count those as matches too.
                if patch_lab == full_lab or (patch_lab == 0 and full_lab != 0):
                    matched += 1
                total += 1
    return matched, total


def load_canonical(results_dir, brain_id):
    hits = glob.glob(os.path.join(results_dir, str(brain_id), "*", "results.csv"))
    return pd.read_csv(sorted(hits)[0], index_col=0) if hits else None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--brain", default="794495")
    ap.add_argument("--cache-dir", default=os.path.join(REPO, "exa-spim-agent/cache"))
    ap.add_argument("--results-dir", default=os.path.join(REPO, "exa-spim-agent/metrics_out"))
    ap.add_argument("--mcl", type=int, default=100,
                    help="min_cable_length of the cache to load (selects "
                         "dataset_cache_<brain>_mcl<MCL>.pkl). Default 100. A smaller "
                         "value (e.g. 10) filters fewer short fragments, so the "
                         "cache-only omit rate runs closer to the canonical pipeline.")
    ap.add_argument("--patch-only", action="store_true",
                    help="Only run the fast patch cross-check (still labels all nodes).")
    args = ap.parse_args()

    cache_path = os.path.join(args.cache_dir, f"dataset_cache_{args.brain}_mcl{args.mcl}.pkl")
    if not os.path.exists(cache_path):
        ap.error(f"cache not found: {cache_path}\n"
                 f"  (check --brain / --mcl; build it via load_skeletons.py if missing)")
    seg_path = segmentation_path_for(args.brain)
    print(f"brain {args.brain}")
    print(f"  cache:        {cache_path}")
    print(f"  segmentation: {seg_path}\n")

    dataset = BrainDataset.load_from_cache(cache_path)
    segmentation = img_util.TensorStoreImage(seg_path)

    # --- full labeling pass (the code under test) ----------------------------
    print("labeling GT nodes from the segmentation ...")
    dataset.label_gt_from_segmentation(seg_path, verbose=True)
    node_label = dataset.gt_graph.node_label
    n_labeled = int((node_label != 0).sum())
    print(f"  labeled {n_labeled:,} / {node_label.size:,} GT nodes "
          f"({100*n_labeled/node_label.size:.1f}%)\n")

    # --- (1) patch cross-check ----------------------------------------------
    print("patch cross-check (voxel/axis convention) ...")
    matched, total = patch_cross_check(dataset, segmentation)
    pct = 100 * matched / total if total else float("nan")
    print(f"  {matched}/{total} in-patch GT nodes agree with direct patch read "
          f"({pct:.1f}%)")
    if total and pct < 95:
        print("  *** LOW AGREEMENT -- likely a voxel/axis-order mismatch in "
              "_label_batch. Adjust before trusting the labels. ***")
    print()

    if args.patch_only:
        return

    # --- (2) whole-brain summary vs canonical results.csv --------------------
    s = cl.summarize(dataset.gt_graph, node_label, dataset.gt_graph.edge_error)
    print("cache-only canonical summary:")
    print(f"  total splits : {s['total_splits']}")
    print(f"  total merges : {s['total_merges']}")
    print(f"  % split edges: {s['pct_split_edges']:.3f}")
    print(f"  % omit edges : {s['pct_omit_edges']:.3f}")
    print(f"  % merged edges:{s['pct_merged_edges']:.3f}")
    print(f"  neurons      : {s['num_neurons']}")

    canon = load_canonical(args.results_dir, args.brain)
    if canon is None:
        print(f"\n  [warn] no results.csv under {args.results_dir}/{args.brain}; "
              "skipping canonical comparison.")
        return
    print("\ncanonical results.csv (sum / mean over neurons):")
    print(f"  total splits : {canon['# Splits'].sum():.0f}")
    print(f"  total merges : {canon['# Merges'].sum():.0f}")
    print(f"  % split edges: {canon['% Split Edges'].mean():.3f}  (per-neuron mean)")
    print(f"  % omit edges : {canon['% Omit Edges'].mean():.3f}  (per-neuron mean)")
    print(f"  % merged edges:{canon['% Merged Edges'].mean():.3f}  (per-neuron mean)")
    print(f"  neurons      : {len(canon)}")
    print("\nNOTE: cache fragments are min_cable_length-filtered while the canonical")
    print("pipeline reads unfiltered skeletons, so expect the cache-only omit rate to")
    print("run higher and counts to differ -- look for same-order-of-magnitude, not")
    print("equality. The patch cross-check above is the strict correctness gate.")


if __name__ == "__main__":
    main()
