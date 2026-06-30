import os
import sys

import agentic_neuron_proofreader

import pickle
import numpy as np
from scipy import stats
from collections import defaultdict

def get_dataset_files():
    # Load the provided dataset directly from $RERUN_PKL
    print("Loading dataset from:", os.environ["RERUN_PKL"])
    return [os.environ["RERUN_PKL"]]

def main():
    caches = get_dataset_files()
    if not caches:
        print("No dataset files found.")
        return

    merging_lengths = []
    non_merging_lengths = []

    # Process all found datasets
    for cache_path in caches:
        with open(cache_path, "rb") as f:
            payload = pickle.load(f)

        gt = payload["gt_graph"]
        node_label = np.asarray(payload["gt_node_canonical_label"])
        merge_labels = set(int(x) for x in payload["gt_merge_labels"])

        node_xyz = gt.node_xyz

        # 1. Group GT edges by their predicted segmentation ID
        segment_lengths = defaultdict(float)

        for u, v in gt.edges:
            lab_u = node_label[u]
            lab_v = node_label[v]
            
            # Process edges where both endpoints share the same valid canonical label (> 0)
            if lab_u == lab_v and lab_u > 0:
                # 2. Calculate physical Euclidean length of the edge in µm
                dist = np.linalg.norm(node_xyz[u] - node_xyz[v])
                segment_lengths[lab_u] += dist

        # 3. Partition into merging and non-merging segment arrays
        for seg_id, length in segment_lengths.items():
            if length > 0:
                if seg_id in merge_labels:
                    merging_lengths.append(length)
                else:
                    non_merging_lengths.append(length)

    merging_lengths = np.array(merging_lengths)
    non_merging_lengths = np.array(non_merging_lengths)

    if len(merging_lengths) == 0 or len(non_merging_lengths) == 0:
        print("Not enough segments to perform t-test.")
        return

    print(f"Number of merging segments: {len(merging_lengths)}")
    print(f"Number of non-merging segments: {len(non_merging_lengths)}")

    print(f"\nMerging segments - Mean length: {np.mean(merging_lengths):.2f} µm, Median length: {np.median(merging_lengths):.2f} µm")
    print(f"Non-merging segments - Mean length: {np.mean(non_merging_lengths):.2f} µm, Median length: {np.median(non_merging_lengths):.2f} µm")

    # 4. Apply a log10 transformation to the lengths
    log_merging = np.log10(merging_lengths)
    log_non_merging = np.log10(non_merging_lengths)

    # 5. Compare the log-length distributions using Welch's t-test
    t_stat, p_val = stats.ttest_ind(log_merging, log_non_merging, equal_var=False)

    print(f"\nWelch's t-test on log-transformed lengths:")
    print(f"t-statistic: {t_stat:.4f}")
    print(f"p-value: {p_val:.4e}")

if __name__ == "__main__":
    main()
