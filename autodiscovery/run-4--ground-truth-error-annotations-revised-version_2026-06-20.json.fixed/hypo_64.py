# CORRECTED TEST for hypothesis id 64 (split edges geodesically closer to branch
# points than correct edges).
#
# FAULTS (verifier MINOR, concrete test faults): large n inflates significance and
# the >1.1M correct edges are NON-INDEPENDENT (edges within a neuron are
# spatially autocorrelated), so the naive Mann-Whitney p=1.32e-197 overstates the
# evidence; part of the mean gap also reflects range truncation (split edges lack
# the long-distance tail). The naive p is therefore not a valid test among
# independent units.
#
# CORRECTION: keep the SAME geodesic distances and direction but (1) report an
# EFFECT SIZE -- Cliff's delta (rank-biserial) with a CLUSTER bootstrap 95% CI
# (resampling whole NEURONS) -- and (2) replace the naive p with a NEURON-CLUSTER
# permutation p-value (permute split/correct label at the neuron level).
#
# ORIGINAL RECORDED NUMBERS (for the driver to compare):
#   6805 split / 1109034 correct, mean dist 516.30 um (split) vs 710.59 um (correct),
#   MW U=2979025451.5, p=1.3239e-197.

import sys
import gc
import os

import agentic_neuron_proofreader

import pickle
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra
import scipy.stats as stats
import matplotlib.pyplot as plt

print("Searching for dataset files...", flush=True)
# Load the provided dataset directly from $RERUN_PKL
print("Loading dataset from:", os.environ["RERUN_PKL"], flush=True)
dataset_paths = [os.environ["RERUN_PKL"]]

print(f"Found {len(dataset_paths)} dataset files: {dataset_paths}", flush=True)

split_distances = []
correct_distances = []
split_neuron = []
correct_neuron = []

EDGE_CORRECT = 0
EDGE_SPLIT = 1

for path in dataset_paths:
    print(f"Loading {os.path.basename(path)}...", flush=True)
    with open(path, "rb") as f:
        payload = pickle.load(f)
    print(f"Finished reading {os.path.basename(path)}. Extracting graphs...", flush=True)

    gt = payload["gt_graph"]
    edge_error = np.asarray(payload["gt_edge_error"])

    payload.pop("fragments_graph", None)
    del payload
    gc.collect()

    edges = list(gt.edges())
    if len(edges) == 0:
        del gt, edge_error
        gc.collect()
        continue

    u_list = [e[0] for e in edges]
    v_list = [e[1] for e in edges]
    u_arr = np.array(u_list)
    v_arr = np.array(v_list)

    # Neuron id per edge (cluster / independent unit), captured before deleting gt.
    neuron_arr = np.array([gt.node_segment_id(n) for n in u_arr])

    weights = np.linalg.norm(gt.node_xyz[u_arr] - gt.node_xyz[v_arr], axis=1)
    N = gt.node_xyz.shape[0]

    degrees = dict(gt.degree())
    branch_points = [n for n, d in degrees.items() if d >= 3]
    print(f"Graph has {len(edges)} edges, {N} max nodes, {len(branch_points)} branch points.", flush=True)

    if len(branch_points) == 0:
        del gt, edge_error, u_arr, v_arr, weights
        gc.collect()
        continue

    bp_arr = np.array(branch_points)
    dummy_node = N
    dummy_u = np.full(len(bp_arr), dummy_node)
    dummy_v = bp_arr
    dummy_w = np.zeros(len(bp_arr))

    u_all = np.concatenate([u_arr, dummy_u])
    v_all = np.concatenate([v_arr, dummy_v])
    w_all = np.concatenate([weights, dummy_w])

    graph_sparse = sp.coo_matrix((w_all, (u_all, v_all)), shape=(N + 1, N + 1))
    dists = dijkstra(graph_sparse, directed=False, indices=dummy_node)
    node_dists = dists[:N]

    dist_u = node_dists[u_arr]
    dist_v = node_dists[v_arr]
    edge_dists = np.minimum(dist_u, dist_v)

    split_mask = (edge_error == EDGE_SPLIT)
    correct_mask = (edge_error == EDGE_CORRECT)
    valid_mask = np.isfinite(edge_dists)

    split_distances.append(edge_dists[split_mask & valid_mask])
    correct_distances.append(edge_dists[correct_mask & valid_mask])
    split_neuron.append(neuron_arr[split_mask & valid_mask])
    correct_neuron.append(neuron_arr[correct_mask & valid_mask])

    del gt, edge_error, u_arr, v_arr, weights, u_all, v_all, w_all
    del graph_sparse, dists, node_dists, dist_u, dist_v, edge_dists
    gc.collect()

split_distances = np.concatenate(split_distances) if split_distances else np.array([])
correct_distances = np.concatenate(correct_distances) if correct_distances else np.array([])
split_neuron = np.concatenate(split_neuron) if len(split_neuron) else np.array([])
correct_neuron = np.concatenate(correct_neuron) if len(correct_neuron) else np.array([])

print(f"Total split edges evaluated: {len(split_distances):,}")
print(f"Total correct edges evaluated: {len(correct_distances):,}")

if len(split_distances) > 0 and len(correct_distances) > 0:
    print(f"Mean distance to branch point (split):   {np.mean(split_distances):.2f} um")
    print(f"Mean distance to branch point (correct): {np.mean(correct_distances):.2f} um")

    # Naive test echoed for comparison only.
    stat, pval = stats.mannwhitneyu(split_distances, correct_distances)
    print(f"Naive Mann-Whitney U={stat}, p={pval:.4e} (for comparison only)")

    # === EFFECT SIZE: Cliff's delta with CLUSTER (neuron) bootstrap 95% CI ===
    def cliffs_delta_sub(a, b, rng, n_sub=4000):
        a_s = a if len(a) <= n_sub else rng.choice(a, n_sub, replace=False)
        b_s = b if len(b) <= n_sub else rng.choice(b, n_sub, replace=False)
        gt_ = sum((x > b_s).sum() for x in a_s)
        lt_ = sum((x < b_s).sum() for x in a_s)
        return (gt_ - lt_) / (len(a_s) * len(b_s))

    rng = np.random.default_rng(42)
    delta = cliffs_delta_sub(split_distances, correct_distances, rng)
    s_df = pd.DataFrame({"neuron": split_neuron, "d": split_distances})
    c_df = pd.DataFrame({"neuron": correct_neuron, "d": correct_distances})
    s_by = {k: g["d"].values for k, g in s_df.groupby("neuron")}
    c_by = {k: g["d"].values for k, g in c_df.groupby("neuron")}
    s_keys = np.array(list(s_by.keys()), dtype=object)
    c_keys = np.array(list(c_by.keys()), dtype=object)
    boots = []
    for _ in range(300):
        bs = np.concatenate([s_by[k] for k in rng.choice(s_keys, len(s_keys), replace=True)])
        bc = np.concatenate([c_by[k] for k in rng.choice(c_keys, len(c_keys), replace=True)])
        boots.append(cliffs_delta_sub(bs, bc, rng, n_sub=2000))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    print("\n=== Effect size (cluster-aware) ===")
    print(f"Cliff's delta (split vs correct dist) = {delta:.4f}  (negative => split closer)")
    print(f"Cluster bootstrap 95% CI = [{lo:.4f}, {hi:.4f}]")

    # === NEURON-CLUSTER permutation p-value ===
    s_means = np.array([np.median(v) for v in s_by.values()])
    c_means = np.array([np.median(v) for v in c_by.values()])
    obs = np.median(s_means) - np.median(c_means)
    pooled = np.concatenate([s_means, c_means])
    n_s = len(s_means)
    n_perm = 5000
    perm = np.empty(n_perm)
    for i in range(n_perm):
        idx = rng.permutation(len(pooled))
        perm[i] = np.median(pooled[idx[:n_s]]) - np.median(pooled[idx[n_s:]])
    perm_p = (np.sum(perm <= obs) + 1) / (n_perm + 1)
    print("\n=== Neuron-cluster permutation test ===")
    print(f"Split-carrying neurons={n_s}, correct neurons={len(c_means)}")
    print(f"Observed neuron-level median distance gap (split-correct) = {obs:.2f} um")
    print(f"Cluster-permutation one-sided p (split closer) = {perm_p:.4g}")
    print("(Compare to ORIGINAL: MW p=1.3239e-197 treating all edges as independent.)")

    plt.figure(figsize=(10, 6))
    plt.violinplot([correct_distances, split_distances], showmeans=True, showmedians=True)
    plt.xticks([1, 2], ['Correct Edges', 'Split Edges'])
    plt.ylabel("Geodesic Distance to Nearest Branch Point (um)")
    plt.title("Proximity to Topological Branch Points: Split vs Correct Edges")
    plt.grid(axis='y', alpha=0.3)
    plt.show()
else:
    print("Not enough data to compute statistics.")
