# CORRECTED TEST for hypothesis id 37 (split errors cluster into localized error
# zones: a split edge has more nearby splits within 30 um than a correct edge).
#
# FAULTS (verifier MINOR, concrete test faults): (1) METRIC ASYMMETRY / built-in
# double-counting -- the neighbor count is computed against the split-edge
# KD-tree, so split edges count OTHER splits in a set they BELONG to, an
# asymmetry that inflates the split-group counts relative to correct edges (which
# are external to the tree). (2) split edges are NOT mutually independent
# observations (edges within a neuron are spatially correlated), so the naive
# Mann-Whitney p (~0) is inflated.
#
# CORRECTION: keep the SAME 30-um split-neighbor metric and direction but
# (1) report an EFFECT SIZE -- Cliff's delta (rank-biserial) with a CLUSTER
# bootstrap 95% CI (resampling whole NEURONS, captured before gt is freed) -- and
# (2) replace the naive p with a NEURON-CLUSTER permutation p-value (permute the
# split/correct label at the neuron level, the independent unit). The leave-one-out
# self-exclusion (-1) for split edges already makes the count symmetric per query;
# we additionally make the test cluster-aware.
#
# ORIGINAL RECORDED NUMBERS (for the driver to compare):
#   6805 split edges, mean split-neighbors 0.97 (split) vs 0.10 (correct),
#   MW U=34661344.5, p~0.

import sys
from unittest.mock import MagicMock
import gc
import os
import glob
import pickle
import numpy as np
import subprocess

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", package])

for pkg in ["tqdm", "networkx", "scipy"]:
    try:
        __import__(pkg)
    except ImportError:
        install(pkg)

class MockException(Exception):
    pass

m_google_auth_exc = MagicMock()
m_google_auth_exc.RefreshError = MockException
m_google_auth_exc.TransportError = MockException

mocked_modules = [
    'tensorstore', 'psutil', 'pandas', 'matplotlib', 'matplotlib.pyplot',
    'matplotlib.colors', 'zarr', 'dask', 's3fs', 'boto3', 'botocore',
    'botocore.client', 'google', 'google.auth', 'google.cloud'
]

for mod in mocked_modules:
    sys.modules[mod] = MagicMock()

sys.modules['google.auth.exceptions'] = m_google_auth_exc

try:
    import agentic_neuron_proofreader
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "--no-deps", "https://github.com/AllenInstitute/agentic-neuron-proofreader/archive/main.zip"])
    import agentic_neuron_proofreader

from scipy.spatial import cKDTree
from scipy.stats import mannwhitneyu

dataset_paths = glob.glob("../*_add.pkl")
if not dataset_paths:
    dataset_paths = glob.glob("./*_add.pkl")
if not dataset_paths:
    dataset_paths = glob.glob("../**/*_add.pkl", recursive=True)
if not dataset_paths:
    dataset_paths = glob.glob("cache/*_add.pkl")

if not dataset_paths:
    print("Dataset not found.")
else:
    dataset_path = dataset_paths[0]

    with open(dataset_path, "rb") as f:
        payload = pickle.load(f)

    gt = payload["gt_graph"]
    edge_error = np.asarray(payload["gt_edge_error"])

    gt_edges_list = list(gt.edges)
    edges_arr = np.array(gt_edges_list)

    node_xyz = gt.node_xyz
    # Capture neuron id per edge (cluster / independent unit) BEFORE freeing gt.
    edge_neuron = np.array([gt.node_segment_id(u) for u in edges_arr[:, 0]]) if len(edges_arr) else np.array([])

    payload.pop("fragments_graph", None)
    del payload
    del gt
    gc.collect()

    if len(edges_arr) > 0:
        u_nodes = edges_arr[:, 0]
        v_nodes = edges_arr[:, 1]
        midpoints = (node_xyz[u_nodes] + node_xyz[v_nodes]) / 2.0

        split_mask_all = (edge_error == 1)
        correct_mask_all = (edge_error == 0)
        split_midpoints = midpoints[split_mask_all]
        correct_midpoints = midpoints[correct_mask_all]
        split_neuron = edge_neuron[split_mask_all]
        correct_neuron_full = edge_neuron[correct_mask_all]

        np.random.seed(42)
        if len(correct_midpoints) > len(split_midpoints) and len(split_midpoints) > 0:
            idx = np.random.choice(len(correct_midpoints), len(split_midpoints), replace=False)
            sampled_correct_midpoints = correct_midpoints[idx]
            sampled_correct_neuron = correct_neuron_full[idx]
        else:
            sampled_correct_midpoints = correct_midpoints
            sampled_correct_neuron = correct_neuron_full

        if len(split_midpoints) > 0:
            tree = cKDTree(split_midpoints)
            radius = 30.0

            split_neighbor_counts = tree.query_ball_point(split_midpoints, r=radius, return_length=True)
            split_counts = np.maximum(0, np.array(split_neighbor_counts) - 1)  # leave-one-out (self excluded)

            if len(sampled_correct_midpoints) > 0:
                correct_counts = np.array(tree.query_ball_point(sampled_correct_midpoints, r=radius, return_length=True))
            else:
                correct_counts = np.array([])

            if len(split_counts) > 0 and len(correct_counts) > 0:
                # Naive test echoed for comparison only.
                u_stat, p_val = mannwhitneyu(split_counts, correct_counts, alternative='greater')

                print("=== Spatial Clustering of Split Errors ===")
                print(f"Number of split edges: {len(split_counts)}")
                print(f"Number of correct edges (sampled): {len(correct_counts)}")
                print(f"Mean split-neighbors within {radius} um (Split):   {np.mean(split_counts):.2f}")
                print(f"Mean split-neighbors within {radius} um (Correct): {np.mean(correct_counts):.2f}")
                print(f"Naive Mann-Whitney U={u_stat}, p={p_val:.2e} (for comparison only)")

                # === EFFECT SIZE: Cliff's delta with CLUSTER (neuron) bootstrap 95% CI ===
                def cliffs_delta_sub(a, b, rng, n_sub=4000):
                    a_s = a if len(a) <= n_sub else rng.choice(a, n_sub, replace=False)
                    b_s = b if len(b) <= n_sub else rng.choice(b, n_sub, replace=False)
                    gt_ = sum((x > b_s).sum() for x in a_s)
                    lt_ = sum((x < b_s).sum() for x in a_s)
                    return (gt_ - lt_) / (len(a_s) * len(b_s))

                rng = np.random.default_rng(42)
                delta = cliffs_delta_sub(split_counts.astype(float), correct_counts.astype(float), rng)

                # Group counts by neuron (plain-numpy grouping; pandas is mocked).
                def group_by(neuron, vals):
                    d = {}
                    for nid, val in zip(neuron, vals):
                        d.setdefault(nid, []).append(val)
                    return {k: np.array(v, float) for k, v in d.items()}

                s_by = group_by(split_neuron, split_counts)
                c_by = group_by(sampled_correct_neuron, correct_counts)
                s_keys = np.array(list(s_by.keys()), dtype=object)
                c_keys = np.array(list(c_by.keys()), dtype=object)
                boots = []
                for _ in range(300):
                    bs = np.concatenate([s_by[k] for k in rng.choice(s_keys, len(s_keys), replace=True)])
                    bc = np.concatenate([c_by[k] for k in rng.choice(c_keys, len(c_keys), replace=True)])
                    boots.append(cliffs_delta_sub(bs, bc, rng, n_sub=2000))
                lo, hi = np.percentile(boots, [2.5, 97.5])
                print("\n=== Effect size (cluster-aware) ===")
                print(f"Cliff's delta (split vs correct neighbor count) = {delta:.4f}")
                print(f"Cluster bootstrap 95% CI = [{lo:.4f}, {hi:.4f}]")

                # === NEURON-CLUSTER permutation p-value ===
                s_means = np.array([np.mean(v) for v in s_by.values()])
                c_means = np.array([np.mean(v) for v in c_by.values()])
                obs = np.mean(s_means) - np.mean(c_means)
                pooled = np.concatenate([s_means, c_means])
                n_s = len(s_means)
                n_perm = 5000
                perm = np.empty(n_perm)
                for i in range(n_perm):
                    pidx = rng.permutation(len(pooled))
                    perm[i] = np.mean(pooled[pidx[:n_s]]) - np.mean(pooled[pidx[n_s:]])
                perm_p = (np.sum(perm >= obs) + 1) / (n_perm + 1)
                print("\n=== Neuron-cluster permutation test ===")
                print(f"Split neurons={n_s}, correct neurons={len(c_means)}")
                print(f"Observed neuron-level mean split-neighbor gap = {obs:.4f}")
                print(f"Cluster-permutation one-sided p (split > correct) = {perm_p:.4g}")
                print("(Compare to ORIGINAL: mean 0.97 vs 0.10, MW p~0; metric is split-set asymmetric.)")
            else:
                print("Not enough edges to compute statistics.")
        else:
            print("No split edges found.")
    else:
        print("No edges found in the graph.")
