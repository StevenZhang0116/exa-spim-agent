# CORRECTED TEST for hypothesis id 61 (split edges closer to merge sites than
# correct edges).
#
# FAULTS (verifier MINOR, concrete test faults): the effect is practically trivial
# -- ~162 um of a ~1900 um median (~8%) -- with the extreme p-values produced
# SOLELY by the >1.1M correct edges; edges are NON-INDEPENDENT and all distances
# reference the same small merge-site set, so the effective n is much smaller than
# reported and the naive Mann-Whitney / Welch p is inflated.
#
# CORRECTION: keep the SAME distances and direction but (1) report an EFFECT SIZE
# -- Cliff's delta (rank-biserial) with a CLUSTER bootstrap 95% CI (resampling
# whole NEURONS) -- and (2) replace the naive p with a NEURON-CLUSTER permutation
# p-value (permute the split/correct label at the neuron level, the independent
# unit).
#
# ORIGINAL RECORDED NUMBERS (for the driver to compare):
#   6805 split / 1109034 correct, split median 1794.64 um vs correct 1956.93 um,
#   MW U=3432660112.0 p=3.50e-38, Welch t=-10.7131 p=7.14e-27.

import subprocess
import sys
import os
import glob

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", package])

for pkg in ["psutil", "pandas", "tensorstore"]:
    try:
        __import__(pkg)
    except ImportError:
        install(pkg)

try:
    import agentic_neuron_proofreader
except ImportError:
    install("https://github.com/AllenInstitute/agentic-neuron-proofreader/archive/refs/heads/main.zip")
    import agentic_neuron_proofreader

import pickle
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import mannwhitneyu, ttest_ind

file_paths = glob.glob("./*_add.pkl")
if not file_paths:
    print("Dataset file not found.")
    sys.exit(1)

EDGE_CORRECT, EDGE_SPLIT = 0, 1

split_distances = []
correct_distances = []
split_neuron = []
correct_neuron = []

for file_path in file_paths:
    with open(file_path, "rb") as f:
        payload = pickle.load(f)

    gt = payload["gt_graph"]
    edge_error = np.asarray(payload["gt_edge_error"])
    merge_sites = payload.get("gt_merge_sites", [])

    if not merge_sites:
        continue
    merge_xyz = np.array([site["xyz"] for site in merge_sites])
    if len(merge_xyz) == 0:
        continue
    merge_tree = cKDTree(merge_xyz)

    edges = list(gt.edges)
    if len(edges) == 0:
        continue

    u = np.array([e[0] for e in edges])
    v = np.array([e[1] for e in edges])

    try:
        u_xyz = gt.node_xyz[u]
        v_xyz = gt.node_xyz[v]
    except (AttributeError, TypeError, IndexError):
        u_xyz = np.array([gt.nodes[n].get('node_xyz', gt.nodes[n].get('xyz')) for n in u])
        v_xyz = np.array([gt.nodes[n].get('node_xyz', gt.nodes[n].get('xyz')) for n in v])

    midpoints = (u_xyz + v_xyz) / 2.0
    distances, _ = merge_tree.query(midpoints)
    neuron_ids = np.array([gt.node_segment_id(n) for n in u])

    split_mask = (edge_error == EDGE_SPLIT)
    correct_mask = (edge_error == EDGE_CORRECT)

    split_distances.extend(distances[split_mask])
    correct_distances.extend(distances[correct_mask])
    split_neuron.extend(neuron_ids[split_mask])
    correct_neuron.extend(neuron_ids[correct_mask])

split_distances = np.array(split_distances)
correct_distances = np.array(correct_distances)
split_neuron = np.array(split_neuron)
correct_neuron = np.array(correct_neuron)

print(f"Number of split edges: {len(split_distances)}")
print(f"Number of correct edges: {len(correct_distances)}")
if len(split_distances) == 0 or len(correct_distances) == 0:
    print("Not enough edges in one of the classes to perform statistical test.")
    sys.exit(0)

print(f"Median split distance:   {np.median(split_distances):.2f} um")
print(f"Median correct distance: {np.median(correct_distances):.2f} um")

# Naive tests echoed for comparison only.
stat, pval = mannwhitneyu(split_distances, correct_distances, alternative='less')
t_stat, t_pval = ttest_ind(split_distances, correct_distances, equal_var=False, alternative='less')
print(f"\nNaive Mann-Whitney U={stat}, p={pval:.2e} (for comparison only)")
print(f"Naive Welch t={t_stat:.4f}, p={t_pval:.2e} (for comparison only)")

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
print(f"Cliff's delta (split vs correct distance) = {delta:.4f}  (negative => split closer)")
print(f"Cluster bootstrap 95% CI = [{lo:.4f}, {hi:.4f}]")
print(f"Median shift = {np.median(split_distances)-np.median(correct_distances):.2f} um "
      f"({100*(np.median(split_distances)-np.median(correct_distances))/np.median(correct_distances):.2f}% of correct)")

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
print("(Compare to ORIGINAL: MW p=3.50e-38, Welch p=7.14e-27 -- significance n-driven.)")
