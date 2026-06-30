# CORRECTED TEST for hypothesis id 36 (omit nodes cluster closer to merge sites
# than correct nodes).
#
# FAULTS (verifier MAJOR): (1) the p=1.6e-66 is driven purely by sample size --
# the median difference is ~147 um on a ~1800 um baseline (~8%), a practically
# trivial shift, yet n~49k per arm forces an astronomically small p. (2) The 49k
# nodes are NOT independent: they are clustered along neurites and all measured to
# the same 67 anchor merge sites, so the effective n is far smaller than 49,295
# and the p is massively overstated.
#
# CORRECTION: keep the SAME distances but (1) report an EFFECT SIZE -- Cliff's
# delta (rank-biserial) with a bootstrap 95% CI -- so the ~8% practical shift is
# explicit, and (2) replace the naive Mann-Whitney p with a NEURON-CLUSTER
# permutation p-value: aggregate each node's distance to its neuron, then permute
# the omit/correct LABEL at the NEURON level (the independent unit) and recompute
# the median-distance difference, building a null that respects within-neuron
# correlation.
#
# ORIGINAL RECORDED NUMBERS (for the driver to compare):
#   67 merge sites, omit median=1812.73 um, correct median=1959.88 um,
#   one-sided Mann-Whitney U=1138194675.0, p=1.6036e-66.

import os
import subprocess
import sys

try:
    import psutil
    import pandas
    import tensorstore
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "psutil", "pandas", "tensorstore"])

try:
    import agentic_neuron_proofreader
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "https://github.com/AllenInstitute/agentic-neuron-proofreader/archive/refs/heads/main.zip"])
    import agentic_neuron_proofreader

import pickle
import numpy as np
from scipy.spatial import KDTree
from scipy.stats import mannwhitneyu

# Load the dataset from the current directory
path = "./dataset_cache_789202_mcl100_add.pkl"
with open(path, "rb") as f:
    payload = pickle.load(f)

gt = payload["gt_graph"]
edge_error = np.asarray(payload["gt_edge_error"])
merge_sites = payload["gt_merge_sites"]

# Extract spatial coordinates for merge sites
merge_xyz = np.array([site["xyz"] for site in merge_sites])

if len(merge_xyz) == 0:
    print("No merge sites found.")
    sys.exit(0)

# Identify omit nodes and correctly reconstructed nodes using the edge errors
omit_nodes = set()
correct_nodes = set()

gt_edges = list(gt.edges)
for i in range(len(gt_edges)):
    u, v = gt_edges[i]
    err = edge_error[i]
    if err == 2:  # EDGE_OMIT
        omit_nodes.add(u)
        omit_nodes.add(v)
    elif err == 0:  # EDGE_CORRECT
        correct_nodes.add(u)
        correct_nodes.add(v)

# Exclude omit nodes from the correct nodes pool to ensure strictly distinct spatial sets
correct_nodes = list(correct_nodes - omit_nodes)
omit_nodes = list(omit_nodes)

# Extract (x,y,z) coordinates strictly using gt.node_xyz parallel array
omit_xyz = np.array([gt.node_xyz[n] for n in omit_nodes])
correct_xyz = np.array([gt.node_xyz[n] for n in correct_nodes])

# Neuron id per node (the cluster / independent unit).
omit_neuron = np.array([gt.node_segment_id(n) for n in omit_nodes])
correct_neuron_full = np.array([gt.node_segment_id(n) for n in correct_nodes])

# Compute distance to nearest merge site for omit nodes
merge_tree = KDTree(merge_xyz)
dist_omit, _ = merge_tree.query(omit_xyz)

# Draw a matched random sample of correct nodes and compute distance to nearest merge site
np.random.seed(42)
if len(correct_xyz) > len(omit_xyz):
    sampled_correct_idx = np.random.choice(len(correct_xyz), size=len(omit_xyz), replace=False)
    sampled_correct_xyz = correct_xyz[sampled_correct_idx]
    sampled_correct_neuron = correct_neuron_full[sampled_correct_idx]
else:
    sampled_correct_xyz = correct_xyz
    sampled_correct_neuron = correct_neuron_full

dist_correct, _ = merge_tree.query(sampled_correct_xyz)

median_dist_omit = np.median(dist_omit)
median_dist_correct = np.median(dist_correct)

# Naive Mann-Whitney echoed for comparison only.
stat, pval = mannwhitneyu(dist_omit, dist_correct, alternative='less')

print("=== Spatial Clustering of Omit Errors vs Merge Errors ===")
print(f"Number of merge sites: {len(merge_xyz)}")
print(f"Number of omit nodes: {len(omit_xyz)}")
print(f"Number of correct nodes (matched sample): {len(sampled_correct_xyz)}")
print(f"Median distance omit -> nearest merge:    {median_dist_omit:.2f} um")
print(f"Median distance correct -> nearest merge: {median_dist_correct:.2f} um")
print(f"Naive Mann-Whitney U={stat}, p={pval:.4e} (for comparison only)")

# === EFFECT SIZE: Cliff's delta (omit vs correct) with bootstrap 95% CI ===
def cliffs_delta(a, b, rng, n_sub=4000):
    # Subsample for tractable O(n_sub^2) computation on large arrays.
    a_s = a if len(a) <= n_sub else rng.choice(a, n_sub, replace=False)
    b_s = b if len(b) <= n_sub else rng.choice(b, n_sub, replace=False)
    gt_ = sum((x > b_s).sum() for x in a_s)
    lt_ = sum((x < b_s).sum() for x in a_s)
    return (gt_ - lt_) / (len(a_s) * len(b_s))

rng = np.random.default_rng(42)
delta = cliffs_delta(dist_omit, dist_correct, rng)
boots = []
for _ in range(500):
    bo = rng.choice(dist_omit, len(dist_omit), replace=True)
    bc = rng.choice(dist_correct, len(dist_correct), replace=True)
    boots.append(cliffs_delta(bo, bc, rng, n_sub=2000))
lo, hi = np.percentile(boots, [2.5, 97.5])
print("\n=== Effect size ===")
print(f"Cliff's delta (omit vs correct distance) = {delta:.4f}  (negative => omit closer)")
print(f"Bootstrap 95% CI = [{lo:.4f}, {hi:.4f}]")
print(f"Median shift = {median_dist_omit - median_dist_correct:.2f} um "
      f"({100*(median_dist_omit-median_dist_correct)/median_dist_correct:.2f}% of correct median)")

# === NEURON-CLUSTER permutation test (neuron = independent unit) ===
# Aggregate distance to one value per (neuron, label) and permute the omit/correct
# label across neuron-units, so within-neuron correlation cannot inflate the test.
print("\n=== Neuron-cluster permutation test (neuron = independent unit) ===")
import pandas as pd
omit_df = pd.DataFrame({"neuron": omit_neuron, "dist": dist_omit, "label": "omit"})
corr_df = pd.DataFrame({"neuron": sampled_correct_neuron, "dist": dist_correct, "label": "correct"})
allg = pd.concat([omit_df, corr_df], ignore_index=True)
# One representative (median distance) per neuron-x-label unit.
unit = allg.groupby(["neuron", "label"])["dist"].median().reset_index()
omit_units = unit[unit["label"] == "omit"]["dist"].values
corr_units = unit[unit["label"] == "correct"]["dist"].values
obs_stat = np.median(omit_units) - np.median(corr_units)

pooled = unit["dist"].values
labels = (unit["label"].values == "omit")
n_omit = labels.sum()
n_perm = 5000
perm_stats = np.empty(n_perm)
for i in range(n_perm):
    perm = rng.permutation(len(pooled))
    sel = perm[:n_omit]
    rest = perm[n_omit:]
    perm_stats[i] = np.median(pooled[sel]) - np.median(pooled[rest])
# One-sided (omit closer => negative stat).
perm_p = (np.sum(perm_stats <= obs_stat) + 1) / (n_perm + 1)
print(f"  Neuron-omit units = {len(omit_units)}, neuron-correct units = {len(corr_units)}")
print(f"  Observed neuron-level median diff (omit-correct) = {obs_stat:.2f} um")
print(f"  Cluster-permutation one-sided p (omit closer) = {perm_p:.4g}")
print("  (Compare to ORIGINAL: U=1138194675.0, p=1.6036e-66 treating 49k nodes as independent.)")
