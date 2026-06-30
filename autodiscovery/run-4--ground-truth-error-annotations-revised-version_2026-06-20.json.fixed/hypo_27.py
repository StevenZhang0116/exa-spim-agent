# CORRECTED TEST for hypothesis id 27 (z-alignment -> split/omit risk).
#
# FAULT (verifier MAJOR): the recorded analysis fit a variational-Bayes mixed
# GLM (BinomialBayesMixedGLM.fit_vb) on a 20k-edge subsample, then MANUFACTURED a
# frequentist two-sided Wald p-value by treating the VB posterior mean/SD as a
# sampling distribution (z = Post.Mean/Post.SD; p = 2*(1-Phi(|z|))). A VB
# posterior is NOT a sampling distribution, so that p (recorded 0.05823) has no
# valid frequentist interpretation. It also discarded ~98% of the 1.16M edges and
# ignored within-neuron edge correlation.
#
# CORRECTION: fit a proper frequentist logistic regression on the FULL edge set
# (is_error ~ z_align_std + length_std), and obtain a VALID p-value with
# CLUSTER-ROBUST standard errors clustered BY NEURON (cov_type='cluster'), which
# accounts for the non-independence of edges within the same neuron. Report the
# odds ratio for z-alignment WITH a 95% confidence interval as the effect size.
# As an independence-respecting cross-check, also report a neuron-level
# permutation p-value: permute the per-neuron mean z-alignment against the
# per-neuron error rate (the cluster is the independent unit).
#
# ORIGINAL RECORDED NUMBERS (for the driver to compare):
#   edges=1160529, errors=4.44%, z_align Post.Mean(std)=-0.0661, Post.SD=0.0349,
#   manufactured Wald p=0.05823, OR=0.8029  -> "does NOT significantly affect".

import subprocess
import sys
import os
import glob
import pickle
import numpy as np
import pandas as pd
import warnings
import scipy.stats
import re

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", package])

# Ensure required packages are installed
packages_to_install = ["psutil", "tqdm", "statsmodels", "tensorstore", "matplotlib"]
for pkg in packages_to_install:
    try:
        __import__(pkg)
    except ImportError:
        install(pkg)

import statsmodels.api as sm
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

try:
    import agentic_neuron_proofreader
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "https://github.com/AllenInstitute/agentic-neuron-proofreader/archive/refs/heads/main.zip"])
    import agentic_neuron_proofreader

def brain_id_from_path(path):
    m = re.search(r"dataset_cache_(\d+)_mcl(\d+)_add\.pkl$", os.path.basename(path))
    return m.group(1) if m else os.path.basename(path)

# 1. Load all available _add.pkl caches
cache_files = glob.glob("./dataset_cache_*_add.pkl")
if not cache_files:
    cache_files = glob.glob("../data/dataset_cache_*_add.pkl")
if not cache_files:
    cache_files = glob.glob("../*/*_add.pkl")

if not cache_files:
    print("No cache files found.")
    sys.exit(0)

edges_data = []

print("Loading dataset caches and extracting edges...")
for path in sorted(cache_files):
    brain_id = brain_id_from_path(path)
    with open(path, "rb") as f:
        payload = pickle.load(f)

    gt = payload["gt_graph"]
    edge_error = np.asarray(payload["gt_edge_error"])
    node_xyz = gt.node_xyz

    edges = list(gt.edges)
    if not edges:
        continue

    # Vectorized computation of edge geometries
    u_nodes = np.array([e[0] for e in edges])
    v_nodes = np.array([e[1] for e in edges])

    xyz_u = node_xyz[u_nodes]
    xyz_v = node_xyz[v_nodes]

    # Euclidean length and absolute Z-difference
    dz = np.abs(xyz_u[:, 2] - xyz_v[:, 2])
    lengths = np.linalg.norm(xyz_u - xyz_v, axis=1)

    # Filter: length > 0, and not 'merged' (which the plan excludes from the binary split/omit classification)
    # 0 = correct, 1 = split, 2 = omit, 3 = merged
    valid_mask = (lengths > 1e-6) & (edge_error != 3)

    valid_u = u_nodes[valid_mask]
    valid_errs = edge_error[valid_mask]
    valid_lengths = lengths[valid_mask]
    valid_dz = dz[valid_mask]

    # 1 if split (1) or omit (2), 0 if correct (0)
    is_err = np.isin(valid_errs, [1, 2]).astype(int)
    z_align = valid_dz / valid_lengths

    # GT neuron names
    neuron_ids = [gt.node_segment_id(u) for u in valid_u]

    df_brain = pd.DataFrame({
        "is_error": is_err,
        "z_alignment": z_align,
        "edge_length": valid_lengths,
        "neuron_id": neuron_ids,
        "brain_id": brain_id
    })
    edges_data.append(df_brain)

    del payload, gt, edge_error, node_xyz

df = pd.concat(edges_data, ignore_index=True)
print(f"Total valid edges loaded: {len(df)}")
print(f"Total errors (splits/omits): {df['is_error'].sum()} ({df['is_error'].mean()*100:.2f}%)\n")

# Standardize continuous variables (same standardization as the original).
df['z_align_std'] = (df['z_alignment'] - df['z_alignment'].mean()) / df['z_alignment'].std()
df['length_std'] = (df['edge_length'] - df['edge_length'].mean()) / df['edge_length'].std()

# Cluster id (neuron within brain) -- the independent unit.
df['cluster'] = df['brain_id'].astype(str) + "::" + df['neuron_id'].astype(str)

# === CORRECTED TEST 1: frequentist logistic regression on the FULL data with
# cluster-robust SEs (clustered by neuron). This gives a VALID Wald p and CI. ===
print("=== Corrected frequentist logistic regression (FULL data, cluster-robust by neuron) ===")
X = sm.add_constant(df[['z_align_std', 'length_std']])
y = df['is_error'].astype(float)
groups = df['cluster'].values

model = sm.GLM(y, X, family=sm.families.Binomial())
res_naive = model.fit()
res = model.fit(cov_type='cluster', cov_kwds={'groups': groups})

coef_z_std = res.params['z_align_std']
p_z_cluster = res.pvalues['z_align_std']
p_z_naive = res_naive.pvalues['z_align_std']

# Convert standardized coefficient back to the per-unit (0..1) scale for OR,
# matching the original report's OR convention.
orig_std_z = df['z_alignment'].std()
coef_orig = coef_z_std / orig_std_z
odds_ratio = float(np.exp(coef_orig))

# 95% CI on the OR using cluster-robust SE (on the standardized coef, then mapped).
ci_lo_std, ci_hi_std = res.conf_int().loc['z_align_std'].values
or_ci_lo = float(np.exp(ci_lo_std / orig_std_z))
or_ci_hi = float(np.exp(ci_hi_std / orig_std_z))

print(f"  N edges (full)                     : {len(df)}")
print(f"  N clusters (neurons)               : {df['cluster'].nunique()}")
print(f"  z_align_std coef                   : {coef_z_std:.4f}")
print(f"  Naive (independent) Wald p         : {p_z_naive:.4g}")
print(f"  CLUSTER-ROBUST Wald p (by neuron)  : {p_z_cluster:.4g}")
print(f"  Odds Ratio (z fully along Z vs XY) : {odds_ratio:.4f}")
print(f"  OR 95% CI (cluster-robust)         : [{or_ci_lo:.4f}, {or_ci_hi:.4f}]")

# === CORRECTED TEST 2: neuron-level permutation test (cluster = independent unit) ===
# Per neuron, compute mean z-alignment and error rate; correlate them, then
# permute the neuron-level z-alignment vector to get a null distribution. This
# treats each neuron (not each edge) as one observation, fully respecting
# independence.
print("\n=== Corrected cluster-level permutation test (neuron = independent unit) ===")
grp = df.groupby('cluster').agg(z_mean=('z_alignment', 'mean'),
                                err_rate=('is_error', 'mean'),
                                n=('is_error', 'size'))
grp = grp[grp['n'] >= 5]  # need a few edges to estimate a neuron's rate
z_arr = grp['z_mean'].values
e_arr = grp['err_rate'].values
obs_rho, _ = scipy.stats.spearmanr(z_arr, e_arr)

rng = np.random.default_rng(42)
n_perm = 5000
perm_rhos = np.empty(n_perm)
for i in range(n_perm):
    perm = rng.permutation(z_arr)
    perm_rhos[i], _ = scipy.stats.spearmanr(perm, e_arr)
# Two-sided permutation p-value.
perm_p = (np.sum(np.abs(perm_rhos) >= abs(obs_rho)) + 1) / (n_perm + 1)
print(f"  Neuron clusters used               : {len(grp)}")
print(f"  Spearman rho(neuron z vs err rate) : {obs_rho:.4f}")
print(f"  Permutation two-sided p            : {perm_p:.4g}")

print("\n=== Interpretation ===")
direction = "INCREASES" if coef_z_std > 0 else "DECREASES"
sig_cluster = p_z_cluster < 0.05
print(f"  Direction of z-alignment effect    : {direction}")
print(f"  Significant under cluster-robust p?: {sig_cluster}")
print("  (Compare to ORIGINAL: invalid VB-Wald p=0.05823, OR=0.8029.)")
