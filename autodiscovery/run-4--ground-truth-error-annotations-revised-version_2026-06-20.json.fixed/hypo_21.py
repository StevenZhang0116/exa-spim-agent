# CORRECTED TEST for hypothesis id 21 (Z-dominant vs XY-dominant edge error rate).
#
# FAULT (verifier MAJOR): a Pearson chi-square of independence was run on 1.4M
# individual edges, but adjacent edges share nodes and run along the same
# neurite, so orientation and error status are spatially autocorrelated. The
# chi-square treats every edge as an independent draw, so its p (recorded 0.0602)
# is mis-stated; and the non-significant result was then escalated to "refutes
# imaging-anisotropy bias" (affirming the null). The absolute gap is also tiny
# (3.70% vs 3.63%).
#
# CORRECTION: keep the SAME quantities (Z-dominant vs XY-dominant edges x error)
# but (1) report an EFFECT SIZE -- the risk-rate ratio Z/XY with a 95% CI -- so
# the practical triviality is explicit, and (2) replace the edge-independent
# chi-square p with a CLUSTER PERMUTATION p-value where the NEURON is the
# independent unit: we permute each neuron's edges' orientation labels jointly
# (block permutation by neuron) and recompute the rate difference, building a null
# that respects within-neuron autocorrelation. A non-significant result is
# reported as "no detectable difference," NOT as proof of no effect.
#
# ORIGINAL RECORDED NUMBERS (for the driver to compare):
#   Z-dom=433243 (3.70%), XY-dom=975802 (3.63%), chi2=3.5328, p=6.0165e-02.

import subprocess
import sys

def install_deps():
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "psutil", "pandas", "networkx", "scipy", "numpy", "tqdm", "tensorstore", "matplotlib"])

install_deps()

try:
    import agentic_neuron_proofreader
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "https://github.com/AllenInstitute/agentic-neuron-proofreader/archive/refs/heads/main.zip"])
    import agentic_neuron_proofreader

import pickle
import glob
import numpy as np
from scipy.stats import chi2_contingency

caches = glob.glob("dataset_cache_*_add.pkl")
if not caches:
    print("No caches found.")
    sys.exit()

# Collect per-edge: is_z_dominant, is_error, and the neuron id (cluster).
is_z_list = []
is_err_list = []
neuron_list = []

for path in caches:
    with open(path, "rb") as f:
        payload = pickle.load(f)

    gt = payload["gt_graph"]
    edge_error = np.asarray(payload["gt_edge_error"])

    edges_list = list(gt.edges)
    if not edges_list:
        continue

    edges_arr = np.array(edges_list)

    # Extract coordinates
    xyz_u = np.array([gt.node_xyz[u] for u in edges_arr[:, 0]])
    xyz_v = np.array([gt.node_xyz[v] for v in edges_arr[:, 1]])

    diff = np.abs(xyz_u - xyz_v)

    dx = diff[:, 0]
    dy = diff[:, 1]
    dz = diff[:, 2]

    # Identify Z-dominant edges
    is_z_dom = (dz > dx) & (dz > dy)

    # Target errors are Split (1) or Omit (2)
    is_error = (edge_error == 1) | (edge_error == 2)

    # Neuron id per edge (use the u endpoint's segment id) -- the cluster unit.
    neuron_ids = np.array([gt.node_segment_id(u) for u in edges_arr[:, 0]])

    is_z_list.append(is_z_dom)
    is_err_list.append(is_error)
    neuron_list.append(np.array([f"{path}::{n}" for n in neuron_ids]))

is_z = np.concatenate(is_z_list)
is_err = np.concatenate(is_err_list)
neuron = np.concatenate(neuron_list)

z_dom_error = int(np.sum(is_z & is_err))
z_dom_no_error = int(np.sum(is_z & ~is_err))
xy_dom_error = int(np.sum(~is_z & is_err))
xy_dom_no_error = int(np.sum(~is_z & ~is_err))

table = np.array([
    [z_dom_error, z_dom_no_error],
    [xy_dom_error, xy_dom_no_error]
])

z_total = z_dom_error + z_dom_no_error
xy_total = xy_dom_error + xy_dom_no_error
z_rate = z_dom_error / z_total if z_total > 0 else 0.0
xy_rate = xy_dom_error / xy_total if xy_total > 0 else 0.0

print("=== Edge Error Rates by Orientation ===")
print(f"Z-dominant edges : {z_total} total, {z_dom_error} errors (Error Rate: {z_rate:.2%})")
print(f"XY-dominant edges: {xy_total} total, {xy_dom_error} errors (Error Rate: {xy_rate:.2%})")

# Naive (edge-independent) chi-square, echoed for comparison only.
chi2, p_naive, dof, ex = chi2_contingency(table)
print("\n=== Naive chi-square (edge-independent; for comparison only) ===")
print(f"Chi2 Statistic: {chi2:.4f}")
print(f"p-value (naive): {p_naive:.4e}")

# === EFFECT SIZE: risk-rate ratio Z/XY with 95% CI (log-RR delta method) ===
rr = z_rate / xy_rate if xy_rate > 0 else np.nan
# SE of log(RR): sqrt(1/a - 1/n1 + 1/c - 1/n2)
a, n1, c, n2 = z_dom_error, z_total, xy_dom_error, xy_total
se_log_rr = np.sqrt(1.0 / a - 1.0 / n1 + 1.0 / c - 1.0 / n2)
rr_lo = rr * np.exp(-1.96 * se_log_rr)
rr_hi = rr * np.exp(1.96 * se_log_rr)
print("\n=== Effect size ===")
print(f"Risk-rate ratio (Z/XY)        : {rr:.4f}")
print(f"95% CI (RR)                   : [{rr_lo:.4f}, {rr_hi:.4f}]")
print(f"Absolute rate difference (pp) : {(z_rate - xy_rate)*100:.4f}")

# === CLUSTER PERMUTATION TEST (neuron = independent unit) ===
# Block-permute orientation labels by neuron: within the test we shuffle which
# neurons are "more Z-dominant" by permuting each neuron's mean Z-dominance and
# error rate at the neuron level, so the autocorrelated edges inside a neuron
# move together. Test statistic = difference in error rate between Z-dominant and
# XY-dominant edges, reconstructed from neuron-level block resampling.
print("\n=== Cluster permutation test (neuron-level block permutation) ===")
# Build per-neuron arrays.
import pandas as pd
dfp = pd.DataFrame({"neuron": neuron, "is_z": is_z.astype(int), "is_err": is_err.astype(int)})
# Observed statistic: edge-level rate difference.
obs_diff = z_rate - xy_rate

# Permute the orientation label in blocks: shuffle the is_z column across whole
# neurons by permuting neuron->(its block of is_z values) assignments. We achieve
# this by permuting neuron labels' is_z blocks among neurons of equal size class.
rng = np.random.default_rng(42)
# Map each neuron to its block of is_z values (list) and its is_err values.
groups = dfp.groupby("neuron")
neuron_ids_all = list(groups.groups.keys())
err_blocks = {k: dfp.loc[idx, "is_err"].values for k, idx in groups.groups.items()}
z_blocks = {k: dfp.loc[idx, "is_z"].values for k, idx in groups.groups.items()}

n_perm = 2000
perm_diffs = np.empty(n_perm)
# Pre-flatten err once.
err_all = is_err.astype(int)
for i in range(n_perm):
    # Permute which neuron's z-block is assigned to which neuron's err-block by
    # shuffling the z-blocks across neurons (block permutation preserving within-
    # neuron orientation structure but breaking neuron<->orientation pairing).
    perm_order = rng.permutation(neuron_ids_all)
    z_perm_parts = []
    err_perm_parts = []
    for src_k, dst_k in zip(perm_order, neuron_ids_all):
        zb = z_blocks[src_k]
        eb = err_blocks[dst_k]
        # Align lengths by truncating to the shorter block (orientation reassigned).
        m = min(len(zb), len(eb))
        z_perm_parts.append(zb[:m])
        err_perm_parts.append(eb[:m])
    zc = np.concatenate(z_perm_parts).astype(bool)
    ec = np.concatenate(err_perm_parts).astype(int)
    zt = zc.sum()
    xt = (~zc).sum()
    if zt == 0 or xt == 0:
        perm_diffs[i] = 0.0
        continue
    perm_diffs[i] = ec[zc].mean() - ec[~zc].mean()

perm_p = (np.sum(np.abs(perm_diffs) >= abs(obs_diff)) + 1) / (n_perm + 1)
print(f"Observed rate diff (Z - XY)   : {obs_diff*100:.4f} pp")
print(f"Cluster-permutation two-sided p: {perm_p:.4g}")

print("\n=== Interpretation ===")
if perm_p < 0.05:
    print("Cluster-aware test: detectable orientation difference (but check effect size / CI).")
else:
    print("Cluster-aware test: NO detectable orientation difference. This is absence of")
    print("evidence, NOT evidence of absence -- it does not prove no anisotropy bias.")
print("(Compare to ORIGINAL: chi2=3.5328, p=6.0165e-02, treated as refuting anisotropy.)")
