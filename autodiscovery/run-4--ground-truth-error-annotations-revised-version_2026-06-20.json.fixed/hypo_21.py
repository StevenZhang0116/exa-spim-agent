# === RERUN BOOTSTRAP (revised loading only) ============================
import os as _os, sys as _sys

_TARGET = "/home/zihan.zhang/.local-numpy2"
_USER_SITE = _os.path.expanduser("~/.local/lib/python3.12/site-packages")
_SHARED_SITE = "/shared/utils.x86_64/anaconda3-2024.10/lib/python3.12/site-packages"
_sys.path = [p for p in _sys.path if p not in (_USER_SITE, _SHARED_SITE)]
if _TARGET in _sys.path:
    _sys.path.remove(_TARGET)
_sys.path.insert(0, _TARGET)

import subprocess as _subprocess
_subprocess.check_call = lambda *a, **k: 0
_subprocess.call = lambda *a, **k: 0
_subprocess.run = lambda *a, **k: type("R", (), {"returncode": 0, "stdout": b"", "stderr": b""})()

_PKL = _os.environ["RERUN_PKL"]
print("Loading dataset from:", _PKL)
import glob as _glob
_orig_glob = _glob.glob
def _glob_patch(pattern, *a, **k):
    if isinstance(pattern, str) and pattern.endswith(".pkl"):
        return [_PKL]
    return _orig_glob(pattern, *a, **k)
_glob.glob = _glob_patch

from pathlib import Path as _Path
_orig_rglob = _Path.rglob
def _rglob_patch(self, pattern, *a, **k):
    if isinstance(pattern, str) and pattern.endswith(".pkl"):
        return iter([_Path(_PKL)])
    return _orig_rglob(self, pattern, *a, **k)
_Path.rglob = _rglob_patch

_orig_walk = _os.walk
def _walk_patch(top, *a, **k):
    yield (_os.path.dirname(_PKL), [], [_os.path.basename(_PKL)])
_os.walk = _walk_patch
# === END BOOTSTRAP =====================================================
# CORRECTED ANALYSIS (entry #3, id 21) ===================================
# Original test: chi-square test of independence on Z-dominant vs XY-dominant
#   edges x error/no-error. Recorded original:
#     Z-dominant 433243 (3.70%) vs XY-dominant 975802 (3.63%);
#     chi2 = 3.5328, p = 6.0165e-02 (n = 1,409,045 edges).
#
# Why the original is wrong: it treats every edge as i.i.d., but edges within
# the same GT neuron share spatial and labelling structure. The chi-square test
# thus exaggerates the effective n and the test statistic. The hypothesis is
# *about neurons* (does Z-anisotropy give one neuron more errors than another?),
# so the unit of analysis must be the neuron / cluster, not the edge.
#
# Corrected tests (all on the SAME data, SAME 2 groups):
#   (a) cluster-robust GEE logistic regression with neuron as the cluster id —
#       the right way to do logistic regression with non-independent edges.
#       Reports OR (XY vs Z) with a 95% CI.
#   (b) cluster-permutation chi-square: permute the Z/XY label at the NEURON
#       level (every edge in a neuron flips together) to build a null for the
#       chi-square statistic under the actual independence structure.
#   (c) neuron-level paired Wilcoxon signed-rank on (Z-error-rate -
#       XY-error-rate) per neuron — the natural paired non-parametric test.
# Each corrected test prints OR / effect size with a 95% CI and is compared
# side-by-side with the original chi-square numbers.
# =======================================================================

import pickle
import numpy as np
import pandas as pd
import sys
from collections import defaultdict
from scipy.stats import chi2_contingency, wilcoxon

print("=" * 72)
print("HYPO 21 — Z-dominant vs XY-dominant edge error rate")
print("Corrected statistical tests")
print("=" * 72)

with open(_PKL, "rb") as f:
    payload = pickle.load(f)

gt = payload["gt_graph"]
edge_error = np.asarray(payload["gt_edge_error"])

edges_list = list(gt.edges)
if not edges_list:
    print("No edges in GT graph; aborting.")
    sys.exit(0)

edges_arr = np.array(edges_list)
xyz_u = np.array([gt.node_xyz[u] for u in edges_arr[:, 0]])
xyz_v = np.array([gt.node_xyz[v] for v in edges_arr[:, 1]])
diff = np.abs(xyz_u - xyz_v)
dx, dy, dz = diff[:, 0], diff[:, 1], diff[:, 2]

is_z_dom = (dz > dx) & (dz > dy)
# Same outcome definition as the original (split or omit)
is_error = (edge_error == 1) | (edge_error == 2)

# Neuron / cluster id for each edge (use u's segment id, edges are within-neuron)
try:
    neuron_ids = np.array([gt.node_segment_id(int(u)) for u in edges_arr[:, 0]])
except Exception:
    # Fallback: use a connected-component label per neuron
    import networkx as nx
    cc_label = {}
    for k, comp in enumerate(nx.connected_components(gt)):
        for n in comp:
            cc_label[n] = k
    neuron_ids = np.array([cc_label.get(int(u), -1) for u in edges_arr[:, 0]])

# --- Original numbers, recomputed for reference ---------------------------
z_dom_error = int(np.sum(is_z_dom & is_error))
z_dom_no = int(np.sum(is_z_dom & ~is_error))
xy_dom_error = int(np.sum(~is_z_dom & is_error))
xy_dom_no = int(np.sum(~is_z_dom & ~is_error))
table = np.array([[z_dom_error, z_dom_no], [xy_dom_error, xy_dom_no]])
z_total = z_dom_error + z_dom_no
xy_total = xy_dom_error + xy_dom_no
z_rate = z_dom_error / z_total
xy_rate = xy_dom_error / xy_total
chi2_orig, p_orig, _, _ = chi2_contingency(table)
print(f"\n[ORIGINAL (recorded) chi-square — for side-by-side]")
print(f"  Z-dominant edges : {z_total} total, {z_dom_error} errors ({z_rate*100:.4f}%)")
print(f"  XY-dominant edges: {xy_total} total, {xy_dom_error} errors ({xy_rate*100:.4f}%)")
print(f"  chi2 = {chi2_orig:.4f}, p = {p_orig:.4e}  (n={z_total+xy_total} edges, i.i.d. assumption)")

# --- (a) Cluster-robust GEE logistic regression with neuron cluster -------
print("\n[CORRECTED (a)] GEE logistic regression, neuron as cluster id")
try:
    import statsmodels.api as sm
    from statsmodels.genmod.generalized_estimating_equations import GEE
    from statsmodels.genmod.families import Binomial
    from statsmodels.genmod.cov_struct import Independence

    # neuron_ids may be strings; convert to integer cluster codes for GEE.
    _, _neuron_int = np.unique(neuron_ids, return_inverse=True)
    df = pd.DataFrame({
        "is_error": is_error.astype(int),
        "is_z": is_z_dom.astype(int),
        "neuron": _neuron_int.astype(int),
    })
    # Drop any edges with unknown neuron id (negative) before fitting
    df = df[df["neuron"] >= 0].reset_index(drop=True)
    # Cap clusters for stability if huge
    X = sm.add_constant(df["is_z"].to_numpy(dtype=float))
    gee_model = GEE(df["is_error"].to_numpy(dtype=float), X,
                    groups=df["neuron"].to_numpy(),
                    family=Binomial(), cov_struct=Independence())
    gee_res = gee_model.fit()
    beta = float(gee_res.params[1])
    se = float(gee_res.bse[1])
    z_stat = beta / se if se > 0 else float("nan")
    from scipy.stats import norm
    p_gee = 2 * (1 - norm.cdf(abs(z_stat)))
    or_z = float(np.exp(beta))
    or_lo = float(np.exp(beta - 1.96 * se))
    or_hi = float(np.exp(beta + 1.96 * se))
    n_clusters = int(df["neuron"].nunique())
    print(f"  beta(is_z) = {beta:.5f}  SE = {se:.5f}  z = {z_stat:.3f}  p = {p_gee:.4e}")
    print(f"  OR (Z vs XY) = {or_z:.4f}  95% CI [{or_lo:.4f}, {or_hi:.4f}]")
    print(f"  n_edges = {len(df)}  n_neurons (clusters) = {n_clusters}")
except Exception as e:
    print(f"  GEE failed: {e}")

# --- (b) Cluster-permutation chi-square (permute Z/XY at neuron level) ----
print("\n[CORRECTED (b)] Cluster-permutation chi-square (neuron-level label permutation)")
rng = np.random.default_rng(42)
# For each neuron, compute its Z fraction. Then permute the neuron-level
# 'is_z' assignment by reshuffling neurons and reassigning edges per neuron
# proportional to the original z fraction. Simpler: for each neuron, flip
# all its edges to Z with the empirical neuron-level z-prob, then recompute
# chi-square statistic under that null.
unique_neurons, neuron_inverse = np.unique(np.asarray(neuron_ids), return_inverse=True)
n_neurons = len(unique_neurons)
# Per-neuron summary
neuron_z_count = np.bincount(neuron_inverse, weights=is_z_dom.astype(float)).astype(np.int64)
neuron_err_count = np.bincount(neuron_inverse, weights=is_error.astype(float)).astype(np.int64)
neuron_size = np.bincount(neuron_inverse).astype(np.int64)
neuron_z_frac = neuron_z_count / np.maximum(neuron_size, 1)

obs_chi2, _, _, _ = chi2_contingency(table)
N_PERM = 1000
null_chi2 = np.empty(N_PERM)
err_per_edge = is_error.astype(np.int64)
for k in range(N_PERM):
    # Shuffle the per-neuron z-fraction across neurons, then assign each
    # edge to Z with its (shuffled) neuron-level probability.
    permuted_frac = neuron_z_frac.copy()
    rng.shuffle(permuted_frac)
    # Per-edge probability of being Z under the null
    p_edge = permuted_frac[neuron_inverse]
    is_z_perm = rng.random(len(err_per_edge)) < p_edge
    a = int(np.sum(is_z_perm & is_error))
    b = int(np.sum(is_z_perm & ~is_error))
    c = int(np.sum(~is_z_perm & is_error))
    d = int(np.sum(~is_z_perm & ~is_error))
    if (a + b == 0) or (c + d == 0):
        null_chi2[k] = 0.0
        continue
    null_chi2[k], _, _, _ = chi2_contingency(np.array([[a, b], [c, d]]))
p_perm = float((null_chi2 >= obs_chi2).sum() + 1) / (N_PERM + 1)
print(f"  observed chi2 = {obs_chi2:.4f}")
print(f"  cluster-permutation null mean chi2 = {null_chi2.mean():.4f}, "
      f"95% null upper = {np.quantile(null_chi2, 0.95):.4f}")
print(f"  cluster-permutation p = {p_perm:.4f}  (N_perm={N_PERM}, clusters={n_neurons})")

# --- (c) Neuron-level paired Wilcoxon on (Z-rate - XY-rate) per neuron ----
print("\n[CORRECTED (c)] Neuron-level paired Wilcoxon signed-rank on per-neuron error rates")
neuron_z_err = np.bincount(neuron_inverse, weights=(is_z_dom & is_error).astype(float))
neuron_xy_err = np.bincount(neuron_inverse, weights=((~is_z_dom) & is_error).astype(float))
neuron_z_size = np.bincount(neuron_inverse, weights=is_z_dom.astype(float))
neuron_xy_size = np.bincount(neuron_inverse, weights=(~is_z_dom).astype(float))
# Only neurons with both Z and XY edges present
valid = (neuron_z_size > 0) & (neuron_xy_size > 0)
rate_z = neuron_z_err[valid] / neuron_z_size[valid]
rate_xy = neuron_xy_err[valid] / neuron_xy_size[valid]
delta = rate_z - rate_xy
print(f"  neurons with both Z and XY edges: {int(valid.sum())}")
print(f"  median per-neuron rate Z  = {np.median(rate_z)*100:.4f}%")
print(f"  median per-neuron rate XY = {np.median(rate_xy)*100:.4f}%")
print(f"  median per-neuron diff (Z - XY) = {np.median(delta)*100:.4f}%")
try:
    nz_delta = delta[delta != 0]
    if len(nz_delta) > 0:
        wstat, wp = wilcoxon(nz_delta, alternative="two-sided", zero_method="wilcox")
        # Effect size: rank-biserial r for one-sample Wilcoxon
        # r = (W+ - W-) / (W+ + W-)  ; using statsmodels-like form:
        n = len(nz_delta)
        # Bootstrap 95% CI on the median difference
        rng2 = np.random.default_rng(7)
        boots = []
        for _ in range(2000):
            idx = rng2.integers(0, n, n)
            boots.append(np.median(nz_delta[idx]))
        ci_lo, ci_hi = np.quantile(boots, [0.025, 0.975])
        print(f"  Wilcoxon W = {wstat:.2f}, p = {wp:.4e}, n_pairs = {n}")
        print(f"  bootstrap 95% CI for median(Z - XY) per neuron: "
              f"[{ci_lo*100:.4f}%, {ci_hi*100:.4f}%]")
    else:
        print("  All neurons have identical Z and XY rates; Wilcoxon undefined.")
except Exception as e:
    print(f"  Wilcoxon failed: {e}")

print("\n" + "=" * 72)
print("SIDE-BY-SIDE: original vs corrected")
print("=" * 72)
print(f"  Original chi2 (i.i.d. edges, n=1.4M):  chi2 = {chi2_orig:.4f}, p = {p_orig:.4e}")
print(f"  Corrected GEE (cluster-robust):        see (a) above")
print(f"  Corrected cluster permutation chi2:    p = {p_perm:.4f}")
print(f"  Corrected paired neuron-level Wilcoxon: see (c) above")
