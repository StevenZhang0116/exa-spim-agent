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
# CORRECTED ANALYSIS (entry #2, id 27) ===================================
# Original test: Bayesian variational mixed-effects logistic regression on a
#   20,000-edge SUBSAMPLE of 1,160,529 valid edges. Recorded result:
#     z_align coef = -0.0661 (standardized), p = 0.05823, OR = 0.8029.
#
# Why the original is wrong / weak:
#   - It throws away ~98% of the available rows for no good reason; that
#     inflates SE and parks the p-value right at the 0.058 borderline.
#   - The conclusion "Z not a driver" is drawn from p > 0.05 (failed-to-reject
#     != null is true).
#
# Corrected test: standard logistic regression on the FULL ~1.16M edges with
#   CLUSTER-ROBUST standard errors (neuron cluster) via GEE. This preserves
#   the non-independence structure the random effect was trying to capture
#   AND uses all the data. We report OR and a 95% CI on the original (un-
#   standardized) z-alignment scale (0 to 1), and compare side-by-side to
#   the recorded subsample numbers.
# =======================================================================
import pickle
import numpy as np
import pandas as pd
import sys
import re
import os
from scipy.stats import norm

print("=" * 72)
print("HYPO 27 — Z-alignment effect on edge errors (corrected)")
print("=" * 72)

with open(_PKL, "rb") as f:
    payload = pickle.load(f)

brain_id = os.path.basename(_PKL)
m = re.search(r"dataset_cache_(\d+)_mcl(\d+)_add\.pkl$", brain_id)
brain_id = m.group(1) if m else brain_id

gt = payload["gt_graph"]
edge_error = np.asarray(payload["gt_edge_error"])
node_xyz = gt.node_xyz

edges = list(gt.edges)
if not edges:
    print("No edges; aborting.")
    sys.exit(0)

u_nodes = np.array([e[0] for e in edges])
v_nodes = np.array([e[1] for e in edges])
xyz_u = node_xyz[u_nodes]
xyz_v = node_xyz[v_nodes]
dz = np.abs(xyz_u[:, 2] - xyz_v[:, 2])
lengths = np.linalg.norm(xyz_u - xyz_v, axis=1)

valid_mask = (lengths > 1e-6) & (edge_error != 3)  # drop merged like original
valid_u = u_nodes[valid_mask]
valid_errs = edge_error[valid_mask]
valid_lengths = lengths[valid_mask]
valid_dz = dz[valid_mask]
is_err = np.isin(valid_errs, [1, 2]).astype(int)
z_align = valid_dz / valid_lengths

try:
    neuron_ids = np.array([gt.node_segment_id(int(u)) for u in valid_u])
except Exception:
    import networkx as nx
    cc_label = {n: k for k, comp in enumerate(nx.connected_components(gt)) for n in comp}
    neuron_ids = np.array([cc_label.get(int(u), -1) for u in valid_u])
_, neuron_int = np.unique(neuron_ids, return_inverse=True)

z_mean = z_align.mean()
z_std = z_align.std() if z_align.std() > 0 else 1.0
len_mean = valid_lengths.mean()
len_std = valid_lengths.std() if valid_lengths.std() > 0 else 1.0

df = pd.DataFrame({
    "is_error": is_err,
    "z_align_std": (z_align - z_mean) / z_std,
    "length_std": (valid_lengths - len_mean) / len_std,
    "neuron": neuron_int,
})
print(f"\nTotal valid edges loaded: {len(df)}")
print(f"Errors (splits/omits): {int(df['is_error'].sum())} "
      f"({df['is_error'].mean()*100:.2f}%)")

# --- ORIGINAL (subsample) for side-by-side -------------------------------
print("\n[ORIGINAL — recorded mixed-effects on a 20k subsample, for reference]")
print(f"  standardized coef = -0.0661, p = 0.05823, OR = 0.8029  (n=20000 of {len(df)})")

# --- CORRECTED: GEE with cluster-robust SE on the FULL dataset -----------
print("\n[CORRECTED] GEE logistic regression on FULL dataset, "
      "cluster-robust SE by neuron")
import statsmodels.api as sm
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.cov_struct import Independence

X = sm.add_constant(df[["z_align_std", "length_std"]].to_numpy(dtype=float))
y = df["is_error"].to_numpy(dtype=float)
groups = df["neuron"].to_numpy()
gee = GEE(y, X, groups=groups, family=Binomial(), cov_struct=Independence())
res = gee.fit()
beta_z_std = float(res.params[1])
se_z_std = float(res.bse[1])
z_stat = beta_z_std / se_z_std if se_z_std > 0 else float("nan")
p_z = 2 * (1 - norm.cdf(abs(z_stat)))
# Map back to the original scale: beta_orig = beta_std / z_std
beta_z_orig = beta_z_std / z_std
or_per_unit = float(np.exp(beta_z_orig))
or_lo = float(np.exp(beta_z_orig - 1.96 * se_z_std / z_std))
or_hi = float(np.exp(beta_z_orig + 1.96 * se_z_std / z_std))
n_clusters = int(df["neuron"].nunique())

print(f"  standardized coef(z_align) = {beta_z_std:.4f}  SE = {se_z_std:.4f}  "
      f"z = {z_stat:.3f}  p = {p_z:.4e}")
print(f"  OR per 1-unit z_align (XY=0 to Z=1): {or_per_unit:.4f}  "
      f"95% CI [{or_lo:.4f}, {or_hi:.4f}]")
print(f"  n_edges = {len(df)}  n_neurons (clusters) = {n_clusters}  "
      f"brain = {brain_id}")

# --- Side-by-side --------------------------------------------------------
print("\n" + "=" * 72)
print("SIDE-BY-SIDE: original vs corrected")
print("=" * 72)
print(f"  Original (20k subsample, BayesGLMM): coef = -0.0661, p = 0.05823, OR = 0.8029")
print(f"  Corrected (full {len(df)} edges, cluster-robust GEE): coef = "
      f"{beta_z_std:.4f}, p = {p_z:.4e}, OR = {or_per_unit:.4f} "
      f"[{or_lo:.4f}, {or_hi:.4f}]")
if p_z < 0.05 and or_per_unit > 1.05:
    verdict = "Z-alignment INCREASES error odds (sig)"
elif p_z < 0.05 and or_per_unit < 0.95:
    verdict = "Z-alignment DECREASES error odds (sig)"
else:
    verdict = "no significant Z-alignment effect (cluster-robust)"
print(f"  Corrected qualitative call: {verdict}")
