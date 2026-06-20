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
# CORRECTED ANALYSIS (entry #12, id 45) ==================================
# Original test: Cochran-Mantel-Haenszel "controlling for brain ID",
#   recorded pooled OR = 2.3561, p ≈ 0 (n = 637,278 nodes).
#
# Why the original is wrong/weak:
#   - On a single-brain pkl the "stratification by brain" is a no-op; the
#     pooled OR collapses to a plain OR and the p≈0 ignores within-neuron
#     non-independence.
#   - Effective n is much smaller than 637k because nodes within a neuron
#     share spatial structure.
#
# Corrected test: GEE logistic with neuron as cluster id on the same
# Extreme-vs-Center node sample; reports OR (Extreme vs Center) + 95% CI
# with cluster-robust SE. Also reports a cluster-bootstrap CI on the OR
# as a non-parametric robustness check.
# =======================================================================
import pickle
import os
import re
import sys
import numpy as np
import pandas as pd
from scipy.stats import norm

print("=" * 72)
print("HYPO 45 — extreme-Z vs center-Z omit rate (corrected)")
print("=" * 72)

with open(_PKL, "rb") as f:
    payload = pickle.load(f)
gt = payload["gt_graph"]
node_label = np.asarray(payload["gt_node_canonical_label"])

brain_id = os.path.basename(_PKL)
m = re.search(r"dataset_cache_(\d+)_mcl(\d+)_add\.pkl$", brain_id)
brain_id = m.group(1) if m else brain_id

nodes = list(gt.nodes)
z_coords = np.array([gt.node_xyz[n][2] for n in nodes], dtype=float)
zmin, zmax = z_coords.min(), z_coords.max()
zrange = zmax - zmin if zmax > zmin else 1.0
z_norm = (z_coords - zmin) / zrange

bin_label = np.array(["Other"] * len(nodes), dtype=object)
bin_label[(z_norm <= 0.1) | (z_norm >= 0.9)] = "Extreme"
bin_label[(z_norm >= 0.4) & (z_norm <= 0.6)] = "Center"
is_omit = np.array([1 if node_label[n] == 0 else 0 for n in nodes], dtype=int)
try:
    neuron_ids = np.array([gt.node_segment_id(int(n)) for n in nodes])
except Exception:
    neuron_ids = np.array([str(n) for n in nodes])
_, neuron_int = np.unique(neuron_ids, return_inverse=True)

df = pd.DataFrame({"bin": bin_label, "is_omit": is_omit, "neuron": neuron_int})
mask = df["bin"].isin(["Extreme", "Center"])
df_t = df[mask].copy().reset_index(drop=True)
df_t["is_extreme"] = (df_t["bin"] == "Extreme").astype(int)

n_ext = int((df_t["bin"] == "Extreme").sum())
n_ctr = int((df_t["bin"] == "Center").sum())
omit_ext = int(df_t[df_t["bin"] == "Extreme"]["is_omit"].sum())
omit_ctr = int(df_t[df_t["bin"] == "Center"]["is_omit"].sum())
rate_ext = omit_ext / max(n_ext, 1)
rate_ctr = omit_ctr / max(n_ctr, 1)
print(f"\nDescriptives:")
print(f"  Extreme bin: {omit_ext}/{n_ext} omit ({rate_ext*100:.4f}%)")
print(f"  Center bin:  {omit_ctr}/{n_ctr} omit ({rate_ctr*100:.4f}%)")

# --- ORIGINAL CMH for reference (single-brain so CMH ~ plain OR) -------
print("\n[ORIGINAL — recorded CMH pooled across brain stratum]")
print(f"  recorded: pooled OR = 2.3561, p ≈ 0  (n = 637,278; single-brain rerun makes "
      f"stratification a no-op)")
from statsmodels.stats.contingency_tables import StratifiedTable
table = np.array([[[omit_ext, n_ext - omit_ext], [omit_ctr, n_ctr - omit_ctr]]])
table = np.transpose(table, (1, 2, 0))  # (2,2,1)
st = StratifiedTable(table)
print(f"  recomputed CMH (this brain): pooled OR = {st.oddsratio_pooled:.4f}, "
      f"p = {st.test_null_odds().pvalue:.4e}")

# --- CORRECTED: GEE logistic with neuron cluster -----------------------
print("\n[CORRECTED (a)] GEE logistic with neuron cluster id (cluster-robust SE)")
import statsmodels.api as sm
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.cov_struct import Independence
X = sm.add_constant(df_t["is_extreme"].to_numpy(dtype=float))
y = df_t["is_omit"].to_numpy(dtype=float)
groups = df_t["neuron"].to_numpy()
gee = GEE(y, X, groups=groups, family=Binomial(), cov_struct=Independence())
res = gee.fit()
beta = float(res.params[1]); se = float(res.bse[1])
z_stat = beta / se if se > 0 else float("nan")
p_g = 2 * (1 - norm.cdf(abs(z_stat)))
or_g = float(np.exp(beta))
or_lo = float(np.exp(beta - 1.96 * se))
or_hi = float(np.exp(beta + 1.96 * se))
n_clusters = int(df_t["neuron"].nunique())
print(f"  beta(is_extreme) = {beta:+.4f}  SE = {se:.4f}  z = {z_stat:.3f}  p = {p_g:.4e}")
print(f"  OR(Extreme vs Center) = {or_g:.4f}  95% CI [{or_lo:.4f}, {or_hi:.4f}]")
print(f"  n_edges = {len(df_t)}  n_neurons (clusters) = {n_clusters}  "
      f"brain = {brain_id}")

# --- (b) Cluster-bootstrap (by neuron) on the OR -----------------------
print("\n[CORRECTED (b)] Cluster-bootstrap (by neuron) on the OR")
rng = np.random.default_rng(7)
unique_n = np.unique(df_t["neuron"].to_numpy())
by_neuron = {n: df_t[df_t["neuron"] == n] for n in unique_n}
n_boot = 500
boots_or = []
for _ in range(n_boot):
    sel = rng.choice(unique_n, size=len(unique_n), replace=True)
    parts = [by_neuron[n] for n in sel]
    sub = pd.concat(parts, ignore_index=True)
    a = int(((sub["bin"] == "Extreme") & (sub["is_omit"] == 1)).sum())
    b = int(((sub["bin"] == "Extreme") & (sub["is_omit"] == 0)).sum())
    c = int(((sub["bin"] == "Center") & (sub["is_omit"] == 1)).sum())
    d = int(((sub["bin"] == "Center") & (sub["is_omit"] == 0)).sum())
    if a == 0 or b == 0 or c == 0 or d == 0:
        continue
    or_b = (a / b) / (c / d)
    boots_or.append(or_b)
if boots_or:
    lo, hi = np.quantile(boots_or, [0.025, 0.975])
    print(f"  cluster-bootstrap 95% CI on OR: [{lo:.4f}, {hi:.4f}]  "
          f"(n_boot={len(boots_or)})")
else:
    print("  bootstrap could not estimate OR (degenerate tables)")

print("\n" + "=" * 72)
print("SIDE-BY-SIDE: original vs corrected")
print("=" * 72)
print(f"  Original CMH:    pooled OR = 2.3561, p ≈ 0   (i.i.d. nodes, single-brain)")
print(f"  Corrected GEE:   OR = {or_g:.4f}  95% CI [{or_lo:.4f}, {or_hi:.4f}], p = {p_g:.4e}")
print(f"  Corrected boot:  95% CI on OR = "
      f"[{(np.quantile(boots_or, 0.025) if boots_or else float('nan')):.4f}, "
      f"{(np.quantile(boots_or, 0.975) if boots_or else float('nan')):.4f}]")
