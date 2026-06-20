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
# CORRECTED ANALYSIS (entry #4, id 139) ==================================
# Original test: plain logistic regression of is_split ~ branch_order on
#   ~1.4M edges treated as i.i.d.; recorded coef = +0.0194, z = 16.37,
#   p < 0.001.
#
# Why the original is wrong: edges within the same neuron share all of their
# branch structure; treating them as i.i.d. inflates evidence by orders of
# magnitude. The "deeper-order branches are more split-prone" claim is also
# a per-neuron biological claim, so the cluster must be the neuron.
#
# Corrected test: same logistic regression of is_split on branch_order but
# fit via GEE with neuron as the cluster id — cluster-robust standard errors
# that account for within-neuron non-independence. Reports:
#   * OR per unit branch_order with 95% CI
#   * GEE p-value
#   * a neuron-level Spearman correlation between neuron branch-order range
#     and its per-neuron split rate (auxiliary cluster-level confirmation)
# Compared side-by-side with the original p<0.001 / coef=0.0194.
# =======================================================================
import pickle
import sys
import gc
import os
import re
from collections import defaultdict
import numpy as np
import networkx as nx
import pandas as pd
from scipy.stats import norm, spearmanr

print("=" * 72)
print("HYPO 139 — branch_order predicts splits (corrected)")
print("=" * 72)

with open(_PKL, "rb") as f:
    payload = pickle.load(f)
gt = payload["gt_graph"]
edge_error = np.asarray(payload["gt_edge_error"])
edges_list = list(gt.edges)
brain_id = os.path.basename(_PKL)
m = re.search(r"dataset_cache_(\d+)_mcl(\d+)_add\.pkl$", brain_id)
brain_id = m.group(1) if m else brain_id

def get_radius(gt, n):
    if hasattr(gt, "node_radius") and gt.node_radius is not None:
        try:
            return float(gt.node_radius[n])
        except Exception:
            pass
    return float(gt.nodes[n].get("node_radius", 1.0))

# --- identical edge classification to original --------------------------
EDGE_SPLIT = 1
edge_error_map = {}
for idx, e in enumerate(edges_list):
    u, v = e
    edge_error_map[(u, v)] = edge_error[idx]
    edge_error_map[(v, u)] = edge_error[idx]

neurons = defaultdict(list)
for n in gt.nodes():
    neurons[gt.node_segment_id(n)].append(n)

rows = []
for neuron_id, nodes in neurons.items():
    subgraph = gt.subgraph(nodes).copy()
    if len(subgraph) == 0:
        continue
    radii = {n: get_radius(gt, n) for n in subgraph.nodes()}
    if not radii:
        continue
    soma = max(radii, key=radii.get)
    if not nx.is_connected(subgraph):
        comp = nx.node_connected_component(subgraph, soma)
        subgraph = subgraph.subgraph(comp).copy()
    branch_orders = {soma: 0}
    visited = {soma}
    queue = [soma]
    while queue:
        curr = queue.pop(0)
        curr_order = branch_orders[curr]
        unvisited_neighbors = [n for n in subgraph.neighbors(curr) if n not in visited]
        next_order = curr_order + 1 if len(unvisited_neighbors) > 1 else curr_order
        for child in unvisited_neighbors:
            visited.add(child)
            branch_orders[child] = next_order
            queue.append(child)
    for u, v in subgraph.edges():
        order = max(branch_orders.get(u, 0), branch_orders.get(v, 0))
        is_split = 1 if edge_error_map.get((u, v), 0) == EDGE_SPLIT else 0
        rows.append((neuron_id, order, is_split))

del payload, gt
gc.collect()

df = pd.DataFrame(rows, columns=["neuron", "branch_order", "is_split"])
print(f"\nTotal edges processed: {len(df)}")
_, neuron_int = np.unique(df["neuron"].astype(str).to_numpy(), return_inverse=True)
df["neuron_int"] = neuron_int

# --- ORIGINAL plain logistic regression for reference -------------------
print("\n[ORIGINAL — plain logistic (i.i.d. edges), recorded]")
print(f"  coef(branch_order) = +0.0194, z = 16.37, p < 0.001  "
      f"(n = {len(df)} edges treated as i.i.d.)")
# Also recompute it for an exact side-by-side
import statsmodels.api as sm
X_plain = sm.add_constant(df["branch_order"].to_numpy(dtype=float))
y_plain = df["is_split"].to_numpy(dtype=float)
try:
    plain = sm.GLM(y_plain, X_plain, family=sm.families.Binomial()).fit()
    print(f"  recomputed plain GLM: coef = {plain.params[1]:+.4f}, "
          f"SE = {plain.bse[1]:.4f}, z = {plain.params[1]/plain.bse[1]:.3f}, "
          f"p = {plain.pvalues[1]:.4e}")
except Exception as e:
    print(f"  plain GLM recompute failed: {e}")

# --- CORRECTED: GEE with neuron cluster ---------------------------------
print("\n[CORRECTED] GEE logistic with neuron as cluster id (robust SE)")
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.cov_struct import Independence
X = sm.add_constant(df["branch_order"].to_numpy(dtype=float))
y = df["is_split"].to_numpy(dtype=float)
groups = df["neuron_int"].to_numpy()
gee = GEE(y, X, groups=groups, family=Binomial(), cov_struct=Independence())
res = gee.fit()
beta = float(res.params[1])
se = float(res.bse[1])
z_stat = beta / se if se > 0 else float("nan")
p_g = 2 * (1 - norm.cdf(abs(z_stat)))
or_per = float(np.exp(beta))
or_lo = float(np.exp(beta - 1.96 * se))
or_hi = float(np.exp(beta + 1.96 * se))
print(f"  coef(branch_order) = {beta:+.5f}  SE = {se:.5f}  z = {z_stat:.3f}  "
      f"p = {p_g:.4e}")
print(f"  OR per +1 branch_order = {or_per:.4f}  95% CI [{or_lo:.4f}, {or_hi:.4f}]")
print(f"  n_edges = {len(df)}  n_neurons (clusters) = {int(df['neuron_int'].nunique())}  "
      f"brain = {brain_id}")

# --- neuron-level Spearman between mean branch_order and split rate -----
print("\n[CORRECTED — auxiliary] neuron-level Spearman: mean branch_order vs split rate")
agg = df.groupby("neuron_int").agg(
    mean_order=("branch_order", "mean"),
    max_order=("branch_order", "max"),
    split_rate=("is_split", "mean"),
    n_edges=("is_split", "count"),
).reset_index()
print(f"  n_neurons used = {len(agg)}")
if len(agg) >= 4:
    rho, p_rho = spearmanr(agg["mean_order"], agg["split_rate"])
    rho_max, p_rho_max = spearmanr(agg["max_order"], agg["split_rate"])
    print(f"  Spearman rho(mean_order, split_rate) = {rho:+.4f}, p = {p_rho:.4e}")
    print(f"  Spearman rho(max_order, split_rate)  = {rho_max:+.4f}, p = {p_rho_max:.4e}")
else:
    print("  Too few neurons for Spearman.")

# --- Side-by-side --------------------------------------------------------
print("\n" + "=" * 72)
print("SIDE-BY-SIDE: original vs corrected")
print("=" * 72)
print(f"  Original (recorded plain logreg, i.i.d.): coef = +0.0194, z = 16.37, p < 0.001")
print(f"  Corrected (GEE, neuron-clustered SE):     coef = {beta:+.5f}, "
      f"z = {z_stat:.3f}, p = {p_g:.4e}, OR = {or_per:.4f} [{or_lo:.4f}, {or_hi:.4f}]")
if p_g < 0.05 and or_per > 1.0:
    qcall = "deeper branches MORE split-prone (sig)"
elif p_g < 0.05 and or_per < 1.0:
    qcall = "deeper branches LESS split-prone (sig)"
else:
    qcall = "no significant branch_order effect once neurons are clustered"
print(f"  Corrected qualitative call: {qcall}")
