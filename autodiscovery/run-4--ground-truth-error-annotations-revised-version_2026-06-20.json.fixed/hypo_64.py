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
# CORRECTED ANALYSIS (entry #17, id 64) =================================
# Original test: Mann-Whitney U on per-edge geodesic distance to nearest GT
#   branch point, n=6,805 split vs 1,109,034 correct.
# Recorded: means 516.30 vs 710.59 um, U=2.98e9, p=1.32e-197.
#
# Why the original is wrong: distances of consecutive edges along the same
# cable to the same branch point are heavily autocorrelated; 1.1M edges
# treated as i.i.d. inflates the effective n. The huge p-value is
# significance-by-sample-size.
#
# Corrected:
#   (a) Cluster-bootstrap by NEURON on median(split - correct) gap and
#       Cliff's delta + cluster-bootstrap p.
#   (b) Per-neuron paired Wilcoxon on neuron-level median distance gaps.
# =======================================================================
import pickle
import sys
import gc
import os
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra
from scipy.stats import mannwhitneyu, wilcoxon

print("=" * 72)
print("HYPO 64 — split edges closer to branch points (corrected)")
print("=" * 72)

with open(_PKL, "rb") as f:
    payload = pickle.load(f)
gt = payload["gt_graph"]
edge_error = np.asarray(payload["gt_edge_error"])
payload.pop("fragments_graph", None)
del payload
gc.collect()

edges = list(gt.edges())
if not edges:
    print("No edges; aborting."); sys.exit(0)
u_arr = np.array([e[0] for e in edges])
v_arr = np.array([e[1] for e in edges])
weights = np.linalg.norm(gt.node_xyz[u_arr] - gt.node_xyz[v_arr], axis=1)
N = gt.node_xyz.shape[0]
degrees = dict(gt.degree())
branch_points = [n for n, d in degrees.items() if d >= 3]
print(f"\n{len(edges)} edges, {N} max nodes, {len(branch_points)} branch points")
if not branch_points:
    print("No branch points; aborting."); sys.exit(0)
bp_arr = np.array(branch_points)
dummy_node = N
u_all = np.concatenate([u_arr, np.full(len(bp_arr), dummy_node)])
v_all = np.concatenate([v_arr, bp_arr])
w_all = np.concatenate([weights, np.zeros(len(bp_arr))])
graph_sparse = sp.coo_matrix((w_all, (u_all, v_all)), shape=(N + 1, N + 1))
dists = dijkstra(graph_sparse, directed=False, indices=dummy_node)
node_dists = dists[:N]
dist_u = node_dists[u_arr]; dist_v = node_dists[v_arr]
edge_dists = np.minimum(dist_u, dist_v)

EDGE_CORRECT, EDGE_SPLIT = 0, 1
split_mask = (edge_error == EDGE_SPLIT)
correct_mask = (edge_error == EDGE_CORRECT)
valid = np.isfinite(edge_dists)
split_d = edge_dists[split_mask & valid]
correct_d = edge_dists[correct_mask & valid]
try:
    edge_neuron = np.array([gt.node_segment_id(int(u)) for u in u_arr])
except Exception:
    edge_neuron = np.array([str(u) for u in u_arr])
_, edge_neuron_int = np.unique(edge_neuron, return_inverse=True)
split_neuron = edge_neuron_int[split_mask & valid]
correct_neuron = edge_neuron_int[correct_mask & valid]

print(f"split edges: {len(split_d)}, correct edges: {len(correct_d)}")
print(f"  mean split = {np.mean(split_d):.2f} um, mean correct = {np.mean(correct_d):.2f} um")
print(f"  median split = {np.median(split_d):.2f} um, median correct = {np.median(correct_d):.2f} um")

# --- ORIGINAL recomputed -----------------------------------------------
print("\n[ORIGINAL — recorded Mann-Whitney U]")
print(f"  recorded: U=2.98e9, p=1.32e-197")
u_o, p_o = mannwhitneyu(split_d, correct_d)
print(f"  recomputed: U = {u_o:.0f}, p = {p_o:.4e}")

# --- (a) Cluster-bootstrap by neuron -----------------------------------
print("\n[CORRECTED (a)] Cluster-bootstrap (by neuron) on median gap & Cliff's delta")
unique_n = np.unique(np.concatenate([split_neuron, correct_neuron]))
print(f"  n_neurons (clusters) = {len(unique_n)}")
split_by_n = {n: split_d[split_neuron == n] for n in unique_n}
correct_by_n = {n: correct_d[correct_neuron == n] for n in unique_n}

def cliffs_delta_fast(a, b, n_max=3000, rng=None):
    rng = rng or np.random.default_rng(0)
    a = np.asarray(a); b = np.asarray(b)
    if len(a) == 0 or len(b) == 0: return float("nan")
    if len(a) > n_max: a = a[rng.choice(len(a), n_max, replace=False)]
    if len(b) > n_max: b = b[rng.choice(len(b), n_max, replace=False)]
    return float(np.sign(a[:, None] - b[None, :]).mean())

rng = np.random.default_rng(19)
obs_gap = float(np.median(split_d) - np.median(correct_d))
obs_d = cliffs_delta_fast(split_d, correct_d, n_max=3000, rng=rng)
print(f"  observed median gap (split - correct) = {obs_gap:.2f} um")
print(f"  observed Cliff's delta = {obs_d:+.4f}  (<0 means split distances SMALLER)")
boots_g, boots_d = [], []
for _ in range(300):
    sel = rng.choice(unique_n, size=len(unique_n), replace=True)
    sl = [split_by_n[n] for n in sel if len(split_by_n[n]) > 0]
    cl = [correct_by_n[n] for n in sel if len(correct_by_n[n]) > 0]
    if not sl or not cl: continue
    s = np.concatenate(sl); c = np.concatenate(cl)
    boots_g.append(float(np.median(s) - np.median(c)))
    boots_d.append(cliffs_delta_fast(s, c, n_max=2000, rng=rng))
if boots_g:
    lo_g, hi_g = np.quantile(boots_g, [0.025, 0.975])
    lo_d, hi_d = np.quantile(boots_d, [0.025, 0.975])
    p_g = 2 * min((np.asarray(boots_g) > 0).mean(), (np.asarray(boots_g) < 0).mean())
    p_d = 2 * min((np.asarray(boots_d) > 0).mean(), (np.asarray(boots_d) < 0).mean())
    print(f"  cluster-bootstrap 95% CI on median gap: [{lo_g:.2f}, {hi_g:.2f}] um  "
          f"(two-sided p = {p_g:.4f})")
    print(f"  cluster-bootstrap 95% CI on Cliff's delta: [{lo_d:+.4f}, {hi_d:+.4f}]  "
          f"(two-sided p = {p_d:.4f})")
else:
    print("  cluster bootstrap failed")
    lo_g = hi_g = lo_d = hi_d = float("nan"); p_g = p_d = float("nan")

# --- (b) Per-neuron paired Wilcoxon ------------------------------------
print("\n[CORRECTED (b)] Per-neuron paired Wilcoxon on neuron-level median distance")
pairs = []
for n in unique_n:
    sn = split_by_n[n]; cn = correct_by_n[n]
    if len(sn) > 0 and len(cn) > 0:
        pairs.append((float(np.median(sn)), float(np.median(cn))))
if pairs:
    arr = np.array(pairs)
    diffs = arr[:, 0] - arr[:, 1]
    diffs_nz = diffs[diffs != 0]
    if len(diffs_nz) > 0:
        wstat, wp = wilcoxon(diffs_nz, alternative="less")
        rng2 = np.random.default_rng(37)
        boots = [np.median(diffs_nz[rng2.integers(0, len(diffs_nz), len(diffs_nz))])
                 for _ in range(2000)]
        ci_lo, ci_hi = np.quantile(boots, [0.025, 0.975])
        print(f"  n_pairs = {len(diffs_nz)}, median(split-correct) per neuron = "
              f"{np.median(diffs_nz):.2f} um")
        print(f"  Wilcoxon (one-sided, split<correct): W = {wstat:.2f}, p = {wp:.4e}")
        print(f"  bootstrap 95% CI on median(split-correct) per neuron: "
              f"[{ci_lo:.2f}, {ci_hi:.2f}] um")
    else:
        print("  No nonzero pairs.")
else:
    print("  No paired neurons.")

print("\n" + "=" * 72)
print("SIDE-BY-SIDE: original vs corrected")
print("=" * 72)
print(f"  Original (i.i.d. edges): U=2.98e9, p=1.32e-197  (gap means 516.30 vs 710.59 um)")
print(f"  Corrected cluster-bootstrap on gap: [{lo_g:.2f}, {hi_g:.2f}] um, p = {p_g:.4f}")
print(f"  Corrected Cliff's delta: [{lo_d:+.4f}, {hi_d:+.4f}], p = {p_d:.4f}")
