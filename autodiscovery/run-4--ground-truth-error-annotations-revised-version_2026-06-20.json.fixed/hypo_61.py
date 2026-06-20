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
# CORRECTED ANALYSIS (entry #15, id 61) =================================
# Original test: one-sided Mann-Whitney U + Welch's t on per-edge distance
#   to nearest merge site, n=6,805 split vs 1,109,034 correct.
# Recorded: medians 1794.64 vs 1956.93 um; U=3.43e9, p=3.50e-38;
#   t=-10.71, p=7.14e-27.
#
# Why the original is wrong: edges along the same cable share distance to
# any given anchor merge site; treating each of 1.1M edges as i.i.d.
# inflates the evidence. Also only 67 merge anchors.
#
# Corrected:
#   (a) cluster-bootstrap by NEURON: 95% CI on median(split - correct) gap
#       and Cliff's delta + cluster-bootstrap p (sign-flip);
#   (b) merge-site bootstrap: resample the 67 merge sites to bound anchor
#       variability.
# =======================================================================
import pickle
import sys
import gc
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import mannwhitneyu, ttest_ind

print("=" * 72)
print("HYPO 61 — split distance to merges < correct (corrected)")
print("=" * 72)

with open(_PKL, "rb") as f:
    payload = pickle.load(f)
gt = payload["gt_graph"]
edge_error = np.asarray(payload["gt_edge_error"])
merge_sites = payload.get("gt_merge_sites", [])
del payload
gc.collect()

if not merge_sites:
    print("No merge sites; aborting."); sys.exit(0)
merge_xyz = np.array([s["xyz"] for s in merge_sites])

edges = list(gt.edges)
u_arr = np.array([e[0] for e in edges]); v_arr = np.array([e[1] for e in edges])
try:
    u_xyz = gt.node_xyz[u_arr]
    v_xyz = gt.node_xyz[v_arr]
except Exception:
    u_xyz = np.array([gt.node_xyz[n] for n in u_arr])
    v_xyz = np.array([gt.node_xyz[n] for n in v_arr])
midpoints = (u_xyz + v_xyz) / 2.0
try:
    edge_neuron = np.array([gt.node_segment_id(int(u)) for u in u_arr])
except Exception:
    edge_neuron = np.array([str(u) for u in u_arr])
_, edge_neuron_int = np.unique(edge_neuron, return_inverse=True)

merge_tree = cKDTree(merge_xyz)
distances, _ = merge_tree.query(midpoints)
EDGE_CORRECT, EDGE_SPLIT = 0, 1
split_mask = (edge_error == EDGE_SPLIT)
correct_mask = (edge_error == EDGE_CORRECT)
split_d = distances[split_mask]; correct_d = distances[correct_mask]
split_n = edge_neuron_int[split_mask]; correct_n = edge_neuron_int[correct_mask]

print(f"\nn_split={len(split_d)}, n_correct={len(correct_d)}")
print(f"  median split = {np.median(split_d):.2f} um, median correct = {np.median(correct_d):.2f} um")

# --- ORIGINAL for reference ---------------------------------------------
print("\n[ORIGINAL — recorded one-sided Mann-Whitney + Welch's t]")
print(f"  recorded: U=3.4327e9, p=3.50e-38; t=-10.71, p=7.14e-27")
u_o, p_o = mannwhitneyu(split_d, correct_d, alternative="less")
t_o, pt_o = ttest_ind(split_d, correct_d, equal_var=False, alternative="less")
print(f"  recomputed: U = {u_o:.0f}, p = {p_o:.4e}; "
      f"t = {t_o:.4f}, p = {pt_o:.4e}")

# --- (a) Cluster-bootstrap by neuron -----------------------------------
print("\n[CORRECTED (a)] Cluster-bootstrap (by neuron) on median gap & Cliff's delta")
unique_n = np.unique(np.concatenate([split_n, correct_n]))
print(f"  n_neurons (clusters) = {len(unique_n)}")
split_by_n = {n: split_d[split_n == n] for n in unique_n}
correct_by_n = {n: correct_d[correct_n == n] for n in unique_n}
obs_gap = float(np.median(split_d) - np.median(correct_d))

def cliffs_delta_fast(a, b, n_max=3000, rng=None):
    rng = rng or np.random.default_rng(0)
    a = np.asarray(a); b = np.asarray(b)
    if len(a) == 0 or len(b) == 0: return float("nan")
    if len(a) > n_max: a = a[rng.choice(len(a), n_max, replace=False)]
    if len(b) > n_max: b = b[rng.choice(len(b), n_max, replace=False)]
    return float(np.sign(a[:, None] - b[None, :]).mean())

rng = np.random.default_rng(17)
obs_d = cliffs_delta_fast(split_d, correct_d, n_max=3000, rng=rng)
print(f"  observed median gap (split - correct) = {obs_gap:.2f} um")
print(f"  observed Cliff's delta = {obs_d:+.4f}  (<0 means split distances are SMALLER)")
boots_g = []; boots_d = []
for _ in range(300):
    sel = rng.choice(unique_n, size=len(unique_n), replace=True)
    sl = [split_by_n[n] for n in sel if len(split_by_n[n]) > 0]
    cl = [correct_by_n[n] for n in sel if len(correct_by_n[n]) > 0]
    if not sl or not cl:
        continue
    s = np.concatenate(sl); c = np.concatenate(cl)
    boots_g.append(float(np.median(s) - np.median(c)))
    boots_d.append(cliffs_delta_fast(s, c, n_max=2000, rng=rng))
if boots_g:
    lo_g, hi_g = np.quantile(boots_g, [0.025, 0.975])
    lo_d, hi_d = np.quantile(boots_d, [0.025, 0.975])
    p_d = 2 * min((np.asarray(boots_d) > 0).mean(), (np.asarray(boots_d) < 0).mean())
    p_g = 2 * min((np.asarray(boots_g) > 0).mean(), (np.asarray(boots_g) < 0).mean())
    print(f"  cluster-bootstrap 95% CI on median gap: [{lo_g:.2f}, {hi_g:.2f}] um  "
          f"(two-sided p = {p_g:.4f})")
    print(f"  cluster-bootstrap 95% CI on Cliff's delta: [{lo_d:+.4f}, {hi_d:+.4f}]  "
          f"(two-sided p = {p_d:.4f})")

# --- (b) Merge-site bootstrap ------------------------------------------
print("\n[CORRECTED (b)] Merge-site bootstrap (resample the merge anchors)")
n_sites = len(merge_xyz)
print(f"  n_merge_sites = {n_sites}")
gaps_ms = []
for _ in range(300):
    sel = rng.choice(n_sites, size=n_sites, replace=True)
    tree = cKDTree(merge_xyz[sel])
    do, _ = tree.query((u_xyz[split_mask] + v_xyz[split_mask]) / 2.0)
    dc, _ = tree.query((u_xyz[correct_mask] + v_xyz[correct_mask]) / 2.0)
    gaps_ms.append(float(np.median(do) - np.median(dc)))
gaps_ms = np.array(gaps_ms)
lo_ms, hi_ms = np.quantile(gaps_ms, [0.025, 0.975])
print(f"  merge-site bootstrap 95% CI on median gap: "
      f"[{lo_ms:.2f}, {hi_ms:.2f}] um")

print("\n" + "=" * 72)
print("SIDE-BY-SIDE: original vs corrected")
print("=" * 72)
print(f"  Original (i.i.d. edges):       U=3.43e9, p=3.5e-38; t=-10.71, p=7.1e-27")
print(f"  Corrected gap CI (cluster):    [{lo_g:.2f}, {hi_g:.2f}] um (observed {obs_gap:.2f})")
print(f"  Corrected Cliff's delta CI:    [{lo_d:+.4f}, {hi_d:+.4f}] (observed {obs_d:+.4f})")
print(f"  Corrected merge-site gap CI:   [{lo_ms:.2f}, {hi_ms:.2f}] um")
