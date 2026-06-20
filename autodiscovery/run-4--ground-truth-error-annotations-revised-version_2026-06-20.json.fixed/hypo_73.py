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
# CORRECTED ANALYSIS (entry #18, id 73) =================================
# Original test: one-sided Mann-Whitney U on per-edge 5-hop tortuosity,
#   n=6,805 split vs 1,109,034 correct.
# Recorded: medians 1.1132 vs 1.0801; U=4.71e9, p=1.66e-276.
#
# Why the original is wrong: 5-hop tortuosity windows for consecutive edges
# overlap by 4 edges; the values are strongly autocorrelated, violating the
# Mann-Whitney i.i.d. assumption even more strongly than the 10-hop variant
# in #14 (id 59).
#
# Corrected:
#   (a) Cluster-bootstrap by NEURON on median gap and Cliff's delta;
#   (b) Per-neuron paired Wilcoxon signed-rank on neuron-level median
#       tortuosity (split vs correct).
# =======================================================================
import pickle
import sys
import gc
import math
from pathlib import Path
import numpy as np
from scipy.stats import mannwhitneyu, wilcoxon

print("=" * 72)
print("HYPO 73 — 5-hop tortuosity replication (corrected)")
print("=" * 72)

def get_extremity(adj, start_node, prev_node, k, node_xyz):
    curr = start_node; prev = prev_node; path_len = 0.0
    for _ in range(k):
        neighbors = [n for n in adj[curr] if n != prev]
        if not neighbors: break
        nxt = neighbors[0]
        dx = node_xyz[nxt][0] - node_xyz[curr][0]
        dy = node_xyz[nxt][1] - node_xyz[curr][1]
        dz = node_xyz[nxt][2] - node_xyz[curr][2]
        path_len += math.sqrt(dx*dx + dy*dy + dz*dz)
        prev = curr; curr = nxt
    return curr, path_len

with open(_PKL, "rb") as f:
    payload = pickle.load(f)
gt = payload["gt_graph"]
edge_error = np.asarray(payload["gt_edge_error"])
edges = list(gt.edges)
node_xyz = gt.node_xyz
adj = {n: list(gt.neighbors(n)) for n in gt.nodes}

try:
    edge_neuron = np.array([gt.node_segment_id(int(u)) for (u, v) in edges])
except Exception:
    edge_neuron = np.array([str(u) for (u, v) in edges])
_, edge_neuron_int = np.unique(edge_neuron, return_inverse=True)

correct_t = []; split_t = []
correct_n = []; split_n = []
k = 5
for idx, (u, v) in enumerate(edges):
    err = int(edge_error[idx])
    if err != 0 and err != 1:
        continue
    u_curr, u_len = get_extremity(adj, u, v, k, node_xyz)
    v_curr, v_len = get_extremity(adj, v, u, k, node_xyz)
    dx = node_xyz[v][0] - node_xyz[u][0]
    dy = node_xyz[v][1] - node_xyz[u][1]
    dz = node_xyz[v][2] - node_xyz[u][2]
    edge_len = math.sqrt(dx*dx + dy*dy + dz*dz)
    path_length = u_len + v_len + edge_len
    dx = node_xyz[v_curr][0] - node_xyz[u_curr][0]
    dy = node_xyz[v_curr][1] - node_xyz[u_curr][1]
    dz = node_xyz[v_curr][2] - node_xyz[u_curr][2]
    euclid = math.sqrt(dx*dx + dy*dy + dz*dz)
    tort = path_length / euclid if euclid > 1e-6 else 1.0
    if err == 0:
        correct_t.append(tort); correct_n.append(int(edge_neuron_int[idx]))
    else:
        split_t.append(tort); split_n.append(int(edge_neuron_int[idx]))

del payload, gt, edge_error
gc.collect()
correct_t = np.array(correct_t); split_t = np.array(split_t)
correct_n = np.array(correct_n); split_n = np.array(split_n)

print(f"\ncorrect: {len(correct_t)}, split: {len(split_t)}")
print(f"  median correct = {np.median(correct_t):.6f}, median split = {np.median(split_t):.6f}")

# --- ORIGINAL recomputed -----------------------------------------------
print("\n[ORIGINAL — recorded one-sided Mann-Whitney U]")
print(f"  recorded: U=4.71e9, p=1.66e-276")
u_o, p_o = mannwhitneyu(split_t, correct_t, alternative="greater")
print(f"  recomputed: U = {u_o:.0f}, p = {p_o:.4e}")

# --- (a) Cluster-bootstrap by neuron ----------------------------------
print("\n[CORRECTED (a)] Cluster-bootstrap (by neuron) on median gap & Cliff's delta")
unique_n = np.unique(np.concatenate([split_n, correct_n]))
print(f"  n_neurons (clusters) = {len(unique_n)}")
split_by_n = {n: split_t[split_n == n] for n in unique_n}
correct_by_n = {n: correct_t[correct_n == n] for n in unique_n}

def cliffs_delta_fast(a, b, n_max=3000, rng=None):
    rng = rng or np.random.default_rng(0)
    a = np.asarray(a); b = np.asarray(b)
    if len(a) == 0 or len(b) == 0: return float("nan")
    if len(a) > n_max: a = a[rng.choice(len(a), n_max, replace=False)]
    if len(b) > n_max: b = b[rng.choice(len(b), n_max, replace=False)]
    return float(np.sign(a[:, None] - b[None, :]).mean())

rng = np.random.default_rng(23)
obs_gap = float(np.median(split_t) - np.median(correct_t))
obs_d = cliffs_delta_fast(split_t, correct_t, n_max=3000, rng=rng)
print(f"  observed median gap = {obs_gap:+.6f}")
print(f"  observed Cliff's delta = {obs_d:+.4f}")
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
    print(f"  cluster-bootstrap 95% CI on median gap: [{lo_g:+.6f}, {hi_g:+.6f}]  "
          f"p = {p_g:.4f}")
    print(f"  cluster-bootstrap 95% CI on Cliff's delta: [{lo_d:+.4f}, {hi_d:+.4f}]  "
          f"p = {p_d:.4f}")
else:
    lo_g = hi_g = lo_d = hi_d = float("nan"); p_g = p_d = float("nan")

# --- (b) Per-neuron paired Wilcoxon ------------------------------------
print("\n[CORRECTED (b)] Per-neuron paired Wilcoxon on neuron-level median tortuosity")
pairs = []
for n in unique_n:
    sn = split_by_n[n]; cn = correct_by_n[n]
    if len(sn) > 0 and len(cn) > 0:
        pairs.append((float(np.median(sn)), float(np.median(cn))))
if pairs:
    arr = np.array(pairs); diffs = arr[:, 0] - arr[:, 1]
    diffs_nz = diffs[diffs != 0]
    if len(diffs_nz) > 0:
        wstat, wp = wilcoxon(diffs_nz, alternative="greater")
        rng2 = np.random.default_rng(43)
        boots = [np.median(diffs_nz[rng2.integers(0, len(diffs_nz), len(diffs_nz))])
                 for _ in range(2000)]
        ci_lo, ci_hi = np.quantile(boots, [0.025, 0.975])
        print(f"  n_pairs = {len(diffs_nz)}, median(split-correct) per neuron = "
              f"{np.median(diffs_nz):+.5f}")
        print(f"  Wilcoxon (one-sided, split>correct): W = {wstat:.2f}, p = {wp:.4e}")
        print(f"  bootstrap 95% CI on median(split-correct) per neuron: "
              f"[{ci_lo:+.5f}, {ci_hi:+.5f}]")
    else:
        print("  No nonzero pairs.")
else:
    print("  No paired neurons.")

print("\n" + "=" * 72)
print("SIDE-BY-SIDE: original vs corrected")
print("=" * 72)
print(f"  Original (i.i.d. edges): U=4.71e9, p=1.66e-276; median gap ~0.033")
print(f"  Corrected cluster-bootstrap on gap: [{lo_g:+.5f}, {hi_g:+.5f}], p = {p_g:.4f}")
print(f"  Corrected Cliff's delta: [{lo_d:+.4f}, {hi_d:+.4f}], p = {p_d:.4f}")
