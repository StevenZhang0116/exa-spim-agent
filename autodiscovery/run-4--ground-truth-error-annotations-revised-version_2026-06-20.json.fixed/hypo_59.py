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
# CORRECTED ANALYSIS (entry #14, id 59) =================================
# Original test: Mann-Whitney U + point-biserial correlation on per-edge
#   tortuosity with 10-hop sliding window, n=6,611 split vs 1,091,075 correct.
# Recorded: medians 1.1115 vs 1.0764, U=4.58e9, p≈0; r=0.0335, p≈0.
#
# Why the original is wrong: edges within the same neuron and along the
# same cable are heavily correlated (consecutive 10-hop windows overlap by
# 9 edges); the effective n is much smaller than 1.1M. r²=0.001 means the
# effect is practically negligible.
#
# Corrected: same data, same metric, but
#   (a) cluster-bootstrap by NEURON on Cliff's delta (split vs correct) +
#       cluster-bootstrap 95% CI;
#   (b) per-neuron paired Wilcoxon signed-rank on neuron-level median
#       tortuosity (split vs correct);
#   (c) cluster-bootstrap 95% CI on the difference of medians.
# Reports the recorded p-value alongside.
# =======================================================================
import pickle
import gc
import sys
import numpy as np
from scipy.stats import wilcoxon

print("=" * 72)
print("HYPO 59 — tortuosity split vs correct (corrected)")
print("=" * 72)

with open(_PKL, "rb") as f:
    payload = pickle.load(f)
gt = payload["gt_graph"]
edge_error = np.asarray(payload["gt_edge_error"])
del payload
gc.collect()

edges_list = list(gt.edges)
node_xyz = gt.node_xyz

# Neuron id for each edge (use u's segment id)
try:
    edge_neuron = np.array([gt.node_segment_id(int(u)) for (u, v) in edges_list])
except Exception:
    edge_neuron = np.array([str(u) for (u, v) in edges_list])
_, edge_neuron_int = np.unique(edge_neuron, return_inverse=True)

# Build adjacency for 10-hop path enumeration
adj = {n: list(gt.neighbors(n)) for n in gt.nodes()}

def get_paths_of_length(G, length, adj):
    paths = []
    for start_node in G.nodes():
        stack = [(start_node, [start_node])]
        while stack:
            curr, path = stack.pop()
            if len(path) == length + 1:
                if start_node < curr:
                    paths.append(path)
                continue
            path_len = len(path)
            for neighbor in adj[curr]:
                if path_len == 1 or neighbor != path[-2]:
                    stack.append((neighbor, path + [neighbor]))
    return paths

edge_info = {}
for idx, (u, v) in enumerate(edges_list):
    fs = frozenset((u, v))
    p_u = np.array(node_xyz[u]); p_v = np.array(node_xyz[v])
    dist = np.linalg.norm(p_u - p_v)
    edge_info[fs] = {'dist': dist, 'error': int(edge_error[idx]),
                     'neuron': int(edge_neuron_int[idx])}

paths = get_paths_of_length(gt, 10, adj)
edge_tort = {fs: [] for fs in edge_info}
for p in paths:
    path_len = 0.0
    for i in range(10):
        fs = frozenset((p[i], p[i + 1]))
        if fs in edge_info:
            path_len += edge_info[fs]['dist']
    p0 = np.array(node_xyz[p[0]]); pK = np.array(node_xyz[p[-1]])
    euclid = np.linalg.norm(p0 - pK)
    tort = path_len / euclid if euclid > 1e-6 else 1.0
    e4 = frozenset((p[4], p[5]))
    e5 = frozenset((p[5], p[6]))
    if e4 in edge_tort:
        edge_tort[e4].append(tort)
    if e5 in edge_tort:
        edge_tort[e5].append(tort)

split_tort = []
correct_tort = []
split_neuron = []
correct_neuron = []
for fs, lst in edge_tort.items():
    if not lst:
        continue
    avg = np.mean(lst)
    info = edge_info[fs]
    if info['error'] == 1:
        split_tort.append(avg); split_neuron.append(info['neuron'])
    elif info['error'] == 0:
        correct_tort.append(avg); correct_neuron.append(info['neuron'])

split_tort = np.array(split_tort, dtype=float)
correct_tort = np.array(correct_tort, dtype=float)
split_neuron = np.array(split_neuron, dtype=int)
correct_neuron = np.array(correct_neuron, dtype=int)

print(f"\nSplit edges analyzed: {len(split_tort)}")
print(f"Correct edges analyzed: {len(correct_tort)}")
print(f"  median split = {np.median(split_tort):.4f}")
print(f"  median correct = {np.median(correct_tort):.4f}")
print(f"  mean split = {np.mean(split_tort):.4f}")
print(f"  mean correct = {np.mean(correct_tort):.4f}")

# --- ORIGINAL recomputed (for side-by-side) ----------------------------
print("\n[ORIGINAL — recorded Mann-Whitney + point-biserial]")
print(f"  recorded: U=4.5781e9, p=0; point-biserial r=0.0335, p=0")
from scipy.stats import mannwhitneyu, pointbiserialr
u_o, p_o = mannwhitneyu(split_tort, correct_tort, alternative="greater")
labels = np.concatenate([np.ones_like(split_tort), np.zeros_like(correct_tort)])
values = np.concatenate([split_tort, correct_tort])
r_pb, p_r = pointbiserialr(labels, values)
print(f"  recomputed: U = {u_o:.4e}, p = {p_o:.4e}; "
      f"point-biserial r = {r_pb:.4f}, p = {p_r:.4e}")

# --- (a) Cluster-bootstrap by neuron on Cliff's delta -----------------
print("\n[CORRECTED (a)] Cluster-bootstrap (by neuron) on Cliff's delta and median gap")
unique_n = np.unique(np.concatenate([split_neuron, correct_neuron]))
print(f"  n_neurons (clusters) = {len(unique_n)}")
split_by_n = {n: split_tort[split_neuron == n] for n in unique_n}
correct_by_n = {n: correct_tort[correct_neuron == n] for n in unique_n}

def cliffs_delta_fast(a, b, n_max=3000, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    a = np.asarray(a); b = np.asarray(b)
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    if len(a) > n_max:
        a = a[rng.choice(len(a), n_max, replace=False)]
    if len(b) > n_max:
        b = b[rng.choice(len(b), n_max, replace=False)]
    diff = a[:, None] - b[None, :]
    return float(np.sign(diff).mean())

rng = np.random.default_rng(11)
obs_delta = cliffs_delta_fast(split_tort, correct_tort, n_max=3000, rng=rng)
obs_gap = float(np.median(split_tort) - np.median(correct_tort))
print(f"  observed Cliff's delta (split - correct) = {obs_delta:+.4f}")
print(f"  observed median gap = {obs_gap:+.6f}")
boots_d = []; boots_g = []
for _ in range(300):
    sel = rng.choice(unique_n, size=len(unique_n), replace=True)
    s_lists = [split_by_n[n] for n in sel if len(split_by_n[n]) > 0]
    c_lists = [correct_by_n[n] for n in sel if len(correct_by_n[n]) > 0]
    if not s_lists or not c_lists:
        continue
    s = np.concatenate(s_lists); c = np.concatenate(c_lists)
    boots_d.append(cliffs_delta_fast(s, c, n_max=2000, rng=rng))
    boots_g.append(float(np.median(s) - np.median(c)))
if boots_d:
    lo_d, hi_d = np.quantile(boots_d, [0.025, 0.975])
    lo_g, hi_g = np.quantile(boots_g, [0.025, 0.975])
    print(f"  cluster-bootstrap 95% CI on Cliff's delta: "
          f"[{lo_d:+.4f}, {hi_d:+.4f}]")
    print(f"  cluster-bootstrap 95% CI on median gap:    "
          f"[{lo_g:+.6f}, {hi_g:+.6f}]")
    p_d = 2 * min((np.asarray(boots_d) > 0).mean(), (np.asarray(boots_d) < 0).mean())
    print(f"  cluster-bootstrap two-sided p (delta sign): {p_d:.4f}")
else:
    print("  cluster bootstrap failed")

# --- (b) Per-neuron paired Wilcoxon on neuron-level medians ----------
print("\n[CORRECTED (b)] Per-neuron paired Wilcoxon on neuron-level median tortuosity")
pairs = []
for n in unique_n:
    sn = split_by_n.get(n, np.array([])); cn = correct_by_n.get(n, np.array([]))
    if len(sn) > 0 and len(cn) > 0:
        pairs.append((float(np.median(sn)), float(np.median(cn))))
if pairs:
    arr = np.array(pairs)
    diffs = arr[:, 0] - arr[:, 1]
    diffs_nz = diffs[diffs != 0]
    if len(diffs_nz) > 0:
        wstat, wp = wilcoxon(diffs_nz, alternative="greater")
        rng2 = np.random.default_rng(31)
        boots = [np.median(diffs_nz[rng2.integers(0, len(diffs_nz), len(diffs_nz))])
                 for _ in range(2000)]
        ci_lo, ci_hi = np.quantile(boots, [0.025, 0.975])
        print(f"  n_pairs = {len(diffs_nz)}, median(split-correct) per neuron = "
              f"{np.median(diffs_nz):+.5f}")
        print(f"  Wilcoxon (one-sided, split>correct): W = {wstat:.2f}, "
              f"p = {wp:.4e}")
        print(f"  bootstrap 95% CI on median(split-correct) per neuron: "
              f"[{ci_lo:+.5f}, {ci_hi:+.5f}]")
    else:
        print("  All neurons have identical split/correct medians.")
else:
    print("  No paired neurons.")

print("\n" + "=" * 72)
print("SIDE-BY-SIDE: original vs corrected")
print("=" * 72)
print(f"  Original Mann-Whitney + r: U=4.58e9, p≈0; r=0.0335, p≈0  "
      f"(treats 1.1M edges as i.i.d.)")
print(f"  Corrected Cliff's delta + cluster-bootstrap CI: see above")
print(f"  Corrected per-neuron Wilcoxon: see above")
