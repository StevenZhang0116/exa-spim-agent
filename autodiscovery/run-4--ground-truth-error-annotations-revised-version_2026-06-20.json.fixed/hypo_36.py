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
# CORRECTED ANALYSIS (entry #8, id 36) ==================================
# Original test: one-sided Mann-Whitney U on per-NODE distance to nearest
#   merge site, n=49,295 omit vs n=49,295 length-matched correct.
# Recorded: medians 1812.73 vs 1959.88 um, U=1.138e9, p=1.60e-66.
#
# Why the original is wrong:
#   - The ~49k nodes are heavily spatially correlated along cable; treating
#     them as i.i.d. enormously inflates the effective n. The colossal p is
#     a sample-size artefact, not an evidence quantity.
#   - The reference set is only 67 merge sites; the test should also account
#     for variability in WHICH sites anchor the comparison.
#
# Corrected tests:
#   (a) Cluster-bootstrap by NEURON: resample neurons with replacement, recompute
#       median(distance_omit) - median(distance_correct) for each bootstrap
#       sample, report 95% CI on the median gap; pivot p-value vs zero.
#   (b) Cliff's delta on the same distance comparison with cluster-bootstrap CI.
#   (c) Merge-site bootstrap: resample the 67 merge sites with replacement and
#       recompute the median gap to bound merge-site variability.
# =======================================================================
import pickle
import sys
import numpy as np
from scipy.spatial import KDTree
from scipy.stats import mannwhitneyu

print("=" * 72)
print("HYPO 36 — omit nodes closer to merges than correct (corrected)")
print("=" * 72)

with open(_PKL, "rb") as f:
    payload = pickle.load(f)
gt = payload["gt_graph"]
edge_error = np.asarray(payload["gt_edge_error"])
merge_sites = payload["gt_merge_sites"]
merge_xyz = np.array([s["xyz"] for s in merge_sites])
if len(merge_xyz) == 0:
    print("No merge sites; aborting.")
    sys.exit(0)

omit_nodes_set = set()
correct_nodes_set = set()
gt_edges = list(gt.edges)
for i, (u, v) in enumerate(gt_edges):
    err = edge_error[i]
    if err == 2:
        omit_nodes_set.add(u); omit_nodes_set.add(v)
    elif err == 0:
        correct_nodes_set.add(u); correct_nodes_set.add(v)
correct_nodes_set -= omit_nodes_set
omit_nodes = list(omit_nodes_set)
correct_nodes = list(correct_nodes_set)
omit_xyz = np.array([gt.node_xyz[n] for n in omit_nodes])
correct_xyz = np.array([gt.node_xyz[n] for n in correct_nodes])

# Per-node neuron id (cluster)
try:
    omit_neuron = np.array([gt.node_segment_id(int(n)) for n in omit_nodes])
    correct_neuron = np.array([gt.node_segment_id(int(n)) for n in correct_nodes])
except Exception:
    omit_neuron = np.array([str(n) for n in omit_nodes])
    correct_neuron = np.array([str(n) for n in correct_nodes])
# Common neuron coding
all_neuron = np.concatenate([omit_neuron, correct_neuron])
_, neuron_int_all = np.unique(all_neuron, return_inverse=True)
omit_neuron_int = neuron_int_all[:len(omit_neuron)]
correct_neuron_int = neuron_int_all[len(omit_neuron):]

# Distances to nearest merge
merge_tree = KDTree(merge_xyz)
d_omit_all, _ = merge_tree.query(omit_xyz)

# Match correct sample to omit count (same RNG seed as original)
rng = np.random.default_rng(42)
if len(correct_xyz) > len(omit_xyz):
    idx = rng.choice(len(correct_xyz), size=len(omit_xyz), replace=False)
    correct_xyz_s = correct_xyz[idx]
    correct_neuron_s = correct_neuron_int[idx]
else:
    correct_xyz_s = correct_xyz
    correct_neuron_s = correct_neuron_int
d_correct_s, _ = merge_tree.query(correct_xyz_s)

# --- ORIGINAL recomputed for reference ---------------------------------
print(f"\n[ORIGINAL — recorded one-sided Mann-Whitney U]")
print(f"  recorded: medians 1812.73 vs 1959.88 um, U=1.138e9, p=1.6036e-66 (n=49295 each)")
median_o = np.median(d_omit_all); median_c = np.median(d_correct_s)
u_orig, p_orig = mannwhitneyu(d_omit_all, d_correct_s, alternative="less")
print(f"  recomputed (this brain, n_omit={len(d_omit_all)}, "
      f"n_correct={len(d_correct_s)}):")
print(f"    medians {median_o:.2f} vs {median_c:.2f} um, "
      f"U = {u_orig:.0f}, p = {p_orig:.4e}")

# --- (a) Cluster-bootstrap by neuron on the median gap -----------------
print("\n[CORRECTED (a)] Cluster-bootstrap by NEURON on median gap")
unique_neurons = np.unique(np.concatenate([omit_neuron_int, correct_neuron_s]))
print(f"  n_neurons (clusters) = {len(unique_neurons)}")
n_boot = 1000
boots_gap = np.empty(n_boot)
omit_by_n = {n: d_omit_all[omit_neuron_int == n] for n in unique_neurons}
correct_by_n = {n: d_correct_s[correct_neuron_s == n] for n in unique_neurons}
for b in range(n_boot):
    sel = rng.choice(unique_neurons, size=len(unique_neurons), replace=True)
    do = np.concatenate([omit_by_n[n] for n in sel if len(omit_by_n[n]) > 0]) if any(len(omit_by_n[n]) for n in sel) else np.array([])
    dc = np.concatenate([correct_by_n[n] for n in sel if len(correct_by_n[n]) > 0]) if any(len(correct_by_n[n]) for n in sel) else np.array([])
    if len(do) == 0 or len(dc) == 0:
        boots_gap[b] = 0.0
    else:
        boots_gap[b] = np.median(do) - np.median(dc)
observed_gap = median_o - median_c
ci_lo, ci_hi = np.quantile(boots_gap, [0.025, 0.975])
# Cluster-bootstrap two-sided p: fraction of boots where the SIGN flips
p_cb = float(min((boots_gap >= 0).sum(), (boots_gap <= 0).sum())) / n_boot * 2
print(f"  observed median(omit) - median(correct) = {observed_gap:.2f} um")
print(f"  cluster-bootstrap 95% CI on the median gap: "
      f"[{ci_lo:.2f}, {ci_hi:.2f}] um  (n_boot={n_boot})")
print(f"  cluster-bootstrap two-sided p (sign-flip): {p_cb:.4f}")

# --- (b) Cliff's delta with cluster-bootstrap CI -----------------------
def cliffs_delta_fast(a, b, n_max=5000, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    a = np.asarray(a); b = np.asarray(b)
    # Subsample for tractability if very large
    if len(a) > n_max:
        a = a[rng.choice(len(a), n_max, replace=False)]
    if len(b) > n_max:
        b = b[rng.choice(len(b), n_max, replace=False)]
    diff = a[:, None] - b[None, :]
    return float((np.sign(diff)).mean())

print("\n[CORRECTED (b)] Cliff's delta (omit - correct) + cluster-bootstrap CI")
rng2 = np.random.default_rng(123)
d0 = cliffs_delta_fast(d_omit_all, d_correct_s, n_max=3000, rng=rng2)
print(f"  observed Cliff's delta = {d0:+.4f}  (<0 means omit distances are SMALLER)")
boots_d = []
for _ in range(300):
    sel = rng2.choice(unique_neurons, size=len(unique_neurons), replace=True)
    do = np.concatenate([omit_by_n[n] for n in sel if len(omit_by_n[n]) > 0]) if any(len(omit_by_n[n]) for n in sel) else np.array([])
    dc = np.concatenate([correct_by_n[n] for n in sel if len(correct_by_n[n]) > 0]) if any(len(correct_by_n[n]) for n in sel) else np.array([])
    if len(do) > 0 and len(dc) > 0:
        boots_d.append(cliffs_delta_fast(do, dc, n_max=2000, rng=rng2))
if boots_d:
    lo_d, hi_d = np.quantile(boots_d, [0.025, 0.975])
    print(f"  cluster-bootstrap 95% CI on Cliff's delta: [{lo_d:+.4f}, {hi_d:+.4f}]  "
          f"(n_boot={len(boots_d)})")
else:
    print("  cluster bootstrap could not compute CI")

# --- (c) Merge-site bootstrap ---------------------------------------------
print("\n[CORRECTED (c)] Merge-site bootstrap (resample the 67 merge sites)")
n_sites = len(merge_xyz)
print(f"  n_merge_sites = {n_sites}")
gaps_ms = []
for _ in range(500):
    sel = rng2.choice(n_sites, size=n_sites, replace=True)
    msubset = merge_xyz[sel]
    tree = KDTree(msubset)
    do, _ = tree.query(omit_xyz)
    dc, _ = tree.query(correct_xyz_s)
    gaps_ms.append(np.median(do) - np.median(dc))
gaps_ms = np.array(gaps_ms)
lo_ms, hi_ms = np.quantile(gaps_ms, [0.025, 0.975])
print(f"  merge-site bootstrap 95% CI on median gap: "
      f"[{lo_ms:.2f}, {hi_ms:.2f}] um  (n_boot=500)")

print("\n" + "=" * 72)
print("SIDE-BY-SIDE: original vs corrected")
print("=" * 72)
print(f"  Original:    medians 1812.73 vs 1959.88 um, U=1.138e9, p=1.6036e-66 "
      f"(assumes 49k i.i.d. nodes)")
print(f"  Corrected:   observed gap = {observed_gap:.2f} um")
print(f"               cluster-bootstrap (by neuron) 95% CI: "
      f"[{ci_lo:.2f}, {ci_hi:.2f}], cluster-p = {p_cb:.4f}")
print(f"               merge-site bootstrap 95% CI on gap: "
      f"[{lo_ms:.2f}, {hi_ms:.2f}]")
