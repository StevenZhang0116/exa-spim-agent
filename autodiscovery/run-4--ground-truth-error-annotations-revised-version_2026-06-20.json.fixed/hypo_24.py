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
# CORRECTED ANALYSIS (entry #5, id 24) ===================================
# Original test: Mann-Whitney U on TOTAL covered cable length
#   (n=24 two-neuron vs n=3 super-merges in the recorded multi-brain run).
# Recorded result: U = 0.0, p = 5.886e-03; medians 6.21 mm vs 35.19 mm.
#
# Why the original is wrong / weak:
#   1. The hypothesis is literally about cable length *per fused neuron*,
#      not total cable length — a super-merge that fuses N neurons must
#      cover at least N times their cable, so the original comparison is
#      partly tautological.
#   2. Severely underpowered: with n=3 in one arm, U=0 is a floor effect.
#      The reported "highly significant" p=0.006 is simply the smallest
#      achievable p for 24 vs 3.
#   3. No effect-size or CI is reported.
#
# Corrected tests:
#   (a) Mann-Whitney U on per-NEURON cable (total / num_fused_neurons) —
#       matches the hypothesis as literally stated.
#   (b) Cliff's delta (rank-biserial) effect size with bootstrap 95% CI
#       (resample within each group).
#   (c) Recompute the original total-cable comparison for side-by-side.
# All on the SAME data.
# =======================================================================
import pickle
import sys
import os
import re
from collections import defaultdict
import numpy as np
from scipy.stats import mannwhitneyu

print("=" * 72)
print("HYPO 24 — super-merges vs 2-neuron merges (corrected)")
print("=" * 72)

with open(_PKL, "rb") as f:
    payload = pickle.load(f)

brain_id = os.path.basename(_PKL)
m = re.search(r"dataset_cache_(\d+)_mcl(\d+)_add\.pkl$", brain_id)
brain_id = m.group(1) if m else brain_id

gt = payload["gt_graph"]
node_label = np.asarray(payload["gt_node_canonical_label"])
merge_labels = set(int(x) for x in payload["gt_merge_labels"])

if hasattr(gt, "node_xyz"):
    def get_xyz(n): return gt.node_xyz[n]
else:
    def get_xyz(n): return gt.nodes[n]["node_xyz"]

# Nodal cable contribution (half of incident edges, mm)
node_cable = defaultdict(float)
for u, v in gt.edges:
    p1 = np.array(get_xyz(u))
    p2 = np.array(get_xyz(v))
    dist = np.linalg.norm(p1 - p2) / 1000.0
    node_cable[u] += dist / 2.0
    node_cable[v] += dist / 2.0

seg_neuron_counts = defaultdict(lambda: defaultdict(int))
seg_cable = defaultdict(float)
for n in gt.nodes:
    lab = int(node_label[n])
    if lab != 0:
        neuron = gt.node_segment_id(n)
        seg_neuron_counts[lab][neuron] += 1
        seg_cable[lab] += node_cable[n]

merges_2_total = []
merges_super_total = []
merges_2_per = []
merges_super_per = []
for lab in merge_labels:
    neurons_covered = [neuron for neuron, count in seg_neuron_counts[lab].items() if count > 50]
    num_neurons = len(neurons_covered)
    if num_neurons >= 2:
        cable_len = seg_cable[lab]
        if num_neurons == 2:
            merges_2_total.append(cable_len)
            merges_2_per.append(cable_len / num_neurons)
        else:
            merges_super_total.append(cable_len)
            merges_super_per.append(cable_len / num_neurons)

def summary(name, data):
    if not data:
        return f"{name}: NO DATA"
    return (f"{name}: n={len(data)}, mean={np.mean(data):.4f} mm, "
            f"median={np.median(data):.4f} mm")

print("\nDescriptives:")
print("  " + summary("2-neuron TOTAL cable", merges_2_total))
print("  " + summary("super-merge TOTAL cable", merges_super_total))
print("  " + summary("2-neuron PER-neuron cable", merges_2_per))
print("  " + summary("super-merge PER-neuron cable", merges_super_per))

# --- (c) ORIGINAL test recomputed -------------------------------------
print("\n[ORIGINAL — recorded Mann-Whitney U on TOTAL cable]")
print(f"  recorded: medians 6.2054 mm vs 35.1873 mm, U=0.0, "
      f"p=5.886e-03 (multi-brain aggregate n=24+3)")
if merges_2_total and merges_super_total:
    u_orig, p_orig = mannwhitneyu(merges_2_total, merges_super_total,
                                  alternative="two-sided")
    print(f"  recomputed on this single brain (n={len(merges_2_total)} 2-merge, "
          f"n={len(merges_super_total)} super): U = {u_orig:.4f}, p = {p_orig:.4e}")
else:
    print("  recomputed on this brain: insufficient data "
          f"(n_2={len(merges_2_total)}, n_super={len(merges_super_total)})")

# --- (a) Corrected MW on PER-NEURON cable -----------------------------
print("\n[CORRECTED (a)] Mann-Whitney U on PER-NEURON cable "
      "(matches hypothesis wording)")
if merges_2_per and merges_super_per:
    u_per, p_per = mannwhitneyu(merges_2_per, merges_super_per,
                                alternative="two-sided")
    print(f"  per-neuron medians: 2-merge = {np.median(merges_2_per):.4f} mm, "
          f"super = {np.median(merges_super_per):.4f} mm")
    print(f"  U = {u_per:.4f}, p = {p_per:.4e}  "
          f"(n_2={len(merges_2_per)}, n_super={len(merges_super_per)})")
else:
    print("  insufficient data")

# --- (b) Cliff's delta + bootstrap 95% CI ------------------------------
def cliffs_delta(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    # vectorised pairwise sign
    diff = b[None, :] - a[:, None]
    return float((np.sign(diff)).mean())

print("\n[CORRECTED (b)] Cliff's delta + bootstrap CI "
      "(per-neuron and total cable)")
rng = np.random.default_rng(42)
def boot_ci(a, b, fn, n_boot=2000):
    a = np.asarray(a); b = np.asarray(b)
    if len(a) == 0 or len(b) == 0:
        return float("nan"), float("nan")
    boots = []
    for _ in range(n_boot):
        sa = a[rng.integers(0, len(a), len(a))]
        sb = b[rng.integers(0, len(b), len(b))]
        boots.append(fn(sa, sb))
    return float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))

for label, a, b in [
    ("TOTAL cable    ", merges_2_total, merges_super_total),
    ("PER-neuron cable", merges_2_per, merges_super_per),
]:
    if len(a) == 0 or len(b) == 0:
        print(f"  {label}: insufficient data "
              f"(n_2={len(a)}, n_super={len(b)})")
        continue
    d = cliffs_delta(a, b)
    lo, hi = boot_ci(a, b, cliffs_delta, n_boot=2000)
    # Magnitude interpretation per Vargha-Delaney convention
    mag = abs(d)
    interp = ("negligible" if mag < 0.147 else
              "small" if mag < 0.33 else
              "medium" if mag < 0.474 else
              "large")
    print(f"  {label}: Cliff's delta = {d:+.4f}  95% CI "
          f"[{lo:+.4f}, {hi:+.4f}]  ({interp})  "
          f"n_2={len(a)}, n_super={len(b)}")

print("\n" + "=" * 72)
print("SIDE-BY-SIDE: original vs corrected")
print("=" * 72)
print(f"  Original (multi-brain TOTAL cable, n=24+3): U=0.0, p=5.886e-03 "
      f"(floor effect — smallest achievable p)")
print(f"  Corrected on this brain TOTAL cable: see above")
print(f"  Corrected PER-NEURON cable (literal hypothesis): see above")
print(f"  Corrected Cliff's delta (effect size) with bootstrap CI: see above")
print(f"  brain = {brain_id}")
