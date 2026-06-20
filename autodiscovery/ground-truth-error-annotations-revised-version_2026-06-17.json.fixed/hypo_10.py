# === RERUN_LOADING_BOOTSTRAP v2 ===
import os as _os, sys as _sys, subprocess as _sp_boot
try:
    import numpy as _np_boot
    _ver = tuple(int(x) for x in _np_boot.__version__.split(".")[:2])
    if _ver < (2, 0):
        raise ImportError("need numpy>=2")
except Exception:
    _sp_boot.check_call([_sys.executable, "-m", "pip", "install", "-q", "--user", "numpy>=2,<2.3"])
    import importlib as _il_boot
    if "numpy" in _sys.modules:
        del _sys.modules["numpy"]
import numpy as _np_chk
print("Loading dataset from:", _os.environ.get("RERUN_PKL", "<unset>"), "| numpy", _np_chk.__version__)

import os as _os2, glob as _glob_mod, subprocess as _sp
from pathlib import Path as _Path_boot
_RERUN_PKL = _os.environ.get("RERUN_PKL", "")
_RERUN_DIR = _os.path.dirname(_RERUN_PKL) or "."
_RERUN_NAME = _os.path.basename(_RERUN_PKL)
def _os_walk_patched(top, *a, **k):
    yield (_RERUN_DIR, [], [_RERUN_NAME])
_os2.walk = _os_walk_patched
_orig_rglob = _Path_boot.rglob
def _rglob_patched(self, pat):
    if isinstance(pat, str) and "pkl" in pat:
        yield _Path_boot(_RERUN_PKL)
        return
    yield from _orig_rglob(self, pat)
_Path_boot.rglob = _rglob_patched
_orig_getoutput = _sp.getoutput
def _getoutput_patched(cmd, *a, **k):
    s = str(cmd)
    if "find" in s and "pkl" in s:
        return _RERUN_PKL
    return _orig_getoutput(cmd, *a, **k)
_sp.getoutput = _getoutput_patched
_orig_check_call = _sp.check_call
def _check_call_patched(args, *a, **k):
    if isinstance(args, list) and len(args) > 3 and args[1:4] == ["-m", "pip", "install"]:
        return 0
    return _orig_check_call(args, *a, **k)
_sp.check_call = _check_call_patched
_orig_run = _sp.run
def _run_patched(args, *a, **k):
    if isinstance(args, list) and len(args) > 3 and args[1:4] == ["-m", "pip", "install"]:
        class _R: returncode = 0; stdout = b""; stderr = b""
        return _R()
    return _orig_run(args, *a, **k)
_sp.run = _run_patched
# === END RERUN_LOADING_BOOTSTRAP ===

# ============================================================
# H10 / Entry 3 — CORRECTED TEST
# Original test: Pearson chi-square on a 2x2 of (branching/linear) x (split/not-split)
#   over n ~ 1.4M edges. Recorded headline: chi2=414.47, p=3.90e-92, rate ratio
#   1.62%/0.47% = 3.4x.
# Problems:
#   - Adjacent edges share endpoints (a branch node feeds 3+ "branching" edges
#     and any "linear" edge it touches), so the independence assumption of the
#     Pearson chi-square is violated; the effective n is much smaller than the
#     raw 1.4M edge count, so chi2 is inflated and the p-value is a floor report.
#   - The discovery is an effect-size claim (rate ratio ~3x), not a p<.001
#     claim; the chi-square gives no uncertainty quantification of that ratio.
# Correction:
#   1. Keep the same partition (branching vs linear edges) and the same SPLIT
#      label.
#   2. Report the RATE RATIO and report a CLUSTER-BOOTSTRAP 95% CI over
#      skeletons (each GT neuron is a cluster) — this respects edge dependence.
#   3. Report a CLUSTER-PERMUTATION p-value where the (branching/linear) label
#      is shuffled WITHIN each skeleton, preserving cluster structure but
#      breaking the cross-class branching x split association under H0.
# ============================================================
import glob
import os
import pickle
import numpy as np
from collections import defaultdict
from scipy.stats import chi2_contingency

EDGE_SPLIT = 1

cache_paths = glob.glob("../dataset_cache_*_add.pkl")
if not cache_paths:
    cache_paths = glob.glob("dataset_cache_*_add.pkl")
if not cache_paths and _RERUN_PKL:
    cache_paths = [_RERUN_PKL]

# Per-skeleton (neuron) edge tables: counts of (branching/linear) x (split/not)
per_skel = defaultdict(lambda: np.zeros((2, 2), dtype=np.int64))

for path in cache_paths:
    with open(path, "rb") as f:
        payload = pickle.load(f)
    gt = payload["gt_graph"]
    edge_error = np.asarray(payload["gt_edge_error"])
    edges = list(gt.edges)
    degrees = dict(gt.degree)
    for k, (u, v) in enumerate(edges):
        is_branching = 1 if (degrees[u] > 2 or degrees[v] > 2) else 0
        is_split = 1 if int(edge_error[k]) == EDGE_SPLIT else 0
        # Skeleton identity: gt_graph component containing edge endpoint u
        try:
            skel = gt.node_segment_id(u)
        except Exception:
            skel = u
        per_skel[skel][is_branching, is_split] += 1

# Pool
pooled = np.zeros((2, 2), dtype=np.int64)
for tbl in per_skel.values():
    pooled += tbl

# unpack (rows: branching=1, linear=0; cols: split=1, not-split=0)
branching_total = int(pooled[1, :].sum())
linear_total = int(pooled[0, :].sum())
branching_split = int(pooled[1, 1])
linear_split = int(pooled[0, 1])

br_rate = branching_split / branching_total if branching_total > 0 else 0
ln_rate = linear_split / linear_total if linear_total > 0 else 0
rate_ratio = br_rate / ln_rate if ln_rate > 0 else float("inf")

# ----- Recorded test (for comparison only) -----
chi2_pearson, p_pearson, _, _ = chi2_contingency(
    np.array([[branching_split, branching_total - branching_split],
              [linear_split, linear_total - linear_split]])
)

# ----- Cluster bootstrap CI on rate ratio over skeletons -----
skels = list(per_skel.keys())
tables = [per_skel[s] for s in skels]
rng = np.random.default_rng(20260618)
n_boot = 1000
boot_ratios = np.empty(n_boot)
for b in range(n_boot):
    idx = rng.integers(0, len(skels), size=len(skels))
    bt = np.zeros((2, 2), dtype=np.int64)
    for j in idx:
        bt += tables[j]
    br_tot = bt[1, :].sum(); ln_tot = bt[0, :].sum()
    br_sp = bt[1, 1]; ln_sp = bt[0, 1]
    if br_tot > 0 and ln_tot > 0 and ln_sp > 0:
        boot_ratios[b] = (br_sp / br_tot) / (ln_sp / ln_tot)
    else:
        boot_ratios[b] = np.nan
boot_ratios_clean = boot_ratios[np.isfinite(boot_ratios)]
ci_low = float(np.percentile(boot_ratios_clean, 2.5))
ci_high = float(np.percentile(boot_ratios_clean, 97.5))

# ----- Cluster permutation p-value: shuffle (branching/linear) labels WITHIN
# each skeleton; recompute observed minus null rate-ratio. This preserves the
# within-skeleton dependence structure.
# To keep CPU bounded we sample edges from each skeleton's existing table.
# Reconstruct per-skeleton edge vectors (label_branch, label_split):
per_skel_arrays = {}
for s, tbl in per_skel.items():
    labels = []  # (is_branching, is_split)
    for br in (0, 1):
        for sp in (0, 1):
            cnt = int(tbl[br, sp])
            if cnt > 0:
                labels.extend([(br, sp)] * cnt)
    per_skel_arrays[s] = np.array(labels, dtype=np.int8) if labels else np.zeros((0, 2), dtype=np.int8)

obs_diff = br_rate - ln_rate
n_perm = 500
perm_diffs = np.empty(n_perm)
for p in range(n_perm):
    pooled_br_sp = 0; pooled_br_tot = 0; pooled_ln_sp = 0; pooled_ln_tot = 0
    for s, arr in per_skel_arrays.items():
        if arr.shape[0] < 2:
            pooled_br_sp += int(arr[arr[:, 0] == 1, 1].sum()) if arr.shape[0] else 0
            pooled_br_tot += int((arr[:, 0] == 1).sum()) if arr.shape[0] else 0
            pooled_ln_sp += int(arr[arr[:, 0] == 0, 1].sum()) if arr.shape[0] else 0
            pooled_ln_tot += int((arr[:, 0] == 0).sum()) if arr.shape[0] else 0
            continue
        # shuffle branching label within skeleton, holding split label fixed
        shuffled_br = rng.permutation(arr[:, 0])
        sp_lbl = arr[:, 1]
        pooled_br_sp += int(sp_lbl[shuffled_br == 1].sum())
        pooled_br_tot += int((shuffled_br == 1).sum())
        pooled_ln_sp += int(sp_lbl[shuffled_br == 0].sum())
        pooled_ln_tot += int((shuffled_br == 0).sum())
    p_br = pooled_br_sp / pooled_br_tot if pooled_br_tot > 0 else 0
    p_ln = pooled_ln_sp / pooled_ln_tot if pooled_ln_tot > 0 else 0
    perm_diffs[p] = p_br - p_ln

perm_pval = float((np.abs(perm_diffs) >= abs(obs_diff)).sum() + 1) / (n_perm + 1)

print("=== H10 corrected: Branching vs Linear split rate with cluster correction ===")
print(f"Skeletons (clusters): {len(skels)}")
print(f"Branching edges: {branching_total} total, {branching_split} split ({100*br_rate:.4f}%)")
print(f"Linear edges:    {linear_total} total, {linear_split} split ({100*ln_rate:.4f}%)")
print(f"Rate ratio (branching/linear): {rate_ratio:.4f}")
print(f"95% cluster-bootstrap CI on rate ratio: [{ci_low:.4f}, {ci_high:.4f}]  (B={n_boot})")
print(f"Cluster-permutation p-value (rate difference, n_perm={n_perm}): {perm_pval:.4e}")
print(f"[recorded for comparison] Pearson chi2={chi2_pearson:.4f}, p={p_pearson:.4e}")
