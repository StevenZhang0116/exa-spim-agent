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
# H30 / Entry 8 — CORRECTED TEST
# Original test: Pearson r between splits_per_mm and pct_omit at n=12 neurons,
#   plus OLS R². Recorded: Pearson r=0.65 (p=0.022), Spearman rho=0.881 (p=1.5e-4).
# Problems:
#   - Pearson is sensitive to outliers and heteroscedasticity, and recorded
#     plot notes show a high-leverage neuron at y~12.6 drives the slope.
#   - n=12 is small; Pearson p sits just below 0.05 and would not survive
#     multiple-comparison correction.
# Correction:
#   1. Promote Spearman rho to the headline statistic (rank-based, robust).
#   2. Report a bootstrap 95% CI on Spearman rho.
#   3. Report a leave-one-out (jackknife) range on Spearman rho to expose
#      outlier leverage. Also report Kendall's tau as a second robust check.
# ============================================================
import glob
import re
import pickle
import gc
import numpy as np
import scipy.stats as stats
from collections import defaultdict

files = glob.glob("../*_add.pkl") + glob.glob("../*/*_add.pkl") + glob.glob("./*_add.pkl")
files = sorted(set(files))

def brain_id_from_path(path):
    m = re.search(r"dataset_cache_(\d+)_mcl(\d+)_add\.pkl$", os.path.basename(path))
    return m.group(1) if m else "unknown"

import os
neurons_data = []
for path in sorted(files):
    brain_id = brain_id_from_path(path)
    print(f"Processing Brain {brain_id} from {os.path.basename(path)}...")
    gc.disable()
    with open(path, "rb") as f:
        payload = pickle.load(f)
    gc.enable()
    gt = payload["gt_graph"]
    node_label = np.asarray(payload["gt_node_canonical_label"])
    edge_error = np.asarray(payload["gt_edge_error"])
    payload.pop("fragments_graph", None)
    gc.collect()

    neuron_segs = defaultdict(set)
    for n in gt.nodes:
        lab = int(node_label[n])
        if lab != 0:
            neuron_segs[gt.node_segment_id(n)].add(lab)
    neuron_splits = {nm: max(len(s) - 1, 0) for nm, s in neuron_segs.items()}

    neuron_edge_lengths = defaultdict(float)
    neuron_omit_counts = defaultdict(int)
    neuron_edge_counts = defaultdict(int)
    for k, (u, v) in enumerate(list(gt.edges)):
        neuron_id = gt.node_segment_id(u)
        xyz_u = gt.node_xyz[u]; xyz_v = gt.node_xyz[v]
        d = float(np.linalg.norm(xyz_u - xyz_v))
        neuron_edge_lengths[neuron_id] += d
        neuron_edge_counts[neuron_id] += 1
        if int(edge_error[k]) == 2:
            neuron_omit_counts[neuron_id] += 1
    for nid in neuron_edge_counts:
        L = neuron_edge_lengths[nid]
        if L < 50:
            continue
        length_mm = L / 1000.0
        sp_per_mm = neuron_splits.get(nid, 0) / length_mm
        pct_omit = (neuron_omit_counts[nid] / neuron_edge_counts[nid]) * 100.0
        neurons_data.append((nid, brain_id, sp_per_mm, pct_omit))
    del payload, gt, node_label, edge_error
    gc.collect()

if not neurons_data:
    print("No qualifying neurons.")
else:
    splits_per_mm = np.array([d[2] for d in neurons_data])
    pct_omit = np.array([d[3] for d in neurons_data])
    n = len(splits_per_mm)
    print(f"\nQualifying neurons n = {n}")

    # Recorded statistics for direct comparison
    pearson_r, pearson_p = stats.pearsonr(splits_per_mm, pct_omit)
    spearman_r, spearman_p = stats.spearmanr(splits_per_mm, pct_omit)
    kendall_t, kendall_p = stats.kendalltau(splits_per_mm, pct_omit)

    # Bootstrap CI on Spearman rho (primary corrected statistic)
    rng = np.random.default_rng(20260618)
    B = 5000
    boot_rho = np.empty(B)
    boot_pearson = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        if len(set(idx)) < 3:
            boot_rho[b] = np.nan; boot_pearson[b] = np.nan; continue
        try:
            br, _ = stats.spearmanr(splits_per_mm[idx], pct_omit[idx])
            bp, _ = stats.pearsonr(splits_per_mm[idx], pct_omit[idx])
        except Exception:
            br = np.nan; bp = np.nan
        boot_rho[b] = br
        boot_pearson[b] = bp
    rho_clean = boot_rho[np.isfinite(boot_rho)]
    pe_clean = boot_pearson[np.isfinite(boot_pearson)]
    rho_ci = (float(np.percentile(rho_clean, 2.5)), float(np.percentile(rho_clean, 97.5)))
    pe_ci = (float(np.percentile(pe_clean, 2.5)), float(np.percentile(pe_clean, 97.5)))

    # Permutation p-value on Spearman rho
    n_perm = 10000
    perm_rho = np.empty(n_perm)
    for p in range(n_perm):
        perm_y = rng.permutation(pct_omit)
        perm_rho[p], _ = stats.spearmanr(splits_per_mm, perm_y)
    perm_p = float((np.abs(perm_rho) >= abs(spearman_r)).sum() + 1) / (n_perm + 1)

    # Leave-one-out on Spearman rho
    loo_rhos = []
    for i in range(n):
        mask = np.arange(n) != i
        if mask.sum() >= 3:
            r, _ = stats.spearmanr(splits_per_mm[mask], pct_omit[mask])
            loo_rhos.append(r)
    loo_min, loo_max = float(min(loo_rhos)), float(max(loo_rhos))

    print("=== H30 corrected: rank-based correlation with bootstrap CI ===")
    print(f"Spearman rho = {spearman_r:.4f}, p = {spearman_p:.4e}  [CORRECTED HEADLINE]")
    print(f"Spearman 95% bootstrap CI: [{rho_ci[0]:.4f}, {rho_ci[1]:.4f}]  (B={B})")
    print(f"Permutation p (Spearman, n_perm={n_perm}): {perm_p:.4e}")
    print(f"Leave-one-out Spearman rho range: [{loo_min:.4f}, {loo_max:.4f}]")
    print(f"Kendall tau = {kendall_t:.4f}, p = {kendall_p:.4e}")
    print(f"[recorded for comparison] Pearson r = {pearson_r:.4f}, p = {pearson_p:.4e}")
    print(f"Pearson 95% bootstrap CI: [{pe_ci[0]:.4f}, {pe_ci[1]:.4f}]")
