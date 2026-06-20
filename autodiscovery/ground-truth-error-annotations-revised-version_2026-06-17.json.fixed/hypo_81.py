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
# H81 / Entry 20 — CORRECTED TEST
# Original test: Welch's two-sample t-test on local fragment-node VOLUMETRIC
#   density (nodes/um^3) at 67 merge sites vs 67 length-weighted controls.
#   Recorded: t = 8.93, p = 2.77e-14, 0.001678 vs 0.001083 nodes/um^3.
# Problems:
#   - Density data is non-negative and right-skewed; with n=67 the Welch's t
#     assumption on the sampling distribution of the mean is less reliable
#     than a rank-based test.
#   - No effect-size CI is reported, only a point estimate of the means.
# Correction:
#   1. Replace Welch's t with Mann-Whitney U (matching the test used for the
#      related H3/H23/H49 variants of this same merge-density signal).
#   2. Report Cliff's delta (rank-biserial) as the effect size and a
#      bootstrap 95% CI.
#   3. Report a permutation p (label shuffle) — does not need a cluster
#      adjustment because the 67 merge sites are independent and the controls
#      are sampled independently.
# ============================================================
import os
import sys
import subprocess
import random
import pickle
import numpy as np
from scipy.spatial import cKDTree
from scipy import stats


def main():
    random.seed(42); np.random.seed(42)
    filename = "dataset_cache_789202_mcl100_add.pkl"
    filepath = os.path.join("..", filename)
    if not os.path.exists(filepath):
        filepath_output = subprocess.getoutput(f'find / -name "{filename}" 2>/dev/null').strip().split('\n')[0]
        if filepath_output and os.path.exists(filepath_output):
            filepath = filepath_output
        else:
            filepath_output = subprocess.getoutput('find / -name "dataset_cache_*_add.pkl" 2>/dev/null').strip().split('\n')[0]
            if filepath_output and os.path.exists(filepath_output):
                filepath = filepath_output

    print(f"Loading dataset from: {filepath}")
    with open(filepath, "rb") as f:
        payload = pickle.load(f)
    fg = payload['fragments_graph']
    gt = payload['gt_graph']
    gt_merge_sites = payload['gt_merge_sites']
    gt_edge_error = np.asarray(payload['gt_edge_error'])

    valid_nodes = list(fg.nodes)
    if hasattr(fg, 'node_xyz') and isinstance(fg.node_xyz, np.ndarray):
        try:
            fg_coords = fg.node_xyz[valid_nodes]
        except Exception:
            fg_coords = np.array([fg.node_xyz[n] for n in valid_nodes])
    elif hasattr(fg, 'node_xyz') and fg.node_xyz is not None:
        fg_coords = np.array([fg.node_xyz[n] for n in valid_nodes])
    else:
        fg_coords = np.array([fg.nodes[n]['xyz'] for n in valid_nodes])
    tree = cKDTree(fg_coords)

    merge_coords = [np.array(site['xyz']) for site in gt_merge_sites]
    radius = 10.0
    volume = (4.0 / 3.0) * np.pi * (radius ** 3)
    merge_counts = tree.query_ball_point(merge_coords, r=radius)
    merge_dens = np.array([len(n) / volume for n in merge_counts])

    edges = list(gt.edges)
    correct_edges = [edges[i] for i, err in enumerate(gt_edge_error) if err == 0]
    edge_lengths = []
    valid_correct_edges = []
    for u, v in correct_edges:
        if hasattr(gt, 'node_xyz') and gt.node_xyz is not None:
            p1 = np.array(gt.node_xyz[u]); p2 = np.array(gt.node_xyz[v])
        else:
            p1 = np.array(gt.nodes[u]['xyz']); p2 = np.array(gt.nodes[v]['xyz'])
        edge_lengths.append(float(np.linalg.norm(p1 - p2)))
        valid_correct_edges.append((p1, p2))
    total_len = sum(edge_lengths); probs = [L / total_len for L in edge_lengths]
    chosen = random.choices(range(len(valid_correct_edges)), weights=probs, k=len(merge_coords))
    control_coords = []
    for idx in chosen:
        p1, p2 = valid_correct_edges[idx]
        t = random.uniform(0, 1)
        control_coords.append(p1 * t + p2 * (1 - t))
    control_counts = tree.query_ball_point(control_coords, r=radius)
    control_dens = np.array([len(n) / volume for n in control_counts])

    # ---- Recorded Welch's t (for comparison) ----
    t_stat, p_t = stats.ttest_ind(merge_dens, control_dens, equal_var=False)

    # ---- Mann-Whitney U (corrected primary test) ----
    U, p_mwu = stats.mannwhitneyu(merge_dens, control_dens, alternative='two-sided')
    # Cliff's delta == rank-biserial = 2*U/(n1*n2) - 1 (in the merge-greater orientation)
    cliffs_delta = 2.0 * U / (len(merge_dens) * len(control_dens)) - 1.0

    # ---- Bootstrap 95% CI on Cliff's delta ----
    rng = np.random.default_rng(20260618)
    B = 5000
    boot_cd = np.empty(B)
    for b in range(B):
        i_m = rng.integers(0, len(merge_dens), size=len(merge_dens))
        i_c = rng.integers(0, len(control_dens), size=len(control_dens))
        Ub, _ = stats.mannwhitneyu(merge_dens[i_m], control_dens[i_c], alternative='two-sided')
        boot_cd[b] = 2.0 * Ub / (len(merge_dens) * len(control_dens)) - 1.0
    ci_lo, ci_hi = float(np.percentile(boot_cd, 2.5)), float(np.percentile(boot_cd, 97.5))

    # ---- Permutation p ----
    n_perm = 10000
    combined = np.concatenate([merge_dens, control_dens])
    perm_cd = np.empty(n_perm)
    for p in range(n_perm):
        perm = rng.permutation(combined)
        a = perm[:len(merge_dens)]; b_ = perm[len(merge_dens):]
        Up, _ = stats.mannwhitneyu(a, b_, alternative='two-sided')
        perm_cd[p] = 2.0 * Up / (len(merge_dens) * len(control_dens)) - 1.0
    perm_p = float((np.abs(perm_cd) >= abs(cliffs_delta)).sum() + 1) / (n_perm + 1)

    print("=== H81 corrected: Mann-Whitney + Cliff's delta + bootstrap CI ===")
    print(f"n merge = {len(merge_dens)}, n control = {len(control_dens)}, radius = {radius} um, volume = {volume:.2f} um^3")
    print(f"Mean merge density: {merge_dens.mean():.6f} nodes/um^3")
    print(f"Mean control density: {control_dens.mean():.6f} nodes/um^3")
    print(f"Median merge density: {np.median(merge_dens):.6f}")
    print(f"Median control density: {np.median(control_dens):.6f}")
    print(f"Mann-Whitney U = {U:.4f}, p = {p_mwu:.4e}")
    print(f"Cliff's delta = {cliffs_delta:.4f}")
    print(f"Cliff's delta 95% bootstrap CI: [{ci_lo:.4f}, {ci_hi:.4f}]  (B={B})")
    print(f"Permutation p (label-shuffle, n_perm={n_perm}): {perm_p:.4e}")
    print(f"[recorded for comparison] Welch's t = {t_stat:.4f}, p = {p_t:.4e}")


if __name__ == "__main__":
    main()
