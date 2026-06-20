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
# H72 / Entry 19 — CORRECTED TEST
# Original test: Mann-Whitney U on topological distance-to-branch over 1.4M
#   edges. Recorded U=2.98e9, p=0 (floor); pooled split mean=518 um vs correct
#   mean=712 um.
# Problems:
#   - Edges within a skeleton are NOT independent; with n > 10^6 even a 5 um
#     difference becomes "p ~ 0", and on the extra brain ds_794491 the
#     effect-size gap collapses to ~5 um while the recorded p still floor-reports.
#   - The headline "median distance about half" is brain-specific; the
#     original test offers no effect-size CI to communicate that.
# Correction:
#   1. Keep the same SPLIT-vs-CORRECT partition and the same distance metric.
#   2. Compute the rank-biserial effect size on the pooled samples (already
#      contained in U).
#   3. Cluster-bootstrap 95% CI on the median shift (split - correct), with
#      each skeleton as a cluster.
#   4. Cluster permutation p where the SPLIT/CORRECT label is shuffled within
#      each skeleton.
# ============================================================
import sys
import glob
import pickle
import gc
import pathlib
import numpy as np
import networkx as nx
from collections import defaultdict
from scipy.stats import mannwhitneyu


def main():
    dataset_path = None
    for p in pathlib.Path('..').rglob('*add.pkl'):
        dataset_path = str(p); break
    if not dataset_path:
        for p in pathlib.Path('.').rglob('*add.pkl'):
            dataset_path = str(p); break
    if not dataset_path:
        print("Dataset not found."); return

    print(f"Loading dataset from: {dataset_path}")
    with open(dataset_path, "rb") as f:
        payload = pickle.load(f)
    gt = payload["gt_graph"]
    edge_error = np.asarray(payload["gt_edge_error"])
    print(f"Graph: {gt.number_of_nodes()} nodes, {gt.number_of_edges()} edges.")

    for u, v, data in gt.edges(data=True):
        data['weight'] = float(np.linalg.norm(gt.node_xyz[u] - gt.node_xyz[v]))

    branch_nodes = {n for n, d in gt.degree() if d >= 3}
    print(f"GT branch nodes: {len(branch_nodes)}")
    if not branch_nodes:
        print("No branch nodes."); return

    dist_to_branch = nx.multi_source_dijkstra_path_length(gt, branch_nodes, weight='weight')

    EDGE_CORRECT = 0; EDGE_SPLIT = 1

    split_all = []; correct_all = []
    per_skel_split = defaultdict(list)
    per_skel_corr = defaultdict(list)
    for i, (u, v) in enumerate(list(gt.edges)):
        err = int(edge_error[i])
        if u in dist_to_branch and v in dist_to_branch:
            ed = (dist_to_branch[u] + dist_to_branch[v]) / 2.0
            try:
                skel = gt.node_segment_id(u)
            except Exception:
                skel = u
            if err == EDGE_SPLIT:
                split_all.append(ed); per_skel_split[skel].append(ed)
            elif err == EDGE_CORRECT:
                correct_all.append(ed); per_skel_corr[skel].append(ed)

    split_arr = np.asarray(split_all)
    correct_arr = np.asarray(correct_all)
    print(f"Split edges: {len(split_arr)}; Correct edges: {len(correct_arr)}")
    if not len(split_arr) or not len(correct_arr):
        print("Not enough data."); return

    # Pooled Mann-Whitney (recorded test) for direct comparison
    U_pool, p_pool = mannwhitneyu(split_arr, correct_arr, alternative='two-sided')
    rbs = 1.0 - 2.0 * U_pool / (len(split_arr) * len(correct_arr))
    median_split = float(np.median(split_arr))
    median_corr = float(np.median(correct_arr))
    shift = median_split - median_corr

    skels = sorted(set(per_skel_split) | set(per_skel_corr))
    rng = np.random.default_rng(20260618)
    B = 1000
    boot_shifts = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, len(skels), size=len(skels))
        s_pool = []; c_pool = []
        for j in idx:
            s = skels[j]
            s_pool.extend(per_skel_split.get(s, []))
            c_pool.extend(per_skel_corr.get(s, []))
        if s_pool and c_pool:
            boot_shifts[b] = float(np.median(s_pool) - np.median(c_pool))
        else:
            boot_shifts[b] = np.nan
    bc = boot_shifts[np.isfinite(boot_shifts)]
    ci_lo, ci_hi = float(np.percentile(bc, 2.5)), float(np.percentile(bc, 97.5))

    # Cluster permutation: shuffle SPLIT/CORRECT labels within each skeleton
    n_perm = 500
    perm_rbs = np.empty(n_perm)
    for p in range(n_perm):
        perm_s = []; perm_c = []
        for s in skels:
            sa = np.asarray(per_skel_split.get(s, []))
            ca = np.asarray(per_skel_corr.get(s, []))
            pool = np.concatenate([sa, ca])
            if pool.size == 0:
                continue
            rng.shuffle(pool)
            n_s = sa.size
            perm_s.extend(pool[:n_s].tolist())
            perm_c.extend(pool[n_s:].tolist())
        if perm_s and perm_c:
            U_p, _ = mannwhitneyu(perm_s, perm_c, alternative='two-sided')
            perm_rbs[p] = 1.0 - 2.0 * U_p / (len(perm_s) * len(perm_c))
        else:
            perm_rbs[p] = 0.0
    perm_p = float((np.abs(perm_rbs) >= abs(rbs)).sum() + 1) / (n_perm + 1)

    print("=== H72 corrected: cluster-aware split-vs-correct distance-to-branch ===")
    print(f"Pooled Mann-Whitney U={U_pool:.4e}, p={p_pool:.4e}")
    print(f"Rank-biserial effect size r_rb = {rbs:.6f}  (negative => split < correct, hypothesis direction)")
    print(f"Pooled median split = {median_split:.4f} um; median correct = {median_corr:.4f} um; shift = {shift:.4f} um")
    print(f"Cluster bootstrap 95% CI on median shift (split - correct): "
          f"[{ci_lo:.4f}, {ci_hi:.4f}]  (B={B}, n_skel={len(skels)})")
    print(f"Cluster permutation p (within-skeleton label shuffle, n_perm={n_perm}): {perm_p:.4f}")
    print(f"[recorded for comparison] U=2.98e9, p=0 floor, mean shift 194 um (origin)")


if __name__ == "__main__":
    main()
