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
# H13 / Entry 5 — CORRECTED TEST
# Original test: Mann-Whitney U on geodesic distance-to-branch for SPLIT vs
#   CORRECT edges. Recorded: U=1.19e10, p=0.0, n=2,218,068 / 13,610 — the n was
#   2x the true n (the recorded script's glob saw two matching paths to the
#   same pkl and processed each edge twice).
# Problems:
#   - Double-counting bug doubles n on each side and inflates U.
#   - p=0.0 is a floor; says nothing about effect-size.
#   - Edges within a skeleton are non-independent (geodesic distance of
#     adjacent edges is tightly correlated), so the nominal p is over-confident.
# Correction:
#   1. Dedup paths so each edge enters exactly once.
#   2. Report cluster-aware median shift with cluster bootstrap CI over
#      skeletons.
#   3. Report rank-biserial effect size and a cluster-permutation p-value.
# ============================================================
import os
import glob
import pickle
import gc
import numpy as np
from collections import defaultdict
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.stats import mannwhitneyu

# DEDUP via absolute paths to fix the recorded double-count bug
raw_files = list(glob.glob("./*_add.pkl") + glob.glob("../data/*_add.pkl")
                 + glob.glob("../*_add.pkl"))
files = sorted(set(os.path.abspath(f) for f in raw_files))
print("Processing files (deduped):", files)

split_dists_all = []
correct_dists_all = []
per_skel_split = defaultdict(list)
per_skel_corr = defaultdict(list)

for file in files:
    gc.disable()
    with open(file, "rb") as f:
        payload = pickle.load(f)
    gc.enable()
    if "fragments_graph" in payload:
        del payload["fragments_graph"]
    gc.collect()

    gt = payload["gt_graph"]
    edge_error = np.asarray(payload["gt_edge_error"])
    nodes = list(gt.nodes())
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    N = len(nodes)

    degrees = dict(gt.degree())
    branch_points = [n for n, deg in degrees.items() if deg >= 3]
    bp_indices = [node_to_idx[n] for n in branch_points]

    edges = list(gt.edges())
    row = []; col = []; w = []
    try:
        node_xyz = gt.node_xyz
        for u, v in edges:
            p1 = node_xyz[u]; p2 = node_xyz[v]
            d = float(np.linalg.norm(p1 - p2))
            i = node_to_idx[u]; j = node_to_idx[v]
            row.extend([i, j]); col.extend([j, i]); w.extend([d, d])
    except Exception:
        for u, v in edges:
            p1 = gt.nodes[u].get('xyz', np.array([0, 0, 0]))
            p2 = gt.nodes[v].get('xyz', np.array([0, 0, 0]))
            d = float(np.linalg.norm(p1 - p2))
            i = node_to_idx[u]; j = node_to_idx[v]
            row.extend([i, j]); col.extend([j, i]); w.extend([d, d])
    for bp in bp_indices:
        row.extend([N, bp]); col.extend([bp, N]); w.extend([0.0, 0.0])

    if len(bp_indices) > 0:
        adj = coo_matrix((w, (row, col)), shape=(N+1, N+1)).tocsr()
        dist_to_bp = dijkstra(adj, directed=False, indices=N)[:N]
    else:
        dist_to_bp = np.full(N, np.inf)

    for k, (u, v) in enumerate(edges):
        err = int(edge_error[k])
        if err in (0, 1):
            i = node_to_idx[u]; j = node_to_idx[v]
            d = float(min(dist_to_bp[i], dist_to_bp[j]))
            if np.isinf(d):
                continue
            try:
                skel = gt.node_segment_id(u)
            except Exception:
                skel = u
            if err == 1:
                split_dists_all.append(d)
                per_skel_split[skel].append(d)
            else:
                correct_dists_all.append(d)
                per_skel_corr[skel].append(d)

    del payload, gt
    if 'adj' in locals():
        del adj
    del dist_to_bp
    gc.collect()

split_arr = np.asarray(split_dists_all)
correct_arr = np.asarray(correct_dists_all)

print(f"Deduped Correct edges: {len(correct_arr)}")
print(f"Deduped Split edges:   {len(split_arr)}")

# ----- Pooled Mann-Whitney (two-sided) for reference -----
U_pool, p_pool = mannwhitneyu(split_arr, correct_arr, alternative='two-sided')
rbs = 1.0 - 2.0 * U_pool / (len(split_arr) * len(correct_arr))
# Note: U here uses scipy ordering; effect direction is "split rank vs correct"
median_split = float(np.median(split_arr))
median_corr = float(np.median(correct_arr))
shift = median_split - median_corr

# ----- Cluster bootstrap over skeletons (each GT skeleton is a cluster) -----
skels = sorted(set(per_skel_split) | set(per_skel_corr))
rng = np.random.default_rng(20260618)
B = 1000
boot_shifts = np.empty(B)
for b in range(B):
    idx = rng.integers(0, len(skels), size=len(skels))
    s_pool = []
    c_pool = []
    for j in idx:
        s = skels[j]
        s_pool.extend(per_skel_split.get(s, []))
        c_pool.extend(per_skel_corr.get(s, []))
    if s_pool and c_pool:
        boot_shifts[b] = float(np.median(s_pool) - np.median(c_pool))
    else:
        boot_shifts[b] = np.nan
boot_clean = boot_shifts[np.isfinite(boot_shifts)]
ci_low, ci_high = float(np.percentile(boot_clean, 2.5)), float(np.percentile(boot_clean, 97.5))

# ----- Cluster permutation: shuffle SPLIT/CORRECT labels within each skeleton
n_perm = 500
perm_rbs = np.empty(n_perm)
for p in range(n_perm):
    perm_s = []
    perm_c = []
    for s in skels:
        sa = np.asarray(per_skel_split.get(s, []))
        ca = np.asarray(per_skel_corr.get(s, []))
        if sa.size == 0 and ca.size == 0:
            continue
        pool = np.concatenate([sa, ca])
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

print("=== H13 corrected: deduped + cluster-aware ===")
print(f"Pooled Mann-Whitney U={U_pool:.4e}, p={p_pool:.4e}")
print(f"Rank-biserial effect size r_rb = {rbs:.6f}  (negative => split < correct distance, hypothesis direction)")
print(f"Pooled median split = {median_split:.4f} um, median correct = {median_corr:.4f} um, shift = {shift:.4f} um")
print(f"Cluster-bootstrap 95% CI on median shift (split - correct): "
      f"[{ci_low:.4f}, {ci_high:.4f}]  (B={B}, n_skel={len(skels)})")
print(f"Cluster-permutation p (within-skeleton label shuffle, n_perm={n_perm}): {perm_p:.4f}")
print(f"[recorded for comparison] U=1.19e10, p=0.0 — recorded n was DOUBLE-COUNTED (2x)")
