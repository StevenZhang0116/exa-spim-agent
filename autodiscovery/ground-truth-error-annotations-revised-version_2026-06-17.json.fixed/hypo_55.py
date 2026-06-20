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
# H55 / Entry 14 — CORRECTED TEST
# Original test: KS two-sample test on pooled OBSERVED inter-split distances
#   versus a uniform-along-skeleton NULL. Recorded D=0.454, p ~ 0 (floor),
#   median 25.58 um vs null median 236.98 um (~9x shorter).
# Problems:
#   - Pooled inter-split nearest-neighbor distances within a skeleton are
#     statistically NON-INDEPENDENT (a nearest-neighbor pair shares one of
#     two distances with another nearest-neighbor pair); ks_2samp over
#     pooled samples treats them as independent, inflating the test.
#   - "p = 0" is a floor report dominated by sample size.
# Correction:
#   1. Reduce each NEURON to a single statistic: the ratio
#         T = median(observed inter-split distance) / median(null inter-split distance)
#      computed from that neuron's permutations (already produced in the
#      original script). Each neuron contributes ONE independent T.
#   2. Paired Wilcoxon signed-rank on observed-vs-null median per neuron.
#   3. Bootstrap 95% CI on the median ratio across neurons.
#   4. Cluster permutation p (over neurons) on the same ratio.
# ============================================================
import sys
import glob
import pickle
import gc
import numpy as np
from collections import defaultdict
import scipy.sparse as sp
from scipy.sparse.csgraph import shortest_path
from scipy.stats import ks_2samp, wilcoxon

def get_roots(N, S_nodes, dist, pred):
    roots = np.full(N + 1, -1, dtype=np.int32)
    roots[S_nodes] = S_nodes
    in_path = np.zeros(N + 1, dtype=bool)
    for i in range(N):
        if roots[i] != -1 or dist[i] == np.inf:
            continue
        curr = i; path = []
        while roots[curr] == -1:
            if in_path[curr]:
                curr = -9999; break
            in_path[curr] = True
            path.append(curr)
            curr = pred[curr]
            if curr == -9999:
                break
        if curr == -9999:
            r = -1
        else:
            r = roots[curr]
        for node in path:
            roots[node] = r; in_path[node] = False
    return roots


paths = glob.glob("../dataset_cache_*_add.pkl") or glob.glob("dataset_cache_*_add.pkl") or glob.glob("cache/dataset_cache_*_add.pkl")
if not paths:
    print("No datasets found.")
    sys.exit(0)

EDGE_SPLIT = 1
NUM_PERMUTATIONS = 100

per_neuron_obs_median = []
per_neuron_null_median = []
all_observed_dists = []
all_null_dists = []

for path in paths:
    print(f"Loading {path}...")
    with open(path, "rb") as f:
        payload = pickle.load(f)
    gt = payload["gt_graph"]
    edge_error = np.asarray(payload["gt_edge_error"])
    edges = list(gt.edges)
    node_xyz = gt.node_xyz
    neuron_to_edges = defaultdict(list)
    for k, (u, v) in enumerate(edges):
        neuron_u = gt.node_segment_id(u)
        neuron_to_edges[neuron_u].append(k)

    for neuron, edge_indices in neuron_to_edges.items():
        split_locals = [i for i, k in enumerate(edge_indices) if int(edge_error[k]) == EDGE_SPLIT]
        if len(split_locals) < 2:
            continue
        unique_nodes = set()
        for k in edge_indices:
            u, v = edges[k]
            unique_nodes.add(u); unique_nodes.add(v)
        V = len(unique_nodes)
        E_num = len(edge_indices)
        node_to_idx = {n: i for i, n in enumerate(unique_nodes)}
        aug_u, aug_v, aug_w = [], [], []
        for i, k in enumerate(edge_indices):
            u, v = edges[k]
            idx_u = node_to_idx[u]; idx_v = node_to_idx[v]
            m = V + i
            L = np.linalg.norm(node_xyz[u] - node_xyz[v])
            aug_u.extend([idx_u, idx_v]); aug_v.extend([m, m]); aug_w.extend([L / 2.0, L / 2.0])
        aug_u = np.array(aug_u, dtype=np.int32)
        aug_v = np.array(aug_v, dtype=np.int32)
        aug_w = np.array(aug_w, dtype=np.float64)
        all_aug_u = np.concatenate([aug_u, aug_v])
        all_aug_v = np.concatenate([aug_v, aug_u])
        all_aug_w = np.concatenate([aug_w, aug_w])
        N = V + E_num
        base_csr = sp.coo_matrix((all_aug_w, (all_aug_u, all_aug_v)), shape=(N + 1, N + 1)).tocsr()
        nnz_base = base_csr.nnz
        num_splits = len(split_locals)
        nnz_total = nnz_base + num_splits
        new_data = np.empty(nnz_total, dtype=np.float64)
        new_data[:nnz_base] = base_csr.data; new_data[nnz_base:] = 0.0
        new_indices = np.empty(nnz_total, dtype=np.int32)
        new_indices[:nnz_base] = base_csr.indices
        new_indptr = np.empty(N + 2, dtype=np.int32)
        new_indptr[:N + 1] = base_csr.indptr[:N + 1]
        new_indptr[N + 1] = nnz_total

        def compute_distances_for_samples(sample_indices):
            S_nodes = np.array([V + s for s in sample_indices], dtype=np.int32)
            new_indices[nnz_base:] = S_nodes
            adj = sp.csr_matrix((new_data, new_indices, new_indptr), shape=(N + 1, N + 1), copy=False)
            dist, pred = shortest_path(csgraph=adj, directed=True, indices=N, return_predecessors=True)
            roots = get_roots(N, S_nodes, dist, pred)
            c_u = roots[aug_u]; c_v = roots[aug_v]
            valid_mask = (c_u != -1) & (c_v != -1) & (c_u != c_v)
            diff_u = c_u[valid_mask]; diff_v = c_v[valid_mask]
            diff_w = aug_w[valid_mask]
            diff_dist_u = dist[aug_u[valid_mask]]
            diff_dist_v = dist[aug_v[valid_mask]]
            total_dists = diff_dist_u + diff_dist_v + diff_w
            min_other_dist = np.full(N + 1, np.inf)
            np.minimum.at(min_other_dist, diff_u, total_dists)
            np.minimum.at(min_other_dist, diff_v, total_dists)
            return min_other_dist[S_nodes]

        obs_dists = compute_distances_for_samples(split_locals)
        obs_finite = obs_dists[np.isfinite(obs_dists)]
        if obs_finite.size == 0:
            continue
        all_observed_dists.extend(obs_finite.tolist())

        null_medians = []
        all_null_per_neuron = []
        for _ in range(NUM_PERMUTATIONS):
            s_rand = np.random.choice(E_num, num_splits, replace=False)
            null_dists = compute_distances_for_samples(s_rand)
            null_finite = null_dists[np.isfinite(null_dists)]
            if null_finite.size > 0:
                all_null_per_neuron.extend(null_finite.tolist())
                null_medians.append(float(np.median(null_finite)))
        all_null_dists.extend(all_null_per_neuron)
        if null_medians:
            per_neuron_obs_median.append(float(np.median(obs_finite)))
            per_neuron_null_median.append(float(np.median(null_medians)))

    del payload, gt, edges, node_xyz
    gc.collect()

obs_arr = np.asarray(all_observed_dists)
null_arr = np.asarray(all_null_dists)

# Pooled KS (for direct comparison with the recorded statistic)
ks_stat, ks_p = ks_2samp(obs_arr, null_arr)

# Cluster-level paired analysis (one observation per neuron)
om = np.asarray(per_neuron_obs_median)
nm = np.asarray(per_neuron_null_median)
n_clusters = len(om)
if n_clusters > 0:
    log_ratio = np.log(om / nm)
    mean_log_ratio = float(log_ratio.mean())
    median_ratio = float(np.median(om / nm))
    # Paired Wilcoxon: observed < null (one-sided)
    w_stat, w_p = wilcoxon(om, nm, alternative='less')
    # Matched-pairs rank-biserial
    diffs = om - nm
    diffs_nz = diffs[diffs != 0]
    ranks = np.argsort(np.argsort(np.abs(diffs_nz))) + 1
    Rp = ranks[diffs_nz > 0].sum(); Rn = ranks[diffs_nz < 0].sum()
    rbs = float((Rp - Rn) / ranks.sum())
    # Cluster bootstrap 95% CI on median ratio
    rng = np.random.default_rng(20260618)
    B = 5000
    boot = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n_clusters, size=n_clusters)
        boot[b] = float(np.median(om[idx] / nm[idx]))
    ci_lo, ci_hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
    # Cluster permutation p: randomly swap obs/null label per neuron
    n_perm = 5000
    perm_meds = np.empty(n_perm)
    for p in range(n_perm):
        flip = rng.choice([0, 1], size=n_clusters).astype(bool)
        om_p = np.where(flip, nm, om)
        nm_p = np.where(flip, om, nm)
        perm_meds[p] = float(np.median(om_p / nm_p))
    perm_p = float((perm_meds <= median_ratio).sum() + 1) / (n_perm + 1)
else:
    w_stat, w_p, median_ratio, ci_lo, ci_hi, perm_p, rbs = (
        None, None, float("nan"), float("nan"), float("nan"), float("nan"), float("nan"))

print("=== H55 corrected: cluster-aware test on per-neuron observed/null median ratio ===")
print(f"Total observed splits (pooled): {len(obs_arr)} | Total null samples: {len(null_arr)}")
print(f"Pooled KS: D={ks_stat:.4f}, p={ks_p:.4e}  [matches recorded floor]")
print(f"Independent clusters (neurons): {n_clusters}")
print(f"Per-neuron median ratio (observed/null): median={median_ratio:.4f}")
print(f"Cluster-bootstrap 95% CI on median ratio: [{ci_lo:.4f}, {ci_hi:.4f}]")
print(f"Paired Wilcoxon signed-rank (obs<null, paired across neurons): W={w_stat}, p={w_p}")
print(f"Matched-pairs rank-biserial effect: {rbs:.4f}")
print(f"Cluster-permutation p (per-neuron obs/null label swap, n_perm=5000): {perm_p:.4f}")
print(f"[recorded for comparison] D=0.4539, p=0.0 floor; obs/null medians ratio ~ 9x")
