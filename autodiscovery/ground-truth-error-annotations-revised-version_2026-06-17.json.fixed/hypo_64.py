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
# H64 / Entry 16 — CORRECTED TEST
# Original test: Pearson chi-square on 2x2 of (terminal/internal) edges x
#   (omit/not-omit) over ~1.4M edges. Recorded chi^2 = 4189.94, p = 0 (floor),
#   rate ratio 4.38%/2.41% = 1.82x.
# Problems:
#   - Edges within a terminal segment share endpoints; adjacent edges along
#     the same maximal degree-2 path are part of the same "segment" and any
#     OMIT in one is highly correlated with OMITs in its neighbors. Pearson
#     chi-square assumes independence; with n ~ 1.4M the effective n is
#     orders of magnitude smaller, so chi^2 is inflated and p is a meaningless
#     floor.
#   - The discovery is a rate-ratio claim (~1.82x), not a p<.001 claim;
#     no CI on the ratio is reported in the original output.
# Correction:
#   1. Keep the same (terminal/internal) partition and the same OMIT label.
#   2. Report rate ratio with a CLUSTER-BOOTSTRAP 95% CI over GT skeletons
#      (each whole skeleton is a cluster — segments are nested within).
#   3. Report a cluster-permutation p where (terminal/internal) labels are
#      shuffled within each skeleton.
# ============================================================
import os
import glob
import pickle
import gc
import numpy as np
from collections import defaultdict
from scipy.stats import chi2_contingency


def get_edge_classifications_per_skel(gt, edge_error):
    """Return per-edge (skeleton_id, is_terminal, is_omit) lists."""
    edges_list = list(gt.edges)
    edge_to_error = {frozenset([u, v]): int(edge_error[i]) for i, (u, v) in enumerate(edges_list)}
    deg = dict(gt.degree())
    visited_edges = set()
    per_skel = defaultdict(lambda: {"t_tot": 0, "t_om": 0, "i_tot": 0, "i_om": 0,
                                    "edges": []})  # edges = list of (is_terminal, is_omit)
    for u0, v0 in edges_list:
        edge_fs0 = frozenset([u0, v0])
        if edge_fs0 in visited_edges:
            continue
        path_edges = [edge_fs0]
        visited_edges.add(edge_fs0)
        visited_nodes = {u0, v0}
        # Walk from v0
        curr = v0; prev = u0
        while deg[curr] == 2:
            neighbors = list(gt.neighbors(curr))
            next_node = neighbors[0] if neighbors[0] != prev else neighbors[1]
            if next_node in visited_nodes:
                break
            visited_nodes.add(next_node)
            ef = frozenset([curr, next_node])
            path_edges.append(ef); visited_edges.add(ef)
            prev = curr; curr = next_node
        end1 = curr
        # Walk from u0
        curr = u0; prev = v0
        while deg[curr] == 2:
            neighbors = list(gt.neighbors(curr))
            next_node = neighbors[0] if neighbors[0] != prev else neighbors[1]
            if next_node in visited_nodes:
                break
            visited_nodes.add(next_node)
            ef = frozenset([curr, next_node])
            path_edges.append(ef); visited_edges.add(ef)
            prev = curr; curr = next_node
        end2 = curr
        is_terminal = 1 if (deg[end1] == 1) or (deg[end2] == 1) else 0
        # Assign skeleton from any endpoint of u0
        try:
            skel = gt.node_segment_id(u0)
        except Exception:
            skel = u0
        for e in path_edges:
            is_omit = 1 if edge_to_error[e] == 2 else 0
            d = per_skel[skel]
            if is_terminal:
                d["t_tot"] += 1
                d["t_om"] += is_omit
            else:
                d["i_tot"] += 1
                d["i_om"] += is_omit
            d["edges"].append((is_terminal, is_omit))
    return per_skel


search_paths = ["../*_add.pkl", "../cache/*_add.pkl", "*_add.pkl", "cache/*_add.pkl"]
files = []
for pat in search_paths:
    files.extend(glob.glob(pat))
files = sorted(set(files))

global_per_skel = defaultdict(lambda: {"t_tot": 0, "t_om": 0, "i_tot": 0, "i_om": 0, "edges": []})

for path in files:
    with open(path, "rb") as f:
        payload = pickle.load(f)
    gt = payload["gt_graph"]
    edge_error = np.asarray(payload["gt_edge_error"])
    per_skel = get_edge_classifications_per_skel(gt, edge_error)
    for s, d in per_skel.items():
        gd = global_per_skel[(path, s)]
        gd["t_tot"] += d["t_tot"]; gd["t_om"] += d["t_om"]
        gd["i_tot"] += d["i_tot"]; gd["i_om"] += d["i_om"]
        gd["edges"].extend(d["edges"])
    del payload, gt, edge_error, per_skel
    gc.collect()

t_tot = sum(d["t_tot"] for d in global_per_skel.values())
t_om = sum(d["t_om"] for d in global_per_skel.values())
i_tot = sum(d["i_tot"] for d in global_per_skel.values())
i_om = sum(d["i_om"] for d in global_per_skel.values())

t_rate = t_om / t_tot if t_tot > 0 else 0
i_rate = i_om / i_tot if i_tot > 0 else 0
rate_ratio = t_rate / i_rate if i_rate > 0 else float("inf")

# Recorded chi-square for direct comparison
chi2_p, p_p, _, _ = chi2_contingency([[t_om, t_tot - t_om], [i_om, i_tot - i_om]])

# Cluster bootstrap CI on rate ratio (clusters = skeletons)
skels = list(global_per_skel.keys())
rng = np.random.default_rng(20260618)
B = 1000
boot_ratios = np.empty(B)
for b in range(B):
    idx = rng.integers(0, len(skels), size=len(skels))
    tt = ot = it = oi = 0
    for j in idx:
        d = global_per_skel[skels[j]]
        tt += d["t_tot"]; ot += d["t_om"]
        it += d["i_tot"]; oi += d["i_om"]
    if tt > 0 and it > 0 and oi > 0:
        boot_ratios[b] = (ot / tt) / (oi / it)
    else:
        boot_ratios[b] = np.nan
br_clean = boot_ratios[np.isfinite(boot_ratios)]
ci_lo, ci_hi = float(np.percentile(br_clean, 2.5)), float(np.percentile(br_clean, 97.5))

# Cluster permutation p: within each skeleton, shuffle (terminal/internal) labels
n_perm = 500
perm_diffs = np.empty(n_perm)
obs_diff = t_rate - i_rate
for p in range(n_perm):
    pt_t_om = 0; pt_t_tot = 0; pt_i_om = 0; pt_i_tot = 0
    for s, d in global_per_skel.items():
        if not d["edges"]:
            continue
        arr = np.asarray(d["edges"], dtype=np.int8)  # cols: is_terminal, is_omit
        if arr.shape[0] < 2:
            continue
        shuffled_term = rng.permutation(arr[:, 0])
        om = arr[:, 1]
        pt_t_om += int(om[shuffled_term == 1].sum())
        pt_t_tot += int((shuffled_term == 1).sum())
        pt_i_om += int(om[shuffled_term == 0].sum())
        pt_i_tot += int((shuffled_term == 0).sum())
    p_t = pt_t_om / pt_t_tot if pt_t_tot > 0 else 0
    p_i = pt_i_om / pt_i_tot if pt_i_tot > 0 else 0
    perm_diffs[p] = p_t - p_i
perm_p = float((np.abs(perm_diffs) >= abs(obs_diff)).sum() + 1) / (n_perm + 1)

print("=== H64 corrected: cluster-aware terminal-vs-internal OMIT rate ===")
print(f"Skeletons used as clusters: {len(skels)}")
print(f"Terminal edges: {t_tot}, OMIT rate: {100*t_rate:.4f}% ({t_om}/{t_tot})")
print(f"Internal edges: {i_tot}, OMIT rate: {100*i_rate:.4f}% ({i_om}/{i_tot})")
print(f"Rate ratio (terminal/internal): {rate_ratio:.4f}")
print(f"Cluster bootstrap 95% CI on rate ratio: [{ci_lo:.4f}, {ci_hi:.4f}]  (B={B})")
print(f"Cluster permutation p (within-skeleton terminal-label shuffle, n_perm={n_perm}): {perm_p:.4f}")
print(f"[recorded for comparison] Pearson chi^2 = {chi2_p:.4f}, p = {p_p:.4e}  (recorded 4189.94 / p=0)")
