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
# H48 / Entry 11 — CORRECTED TEST
# Original test: chi-square on a 2x2 transition table computed by counting each
#   adjacency twice (undirected edge (A,B) counted as both (A,B) and (B,A)).
#   Recorded chi^2 = 2,226,077.87 ("p ~ 0 floor"); marginal P(OMIT)=0.0317,
#   conditional P(OMIT | OMIT neighbor) = 0.8908, ratio 28.09x.
# Problems:
#   - Reciprocal counting exactly doubles every cell, which roughly doubles
#     chi^2 — the recorded test statistic is artefactually inflated.
#   - At nominal n in the millions the p is a floor; the EFFECT SIZE (ratio
#     28x vs marginal) is the meaningful quantity, but the test was not built
#     to provide a CI on that ratio.
#   - Within-skeleton clustering of OMITs is the very phenomenon being tested,
#     so a simple chi^2 of independence over reciprocal-counted unordered
#     pairs is doubly invalid as an inference about clustering "above
#     baseline".
# Correction:
#   1. Count each UNORDERED adjacent edge pair exactly ONCE (no reciprocal
#      double-counting) — build a 2x2 of (state_i, state_j) where (i,j) is an
#      unordered pair of adjacent edges at a shared node, with i<j in some
#      arbitrary ordering.
#   2. Compute the rate ratio P(OMIT | OMIT neighbor) / P(OMIT) on the deduped
#      table.
#   3. Cluster bootstrap CI on the ratio over skeletons (each GT skeleton is
#      a cluster).
#   4. Skeleton-permutation p-value: shuffle the OMIT label of each edge
#      WITHIN its skeleton, then recompute the conditional / marginal ratio.
# ============================================================
import glob
import pickle
import gc
import numpy as np
from collections import defaultdict
from scipy.stats import chi2_contingency

files = []
for pattern in ["../*_add.pkl", "../cache/*_add.pkl", "cache/*_add.pkl", "*_add.pkl"]:
    files = glob.glob(pattern)
    if files:
        break

per_skel = defaultdict(lambda: {
    "n_edge": 0, "n_omit": 0,
    "T": np.zeros((2, 2), dtype=np.int64),  # unordered (i<j) adjacent-edge pairs
    "edges": [],  # list of (edge_idx, is_omit) for permutation
    "adj": []     # list of (edge_idx_i, edge_idx_j) unordered with i<j
})

for fpath in sorted(files):
    with open(fpath, "rb") as f:
        payload = pickle.load(f)
    gt = payload["gt_graph"]
    edge_error = np.asarray(payload["gt_edge_error"])
    edges = list(gt.edges)
    edge_states = np.array([1 if int(e) == 2 else 0 for e in edge_error], dtype=np.int8)

    # Map each node to its incident edge indices
    node_to_edge_idx = defaultdict(list)
    for k, (u, v) in enumerate(edges):
        node_to_edge_idx[u].append(k)
        node_to_edge_idx[v].append(k)

    # Build UNORDERED adjacency between edges (no reciprocal double-count)
    skel_of_edge = {}
    for k, (u, v) in enumerate(edges):
        try:
            skel_of_edge[k] = gt.node_segment_id(u)
        except Exception:
            skel_of_edge[k] = u

    pair_seen = set()
    for node, eidxs in node_to_edge_idx.items():
        if len(eidxs) < 2:
            continue
        for a in range(len(eidxs)):
            for b in range(a + 1, len(eidxs)):
                ei, ej = eidxs[a], eidxs[b]
                if ei > ej:
                    ei, ej = ej, ei
                key = (ei, ej)
                if key in pair_seen:
                    continue
                pair_seen.add(key)
                si = int(edge_states[ei]); sj = int(edge_states[ej])
                skel = skel_of_edge[ei]
                per_skel[skel]["T"][si, sj] += 1
                if si != sj:
                    per_skel[skel]["T"][sj, si] += 0  # store symmetric form below
                per_skel[skel]["adj"].append((ei, ej))

    for k, st in enumerate(edge_states):
        skel = skel_of_edge[k]
        per_skel[skel]["n_edge"] += 1
        per_skel[skel]["n_omit"] += int(st)
        per_skel[skel]["edges"].append((k, int(st)))

    del payload, gt, edge_error, edges, edge_states, node_to_edge_idx, pair_seen
    gc.collect()

# Symmetrize T cleanly for printing (T_ij + T_ji counted once)
def symmetric_pair_table(T):
    out = np.zeros((2, 2), dtype=np.int64)
    out[0, 0] = T[0, 0]
    out[1, 1] = T[1, 1]
    out[0, 1] = T[0, 1] + T[1, 0]
    out[1, 0] = out[0, 1]
    return out

pooled_T = np.zeros((2, 2), dtype=np.int64)
total_edges = 0; total_omit = 0
for s, d in per_skel.items():
    pooled_T += d["T"]
    total_edges += d["n_edge"]
    total_omit += d["n_omit"]

# For the conditional/marginal calculation, build a row-normalised conditional:
# P(OMIT | neighbor is OMIT). With unordered pairs we have T[0,1] / 2 of each
# directed permutation; the conditional must be over directed adjacencies so
# we count both directions exactly once.
n00 = int(pooled_T[0, 0]); n11 = int(pooled_T[1, 1])
n01 = int(pooled_T[0, 1]); n10 = int(pooled_T[1, 0])
n_unordered_diff = n01 + n10  # already unordered

# Directed adjacency counts (each unordered pair contributes two directed edges)
dir_00 = 2 * n00
dir_11 = 2 * n11
dir_01 = n_unordered_diff
dir_10 = n_unordered_diff

marg_p_omit = total_omit / total_edges if total_edges > 0 else 0
# Conditional: of all directed pairs where source is OMIT, fraction with target OMIT
denom = dir_10 + dir_11  # directed pairs from OMIT
cond_p = dir_11 / denom if denom > 0 else 0
ratio = cond_p / marg_p_omit if marg_p_omit > 0 else float("inf")

# Chi-square on the UNORDERED 2x2 (no double-count) for direct comparison
table_unordered = np.array([[n00, n01 + n10], [n10 + n01, n11]])
# Build a proper test-of-independence 2x2 (we use the symmetric merged form):
# rows = state_i, cols = state_j
table_test = np.array([[n00, n01 + n10], [0, n11]], dtype=np.int64)
# A proper chi^2 of independence needs a contingency on (state_i, state_j) with
# both i and j enumerated once; using the unordered (i<j) form yields:
chi2_unord, p_unord, dof_unord, _ = chi2_contingency(np.array([
    [n00, n01],
    [n10, n11]
]) + 1)  # +1 smoothing to avoid empty cell on edge cases
# Note: above table is identical to the deduped version of the recorded test.

# Cluster bootstrap CI on conditional/marginal ratio
skels = list(per_skel.keys())
rng = np.random.default_rng(20260618)
B = 1000
boot_ratios = np.empty(B)
for b in range(B):
    idx = rng.integers(0, len(skels), size=len(skels))
    Tb = np.zeros((2, 2), dtype=np.int64)
    n_e = 0; n_o = 0
    for j in idx:
        d = per_skel[skels[j]]
        Tb += d["T"]
        n_e += d["n_edge"]
        n_o += d["n_omit"]
    dir_11b = 2 * Tb[1, 1]
    dir_10b = Tb[0, 1] + Tb[1, 0]
    p_o = n_o / n_e if n_e > 0 else 0
    p_c = dir_11b / (dir_10b + dir_11b) if (dir_10b + dir_11b) > 0 else 0
    boot_ratios[b] = (p_c / p_o) if p_o > 0 else np.nan
br_clean = boot_ratios[np.isfinite(boot_ratios)]
ci_lo, ci_hi = float(np.percentile(br_clean, 2.5)), float(np.percentile(br_clean, 97.5))

# Skeleton-permutation: shuffle OMIT label of edges WITHIN each skeleton;
# recompute the conditional/marginal ratio. Preserves per-skeleton OMIT base
# rate but breaks within-skeleton clustering of OMITs.
n_perm = 300
perm_ratios = np.empty(n_perm)
for p in range(n_perm):
    # Build a global state per edge using within-skeleton shuffle
    perm_state = {}
    for s, d in per_skel.items():
        edge_ids = [e for e, _ in d["edges"]]
        states = np.array([st for _, st in d["edges"]], dtype=np.int8)
        rng.shuffle(states)
        for ei, st in zip(edge_ids, states):
            perm_state[ei] = int(st)
    # Recompute the pair table from the FIXED adjacency list with shuffled states
    Tp = np.zeros((2, 2), dtype=np.int64)
    for s, d in per_skel.items():
        for (ei, ej) in d["adj"]:
            si = perm_state[ei]; sj = perm_state[ej]
            Tp[si, sj] += 1
    dir_11p = 2 * Tp[1, 1]
    dir_10p = Tp[0, 1] + Tp[1, 0]
    p_o = marg_p_omit
    denomp = dir_10p + dir_11p
    perm_ratios[p] = (dir_11p / denomp / p_o) if (denomp > 0 and p_o > 0) else np.nan
pr_clean = perm_ratios[np.isfinite(perm_ratios)]
perm_p = float((pr_clean >= ratio).sum() + 1) / (pr_clean.size + 1)

print("=== H48 corrected: dedup + cluster-aware OMIT-streak test ===")
print("Unordered pair table (state_i, state_j):")
print(np.array([[n00, n01], [n10, n11]]))
print(f"Marginal P(OMIT) = {marg_p_omit:.4f}")
print(f"P(OMIT | neighbor=OMIT) = {cond_p:.4f}  [directed adjacency, no double-count]")
print(f"Ratio (conditional/marginal) = {ratio:.4f}")
print(f"Cluster bootstrap 95% CI on ratio: [{ci_lo:.4f}, {ci_hi:.4f}]  (B={B}, n_skel={len(skels)})")
print(f"Cluster-permutation p (within-skeleton shuffle, n_perm={n_perm}): {perm_p:.4f}")
print(f"Deduped chi-square (Pearson) on (state_i, state_j) table: chi2={chi2_unord:.4f}, p={p_unord:.4e}")
print(f"[recorded for comparison] reciprocal-counted chi^2 = 2,226,077.87 (p=0 floor); ratio 28.09")
