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
# H47 / Entry 1 — CORRECTED TEST
# Original test: statsmodels Logit on n=1214 (1,072 positives / 142 negatives)
#   with predictors x1=dist, x2=angle, x3=dist*angle. Recorded interaction
#   coef = -5.96, p = 0.014 (just below 0.05), LLR p = 1.63e-17.
# Problems:
#   - 7.5:1 class imbalance with only 142 negatives makes the SE on the
#     interaction term sensitive to small reshuffles; a single Wald p ~ 0.014
#     is a fragile claim and the verifier requested an effect-size CI for
#     the interaction.
#   - No CI on the interaction coefficient is reported; the original test
#     also did not include a likelihood-ratio comparison of (interaction
#     model) vs (additive-only model).
# Correction:
#   1. Keep the same Logit model and the same data (no rebalancing).
#   2. Add a likelihood-ratio test comparing additive (x1, x2) vs interaction
#      (x1, x2, x3) — this tests the SPECIFIC claim that an interaction is
#      needed beyond an additive distance + angle model.
#   3. Add a bootstrap 95% CI on the interaction coefficient (stratified by
#      class so the imbalance is preserved in resamples).
#   4. Print the corrected LR-test p, the bootstrap CI on x3, and a
#      permutation p-value where the class label is shuffled while preserving
#      class proportions.
# ============================================================
import re
import pickle
import gc
import glob
import numpy as np
from collections import defaultdict
import statsmodels.api as sm
from scipy.spatial import cKDTree
from scipy.stats import chi2

def get_xyz(fg, n):
    if hasattr(fg, 'node_xyz'):
        return np.array(fg.node_xyz[n])
    return np.array(fg.nodes[n]['node_xyz'])

def get_seg_id(fg, n):
    try:
        comp_id = fg.node_component_id[n] if hasattr(fg, 'node_component_id') else fg.nodes[n]['node_component_id']
        swc_arr = fg.component_id_to_swc_id if hasattr(fg, 'component_id_to_swc_id') else fg.graph.get('component_id_to_swc_id')
        swc_str = str(swc_arr[comp_id])
    except Exception:
        swc_str = str(fg.nodes[n].get('component_id_to_swc_id', ''))
    swc_str = swc_str.replace('.swc', '')
    try:
        return int(float(swc_str))
    except ValueError:
        pass
    m = re.search(r'(\d+)', swc_str)
    if m:
        return int(m.group(1))
    return -1

def get_terminal_vector(fg, term_node, dist_th=5.0):
    curr = term_node
    prev = None
    path_len = 0.0
    while True:
        neighbors = list(fg.neighbors(curr))
        if prev is not None and prev in neighbors:
            neighbors.remove(prev)
        if len(neighbors) != 1:
            break
        next_node = neighbors[0]
        d = np.linalg.norm(get_xyz(fg, curr) - get_xyz(fg, next_node))
        path_len += d
        prev = curr
        curr = next_node
        if path_len >= dist_th:
            break
    vec = get_xyz(fg, term_node) - get_xyz(fg, curr)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    else:
        vec = np.array([1.0, 0.0, 0.0])
    return vec

files = glob.glob("../dataset_cache_*_add.pkl") or glob.glob("../*_add.pkl") or glob.glob("dataset_cache_*_add.pkl")

X_dist, X_angle, Y = [], [], []
for path in files:
    print(f"Processing {path}...")
    with open(path, "rb") as f:
        payload = pickle.load(f)
    gt = payload["gt_graph"]
    fg = payload["fragments_graph"]
    node_label = np.asarray(payload["gt_node_canonical_label"])
    edge_error = np.asarray(payload["gt_edge_error"])
    nodes_list = list(gt.nodes)
    node_to_idx = {n: i for i, n in enumerate(nodes_list)}
    seg_to_gt = {}
    for i, n in enumerate(nodes_list):
        try:
            lab = int(node_label[n])
        except IndexError:
            lab = int(node_label[i])
        if lab != 0:
            seg_to_gt[lab] = gt.node_segment_id(n)
    split_pairs = set()
    edges_list = list(gt.edges)
    for k, (u, v) in enumerate(edges_list):
        if int(edge_error[k]) == 1:
            try:
                lab_u = int(node_label[u]); lab_v = int(node_label[v])
            except IndexError:
                idx_u = node_to_idx[u]; idx_v = node_to_idx[v]
                lab_u = int(node_label[idx_u]); lab_v = int(node_label[idx_v])
            if lab_u != 0 and lab_v != 0 and lab_u != lab_v:
                split_pairs.add(tuple(sorted([lab_u, lab_v])))
    degrees = dict(fg.degree())
    terminals = [n for n, d in degrees.items() if d == 1]
    seg_terminals = defaultdict(list)
    term_vectors = {}
    for t in terminals:
        seg_id = get_seg_id(fg, t)
        if seg_id != -1:
            seg_terminals[seg_id].append(t)
        term_vectors[t] = get_terminal_vector(fg, t)
    for (lab_u, lab_v) in split_pairs:
        terms_u = seg_terminals.get(lab_u, [])
        terms_v = seg_terminals.get(lab_v, [])
        best_dist = float('inf'); best_angle = None
        for tu in terms_u:
            for tv in terms_v:
                d = np.linalg.norm(get_xyz(fg, tu) - get_xyz(fg, tv))
                if d < best_dist:
                    best_dist = d
                    vu = term_vectors[tu]; vv = term_vectors[tv]
                    dot = np.clip(np.dot(vu, -vv), -1.0, 1.0)
                    best_angle = np.degrees(np.arccos(dot))
        if best_dist <= 15.0 and best_angle is not None:
            X_dist.append(best_dist); X_angle.append(best_angle); Y.append(1)
    term_coords = np.array([get_xyz(fg, t) for t in terminals])
    if len(term_coords) > 0:
        tree = cKDTree(term_coords)
        pairs = tree.query_pairs(15.0)
        for i, j in pairs:
            t_a = terminals[i]; t_b = terminals[j]
            seg_a = get_seg_id(fg, t_a); seg_b = get_seg_id(fg, t_b)
            if seg_a == -1 or seg_b == -1 or seg_a == seg_b:
                continue
            pt = tuple(sorted([seg_a, seg_b]))
            if pt in split_pairs:
                continue
            gt_a = seg_to_gt.get(seg_a); gt_b = seg_to_gt.get(seg_b)
            if gt_a != gt_b and (gt_a is not None or gt_b is not None):
                d = np.linalg.norm(get_xyz(fg, t_a) - get_xyz(fg, t_b))
                va = term_vectors[t_a]; vb = term_vectors[t_b]
                dot = np.clip(np.dot(va, -vb), -1.0, 1.0)
                angle = np.degrees(np.arccos(dot))
                X_dist.append(d); X_angle.append(angle); Y.append(0)
    del payload, gt, fg
    gc.collect()

X_dist = np.array(X_dist); X_angle = np.array(X_angle); Y = np.array(Y)
print(f"Collected {(Y == 1).sum()} positives and {(Y == 0).sum()} negatives.")

if len(np.unique(Y)) < 2:
    print("Not enough classes."); sys_exit = True
else:
    X_dist_n = X_dist / 15.0
    X_angle_n = X_angle / 180.0
    X_inter_n = X_dist_n * X_angle_n

    # FULL model (with interaction)
    X_full = sm.add_constant(np.column_stack((X_dist_n, X_angle_n, X_inter_n)))
    model_full = sm.Logit(Y, X_full)
    res_full = model_full.fit(method='bfgs', disp=0)
    llf_full = res_full.llf
    coef_inter = float(res_full.params[3])
    p_inter = float(res_full.pvalues[3])
    # Wald CI on x3 from statsmodels
    conf_full = res_full.conf_int(alpha=0.05)
    wald_ci_lo, wald_ci_hi = float(conf_full[3][0]), float(conf_full[3][1])

    # REDUCED model (additive only)
    X_add = sm.add_constant(np.column_stack((X_dist_n, X_angle_n)))
    model_add = sm.Logit(Y, X_add)
    res_add = model_add.fit(method='bfgs', disp=0)
    llf_add = res_add.llf

    # Likelihood-ratio test on the interaction term
    lr_stat = 2.0 * (llf_full - llf_add)
    lr_p = float(chi2.sf(lr_stat, df=1))

    # Stratified bootstrap CI on x3 (preserve per-class n)
    rng = np.random.default_rng(20260618)
    pos_idx = np.where(Y == 1)[0]; neg_idx = np.where(Y == 0)[0]
    B = 2000
    boot_x3 = np.empty(B)
    for b in range(B):
        ipos = rng.choice(pos_idx, size=len(pos_idx), replace=True)
        ineg = rng.choice(neg_idx, size=len(neg_idx), replace=True)
        idx = np.concatenate([ipos, ineg])
        try:
            res_b = sm.Logit(Y[idx], X_full[idx]).fit(method='bfgs', disp=0)
            boot_x3[b] = float(res_b.params[3])
        except Exception:
            boot_x3[b] = np.nan
    boot_clean = boot_x3[np.isfinite(boot_x3)]
    boot_ci_lo, boot_ci_hi = float(np.percentile(boot_clean, 2.5)), float(np.percentile(boot_clean, 97.5))

    # Permutation p-value on Wald z for x3
    n_perm = 1000
    perm_x3 = np.empty(n_perm)
    for p in range(n_perm):
        Y_perm = rng.permutation(Y)
        try:
            res_p = sm.Logit(Y_perm, X_full).fit(method='bfgs', disp=0)
            perm_x3[p] = float(res_p.params[3])
        except Exception:
            perm_x3[p] = np.nan
    pp_clean = perm_x3[np.isfinite(perm_x3)]
    perm_p = float((np.abs(pp_clean) >= abs(coef_inter)).sum() + 1) / (pp_clean.size + 1)

    print("=== H47 corrected: LR test of interaction + bootstrap CI on x3 ===")
    print(f"n={len(Y)} ({(Y == 1).sum()} positives / {(Y == 0).sum()} negatives)")
    print(f"Full model LLF = {llf_full:.4f}; Additive model LLF = {llf_add:.4f}")
    print(f"Likelihood-ratio test of interaction (df=1): LR={lr_stat:.4f}, p={lr_p:.4e}")
    print(f"x3 (interaction) coef = {coef_inter:.4f}, Wald p = {p_inter:.4e}")
    print(f"x3 Wald 95% CI: [{wald_ci_lo:.4f}, {wald_ci_hi:.4f}]")
    print(f"x3 stratified bootstrap 95% CI: [{boot_ci_lo:.4f}, {boot_ci_hi:.4f}]  (B={B})")
    print(f"x3 permutation p (label-shuffled, n_perm={n_perm}): {perm_p:.4e}")
    print(f"[recorded for comparison] interaction coef=-5.96, Wald p=0.0139, LLR p=1.63e-17 "
          f"(LLR was vs intercept-only, NOT vs the additive-only model tested here)")
