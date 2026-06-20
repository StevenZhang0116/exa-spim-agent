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
# H39 / Entry 10 — CORRECTED TEST
# Original test: descriptive only — per-neuron baseline -> after-cut percentage
#   of merged edges; reported a single point estimate "~86% reduction" with
#   no inferential test and no uncertainty quantification.
# Problems:
#   - No paired Wilcoxon / sign test on the per-neuron deltas.
#   - No CI on the global reduction proportion; the n=12-neuron headline
#     "~86% reduction" is presented as a property of the intervention without
#     uncertainty.
#   - The reduction is heavily influenced by two extreme neurons (N013, N018);
#     a small-n point estimate without CI is misleading.
# Correction:
#   1. Paired Wilcoxon signed-rank on (baseline %merged) vs (after-cut %merged)
#      across neurons (paired clusters).
#   2. Paired Wilcoxon on (baseline edge-accuracy) vs (after-cut edge-accuracy).
#   3. Cluster bootstrap 95% CI on the global "% merge-edge reduction".
#   4. Report rank-biserial effect size for the paired comparison.
# ============================================================
import os
import sys
import pickle
import numpy as np
import networkx as nx
import scipy.spatial
from collections import defaultdict
from scipy.stats import wilcoxon

import agentic_neuron_proofreader
from agentic_neuron_proofreader.data_modules import canonical_labeling as cl

dataset_path = None
for root, dirs, files in os.walk(".."):
    for file in files:
        if file.endswith("_add.pkl"):
            dataset_path = os.path.join(root, file)
            break
    if dataset_path:
        break
if not dataset_path:
    print("Dataset not found.")
    sys.exit(1)

print(f"Loading {dataset_path}...")
with open(dataset_path, "rb") as f:
    payload = pickle.load(f)

gt = payload["gt_graph"]
node_label = np.asarray(payload["gt_node_canonical_label"])
edge_error = np.asarray(payload["gt_edge_error"])
merge_labels = set(int(x) for x in payload["gt_merge_labels"])
merge_sites = payload["gt_merge_sites"]
fragments_graph = payload["fragments_graph"]

EDGE_CORRECT, EDGE_SPLIT, EDGE_OMIT, EDGE_MERGED = 0, 1, 2, 3

def compute_metrics(gt_graph, node_lbls, edge_errs, m_labels):
    edges = list(gt_graph.edges)
    neuron_class = defaultdict(lambda: defaultdict(int))
    for k, (i, j) in enumerate(edges):
        neuron_class[gt_graph.node_segment_id(i)][int(edge_errs[k])] += 1
    seg_neuron_counts = cl.segment_neuron_node_counts(gt_graph, node_lbls)
    neuron_merged = defaultdict(int)
    for lab in m_labels:
        if lab in seg_neuron_counts:
            for neuron, c in seg_neuron_counts[lab].items():
                neuron_merged[neuron] += max(c - 1, 0)
    metrics = {}
    for neuron, classes in neuron_class.items():
        tot = sum(classes.values())
        if tot == 0:
            continue
        metrics[neuron] = {
            "edge_acc": 100 * classes[EDGE_CORRECT] / tot,
            "pct_merged": 100 * neuron_merged[neuron] / tot,
            "tot_edges": tot,
        }
    return metrics

baseline_metrics = compute_metrics(gt, node_label, edge_error, merge_labels)

removed_nodes = set()
for site in merge_sites:
    xyz = site["xyz"]
    d, idx = fragments_graph.kdtree.query(xyz)
    removed_nodes.add(int(idx))

affected_components = defaultdict(list)
for idx in removed_nodes:
    comp_id = fragments_graph.node_component_id[idx]
    affected_components[comp_id].append(idx)

pseudo_id_map = {}
pseudo_id_threshold = int(np.max(node_label)) + 1
next_pseudo_id = pseudo_id_threshold
cuts_that_broke = 0
for comp_id, target_nodes in affected_components.items():
    comp_nodes = np.where(fragments_graph.node_component_id == comp_id)[0]
    subgraph = fragments_graph.subgraph(comp_nodes).copy()
    subgraph.remove_nodes_from(target_nodes)
    ccs = list(nx.connected_components(subgraph))
    if len(ccs) > 1:
        cuts_that_broke += 1
        for cc in ccs:
            for n in cc:
                pseudo_id_map[n] = next_pseudo_id
            next_pseudo_id += 1

new_node_label = node_label.copy()
remapped_count = 0
gt_nodes_to_remap = [n for n in gt.nodes if node_label[n] in merge_labels]
for n in gt_nodes_to_remap:
    xyz = gt.node_xyz[n]
    dists, idxs = fragments_graph.kdtree.query(xyz, k=10)
    idxs = np.atleast_1d(idxs)
    for idx in idxs:
        idx = int(idx)
        if idx not in removed_nodes:
            if idx in pseudo_id_map:
                new_node_label[n] = pseudo_id_map[idx]
                remapped_count += 1
            break

seg_neuron_counts_new = cl.segment_neuron_node_counts(gt, new_node_label)
seg_neuron_counts_old = cl.segment_neuron_node_counts(gt, node_label)
new_merge_labels = set()
for seg, neuron_counts in seg_neuron_counts_new.items():
    if seg == 0:
        continue
    if len(neuron_counts) < 2:
        continue
    if seg >= pseudo_id_threshold:
        count_large = sum(1 for c in neuron_counts.values() if c > 50)
        if count_large >= 2:
            new_merge_labels.add(seg)
    else:
        old_counts = seg_neuron_counts_old.get(seg, {})
        changed = any(neuron_counts.get(neuron, 0) != count for neuron, count in old_counts.items())
        if changed:
            count_large = sum(1 for c in neuron_counts.values() if c > 50)
            if count_large >= 2:
                new_merge_labels.add(seg)
        else:
            if seg in merge_labels:
                new_merge_labels.add(seg)

new_edge_error = np.zeros_like(edge_error)
edges = list(gt.edges)
for k, (u, v) in enumerate(edges):
    lbl_u = new_node_label[u]; lbl_v = new_node_label[v]
    if lbl_u == 0 or lbl_v == 0:
        new_edge_error[k] = EDGE_OMIT
    elif lbl_u != lbl_v:
        new_edge_error[k] = EDGE_SPLIT
    else:
        if lbl_u in new_merge_labels:
            new_edge_error[k] = EDGE_MERGED
        else:
            new_edge_error[k] = EDGE_CORRECT

new_metrics = compute_metrics(gt, new_node_label, new_edge_error, new_merge_labels)

neurons = sorted(baseline_metrics.keys())
base_merg = np.array([baseline_metrics[n]['pct_merged'] for n in neurons])
new_merg = np.array([new_metrics[n]['pct_merged'] for n in neurons])
base_acc = np.array([baseline_metrics[n]['edge_acc'] for n in neurons])
new_acc = np.array([new_metrics[n]['edge_acc'] for n in neurons])
# Track raw merged-edge counts and totals for an exact total-reduction CI
total_edges_arr = np.array([baseline_metrics[n]['tot_edges'] for n in neurons], dtype=np.int64)
base_merg_count = (base_merg / 100.0) * total_edges_arr
new_merg_count = (new_merg / 100.0) * total_edges_arr

print(f"\nNeurons analyzed: {len(neurons)}")
print(f"Mean baseline %merged: {base_merg.mean():.4f}%")
print(f"Mean after-cut %merged: {new_merg.mean():.4f}%")
print(f"Mean baseline edge-accuracy: {base_acc.mean():.4f}%")
print(f"Mean after-cut edge-accuracy: {new_acc.mean():.4f}%")

# Paired Wilcoxon on per-neuron deltas
try:
    w_merg, p_merg = wilcoxon(base_merg, new_merg, alternative='greater')  # baseline > after
except Exception as e:
    w_merg, p_merg = float("nan"), None
    print(f"Wilcoxon (pct_merged) failed: {e}")
try:
    w_acc, p_acc = wilcoxon(new_acc, base_acc, alternative='greater')  # after > baseline
except Exception as e:
    w_acc, p_acc = float("nan"), None
    print(f"Wilcoxon (edge_acc) failed: {e}")

# Rank-biserial effect size for paired tests (matched-pairs version)
def matched_rbs(x, y):
    diffs = x - y
    diffs = diffs[diffs != 0]
    if diffs.size == 0:
        return float("nan")
    ranks = np.argsort(np.argsort(np.abs(diffs))) + 1
    Rp = ranks[diffs > 0].sum()
    Rn = ranks[diffs < 0].sum()
    return float((Rp - Rn) / ranks.sum())

rbs_merg = matched_rbs(base_merg, new_merg)
rbs_acc = matched_rbs(new_acc, base_acc)

# Cluster bootstrap CI on global "% merge-edge reduction"
rng = np.random.default_rng(20260618)
B = 5000
boot_red = np.empty(B)
boot_acc_gain = np.empty(B)
boot_pct_red = np.empty(B)
for b in range(B):
    idx = rng.integers(0, len(neurons), size=len(neurons))
    bm = base_merg_count[idx].sum(); nm = new_merg_count[idx].sum()
    if bm > 0:
        boot_red[b] = (bm - nm) / bm  # fraction of merged edges removed
    else:
        boot_red[b] = np.nan
    boot_acc_gain[b] = float(new_acc[idx].mean() - base_acc[idx].mean())
    if base_merg[idx].sum() > 0:
        boot_pct_red[b] = float(1 - new_merg[idx].mean() / base_merg[idx].mean())
    else:
        boot_pct_red[b] = np.nan
br_clean = boot_red[np.isfinite(boot_red)]
ag_clean = boot_acc_gain[np.isfinite(boot_acc_gain)]
pr_clean = boot_pct_red[np.isfinite(boot_pct_red)]
ci_red = (float(np.percentile(br_clean, 2.5)), float(np.percentile(br_clean, 97.5)))
ci_acc = (float(np.percentile(ag_clean, 2.5)), float(np.percentile(ag_clean, 97.5)))
ci_pctred = (float(np.percentile(pr_clean, 2.5)), float(np.percentile(pr_clean, 97.5)))

total_red = (base_merg_count.sum() - new_merg_count.sum()) / max(base_merg_count.sum(), 1e-12)

print("=== H39 corrected: paired-test + bootstrap CI on intervention effect ===")
print(f"Total merged edges baseline: {int(base_merg_count.sum())}, after-cut: {int(new_merg_count.sum())}")
print(f"Total fraction of merged edges removed: {total_red:.4f}")
print(f"Cluster-bootstrap 95% CI on fraction removed (paired across neurons): "
      f"[{ci_red[0]:.4f}, {ci_red[1]:.4f}]  (B={B})")
print(f"Cluster-bootstrap 95% CI on per-neuron mean %merged ratio (1 - mean_after/mean_base): "
      f"[{ci_pctred[0]:.4f}, {ci_pctred[1]:.4f}]")
print(f"Paired Wilcoxon (baseline > after, pct_merged): W={w_merg}, p={p_merg}")
print(f"Matched-pairs rank-biserial effect for pct_merged: {rbs_merg:.4f}")
print(f"Paired Wilcoxon (after > baseline, edge_acc): W={w_acc}, p={p_acc}")
print(f"Matched-pairs rank-biserial effect for edge_acc: {rbs_acc:.4f}")
print(f"Cluster-bootstrap 95% CI on mean per-neuron edge-accuracy gain (%pts): "
      f"[{ci_acc[0]:.4f}, {ci_acc[1]:.4f}]")
print(f"[recorded for comparison] mean %merged 13.25 -> 1.83 (~86% reduction), "
      f"mean edge_acc 82.27 -> 93.66 (no test, no CI)")
