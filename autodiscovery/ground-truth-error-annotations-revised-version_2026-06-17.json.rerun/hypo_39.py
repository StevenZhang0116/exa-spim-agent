# === RERUN_LOADING_BOOTSTRAP v2 ===
import os as _os, sys as _sys, subprocess as _sp_boot
# Ensure NumPy >= 2 (the pkl was written with NumPy 2.x). Do this BEFORE any
# import that brings in numpy transitively. Quiet, single-shot.
try:
    import numpy as _np_boot
    _ver = tuple(int(x) for x in _np_boot.__version__.split(".")[:2])
    if _ver < (2, 0):
        raise ImportError("need numpy>=2")
except Exception:
    _sp_boot.check_call([_sys.executable, "-m", "pip", "install", "-q", "--user", "numpy>=2,<2.3"])
    # invalidate caches so a fresh numpy 2.x is loaded
    import importlib as _il_boot
    if "numpy" in _sys.modules:
        del _sys.modules["numpy"]
import numpy as _np_chk
print("Loading dataset from:", _os.environ.get("RERUN_PKL", "<unset>"), "| numpy", _np_chk.__version__)

# Make any dataset-search return RERUN_PKL (the runner's open() patch then
# redirects the actual file open). Cover os.walk, Path.rglob, subprocess.getoutput.
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
# Short-circuit ALL in-script pip installs. The host environment already has
# numpy>=2, pandas, scipy, matplotlib, statsmodels, networkx, agentic_neuron_proofreader,
# etc. Re-running pip-install inside the subprocess only downgrades numpy and
# breaks the install, so we no-op every pip install here.
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

import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", package])

# Ensure all dependencies are installed before importing
install("https://github.com/AllenInstitute/agentic-neuron-proofreader/archive/refs/heads/main.zip")
for pkg in ["tensorstore", "psutil", "pandas", "tqdm", "networkx", "scipy", "matplotlib"]:
    install(pkg)

import os
import pickle
import numpy as np
import networkx as nx
from collections import defaultdict
import matplotlib.pyplot as plt
import scipy.spatial

import agentic_neuron_proofreader
from agentic_neuron_proofreader.data_modules import canonical_labeling as cl

# 1. Dataset Localization & Loading
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
        if tot == 0: continue
        metrics[neuron] = {
            "edge_acc": 100 * classes[EDGE_CORRECT] / tot,
            "pct_merged": 100 * neuron_merged[neuron] / tot,
            "tot_edges": tot,
        }
    return metrics

print("Computing baseline metrics...")
baseline_metrics = compute_metrics(gt, node_label, edge_error, merge_labels)

# 2. Use existing KDTree to target physical nodes
print("Querying KDTree for merge sites...")
removed_nodes = set()
for site in merge_sites:
    xyz = site["xyz"]
    d, idx = fragments_graph.kdtree.query(xyz)
    removed_nodes.add(int(idx))

print(f"Targeted {len(removed_nodes)} unique physical nodes from {len(merge_sites)} merge sites.")

# 3. Component Grouping and Breaking
affected_components = defaultdict(list)
for idx in removed_nodes:
    comp_id = fragments_graph.node_component_id[idx]
    affected_components[comp_id].append(idx)

print(f"Affected {len(affected_components)} unique fragment components.")

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

print(f"Successfully broke {cuts_that_broke} out of {len(affected_components)} targeted components.")

# 4. Remapping GT Nodes
print("Remapping GT nodes nearest to the broken fragments...")
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

print(f"Remapped {remapped_count} GT nodes to new pseudo-segments.")

# 5. Re-evaluating Merge Status
print("Re-evaluating canonical merge status for segments...")
seg_neuron_counts_new = cl.segment_neuron_node_counts(gt, new_node_label)
seg_neuron_counts_old = cl.segment_neuron_node_counts(gt, node_label)
new_merge_labels = set()

for seg, neuron_counts in seg_neuron_counts_new.items():
    if seg == 0: continue
    
    if len(neuron_counts) < 2:
        continue
        
    if seg >= pseudo_id_threshold:
        count_large = sum(1 for c in neuron_counts.values() if c > 50)
        if count_large >= 2:
            new_merge_labels.add(seg)
    else:
        old_counts = seg_neuron_counts_old.get(seg, {})
        changed = False
        for neuron, count in old_counts.items():
            if neuron_counts.get(neuron, 0) != count:
                changed = True
                break
                
        if changed:
            count_large = sum(1 for c in neuron_counts.values() if c > 50)
            if count_large >= 2:
                new_merge_labels.add(seg)
        else:
            if seg in merge_labels:
                new_merge_labels.add(seg)

print("Re-evaluating Ground-Truth edge errors...")
new_edge_error = np.zeros_like(edge_error)
edges = list(gt.edges)

for k, (u, v) in enumerate(edges):
    lbl_u = new_node_label[u]
    lbl_v = new_node_label[v]
    
    if lbl_u == 0 or lbl_v == 0:
        new_edge_error[k] = EDGE_OMIT
    elif lbl_u != lbl_v:
        new_edge_error[k] = EDGE_SPLIT
    else:
        if lbl_u in new_merge_labels:
            new_edge_error[k] = EDGE_MERGED
        else:
            new_edge_error[k] = EDGE_CORRECT

print("Computing adjusted metrics...")
new_metrics = compute_metrics(gt, new_node_label, new_edge_error, new_merge_labels)

# 6. Analysis and Delivery
neurons = sorted(baseline_metrics.keys())
print(f"\n{'Neuron':<20} | {'Base Acc':<8} | {'New Acc':<8} | {'Base Merg':<9} | {'New Merg':<9}")
print("-" * 65)

base_acc_list, new_acc_list = [], []
base_merg_list, new_merg_list = [], []

for n in neurons:
    b = baseline_metrics[n]
    a = new_metrics[n]
    print(f"{n:<20} | {b['edge_acc']:>8.2f} | {a['edge_acc']:>8.2f} | {b['pct_merged']:>9.2f} | {a['pct_merged']:>9.2f}")
    
    base_acc_list.append(b['edge_acc'])
    new_acc_list.append(a['edge_acc'])
    base_merg_list.append(b['pct_merged'])
    new_merg_list.append(a['pct_merged'])

mean_base_merg = np.mean(base_merg_list)
mean_new_merg = np.mean(new_merg_list)
mean_base_acc = np.mean(base_acc_list)
mean_new_acc = np.mean(new_acc_list)

print(f"\nGlobal Metrics (Averaged across {len(neurons)} neurons):")
print(f"Mean % Merged Edges: Baseline {mean_base_merg:.2f}% -> After Cut {mean_new_merg:.2f}%")
print(f"Mean Edge Accuracy : Baseline {mean_base_acc:.2f}% -> After Cut {mean_new_acc:.2f}%")

# Visual comparison chart
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

x = np.arange(len(neurons))
width = 0.35

ax1.bar(x - width/2, base_merg_list, width, label='Baseline', color='tomato')
ax1.bar(x + width/2, new_merg_list, width, label='After Cut', color='mediumseagreen')
ax1.set_ylabel('% Merged Edges')
ax1.set_title('Reduction in % Merged Edges per Neuron')
ax1.set_xticks(x)
ax1.set_xticklabels(neurons, rotation=45, ha="right")
ax1.legend()

ax2.bar(x - width/2, base_acc_list, width, label='Baseline', color='steelblue')
ax2.bar(x + width/2, new_acc_list, width, label='After Cut', color='mediumseagreen')
ax2.set_ylabel('Edge Accuracy (%)')
ax2.set_title('Change in Edge Accuracy per Neuron')
ax2.set_xticks(x)
ax2.set_xticklabels(neurons, rotation=45, ha="right")
ax2.legend()

plt.tight_layout()
plt.show()
