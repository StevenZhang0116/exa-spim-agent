import subprocess
import sys

# ---------------------------------------------------------------------------
# CORRECTED TEST (id 39)
# Original (recorded/rerun): NO statistical test. Only descriptive before/after
#   GLOBAL means over 12 neurons:
#     mean %merged 13.25% -> 1.83%, mean edge accuracy 82.27% -> 93.66%.
#   The hypothesis is stated PER NEURON (">80% of merge errors per neuron") but
#   was evaluated as a global mean, with no p-value / CI / paired test, masking
#   per-neuron failures (e.g. N020 0%, N006 ~55%).
# FIX (test-level only, analysis quantities unchanged):
#   * Add a paired Wilcoxon signed-rank test across the 12 per-neuron deltas for
#     (a) %merged reduction and (b) edge-accuracy gain -- the right paired,
#     non-parametric test for before/after on the same neurons.
#   * Report a per-unit effect (median per-neuron delta) WITH a bootstrap 95% CI,
#     not just the global mean.
#   * Compute the PER-NEURON pass rate of the ">80% merge reduction" claim
#     (only neurons with non-trivial baseline merges), instead of a global mean,
#     so the actual hypothesis is evaluated as stated.
#   Everything above the final block is byte-for-byte the loading-fixed script.
# ---------------------------------------------------------------------------

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", package])

# Ensure all dependencies are installed before importing (install once if missing)
for pkg in ["tensorstore", "psutil", "pandas", "tqdm", "networkx", "scipy", "matplotlib"]:
    try:
        __import__(pkg)
    except ImportError:
        install(pkg)

try:
    import agentic_neuron_proofreader
except ImportError:
    install("https://github.com/AllenInstitute/agentic-neuron-proofreader/archive/refs/heads/main.zip")
    import agentic_neuron_proofreader

import os
import pickle
import numpy as np
import networkx as nx
from collections import defaultdict
import matplotlib.pyplot as plt
import scipy.spatial
from scipy import stats

import agentic_neuron_proofreader
from agentic_neuron_proofreader.data_modules import canonical_labeling as cl

# 1. Dataset Localization & Loading
dataset_path = os.environ["RERUN_PKL"]
print(f"Loading dataset from: {dataset_path}")
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
# recorded: mean %merged 13.25% -> 1.83%, mean edge accuracy 82.27% -> 93.66%; NO statistical test.

# ---------------------------------------------------------------------------
# CORRECTED STATISTICAL TEST: paired across the per-neuron before/after deltas
# ---------------------------------------------------------------------------
base_merg = np.asarray(base_merg_list, dtype=float)
new_merg = np.asarray(new_merg_list, dtype=float)
base_acc = np.asarray(base_acc_list, dtype=float)
new_acc = np.asarray(new_acc_list, dtype=float)

merge_reduction = base_merg - new_merg          # positive = fewer merged edges
acc_gain = new_acc - base_acc                   # positive = higher accuracy
n_neurons = len(neurons)

def boot_ci_median(vals, rng, n_boot=10000):
    vals = np.asarray(vals, dtype=float)
    m = len(vals)
    bs = [np.median(rng.choice(vals, size=m, replace=True)) for _ in range(n_boot)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return float(np.median(vals)), float(lo), float(hi)

rng = np.random.default_rng(42)

print("\n=== CORRECTED: paired before/after statistical test across neurons ===")
print(f"n = {n_neurons} paired neurons")

# Wilcoxon signed-rank on per-neuron %merged reduction (one-sided: reduction > 0)
try:
    w_merg, p_merg = stats.wilcoxon(base_merg, new_merg, alternative='greater', zero_method='wilcox')
    print(f"Wilcoxon signed-rank (%merged baseline > after): W={w_merg:.1f}, p={p_merg:.4e}")
except ValueError as e:
    print(f"Wilcoxon (%merged) could not run: {e}")

med, lo, hi = boot_ci_median(merge_reduction, rng)
print(f"Per-neuron %merged reduction: median={med:.2f} pct-points (95% bootstrap CI [{lo:.2f}, {hi:.2f}])")

# Wilcoxon signed-rank on per-neuron accuracy gain (one-sided: gain > 0)
try:
    w_acc, p_acc = stats.wilcoxon(new_acc, base_acc, alternative='greater', zero_method='wilcox')
    print(f"Wilcoxon signed-rank (accuracy after > baseline): W={w_acc:.1f}, p={p_acc:.4e}")
except ValueError as e:
    print(f"Wilcoxon (accuracy) could not run: {e}")

med, lo, hi = boot_ci_median(acc_gain, rng)
print(f"Per-neuron edge-accuracy gain: median={med:.2f} pct-points (95% bootstrap CI [{lo:.2f}, {hi:.2f}])")

# Per-neuron evaluation of the ">80% merge-error reduction" claim (as stated),
# restricted to neurons with a non-trivial baseline merge burden.
eps = 1e-9
nontrivial = base_merg > 1.0  # at least ~1% baseline merged edges to be meaningful
per_neuron_frac = np.where(base_merg > eps, merge_reduction / np.maximum(base_merg, eps), np.nan)
considered = int(np.sum(nontrivial))
passed = int(np.sum((per_neuron_frac[nontrivial] >= 0.80)))
print(f"\nPer-neuron '>80% merge reduction' claim (as stated, baseline %merged > 1%):")
print(f"  {passed}/{considered} neurons individually achieve >=80% merge-error reduction.")
frac_vals = per_neuron_frac[nontrivial & np.isfinite(per_neuron_frac)]
if len(frac_vals) > 0:
    fm, flo, fhi = boot_ci_median(frac_vals * 100.0, rng)
    print(f"  Median per-neuron merge reduction = {fm:.1f}% (95% bootstrap CI [{flo:.1f}%, {fhi:.1f}%])")

print("\nConclusion (corrected): the per-neuron reduction in %merged edges and the gain in")
print("edge accuracy are tested with a paired Wilcoxon signed-rank test (not a bare global mean),")
print("and the '>80% per neuron' claim is now evaluated per neuron rather than via the global mean.")

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
