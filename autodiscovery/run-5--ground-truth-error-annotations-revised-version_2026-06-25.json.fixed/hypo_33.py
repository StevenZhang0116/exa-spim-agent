import subprocess
import sys
import os

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", package])

# Install missing dependencies
for pkg in ["psutil", "pandas", "tensorstore"]:
    try:
        __import__(pkg)
    except ImportError:
        install(pkg)

try:
    import agentic_neuron_proofreader
except ImportError:
    install("https://github.com/AllenInstitute/agentic-neuron-proofreader/archive/refs/heads/main.zip")

import pickle
import numpy as np
import networkx as nx
import glob
from scipy.stats import mannwhitneyu
import matplotlib.pyplot as plt

def main():
    paths = glob.glob("../*add.pkl") + glob.glob("../../*add.pkl") + glob.glob("../*/*add.pkl")
    if not paths:
        paths = glob.glob("**/*add.pkl", recursive=True)
        if not paths:
            print("Dataset not found.")
            return

    target_paths = [p for p in paths if '794495' in p]
    path = target_paths[0] if target_paths else paths[0]

    print(f"Loading dataset from: {path}")
    with open(path, "rb") as f:
        payload = pickle.load(f)

    gt = payload["gt_graph"]
    edge_error = np.asarray(payload["gt_edge_error"])

    edges = list(gt.edges)
    nodes_u = [e[0] for e in edges]
    nodes_v = [e[1] for e in edges]

    try:
        xyz_u = gt.node_xyz[nodes_u]
        xyz_v = gt.node_xyz[nodes_v]
    except Exception:
        try:
            xyz_u = np.array([gt.nodes[u]['node_xyz'] for u in nodes_u])
            xyz_v = np.array([gt.nodes[v]['node_xyz'] for v in nodes_v])
        except Exception:
            xyz_u = np.array([gt.node_xyz[u] for u in nodes_u])
            xyz_v = np.array([gt.node_xyz[v] for v in nodes_v])

    lengths = np.linalg.norm(xyz_u - xyz_v, axis=1)

    for (u, v), length in zip(edges, lengths):
        gt[u][v]['weight'] = length

    degrees = dict(gt.degree())
    branch_nodes = {n for n, d in degrees.items() if d >= 3}

    distances_to_branch = {}
    for comp in nx.connected_components(gt):
        comp_branch_nodes = branch_nodes.intersection(comp)
        if not comp_branch_nodes:
            continue

        subg = gt.subgraph(comp)
        dist_dict = nx.multi_source_dijkstra_path_length(subg, comp_branch_nodes, weight='weight')
        distances_to_branch.update(dist_dict)

    split_distances = []
    correct_distances = []
    # CORRECTED: also track the per-neuron cluster id of every edge so we can run
    # a cluster-aware (per-neuron) test; this does NOT change the quantities measured.
    split_neuron = []
    correct_neuron = []

    EDGE_CORRECT = 0
    EDGE_SPLIT = 1

    for k, (u, v) in enumerate(edges):
        if u in distances_to_branch and v in distances_to_branch:
            d_u = distances_to_branch[u]
            d_v = distances_to_branch[v]

            edge_dist = min(d_u, d_v) + lengths[k] / 2

            c = edge_error[k]
            neuron = gt.node_segment_id(u)
            if c == EDGE_SPLIT:
                split_distances.append(edge_dist)
                split_neuron.append(neuron)
            elif c == EDGE_CORRECT:
                correct_distances.append(edge_dist)
                correct_neuron.append(neuron)

    split_distances = np.array(split_distances)
    correct_distances = np.array(correct_distances)
    split_neuron = np.array(split_neuron)
    correct_neuron = np.array(correct_neuron)

    print(f"Number of split edges near branches analyzed: {len(split_distances)}")
    print(f"Number of correct edges near branches analyzed: {len(correct_distances)}")

    if len(split_distances) == 0 or len(correct_distances) == 0:
        print("Not enough edges to compare.")
        return

    # ------------------------------------------------------------------
    # Original (recorded) test: Mann-Whitney U on split vs correct edge
    #   distance-to-branch-node, U = 3,472,848,384.0, p = 4.637e-27,
    #   split median = 216.42 um, correct median = 246.43 um (n_split=7,988,
    #   n_correct=934,849).
    # Faults flagged:
    #   (a) HUGE-n TRIVIAL EFFECT: p~0 is driven almost entirely by n~940k while
    #       the median difference is only ~30 um -> significance != importance.
    #   (b) NON-INDEPENDENCE: edges within the same neuron are spatially
    #       autocorrelated, so treating ~10^6 edges as independent inflates the
    #       test statistic / understates the p-value.
    # CORRECTION:
    #   1. Keep the MWU but ALSO report a proper EFFECT SIZE with a CI:
    #      rank-biserial r = 1 - 2U/(n1*n2) and Cliff's delta (= -rank-biserial),
    #      with a CLUSTER bootstrap CI (resample whole neurons, not edges).
    #   2. Add a CLUSTER-PERMUTATION p-value: the independent unit is the NEURON.
    #      We aggregate to per-neuron median distance for split vs correct edges
    #      and run a paired sign-flip / label-permutation test across neurons.
    # ------------------------------------------------------------------
    stat, pval = mannwhitneyu(split_distances, correct_distances, alternative='two-sided')
    print(f"Mann-Whitney U test statistic: {stat}")
    print(f"p-value (UNCORRECTED, treats all edges as independent): {pval}")
    print(f"Median distance for split edges: {np.median(split_distances):.2f} µm")
    print(f"Median distance for correct edges: {np.median(correct_distances):.2f} µm")

    n1 = len(split_distances)
    n2 = len(correct_distances)
    # rank-biserial correlation as effect size (probability of superiority based)
    rank_biserial = 1.0 - (2.0 * stat) / (n1 * n2)
    cliffs_delta = -rank_biserial  # split - correct orientation
    print()
    print("=== CORRECTED: effect size (the ~30 um median gap is tiny) ===")
    print(f"Rank-biserial r = {rank_biserial:.4f} (|r|<0.1 negligible, <0.3 small)")
    print(f"Cliff's delta (split vs correct) = {cliffs_delta:.4f}")
    print(f"Median difference = {np.median(split_distances)-np.median(correct_distances):.2f} µm "
          f"(split - correct); practically small vs typical branch spacing.")

    # --- Cluster bootstrap CI for Cliff's delta: resample whole neurons ---
    rng = np.random.default_rng(42)
    uniq_split_neurons = np.unique(split_neuron)
    uniq_correct_neurons = np.unique(correct_neuron)
    # index edges by neuron for fast resampling
    split_by_neuron = {nrn: split_distances[split_neuron == nrn] for nrn in uniq_split_neurons}
    correct_by_neuron = {nrn: correct_distances[correct_neuron == nrn] for nrn in uniq_correct_neurons}

    def cliffs_from_samples(a, b, max_n=4000):
        # subsample for tractable O(n*m) on huge correct set
        if len(a) > max_n:
            a = rng.choice(a, max_n, replace=False)
        if len(b) > max_n:
            b = rng.choice(b, max_n, replace=False)
        gt_count = 0
        lt_count = 0
        b_sorted = np.sort(b)
        for x in a:
            gt_count += np.searchsorted(b_sorted, x, side='left')
            lt_count += len(b_sorted) - np.searchsorted(b_sorted, x, side='right')
        return (gt_count - lt_count) / (len(a) * len(b))

    B = 1000
    boot_deltas = []
    for _ in range(B):
        sn = rng.choice(uniq_split_neurons, len(uniq_split_neurons), replace=True)
        cn = rng.choice(uniq_correct_neurons, len(uniq_correct_neurons), replace=True)
        a = np.concatenate([split_by_neuron[x] for x in sn]) if len(sn) else np.array([])
        b = np.concatenate([correct_by_neuron[x] for x in cn]) if len(cn) else np.array([])
        if len(a) == 0 or len(b) == 0:
            continue
        boot_deltas.append(cliffs_from_samples(a, b))
    boot_deltas = np.array(boot_deltas)
    d_lo, d_hi = np.percentile(boot_deltas, [2.5, 97.5])
    print(f"Cluster bootstrap (resample neurons) 95% CI for Cliff's delta: [{d_lo:.4f}, {d_hi:.4f}]")

    # --- Cluster-level (per-neuron) test: median split vs median correct per neuron ---
    common = np.intersect1d(uniq_split_neurons, uniq_correct_neurons)
    paired_diffs = []
    for nrn in common:
        ms = np.median(split_by_neuron[nrn])
        mc = np.median(correct_by_neuron[nrn])
        paired_diffs.append(ms - mc)
    paired_diffs = np.array(paired_diffs)
    n_neurons = len(paired_diffs)
    print()
    print(f"=== CORRECTED: cluster-level test, independent unit = neuron (n={n_neurons}) ===")
    if n_neurons >= 2:
        # sign-flip permutation test on the per-neuron median differences
        observed = np.mean(paired_diffs)
        n_perm = 20000
        count = 0
        for _ in range(n_perm):
            signs = rng.choice([-1.0, 1.0], size=n_neurons)
            if abs(np.mean(signs * paired_diffs)) >= abs(observed):
                count += 1
        perm_p = (count + 1) / (n_perm + 1)
        frac_neg = np.mean(paired_diffs < 0)
        print(f"Per-neuron mean(median_split - median_correct) = {observed:.2f} µm")
        print(f"Fraction of neurons with split median < correct median: {frac_neg:.2%}")
        print(f"Sign-flip permutation p (cluster = neuron): {perm_p:.4e}")
    else:
        print("Too few neurons with both split and correct edges for a cluster test.")

    plt.figure(figsize=(8, 6))
    plt.boxplot([correct_distances, split_distances], labels=['Correct Edges', 'Split Edges'], showfliers=False)
    plt.ylabel('Distance to Nearest Branch Node (µm)')
    plt.title('Distance from Edges to Nearest Branch Node\n(Outliers Hidden)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

if __name__ == "__main__":
    main()
