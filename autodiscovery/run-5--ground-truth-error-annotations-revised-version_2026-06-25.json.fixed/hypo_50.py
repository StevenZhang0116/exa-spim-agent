import sys
import subprocess
import glob
import pickle
import random
import os
import heapq
from collections import defaultdict

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", package])

try:
    import pandas as pd
except ImportError:
    install("pandas")
    import pandas as pd

try:
    import psutil
except ImportError:
    install("psutil")
    import psutil

try:
    import tensorstore as ts
except ImportError:
    install("tensorstore")
    import tensorstore as ts

try:
    import agentic_neuron_proofreader
except ImportError:
    install("https://github.com/AllenInstitute/agentic-neuron-proofreader/archive/refs/heads/main.zip")
    import agentic_neuron_proofreader

try:
    import numpy as np
except ImportError:
    install("numpy")
    import numpy as np

try:
    import networkx as nx
except ImportError:
    install("networkx")
    import networkx as nx

try:
    import scipy.stats as stats
except ImportError:
    install("scipy")
    import scipy.stats as stats

try:
    import matplotlib.pyplot as plt
except ImportError:
    install("matplotlib")
    import matplotlib.pyplot as plt

try:
    import seaborn as sns
except ImportError:
    install("seaborn")
    import seaborn as sns

def get_edge_nn_distances_fast(H, target_edges, target_set=None):
    # target_set: optional set of edge indices that count as "the same class"
    # to find nearest neighbor among. If None, use all target_edges (original behavior).
    node_to_edge_idx = defaultdict(list)
    for i, (u, v) in enumerate(target_edges):
        node_to_edge_idx[u].append(i)
        node_to_edge_idx[v].append(i)

    nn_distances = []

    for i, (u, v) in enumerate(target_edges):
        queue = [(0, u), (0, v)]
        visited = set()
        min_dist = float('inf')

        while queue:
            d, curr = heapq.heappop(queue)

            if curr in visited:
                continue
            visited.add(curr)

            if curr in node_to_edge_idx:
                other_edges = [idx for idx in node_to_edge_idx[curr] if idx != i]
                if other_edges:
                    min_dist = d
                    break

            for neighbor, edge_data in H[curr].items():
                if neighbor not in visited:
                    weight = edge_data.get('length', 1.0)
                    heapq.heappush(queue, (d + weight, neighbor))

        if min_dist != float('inf'):
            nn_distances.append(min_dist)

    return nn_distances


def mean_nn_for_subset(H, subset_edges):
    """Mean omit-to-omit NN distance for a subset of edges within neuron graph H."""
    if len(subset_edges) < 2:
        return None
    d = get_edge_nn_distances_fast(H, subset_edges)
    if not d:
        return None
    return float(np.mean(d))


def main():
    random.seed(42)
    np.random.seed(42)

    print("Loading dataset from:", os.environ["RERUN_PKL"])
    datasets = [os.path.join('..', 'data', os.path.basename(os.environ["RERUN_PKL"])),
                os.path.join('.', os.path.basename(os.environ["RERUN_PKL"]))]

    if not datasets:
        print("No datasets found.")
        return

    EDGE_OMIT = 2
    EDGE_CORRECT = 0

    omit_nn_distances = []
    random_nn_distances = []

    # CORRECTED-test bookkeeping: keep per-neuron graphs + edge lists so we can run a
    # within-neuron LABEL-PERMUTATION null (the correct null for "do omits cluster
    # beyond their marginal rate / run structure?").
    neuron_store = {}  # neuron -> (H, all_intra_edges, omit_edges)

    for dataset_path in datasets:
        print(f"Processing {dataset_path}...")
        with open(dataset_path, "rb") as f:
            payload = pickle.load(f)

        gt = payload["gt_graph"]
        edge_error = np.asarray(payload["gt_edge_error"])
        node_xyz = gt.node_xyz

        edges = list(gt.edges)

        neuron_graphs = defaultdict(nx.Graph)
        neuron_omit_edges = defaultdict(list)
        neuron_correct_edges = defaultdict(list)
        neuron_all_edges = defaultdict(list)

        for k, (u, v) in enumerate(edges):
            u_neuron = gt.node_segment_id(u)
            v_neuron = gt.node_segment_id(v)
            if u_neuron != v_neuron:
                continue

            error = int(edge_error[k])

            p1 = node_xyz[u]
            p2 = node_xyz[v]
            length = np.linalg.norm(p1 - p2)

            neuron_graphs[u_neuron].add_edge(u, v, length=length)
            neuron_all_edges[u_neuron].append((u, v))

            if error == EDGE_OMIT:
                neuron_omit_edges[u_neuron].append((u, v))
            elif error == EDGE_CORRECT:
                neuron_correct_edges[u_neuron].append((u, v))

        for neuron, omit_edges in neuron_omit_edges.items():
            H = neuron_graphs[neuron]
            n_omit = len(omit_edges)
            if n_omit < 2:
                continue

            correct_edges = neuron_correct_edges[neuron]
            if len(correct_edges) < n_omit:
                continue
            sampled_correct = random.sample(correct_edges, n_omit)

            omit_nn = get_edge_nn_distances_fast(H, omit_edges)
            omit_nn_distances.extend(omit_nn)

            rand_nn = get_edge_nn_distances_fast(H, sampled_correct)
            random_nn_distances.extend(rand_nn)

            neuron_store[neuron] = (H, neuron_all_edges[neuron], omit_edges)

    print(f"\nTotal omit edge NN pairs: {len(omit_nn_distances)}")
    print(f"Total random edge NN pairs: {len(random_nn_distances)}")

    if len(omit_nn_distances) == 0 or len(random_nn_distances) == 0:
        print("Not enough data to compute statistics.")
        return

    # ------------------------------------------------------------------
    # Original (recorded) test: Mann-Whitney U of omit NN distance (100% = 0 um)
    #   vs random correct-edge NN distance (mean 68.34 um), U = 123,107,614.0,
    #   p = 0.0, n = 57,286 omit NN pairs.
    # Fault flagged (MAJOR): the omit group is a DEGENERATE POINT MASS -- every
    #   omit NN distance is EXACTLY 0 because omits occur in connected multi-edge
    #   runs, so by construction almost every omit edge shares a node with another
    #   omit edge. Comparing a constant-0 group to a positive group is tautological;
    #   p=0.0 carries no evidential weight. It tests a foregone conclusion, not
    #   clustering beyond what run-structure guarantees.
    # CORRECTION: test whether omits cluster MORE than expected given the same
    #   per-neuron omit COUNT, using a WITHIN-NEURON LABEL-PERMUTATION null:
    #   for each neuron, randomly relabel which edges are "omit" (keeping the
    #   observed omit count) and recompute the mean omit-to-omit NN distance. If
    #   the observed mean NN distance is significantly SMALLER than this null,
    #   omits cluster beyond chance. The independent unit is the NEURON (cluster),
    #   so we aggregate to per-neuron mean NN distance and run a cluster-level
    #   sign-flip / permutation test. Effect size = standardized gap with CI.
    # ------------------------------------------------------------------

    # Original (now-flagged) MWU, reported for the record only:
    stat, p_value = stats.mannwhitneyu(omit_nn_distances, random_nn_distances, alternative='less')
    print("\n=== Original Mann-Whitney U Test (DEGENERATE point-mass, reported for comparison) ===")
    print(f"U statistic: {stat}")
    print(f"p-value: {p_value}")
    print(f"Omit NN: mean={np.mean(omit_nn_distances):.2f}, "
          f"0-dist fraction={sum(1 for d in omit_nn_distances if d == 0)/len(omit_nn_distances):.1%}")
    print(f"Random correct NN: mean={np.mean(random_nn_distances):.2f}")

    # --- CORRECTED: within-neuron label-permutation null, cluster = neuron ---
    rng = np.random.default_rng(42)
    N_PERM = 200
    obs_means = []      # per-neuron observed mean omit NN distance
    null_means = []     # per-neuron expected (mean over permutations) omit NN distance
    z_per_neuron = []
    for neuron, (H, all_edges, omit_edges) in neuron_store.items():
        n_omit = len(omit_edges)
        if n_omit < 2 or len(all_edges) <= n_omit:
            continue
        obs = mean_nn_for_subset(H, omit_edges)
        if obs is None:
            continue
        perm_vals = []
        for _ in range(N_PERM):
            idx = rng.choice(len(all_edges), size=n_omit, replace=False)
            subset = [all_edges[j] for j in idx]
            m = mean_nn_for_subset(H, subset)
            if m is not None:
                perm_vals.append(m)
        if len(perm_vals) < 10:
            continue
        perm_vals = np.array(perm_vals)
        obs_means.append(obs)
        null_means.append(perm_vals.mean())
        sd = perm_vals.std(ddof=1)
        if sd > 0:
            z_per_neuron.append((obs - perm_vals.mean()) / sd)

    obs_means = np.array(obs_means)
    null_means = np.array(null_means)
    n_neurons = len(obs_means)

    print(f"\n=== CORRECTED: within-neuron label-permutation test (cluster = neuron, n={n_neurons}) ===")
    if n_neurons >= 2:
        diffs = obs_means - null_means  # negative => observed omits closer than chance
        observed_stat = np.mean(diffs)
        # cluster-level sign-flip permutation p-value (two-sided)
        n_flip = 20000
        cnt = 0
        for _ in range(n_flip):
            signs = rng.choice([-1.0, 1.0], size=n_neurons)
            if abs(np.mean(signs * diffs)) >= abs(observed_stat):
                cnt += 1
        cluster_p = (cnt + 1) / (n_flip + 1)
        frac_closer = np.mean(diffs < 0)
        # effect size: bootstrap CI on the mean per-neuron (obs - null) gap
        B = 5000
        boot = np.empty(B)
        for b in range(B):
            s = rng.choice(diffs, size=n_neurons, replace=True)
            boot[b] = s.mean()
        lo, hi = np.percentile(boot, [2.5, 97.5])
        print(f"Mean per-neuron (observed - permuted) omit NN distance = {observed_stat:.2f} µm")
        print(f"  (negative => omits genuinely cluster tighter than their own count predicts)")
        print(f"Fraction of neurons with observed < permuted-null: {frac_closer:.2%}")
        print(f"Cluster sign-flip permutation p-value: {cluster_p:.4e}")
        print(f"Bootstrap 95% CI for mean gap: [{lo:.2f}, {hi:.2f}] µm")
        if z_per_neuron:
            print(f"Mean per-neuron z-score vs own null: {np.mean(z_per_neuron):.2f} (n={len(z_per_neuron)})")
        print("(Original degenerate MWU reported U=123,107,614.0, p=0.0 on a constant-0 group.)")
    else:
        print("Too few neurons with >=2 omit edges and a permutable pool for a cluster test.")

    plt.figure(figsize=(10, 6))

    omit_valid = [d for d in omit_nn_distances if np.isfinite(d)]
    rand_valid = [d for d in random_nn_distances if np.isfinite(d)]

    all_valid = omit_valid + rand_valid
    if all_valid:
        p99 = np.percentile(all_valid, 98)
        if p99 == 0:
            p99 = max(all_valid) if max(all_valid) > 0 else 1

        bins = np.linspace(0, p99, 50)

        sns.histplot(omit_valid, color='red', label='Omit Edges', stat='density', alpha=0.5, bins=bins, kde=False)
        sns.histplot(rand_valid, color='blue', label='Random Correct Edges', stat='density', alpha=0.5, bins=bins, kde=False)

        plt.xlabel('Nearest-Neighbor Network Distance (µm)')
        plt.ylabel('Density')
        plt.title('Nearest-Neighbor Distances: Omit vs Random Correct Edges')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xlim(-(p99 * 0.05), p99 * 1.05)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()
