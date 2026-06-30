import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", package])

# Install required dependencies to ensure unpickling succeeds
for pkg in ["psutil", "pandas", "tqdm", "networkx", "matplotlib", "scipy", "tensorstore"]:
    try:
        __import__(pkg)
    except ImportError:
        install(pkg)

try:
    import agentic_neuron_proofreader
except ImportError:
    install("https://github.com/AllenInstitute/agentic-neuron-proofreader/archive/refs/heads/main.zip")
    import agentic_neuron_proofreader

import pickle
import numpy as np
import networkx as nx
import glob
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from collections import defaultdict
import random
import heapq
import itertools

def nearest_target_dists(H, targets):
    target_set = set(targets)
    dists = []
    
    for start in targets:
        counter = itertools.count()
        queue = [(0, next(counter), start)]
        min_dists = {start: 0}
        found_dist = None
        
        while queue:
            dist, _, u = heapq.heappop(queue)
            
            if dist > min_dists.get(u, float('inf')):
                continue
                
            if u != start and u in target_set:
                found_dist = dist
                break
                
            for v, edata in H[u].items():
                w = edata.get('weight', 1.0)
                new_dist = dist + w
                if new_dist < min_dists.get(v, float('inf')):
                    min_dists[v] = new_dist
                    heapq.heappush(queue, (new_dist, next(counter), v))
                    
        if found_dist is not None:
            dists.append(found_dist)
            
    return dists

def main():
    # Dataset files are present one level above the current working directory.
    cache_files = glob.glob("../dataset_cache_*_add.pkl")
    if not cache_files:
        cache_files = glob.glob("dataset_cache_*_add.pkl")
        
    if not cache_files:
        print("No dataset cache files found.")
        return
        
    observed_nn_dists = []
    random_nn_dists = []

    # Ensure reproducibility
    random.seed(42)
    np.random.seed(42)

    for cache_file in cache_files:
        print(f"Loading cache file: {cache_file}")
        with open(cache_file, "rb") as f:
            payload = pickle.load(f)

        gt = payload["gt_graph"]
        edge_error = np.asarray(payload["gt_edge_error"])

        EDGE_SPLIT = 1

        edges = list(gt.edges)
        neuron_edges = defaultdict(list)
        neuron_edge_indices = defaultdict(list)

        print("Grouping edges by neuron...")
        for k, (u, v) in enumerate(edges):
            neuron1 = gt.node_segment_id(u)
            neuron_edges[neuron1].append((u, v))
            neuron_edge_indices[neuron1].append(k)

        print("Computing distances for each neuron...")
        for neuron, n_edges in neuron_edges.items():
            H = nx.Graph()
            edge_nodes = []
            split_nodes = []
            
            edges_to_add = []
            for i, (u, v) in enumerate(n_edges):
                idx = neuron_edge_indices[neuron][i]
                is_split = (edge_error[idx] == EDGE_SPLIT)
                
                xyz_u = gt.node_xyz[u]
                xyz_v = gt.node_xyz[v]
                L = np.linalg.norm(xyz_u - xyz_v)
                
                edge_node = f"e_{i}"
                edge_nodes.append(edge_node)
                
                edges_to_add.append((u, edge_node, {'weight': L / 2.0}))
                edges_to_add.append((v, edge_node, {'weight': L / 2.0}))
                
                if is_split:
                    split_nodes.append(edge_node)
                    
            H.add_edges_from(edges_to_add)
            
            num_splits = len(split_nodes)
            if num_splits < 2:
                continue
                
            # Observed distances
            obs_dists = nearest_target_dists(H, split_nodes)
            observed_nn_dists.extend(obs_dists)
            
            # Random null model
            # Randomly distribute the exact same number of split edges
            # Repeat 5 times for a robust random distribution
            for _ in range(5):
                rand_nodes = random.sample(edge_nodes, num_splits)
                rand_dists = nearest_target_dists(H, rand_nodes)
                random_nn_dists.extend(rand_dists)

    if not observed_nn_dists:
        print("No neurons with >= 2 splits found to compute distances.")
        return

    print("\nRunning Kolmogorov-Smirnov test...")
    ks_stat, p_value = ks_2samp(observed_nn_dists, random_nn_dists)
    
    print(f"\n=== Kolmogorov-Smirnov Test Results ===")
    print(f"KS Statistic: {ks_stat:.4f}")
    print(f"P-value: {p_value:.4e}")
    
    print(f"\nMean observed nearest-neighbor distance: {np.mean(observed_nn_dists):.2f} µm")
    print(f"Mean random nearest-neighbor distance:   {np.mean(random_nn_dists):.2f} µm")
    print(f"Total split edges analyzed: {len(observed_nn_dists)}")

    # Plotting
    print("\nGenerating histogram...")
    plt.figure(figsize=(10, 6))
    
    # Exclude extreme outliers for better visualization
    max_val = np.percentile(observed_nn_dists + random_nn_dists, 95)
    bins = np.linspace(0, max_val, 50)
    
    plt.hist(observed_nn_dists, bins=bins, alpha=0.5, density=True, label='Observed Split Edges')
    plt.hist(random_nn_dists, bins=bins, alpha=0.5, density=True, label='Random Null Model')
    
    plt.title('Nearest-Neighbor Distance of Split Edges (Observed vs Random)')
    plt.xlabel('Distance to Nearest Split Edge (µm)')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
