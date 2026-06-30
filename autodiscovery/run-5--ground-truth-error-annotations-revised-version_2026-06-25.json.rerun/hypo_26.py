import subprocess
import sys
import os
import glob
import heapq

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", package])

# Install standard packages needed for analysis
packages = {
    'seaborn': 'seaborn',
    'scipy': 'scipy',
    'scikit-learn': 'sklearn',
    'matplotlib': 'matplotlib',
    'pandas': 'pandas',
    'psutil': 'psutil',
    'tensorstore': 'tensorstore'
}

for pkg, imp in packages.items():
    try:
        __import__(imp)
    except ImportError:
        install(pkg)

# Install agentic_neuron_proofreader without relying on a local git installation
try:
    import agentic_neuron_proofreader
except ImportError:
    install("https://github.com/AllenInstitute/agentic-neuron-proofreader/archive/refs/heads/main.zip")
    import agentic_neuron_proofreader

import pickle
import numpy as np
import networkx as nx
from collections import defaultdict
import random
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Locate and load the dataset
cache_files = glob.glob("../dataset_cache_*_add.pkl")
if not cache_files:
    cache_files = glob.glob("dataset_cache_*_add.pkl")
if not cache_files:
    cache_files = glob.glob("../cache/dataset_cache_*_add.pkl")
if not cache_files:
    cache_files = glob.glob("cache/dataset_cache_*_add.pkl")
    
if not cache_files:
    raise FileNotFoundError("Could not find dataset cache file.")

print(f"Loading {cache_files[0]}...")
with open(cache_files[0], "rb") as f:
    payload = pickle.load(f)

gt = payload["gt_graph"]
edge_error = np.asarray(payload["gt_edge_error"])

edges_list = list(gt.edges)

neuron_edges = defaultdict(list)
neuron_splits = defaultdict(list)

# Safely extract node coordinates
def get_node_xyz(g, n):
    if hasattr(g, 'node_xyz'):
        return np.array(g.node_xyz[n])
    elif 'node_xyz' in g.nodes[n]:
        return np.array(g.nodes[n]['node_xyz'])
    else:
        raise ValueError(f"Cannot find node_xyz for node {n}")

print("Extracting edges by neuron...")
for idx, (u, v) in enumerate(edges_list):
    # Determine the neuron this edge belongs to
    u_neuron = gt.node_segment_id(u)
    v_neuron = gt.node_segment_id(v)
    if u_neuron == v_neuron:
        neuron_name = u_neuron
        neuron_edges[neuron_name].append((idx, u, v))
        if edge_error[idx] == 1: # EDGE_SPLIT
            neuron_splits[neuron_name].append((idx, u, v))

# Consider only neurons with at least 3 splits to ensure valid distances and clustering
qualifying_neurons = [n for n in neuron_splits if len(neuron_splits[n]) >= 3]
print(f"Found {len(qualifying_neurons)} qualifying neurons with >= 3 splits.\n")

obs_means = []
sim_means = []

np.random.seed(42)
random.seed(42)

for neuron_idx, neuron in enumerate(qualifying_neurons):
    n_edges = neuron_edges[neuron]
    n_splits = neuron_splits[neuron]
    num_splits = len(n_splits)
    
    all_nodes = set()
    for idx, u, v in n_edges:
        all_nodes.add(u)
        all_nodes.add(v)
        
    nodes = list(all_nodes)
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    N = len(nodes)
    
    adj_list = [[] for _ in range(N)]
    edges_with_weights = []
    
    for idx, u, v in n_edges:
        i, j = node_to_idx[u], node_to_idx[v]
        dist = np.linalg.norm(get_node_xyz(gt, u) - get_node_xyz(gt, v))
        adj_list[i].append((j, dist))
        adj_list[j].append((i, dist))
        edges_with_weights.append((i, j, dist))
        
    n_splits_idx = [(node_to_idx[u], node_to_idx[v]) for _, u, v in n_splits]
    n_edges_idx = [(node_to_idx[u], node_to_idx[v]) for _, u, v in n_edges]
    
    # Highly optimized algorithm using a Multi-Source Dijkstra (Voronoi Graph Partition) 
    # to find nearest neighbors simultaneously across all targets.
    def get_nn_dists(selected_edges):
        if len(selected_edges) < 2:
            return []
        
        K = len(selected_edges)
        dist = np.full(N, np.inf)
        source = np.full(N, -1, dtype=np.int32)
        
        pq = []
        
        node_to_sources = defaultdict(list)
        for i, (u, v) in enumerate(selected_edges):
            node_to_sources[u].append(i)
            node_to_sources[v].append(i)
            
        min_dist = np.full(K, np.inf)
        
        for u, srcs in node_to_sources.items():
            if len(srcs) > 1:
                for s in srcs:
                    min_dist[s] = 0.0
            dist[u] = 0.0
            source[u] = srcs[0]
            heapq.heappush(pq, (0.0, u))
            
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist[u]:
                continue
                
            s_u = source[u]
            for v, w in adj_list[u]:
                new_d = d + w
                if new_d < dist[v]:
                    dist[v] = new_d
                    source[v] = s_u
                    heapq.heappush(pq, (new_d, v))
                    
        for u, v, w in edges_with_weights:
            s_u = source[u]
            s_v = source[v]
            if s_u != -1 and s_v != -1 and s_u != s_v:
                d = dist[u] + dist[v] + w
                if d < min_dist[s_u]:
                    min_dist[s_u] = d
                if d < min_dist[s_v]:
                    min_dist[s_v] = d
                    
        valid_dists = min_dist[~np.isinf(min_dist)]
        return valid_dists.tolist()

    # Actual observed splits spatial distance mean
    obs_nns = get_nn_dists(n_splits_idx)
    if not obs_nns:
        continue
    obs_mean = np.mean(obs_nns)
    
    sim_mean_nns = []
    
    # Monte Carlo simulation (100 iterations)
    for _ in range(100):
        sim_edges = random.sample(n_edges_idx, num_splits)
        sim_nns = get_nn_dists(sim_edges)
        if sim_nns:
            sim_mean_nns.append(np.mean(sim_nns))
            
    if sim_mean_nns:
        obs_means.append(obs_mean)
        sim_means.append(np.mean(sim_mean_nns))

if len(obs_means) < 2:
    print("Not enough qualifying neurons to perform paired t-test.")
else:
    t_stat, p_val = stats.ttest_rel(obs_means, sim_means, alternative='less')

    print(f"--- Statistical Test Results ---")
    print(f"Paired t-test t-statistic: {t_stat:.4f}")
    print(f"Paired t-test p-value: {p_val:.4e}")
    print(f"Mean of observed NN distances: {np.mean(obs_means):.4f} \u00b5m")
    print(f"Mean of simulated random NN distances: {np.mean(sim_means):.4f} \u00b5m")
    if p_val < 0.05:
        print("Conclusion: Split errors are significantly clustered (observed distances are smaller).")
    else:
        print("Conclusion: Split errors are NOT significantly clustered.")

    # Visualization
    plt.figure(figsize=(8, 6))
    plot_data = pd.DataFrame({
        'Distance (\u00b5m)': obs_means + sim_means,
        'Type': ['Observed']*len(obs_means) + ['Simulated Random']*len(sim_means)
    })
    sns.violinplot(x='Type', y='Distance (\u00b5m)', data=plot_data, palette="Set2")
    plt.title(f'Spatial Clustering of Split Errors\nPaired t-test p-value: {p_val:.4e}')
    plt.ylabel('Mean Nearest-Neighbor Distance per Neuron (\u00b5m)')
    plt.tight_layout()
    plt.show()
