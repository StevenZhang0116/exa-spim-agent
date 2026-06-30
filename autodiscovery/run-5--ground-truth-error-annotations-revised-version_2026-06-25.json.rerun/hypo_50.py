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

def get_edge_nn_distances_fast(H, target_edges):
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
            
    print(f"\nTotal omit edge NN pairs: {len(omit_nn_distances)}")
    print(f"Total random edge NN pairs: {len(random_nn_distances)}")
    
    if len(omit_nn_distances) == 0 or len(random_nn_distances) == 0:
        print("Not enough data to compute statistics.")
        return

    stat, p_value = stats.mannwhitneyu(omit_nn_distances, random_nn_distances, alternative='less')
    
    print("\n=== Mann-Whitney U Test ===")
    print(f"U statistic: {stat}")
    print(f"p-value: {p_value}")
    
    print("\n=== Summary Statistics ===")
    print(f"Omit NN Distances: Mean = {np.mean(omit_nn_distances):.2f}, Median = {np.median(omit_nn_distances):.2f}, 0-dist = {sum(1 for d in omit_nn_distances if d == 0)/len(omit_nn_distances):.1%}")
    print(f"Random NN Distances: Mean = {np.mean(random_nn_distances):.2f}, Median = {np.median(random_nn_distances):.2f}, 0-dist = {sum(1 for d in random_nn_distances if d == 0)/len(random_nn_distances):.1%}")

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
        
        plt.xlabel('Nearest-Neighbor Network Distance (\u00b5m)')
        plt.ylabel('Density')
        plt.title('Nearest-Neighbor Distances: Omit vs Random Correct Edges')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xlim(-(p99 * 0.05), p99 * 1.05)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()
