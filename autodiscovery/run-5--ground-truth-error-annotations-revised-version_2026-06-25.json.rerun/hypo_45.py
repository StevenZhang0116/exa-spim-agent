import sys
import subprocess
import os

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", package])

for pkg in ["psutil", "pandas", "networkx", "scipy", "tqdm", "matplotlib", "tensorstore"]:
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
import random
import heapq
import scipy.stats as stats
import matplotlib.pyplot as plt
from collections import defaultdict

def get_xyz(gt, u):
    try:
        return gt.node_xyz(u)
    except (TypeError, KeyError, AttributeError):
        pass
    try:
        return gt.node_xyz[u]
    except (TypeError, KeyError, AttributeError):
        return gt.nodes[u]['node_xyz']

def get_all_nn_distances(S_edge_indices, edge_list, adj):
    # Highly optimized Voronoi-style Multi-Source Dijkstra.
    # Finds the exact nearest-neighbor distance for every source simultaneously in O(N log N) time.
    pq = []
    node_visited = {} 
    nn_dist_for_edge = {e: float('inf') for e in S_edge_indices}
    
    for e_idx in S_edge_indices:
        u, v = edge_list[e_idx]
        heapq.heappush(pq, (0.0, u, e_idx))
        heapq.heappush(pq, (0.0, v, e_idx))
        
    while pq:
        dist, curr, src_e = heapq.heappop(pq)
                
        if curr in node_visited:
            prev_dist, prev_src = node_visited[curr]
            if prev_src != src_e:
                meet_dist = dist + prev_dist
                if meet_dist < nn_dist_for_edge[src_e]:
                    nn_dist_for_edge[src_e] = meet_dist
                if meet_dist < nn_dist_for_edge[prev_src]:
                    nn_dist_for_edge[prev_src] = meet_dist
            continue
            
        node_visited[curr] = (dist, src_e)
        
        for neighbor, weight in adj[curr]:
            if neighbor not in node_visited:
                heapq.heappush(pq, (dist + weight, neighbor, src_e))
            else:
                prev_dist, prev_src = node_visited[neighbor]
                if prev_src != src_e:
                    meet_dist = dist + weight + prev_dist
                    if meet_dist < nn_dist_for_edge[src_e]:
                        nn_dist_for_edge[src_e] = meet_dist
                    if meet_dist < nn_dist_for_edge[prev_src]:
                        nn_dist_for_edge[prev_src] = meet_dist
                        
    return [nn_dist_for_edge[e] for e in S_edge_indices]

def safe_mean(dists):
    valid = [d for d in dists if not np.isinf(d) and not np.isnan(d)]
    return np.mean(valid) if valid else float('nan')

def main():
    dataset_paths = [os.environ["RERUN_PKL"]]
    print("Loading dataset from:", os.environ["RERUN_PKL"])

    if not dataset_paths:
        print("No datasets found in .. or its subdirectories.")
        return

    print(f"Found datasets: {dataset_paths}")

    neuron_results = []

    for path in dataset_paths:
        print(f"Processing {path}...")
        with open(path, "rb") as f:
            payload = pickle.load(f)
        
        gt = payload["gt_graph"]
        edge_error = np.asarray(payload["gt_edge_error"])
        gt_edges = list(gt.edges)
        
        neuron_edges = defaultdict(list)
        for k, (u, v) in enumerate(gt_edges):
            seg_u = gt.node_segment_id(u)
            seg_v = gt.node_segment_id(v)
            if seg_u == seg_v:
                neuron_edges[seg_u].append(k)
                
        for neuron, edge_idx_list in neuron_edges.items():
            split_edge_idxs = [k for k in edge_idx_list if edge_error[k] == 1]
            K = len(split_edge_idxs)
            if K < 2:
                continue
                
            local_nodes = set()
            for k in edge_idx_list:
                u, v = gt_edges[k]
                local_nodes.add(u)
                local_nodes.add(v)
                
            local_node_to_idx = {n: i for i, n in enumerate(local_nodes)}
            N_local = len(local_nodes)
            
            adj = [[] for _ in range(N_local)]
            local_edge_list = []
            local_split_indices = []
            
            for local_e_idx, k in enumerate(edge_idx_list):
                u, v = gt_edges[k]
                lu, lv = local_node_to_idx[u], local_node_to_idx[v]
                
                xyz_u = np.asarray(get_xyz(gt, u))
                xyz_v = np.asarray(get_xyz(gt, v))
                weight = float(np.linalg.norm(xyz_u - xyz_v))
                
                adj[lu].append((lv, weight))
                adj[lv].append((lu, weight))
                
                local_edge_list.append((lu, lv))
                if edge_error[k] == 1:
                    local_split_indices.append(local_e_idx)
                    
            actual_nn_dists = get_all_nn_distances(local_split_indices, local_edge_list, adj)
            mean_actual_dist = safe_mean(actual_nn_dists)
            
            if np.isnan(mean_actual_dist):
                continue
                
            M = len(edge_idx_list)
            all_edge_indices = list(range(M))
            
            simulated_means = []
            for _ in range(100):
                sim_splits = random.sample(all_edge_indices, K)
                sim_nn_dists = get_all_nn_distances(sim_splits, local_edge_list, adj)
                m = safe_mean(sim_nn_dists)
                if not np.isnan(m):
                    simulated_means.append(m)
            
            if not simulated_means:
                continue
                
            expected_mean = np.mean(simulated_means)
            
            neuron_results.append({
                'neuron': neuron,
                'actual': mean_actual_dist,
                'expected': expected_mean,
                'k_splits': K,
                'm_edges': M
            })
            
            print(f"Neuron {neuron}: Actual NN = {mean_actual_dist:.2f} um, Expected = {expected_mean:.2f} um (Splits: {K}/{M})")
            
    if not neuron_results:
        print("No valid neurons with >= 2 split edges found for analysis.")
        return
        
    actuals = [r['actual'] for r in neuron_results]
    expecteds = [r['expected'] for r in neuron_results]
    
    t_stat, p_val = stats.ttest_rel(actuals, expecteds)
    
    print("\n=== Spatial Clustering of Split Errors ===")
    print(f"Number of neurons analyzed: {len(neuron_results)}")
    print(f"Mean Actual Nearest-Neighbor Distance:   {np.mean(actuals):.2f} um")
    print(f"Mean Expected Nearest-Neighbor Distance: {np.mean(expecteds):.2f} um")
    print(f"Paired t-test results: t = {t_stat:.4f}, p = {p_val:.4e}")
    
    plt.figure(figsize=(8, 6))
    plt.hist(actuals, bins=20, alpha=0.6, label='Actual Mean NN Dist')
    plt.hist(expecteds, bins=20, alpha=0.6, label='Expected Mean NN Dist')
    plt.xlabel('Mean Nearest-Neighbor Distance (um)')
    plt.ylabel('Number of Neurons')
    plt.title('Distribution of Actual vs. Expected Split Edge Spacing')
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
