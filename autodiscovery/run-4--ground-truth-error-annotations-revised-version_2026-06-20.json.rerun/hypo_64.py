import sys
import gc
import os

import agentic_neuron_proofreader

import pickle
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra
import scipy.stats as stats
import matplotlib.pyplot as plt

print("Searching for dataset files...", flush=True)
# Load the provided dataset directly from $RERUN_PKL
print("Loading dataset from:", os.environ["RERUN_PKL"], flush=True)
dataset_paths = [os.environ["RERUN_PKL"]]

print(f"Found {len(dataset_paths)} dataset files: {dataset_paths}", flush=True)

split_distances = []
correct_distances = []

EDGE_CORRECT = 0
EDGE_SPLIT = 1

for path in dataset_paths:
    print(f"Loading {os.path.basename(path)}...", flush=True)
    with open(path, "rb") as f:
        payload = pickle.load(f)
    print(f"Finished reading {os.path.basename(path)}. Extracting graphs...", flush=True)
    
    gt = payload["gt_graph"]
    edge_error = np.asarray(payload["gt_edge_error"])
    
    # Aggressively free memory associated with unused structures
    payload.pop("fragments_graph", None)
    del payload
    gc.collect()
    
    edges = list(gt.edges())
    if len(edges) == 0:
        del gt, edge_error
        gc.collect()
        continue
        
    u_list = [e[0] for e in edges]
    v_list = [e[1] for e in edges]
    u_arr = np.array(u_list)
    v_arr = np.array(v_list)
    
    # Compute lengths of all edges in micrometers
    weights = np.linalg.norm(gt.node_xyz[u_arr] - gt.node_xyz[v_arr], axis=1)
    
    # Graph size N defined by the length of the node_xyz parallel array
    N = gt.node_xyz.shape[0]
    
    # Identify topological branch points (degree >= 3 in the ground truth graph)
    degrees = dict(gt.degree())
    branch_points = [n for n, d in degrees.items() if d >= 3]
    
    print(f"Graph has {len(edges)} edges, {N} max nodes, {len(branch_points)} branch points.", flush=True)
    
    if len(branch_points) == 0:
        del gt, edge_error, u_arr, v_arr, weights
        gc.collect()
        continue
    
    # Multi-source Dijkstra trick: Add a dummy node connected to all branch points with 0 distance
    bp_arr = np.array(branch_points)
    dummy_node = N
    dummy_u = np.full(len(bp_arr), dummy_node)
    dummy_v = bp_arr
    dummy_w = np.zeros(len(bp_arr))
    
    u_all = np.concatenate([u_arr, dummy_u])
    v_all = np.concatenate([v_arr, dummy_v])
    w_all = np.concatenate([weights, dummy_w])
    
    # Build sparse undirected graph with shape N+1 to include dummy node
    graph_sparse = sp.coo_matrix((w_all, (u_all, v_all)), shape=(N+1, N+1))
    
    # Compute distances from the dummy node (representing all branch points) to all other nodes
    dists = dijkstra(graph_sparse, directed=False, indices=dummy_node)
    
    node_dists = dists[:N]
    
    # Fetch distance to nearest branch point for each edge endpoint
    dist_u = node_dists[u_arr]
    dist_v = node_dists[v_arr]
    
    # Take minimum distance of its two endpoints
    edge_dists = np.minimum(dist_u, dist_v)
    
    split_mask = (edge_error == EDGE_SPLIT)
    correct_mask = (edge_error == EDGE_CORRECT)
    valid_mask = np.isfinite(edge_dists)
    
    split_dists = edge_dists[split_mask & valid_mask]
    correct_dists = edge_dists[correct_mask & valid_mask]
    
    split_distances.append(split_dists)
    correct_distances.append(correct_dists)

    print(f"Processed {os.path.basename(path)}. Splits: {len(split_dists)}, Correct: {len(correct_dists)}\n", flush=True)

    # Free memory before next iteration
    del gt, edge_error, u_arr, v_arr, weights, u_all, v_all, w_all
    del graph_sparse, dists, node_dists, dist_u, dist_v, edge_dists
    del split_mask, correct_mask, valid_mask, split_dists, correct_dists
    gc.collect()

# Combine data across all pooled brains
split_distances = np.concatenate(split_distances) if split_distances else np.array([])
correct_distances = np.concatenate(correct_distances) if correct_distances else np.array([])

print(f"Total split edges evaluated: {len(split_distances):,}")
print(f"Total correct edges evaluated: {len(correct_distances):,}")

if len(split_distances) > 0 and len(correct_distances) > 0:
    mean_split = np.mean(split_distances)
    mean_correct = np.mean(correct_distances)
    print(f"Mean distance to branch point (split): {mean_split:.2f} um")
    print(f"Mean distance to branch point (correct): {mean_correct:.2f} um")
    
    # Perform Mann-Whitney U test to compare distributions
    stat, pval = stats.mannwhitneyu(split_distances, correct_distances)
    print(f"Mann-Whitney U statistic: {stat}, p-value: {pval:.4e}")
    
    # Render comparative violin plots
    plt.figure(figsize=(10, 6))
    plt.violinplot([correct_distances, split_distances], showmeans=True, showmedians=True)
    plt.xticks([1, 2], ['Correct Edges', 'Split Edges'])
    plt.ylabel("Geodesic Distance to Nearest Branch Point (um)")
    plt.title("Proximity to Topological Branch Points: Split vs Correct Edges")
    plt.grid(axis='y', alpha=0.3)
    plt.show()
else:
    print("Not enough data to compute statistics.")
