import sys
import subprocess
import os

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", package])

try:
    import psutil
except ImportError:
    install("psutil")

try:
    import tensorstore
except ImportError:
    install("tensorstore")

try:
    import agentic_neuron_proofreader
except ImportError:
    # Using the direct URL to the zip archive bypasses the need for local write permissions and git
    install("https://github.com/AllenInstitute/agentic-neuron-proofreader/archive/refs/heads/main.zip")
    import agentic_neuron_proofreader

import pickle
import glob
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import random
import heapq

# Load dataset
paths = glob.glob("../dataset_cache_*_add.pkl")
if not paths:
    paths = glob.glob("dataset_cache_*_add.pkl")
path = paths[0]
with open(path, "rb") as f:
    payload = pickle.load(f)

gt = payload["gt_graph"]
edge_error = np.asarray(payload["gt_edge_error"])

EDGE_CORRECT, EDGE_SPLIT = 0, 1

edges = list(gt.edges)
split_edges = [edges[i] for i, c in enumerate(edge_error) if c == EDGE_SPLIT]
correct_edges = [edges[i] for i, c in enumerate(edge_error) if c == EDGE_CORRECT]

# Balance classes by drawing a random sample of correct edges
random.seed(42)
if len(correct_edges) > len(split_edges):
    correct_edges = random.sample(correct_edges, len(split_edges))

def get_xyz(n):
    return gt.node_xyz[n]

def get_furthest_node(start_node, avoid_node, max_dist):
    """Finds the furthest reachable node up to max_dist along the skeleton without passing through avoid_node."""
    queue = [(0.0, start_node)]
    visited = {avoid_node}
    
    furthest_node = start_node
    max_d = 0.0
    
    while queue:
        d, curr = heapq.heappop(queue)
        if curr in visited:
            continue
        visited.add(curr)
        
        if d > max_d:
            max_d = d
            furthest_node = curr
            
        for neighbor in gt.neighbors(curr):
            if neighbor not in visited:
                pos1 = get_xyz(curr)
                pos2 = get_xyz(neighbor)
                weight = np.linalg.norm(np.array(pos1) - np.array(pos2))
                new_d = d + weight
                if new_d <= max_dist:
                    heapq.heappush(queue, (new_d, neighbor))
    return furthest_node, max_d

def calculate_tortuosity(u, v, max_dist=10.0):
    """Calculates tortuosity across an edge within a local skeleton window."""
    # Trace outwards from both endpoints up to max_dist path length
    end_u, dist_u = get_furthest_node(u, v, max_dist)
    end_v, dist_v = get_furthest_node(v, u, max_dist)
    
    pos_u = np.array(get_xyz(u))
    pos_v = np.array(get_xyz(v))
    edge_len = np.linalg.norm(pos_u - pos_v)
    
    total_path_length = dist_u + edge_len + dist_v
    
    pos_end_u = np.array(get_xyz(end_u))
    pos_end_v = np.array(get_xyz(end_v))
    euclidean_dist = np.linalg.norm(pos_end_u - pos_end_v)
    
    if euclidean_dist < 1e-5:
        return None
        
    return total_path_length / euclidean_dist

split_tort = []
for u, v in split_edges:
    t = calculate_tortuosity(u, v)
    if t is not None:
        split_tort.append(t)

correct_tort = []
for u, v in correct_edges:
    t = calculate_tortuosity(u, v)
    if t is not None:
        correct_tort.append(t)

# Ensure values are valid
split_tort = [t for t in split_tort if np.isfinite(t)]
correct_tort = [t for t in correct_tort if np.isfinite(t)]

t_stat, p_val = stats.ttest_ind(split_tort, correct_tort, equal_var=False)

print("=== Tortuosity Comparison (10 \u00b5m window) ===")
print(f"Sample size - Split Edges: {len(split_tort)}")
print(f"Sample size - Correct Edges: {len(correct_tort)}")
print(f"Mean Tortuosity - Split Edges:   {np.mean(split_tort):.4f} \u00b1 {np.std(split_tort):.4f}")
print(f"Mean Tortuosity - Correct Edges: {np.mean(correct_tort):.4f} \u00b1 {np.std(correct_tort):.4f}")
print(f"T-test: t = {t_stat:.4f}, p = {p_val:.4e}")

# Filter extreme outliers for plotting
p99_split = np.percentile(split_tort, 99) if split_tort else 2.0
p99_correct = np.percentile(correct_tort, 99) if correct_tort else 2.0
max_val = max(p99_split, p99_correct) * 1.5

plt.figure(figsize=(8, 6))

x_split = np.random.normal(1, 0.04, size=len(split_tort))
x_correct = np.random.normal(2, 0.04, size=len(correct_tort))

plt.scatter(x_split, split_tort, alpha=0.5, label='Split Edges')
plt.scatter(x_correct, correct_tort, alpha=0.5, label='Correct Edges')

# Formatting the plot
plt.xticks([1, 2], ['Split Edges', 'Correct Edges'])
plt.title('Local Skeleton Tortuosity (10 \u00b5m Window)')
plt.ylabel('Tortuosity (Path Length / Euclidean Distance)')
plt.ylim(0.9, max(2.0, max_val))
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend()
plt.show()
