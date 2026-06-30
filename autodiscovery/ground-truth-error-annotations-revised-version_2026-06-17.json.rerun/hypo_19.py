import sys
import subprocess

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", package])

# Install standard dependencies if not present
for pkg in ["psutil", "tensorstore", "pandas", "tqdm", "networkx", "scipy", "numpy", "matplotlib"]:
    try:
        __import__(pkg)
    except ImportError:
        install(pkg)

# Install the agentic_neuron_proofreader package directly from GitHub archive
try:
    import agentic_neuron_proofreader
except ImportError:
    install("https://github.com/AllenInstitute/agentic-neuron-proofreader/archive/refs/heads/main.zip")
    import agentic_neuron_proofreader

import pickle
import numpy as np
import networkx as nx
from scipy.spatial import cKDTree
from scipy.stats import mannwhitneyu
import matplotlib.pyplot as plt

# 2. Load dataset
dataset_path = "dataset_cache_789202_mcl100_add.pkl"
with open(dataset_path, "rb") as f:
    payload = pickle.load(f)

gt = payload["gt_graph"]
edge_error = np.asarray(payload["gt_edge_error"])

# 3. Identify branch nodes (degree >= 3)
branch_nodes = [n for n, d in gt.degree() if d >= 3]

# Group nodes and branch nodes by GT neuron ID
neuron_nodes = {}
for n in gt.nodes:
    neuron_id = gt.node_segment_id(n)
    if neuron_id not in neuron_nodes:
        neuron_nodes[neuron_id] = []
    neuron_nodes[neuron_id].append(n)

neuron_branch_nodes = {}
for n in branch_nodes:
    neuron_id = gt.node_segment_id(n)
    if neuron_id not in neuron_branch_nodes:
        neuron_branch_nodes[neuron_id] = []
    neuron_branch_nodes[neuron_id].append(n)

# 4. Compute Euclidean distance from every node to the nearest branch node on the same neuron
node_to_dist = {}
for neuron_id, n_list in neuron_nodes.items():
    b_list = neuron_branch_nodes.get(neuron_id, [])
    if len(b_list) == 0:
        for n in n_list:
            node_to_dist[n] = float('inf')
        continue
    
    b_coords = np.array([gt.node_xyz[b] for b in b_list])
    tree = cKDTree(b_coords)
    
    n_coords = np.array([gt.node_xyz[n] for n in n_list])
    dists, _ = tree.query(n_coords)
    for n, dist in zip(n_list, dists):
        node_to_dist[n] = dist

# 5. Assign distance to edges as the minimum distance of its endpoints
EDGE_CORRECT = 0
EDGE_SPLIT = 1

split_dists = []
correct_dists = []

edges = list(gt.edges)
for k, (u, v) in enumerate(edges):
    error_type = edge_error[k]
    if error_type in (EDGE_SPLIT, EDGE_CORRECT):
        dist = min(node_to_dist[u], node_to_dist[v])
        if dist == float('inf'):
            continue
        if error_type == EDGE_SPLIT:
            split_dists.append(dist)
        elif error_type == EDGE_CORRECT:
            correct_dists.append(dist)

# 6. Statistical testing
stat, pval = mannwhitneyu(split_dists, correct_dists, alternative='two-sided')
mean_split = np.mean(split_dists)
mean_correct = np.mean(correct_dists)
median_split = np.median(split_dists)
median_correct = np.median(correct_dists)

print("=== Spatial Correlation of Split Errors to Branch Nodes ===")
print(f"Mean distance to branch point (Split):   {mean_split:.2f} \u00b5m (Median: {median_split:.2f} \u00b5m)")
print(f"Mean distance to branch point (Correct): {mean_correct:.2f} \u00b5m (Median: {median_correct:.2f} \u00b5m)")
print(f"Mann-Whitney U test: U={stat}, p-value={pval}")

# 7. Visualization
plt.figure(figsize=(10, 6))
all_dists = correct_dists + split_dists
max_dist = min(max(all_dists), 200) if all_dists else 200  # Cap the visualization at 200 um for clarity
bins = np.linspace(0, max_dist, 50)

plt.hist(correct_dists, bins=bins, alpha=0.5, label=f'Correct (n={len(correct_dists)})', density=True)
plt.hist(split_dists, bins=bins, alpha=0.5, label=f'Split (n={len(split_dists)})', density=True)
plt.xlabel("Distance to nearest branch point (\u00b5m)")
plt.ylabel("Density")
plt.legend()
plt.title("Distance to Nearest Branch Point: Split vs Correct Edges")
plt.tight_layout()
plt.show()
