import os
import glob
import pickle
import sys
import subprocess
import gc
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.stats import mannwhitneyu

def install_deps():
    deps = ["psutil", "tensorstore", "networkx", "pandas", "statsmodels", "matplotlib", "scipy"]
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet"] + deps)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "https://github.com/AllenInstitute/agentic-neuron-proofreader/archive/main.zip"])

try:
    import psutil
    import agentic_neuron_proofreader
except ImportError:
    install_deps()
    import agentic_neuron_proofreader

# Locate dataset files
files = list(set(glob.glob("./*_add.pkl") + glob.glob("../data/*_add.pkl")))
print("Processing files:", files)

split_distances = []
correct_distances = []

for file in files:
    # Load aggressively managing memory
    gc.disable()
    with open(file, "rb") as f:
        payload = pickle.load(f)
    gc.enable()
    
    if "fragments_graph" in payload:
        del payload["fragments_graph"]
    gc.collect()
    
    gt = payload["gt_graph"]
    edge_error = np.asarray(payload["gt_edge_error"])
    
    nodes = list(gt.nodes())
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    N = len(nodes)
    
    # Identify structural branch points
    degrees = dict(gt.degree())
    branch_points = [n for n, deg in degrees.items() if deg >= 3]
    bp_indices = [node_to_idx[n] for n in branch_points]
    
    edges = list(gt.edges())
    row = []
    col = []
    edge_weights = []
    
    # Extract structural positions to build distance graph
    try:
        node_xyz = gt.node_xyz
        for u, v in edges:
            p1 = node_xyz[u]
            p2 = node_xyz[v]
            dist = np.linalg.norm(p1 - p2)
            i = node_to_idx[u]
            j = node_to_idx[v]
            row.extend([i, j])
            col.extend([j, i])
            edge_weights.extend([dist, dist])
    except Exception:
        for u, v in edges:
            p1 = gt.nodes[u].get('xyz', np.array([0,0,0]))
            p2 = gt.nodes[v].get('xyz', np.array([0,0,0]))
            dist = np.linalg.norm(p1 - p2)
            i = node_to_idx[u]
            j = node_to_idx[v]
            row.extend([i, j])
            col.extend([j, i])
            edge_weights.extend([dist, dist])
            
    # Connect a universal dummy node to all branch points with 0 distance
    # This allows computing distance to NEAREST branch point efficiently in one pass
    for bp in bp_indices:
        row.extend([N, bp])
        col.extend([bp, N])
        edge_weights.extend([0.0, 0.0])
        
    adj = None
    if len(bp_indices) > 0:
        adj = coo_matrix((edge_weights, (row, col)), shape=(N+1, N+1)).tocsr()
        dist_to_bp = dijkstra(adj, directed=False, indices=N)
        dist_to_bp = dist_to_bp[:N]
    else:
        dist_to_bp = np.full(N, np.inf)
        
    # Evaluate distance for each edge
    for k, (u, v) in enumerate(edges):
        err = edge_error[k]
        if err in (0, 1): # 0 = Correct, 1 = Split
            i = node_to_idx[u]
            j = node_to_idx[v]
            # Determine nearest distance using endpoints of the edge
            d = min(dist_to_bp[i], dist_to_bp[j])
            
            if np.isinf(d):
                continue
                
            if err == 0:
                correct_distances.append(d)
            elif err == 1:
                split_distances.append(d)
                
    # Perform cleanup to avoid hitting memory limits across loops
    del payload
    del gt
    if adj is not None:
        del adj
    del dist_to_bp
    gc.collect()

split_distances = np.array(split_distances)
correct_distances = np.array(correct_distances)

print(f"Total Correct edges analyzed: {len(correct_distances)}")
print(f"Total Split edges analyzed: {len(split_distances)}")

if len(split_distances) > 0 and len(correct_distances) > 0:
    stat, p_val = mannwhitneyu(split_distances, correct_distances, alternative='two-sided')
    print(f"Mann-Whitney U statistic: {stat}")
    print(f"p-value: {p_val:.4e}")

    # Generate side-by-side violin plots
    fig, ax = plt.subplots(figsize=(8, 6))
    
    parts = ax.violinplot([correct_distances, split_distances], showmeans=True, showmedians=True)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['Correct Edges', 'Split Edges'])
    ax.set_ylabel('Geodesic Distance to Nearest Branch Point (\u00b5m)')
    ax.set_title('Topological Clustering: Distance to Nearest Branch Point')
    
    # Limit the y-axis to the 99th percentile to prevent distant outliers from squishing the violins
    all_dists = np.concatenate([correct_distances, split_distances])
    p99 = np.percentile(all_dists, 99)
    if p99 > 0:
        ax.set_ylim(-p99 * 0.05, p99)
    
    plt.tight_layout()
    plt.show()
else:
    print("Not enough data to run tests and plot.")
