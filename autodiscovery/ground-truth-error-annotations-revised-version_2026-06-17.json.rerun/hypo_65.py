import subprocess
import sys
import os
import glob
import pickle
import gc

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", package])

# Ensure dependencies are installed
try:
    import agentic_neuron_proofreader
except ImportError:
    install("tensorstore")
    install("psutil")
    install("pandas")
    install("tqdm")
    install("networkx")
    install("scipy")
    install("matplotlib")
    install("https://github.com/AllenInstitute/agentic-neuron-proofreader/archive/refs/heads/main.zip")

import agentic_neuron_proofreader
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx

EDGE_CORRECT = 0
EDGE_SPLIT = 1

data = []

# Safely locate the dataset files
search_dirs = ['.', '..', '/data', '/workspace']
pkl_files = []
for d in search_dirs:
    pkl_files.extend(glob.glob(os.path.join(d, '*_add.pkl')))
pkl_files = list(set(pkl_files))

if not pkl_files:
    print("No dataset files found.")
    sys.exit(1)

# Process each dataset file
for path in pkl_files:
    print(f"Processing {os.path.basename(path)}...")
    basename = os.path.basename(path)
    parts = basename.split('_')
    brain_id = parts[2] if len(parts) > 2 else 'unknown'
    
    with open(path, 'rb') as f:
        payload = pickle.load(f)
    
    gt = payload["gt_graph"]
    edge_error = np.asarray(payload["gt_edge_error"])
    edges = list(gt.edges)
    node_xyz = gt.node_xyz
    
    # 2. Build weighted graph for morphological proxy calculation
    G = nx.Graph()
    edges_with_weights = []
    for u, v in edges:
        w = np.linalg.norm(node_xyz[u] - node_xyz[v])
        edges_with_weights.append((u, v, w))
    G.add_weighted_edges_from(edges_with_weights)
    
    # Identify terminal processes (leaves)
    leaves = [n for n, d in G.degree() if d <= 1]
    
    if leaves:
        # Compute topological distance to the nearest leaf for all reachable nodes
        dist_dict = nx.multi_source_dijkstra_path_length(G, set(leaves), weight='weight')
        
        # 3. Aggregate proxy thickness and edge errors
        for k, (u, v) in enumerate(edges):
            err = edge_error[k]
            if err in (EDGE_CORRECT, EDGE_SPLIT):
                if u in dist_dict and v in dist_dict:
                    dist_u = dist_dict[u]
                    dist_v = dist_dict[v]
                    avg_dist = (dist_u + dist_v) / 2.0
                    data.append((avg_dist, err, brain_id))
    
    # Manual cleanup to handle memory limits across multiple brains
    del G
    del edges_with_weights
    del gt
    del payload
    gc.collect()

# Combine and format data
df = pd.DataFrame(data, columns=['dist', 'error_class', 'brain_id']).dropna()
split_df = df[df['error_class'] == EDGE_SPLIT]
correct_df = df[df['error_class'] == EDGE_CORRECT]

print(f"Total Correct Edges (Before downsampling): {len(correct_df)}")
print(f"Total Split Edges: {len(split_df)}")

if len(split_df) == 0:
    print("No split edges found. Cannot perform statistical test.")
    sys.exit(0)

# 4. Statistical downsampling for robust testing without overflow
if len(correct_df) > 50000:
    correct_df = correct_df.sample(n=50000, random_state=42)

print(f"Total Correct Edges (After downsampling): {len(correct_df)}")

split_dist = split_df['dist']
correct_dist = correct_df['dist']

print("\nSummary Statistics:")
print(f"Correct Edges - Mean Distance-to-Leaf: {correct_dist.mean():.4f} µm, Median: {correct_dist.median():.4f} µm")
print(f"Split Edges   - Mean Distance-to-Leaf: {split_dist.mean():.4f} µm, Median: {split_dist.median():.4f} µm")

# 5. Statistical Testing
stat, p = stats.mannwhitneyu(split_dist, correct_dist, alternative='less')
print(f"\nMann-Whitney U Test (Split < Correct): U={stat}, p-value={p:.4e}")

# 6. Visualization
plt.figure(figsize=(10, 6))
plt.hist(correct_dist, bins=50, density=True, alpha=0.5, label='Correct Edges', color='blue')
plt.hist(split_dist, bins=50, density=True, alpha=0.5, label='Split Edges', color='red')
plt.xlabel('Topological Distance to Leaf (µm)')
plt.ylabel('Density')
plt.title('Morphological Proxy: Distance-to-Leaf for Correct vs. Split Edges')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
