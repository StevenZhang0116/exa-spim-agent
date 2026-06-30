import sys
import subprocess

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", package])

for pkg in ["psutil", "tensorstore", "scipy", "statsmodels"]:
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
import glob
import os
import re
import gc
import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf

def brain_id_from_path(path):
    m = re.search(r"dataset_cache_(\d+)_mcl(\d+)_add\.pkl$", os.path.basename(path))
    return m.group(1) if m else os.path.basename(path)

data = []

for path in sorted(glob.glob("../data/*_add.pkl")):
    brain_id = brain_id_from_path(path)
    print(f"Processing {brain_id}...")
    with open(path, "rb") as f:
        payload = pickle.load(f)
    
    gt = payload["gt_graph"]
    edge_error = np.asarray(payload["gt_edge_error"])
    
    edges = list(gt.edges)
    node_xyz = gt.node_xyz
    
    # Pre-calculate spatial distances and populate them as edge weights
    for u, v in edges:
        dist = np.linalg.norm(node_xyz[u] - node_xyz[v])
        gt[u][v]['weight'] = dist
        
    # Leaf nodes are degree 1
    leaves = [n for n, d in gt.degree() if d == 1]
    
    # Shortest path to nearest leaf from every node
    if leaves:
        dists = nx.multi_source_dijkstra_path_length(gt, leaves, weight='weight')
    else:
        dists = {}
        
    for i, (u, v) in enumerate(edges):
        du = dists.get(u, np.inf)
        dv = dists.get(v, np.inf)
        if np.isinf(du) and np.isinf(dv):
            continue
        
        weight = gt[u][v]['weight']
        # Midpoint shortest distance to leaf
        d_leaf = min(du, dv) + weight / 2.0
        
        # split class encoded as 1
        is_split = 1 if edge_error[i] == 1 else 0
        neuron_id = gt.node_segment_id(u)
        
        data.append({
            "brain_id": brain_id,
            "neuron_id": neuron_id,
            "dist_to_leaf": d_leaf,
            "is_split": is_split
        })
        
    # Free up memory
    del gt
    del payload
    gc.collect()

df = pd.DataFrame(data)
print(f"\nTotal valid edges processed: {len(df)}\n")

# --- 1. Split Error Rate by Descriptive Bins ---
desc_bins = [-1, 50, 200, np.inf]
desc_labels = ["<50um", "50-200um", ">200um"]
df['desc_bin'] = pd.cut(df['dist_to_leaf'], bins=desc_bins, labels=desc_labels)
desc_stats = df.groupby('desc_bin', observed=True)['is_split'].agg(['mean', 'count']).reset_index()

print("Split Error Rate by Descriptive Categories:")
for _, row in desc_stats.iterrows():
    print(f"  {row['desc_bin']:>8s}: {row['mean']*100:.2f}% splits (n={row['count']})")
print()

# --- 2. Line Plot ---
bins = list(range(0, 501, 50)) + [np.inf]
labels = [f"{bins[i]}-{bins[i+1]}" for i in range(len(bins)-2)] + [">500"]
df['plot_bin'] = pd.cut(df['dist_to_leaf'], bins=bins, labels=labels, right=False)
plot_stats = df.groupby('plot_bin', observed=True)['is_split'].agg(['mean', 'count']).reset_index()

plt.figure(figsize=(10, 6))
plt.plot(plot_stats['plot_bin'].astype(str), plot_stats['mean'] * 100, marker='o', linestyle='-', color='indigo')
plt.xlabel("Distance to Nearest Leaf Tip (um)")
plt.ylabel("Split Rate (%)")
plt.title("Edge Split Rate as a Function of Distance to Leaf Tip")
plt.xticks(rotation=45)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# --- 3. Mixed-effects Logistic Regression ---
# Standardize continuous predictor for modeling stability
df['dist_to_leaf_std'] = (df['dist_to_leaf'] - df['dist_to_leaf'].mean()) / df['dist_to_leaf'].std()
df['group_id'] = df['brain_id'].astype(str) + "_" + df['neuron_id'].astype(str)

print("Fitting Mixed-Effects (GEE) Logistic Regression predicting splits...")
try:
    # Using GEE to natively handle clustered correlations
    gee_model = smf.gee("is_split ~ dist_to_leaf_std", df, groups=df["group_id"], family=sm.families.Binomial())
    gee_result = gee_model.fit()
    print(gee_result.summary())
except Exception as e:
    print("Could not fit GEE model:", e)
    print("Falling back to standard logistic regression...")
    try:
        logit_model = smf.logit("is_split ~ dist_to_leaf_std", df)
        logit_result = logit_model.fit()
        print(logit_result.summary())
    except Exception as e2:
        print("Could not fit logit model:", e2)
