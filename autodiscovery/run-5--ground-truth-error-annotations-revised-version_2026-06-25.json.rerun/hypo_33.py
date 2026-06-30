import subprocess
import sys
import os

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", package])

# Install missing dependencies
for pkg in ["psutil", "pandas", "tensorstore"]:
    try:
        __import__(pkg)
    except ImportError:
        install(pkg)

try:
    import agentic_neuron_proofreader
except ImportError:
    install("https://github.com/AllenInstitute/agentic-neuron-proofreader/archive/refs/heads/main.zip")

import pickle
import numpy as np
import networkx as nx
import glob
from scipy.stats import mannwhitneyu
import matplotlib.pyplot as plt

def main():
    paths = glob.glob("../*add.pkl") + glob.glob("../../*add.pkl") + glob.glob("../*/*add.pkl")
    if not paths:
        paths = glob.glob("**/*add.pkl", recursive=True)
        if not paths:
            print("Dataset not found.")
            return
            
    target_paths = [p for p in paths if '794495' in p]
    path = target_paths[0] if target_paths else paths[0]
    
    print(f"Loading dataset from: {path}")
    with open(path, "rb") as f:
        payload = pickle.load(f)
        
    gt = payload["gt_graph"]
    edge_error = np.asarray(payload["gt_edge_error"])
    
    edges = list(gt.edges)
    nodes_u = [e[0] for e in edges]
    nodes_v = [e[1] for e in edges]
    
    try:
        xyz_u = gt.node_xyz[nodes_u]
        xyz_v = gt.node_xyz[nodes_v]
    except Exception:
        try:
            xyz_u = np.array([gt.nodes[u]['node_xyz'] for u in nodes_u])
            xyz_v = np.array([gt.nodes[v]['node_xyz'] for v in nodes_v])
        except Exception:
            xyz_u = np.array([gt.node_xyz[u] for u in nodes_u])
            xyz_v = np.array([gt.node_xyz[v] for v in nodes_v])
        
    lengths = np.linalg.norm(xyz_u - xyz_v, axis=1)
    
    for (u, v), length in zip(edges, lengths):
        gt[u][v]['weight'] = length
        
    degrees = dict(gt.degree())
    branch_nodes = {n for n, d in degrees.items() if d >= 3}
    
    distances_to_branch = {}
    for comp in nx.connected_components(gt):
        comp_branch_nodes = branch_nodes.intersection(comp)
        if not comp_branch_nodes:
            continue
            
        subg = gt.subgraph(comp)
        dist_dict = nx.multi_source_dijkstra_path_length(subg, comp_branch_nodes, weight='weight')
        distances_to_branch.update(dist_dict)
        
    split_distances = []
    correct_distances = []
    
    EDGE_CORRECT = 0
    EDGE_SPLIT = 1
    
    for k, (u, v) in enumerate(edges):
        if u in distances_to_branch and v in distances_to_branch:
            d_u = distances_to_branch[u]
            d_v = distances_to_branch[v]
            
            edge_dist = min(d_u, d_v) + lengths[k] / 2
            
            c = edge_error[k]
            if c == EDGE_SPLIT:
                split_distances.append(edge_dist)
            elif c == EDGE_CORRECT:
                correct_distances.append(edge_dist)
                
    split_distances = np.array(split_distances)
    correct_distances = np.array(correct_distances)
    
    print(f"Number of split edges near branches analyzed: {len(split_distances)}")
    print(f"Number of correct edges near branches analyzed: {len(correct_distances)}")
    
    if len(split_distances) == 0 or len(correct_distances) == 0:
        print("Not enough edges to compare.")
        return
        
    stat, pval = mannwhitneyu(split_distances, correct_distances, alternative='two-sided')
    
    print(f"Mann-Whitney U test statistic: {stat}")
    print(f"p-value: {pval}")
    print(f"Median distance for split edges: {np.median(split_distances):.2f} \u00b5m")
    print(f"Median distance for correct edges: {np.median(correct_distances):.2f} \u00b5m")
    
    plt.figure(figsize=(8, 6))
    plt.boxplot([correct_distances, split_distances], labels=['Correct Edges', 'Split Edges'], showfliers=False)
    plt.ylabel('Distance to Nearest Branch Node (\u00b5m)')
    plt.title('Distance from Edges to Nearest Branch Node\n(Outliers Hidden)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

if __name__ == "__main__":
    main()
