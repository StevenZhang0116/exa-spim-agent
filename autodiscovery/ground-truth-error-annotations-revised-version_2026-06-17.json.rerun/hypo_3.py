import subprocess
import sys
import os

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", package])

for pkg in ["psutil", "pandas", "tensorstore", "scipy", "networkx"]:
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
import scipy.stats as stats
import random
from scipy.spatial import KDTree

def main():
    # Revised loader: load $RERUN_PKL directly (os.walk path-search is not
    # intercepted by the runner's open/glob monkeypatches).
    path = os.environ["RERUN_PKL"]
    print(f"Loading dataset from: {path}")
    with open(path, "rb") as f:
        payload = pickle.load(f)
    
    gt = payload["gt_graph"]
    fragments = payload["fragments_graph"]
    edge_error = np.asarray(payload["gt_edge_error"])
    merge_sites = payload["gt_merge_sites"]
    
    # 1. Extract xyz spatial coordinates from gt_merge_sites
    merge_xyz = [ms["xyz"] for ms in merge_sites]
    
    # 2. Define control set
    EDGE_CORRECT = 0
    node_edge_errors = {n: [] for n in gt.nodes}
    for (u, v), err in zip(list(gt.edges), edge_error):
        node_edge_errors[u].append(err)
        node_edge_errors[v].append(err)
        
    correct_nodes = []
    for n, errs in node_edge_errors.items():
        if len(errs) > 0 and all(e == EDGE_CORRECT for e in errs):
            correct_nodes.append(n)
            
    random.seed(42)
    if len(correct_nodes) > len(merge_xyz):
        control_nodes = random.sample(correct_nodes, len(merge_xyz))
    else:
        control_nodes = correct_nodes
        
    control_xyz = [gt.node_xyz[n] for n in control_nodes]
    
    # 3. Query fragments_graph.kdtree for nodes within 10 um radius
    try:
        tree = fragments.kdtree
    except AttributeError:
        tree = KDTree(fragments.node_xyz)
        
    radius = 10.0
    
    merge_counts = []
    for xyz in merge_xyz:
        indices = tree.query_ball_point(xyz, r=radius)
        if isinstance(indices, list):
            merge_counts.append(len(indices))
        else:
            merge_counts.append(indices.size)
        
    control_counts = []
    for xyz in control_xyz:
        indices = tree.query_ball_point(xyz, r=radius)
        if isinstance(indices, list):
            control_counts.append(len(indices))
        else:
            control_counts.append(indices.size)
            
    # 4. Perform Mann-Whitney U test
    u_stat, p_val = stats.mannwhitneyu(merge_counts, control_counts, alternative='greater')
    
    print(f"Total merge sites analyzed: {len(merge_xyz)}")
    print(f"Total control sites analyzed: {len(control_xyz)}")
    print(f"Average local node density (merge sites): {np.mean(merge_counts):.2f} nodes / 10 µm radius")
    print(f"Average local node density (control sites): {np.mean(control_counts):.2f} nodes / 10 µm radius")
    print(f"Mann-Whitney U statistic: {u_stat}")
    print(f"p-value: {p_val:.4e}")

if __name__ == "__main__":
    main()
