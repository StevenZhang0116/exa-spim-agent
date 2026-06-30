import subprocess
import sys
import os

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", package])

# Ensure dependencies that might be lazily imported during unpickling are available
for pkg in ["psutil", "pandas", "tensorstore"]:
    try:
        __import__(pkg)
    except ImportError:
        try:
            install(pkg)
        except Exception as e:
            print(f"Failed to install {pkg}: {e}")
            if pkg == "tensorstore":
                # tensorstore might fail to install on Python 3.13 due to lack of wheels.
                # We mock it since it's only imported in a utility module we don't actively use.
                import types
                sys.modules['tensorstore'] = types.ModuleType('tensorstore')

try:
    import agentic_neuron_proofreader
except ImportError:
    # Install from zip archive since git might not be available in the environment
    install("https://github.com/AllenInstitute/agentic-neuron-proofreader/archive/refs/heads/main.zip")
    import agentic_neuron_proofreader

import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
from collections import defaultdict

# Constants for edge error classes
EDGE_SPLIT = 1
EDGE_OMIT = 2

# Find dataset caches
paths = glob.glob("../dataset_cache_*_add.pkl")
if not paths:
    paths = glob.glob("dataset_cache_*_add.pkl")
    if not paths:
        paths = glob.glob("cache/dataset_cache_*_add.pkl")
        if not paths:
            print("No dataset caches found.")
            sys.exit(1)

baseline_rates = []
conditional_rates = []
neuron_names = []

for path in paths:
    with open(path, "rb") as f:
        payload = pickle.load(f)
    
    gt = payload["gt_graph"]
    edge_error = np.asarray(payload["gt_edge_error"])
    edges = list(gt.edges)
    
    # Group edges by neuron
    neuron_edges = defaultdict(list)
    for k, (u, v) in enumerate(edges):
        neuron = gt.node_segment_id(u)
        neuron_edges[neuron].append(k)
        
    for neuron, edge_indices in neuron_edges.items():
        types = edge_error[edge_indices]
        total_edges = len(edge_indices)
        if total_edges == 0:
            continue
            
        num_omits = np.count_nonzero(types == EDGE_OMIT)
        num_splits = np.count_nonzero(types == EDGE_SPLIT)
        
        # Filter out neurons with no split edges
        if num_splits == 0:
            continue
            
        # Baseline omit rate
        baseline_rate = num_omits / total_edges
        
        # Find split edges and the nodes they touch
        split_edge_indices = set(idx for idx in edge_indices if edge_error[idx] == EDGE_SPLIT)
        split_nodes = set()
        for idx in split_edge_indices:
            u, v = edges[idx]
            split_nodes.add(u)
            split_nodes.add(v)
            
        # Find adjacent edges (exclude split edges themselves to avoid mechanically lowering omit rate)
        adjacent_edge_indices = []
        for idx in edge_indices:
            if idx not in split_edge_indices:
                u, v = edges[idx]
                if u in split_nodes or v in split_nodes:
                    adjacent_edge_indices.append(idx)
                    
        if len(adjacent_edge_indices) == 0:
            continue
            
        # Conditional omit rate
        adjacent_types = edge_error[adjacent_edge_indices]
        conditional_rate = np.count_nonzero(adjacent_types == EDGE_OMIT) / len(adjacent_edge_indices)
        
        baseline_rates.append(baseline_rate)
        conditional_rates.append(conditional_rate)
        neuron_names.append(neuron)

baseline_rates = np.array(baseline_rates)
conditional_rates = np.array(conditional_rates)

if len(baseline_rates) == 0:
    print("No valid neurons found for analysis.")
else:
    differences = conditional_rates - baseline_rates
    non_zero_diffs = differences[differences != 0]
    
    print("=== Omit Rates near Split Errors ===")
    print(f"Number of neurons analyzed: {len(baseline_rates)}")
    print(f"Mean Baseline Omit Rate: {np.mean(baseline_rates):.4f}")
    print(f"Mean Conditional Omit Rate: {np.mean(conditional_rates):.4f}")
    
    if len(non_zero_diffs) == 0:
        print("All conditional omit rates are exactly equal to baseline omit rates. p-value: 1.0")
    else:
        # Perform Wilcoxon signed-rank test
        stat, p_val = wilcoxon(baseline_rates, conditional_rates)
        print(f"Wilcoxon signed-rank test statistic: {stat}")
        print(f"p-value: {p_val:.4e}")

    # Scatter plot generation
    plt.figure(figsize=(7, 7))
    plt.scatter(baseline_rates, conditional_rates, alpha=0.7, edgecolors='k', c='royalblue')
    
    max_val = max(np.max(baseline_rates), np.max(conditional_rates)) * 1.1
    if max_val == 0:
        max_val = 1.0
        
    plt.plot([0, max_val], [0, max_val], 'r--', label='y=x (No Difference)')
    plt.xlim(0, max_val)
    plt.ylim(0, max_val)
    
    plt.xlabel("Baseline Omit Rate")
    plt.ylabel("Conditional Omit Rate (Adjacent to Splits)")
    plt.title("Baseline vs Conditional Omit Rates per Neuron")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()
