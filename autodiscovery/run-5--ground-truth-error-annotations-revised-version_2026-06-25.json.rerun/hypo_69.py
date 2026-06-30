import subprocess
import sys
import glob
import os
import pickle

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", package])

# Ensure all dependencies are available, specifically checking modules that could fail during unpickling
dependencies = ["numpy", "networkx", "matplotlib", "scipy", "pandas", "tqdm", "psutil", "tensorstore"]
for dep in dependencies:
    try:
        __import__(dep)
    except ImportError:
        install(dep)

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

try:
    from agentic_neuron_proofreader.data_modules import graph_classes
except ImportError:
    install("https://github.com/AllenInstitute/agentic-neuron-proofreader/archive/refs/heads/main.zip")
    from agentic_neuron_proofreader.data_modules import graph_classes

import agentic_neuron_proofreader

# Load datasets
paths = glob.glob("../*add.pkl")
if not paths:
    paths = glob.glob("*add.pkl")
if not paths:
    paths = glob.glob("cache/*add.pkl")
if not paths:
    print("No datasets found.")
    sys.exit(0)

EDGE_OMIT = 2
MIN_CABLE_LENGTH = 100.0

all_omit_lengths = []
total_omit_length = 0.0
short_omit_length = 0.0

for path in paths:
    with open(path, "rb") as f:
        payload = pickle.load(f)
    
    gt = payload["gt_graph"]
    edge_error = np.asarray(payload["gt_edge_error"])
    node_xyz = gt.node_xyz
    
    # Extract omit edges
    edges = list(gt.edges)
    omit_edges = [edges[k] for k, c in enumerate(edge_error) if c == EDGE_OMIT]
    
    # Build subgraph of omit edges
    omit_graph = nx.Graph()
    omit_graph.add_edges_from(omit_edges)
    
    # Find connected components (continuous omitted stretches)
    components = list(nx.connected_components(omit_graph))
    
    for comp in components:
        comp_subgraph = omit_graph.subgraph(comp)
        
        comp_length = 0.0
        for u, v in comp_subgraph.edges:
            p1 = node_xyz[u]
            p2 = node_xyz[v]
            comp_length += np.linalg.norm(p1 - p2)
            
        all_omit_lengths.append(comp_length)
        total_omit_length += comp_length
        if comp_length < MIN_CABLE_LENGTH:
            short_omit_length += comp_length

# Summary Statistics
num_components = len(all_omit_lengths)
short_components = [l for l in all_omit_lengths if l < MIN_CABLE_LENGTH]
num_short = len(short_components)

if total_omit_length > 0:
    pct_length_short = (short_omit_length / total_omit_length) * 100
else:
    pct_length_short = 0.0

if num_components > 0:
    pct_count_short = (num_short / num_components) * 100
else:
    pct_count_short = 0.0

print("=== Omit Errors Stretch Summary ===")
print(f"Total omitted stretches: {num_components}")
print(f"Total omitted cable length: {total_omit_length:.2f} µm")
print(f"Stretches < {MIN_CABLE_LENGTH} µm: {num_short} ({pct_count_short:.2f}%)")
print(f"Length from stretches < {MIN_CABLE_LENGTH} µm: {short_omit_length:.2f} µm ({pct_length_short:.2f}%)")

# Plotting
if num_components > 0:
    plt.figure(figsize=(10, 6))
    plt.hist(all_omit_lengths, bins=50, color='skyblue', edgecolor='black')
    plt.axvline(x=MIN_CABLE_LENGTH, color='red', linestyle='dashed', linewidth=2, label=f'Threshold ({MIN_CABLE_LENGTH} µm)')
    plt.title('Distribution of Omitted Stretch Lengths')
    plt.xlabel('Length of Continuous Omitted Stretch (µm)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.tight_layout()
    plt.show()
