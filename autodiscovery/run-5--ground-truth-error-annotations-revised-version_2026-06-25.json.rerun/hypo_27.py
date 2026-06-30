import sys
import subprocess
import glob
import pickle
import numpy as np
from collections import defaultdict

# Helper for installing packages if missing
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", package])

# Ensure all dependencies are met before importing
for pkg in ["psutil", "pandas", "scipy", "networkx", "matplotlib", "tqdm", "tensorstore"]:
    try:
        __import__(pkg)
    except ImportError:
        install(pkg)

try:
    import agentic_neuron_proofreader
except ImportError:
    install("https://github.com/AllenInstitute/agentic-neuron-proofreader/archive/refs/heads/main.zip")
    import agentic_neuron_proofreader

import networkx as nx
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

# --- 1. Load Dataset ---
cache_files = glob.glob("../dataset_cache_*_add.pkl")
if not cache_files:
    cache_files = glob.glob("dataset_cache_*_add.pkl")
if not cache_files:
    print("No dataset files found.")
    sys.exit(1)

EDGE_CORRECT = 0
EDGE_SPLIT = 1
EDGE_OMIT = 2
EDGE_MERGED = 3

p_baseline = []
p_conditional = []
all_omit_run_lengths = []
transition_matrix = np.zeros((4, 4), dtype=int)

total_neurons_analyzed = 0

# --- 2. Process Data ---
for path_to_cache in cache_files:
    with open(path_to_cache, "rb") as f:
        payload = pickle.load(f)

    gt = payload["gt_graph"]
    edge_error = np.asarray(payload["gt_edge_error"])
    edge_list = list(gt.edges)
    
    # Group edges by neuron ID
    neuron_edges = defaultdict(list)
    for i, e in enumerate(edge_list):
        u, v = e
        neuron_id = gt.node_segment_id(u)
        neuron_edges[neuron_id].append((e, edge_error[i]))
        
    for neuron_id, edges_and_errors in neuron_edges.items():
        total_edges = len(edges_and_errors)
        if total_edges < 2:
            continue
            
        # Create a fast lookup for error class by sorted edge tuple
        local_err = {}
        for e, err in edges_and_errors:
            e_sorted = tuple(sorted(e))
            local_err[e_sorted] = err
            
        # Baseline P(Omit)
        omit_count = sum(1 for _, err in edges_and_errors if err == EDGE_OMIT)
        base_p = omit_count / total_edges
        
        # Build adjacency mapping for edges on this neuron
        incidence = defaultdict(list)
        for e, _ in edges_and_errors:
            incidence[e[0]].append(e)
            incidence[e[1]].append(e)
            
        total_neighbors_of_omit = 0
        omit_neighbors_of_omit = 0
        
        # Calculate edge-to-edge transition probabilities
        for node, adj_edges in incidence.items():
            for i in range(len(adj_edges)):
                for j in range(i + 1, len(adj_edges)):
                    e1 = tuple(sorted(adj_edges[i]))
                    e2 = tuple(sorted(adj_edges[j]))
                    err1 = local_err[e1]
                    err2 = local_err[e2]
                    
                    # Symmetrically increment transition matrix for adjacent edge pairs
                    transition_matrix[err1, err2] += 1
                    transition_matrix[err2, err1] += 1
                    
                    if err1 == EDGE_OMIT:
                        total_neighbors_of_omit += 1
                        if err2 == EDGE_OMIT:
                            omit_neighbors_of_omit += 1
                            
                    if err2 == EDGE_OMIT:
                        total_neighbors_of_omit += 1
                        if err1 == EDGE_OMIT:
                            omit_neighbors_of_omit += 1
                            
        # Only include neurons that have at least one valid adjacency originating from an omit edge
        if total_neighbors_of_omit > 0:
            cond_p = omit_neighbors_of_omit / total_neighbors_of_omit
            p_baseline.append(base_p)
            p_conditional.append(cond_p)
            total_neurons_analyzed += 1
            
        # Omit edge run lengths (measured as edge counts in omit-connected subgraphs)
        omit_edges = [e for e, err in edges_and_errors if err == EDGE_OMIT]
        if omit_edges:
            G_omit = nx.Graph()
            G_omit.add_edges_from(omit_edges)
            for cc in nx.connected_components(G_omit):
                run_length = G_omit.subgraph(cc).number_of_edges()
                if run_length > 0:
                    all_omit_run_lengths.append(run_length)
                    
    del payload
    del gt
    del edge_error

# --- 3. Statistical Analysis ---
p_baseline = np.array(p_baseline)
p_conditional = np.array(p_conditional)

if len(p_baseline) > 1:
    diffs = p_conditional - p_baseline
    if np.all(diffs == 0):
        stat, pval = 0.0, 1.0
    else:
        stat, pval = wilcoxon(p_baseline, p_conditional)
else:
    stat, pval = float('nan'), float('nan')

print("=== Omit Error Clustering Analysis ===")
print(f"Number of neurons analyzed (with valid omit adjacencies): {total_neurons_analyzed}")
if total_neurons_analyzed > 0:
    print(f"Mean Baseline P(Omit): {np.mean(p_baseline):.4f}")
    print(f"Mean Conditional P(Omit | adjacent is Omit): {np.mean(p_conditional):.4f}")
    print(f"Wilcoxon signed-rank test: W={stat}, p={pval:.4e}")

print("\n=== Edge Error Transition Probability Matrix ===")
classes = ["Correct", "Split", "Omit", "Merged"]
print(f"{'':>10} | " + " | ".join(f"{c:>8}" for c in classes))
print("-" * 55)
for i, row_name in enumerate(classes):
    row_str = f"{row_name:>10} | "
    row_total = sum(transition_matrix[i, j] for j in range(4))
    if row_total > 0:
        row_str += " | ".join(f"{transition_matrix[i, j]/row_total:>8.3f}" for j in range(4))
    else:
        row_str += " | ".join(f"{0:>8.3f}" for j in range(4))
    print(row_str)

# --- 4. Plot Results ---
if all_omit_run_lengths:
    print(f"\n=== Omit Run-Length Distribution ===")
    print(f"Total omit runs: {len(all_omit_run_lengths)}")
    print(f"Max run length: {max(all_omit_run_lengths)}")
    print(f"Mean run length: {np.mean(all_omit_run_lengths):.2f}")
    
    plt.figure(figsize=(12, 5))
    
    # Boxplot of Probabilities
    plt.subplot(1, 2, 1)
    if len(p_baseline) > 0:
        plt.boxplot([p_baseline, p_conditional], labels=["Baseline\nP(Omit)", "Conditional\nP(Omit|Omit)"])
        plt.ylabel("Probability")
        plt.title("Omit Probabilities per Neuron")
    else:
        plt.text(0.5, 0.5, "No data for boxplot", ha='center', va='center')
        
    # Histogram of Run Lengths
    plt.subplot(1, 2, 2)
    max_rl = min(max(all_omit_run_lengths), 50)
    bins = np.arange(1, max_rl + 2) - 0.5
    plt.hist(all_omit_run_lengths, bins=bins, edgecolor='black', color='skyblue')
    
    runs_exceeding = sum(x > max_rl for x in all_omit_run_lengths)
    if runs_exceeding > 0:
        plt.text(0.5, 0.9, f"{runs_exceeding} runs > {max_rl}", 
                 transform=plt.gca().transAxes, ha='center')
    plt.xlabel("Consecutive Omit Edges (Run Length)")
    plt.ylabel("Frequency")
    plt.title("Histogram of Omit Run-Lengths")
    plt.xlim(0, max_rl + 1)
    
    plt.tight_layout()
    plt.show()
else:
    print("\nNo omit runs found.")
