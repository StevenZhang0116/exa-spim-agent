import sys
import subprocess

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", package])

# Install required dependencies
for pkg in ["psutil", "pandas", "networkx", "scipy", "tensorstore", "matplotlib"]:
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
import networkx as nx
import glob
from scipy.stats import mannwhitneyu

def main():
    files = glob.glob("*_add.pkl")
    if not files:
        print("No dataset files found.")
        return

    bridged_lengths = []
    broken_lengths = []

    for file in files:
        with open(file, "rb") as f:
            payload = pickle.load(f)
            
        gt = payload["gt_graph"]
        edge_error = np.asarray(payload["gt_edge_error"])
        node_label = np.asarray(payload["gt_node_canonical_label"])
        
        EDGE_OMIT = 2
        edges = list(gt.edges)
        
        # Filter for omit edges
        omit_edges = [e for i, e in enumerate(edges) if edge_error[i] == EDGE_OMIT]
        
        # Create an induced subgraph of omit edges
        omit_subgraph = nx.Graph()
        omit_subgraph.add_edges_from(omit_edges)
        
        # Extract continuous omit paths as connected components
        components = list(nx.connected_components(omit_subgraph))
        
        for comp in components:
            comp_subgraph = omit_subgraph.subgraph(comp)
            
            # Calculate the total physical length of the omit path
            path_length = 0.0
            for u, v in comp_subgraph.edges:
                p1 = gt.node_xyz[u]
                p2 = gt.node_xyz[v]
                dist = np.linalg.norm(p1 - p2)
                path_length += dist
                
            # Identify the non-omit GT nodes immediately preceding and succeeding the omit path
            flanking_nodes = set()
            for node in comp:
                for neighbor in gt.neighbors(node):
                    if neighbor not in comp:
                        flanking_nodes.add(neighbor)
            
            # Classify path as bridged or broken based on its flanking nodes
            if len(flanking_nodes) >= 2:
                segs = [int(node_label[n]) for n in flanking_nodes]
                # Bridged if all flanking segment IDs are identical and non-zero
                if len(set(segs)) == 1 and segs[0] != 0:
                    bridged_lengths.append(path_length)
                else:
                    broken_lengths.append(path_length)
            else:
                # If there are <2 flanking nodes, it represents a true termination or completely omitted structure
                broken_lengths.append(path_length)
                
    print(f"Total Bridged Omit Paths: {len(bridged_lengths)}")
    if bridged_lengths:
        print(f"Mean Length of Bridged Omit Paths: {np.mean(bridged_lengths):.2f} µm")
        print(f"Median Length of Bridged Omit Paths: {np.median(bridged_lengths):.2f} µm")
        
    print(f"\nTotal Broken Omit Paths: {len(broken_lengths)}")
    if broken_lengths:
        print(f"Mean Length of Broken Omit Paths: {np.mean(broken_lengths):.2f} µm")
        print(f"Median Length of Broken Omit Paths: {np.median(broken_lengths):.2f} µm")
        
    if bridged_lengths and broken_lengths:
        stat, p = mannwhitneyu(bridged_lengths, broken_lengths, alternative='less')
        print(f"\nMann-Whitney U test (Bridged < Broken): U = {stat}, p = {p:.4e}")
        if p < 0.05:
            print("Conclusion: Bridged omit paths are significantly shorter than broken omit paths.")
        else:
            print("Conclusion: Bridged omit paths are not significantly shorter than broken omit paths.")

if __name__ == "__main__":
    main()
