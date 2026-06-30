import subprocess
import sys
import os

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", package])

# Install missing dependencies identified from the traceback and previous explorations
for pkg in ["psutil", "pandas", "tensorstore", "statsmodels"]:
    try:
        __import__(pkg)
    except ImportError:
        install(pkg)

# Ensure agentic_neuron_proofreader is installed
try:
    import agentic_neuron_proofreader
except ImportError:
    # Since git is not available in the environment, install directly from the GitHub zip archive
    try:
        install("https://github.com/AllenInstitute/agentic-neuron-proofreader/archive/refs/heads/main.zip")
    except subprocess.CalledProcessError:
        install("https://github.com/AllenInstitute/agentic-neuron-proofreader/archive/refs/heads/master.zip")
    import agentic_neuron_proofreader

# Ensure matplotlib is installed for plotting
try:
    import matplotlib.pyplot as plt
except ImportError:
    install("matplotlib")
    import matplotlib.pyplot as plt

import glob
import pickle
import numpy as np
import networkx as nx
from scipy.stats import mannwhitneyu
import pathlib

def main():
    # 1. Locate and load the dataset
    dataset_path = os.environ["RERUN_PKL"]
    print(f"Loading dataset from: {dataset_path}")

    with open(dataset_path, "rb") as f:
        payload = pickle.load(f)

    gt = payload["gt_graph"]
    edge_error = np.asarray(payload["gt_edge_error"])
    
    print(f"Graph has {gt.number_of_nodes()} nodes and {gt.number_of_edges()} edges.")

    # 2. Add spatial weights to edges (Euclidean distance in \u00B5m)
    print("Calculating edge weights based on spatial distance...")
    for u, v, data in gt.edges(data=True):
        dist = np.linalg.norm(gt.node_xyz[u] - gt.node_xyz[v])
        data['weight'] = dist

    # 3. Identify ground-truth branch points
    branch_nodes = {n for n, d in gt.degree() if d >= 3}
    print(f"Found {len(branch_nodes)} GT branch nodes.")
    if not branch_nodes:
        print("No branch nodes found in gt_graph. Cannot test hypothesis.")
        return

    # 4. Compute shortest topological paths to the nearest branch node
    print("Computing shortest topological paths to nearest branch node...")
    dist_to_branch = nx.multi_source_dijkstra_path_length(gt, branch_nodes, weight='weight')
    print(f"Computed distances for {len(dist_to_branch)} nodes.")

    # 5. Extract edge distances to nearest branch point
    EDGE_CORRECT = 0
    EDGE_SPLIT = 1

    split_distances = []
    correct_distances = []

    print("Classifying edges and extracting distances...")
    for i, (u, v) in enumerate(list(gt.edges)):
        err = edge_error[i]
        # Only consider edges where both endpoints are in a connected component with a branch node
        if u in dist_to_branch and v in dist_to_branch:
            # Assign distance as the average of its endpoints' distances to the nearest branch
            edge_dist = (dist_to_branch[u] + dist_to_branch[v]) / 2.0
            if err == EDGE_SPLIT:
                split_distances.append(edge_dist)
            elif err == EDGE_CORRECT:
                correct_distances.append(edge_dist)

    split_distances = np.array(split_distances)
    correct_distances = np.array(correct_distances)

    print(f"Number of EDGE_SPLIT edges analyzed: {len(split_distances)}")
    print(f"Number of EDGE_CORRECT edges analyzed: {len(correct_distances)}")

    if len(split_distances) == 0 or len(correct_distances) == 0:
        print("Not enough data in one of the categories to perform statistical test.")
        return

    # 6. Perform statistical test
    print("\nPerforming Mann-Whitney U test...")
    stat, p_value = mannwhitneyu(split_distances, correct_distances, alternative='two-sided')
    
    print("\n--- Results ---")
    print(f"Mann-Whitney U statistic: {stat}")
    print(f"p-value: {p_value:.5e}")
    print(f"Mean distance to branch (EDGE_SPLIT): {np.mean(split_distances):.2f} \u00B5m")
    print(f"Mean distance to branch (EDGE_CORRECT): {np.mean(correct_distances):.2f} \u00B5m")
    print(f"Median distance to branch (EDGE_SPLIT): {np.median(split_distances):.2f} \u00B5m")
    print(f"Median distance to branch (EDGE_CORRECT): {np.median(correct_distances):.2f} \u00B5m")

    # 7. Plot Cumulative Density Function (CDF)
    print("\nPlotting CDF...")
    plt.figure(figsize=(10, 6))

    x_split = np.sort(split_distances)
    y_split = np.arange(1, len(x_split) + 1) / len(x_split)

    x_correct = np.sort(correct_distances)
    y_correct = np.arange(1, len(x_correct) + 1) / len(x_correct)

    plt.plot(x_split, y_split, label='EDGE_SPLIT', color='red', linewidth=2)
    plt.plot(x_correct, y_correct, label='EDGE_CORRECT', color='blue', linewidth=2)

    plt.title('CDF of Topological Distance to Nearest GT Branch Point')
    plt.xlabel('Distance to Nearest Branch (\u00B5m)')
    plt.ylabel('Cumulative Probability')
    
    # Cap x-axis to 95th percentile to clearly visualize the main body of the distribution without the long tail obscuring it
    max_val = max(np.percentile(split_distances, 95), np.percentile(correct_distances, 95))
    plt.xlim(0, max_val * 1.1)
    
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
