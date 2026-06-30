import subprocess
import sys
import os

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", package])

# Install missing dependencies required by the agentic_neuron_proofreader package
packages = ['psutil', 'tensorstore', 'pandas', 'networkx', 'scipy', 'tqdm', 'matplotlib']
for pkg in packages:
    try:
        __import__(pkg)
    except ImportError:
        install(pkg)

try:
    import agentic_neuron_proofreader
except ImportError:
    install("https://github.com/AllenInstitute/agentic-neuron-proofreader/archive/main.zip")
    import agentic_neuron_proofreader

import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt

# Locate dataset cache files (one level above as indicated)
dataset_paths = glob.glob("../dataset_cache_*_add.pkl")
if not dataset_paths:
    # Fallback to current directory
    dataset_paths = glob.glob("dataset_cache_*_add.pkl")

split_distances = []

for path in dataset_paths:
    with open(path, "rb") as f:
        payload = pickle.load(f)
    
    gt = payload["gt_graph"]
    node_label = np.asarray(payload["gt_node_canonical_label"])
    
    # Iterate through ground truth edges
    for u, v in gt.edges:
        label_u = node_label[u]
        label_v = node_label[v]
        
        # Check for split transition (different non-zero predicted segment IDs)
        if label_u != label_v and label_u > 0 and label_v > 0:
            # Access node coordinates
            if hasattr(gt, "node_xyz"):
                xyz_u = gt.node_xyz[u]
                xyz_v = gt.node_xyz[v]
            elif "xyz" in gt.nodes[u]:
                xyz_u = gt.nodes[u]["xyz"]
                xyz_v = gt.nodes[v]["xyz"]
            else:
                xyz_u = gt.nodes[u]["node_xyz"]
                xyz_v = gt.nodes[v]["node_xyz"]
            
            # Calculate Euclidean distance in micrometers
            dist = np.linalg.norm(np.array(xyz_u) - np.array(xyz_v))
            split_distances.append(dist)

if not split_distances:
    print("No split transitions found in the dataset.")
else:
    split_distances = np.array(split_distances)
    percent_under_15 = np.mean(split_distances < 15) * 100
    
    # Deliverables: Statistics
    print("=== Split Error Gap Analysis ===")
    print(f"Total split gaps analyzed: {len(split_distances)}")
    print(f"Percentage of split gaps < 15 µm: {percent_under_15:.2f}%")
    print("\n--- Hypothesis Evaluation ---")
    if percent_under_15 >= 80:
        print("Hypothesis CONFIRMED: Over 80% of split gaps are under 15 µm.")
    else:
        print("Hypothesis REJECTED: Less than 80% of split gaps are under 15 µm.")
    
    # Deliverables: Recommendation
    print("\n--- Recommendation ---")
    p90 = np.percentile(split_distances, 90)
    p95 = np.percentile(split_distances, 95)
    p99 = np.percentile(split_distances, 99)
    print(f"To capture the vast majority of split gaps, the recommended maximum search radius for an automated split-repair algorithm is:")
    print(f"- ~{p95:.2f} µm (covers 95% of splits)")
    print(f"- ~{p99:.2f} µm (covers 99% of splits)")
    
    # Deliverables: ECDF Plot
    sorted_dists = np.sort(split_distances)
    yvals = np.arange(1, len(sorted_dists) + 1) / len(sorted_dists)
    
    plt.figure(figsize=(9, 6))
    plt.plot(sorted_dists, yvals, marker='.', linestyle='none', color='b', alpha=0.5, label='Split Gap Sizes')
    plt.axvline(15, color='r', linestyle='--', linewidth=2, label='15 µm Threshold')
    plt.title('Empirical Cumulative Distribution Function (ECDF) of Split Gap Distances', fontsize=14)
    plt.xlabel('Distance between Split Segments (µm)', fontsize=12)
    plt.ylabel('Cumulative Probability', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.show()
