import sys
import subprocess
import os
import glob

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", package])

# Install necessary dependencies
packages = ["psutil", "pandas", "scipy", "matplotlib", "tensorstore"]
for pkg in packages:
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
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
from collections import defaultdict

# Find dataset paths
file_paths = glob.glob("*_add.pkl")
if not file_paths:
    print("No dataset files found.")
    sys.exit(1)

merge_lengths = []
non_merge_lengths = []

for file_path in file_paths:
    with open(file_path, "rb") as f:
        payload = pickle.load(f)
    
    fg = payload["fragments_graph"]
    gt_merge_labels = set(int(float(x)) for x in payload["gt_merge_labels"])
    
    node_component_id = fg.node_component_id
    node_xyz = fg.node_xyz
    comp_to_swc = fg.component_id_to_swc_id
    
    segment_lengths = defaultdict(float)
    
    # Calculate total cable length for each unique globally predicted segment id (swc_id)
    for u, v in fg.edges:
        comp_u = int(node_component_id[u])
        
        # Map local component id to global swc_id
        if isinstance(comp_to_swc, dict):
            swc_id = comp_to_swc.get(comp_u)
        else:
            swc_id = comp_to_swc[comp_u]
            
        if swc_id is None:
            continue
            
        swc_id = int(float(swc_id))
        
        # Euclidean distance for edge
        dist = np.linalg.norm(node_xyz[u] - node_xyz[v])
        segment_lengths[swc_id] += dist
        
    # Group the segment lengths by checking against gt_merge_labels
    for seg_id, length in segment_lengths.items():
        if seg_id in gt_merge_labels:
            merge_lengths.append(length)
        else:
            non_merge_lengths.append(length)

merge_median = np.median(merge_lengths) if merge_lengths else 0
non_merge_median = np.median(non_merge_lengths) if non_merge_lengths else 0

print(f"Number of merge segments: {len(merge_lengths)}")
print(f"Number of non-merge segments: {len(non_merge_lengths)}")
print(f"Median length of merge segments: {merge_median:.2f} \u00B5m")
print(f"Median length of non-merge segments: {non_merge_median:.2f} \u00B5m")

# Perform Mann-Whitney U test
if len(merge_lengths) > 0 and len(non_merge_lengths) > 0:
    stat, pval = mannwhitneyu(merge_lengths, non_merge_lengths, alternative='two-sided')
    print(f"Mann-Whitney U test statistic: {stat}, p-value: {pval}")
else:
    print("Not enough data to perform Mann-Whitney U test.")

# Boxplot generation
if merge_lengths and non_merge_lengths:
    plt.figure(figsize=(8, 6))
    plt.boxplot([merge_lengths, non_merge_lengths], labels=["Merge Segments", "Non-Merge Segments"])
    plt.yscale('log')
    plt.ylabel("Total Cable Length (\u00B5m) (log scale)")
    plt.title("Cable Length of Merge vs. Non-Merge Segments")
    plt.show()
