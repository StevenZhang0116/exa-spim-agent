import subprocess
import sys
import os
import glob
import pickle
import numpy as np
import random
import gc
from scipy.spatial import cKDTree
from scipy.stats import mannwhitneyu
import matplotlib.pyplot as plt

def setup():
    # Install missing dependencies explicitly before loading the package
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "psutil", "pandas", "tensorstore"])
    try:
        import agentic_neuron_proofreader
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "https://github.com/AllenInstitute/agentic-neuron-proofreader/archive/refs/heads/main.zip"])

setup()
import agentic_neuron_proofreader

merge_dists = []
random_dists = []

# Search for the datasets iteratively in common locations
cache_files = glob.glob("/data/**/*_add.pkl", recursive=True)
if not cache_files:
    cache_files = glob.glob("**/*_add.pkl", recursive=True)
if not cache_files:
    cache_files = glob.glob("../*_add.pkl")

# Remove duplicates and sort for deterministic processing
cache_files = sorted(list(set(cache_files)))
print(f"Found {len(cache_files)} cache files.\n")

for path in cache_files:
    print(f"Processing {os.path.basename(path)}...")
    with open(path, "rb") as f:
        payload = pickle.load(f)
    
    gt_merge_sites = payload.get("gt_merge_sites", [])
    if not gt_merge_sites:
        print("  No merge sites found. Skipping.\n")
        del payload
        gc.collect()
        continue
        
    fragments_graph = payload["fragments_graph"]
    degrees = dict(fragments_graph.degree())
    
    # Identify branch nodes (degree >= 3) and regular nodes (degree <= 2)
    branch_nodes = [n for n, d in degrees.items() if d >= 3]
    non_branch_nodes = [n for n, d in degrees.items() if d <= 2]
    
    if not branch_nodes:
        print("  No branch nodes found in fragments_graph. Skipping.\n")
        del payload, fragments_graph, degrees
        gc.collect()
        continue
        
    # Extract XYZ coordinates for branch nodes and build a KDTree
    branch_xyz = np.array([fragments_graph.node_xyz[n] for n in branch_nodes])
    kdtree = cKDTree(branch_xyz)
    
    # Extract XYZ coordinates for merge sites and query nearest branch point
    merge_xyz = np.array([site["xyz"] for site in gt_merge_sites])
    dists, _ = kdtree.query(merge_xyz)
    merge_dists.extend(dists.tolist())
    
    # Sample an equal number of non-branch nodes to form the null distribution
    num_samples = len(gt_merge_sites)
    if num_samples > len(non_branch_nodes):
        sampled_non_branch = non_branch_nodes
    else:
        sampled_non_branch = random.sample(non_branch_nodes, num_samples)
        
    if sampled_non_branch:
        random_xyz = np.array([fragments_graph.node_xyz[n] for n in sampled_non_branch])
        dists_rand, _ = kdtree.query(random_xyz)
        random_dists.extend(dists_rand.tolist())
        
    # Clear memory for multi-GB graphs
    del payload, fragments_graph, degrees, branch_nodes, non_branch_nodes, branch_xyz, kdtree
    gc.collect()
    print("  Done.\n")

merge_dists = np.array(merge_dists)
random_dists = np.array(random_dists)

print(f"Total merge sites analyzed: {len(merge_dists)}")
print(f"Total random nodes analyzed: {len(random_dists)}")

if len(merge_dists) > 0 and len(random_dists) > 0:
    mean_merge = np.mean(merge_dists)
    mean_rand = np.mean(random_dists)
    median_merge = np.median(merge_dists)
    median_rand = np.median(random_dists)
    
    print(f"\nMean distance (Merge): {mean_merge:.2f} um (Median: {median_merge:.2f} um)")
    print(f"Mean distance (Random): {mean_rand:.2f} um (Median: {median_rand:.2f} um)")

    # Mann-Whitney U test (testing if merge dists are significantly smaller than random dists)
    stat, p_val = mannwhitneyu(merge_dists, random_dists, alternative='less')
    print(f"\nMann-Whitney U Test (Merge < Random): U={stat}, p-value={p_val:.2e}")

    # Generate CDF plot
    plt.figure(figsize=(9, 6))
    
    merge_sorted = np.sort(merge_dists)
    random_sorted = np.sort(random_dists)
    
    p_merge = np.arange(1, len(merge_sorted) + 1) / len(merge_sorted)
    p_random = np.arange(1, len(random_sorted) + 1) / len(random_sorted)
    
    plt.plot(merge_sorted, p_merge, label='Merge Sites', color='#E41A1C', linewidth=2.5)
    plt.plot(random_sorted, p_random, label='Random Fragment Nodes', color='#377EB8', linewidth=2.5, linestyle='--')
    
    # Use symlog for better visual separation across orders of magnitude while supporting 0
    plt.xscale('symlog', linthresh=1.0)
    plt.xlabel('Distance to Nearest Fragments Branch Point (\u00b5m)')
    plt.ylabel('Cumulative Distribution Function (CDF)')
    plt.title('Proximity to Nearest Branch Point: Merge Sites vs Random Fragments Nodes')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.tight_layout()
    plt.show()
else:
    print("\nNot enough data to compute statistics or plot.")
