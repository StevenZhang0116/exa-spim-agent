import os
import sys
import pickle
import numpy as np
import re
from collections import defaultdict

import agentic_neuron_proofreader

from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

# Load the provided dataset directly
print("Loading dataset from:", os.environ["RERUN_PKL"])
pkl_files = [os.environ["RERUN_PKL"]]

if not pkl_files:
    print("No dataset files found.")
    sys.exit(1)

all_gaps = []

# Extract gap distances for all true splits
for pkl_file in pkl_files:
    print(f"Processing {pkl_file}...")
    try:
        with open(pkl_file, "rb") as f:
            payload = pickle.load(f)
    except Exception as e:
        print(f"Error loading {pkl_file}: {e}")
        continue
        
    gt = payload["gt_graph"]
    fg = payload["fragments_graph"]
    node_label = np.asarray(payload["gt_node_canonical_label"])
    
    comp_to_swc = fg.component_id_to_swc_id
    node_comp = fg.node_component_id
    node_xyz = fg.node_xyz
    
    valid_segment_ids = set(int(x) for x in node_label if x != 0)
    
    swc_to_endpoints = defaultdict(list)
    all_endpoints = []
    
    # Get node degrees from networkx graph
    try:
        fg_degrees = dict(fg.degree)
    except TypeError:
        fg_degrees = dict(fg.degree())
        
    # Precompute fragment endpoints (nodes with degree <= 1)
    for n, d in fg_degrees.items():
        if d <= 1:
            xyz = node_xyz[n]
            all_endpoints.append(xyz)
            
            comp_id = node_comp[n]
            swc_id = comp_to_swc.get(comp_id) if isinstance(comp_to_swc, dict) else comp_to_swc[comp_id]
            if swc_id is not None:
                # extract all numeric values to match against U-Net segment IDs safely
                nums = re.findall(r'\d+', str(swc_id))
                for num in nums:
                    clean_id = int(num)
                    if clean_id in valid_segment_ids:
                        swc_to_endpoints[clean_id].append(xyz)
                        
    # Convert endpoints to KD-Trees for fast spatial mapping
    swc_to_kdtree = {}
    for k, v in swc_to_endpoints.items():
        if len(v) > 0:
            swc_to_kdtree[k] = cKDTree(np.array(v))
            
    if len(all_endpoints) > 0:
        global_kdtree = cKDTree(np.array(all_endpoints))
    else:
        print("No endpoints found in fragments_graph.")
        continue

    split_pairs = 0
    matched_by_id = 0
    
    # Identify true splits in ground truth edges
    for u, v in gt.edges:
        lab_u = int(node_label[u])
        lab_v = int(node_label[v])
        
        # A 'true split' is an edge where adjacent nodes belong to different valid segment IDs
        if lab_u != lab_v and lab_u != 0 and lab_v != 0:
            split_pairs += 1
            xyz_u = gt.node_xyz[u]
            xyz_v = gt.node_xyz[v]
            
            # Map GT node `u` to its closest fragment endpoint (filtered by ID if possible, else global spatial closest)
            if lab_u in swc_to_kdtree:
                dist_u, idx_u = swc_to_kdtree[lab_u].query(xyz_u)
                ep_u = swc_to_endpoints[lab_u][idx_u]
                matched_by_id += 1
            else:
                dist_u, idx_u = global_kdtree.query(xyz_u)
                ep_u = all_endpoints[idx_u]
                
            # Map GT node `v` to its closest fragment endpoint
            if lab_v in swc_to_kdtree:
                dist_v, idx_v = swc_to_kdtree[lab_v].query(xyz_v)
                ep_v = swc_to_endpoints[lab_v][idx_v]
                matched_by_id += 1
            else:
                dist_v, idx_v = global_kdtree.query(xyz_v)
                ep_v = all_endpoints[idx_v]
                
            # Compute pairwise Euclidean distances between the corresponding fragment endpoints
            gap_size = np.linalg.norm(ep_u - ep_v)
            all_gaps.append(gap_size)
            
    print(f"Found {split_pairs} true split edges. Matched {matched_by_id} out of {split_pairs * 2} fragment endpoints explicitly by segment ID.")
    
    # Memory cleanup for next dataset in iteration
    del payload, gt, fg, node_label, comp_to_swc, node_comp, node_xyz, swc_to_endpoints, all_endpoints, swc_to_kdtree, global_kdtree

print(f"\nTotal valid split gaps found: {len(all_gaps)}")

if len(all_gaps) < 2:
    print("Not enough split gaps found across datasets to fit a 2-component GMM.")
    sys.exit(0)

# Fit a Gaussian Mixture Model (GMM) to test for bimodality
X = np.array(all_gaps).reshape(-1, 1)
gmm = GaussianMixture(n_components=2, random_state=42)
gmm.fit(X)

print("\n--- GMM Component Parameters ---")
for i in range(gmm.n_components):
    weight = gmm.weights_[i]
    mean = gmm.means_[i, 0]
    covar = gmm.covariances_[i, 0, 0]
    print(f"Component {i+1}:")
    print(f"  Weight: {weight:.4f}")
    print(f"  Mean:   {mean:.4f} \u00b5m")
    print(f"  StdDev: {np.sqrt(covar):.4f} \u00b5m")

# Plot the density distribution
plt.figure(figsize=(10, 6))
plt.hist(all_gaps, bins=50, density=True, alpha=0.6, color='skyblue', edgecolor='black', label='Gap Sizes Histogram')

# To safely evaluate the pdf across the range of data
x_min, x_max = min(all_gaps), max(all_gaps)
if x_min == x_max:
    x_min, x_max = x_min - 1, x_max + 1
    
x_range = np.linspace(x_min, x_max, 1000).reshape(-1, 1)
logprob = gmm.score_samples(x_range)
pdf = np.exp(logprob)
plt.plot(x_range, pdf, '-k', linewidth=2, label='GMM Total PDF')

for i in range(gmm.n_components):
    weight = gmm.weights_[i]
    mean = gmm.means_[i, 0]
    covar = gmm.covariances_[i, 0, 0]
    if covar > 0:
        comp_pdf = weight * (1.0 / np.sqrt(2 * np.pi * covar)) * np.exp(-0.5 * ((x_range.flatten() - mean) ** 2) / covar)
        plt.plot(x_range, comp_pdf, '--', linewidth=2, label=f'Component {i+1} (mean: {mean:.2f})')

plt.title('Density Distribution of Split Gap Distances')
plt.xlabel('Euclidean Distance (\u00b5m)')
plt.ylabel('Density')
plt.legend()
plt.tight_layout()
plt.show()
