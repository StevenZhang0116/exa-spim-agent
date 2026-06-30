import sys
import subprocess

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", package])

for pkg in ["psutil", "scipy", "pandas", "tensorstore", "scikit-learn", "matplotlib", "networkx"]:
    install(pkg)

try:
    import agentic_neuron_proofreader
except ImportError:
    install("https://github.com/AllenInstitute/agentic-neuron-proofreader/archive/refs/heads/main.zip")
    import agentic_neuron_proofreader

import glob
import pickle
import numpy as np
import networkx as nx
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

# 1. Locate and load the dataset
files = glob.glob("../dataset_cache_*_add.pkl") + glob.glob("dataset_cache_*_add.pkl")
if not files:
    raise FileNotFoundError("Could not find the dataset _add.pkl file.")

with open(files[0], "rb") as f:
    payload = pickle.load(f)

gt = payload["gt_graph"]
node_label = np.asarray(payload["gt_node_canonical_label"])
merge_sites = payload["gt_merge_sites"]

# Function to compute the tangent vector at a specific node
def get_tangent(graph, node, window=5):
    # Extract local neighborhood within 'window' edges along the skeleton
    neighborhood = nx.single_source_shortest_path_length(graph, node, cutoff=window)
    nodes = list(neighborhood.keys())
    
    coords = np.array([graph.node_xyz[n] for n in nodes])
    coords = np.unique(coords, axis=0)
    
    if len(coords) < 2:
        return np.array([1.0, 0.0, 0.0])
    
    if len(coords) == 2:
        vec = coords[1] - coords[0]
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else np.array([1.0, 0.0, 0.0])
    
    # Calculate tangent via PCA over the spatial neighborhood
    pca = PCA(n_components=1)
    pca.fit(coords)
    return pca.components_[0]

# Pre-group nodes by their predicted segment_id for rapid lookups
seg_id_to_nodes = {}
for n in gt.nodes:
    lab = int(node_label[n])
    if lab == 0:  # Skip unlabelled (omit) space
        continue
    if lab not in seg_id_to_nodes:
        seg_id_to_nodes[lab] = []
    seg_id_to_nodes[lab].append(n)

angles = []

# 2. Iterate through merge sites and evaluate crossing geometries
for site in merge_sites:
    seg_id = int(site["segment_id"])
    xyz = np.array(site["xyz"])
    
    if seg_id not in seg_id_to_nodes:
        continue
        
    nodes_with_seg = seg_id_to_nodes[seg_id]
    
    # Group involved GT nodes by their ground-truth neuron identity
    neuron_to_nodes = {}
    for n in nodes_with_seg:
        neuron_name = gt.node_segment_id(n)
        if neuron_name not in neuron_to_nodes:
            neuron_to_nodes[neuron_name] = []
        neuron_to_nodes[neuron_name].append(n)
        
    # A merge must involve at least 2 distinct ground-truth neurons
    if len(neuron_to_nodes) < 2:
        continue
        
    # Find the single closest node to the merge site on each GT neuron
    closest_nodes = []
    for neuron_name, n_list in neuron_to_nodes.items():
        coords = np.array([gt.node_xyz[n] for n in n_list])
        dists = np.linalg.norm(coords - xyz, axis=1)
        best_idx = np.argmin(dists)
        closest_nodes.append((dists[best_idx], n_list[best_idx], neuron_name))
        
    # Identify the top 2 GT neurons geometrically closest to the merge site
    closest_nodes.sort(key=lambda x: x[0])
    node1 = closest_nodes[0][1]
    node2 = closest_nodes[1][1]
    
    # 3. Compute local tangents for both intersecting neurons
    v1 = get_tangent(gt, node1, window=5)
    v2 = get_tangent(gt, node2, window=5)
    
    # 4. Calculate acute 3D intersection angle
    cos_theta = np.clip(np.abs(np.dot(v1, v2)), 0.0, 1.0)
    angle = np.degrees(np.arccos(cos_theta))
    angles.append(angle)

angles = np.array(angles)
print(f"Computed 3D crossing angles for {len(angles)} valid merge sites.\n")

if len(angles) > 0:
    # 5. One-sample Wilcoxon signed-rank test against 45 degrees
    diffs = angles - 45.0
    if np.all(diffs == 0):
        stat, p = 0.0, 1.0
    else:
        stat, p = wilcoxon(diffs, alternative='greater')
    
    # Present test results and deliverables
    print("--- Wilcoxon signed-rank test (H0: median <= 45° vs H1: median > 45°) ---")
    print(f"Statistic = {stat:.2f}")
    print(f"p-value   = {p:.4e}")
    print(f"Mean angle   = {np.mean(angles):.2f}°")
    print(f"Median angle = {np.median(angles):.2f}°\n")
    
    plt.figure(figsize=(9, 5))
    plt.hist(angles, bins=20, range=(0, 90), color='mediumpurple', edgecolor='black', alpha=0.8)
    plt.axvline(45, color='crimson', linestyle='dashed', linewidth=2.5, label='45° Benchmark')
    plt.axvline(np.median(angles), color='darkgreen', linestyle='dotted', linewidth=2.5, label=f'Median: {np.median(angles):.1f}°')
    plt.title('Distribution of 3D Crossing Angles at Ground-Truth Merge Sites')
    plt.xlabel('Acute Intersection Angle (Degrees)')
    plt.ylabel('Frequency (Merge Events)')
    plt.legend()
    plt.grid(axis='y', alpha=0.4)
    plt.tight_layout()
    plt.show()
else:
    print("No geometrically valid merge sites were found to compute crossing angles.")
