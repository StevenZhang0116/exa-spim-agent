import sys
import subprocess
import glob

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", package])

# Install all necessary dependencies for unpickling and data analysis
install("psutil")
install("pandas")
install("scikit-learn")
install("matplotlib")
install("tensorstore")

try:
    import agentic_neuron_proofreader
except ImportError:
    install("https://github.com/AllenInstitute/agentic-neuron-proofreader/archive/refs/heads/main.zip")
    import agentic_neuron_proofreader

import pickle
import numpy as np
from scipy.spatial import KDTree
from scipy.stats import wilcoxon, ttest_rel
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

# Find the dataset robustly
possible_files = glob.glob("*_add.pkl") + glob.glob("../*_add.pkl") + glob.glob("*/*_add.pkl") + glob.glob("../*/*_add.pkl")
if not possible_files:
    print("Error: Dataset not found.")
    sys.exit(1)

dataset_path = possible_files[0]
print(f"Loading dataset: {dataset_path}")

with open(dataset_path, "rb") as f:
    payload = pickle.load(f)

fg = payload["fragments_graph"]
merge_sites = payload["gt_merge_sites"]

fg_xyz = fg.node_xyz
fg_comp = fg.node_component_id

print("Precomputing node degrees...")
# Calculate degrees directly from fg.degree() to efficiently identify branch nodes
fg_degrees = np.zeros(len(fg_xyz), dtype=int)
for n, d in fg.degree():
    if isinstance(n, int) and n < len(fg_degrees):
        fg_degrees[n] = d

merge_densities = []
control_densities = []

np.random.seed(42)

for ms in merge_sites:
    seg_id = ms["segment_id"]
    xyz = ms["xyz"]
    
    # Valid nodes for the segment_id
    valid_nodes = np.where(fg_comp == seg_id)[0]
    
    # Fallback to KD-tree mapping if valid_nodes for the segment is empty (mismatched ID label)
    if len(valid_nodes) == 0:
        d, mapped_node = fg.kdtree.query(xyz)
        seg_id_to_use = fg_comp[mapped_node]
        valid_nodes = np.where(fg_comp == seg_id_to_use)[0]
    else:
        valid_xyz = fg_xyz[valid_nodes]
        dists = np.linalg.norm(valid_xyz - xyz, axis=1)
        mapped_node = valid_nodes[np.argmin(dists)]
        seg_id_to_use = seg_id
        
    if len(valid_nodes) == 0:
        continue
        
    # 1. Local branch density at merge site (15-micron radius)
    neighbors = fg.kdtree.query_ball_point(fg_xyz[mapped_node], 15.0)
    # Count branch nodes (degree > 2) in the same segment
    m_density = sum(1 for n in neighbors if fg_comp[n] == seg_id_to_use and fg_degrees[n] > 2)
    
    # 2. Local branch density at control site (randomly sample non-merge nodes on same segment)
    mapped_xyz = fg_xyz[mapped_node]
    dists_to_mapped = np.linalg.norm(fg_xyz[valid_nodes] - mapped_xyz, axis=1)
    
    # Find a distant node on the same segment to act as a proper control
    far_nodes = valid_nodes[dists_to_mapped > 30.0]
    if len(far_nodes) > 0:
        control_node = np.random.choice(far_nodes)
    else:
        control_node = np.random.choice(valid_nodes)
        
    c_neighbors = fg.kdtree.query_ball_point(fg_xyz[control_node], 15.0)
    c_density = sum(1 for n in c_neighbors if fg_comp[n] == seg_id_to_use and fg_degrees[n] > 2)
    
    merge_densities.append(m_density)
    control_densities.append(c_density)

merge_densities = np.array(merge_densities)
control_densities = np.array(control_densities)

print(f"\nAnalyzed {len(merge_densities)} merge sites.")
print(f"Mean branch density at merge sites:   {np.mean(merge_densities):.2f} branches / 15um radius")
print(f"Mean branch density at control sites: {np.mean(control_densities):.2f} branches / 15um radius")

if len(merge_densities) > 1:
    stat, p_val = ttest_rel(merge_densities, control_densities)
    print(f"\nPaired t-test: t = {stat:.4f}, p = {p_val:.4e}")
    
    diffs = merge_densities - control_densities
    if np.any(diffs != 0):
        w_stat, w_p_val = wilcoxon(merge_densities, control_densities)
        print(f"Wilcoxon signed-rank test: W = {w_stat:.4f}, p = {w_p_val:.4e}")
    else:
        print("Wilcoxon signed-rank test cannot be performed (all differences are zero).")
    
    labels = np.concatenate([np.ones(len(merge_densities)), np.zeros(len(control_densities))])
    scores = np.concatenate([merge_densities, control_densities])
    try:
        auc = roc_auc_score(labels, scores)
        print(f"Predictive power for identifying merge coordinates (ROC-AUC): {auc:.4f}")
    except ValueError:
        print("ROC-AUC cannot be calculated.")
else:
    print("Not enough merge sites to perform statistical tests.")

# Visualization
plt.figure(figsize=(8, 6))
box = plt.boxplot([merge_densities, control_densities], labels=['Merge Sites', 'Control Sites'], patch_artist=True)

colors = ['lightcoral', 'lightblue']
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)

plt.title('Local Branch Density in UNet Fragments Graph')
plt.ylabel('Number of Branches (15um radius)')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
