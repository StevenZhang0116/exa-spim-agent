import subprocess
import sys
import os

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", package])

def ensure_package(pkg_name, import_name=None):
    if import_name is None:
        import_name = pkg_name
    try:
        __import__(import_name)
    except ImportError:
        install(pkg_name)

# Ensure all dependencies are installed prior to loading the custom module
ensure_package("psutil")
ensure_package("pandas")
ensure_package("tensorstore")
ensure_package("scikit-learn", "sklearn")
ensure_package("tqdm")
ensure_package("scipy")
ensure_package("networkx")
ensure_package("numpy")

try:
    import agentic_neuron_proofreader
except ImportError:
    # Install from zip archive since git might not be available in the environment
    install("https://github.com/AllenInstitute/agentic-neuron-proofreader/archive/refs/heads/main.zip")

import pickle
import numpy as np
import glob
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from sklearn.metrics import roc_curve, auc

def main():
    # 1. Load the dataset (from the current directory)
    paths = glob.glob("*_add.pkl")
    if not paths:
        raise FileNotFoundError("Could not find any *_add.pkl file in the current directory.")
    
    path = paths[0]
    print(f"Using dataset: {path}")

    with open(path, "rb") as f:
        payload = pickle.load(f)

    gt = payload["gt_graph"]
    edge_error = np.asarray(payload["gt_edge_error"])
    
    nodes = list(gt.nodes)
    # node_xyz is a parallel numpy array on the SkeletonGraph
    node_coords = np.array([gt.node_xyz[n] for n in nodes])
    node_gt_ids = np.array([gt.node_segment_id(n) for n in nodes])
    
    unique_gt_ids = np.unique(node_gt_ids)
    
    # Build a KDTree for each GT neuron to quickly find false candidates
    trees = {}
    node_xyz_dict = {}
    for gt_id in unique_gt_ids:
        mask = node_gt_ids == gt_id
        if np.any(mask):
            xyz_subset = node_coords[mask]
            trees[gt_id] = KDTree(xyz_subset)
            node_xyz_dict[gt_id] = xyz_subset

    EDGE_SPLIT = 1
    edges = list(gt.edges)

    true_angles = []
    false_angles = []

    def get_angle(v1, v2):
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 == 0 or n2 == 0:
            return 0.0
        cos_theta = np.dot(v1, v2) / (n1 * n2)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        return np.degrees(np.arccos(cos_theta))

    # 2 & 3. Iterate through edges to process split errors
    for k, (u, v) in enumerate(edges):
        if edge_error[k] == EDGE_SPLIT:
            # Process both endpoints of the split edge as individual split nodes
            for current_node, other_node in [(u, v), (v, u)]:
                neighbors = [n for n in gt.neighbors(current_node) if n != other_node]
                if not neighbors:
                    continue
                
                # Get the true incoming edge (if branch point, pick the first neighbor)
                w = neighbors[0]
                
                c_xyz = gt.node_xyz[current_node]
                o_xyz = gt.node_xyz[other_node]
                w_xyz = gt.node_xyz[w]
                
                # Vectors pointing away from the current node 
                vec_in = w_xyz - c_xyz
                vec_true_out = o_xyz - c_xyz
                
                angle_true = get_angle(vec_in, vec_true_out)
                
                # Find the closest false candidate node from a DIFFERENT GT neuron
                c_seg = gt.node_segment_id(current_node)
                min_dist = float('inf')
                found_z = None
                
                for gt_id, tree in trees.items():
                    if gt_id == c_seg:
                        continue
                    dist, idx = tree.query(c_xyz, k=1)
                    if dist < min_dist:
                        min_dist = dist
                        found_z = node_xyz_dict[gt_id][idx]
                        
                if found_z is not None:
                    vec_false_out = found_z - c_xyz
                    angle_false = get_angle(vec_in, vec_false_out)
                    
                    true_angles.append(angle_true)
                    false_angles.append(angle_false)
                    
    if not true_angles:
        print("No valid split configurations found to compare.")
        return

    true_angles = np.array(true_angles)
    false_angles = np.array(false_angles)

    # 4. Statistical comparison and Deliverables
    print(f"Analyzed {len(true_angles)} split node configurations.")
    print(f"Mean true deviation angle: {np.mean(true_angles):.2f} degrees")
    print(f"Mean false candidate angle: {np.mean(false_angles):.2f} degrees\n")

    stat, p_value = ks_2samp(true_angles, false_angles)
    print(f"KS test statistic: {stat:.4f}, p-value: {p_value:.4e}")

    # Generate ROC curve showing discriminative power of directional angular thresholds
    # A score closer to 180 degrees indicates a better, straighter continuation (so we use angles as scores)
    y_true = np.concatenate([np.ones_like(true_angles), np.zeros_like(false_angles)])
    y_scores = np.concatenate([true_angles, false_angles])

    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    print(f"ROC AUC: {roc_auc:.4f}")

    # Plot ROC Curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC for Angular Alignment at Splits')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
