import subprocess
import sys
import os
import glob

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", package])

# Install standard dependencies if missing
for pkg in ["statsmodels", "matplotlib", "scipy", "networkx", "tqdm", "numpy", "pandas", "psutil", "tensorstore"]:
    try:
        __import__(pkg)
    except ImportError:
        install(pkg)

# Install agentic_neuron_proofreader using archive URL since git is not available
try:
    import agentic_neuron_proofreader
except ImportError:
    install("https://github.com/AllenInstitute/agentic-neuron-proofreader/archive/refs/heads/main.zip")
    import agentic_neuron_proofreader

import pickle
import numpy as np
import re
from collections import defaultdict
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.spatial import cKDTree
import gc

def get_xyz(fg, n):
    if hasattr(fg, 'node_xyz'):
        return np.array(fg.node_xyz[n])
    return np.array(fg.nodes[n]['node_xyz'])

def get_seg_id(fg, n):
    try:
        comp_id = fg.node_component_id[n] if hasattr(fg, 'node_component_id') else fg.nodes[n]['node_component_id']
        swc_arr = fg.component_id_to_swc_id if hasattr(fg, 'component_id_to_swc_id') else fg.graph.get('component_id_to_swc_id')
        swc_str = str(swc_arr[comp_id])
    except Exception:
        swc_str = str(fg.nodes[n].get('component_id_to_swc_id', ''))
    
    # Parse ID as per instructions: int(float(swc_string))
    swc_str = swc_str.replace('.swc', '')
    try:
        return int(float(swc_str))
    except ValueError:
        pass
        
    m = re.search(r'(\d+)', swc_str)
    if m:
        return int(m.group(1))
    return -1

def get_terminal_vector(fg, term_node, dist_th=5.0):
    curr = term_node
    prev = None
    path_len = 0.0
    while True:
        neighbors = list(fg.neighbors(curr))
        if prev is not None and prev in neighbors:
            neighbors.remove(prev)
        if len(neighbors) != 1:
            break
        next_node = neighbors[0]
        d = np.linalg.norm(get_xyz(fg, curr) - get_xyz(fg, next_node))
        path_len += d
        prev = curr
        curr = next_node
        if path_len >= dist_th:
            break
    vec = get_xyz(fg, term_node) - get_xyz(fg, curr)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    else:
        vec = np.array([1.0, 0.0, 0.0])
    return vec

# Find dataset files
files = glob.glob("../dataset_cache_*_add.pkl")
if not files:
    files = glob.glob("../*_add.pkl")
if not files:
    files = glob.glob("dataset_cache_*_add.pkl")

X_dist = []
X_angle = []
Y = []

for path in files:
    print(f"Processing {path}...")
    with open(path, "rb") as f:
        payload = pickle.load(f)
    
    gt = payload["gt_graph"]
    fg = payload["fragments_graph"]
    node_label = np.asarray(payload["gt_node_canonical_label"])
    edge_error = np.asarray(payload["gt_edge_error"])
    
    nodes_list = list(gt.nodes)
    node_to_idx = {n: i for i, n in enumerate(nodes_list)}
    
    seg_to_gt = {}
    for i, n in enumerate(nodes_list):
        try:
            lab = int(node_label[n])
        except IndexError:
            lab = int(node_label[i])
            
        if lab != 0:
            seg_to_gt[lab] = gt.node_segment_id(n)
            
    split_pairs = set()
    edges_list = list(gt.edges)
    for k, (u, v) in enumerate(edges_list):
        if edge_error[k] == 1: # EDGE_SPLIT
            try:
                lab_u = int(node_label[u])
                lab_v = int(node_label[v])
            except IndexError:
                idx_u = node_to_idx[u]
                idx_v = node_to_idx[v]
                lab_u = int(node_label[idx_u])
                lab_v = int(node_label[idx_v])
                
            if lab_u != 0 and lab_v != 0 and lab_u != lab_v:
                split_pairs.add(tuple(sorted([lab_u, lab_v])))
                
    degrees = dict(fg.degree())
    terminals = [n for n, d in degrees.items() if d == 1]
    
    seg_terminals = defaultdict(list)
    term_vectors = {}
    for t in terminals:
        seg_id = get_seg_id(fg, t)
        if seg_id != -1:
            seg_terminals[seg_id].append(t)
        term_vectors[t] = get_terminal_vector(fg, t)
        
    for (lab_u, lab_v) in split_pairs:
        terms_u = seg_terminals.get(lab_u, [])
        terms_v = seg_terminals.get(lab_v, [])
        best_dist = float('inf')
        best_angle = None
        for tu in terms_u:
            for tv in terms_v:
                dist = np.linalg.norm(get_xyz(fg, tu) - get_xyz(fg, tv))
                if dist < best_dist:
                    best_dist = dist
                    vu = term_vectors[tu]
                    vv = term_vectors[tv]
                    dot = np.clip(np.dot(vu, -vv), -1.0, 1.0)
                    best_angle = np.degrees(np.arccos(dot))
        if best_dist <= 15.0 and best_angle is not None:
            X_dist.append(best_dist)
            X_angle.append(best_angle)
            Y.append(1)
            
    term_coords = np.array([get_xyz(fg, t) for t in terminals])
    if len(term_coords) > 0:
        tree = cKDTree(term_coords)
        pairs = tree.query_pairs(15.0)
        
        for i, j in pairs:
            t_a = terminals[i]
            t_b = terminals[j]
            seg_a = get_seg_id(fg, t_a)
            seg_b = get_seg_id(fg, t_b)
            if seg_a == -1 or seg_b == -1 or seg_a == seg_b:
                continue
                
            pair_tuple = tuple(sorted([seg_a, seg_b]))
            if pair_tuple in split_pairs:
                continue
                
            gt_a = seg_to_gt.get(seg_a)
            gt_b = seg_to_gt.get(seg_b)
            
            # More robust False Merge sampling: 
            # Include pairs where they don't map to the SAME gt neuron,
            # and at least one maps to a GT neuron (to evaluate biologically relevant gaps).
            if gt_a != gt_b and (gt_a is not None or gt_b is not None):
                dist = np.linalg.norm(get_xyz(fg, t_a) - get_xyz(fg, t_b))
                va = term_vectors[t_a]
                vb = term_vectors[t_b]
                dot = np.clip(np.dot(va, -vb), -1.0, 1.0)
                angle = np.degrees(np.arccos(dot))
                X_dist.append(dist)
                X_angle.append(angle)
                Y.append(0)
                
    del payload
    del gt
    del fg
    gc.collect()

X_dist = np.array(X_dist)
X_angle = np.array(X_angle)
Y = np.array(Y)

print(f"Collected {np.sum(Y==1)} positives and {np.sum(Y==0)} negatives.")

if len(np.unique(Y)) < 2:
    print("Not enough classes found to run Logistic Regression.")
else:
    # Normalization prevents exponential overflow in sm.Logit
    X_dist_norm = X_dist / 15.0
    X_angle_norm = X_angle / 180.0
    X_inter_norm = X_dist_norm * X_angle_norm
    
    X = np.column_stack((X_dist_norm, X_angle_norm, X_inter_norm))
    X = sm.add_constant(X)
    
    try:
        model = sm.Logit(Y, X)
        result = model.fit(method='bfgs', disp=0)
        print("\n--- Logistic Regression Results ---")
        print(result.summary())
        print("\nP-values of regression coefficients:")
        print(result.pvalues)
        
        beta = result.params
        
        plt.figure(figsize=(8, 6))
        plt.scatter(X_dist[Y==0], X_angle[Y==0], color='red', alpha=0.5, label='False Merge (Negative)', s=10)
        plt.scatter(X_dist[Y==1], X_angle[Y==1], color='blue', alpha=0.5, label='True Split (Positive)', s=10)
        
        dist_range = np.linspace(0, 15, 100)
        d_norm = dist_range / 15.0
        
        denom = beta[2] + beta[3] * d_norm
        denom[denom == 0] = 1e-9
        a_norm_boundary = -(beta[0] + beta[1] * d_norm) / denom
        angle_boundary = a_norm_boundary * 180.0
        
        valid = (angle_boundary >= -10) & (angle_boundary <= 190)
        
        plt.plot(dist_range[valid], angle_boundary[valid], color='black', linewidth=2, label='Decision Boundary (P=0.5)')
        
        plt.xlabel('Euclidean Gap Distance (\u00b5m)')
        plt.ylabel('Turning Angle (degrees)')
        plt.title('Non-Linear Trade-off: Gap Distance vs Turning Angle')
        plt.xlim(0, 15)
        plt.ylim(0, 180)
        plt.legend()
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print("Error during modeling or plotting:", e)
