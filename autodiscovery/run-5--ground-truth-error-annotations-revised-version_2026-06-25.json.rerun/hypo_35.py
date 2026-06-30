import os
import sys
import subprocess

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", package])

dependencies = ["psutil", "pandas", "tensorstore", "scipy", "networkx", "matplotlib"]
for pkg in dependencies:
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
import glob
from collections import defaultdict

def main():
    # Find dataset cache
    cache_files = glob.glob("../dataset_cache_*_add.pkl") + glob.glob("dataset_cache_*_add.pkl") + glob.glob("cache/dataset_cache_*_add.pkl")
    if not cache_files:
        print("Could not find dataset cache file")
        return
    cache_path = cache_files[0]
    
    with open(cache_path, "rb") as f:
        payload = pickle.load(f)
    
    gt = payload["gt_graph"]
    node_label = np.asarray(payload["gt_node_canonical_label"])
    edge_error = np.asarray(payload["gt_edge_error"])
    
    # 1. Compute total GT cable length associated with each predicted segment ID
    seg_lengths = defaultdict(float)
    edges = list(gt.edges)
    
    for u, v in edges:
        u_lab = int(node_label[u])
        v_lab = int(node_label[v])
        
        # Get coordinates in um
        xyz_u = gt.node_xyz[u]
        xyz_v = gt.node_xyz[v]
            
        dist = np.linalg.norm(np.array(xyz_u) - np.array(xyz_v))
        
        # Distribute length to the segments
        if u_lab == v_lab and u_lab != 0:
            seg_lengths[u_lab] += dist
        else:
            if u_lab != 0:
                seg_lengths[u_lab] += dist / 2.0
            if v_lab != 0:
                seg_lengths[v_lab] += dist / 2.0
                
    # 2. Identify all split events
    split_events = []
    EDGE_SPLIT = 1
    for k, (u, v) in enumerate(edges):
        if edge_error[k] == EDGE_SPLIT:
            u_lab = int(node_label[u])
            v_lab = int(node_label[v])
            # A split implies different valid segments
            if u_lab != 0 and v_lab != 0 and u_lab != v_lab:
                split_events.append((u_lab, v_lab))
            
    # 3. Check for micro-fragments (< 100 um)
    micro_fragment_threshold = 100.0
    micro_fragment_splits = 0
    
    for u_lab, v_lab in split_events:
        len_u = seg_lengths[u_lab]
        len_v = seg_lengths[v_lab]
        if len_u < micro_fragment_threshold or len_v < micro_fragment_threshold:
            micro_fragment_splits += 1
            
    # 4. Calculate overall percentage
    total_splits = len(split_events)
    if total_splits > 0:
        percentage = (micro_fragment_splits / total_splits) * 100.0
    else:
        percentage = 0.0
        
    print(f"Total split events: {total_splits}")
    print(f"Split events involving a micro-fragment (< 100 um): {micro_fragment_splits}")
    print(f"Percentage: {percentage:.2f}%")

if __name__ == "__main__":
    main()
