import subprocess
import sys
import os

def install_deps():
    try:
        subprocess.check_call(["apt-get", "update", "-y"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.check_call(["apt-get", "install", "-y", "git"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "psutil", "pandas", "scipy", "networkx", "tensorstore"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "git+https://github.com/AllenInstitute/agentic-neuron-proofreader.git"])

install_deps()

import pickle
import numpy as np
import warnings
from collections import defaultdict
from scipy.stats import mannwhitneyu
import gc
import agentic_neuron_proofreader

warnings.filterwarnings('ignore')

def process_dataset(path):
    with open(path, "rb") as f:
        payload = pickle.load(f)
    
    gt = payload["gt_graph"]
    fg = payload["fragments_graph"]
    
    node_label = np.asarray(payload["gt_node_canonical_label"])
    edge_error = np.asarray(payload["gt_edge_error"])
    merge_labels = set(int(x) for x in payload["gt_merge_labels"])
    
    # Identify which error classes apply to each segment via GT edges
    edge_error_by_seg = defaultdict(set)
    edges = list(gt.edges)
    for k, (u, v) in enumerate(edges):
        seg_u = int(node_label[u])
        seg_v = int(node_label[v])
        err = int(edge_error[k])
        if seg_u != 0:
            edge_error_by_seg[seg_u].add(err)
        if seg_v != 0:
            edge_error_by_seg[seg_v].add(err)
            
    # Control segments: have at least one correct edge mapping and are not involved in merges
    control_segments = {seg for seg, errs in edge_error_by_seg.items() if 0 in errs and 3 not in errs} - merge_labels
    
    # Build mapping from swc_id (segment id) to component ids
    swc_to_comp_ids = defaultdict(list)
    for comp_id, swc in fg.component_id_to_swc_id.items():
        swc_to_comp_ids[int(float(swc))].append(comp_id)
        
    # Group fragments_graph nodes by component id
    comp_id_to_nodes = defaultdict(list)
    for n in fg.nodes:
        comp_id_to_nodes[int(fg.node_component_id[n])].append(n)
            
    target_segs = merge_labels.union(control_segments)
    m_dens = []
    c_dens = []
    
    # Calculate length, branches, and density for each targeted segment subgraph
    for seg_id in target_segs:
        comp_ids = swc_to_comp_ids.get(seg_id, [])
        nodes = []
        for c_id in comp_ids:
            nodes.extend(comp_id_to_nodes[c_id])
            
        if len(nodes) < 2:
            continue
            
        sub = fg.subgraph(nodes)
        cable_length = 0.0
        for u, v in sub.edges:
            cable_length += np.linalg.norm(fg.node_xyz[u] - fg.node_xyz[v])
            
        if cable_length > 0:
            # Branch nodes have degree > 2
            branch_nodes = sum(1 for n, d in sub.degree() if d > 2)
            # Density per 100 microns of cable
            density = (branch_nodes / cable_length) * 100
            
            if seg_id in merge_labels:
                m_dens.append(density)
            elif seg_id in control_segments:
                c_dens.append(density)
                
    return m_dens, c_dens

def main():
    # Revised loader: use $RERUN_PKL directly (os.walk("..") path-search is
    # not intercepted by the runner's open/glob monkeypatches).
    pkl_paths = [os.environ["RERUN_PKL"]]
    print(f"Loading dataset from: {pkl_paths[0]}")

    merge_densities = []
    control_densities = []

    for path in pkl_paths:
        m, c = process_dataset(path)
        merge_densities.extend(m)
        control_densities.extend(c)
        gc.collect()  # Ensure cleanup between datasets to control memory footprint
        
    if merge_densities and control_densities:
        u_stat, p_val = mannwhitneyu(merge_densities, control_densities, alternative='two-sided')
        print(f"Analyzed {len(merge_densities)} merge segments and {len(control_densities)} control segments.")
        print(f"Average branch point density (merges): {np.mean(merge_densities):.4f} branches per 100 um")
        print(f"Average branch point density (controls): {np.mean(control_densities):.4f} branches per 100 um")
        print(f"Mann-Whitney U stat: {u_stat}")
        print(f"P-value: {p_val:.4e}")
    else:
        print(f"Not enough data computed. Merges found: {len(merge_densities)}, Controls found: {len(control_densities)}")

if __name__ == "__main__":
    main()
