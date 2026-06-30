import glob
import os
import sys
import pickle
import math
import numpy as np
import gc
from pathlib import Path

import agentic_neuron_proofreader

from scipy.stats import mannwhitneyu

def get_extremity(adj, start_node, prev_node, k, node_xyz):
    curr = start_node
    prev = prev_node
    path_len = 0.0
    for _ in range(k):
        neighbors = [n for n in adj[curr] if n != prev]
        if not neighbors:
            break
        # In case of branching, just follow the first branch.
        nxt = neighbors[0]
        dx = node_xyz[nxt][0] - node_xyz[curr][0]
        dy = node_xyz[nxt][1] - node_xyz[curr][1]
        dz = node_xyz[nxt][2] - node_xyz[curr][2]
        path_len += math.sqrt(dx*dx + dy*dy + dz*dz)
        prev = curr
        curr = nxt
    return curr, path_len

def get_tortuosity(gt, edges, edge_errors, k=5):
    try:
        node_xyz = gt.node_xyz
        _ = node_xyz[next(iter(gt.nodes))]
    except (AttributeError, TypeError, IndexError):
        node_xyz = {n: gt.nodes[n].get('node_xyz', gt.nodes[n].get('xyz')) for n in gt.nodes}
        
    adj = {n: list(gt.neighbors(n)) for n in gt.nodes}
    
    correct_tortuosities = []
    split_tortuosities = []
    
    for idx, (u, v) in enumerate(edges):
        err = edge_errors[idx]
        if err != 0 and err != 1:
            continue
            
        u_curr, u_len = get_extremity(adj, u, v, k, node_xyz)
        v_curr, v_len = get_extremity(adj, v, u, k, node_xyz)
        
        dx_edge = node_xyz[v][0] - node_xyz[u][0]
        dy_edge = node_xyz[v][1] - node_xyz[u][1]
        dz_edge = node_xyz[v][2] - node_xyz[u][2]
        edge_len = math.sqrt(dx_edge*dx_edge + dy_edge*dy_edge + dz_edge*dz_edge)
        
        path_length = u_len + v_len + edge_len
        
        dx_end = node_xyz[v_curr][0] - node_xyz[u_curr][0]
        dy_end = node_xyz[v_curr][1] - node_xyz[u_curr][1]
        dz_end = node_xyz[v_curr][2] - node_xyz[u_curr][2]
        euclidean_dist = math.sqrt(dx_end*dx_end + dy_end*dy_end + dz_end*dz_end)
        
        if euclidean_dist > 1e-6:
            tortuosity = path_length / euclidean_dist
        else:
            tortuosity = 1.0
            
        if err == 0:
            correct_tortuosities.append(tortuosity)
        elif err == 1:
            split_tortuosities.append(tortuosity)
            
    return correct_tortuosities, split_tortuosities

def main():
    # Load the provided dataset directly from $RERUN_PKL
    print("Loading dataset from:", os.environ["RERUN_PKL"])
    files = [Path(os.environ["RERUN_PKL"])]

    if not files:
        print("No dataset files found matching *_add.pkl")
        sys.exit(1)
        
    all_correct = []
    all_split = []
    
    for fpath in files:
        print(f"Loading {fpath}...")
        with open(fpath, "rb") as f:
            payload = pickle.load(f)
        
        gt = payload["gt_graph"]
        edge_error = np.asarray(payload["gt_edge_error"])
        edges = list(gt.edges)
        
        c_tort, s_tort = get_tortuosity(gt, edges, edge_error, k=5)
        all_correct.extend(c_tort)
        all_split.extend(s_tort)
        
        # Free memory to prevent OOM errors with multiple large caches
        del payload
        del gt
        del edge_error
        del edges
        gc.collect()
        
    print(f"\nProcessed {len(all_correct)} correct edges and {len(all_split)} split edges.")
    
    if len(all_correct) == 0 or len(all_split) == 0:
        print("Not enough edges to compare.")
    else:
        median_correct = np.median(all_correct)
        median_split = np.median(all_split)
        
        print(f"Median Tortuosity for CORRECT edges: {median_correct:.6f}")
        print(f"Median Tortuosity for SPLIT edges:   {median_split:.6f}")
        print(f"Mean Tortuosity for CORRECT edges:   {np.mean(all_correct):.6f}")
        print(f"Mean Tortuosity for SPLIT edges:     {np.mean(all_split):.6f}")
        
        # Mann-Whitney U test (Alternative: split tortuosities are greater than correct ones)
        stat, p_val = mannwhitneyu(all_split, all_correct, alternative='greater')
        print(f"\nMann-Whitney U statistic: {stat}")
        print(f"p-value: {p_val:.4e}")
        
        if p_val < 0.05:
            print("Result: Statistically significant. Split edges occur on more tortuous segments.")
        else:
            print("Result: Not statistically significant.")

if __name__ == "__main__":
    main()
