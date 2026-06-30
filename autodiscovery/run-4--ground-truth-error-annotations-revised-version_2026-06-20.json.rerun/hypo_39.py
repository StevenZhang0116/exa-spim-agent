import subprocess
import sys
import os

try:
    import agentic_neuron_proofreader
    import psutil
    import pandas
    import tensorstore
except ImportError:
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "--quiet",
        "psutil", "pandas", "tensorstore", "matplotlib", "scipy", "tqdm",
        "https://github.com/AllenInstitute/agentic-neuron-proofreader/archive/main.zip"
    ])
    import agentic_neuron_proofreader

import pickle
import numpy as np
import glob
import networkx as nx
import heapq

def run_experiment():
    # Locate the dataset cache file
    files = glob.glob("../dataset_cache_*_add.pkl") + glob.glob("dataset_cache_*_add.pkl")
    if not files:
        print("Dataset not found.")
        return
    path = files[0]
    
    print(f"Loading {path}...")
    with open(path, "rb") as f:
        payload = pickle.load(f)
        
    gt = payload["gt_graph"]
    fg = payload["fragments_graph"]
    node_label = np.asarray(payload["gt_node_canonical_label"])
    edge_error = np.asarray(payload["gt_edge_error"])
    
    EDGE_CORRECT, EDGE_SPLIT, EDGE_OMIT, EDGE_MERGED = 0, 1, 2, 3
    
    total_edges = len(edge_error)
    correct_edges_count = np.count_nonzero(edge_error == EDGE_CORRECT)
    
    # Identify all GT split edges
    split_edges = [(u, v) for (u, v), c in zip(gt.edges, edge_error) if c == EDGE_SPLIT]
    
    print(f"Total GT edges: {total_edges}")
    print(f"Original Correct edges: {correct_edges_count} ({correct_edges_count/total_edges*100:.2f}%)")
    print(f"Original Split edges: {len(split_edges)} ({len(split_edges)/total_edges*100:.2f}%)")
    
    resolved_count = 0
    target_splits = 0
    
    print(f"\nStarting A* path-finding on split edges...")
    for idx, (u, v) in enumerate(split_edges):
        if idx > 0 and idx % 100 == 0:
            print(f"Processed {idx}/{len(split_edges)} splits...")
            
        lu = int(node_label[u])
        lv = int(node_label[v])
        
        # Only target split edges between two identified, distinct U-Net segment IDs
        if lu == 0 or lv == 0 or lu == lv:
            continue
            
        target_splits += 1
        target_neuron = gt.node_segment_id(u)
        
        # Map GT nodes to the closest Fragments Graph (fg) nodes
        _, fg_u = fg.kdtree.query(gt.node_xyz[u])
        _, fg_v = fg.kdtree.query(gt.node_xyz[v])
        
        if fg_u == fg_v:
            # They map to the same node in fg (trivial connection)
            dist_gt, gt_n = gt.kdtree.query(fg.node_xyz[fg_u])
            if dist_gt < 10.0 and gt.node_segment_id(gt_n) != target_neuron:
                pass # Induces a merge
            else:
                resolved_count += 1
            continue
            
        # A* Search directly on fragments_graph
        start_state = (fg_u, None)
        open_set = []
        # Using tuple: (f_score, g_score, curr_node_id, prev_node_id, state)
        heapq.heappush(open_set, (0.0, 0.0, fg_u, -1, start_state))
        
        g_score = {start_state: 0.0}
        came_from = {}
        
        goal_xyz = fg.node_xyz[fg_v]
        nodes_explored = 0
        path_found = None
        
        while open_set:
            f_val, _, curr, prev, current_state = heapq.heappop(open_set)
            
            if curr == fg_v:
                path_found = []
                while current_state in came_from:
                    path_found.append(current_state[0])
                    current_state = came_from[current_state]
                path_found.append(fg_u)
                path_found = path_found[::-1]
                break
                
            nodes_explored += 1
            if nodes_explored > 5000:  # Prevent infinite expansion in huge graphs
                break
                
            curr_xyz = fg.node_xyz[curr]
            
            for nxt in fg.neighbors(curr):
                nxt_xyz = fg.node_xyz[nxt]
                dist = np.linalg.norm(nxt_xyz - curr_xyz)
                
                # Compute turn angle
                angle = 0.0
                if prev != -1 and prev is not None:
                    prev_xyz = fg.node_xyz[prev]
                    v1 = curr_xyz - prev_xyz
                    v2 = nxt_xyz - curr_xyz
                    n1 = np.linalg.norm(v1)
                    n2 = np.linalg.norm(v2)
                    if n1 > 1e-5 and n2 > 1e-5:
                        cos_t = np.dot(v1, v2) / (n1 * n2)
                        cos_t = np.clip(cos_t, -1.0, 1.0)
                        angle = np.degrees(np.arccos(cos_t))
                
                # Cost function definition heavily penalizing sharp deviations > 45 deg
                turn_penalty = 100.0 * dist if angle > 45 else 0.0
                rad_diff = abs(fg.node_radius[curr] - fg.node_radius[nxt])
                rad_penalty = 10.0 * rad_diff
                
                edge_cost = dist + turn_penalty + rad_penalty
                
                new_g = g_score[current_state] + edge_cost
                nxt_state = (nxt, curr)
                
                if nxt_state not in g_score or new_g < g_score[nxt_state]:
                    g_score[nxt_state] = new_g
                    h = np.linalg.norm(nxt_xyz - goal_xyz)
                    came_from[nxt_state] = current_state
                    heapq.heappush(open_set, (new_g + h, new_g, nxt, curr, nxt_state))
        
        # Validate the recovered path
        if path_found is not None:
            is_valid = True
            for n in path_found:
                xyz = fg.node_xyz[n]
                dist_gt, gt_n = gt.kdtree.query(xyz)
                # If the path traverses close to a DIFFERENT traced GT neuron, it induces a merge
                if dist_gt < 10.0:
                    gn = gt.node_segment_id(gt_n)
                    if gn != target_neuron:
                        is_valid = False
                        break
            if is_valid:
                resolved_count += 1

    print(f"\nTarget split edges (distinct predicted segments): {target_splits}")
    print(f"Successfully resolved split edges: {resolved_count}")
    
    if target_splits > 0:
        success_rate = resolved_count / target_splits * 100
        print(f"Success rate on targeted splits: {success_rate:.2f}%")
        
    if len(split_edges) > 0:
        overall_success_rate = resolved_count / len(split_edges) * 100
        print(f"Percentage of ALL split edges resolved: {overall_success_rate:.2f}%")
        
    new_correct = correct_edges_count + resolved_count
    new_accuracy = new_correct / total_edges * 100
    old_accuracy = correct_edges_count / total_edges * 100
    
    print(f"\nOriginal Edge Accuracy: {old_accuracy:.2f}%")
    print(f"New Edge Accuracy (Simulated): {new_accuracy:.2f}%")
    print(f"Simulated gain in Edge Accuracy: {new_accuracy - old_accuracy:.2f}%")

if __name__ == "__main__":
    run_experiment()
