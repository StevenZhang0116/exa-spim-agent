import os
import sys
import pickle
import gc

import agentic_neuron_proofreader

import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Load the provided dataset directly from $RERUN_PKL
print("Loading dataset from:", os.environ["RERUN_PKL"])
dataset_paths = [os.environ["RERUN_PKL"]]

if not dataset_paths:
    print("No dataset files found.")
    sys.exit(1)

def get_radius(gt, n):
    # Node attributes are parallel NumPy arrays indexed by integer node id
    if hasattr(gt, "node_radius") and gt.node_radius is not None:
        try:
            return float(gt.node_radius[n])
        except Exception:
            pass
    return float(gt.nodes[n].get("node_radius", 1.0))

EDGE_SPLIT = 1
rows = []

for path in dataset_paths:
    print(f"Loading {os.path.basename(path)}...")
    with open(path, "rb") as f:
        payload = pickle.load(f)
    
    gt = payload["gt_graph"]
    edge_error = np.asarray(payload["gt_edge_error"])
    edges_list = list(gt.edges)
    
    # Map edges to their error class, handling both directions as it's an undirected graph
    edge_error_map = {}
    for idx, e in enumerate(edges_list):
        u, v = e
        edge_error_map[(u, v)] = edge_error[idx]
        edge_error_map[(v, u)] = edge_error[idx]
        
    from collections import defaultdict
    neurons = defaultdict(list)
    for n in gt.nodes():
        neurons[gt.node_segment_id(n)].append(n)
        
    for neuron_id, nodes in neurons.items():
        subgraph = gt.subgraph(nodes).copy()
        if len(subgraph) == 0:
            continue
            
        radii = {n: get_radius(gt, n) for n in subgraph.nodes()}
        if not radii:
            continue
            
        # 1. Identify the presumptive soma (node with max radius)
        soma = max(radii, key=radii.get)
        
        # Restrict traversal to the main connected component connected to the soma
        if not nx.is_connected(subgraph):
            comp = nx.node_connected_component(subgraph, soma)
            subgraph = subgraph.subgraph(comp).copy()
            
        # 2. Traverse the graph outwards from the soma to assign a centrifugal branch order
        branch_orders = {soma: 0}
        visited = {soma}
        queue = [soma]
        
        while queue:
            curr = queue.pop(0)
            curr_order = branch_orders[curr]
            
            # Identify children (unvisited neighbors)
            unvisited_neighbors = [n for n in subgraph.neighbors(curr) if n not in visited]
            
            # Increment branch order if the current node is a branching point (>1 child)
            next_order = curr_order + 1 if len(unvisited_neighbors) > 1 else curr_order
            
            for child in unvisited_neighbors:
                visited.add(child)
                branch_orders[child] = next_order
                queue.append(child)
                    
        max_thickness = max(radii.values()) if radii and max(radii.values()) > 0 else 1.0
        
        # 3. For every edge, record its branch order, normalized thickness, and EDGE_SPLIT flag
        for u, v in subgraph.edges():
            # Assign branch order of the edge based on the distal node (deeper topological order)
            order = max(branch_orders.get(u, 0), branch_orders.get(v, 0))
            thickness = (radii.get(u, 1.0) + radii.get(v, 1.0)) / 2.0
            norm_thickness = thickness / max_thickness
            
            is_split = 1 if edge_error_map.get((u, v), 0) == EDGE_SPLIT else 0
            
            rows.append({
                "branch_order": order,
                "norm_thickness": norm_thickness,
                "is_split": is_split
            })

    # Free memory between iterations
    del payload
    del gt
    gc.collect()

# 4. Pool all edges across caches
df = pd.DataFrame(rows)
print(f"\nTotal edges processed: {len(df)}")

if len(df) == 0:
    print("No edges found.")
    sys.exit(1)

# 5. Multiple Logistic Regression predicting EDGE_SPLIT
df['const'] = 1.0
predictors = ['const', 'branch_order']

# Check if norm_thickness has variance to avoid Singular Matrix error
if df['norm_thickness'].nunique() > 1:
    predictors.append('norm_thickness')
else:
    print("\n[Warning] 'norm_thickness' has zero variance (constant radius). Dropping it from the regression model to prevent Singular Matrix error.")

model = sm.Logit(df['is_split'], df[predictors])
try:
    result = model.fit(disp=0)
    print("\n=== Logistic Regression predicting EDGE_SPLIT ===")
    print(result.summary().tables[1])
except Exception as e:
    print("Regression failed:", e)

# Deliverables: Grouping for visual chart of split rates aggregated by branch order
agg = df.groupby('branch_order')['is_split'].agg(['mean', 'count']).reset_index()
# Filter out noise from extreme topological depths with very sparse data for clearer plotting
agg = agg[agg['count'] >= 50] 

plt.figure(figsize=(10, 6))
plt.bar(agg['branch_order'], agg['mean'] * 100, color='coral', edgecolor='black')
plt.xlabel('Centrifugal Branch Order (Topological Depth)')
plt.ylabel('% Split Edges')
plt.title('Split Error Rate by Topological Branch Order')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
