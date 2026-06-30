# CORRECTED TEST for hypothesis id 139 (branch order -> split risk).
#
# FAULT (verifier MAJOR): the logistic regression of is_split on branch_order was
# fit over 1.4M edges treated as independent, but edges within a neuron are
# spatially correlated, so the reported z=16.369 (p<0.001) overstates the
# effective information -- the true SE is larger. (The "independent of thickness"
# clause is separately untestable because norm_thickness has zero variance in
# this pkl, so it is dropped; we cannot manufacture a thickness control that does
# not exist, but we DO fix the independence fault on the slope itself.)
#
# CORRECTION: keep the SAME predictor (branch_order) and SAME outcome (is_split),
# but obtain a VALID significance for the slope with CLUSTER-ROBUST standard
# errors clustered BY NEURON (cov_type='cluster'), and report the odds ratio per
# +1 branch order WITH a 95% CI as the effect size. Also report a NEURON-LEVEL
# permutation p-value (permute neuron mean-branch-order against neuron split-rate;
# the neuron is the independent unit).
#
# ORIGINAL RECORDED NUMBERS (for the driver to compare):
#   edges=1409045, branch_order coef=0.0194, z=16.369, p<0.001 (norm_thickness dropped).

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
import scipy.stats

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

            unvisited_neighbors = [n for n in subgraph.neighbors(curr) if n not in visited]
            next_order = curr_order + 1 if len(unvisited_neighbors) > 1 else curr_order

            for child in unvisited_neighbors:
                visited.add(child)
                branch_orders[child] = next_order
                queue.append(child)

        max_thickness = max(radii.values()) if radii and max(radii.values()) > 0 else 1.0

        # 3. For every edge, record its branch order, normalized thickness, EDGE_SPLIT flag,
        #    AND the neuron id (the cluster/independent unit).
        for u, v in subgraph.edges():
            order = max(branch_orders.get(u, 0), branch_orders.get(v, 0))
            thickness = (radii.get(u, 1.0) + radii.get(v, 1.0)) / 2.0
            norm_thickness = thickness / max_thickness

            is_split = 1 if edge_error_map.get((u, v), 0) == EDGE_SPLIT else 0

            rows.append({
                "branch_order": order,
                "norm_thickness": norm_thickness,
                "is_split": is_split,
                "neuron_id": neuron_id,
            })

    del payload
    del gt
    gc.collect()

# 4. Pool all edges across caches
df = pd.DataFrame(rows)
print(f"\nTotal edges processed: {len(df)}")

if len(df) == 0:
    print("No edges found.")
    sys.exit(1)

# 5. Logistic regression predicting EDGE_SPLIT with branch_order.
df['const'] = 1.0
predictors = ['const', 'branch_order']
if df['norm_thickness'].nunique() > 1:
    predictors.append('norm_thickness')
else:
    print("\n[Warning] 'norm_thickness' has zero variance (constant radius); dropping it"
          " (the 'independent of thickness' clause remains structurally untestable in this pkl).")

X = df[predictors]
y = df['is_split'].astype(float)
groups = df['neuron_id'].astype(str).values

model = sm.Logit(y, X)
res_naive = model.fit(disp=0)
res_cluster = model.fit(disp=0, cov_type='cluster', cov_kwds={'groups': groups})

coef_bo = res_cluster.params['branch_order']
z_naive = res_naive.tvalues['branch_order']
p_naive = res_naive.pvalues['branch_order']
z_cluster = res_cluster.tvalues['branch_order']
p_cluster = res_cluster.pvalues['branch_order']
ci_lo, ci_hi = res_cluster.conf_int().loc['branch_order'].values
or_bo = float(np.exp(coef_bo))
or_lo = float(np.exp(ci_lo))
or_hi = float(np.exp(ci_hi))

print("\n=== Corrected logistic regression (branch_order; cluster-robust by neuron) ===")
print(f"  N edges                            : {len(df)}")
print(f"  N clusters (neurons)               : {df['neuron_id'].nunique()}")
print(f"  branch_order coef                  : {coef_bo:.5f}")
print(f"  Naive z / p (edge-independent)     : z={z_naive:.3f}, p={p_naive:.4g}")
print(f"  CLUSTER-ROBUST z / p (by neuron)   : z={z_cluster:.3f}, p={p_cluster:.4g}")
print(f"  Odds ratio per +1 branch order     : {or_bo:.4f}")
print(f"  OR 95% CI (cluster-robust)         : [{or_lo:.4f}, {or_hi:.4f}]")

# === NEURON-LEVEL permutation cross-check (neuron = independent unit) ===
grp = df.groupby('neuron_id').agg(bo_mean=('branch_order', 'mean'),
                                  split_rate=('is_split', 'mean'),
                                  n=('is_split', 'size'))
grp = grp[grp['n'] >= 5]
bo = grp['bo_mean'].values
sr = grp['split_rate'].values
obs_rho, _ = scipy.stats.spearmanr(bo, sr)
rng = np.random.default_rng(42)
n_perm = 5000
perm = np.empty(n_perm)
for i in range(n_perm):
    perm[i], _ = scipy.stats.spearmanr(rng.permutation(bo), sr)
perm_p = (np.sum(np.abs(perm) >= abs(obs_rho)) + 1) / (n_perm + 1)
print("\n=== Neuron-level permutation cross-check ===")
print(f"  Neuron clusters used               : {len(grp)}")
print(f"  Spearman rho(neuron BO vs split rt): {obs_rho:.4f}")
print(f"  Permutation two-sided p            : {perm_p:.4g}")
print("  (Compare to ORIGINAL: coef=0.0194, z=16.369, p<0.001, treated as independent.)")
