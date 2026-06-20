# === RERUN BOOTSTRAP (revised loading only) ============================
# Recorded code originally hunted for .pkl files in the CWD/parents and
# pulled in numpy-1.x-incompatible pickles + several heavy pip installs.
# Here we (a) install a numpy>=2 + scientific stack in a vendored dir up
# front, (b) make sure glob/Path/os.walk all redirect any .pkl search to
# the single RERUN_PKL provided by the orchestrator. The downstream
# analysis is unchanged.
import os as _os, sys as _sys

_TARGET = "/home/zihan.zhang/.local-numpy2"
_USER_SITE = _os.path.expanduser("~/.local/lib/python3.12/site-packages")
_SHARED_SITE = "/shared/utils.x86_64/anaconda3-2024.10/lib/python3.12/site-packages"
# Drop the paths that hold numpy 1.x and prepend our numpy 2.x stack.
_sys.path = [p for p in _sys.path if p not in (_USER_SITE, _SHARED_SITE)]
if _TARGET in _sys.path:
    _sys.path.remove(_TARGET)
_sys.path.insert(0, _TARGET)

# Defang the recorded code's `pip install` / repo-fetch retry loops.
import subprocess as _subprocess
_subprocess.check_call = lambda *a, **k: 0
_subprocess.call = lambda *a, **k: 0
_subprocess.run = lambda *a, **k: type("R", (), {"returncode": 0, "stdout": b"", "stderr": b""})()

# Force every .pkl search the script does to resolve to RERUN_PKL.
_PKL = _os.environ["RERUN_PKL"]
print("Loading dataset from:", _PKL)
import glob as _glob
_orig_glob = _glob.glob
def _glob_patch(pattern, *a, **k):
    if isinstance(pattern, str) and pattern.endswith(".pkl"):
        return [_PKL]
    return _orig_glob(pattern, *a, **k)
_glob.glob = _glob_patch

from pathlib import Path as _Path
_orig_rglob = _Path.rglob
def _rglob_patch(self, pattern, *a, **k):
    if isinstance(pattern, str) and pattern.endswith(".pkl"):
        return iter([_Path(_PKL)])
    return _orig_rglob(self, pattern, *a, **k)
_Path.rglob = _rglob_patch

_orig_walk = _os.walk
def _walk_patch(top, *a, **k):
    # The recorded code does `for root, dirs, files in os.walk('..')` and then
    # filters files ending in '_add.pkl'. Return one synthetic walk entry
    # whose file is the single RERUN_PKL.
    yield (_os.path.dirname(_PKL), [], [_os.path.basename(_PKL)])
_os.walk = _walk_patch
# === END BOOTSTRAP =====================================================
import os
import sys
import pickle
import subprocess
import gc

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", package])

# Ensure all necessary packages are present before importing
packages = ["numpy", "networkx", "pandas", "matplotlib", "statsmodels", "psutil", "scipy", "tqdm", "tensorstore"]
for pkg in packages:
    try:
        __import__(pkg)
    except ImportError:
        install(pkg)

try:
    import agentic_neuron_proofreader
except ImportError:
    # Install directly from the repo archive to avoid requiring git
    install("https://github.com/AllenInstitute/agentic-neuron-proofreader/archive/refs/heads/main.zip")
    import agentic_neuron_proofreader

import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Recursively find dataset files one level above the current working directory
dataset_paths = []
for root, dirs, files in os.walk('..'):
    for f in files:
        if f.endswith('_add.pkl'):
            dataset_paths.append(os.path.join(root, f))

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
