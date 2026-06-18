# === RERUN_LOADING_BOOTSTRAP v2 ===
import os as _os, sys as _sys, subprocess as _sp_boot
# Ensure NumPy >= 2 (the pkl was written with NumPy 2.x). Do this BEFORE any
# import that brings in numpy transitively. Quiet, single-shot.
try:
    import numpy as _np_boot
    _ver = tuple(int(x) for x in _np_boot.__version__.split(".")[:2])
    if _ver < (2, 0):
        raise ImportError("need numpy>=2")
except Exception:
    _sp_boot.check_call([_sys.executable, "-m", "pip", "install", "-q", "--user", "numpy>=2,<2.3"])
    # invalidate caches so a fresh numpy 2.x is loaded
    import importlib as _il_boot
    if "numpy" in _sys.modules:
        del _sys.modules["numpy"]
import numpy as _np_chk
print("Loading dataset from:", _os.environ.get("RERUN_PKL", "<unset>"), "| numpy", _np_chk.__version__)

# Make any dataset-search return RERUN_PKL (the runner's open() patch then
# redirects the actual file open). Cover os.walk, Path.rglob, subprocess.getoutput.
import os as _os2, glob as _glob_mod, subprocess as _sp
from pathlib import Path as _Path_boot
_RERUN_PKL = _os.environ.get("RERUN_PKL", "")
_RERUN_DIR = _os.path.dirname(_RERUN_PKL) or "."
_RERUN_NAME = _os.path.basename(_RERUN_PKL)
def _os_walk_patched(top, *a, **k):
    yield (_RERUN_DIR, [], [_RERUN_NAME])
_os2.walk = _os_walk_patched
_orig_rglob = _Path_boot.rglob
def _rglob_patched(self, pat):
    if isinstance(pat, str) and "pkl" in pat:
        yield _Path_boot(_RERUN_PKL)
        return
    yield from _orig_rglob(self, pat)
_Path_boot.rglob = _rglob_patched
_orig_getoutput = _sp.getoutput
def _getoutput_patched(cmd, *a, **k):
    s = str(cmd)
    if "find" in s and "pkl" in s:
        return _RERUN_PKL
    return _orig_getoutput(cmd, *a, **k)
_sp.getoutput = _getoutput_patched
# Short-circuit ALL in-script pip installs. The host environment already has
# numpy>=2, pandas, scipy, matplotlib, statsmodels, networkx, agentic_neuron_proofreader,
# etc. Re-running pip-install inside the subprocess only downgrades numpy and
# breaks the install, so we no-op every pip install here.
_orig_check_call = _sp.check_call
def _check_call_patched(args, *a, **k):
    if isinstance(args, list) and len(args) > 3 and args[1:4] == ["-m", "pip", "install"]:
        return 0
    return _orig_check_call(args, *a, **k)
_sp.check_call = _check_call_patched
_orig_run = _sp.run
def _run_patched(args, *a, **k):
    if isinstance(args, list) and len(args) > 3 and args[1:4] == ["-m", "pip", "install"]:
        class _R: returncode = 0; stdout = b""; stderr = b""
        return _R()
    return _orig_run(args, *a, **k)
_sp.run = _run_patched
# === END RERUN_LOADING_BOOTSTRAP ===

import subprocess
import sys
import os

def install_packages(packages):
    for pkg in packages:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", pkg])

# Ensure all dependencies, including those from the previous experiment instructions, are installed
install_packages([
    "psutil", 
    "tensorstore", 
    "pandas", 
    "networkx", 
    "scipy", 
    "numpy", 
    "tqdm", 
    "matplotlib"
])

try:
    import agentic_neuron_proofreader
except ImportError:
    install_packages(["https://github.com/AllenInstitute/agentic-neuron-proofreader/archive/refs/heads/main.zip"])
    import agentic_neuron_proofreader

import pickle
import numpy as np
import networkx as nx
from scipy.stats import mannwhitneyu
import matplotlib.pyplot as plt

def main():
    possible_paths = [
        "../dataset_cache_789202_mcl100_add.pkl",
        "dataset_cache_789202_mcl100_add.pkl",
        "cache/dataset_cache_789202_mcl100_add.pkl",
        "../cache/dataset_cache_789202_mcl100_add.pkl"
    ]
    dataset_path = None
    for p in possible_paths:
        if os.path.exists(p):
            dataset_path = p
            break
            
    if dataset_path is None:
        raise FileNotFoundError("Could not find the dataset file in any of the expected locations.")
    
    with open(dataset_path, "rb") as f:
        payload = pickle.load(f)

    gt = payload["gt_graph"]
    edge_error = np.asarray(payload["gt_edge_error"])
    node_xyz = gt.node_xyz

    # 1. Identify graph edges and weights based on Euclidean distance
    G_weighted = nx.Graph()
    G_weighted.add_nodes_from(gt.nodes)
    
    gt_edges = list(gt.edges)
    edges_with_weight = []
    
    for u, v in gt_edges:
        dist = np.linalg.norm(node_xyz[u] - node_xyz[v])
        edges_with_weight.append((u, v, dist))
        
    G_weighted.add_weighted_edges_from(edges_with_weight)

    # 2. Identify all leaf nodes (nodes with degree 1)
    leaves = [n for n, d in G_weighted.degree() if d == 1]
    
    if not leaves:
        print("No leaves found in the graph.")
        return

    # 3. Compute shortest path distance from all leaves to every node
    dist_to_leaf = nx.multi_source_dijkstra_path_length(G_weighted, leaves, weight='weight')

    EDGE_CORRECT = 0
    EDGE_SPLIT = 1

    split_distances = []
    correct_distances = []

    # 4. Group calculated distances by edge classification
    for i, (u, v) in enumerate(gt_edges):
        err = edge_error[i]
        if err in (EDGE_CORRECT, EDGE_SPLIT):
            if u in dist_to_leaf and v in dist_to_leaf:
                # Distance from an edge to the nearest leaf is the min distance from either endpoint
                d = min(dist_to_leaf[u], dist_to_leaf[v])
                if err == EDGE_CORRECT:
                    correct_distances.append(d)
                elif err == EDGE_SPLIT:
                    split_distances.append(d)

    split_distances = np.array(split_distances)
    correct_distances = np.array(correct_distances)

    mean_split = np.mean(split_distances) if len(split_distances) > 0 else 0
    mean_correct = np.mean(correct_distances) if len(correct_distances) > 0 else 0

    print("=== RESULTS ===")
    print(f"Number of split edges evaluated: {len(split_distances)}")
    print(f"Number of correct edges evaluated: {len(correct_distances)}")
    print(f"Mean distance to nearest leaf for split edges: {mean_split:.2f} µm")
    print(f"Mean distance to nearest leaf for correct edges: {mean_correct:.2f} µm")

    # 5. Compare the distance-to-leaf distributions
    if len(split_distances) > 0 and len(correct_distances) > 0:
        stat, pval = mannwhitneyu(split_distances, correct_distances, alternative='two-sided')
        print(f"Mann-Whitney U test p-value: {pval:.4e}")
    else:
        print("Not enough data to perform Mann-Whitney U test.")

    # 6. Plot the histogram overlaying the two distributions
    plt.figure(figsize=(10, 6))
    plt.hist(correct_distances, bins=60, alpha=0.5, density=True, label='Correct Edges', color='blue')
    plt.hist(split_distances, bins=60, alpha=0.5, density=True, label='Split Edges', color='orange')
    plt.xlabel('Distance to Nearest Leaf (µm)')
    plt.ylabel('Density')
    plt.title('Distance to Nearest Leaf: Split vs. Correct Edges')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
