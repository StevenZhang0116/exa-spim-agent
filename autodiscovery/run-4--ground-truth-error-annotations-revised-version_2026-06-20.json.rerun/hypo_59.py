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
import glob
import os
import pickle
import numpy as np
import gc
from scipy.stats import mannwhitneyu, pointbiserialr
import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", package])

# Install missing dependencies required by the package
try:
    import psutil
except ImportError:
    install("psutil")

try:
    import pandas
except ImportError:
    install("pandas")

try:
    import tensorstore
except ImportError:
    install("tensorstore")

try:
    import networkx
except ImportError:
    install("networkx")

try:
    import scipy
except ImportError:
    install("scipy")

try:
    import agentic_neuron_proofreader
except ImportError:
    install("https://github.com/AllenInstitute/agentic-neuron-proofreader/archive/main.zip")
    import agentic_neuron_proofreader

def get_paths_of_length(G, length):
    """
    Extracts all simple paths of exactly the specified length from a tree/forest.
    """
    paths = []
    # Pre-build adjacency list for faster traversal
    adj = {n: list(G.neighbors(n)) for n in G.nodes()}
    for start_node in G.nodes():
        stack = [(start_node, [start_node])]
        while stack:
            curr, path = stack.pop()
            if len(path) == length + 1:
                if start_node < curr:  # Avoid duplicate reverse paths
                    paths.append(path)
                continue
            
            path_len = len(path)
            for neighbor in adj[curr]:
                if path_len == 1 or neighbor != path[-2]:
                    stack.append((neighbor, path + [neighbor]))
    return paths

def main():
    files = glob.glob("*_add.pkl")
    if not files:
        print("No _add.pkl files found.")
        return

    split_tortuosity = []
    correct_tortuosity = []

    for fpath in files:
        with open(fpath, "rb") as f:
            payload = pickle.load(f)
        
        gt = payload["gt_graph"]
        edge_error = np.asarray(payload["gt_edge_error"])
        
        # Delete payload to free up memory before allocating new large structures
        del payload
        gc.collect()

        edges_list = list(gt.edges)
        
        # Access node coordinates properly based on the object structure
        node_xyz = {}
        for n in gt.nodes:
            if 'xyz' in gt.nodes[n]:
                node_xyz[n] = gt.nodes[n]['xyz']
            elif hasattr(gt, 'node_xyz'):
                node_xyz[n] = gt.node_xyz[n]
            else:
                raise ValueError("Could not find xyz coordinates for node.")

        edge_info = {}
        for idx, (u, v) in enumerate(edges_list):
            fs = frozenset((u, v))
            err = edge_error[idx]
            
            p_u = np.array(node_xyz[u])
            p_v = np.array(node_xyz[v])
                
            dist = np.linalg.norm(p_u - p_v)
            edge_info[fs] = {'dist': dist, 'error': err}
            
        # Get all paths of 10 edges (11 nodes) along the neuron branches
        paths = get_paths_of_length(gt, 10)
        
        # Store tortuosity scores to average per central edge later
        edge_tortuosities = {fs: [] for fs in edge_info}
        
        for p in paths:
            path_len = 0.0
            for i in range(10):
                fs = frozenset((p[i], p[i+1]))
                if fs in edge_info:
                    path_len += edge_info[fs]['dist']
                
            p0 = np.array(node_xyz[p[0]])
            p10 = np.array(node_xyz[p[-1]])
                
            euclidean = np.linalg.norm(p0 - p10)
            # If euclidean is effectively zero, path is completely folded, tortuosity is arbitrarily high or 1.0
            tortuosity = path_len / euclidean if euclidean > 1e-6 else 1.0
            
            # Map to the two central-most edges in the 10-edge sequence (indices 4 & 5)
            e4 = frozenset((p[4], p[5]))
            e5 = frozenset((p[5], p[6]))
            
            if e4 in edge_tortuosities:
                edge_tortuosities[e4].append(tortuosity)
            if e5 in edge_tortuosities:
                edge_tortuosities[e5].append(tortuosity)
            
        for fs, tort_list in edge_tortuosities.items():
            if not tort_list:
                continue
            avg_tort = np.mean(tort_list)
            err = edge_info[fs]['error']
            if err == 1: # EDGE_SPLIT
                split_tortuosity.append(avg_tort)
            elif err == 0: # EDGE_CORRECT
                correct_tortuosity.append(avg_tort)
                
        # Force cleanup before next iteration to save memory
        del gt
        del edge_info
        del edge_tortuosities
        del paths
        gc.collect()
                
    if not split_tortuosity or not correct_tortuosity:
        print("Not enough data to compare.")
        return
        
    split_tortuosity = np.array(split_tortuosity)
    correct_tortuosity = np.array(correct_tortuosity)
    
    print(f"Number of split edges analyzed: {len(split_tortuosity)}")
    print(f"Number of correct edges analyzed: {len(correct_tortuosity)}")
    print("-" * 40)
    print(f"Median Tortuosity (Split)   : {np.median(split_tortuosity):.4f}")
    print(f"Median Tortuosity (Correct) : {np.median(correct_tortuosity):.4f}")
    print(f"Mean Tortuosity (Split)     : {np.mean(split_tortuosity):.4f}")
    print(f"Mean Tortuosity (Correct)   : {np.mean(correct_tortuosity):.4f}")
    print("-" * 40)
    
    # Non-parametric Mann-Whitney U Test (comparison of distributions)
    # Tests if local tortuosity for split edges is stochastically greater than for correct edges
    stat, p_val = mannwhitneyu(split_tortuosity, correct_tortuosity, alternative='greater')
    print(f"Mann-Whitney U statistic: {stat:.4e}, p-value: {p_val:.4e}")
    
    # Point-biserial correlation (dichotomous vs continuous variables)
    labels = np.concatenate([np.ones_like(split_tortuosity), np.zeros_like(correct_tortuosity)])
    values = np.concatenate([split_tortuosity, correct_tortuosity])
    
    r, p_r = pointbiserialr(labels, values) 
    print(f"Point-biserial correlation: {r:.4f}, p-value: {p_r:.4e}")

if __name__ == "__main__":
    main()
