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
import glob
import pickle
import gc
from collections import deque

# Helper to install missing packages
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", package])

# Install all necessary dependencies for the environment and unpickling
install("psutil")
install("pandas")
install("tensorstore")
install("numpy")
install("networkx")
install("scipy")

# Try to install the proofreader package via zip since git may be unavailable
try:
    import agentic_neuron_proofreader
except ImportError:
    install("https://github.com/AllenInstitute/agentic-neuron-proofreader/archive/refs/heads/main.zip")
    import agentic_neuron_proofreader

import numpy as np
import networkx as nx
from scipy.stats import mannwhitneyu

def main():
    # Dataset files are present one level above the current working directory.
    # Avoid recursive search to prevent filesystem traversal timeouts.
    search_paths = [
        "../*_add.pkl",
        "./*_add.pkl",
        "../dataset_cache_*_add.pkl"
    ]
    
    files = []
    for path in search_paths:
        files.extend(glob.glob(path))
        
    # Deduplicate file paths
    files = list(set(os.path.abspath(f) for f in files if f.endswith("_add.pkl")))
    files.sort()
    
    if not files:
        print("No dataset files found.")
        return
        
    omit_distances = []
    correct_distances = []
    
    for fpath in files:
        with open(fpath, "rb") as f:
            payload = pickle.load(f)
            
        gt = payload["gt_graph"]
        edge_error = np.asarray(payload["gt_edge_error"])
        edges = list(gt.edges)
        
        # Identify all leaf nodes (degree == 1) in the ground truth graph
        leaf_nodes = [n for n, d in gt.degree() if d == 1]
        
        # Multi-source BFS for unweighted shortest topological path length
        # This computes the shortest topological distance from any node to its nearest leaf node
        dist_to_leaf = {}
        queue = deque()
        for leaf in leaf_nodes:
            dist_to_leaf[leaf] = 0
            queue.append(leaf)
            
        while queue:
            curr = queue.popleft()
            d = dist_to_leaf[curr]
            for nbr in gt.neighbors(curr):
                if nbr not in dist_to_leaf:
                    dist_to_leaf[nbr] = d + 1
                    queue.append(nbr)
        
        for k, (u, v) in enumerate(edges):
            c = edge_error[k]
            if c in (0, 2):  # Correct (0) or Omit (2)
                du = dist_to_leaf.get(u, None)
                dv = dist_to_leaf.get(v, None)
                
                # Ensure that both endpoints are within a connected component containing at least one leaf node
                if du is not None and dv is not None:
                    # Compute the topological distance from the edge's midpoint to the nearest leaf.
                    # Since adjacent nodes differ by at most 1 in distance, the midpoint is exactly 
                    # min(du, dv) + 0.5
                    dist = min(du, dv) + 0.5
                    
                    if c == 2:
                        omit_distances.append(dist)
                    elif c == 0:
                        correct_distances.append(dist)
                        
        # Release memory to prevent OutOfMemory/Timeout issues across multiple large files
        del payload
        del gt
        del edge_error
        del edges
        del dist_to_leaf
        del leaf_nodes
        del queue
        gc.collect()
                        
    if not omit_distances or not correct_distances:
        print("Missing valid data for omit or correct edges across the datasets.")
        return
        
    omit_distances = np.array(omit_distances)
    correct_distances = np.array(correct_distances)
    
    mean_omit = np.mean(omit_distances)
    median_omit = np.median(omit_distances)
    mean_correct = np.mean(correct_distances)
    median_correct = np.median(correct_distances)
    
    # Test hypothesis: omission errors occur closer to the terminal leaves (distance is stochastically less)
    stat, pval = mannwhitneyu(omit_distances, correct_distances, alternative='less')
    
    print("=== Distance to Nearest Leaf Node (Topological Edges) ===")
    print(f"Omit Edges    - Count: {len(omit_distances)}, Mean: {mean_omit:.4f}, Median: {median_omit:.4f}")
    print(f"Correct Edges - Count: {len(correct_distances)}, Mean: {mean_correct:.4f}, Median: {median_correct:.4f}")
    print(f"Mann-Whitney U Test (Omit < Correct): U-statistic = {stat}, p-value = {pval:.4e}")

if __name__ == "__main__":
    main()
