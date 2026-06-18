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

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", package])

# Install missing dependencies identified from the traceback and previous explorations
for pkg in ["psutil", "pandas", "tensorstore", "statsmodels"]:
    try:
        __import__(pkg)
    except ImportError:
        install(pkg)

# Ensure agentic_neuron_proofreader is installed
try:
    import agentic_neuron_proofreader
except ImportError:
    # Since git is not available in the environment, install directly from the GitHub zip archive
    try:
        install("https://github.com/AllenInstitute/agentic-neuron-proofreader/archive/refs/heads/main.zip")
    except subprocess.CalledProcessError:
        install("https://github.com/AllenInstitute/agentic-neuron-proofreader/archive/refs/heads/master.zip")
    import agentic_neuron_proofreader

# Ensure matplotlib is installed for plotting
try:
    import matplotlib.pyplot as plt
except ImportError:
    install("matplotlib")
    import matplotlib.pyplot as plt

import glob
import pickle
import numpy as np
import networkx as nx
from scipy.stats import mannwhitneyu
import pathlib

def main():
    # 1. Locate and load the dataset
    dataset_path = None
    # Search for the dataset recursively in parent and current directory
    for p in pathlib.Path('..').rglob('*add.pkl'):
        dataset_path = str(p)
        break
    if not dataset_path:
        for p in pathlib.Path('.').rglob('*add.pkl'):
            dataset_path = str(p)
            break
            
    if not dataset_path:
        print("Dataset not found. Please ensure the dataset is available.")
        return
        
    print(f"Loading dataset from: {dataset_path}")

    with open(dataset_path, "rb") as f:
        payload = pickle.load(f)

    gt = payload["gt_graph"]
    edge_error = np.asarray(payload["gt_edge_error"])
    
    print(f"Graph has {gt.number_of_nodes()} nodes and {gt.number_of_edges()} edges.")

    # 2. Add spatial weights to edges (Euclidean distance in \u00B5m)
    print("Calculating edge weights based on spatial distance...")
    for u, v, data in gt.edges(data=True):
        dist = np.linalg.norm(gt.node_xyz[u] - gt.node_xyz[v])
        data['weight'] = dist

    # 3. Identify ground-truth branch points
    branch_nodes = {n for n, d in gt.degree() if d >= 3}
    print(f"Found {len(branch_nodes)} GT branch nodes.")
    if not branch_nodes:
        print("No branch nodes found in gt_graph. Cannot test hypothesis.")
        return

    # 4. Compute shortest topological paths to the nearest branch node
    print("Computing shortest topological paths to nearest branch node...")
    dist_to_branch = nx.multi_source_dijkstra_path_length(gt, branch_nodes, weight='weight')
    print(f"Computed distances for {len(dist_to_branch)} nodes.")

    # 5. Extract edge distances to nearest branch point
    EDGE_CORRECT = 0
    EDGE_SPLIT = 1

    split_distances = []
    correct_distances = []

    print("Classifying edges and extracting distances...")
    for i, (u, v) in enumerate(list(gt.edges)):
        err = edge_error[i]
        # Only consider edges where both endpoints are in a connected component with a branch node
        if u in dist_to_branch and v in dist_to_branch:
            # Assign distance as the average of its endpoints' distances to the nearest branch
            edge_dist = (dist_to_branch[u] + dist_to_branch[v]) / 2.0
            if err == EDGE_SPLIT:
                split_distances.append(edge_dist)
            elif err == EDGE_CORRECT:
                correct_distances.append(edge_dist)

    split_distances = np.array(split_distances)
    correct_distances = np.array(correct_distances)

    print(f"Number of EDGE_SPLIT edges analyzed: {len(split_distances)}")
    print(f"Number of EDGE_CORRECT edges analyzed: {len(correct_distances)}")

    if len(split_distances) == 0 or len(correct_distances) == 0:
        print("Not enough data in one of the categories to perform statistical test.")
        return

    # 6. Perform statistical test
    print("\nPerforming Mann-Whitney U test...")
    stat, p_value = mannwhitneyu(split_distances, correct_distances, alternative='two-sided')
    
    print("\n--- Results ---")
    print(f"Mann-Whitney U statistic: {stat}")
    print(f"p-value: {p_value:.5e}")
    print(f"Mean distance to branch (EDGE_SPLIT): {np.mean(split_distances):.2f} \u00B5m")
    print(f"Mean distance to branch (EDGE_CORRECT): {np.mean(correct_distances):.2f} \u00B5m")
    print(f"Median distance to branch (EDGE_SPLIT): {np.median(split_distances):.2f} \u00B5m")
    print(f"Median distance to branch (EDGE_CORRECT): {np.median(correct_distances):.2f} \u00B5m")

    # 7. Plot Cumulative Density Function (CDF)
    print("\nPlotting CDF...")
    plt.figure(figsize=(10, 6))

    x_split = np.sort(split_distances)
    y_split = np.arange(1, len(x_split) + 1) / len(x_split)

    x_correct = np.sort(correct_distances)
    y_correct = np.arange(1, len(x_correct) + 1) / len(x_correct)

    plt.plot(x_split, y_split, label='EDGE_SPLIT', color='red', linewidth=2)
    plt.plot(x_correct, y_correct, label='EDGE_CORRECT', color='blue', linewidth=2)

    plt.title('CDF of Topological Distance to Nearest GT Branch Point')
    plt.xlabel('Distance to Nearest Branch (\u00B5m)')
    plt.ylabel('Cumulative Probability')
    
    # Cap x-axis to 95th percentile to clearly visualize the main body of the distribution without the long tail obscuring it
    max_val = max(np.percentile(split_distances, 95), np.percentile(correct_distances, 95))
    plt.xlim(0, max_val * 1.1)
    
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
