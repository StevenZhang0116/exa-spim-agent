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

import sys
import subprocess

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", package])

# Install standard dependencies if not present
for pkg in ["psutil", "tensorstore", "pandas", "tqdm", "networkx", "scipy", "numpy", "matplotlib"]:
    try:
        __import__(pkg)
    except ImportError:
        install(pkg)

# Install the agentic_neuron_proofreader package directly from GitHub archive
try:
    import agentic_neuron_proofreader
except ImportError:
    install("https://github.com/AllenInstitute/agentic-neuron-proofreader/archive/refs/heads/main.zip")
    import agentic_neuron_proofreader

import pickle
import numpy as np
import networkx as nx
from scipy.spatial import cKDTree
from scipy.stats import mannwhitneyu
import matplotlib.pyplot as plt

# 2. Load dataset
dataset_path = "dataset_cache_789202_mcl100_add.pkl"
with open(dataset_path, "rb") as f:
    payload = pickle.load(f)

gt = payload["gt_graph"]
edge_error = np.asarray(payload["gt_edge_error"])

# 3. Identify branch nodes (degree >= 3)
branch_nodes = [n for n, d in gt.degree() if d >= 3]

# Group nodes and branch nodes by GT neuron ID
neuron_nodes = {}
for n in gt.nodes:
    neuron_id = gt.node_segment_id(n)
    if neuron_id not in neuron_nodes:
        neuron_nodes[neuron_id] = []
    neuron_nodes[neuron_id].append(n)

neuron_branch_nodes = {}
for n in branch_nodes:
    neuron_id = gt.node_segment_id(n)
    if neuron_id not in neuron_branch_nodes:
        neuron_branch_nodes[neuron_id] = []
    neuron_branch_nodes[neuron_id].append(n)

# 4. Compute Euclidean distance from every node to the nearest branch node on the same neuron
node_to_dist = {}
for neuron_id, n_list in neuron_nodes.items():
    b_list = neuron_branch_nodes.get(neuron_id, [])
    if len(b_list) == 0:
        for n in n_list:
            node_to_dist[n] = float('inf')
        continue
    
    b_coords = np.array([gt.node_xyz[b] for b in b_list])
    tree = cKDTree(b_coords)
    
    n_coords = np.array([gt.node_xyz[n] for n in n_list])
    dists, _ = tree.query(n_coords)
    for n, dist in zip(n_list, dists):
        node_to_dist[n] = dist

# 5. Assign distance to edges as the minimum distance of its endpoints
EDGE_CORRECT = 0
EDGE_SPLIT = 1

split_dists = []
correct_dists = []

edges = list(gt.edges)
for k, (u, v) in enumerate(edges):
    error_type = edge_error[k]
    if error_type in (EDGE_SPLIT, EDGE_CORRECT):
        dist = min(node_to_dist[u], node_to_dist[v])
        if dist == float('inf'):
            continue
        if error_type == EDGE_SPLIT:
            split_dists.append(dist)
        elif error_type == EDGE_CORRECT:
            correct_dists.append(dist)

# 6. Statistical testing
stat, pval = mannwhitneyu(split_dists, correct_dists, alternative='two-sided')
mean_split = np.mean(split_dists)
mean_correct = np.mean(correct_dists)
median_split = np.median(split_dists)
median_correct = np.median(correct_dists)

print("=== Spatial Correlation of Split Errors to Branch Nodes ===")
print(f"Mean distance to branch point (Split):   {mean_split:.2f} \u00b5m (Median: {median_split:.2f} \u00b5m)")
print(f"Mean distance to branch point (Correct): {mean_correct:.2f} \u00b5m (Median: {median_correct:.2f} \u00b5m)")
print(f"Mann-Whitney U test: U={stat}, p-value={pval}")

# 7. Visualization
plt.figure(figsize=(10, 6))
all_dists = correct_dists + split_dists
max_dist = min(max(all_dists), 200) if all_dists else 200  # Cap the visualization at 200 um for clarity
bins = np.linspace(0, max_dist, 50)

plt.hist(correct_dists, bins=bins, alpha=0.5, label=f'Correct (n={len(correct_dists)})', density=True)
plt.hist(split_dists, bins=bins, alpha=0.5, label=f'Split (n={len(split_dists)})', density=True)
plt.xlabel("Distance to nearest branch point (\u00b5m)")
plt.ylabel("Density")
plt.legend()
plt.title("Distance to Nearest Branch Point: Split vs Correct Edges")
plt.tight_layout()
plt.show()
