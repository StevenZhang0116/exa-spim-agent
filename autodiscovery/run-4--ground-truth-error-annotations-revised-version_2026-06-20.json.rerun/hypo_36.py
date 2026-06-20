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
import subprocess
import sys

# Install required dependencies before importing agentic_neuron_proofreader
try:
    import psutil
    import pandas
    import tensorstore
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "psutil", "pandas", "tensorstore"])

try:
    import agentic_neuron_proofreader
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "https://github.com/AllenInstitute/agentic-neuron-proofreader/archive/refs/heads/main.zip"])
    import agentic_neuron_proofreader

import pickle
import numpy as np
from scipy.spatial import KDTree
from scipy.stats import mannwhitneyu

# Load the dataset from the current directory
path = "./dataset_cache_789202_mcl100_add.pkl"
with open(path, "rb") as f:
    payload = pickle.load(f)

gt = payload["gt_graph"]
edge_error = np.asarray(payload["gt_edge_error"])
merge_sites = payload["gt_merge_sites"]

# Extract spatial coordinates for merge sites
merge_xyz = np.array([site["xyz"] for site in merge_sites])

if len(merge_xyz) == 0:
    print("No merge sites found.")
    sys.exit(0)

# Identify omit nodes and correctly reconstructed nodes using the edge errors
omit_nodes = set()
correct_nodes = set()

gt_edges = list(gt.edges)
for i in range(len(gt_edges)):
    u, v = gt_edges[i]
    err = edge_error[i]
    if err == 2: # EDGE_OMIT
        omit_nodes.add(u)
        omit_nodes.add(v)
    elif err == 0: # EDGE_CORRECT
        correct_nodes.add(u)
        correct_nodes.add(v)

# Exclude omit nodes from the correct nodes pool to ensure strictly distinct spatial sets
correct_nodes = list(correct_nodes - omit_nodes)
omit_nodes = list(omit_nodes)

# Extract (x,y,z) coordinates strictly using gt.node_xyz parallel array
omit_xyz = np.array([gt.node_xyz[n] for n in omit_nodes])
correct_xyz = np.array([gt.node_xyz[n] for n in correct_nodes])

# Compute distance to nearest merge site for omit nodes
merge_tree = KDTree(merge_xyz)
dist_omit, _ = merge_tree.query(omit_xyz)

# Draw a matched random sample of correct nodes and compute distance to nearest merge site
np.random.seed(42)
if len(correct_xyz) > len(omit_xyz):
    sampled_correct_idx = np.random.choice(len(correct_xyz), size=len(omit_xyz), replace=False)
    sampled_correct_xyz = correct_xyz[sampled_correct_idx]
else:
    sampled_correct_xyz = correct_xyz

dist_correct, _ = merge_tree.query(sampled_correct_xyz)

# Statistical comparison
median_dist_omit = np.median(dist_omit)
median_dist_correct = np.median(dist_correct)

# We test the alternative hypothesis that omit nodes are closer to merge sites (less distance)
stat, pval = mannwhitneyu(dist_omit, dist_correct, alternative='less')

print("=== Spatial Clustering of Omit Errors vs Merge Errors ===")
print(f"Number of merge sites: {len(merge_xyz)}")
print(f"Number of omit nodes: {len(omit_xyz)}")
print(f"Number of correctly reconstructed nodes (matched sample): {len(sampled_correct_xyz)}")
print(f"Median distance of omit nodes to nearest merge site: {median_dist_omit:.2f} µm")
print(f"Median distance of correct nodes to nearest merge site: {median_dist_correct:.2f} µm")
print(f"Mann-Whitney U statistic: {stat}")
print(f"p-value: {pval:.4e}")
