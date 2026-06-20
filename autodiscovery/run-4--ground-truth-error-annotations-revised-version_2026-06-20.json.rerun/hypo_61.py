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
import subprocess
import sys
import os
import glob

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", package])

# Ensure all dependencies that might be triggered by pickle.load are installed
for pkg in ["psutil", "pandas", "tensorstore"]:
    try:
        __import__(pkg)
    except ImportError:
        install(pkg)

try:
    import agentic_neuron_proofreader
except ImportError:
    install("https://github.com/AllenInstitute/agentic-neuron-proofreader/archive/refs/heads/main.zip")
    import agentic_neuron_proofreader

import pickle
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import mannwhitneyu, ttest_ind

# Find dataset file directly in the current directory
file_paths = glob.glob("./*_add.pkl")
if not file_paths:
    print("Dataset file not found.")
    sys.exit(1)

EDGE_CORRECT, EDGE_SPLIT = 0, 1

split_distances = []
correct_distances = []

for file_path in file_paths:
    with open(file_path, "rb") as f:
        payload = pickle.load(f)
    
    gt = payload["gt_graph"]
    edge_error = np.asarray(payload["gt_edge_error"])
    merge_sites = payload.get("gt_merge_sites", [])
    
    # 1. Extract 3D coordinates of all merge sites
    if not merge_sites:
        continue
    
    merge_xyz = np.array([site["xyz"] for site in merge_sites])
    if len(merge_xyz) == 0:
        continue
        
    merge_tree = cKDTree(merge_xyz)
    
    edges = list(gt.edges)
    if len(edges) == 0:
        continue
        
    # Vectorized edge array preparation
    u = np.array([e[0] for e in edges])
    v = np.array([e[1] for e in edges])
    
    # 2. Vectorized computation of 3D midpoints for each GT edge
    try:
        # Access parallel NumPy array if available
        u_xyz = gt.node_xyz[u]
        v_xyz = gt.node_xyz[v]
    except (AttributeError, TypeError, IndexError):
        # Fallback if node_xyz isn't directly indexable with numpy arrays
        u_xyz = np.array([gt.nodes[n].get('node_xyz', gt.nodes[n].get('xyz')) for n in u])
        v_xyz = np.array([gt.nodes[n].get('node_xyz', gt.nodes[n].get('xyz')) for n in v])
    
    midpoints = (u_xyz + v_xyz) / 2.0
    
    # 3. Vectorized KDTree query to find the nearest merge site for all midpoints
    distances, _ = merge_tree.query(midpoints)
    
    # 4. Group distances by edge class (Split vs Correct)
    split_mask = (edge_error == EDGE_SPLIT)
    correct_mask = (edge_error == EDGE_CORRECT)
    
    split_distances.extend(distances[split_mask])
    correct_distances.extend(distances[correct_mask])

split_distances = np.array(split_distances)
correct_distances = np.array(correct_distances)

print(f"Number of split edges: {len(split_distances)}")
print(f"Number of correct edges: {len(correct_distances)}")

if len(split_distances) == 0 or len(correct_distances) == 0:
    print("Not enough edges in one of the classes to perform statistical test.")
    sys.exit(0)

print(f"\nMean distance to nearest merge site - Split edges:   {np.mean(split_distances):.2f} \u00b5m")
print(f"Mean distance to nearest merge site - Correct edges: {np.mean(correct_distances):.2f} \u00b5m")
print(f"Median distance to nearest merge site - Split edges:   {np.median(split_distances):.2f} \u00b5m")
print(f"Median distance to nearest merge site - Correct edges: {np.median(correct_distances):.2f} \u00b5m")

# 5. Perform Statistical Tests
# Mann-Whitney U test (non-parametric, robust to outliers since distance distributions are likely skewed)
stat, pval = mannwhitneyu(split_distances, correct_distances, alternative='less')
print("\nMann-Whitney U Test (Alternative Hypothesis: Split distances < Correct distances):")
print(f"Statistic: {stat}")
print(f"p-value:   {pval:.2e}")

# Welch's t-test
t_stat, t_pval = ttest_ind(split_distances, correct_distances, equal_var=False, alternative='less')
print("\nWelch's t-test (Alternative Hypothesis: Split distances < Correct distances):")
print(f"t-statistic: {t_stat:.4f}")
print(f"p-value:     {t_pval:.2e}")

if pval < 0.05 or t_pval < 0.05:
    print("\nResult: Significant evidence that split edges are spatially closer to merge sites than correct edges.")
else:
    print("\nResult: No significant evidence that split edges are spatially closer to merge sites than correct edges.")
