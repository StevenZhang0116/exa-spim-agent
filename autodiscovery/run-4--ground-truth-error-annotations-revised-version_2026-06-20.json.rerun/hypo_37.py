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
import sys
from unittest.mock import MagicMock
import gc
import os
import glob
import pickle
import numpy as np
import subprocess

# Ensure essential lightweight dependencies are installed
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", package])

for pkg in ["tqdm", "networkx", "scipy"]:
    try:
        __import__(pkg)
    except ImportError:
        install(pkg)

# Mock heavy, unnecessary dependencies to prevent long builds and timeouts
class MockException(Exception): pass

m_google_auth_exc = MagicMock()
m_google_auth_exc.RefreshError = MockException
m_google_auth_exc.TransportError = MockException

mocked_modules = [
    'tensorstore', 'psutil', 'pandas', 'matplotlib', 'matplotlib.pyplot',
    'matplotlib.colors', 'zarr', 'dask', 's3fs', 'boto3', 'botocore',
    'botocore.client', 'google', 'google.auth', 'google.cloud'
]

for mod in mocked_modules:
    sys.modules[mod] = MagicMock()

sys.modules['google.auth.exceptions'] = m_google_auth_exc

try:
    import agentic_neuron_proofreader
except ImportError:
    # Use --no-deps to prevent compiling huge packages from source on Python 3.13
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "--no-deps", "https://github.com/AllenInstitute/agentic-neuron-proofreader/archive/main.zip"])
    import agentic_neuron_proofreader

from scipy.spatial import cKDTree
from scipy.stats import mannwhitneyu

# Find dataset safely, prioritizing the known parent directory location
dataset_paths = glob.glob("../*_add.pkl")
if not dataset_paths:
    dataset_paths = glob.glob("./*_add.pkl")
if not dataset_paths:
    dataset_paths = glob.glob("../**/*_add.pkl", recursive=True)
if not dataset_paths:
    dataset_paths = glob.glob("cache/*_add.pkl")

if not dataset_paths:
    print("Dataset not found.")
else:
    dataset_path = dataset_paths[0]
    
    # Load dataset payload
    with open(dataset_path, "rb") as f:
        payload = pickle.load(f)

    gt = payload["gt_graph"]
    edge_error = np.asarray(payload["gt_edge_error"])
    
    gt_edges_list = list(gt.edges)
    edges_arr = np.array(gt_edges_list)
    
    # Extract needed attributes directly
    node_xyz = gt.node_xyz
    
    # Free up multi-gigabyte memory immediately to prevent swapping and thrashing timeouts
    payload.pop("fragments_graph", None)
    del payload
    del gt
    gc.collect()
    
    if len(edges_arr) > 0:
        # Vectorized extraction of midpoints to prevent loop timeouts
        u_nodes = edges_arr[:, 0]
        v_nodes = edges_arr[:, 1]
        
        midpoints = (node_xyz[u_nodes] + node_xyz[v_nodes]) / 2.0
        
        split_midpoints = midpoints[edge_error == 1]
        correct_midpoints = midpoints[edge_error == 0]
        
        # Downsample the correct control group to match the split group size
        np.random.seed(42)
        if len(correct_midpoints) > len(split_midpoints) and len(split_midpoints) > 0:
            idx = np.random.choice(len(correct_midpoints), len(split_midpoints), replace=False)
            sampled_correct_midpoints = correct_midpoints[idx]
        else:
            sampled_correct_midpoints = correct_midpoints

        if len(split_midpoints) > 0:
            # Build KD-Tree for split midpoints
            tree = cKDTree(split_midpoints)
            radius = 30.0

            # Query KD-Tree with return_length=True for extreme C-level speed
            split_neighbor_counts = tree.query_ball_point(split_midpoints, r=radius, return_length=True)
            split_counts = np.maximum(0, np.array(split_neighbor_counts) - 1)  # Exclude self
            
            if len(sampled_correct_midpoints) > 0:
                correct_counts = np.array(tree.query_ball_point(sampled_correct_midpoints, r=radius, return_length=True))
            else:
                correct_counts = np.array([])

            if len(split_counts) > 0 and len(correct_counts) > 0:
                # Use a Mann-Whitney U test to compare the neighbor counts between the two groups
                u_stat, p_val = mannwhitneyu(split_counts, correct_counts, alternative='greater')

                print("=== Spatial Clustering of Split Errors ===")
                print(f"Number of split edges: {len(split_counts)}")
                print(f"Number of correct edges (sampled): {len(correct_counts)}")
                print(f"Mean split neighbors within {radius} µm for Split edges: {np.mean(split_counts):.2f}")
                print(f"Mean split neighbors within {radius} µm for Correct edges: {np.mean(correct_counts):.2f}")
                print(f"Mann-Whitney U statistic: {u_stat}")
                print(f"p-value: {p_val:.2e}")
            else:
                print("Not enough edges to compute statistics.")
        else:
            print("No split edges found.")
    else:
        print("No edges found in the graph.")
