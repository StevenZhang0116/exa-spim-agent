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

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", package])

# Ensure dependencies are installed
try:
    import agentic_neuron_proofreader
except ImportError:
    install("tensorstore")
    install("psutil")
    install("pandas")
    install("tqdm")
    install("networkx")
    install("scipy")
    install("matplotlib")
    install("https://github.com/AllenInstitute/agentic-neuron-proofreader/archive/refs/heads/main.zip")

import agentic_neuron_proofreader
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx

EDGE_CORRECT = 0
EDGE_SPLIT = 1

data = []

# Safely locate the dataset files
search_dirs = ['.', '..', '/data', '/workspace']
pkl_files = []
for d in search_dirs:
    pkl_files.extend(glob.glob(os.path.join(d, '*_add.pkl')))
pkl_files = list(set(pkl_files))

if not pkl_files:
    print("No dataset files found.")
    sys.exit(1)

# Process each dataset file
for path in pkl_files:
    print(f"Processing {os.path.basename(path)}...")
    basename = os.path.basename(path)
    parts = basename.split('_')
    brain_id = parts[2] if len(parts) > 2 else 'unknown'
    
    with open(path, 'rb') as f:
        payload = pickle.load(f)
    
    gt = payload["gt_graph"]
    edge_error = np.asarray(payload["gt_edge_error"])
    edges = list(gt.edges)
    node_xyz = gt.node_xyz
    
    # 2. Build weighted graph for morphological proxy calculation
    G = nx.Graph()
    edges_with_weights = []
    for u, v in edges:
        w = np.linalg.norm(node_xyz[u] - node_xyz[v])
        edges_with_weights.append((u, v, w))
    G.add_weighted_edges_from(edges_with_weights)
    
    # Identify terminal processes (leaves)
    leaves = [n for n, d in G.degree() if d <= 1]
    
    if leaves:
        # Compute topological distance to the nearest leaf for all reachable nodes
        dist_dict = nx.multi_source_dijkstra_path_length(G, set(leaves), weight='weight')
        
        # 3. Aggregate proxy thickness and edge errors
        for k, (u, v) in enumerate(edges):
            err = edge_error[k]
            if err in (EDGE_CORRECT, EDGE_SPLIT):
                if u in dist_dict and v in dist_dict:
                    dist_u = dist_dict[u]
                    dist_v = dist_dict[v]
                    avg_dist = (dist_u + dist_v) / 2.0
                    data.append((avg_dist, err, brain_id))
    
    # Manual cleanup to handle memory limits across multiple brains
    del G
    del edges_with_weights
    del gt
    del payload
    gc.collect()

# Combine and format data
df = pd.DataFrame(data, columns=['dist', 'error_class', 'brain_id']).dropna()
split_df = df[df['error_class'] == EDGE_SPLIT]
correct_df = df[df['error_class'] == EDGE_CORRECT]

print(f"Total Correct Edges (Before downsampling): {len(correct_df)}")
print(f"Total Split Edges: {len(split_df)}")

if len(split_df) == 0:
    print("No split edges found. Cannot perform statistical test.")
    sys.exit(0)

# 4. Statistical downsampling for robust testing without overflow
if len(correct_df) > 50000:
    correct_df = correct_df.sample(n=50000, random_state=42)

print(f"Total Correct Edges (After downsampling): {len(correct_df)}")

split_dist = split_df['dist']
correct_dist = correct_df['dist']

print("\nSummary Statistics:")
print(f"Correct Edges - Mean Distance-to-Leaf: {correct_dist.mean():.4f} µm, Median: {correct_dist.median():.4f} µm")
print(f"Split Edges   - Mean Distance-to-Leaf: {split_dist.mean():.4f} µm, Median: {split_dist.median():.4f} µm")

# 5. Statistical Testing
stat, p = stats.mannwhitneyu(split_dist, correct_dist, alternative='less')
print(f"\nMann-Whitney U Test (Split < Correct): U={stat}, p-value={p:.4e}")

# 6. Visualization
plt.figure(figsize=(10, 6))
plt.hist(correct_dist, bins=50, density=True, alpha=0.5, label='Correct Edges', color='blue')
plt.hist(split_dist, bins=50, density=True, alpha=0.5, label='Split Edges', color='red')
plt.xlabel('Topological Distance to Leaf (µm)')
plt.ylabel('Density')
plt.title('Morphological Proxy: Distance-to-Leaf for Correct vs. Split Edges')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
