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
import sys
import pickle
import math
import numpy as np
import subprocess
import gc
from pathlib import Path

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", package])

# Pre-install dependencies that might be missing in the environment
for pkg in ['scipy', 'pandas', 'psutil', 'networkx', 'tqdm', 'tensorstore', 'matplotlib']:
    try:
        __import__(pkg)
    except ImportError:
        install(pkg)

# Install agentic_neuron_proofreader if needed using zip archive to bypass git dependency
try:
    import agentic_neuron_proofreader
except ImportError:
    install("https://github.com/AllenInstitute/agentic-neuron-proofreader/archive/refs/heads/main.zip")
    import agentic_neuron_proofreader

from scipy.stats import mannwhitneyu

def get_extremity(adj, start_node, prev_node, k, node_xyz):
    curr = start_node
    prev = prev_node
    path_len = 0.0
    for _ in range(k):
        neighbors = [n for n in adj[curr] if n != prev]
        if not neighbors:
            break
        # In case of branching, just follow the first branch.
        nxt = neighbors[0]
        dx = node_xyz[nxt][0] - node_xyz[curr][0]
        dy = node_xyz[nxt][1] - node_xyz[curr][1]
        dz = node_xyz[nxt][2] - node_xyz[curr][2]
        path_len += math.sqrt(dx*dx + dy*dy + dz*dz)
        prev = curr
        curr = nxt
    return curr, path_len

def get_tortuosity(gt, edges, edge_errors, k=5):
    try:
        node_xyz = gt.node_xyz
        _ = node_xyz[next(iter(gt.nodes))]
    except (AttributeError, TypeError, IndexError):
        node_xyz = {n: gt.nodes[n].get('node_xyz', gt.nodes[n].get('xyz')) for n in gt.nodes}
        
    adj = {n: list(gt.neighbors(n)) for n in gt.nodes}
    
    correct_tortuosities = []
    split_tortuosities = []
    
    for idx, (u, v) in enumerate(edges):
        err = edge_errors[idx]
        if err != 0 and err != 1:
            continue
            
        u_curr, u_len = get_extremity(adj, u, v, k, node_xyz)
        v_curr, v_len = get_extremity(adj, v, u, k, node_xyz)
        
        dx_edge = node_xyz[v][0] - node_xyz[u][0]
        dy_edge = node_xyz[v][1] - node_xyz[u][1]
        dz_edge = node_xyz[v][2] - node_xyz[u][2]
        edge_len = math.sqrt(dx_edge*dx_edge + dy_edge*dy_edge + dz_edge*dz_edge)
        
        path_length = u_len + v_len + edge_len
        
        dx_end = node_xyz[v_curr][0] - node_xyz[u_curr][0]
        dy_end = node_xyz[v_curr][1] - node_xyz[u_curr][1]
        dz_end = node_xyz[v_curr][2] - node_xyz[u_curr][2]
        euclidean_dist = math.sqrt(dx_end*dx_end + dy_end*dy_end + dz_end*dz_end)
        
        if euclidean_dist > 1e-6:
            tortuosity = path_length / euclidean_dist
        else:
            tortuosity = 1.0
            
        if err == 0:
            correct_tortuosities.append(tortuosity)
        elif err == 1:
            split_tortuosities.append(tortuosity)
            
    return correct_tortuosities, split_tortuosities

def main():
    # Search for files dynamically
    files = list(Path("..").rglob("*_add.pkl"))
    if not files:
        files = list(Path(".").rglob("*_add.pkl"))
        
    if not files:
        print("No dataset files found matching *_add.pkl")
        sys.exit(1)
        
    all_correct = []
    all_split = []
    
    for fpath in files:
        print(f"Loading {fpath}...")
        with open(fpath, "rb") as f:
            payload = pickle.load(f)
        
        gt = payload["gt_graph"]
        edge_error = np.asarray(payload["gt_edge_error"])
        edges = list(gt.edges)
        
        c_tort, s_tort = get_tortuosity(gt, edges, edge_error, k=5)
        all_correct.extend(c_tort)
        all_split.extend(s_tort)
        
        # Free memory to prevent OOM errors with multiple large caches
        del payload
        del gt
        del edge_error
        del edges
        gc.collect()
        
    print(f"\nProcessed {len(all_correct)} correct edges and {len(all_split)} split edges.")
    
    if len(all_correct) == 0 or len(all_split) == 0:
        print("Not enough edges to compare.")
    else:
        median_correct = np.median(all_correct)
        median_split = np.median(all_split)
        
        print(f"Median Tortuosity for CORRECT edges: {median_correct:.6f}")
        print(f"Median Tortuosity for SPLIT edges:   {median_split:.6f}")
        print(f"Mean Tortuosity for CORRECT edges:   {np.mean(all_correct):.6f}")
        print(f"Mean Tortuosity for SPLIT edges:     {np.mean(all_split):.6f}")
        
        # Mann-Whitney U test (Alternative: split tortuosities are greater than correct ones)
        stat, p_val = mannwhitneyu(all_split, all_correct, alternative='greater')
        print(f"\nMann-Whitney U statistic: {stat}")
        print(f"p-value: {p_val:.4e}")
        
        if p_val < 0.05:
            print("Result: Statistically significant. Split edges occur on more tortuous segments.")
        else:
            print("Result: Not statistically significant.")

if __name__ == "__main__":
    main()
