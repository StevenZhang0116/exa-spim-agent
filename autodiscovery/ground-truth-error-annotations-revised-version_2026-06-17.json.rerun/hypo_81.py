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
import os

def setup_env():
    packages = ["psutil", "pandas", "tensorstore", "scipy", "networkx", "https://github.com/AllenInstitute/agentic-neuron-proofreader/archive/refs/heads/main.zip"]
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet"] + packages)

setup_env()

import agentic_neuron_proofreader
import pickle
import numpy as np
from scipy.spatial import cKDTree
from scipy import stats
import random

def main():
    random.seed(42)
    np.random.seed(42)

    filename = "dataset_cache_789202_mcl100_add.pkl"
    # Default to one level above the current working directory as per instruction
    filepath = os.path.join("..", filename)
    
    if not os.path.exists(filepath):
        # Fallback to system search if the directory structure differs
        filepath_output = subprocess.getoutput(f'find / -name "{filename}" 2>/dev/null').strip().split('\n')[0]
        if filepath_output and os.path.exists(filepath_output):
            filepath = filepath_output
        else:
            filepath_output = subprocess.getoutput('find / -name "dataset_cache_*_add.pkl" 2>/dev/null').strip().split('\n')[0]
            if filepath_output and os.path.exists(filepath_output):
                filepath = filepath_output

    print(f"Loading dataset from: {filepath}")

    with open(filepath, 'rb') as f:
        payload = pickle.load(f)

    fg = payload['fragments_graph']
    gt = payload['gt_graph']
    gt_merge_sites = payload['gt_merge_sites']
    gt_edge_error = np.asarray(payload['gt_edge_error'])

    # 1. Build KD-tree over fragment nodes
    valid_nodes = list(fg.nodes)
    if hasattr(fg, 'node_xyz') and isinstance(fg.node_xyz, np.ndarray):
        try:
            fg_coords = fg.node_xyz[valid_nodes]
        except Exception:
            fg_coords = np.array([fg.node_xyz[n] for n in valid_nodes])
    elif hasattr(fg, 'node_xyz') and fg.node_xyz is not None:
        fg_coords = np.array([fg.node_xyz[n] for n in valid_nodes])
    else:
        fg_coords = np.array([fg.nodes[n]['xyz'] for n in valid_nodes])

    tree = cKDTree(fg_coords)

    # 2. Local density for merge locations
    merge_coords = [np.array(site['xyz']) for site in gt_merge_sites]
    radius = 10.0
    volume = (4.0 / 3.0) * np.pi * (radius ** 3)

    merge_counts = tree.query_ball_point(merge_coords, r=radius)
    merge_densities = [len(neighbors) / volume for neighbors in merge_counts]

    # 3. Local density for correctly reconstructed GT edges
    edges = list(gt.edges)
    correct_edges = [edges[i] for i, err in enumerate(gt_edge_error) if err == 0]

    edge_lengths = []
    valid_correct_edges = []
    for u, v in correct_edges:
        if hasattr(gt, 'node_xyz') and gt.node_xyz is not None:
            p1 = np.array(gt.node_xyz[u])
            p2 = np.array(gt.node_xyz[v])
        else:
            p1 = np.array(gt.nodes[u]['xyz'])
            p2 = np.array(gt.nodes[v]['xyz'])
        
        dist = np.linalg.norm(p1 - p2)
        edge_lengths.append(dist)
        valid_correct_edges.append((p1, p2))

    total_len = sum(edge_lengths)
    probs = [l / total_len for l in edge_lengths]

    control_coords = []
    # Sample randomly, weighting by physical edge length to avoid bias
    chosen_edges_idx = random.choices(range(len(valid_correct_edges)), weights=probs, k=len(merge_coords))

    for idx in chosen_edges_idx:
        p1, p2 = valid_correct_edges[idx]
        t = random.uniform(0, 1)
        p = p1 * t + p2 * (1 - t)
        control_coords.append(p)

    control_counts = tree.query_ball_point(control_coords, r=radius)
    control_densities = [len(neighbors) / volume for neighbors in control_counts]

    # 4. Compare distributions
    t_stat, p_val = stats.ttest_ind(merge_densities, control_densities, equal_var=False)

    print("\n=== Local Graph Density Analysis ===")
    print(f"Number of merge sites evaluated: {len(merge_coords)}")
    print(f"Number of control regions evaluated: {len(control_coords)}")
    print(f"Local neighborhood radius: {radius} um (Volume: {volume:.2f} um^3)")
    print(f"Average fragment node density at merge sites: {np.mean(merge_densities):.6f} nodes/um^3")
    print(f"Average fragment node density at correct regions: {np.mean(control_densities):.6f} nodes/um^3")
    print(f"Independent t-test (Welch's) - t-statistic: {t_stat:.4f}, p-value: {p_val:.4e}")

    if p_val < 0.05:
        print("Conclusion: Significant difference in local fragment density between merge sites and correctly reconstructed control regions.")
    else:
        print("Conclusion: No significant difference in local fragment density between merge sites and correctly reconstructed control regions.")

if __name__ == "__main__":
    main()
