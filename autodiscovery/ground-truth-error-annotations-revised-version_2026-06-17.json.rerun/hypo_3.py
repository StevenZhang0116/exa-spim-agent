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

for pkg in ["psutil", "pandas", "tensorstore", "scipy", "networkx"]:
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
import scipy.stats as stats
import random
from scipy.spatial import KDTree

def main():
    path = None
    for root, dirs, files in os.walk(".."):
        for f in files:
            if f.endswith("_add.pkl"):
                path = os.path.join(root, f)
                break
        if path:
            break

    if not path:
        print("Dataset not found.")
        return
    
    with open(path, "rb") as f:
        payload = pickle.load(f)
    
    gt = payload["gt_graph"]
    fragments = payload["fragments_graph"]
    edge_error = np.asarray(payload["gt_edge_error"])
    merge_sites = payload["gt_merge_sites"]
    
    # 1. Extract xyz spatial coordinates from gt_merge_sites
    merge_xyz = [ms["xyz"] for ms in merge_sites]
    
    # 2. Define control set
    EDGE_CORRECT = 0
    node_edge_errors = {n: [] for n in gt.nodes}
    for (u, v), err in zip(list(gt.edges), edge_error):
        node_edge_errors[u].append(err)
        node_edge_errors[v].append(err)
        
    correct_nodes = []
    for n, errs in node_edge_errors.items():
        if len(errs) > 0 and all(e == EDGE_CORRECT for e in errs):
            correct_nodes.append(n)
            
    random.seed(42)
    if len(correct_nodes) > len(merge_xyz):
        control_nodes = random.sample(correct_nodes, len(merge_xyz))
    else:
        control_nodes = correct_nodes
        
    control_xyz = [gt.node_xyz[n] for n in control_nodes]
    
    # 3. Query fragments_graph.kdtree for nodes within 10 um radius
    try:
        tree = fragments.kdtree
    except AttributeError:
        tree = KDTree(fragments.node_xyz)
        
    radius = 10.0
    
    merge_counts = []
    for xyz in merge_xyz:
        indices = tree.query_ball_point(xyz, r=radius)
        if isinstance(indices, list):
            merge_counts.append(len(indices))
        else:
            merge_counts.append(indices.size)
        
    control_counts = []
    for xyz in control_xyz:
        indices = tree.query_ball_point(xyz, r=radius)
        if isinstance(indices, list):
            control_counts.append(len(indices))
        else:
            control_counts.append(indices.size)
            
    # 4. Perform Mann-Whitney U test
    u_stat, p_val = stats.mannwhitneyu(merge_counts, control_counts, alternative='greater')
    
    print(f"Total merge sites analyzed: {len(merge_xyz)}")
    print(f"Total control sites analyzed: {len(control_xyz)}")
    print(f"Average local node density (merge sites): {np.mean(merge_counts):.2f} nodes / 10 µm radius")
    print(f"Average local node density (control sites): {np.mean(control_counts):.2f} nodes / 10 µm radius")
    print(f"Mann-Whitney U statistic: {u_stat}")
    print(f"p-value: {p_val:.4e}")

if __name__ == "__main__":
    main()
