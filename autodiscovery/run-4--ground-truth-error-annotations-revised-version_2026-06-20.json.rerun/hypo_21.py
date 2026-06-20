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

def install_deps():
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "psutil", "pandas", "networkx", "scipy", "numpy", "tqdm", "tensorstore", "matplotlib"])

install_deps()

try:
    import agentic_neuron_proofreader
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "https://github.com/AllenInstitute/agentic-neuron-proofreader/archive/refs/heads/main.zip"])
    import agentic_neuron_proofreader

import pickle
import glob
import numpy as np
from scipy.stats import chi2_contingency

caches = glob.glob("dataset_cache_*_add.pkl")
if not caches:
    print("No caches found.")
    sys.exit()

z_dom_error = 0
z_dom_no_error = 0
xy_dom_error = 0
xy_dom_no_error = 0

for path in caches:
    with open(path, "rb") as f:
        payload = pickle.load(f)
        
    gt = payload["gt_graph"]
    edge_error = np.asarray(payload["gt_edge_error"])
    
    edges_list = list(gt.edges)
    if not edges_list:
        continue
    
    edges_arr = np.array(edges_list)
    
    # Extract coordinates
    xyz_u = np.array([gt.node_xyz[u] for u in edges_arr[:, 0]])
    xyz_v = np.array([gt.node_xyz[v] for v in edges_arr[:, 1]])
        
    diff = np.abs(xyz_u - xyz_v)
    
    dx = diff[:, 0]
    dy = diff[:, 1]
    dz = diff[:, 2]
    
    # Identify Z-dominant edges
    is_z_dom = (dz > dx) & (dz > dy)
    
    # Target errors are Split (1) or Omit (2)
    is_error = (edge_error == 1) | (edge_error == 2)
    
    z_dom_error += int(np.sum(is_z_dom & is_error))
    z_dom_no_error += int(np.sum(is_z_dom & ~is_error))
    
    xy_dom_error += int(np.sum(~is_z_dom & is_error))
    xy_dom_no_error += int(np.sum(~is_z_dom & ~is_error))

# Create contingency table
table = np.array([
    [z_dom_error, z_dom_no_error],
    [xy_dom_error, xy_dom_no_error]
])

z_total = z_dom_error + z_dom_no_error
xy_total = xy_dom_error + xy_dom_no_error

z_rate = z_dom_error / z_total if z_total > 0 else 0
xy_rate = xy_dom_error / xy_total if xy_total > 0 else 0

print("=== Edge Error Rates by Orientation ===")
print(f"Z-dominant edges : {z_total} total, {z_dom_error} errors (Error Rate: {z_rate:.2%})")
print(f"XY-dominant edges: {xy_total} total, {xy_dom_error} errors (Error Rate: {xy_rate:.2%})")

if z_total > 0 and xy_total > 0:
    chi2, p, dof, ex = chi2_contingency(table)
    print("\n=== Chi-Square Test ===")
    print(f"Chi2 Statistic: {chi2:.4f}")
    print(f"p-value       : {p:.4e}")
else:
    print("\n=== Chi-Square Test ===")
    print("Not enough data to perform Chi-Square test.")
