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

import glob
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
import sys
import subprocess

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", package])

for pkg in ["pandas", "psutil", "tensorstore", "tqdm"]:
    try:
        __import__(pkg)
    except ImportError:
        install(pkg)

try:
    import agentic_neuron_proofreader
except ImportError:
    install("https://github.com/AllenInstitute/agentic-neuron-proofreader/archive/refs/heads/main.zip")
    import agentic_neuron_proofreader

EDGE_SPLIT = 1

branching_split = 0
branching_total = 0
linear_split = 0
linear_total = 0

# Load all dataset caches one level above the current directory
cache_paths = glob.glob("../dataset_cache_*_add.pkl")
if not cache_paths:
    # Fallback to current directory just in case
    cache_paths = glob.glob("dataset_cache_*_add.pkl")

for path in cache_paths:
    with open(path, "rb") as f:
        payload = pickle.load(f)
    
    gt = payload["gt_graph"]
    edge_error = np.asarray(payload["gt_edge_error"])
    
    edges = list(gt.edges)
    degrees = dict(gt.degree)
    
    for k, (u, v) in enumerate(edges):
        # Classify as branching if at least one endpoint has degree > 2
        is_branching = degrees[u] > 2 or degrees[v] > 2
        is_split = (edge_error[k] == EDGE_SPLIT)
        
        if is_branching:
            branching_total += 1
            if is_split:
                branching_split += 1
        else:
            linear_total += 1
            if is_split:
                linear_split += 1

# Prepare contingency table for Chi-square test
# Table format: [[Branching Split, Branching Not Split],
#                [Linear Split, Linear Not Split]]
branching_not_split = branching_total - branching_split
linear_not_split = linear_total - linear_split

contingency_table = np.array([
    [branching_split, branching_not_split],
    [linear_split, linear_not_split]
])

chi2, p_val, dof, expected = chi2_contingency(contingency_table)

branching_split_pct = (branching_split / branching_total * 100) if branching_total > 0 else 0
linear_split_pct = (linear_split / linear_total * 100) if linear_total > 0 else 0

print("=== Split Errors on Branching vs. Linear Edges ===")
print(f"Branching edges: {branching_total} total, {branching_split} split ({branching_split_pct:.2f}%)")
print(f"Linear edges   : {linear_total} total, {linear_split} split ({linear_split_pct:.2f}%)")
print(f"\nChi-square test statistic: {chi2:.4f}")
print(f"p-value                  : {p_val:.4e}")

# Generate Grouped Bar Chart
labels = ['Branching Edges', 'Linear Edges']
percentages = [branching_split_pct, linear_split_pct]

plt.figure(figsize=(7, 6))
bars = plt.bar(labels, percentages, color=['#d62728', '#1f77b4'], width=0.5)
plt.ylabel('Split Error Percentage (%)')
plt.title('Split Error Frequency: Branching vs. Linear Edges')

# Add percentage text above the bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + (max(percentages)*0.02), f'{yval:.2f}%', ha='center', va='bottom')

plt.ylim(0, max(percentages) * 1.2 if max(percentages) > 0 else 1)
plt.tight_layout()
plt.show()
