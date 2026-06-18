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

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", package])

# Ensure dependencies are installed
dependencies = ["psutil", "tensorstore", "networkx", "scipy", "numpy", "pandas", "tqdm", "matplotlib"]
for pkg in dependencies:
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
import matplotlib.pyplot as plt

# Load dataset from the current directory
data_path = "dataset_cache_789202_mcl100_add.pkl"
with open(data_path, "rb") as f:
    payload = pickle.load(f)

gt = payload["gt_graph"]
fragments = payload["fragments_graph"]
merge_sites = payload["gt_merge_sites"]
node_label = np.asarray(payload["gt_node_canonical_label"])
merge_labels = set(int(x) for x in payload["gt_merge_labels"])

# 1. Extract the xyz coordinates of all merge sites
merge_coords = [site["xyz"] for site in merge_sites]

# 2. Randomly sample an equal number of nodes from gt_graph not associated with merge errors
non_merge_coords = []
for n in gt.nodes:
    if int(node_label[n]) not in merge_labels:
        non_merge_coords.append(gt.node_xyz[n])

random.seed(42)
if len(non_merge_coords) > len(merge_coords):
    sampled_non_merge_coords = random.sample(non_merge_coords, len(merge_coords))
else:
    sampled_non_merge_coords = non_merge_coords

# 3. Query the fragments_graph's KDTree for fragment nodes within 15 µm radius
radius = 15

if not merge_coords:
    print("No merge sites found in the dataset.")
else:
    merge_counts = []
    for coord in merge_coords:
        indices = fragments.kdtree.query_ball_point(coord, r=radius)
        merge_counts.append(len(indices))

    non_merge_counts = []
    for coord in sampled_non_merge_coords:
        indices = fragments.kdtree.query_ball_point(coord, r=radius)
        non_merge_counts.append(len(indices))

    # 4. Perform a statistical test (Mann-Whitney U Test)
    statistic, p_value = stats.mannwhitneyu(merge_counts, non_merge_counts)

    mean_merge_density = np.mean(merge_counts)
    mean_non_merge_density = np.mean(non_merge_counts)

    # Output deliverables
    print(f"Number of merge sites evaluated: {len(merge_counts)}")
    print(f"Mean fragment node count within {radius} µm for merge sites: {mean_merge_density:.2f}")
    print(f"Mean fragment node count within {radius} µm for control sites: {mean_non_merge_density:.2f}")
    print(f"Mann-Whitney U Test: statistic={statistic}, p-value={p_value:.2e}")

    # 5. Boxplot for density comparison
    plt.figure(figsize=(8, 6))
    plt.boxplot([merge_counts, non_merge_counts], labels=['Merge Sites', 'Control Sites'])
    plt.ylabel(f'Fragment Node Count (within {radius} µm)')
    plt.title('Local Fragment Density: Merge Sites vs Control Sites')
    plt.show()
