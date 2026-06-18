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
import re
import pickle
import numpy as np
import gc
import subprocess
import sys

# Helper to install missing packages quietly

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", package])

# Install all required packages sequentially to satisfy implicit dependencies
for pkg in ["psutil", "tensorstore", "networkx", "pandas", "statsmodels", "matplotlib", "scipy"]:
    try:
        __import__(pkg)
    except ImportError:
        install(pkg)

try:
    import agentic_neuron_proofreader
except ImportError:
    install("https://github.com/AllenInstitute/agentic-neuron-proofreader/archive/refs/heads/main.zip")
    import agentic_neuron_proofreader

import pandas as pd
import statsmodels.formula.api as smf
import scipy.stats as stats
import matplotlib.pyplot as plt
from collections import defaultdict

# Locate dataset cache files (files are one level above cwd)
files = glob.glob("../*_add.pkl") + glob.glob("../*/*_add.pkl")
files = list(set(files))

def brain_id_from_path(path):
    m = re.search(r"dataset_cache_(\d+)_mcl(\d+)_add\.pkl$", os.path.basename(path))
    return m.group(1) if m else "unknown"

data = []

for path in sorted(files):
    brain_id = brain_id_from_path(path)
    print(f"Processing Brain ID: {brain_id} from {os.path.basename(path)}...")
    
    # Disable garbage collection temporarily to speed up load time
    gc.disable()
    with open(path, "rb") as f:
        payload = pickle.load(f)
    gc.enable()
    
    # Extract required graphs and arrays
    gt = payload["gt_graph"]
    node_label = np.asarray(payload["gt_node_canonical_label"])
    edge_error = np.asarray(payload["gt_edge_error"])
    
    # Drop massive automated graph to free memory immediately
    payload.pop("fragments_graph", None)
    gc.collect()
    
    # Step 3: Count distinct segments overlapping each neuron to compute total splits
    neuron_segs = defaultdict(set)
    for n in gt.nodes:
        lab = int(node_label[n])
        if lab != 0:
            neuron_segs[gt.node_segment_id(n)].add(lab)
            
    neuron_splits = {nm: max(len(s) - 1, 0) for nm, s in neuron_segs.items()}
    
    # Variables to track length and edge classes per neuron
    neuron_edge_lengths = defaultdict(float)
    neuron_omit_counts = defaultdict(int)
    neuron_edge_counts = defaultdict(int)
    
    gt_edges = list(gt.edges)
    
    # Step 2 & 5: Calculate total cable length and omit error frequency per neuron
    for k, (u, v) in enumerate(gt_edges):
        neuron_id = gt.node_segment_id(u)
        
        xyz_u = gt.node_xyz[u]
        xyz_v = gt.node_xyz[v]
        dist = np.linalg.norm(xyz_u - xyz_v)
        
        neuron_edge_lengths[neuron_id] += dist
        neuron_edge_counts[neuron_id] += 1
        if edge_error[k] == 2:  # EDGE_OMIT
            neuron_omit_counts[neuron_id] += 1
            
    # Formulate metrics for regression
    for neuron_id in neuron_edge_counts.keys():
        length_um = neuron_edge_lengths[neuron_id]
        
        # Step 6: Filter out short fragments (< 50 um length)
        if length_um < 50:
            continue
            
        length_mm = length_um / 1000.0
        splits = neuron_splits.get(neuron_id, 0)
        splits_per_mm = splits / length_mm
        
        omit_count = neuron_omit_counts[neuron_id]
        total_edges = neuron_edge_counts[neuron_id]
        pct_omit = (omit_count / total_edges) * 100.0
        
        data.append({
            "neuron_id": neuron_id,
            "brain_id": "B" + str(brain_id),  # Prefix string to safely use as categorical variable
            "length_mm": length_mm,
            "splits_per_mm": splits_per_mm,
            "pct_omit": pct_omit
        })
        
    # Explicit memory cleanup per file
    del payload
    del gt
    del node_label
    del edge_error
    del gt_edges
    gc.collect()

df = pd.DataFrame(data)
print(f"\nSuccessfully processed {len(df)} neurons meeting the length criteria.")

if len(df) > 0:
    # Step 7: Compute rank/linear correlations
    pearson_r, pearson_p = stats.pearsonr(df['splits_per_mm'], df['pct_omit'])
    spearman_r, spearman_p = stats.spearmanr(df['splits_per_mm'], df['pct_omit'])
    print(f"\n--- Correlation Statistics ---")
    print(f"Pearson r : {pearson_r:.4f} (p-value: {pearson_p:.4e})")
    print(f"Spearman r: {spearman_r:.4f} (p-value: {spearman_p:.4e})")
    
    # Fit multiple linear regression controlling for brain_id
    print("\n--- Multiple Linear Regression (Predicting % Omitted Edges) ---")
    if df['brain_id'].nunique() > 1:
        model = smf.ols('pct_omit ~ splits_per_mm + C(brain_id)', data=df).fit()
    else:
        model = smf.ols('pct_omit ~ splits_per_mm', data=df).fit()
    
    print(model.summary())
    
    # Step 8: Generate visual deliverables
    fig, ax = plt.subplots(figsize=(10, 6))
    brains = df['brain_id'].unique()
    colors = plt.cm.get_cmap('tab10', max(len(brains), 10))
    
    # Scatter plot color-coded by brain ID
    for i, brain in enumerate(brains):
        subset = df[df['brain_id'] == brain]
        ax.scatter(subset['splits_per_mm'], subset['pct_omit'], 
                   label=f"Brain {brain.replace('B', '')}", 
                   color=colors(i), alpha=0.7, edgecolor='k')
    
    # Compute and plot global linear regression trend line
    m, b = np.polyfit(df['splits_per_mm'], df['pct_omit'], 1)
    x_vals = np.array([df['splits_per_mm'].min(), df['splits_per_mm'].max()])
    ax.plot(x_vals, m * x_vals + b, color='black', linewidth=2.5, linestyle='--', label='Global Trend Line')
    
    ax.set_title('Neuron-Level Defect Modalities: Splits per mm vs. % Omitted Edges', fontsize=14)
    ax.set_xlabel('Fragmentation Rate (Splits per mm)', fontsize=12)
    ax.set_ylabel('Omission Rate (% of Edges Omitted)', fontsize=12)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(title='Dataset / Element')
    plt.tight_layout()
    plt.show()
else:
    print("\nInsufficient valid neuron data to compute correlations or plot.")
