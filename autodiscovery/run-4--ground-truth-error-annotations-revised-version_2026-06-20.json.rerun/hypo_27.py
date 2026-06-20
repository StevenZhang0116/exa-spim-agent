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
import pickle
import numpy as np
import pandas as pd
import warnings
import scipy.stats
import re

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", package])

# Ensure required packages are installed
packages_to_install = ["psutil", "tqdm", "statsmodels", "tensorstore", "matplotlib"]
for pkg in packages_to_install:
    try:
        __import__(pkg)
    except ImportError:
        install(pkg)

import statsmodels.api as sm
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

try:
    import agentic_neuron_proofreader
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "https://github.com/AllenInstitute/agentic-neuron-proofreader/archive/refs/heads/main.zip"])
    import agentic_neuron_proofreader

def brain_id_from_path(path):
    m = re.search(r"dataset_cache_(\d+)_mcl(\d+)_add\.pkl$", os.path.basename(path))
    return m.group(1) if m else os.path.basename(path)

# 1. Load all available _add.pkl caches
cache_files = glob.glob("./dataset_cache_*_add.pkl")
if not cache_files:
    cache_files = glob.glob("../data/dataset_cache_*_add.pkl")
if not cache_files:
    cache_files = glob.glob("../*/*_add.pkl")

if not cache_files:
    print("No cache files found.")
    sys.exit(0)

edges_data = []

print("Loading dataset caches and extracting edges...")
for path in sorted(cache_files):
    brain_id = brain_id_from_path(path)
    with open(path, "rb") as f:
        payload = pickle.load(f)
    
    gt = payload["gt_graph"]
    edge_error = np.asarray(payload["gt_edge_error"])
    node_xyz = gt.node_xyz
    
    edges = list(gt.edges)
    if not edges:
        continue
        
    # Vectorized computation of edge geometries
    u_nodes = np.array([e[0] for e in edges])
    v_nodes = np.array([e[1] for e in edges])
    
    xyz_u = node_xyz[u_nodes]
    xyz_v = node_xyz[v_nodes]
    
    # Euclidean length and absolute Z-difference
    dz = np.abs(xyz_u[:, 2] - xyz_v[:, 2])
    lengths = np.linalg.norm(xyz_u - xyz_v, axis=1)
    
    # Filter: length > 0, and not 'merged' (which the plan excludes from the binary split/omit classification)
    # 0 = correct, 1 = split, 2 = omit, 3 = merged
    valid_mask = (lengths > 1e-6) & (edge_error != 3)
    
    valid_u = u_nodes[valid_mask]
    valid_errs = edge_error[valid_mask]
    valid_lengths = lengths[valid_mask]
    valid_dz = dz[valid_mask]
    
    # 1 if split (1) or omit (2), 0 if correct (0)
    is_err = np.isin(valid_errs, [1, 2]).astype(int)
    z_align = valid_dz / valid_lengths
    
    # GT neuron names
    neuron_ids = [gt.node_segment_id(u) for u in valid_u]
    
    df_brain = pd.DataFrame({
        "is_error": is_err,
        "z_alignment": z_align,
        "edge_length": valid_lengths,
        "neuron_id": neuron_ids,
        "brain_id": brain_id
    })
    edges_data.append(df_brain)
    
    # To save memory, let garbage collection clean up
    del payload, gt, edge_error, node_xyz

df = pd.concat(edges_data, ignore_index=True)
print(f"Total valid edges loaded: {len(df)}")
print(f"Total errors (splits/omits): {df['is_error'].sum()} ({df['is_error'].mean()*100:.2f}%)\n")

print("Descriptive Statistics for Z-alignment (0=XY-plane, 1=Z-axis):")
print(f"  Mean Z-alignment (all edges): {df['z_alignment'].mean():.4f}")
if df['is_error'].sum() > 0:
    print(f"  Mean Z-alignment (errors): {df[df['is_error']==1]['z_alignment'].mean():.4f}")
if (len(df) - df['is_error'].sum()) > 0:
    print(f"  Mean Z-alignment (correct): {df[df['is_error']==0]['z_alignment'].mean():.4f}\n")

# Standardize continuous variables for model convergence
df['z_align_std'] = (df['z_alignment'] - df['z_alignment'].mean()) / df['z_alignment'].std()
df['length_std'] = (df['edge_length'] - df['edge_length'].mean()) / df['edge_length'].std()

# Limit dataset size to prevent excessive computation time with Variational Bayes
max_samples = 20000
if len(df) > max_samples:
    print(f"Sampling down to {max_samples} edges for timely convergence...")
    errs_df = df[df["is_error"] == 1]
    ok_df = df[df["is_error"] == 0]
    
    err_frac = len(errs_df) / len(df)
    n_errs = int(max_samples * err_frac)
    n_ok = max_samples - n_errs
    
    sampled_errs = errs_df.sample(n=min(n_errs, len(errs_df)), random_state=42)
    sampled_ok = ok_df.sample(n=min(n_ok, len(ok_df)), random_state=42)
    
    df_fit = pd.concat([sampled_errs, sampled_ok]).sample(frac=1, random_state=42).reset_index(drop=True)
else:
    df_fit = df

print("\nFitting Logistic Regression Model...")
vc_formulas = {}
if df_fit['brain_id'].nunique() > 1:
    vc_formulas["brain"] = "0 + C(brain_id)"
if df_fit['neuron_id'].nunique() > 1:
    vc_formulas["neuron"] = "0 + C(neuron_id)"

is_mixed = len(vc_formulas) > 0

def fit_standard_glm():
    model = sm.GLM.from_formula("is_error ~ z_align_std + length_std", family=sm.families.Binomial(), data=df_fit)
    result = model.fit()
    print(result.summary())
    coef_z = result.params.get("z_align_std", 0)
    p_val_z = result.pvalues.get("z_align_std", 1)
    sd_z = result.bse.get("z_align_std", 0)
    return coef_z, sd_z, p_val_z

coef_z, sd_z, p_val_z = None, None, None

if not is_mixed:
    print("Not enough grouping levels for mixed effects. Falling back to standard GLM.")
    coef_z, sd_z, p_val_z = fit_standard_glm()
else:
    model = BinomialBayesMixedGLM.from_formula("is_error ~ z_align_std + length_std", vc_formulas=vc_formulas, data=df_fit)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            result = model.fit_vb()
            summary_str = str(result.summary())
            print(summary_str)
            
            # Parse summary
            match = re.search(r"z_align_std\s+[A-Za-z]+\s+([-\.\d]+(?:[eE][-\+]?\d+)?)\s+([-\.\d]+(?:[eE][-\+]?\d+)?)", summary_str)
            if match:
                coef_z = float(match.group(1))
                sd_z = float(match.group(2))
                if sd_z > 0:
                    z_stat = coef_z / sd_z
                    p_val_z = 2 * (1 - scipy.stats.norm.cdf(abs(z_stat)))
                else:
                    p_val_z = 1.0
            else:
                print("Could not parse mixed-effects summary. Falling back to GLM.")
                coef_z, sd_z, p_val_z = fit_standard_glm()
        except Exception as e:
            print(f"Mixed-effects GLM failed: {e}\nFalling back to standard GLM.")
            coef_z, sd_z, p_val_z = fit_standard_glm()

if coef_z is not None:
    orig_std_z = df['z_alignment'].std()
    coef_orig = coef_z / orig_std_z
    odds_ratio = np.exp(coef_orig)
    
    print("\n=== Model Findings ===")
    print(f"Standardized Coefficient (Z-alignment): {coef_z:.4f} (p-value = {p_val_z:.4g})")
    print(f"Original Scale Coefficient (per 1 unit [0 to 1]): {coef_orig:.4f}")
    print(f"Odds Ratio (Z-alignment perfectly along Z vs XY-plane): {odds_ratio:.4f}")
    
    if coef_z > 0 and p_val_z < 0.05:
        print("\nConclusion: Z-axis alignment significantly INCREASES the risk of topological errors (splits/omissions). Highly anisotropic segments are harder to segment correctly.")
    elif coef_z < 0 and p_val_z < 0.05:
        print("\nConclusion: Z-axis alignment significantly DECREASES the risk of topological errors. (This would be unexpected given typical resolution anisotropy).")
    else:
        print("\nConclusion: Z-axis alignment does NOT significantly affect the risk of topological errors based on this dataset.")
