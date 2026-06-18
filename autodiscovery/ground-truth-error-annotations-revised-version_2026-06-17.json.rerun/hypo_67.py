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

import os
import sys
import urllib.request
import zipfile
import subprocess

# 1. Download and install to a writable target directory
zip_url = "https://github.com/AllenInstitute/agentic-neuron-proofreader/archive/refs/heads/main.zip"
zip_path = "/tmp/agentic.zip"
extract_path = "/tmp/agentic-neuron-proofreader-main"
lib_path = "/tmp/lib"

if not os.path.exists(lib_path):
    print("Downloading agentic-neuron-proofreader to /tmp/...", flush=True)
    urllib.request.urlretrieve(zip_url, zip_path)
    
    print("Extracting...", flush=True)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall("/tmp/")
        
    print("Installing to /tmp/lib...", flush=True)
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", 
        "--quiet", "--target", lib_path, "--no-cache-dir", 
        extract_path, "psutil", "tensorstore"
    ])

if lib_path not in sys.path:
    sys.path.insert(0, lib_path)

import agentic_neuron_proofreader
import pickle
import numpy as np
from scipy.spatial import KDTree
from scipy.stats import mannwhitneyu, gaussian_kde
import matplotlib.pyplot as plt

# 2. Load dataset
path = "../dataset_cache_789202_mcl100_add.pkl"
if not os.path.exists(path):
    path = "dataset_cache_789202_mcl100_add.pkl"
    if not os.path.exists(path):
        raise FileNotFoundError("Dataset not found in expected locations.")

print(f"Loading dataset from: {path}", flush=True)
with open(path, "rb") as f:
    payload = pickle.load(f)

fragments_graph = payload["fragments_graph"]
gt_graph = payload["gt_graph"]
node_label = np.asarray(payload["gt_node_canonical_label"])
edge_error = np.asarray(payload["gt_edge_error"])

print("Identifying GT split segment pairs...", flush=True)
gt_split_segment_pairs = set()
edges_list = list(gt_graph.edges)
for k, (u, v) in enumerate(edges_list):
    if edge_error[k] == 1:
        s1 = int(node_label[u])
        s2 = int(node_label[v])
        if s1 != 0 and s2 != 0 and s1 != s2:
            gt_split_segment_pairs.add(frozenset({s1, s2}))

print("Finding terminals in fragments_graph...", flush=True)
terminals = np.array([n for n, d in fragments_graph.degree() if d == 1], dtype=int)

print("Extracting terminal coordinates...", flush=True)
try:
    terminal_xyz = fragments_graph.node_xyz[terminals]
except (TypeError, AttributeError, IndexError):
    terminal_xyz = np.array([fragments_graph.nodes[n]['node_xyz'] for n in terminals])

print(f"Building KDTree for {len(terminals)} terminals...", flush=True)
kdtree = KDTree(terminal_xyz)

def get_comp(n):
    try:
        return int(fragments_graph.node_component_id[n])
    except (TypeError, AttributeError, IndexError):
        return int(fragments_graph.nodes[n]['node_component_id'])

def get_vector(n):
    try:
        neighbor = next(fragments_graph.neighbors(n))
        try:
            pos_n = fragments_graph.node_xyz[n]
            pos_nb = fragments_graph.node_xyz[neighbor]
        except (TypeError, AttributeError, IndexError):
            pos_n = fragments_graph.nodes[n]['node_xyz']
            pos_nb = fragments_graph.nodes[neighbor]['node_xyz']
            
        v = pos_nb - pos_n
        norm = np.linalg.norm(v)
        if norm > 0:
            return v / norm
        return None
    except StopIteration:
        return None

print("Identifying exact split pairs from GT endpoints...", flush=True)
split_pairs_exact = set()

for k, (u, v) in enumerate(edges_list):
    if edge_error[k] == 1:
        try:
            pos_u = gt_graph.node_xyz[u]
            pos_v = gt_graph.node_xyz[v]
        except (TypeError, AttributeError, IndexError):
            pos_u = gt_graph.nodes[u]['node_xyz']
            pos_v = gt_graph.nodes[v]['node_xyz']
            
        dist_u, idx_u = kdtree.query(pos_u)
        dist_v, idx_v = kdtree.query(pos_v)
        
        if dist_u <= 15.0 and dist_v <= 15.0:
            n1 = terminals[idx_u]
            n2 = terminals[idx_v]
            if n1 != n2:
                s1 = get_comp(n1)
                s2 = get_comp(n2)
                if s1 != s2:
                    split_pairs_exact.add(frozenset({n1, n2}))

split_cos_sims = []
for pair_set in split_pairs_exact:
    n1, n2 = tuple(pair_set)
    v1 = get_vector(n1)
    v2 = get_vector(n2)
    
    if v1 is not None and v2 is not None:
        cos_sim = np.dot(v1, v2)
        split_cos_sims.append(cos_sim)

print(f"Computed cosine similarity for {len(split_cos_sims)} split endpoint pairs.", flush=True)

print("Querying control pairs within 15um using targeted sampling...", flush=True)
control_cos_sims = []
MAX_CONTROLS = 10000

np.random.seed(42)
sampled_indices = np.random.choice(len(terminals), min(50000, len(terminals)), replace=False)

for i in sampled_indices:
    if len(control_cos_sims) >= MAX_CONTROLS:
        break
        
    neighbors_idx = kdtree.query_ball_point(terminal_xyz[i], r=15.0)
    
    for j in neighbors_idx:
        if i >= j:
            continue
            
        n1 = terminals[i]
        n2 = terminals[j]
        
        s1 = get_comp(n1)
        s2 = get_comp(n2)
        
        if s1 == s2:
            continue
            
        if frozenset({s1, s2}) in gt_split_segment_pairs:
            continue
            
        v1 = get_vector(n1)
        v2 = get_vector(n2)
        
        if v1 is not None and v2 is not None:
            cos_sim = np.dot(v1, v2)
            control_cos_sims.append(cos_sim)
            
        if len(control_cos_sims) >= MAX_CONTROLS:
            break

print(f"Computed cosine similarity for {len(control_cos_sims)} control endpoint pairs.", flush=True)

if len(split_cos_sims) > 0 and len(control_cos_sims) > 0:
    stat, pval = mannwhitneyu(split_cos_sims, control_cos_sims, alternative='two-sided')
    print(f"\nResults:")
    print(f"Mann-Whitney U statistic: {stat}")
    print(f"p-value: {pval:.2e}")
    print(f"Mean split cos sim: {np.mean(split_cos_sims):.4f} (std: {np.std(split_cos_sims):.4f})")
    print(f"Mean control cos sim: {np.mean(control_cos_sims):.4f} (std: {np.std(control_cos_sims):.4f})")

    plt.figure(figsize=(10, 6))
    
    xs = np.linspace(-1.2, 1.2, 300)
    
    if len(split_cos_sims) > 1 and np.std(split_cos_sims) > 1e-5:
        kde1 = gaussian_kde(split_cos_sims)
        plt.plot(xs, kde1(xs), label=f'Split Endpoints (n={len(split_cos_sims)})', color='blue', linewidth=2)
        plt.fill_between(xs, kde1(xs), alpha=0.3, color='blue')
        
    if len(control_cos_sims) > 1 and np.std(control_cos_sims) > 1e-5:
        kde2 = gaussian_kde(control_cos_sims)
        plt.plot(xs, kde2(xs), label=f'Control Endpoints (n={len(control_cos_sims)})', color='orange', linewidth=2)
        plt.fill_between(xs, kde2(xs), alpha=0.3, color='orange')

    plt.title('Collinearity (Cosine Similarity) of Opposing Fragment Endpoints')
    plt.xlabel('Cosine Similarity (Anti-parallel = -1, Parallel = +1)')
    plt.ylabel('Density')
    plt.xlim(-1.1, 1.1)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_path = '/tmp/collinearity_plot.png'
    try:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}", flush=True)
    except Exception as e:
        print(f"Failed to save plot: {e}", flush=True)
else:
    print("Not enough data points to compute statistics or plot density.", flush=True)
