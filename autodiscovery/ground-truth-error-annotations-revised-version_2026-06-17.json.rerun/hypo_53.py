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

def install_deps():
    try:
        subprocess.check_call(["apt-get", "update", "-y"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.check_call(["apt-get", "install", "-y", "git"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "psutil", "pandas", "scipy", "networkx", "tensorstore"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "git+https://github.com/AllenInstitute/agentic-neuron-proofreader.git"])

install_deps()

import pickle
import numpy as np
import warnings
from collections import defaultdict
from scipy.stats import mannwhitneyu
import gc
import agentic_neuron_proofreader

warnings.filterwarnings('ignore')

def process_dataset(path):
    with open(path, "rb") as f:
        payload = pickle.load(f)
    
    gt = payload["gt_graph"]
    fg = payload["fragments_graph"]
    
    node_label = np.asarray(payload["gt_node_canonical_label"])
    edge_error = np.asarray(payload["gt_edge_error"])
    merge_labels = set(int(x) for x in payload["gt_merge_labels"])
    
    # Identify which error classes apply to each segment via GT edges
    edge_error_by_seg = defaultdict(set)
    edges = list(gt.edges)
    for k, (u, v) in enumerate(edges):
        seg_u = int(node_label[u])
        seg_v = int(node_label[v])
        err = int(edge_error[k])
        if seg_u != 0:
            edge_error_by_seg[seg_u].add(err)
        if seg_v != 0:
            edge_error_by_seg[seg_v].add(err)
            
    # Control segments: have at least one correct edge mapping and are not involved in merges
    control_segments = {seg for seg, errs in edge_error_by_seg.items() if 0 in errs and 3 not in errs} - merge_labels
    
    # Build mapping from swc_id (segment id) to component ids
    swc_to_comp_ids = defaultdict(list)
    for comp_id, swc in fg.component_id_to_swc_id.items():
        swc_to_comp_ids[int(float(swc))].append(comp_id)
        
    # Group fragments_graph nodes by component id
    comp_id_to_nodes = defaultdict(list)
    for n in fg.nodes:
        comp_id_to_nodes[int(fg.node_component_id[n])].append(n)
            
    target_segs = merge_labels.union(control_segments)
    m_dens = []
    c_dens = []
    
    # Calculate length, branches, and density for each targeted segment subgraph
    for seg_id in target_segs:
        comp_ids = swc_to_comp_ids.get(seg_id, [])
        nodes = []
        for c_id in comp_ids:
            nodes.extend(comp_id_to_nodes[c_id])
            
        if len(nodes) < 2:
            continue
            
        sub = fg.subgraph(nodes)
        cable_length = 0.0
        for u, v in sub.edges:
            cable_length += np.linalg.norm(fg.node_xyz[u] - fg.node_xyz[v])
            
        if cable_length > 0:
            # Branch nodes have degree > 2
            branch_nodes = sum(1 for n, d in sub.degree() if d > 2)
            # Density per 100 microns of cable
            density = (branch_nodes / cable_length) * 100
            
            if seg_id in merge_labels:
                m_dens.append(density)
            elif seg_id in control_segments:
                c_dens.append(density)
                
    return m_dens, c_dens

def main():
    pkl_paths = []
    for root, dirs, files in os.walk(".."):
        for f in files:
            if f.endswith("_add.pkl"):
                pkl_paths.append(os.path.join(root, f))
                
    pkl_paths = sorted(pkl_paths)
    if not pkl_paths:
        print("No dataset files found.")
        return
        
    merge_densities = []
    control_densities = []
    
    for path in pkl_paths:
        m, c = process_dataset(path)
        merge_densities.extend(m)
        control_densities.extend(c)
        gc.collect()  # Ensure cleanup between datasets to control memory footprint
        
    if merge_densities and control_densities:
        u_stat, p_val = mannwhitneyu(merge_densities, control_densities, alternative='two-sided')
        print(f"Analyzed {len(merge_densities)} merge segments and {len(control_densities)} control segments.")
        print(f"Average branch point density (merges): {np.mean(merge_densities):.4f} branches per 100 um")
        print(f"Average branch point density (controls): {np.mean(control_densities):.4f} branches per 100 um")
        print(f"Mann-Whitney U stat: {u_stat}")
        print(f"P-value: {p_val:.4e}")
    else:
        print(f"Not enough data computed. Merges found: {len(merge_densities)}, Controls found: {len(control_densities)}")

if __name__ == "__main__":
    main()
