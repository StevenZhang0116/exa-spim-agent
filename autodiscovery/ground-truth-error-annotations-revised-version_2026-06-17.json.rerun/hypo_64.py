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

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", package])

try:
    import psutil
except ImportError:
    install("psutil")

try:
    import tensorstore
except ImportError:
    install("tensorstore")

try:
    import agentic_neuron_proofreader
except ImportError:
    # Install from zip to avoid requiring git
    install("https://github.com/AllenInstitute/agentic-neuron-proofreader/archive/refs/heads/main.zip")
    import agentic_neuron_proofreader

import os
import glob
import pickle
import numpy as np
import networkx as nx
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt

def get_edge_classifications(gt, edge_error):
    edges_list = list(gt.edges)
    edge_to_error = {frozenset([u, v]): edge_error[i] for i, (u, v) in enumerate(edges_list)}
    
    deg = dict(gt.degree())
    visited_edges = set()
    
    t_omit = 0
    t_tot = 0
    i_omit = 0
    i_tot = 0
    
    for u, v in edges_list:
        edge_fs = frozenset([u, v])
        if edge_fs in visited_edges:
            continue
            
        path_edges = [edge_fs]
        visited_edges.add(edge_fs)
        visited_nodes_in_path = {u, v}
        
        # Walk from v
        curr = v
        prev = u
        while deg[curr] == 2:
            neighbors = list(gt.neighbors(curr))
            next_node = neighbors[0] if neighbors[0] != prev else neighbors[1]
            if next_node in visited_nodes_in_path:
                break
            visited_nodes_in_path.add(next_node)
            
            next_edge_fs = frozenset([curr, next_node])
            path_edges.append(next_edge_fs)
            visited_edges.add(next_edge_fs)
            
            prev = curr
            curr = next_node
        end1 = curr
            
        # Walk from u
        curr = u
        prev = v
        while deg[curr] == 2:
            neighbors = list(gt.neighbors(curr))
            next_node = neighbors[0] if neighbors[0] != prev else neighbors[1]
            if next_node in visited_nodes_in_path:
                break
            visited_nodes_in_path.add(next_node)
            
            next_edge_fs = frozenset([curr, next_node])
            path_edges.append(next_edge_fs)
            visited_edges.add(next_edge_fs)
            
            prev = curr
            curr = next_node
        end2 = curr
            
        is_terminal = (deg[end1] == 1) or (deg[end2] == 1)
        
        for e in path_edges:
            is_omit = (edge_to_error[e] == 2) # EDGE_OMIT == 2
            
            if is_terminal:
                t_tot += 1
                if is_omit:
                    t_omit += 1
            else:
                i_tot += 1
                if is_omit:
                    i_omit += 1
                    
    return t_omit, t_tot, i_omit, i_tot

def main():
    search_paths = [
        "../*_add.pkl",
        "../cache/*_add.pkl",
        "*_add.pkl",
        "cache/*_add.pkl"
    ]
    files = []
    for pattern in search_paths:
        files.extend(glob.glob(pattern))
    files = sorted(list(set(files)))
    
    if not files:
        print("No cache files found.")
        return
        
    terminal_omit_total = 0
    terminal_count_total = 0
    internal_omit_total = 0
    internal_count_total = 0
    
    for path in files:
        with open(path, "rb") as f:
            payload = pickle.load(f)
        
        gt = payload["gt_graph"]
        edge_error = np.asarray(payload["gt_edge_error"])
        
        if len(edge_error) != gt.number_of_edges():
            print(f"Warning: Edge count mismatch in {path}.")
            continue
            
        t_omit, t_tot, i_omit, i_tot = get_edge_classifications(gt, edge_error)
        
        terminal_omit_total += t_omit
        terminal_count_total += t_tot
        internal_omit_total += i_omit
        internal_count_total += i_tot

    if terminal_count_total == 0 or internal_count_total == 0:
        print("Not enough edges found for Terminal/Internal classification.")
        return

    A = terminal_omit_total
    B = terminal_count_total - terminal_omit_total
    C = internal_omit_total
    D = internal_count_total - internal_omit_total

    table = [[A, B], [C, D]]
    chi2, p_val, dof, expected = chi2_contingency(table)
    
    t_rate = (A / terminal_count_total) * 100
    i_rate = (C / internal_count_total) * 100
    
    print("\n--- Edge Distribution ---")
    print(f"Terminal edges total: {terminal_count_total}")
    print(f"Internal edges total: {internal_count_total}")

    print("\n--- Summary Statistics ---")
    print(f"Terminal OMIT rate: {t_rate:.2f}% ({A}/{terminal_count_total})")
    print(f"Internal OMIT rate: {i_rate:.2f}% ({C}/{internal_count_total})")
    print(f"Chi-square statistic: {chi2:.4f}")
    print(f"p-value: {p_val:.4e}")

    labels = ['Terminal', 'Internal']
    rates = [t_rate, i_rate]

    plt.figure(figsize=(7, 6))
    bars = plt.bar(labels, rates, color=['#1f77b4', '#ff7f0e'], edgecolor='black')
    plt.ylabel('OMIT Error Rate (%)', fontsize=12)
    plt.title('OMIT Error Rates: Terminal vs Internal Branches', fontsize=14)
    plt.ylim(0, max(rates) * 1.2 if max(rates) > 0 else 10)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + (max(rates)*0.02 if max(rates) > 0 else 0.5), 
                 f"{yval:.2f}%", ha='center', va='bottom', fontweight='bold', fontsize=11)
                 
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
