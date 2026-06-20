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
import sys
import subprocess
import glob
import os
import urllib.request
import zipfile

def install_and_import(package):
    try:
        __import__(package)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", package])

install_and_import('networkx')
install_and_import('scipy')
install_and_import('matplotlib')
install_and_import('numpy')
install_and_import('psutil')
install_and_import('pandas')
install_and_import('tqdm')
install_and_import('tensorstore')

try:
    import agentic_neuron_proofreader
except ImportError:
    temp_dir = "/tmp"
    repo_dir = os.path.join(temp_dir, "agentic-neuron-proofreader-main")
    if not os.path.exists(repo_dir):
        zip_path = os.path.join(temp_dir, "main.zip")
        urllib.request.urlretrieve("https://github.com/AllenInstitute/agentic-neuron-proofreader/archive/refs/heads/main.zip", zip_path)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", repo_dir])
    import agentic_neuron_proofreader

import pickle
import numpy as np
import networkx as nx
import scipy.stats as stats
import matplotlib.pyplot as plt

def main():
    dataset_paths = glob.glob("../dataset_cache_*_add.pkl")
    if not dataset_paths:
        dataset_paths = glob.glob("*_add.pkl")
        
    terminal_omit = 0
    terminal_non = 0
    internal_omit = 0
    internal_non = 0

    EDGE_OMIT = 2

    for path in dataset_paths:
        with open(path, "rb") as f:
            payload = pickle.load(f)

        gt_graph = payload["gt_graph"]
        edge_error = np.asarray(payload["gt_edge_error"])

        terminal_edges = set()

        # Identify terminal branches
        for node in gt_graph.nodes:
            if gt_graph.degree(node) == 1:
                curr = node
                prev = None
                # Traverse inwards until hitting a branch point (degree >= 3)
                while True:
                    if gt_graph.degree(curr) >= 3:
                        break
                        
                    neighbors = list(gt_graph.neighbors(curr))
                    if prev is not None and prev in neighbors:
                        neighbors.remove(prev)
                    
                    if not neighbors:
                        break
                        
                    nxt = neighbors[0]
                    # Store edge as frozenset so (u, v) == (v, u)
                    terminal_edges.add(frozenset([curr, nxt]))
                    
                    prev = curr
                    curr = nxt

        edges = list(gt_graph.edges)
        
        # Cross-tabulate edge types (terminal vs internal) with edge error status (Omit vs Non-Omit)
        for k, (u, v) in enumerate(edges):
            is_term = frozenset([u, v]) in terminal_edges
            is_omit = (edge_error[k] == EDGE_OMIT)
            
            if is_term:
                if is_omit:
                    terminal_omit += 1
                else:
                    terminal_non += 1
            else:
                if is_omit:
                    internal_omit += 1
                else:
                    internal_non += 1

    print("=== Omit Errors by Branch Location ===")
    print(f"Terminal edges: Omit={terminal_omit}, Non-Omit={terminal_non}")
    print(f"Internal edges: Omit={internal_omit}, Non-Omit={internal_non}")

    contingency_table = [[terminal_omit, terminal_non],
                         [internal_omit, internal_non]]

    # Perform chi-square contingency test
    chi2, p_val, dof, expected = stats.chi2_contingency(contingency_table)

    print("\n=== Chi-Square Contingency Test ===")
    print(f"Chi-square statistic: {chi2:.4f}")
    print(f"p-value: {p_val:.4e}")

    term_pct = 100 * terminal_omit / (terminal_omit + terminal_non) if (terminal_omit + terminal_non) > 0 else 0
    int_pct = 100 * internal_omit / (internal_omit + internal_non) if (internal_omit + internal_non) > 0 else 0

    print(f"\nTerminal Omit Percentage: {term_pct:.2f}%")
    print(f"Internal Omit Percentage: {int_pct:.2f}%")

    labels = ['Terminal Edges', 'Internal Edges']
    omit_pct = [term_pct, int_pct]

    plt.figure(figsize=(8, 6))
    bars = plt.bar(labels, omit_pct, color=['#FF9999', '#99CCFF'])
    plt.ylabel('% Omit Errors')
    plt.title('Percentage of Omit Errors: Terminal vs Internal Edges')
    
    max_y = max(omit_pct)
    plt.ylim(0, max_y * 1.2 if max_y > 0 else 100)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + (max_y * 0.02), f'{yval:.2f}%', ha='center', va='bottom')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
