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

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", package])

# Install required dependencies
for pkg in ["psutil", "pandas", "networkx", "scipy", "tensorstore", "matplotlib"]:
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
import networkx as nx
import glob
from scipy.stats import mannwhitneyu

def main():
    files = glob.glob("*_add.pkl")
    if not files:
        print("No dataset files found.")
        return

    bridged_lengths = []
    broken_lengths = []

    for file in files:
        with open(file, "rb") as f:
            payload = pickle.load(f)
            
        gt = payload["gt_graph"]
        edge_error = np.asarray(payload["gt_edge_error"])
        node_label = np.asarray(payload["gt_node_canonical_label"])
        
        EDGE_OMIT = 2
        edges = list(gt.edges)
        
        # Filter for omit edges
        omit_edges = [e for i, e in enumerate(edges) if edge_error[i] == EDGE_OMIT]
        
        # Create an induced subgraph of omit edges
        omit_subgraph = nx.Graph()
        omit_subgraph.add_edges_from(omit_edges)
        
        # Extract continuous omit paths as connected components
        components = list(nx.connected_components(omit_subgraph))
        
        for comp in components:
            comp_subgraph = omit_subgraph.subgraph(comp)
            
            # Calculate the total physical length of the omit path
            path_length = 0.0
            for u, v in comp_subgraph.edges:
                p1 = gt.node_xyz[u]
                p2 = gt.node_xyz[v]
                dist = np.linalg.norm(p1 - p2)
                path_length += dist
                
            # Identify the non-omit GT nodes immediately preceding and succeeding the omit path
            flanking_nodes = set()
            for node in comp:
                for neighbor in gt.neighbors(node):
                    if neighbor not in comp:
                        flanking_nodes.add(neighbor)
            
            # Classify path as bridged or broken based on its flanking nodes
            if len(flanking_nodes) >= 2:
                segs = [int(node_label[n]) for n in flanking_nodes]
                # Bridged if all flanking segment IDs are identical and non-zero
                if len(set(segs)) == 1 and segs[0] != 0:
                    bridged_lengths.append(path_length)
                else:
                    broken_lengths.append(path_length)
            else:
                # If there are <2 flanking nodes, it represents a true termination or completely omitted structure
                broken_lengths.append(path_length)
                
    print(f"Total Bridged Omit Paths: {len(bridged_lengths)}")
    if bridged_lengths:
        print(f"Mean Length of Bridged Omit Paths: {np.mean(bridged_lengths):.2f} µm")
        print(f"Median Length of Bridged Omit Paths: {np.median(bridged_lengths):.2f} µm")
        
    print(f"\nTotal Broken Omit Paths: {len(broken_lengths)}")
    if broken_lengths:
        print(f"Mean Length of Broken Omit Paths: {np.mean(broken_lengths):.2f} µm")
        print(f"Median Length of Broken Omit Paths: {np.median(broken_lengths):.2f} µm")
        
    if bridged_lengths and broken_lengths:
        stat, p = mannwhitneyu(bridged_lengths, broken_lengths, alternative='less')
        print(f"\nMann-Whitney U test (Bridged < Broken): U = {stat}, p = {p:.4e}")
        if p < 0.05:
            print("Conclusion: Bridged omit paths are significantly shorter than broken omit paths.")
        else:
            print("Conclusion: Bridged omit paths are not significantly shorter than broken omit paths.")

if __name__ == "__main__":
    main()
