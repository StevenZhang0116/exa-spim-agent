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
import glob
import pickle
import numpy as np
import scipy.stats as stats
import subprocess
import sys
import gc
from collections import defaultdict

def install_deps():
    # Install required dependencies explicitly as they are required by agentic_neuron_proofreader utils
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "psutil", "tensorstore"])
    try:
        import agentic_neuron_proofreader
    except ImportError:
        url = "https://github.com/AllenInstitute/agentic-neuron-proofreader/archive/refs/heads/main.zip"
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", url])

install_deps()
import agentic_neuron_proofreader

def main():
    # Locate dataset caches based on expected file structure
    files = []
    for pattern in [
        "../*_add.pkl",
        "../cache/*_add.pkl",
        "cache/*_add.pkl",
        "*_add.pkl"
    ]:
        files = glob.glob(pattern)
        if files:
            break
            
    if not files:
        print("No dataset files found.")
        return
        
    total_edges = 0
    total_omit_edges = 0
    transition_matrix = np.zeros((2, 2), dtype=int)
    
    for fpath in sorted(files):
        with open(fpath, "rb") as f:
            payload = pickle.load(f)
        
        gt = payload["gt_graph"]
        edge_error = np.asarray(payload["gt_edge_error"])
        
        total_edges += len(edge_error)
        total_omit_edges += np.count_nonzero(edge_error == 2)
        
        # Map incident edges to states (0: OTHER, 1: OMIT) per node
        node_states = defaultdict(list)
        for (u, v), err in zip(gt.edges, edge_error):
            state = 1 if err == 2 else 0
            node_states[u].append(state)
            node_states[v].append(state)
            
        # Efficiently compute combinations of adjacent edges for the transition matrix
        for states in node_states.values():
            n_states = len(states)
            if n_states < 2:
                continue
                
            c1 = sum(states)
            c0 = n_states - c1
            
            # For each pair of adjacent edges at this node, we add to the symmetric matrix:
            # Since we count undirected transitions both ways [i,j] and [j,i], we effectively
            # add 2 for identical state pairs, which resolves to c * (c - 1)
            transition_matrix[0, 0] += c0 * (c0 - 1)
            transition_matrix[1, 1] += c1 * (c1 - 1)
            transition_matrix[0, 1] += c0 * c1
            transition_matrix[1, 0] += c0 * c1
            
        # Aggressively release memory before loading the next graph to prevent OOM/timeouts
        del payload
        del gt
        del edge_error
        del node_states
        gc.collect()

    # 4. Compute the marginal probability of an edge being an OMIT error 
    marg_prob = total_omit_edges / total_edges if total_edges > 0 else 0
    
    # 5. Compute the conditional probability of an edge being an OMIT error given that an adjacent edge is an OMIT error.
    row1_sum = transition_matrix[1, 0] + transition_matrix[1, 1]
    cond_prob = transition_matrix[1, 1] / row1_sum if row1_sum > 0 else 0
    
    print("Edge-state transition matrix (0: OTHER, 1: OMIT):")
    print(transition_matrix)
    print(f"\nMarginal probability of OMIT: {marg_prob:.4f}")
    print(f"Conditional probability of OMIT given adjacent is OMIT: {cond_prob:.4f}")
    
    # 7. Compare the conditional probability to the marginal probability 
    if marg_prob > 0:
        ratio = cond_prob / marg_prob
        print(f"Ratio (Conditional / Marginal): {ratio:.2f}")
        if ratio >= 3:
            print("Hypothesis validated: Transition probability is at least 3 times higher than the marginal probability.")
        else:
            print("Hypothesis not validated: The ratio did not meet the 3x threshold.")
    
    # 6. Perform a chi-square test of independence on the transition counts to statistically confirm clustering.
    try:
        chi2, p_val, dof, expected = stats.chi2_contingency(transition_matrix)
        print(f"\nChi-square test statistic: {chi2:.4f}")
        print(f"p-value: {p_val:.4e}")
    except ValueError as e:
        print(f"\nCould not compute chi-square test: {e}")

if __name__ == "__main__":
    main()
