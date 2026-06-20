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
import os
import sys
import subprocess

# Install missing dependencies
def install(*packages):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet"] + list(packages))

# Install required packages for the environment and for the module to load
install("psutil", "pandas", "tensorstore", "networkx", "scipy", "matplotlib")

try:
    import agentic_neuron_proofreader
except ImportError:
    install("https://github.com/AllenInstitute/agentic-neuron-proofreader/archive/refs/heads/main.zip")
    import agentic_neuron_proofreader

import pickle
import numpy as np
from scipy import stats
from collections import defaultdict

def get_dataset_files():
    files = []
    search_dirs = ["..", "."]
    for d in search_dirs:
        for root, _, filenames in os.walk(d):
            for f in filenames:
                if f.endswith("_add.pkl"):
                    files.append(os.path.join(root, f))
        if files:
            break
    return list({os.path.abspath(f) for f in files})

def main():
    caches = get_dataset_files()
    if not caches:
        print("No dataset files found.")
        return

    merging_lengths = []
    non_merging_lengths = []

    # Process all found datasets
    for cache_path in caches:
        with open(cache_path, "rb") as f:
            payload = pickle.load(f)

        gt = payload["gt_graph"]
        node_label = np.asarray(payload["gt_node_canonical_label"])
        merge_labels = set(int(x) for x in payload["gt_merge_labels"])

        node_xyz = gt.node_xyz

        # 1. Group GT edges by their predicted segmentation ID
        segment_lengths = defaultdict(float)

        for u, v in gt.edges:
            lab_u = node_label[u]
            lab_v = node_label[v]
            
            # Process edges where both endpoints share the same valid canonical label (> 0)
            if lab_u == lab_v and lab_u > 0:
                # 2. Calculate physical Euclidean length of the edge in µm
                dist = np.linalg.norm(node_xyz[u] - node_xyz[v])
                segment_lengths[lab_u] += dist

        # 3. Partition into merging and non-merging segment arrays
        for seg_id, length in segment_lengths.items():
            if length > 0:
                if seg_id in merge_labels:
                    merging_lengths.append(length)
                else:
                    non_merging_lengths.append(length)

    merging_lengths = np.array(merging_lengths)
    non_merging_lengths = np.array(non_merging_lengths)

    if len(merging_lengths) == 0 or len(non_merging_lengths) == 0:
        print("Not enough segments to perform t-test.")
        return

    print(f"Number of merging segments: {len(merging_lengths)}")
    print(f"Number of non-merging segments: {len(non_merging_lengths)}")

    print(f"\nMerging segments - Mean length: {np.mean(merging_lengths):.2f} µm, Median length: {np.median(merging_lengths):.2f} µm")
    print(f"Non-merging segments - Mean length: {np.mean(non_merging_lengths):.2f} µm, Median length: {np.median(non_merging_lengths):.2f} µm")

    # 4. Apply a log10 transformation to the lengths
    log_merging = np.log10(merging_lengths)
    log_non_merging = np.log10(non_merging_lengths)

    # 5. Compare the log-length distributions using Welch's t-test
    t_stat, p_val = stats.ttest_ind(log_merging, log_non_merging, equal_var=False)

    print(f"\nWelch's t-test on log-transformed lengths:")
    print(f"t-statistic: {t_stat:.4f}")
    print(f"p-value: {p_val:.4e}")

if __name__ == "__main__":
    main()
