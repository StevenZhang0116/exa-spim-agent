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
import pickle
import gc
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

def install(package):
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--quiet', package])

# Ensure all possible runtime dependencies are installed before importing the package
for pkg in ['psutil', 'pandas', 'tensorstore', 'scikit-learn']:
    try:
        # Check if installed (scikit-learn imports as sklearn)
        __import__(pkg.replace('scikit-learn', 'sklearn'))
    except ImportError:
        install(pkg)

from sklearn.metrics import roc_curve, auc

try:
    import agentic_neuron_proofreader
except ImportError:
    try:
        install('https://github.com/AllenInstitute/agentic-neuron-proofreader/archive/refs/heads/main.zip')
    except Exception:
        install('https://github.com/AllenInstitute/agentic-neuron-proofreader/archive/refs/heads/master.zip')
    import agentic_neuron_proofreader

def get_xyz(gt, n):
    if hasattr(gt, 'node_xyz'):
        return np.array(gt.node_xyz[n])
    else:
        return np.array(gt.nodes[n]['node_xyz'])

def get_false_gaps(tree, coords, labeled_nodes, gt, node_label, radius):
    f_gaps = []
    # query_pairs gives unique pairs within the given Euclidean distance radius
    pairs = tree.query_pairs(radius)
    for i, j in pairs:
        u = labeled_nodes[i]
        v = labeled_nodes[j]
        seg_u = gt.node_segment_id(u)
        seg_v = gt.node_segment_id(v)
        lab_u = node_label[u]
        lab_v = node_label[v]
        
        # Must belong to different GT neurons and have different predicted segment labels
        if seg_u != seg_v and lab_u != lab_v:
            dist = np.linalg.norm(coords[i] - coords[j])
            f_gaps.append(dist)
    return f_gaps

def main():
    true_gaps = []
    false_gaps = []
    
    # Look for the dataset caches in the current directory
    files = sorted(glob.glob('*_add.pkl'))
    if not files:
        print('No dataset files found!')
        return
        
    for path in files:
        with open(path, 'rb') as f:
            payload = pickle.load(f)
        
        gt = payload['gt_graph']
        node_label = np.asarray(payload['gt_node_canonical_label'])
        edge_error = np.asarray(payload['gt_edge_error'])
        
        # Filter for labeled nodes to compute false positive gaps
        labeled_nodes = [n for n in gt.nodes if node_label[n] != 0]
        if not labeled_nodes:
            del payload, gt, node_label, edge_error
            gc.collect()
            continue
            
        coords = np.array([get_xyz(gt, n) for n in labeled_nodes])
        tree = KDTree(coords)
        
        # Generate a set of 'false positive gaps' restricted to a 20 micrometer radius
        f_gaps = get_false_gaps(tree, coords, labeled_nodes, gt, node_label, 20.0)
        false_gaps.extend(f_gaps)
        
        # Find 'true split gaps' directly from EDGE_SPLIT tagged edges
        edges = list(gt.edges)
        for k, (u, v) in enumerate(edges):
            if edge_error[k] == 1:  # EDGE_SPLIT = 1
                lab_u = node_label[u]
                lab_v = node_label[v]
                # Both ends must be labeled with different segment ids (true split gaps)
                if lab_u != 0 and lab_v != 0 and lab_u != lab_v:
                    xyz_u = get_xyz(gt, u)
                    xyz_v = get_xyz(gt, v)
                    dist = np.linalg.norm(xyz_u - xyz_v)
                    true_gaps.append(dist)
                    
        # Explicitly free memory to prevent bloat across multiple large brains
        del tree, coords, labeled_nodes, edges
        del payload, gt, node_label, edge_error
        gc.collect()
                    
    true_gaps = np.array(true_gaps)
    false_gaps = np.array(false_gaps)
    
    if len(true_gaps) == 0 or len(false_gaps) == 0:
        print(f'Not enough data: {len(true_gaps)} true gaps, {len(false_gaps)} false gaps.')
        return
        
    # Setup for ROC curve
    # True splits are positives (1), false positive inter-neuron gaps are negatives (0)
    y_true = np.concatenate([np.ones(len(true_gaps)), np.zeros(len(false_gaps))])
    distances = np.concatenate([true_gaps, false_gaps])
    
    # A smaller gap distance indicates a stronger likelihood of being a true split.
    # Using negative distances allows the standard ROC curve to consider greater values as more positive.
    y_scores = -distances
    
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    # Calculate F1 Score to find optimal threshold
    P = len(true_gaps)
    N = len(false_gaps)
    TPs = tpr * P
    FPs = fpr * N
    
    # Prevent division by zero during precision & recall calculation
    precision = np.divide(TPs, TPs + FPs, out=np.zeros_like(TPs), where=(TPs + FPs) != 0)
    recall = TPs / P if P > 0 else np.zeros_like(TPs)
    f1_scores = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(precision), where=(precision + recall) != 0)
    
    best_idx = np.argmax(f1_scores)
    best_f1 = f1_scores[best_idx]
    best_thresh_dist = -thresholds[best_idx]  # Inverting back to a positive distance threshold in um
    
    print(f'Number of true split gaps: {len(true_gaps)}')
    print(f'Number of false positive gaps: {len(false_gaps)}')
    print(f'ROC AUC: {roc_auc:.4f}')
    print(f'Optimal Distance Threshold: {best_thresh_dist:.2f} um')
    print(f'Max F1 Score: {best_f1:.4f}')
    
    # Visualizations
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].hist(true_gaps, bins=30, alpha=0.5, label='True Splits', density=True)
    axes[0].hist(false_gaps, bins=30, alpha=0.5, label='False Positives', density=True)
    axes[0].set_xlabel('Gap Distance (um)')
    axes[0].set_ylabel('Density')
    axes[0].set_title('Distance Distributions (<= 20um)')
    axes[0].legend()
    
    axes[1].plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
    axes[1].plot([0, 1], [0, 1], 'k--')
    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True Positive Rate')
    axes[1].set_title('ROC Curve for Gap Distance Threshold')
    axes[1].legend(loc='lower right')
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
