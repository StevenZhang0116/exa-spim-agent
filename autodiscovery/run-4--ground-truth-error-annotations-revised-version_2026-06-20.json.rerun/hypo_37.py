import sys
from unittest.mock import MagicMock
import gc
import os
import glob
import pickle
import numpy as np
import subprocess

# Ensure essential lightweight dependencies are installed
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", package])

for pkg in ["tqdm", "networkx", "scipy"]:
    try:
        __import__(pkg)
    except ImportError:
        install(pkg)

# Mock heavy, unnecessary dependencies to prevent long builds and timeouts
class MockException(Exception): pass

m_google_auth_exc = MagicMock()
m_google_auth_exc.RefreshError = MockException
m_google_auth_exc.TransportError = MockException

mocked_modules = [
    'tensorstore', 'psutil', 'pandas', 'matplotlib', 'matplotlib.pyplot',
    'matplotlib.colors', 'zarr', 'dask', 's3fs', 'boto3', 'botocore',
    'botocore.client', 'google', 'google.auth', 'google.cloud'
]

for mod in mocked_modules:
    sys.modules[mod] = MagicMock()

sys.modules['google.auth.exceptions'] = m_google_auth_exc

try:
    import agentic_neuron_proofreader
except ImportError:
    # Use --no-deps to prevent compiling huge packages from source on Python 3.13
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "--no-deps", "https://github.com/AllenInstitute/agentic-neuron-proofreader/archive/main.zip"])
    import agentic_neuron_proofreader

from scipy.spatial import cKDTree
from scipy.stats import mannwhitneyu

# Find dataset safely, prioritizing the known parent directory location
dataset_paths = glob.glob("../*_add.pkl")
if not dataset_paths:
    dataset_paths = glob.glob("./*_add.pkl")
if not dataset_paths:
    dataset_paths = glob.glob("../**/*_add.pkl", recursive=True)
if not dataset_paths:
    dataset_paths = glob.glob("cache/*_add.pkl")

if not dataset_paths:
    print("Dataset not found.")
else:
    dataset_path = dataset_paths[0]
    
    # Load dataset payload
    with open(dataset_path, "rb") as f:
        payload = pickle.load(f)

    gt = payload["gt_graph"]
    edge_error = np.asarray(payload["gt_edge_error"])
    
    gt_edges_list = list(gt.edges)
    edges_arr = np.array(gt_edges_list)
    
    # Extract needed attributes directly
    node_xyz = gt.node_xyz
    
    # Free up multi-gigabyte memory immediately to prevent swapping and thrashing timeouts
    payload.pop("fragments_graph", None)
    del payload
    del gt
    gc.collect()
    
    if len(edges_arr) > 0:
        # Vectorized extraction of midpoints to prevent loop timeouts
        u_nodes = edges_arr[:, 0]
        v_nodes = edges_arr[:, 1]
        
        midpoints = (node_xyz[u_nodes] + node_xyz[v_nodes]) / 2.0
        
        split_midpoints = midpoints[edge_error == 1]
        correct_midpoints = midpoints[edge_error == 0]
        
        # Downsample the correct control group to match the split group size
        np.random.seed(42)
        if len(correct_midpoints) > len(split_midpoints) and len(split_midpoints) > 0:
            idx = np.random.choice(len(correct_midpoints), len(split_midpoints), replace=False)
            sampled_correct_midpoints = correct_midpoints[idx]
        else:
            sampled_correct_midpoints = correct_midpoints

        if len(split_midpoints) > 0:
            # Build KD-Tree for split midpoints
            tree = cKDTree(split_midpoints)
            radius = 30.0

            # Query KD-Tree with return_length=True for extreme C-level speed
            split_neighbor_counts = tree.query_ball_point(split_midpoints, r=radius, return_length=True)
            split_counts = np.maximum(0, np.array(split_neighbor_counts) - 1)  # Exclude self
            
            if len(sampled_correct_midpoints) > 0:
                correct_counts = np.array(tree.query_ball_point(sampled_correct_midpoints, r=radius, return_length=True))
            else:
                correct_counts = np.array([])

            if len(split_counts) > 0 and len(correct_counts) > 0:
                # Use a Mann-Whitney U test to compare the neighbor counts between the two groups
                u_stat, p_val = mannwhitneyu(split_counts, correct_counts, alternative='greater')

                print("=== Spatial Clustering of Split Errors ===")
                print(f"Number of split edges: {len(split_counts)}")
                print(f"Number of correct edges (sampled): {len(correct_counts)}")
                print(f"Mean split neighbors within {radius} µm for Split edges: {np.mean(split_counts):.2f}")
                print(f"Mean split neighbors within {radius} µm for Correct edges: {np.mean(correct_counts):.2f}")
                print(f"Mann-Whitney U statistic: {u_stat}")
                print(f"p-value: {p_val:.2e}")
            else:
                print("Not enough edges to compute statistics.")
        else:
            print("No split edges found.")
    else:
        print("No edges found in the graph.")
