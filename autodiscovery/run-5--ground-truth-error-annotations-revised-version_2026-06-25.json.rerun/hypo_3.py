import subprocess
import sys
import glob
import pickle
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import chi2_contingency

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", package])

try:
    import agentic_neuron_proofreader
except ImportError:
    for pkg in ["psutil", "pandas", "networkx", "scipy", "tensorstore", "matplotlib", "tqdm"]:
        install(pkg)
    install("https://github.com/AllenInstitute/agentic-neuron-proofreader/archive/refs/heads/main.zip")
    import agentic_neuron_proofreader

files = glob.glob("../*_add.pkl") + glob.glob("**/*_add.pkl", recursive=True)
files = list(set(files))

total_proximal_edges = 0
total_proximal_splits = 0
total_distal_edges = 0
total_distal_splits = 0

for path in files:
    with open(path, "rb") as f:
        payload = pickle.load(f)
    
    gt = payload["gt_graph"]
    edge_errors = np.asarray(payload["gt_edge_error"])
    edges = list(gt.edges)
    
    branch_nodes = [n for n, d in gt.degree() if d > 2]
    if not branch_nodes:
        continue
        
    branch_xyz = np.array([gt.node_xyz[n] for n in branch_nodes])
    tree = cKDTree(branch_xyz)
    
    u_nodes = np.array([u for u, v in edges])
    v_nodes = np.array([v for u, v in edges])
    
    u_xyz = gt.node_xyz[u_nodes]
    v_xyz = gt.node_xyz[v_nodes]
    
    midpoints = (u_xyz + v_xyz) / 2.0
    
    distances, _ = tree.query(midpoints)
    
    proximal_mask = distances <= 15.0
    distal_mask = distances > 15.0
    
    total_proximal_edges += np.sum(proximal_mask)
    total_proximal_splits += np.sum(edge_errors[proximal_mask] == 1)
    
    total_distal_edges += np.sum(distal_mask)
    total_distal_splits += np.sum(edge_errors[distal_mask] == 1)

if total_proximal_edges == 0 and total_distal_edges == 0:
    print("No edges found.")
else:
    proximal_rate = total_proximal_splits / total_proximal_edges if total_proximal_edges > 0 else 0
    distal_rate = total_distal_splits / total_distal_edges if total_distal_edges > 0 else 0
    
    print(f"Branch-Proximal (<= 15 um) Edges: {total_proximal_edges}")
    print(f"  Splits: {total_proximal_splits} ({proximal_rate*100:.2f}%)")
    print(f"Branch-Distal (> 15 um) Edges: {total_distal_edges}")
    print(f"  Splits: {total_distal_splits} ({distal_rate*100:.2f}%)")
    
    if distal_rate > 0:
        rr = proximal_rate / distal_rate
        print(f"Relative Risk: {rr:.4f}")
    else:
        print("Relative Risk: inf")
        
    contingency_table = [
        [total_proximal_splits, total_proximal_edges - total_proximal_splits],
        [total_distal_splits, total_distal_edges - total_distal_splits]
    ]
    
    chi2, p_val, dof, expected = chi2_contingency(contingency_table)
    print(f"Chi-square p-value: {p_val:.4e}")
