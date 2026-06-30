import subprocess
import sys
import os

def install_and_import():
    # Install required dependencies to avoid ModuleNotFoundError during unpickling
    packages = ["psutil", "pandas", "scipy", "networkx", "tensorstore", "matplotlib", "tqdm"]
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet"] + packages)
    
    try:
        import agentic_neuron_proofreader
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "https://github.com/AllenInstitute/agentic-neuron-proofreader/archive/refs/heads/main.zip"])
        import agentic_neuron_proofreader

install_and_import()

import pickle
import numpy as np
import networkx as nx
from scipy.stats import chi2_contingency
import agentic_neuron_proofreader  # noqa: F401

dataset_path = "dataset_cache_789202_mcl100_add.pkl"

with open(dataset_path, "rb") as f:
    payload = pickle.load(f)

gt = payload["gt_graph"]
node_label = np.asarray(payload["gt_node_canonical_label"])

branch_omitted = 0
branch_labeled = 0
linear_omitted = 0
linear_labeled = 0

# Compute node degrees and classify nodes into Branch (degree > 2) and Linear (degree == 2)
for n in gt.nodes:
    deg = gt.degree[n]
    lab = int(node_label[n])
    
    if deg > 2:
        if lab == 0:
            branch_omitted += 1
        else:
            branch_labeled += 1
    elif deg == 2:
        if lab == 0:
            linear_omitted += 1
        else:
            linear_labeled += 1

# Construct contingency table
contingency_table = np.array([
    [branch_omitted, branch_labeled],
    [linear_omitted, linear_labeled]
])

branch_total = branch_omitted + branch_labeled
linear_total = linear_omitted + linear_labeled

branch_omission_rate = branch_omitted / branch_total if branch_total > 0 else 0
linear_omission_rate = linear_omitted / linear_total if linear_total > 0 else 0

# Perform Chi-square test of independence
chi2, p, dof, expected = chi2_contingency(contingency_table)

# Output Deliverables
print("=== Contingency Table ===")
print(f"{'Type':<10} {'Omitted':<10} {'Labeled':<10} {'Total':<10}")
print(f"{'Branch':<10} {branch_omitted:<10} {branch_labeled:<10} {branch_total:<10}")
print(f"{'Linear':<10} {linear_omitted:<10} {linear_labeled:<10} {linear_total:<10}")

print("\n=== Omission Rates ===")
print(f"Branch Nodes (degree > 2): {branch_omission_rate:.4%}")
print(f"Linear Nodes (degree = 2): {linear_omission_rate:.4%}")

print("\n=== Chi-square Test Results ===")
print(f"Chi-square statistic: {chi2:.4f}")
print(f"p-value: {p:.4e}")
print(f"Degrees of freedom: {dof}")
