import glob
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
import sys
import subprocess

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", package])

for pkg in ["pandas", "psutil", "tensorstore", "tqdm"]:
    try:
        __import__(pkg)
    except ImportError:
        install(pkg)

try:
    import agentic_neuron_proofreader
except ImportError:
    install("https://github.com/AllenInstitute/agentic-neuron-proofreader/archive/refs/heads/main.zip")
    import agentic_neuron_proofreader

EDGE_SPLIT = 1

branching_split = 0
branching_total = 0
linear_split = 0
linear_total = 0

# Load all dataset caches one level above the current directory
cache_paths = glob.glob("../dataset_cache_*_add.pkl")
if not cache_paths:
    # Fallback to current directory just in case
    cache_paths = glob.glob("dataset_cache_*_add.pkl")

for path in cache_paths:
    with open(path, "rb") as f:
        payload = pickle.load(f)
    
    gt = payload["gt_graph"]
    edge_error = np.asarray(payload["gt_edge_error"])
    
    edges = list(gt.edges)
    degrees = dict(gt.degree)
    
    for k, (u, v) in enumerate(edges):
        # Classify as branching if at least one endpoint has degree > 2
        is_branching = degrees[u] > 2 or degrees[v] > 2
        is_split = (edge_error[k] == EDGE_SPLIT)
        
        if is_branching:
            branching_total += 1
            if is_split:
                branching_split += 1
        else:
            linear_total += 1
            if is_split:
                linear_split += 1

# Prepare contingency table for Chi-square test
# Table format: [[Branching Split, Branching Not Split],
#                [Linear Split, Linear Not Split]]
branching_not_split = branching_total - branching_split
linear_not_split = linear_total - linear_split

contingency_table = np.array([
    [branching_split, branching_not_split],
    [linear_split, linear_not_split]
])

chi2, p_val, dof, expected = chi2_contingency(contingency_table)

branching_split_pct = (branching_split / branching_total * 100) if branching_total > 0 else 0
linear_split_pct = (linear_split / linear_total * 100) if linear_total > 0 else 0

print("=== Split Errors on Branching vs. Linear Edges ===")
print(f"Branching edges: {branching_total} total, {branching_split} split ({branching_split_pct:.2f}%)")
print(f"Linear edges   : {linear_total} total, {linear_split} split ({linear_split_pct:.2f}%)")
print(f"\nChi-square test statistic: {chi2:.4f}")
print(f"p-value                  : {p_val:.4e}")

# Generate Grouped Bar Chart
labels = ['Branching Edges', 'Linear Edges']
percentages = [branching_split_pct, linear_split_pct]

plt.figure(figsize=(7, 6))
bars = plt.bar(labels, percentages, color=['#d62728', '#1f77b4'], width=0.5)
plt.ylabel('Split Error Percentage (%)')
plt.title('Split Error Frequency: Branching vs. Linear Edges')

# Add percentage text above the bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + (max(percentages)*0.02), f'{yval:.2f}%', ha='center', va='bottom')

plt.ylim(0, max(percentages) * 1.2 if max(percentages) > 0 else 1)
plt.tight_layout()
plt.show()
