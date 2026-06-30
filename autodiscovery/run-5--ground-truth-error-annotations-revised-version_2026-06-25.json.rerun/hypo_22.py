import sys
import subprocess
import os
import glob

def install_packages():
    # Install missing dependencies first
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "psutil", "tensorstore"])
    try:
        import agentic_neuron_proofreader
    except ImportError:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "--quiet", 
            "https://github.com/AllenInstitute/agentic-neuron-proofreader/archive/refs/heads/main.zip"
        ])

install_packages()

import pickle
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from collections import defaultdict

# Locate dataset cache files (search up to one level deep to prevent timeouts)
files = glob.glob("../*_add.pkl") + glob.glob("../*/*_add.pkl") + glob.glob("*_add.pkl") + glob.glob("*/*_add.pkl")
files = list(set([os.path.abspath(f) for f in files]))
print(f"Found {len(files)} dataset cache file(s).")

overlap_ratios = []

for path in files:
    with open(path, "rb") as f:
        payload = pickle.load(f)
    
    gt = payload["gt_graph"]
    node_label = np.asarray(payload["gt_node_canonical_label"])
    merge_labels = set(int(x) for x in payload["gt_merge_labels"])
    
    counts = defaultdict(lambda: defaultdict(int))
    
    for n in gt.nodes:
        lab = int(node_label[n])
        if lab in merge_labels:
            neuron_name = gt.node_segment_id(n)
            counts[lab][neuron_name] += 1
            
    for lab, neuron_counts in counts.items():
        vals = list(neuron_counts.values())
        if sum(vals) > 0:
            max_count = max(vals)
            total_count = sum(vals)
            overlap_ratios.append(max_count / total_count)

if not overlap_ratios:
    print("No merging segments found.")
else:
    overlap_ratios = np.array(overlap_ratios)
    mean_ratio = np.mean(overlap_ratios)
    median_ratio = np.median(overlap_ratios)
    
    # One-sample t-test to check if the mean overlap ratio is significantly greater than 0.5
    t_stat, p_val = stats.ttest_1samp(overlap_ratios, 0.5, alternative='greater')
    
    print(f"Total merging segments: {len(overlap_ratios)}")
    print(f"Mean overlap ratio: {mean_ratio:.4f}")
    print(f"Median overlap ratio: {median_ratio:.4f}")
    print(f"One-sample t-test (mu > 0.5): t-statistic = {t_stat:.4f}, p-value = {p_val:.4e}")
    
    # Plot histogram of overlap ratios
    plt.figure(figsize=(8, 6))
    plt.hist(overlap_ratios, bins=20, range=(0, 1), color='skyblue', edgecolor='black')
    plt.axvline(mean_ratio, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_ratio:.2f}')
    plt.axvline(median_ratio, color='green', linestyle='dashed', linewidth=2, label=f'Median: {median_ratio:.2f}')
    plt.axvline(0.5, color='gray', linestyle='dotted', linewidth=2, label='0.5 (Balanced Split)')
    plt.title("Histogram of Overlap Ratios for Merging Segments")
    plt.xlabel("Overlap Ratio (Max Nodes on One Neuron / Total Nodes Covered)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.show()
