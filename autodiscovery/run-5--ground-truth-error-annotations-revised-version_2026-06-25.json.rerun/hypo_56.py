import subprocess
import sys
import glob

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", package])

install("psutil")
install("pandas")
install("tensorstore")
install("https://github.com/AllenInstitute/agentic-neuron-proofreader/archive/refs/heads/main.zip")
install("scipy")
install("matplotlib")

import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
import agentic_neuron_proofreader
from collections import defaultdict

# 1. Load the dataset
paths = glob.glob("../*add.pkl") + glob.glob("../cache/*add.pkl") + glob.glob("./*add.pkl")
with open(paths[0], "rb") as f:
    payload = pickle.load(f)

# As noted in debugging, `fragments_graph` component IDs don't correspond to dense segment IDs.
# We use the GT node mappings `gt_node_canonical_label` to correctly estimate segment size.
node_label = np.asarray(payload["gt_node_canonical_label"])
merge_labels = set(int(x) for x in payload["gt_merge_labels"])
node_spacing = payload.get("node_spacing", 5)  # target µm spacing between skeleton nodes

# 2 & 3. Group GT nodes by predicted segment ID to estimate their size
seg_gt_counts = defaultdict(int)
for lbl in node_label:
    if lbl != 0:
        seg_gt_counts[lbl] += 1

# 4 & 5. Divide into merging and non-merging segments, excluding small fragments (<10 nodes)
merges_nc = []
non_merges_nc = []

for lbl, nc in seg_gt_counts.items():
    if nc >= 10:
        if lbl in merge_labels:
            merges_nc.append(nc)
        else:
            non_merges_nc.append(nc)

# 6. Compare using a Mann-Whitney U test
stat, pval = mannwhitneyu(merges_nc, non_merges_nc, alternative='two-sided')

# Deliverables
print("=== Descriptive Statistics ===")
print(f"Merging Segments (N={len(merges_nc)}):")
print(f"  Mean Node Count: {np.mean(merges_nc):.2f}")
print(f"  Median Node Count: {np.median(merges_nc):.2f}")
print(f"  Mean Est. Cable Length: {np.mean(merges_nc) * node_spacing:.2f} µm")

print(f"\nNon-Merging Segments (N={len(non_merges_nc)}):")
print(f"  Mean Node Count: {np.mean(non_merges_nc):.2f}")
print(f"  Median Node Count: {np.median(non_merges_nc):.2f}")
print(f"  Mean Est. Cable Length: {np.mean(non_merges_nc) * node_spacing:.2f} µm")

print("\n=== Statistical Significance ===")
print(f"Mann-Whitney U Test: statistic={stat}, p-value={pval:.2e}")

# Visualization: Log-scaled boxplot
plt.figure(figsize=(8, 6))
plt.boxplot([merges_nc, non_merges_nc], labels=["Merging Segments", "Non-Merging Segments"])
plt.yscale('log')
plt.ylabel('Node Count (Log Scale)')
plt.title('Size Distribution of Predicted Segments\n(Merging vs Non-Merging)')
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.show()
