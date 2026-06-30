import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", package])

# Ensure dependencies are installed
dependencies = ["psutil", "tensorstore", "networkx", "scipy", "numpy", "pandas", "tqdm", "matplotlib"]
for pkg in dependencies:
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
import scipy.stats as stats
import random
import matplotlib.pyplot as plt

# Load dataset from the current directory
data_path = "dataset_cache_789202_mcl100_add.pkl"
with open(data_path, "rb") as f:
    payload = pickle.load(f)

gt = payload["gt_graph"]
fragments = payload["fragments_graph"]
merge_sites = payload["gt_merge_sites"]
node_label = np.asarray(payload["gt_node_canonical_label"])
merge_labels = set(int(x) for x in payload["gt_merge_labels"])

# 1. Extract the xyz coordinates of all merge sites
merge_coords = [site["xyz"] for site in merge_sites]

# 2. Randomly sample an equal number of nodes from gt_graph not associated with merge errors
non_merge_coords = []
for n in gt.nodes:
    if int(node_label[n]) not in merge_labels:
        non_merge_coords.append(gt.node_xyz[n])

random.seed(42)
if len(non_merge_coords) > len(merge_coords):
    sampled_non_merge_coords = random.sample(non_merge_coords, len(merge_coords))
else:
    sampled_non_merge_coords = non_merge_coords

# 3. Query the fragments_graph's KDTree for fragment nodes within 15 µm radius
radius = 15

if not merge_coords:
    print("No merge sites found in the dataset.")
else:
    merge_counts = []
    for coord in merge_coords:
        indices = fragments.kdtree.query_ball_point(coord, r=radius)
        merge_counts.append(len(indices))

    non_merge_counts = []
    for coord in sampled_non_merge_coords:
        indices = fragments.kdtree.query_ball_point(coord, r=radius)
        non_merge_counts.append(len(indices))

    # 4. Perform a statistical test (Mann-Whitney U Test)
    statistic, p_value = stats.mannwhitneyu(merge_counts, non_merge_counts)

    mean_merge_density = np.mean(merge_counts)
    mean_non_merge_density = np.mean(non_merge_counts)

    # Output deliverables
    print(f"Number of merge sites evaluated: {len(merge_counts)}")
    print(f"Mean fragment node count within {radius} µm for merge sites: {mean_merge_density:.2f}")
    print(f"Mean fragment node count within {radius} µm for control sites: {mean_non_merge_density:.2f}")
    print(f"Mann-Whitney U Test: statistic={statistic}, p-value={p_value:.2e}")

    # 5. Boxplot for density comparison
    plt.figure(figsize=(8, 6))
    plt.boxplot([merge_counts, non_merge_counts], labels=['Merge Sites', 'Control Sites'])
    plt.ylabel(f'Fragment Node Count (within {radius} µm)')
    plt.title('Local Fragment Density: Merge Sites vs Control Sites')
    plt.show()
