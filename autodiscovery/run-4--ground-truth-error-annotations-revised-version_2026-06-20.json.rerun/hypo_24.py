import sys
import subprocess
import os
import glob
import pickle
import numpy as np
from collections import defaultdict
import scipy.stats as stats

def install(*packages):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet"] + list(packages))

try:
    import agentic_neuron_proofreader
except ImportError:
    install("psutil", "pandas", "networkx", "scipy", "tqdm", "tensorstore", "matplotlib", "boto3", "aiohttp")
    install("https://github.com/AllenInstitute/agentic-neuron-proofreader/archive/refs/heads/main.zip")
    import agentic_neuron_proofreader

def main():
    # Find dataset files robustly
    search_paths = [
        "**/*_add.pkl",
        "../*_add.pkl",
        "../*/*_add.pkl",
        "/*_add.pkl",
        "/*/*_add.pkl"
    ]
    cache_files = set()
    for pattern in search_paths:
        for match in glob.glob(pattern, recursive=True):
            cache_files.add(match)
            
    cache_files = list(cache_files)
    if not cache_files:
        print("No dataset files found.")
        return

    merges_2_neuron = []
    merges_super = []

    for file_path in cache_files:
        with open(file_path, "rb") as f:
            payload = pickle.load(f)
            
        gt = payload["gt_graph"]
        node_label = np.asarray(payload["gt_node_canonical_label"])
        merge_labels = set(int(x) for x in payload["gt_merge_labels"])
        
        # Access node coordinates securely whether they are a parallel array or networkx attributes
        if hasattr(gt, 'node_xyz'):
            get_xyz = lambda n: gt.node_xyz[n]
        else:
            get_xyz = lambda n: gt.nodes[n]['node_xyz']
            
        # Calculate nodal cable length contribution (half of incident edge lengths)
        node_cable = defaultdict(float)
        for u, v in gt.edges:
            p1 = np.array(get_xyz(u))
            p2 = np.array(get_xyz(v))
            dist = np.linalg.norm(p1 - p2) / 1000.0  # mm
            node_cable[u] += dist / 2.0
            node_cable[v] += dist / 2.0
            
        # Group nodes by label and count distinct GT neurons with > 50 nodes
        seg_neuron_counts = defaultdict(lambda: defaultdict(int))
        seg_cable = defaultdict(float)
        
        for n in gt.nodes:
            lab = int(node_label[n])
            if lab != 0:
                neuron = gt.node_segment_id(n)
                seg_neuron_counts[lab][neuron] += 1
                seg_cable[lab] += node_cable[n]
                
        # Classify merge errors based on severity
        for lab in merge_labels:
            neurons_covered = [neuron for neuron, count in seg_neuron_counts[lab].items() if count > 50]
            num_neurons = len(neurons_covered)
            
            if num_neurons >= 2:
                cable_len = seg_cable[lab]
                
                if num_neurons == 2:
                    merges_2_neuron.append(cable_len)
                else:
                    merges_super.append(cable_len)

    def print_summary(name, data):
        if not data:
            print(f"{name}: No data")
            return
        q25, median, q75 = np.percentile(data, [25, 50, 75])
        iqr = q75 - q25
        print(f"{name} (n={len(data)}):")
        print(f"  Median Covered Cable Length: {median:.4f} mm")
        print(f"  IQR: {iqr:.4f} mm")

    print("\n=== Merge Severity Cable Length Analysis ===")
    print_summary("2-Neuron Merges", merges_2_neuron)
    print_summary("Super-merges (>2 neurons)", merges_super)

    if len(merges_2_neuron) > 0 and len(merges_super) > 0:
        stat, p_val = stats.mannwhitneyu(merges_2_neuron, merges_super, alternative='two-sided')
        print("\n=== Mann-Whitney U Test ===")
        print(f"Statistic: {stat}")
        print(f"P-value: {p_val:.4e}")
    else:
        print("\nNot enough data to perform Mann-Whitney U Test.")

if __name__ == "__main__":
    main()
