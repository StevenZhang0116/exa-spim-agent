# CORRECTED TEST for hypothesis id 24 (super-merges cover disproportionately
# more GT cable than 2-neuron merges).
#
# FAULTS (verifier MAJOR): (1) CONSTRUCT MISMATCH -- the hypothesis is about cable
# PER NEURON, but the code compared TOTAL covered cable per merge label; total
# cable is mechanically larger for super-merges because they span more neurons,
# so the test partly tests its own definition. (2) UNDERPOWERED / no effect size
# -- n=3 super-merges with a two-sided Mann-Whitney (U=0.0) is the minimum
# configuration that can reach p<0.01, and it DIVERGED on rerun to n=1 / p=0.222.
#
# CORRECTION: keep the SAME merge classes but test the quantity the hypothesis
# actually claims -- cable PER NEURON (total covered cable / number of GT neurons
# covered). Report an exact Mann-Whitney p, the rank-biserial correlation (Cliff's
# delta) as an EFFECT SIZE with a bootstrap 95% CI, and clearly flag the sample
# size. We test BOTH per-neuron (corrected construct) and total (original construct)
# so the driver can compare.
#
# ORIGINAL RECORDED NUMBERS (for the driver to compare):
#   2-neuron merges n=24, super-merges n=3, Mann-Whitney U=0.0, p=5.8861e-03,
#   medians 6.2054 mm (2-neuron total) vs 35.1873 mm (super total).

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

def cliffs_delta_and_rbc(a, b):
    """Cliff's delta = P(a>b)-P(a<b); equals rank-biserial r for MW. a,b arrays."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    gt = sum((x > b).sum() for x in a)
    lt = sum((x < b).sum() for x in a)
    n = len(a) * len(b)
    return (gt - lt) / n if n else np.nan

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

    # Per-merge totals and per-neuron values for the two classes.
    total_2, total_super = [], []      # original construct: total cable
    pern_2, pern_super = [], []        # corrected construct: cable per neuron

    for file_path in cache_files:
        with open(file_path, "rb") as f:
            payload = pickle.load(f)

        gt = payload["gt_graph"]
        node_label = np.asarray(payload["gt_node_canonical_label"])
        merge_labels = set(int(x) for x in payload["gt_merge_labels"])

        if hasattr(gt, 'node_xyz'):
            get_xyz = lambda n: gt.node_xyz[n]
        else:
            get_xyz = lambda n: gt.nodes[n]['node_xyz']

        node_cable = defaultdict(float)
        for u, v in gt.edges:
            p1 = np.array(get_xyz(u))
            p2 = np.array(get_xyz(v))
            dist = np.linalg.norm(p1 - p2) / 1000.0  # mm
            node_cable[u] += dist / 2.0
            node_cable[v] += dist / 2.0

        seg_neuron_counts = defaultdict(lambda: defaultdict(int))
        seg_cable = defaultdict(float)

        for n in gt.nodes:
            lab = int(node_label[n])
            if lab != 0:
                neuron = gt.node_segment_id(n)
                seg_neuron_counts[lab][neuron] += 1
                seg_cable[lab] += node_cable[n]

        for lab in merge_labels:
            neurons_covered = [neuron for neuron, count in seg_neuron_counts[lab].items() if count > 50]
            num_neurons = len(neurons_covered)

            if num_neurons >= 2:
                cable_len = seg_cable[lab]
                cable_per_neuron = cable_len / num_neurons  # CORRECTED quantity
                if num_neurons == 2:
                    total_2.append(cable_len)
                    pern_2.append(cable_per_neuron)
                else:
                    total_super.append(cable_len)
                    pern_super.append(cable_per_neuron)

    def summarize(name, data):
        if not data:
            print(f"{name}: No data (n=0)")
            return
        q25, median, q75 = np.percentile(data, [25, 50, 75])
        print(f"{name} (n={len(data)}): median={median:.4f} mm, IQR={q75-q25:.4f} mm")

    print("\n=== Merge severity: PER-NEURON cable (corrected construct) ===")
    summarize("2-Neuron merges (per-neuron)", pern_2)
    summarize("Super-merges (per-neuron)", pern_super)

    def test_pair(label, two, sup):
        print(f"\n--- {label} ---")
        if len(two) > 0 and len(sup) > 0:
            # Exact two-sided Mann-Whitney (appropriate for tiny n).
            method = "exact" if (len(two) * len(sup) < 1e5) else "auto"
            try:
                stat, p = stats.mannwhitneyu(sup, two, alternative='two-sided', method=method)
            except TypeError:
                stat, p = stats.mannwhitneyu(sup, two, alternative='two-sided')
            delta = cliffs_delta_and_rbc(sup, two)  # >0 means super > two
            # Bootstrap 95% CI for Cliff's delta.
            rng = np.random.default_rng(42)
            boots = []
            sup_a = np.asarray(sup, float); two_a = np.asarray(two, float)
            for _ in range(2000):
                bs = rng.choice(sup_a, size=len(sup_a), replace=True)
                bt = rng.choice(two_a, size=len(two_a), replace=True)
                boots.append(cliffs_delta_and_rbc(bs, bt))
            lo, hi = np.percentile(boots, [2.5, 97.5])
            print(f"  n_super={len(sup)}, n_2neuron={len(two)}")
            print(f"  Mann-Whitney U={stat}, exact two-sided p={p:.4e}")
            print(f"  Cliff's delta (super vs 2-neuron) = {delta:.4f}  [rank-biserial effect size]")
            print(f"  Bootstrap 95% CI for Cliff's delta = [{lo:.4f}, {hi:.4f}]")
            if len(sup) < 3:
                print(f"  [WARNING] n_super={len(sup)} is too small for a reliable inference.")
        else:
            print("  Not enough data to perform the test (one class is empty).")

    test_pair("PER-NEURON cable (corrected: hypothesis quantity)", pern_2, pern_super)

    print("\n=== Original construct (TOTAL cable) -- echoed for comparison ===")
    summarize("2-Neuron merges (total)", total_2)
    summarize("Super-merges (total)", total_super)
    test_pair("TOTAL cable (original construct)", total_2, total_super)

if __name__ == "__main__":
    main()
