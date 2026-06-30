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

    # ------------------------------------------------------------------
    # CORRECTED TEST
    # Original (FAULTY) test:
    #   one-sample t-test that mean overlap ratio > 0.5,
    #   t = 24.8654, p = 3.5768e-44, n = 98 (mean = 0.9129, median = 1.0000).
    # Why wrong: overlap_ratio is a proportion bounded on (0, 1], heavily
    #   left-skewed and piled at the 1.0 CEILING (median = 1.0). A one-sample
    #   t-test assumes approximate normality, which a ceiling-bounded mass at
    #   1.0 violates; the t-statistic / p-value are not trustworthy and the
    #   0.5 reference is far from where the data sit.
    # Correct approach for a skewed, ceiling-bounded one-sample location test:
    #   (1) one-sample Wilcoxon signed-rank test of (ratio - 0.5) [nonparametric],
    #   (2) an exact sign test (binomial) for #{ratio > 0.5} [no distributional
    #       assumption at all], and
    #   (3) report the MEDIAN with a bootstrap 95% CI as the effect size, so the
    #       conclusion rests on a distribution-free estimate rather than a mean.
    # ------------------------------------------------------------------
    n = len(overlap_ratios)
    REF = 0.5

    # (1) One-sample Wilcoxon signed-rank test against 0.5 (one-sided, greater).
    diffs = overlap_ratios - REF
    nonzero = diffs[diffs != 0]
    n_zero = int(np.sum(diffs == 0))
    try:
        w_stat, w_p = stats.wilcoxon(nonzero, alternative='greater', zero_method='wilcox')
    except Exception as e:
        w_stat, w_p = float('nan'), float('nan')
        print(f"Wilcoxon could not run: {e}")

    # (2) Exact sign test: number of ratios strictly above 0.5 vs binomial(0.5).
    n_above = int(np.sum(overlap_ratios > REF))
    n_below = int(np.sum(overlap_ratios < REF))
    n_eff = n_above + n_below
    sign_p = stats.binomtest(n_above, n_eff, 0.5, alternative='greater').pvalue if n_eff > 0 else float('nan')

    # (3) Bootstrap 95% CI for the median overlap ratio (distribution-free effect size).
    rng = np.random.default_rng(42)
    B = 10000
    boot_medians = np.empty(B)
    for b in range(B):
        sample = rng.choice(overlap_ratios, size=n, replace=True)
        boot_medians[b] = np.median(sample)
    med_lo, med_hi = np.percentile(boot_medians, [2.5, 97.5])
    # Bootstrap CI for the mean too, for direct comparison with the original t-test target.
    boot_means = np.empty(B)
    for b in range(B):
        sample = rng.choice(overlap_ratios, size=n, replace=True)
        boot_means[b] = np.mean(sample)
    mean_lo, mean_hi = np.percentile(boot_means, [2.5, 97.5])

    print(f"Total merging segments: {n}")
    print(f"Mean overlap ratio: {mean_ratio:.4f}")
    print(f"Median overlap ratio: {median_ratio:.4f}")
    print(f"Fraction of segments with overlap_ratio > 0.5: {n_above}/{n_eff} (ties at 0.5: {n_zero})")
    print()
    print("=== CORRECTED tests (overlap ratio is ceiling-bounded / left-skewed) ===")
    print(f"One-sample Wilcoxon signed-rank (median > 0.5): W = {w_stat}, p = {w_p:.4e} (n_nonzero={len(nonzero)})")
    print(f"Exact sign test (#ratio>0.5, one-sided greater): p = {sign_p:.4e}")
    print(f"Effect size - median overlap ratio = {median_ratio:.4f}, bootstrap 95% CI = [{med_lo:.4f}, {med_hi:.4f}]")
    print(f"Effect size - mean   overlap ratio = {mean_ratio:.4f}, bootstrap 95% CI = [{mean_lo:.4f}, {mean_hi:.4f}]")
    print("(Original one-sample t-test reported t=24.8654, p=3.5768e-44.)")

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
