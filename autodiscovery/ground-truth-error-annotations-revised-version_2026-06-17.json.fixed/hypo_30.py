import glob
import os
import re
import pickle
import numpy as np
import gc
import subprocess
import sys

# ---------------------------------------------------------------------------
# CORRECTED TEST (id 30)
# Original (recorded/rerun): Pearson r=0.6500 (p=2.2134e-02) plus OLS
#   R^2=0.422, F=7.315 (p=0.0221), n=12 neurons; Spearman rho=0.8811
#   (p=1.5267e-04) was also reported.
# WHY WRONG: n=12 is tiny and the OLS residuals are badly non-normal
#   (Jarque-Bera p=0.00036, skew=2.06, kurtosis=6.84) with one high-leverage
#   outlier (~2.5 splits/mm, 12.6% omit) that inflates the Pearson slope. The
#   parametric Pearson/OLS p=0.022 is borderline and assumption-violating.
# FIX: report the rank-based Spearman rho as the headline statistic with a
#   *permutation* p-value (exact null by shuffling one variable's labels,
#   robust to non-normality and the leverage point), plus a bootstrap 95% CI
#   for rho. Same per-neuron quantities (splits_per_mm, pct_omit), same
#   filtering, same loading. The Pearson/OLS are still echoed for comparison.
# ---------------------------------------------------------------------------

# Helper to install missing packages quietly

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", package])

# Install all required packages sequentially to satisfy implicit dependencies
for pkg in ["psutil", "tensorstore", "networkx", "pandas", "statsmodels", "matplotlib", "scipy"]:
    try:
        __import__(pkg)
    except ImportError:
        install(pkg)

try:
    import agentic_neuron_proofreader
except ImportError:
    install("https://github.com/AllenInstitute/agentic-neuron-proofreader/archive/refs/heads/main.zip")
    import agentic_neuron_proofreader

import pandas as pd
import statsmodels.formula.api as smf
import scipy.stats as stats
import matplotlib.pyplot as plt
from collections import defaultdict

# Locate dataset cache files. Loading fix: prefer $RERUN_PKL directly; the
# harness also redirects any *_add.pkl glob to the provided dataset.
rerun_pkl = os.environ.get("RERUN_PKL")
if rerun_pkl and os.path.exists(rerun_pkl):
    files = [rerun_pkl]
else:
    files = glob.glob("../*_add.pkl") + glob.glob("../*/*_add.pkl")
    files = list(set(files))

def brain_id_from_path(path):
    m = re.search(r"dataset_cache_(\d+)_mcl(\d+)_add\.pkl$", os.path.basename(path))
    return m.group(1) if m else "unknown"

data = []

for path in sorted(files):
    brain_id = brain_id_from_path(path)
    print(f"Processing Brain ID: {brain_id} from {os.path.basename(path)}...")

    # Disable garbage collection temporarily to speed up load time
    gc.disable()
    with open(path, "rb") as f:
        payload = pickle.load(f)
    gc.enable()

    # Extract required graphs and arrays
    gt = payload["gt_graph"]
    node_label = np.asarray(payload["gt_node_canonical_label"])
    edge_error = np.asarray(payload["gt_edge_error"])

    # Drop massive automated graph to free memory immediately
    payload.pop("fragments_graph", None)
    gc.collect()

    # Step 3: Count distinct segments overlapping each neuron to compute total splits
    neuron_segs = defaultdict(set)
    for n in gt.nodes:
        lab = int(node_label[n])
        if lab != 0:
            neuron_segs[gt.node_segment_id(n)].add(lab)

    neuron_splits = {nm: max(len(s) - 1, 0) for nm, s in neuron_segs.items()}

    # Variables to track length and edge classes per neuron
    neuron_edge_lengths = defaultdict(float)
    neuron_omit_counts = defaultdict(int)
    neuron_edge_counts = defaultdict(int)

    gt_edges = list(gt.edges)

    # Step 2 & 5: Calculate total cable length and omit error frequency per neuron
    for k, (u, v) in enumerate(gt_edges):
        neuron_id = gt.node_segment_id(u)

        xyz_u = gt.node_xyz[u]
        xyz_v = gt.node_xyz[v]
        dist = np.linalg.norm(xyz_u - xyz_v)

        neuron_edge_lengths[neuron_id] += dist
        neuron_edge_counts[neuron_id] += 1
        if edge_error[k] == 2:  # EDGE_OMIT
            neuron_omit_counts[neuron_id] += 1

    # Formulate metrics for regression
    for neuron_id in neuron_edge_counts.keys():
        length_um = neuron_edge_lengths[neuron_id]

        # Step 6: Filter out short fragments (< 50 um length)
        if length_um < 50:
            continue

        length_mm = length_um / 1000.0
        splits = neuron_splits.get(neuron_id, 0)
        splits_per_mm = splits / length_mm

        omit_count = neuron_omit_counts[neuron_id]
        total_edges = neuron_edge_counts[neuron_id]
        pct_omit = (omit_count / total_edges) * 100.0

        data.append({
            "neuron_id": neuron_id,
            "brain_id": "B" + str(brain_id),  # Prefix string to safely use as categorical variable
            "length_mm": length_mm,
            "splits_per_mm": splits_per_mm,
            "pct_omit": pct_omit
        })

    # Explicit memory cleanup per file
    del payload
    del gt
    del node_label
    del edge_error
    del gt_edges
    gc.collect()

df = pd.DataFrame(data)
print(f"\nSuccessfully processed {len(df)} neurons meeting the length criteria.")

if len(df) > 0:
    x = df['splits_per_mm'].to_numpy(dtype=float)
    y = df['pct_omit'].to_numpy(dtype=float)
    n = len(df)

    # --- Echo the original (fragile parametric) statistics for comparison ---
    pearson_r, pearson_p = stats.pearsonr(x, y)
    spearman_r, spearman_p = stats.spearmanr(x, y)
    print("\n--- Original (fragile) parametric statistics, echoed for comparison ---")
    print(f"Pearson r : {pearson_r:.4f} (p-value: {pearson_p:.4e})  [assumption-violating at n=12]")
    print(f"Spearman r (asymptotic): {spearman_r:.4f} (p-value: {spearman_p:.4e})")
    # recorded: Pearson r=0.6500 (p=2.2134e-02), Spearman rho=0.8811 (p=1.5267e-04),
    #           OLS R^2=0.422, F=7.315, splits_per_mm coef=2.2771, n=12

    # --- CORRECTED HEADLINE TEST: Spearman rho with a PERMUTATION p-value ---
    # Robust to non-normality and the high-leverage outlier; exact small-n null.
    rng = np.random.default_rng(42)
    n_perm = 50000
    obs_rho = stats.spearmanr(x, y).correlation
    perm_rhos = np.empty(n_perm)
    for i in range(n_perm):
        yp = rng.permutation(y)
        perm_rhos[i] = stats.spearmanr(x, yp).correlation
    # two-sided permutation p (add-one correction)
    perm_p = (np.sum(np.abs(perm_rhos) >= abs(obs_rho)) + 1) / (n_perm + 1)

    # Bootstrap 95% CI for Spearman rho (resample neuron pairs with replacement).
    boot_rhos = []
    for _ in range(10000):
        idx = rng.integers(0, n, size=n)
        bx, by = x[idx], y[idx]
        if np.std(bx) == 0 or np.std(by) == 0:
            continue
        boot_rhos.append(stats.spearmanr(bx, by).correlation)
    boot_rhos = np.asarray(boot_rhos)
    ci_lo, ci_hi = np.percentile(boot_rhos, [2.5, 97.5])

    print("\n--- CORRECTED headline: Spearman rho with permutation p-value ---")
    print(f"Spearman rho = {obs_rho:.4f}, permutation p = {perm_p:.4e} ({n_perm} permutations), n={n}")
    print(f"Effect size (Spearman rho) 95% bootstrap CI [{ci_lo:.4f}, {ci_hi:.4f}]")

    if perm_p < 0.05:
        print("Conclusion: Splits/mm and omit rate are significantly positively associated (rank-based, robust to n=12 non-normality).")
    else:
        print("Conclusion: No significant rank association between splits/mm and omit rate under the permutation test.")
else:
    print("\nInsufficient valid neuron data to compute correlations or plot.")
