import sys
import subprocess

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", package])

for pkg in ["psutil", "tensorstore", "scipy", "statsmodels"]:
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
import glob
import os
import re
import gc
import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf

def brain_id_from_path(path):
    m = re.search(r"dataset_cache_(\d+)_mcl(\d+)_add\.pkl$", os.path.basename(path))
    return m.group(1) if m else os.path.basename(path)

data = []

for path in sorted(glob.glob("../data/*_add.pkl")):
    brain_id = brain_id_from_path(path)
    print(f"Processing {brain_id}...")
    with open(path, "rb") as f:
        payload = pickle.load(f)

    gt = payload["gt_graph"]
    edge_error = np.asarray(payload["gt_edge_error"])

    edges = list(gt.edges)
    node_xyz = gt.node_xyz

    # Pre-calculate spatial distances and populate them as edge weights
    for u, v in edges:
        dist = np.linalg.norm(node_xyz[u] - node_xyz[v])
        gt[u][v]['weight'] = dist

    # Leaf nodes are degree 1
    leaves = [n for n, d in gt.degree() if d == 1]

    # Shortest path to nearest leaf from every node
    if leaves:
        dists = nx.multi_source_dijkstra_path_length(gt, leaves, weight='weight')
    else:
        dists = {}

    for i, (u, v) in enumerate(edges):
        du = dists.get(u, np.inf)
        dv = dists.get(v, np.inf)
        if np.isinf(du) and np.isinf(dv):
            continue

        weight = gt[u][v]['weight']
        # Midpoint shortest distance to leaf
        d_leaf = min(du, dv) + weight / 2.0

        # split class encoded as 1
        is_split = 1 if edge_error[i] == 1 else 0
        neuron_id = gt.node_segment_id(u)

        data.append({
            "brain_id": brain_id,
            "neuron_id": neuron_id,
            "dist_to_leaf": d_leaf,
            "is_split": is_split
        })

    # Free up memory
    del gt
    del payload
    gc.collect()

df = pd.DataFrame(data)
print(f"\nTotal valid edges processed: {len(df)}\n")

# --- 1. Split Error Rate by Descriptive Bins ---
desc_bins = [-1, 50, 200, np.inf]
desc_labels = ["<50um", "50-200um", ">200um"]
df['desc_bin'] = pd.cut(df['dist_to_leaf'], bins=desc_bins, labels=desc_labels)
desc_stats = df.groupby('desc_bin', observed=True)['is_split'].agg(['mean', 'count']).reset_index()

print("Split Error Rate by Descriptive Categories:")
for _, row in desc_stats.iterrows():
    print(f"  {row['desc_bin']:>8s}: {row['mean']*100:.2f}% splits (n={row['count']})")
print()

# --- 2. Line Plot ---
bins = list(range(0, 501, 50)) + [np.inf]
labels = [f"{bins[i]}-{bins[i+1]}" for i in range(len(bins)-2)] + [">500"]
df['plot_bin'] = pd.cut(df['dist_to_leaf'], bins=bins, labels=labels, right=False)
plot_stats = df.groupby('plot_bin', observed=True)['is_split'].agg(['mean', 'count']).reset_index()

plt.figure(figsize=(10, 6))
plt.plot(plot_stats['plot_bin'].astype(str), plot_stats['mean'] * 100, marker='o', linestyle='-', color='indigo')
plt.xlabel("Distance to Nearest Leaf Tip (um)")
plt.ylabel("Split Rate (%)")
plt.title("Edge Split Rate as a Function of Distance to Leaf Tip")
plt.xticks(rotation=45)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Standardize continuous predictor for modeling stability
df['dist_to_leaf_std'] = (df['dist_to_leaf'] - df['dist_to_leaf'].mean()) / df['dist_to_leaf'].std()
df['group_id'] = df['brain_id'].astype(str) + "_" + df['neuron_id'].astype(str)

# ----------------------------------------------------------------------
# Original (recorded) test: the intended Mixed-Effects GEE crashed with
#   "GEE.from_formula() got multiple values for argument 'groups'" (groups was
#   passed BOTH implicitly via smf.gee and again as a keyword), so the analysis
#   FELL BACK to a pooled logistic regression: coef = -0.1361, z = -9.92,
#   p < 0.001, n = 1,363,789 edges, pseudo R^2 = 0.0011.
# Fault flagged (MAJOR): PSEUDO-REPLICATION / violated independence -- the
#   pooled logit treats ~1.36M WITHIN-NEURON-CORRELATED edges as independent,
#   so the effective n is hugely overstated and the slope's certainty is
#   spurious. The fallback uses a nonrobust covariance.
# CORRECTION:
#   1. Run the GEE CORRECTLY (cluster = neuron) with an exchangeable working
#      correlation -> robust ("sandwich") SEs that respect within-neuron
#      clustering. This is the analysis that was intended but crashed.
#   2. As an independent cross-check that does NOT rely on edge-level n at all,
#      aggregate to the NEURON level: fit a per-neuron logistic slope is not
#      stable on rare events, so instead run a CLUSTER-LEVEL test on per-neuron
#      split-rate gradients -- correlate each neuron's bin split-rates with
#      distance via a per-neuron Spearman / sign test across neurons, giving a
#      p-value whose unit of replication is the neuron, not the edge.
# ----------------------------------------------------------------------

print("=== CORRECTED 1: GEE logistic regression, cluster = neuron (robust SEs) ===")
gee_ok = False
try:
    # CORRECT call signature: smf.gee(formula, groups, data, ...). 'groups' is the
    # SECOND positional argument and must be a COLUMN NAME (string), not a Series,
    # which is exactly what caused the original "multiple values for 'groups'".
    gee_model = smf.gee(
        "is_split ~ dist_to_leaf_std",
        "group_id",
        data=df,
        family=sm.families.Binomial(),
        cov_struct=sm.cov_struct.Exchangeable(),
    )
    gee_result = gee_model.fit(maxiter=50)
    coef = gee_result.params["dist_to_leaf_std"]
    se = gee_result.bse["dist_to_leaf_std"]
    z = gee_result.tvalues["dist_to_leaf_std"]
    p = gee_result.pvalues["dist_to_leaf_std"]
    ci = gee_result.conf_int().loc["dist_to_leaf_std"]
    n_clusters = df["group_id"].nunique()
    print(f"GEE coef (dist_to_leaf_std) = {coef:.4f}, robust SE = {se:.4f}")
    print(f"GEE z = {z:.3f}, p = {p:.4e}")
    print(f"GEE robust 95% CI = [{ci[0]:.4f}, {ci[1]:.4f}]  (odds-ratio per SD = {np.exp(coef):.4f})")
    print(f"Number of independent clusters (neurons) = {n_clusters}")
    print("(Original pooled-logit fallback: coef=-0.1361, z=-9.92, p<0.001, treated 1.36M edges as independent.)")
    gee_ok = True
except Exception as e:
    print("GEE still could not fit:", e)

# CORRECTED 2: cluster-level test, unit of replication = neuron
print("\n=== CORRECTED 2: cluster-level slope test, independent unit = neuron ===")
try:
    from scipy.stats import spearmanr, wilcoxon
    # Per-neuron split-rate vs distance: bin each neuron's edges and take the
    # within-neuron Spearman correlation between distance bin midpoint and split rate.
    # Then test the DISTRIBUTION of per-neuron slopes/correlations across neurons.
    neuron_corrs = []
    bin_edges = np.array([0, 25, 50, 100, 200, 400, np.inf])
    for gid, sub in df.groupby("group_id"):
        if len(sub) < 50 or sub["is_split"].sum() < 2:
            continue
        sub = sub.copy()
        sub["b"] = pd.cut(sub["dist_to_leaf"], bins=bin_edges, labels=False)
        grp = sub.groupby("b")["is_split"].mean()
        if grp.shape[0] < 3:
            continue
        rho, _ = spearmanr(grp.index.values.astype(float), grp.values)
        if np.isfinite(rho):
            neuron_corrs.append(rho)
    neuron_corrs = np.array(neuron_corrs)
    n_neurons = len(neuron_corrs)
    if n_neurons >= 2:
        median_rho = np.median(neuron_corrs)
        frac_neg = np.mean(neuron_corrs < 0)
        # Wilcoxon signed-rank test that per-neuron rho distribution is centered < 0
        try:
            w_stat, w_p = wilcoxon(neuron_corrs, alternative='less')
        except Exception:
            w_stat, w_p = float('nan'), float('nan')
        # bootstrap CI for the median per-neuron Spearman rho
        rng = np.random.default_rng(42)
        B = 5000
        boot = np.array([np.median(rng.choice(neuron_corrs, n_neurons, replace=True)) for _ in range(B)])
        lo, hi = np.percentile(boot, [2.5, 97.5])
        print(f"Per-neuron Spearman rho (split-rate vs distance-bin), n_neurons = {n_neurons}")
        print(f"Median rho = {median_rho:.4f}, bootstrap 95% CI = [{lo:.4f}, {hi:.4f}]")
        print(f"Fraction of neurons with NEGATIVE slope (split rate falls with distance): {frac_neg:.2%}")
        print(f"Wilcoxon signed-rank (rho < 0 across neurons): W = {w_stat}, p = {w_p:.4e}")
    else:
        print("Too few neurons with enough edges/splits for a cluster-level slope test.")
except Exception as e:
    print("Cluster-level test could not run:", e)
