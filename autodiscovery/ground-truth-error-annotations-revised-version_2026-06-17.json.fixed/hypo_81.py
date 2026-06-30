import sys
import subprocess
import os

def setup_env():
    packages = ["psutil", "pandas", "tensorstore", "scipy", "networkx", "https://github.com/AllenInstitute/agentic-neuron-proofreader/archive/refs/heads/main.zip"]
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet"] + packages)

setup_env()

import agentic_neuron_proofreader
import pickle
import numpy as np
from scipy.spatial import cKDTree
from scipy import stats
import random

# ---------------------------------------------------------------------------
# CORRECTED TEST (id 81)
# Original (recorded/rerun): Welch's two-sample t-test on volumetric fragment
#   density (nodes/um^3) at merge sites vs control regions.
#   recorded: Welch t=8.9288, p=2.7673e-14, n=67/67,
#             merge mean 0.001678 vs control mean 0.001083 nodes/um^3.
# WHY WRONG: the "density" is a within-sphere node *count* divided by a constant
#   volume -> discrete, non-negative, right-skewed count data, NOT normal. A
#   parametric t-test assumes normality (questionable at n=67). The
#   substantively identical findings (ids 2 and 23) correctly used Mann-Whitney.
# FIX: Mann-Whitney U (rank-based, two-sided) on the SAME densities, plus a
#   rank-biserial correlation effect size with a bootstrap 95% CI. Same data,
#   same grouping, same loading, same quantities being compared.
# ---------------------------------------------------------------------------

def main():
    random.seed(42)
    np.random.seed(42)

    filename = "dataset_cache_789202_mcl100_add.pkl"
    # Loading fix: load directly from $RERUN_PKL (harness also redirects pkl paths).
    filepath = os.environ.get("RERUN_PKL")
    if not filepath or not os.path.exists(filepath):
        filepath = os.path.join("..", filename)

    print(f"Loading dataset from: {filepath}")

    with open(filepath, 'rb') as f:
        payload = pickle.load(f)

    fg = payload['fragments_graph']
    gt = payload['gt_graph']
    gt_merge_sites = payload['gt_merge_sites']
    gt_edge_error = np.asarray(payload['gt_edge_error'])

    # 1. Build KD-tree over fragment nodes
    valid_nodes = list(fg.nodes)
    if hasattr(fg, 'node_xyz') and isinstance(fg.node_xyz, np.ndarray):
        try:
            fg_coords = fg.node_xyz[valid_nodes]
        except Exception:
            fg_coords = np.array([fg.node_xyz[n] for n in valid_nodes])
    elif hasattr(fg, 'node_xyz') and fg.node_xyz is not None:
        fg_coords = np.array([fg.node_xyz[n] for n in valid_nodes])
    else:
        fg_coords = np.array([fg.nodes[n]['xyz'] for n in valid_nodes])

    tree = cKDTree(fg_coords)

    # 2. Local density for merge locations
    merge_coords = [np.array(site['xyz']) for site in gt_merge_sites]
    radius = 10.0
    volume = (4.0 / 3.0) * np.pi * (radius ** 3)

    merge_counts = tree.query_ball_point(merge_coords, r=radius)
    merge_densities = [len(neighbors) / volume for neighbors in merge_counts]

    # 3. Local density for correctly reconstructed GT edges
    edges = list(gt.edges)
    correct_edges = [edges[i] for i, err in enumerate(gt_edge_error) if err == 0]

    edge_lengths = []
    valid_correct_edges = []
    for u, v in correct_edges:
        if hasattr(gt, 'node_xyz') and gt.node_xyz is not None:
            p1 = np.array(gt.node_xyz[u])
            p2 = np.array(gt.node_xyz[v])
        else:
            p1 = np.array(gt.nodes[u]['xyz'])
            p2 = np.array(gt.nodes[v]['xyz'])

        dist = np.linalg.norm(p1 - p2)
        edge_lengths.append(dist)
        valid_correct_edges.append((p1, p2))

    total_len = sum(edge_lengths)
    probs = [l / total_len for l in edge_lengths]

    control_coords = []
    # Sample randomly, weighting by physical edge length to avoid bias
    chosen_edges_idx = random.choices(range(len(valid_correct_edges)), weights=probs, k=len(merge_coords))

    for idx in chosen_edges_idx:
        p1, p2 = valid_correct_edges[idx]
        t = random.uniform(0, 1)
        p = p1 * t + p2 * (1 - t)
        control_coords.append(p)

    control_counts = tree.query_ball_point(control_coords, r=radius)
    control_densities = [len(neighbors) / volume for neighbors in control_counts]

    merge_arr = np.asarray(merge_densities, dtype=float)
    control_arr = np.asarray(control_densities, dtype=float)
    n1, n2 = len(merge_arr), len(control_arr)

    # 4. CORRECTED TEST: Mann-Whitney U (two-sided), the right non-parametric
    #    test for skewed count-derived densities (matches ids 2/23).
    U, p_val = stats.mannwhitneyu(merge_arr, control_arr, alternative='two-sided')

    # Rank-biserial correlation effect size: r_rb = 1 - 2U/(n1*n2)
    # (scipy returns U for the first sample; convention here gives positive r_rb
    #  when merge densities tend to exceed control densities).
    r_rb = 2.0 * U / (n1 * n2) - 1.0

    # Bootstrap 95% CI for the rank-biserial effect size (resample within group).
    rng = np.random.default_rng(42)
    boot = []
    for _ in range(5000):
        bm = rng.choice(merge_arr, size=n1, replace=True)
        bc = rng.choice(control_arr, size=n2, replace=True)
        bU, _ = stats.mannwhitneyu(bm, bc, alternative='two-sided')
        boot.append(2.0 * bU / (n1 * n2) - 1.0)
    ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])

    print("\n=== Local Graph Density Analysis (CORRECTED: Mann-Whitney U) ===")
    print(f"Number of merge sites evaluated: {n1}")
    print(f"Number of control regions evaluated: {n2}")
    print(f"Local neighborhood radius: {radius} um (Volume: {volume:.2f} um^3)")
    print(f"Average fragment node density at merge sites: {np.mean(merge_arr):.6f} nodes/um^3")
    print(f"Average fragment node density at correct regions: {np.mean(control_arr):.6f} nodes/um^3")
    print(f"Median fragment node density at merge sites: {np.median(merge_arr):.6f} nodes/um^3")
    print(f"Median fragment node density at correct regions: {np.median(control_arr):.6f} nodes/um^3")
    print(f"Mann-Whitney U (two-sided): U={U:.1f}, p={p_val:.4e}, n={n1}/{n2}")
    print(f"Effect size rank-biserial r_rb = {r_rb:.4f} (95% bootstrap CI [{ci_lo:.4f}, {ci_hi:.4f}])")
    # recorded (original, WRONG test): Welch t=8.9288, p=2.7673e-14, n=67/67
    print("# recorded (original Welch t-test): t=8.9288, p=2.7673e-14, n=67/67")

    if p_val < 0.05:
        print("Conclusion: Significant difference in local fragment density between merge sites and correctly reconstructed control regions.")
    else:
        print("Conclusion: No significant difference in local fragment density between merge sites and correctly reconstructed control regions.")

if __name__ == "__main__":
    main()
