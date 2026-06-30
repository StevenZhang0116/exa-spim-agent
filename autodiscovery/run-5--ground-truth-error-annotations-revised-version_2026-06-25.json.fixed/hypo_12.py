import os
import sys
import pickle
import numpy as np
import re
from collections import defaultdict

import agentic_neuron_proofreader

from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

# Load the provided dataset directly
print("Loading dataset from:", os.environ["RERUN_PKL"])
pkl_files = [os.environ["RERUN_PKL"]]

if not pkl_files:
    print("No dataset files found.")
    sys.exit(1)

all_gaps = []

# Extract gap distances for all true splits
for pkl_file in pkl_files:
    print(f"Processing {pkl_file}...")
    try:
        with open(pkl_file, "rb") as f:
            payload = pickle.load(f)
    except Exception as e:
        print(f"Error loading {pkl_file}: {e}")
        continue

    gt = payload["gt_graph"]
    fg = payload["fragments_graph"]
    node_label = np.asarray(payload["gt_node_canonical_label"])

    comp_to_swc = fg.component_id_to_swc_id
    node_comp = fg.node_component_id
    node_xyz = fg.node_xyz

    valid_segment_ids = set(int(x) for x in node_label if x != 0)

    swc_to_endpoints = defaultdict(list)
    all_endpoints = []

    # Get node degrees from networkx graph
    try:
        fg_degrees = dict(fg.degree)
    except TypeError:
        fg_degrees = dict(fg.degree())

    # Precompute fragment endpoints (nodes with degree <= 1)
    for n, d in fg_degrees.items():
        if d <= 1:
            xyz = node_xyz[n]
            all_endpoints.append(xyz)

            comp_id = node_comp[n]
            swc_id = comp_to_swc.get(comp_id) if isinstance(comp_to_swc, dict) else comp_to_swc[comp_id]
            if swc_id is not None:
                # extract all numeric values to match against U-Net segment IDs safely
                nums = re.findall(r'\d+', str(swc_id))
                for num in nums:
                    clean_id = int(num)
                    if clean_id in valid_segment_ids:
                        swc_to_endpoints[clean_id].append(xyz)

    # Convert endpoints to KD-Trees for fast spatial mapping
    swc_to_kdtree = {}
    for k, v in swc_to_endpoints.items():
        if len(v) > 0:
            swc_to_kdtree[k] = cKDTree(np.array(v))

    if len(all_endpoints) > 0:
        global_kdtree = cKDTree(np.array(all_endpoints))
    else:
        print("No endpoints found in fragments_graph.")
        continue

    split_pairs = 0
    matched_by_id = 0

    # Identify true splits in ground truth edges
    for u, v in gt.edges:
        lab_u = int(node_label[u])
        lab_v = int(node_label[v])

        # A 'true split' is an edge where adjacent nodes belong to different valid segment IDs
        if lab_u != lab_v and lab_u != 0 and lab_v != 0:
            split_pairs += 1
            xyz_u = gt.node_xyz[u]
            xyz_v = gt.node_xyz[v]

            # Map GT node `u` to its closest fragment endpoint (filtered by ID if possible, else global spatial closest)
            if lab_u in swc_to_kdtree:
                dist_u, idx_u = swc_to_kdtree[lab_u].query(xyz_u)
                ep_u = swc_to_endpoints[lab_u][idx_u]
                matched_by_id += 1
            else:
                dist_u, idx_u = global_kdtree.query(xyz_u)
                ep_u = all_endpoints[idx_u]

            # Map GT node `v` to its closest fragment endpoint
            if lab_v in swc_to_kdtree:
                dist_v, idx_v = swc_to_kdtree[lab_v].query(xyz_v)
                ep_v = swc_to_endpoints[lab_v][idx_v]
                matched_by_id += 1
            else:
                dist_v, idx_v = global_kdtree.query(xyz_v)
                ep_v = all_endpoints[idx_v]

            # Compute pairwise Euclidean distances between the corresponding fragment endpoints
            gap_size = np.linalg.norm(ep_u - ep_v)
            all_gaps.append(gap_size)

    print(f"Found {split_pairs} true split edges. Matched {matched_by_id} out of {split_pairs * 2} fragment endpoints explicitly by segment ID.")

    # Memory cleanup for next dataset in iteration
    del payload, gt, fg, node_label, comp_to_swc, node_comp, node_xyz, swc_to_endpoints, all_endpoints, swc_to_kdtree, global_kdtree

print(f"\nTotal valid split gaps found: {len(all_gaps)}")

if len(all_gaps) < 2:
    print("Not enough split gaps found across datasets to fit a 2-component GMM.")
    sys.exit(0)

X = np.array(all_gaps).reshape(-1, 1)

# ----------------------------------------------------------------------
# Original (recorded) "test": a 2-component GMM was FIT and its two
#   components reported (Comp1 weight 0.6357 mean ~0.0 um std ~0.001;
#   Comp2 weight 0.3643 mean 159.18 um std 259.01 um, n = 7,988).
# Fault flagged: the k=2 GMM is IMPOSED, not SELECTED. Fitting k=2 to any
#   continuous distribution returns two components regardless of true
#   modality, so "bimodality" is asserted by construction. No model-selection
#   statistic (BIC/AIC vs k=1, or a dip test) was reported.
# CORRECTION: add an explicit MODEL-SELECTION TEST for the number of modes:
#   1. Fit GMMs for k = 1..5 and compare BIC and AIC (lower = better). If
#      k=2 (or more) genuinely beats k=1, bimodality is supported; if k=1
#      wins, the "two distinct modes" claim is not supported.
#   2. Likelihood-ratio statistic 2*(LL_2 - LL_1) for k=2 vs k=1.
#   3. Hartigan-style dip test for unimodality (implemented inline) on the
#      empirical CDF: large dip => reject unimodality.
# The original 2-component fit is still reported for comparison.
# ----------------------------------------------------------------------

print("\n=== CORRECTED: model-selection test for number of modes ===")
ks = [1, 2, 3, 4, 5]
bics, aics, lls = {}, {}, {}
fitted = {}
for k in ks:
    g = GaussianMixture(n_components=k, random_state=42, n_init=3)
    g.fit(X)
    fitted[k] = g
    bics[k] = g.bic(X)
    aics[k] = g.aic(X)
    lls[k] = g.score(X) * len(X)  # total log-likelihood
    print(f"  k={k}: BIC={bics[k]:.1f}  AIC={aics[k]:.1f}  logL={lls[k]:.1f}")

best_bic_k = min(ks, key=lambda k: bics[k])
best_aic_k = min(ks, key=lambda k: aics[k])
print(f"Best k by BIC = {best_bic_k}; best k by AIC = {best_aic_k}")
print(f"BIC(k=1) - BIC(k=2) = {bics[1]-bics[2]:.1f} (positive => k=2 preferred over k=1)")
lr_stat = 2.0 * (lls[2] - lls[1])
print(f"Likelihood-ratio 2*(LL_2 - LL_1) = {lr_stat:.1f} (large => k=2 fits much better)")
print(f"Bimodality supported by model selection: {best_bic_k >= 2 or best_aic_k >= 2}")

# --- Hartigan dip test for unimodality (inline implementation) ---
def dip_test(x):
    """Return Hartigan's dip statistic for a 1D sample (test of unimodality)."""
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    if n < 4:
        return 0.0
    # Empirical CDF values at the sorted points.
    xcdf = np.arange(1, n + 1) / n
    # Greatest convex minorant (GCM) and least concave majorant (LCM) of the ECDF.
    def gcm(xv, yv):
        # lower convex hull of points (xv, yv)
        idx = [0]
        for i in range(1, len(xv)):
            while len(idx) >= 2:
                j, k = idx[-2], idx[-1]
                # cross product to keep convex (lower) hull
                if (yv[i] - yv[j]) * (xv[k] - xv[j]) <= (yv[k] - yv[j]) * (xv[i] - xv[j]):
                    idx.pop()
                else:
                    break
            idx.append(i)
        return idx
    def lcm(xv, yv):
        idx = gcm(xv, -yv)
        return idx
    lower = np.concatenate(([0.0], xcdf[:-1]))  # ECDF jumps: value just left of each point
    upper = xcdf
    gi = gcm(x, lower)
    li = lcm(x, upper)
    g_interp = np.interp(x, x[gi], lower[gi])
    l_interp = np.interp(x, x[li], upper[li])
    dip = np.max(np.abs(upper - g_interp))
    dip = max(dip, np.max(np.abs(lower - l_interp)))
    return dip / 2.0

obs_dip = dip_test(X.ravel())
# Bootstrap p-value for the dip test against a uniform-null reference scaled to n.
rng = np.random.default_rng(42)
nb = 500
null_dips = np.empty(nb)
n = len(X)
for i in range(nb):
    u = rng.uniform(0, 1, size=n)
    null_dips[i] = dip_test(u)
dip_p = (np.sum(null_dips >= obs_dip) + 1) / (nb + 1)
print(f"Hartigan dip statistic = {obs_dip:.6f}, bootstrap p (unimodal null) = {dip_p:.4f} "
      f"(p<0.05 => reject unimodality)")

# --- Report the original 2-component GMM fit for comparison ---
gmm = fitted[2]
print("\n--- GMM Component Parameters (k=2, original fit) ---")
for i in range(gmm.n_components):
    weight = gmm.weights_[i]
    mean = gmm.means_[i, 0]
    covar = gmm.covariances_[i, 0, 0]
    print(f"Component {i+1}:")
    print(f"  Weight: {weight:.4f}")
    print(f"  Mean:   {mean:.4f} µm")
    print(f"  StdDev: {np.sqrt(covar):.4f} µm")

# Plot the density distribution
plt.figure(figsize=(10, 6))
plt.hist(all_gaps, bins=50, density=True, alpha=0.6, color='skyblue', edgecolor='black', label='Gap Sizes Histogram')

# To safely evaluate the pdf across the range of data
x_min, x_max = min(all_gaps), max(all_gaps)
if x_min == x_max:
    x_min, x_max = x_min - 1, x_max + 1

x_range = np.linspace(x_min, x_max, 1000).reshape(-1, 1)
logprob = gmm.score_samples(x_range)
pdf = np.exp(logprob)
plt.plot(x_range, pdf, '-k', linewidth=2, label='GMM Total PDF')

for i in range(gmm.n_components):
    weight = gmm.weights_[i]
    mean = gmm.means_[i, 0]
    covar = gmm.covariances_[i, 0, 0]
    if covar > 0:
        comp_pdf = weight * (1.0 / np.sqrt(2 * np.pi * covar)) * np.exp(-0.5 * ((x_range.flatten() - mean) ** 2) / covar)
        plt.plot(x_range, comp_pdf, '--', linewidth=2, label=f'Component {i+1} (mean: {mean:.2f})')

plt.title('Density Distribution of Split Gap Distances')
plt.xlabel('Euclidean Distance (µm)')
plt.ylabel('Density')
plt.legend()
plt.tight_layout()
plt.show()
