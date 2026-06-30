# CORRECTED TEST for hypothesis id 59 (high local tortuosity -> more split errors,
# 10-edge window).
#
# FAULTS (verifier MINOR, concrete test faults): (1) HUGE-n significance of a
# TRIVIAL effect -- the p~0 is driven entirely by the >1M-edge sample; the
# point-biserial r=0.0335 (~0.1% of variance) means curvature explains almost
# nothing. (2) EDGE NON-INDEPENDENCE -- edges within a neuron are spatially
# autocorrelated, further inflating the nominal significance, so the naive
# Mann-Whitney p is not a valid significance test of an effect among independent
# units.
#
# CORRECTION: keep the SAME tortuosity quantities and direction but (1) report an
# EFFECT SIZE -- Cliff's delta (rank-biserial) with a CLUSTER bootstrap 95% CI
# (resampling whole NEURONS, the independent unit) -- and (2) replace the naive p
# with a NEURON-CLUSTER permutation p-value (permute the split/correct label at
# the neuron level). The original point-biserial r is still printed for continuity.
#
# ORIGINAL RECORDED NUMBERS (for the driver to compare):
#   6611 split / 1091075 correct, median 1.1115 vs 1.0764, MW U=4.5781e9 p~0,
#   point-biserial r=0.0335.

import glob
import os
import pickle
import numpy as np
import gc
from scipy.stats import mannwhitneyu, pointbiserialr
import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", package])

for pkg in ["psutil", "pandas", "tensorstore", "networkx", "scipy"]:
    try:
        __import__(pkg)
    except ImportError:
        install(pkg)

try:
    import agentic_neuron_proofreader
except ImportError:
    install("https://github.com/AllenInstitute/agentic-neuron-proofreader/archive/main.zip")
    import agentic_neuron_proofreader

def get_paths_of_length(G, length):
    paths = []
    adj = {n: list(G.neighbors(n)) for n in G.nodes()}
    for start_node in G.nodes():
        stack = [(start_node, [start_node])]
        while stack:
            curr, path = stack.pop()
            if len(path) == length + 1:
                if start_node < curr:
                    paths.append(path)
                continue
            path_len = len(path)
            for neighbor in adj[curr]:
                if path_len == 1 or neighbor != path[-2]:
                    stack.append((neighbor, path + [neighbor]))
    return paths

def cliffs_delta_sub(a, b, rng, n_sub=4000):
    a_s = a if len(a) <= n_sub else rng.choice(a, n_sub, replace=False)
    b_s = b if len(b) <= n_sub else rng.choice(b, n_sub, replace=False)
    gt_ = sum((x > b_s).sum() for x in a_s)
    lt_ = sum((x < b_s).sum() for x in a_s)
    return (gt_ - lt_) / (len(a_s) * len(b_s))

def main():
    files = glob.glob("*_add.pkl")
    if not files:
        print("No _add.pkl files found.")
        return

    split_tort, correct_tort = [], []
    split_neuron, correct_neuron = [], []

    for fpath in files:
        with open(fpath, "rb") as f:
            payload = pickle.load(f)

        gt = payload["gt_graph"]
        edge_error = np.asarray(payload["gt_edge_error"])
        del payload
        gc.collect()

        edges_list = list(gt.edges)

        node_xyz = {}
        for n in gt.nodes:
            if 'xyz' in gt.nodes[n]:
                node_xyz[n] = gt.nodes[n]['xyz']
            elif hasattr(gt, 'node_xyz'):
                node_xyz[n] = gt.node_xyz[n]
            else:
                raise ValueError("Could not find xyz coordinates for node.")

        edge_info = {}
        for idx, (u, v) in enumerate(edges_list):
            fs = frozenset((u, v))
            err = edge_error[idx]
            p_u = np.array(node_xyz[u])
            p_v = np.array(node_xyz[v])
            dist = np.linalg.norm(p_u - p_v)
            # Record neuron id (cluster) for this edge via the u endpoint.
            edge_info[fs] = {'dist': dist, 'error': err, 'neuron': gt.node_segment_id(u)}

        paths = get_paths_of_length(gt, 10)
        edge_tortuosities = {fs: [] for fs in edge_info}

        for p in paths:
            path_len = 0.0
            for i in range(10):
                fs = frozenset((p[i], p[i + 1]))
                if fs in edge_info:
                    path_len += edge_info[fs]['dist']
            p0 = np.array(node_xyz[p[0]])
            p10 = np.array(node_xyz[p[-1]])
            euclidean = np.linalg.norm(p0 - p10)
            tortuosity = path_len / euclidean if euclidean > 1e-6 else 1.0
            e4 = frozenset((p[4], p[5]))
            e5 = frozenset((p[5], p[6]))
            if e4 in edge_tortuosities:
                edge_tortuosities[e4].append(tortuosity)
            if e5 in edge_tortuosities:
                edge_tortuosities[e5].append(tortuosity)

        for fs, tort_list in edge_tortuosities.items():
            if not tort_list:
                continue
            avg_tort = np.mean(tort_list)
            err = edge_info[fs]['error']
            neuron = edge_info[fs]['neuron']
            if err == 1:
                split_tort.append(avg_tort); split_neuron.append(neuron)
            elif err == 0:
                correct_tort.append(avg_tort); correct_neuron.append(neuron)

        del gt, edge_info, edge_tortuosities, paths
        gc.collect()

    if not split_tort or not correct_tort:
        print("Not enough data to compare.")
        return

    split_tort = np.array(split_tort)
    correct_tort = np.array(correct_tort)
    split_neuron = np.array(split_neuron)
    correct_neuron = np.array(correct_neuron)

    print(f"Number of split edges analyzed: {len(split_tort)}")
    print(f"Number of correct edges analyzed: {len(correct_tort)}")
    print(f"Median Tortuosity (Split)   : {np.median(split_tort):.4f}")
    print(f"Median Tortuosity (Correct) : {np.median(correct_tort):.4f}")

    # Naive tests echoed for comparison only.
    stat, p_val = mannwhitneyu(split_tort, correct_tort, alternative='greater')
    labels = np.concatenate([np.ones_like(split_tort), np.zeros_like(correct_tort)])
    values = np.concatenate([split_tort, correct_tort])
    r, p_r = pointbiserialr(labels, values)
    print(f"\nNaive Mann-Whitney U={stat:.4e}, p={p_val:.4e} (for comparison only)")
    print(f"Point-biserial r={r:.4f} (effect size, as in original)")

    # === EFFECT SIZE: Cliff's delta with CLUSTER (neuron) bootstrap 95% CI ===
    rng = np.random.default_rng(42)
    delta = cliffs_delta_sub(split_tort, correct_tort, rng)
    # Cluster bootstrap: resample whole neurons.
    import pandas as pd
    s_df = pd.DataFrame({"neuron": split_neuron, "t": split_tort})
    c_df = pd.DataFrame({"neuron": correct_neuron, "t": correct_tort})
    s_by = {k: g["t"].values for k, g in s_df.groupby("neuron")}
    c_by = {k: g["t"].values for k, g in c_df.groupby("neuron")}
    s_keys = np.array(list(s_by.keys()), dtype=object)
    c_keys = np.array(list(c_by.keys()), dtype=object)
    boots = []
    for _ in range(300):
        bs = np.concatenate([s_by[k] for k in rng.choice(s_keys, len(s_keys), replace=True)])
        bc = np.concatenate([c_by[k] for k in rng.choice(c_keys, len(c_keys), replace=True)])
        boots.append(cliffs_delta_sub(bs, bc, rng, n_sub=2000))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    print("\n=== Effect size (cluster-aware) ===")
    print(f"Cliff's delta (split vs correct tortuosity) = {delta:.4f}")
    print(f"Cluster bootstrap 95% CI = [{lo:.4f}, {hi:.4f}]  (small delta => trivial effect)")

    # === NEURON-CLUSTER permutation p-value (neuron = independent unit) ===
    # Per neuron: mean tortuosity and whether it carries a split. Permute the
    # split/correct neuron labels and recompute the median-tortuosity gap.
    s_means = np.array([np.mean(v) for v in s_by.values()])
    c_means = np.array([np.mean(v) for v in c_by.values()])
    obs = np.median(s_means) - np.median(c_means)
    pooled = np.concatenate([s_means, c_means])
    n_s = len(s_means)
    n_perm = 5000
    perm = np.empty(n_perm)
    for i in range(n_perm):
        idx = rng.permutation(len(pooled))
        perm[i] = np.median(pooled[idx[:n_s]]) - np.median(pooled[idx[n_s:]])
    perm_p = (np.sum(perm >= obs) + 1) / (n_perm + 1)
    print("\n=== Neuron-cluster permutation test ===")
    print(f"Split-carrying neurons={n_s}, correct neurons={len(c_means)}")
    print(f"Observed neuron-level median tortuosity gap = {obs:.4f}")
    print(f"Cluster-permutation one-sided p (split > correct) = {perm_p:.4g}")
    print("(Compare to ORIGINAL: MW p~0, r=0.0335 -- significance was n-driven.)")

if __name__ == "__main__":
    main()
