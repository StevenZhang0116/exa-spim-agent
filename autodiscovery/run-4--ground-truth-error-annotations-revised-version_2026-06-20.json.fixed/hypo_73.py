# CORRECTED TEST for hypothesis id 73 (split edges on more tortuous segments,
# 5-hop window).
#
# FAULTS (verifier MINOR, concrete test faults): same tiny-effect / over-power
# pattern as id 59 -- the median gap is ~0.03 and the p~0 comes from >1.1M edges;
# UNLIKE id 59, this run reports NO effect-size statistic, so the practical
# smallness is hidden, and edges are NON-INDEPENDENT, inflating the naive
# Mann-Whitney p.
#
# CORRECTION: keep the SAME 5-hop tortuosity quantities and direction but (1) add
# an EFFECT SIZE -- Cliff's delta (rank-biserial) with a CLUSTER bootstrap 95% CI
# (resampling whole NEURONS) -- and (2) replace the naive p with a NEURON-CLUSTER
# permutation p-value (permute the split/correct label at the neuron level).
#
# ORIGINAL RECORDED NUMBERS (for the driver to compare):
#   1109034 correct / 6805 split, median 1.0801 vs 1.1132, MW U=4714207979.0,
#   p=1.6599e-276 (no effect size reported originally).

import glob
import os
import sys
import pickle
import math
import numpy as np
import pandas as pd
import gc
from pathlib import Path

import agentic_neuron_proofreader

from scipy.stats import mannwhitneyu

def get_extremity(adj, start_node, prev_node, k, node_xyz):
    curr = start_node
    prev = prev_node
    path_len = 0.0
    for _ in range(k):
        neighbors = [n for n in adj[curr] if n != prev]
        if not neighbors:
            break
        nxt = neighbors[0]
        dx = node_xyz[nxt][0] - node_xyz[curr][0]
        dy = node_xyz[nxt][1] - node_xyz[curr][1]
        dz = node_xyz[nxt][2] - node_xyz[curr][2]
        path_len += math.sqrt(dx*dx + dy*dy + dz*dz)
        prev = curr
        curr = nxt
    return curr, path_len

def get_tortuosity(gt, edges, edge_errors, k=5):
    try:
        node_xyz = gt.node_xyz
        _ = node_xyz[next(iter(gt.nodes))]
    except (AttributeError, TypeError, IndexError):
        node_xyz = {n: gt.nodes[n].get('node_xyz', gt.nodes[n].get('xyz')) for n in gt.nodes}

    adj = {n: list(gt.neighbors(n)) for n in gt.nodes}

    correct_tort, split_tort = [], []
    correct_neuron, split_neuron = [], []

    for idx, (u, v) in enumerate(edges):
        err = edge_errors[idx]
        if err != 0 and err != 1:
            continue

        u_curr, u_len = get_extremity(adj, u, v, k, node_xyz)
        v_curr, v_len = get_extremity(adj, v, u, k, node_xyz)

        dx_edge = node_xyz[v][0] - node_xyz[u][0]
        dy_edge = node_xyz[v][1] - node_xyz[u][1]
        dz_edge = node_xyz[v][2] - node_xyz[u][2]
        edge_len = math.sqrt(dx_edge*dx_edge + dy_edge*dy_edge + dz_edge*dz_edge)

        path_length = u_len + v_len + edge_len

        dx_end = node_xyz[v_curr][0] - node_xyz[u_curr][0]
        dy_end = node_xyz[v_curr][1] - node_xyz[u_curr][1]
        dz_end = node_xyz[v_curr][2] - node_xyz[u_curr][2]
        euclidean_dist = math.sqrt(dx_end*dx_end + dy_end*dy_end + dz_end*dz_end)

        tortuosity = path_length / euclidean_dist if euclidean_dist > 1e-6 else 1.0
        neuron = gt.node_segment_id(u)  # cluster / independent unit

        if err == 0:
            correct_tort.append(tortuosity); correct_neuron.append(neuron)
        elif err == 1:
            split_tort.append(tortuosity); split_neuron.append(neuron)

    return correct_tort, split_tort, correct_neuron, split_neuron

def cliffs_delta_sub(a, b, rng, n_sub=4000):
    a_s = a if len(a) <= n_sub else rng.choice(a, n_sub, replace=False)
    b_s = b if len(b) <= n_sub else rng.choice(b, n_sub, replace=False)
    gt_ = sum((x > b_s).sum() for x in a_s)
    lt_ = sum((x < b_s).sum() for x in a_s)
    return (gt_ - lt_) / (len(a_s) * len(b_s))

def main():
    # Load the provided dataset directly from $RERUN_PKL
    print("Loading dataset from:", os.environ["RERUN_PKL"])
    files = [Path(os.environ["RERUN_PKL"])]

    if not files:
        print("No dataset files found matching *_add.pkl")
        sys.exit(1)

    all_correct, all_split = [], []
    all_correct_neuron, all_split_neuron = [], []

    for fpath in files:
        print(f"Loading {fpath}...")
        with open(fpath, "rb") as f:
            payload = pickle.load(f)

        gt = payload["gt_graph"]
        edge_error = np.asarray(payload["gt_edge_error"])
        edges = list(gt.edges)

        c_tort, s_tort, c_neu, s_neu = get_tortuosity(gt, edges, edge_error, k=5)
        all_correct.extend(c_tort); all_split.extend(s_tort)
        all_correct_neuron.extend(c_neu); all_split_neuron.extend(s_neu)

        del payload, gt, edge_error, edges
        gc.collect()

    print(f"\nProcessed {len(all_correct)} correct edges and {len(all_split)} split edges.")

    if len(all_correct) == 0 or len(all_split) == 0:
        print("Not enough edges to compare.")
        return

    all_correct = np.array(all_correct); all_split = np.array(all_split)
    all_correct_neuron = np.array(all_correct_neuron); all_split_neuron = np.array(all_split_neuron)

    print(f"Median Tortuosity for CORRECT edges: {np.median(all_correct):.6f}")
    print(f"Median Tortuosity for SPLIT edges:   {np.median(all_split):.6f}")

    # Naive test echoed for comparison only.
    stat, p_val = mannwhitneyu(all_split, all_correct, alternative='greater')
    print(f"\nNaive Mann-Whitney U={stat}, p={p_val:.4e} (for comparison only)")

    # === EFFECT SIZE: Cliff's delta with CLUSTER (neuron) bootstrap 95% CI ===
    rng = np.random.default_rng(42)
    delta = cliffs_delta_sub(all_split, all_correct, rng)
    s_df = pd.DataFrame({"neuron": all_split_neuron, "t": all_split})
    c_df = pd.DataFrame({"neuron": all_correct_neuron, "t": all_correct})
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
    print(f"Cliff's delta (split vs correct tortuosity) = {delta:.4f}  (small => trivial effect)")
    print(f"Cluster bootstrap 95% CI = [{lo:.4f}, {hi:.4f}]")

    # === NEURON-CLUSTER permutation p-value ===
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
    print("(Compare to ORIGINAL: MW p=1.6599e-276, no effect size reported.)")

if __name__ == "__main__":
    main()
