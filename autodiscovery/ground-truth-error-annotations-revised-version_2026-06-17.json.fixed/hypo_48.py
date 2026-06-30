import os
import glob
import pickle
import numpy as np
import scipy.stats as stats
import subprocess
import sys
import gc
from collections import defaultdict

# ---------------------------------------------------------------------------
# CORRECTED TEST (id 48)
# Original (recorded/rerun): chi-square test of independence on a 2x2 edge-state
#   transition matrix [[2727004,9999],[9999,81558]],
#   chi2=2226077.8739, p=0.0000e+00; descriptive marginal OMIT=0.0317,
#   conditional=0.8908, ratio=28.09.
# WHY WRONG: each edge is reused in many adjacent-pair "observations" (a node of
#   degree d contributes d*(d-1) ordered pairs), so the ~2.8M-cell transition
#   total derives from only ~1.4M edges. The pairs are massively
#   pseudo-replicated / non-independent, so the chi-square statistic is inflated
#   and its p-value is NOT interpretable as a formal independence test.
# FIX: keep the (valid) descriptive conditional/marginal ratio, but obtain a
#   proper p-value from a PERMUTATION test that respects the EDGE as the unit of
#   independence: randomly permute the per-edge OMIT labels across all edges
#   (preserving the marginal omit count) and recompute the conditional/marginal
#   ratio many times; the permutation null destroys spatial contiguity while
#   keeping the same number of omit edges. Also report a block (edge-level)
#   bootstrap 95% CI for the observed ratio.
# ---------------------------------------------------------------------------

def install_deps():
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "psutil", "tensorstore"])
    try:
        import agentic_neuron_proofreader
    except ImportError:
        url = "https://github.com/AllenInstitute/agentic-neuron-proofreader/archive/refs/heads/main.zip"
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", url])

install_deps()
import agentic_neuron_proofreader


def transition_counts_from_labels(edge_omit, node_edge_idx):
    """Build the 2x2 adjacent-edge transition matrix from per-edge OMIT labels.

    ``edge_omit`` is a 0/1 array over edges; ``node_edge_idx`` maps each node to
    the list of incident edge indices. Counts the same ordered adjacent pairs as
    the original analysis (c*(c-1) on the diagonal, c0*c1 off-diagonal).
    """
    tm = np.zeros((2, 2), dtype=np.int64)
    for idxs in node_edge_idx:
        n_states = len(idxs)
        if n_states < 2:
            continue
        c1 = int(edge_omit[idxs].sum())
        c0 = n_states - c1
        tm[0, 0] += c0 * (c0 - 1)
        tm[1, 1] += c1 * (c1 - 1)
        tm[0, 1] += c0 * c1
        tm[1, 0] += c0 * c1
    return tm


def ratio_from_tm(tm, marg_prob):
    row1_sum = tm[1, 0] + tm[1, 1]
    cond_prob = tm[1, 1] / row1_sum if row1_sum > 0 else 0.0
    return (cond_prob / marg_prob) if marg_prob > 0 else 0.0, cond_prob


def main():
    rng = np.random.default_rng(42)

    # Loading fix: prefer $RERUN_PKL directly; harness also redirects pkl globs.
    rerun_pkl = os.environ.get("RERUN_PKL")
    if rerun_pkl and os.path.exists(rerun_pkl):
        files = [rerun_pkl]
    else:
        files = []
        for pattern in ["../*_add.pkl", "../cache/*_add.pkl", "cache/*_add.pkl", "*_add.pkl"]:
            files = glob.glob(pattern)
            if files:
                break
    if not files:
        print("No dataset files found.")
        return

    total_edges = 0
    total_omit_edges = 0

    # Collect per-edge OMIT labels and node->incident-edge-index mapping so we can
    # permute/bootstrap at the EDGE level (the true unit of independence).
    all_edge_omit = []
    node_edge_idx = []
    edge_offset = 0

    for fpath in sorted(files):
        print(f"Loading dataset from: {fpath}")
        with open(fpath, "rb") as f:
            payload = pickle.load(f)

        gt = payload["gt_graph"]
        edge_error = np.asarray(payload["gt_edge_error"])

        total_edges += len(edge_error)
        total_omit_edges += int(np.count_nonzero(edge_error == 2))

        edge_omit = (edge_error == 2).astype(np.int8)
        all_edge_omit.append(edge_omit)

        node_to_edges = defaultdict(list)
        for ei, (u, v) in enumerate(gt.edges):
            gidx = edge_offset + ei
            node_to_edges[u].append(gidx)
            node_to_edges[v].append(gidx)
        for u, idxs in node_to_edges.items():
            node_edge_idx.append(np.asarray(idxs, dtype=np.int64))
        edge_offset += len(edge_error)

        del payload, gt, edge_error
        gc.collect()

    edge_omit = np.concatenate(all_edge_omit).astype(np.int8)

    marg_prob = total_omit_edges / total_edges if total_edges > 0 else 0.0

    # Observed transition matrix and descriptive ratio (same quantity as recorded).
    tm = transition_counts_from_labels(edge_omit, node_edge_idx)
    obs_ratio, cond_prob = ratio_from_tm(tm, marg_prob)

    print("\nEdge-state transition matrix (0: OTHER, 1: OMIT):")
    print(tm)
    print(f"\nMarginal probability of OMIT: {marg_prob:.4f}")
    print(f"Conditional probability of OMIT given adjacent is OMIT: {cond_prob:.4f}")
    print(f"Ratio (Conditional / Marginal): {obs_ratio:.2f}")

    # --- Echo the original (unsound, pseudo-replicated) chi-square ---
    try:
        chi2, p_chi, dof, _ = stats.chi2_contingency(tm)
        print(f"\n[Original unsound] Pseudo-replicated chi-square: chi2={chi2:.4f}, p={p_chi:.4e} (NOT interpretable)")
    except ValueError as e:
        print(f"\n[Original unsound] Could not compute chi-square: {e}")
    # recorded: chi2=2226077.8739, p=0.0000e+00; ratio=28.09, marginal=0.0317, conditional=0.8908

    # --- CORRECTED p-value: edge-label PERMUTATION test on the ratio ---
    # Null: omit labels are spatially exchangeable across edges (same omit count).
    n_perm = 1000
    perm_ratios = np.empty(n_perm)
    base = edge_omit.copy()
    for i in range(n_perm):
        perm = rng.permutation(base)
        tm_p = transition_counts_from_labels(perm, node_edge_idx)
        perm_ratios[i], _ = ratio_from_tm(tm_p, marg_prob)
    # one-sided (clustering => ratio >> null); add-one correction
    perm_p = (np.sum(perm_ratios >= obs_ratio) + 1) / (n_perm + 1)
    null_mean = float(np.mean(perm_ratios))
    null_hi = float(np.percentile(perm_ratios, 97.5))

    print("\n--- CORRECTED: edge-label permutation test (respects edge as unit) ---")
    print(f"Observed ratio = {obs_ratio:.2f}; permutation null mean ratio = {null_mean:.3f} "
          f"(97.5th pct {null_hi:.3f}), n_perm={n_perm}")
    print(f"Permutation p-value (ratio >= observed) = {perm_p:.4e}")

    # --- Effect-size CI: block bootstrap resampling NODES (clusters of edges) ---
    n_boot = 1000
    n_nodes = len(node_edge_idx)
    boot_ratios = []
    for _ in range(n_boot):
        sel = rng.integers(0, n_nodes, size=n_nodes)
        tm_b = np.zeros((2, 2), dtype=np.int64)
        for s in sel:
            idxs = node_edge_idx[s]
            ns = len(idxs)
            if ns < 2:
                continue
            c1 = int(edge_omit[idxs].sum())
            c0 = ns - c1
            tm_b[1, 1] += c1 * (c1 - 1)
            tm_b[1, 0] += c0 * c1
        row1 = tm_b[1, 0] + tm_b[1, 1]
        cp = tm_b[1, 1] / row1 if row1 > 0 else 0.0
        boot_ratios.append((cp / marg_prob) if marg_prob > 0 else 0.0)
    boot_ratios = np.asarray(boot_ratios)
    b_lo, b_hi = np.percentile(boot_ratios, [2.5, 97.5])
    print(f"Conditional/marginal ratio 95% block-bootstrap CI [{b_lo:.2f}, {b_hi:.2f}] (node-cluster resample)")

    if perm_p < 0.05 and obs_ratio >= 3:
        print("Hypothesis validated: OMIT clustering ratio far exceeds the spatially-exchangeable null (>=3x), p<0.05 by permutation.")
    else:
        print("Hypothesis not validated under the corrected permutation test.")


if __name__ == "__main__":
    main()
