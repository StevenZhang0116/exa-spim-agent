import sys
import subprocess
import importlib

def install(package):
    print(f"Installing {package}...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", package])

# Proactively install common dependencies to prevent unpickling failures
packages_to_check = ["psutil", "pandas", "tensorstore", "scipy", "matplotlib", "networkx", "tqdm"]
for pkg in packages_to_check:
    try:
        importlib.import_module(pkg)
    except ImportError:
        install(pkg)

try:
    import agentic_neuron_proofreader
except ImportError:
    # Download using zip URL to bypass any missing git dependencies
    install("https://github.com/AllenInstitute/agentic-neuron-proofreader/archive/refs/heads/main.zip")
    import agentic_neuron_proofreader

import pickle
import numpy as np
import glob
import matplotlib.pyplot as plt
from collections import defaultdict
import scipy.sparse as sp
from scipy.sparse.csgraph import shortest_path
from scipy.stats import ks_2samp
import gc

def load_pickle_robustly(path):
    """
    Loads the pickle while dynamically catching and installing any missing 
    deep dependencies triggered during class reconstruction.
    """
    while True:
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except ModuleNotFoundError as e:
            missing_mod = e.name.split('.')[0]
            print(f"Missing module '{missing_mod}' detected during unpickling. Installing...")
            install(missing_mod)

def get_roots(N, S_nodes, dist, pred):
    """
    Fast O(N) tree traceback to find the closest sampled root for every node,
    safeguarded against infinite loops from cycles caused by 0-length edges.
    """
    roots = np.full(N+1, -1, dtype=np.int32)
    roots[S_nodes] = S_nodes
    in_path = np.zeros(N+1, dtype=bool)
    
    for i in range(N):
        if roots[i] != -1 or dist[i] == np.inf:
            continue
            
        curr = i
        path = []
        while roots[curr] == -1:
            if in_path[curr]:
                # Cycle detected!
                curr = -9999
                break
            in_path[curr] = True
            path.append(curr)
            curr = pred[curr]
            if curr == -9999:
                break
                
        if curr == -9999:
            r = -1
        else:
            r = roots[curr]
            
        for node in path:
            roots[node] = r
            in_path[node] = False
            
    return roots

def main():
    # Resolve dataset paths (checks multiple possible locations)
    paths = glob.glob("../dataset_cache_*_add.pkl")
    if not paths:
        paths = glob.glob("dataset_cache_*_add.pkl")
    if not paths:
        paths = glob.glob("cache/dataset_cache_*_add.pkl")
    if not paths:
        print("No datasets found.")
        return

    EDGE_SPLIT = 1
    NUM_PERMUTATIONS = 100  # Reduced to prevent timeout

    all_observed_dists = []
    all_null_dists = []

    for path in paths:
        print(f"Loading {path}...")
        payload = load_pickle_robustly(path)
            
        gt = payload["gt_graph"]
        edge_error = np.asarray(payload["gt_edge_error"])
        edges = list(gt.edges)
        node_xyz = gt.node_xyz
        
        # Group edge indices by GT neuron
        neuron_to_edges = defaultdict(list)
        for k, (u, v) in enumerate(edges):
            neuron_u = gt.node_segment_id(u)
            neuron_to_edges[neuron_u].append(k)
            
        for neuron, edge_indices in neuron_to_edges.items():
            split_locals = [i for i, k in enumerate(edge_indices) if edge_error[k] == EDGE_SPLIT]
            if len(split_locals) < 2:
                continue  # Need at least 2 splits to compute inter-split distance
                
            # Subgraph setup for the specific neuron
            unique_nodes = set()
            for k in edge_indices:
                u, v = edges[k]
                unique_nodes.add(u)
                unique_nodes.add(v)
                
            V = len(unique_nodes)
            E_num = len(edge_indices)
            node_to_idx = {n: i for i, n in enumerate(unique_nodes)}
            
            aug_u, aug_v, aug_w = [], [], []
            
            # Construct Augmented Graph: 
            # Nodes 0 to V-1: Original Nodes
            # Nodes V to V+E_num-1: Edge midpoints (representing the edges themselves)
            for i, k in enumerate(edge_indices):
                u, v = edges[k]
                idx_u = node_to_idx[u]
                idx_v = node_to_idx[v]
                m = V + i
                
                L = np.linalg.norm(node_xyz[u] - node_xyz[v])
                
                # Create edges pointing from original nodes to midpoints
                aug_u.extend([idx_u, idx_v])
                aug_v.extend([m, m])
                aug_w.extend([L/2.0, L/2.0])
                
            aug_u = np.array(aug_u, dtype=np.int32)
            aug_v = np.array(aug_v, dtype=np.int32)
            aug_w = np.array(aug_w, dtype=np.float64)
            
            all_aug_u = np.concatenate([aug_u, aug_v])
            all_aug_v = np.concatenate([aug_v, aug_u])
            all_aug_w = np.concatenate([aug_w, aug_w])
            
            N = V + E_num
            # Base undirected CSR without super source
            base_coo = sp.coo_matrix((all_aug_w, (all_aug_u, all_aug_v)), shape=(N+1, N+1))
            base_csr = base_coo.tocsr()
            
            # Preallocate the mutated CSR structure to reuse heavily inside the loop
            nnz_base = base_csr.nnz
            num_splits = len(split_locals)
            nnz_total = nnz_base + num_splits
            
            new_data = np.empty(nnz_total, dtype=np.float64)
            new_data[:nnz_base] = base_csr.data
            new_data[nnz_base:] = 0.0
            
            new_indices = np.empty(nnz_total, dtype=np.int32)
            new_indices[:nnz_base] = base_csr.indices
            
            new_indptr = np.empty(N+2, dtype=np.int32)
            new_indptr[:N+1] = base_csr.indptr[:N+1]
            new_indptr[N+1] = nnz_total
            
            def compute_distances_for_samples(sample_indices):
                S_nodes = np.array([V + s for s in sample_indices], dtype=np.int32)
                
                # Mutate in-place to append edges from super source 'N' to the sampled nodes
                new_indices[nnz_base:] = S_nodes
                
                adj = sp.csr_matrix((new_data, new_indices, new_indptr), shape=(N+1, N+1), copy=False)
                dist, pred = shortest_path(csgraph=adj, directed=True, indices=N, return_predecessors=True)
                
                # Reconstruct Voronoi cells per sampled edge midpoint
                roots = get_roots(N, S_nodes, dist, pred)
                
                c_u = roots[aug_u]
                c_v = roots[aug_v]
                
                # Detect cell boundaries (collisions between different split instances)
                valid_mask = (c_u != -1) & (c_v != -1) & (c_u != c_v)
                
                diff_u = c_u[valid_mask]
                diff_v = c_v[valid_mask]
                diff_w = aug_w[valid_mask]
                diff_dist_u = dist[aug_u[valid_mask]]
                diff_dist_v = dist[aug_v[valid_mask]]
                
                total_dists = diff_dist_u + diff_dist_v + diff_w
                
                min_other_dist = np.full(N+1, np.inf)
                np.minimum.at(min_other_dist, diff_u, total_dists)
                np.minimum.at(min_other_dist, diff_v, total_dists)
                
                return min_other_dist[S_nodes]
                
            # Analyze observed layout
            obs_dists = compute_distances_for_samples(split_locals)
            obs_finite = obs_dists[np.isfinite(obs_dists)]
            all_observed_dists.extend(obs_finite)
            
            # Analyze simulated random layouts 
            for _ in range(NUM_PERMUTATIONS):
                s_rand = np.random.choice(E_num, num_splits, replace=False)
                null_dists = compute_distances_for_samples(s_rand)
                null_finite = null_dists[np.isfinite(null_dists)]
                all_null_dists.extend(null_finite)

        # Free memory associated with the multi-GB payload before next iteration
        del payload
        del gt
        del edges
        del node_xyz
        gc.collect()
                
    if len(all_observed_dists) == 0:
        print("No valid split pairs found in the dataset.")
        return
    if len(all_null_dists) == 0:
        print("No valid null splits found in the dataset.")
        return

    # Generate statistics & visualizations
    obs_sorted = np.sort(all_observed_dists)
    null_sorted = np.sort(all_null_dists)

    obs_cdf = np.arange(1, len(obs_sorted) + 1) / len(obs_sorted)
    null_cdf = np.arange(1, len(null_sorted) + 1) / len(null_sorted)

    stat, pval = ks_2samp(all_observed_dists, all_null_dists)

    print("\n=== Split Cascading Analysis ===")
    print(f"Total observed splits analyzed: {len(all_observed_dists)}")
    print(f"Total null random samples: {len(all_null_dists)}")
    print(f"KS test D-statistic: {stat:.4f}")
    print(f"KS test p-value: {pval:.4e}")
    print(f"Mean observed inter-split distance: {np.mean(all_observed_dists):.2f} um")
    print(f"Mean random inter-split distance: {np.mean(all_null_dists):.2f} um")
    print(f"Median observed: {np.median(all_observed_dists):.2f} um")
    print(f"Median random: {np.median(all_null_dists):.2f} um")

    plt.figure(figsize=(8, 6))
    plt.plot(obs_sorted, obs_cdf, label='Observed Splits', color='red', lw=2)
    plt.plot(null_sorted, null_cdf, label='Null Model (Random)', color='blue', lw=2, linestyle='--')
    plt.xlabel("Nearest Neighbor Inter-split distance (um)")
    plt.ylabel("Cumulative Probability")
    plt.title(f"CDF of Nearest Neighbor Inter-split Distances\nKS Stat: {stat:.4f}, p-val: {pval:.4e}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
