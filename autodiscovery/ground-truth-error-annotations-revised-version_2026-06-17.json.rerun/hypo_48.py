import os
import glob
import pickle
import numpy as np
import scipy.stats as stats
import subprocess
import sys
import gc
from collections import defaultdict

def install_deps():
    # Install required dependencies explicitly as they are required by agentic_neuron_proofreader utils
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "psutil", "tensorstore"])
    try:
        import agentic_neuron_proofreader
    except ImportError:
        url = "https://github.com/AllenInstitute/agentic-neuron-proofreader/archive/refs/heads/main.zip"
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", url])

install_deps()
import agentic_neuron_proofreader

def main():
    # Locate dataset caches based on expected file structure
    files = []
    for pattern in [
        "../*_add.pkl",
        "../cache/*_add.pkl",
        "cache/*_add.pkl",
        "*_add.pkl"
    ]:
        files = glob.glob(pattern)
        if files:
            break
            
    if not files:
        print("No dataset files found.")
        return
        
    total_edges = 0
    total_omit_edges = 0
    transition_matrix = np.zeros((2, 2), dtype=int)
    
    for fpath in sorted(files):
        with open(fpath, "rb") as f:
            payload = pickle.load(f)
        
        gt = payload["gt_graph"]
        edge_error = np.asarray(payload["gt_edge_error"])
        
        total_edges += len(edge_error)
        total_omit_edges += np.count_nonzero(edge_error == 2)
        
        # Map incident edges to states (0: OTHER, 1: OMIT) per node
        node_states = defaultdict(list)
        for (u, v), err in zip(gt.edges, edge_error):
            state = 1 if err == 2 else 0
            node_states[u].append(state)
            node_states[v].append(state)
            
        # Efficiently compute combinations of adjacent edges for the transition matrix
        for states in node_states.values():
            n_states = len(states)
            if n_states < 2:
                continue
                
            c1 = sum(states)
            c0 = n_states - c1
            
            # For each pair of adjacent edges at this node, we add to the symmetric matrix:
            # Since we count undirected transitions both ways [i,j] and [j,i], we effectively
            # add 2 for identical state pairs, which resolves to c * (c - 1)
            transition_matrix[0, 0] += c0 * (c0 - 1)
            transition_matrix[1, 1] += c1 * (c1 - 1)
            transition_matrix[0, 1] += c0 * c1
            transition_matrix[1, 0] += c0 * c1
            
        # Aggressively release memory before loading the next graph to prevent OOM/timeouts
        del payload
        del gt
        del edge_error
        del node_states
        gc.collect()

    # 4. Compute the marginal probability of an edge being an OMIT error 
    marg_prob = total_omit_edges / total_edges if total_edges > 0 else 0
    
    # 5. Compute the conditional probability of an edge being an OMIT error given that an adjacent edge is an OMIT error.
    row1_sum = transition_matrix[1, 0] + transition_matrix[1, 1]
    cond_prob = transition_matrix[1, 1] / row1_sum if row1_sum > 0 else 0
    
    print("Edge-state transition matrix (0: OTHER, 1: OMIT):")
    print(transition_matrix)
    print(f"\nMarginal probability of OMIT: {marg_prob:.4f}")
    print(f"Conditional probability of OMIT given adjacent is OMIT: {cond_prob:.4f}")
    
    # 7. Compare the conditional probability to the marginal probability 
    if marg_prob > 0:
        ratio = cond_prob / marg_prob
        print(f"Ratio (Conditional / Marginal): {ratio:.2f}")
        if ratio >= 3:
            print("Hypothesis validated: Transition probability is at least 3 times higher than the marginal probability.")
        else:
            print("Hypothesis not validated: The ratio did not meet the 3x threshold.")
    
    # 6. Perform a chi-square test of independence on the transition counts to statistically confirm clustering.
    try:
        chi2, p_val, dof, expected = stats.chi2_contingency(transition_matrix)
        print(f"\nChi-square test statistic: {chi2:.4f}")
        print(f"p-value: {p_val:.4e}")
    except ValueError as e:
        print(f"\nCould not compute chi-square test: {e}")

if __name__ == "__main__":
    main()
