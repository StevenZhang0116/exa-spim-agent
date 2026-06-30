import sys
import subprocess
import glob
import os
import urllib.request
import zipfile

def install_and_import(package):
    try:
        __import__(package)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", package])

install_and_import('networkx')
install_and_import('scipy')
install_and_import('matplotlib')
install_and_import('numpy')
install_and_import('psutil')
install_and_import('pandas')
install_and_import('tqdm')
install_and_import('tensorstore')

try:
    import agentic_neuron_proofreader
except ImportError:
    temp_dir = "/tmp"
    repo_dir = os.path.join(temp_dir, "agentic-neuron-proofreader-main")
    if not os.path.exists(repo_dir):
        zip_path = os.path.join(temp_dir, "main.zip")
        urllib.request.urlretrieve("https://github.com/AllenInstitute/agentic-neuron-proofreader/archive/refs/heads/main.zip", zip_path)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", repo_dir])
    import agentic_neuron_proofreader

import pickle
import numpy as np
import networkx as nx
import scipy.stats as stats
import matplotlib.pyplot as plt

def main():
    dataset_paths = glob.glob("../dataset_cache_*_add.pkl")
    if not dataset_paths:
        dataset_paths = glob.glob("*_add.pkl")
        
    terminal_omit = 0
    terminal_non = 0
    internal_omit = 0
    internal_non = 0

    EDGE_OMIT = 2

    for path in dataset_paths:
        with open(path, "rb") as f:
            payload = pickle.load(f)

        gt_graph = payload["gt_graph"]
        edge_error = np.asarray(payload["gt_edge_error"])

        terminal_edges = set()

        # Identify terminal branches
        for node in gt_graph.nodes:
            if gt_graph.degree(node) == 1:
                curr = node
                prev = None
                # Traverse inwards until hitting a branch point (degree >= 3)
                while True:
                    if gt_graph.degree(curr) >= 3:
                        break
                        
                    neighbors = list(gt_graph.neighbors(curr))
                    if prev is not None and prev in neighbors:
                        neighbors.remove(prev)
                    
                    if not neighbors:
                        break
                        
                    nxt = neighbors[0]
                    # Store edge as frozenset so (u, v) == (v, u)
                    terminal_edges.add(frozenset([curr, nxt]))
                    
                    prev = curr
                    curr = nxt

        edges = list(gt_graph.edges)
        
        # Cross-tabulate edge types (terminal vs internal) with edge error status (Omit vs Non-Omit)
        for k, (u, v) in enumerate(edges):
            is_term = frozenset([u, v]) in terminal_edges
            is_omit = (edge_error[k] == EDGE_OMIT)
            
            if is_term:
                if is_omit:
                    terminal_omit += 1
                else:
                    terminal_non += 1
            else:
                if is_omit:
                    internal_omit += 1
                else:
                    internal_non += 1

    print("=== Omit Errors by Branch Location ===")
    print(f"Terminal edges: Omit={terminal_omit}, Non-Omit={terminal_non}")
    print(f"Internal edges: Omit={internal_omit}, Non-Omit={internal_non}")

    contingency_table = [[terminal_omit, terminal_non],
                         [internal_omit, internal_non]]

    # Perform chi-square contingency test
    chi2, p_val, dof, expected = stats.chi2_contingency(contingency_table)

    print("\n=== Chi-Square Contingency Test ===")
    print(f"Chi-square statistic: {chi2:.4f}")
    print(f"p-value: {p_val:.4e}")

    term_pct = 100 * terminal_omit / (terminal_omit + terminal_non) if (terminal_omit + terminal_non) > 0 else 0
    int_pct = 100 * internal_omit / (internal_omit + internal_non) if (internal_omit + internal_non) > 0 else 0

    print(f"\nTerminal Omit Percentage: {term_pct:.2f}%")
    print(f"Internal Omit Percentage: {int_pct:.2f}%")

    labels = ['Terminal Edges', 'Internal Edges']
    omit_pct = [term_pct, int_pct]

    plt.figure(figsize=(8, 6))
    bars = plt.bar(labels, omit_pct, color=['#FF9999', '#99CCFF'])
    plt.ylabel('% Omit Errors')
    plt.title('Percentage of Omit Errors: Terminal vs Internal Edges')
    
    max_y = max(omit_pct)
    plt.ylim(0, max_y * 1.2 if max_y > 0 else 100)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + (max_y * 0.02), f'{yval:.2f}%', ha='center', va='bottom')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
