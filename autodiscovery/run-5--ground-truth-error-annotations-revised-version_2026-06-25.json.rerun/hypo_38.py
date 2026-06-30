import subprocess
import sys
import os
import urllib.request
import zipfile
import importlib
import site
import gc

def install_deps():
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "psutil", "tensorstore", "scipy", "pandas", "matplotlib", "networkx"])
    url = "https://github.com/AllenInstitute/agentic-neuron-proofreader/archive/refs/heads/main.zip"
    
    dest_dir = "/tmp/agentic-repo"
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
        zip_path = os.path.join(dest_dir, "repo.zip")
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(dest_dir)
    
    repo_path = os.path.join(dest_dir, "agentic-neuron-proofreader-main")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", repo_path])

try:
    import agentic_neuron_proofreader
except ImportError:
    install_deps()
    importlib.invalidate_caches()
    for sp in site.getsitepackages():
        if sp not in sys.path:
            sys.path.append(sp)
    if hasattr(site, 'getusersitepackages'):
        usp = site.getusersitepackages()
        if usp not in sys.path:
            sys.path.append(usp)
    import sysconfig
    for p in [sysconfig.get_path('purelib'), sysconfig.get_path('platlib')]:
        if p and p not in sys.path:
            sys.path.append(p)
            
    import agentic_neuron_proofreader

import pickle
import glob
import numpy as np
from collections import defaultdict
import networkx as nx
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

def main():
    cache_files = glob.glob("../dataset_cache_*_add.pkl")
    if not cache_files:
        cache_files = glob.glob("../*_add.pkl")
    if not cache_files:
        cache_files = glob.glob("*_add.pkl")
        
    terminal_split = defaultdict(int)
    terminal_total = defaultdict(int)
    internal_split = defaultdict(int)
    internal_total = defaultdict(int)

    for path in cache_files:
        with open(path, "rb") as f:
            payload = pickle.load(f)
        gt = payload["gt_graph"]
        edge_error = np.asarray(payload["gt_edge_error"])
        
        terminal_edges = set()
        for comp in nx.connected_components(gt):
            subgraph = gt.subgraph(comp)
            degrees = dict(subgraph.degree())
            leaves = [n for n, d in degrees.items() if d == 1]
            branch_points = set(n for n, d in degrees.items() if d >= 3)
            
            if not branch_points:
                for u, v in subgraph.edges():
                    terminal_edges.add(frozenset((u, v)))
                continue
                
            for leaf in leaves:
                curr = leaf
                prev = None
                while True:
                    neighbors = list(subgraph.neighbors(curr))
                    if prev is not None:
                        if prev in neighbors:
                            neighbors.remove(prev)
                    
                    if not neighbors:
                        break
                        
                    nxt = neighbors[0]
                    terminal_edges.add(frozenset((curr, nxt)))
                    
                    if nxt in branch_points:
                        break
                        
                    prev = curr
                    curr = nxt
                    
        gt_edges = list(gt.edges)
        for k, (u, v) in enumerate(gt_edges):
            edge_set = frozenset((u, v))
            neuron = gt.node_segment_id(u)
            is_split = (edge_error[k] == 1)
            
            if edge_set in terminal_edges:
                terminal_total[neuron] += 1
                if is_split:
                    terminal_split[neuron] += 1
            else:
                internal_total[neuron] += 1
                if is_split:
                    internal_split[neuron] += 1
                    
        del gt
        del edge_error
        del payload
        gc.collect()

    data = []
    for neuron in set(terminal_total.keys()).union(internal_total.keys()):
        t_tot = terminal_total[neuron]
        i_tot = internal_total[neuron]
        
        if t_tot > 0 and i_tot > 0:
            t_rate = terminal_split[neuron] / t_tot
            i_rate = internal_split[neuron] / i_tot
            data.append({
                'neuron': neuron,
                'terminal_rate': t_rate,
                'internal_rate': i_rate
            })

    df = pd.DataFrame(data)

    if not df.empty:
        df['diff'] = df['terminal_rate'] - df['internal_rate']
        t_stat, p_val = stats.ttest_rel(df['terminal_rate'], df['internal_rate'])
        mean_diff = df['diff'].mean()

        print(f"Number of neurons analyzed: {len(df)}")
        print(f"Mean Internal Split Rate: {df['internal_rate'].mean():.4f}")
        print(f"Mean Terminal Split Rate: {df['terminal_rate'].mean():.4f}")
        print(f"Mean difference (Terminal - Internal split rate): {mean_diff:.4f}")
        print(f"Paired t-test t-statistic: {t_stat:.4f}")
        print(f"Paired t-test p-value: {p_val:.4e}")

        plt.figure(figsize=(6, 8))
        for _, row in df.iterrows():
            plt.plot([1, 2], [row['internal_rate'], row['terminal_rate']], marker='o', color='gray', alpha=0.5)

        mean_internal = df['internal_rate'].mean()
        mean_terminal = df['terminal_rate'].mean()
        plt.plot([1, 2], [mean_internal, mean_terminal], marker='o', color='red', linewidth=2, label='Mean')

        plt.xticks([1, 2], ['Internal Branches', 'Terminal Branches'])
        plt.ylabel('Split Error Rate')
        plt.title('Split Error Rate: Internal vs Terminal Compartments')
        plt.xlim(0.8, 2.2)
        plt.legend()
        plt.tight_layout()
        plt.show()
    else:
        print("Not enough data to run paired t-test.")

if __name__ == '__main__':
    main()
