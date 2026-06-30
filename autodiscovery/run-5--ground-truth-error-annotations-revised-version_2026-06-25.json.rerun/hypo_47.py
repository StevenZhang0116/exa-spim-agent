import subprocess
import sys
import glob
import collections
import numpy as np
import pickle

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", package])

# Install dependencies if needed
for pkg in ["psutil", "pandas", "networkx", "scipy", "tensorstore", "matplotlib", "tqdm"]:
    try:
        __import__(pkg)
    except ImportError:
        install(pkg)

try:
    import agentic_neuron_proofreader
except ImportError:
    # Use archive to avoid git requirement
    install("https://github.com/AllenInstitute/agentic-neuron-proofreader/archive/refs/heads/main.zip")
    import agentic_neuron_proofreader

import networkx as nx
from scipy.spatial import cKDTree

def log(msg):
    print(msg)
    sys.stdout.flush()

def main():
    log("Searching for dataset...")
    # Safe non-recursive search to prevent globbing timeouts
    cache_paths = glob.glob("../*_add.pkl") + glob.glob("../*/*_add.pkl") + glob.glob("*_add.pkl") + glob.glob("*/*_add.pkl")
    if not cache_paths:
        log("Dataset cache not found.")
        return
    cache_path = cache_paths[0]
    
    log(f"Loading dataset from {cache_path}...")
    with open(cache_path, "rb") as f:
        payload = pickle.load(f)
        
    fg = payload["fragments_graph"]
    gt = payload["gt_graph"]
    node_label = np.asarray(payload["gt_node_canonical_label"])
    
    log("Extracting leaf nodes...")
    # Fast degree calculation using underlying adjacency dict instead of pure python networkx DegreeView wrapper
    leaf_nodes = [n for n, nbrs in fg._adj.items() if len(nbrs) == 1]
    
    log(f"Found {len(leaf_nodes)} leaf nodes. Extracting features...")
    leaf_nodes = np.array(leaf_nodes)
    
    xyz = fg.node_xyz[leaf_nodes]
    radii = fg.node_radius[leaf_nodes]
    comp_ids = fg.node_component_id[leaf_nodes]
    
    log("Mapping components to segment IDs...")
    swc_ids = [fg.component_id_to_swc_id[c] for c in comp_ids]
    seg_ids = np.array([int(float(x)) for x in swc_ids])
    
    log("Building KD-tree and querying pairs within 15 µm...")
    tree = cKDTree(xyz)
    pairs = tree.query_pairs(15.0)
    log(f"KD-tree returned {len(pairs)} raw pairs. Filtering baseline pairs...")
    
    baseline_pairs = []
    for i, j in pairs:
        seg_i = seg_ids[i]
        seg_j = seg_ids[j]
        if seg_i != seg_j and seg_i != 0 and seg_j != 0:
            baseline_pairs.append((seg_i, seg_j, radii[i], radii[j]))
            
    log(f"Found {len(baseline_pairs)} baseline segment pairs.")
    
    log("Building ground-truth segment mappings...")
    seg_to_neurons = collections.defaultdict(lambda: collections.defaultdict(int))
    for n in gt.nodes:
        lab = int(node_label[n])
        if lab != 0:
            neuron_name = gt.node_segment_id(n)
            seg_to_neurons[lab][neuron_name] += 1
            
    log("Calculating initial splits...")
    neuron_to_segs = collections.defaultdict(set)
    for seg, neurons in seg_to_neurons.items():
        for neuron, count in neurons.items():
            if count > 0:
                neuron_to_segs[neuron].add(seg)
                
    initial_splits = sum(max(len(segs) - 1, 0) for segs in neuron_to_segs.values())
    
    log("Calculating initial merges...")
    def count_merges(seg_to_neu_dict):
        total_merges = 0
        for seg, neurons in seg_to_neu_dict.items():
            valid_neurons = [n for n, c in neurons.items() if c > 50]
            total_merges += max(len(valid_neurons) - 1, 0)
        return total_merges
        
    initial_merges = count_merges(seg_to_neurons)
    
    log(f"Initial splits: {initial_splits}")
    log(f"Initial canonical merges: {initial_merges}")
    
    log("Evaluating valid segments for false merge pairs...")
    valid_seg_to_neurons = collections.defaultdict(set)
    for seg, n_counts in seg_to_neurons.items():
        for neuron, count in n_counts.items():
            if count > 50:
                valid_seg_to_neurons[seg].add(neuron)
                
    def evaluate_heuristic(pairs_list):
        G = nx.Graph()
        for seg_i, seg_j, _, _ in pairs_list:
            G.add_edge(seg_i, seg_j)
            
        super_segments = list(nx.connected_components(G))
        
        seg_to_super = {}
        for comp in super_segments:
            rep = min(comp)
            for seg in comp:
                seg_to_super[seg] = rep
                
        new_neuron_to_segs = collections.defaultdict(set)
        for neuron, segs in neuron_to_segs.items():
            for seg in segs:
                new_seg = seg_to_super.get(seg, seg)
                new_neuron_to_segs[neuron].add(new_seg)
                
        new_splits = sum(max(len(segs) - 1, 0) for segs in new_neuron_to_segs.values())
        resolved_splits = initial_splits - new_splits
        
        new_seg_to_neurons = collections.defaultdict(lambda: collections.defaultdict(int))
        for seg, neurons in seg_to_neurons.items():
            super_seg = seg_to_super.get(seg, seg)
            for neuron, count in neurons.items():
                new_seg_to_neurons[super_seg][neuron] += count
                
        new_merges = count_merges(new_seg_to_neurons)
        false_merges_introduced = new_merges - initial_merges
        
        bad_pairs = 0
        for seg_i, seg_j, _, _ in pairs_list:
            ni = valid_seg_to_neurons.get(seg_i, set())
            nj = valid_seg_to_neurons.get(seg_j, set())
            if ni and nj and len(ni.union(nj)) > 1:
                bad_pairs += 1
                
        return resolved_splits, false_merges_introduced, bad_pairs

    log("Evaluating baseline heuristic...")
    baseline_resolved, baseline_false_merges, baseline_bad_pairs = evaluate_heuristic(baseline_pairs)
    
    log("Applying thickness constraint...")
    constrained_pairs = []
    for seg_i, seg_j, r_i, r_j in baseline_pairs:
        max_r = max(r_i, r_j)
        if max_r > 0:
            diff = abs(r_i - r_j) / max_r
            if diff < 0.25:
                constrained_pairs.append((seg_i, seg_j, r_i, r_j))
        else:
            constrained_pairs.append((seg_i, seg_j, r_i, r_j))
            
    log(f"Found {len(constrained_pairs)} constrained segment pairs.")
    
    log("Evaluating constrained heuristic...")
    constrained_resolved, constrained_false_merges, constrained_bad_pairs = evaluate_heuristic(constrained_pairs)
    
    pct_reduction_canonical = 100.0 * (baseline_false_merges - constrained_false_merges) / baseline_false_merges if baseline_false_merges else 0.0
    pct_reduction_pairs = 100.0 * (baseline_bad_pairs - constrained_bad_pairs) / baseline_bad_pairs if baseline_bad_pairs else 0.0
    pct_splits_retained = 100.0 * constrained_resolved / baseline_resolved if baseline_resolved else 0.0
    
    log("\n--- Results ---")
    log("Baseline Heuristic (Proximity <= 15 µm):")
    log(f"  Resolved True Splits: {baseline_resolved}")
    log(f"  False Merges Introduced (Canonical): {baseline_false_merges}")
    log(f"  False Merge Pairs (Direct): {baseline_bad_pairs}")
    
    log("\nConstrained Heuristic (Proximity + Radius Diff < 25%):")
    log(f"  Resolved True Splits: {constrained_resolved}")
    log(f"  False Merges Introduced (Canonical): {constrained_false_merges}")
    log(f"  False Merge Pairs (Direct): {constrained_bad_pairs}")
    
    log("\nSummary:")
    log(f"  Percentage Reduction in False Merges (Canonical): {pct_reduction_canonical:.2f}%")
    log(f"  Percentage Reduction in False Merge Pairs (Direct): {pct_reduction_pairs:.2f}%")
    log(f"  Impact: Retained {pct_splits_retained:.2f}% of the splits that were resolved by the baseline heuristic")

if __name__ == "__main__":
    main()
