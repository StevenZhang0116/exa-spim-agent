# === RERUN_LOADING_BOOTSTRAP v2 ===
import os as _os, sys as _sys, subprocess as _sp_boot
# Ensure NumPy >= 2 (the pkl was written with NumPy 2.x). Do this BEFORE any
# import that brings in numpy transitively. Quiet, single-shot.
try:
    import numpy as _np_boot
    _ver = tuple(int(x) for x in _np_boot.__version__.split(".")[:2])
    if _ver < (2, 0):
        raise ImportError("need numpy>=2")
except Exception:
    _sp_boot.check_call([_sys.executable, "-m", "pip", "install", "-q", "--user", "numpy>=2,<2.3"])
    # invalidate caches so a fresh numpy 2.x is loaded
    import importlib as _il_boot
    if "numpy" in _sys.modules:
        del _sys.modules["numpy"]
import numpy as _np_chk
print("Loading dataset from:", _os.environ.get("RERUN_PKL", "<unset>"), "| numpy", _np_chk.__version__)

# Make any dataset-search return RERUN_PKL (the runner's open() patch then
# redirects the actual file open). Cover os.walk, Path.rglob, subprocess.getoutput.
import os as _os2, glob as _glob_mod, subprocess as _sp
from pathlib import Path as _Path_boot
_RERUN_PKL = _os.environ.get("RERUN_PKL", "")
_RERUN_DIR = _os.path.dirname(_RERUN_PKL) or "."
_RERUN_NAME = _os.path.basename(_RERUN_PKL)
def _os_walk_patched(top, *a, **k):
    yield (_RERUN_DIR, [], [_RERUN_NAME])
_os2.walk = _os_walk_patched
_orig_rglob = _Path_boot.rglob
def _rglob_patched(self, pat):
    if isinstance(pat, str) and "pkl" in pat:
        yield _Path_boot(_RERUN_PKL)
        return
    yield from _orig_rglob(self, pat)
_Path_boot.rglob = _rglob_patched
_orig_getoutput = _sp.getoutput
def _getoutput_patched(cmd, *a, **k):
    s = str(cmd)
    if "find" in s and "pkl" in s:
        return _RERUN_PKL
    return _orig_getoutput(cmd, *a, **k)
_sp.getoutput = _getoutput_patched
# Short-circuit ALL in-script pip installs. The host environment already has
# numpy>=2, pandas, scipy, matplotlib, statsmodels, networkx, agentic_neuron_proofreader,
# etc. Re-running pip-install inside the subprocess only downgrades numpy and
# breaks the install, so we no-op every pip install here.
_orig_check_call = _sp.check_call
def _check_call_patched(args, *a, **k):
    if isinstance(args, list) and len(args) > 3 and args[1:4] == ["-m", "pip", "install"]:
        return 0
    return _orig_check_call(args, *a, **k)
_sp.check_call = _check_call_patched
_orig_run = _sp.run
def _run_patched(args, *a, **k):
    if isinstance(args, list) and len(args) > 3 and args[1:4] == ["-m", "pip", "install"]:
        class _R: returncode = 0; stdout = b""; stderr = b""
        return _R()
    return _orig_run(args, *a, **k)
_sp.run = _run_patched
# === END RERUN_LOADING_BOOTSTRAP ===

import sys
import subprocess
import glob

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", package])

# Install missing dependencies first to prevent unpickling issues
for pkg in ["networkx", "scipy", "matplotlib", "psutil", "pandas", "tensorstore"]:
    try:
        __import__(pkg)
    except ImportError:
        install(pkg)

try:
    import agentic_neuron_proofreader
except ImportError:
    install("https://github.com/AllenInstitute/agentic-neuron-proofreader/archive/refs/heads/main.zip")
    import agentic_neuron_proofreader

import pickle
import numpy as np
import re
from collections import defaultdict
import networkx as nx
from scipy.spatial.distance import cdist
from scipy.stats import mannwhitneyu
import matplotlib.pyplot as plt

# Load dataset
paths = glob.glob("../*add.pkl") + glob.glob("../cache/*add.pkl") + glob.glob("*add.pkl") + glob.glob("cache/*add.pkl")
if not paths:
    print("Dataset not found.")
    sys.exit(1)

path = paths[0]
with open(path, "rb") as f:
    payload = pickle.load(f)

gt = payload["gt_graph"]
fragments = payload["fragments_graph"]
node_label = np.asarray(payload["gt_node_canonical_label"])

# 1. Group predicted segment IDs by GT neuron
neuron_segs = defaultdict(set)
for n in gt.nodes:
    lab = int(node_label[n])
    if lab != 0:
        neuron_segs[gt.node_segment_id(n)].add(lab)

# Isolate neurons suffering from splits
split_neurons = {neuron: segs for neuron, segs in neuron_segs.items() if len(segs) > 1}

split_segs = set()
for segs in split_neurons.values():
    split_segs.update(segs)

print(f"Number of GT neurons with splits: {len(split_neurons)}")
print(f"Total number of split segment IDs involved: {len(split_segs)}")

# Resolve the mapping from node to SWC ID (predicted segment ID)
mapping = fragments.component_id_to_swc_id
parsed_mapping = {}

if isinstance(mapping, dict):
    iterator = mapping.items()
else:
    iterator = enumerate(mapping)

for comp_id, swc_val in iterator:
    if isinstance(swc_val, int):
        num = swc_val
    elif isinstance(swc_val, str):
        # Extract the segment ID, usually the filename without extension
        m = re.search(r'(\d+)', swc_val)
        if m:
            num = int(m.group(1))
        else:
            num = -1
    else:
        num = -1
    parsed_mapping[comp_id] = num

frag_swc_ids = np.array([parsed_mapping.get(c, -1) for c in fragments.node_component_id])

# 2 & 4. Filter fragments related to split segments and find terminal nodes
split_segs_arr = np.array(list(split_segs))
mask = np.isin(frag_swc_ids, split_segs_arr)
valid_nodes = np.where(mask)[0]

print(f"Number of valid fragments_graph nodes belonging to split segments: {len(valid_nodes)}")

frag_degrees = dict(fragments.degree())
swc_terminals = defaultdict(list)

for n in valid_nodes:
    if frag_degrees[n] <= 1:
        swc_terminals[frag_swc_ids[n]].append(n)

# Calculate intra-segment internal edge lengths
internal_edge_lengths = []
subgraph = fragments.subgraph(valid_nodes)
sub_edges = list(subgraph.edges)

if sub_edges:
    u_nodes = [e[0] for e in sub_edges]
    v_nodes = [e[1] for e in sub_edges]
    
    p1 = fragments.node_xyz[u_nodes]
    p2 = fragments.node_xyz[v_nodes]
    dists = np.linalg.norm(p1 - p2, axis=1)
    internal_edge_lengths.extend(dists.tolist())

gap_distances = []

# 3. Calculate pairwise Euclidean distance to find bridging gaps
for neuron, segs in split_neurons.items():
    segs = list(segs)
    G_segs = nx.Graph()
    for i in range(len(segs)):
        G_segs.add_node(segs[i])
        for j in range(i + 1, len(segs)):
            s1 = segs[i]
            s2 = segs[j]
            t1 = swc_terminals.get(s1, [])
            t2 = swc_terminals.get(s2, [])
            
            if not t1 or not t2:
                continue
            
            coords1 = fragments.node_xyz[t1]
            coords2 = fragments.node_xyz[t2]
            
            dists_ij = cdist(coords1, coords2)
            min_dist = np.min(dists_ij)
            G_segs.add_edge(s1, s2, weight=min_dist)
            
    if G_segs.number_of_nodes() > 0:
        for component in nx.connected_components(G_segs):
            sub_G = G_segs.subgraph(component)
            if sub_G.number_of_nodes() > 1:
                mst = nx.minimum_spanning_tree(sub_G)
                for u, v, data in mst.edges(data=True):
                    gap_distances.append(data['weight'])

# 5. Statistical comparison
gap_distances = np.array(gap_distances)
internal_edge_lengths = np.array(internal_edge_lengths)

if len(gap_distances) > 0:
    median_gap = np.median(gap_distances)
else:
    median_gap = np.nan

if len(internal_edge_lengths) > 0:
    p95_internal = np.percentile(internal_edge_lengths, 95)
else:
    p95_internal = np.nan

print(f"\nNumber of split bridging gaps found: {len(gap_distances)}")
print(f"Median inter-segment gap distance: {median_gap:.2f} µm" if not np.isnan(median_gap) else "Median inter-segment gap distance: NaN")
print(f"95th percentile of internal edge lengths: {p95_internal:.2f} µm" if not np.isnan(p95_internal) else "95th percentile of internal edge lengths: NaN")

if len(gap_distances) > 0 and len(internal_edge_lengths) > 0:
    stat, p_val = mannwhitneyu(gap_distances, internal_edge_lengths, alternative='greater')
    print(f"\nMann-Whitney U test (gaps > internal edges): U={stat}, p-value={p_val:.4e}")

    plt.figure(figsize=(10, 6))
    
    max_plot_dist = np.percentile(gap_distances, 95) + 5 if len(gap_distances) > 0 else 50
    bins = np.linspace(0, max(max_plot_dist, (p95_internal if not np.isnan(p95_internal) else 0) + 5), 50)
    
    plt.hist(internal_edge_lengths, bins=bins, density=True, alpha=0.6, label='Intra-segment Edge Lengths', color='blue')
    plt.hist(gap_distances, bins=bins, density=True, alpha=0.6, label='Inter-segment Gap Distances', color='red')
    
    plt.axvline(median_gap, color='red', linestyle='dashed', linewidth=2, label=f'Median Gap: {median_gap:.2f} µm')
    plt.axvline(p95_internal, color='blue', linestyle='dashed', linewidth=2, label=f'95th Pctl Internal: {p95_internal:.2f} µm')
    
    plt.xlabel('Physical Distance (µm)')
    plt.ylabel('Density')
    plt.title('Inter-segment Gap Distances vs Intra-segment Edge Lengths')
    plt.legend()
    plt.tight_layout()
    plt.show()
else:
    print("Not enough data to perform Mann-Whitney U test or generate plots.")
