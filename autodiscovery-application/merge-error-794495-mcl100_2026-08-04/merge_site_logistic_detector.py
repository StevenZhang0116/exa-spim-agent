#!/usr/bin/env python3
"""
merge_site_logistic_detector.py — Logistic regression merge detector.

Combines the 10 confirmed features from the AutoDiscovery run
merge-error-794495-mcl100_2026-08-04 into a single calibrated probability
per segment.  The model uses class_weight='balanced' to compensate for the
97:1 class imbalance (98 merged vs 9 525 clean adjudicable segments).

Feature summary (discovery run verdicts):
  1  max_windowed_tortuosity   — SOUND, AUC 0.94  (id 1)
  2  caliber_asymmetry         — SOUND, AUC 0.91  (id 4)
  3  pseudo_diam_tortuosity    — WEAK,  AUC 0.93  (id 20, size confound noted)
  4  max_edge_jump             — MINOR, AUC 0.81  (id 8)
  5  min_junction_angle        — WEAK,  AUC 0.92  (id 3; impute 180° if no junction)
  6  junction_tort_var         — WEAK,  ---       (id 13; impute 0.0 if no junction)
  7  curvature_spike_density   — MINOR→UPHELD corrected AUC≈0.74  (id 29)
  8  log10_fill_factor         — WEAK,  ---       (id 50; impute column mean)
  9  bimodality_score          — WEAK,  ---       (id 14; impute 0.0)
 10  log10_terminal_tortuosity — WEAK,  pseudo-R²=0.08  (id 16; impute 0.0)

Provenance: derived from the AutoDiscovery run
`autodiscovery/merge-error-794495-mcl100_2026-08-04.json` — see the sibling
README.md for the hypothesis-id → feature mapping and the run's own verdicts.

Usage (run from the repo root):
    conda activate panda
    RUN=autodiscovery-application/merge-error-794495-mcl100_2026-08-04
    python $RUN/merge_site_logistic_detector.py \\
        cache/dataset_cache_794495_mcl100_add.pkl

    # with explicit output path:
    python $RUN/merge_site_logistic_detector.py \\
        cache/dataset_cache_794495_mcl100_add.pkl \\
        --out-csv $RUN/merge_detector_794495.csv

Outputs (written next to this script by default):
  - Console: feature stats by class, 5-fold CV, coefficients, threshold sweep
  - CSV:     per-segment features + merge_probability + is_merge label

Runtime: ~3 min on a compute node for 794495 (measured 2026-08-03: 65 s pkl
load + 118 s feature extraction over 3 227 adjudicable components).
Memory: >20 GB for the pkl; allocate 80 GB to be safe. The login node OOMs —
this must run on a compute node.

Slurm template:
    #!/bin/bash
    #SBATCH --mem=80G --time=00:30:00
    source /shared/utils.x86_64/anaconda3-2024.10/etc/profile.d/conda.sh
    conda activate panda
    cd /allen/programs/mindscope/workgroups/auto-model/zihan.zhang/exaspim-agent/exa-spim-agent
    python autodiscovery-application/merge-error-794495-mcl100_2026-08-04/\\
merge_site_logistic_detector.py cache/dataset_cache_794495_mcl100_add.pkl
"""

import argparse
import heapq
import math
import os
import pickle
import re
import time
from collections import defaultdict, deque

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Pickle loader
# ─────────────────────────────────────────────────────────────────────────────

def _load_payload(pkl_path: str) -> dict:
    """Load *_add.pkl; prefer the real package, fall back to a mock unpickler."""
    try:
        import agentic_neuron_proofreader  # noqa: F401  registers SkeletonGraph
        with open(pkl_path, "rb") as f:
            return pickle.load(f)
    except (ImportError, ModuleNotFoundError):
        print("[warn] agentic_neuron_proofreader not importable; using fallback unpickler")
        import types

        _sg_base = nx.Graph

        class _SG(_sg_base):
            pass

        class _UP(pickle.Unpickler):
            def find_class(self, module, name):
                if module.startswith("agentic_neuron_proofreader"):
                    if name == "SkeletonGraph":
                        return _SG
                    class _D:
                        def __init__(self, *a, **k):
                            pass
                        def __setstate__(self, s):
                            if isinstance(s, dict):
                                self.__dict__.update(s)
                    _D.__name__ = name
                    return _D
                return super().find_class(module, name)

        with open(pkl_path, "rb") as f:
            return _UP(f).load()


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Low-level helpers (adapted verbatim from hypothesis rerun scripts)
# ─────────────────────────────────────────────────────────────────────────────

def _branch_direction(node_xyz, g, node, nbr, reach_um=15.0):
    """Unit vector from node outward along the branch that starts toward nbr."""
    prev, cur = node, nbr
    acc = float(np.linalg.norm(node_xyz[node] - node_xyz[nbr]))
    while acc < reach_um:
        nxt = [k for k in g.neighbors(cur) if k != prev]
        if len(nxt) != 1:
            break
        prev, cur = cur, nxt[0]
        acc += float(np.linalg.norm(node_xyz[prev] - node_xyz[cur]))
    v = node_xyz[cur] - node_xyz[node]
    nrm = float(np.linalg.norm(v))
    return (v / nrm) if nrm > 1e-9 else np.zeros(3, dtype=np.float32)


def _branch_radius(node_xyz, node_radius, g, node, nbr, reach_um=15.0):
    """Mean radius along the branch from node toward nbr, up to reach_um µm."""
    prev, cur = node, nbr
    acc = float(np.linalg.norm(node_xyz[node] - node_xyz[nbr]))
    radii = [float(node_radius[cur])]
    while acc < reach_um:
        nxt = [k for k in g.neighbors(cur) if k != prev]
        if len(nxt) != 1:
            break
        prev, cur = cur, nxt[0]
        acc += float(np.linalg.norm(node_xyz[prev] - node_xyz[cur]))
        radii.append(float(node_radius[cur]))
    return float(np.mean(radii))


def _branch_tortuosity(node_xyz, g, node, nbr, reach_um=20.0):
    """Tortuosity (path_len / euclidean) of the branch from node toward nbr."""
    prev, cur = node, nbr
    acc = float(np.linalg.norm(node_xyz[node] - node_xyz[nbr]))
    while acc < reach_um:
        nxt = [k for k in g.neighbors(cur) if k != prev]
        if len(nxt) != 1:
            break
        acc += float(np.linalg.norm(node_xyz[cur] - node_xyz[nxt[0]]))
        prev, cur = cur, nxt[0]
    euclid = float(np.linalg.norm(node_xyz[cur] - node_xyz[node]))
    return acc / euclid if euclid > 1e-9 else 1.0


def _extract_chains(g, degrees):
    """
    Decompose the graph into maximal chains (paths through consecutive
    degree-2 nodes, bounded by non-degree-2 endpoints).  Mirrors hypo_1.py.
    """
    end_nodes = [n for n, d in degrees.items() if d != 2]
    visited = set()
    paths = []

    for n in end_nodes:
        for nbr in g.neighbors(n):
            if degrees.get(nbr, 0) != 2:
                if n < nbr:
                    paths.append([n, nbr])
                continue
            if nbr in visited:
                continue
            path = [n, nbr]
            visited.add(nbr)
            curr, prev = nbr, n
            while True:
                nxt = [k for k in g.neighbors(curr) if k != prev]
                if not nxt:
                    break
                nxt = nxt[0]
                path.append(nxt)
                if degrees.get(nxt, 0) != 2:
                    break
                visited.add(nxt)
                prev, curr = curr, nxt
            paths.append(path)

    # pure degree-2 cycles
    for n, d in degrees.items():
        if d != 2 or n in visited:
            continue
        path = [n]
        visited.add(n)
        curr, prev_ = n, None
        while True:
            nxt = [k for k in g.neighbors(curr) if k != prev_]
            if not nxt:
                break
            nxt = nxt[0]
            if nxt in visited:
                path.append(nxt)
                break
            path.append(nxt)
            visited.add(nxt)
            prev_, curr = curr, nxt
        paths.append(path)

    return paths


def _windowed_tortuosity(node_xyz, path, window_um=15.0, min_len=10.0):
    """Max path-length / Euclidean in a sliding window of ≥window_um along path."""
    if len(path) < 2:
        return 1.0
    pts = node_xyz[path]
    diffs = pts[1:] - pts[:-1]
    step_d = np.linalg.norm(diffs, axis=1)
    cum = np.zeros(len(path))
    cum[1:] = np.cumsum(step_d)
    max_t = 1.0
    n = len(path)
    for i in range(n - 1):
        j = i + 1
        while j < n and cum[j] - cum[i] < window_um:
            j += 1
        if j == n:
            j = n - 1
        L = cum[j] - cum[i]
        if L >= min_len:
            E = float(np.linalg.norm(pts[j] - pts[i]))
            if E > 1e-5:
                t = L / E
                if t > max_t:
                    max_t = t
    return max_t


def _double_dijkstra(g, nodes_set, node_xyz):
    """
    Weighted pseudo-diameter via double-Dijkstra.
    Returns (n_a, n_b, geodesic_dist, euclidean_dist).
    """
    if len(nodes_set) < 2:
        return None, None, 0.0, 0.0

    def _dijkstra(start):
        dist = {start: 0.0}
        pq = [(0.0, start)]
        far, max_d = start, 0.0
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist.get(u, 1e18):
                continue
            if d > max_d:
                max_d, far = d, u
            for v in g.neighbors(u):
                if v not in nodes_set:
                    continue
                w = float(np.linalg.norm(node_xyz[u] - node_xyz[v]))
                nd = d + w
                if nd < dist.get(v, 1e18):
                    dist[v] = nd
                    heapq.heappush(pq, (nd, v))
        return far, max_d

    start = next(iter(nodes_set))
    n_a, _ = _dijkstra(start)
    n_b, geo = _dijkstra(n_a)
    euc = float(np.linalg.norm(node_xyz[n_a] - node_xyz[n_b])) if n_b is not None else 0.0
    return n_a, n_b, geo, euc


def _bfs_longest_path(g, nodes_set, node_xyz):
    """Approximate longest path (diameter) via double-BFS (hop-weighted)."""
    if len(nodes_set) < 2:
        return []

    def _bfs_far(start):
        vis = {start: None}
        q = deque([start])
        far = start
        while q:
            u = q.popleft()
            far = u
            for v in g.neighbors(u):
                if v in nodes_set and v not in vis:
                    vis[v] = u
                    q.append(v)
        return far, vis

    start = next(iter(nodes_set))
    a, _ = _bfs_far(start)
    b, vis = _bfs_far(a)
    path, cur = [], b
    while cur is not None:
        path.append(cur)
        cur = vis[cur]
    return path


def _curvature_spike_density(path, node_xyz):
    """Curvature spike rate: count of turns >90° per 100 µm along path."""
    if len(path) < 3:
        return 0.0
    coords = node_xyz[path].astype(np.float64)
    diffs = coords[1:] - coords[:-1]
    lens = np.linalg.norm(diffs, axis=1)
    path_um = float(np.sum(lens))
    if path_um < 1.0:
        return 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        du = np.where(lens[:, None] > 1e-9, diffs / lens[:, None], 0.0)
    dots = np.sum(du[:-1] * du[1:], axis=1)
    spikes = int(np.sum((dots < 0) & ~np.isnan(dots)))
    return (spikes / path_um) * 100.0


def _bimodality_score(xyz_nodes):
    """BIC(1-GMM) − BIC(2-GMM) on node-to-centroid distances. Positive → bimodal."""
    if len(xyz_nodes) < 50:
        return 0.0
    centroid = xyz_nodes.mean(axis=0)
    dists = np.linalg.norm(xyz_nodes - centroid, axis=1).reshape(-1, 1)
    try:
        g1 = GaussianMixture(n_components=1, random_state=42).fit(dists)
        g2 = GaussianMixture(n_components=2, random_state=42).fit(dists)
        return float(g1.bic(dists) - g2.bic(dists))
    except Exception:
        return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Feature extraction — five computation phases
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_COLS = [
    "max_windowed_tortuosity",
    "caliber_asymmetry",
    "pseudo_diam_tortuosity",
    "max_edge_jump",
    "min_junction_angle",
    "junction_tort_var",
    "curvature_spike_density",
    "log10_fill_factor",
    "bimodality_score",
    "log10_terminal_tortuosity",
]


def extract_features(payload: dict) -> pd.DataFrame:
    frag         = payload["fragments_graph"]
    node_xyz     = frag.node_xyz
    node_radius  = frag.node_radius
    node_comp_id = frag.node_component_id
    comp_to_swc  = frag.component_id_to_swc_id
    min_cable    = int(payload.get("min_cable_length", 100))

    gt_merge_labels = set(int(x) for x in payload["gt_merge_labels"])
    node_label      = np.asarray(payload["gt_node_canonical_label"])
    adjudicable     = set(int(x) for x in np.unique(node_label) if int(x) != 0)

    # Map component_id → segment_id
    comp_to_seg = {int(c): int(s.split(".")[0]) for c, s in comp_to_swc.items()}

    # ── Build per-segment node and component index ────────────────────────
    t0 = time.time()
    print("  Building component/segment index...")

    # node_seg_id: int64 array of segment_id per node (-1 if not adjudicable)
    max_node_id = int(max(frag.nodes)) + 1
    node_seg_id = np.full(max_node_id, -1, dtype=np.int64)
    comp_nodes  = defaultdict(list)   # comp_id → [node, ...]
    seg_to_comps = defaultdict(list)  # seg_id  → [comp_id, ...]

    for n in frag.nodes:
        c = int(node_comp_id[n])
        s = comp_to_seg.get(c, -1)
        if s != -1 and s in adjudicable:
            node_seg_id[n] = s
            comp_nodes[c].append(n)
            seg_to_comps[s].append(c)

    print(f"    {len(comp_nodes)} adjudicable components, "
          f"{len(seg_to_comps)} segments  ({time.time()-t0:.0f}s)")

    degrees = dict(frag.degree())

    # Per-segment feature accumulators (default values match the imputation
    # used in train_and_evaluate; they are the "no signal" baseline)
    seg_max_wt   = {}  # max windowed tortuosity  — default 1.0
    seg_asym     = {}  # caliber asymmetry         — default 0.0
    seg_pdt      = {}  # pseudo-diam tortuosity    — default 1.0
    seg_edge_jmp = {}  # max edge jump             — default 0.0
    seg_min_ang  = {}  # min junction angle        — default 180.0
    seg_jt_var   = {}  # junction tort variance    — default 0.0
    seg_spk_rate = {}  # curvature spike density   — default 0.0
    seg_cable    = {}  # cable length (µm)         — internal, for fill_factor
    seg_bimod    = {}  # bimodality score          — default 0.0
    seg_term_t   = {}  # log10 terminal tortuosity — default NaN (imputed later)
    seg_fill     = {}  # log10 fill factor         — default NaN (imputed later)

    # ── Phase A: vectorised edge-level features ───────────────────────────
    t0 = time.time()
    print("  [A] Edge-level: max_edge_jump, cable_length ...")

    all_edges = np.array(list(frag.edges), dtype=np.int64)
    if len(all_edges):
        u_arr = all_edges[:, 0]
        v_arr = all_edges[:, 1]
        eu    = np.linalg.norm(node_xyz[u_arr] - node_xyz[v_arr], axis=1)
        su    = node_seg_id[u_arr]
        sv    = node_seg_id[v_arr]

        # max edge jump: any edge whose source is adjudicable
        adj_mask = su != -1
        eu_adj   = eu[adj_mask]
        su_adj   = su[adj_mask]
        order    = np.argsort(su_adj)
        su_adj   = su_adj[order]
        eu_adj   = eu_adj[order]
        unique_s, starts = np.unique(su_adj, return_index=True)
        for i, s in enumerate(unique_s):
            sl = slice(starts[i], starts[i + 1] if i + 1 < len(starts) else None)
            seg_edge_jmp[int(s)] = float(eu_adj[sl].max())

        # cable length: only intra-segment edges (same seg on both ends)
        intra = (su == sv) & (su != -1)
        eu_intra = eu[intra]
        su_intra = su[intra]
        order    = np.argsort(su_intra)
        su_intra = su_intra[order]
        eu_intra = eu_intra[order]
        unique_s, starts = np.unique(su_intra, return_index=True)
        for i, s in enumerate(unique_s):
            sl = slice(starts[i], starts[i + 1] if i + 1 < len(starts) else None)
            seg_cable[int(s)] = float(eu_intra[sl].sum())

    print(f"    done  ({time.time()-t0:.0f}s)")

    # ── Phase B: global chain extraction → windowed tortuosity ───────────
    t0 = time.time()
    print("  [B] Global chains → max_windowed_tortuosity ...")

    chains = _extract_chains(frag, degrees)
    print(f"    {len(chains)} chains  ({time.time()-t0:.0f}s)")

    t0 = time.time()
    for ch in chains:
        if len(ch) < 2:
            continue
        s = node_seg_id[ch[0]]
        if s == -1:
            continue
        t = _windowed_tortuosity(node_xyz, ch)
        if t > seg_max_wt.get(s, 1.0):
            seg_max_wt[int(s)] = t

    print(f"    windowed tortuosity done  ({time.time()-t0:.0f}s)")

    # ── Phase C: junction-level features ─────────────────────────────────
    t0 = time.time()
    print("  [C] Junction nodes → caliber_asymmetry, min_junction_angle, "
          "junction_tort_var ...")

    n_junc = 0
    for n in frag.nodes:
        if degrees.get(n, 0) < 3:
            continue
        s = int(node_seg_id[n])
        if s == -1:
            continue
        n_junc += 1
        nbrs = list(frag.neighbors(n))

        # caliber asymmetry (id 4): thickest daughter / parent
        rads = sorted(_branch_radius(node_xyz, node_radius, frag, n, nb)
                      for nb in nbrs)
        if rads[-1] > 1e-5:
            asym = rads[-2] / rads[-1]
            if asym > seg_asym.get(s, 0.0):
                seg_asym[s] = asym

        # min junction angle (id 3)
        dirs = [_branch_direction(node_xyz, frag, n, nb) for nb in nbrs]
        for i in range(len(dirs)):
            for j in range(i + 1, len(dirs)):
                dot = max(-1.0, min(1.0, float(np.dot(dirs[i], dirs[j]))))
                ang = math.degrees(math.acos(dot))
                if ang < seg_min_ang.get(s, 180.0):
                    seg_min_ang[s] = ang

        # junction tortuosity variance (id 13)
        if len(nbrs) >= 3:
            torts = [_branch_tortuosity(node_xyz, frag, n, nb) for nb in nbrs]
            tv = float(np.var(torts))
            if tv > seg_jt_var.get(s, 0.0):
                seg_jt_var[s] = tv

    print(f"    {n_junc} junctions  ({time.time()-t0:.0f}s)")

    # ── Phase D: per-component features ──────────────────────────────────
    t0 = time.time()
    n_comps = len(comp_nodes)
    print(f"  [D] Per-component ({n_comps} comps): pseudo_diam_tortuosity, "
          "curvature_spike_density, terminal_tortuosity, bbox ...")

    seg_all_xyz   = defaultdict(list)  # all xyz per segment (for bimodality + fill)
    all_term_t    = defaultdict(list)  # terminal tortuosity samples

    for idx, (comp_id, nodes) in enumerate(comp_nodes.items()):
        if idx % 10_000 == 0 and idx > 0:
            print(f"    {idx}/{n_comps} components  ({time.time()-t0:.0f}s)")
        if not nodes:
            continue
        s = int(comp_to_seg.get(comp_id, -1))
        if s == -1:
            continue

        nodes_set = set(nodes)
        xyz_nodes = node_xyz[nodes]
        seg_all_xyz[s].append(xyz_nodes)

        # pseudo-diameter tortuosity (id 20)
        if len(nodes) >= 2:
            _, _, geo, euc = _double_dijkstra(frag, nodes_set, node_xyz)
            if geo > 0 and euc > 1e-5:
                t = geo / euc
                if t > seg_pdt.get(s, 1.0):
                    seg_pdt[s] = t

        # curvature spike density (id 29, corrected)
        if len(nodes) >= 3:
            path = _bfs_longest_path(frag, nodes_set, node_xyz)
            rate = _curvature_spike_density(path, node_xyz)
            if rate > seg_spk_rate.get(s, 0.0):
                seg_spk_rate[s] = rate

        # terminal tortuosity (id 16)
        local_deg  = {n: degrees[n] for n in nodes}
        branch_pts = [n for n in nodes if local_deg[n] > 2]
        leaves     = [n for n in nodes if local_deg[n] == 1]
        if branch_pts and leaves and len(nodes) >= 20:
            bp_xyz = node_xyz[branch_pts]
            subg   = frag.subgraph(nodes).copy()
            for u, v, d in subg.edges(data=True):
                d["weight"] = float(np.linalg.norm(node_xyz[u] - node_xyz[v]))
            # cap at 50 leaves for speed
            if len(leaves) > 50:
                rng    = np.random.default_rng(comp_id)
                leaves = rng.choice(leaves, 50, replace=False).tolist()
            for leaf in leaves:
                l_xyz   = node_xyz[leaf]
                dist_bp = np.linalg.norm(bp_xyz - l_xyz, axis=1)
                nearest = branch_pts[int(np.argmin(dist_bp))]
                euc_d   = float(np.min(dist_bp))
                if euc_d > 0.1:
                    try:
                        geo_d = nx.dijkstra_path_length(
                            subg, source=nearest, target=leaf, weight="weight")
                        if geo_d > 0:
                            all_term_t[s].append(geo_d / euc_d)
                    except nx.NetworkXNoPath:
                        pass

    print(f"    per-component done  ({time.time()-t0:.0f}s)")

    # ── Phase E: segment-level aggregation ───────────────────────────────
    t0 = time.time()
    print("  [E] Segment-level aggregation: bimodality, fill_factor ...")

    for s, xyzs in seg_all_xyz.items():
        all_xyz = np.vstack(xyzs)
        seg_bimod[s] = _bimodality_score(all_xyz)

        cable = seg_cable.get(s, 0.0)
        if cable >= 5 * min_cable:
            dims = np.maximum(all_xyz.max(axis=0) - all_xyz.min(axis=0), 1.0)
            vol  = float(np.prod(dims))
            if vol > 0:
                seg_fill[s] = math.log10(cable / vol)

    for s, torts in all_term_t.items():
        if torts:
            avg = float(np.mean(torts))
            seg_term_t[s] = math.log10(avg) if avg > 1e-9 else 0.0

    print(f"    done  ({time.time()-t0:.0f}s)")

    # ── Assemble DataFrame ────────────────────────────────────────────────
    print("  Assembling feature DataFrame...")
    rows = []
    for s in adjudicable:
        rows.append({
            "segment_id":               int(s),
            "is_merge":                 int(s in gt_merge_labels),
            "max_windowed_tortuosity":  seg_max_wt.get(s, 1.0),
            "caliber_asymmetry":        seg_asym.get(s, 0.0),
            "pseudo_diam_tortuosity":   seg_pdt.get(s, 1.0),
            "max_edge_jump":            seg_edge_jmp.get(s, 0.0),
            "min_junction_angle":       seg_min_ang.get(s, 180.0),
            "junction_tort_var":        seg_jt_var.get(s, 0.0),
            "curvature_spike_density":  seg_spk_rate.get(s, 0.0),
            "log10_fill_factor":        seg_fill.get(s, float("nan")),
            "bimodality_score":         seg_bimod.get(s, 0.0),
            "log10_terminal_tortuosity": seg_term_t.get(s, float("nan")),
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Model: train, cross-validate, evaluate
# ─────────────────────────────────────────────────────────────────────────────

def train_and_evaluate(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Imputation
    fill_mean = df["log10_fill_factor"].mean()
    df["log10_fill_factor"] = df["log10_fill_factor"].fillna(
        fill_mean if not math.isnan(fill_mean) else 0.0)
    df["log10_terminal_tortuosity"] = df["log10_terminal_tortuosity"].fillna(0.0)

    X = df[FEATURE_COLS].values.astype(np.float64)
    y = df["is_merge"].values

    n_merged = int(y.sum())
    n_clean  = int((y == 0).sum())
    print(f"\nDataset: {len(df)} adjudicable segments — "
          f"{n_merged} merged, {n_clean} clean ({n_clean // n_merged}:1 imbalance)")

    # Per-class feature summary
    print("\n--- Feature means by class ---")
    print(f"  {'Feature':<32}  {'Merged':>10}  {'Clean':>10}")
    print("  " + "─" * 56)
    for col in FEATURE_COLS:
        m = df.loc[df.is_merge == 1, col].mean()
        c = df.loc[df.is_merge == 0, col].mean()
        print(f"  {col:<32}  {m:>10.4f}  {c:>10.4f}")

    # Model: StandardScaler + logistic regression with balanced class weight
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            class_weight="balanced",
            C=0.1,
            max_iter=1000,
            solver="lbfgs",
        ),
    )

    # 5-fold stratified cross-validation
    cv  = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    res = cross_validate(clf, X, y, cv=cv,
                         scoring=["roc_auc", "average_precision"],
                         return_train_score=False)
    print(f"\n--- 5-fold stratified cross-validation ---")
    print(f"  ROC-AUC:          {res['test_roc_auc'].mean():.4f} "
          f"± {res['test_roc_auc'].std():.4f}")
    print(f"  Avg Precision:    {res['test_average_precision'].mean():.4f} "
          f"± {res['test_average_precision'].std():.4f}")

    # Fit on full data to inspect coefficients
    clf.fit(X, y)
    lr = clf.named_steps["logisticregression"]
    print("\n--- Coefficients (z-scored features, sorted by |coef|) ---")
    print(f"  {'Feature':<32}  {'coef':>8}")
    print("  " + "─" * 44)
    for feat, coef in sorted(zip(FEATURE_COLS, lr.coef_[0]),
                              key=lambda x: -abs(x[1])):
        print(f"  {feat:<32}  {coef:>8.4f}")

    probs = clf.predict_proba(X)[:, 1]
    df["merge_probability"] = probs

    print(f"\n  Full-data ROC-AUC:  {roc_auc_score(y, probs):.4f}")
    print(f"  Full-data AvgPrec:  {average_precision_score(y, probs):.4f}")

    # Threshold sweep: recall / precision / number of flagged segments
    print("\n--- Threshold sweep (full-data, for calibration reference) ---")
    print(f"  {'Threshold':>9}  {'Recall':>7}  {'Precision':>10}  {'N flagged':>10}")
    for thr in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        preds = (probs >= thr).astype(int)
        tp  = int(((preds == 1) & (y == 1)).sum())
        fp  = int(((preds == 1) & (y == 0)).sum())
        fn  = int(((preds == 0) & (y == 1)).sum())
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        print(f"  {thr:>9.1f}  {rec:>7.3f}  {prec:>10.3f}  {(preds==1).sum():>10}")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Logistic regression merge detector — 10 AutoDiscovery features",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("pkl", help="Path to *_add.pkl cache file")
    parser.add_argument(
        "--out-csv", default=None,
        help="Output CSV path (default: <this script's dir>/merge_detector_<brain>.csv)",
    )
    parser.add_argument(
        "--exclude-empty", action="store_true",
        help="Drop adjudicable segments that have NO fragment component before "
             "fitting. Those segments get every feature at its default value, "
             "are never merges, and so act as a trivially-separable class that "
             "inflates ROC-AUC (+0.105 on 794495: 0.946 -> 0.841). Use this "
             "flag for the honest discriminative number; omit it to reproduce "
             "the whole-adjudicable-universe numbers in README.md.",
    )
    args = parser.parse_args()

    pkl_path = args.pkl
    m = re.search(r"dataset_cache_(\d+)", os.path.basename(pkl_path))
    brain_id = m.group(1) if m else "unknown"

    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_csv = args.out_csv or os.path.join(out_dir, f"merge_detector_{brain_id}.csv")

    print(f"Loading {pkl_path} ...")
    t_start = time.time()
    payload = _load_payload(pkl_path)
    print(f"  loaded in {time.time()-t_start:.0f}s")

    print("\nExtracting features ...")
    t_feat = time.time()
    df = extract_features(payload)
    print(f"  done in {time.time()-t_feat:.0f}s  ({len(df)} segments)")

    # Segments with no fragment component carry every feature at its default,
    # never carry a merge label, and therefore form a trivially-separable class.
    empty = df["max_windowed_tortuosity"] <= 1.001
    n_empty = int(empty.sum())
    n_empty_merged = int(df.loc[empty, "is_merge"].sum())
    print(f"\n  {n_empty} of {len(df)} adjudicable segments have NO fragment "
          f"component ({n_empty_merged} of them merged)")
    if args.exclude_empty:
        df = df.loc[~empty].reset_index(drop=True)
        print(f"  --exclude-empty: dropped them; fitting on {len(df)} segments")
    else:
        print("  keeping them (pass --exclude-empty for the deflated, honest AUC)")

    df_out = train_and_evaluate(df)

    top = df_out.sort_values("merge_probability", ascending=False).head(20)
    print("\n--- Top 20 predicted merges (by probability) ---")
    cols_show = [
        "segment_id", "merge_probability", "is_merge",
        "max_windowed_tortuosity", "caliber_asymmetry",
        "pseudo_diam_tortuosity", "min_junction_angle",
    ]
    print(top[cols_show].to_string(index=False))

    df_out.to_csv(out_csv, index=False)
    print(f"\nFull feature matrix + probabilities → {out_csv}")
    print(f"Total wall time: {time.time()-t_start:.0f}s")


if __name__ == "__main__":
    main()
