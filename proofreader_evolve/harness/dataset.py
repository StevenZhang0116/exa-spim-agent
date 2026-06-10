"""
Data access for the proofreader evolution loop.

Everything here is built on the *cached* ``BrainDataset`` pickle produced by
``notebooks/load_skeletons.ipynb`` (``cache/dataset_cache_<brain>_mcl<N>.pkl``),
so it loads in seconds rather than re-reading ~10k SWCs from GCS (~15 min).

Two jobs:

  1. ``train_heldout_split`` — partition the ground-truth skeletons into a
     *train* set (the agent sees its mistakes here and revises) and a
     *held-out* set (used only to gate whether a revision is kept). Splitting
     by whole GT skeleton makes "improves on held-out" leak-free.

  2. ``candidate_split_sites`` — enumerate places in the fragment graph where
     two fragments are tip-to-tip close in space but carry *different* labels.
     These are the split-error candidates an "identify + correct" policy reasons
     over: each is a potential ``(label_a, label_b)`` unification edit. The
     evolved ``heuristics.propose_edits`` decides which to accept.

The geometry here is intentionally simple and dependency-free — it gives the
agent a concrete, inspectable candidate set to start from. The evolved
heuristics are free to compute richer features (angles, radii, image patches).
"""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass

import numpy as np
from scipy.spatial import KDTree


def default_cache_path(brain_id: str, min_cable_length: int = 100) -> str:
    """Path to the BrainDataset cache for a brain, relative to the project root."""
    here = os.path.dirname(__file__)
    return os.path.abspath(
        os.path.join(
            here, "..", "..", "cache",
            f"dataset_cache_{brain_id}_mcl{min_cable_length}.pkl",
        )
    )


def load_cached_graphs(cache_path: str):
    """Load just the two SkeletonGraphs from a BrainDataset cache pickle.

    We read the pickle payload directly (rather than via BrainDataset.load_from_cache)
    so this module has no dependency on the TensorStore image reader — the
    evolution loop's candidate-site geometry only needs the graphs.
    """
    with open(cache_path, "rb") as f:
        payload = pickle.load(f)
    return payload["fragments_graph"], payload["gt_graph"], payload


def train_heldout_split(
    gt_swc_names: list[str], heldout_fraction: float = 0.33, seed: int = 0
) -> tuple[list[str], list[str]]:
    """Deterministically split GT skeleton names into (train, heldout).

    Splitting by whole skeleton (not by edge) is what makes the held-out metric
    an honest generalization signal: no skeleton contributes to both feedback
    and gating.

    Parameters
    ----------
    gt_swc_names : list of str
        All ground-truth skeleton names (e.g. evaluate()'s per_swc index).
    heldout_fraction : float
        Fraction of skeletons reserved for gating. Default 1/3.
    seed : int
        RNG seed for the shuffle (fixed so runs are reproducible).
    """
    names = sorted(gt_swc_names)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(names))
    n_heldout = max(1, round(len(names) * heldout_fraction))
    heldout_idx = set(order[:n_heldout].tolist())
    train = [names[i] for i in range(len(names)) if i not in heldout_idx]
    heldout = [names[i] for i in range(len(names)) if i in heldout_idx]
    return train, heldout


@dataclass
class SplitSite:
    """A candidate split-error: two nearby fragment tips with different labels.

    Attributes
    ----------
    label_a, label_b : str
        Fragment labels (segment IDs) on either side of the candidate gap.
    gap_um : float
        Physical distance (microns) between the two tips.
    node_a, node_b : int
        The fragment-graph node ids of the two tips. Use these directly with the
        graph API the policy reasons over — ``g.neighbors(node_a)``,
        ``g.node_xyz[node_a]``, ``g.rooted_subgraph(node_a, radius)`` — to build
        tangent-direction, endpoint-degree, and local-continuity features. No
        coordinate reverse-lookup needed.
    xyz_a, xyz_b : tuple[float, float, float]
        Physical coordinates of the two tips (for image-patch lookups / display).
        These equal ``g.node_xyz[node_a]`` / ``g.node_xyz[node_b]``.
    """

    label_a: str
    label_b: str
    gap_um: float
    node_a: int
    node_b: int
    xyz_a: tuple
    xyz_b: tuple

    def as_edit(self) -> tuple:
        """The label pair this site would unify if accepted."""
        return (self.label_a, self.label_b)


def candidate_split_sites(
    fragments_graph,
    max_gap_um: float = 15.0,
    max_sites: int = 5000,
    tip_to_shaft: bool = True,
) -> list[SplitSite]:
    """Enumerate candidate split-repair sites: a fragment tip near a *differently
    labelled* node.

    A split error is a single neuron broken into multiple fragments. A repair
    unifies the two labels. The reconnection partner is not always another
    fragment's tip — a true split often lands a tip mid-shaft or at a branch
    point of the other fragment (tip-to-shaft / branch-point breakage). So:

    - ``node_a`` (the anchor) is always a **tip** (degree-1 node) — that is where
      a fragment ends and a repair must originate.
    - ``node_b`` (the partner) is the nearest **differently-labelled** node within
      ``max_gap_um``. With ``tip_to_shaft=True`` (default) the partner may be any
      node — tip, shaft (degree 2), or branch (degree 3+); with ``False`` only
      other tips qualify (the old tip-to-tip behavior).

    Broadening the partner set raises recall (tip-to-shaft and branch-point
    splits become reachable) at the cost of more, noisier candidates — so the
    evolved policy must lean harder on its precision checks (tangent agreement,
    endpoint degree, continuity), which is exactly what ``node_a``/``node_b``
    enable. Deciding which candidates to accept is the policy's job.

    Parameters
    ----------
    fragments_graph : SkeletonGraph
        The cached fragment graph (``dataset.fragments_graph``).
    max_gap_um : float
        Only return tip→node pairs closer than this physical distance.
    max_sites : int
        Cap on returned sites (closest gaps first), to bound the agent's input.
    tip_to_shaft : bool
        If True, the partner node may be any node (tip/shaft/branch). If False,
        partners are restricted to other tips (legacy tip-to-tip enumeration).

    Returns
    -------
    list[SplitSite] sorted by ascending gap, deduplicated per unordered
    label pair (closest gap kept).
    """
    g = fragments_graph
    all_nodes = list(g.nodes)
    if not all_nodes:
        return []

    # In agentic_neuron_proofreader.SkeletonGraph (the class that built the cache
    # pickle): node_xyz is an (N, 3) array ALREADY in microns, so we index it
    # directly and do NOT multiply by anisotropy. Per-node fragment identity is
    # the segment id, exposed as the node_segment_id(node) METHOD (there is no
    # node_label array on this class — that lives on the metrics LabeledGraph).
    node_arr = np.array(all_nodes)
    node_coords = g.node_xyz[node_arr]
    node_labels = np.array([str(g.node_segment_id(n)) for n in all_nodes])

    # Anchors are tips (degree-1). Partners come from the full node set (or just
    # tips when tip_to_shaft is False) via a kd-tree ball query.
    is_tip = np.array([g.degree[n] == 1 for n in all_nodes])
    tip_idx = np.flatnonzero(is_tip)
    if tip_idx.size == 0:
        return []

    partner_mask = np.ones(len(all_nodes), dtype=bool) if tip_to_shaft else is_tip
    partner_idx = np.flatnonzero(partner_mask)
    partner_tree = KDTree(node_coords[partner_idx])

    # For each tip, all partner nodes within max_gap_um.
    neighbor_lists = partner_tree.query_ball_point(node_coords[tip_idx], r=max_gap_um)

    # Keep the closest valid candidate per unordered label pair.
    best: dict[frozenset, tuple] = {}  # {label_pair: (gap, node_a, node_b, xyz_a, xyz_b)}
    for ti, neighbors in zip(tip_idx, neighbor_lists):
        la = node_labels[ti]
        if la == "0":
            continue
        ai = int(node_arr[ti])
        a_xyz = node_coords[ti]
        for pj in neighbors:
            gi = int(partner_idx[pj])
            if gi == ti:
                continue  # the tip itself
            lb = node_labels[gi]
            if lb == "0" or lb == la:
                continue  # unlabelled or same fragment — not a split-repair candidate
            gap = float(np.linalg.norm(a_xyz - node_coords[gi]))
            key = frozenset((la, lb))
            prev = best.get(key)
            if prev is None or gap < prev[0]:
                best[key] = (
                    gap,
                    ai,                                 # node_a: always the tip
                    int(node_arr[gi]),                  # node_b: tip/shaft/branch partner
                    tuple(map(float, a_xyz)),
                    tuple(map(float, node_coords[gi])),
                )

    sites: list[SplitSite] = []
    for v in best.values():
        gap, ai, bi, axyz, bxyz = v
        sites.append(
            SplitSite(
                label_a=str(g.node_segment_id(ai)),
                label_b=str(g.node_segment_id(bi)),
                gap_um=gap,
                node_a=ai,                              # always the tip
                node_b=bi,                              # tip / shaft / branch partner
                xyz_a=axyz,
                xyz_b=bxyz,
            )
        )
    sites.sort(key=lambda s: s.gap_um)
    return sites[:max_sites]


def list_fragment_labels(fragments_graph) -> list[str]:
    """All distinct fragment labels (the LabelHandler universe for scoring).

    Labels are per-node segment ids via the node_segment_id(node) method; there
    is no node_label array on agentic_neuron_proofreader.SkeletonGraph.
    """
    g = fragments_graph
    labels = {str(g.node_segment_id(n)) for n in g.nodes}
    labels.discard("0")
    return sorted(labels)


if __name__ == "__main__":
    # Smoke test against a cache pickle:
    #   python proofreader_evolve/harness/dataset.py 789202
    import sys

    brain = sys.argv[1] if len(sys.argv) > 1 else "789202"
    path = default_cache_path(brain)
    print(f"Loading {path} ...")
    frags, gt, _ = load_cached_graphs(path)
    print(frags.summary(prefix="Fragments") if hasattr(frags, "summary") else frags)
    sites = candidate_split_sites(frags)
    print(f"{len(sites)} candidate split sites (<=15um). Closest 5:")
    for s in sites[:5]:
        print(f"  {s.label_a} <-> {s.label_b}  gap={s.gap_um:.2f}um")
    print(f"{len(list_fragment_labels(frags))} distinct fragment labels.")
