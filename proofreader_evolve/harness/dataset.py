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
from dataclasses import dataclass, field

import networkx as nx
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

    kind: str = field(default="split", init=False)  # site-type tag for the policy

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


@dataclass
class MergeSite:
    """A candidate merge-error: one label fused across two neurites at a branch.

    A merge error is a single raw segment id ``L`` whose skeleton actually covers
    two (or more) distinct neurons that physically touch/cross. The repair is a
    ``split_label`` that partitions ``L`` into pseudo-labels by location. This site
    proposes one such cut: a suspicious branch node where two long arms meet, plus
    a seed point deep inside each arm (so the score-time nearest-seed assignment
    has a clean margin).

    DEPLOYABLE: every field is derived from fragment geometry alone (no ground
    truth, no image), so a policy may act on it on held-out/test data. The features
    are advisory signals the evolved ``propose_edits`` can threshold on; the
    generator does NOT decide — it only enumerates plausible cuts.

    Attributes
    ----------
    label : str
        The raw fragment/segment id suspected of fusing two neurites.
    cut_node : int
        The branch node (degree >= 3) where the two arms meet — the cut location.
    cut_xyz : tuple[float, float, float]
        Physical coordinate (microns) of ``cut_node``.
    seed_a_node, seed_b_node : int
        A node well inside each of the two longest arms (>= ``seed_depth_um`` from
        the cut where possible). These define the split's two sides.
    seed_a_xyz, seed_b_xyz : tuple[float, float, float]
        Physical coordinates (microns) of the two seeds. These are what a
        ``split_label`` edit consumes today (nearest-seed assignment).
    branch_degree : int
        Degree of ``cut_node`` (3 = simple bifurcation, 4 = X-crossing, ...). A
        higher degree at a single node is a stronger merge signal.
    angle_deg : float
        Angle (degrees) between the two arms' tangents at the cut. Two arms of ONE
        neuron pass through ~straight (near 180); a merge of two unrelated neurites
        tends to a sharper, more arbitrary angle. NaN if a tangent is undefined.
    radius_ratio : float | None
        max(r_a, r_b) / min(r_a, r_b) of the two arms' mean neurite radius. Far
        from 1.0 suggests two different cable calibers fused. None if unavailable.
    cable_a_um, cable_b_um : float
        Cable length (microns) of each arm — both being long is what makes the cut
        meaningful (a short spur is not a merge of two real neurons).
    detector : str
        Which GT-free detector proposed this site, so the policy / diagnosis can
        treat the topologies differently:
          "branch"    — a degree>=3 node where two long arms meet (the original).
          "bridge"    — a thin degree-2 neck whose removal splits the label into two
                        long components (a fusion with NO branch node — the topology
                        the branch detector structurally cannot see). ``cut_node`` is
                        the neck's midpoint; ``branch_degree`` is 2.
          "component" — one raw label spread over >=2 DISCONNECTED graph components,
                        each substantial (a label fused across unconnected neurites).
                        ``cut_node`` is the nearest-pair midpoint; ``angle_deg`` is
                        NaN (no shared vertex to take a tangent at).
    """

    kind: str = field(default="merge", init=False)  # site-type tag for the policy

    label: str
    cut_node: int
    cut_xyz: tuple
    seed_a_node: int
    seed_b_node: int
    seed_a_xyz: tuple
    seed_b_xyz: tuple
    branch_degree: int
    angle_deg: float
    radius_ratio: float | None
    cable_a_um: float
    cable_b_um: float
    detector: str = "branch"

    def as_edit(self) -> dict:
        """The ``split_label`` edit this site proposes if accepted.

        Emits the multi-seed ``seeds`` schema (with the two arm seeds and their
        originating fragment-graph node ids), which ``EditHandler`` partitions by
        GRAPH path distance when fragment graphs are available (and falls back to
        Euclidean nearest-seed otherwise). Multiple ``MergeSite``s on the same raw
        label are COMPOSED into one multi-seed partition by the handler — no
        last-one-wins — so several branch cuts on one fused segment cooperate.
        The legacy ``seed_a_xyz`` / ``seed_b_xyz`` keys are kept alongside for
        backward compatibility with any consumer that reads them directly.
        """
        return {
            "kind": "split_label",
            "label": self.label,
            "seeds": [
                {"suffix": "a", "xyz": self.seed_a_xyz, "node": self.seed_a_node},
                {"suffix": "b", "xyz": self.seed_b_xyz, "node": self.seed_b_node},
            ],
            # Legacy two-seed keys (still accepted by EditHandler).
            "seed_a_xyz": self.seed_a_xyz,
            "seed_b_xyz": self.seed_b_xyz,
        }


def _arm_from_branch(g, branch, neighbor, max_depth_um):
    """Walk one arm out of ``branch`` starting toward ``neighbor``.

    Follows the component away from the branch node, stopping at ``max_depth_um``
    cable or at the next branch/tip. Returns (nodes_in_order, cable_um). The walk
    refuses to cross back through ``branch`` so the two arms of a bifurcation stay
    disjoint. At an internal branch (degree>2 reached mid-arm) it stops — we only
    characterize the arm up to the first complication.
    """
    arm = [branch, neighbor]
    cable = g.dist(branch, neighbor)
    prev, cur = branch, neighbor
    while cable < max_depth_um:
        nbrs = [n for n in g.neighbors(cur) if n != prev]
        if len(nbrs) != 1:  # tip (0) or branch (>=2) — stop characterizing here
            break
        nxt = nbrs[0]
        cable += g.dist(cur, nxt)
        arm.append(nxt)
        prev, cur = cur, nxt
    return arm, float(cable)


def candidate_merge_sites(
    fragments_graph,
    min_arm_cable_um: float = 10.0,
    seed_depth_um: float = 8.0,
    max_sites: int = 5000,
    max_per_label: int = 8,
) -> list[MergeSite]:
    """Enumerate candidate merge-repair sites from fragment geometry alone.

    DEPLOYABLE & GT-FREE (see MergeSite). A merge is one segment id covering two real
    neurites; that shows up in THREE skeleton topologies, each enumerated by its own
    detector and tagged on ``MergeSite.detector`` (P1-1 — the branch scan alone
    structurally misses the latter two):

      * "branch"    — a degree>=3 node where two long arms meet (touch / crossing).
      * "bridge"    — a thin degree-2 NECK with a sharp kink between two long sides
                      (two neurites fused end-to-end; there is NO branch node).
      * "component" — one label spread over >=2 DISCONNECTED graph components, each
                      substantial (a fusion the skeletonization never joined).

    In every case both sides must exceed ``min_arm_cable_um`` (a short spur is not a
    merge), each side is seeded ``seed_depth_um`` in, and advisory features (degree,
    angle, radius ratio, cables) are attached. The generator only PROPOSES; the
    evolved policy thresholds on the features and may branch on ``detector``.

    Parameters
    ----------
    fragments_graph : SkeletonGraph (an nx.Graph subclass)
        The cached fragment graph (``dataset.fragments_graph``).
    min_arm_cable_um : float
        Both of a branch's two longest arms must reach this cable length for the
        branch to be a merge candidate. Filters spurs/noise.
    seed_depth_um : float
        Target distance from the cut to place each seed (a clean assignment margin).
    max_sites : int
        Global cap on returned sites (strongest signal first).
    max_per_label : int
        Cap on candidates emitted per label, so one tangled segment can't flood the
        candidate set. Strongest (by min-arm cable) kept.

    Returns
    -------
    list[MergeSite] sorted by descending ``min(cable_a_um, cable_b_um)`` — the
    candidates whose two arms are both longest (most confidently two real neurons)
    come first.
    """
    g = fragments_graph
    if g.number_of_nodes() == 0:
        return []

    radius = getattr(g, "node_radius", None)

    def tangent(arm):
        """Unit direction from the cut along an arm (cut node is arm[0])."""
        if len(arm) < 2:
            return None
        v = np.asarray(g.node_xyz[arm[-1]], dtype=float) - np.asarray(
            g.node_xyz[arm[0]], dtype=float)
        n = np.linalg.norm(v)
        return v / n if n > 0 else None

    def seed_in_arm(arm):
        """Node ~seed_depth_um into the arm (clamp to far end if arm is shorter)."""
        cable = 0.0
        for k in range(1, len(arm)):
            cable += g.dist(arm[k - 1], arm[k])
            if cable >= seed_depth_um:
                return arm[k]
        return arm[-1]

    def arm_mean_radius(arm):
        if radius is None or len(arm) == 0:
            return None
        vals = [float(radius[n]) for n in arm if radius[n] > 0]
        return float(np.mean(vals)) if vals else None

    per_label: dict[str, list[MergeSite]] = {}
    # Branch nodes only — degree>=3 is where two arms can meet within one label.
    for node in g.nodes:
        deg = g.degree[node]
        if deg < 3:
            continue
        label = str(g.node_segment_id(node))
        if label == "0":
            continue

        # Characterize each arm out of this branch; keep those long enough.
        arms = []
        for nbr in g.neighbors(node):
            arm, cable = _arm_from_branch(g, node, nbr, seed_depth_um * 3)
            arms.append((cable, arm))
        arms.sort(key=lambda t: t[0], reverse=True)
        if len(arms) < 2:
            continue
        (cable_a, arm_a), (cable_b, arm_b) = arms[0], arms[1]
        if cable_b < min_arm_cable_um:  # second-longest arm too short -> spur, not merge
            continue

        ta, tb = tangent(arm_a), tangent(arm_b)
        if ta is not None and tb is not None:
            cos = float(np.clip(np.dot(ta, tb), -1.0, 1.0))
            angle_deg = float(np.degrees(np.arccos(cos)))
        else:
            angle_deg = float("nan")

        ra, rb = arm_mean_radius(arm_a), arm_mean_radius(arm_b)
        if ra and rb and min(ra, rb) > 0:
            radius_ratio = max(ra, rb) / min(ra, rb)
        else:
            radius_ratio = None

        sa, sb = seed_in_arm(arm_a), seed_in_arm(arm_b)
        site = MergeSite(
            label=label,
            cut_node=int(node),
            cut_xyz=tuple(map(float, g.node_xyz[node])),
            seed_a_node=int(sa),
            seed_b_node=int(sb),
            seed_a_xyz=tuple(map(float, g.node_xyz[sa])),
            seed_b_xyz=tuple(map(float, g.node_xyz[sb])),
            branch_degree=int(deg),
            angle_deg=angle_deg,
            radius_ratio=radius_ratio,
            cable_a_um=float(cable_a),
            cable_b_um=float(cable_b),
            detector="branch",
        )
        per_label.setdefault(label, []).append(site)

    # --- Additional GT-free detectors (P1-1): topologies the branch scan misses ---
    # A real merge often has NO degree>=3 node — two neurites fused through a thin
    # degree-2 neck, or one label spread over disconnected components. Both are
    # detectable from fragment geometry alone, so they are deployable on held-out.
    for site in _bridge_merge_sites(g, min_arm_cable_um, seed_depth_um):
        per_label.setdefault(site.label, []).append(site)
    for site in _component_merge_sites(g, min_arm_cable_um, seed_depth_um):
        per_label.setdefault(site.label, []).append(site)

    # Per-label cap (strongest = both arms longest), then global sort + cap.
    sites: list[MergeSite] = []
    for label, label_sites in per_label.items():
        label_sites.sort(key=lambda s: min(s.cable_a_um, s.cable_b_um), reverse=True)
        sites.extend(label_sites[:max_per_label])
    sites.sort(key=lambda s: min(s.cable_a_um, s.cable_b_um), reverse=True)
    return sites[:max_sites]


def _walk_until(g, start, prev, target_um):
    """Walk a degree-<=2 chain from ``start`` (came from ``prev``) up to ``target_um``.

    Stops at ``target_um`` cable, or at the first tip / branch (degree != 2).
    Returns ``(nodes_in_order_including_start, cable_um)``. Used by the bridge
    detector to measure how much cable lies on each side of a candidate neck and to
    place a seed deep into each side.
    """
    chain = [start]
    cable = 0.0
    p, cur = prev, start
    while cable < target_um:
        nbrs = [n for n in g.neighbors(cur) if n != p]
        if len(nbrs) != 1:  # tip (0) or branch (>=2): chain ends here
            break
        nxt = nbrs[0]
        cable += g.dist(cur, nxt)
        chain.append(nxt)
        p, cur = cur, nxt
    return chain, float(cable)


def _bridge_merge_sites(g, min_arm_cable_um, seed_depth_um,
                        max_kink_angle_deg: float = 120.0,
                        min_separation_um: float = 20.0):
    """GT-free: merges with NO branch node — two neurites fused at a thin neck.

    A fusion between two neurites that meet end-to-end leaves a degree-2 node (a
    "neck"), not a branch, so the degree>=3 scan in ``candidate_merge_sites`` cannot
    see it. In a tree-like skeleton almost every interior degree-2 node trivially
    separates two long arms, so "both sides long" alone would flood; the actual
    merge signature at a neck is a GEOMETRIC KINK — two otherwise-straight neurites
    joined at a sharp turn. We therefore flag a degree-2 node only when:
      * both sides carry >= ``min_arm_cable_um`` cable (two real neurites), AND
      * the turn angle through the node is sharp (< ``max_kink_angle_deg`` between
        the two side tangents — a single neuron runs ~straight, ~180°),
    then suppress near-duplicate kinks within ``min_separation_um`` (keep sharpest).

    Yields ``MergeSite(detector="bridge")``. ``cut_node`` is the neck; the two seeds
    sit ``seed_depth_um`` into each side. Deployable (geometry only).
    """
    radius = getattr(g, "node_radius", None)
    candidates = []  # (sharpness, node, side_a_chain, side_b_chain, cable_a, cable_b)
    for node in g.nodes:
        if g.degree[node] != 2:
            continue
        label = str(g.node_segment_id(node))
        if label == "0":
            continue
        a, b = list(g.neighbors(node))
        chain_a, cable_a = _walk_until(g, a, node, seed_depth_um * 3)
        chain_b, cable_b = _walk_until(g, b, node, seed_depth_um * 3)
        if cable_a < min_arm_cable_um or cable_b < min_arm_cable_um:
            continue
        # Turn angle at the neck: tangents from the node out along each side.
        va = np.asarray(g.node_xyz[chain_a[-1]], float) - np.asarray(g.node_xyz[node], float)
        vb = np.asarray(g.node_xyz[chain_b[-1]], float) - np.asarray(g.node_xyz[node], float)
        na, nb = np.linalg.norm(va), np.linalg.norm(vb)
        if na == 0 or nb == 0:
            continue
        cos = float(np.clip(np.dot(va, vb) / (na * nb), -1.0, 1.0))
        angle_deg = float(np.degrees(np.arccos(cos)))  # ~180 = straight, low = kink
        if angle_deg > max_kink_angle_deg:
            continue  # too straight to be a merge neck
        sharpness = max_kink_angle_deg - angle_deg
        candidates.append((sharpness, node, chain_a, chain_b, cable_a, cable_b, angle_deg, label))

    # Suppress near-duplicate necks (a gentle bend spans several degree-2 nodes):
    # keep the sharpest within min_separation_um.
    candidates.sort(reverse=True, key=lambda t: t[0])
    kept_xyz: list = []
    for sharp, node, chain_a, chain_b, cable_a, cable_b, angle_deg, label in candidates:
        xyz = np.asarray(g.node_xyz[node], float)
        if any(np.linalg.norm(xyz - k) < min_separation_um for k in kept_xyz):
            continue
        kept_xyz.append(xyz)

        def seed_of(chain):
            cable = 0.0
            for k in range(1, len(chain)):
                cable += g.dist(chain[k - 1], chain[k])
                if cable >= seed_depth_um:
                    return chain[k]
            return chain[-1]

        sa, sb = seed_of(chain_a), seed_of(chain_b)
        ra = _mean_radius(radius, chain_a)
        rb = _mean_radius(radius, chain_b)
        radius_ratio = (max(ra, rb) / min(ra, rb)) if (ra and rb and min(ra, rb) > 0) else None
        yield MergeSite(
            label=label,
            cut_node=int(node),
            cut_xyz=tuple(map(float, g.node_xyz[node])),
            seed_a_node=int(sa),
            seed_b_node=int(sb),
            seed_a_xyz=tuple(map(float, g.node_xyz[sa])),
            seed_b_xyz=tuple(map(float, g.node_xyz[sb])),
            branch_degree=2,
            angle_deg=angle_deg,
            radius_ratio=radius_ratio,
            cable_a_um=float(cable_a),
            cable_b_um=float(cable_b),
            detector="bridge",
        )


def _component_merge_sites(g, min_arm_cable_um, seed_depth_um):
    """GT-free: one raw label spread over >=2 DISCONNECTED graph components.

    If a single segment id labels two skeleton pieces that are NOT graph-connected,
    the segmentation fused two neurites that the skeletonization never joined — a
    merge with no shared vertex at all. We group each label's nodes by connected
    component, keep components with >= ``min_arm_cable_um`` cable, and emit one site
    per adjacent substantial pair (seed = the node of each component nearest the
    other component, so the split plane sits at the contact). ``angle_deg`` is NaN
    (no shared vertex). Deployable (geometry only).
    """
    radius = getattr(g, "node_radius", None)
    # Group nodes by label, then by connected component within the label.
    label_nodes: dict[str, list[int]] = {}
    for n in g.nodes:
        lab = str(g.node_segment_id(n))
        if lab == "0":
            continue
        label_nodes.setdefault(lab, []).append(n)

    for label, nodes in label_nodes.items():
        node_set = set(nodes)
        seen: set = set()
        comps: list[list[int]] = []
        for n in nodes:
            if n in seen:
                continue
            # BFS within this label's node set only.
            stack, comp = [n], []
            seen.add(n)
            while stack:
                u = stack.pop()
                comp.append(u)
                for v in g.neighbors(u):
                    if v in node_set and v not in seen:
                        seen.add(v)
                        stack.append(v)
            comps.append(comp)
        if len(comps) < 2:
            continue  # connected — not a multi-component merge

        # Component cable length (sum of incident-edge half-lengths within comp).
        def comp_cable(comp):
            cs = set(comp)
            total = 0.0
            for u in comp:
                for v in g.neighbors(u):
                    if v in cs and v > u:
                        total += g.dist(u, v)
            return total

        sized = [(comp_cable(c), c) for c in comps]
        sized = [(cl, c) for cl, c in sized if cl >= min_arm_cable_um]
        if len(sized) < 2:
            continue
        sized.sort(reverse=True, key=lambda t: t[0])

        # Emit sites for the largest component paired with each other substantial
        # one (the per-label cap downstream trims if there are many).
        cable_a, comp_a = sized[0]
        coords_a = np.array([g.node_xyz[u] for u in comp_a], float)
        tree_a = KDTree(coords_a)
        for cable_b, comp_b in sized[1:]:
            coords_b = np.array([g.node_xyz[u] for u in comp_b], float)
            # Closest cross-component node pair -> the contact location (KD-tree so
            # this stays cheap even for large components).
            dists, idx_a = tree_a.query(coords_b, k=1)
            ib = int(np.argmin(dists))
            ia = int(idx_a[ib])
            na_node, nb_node = comp_a[ia], comp_b[ib]
            mid = tuple((np.asarray(g.node_xyz[na_node], float)
                         + np.asarray(g.node_xyz[nb_node], float)) / 2.0)
            ra = _mean_radius(radius, comp_a)
            rb = _mean_radius(radius, comp_b)
            rr = (max(ra, rb) / min(ra, rb)) if (ra and rb and min(ra, rb) > 0) else None
            yield MergeSite(
                label=label,
                cut_node=int(na_node),
                cut_xyz=tuple(map(float, mid)),
                seed_a_node=int(na_node),
                seed_b_node=int(nb_node),
                seed_a_xyz=tuple(map(float, g.node_xyz[na_node])),
                seed_b_xyz=tuple(map(float, g.node_xyz[nb_node])),
                branch_degree=0,
                angle_deg=float("nan"),
                radius_ratio=rr,
                cable_a_um=float(cable_a),
                cable_b_um=float(cable_b),
                detector="component",
            )


def _mean_radius(radius, nodes):
    """Mean positive neurite radius over ``nodes`` (None if unavailable)."""
    if radius is None or len(nodes) == 0:
        return None
    vals = [float(radius[n]) for n in nodes if radius[n] > 0]
    return float(np.mean(vals)) if vals else None


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

    merges = candidate_merge_sites(frags)
    from collections import Counter
    by_det = Counter(s.detector for s in merges)
    print(f"\n{len(merges)} candidate merge sites by detector: {dict(by_det)}. "
          f"Strongest 5:")
    for s in merges[:5]:
        rr = f"{s.radius_ratio:.2f}" if s.radius_ratio else "n/a"
        print(f"  [{s.detector}] label {s.label}  deg={s.branch_degree}  "
              f"arms={s.cable_a_um:.1f}/{s.cable_b_um:.1f}um  "
              f"angle={s.angle_deg:.0f}deg  rratio={rr}")

    print(f"\n{len(list_fragment_labels(frags))} distinct fragment labels.")
