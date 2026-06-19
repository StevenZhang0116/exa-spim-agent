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


# --- Evolvable enumeration priors -------------------------------------------
# These knobs decide WHAT COUNTS AS A CANDIDATE (the framework's prior on "what a
# split/merge error looks like"), as opposed to which candidates the policy then
# accepts. They used to be hardcoded defaults — a HARD prior the evolved policy
# could not move. Exposing them turns that into a SOFT, evolvable prior: the policy
# artifact may define a module-level ``ENUM_PARAMS`` dict to widen/narrow the
# candidate stream (e.g. raise ``max_gap_um`` to reach longer true gaps, lower
# ``min_arm_cable_um`` to surface shorter merges, flip ``tip_to_shaft``).
#
# Each entry is (default, lo, hi). Values are CLAMPED to [lo, hi] — the bounds are a
# safety rail: the enumeration is a whole-brain geometric scan whose cost grows with
# the gap radius and candidate count, so an unbounded ``max_gap_um`` could make the
# scan explode (the same reason ``max_class_size`` caps merges). Out-of-range or
# unknown keys are clamped/ignored, never crash the run.
ENUM_PARAM_SPEC = {
    # candidate_split_sites
    "max_gap_um":       (15.0,  1.0,  40.0),   # tip->partner search radius (µm)
    "split_max_sites":  (5000,  100,  50000),  # cap on split candidates
    "tip_to_shaft":     (True,  None, None),   # bool: partners may be shaft/branch
    # candidate_merge_sites
    "min_arm_cable_um": (10.0,  2.0,  50.0),   # both arms must reach this (µm)
    "seed_depth_um":    (8.0,   2.0,  30.0),   # seed placement depth into each arm
    "merge_max_sites":  (5000,  100,  50000),  # global cap on merge candidates
    "max_per_label":    (8,     1,    100),    # cap on merge sites per raw label
}


def resolve_enum_params(raw: dict | None) -> dict:
    """Validate + clamp a policy-supplied ``ENUM_PARAMS`` dict to the safe schema.

    Returns a full param dict (every key present) with each value clamped to its
    ``[lo, hi]`` rail; missing keys take the default; unknown keys are dropped.
    Booleans pass through by ``bool()``. Non-numeric / unparseable values fall back
    to the default rather than raising — a bad policy must never crash enumeration.
    """
    raw = raw or {}
    out = {}
    for key, (default, lo, hi) in ENUM_PARAM_SPEC.items():
        if key not in raw:
            out[key] = default
            continue
        v = raw[key]
        if lo is None:  # boolean knob
            try:
                out[key] = bool(v)
            except Exception:
                out[key] = default
            continue
        try:
            v = type(default)(v)  # coerce to int/float like the default
        except (TypeError, ValueError):
            out[key] = default
            continue
        out[key] = max(lo, min(hi, v))  # clamp to the safety rail
    return out


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


def kfold_split(gt_swc_names: list[str], k: int = 4, seed: int = 0) -> list[dict]:
    """Partition GT skeleton names into ``k`` leak-free cross-validation folds.

    Each fold is ``{"fold": i, "heldout": [...], "train": [...]}`` where ``heldout``
    is one of ``k`` disjoint, near-equal-size groups and ``train`` is everything
    else. Splitting by WHOLE skeleton (never an edge) keeps each fold's held-out
    metric an honest generalization signal — no skeleton is in both its own train
    and held-out.

    Why K-fold instead of one random split: with few GT neurons (here ~12) a single
    4-neuron held-out gives a near-quantized fitness whose value is dominated by
    WHICH 4 were drawn, so a real policy improvement is invisible against split
    noise. Scoring on ``k`` rotating held-out groups and gating on the AGGREGATE
    turns ``k`` noisy points into a stable mean (and exposes per-fold variance), so
    a small true improvement becomes detectable. Scoring cost does not grow: the
    scorer already returns every skeleton's row in one pass, so the folds are just
    row slices of that one result.

    ``k`` is clamped to ``[1, len(names)]``; ``k == 1`` degenerates to a single fold
    whose held-out is the WHOLE set (used by callers that want the legacy single
    train/held-out split via ``train_heldout_split`` instead — see run_evolution).
    """
    names = sorted(gt_swc_names)
    n = len(names)
    if n == 0:
        return []
    k = max(1, min(int(k), n))
    rng = np.random.default_rng(seed)
    order = rng.permutation(n).tolist()
    # Near-equal contiguous chunks of the shuffled order (sizes differ by <=1).
    folds = []
    for i in range(k):
        held_pos = order[i::k]  # stride partition -> balanced, deterministic
        held = sorted(names[p] for p in held_pos)
        held_set = set(held)
        train = [nm for nm in names if nm not in held_set]
        folds.append({"fold": i, "heldout": held, "train": train})
    return folds


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
    arms_reconverge : bool | None
        Whether the two arms re-join downstream within the same label. True is a
        STRONG non-merge signal — two truly separate neurons never reconnect, but
        one neuron's own branches (a real bifurcation) or a loop do. None when not
        computed (the "component" detector, or no usable fragment topology). A
        precise policy should be far more reluctant to cut when this is True.
    extra_seeds : list[dict]
        Third+ arm seeds for a high-degree fusion (``branch_degree >= 4``, e.g. an
        X-crossing that must be cut into 3+ pieces). Each dict is
        ``{"suffix": str, "xyz": (x,y,z), "node": int, "cable_um": float}``. Empty
        for an ordinary bifurcation (``seed_a``/``seed_b`` already cover both
        sides). ``as_edit()`` folds these into the multi-seed ``seeds`` list so the
        handler partitions the label into one side per arm.
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

    # --- Stronger discriminative signals (added; default-safe for all detectors) ---
    # arms_reconverge: do the two arms re-join downstream within the same label?
    # A True here is a strong NON-merge signal: a real merge fuses two SEPARATE
    # neurons whose arms never reconnect, whereas one neuron's own branches (a true
    # bifurcation, or a loop) come back together. None = not computed (component
    # detector / no fragment topology). The policy should be far more reluctant to
    # cut when this is True.
    arms_reconverge: bool | None = None
    # extra_seeds: third+ arm seeds for a high-degree fusion (branch_degree >= 4,
    # e.g. an X-crossing that must be cut into 3+ pieces). Each is
    # {"suffix": str, "xyz": (x,y,z), "node": int, "cable_um": float}. Empty for the
    # common bifurcation case (seed_a/seed_b already cover it). as_edit() folds these
    # into the multi-seed `seeds` list so EditHandler partitions into >2 sides.
    extra_seeds: list = field(default_factory=list)
    # seed_groups: an explicit arm->neurite GROUPING for branch/crossing fusions.
    # A degree-4 X is two neurites passing through, NOT four separate sides: the two
    # collinear (opposite-tangent) arms belong to ONE neuron. When the branch
    # detector can pair arms by tangent it fills this with one entry per arm, each
    # {"suffix": str, "xyz": (x,y,z), "node": int}, where arms of the SAME neurite
    # SHARE a suffix (e.g. opposite arms both "a", the crossing pair both "b"). When
    # non-empty, as_edit() emits exactly these seeds (so EditHandler's multi-source
    # Dijkstra cuts the label into one side PER NEURITE, not per arm). Empty =>
    # fall back to the seed_a/seed_b/extra_seeds path (bridge/component, or when no
    # confident pairing exists). Advisory: the policy still decides whether to cut.
    seed_groups: list = field(default_factory=list)

    def as_edit(self) -> dict:
        """The ``split_label`` edit this site proposes if accepted.

        Emits the multi-seed ``seeds`` schema (with the two arm seeds and their
        originating fragment-graph node ids), which ``EditHandler`` partitions by
        GRAPH path distance when fragment graphs are available (and falls back to
        Euclidean nearest-seed otherwise). Multiple ``MergeSite``s on the same raw
        label are COMPOSED into one multi-seed partition by the handler — no
        last-one-wins — so several branch cuts on one fused segment cooperate.

        When ``extra_seeds`` is non-empty (a high-degree fusion), those arm seeds are
        appended so the handler partitions the label into >2 sides — one per arm of
        the crossing. The legacy ``seed_a_xyz`` / ``seed_b_xyz`` keys are kept
        alongside for backward compatibility with any consumer that reads them.
        """
        # Preferred: an explicit arm->neurite grouping (shared suffix => same
        # neurite). EditHandler shares a suffix across Dijkstra sources, so this cuts
        # the label into one side per NEURITE rather than per arm.
        if self.seed_groups:
            seeds = [
                {"suffix": spec["suffix"], "xyz": spec["xyz"], "node": spec.get("node")}
                for spec in self.seed_groups
            ]
            return {
                "kind": "split_label",
                "label": self.label,
                "seeds": seeds,
                "seed_a_xyz": self.seed_a_xyz,
                "seed_b_xyz": self.seed_b_xyz,
            }
        seeds = [
            {"suffix": "a", "xyz": self.seed_a_xyz, "node": self.seed_a_node},
            {"suffix": "b", "xyz": self.seed_b_xyz, "node": self.seed_b_node},
        ]
        for k, spec in enumerate(self.extra_seeds):
            seeds.append({
                "suffix": spec.get("suffix") or chr(ord("c") + k),
                "xyz": spec["xyz"],
                "node": spec.get("node"),
            })
        return {
            "kind": "split_label",
            "label": self.label,
            "seeds": seeds,
            # Legacy two-seed keys (still accepted by EditHandler).
            "seed_a_xyz": self.seed_a_xyz,
            "seed_b_xyz": self.seed_b_xyz,
        }


def _pair_arms_into_neurites(g, long_arms, tangent, seed_in_arm,
                             min_pair_cos: float = 0.7):
    """Group a branch's long arms into neurites by tangent, return grouped seeds.

    An X-crossing (degree 4) is two neurites passing through: the two arms whose
    tangents are most anti-parallel (continue straight across the node) are ONE
    neuron. Greedily pair arms by most-opposite tangent; each confident pair shares
    a suffix (so EditHandler cuts the label into one side per NEURITE, not per arm).

    Returns a ``seed_groups`` list (``[{"suffix", "xyz", "node"}, ...]``) when a
    confident, complete pairing exists (an even number of long arms, every pair
    anti-parallel enough: ``-dot >= min_pair_cos``). Returns ``[]`` otherwise — an
    odd arm count, a degree-3 bifurcation (no through-pairing to infer), or any pair
    too perpendicular to call — so the caller falls back to per-arm seeds.
    """
    # Need tangents for every long arm; a degree-3 (3 long arms) has no clean
    # through-pairing, and an odd count can't pair fully — bail to per-arm seeds.
    if len(long_arms) < 4 or len(long_arms) % 2 != 0:
        return []
    tans = []
    for cable, arm in long_arms:
        t = tangent(arm)
        if t is None:
            return []
        tans.append((t, arm))

    remaining = list(range(len(tans)))
    pairs = []
    while remaining:
        i = remaining.pop(0)
        ti = tans[i][0]
        # Best partner = most anti-parallel tangent (a straight pass-through).
        best_j, best_score = None, -2.0
        for j in remaining:
            score = -float(np.dot(ti, tans[j][0]))  # 1.0 = perfectly opposite
            if score > best_score:
                best_score, best_j = score, j
        if best_j is None or best_score < min_pair_cos:
            return []  # no confident through-partner -> don't guess a grouping
        remaining.remove(best_j)
        pairs.append((i, best_j))

    seed_groups = []
    for p, (i, j) in enumerate(pairs):
        suffix = chr(ord("a") + p)
        for idx in (i, j):
            _, arm = tans[idx]
            sk = seed_in_arm(arm)
            seed_groups.append({
                "suffix": suffix,
                "xyz": tuple(map(float, g.node_xyz[sk])),
                "node": int(sk),
            })
    return seed_groups


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


def _arms_reconverge(g, branch, first_a, first_b, max_explore_um, max_steps=400):
    """Do two arms of ``branch`` re-join downstream within the SAME label?

    Returns True if, with ``branch`` REMOVED, the arm that starts at ``first_a``
    (the branch's first step into arm A) can still reach ``first_b`` (the first step
    into arm B) along same-label edges. Reaching it means the two arms reconnect
    somewhere downstream — a loop, or two branches of ONE neuron — which is a strong
    NON-merge signal (two separate fused neurons never reconnect).

    PERFORMANCE: this runs once per branch node over a whole-brain fragment graph
    (hundreds of thousands of fragments), and a genuine merge does NOT reconverge,
    so the common case must FAIL FAST. The search is therefore tightly bounded:
    ``max_steps`` node-visits AND ``max_explore_um`` cable. A reconvergence that
    exists is almost always LOCAL (a small loop / a nearby re-branch), so a tight
    bound keeps near-100% of true re-joins while turning the dominant "no re-join"
    case into a cheap bounded probe instead of a brain-wide flood. A re-join farther
    than the bound is reported False (missed) — an acceptable, conservative miss:
    it only means the policy loses one advisory hint, never a correctness error.
    """
    label = g.node_segment_id(branch)
    seen = {branch, first_a}
    frontier = [(first_a, 0.0)]
    steps = 0
    while frontier and steps < max_steps:
        cur, dist = frontier.pop()
        steps += 1
        if cur == first_b:
            return True
        if dist > max_explore_um:
            continue
        for nbr in g.neighbors(cur):
            if nbr in seen or nbr == branch:
                continue  # never route back through the cut node
            if g.node_segment_id(nbr) != label:
                continue  # stay within the same fused label
            seen.add(nbr)
            frontier.append((nbr, dist + g.dist(cur, nbr)))
    return False


def candidate_merge_sites(
    fragments_graph,
    min_arm_cable_um: float = 10.0,
    seed_depth_um: float = 8.0,
    max_sites: int = 5000,
    max_per_label: int = 8,
    check_reconvergence: bool = False,
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
    check_reconvergence : bool
        Compute ``MergeSite.arms_reconverge`` (a per-branch bounded BFS). OFF by
        default: on a whole-brain graph it dominates enumeration time (a genuine
        merge never re-joins, so every check runs to its bound), while the signal is
        only advisory. Leave it off for the evolution loop; turn it on only for
        small-scope analysis. When off, ``arms_reconverge`` is None on every site.

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

        # Reconvergence check: do the two arms re-join downstream (one neuron's
        # branches / a loop) rather than belong to two fused neurons? This is a
        # bounded BFS PER BRANCH NODE; on a whole-brain fragment graph (millions of
        # nodes, many branch nodes) it dominates enumeration time even when bounded,
        # because a genuine merge never re-joins and so always runs to the bound.
        # OFF by default for that reason — the signal is advisory, never required for
        # correctness. A policy that wants it can re-enumerate with
        # check_reconvergence=True on a smaller scope, or compute it itself from
        # ctx["fragments_graph"] for just the few sites it is actually weighing.
        if check_reconvergence:
            arms_reconverge = _arms_reconverge(
                g, node, arm_a[1], arm_b[1], max_explore_um=seed_depth_um * 8
            )
        else:
            arms_reconverge = None

        # High-degree fusion (X-crossing, deg>=4): a single cut into two sides is
        # not enough — emit a seed for each ADDITIONAL long arm so EditHandler can
        # partition the label into one side per arm. Only arms clearing the cable
        # floor count (short spurs are not separate neurites).
        extra_seeds = []
        for k, (cable_k, arm_k) in enumerate(arms[2:]):
            if cable_k < min_arm_cable_um:
                break  # arms are cable-sorted desc; the rest are shorter spurs
            sk = seed_in_arm(arm_k)
            extra_seeds.append({
                "suffix": chr(ord("c") + k),
                "xyz": tuple(map(float, g.node_xyz[sk])),
                "node": int(sk),
                "cable_um": float(cable_k),
            })

        # An X-crossing (degree 4) is two neurites passing THROUGH the node, not four
        # separate sides: the two arms that continue roughly straight across (tangents
        # closest to anti-parallel) belong to ONE neuron. Pair the long arms by
        # tangent so a confident grouping cuts the label into one side per NEURITE.
        # Only attempted for an even count of long arms with usable tangents; left
        # empty otherwise (falls back to per-arm seeds, the prior behavior).
        seed_groups = _pair_arms_into_neurites(
            g, [(cable, arm) for cable, arm in arms if cable >= min_arm_cable_um],
            tangent, seed_in_arm,
        )

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
            arms_reconverge=arms_reconverge,
            extra_seeds=extra_seeds,
            seed_groups=seed_groups,
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
