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

import hashlib
import os
import pickle
import time
from dataclasses import dataclass, field

import networkx as nx
import numpy as np
from scipy.spatial import KDTree


# --- Coordinate-frame bridge: agentic SkeletonGraph -> metrics LabeledGraph --
# The agentic cache and the metrics package read the SAME ground-truth SWC voxel
# files with the SAME anisotropy (verified: cache anisotropy == BrainPaths
# anisotropy == (0.748, 0.748, 1.0)), but store the resulting physical coordinate
# in DIFFERENT axis orders:
#
#   * agentic SkeletonGraph.node_xyz  = anisotropy * swc_columns          (no reversal)
#         (utils/swc_util.Reader.read_coordinate: a * (s + offset), in file order)
#   * metrics  LabeledGraph.node_xyz(i) = node_voxel[i][::-1] * anisotropy (REVERSED)
#         (data_handling/graph_classes.py: voxels stored in file order, then reversed)
#
# So for one physical point with SWC columns (c0, c1, c2) and anisotropy a:
#       agentic = (c0*a0, c1*a1, c2*a2)
#       metrics = (c2*a0, c1*a1, c0*a2)
# i.e. axes 0 and 2 are swapped (and each keeps the anisotropy factor of its NEW
# position). A split_label seed is read from the agentic graph (g.node_xyz[node])
# but consumed by the scorer in the metrics frame (EditHandler snaps it against the
# metrics fragment graphs' node_xyz), so it MUST be converted at that boundary or it
# lands tens of thousands of microns away (verify.py's split-contract check (a)).
#
# The conversion is anisotropy-exact (no hardcoded 0.748): recover the voxel
# (divide by a), reverse the axes, re-apply a. With matching anisotropy this is a
# pure (x,y,z) -> (z*a0/a2, y, x*a2/a0) remap.
_METRICS_ANISOTROPY = (0.748, 0.748, 1.0)


def agentic_xyz_to_metrics_frame(xyz, anisotropy=_METRICS_ANISOTROPY):
    """Convert an agentic-graph physical coordinate into the metrics-graph frame.

    See the module note above for the derivation. ``xyz`` is an agentic
    ``node_xyz`` (microns, file-axis order); the return is the same physical point
    expressed the way ``segmentation_skeleton_metrics`` ``LabeledGraph.node_xyz(i)``
    would, so a ``split_label`` seed snaps to the correct fragment node.

    Identity when ``anisotropy`` is isotropic AND the caller's frames already agree;
    the reversal is always applied (it is the axis-order half of the mismatch, which
    is present regardless of anisotropy).
    """
    a = np.asarray(anisotropy, dtype=float)
    p = np.asarray(xyz, dtype=float)
    voxel = p / a                 # back to (file-order) voxel coordinates
    return tuple((voxel[::-1] * a).tolist())   # reverse axes, re-apply anisotropy


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
    "split_max_sites":  (5000,  100,  50000),  # GLOBAL cap on split candidates (final)
    "split_per_tip_k":  (4,     1,    32),     # PER-TIP quota: keep only each tip's k
                                               # closest differently-labelled partners
                                               # BEFORE the global split_max_sites cap,
                                               # so a dense region cannot consume the
                                               # whole budget and starve a sparse tip's
                                               # only true partner. Set high (e.g. 32) to
                                               # approximate the old global-only behavior.
    "split_image_rescue":     (0,   0,  5000), # IMAGE-GUIDED RESCUE: probe this many of
                                               # the FARTHEST dropped (truncated) pairs
                                               # with gap_bridge_evidence and re-add any
                                               # with a bright continuous bridge. 0 = off
                                               # (default; no cloud reads). Raises the
                                               # enumeration CEILING beyond geometry.
                                               # Each probe is ~5-25 cloud reads, so keep
                                               # small; needs a live image reader.
    "split_image_rescue_min_bridge": (0.7, 0.0, 1.0),  # bridge_ratio floor to rescue a
                                               # dropped pair (higher = stricter).
    "tip_to_shaft":     (True,  None, None),   # bool: partners may be shaft/branch
    "split_alt_per_pair": (1,    1,    10),    # gaps kept per label pair (>1 attaches
                                               # extra evidence gaps as SplitSite.alt_gaps)
    # candidate_merge_sites
    "min_arm_cable_um": (10.0,  2.0,  50.0),   # both arms must reach this (µm)
    "seed_depth_um":    (8.0,   2.0,  30.0),   # seed placement depth into each arm
    "merge_max_sites":  (5000,  100,  50000),  # global cap on merge candidates
    "max_per_label":    (4,     1,    100),    # cap on merge sites per raw label. Default
                                               # 4 (was 8): a policy that omits this now
                                               # gets 4 merge candidates/label — halving
                                               # the per-label budget vs the old default.
                                               # A policy MAY still set any value in [1,100].
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


def load_cached_graphs(cache_path: str, expect_brain=None, expect_mcl=None):
    """Load just the two SkeletonGraphs from a BrainDataset cache pickle.

    We read the pickle payload directly (rather than via BrainDataset.load_from_cache)
    so this module has no dependency on the TensorStore image reader — the
    evolution loop's candidate-site geometry only needs the graphs.

    Content validation (opt-in): the path follows the
    ``dataset_cache_<brain>_mcl<mcl>.pkl`` filename CONVENTION, but nothing
    guarantees the file's CONTENTS match that name — a wrong-but-existing pickle (a
    renamed/stale build, a swapped brain) would otherwise load silently and the run
    would train/gate on the wrong data. When ``expect_brain`` / ``expect_mcl`` are
    given we cross-check them against the payload's own metadata and FAIL FAST:
      * ``expect_mcl`` vs the payload's ``min_cable_length`` (stored at build time);
      * ``expect_brain`` vs the brain id embedded in the payload's
        ``fragments_path`` / ``gt_path`` / ``img_path`` (e.g. ``/789202/``).
    Either is skipped if the corresponding metadata is absent (older caches), so the
    check only ever tightens, never breaks, an otherwise-valid load.
    """
    with open(cache_path, "rb") as f:
        payload = pickle.load(f)

    # --- mcl content check: payload's own min_cable_length must match the request.
    if expect_mcl is not None and payload.get("min_cable_length") is not None:
        got_mcl = payload["min_cable_length"]
        if int(got_mcl) != int(expect_mcl):
            raise SystemExit(
                f"cache mcl mismatch: {cache_path} was built with "
                f"min_cable_length={got_mcl}, but the run requested mcl={expect_mcl}. "
                f"The filename convention (dataset_cache_<brain>_mcl<mcl>.pkl) and the "
                f"file CONTENTS disagree — refusing to enumerate candidates over the "
                f"wrong cache. Load the matching cache or rebuild it."
            )

    # --- brain content check: the brain id embedded in the source paths must match.
    if expect_brain is not None:
        eb = str(expect_brain)
        srcs = [payload.get(k) for k in ("fragments_path", "gt_path", "img_path")]
        srcs = [str(s) for s in srcs if s]
        # Only assert when at least one source path is present AND some path actually
        # encodes a brain id, so older/atypical caches don't trip a false alarm.
        if srcs and not any(f"/{eb}/" in s or f"_{eb}_" in s for s in srcs):
            raise SystemExit(
                f"cache brain mismatch: {cache_path} requested brain={eb}, but none "
                f"of the payload's source paths reference it "
                f"({'; '.join(srcs) or 'no source paths in payload'}). The filename "
                f"says brain {eb} but the contents look like a different brain — "
                f"refusing to train/gate on mismatched data."
            )

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
    alt_gaps : list[dict]
        OTHER nearby gaps between the SAME label pair (when ``alt_per_pair > 1`` was
        requested). The site itself carries the CLOSEST gap (``node_a``/``node_b``/
        ``gap_um``); ``alt_gaps`` holds up to ``alt_per_pair - 1`` additional gaps,
        each ``{"gap_um", "node_a", "node_b", "xyz_a", "xyz_b"}`` sorted by gap. Two
        fragments can be near each other in MORE than one place (e.g. parallel-running
        neurites), and the closest gap is not always the most decisive — its geometry
        may be a sideways graze while another gap is a clean colinear continuation. A
        policy can inspect ``alt_gaps`` to find the most convincing evidence point
        before deciding. It does NOT change the action: ``as_edit()`` still emits the
        single ``merge_labels(label_a, label_b)`` (one merge unifies the pair across
        ALL their gaps via union-find), so edit count / scoring are unchanged. Empty
        in the default (``alt_per_pair == 1``) behavior.
    recip_rank_a : int
        Reciprocal-neighbor rank of the PARTNER (``label_b``) among the anchor tip
        (``node_a``)'s differently-labelled partners, ordered by gap. ``1`` means the
        partner is this tip's CLOSEST other-label neighbor; larger = farther down the
        tip's preference list; ``0`` = not ranked (undefined). A low rank says the tip
        genuinely "points at" this partner rather than merely being near it.
    recip_rank_b : int
        The mirror: rank of the anchor (``label_a``) among the partner node
        (``node_b``)'s nearest differently-labelled TIPS, ordered by gap. Edges run
        tip→partner, so this asks whether the partner reciprocally picks this tip.
        ``1`` = the partner's closest tip; ``0`` = not ranked.
    mutual_nearest : bool
        True iff ``recip_rank_a == 1 and recip_rank_b == 1`` — the two endpoints are
        each other's #1 reconnection choice (a reciprocal / mutual-nearest-neighbor
        pair). This is a strong TRUE-split signal: two fragment ends that each select
        the other as their nearest partner are far more likely one broken neuron than
        a tip grazing an unrelated neuron that does not point back. A cheap,
        GT-free precision discriminator the policy can gate merges on.
    """

    kind: str = field(default="split", init=False)  # site-type tag for the policy

    label_a: str
    label_b: str
    gap_um: float
    node_a: int
    node_b: int
    xyz_a: tuple
    xyz_b: tuple
    alt_gaps: list = field(default_factory=list)
    # Reciprocal-neighbor features (see class docstring). Defaults keep every existing
    # SplitSite construction valid and mean "undefined / not mutual".
    recip_rank_a: int = 0
    recip_rank_b: int = 0
    mutual_nearest: bool = False
    # True iff this site was NOT reachable by geometry (dropped at the global cap /
    # beyond the geometric budget) and was RESCUED by image evidence — a bright
    # continuous bridge across the gap. False for ordinary geometric candidates. Lets a
    # policy treat image-rescued far-gap sites with appropriate caution.
    image_rescued: bool = False
    bridge_ratio: float = float("nan")  # gap_bridge_evidence.bridge_ratio when rescued

    def as_edit(self) -> tuple:
        """The label pair this site would unify if accepted.

        Independent of ``alt_gaps``: a single ``merge_labels`` unifies the pair
        across every gap between them, so multiple evidence gaps still yield ONE edit.
        """
        return (self.label_a, self.label_b)


def candidate_split_sites(
    fragments_graph,
    max_gap_um: float = 15.0,
    max_sites: int = 5000,
    tip_to_shaft: bool = True,
    alt_per_pair: int = 1,
    per_tip_k: int = 4,
    return_stats: bool = False,
    image_reader=None,
    image_rescue_max: int = 0,
    image_rescue_min_bridge: float = 0.7,
    image_rescue_shape=(16, 16, 16),
):
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
        GLOBAL cap on returned sites (closest gaps first), applied AFTER the per-tip
        quota, to bound the agent's input.
    per_tip_k : int
        PER-TIP quota: from each anchor tip, keep only its ``per_tip_k`` closest
        differently-labelled partners before the pair-level dedup and the global
        ``max_sites`` cap. This stops a dense region's tips from filling the entire
        global budget with their many near-partners and starving a sparse tip whose
        ONE true reconnection partner would otherwise be ranked out (the failure mode
        behind the observed ~0.3% reachable-real recall: the global top-5000 kept only
        gaps < ~1.87 µm, so a sparse tip's slightly-farther true partner never made the
        list). Clamped to >= 1; set very high (>= max tip degree) to recover the old
        global-only behavior. The reciprocal-rank / mutual-nearest features are computed
        over the SAME per-tip ordering.
    tip_to_shaft : bool
        If True, the partner node may be any node (tip/shaft/branch). If False,
        partners are restricted to other tips (legacy tip-to-tip enumeration).
    image_reader : LazyImagePatchReader or None
        Optional image reader for IMAGE-GUIDED candidate RESCUE. Geometry alone drops
        far-gap pairs at the ``max_sites`` cap (and never proposes beyond ``max_gap_um``),
        so a true split whose continuation is faint/long is unreachable by ANY policy.
        When a reader is given and ``image_rescue_max > 0``, the FARTHEST pairs the
        global cap dropped (the ``truncated`` tail) are probed with
        ``gap_bridge_evidence``; any whose ``bridge_ratio >= image_rescue_min_bridge``
        (a continuous bright bridge across the gap ⇒ likely one neuron) are RESCUED back
        into the returned sites, tagged ``image_rescued=True``. This lets image raise the
        enumeration CEILING, not just filter within it. None (default) ⇒ geometry only,
        no rescue, no cloud reads.
    image_rescue_max : int
        Cap on how many dropped pairs to image-probe (each is ~5-25 cloud reads, so this
        bounds cost). 0 (default) disables rescue even if a reader is passed. The tail is
        probed CLOSEST-dropped-first (most likely a real continuation).
    image_rescue_min_bridge : float
        ``bridge_ratio`` threshold to rescue a dropped pair (default 0.7). Higher =
        stricter (fewer, higher-confidence rescues).
    image_rescue_shape : tuple
        Image patch shape passed to ``gap_bridge_evidence``.
    alt_per_pair : int
        How many gaps to retain PER unordered label pair (default 1 = legacy: keep
        only the single closest gap). With ``alt_per_pair > 1``, the closest gap
        still defines the site (``node_a``/``node_b``/``gap_um``) and the next
        ``alt_per_pair - 1`` closest gaps between the SAME pair are attached as
        ``SplitSite.alt_gaps`` (additional evidence points the policy can inspect).
        The returned site COUNT is unchanged (still one per label pair) — this only
        enriches each site, so edit count and scoring are unaffected. Clamped to >= 1.

    return_stats : bool
        When True, return ``(sites, stats)`` instead of just ``sites``. ``stats`` is
        a dict describing the ENUMERATION CEILING — what the geometric scan dropped
        BEFORE the policy ever saw it, so the failure report can tell the reviser when
        its recall is bounded by enumeration (fixable only via ENUM_PARAMS) rather
        than by its own accept/reject thresholds. Keys:
          * ``n_pairs_enumerated`` — distinct unordered label pairs found within
            ``max_gap_um`` (== number of sites BEFORE the ``max_sites`` cap);
          * ``n_returned`` — sites actually returned (after the cap);
          * ``n_truncated`` — pairs DROPPED by the ``max_sites`` cap (0 if uncapped);
          * ``truncated_at_gap_um`` — the gap of the closest dropped pair (the cap
            keeps the closest gaps, so everything at/after this gap was discarded);
            ``None`` when nothing was truncated;
          * ``max_truncated_gap_um`` — the farthest dropped pair's gap (``None`` if
            none), so the reviser sees the gap RANGE it is losing to truncation;
          * ``max_gap_um`` / ``max_sites`` / ``tip_to_shaft`` — the params in effect.
        This is GT-FREE (pure geometry), so it is leak-free and safe to surface.

    Returns
    -------
    list[SplitSite] sorted by ascending gap, deduplicated per unordered
    label pair (closest gap kept). If ``return_stats`` is True, returns
    ``(list[SplitSite], stats_dict)`` instead (see ``return_stats`` above).
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

    alt_per_pair = max(1, int(alt_per_pair))
    per_tip_k = max(1, int(per_tip_k))

    # id(node) -> its index into the aligned arrays (node_coords / node_labels), so a
    # SplitSite's node_a / node_b (fragment-graph node ids) can be turned back into an
    # array index for the reciprocal-rank queries below.
    id_to_idx = {int(nid): k for k, nid in enumerate(node_arr)}

    # A node's ranking of its differently-labelled partner LABELS by closest gap,
    # computed over the SAME partner set the enumeration uses (partner_tree). Returns
    # ``{partner_label: rank}`` with rank 1 = the node's closest other-label partner.
    # Memoized per node index: each tip and each partner node is queried at most once,
    # and the whole enumeration is itself cached once per (graph, params) — so this is
    # paid once per run. This one function backs BOTH per-tip quota (via the anchor's
    # ranking) AND the reciprocal-neighbor features, so the two are always consistent.
    _rank_memo: dict[int, dict[str, int]] = {}

    def _partner_label_rank(node_index: int) -> dict[str, int]:
        cached = _rank_memo.get(node_index)
        if cached is not None:
            return cached
        own = node_labels[node_index]
        nbrs = partner_tree.query_ball_point(node_coords[node_index], r=max_gap_um)
        best_by_label: dict[str, float] = {}
        for pj in nbrs:
            gi = int(partner_idx[pj])
            if gi == node_index:
                continue
            lb = node_labels[gi]
            if lb == "0" or lb == own:
                continue
            gap = float(np.linalg.norm(node_coords[node_index] - node_coords[gi]))
            if lb not in best_by_label or gap < best_by_label[lb]:
                best_by_label[lb] = gap
        ranked = sorted(best_by_label, key=best_by_label.get)
        out = {lb: r + 1 for r, lb in enumerate(ranked)}
        _rank_memo[node_index] = out
        return out

    # Collect candidate gaps per unordered label pair. When alt_per_pair == 1 we keep
    # only the single closest gap (legacy behavior); otherwise we accumulate all gaps
    # for a pair and keep the closest ``alt_per_pair`` at the end. Each gap entry is
    # (gap, node_a, node_b, xyz_a, xyz_b); node_a is always the tip.
    #
    # PER-TIP QUOTA: from each tip we keep only its ``per_tip_k`` closest partner
    # LABELS (ranked by their closest gap). A pair survives if EITHER of its tips ranks
    # the other within top-k (union), so a sparse tip's single true partner is
    # preserved even when the partner's dense side would rank it out. This is what
    # stops a dense region from consuming the whole global ``max_sites`` budget and
    # starving sparse tips (the mechanism behind the ~0.3% reachable-real recall: the
    # global top-N kept only the very closest gaps brain-wide).
    gaps_by_pair: dict[frozenset, list] = {}
    for ti, neighbors in zip(tip_idx, neighbor_lists):
        la = node_labels[ti]
        if la == "0":
            continue
        ai = int(node_arr[ti])
        a_xyz = node_coords[ti]
        # This tip's top-k partner labels (rank 1..per_tip_k) — its own quota.
        tip_rank = _partner_label_rank(ti)
        kept_labels = {lb for lb, rk in tip_rank.items() if rk <= per_tip_k}
        if not kept_labels:
            continue
        for pj in neighbors:
            gi = int(partner_idx[pj])
            if gi == ti:
                continue  # the tip itself
            lb = node_labels[gi]
            if lb == "0" or lb == la:
                continue  # unlabelled or same fragment — not a split-repair candidate
            if lb not in kept_labels:
                continue  # beyond this tip's per-tip quota — drop before the global cap
            gap = float(np.linalg.norm(a_xyz - node_coords[gi]))
            entry = (
                gap,
                ai,                                 # node_a: always the tip
                int(node_arr[gi]),                  # node_b: tip/shaft/branch partner
                tuple(map(float, a_xyz)),
                tuple(map(float, node_coords[gi])),
            )
            key = frozenset((la, lb))
            bucket = gaps_by_pair.get(key)
            if bucket is None:
                gaps_by_pair[key] = [entry]
            elif alt_per_pair == 1:
                # Legacy: keep only the closest gap for this pair (no list growth).
                if gap < bucket[0][0]:
                    bucket[0] = entry
            else:
                bucket.append(entry)

    sites: list[SplitSite] = []
    for bucket in gaps_by_pair.values():
        # Closest gap defines the site; the next-closest become alt_gaps evidence.
        bucket.sort(key=lambda t: t[0])
        gap, ai, bi, axyz, bxyz = bucket[0]
        alt_gaps = [
            {"gap_um": g_, "node_a": a_, "node_b": b_, "xyz_a": xa_, "xyz_b": xb_}
            for (g_, a_, b_, xa_, xb_) in bucket[1:alt_per_pair]
        ]
        la_s = str(g.node_segment_id(ai))
        lb_s = str(g.node_segment_id(bi))
        # Reciprocal-neighbor features (GT-free precision signal). recip_rank_a is how
        # the anchor tip ranks the partner's label; recip_rank_b is how the partner
        # node ranks the anchor's label — both over the same partner set, via the
        # memoized ranker. mutual_nearest = each is the other's #1 choice.
        rank_a = _partner_label_rank(id_to_idx[ai]).get(lb_s, 0) if ai in id_to_idx else 0
        rank_b = _partner_label_rank(id_to_idx[bi]).get(la_s, 0) if bi in id_to_idx else 0
        sites.append(
            SplitSite(
                label_a=la_s,
                label_b=lb_s,
                gap_um=gap,
                node_a=ai,                              # always the tip
                node_b=bi,                              # tip / shaft / branch partner
                xyz_a=axyz,
                xyz_b=bxyz,
                alt_gaps=alt_gaps,
                recip_rank_a=int(rank_a),
                recip_rank_b=int(rank_b),
                mutual_nearest=bool(rank_a == 1 and rank_b == 1),
            )
        )
    sites.sort(key=lambda s: s.gap_um)

    # Enumeration-ceiling stats (GT-free): the GLOBAL cap keeps the CLOSEST gaps, so any
    # pair beyond index ``max_sites`` is silently dropped before the policy sees it.
    # Surface what was lost so the failure report can distinguish a recall miss the
    # policy CAN fix (a site it rejected) from one only ENUM_PARAMS can (a site that
    # was never enumerated / was truncated away). NOTE: the per-tip quota already ran,
    # so ``n_pairs_enumerated`` counts pairs that survived BOTH the max_gap radius AND
    # each tip's per_tip_k quota; a pair dropped by the quota is not counted here (it is
    # a per_tip_k limit, reported separately via that knob). Computed before the slice.
    n_pairs = len(sites)
    n_returned = min(n_pairs, max_sites)
    truncated = sites[max_sites:] if n_pairs > max_sites else []
    n_mutual = sum(1 for s in sites[:max_sites] if s.mutual_nearest)

    # IMAGE-GUIDED RESCUE (option B): the global cap just dropped ``truncated`` — the
    # farthest-gap pairs — which is exactly where an isolated fragment's faint/long true
    # continuation hides. If an image reader is provided, probe the closest-dropped of
    # those with gap_bridge_evidence and RESCUE any with a bright continuous bridge
    # (bridge_ratio >= threshold), appending them to the kept set tagged image_rescued.
    # This raises the enumeration CEILING (adds candidates geometry never would), unlike
    # the accept/reject policy which only filters within the geometric set. Cost is
    # bounded by ``image_rescue_max`` (each probe is a handful of cloud reads).
    kept = sites[:max_sites]
    rescued: list[SplitSite] = []
    n_rescue_probed = 0
    if (image_reader is not None and image_rescue_max > 0 and truncated):
        # Probe the CLOSEST dropped pairs first (most likely a real continuation).
        for s in truncated[:image_rescue_max]:
            n_rescue_probed += 1
            try:
                ev = image_reader.gap_bridge_evidence(
                    s.node_a, s.node_b, shape=image_rescue_shape)
                br = float(ev.get("bridge_ratio", float("nan")))
            except Exception:
                br = float("nan")
            if br == br and br >= image_rescue_min_bridge:   # not NaN and bright bridge
                s.image_rescued = True
                s.bridge_ratio = br
                rescued.append(s)
    if rescued:
        kept = kept + rescued
        kept.sort(key=lambda s: s.gap_um)

    stats = {
        "n_pairs_enumerated": n_pairs,
        "n_returned": len(kept),
        "n_truncated": len(truncated),
        "truncated_at_gap_um": float(truncated[0].gap_um) if truncated else None,
        "max_truncated_gap_um": float(truncated[-1].gap_um) if truncated else None,
        "max_gap_um": float(max_gap_um),
        "max_sites": int(max_sites),
        "per_tip_k": int(per_tip_k),
        "n_mutual_nearest": int(n_mutual),  # returned sites that are reciprocal #1 pairs
        "tip_to_shaft": bool(tip_to_shaft),
        # image-rescue accounting (0 / absent when no reader or rescue disabled)
        "image_rescue_probed": int(n_rescue_probed),
        "image_rescued": int(len(rescued)),
        "image_rescue_min_bridge": float(image_rescue_min_bridge) if image_reader else None,
    }
    sites = kept
    if return_stats:
        return sites, stats
    return sites


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
        # Seed coordinates are stored in the AGENTIC graph frame but the scorer
        # (EditHandler) snaps them against the METRICS fragment graphs, so every
        # seed xyz must cross into the metrics frame here — the one boundary where
        # agentic-frame data enters the scorer. See agentic_xyz_to_metrics_frame.
        to_metrics = agentic_xyz_to_metrics_frame

        # Preferred: an explicit arm->neurite grouping (shared suffix => same
        # neurite). EditHandler shares a suffix across Dijkstra sources, so this cuts
        # the label into one side per NEURITE rather than per arm.
        if self.seed_groups:
            seeds = [
                {"suffix": spec["suffix"], "xyz": to_metrics(spec["xyz"]),
                 "node": spec.get("node")}
                for spec in self.seed_groups
            ]
            return {
                "kind": "split_label",
                "label": self.label,
                "seeds": seeds,
                "seed_a_xyz": to_metrics(self.seed_a_xyz),
                "seed_b_xyz": to_metrics(self.seed_b_xyz),
            }
        seeds = [
            {"suffix": "a", "xyz": to_metrics(self.seed_a_xyz), "node": self.seed_a_node},
            {"suffix": "b", "xyz": to_metrics(self.seed_b_xyz), "node": self.seed_b_node},
        ]
        for k, spec in enumerate(self.extra_seeds):
            seeds.append({
                "suffix": spec.get("suffix") or chr(ord("c") + k),
                "xyz": to_metrics(spec["xyz"]),
                "node": spec.get("node"),
            })
        return {
            "kind": "split_label",
            "label": self.label,
            "seeds": seeds,
            # Legacy two-seed keys (still accepted by EditHandler).
            "seed_a_xyz": to_metrics(self.seed_a_xyz),
            "seed_b_xyz": to_metrics(self.seed_b_xyz),
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


# --- Parallel MergeSite enumeration (#1) --------------------------------------------
# The whole-brain scan spends most of its time in PER-NODE graph walks (branch arms,
# bridge necks) — embarrassingly parallel across nodes. We parallelize ONLY that per-node
# GENERATION, by contiguous node-block, and keep the FINALIZE (sort / global cross-label
# dedup / cap) SERIAL in the parent. This is what preserves byte-identity: concatenating
# per-block results in block order reproduces the exact serial node-iteration order, so
# the stable sorts, the global bridge dedup, and the sort+cap all see the identical input
# and produce the identical output. `_candidate_merge_sites_impl` (the serial reference)
# is NEVER touched; `n_workers <= 1` dispatches to it unchanged.
#
# BACKEND: fork + copy-on-write. The fragment graph is published to a module global BEFORE
# the pool forks, so workers inherit it read-only (no multi-GB pickle per task). CAVEAT 1
# (memory): reading networkx adjacency touches refcounts -> COW copies pages over time, so
# per-worker RSS grows; pick `n_workers` for the node's free RAM (user-controlled). CAVEAT
# 2 (deadlock): fork-after-threads with an open TensorStore/gRPC client wedges (see
# incremental_scoring._cap_graph_loading_workers). candidate_merge_sites reads ONLY the
# fragment graph (no image), so this is safe at enumeration time — but never call the
# parallel path after opening a TensorStore reader in the same process.
_WORKER_GRAPH = None   # set in the parent before forking; inherited COW by each worker
_WORKER_NODES = None   # the node list (order == serial iteration); COW-inherited, not pickled


def _branch_site_for_node(g, node, min_arm_cable_um, seed_depth_um, check_reconvergence):
    """The branch-detector body for ONE node — a MergeSite or None. Extracted so the
    parallel worker and (via the verifier) the serial reference compute identically.
    Mirrors the inline branch loop in `_candidate_merge_sites_impl` exactly."""
    radius = getattr(g, "node_radius", None)

    def tangent(arm):
        if len(arm) < 2:
            return None
        v = np.asarray(g.node_xyz[arm[-1]], dtype=float) - np.asarray(
            g.node_xyz[arm[0]], dtype=float)
        n = np.linalg.norm(v)
        return v / n if n > 0 else None

    def seed_in_arm(arm):
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

    deg = g.degree[node]
    if deg < 3:
        return None
    label = str(g.node_segment_id(node))
    if label == "0":
        return None
    arms = []
    for nbr in g.neighbors(node):
        arm, cable = _arm_from_branch(g, node, nbr, seed_depth_um * 3)
        arms.append((cable, arm))
    arms.sort(key=lambda t: t[0], reverse=True)
    if len(arms) < 2:
        return None
    (cable_a, arm_a), (cable_b, arm_b) = arms[0], arms[1]
    if cable_b < min_arm_cable_um:
        return None
    ta, tb = tangent(arm_a), tangent(arm_b)
    if ta is not None and tb is not None:
        cos = float(np.clip(np.dot(ta, tb), -1.0, 1.0))
        angle_deg = float(np.degrees(np.arccos(cos)))
    else:
        angle_deg = float("nan")
    ra, rb = arm_mean_radius(arm_a), arm_mean_radius(arm_b)
    radius_ratio = (max(ra, rb) / min(ra, rb)) if (ra and rb and min(ra, rb) > 0) else None
    sa, sb = seed_in_arm(arm_a), seed_in_arm(arm_b)
    if check_reconvergence:
        arms_reconverge = _arms_reconverge(
            g, node, arm_a[1], arm_b[1], max_explore_um=seed_depth_um * 8)
    else:
        arms_reconverge = None
    extra_seeds = []
    for k, (cable_k, arm_k) in enumerate(arms[2:]):
        if cable_k < min_arm_cable_um:
            break
        sk = seed_in_arm(arm_k)
        extra_seeds.append({"suffix": chr(ord("c") + k),
                            "xyz": tuple(map(float, g.node_xyz[sk])),
                            "node": int(sk), "cable_um": float(cable_k)})
    seed_groups = _pair_arms_into_neurites(
        g, [(cable, arm) for cable, arm in arms if cable >= min_arm_cable_um],
        tangent, seed_in_arm)
    return MergeSite(
        label=label, cut_node=int(node), cut_xyz=tuple(map(float, g.node_xyz[node])),
        seed_a_node=int(sa), seed_b_node=int(sb),
        seed_a_xyz=tuple(map(float, g.node_xyz[sa])),
        seed_b_xyz=tuple(map(float, g.node_xyz[sb])),
        branch_degree=int(deg), angle_deg=angle_deg, radius_ratio=radius_ratio,
        cable_a_um=float(cable_a), cable_b_um=float(cable_b), detector="branch",
        arms_reconverge=arms_reconverge, extra_seeds=extra_seeds, seed_groups=seed_groups)


def _bridge_candidate_for_node(g, node, min_arm_cable_um, seed_depth_um,
                               max_kink_angle_deg=120.0):
    """The bridge-detector's PER-NODE candidate generation (the expensive walk part), or
    None. Returns a picklable tuple identical to the one the serial `_bridge_merge_sites`
    appends to its `candidates` list — the serial sort/dedup/construction (which is global
    and stays in the parent) is unchanged and consumes these."""
    if g.degree[node] != 2:
        return None
    label = str(g.node_segment_id(node))
    if label == "0":
        return None
    a, b = list(g.neighbors(node))
    chain_a, cable_a = _walk_until(g, a, node, seed_depth_um * 3)
    chain_b, cable_b = _walk_until(g, b, node, seed_depth_um * 3)
    if cable_a < min_arm_cable_um or cable_b < min_arm_cable_um:
        return None
    va = np.asarray(g.node_xyz[chain_a[-1]], float) - np.asarray(g.node_xyz[node], float)
    vb = np.asarray(g.node_xyz[chain_b[-1]], float) - np.asarray(g.node_xyz[node], float)
    na, nb = np.linalg.norm(va), np.linalg.norm(vb)
    if na == 0 or nb == 0:
        return None
    cos = float(np.clip(np.dot(va, vb) / (na * nb), -1.0, 1.0))
    angle_deg = float(np.degrees(np.arccos(cos)))
    if angle_deg > max_kink_angle_deg:
        return None
    sharpness = max_kink_angle_deg - angle_deg
    return (sharpness, node, chain_a, chain_b, cable_a, cable_b, angle_deg, label)


def _merge_worker(args):
    """Runs in a forked worker: process node-block [lo, hi) of the module-global graph +
    node list, returning ORDERED (branch_sites, bridge_candidates) for that block. Only
    the small ``args`` tuple (bounds + scalars) is pickled to the worker; the ~21M-node
    graph AND node list are read from module globals (fork+COW inherited, NOT pickled) —
    that is what keeps dispatch cheap at whole-brain scale. Order within the block == node
    order, so parent concatenation reproduces the serial sequence exactly."""
    lo, hi, min_arm, seed_depth, check_recon = args
    g, node_list = _WORKER_GRAPH, _WORKER_NODES
    branch, bridge = [], []
    for node in node_list[lo:hi]:
        s = _branch_site_for_node(g, node, min_arm, seed_depth, check_recon)
        if s is not None:
            branch.append(s)
        c = _bridge_candidate_for_node(g, node, min_arm, seed_depth)
        if c is not None:
            bridge.append(c)
    return branch, bridge


def _candidate_merge_sites_parallel(g, min_arm_cable_um, seed_depth_um, max_sites,
                                    max_per_label, check_reconvergence, n_workers):
    """Byte-identical parallel twin of `_candidate_merge_sites_impl`: parallel per-node
    generation (branch + bridge candidates) by node-block, then the SAME serial finalize."""
    import concurrent.futures as _cf
    import multiprocessing as _mp
    global _WORKER_GRAPH, _WORKER_NODES

    node_list = list(g.nodes)                         # fixed order == serial iteration
    n = len(node_list)
    if n == 0:
        return []
    # Contiguous blocks preserve global node order under concatenation. Tasks carry ONLY
    # (lo, hi) + scalars — NOT node_list — so ex.map pickles a handful of ints per task
    # instead of the full ~21M-id list to every worker (the whole-brain dispatch cost fix).
    n_blocks = max(1, int(n_workers))
    step = (n + n_blocks - 1) // n_blocks
    tasks = [(lo, min(lo + step, n), min_arm_cable_um, seed_depth_um, check_reconvergence)
             for lo in range(0, n, step)]

    # Progress log so it's OBVIOUS the parallel path fired and with how many workers (the
    # scan is otherwise silent for tens of minutes). SLURM cgroup-pins the job, so also
    # report the CPU budget: if n_blocks > allocated CPUs the workers oversubscribe (no
    # speedup) — the line makes that mismatch visible. flush=True survives log capture.
    try:
        _avail = len(os.sched_getaffinity(0))         # CPUs this process may actually use
    except (AttributeError, OSError):
        _avail = os.cpu_count() or 0
    _t_par = time.monotonic()
    print(f"[candidate_merge_sites] PARALLEL: {n_blocks} worker(s) over {n:,} nodes "
          f"({step:,} nodes/block); this process is pinned to {_avail} CPU(s)"
          + ("" if n_blocks <= _avail else
             f" — WARNING: {n_blocks} workers > {_avail} CPUs, will OVERSUBSCRIBE (no speedup)")
          + ".", flush=True)

    # Publish graph + node list as globals BEFORE forking so workers inherit them read-only
    # via copy-on-write (no pickle). Restored in finally so we never leak the big graph.
    _WORKER_GRAPH, _WORKER_NODES = g, node_list
    try:
        ctx = _mp.get_context("fork")
        with _cf.ProcessPoolExecutor(max_workers=n_blocks, mp_context=ctx) as ex:
            results = list(ex.map(_merge_worker, tasks))   # ordered: map preserves task order
    finally:
        _WORKER_GRAPH = _WORKER_NODES = None
    print(f"[candidate_merge_sites] PARALLEL: {n_blocks} worker(s) finished branch+bridge "
          f"GENERATION in {time.monotonic() - _t_par:.1f}s. Now the SERIAL finalize "
          f"(bridge dedup + component detector) runs — this is NOT parallelized, so "
          f"expect more time below.", flush=True)

    # Reassemble in block order == serial node order.
    _n_branch = sum(len(b) for b, _ in results)
    _n_bridge_cand = sum(len(c) for _, c in results)
    per_label: dict[str, list] = {}
    for branch_sites, _bridge in results:             # branch first (serial did branch loop first)
        for site in branch_sites:
            per_label.setdefault(site.label, []).append(site)

    # Bridge: concatenate per-block candidates in order, then run the SERIAL finalize
    # (sort by sharpness + global cross-label dedup + seed/construct) UNCHANGED, so the
    # global dedup is byte-identical to _bridge_merge_sites.
    _t = time.monotonic()
    bridge_candidates = [c for _b, bridge in results for c in bridge]
    _n_bridge = 0
    for site in _bridge_finalize(g, bridge_candidates, seed_depth_um):
        per_label.setdefault(site.label, []).append(site)
        _n_bridge += 1
    print(f"[candidate_merge_sites] SERIAL bridge finalize: {_n_bridge} sites from "
          f"{_n_bridge_cand:,} candidates in {time.monotonic() - _t:.1f}s "
          f"(branch sites: {_n_branch}).", flush=True)

    # Component detector stays serial (order-safe parallelization is harder; profile first).
    _t = time.monotonic()
    _n_comp = 0
    for site in _component_merge_sites(g, min_arm_cable_um, seed_depth_um):
        per_label.setdefault(site.label, []).append(site)
        _n_comp += 1
    print(f"[candidate_merge_sites] SERIAL component detector: {_n_comp} sites in "
          f"{time.monotonic() - _t:.1f}s.", flush=True)

    sites: list = []
    for label, label_sites in per_label.items():
        label_sites.sort(key=lambda s: min(s.cable_a_um, s.cable_b_um), reverse=True)
        sites.extend(label_sites[:max_per_label])
    sites.sort(key=lambda s: min(s.cable_a_um, s.cable_b_um), reverse=True)
    _out = sites[:max_sites]
    print(f"[candidate_merge_sites] DONE: {len(_out)} sites returned "
          f"(after per-label + global cap).", flush=True)
    return _out


# --- Persistent cross-run MergeSite cache (B) ---------------------------------------
# The whole-brain candidate_merge_sites scan is ~tens of minutes; without this every run
# re-pays it. We memoize the RESULT LIST to a shared pickle per (graph content, merge
# params, schema). Mirrors incremental_scoring.get_or_build's discipline: a content
# fingerprint + a schema version guard the artifact so a stale/mismatched/corrupt file is
# recomputed rather than trusted. This caches OUTPUT ONLY (no recomputation on hit), so a
# hit is trivially byte-identical to a fresh scan.
MERGE_CACHE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "merge_site_cache"))

# Bump when MergeSite fields or the enumeration logic change in a way that makes an OLD
# cached list incompatible with current code (new/renamed fields, a changed detector).
MERGE_CACHE_SCHEMA = 1


def _graph_fingerprint(g) -> str:
    """Order-invariant content hash of a fragment graph — the CACHE IDENTITY.

    Two loads of the SAME fragment cache hash EQUAL; a different brain / mcl / a dropped
    fragment hashes DIFFERENT. Folds node count, edge count, and digests of the
    node_xyz coords + per-node segment ids (both sorted by node id so insertion order
    never matters). Cheap relative to the ~30-min scan it guards."""
    h = hashlib.md5()
    nodes = np.fromiter((int(n) for n in g.nodes), dtype=np.int64)
    nodes.sort()
    h.update(b"n"); h.update(nodes.tobytes())
    h.update(b"e"); h.update(np.int64(g.number_of_edges()).tobytes())
    xyz = np.asarray(g.node_xyz, dtype=np.float64)[nodes]   # rows in sorted-node order
    h.update(b"xyz"); h.update(np.ascontiguousarray(xyz).tobytes())
    # segment ids in the same order (the label identity the sites carry).
    segs = "|".join(str(g.node_segment_id(int(n))) for n in nodes)
    h.update(b"seg"); h.update(segs.encode("utf-8", "replace"))
    return h.hexdigest()


def _merge_cache_key(g, min_arm_cable_um, seed_depth_um, max_sites, max_per_label) -> dict:
    """Identity of one cached MergeSite result: graph fingerprint + exact params + schema."""
    return {
        "schema": MERGE_CACHE_SCHEMA,
        "fingerprint": _graph_fingerprint(g),
        "min_arm_cable_um": float(min_arm_cable_um),
        "seed_depth_um": float(seed_depth_um),
        "max_sites": int(max_sites),
        "max_per_label": int(max_per_label),
    }


def _merge_cache_path(key: dict) -> str:
    """Content-addressed filename for a key (param+fingerprint digest)."""
    digest = hashlib.md5(
        "|".join(f"{k}={key[k]}" for k in sorted(key)).encode()).hexdigest()
    return os.path.join(MERGE_CACHE_DIR, f"merge_{digest}.pkl")


def _load_merge_cache(key: dict):
    """Return the cached MergeSite list if a VALID artifact exists for ``key``, else None.

    Validates the stored key matches (schema + fingerprint + every param) before
    trusting the payload, so a hash collision or a hand-edited file can't feed wrong
    sites. Any read/unpickle error -> None (recompute), never a crash."""
    path = _merge_cache_path(key)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            payload = pickle.load(f)
    except (pickle.UnpicklingError, EOFError, OSError, AttributeError, ModuleNotFoundError):
        return None
    if not (isinstance(payload, dict) and payload.get("key") == key
            and isinstance(payload.get("sites"), list)):
        return None
    return payload["sites"]


def _save_merge_cache(key: dict, sites: list) -> None:
    """Persist ``sites`` for ``key`` atomically (temp + replace). Best-effort: a write
    failure (disk/permissions) is swallowed — the scan already ran, the cache is only an
    optimization."""
    try:
        os.makedirs(MERGE_CACHE_DIR, exist_ok=True)
        path = _merge_cache_path(key)
        tmp = f"{path}.tmp.{os.getpid()}"
        with open(tmp, "wb") as f:
            pickle.dump({"key": key, "sites": sites}, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)
    except OSError:
        pass


def candidate_merge_sites(
    fragments_graph,
    min_arm_cable_um: float = 10.0,
    seed_depth_um: float = 8.0,
    max_sites: int = 5000,
    max_per_label: int = 8,
    check_reconvergence: bool = False,
    n_workers: int = 1,
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

    PERSISTENT CACHE (B): the whole-brain scan is ~tens of minutes, and every run
    re-pays it from scratch even for identical params. So the RESULT is memoized to a
    shared cross-run pickle keyed by the graph's CONTENT fingerprint + the exact merge
    params (+ a schema version). A validated hit returns the stored list in ms; a miss
    (new graph, new params, stale schema, or corruption) recomputes via
    ``_candidate_merge_sites_impl`` — byte-for-byte the pre-cache result — and saves it.
    ``check_reconvergence=True`` is NOT cached (it changes the sites and is a rare
    diagnostic path). Disable entirely with env ``PE_MERGE_CACHE=0``.

    PARALLEL (#1): ``n_workers > 1`` runs the per-node generation across that many forked
    workers (``_candidate_merge_sites_parallel``) and returns a BYTE-IDENTICAL result to
    the serial ``n_workers=1`` path (parallelism is by node-block with the finalize kept
    serial; verified). ``n_workers`` is SPEED-ONLY, so it is deliberately NOT part of the
    cache key — a cached result from any worker count is reused. Default 1 = the untouched
    serial reference. Do not use >1 after a TensorStore reader is open in this process
    (fork-after-threads deadlock; enumeration runs before that, so it is safe there).
    """
    def _compute():
        if int(n_workers) > 1:
            return _candidate_merge_sites_parallel(
                fragments_graph, min_arm_cable_um, seed_depth_um, max_sites,
                max_per_label, check_reconvergence, int(n_workers))
        return _candidate_merge_sites_impl(
            fragments_graph, min_arm_cable_um=min_arm_cable_um, seed_depth_um=seed_depth_um,
            max_sites=max_sites, max_per_label=max_per_label,
            check_reconvergence=check_reconvergence)

    # check_reconvergence changes site content and is a rare off-path diagnostic; never
    # serve/persist it from the cache (which is keyed only by the standard params).
    if check_reconvergence or os.environ.get("PE_MERGE_CACHE") == "0":
        return _compute()

    key = _merge_cache_key(fragments_graph, min_arm_cable_um, seed_depth_um,
                           max_sites, max_per_label)
    cached = _load_merge_cache(key)
    if cached is not None:
        return cached
    sites = _compute()
    _save_merge_cache(key, sites)
    return sites


def _candidate_merge_sites_impl(
    fragments_graph,
    min_arm_cable_um: float = 10.0,
    seed_depth_um: float = 8.0,
    max_sites: int = 5000,
    max_per_label: int = 8,
    check_reconvergence: bool = False,
) -> list[MergeSite]:
    """The whole-brain MergeSite scan (unchanged body). ``candidate_merge_sites`` wraps
    this with the optional persistent cross-run cache; keeping the compute path in its
    own function guarantees a cache MISS reproduces the pre-cache result byte-for-byte."""
    g = fragments_graph
    if g.number_of_nodes() == 0:
        return []

    per_label: dict[str, list[MergeSite]] = {}
    # Branch detector: degree>=3 is where two arms can meet within one label. The
    # SINGLE SOURCE OF TRUTH for the per-node branch logic is _branch_site_for_node —
    # the parallel path (`_merge_worker`) calls the SAME helper, so serial and parallel
    # cannot drift (previously this loop was inlined + duplicated; see Concern A).
    for node in g.nodes:
        site = _branch_site_for_node(g, node, min_arm_cable_um, seed_depth_um,
                                     check_reconvergence)
        if site is not None:
            per_label.setdefault(site.label, []).append(site)

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
    # Per-node candidate generation (the expensive walks). Kept in its own helper
    # (_bridge_candidate_for_node) so the parallel path can generate these by node-block
    # and hand the SAME candidate tuples to the SAME finalize below — byte-identical.
    candidates = []  # (sharpness, node, side_a_chain, side_b_chain, cable_a, cable_b, angle, label)
    for node in g.nodes:
        c = _bridge_candidate_for_node(g, node, min_arm_cable_um, seed_depth_um,
                                       max_kink_angle_deg)
        if c is not None:
            candidates.append(c)
    yield from _bridge_finalize(g, candidates, seed_depth_um, min_separation_um)


def _bridge_finalize(g, candidates, seed_depth_um, min_separation_um: float = 20.0):
    """Serial finalize for bridge candidates: sort by sharpness, suppress near-duplicate
    necks within ``min_separation_um`` (keep sharpest, GLOBAL across labels), and build the
    MergeSite for each survivor. Shared verbatim by the serial and parallel paths, so the
    order-dependent global dedup + construction is identical. ``candidates`` MUST already
    be in serial node order (the parallel path concatenates blocks in order to match)."""
    radius = getattr(g, "node_radius", None)
    candidates = list(candidates)
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
