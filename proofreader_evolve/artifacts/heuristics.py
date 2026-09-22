"""
THE EVOLVED PROGRAM (executable policy).  <-- the evolution loop edits THIS file.

``propose_edits`` is the proofreader's decision policy: given a UNIFIED stream of
candidate sites enumerated from the fragment graph, decide which proofreading edits
to make. The harness feeds the returned edits to the scoring framework.

TWO-PHASE OPERATION (this seed is built for ``--two-phase`` runs): each iteration
first repairs MERGE errors, then repairs SPLIT errors on the post-edit surface.
``ctx["phase"]`` says which pass this call is:

  - ``"merge_repair"``  (pass 1): the stream holds the RAW brain's SplitSites AND
    MergeSites; only ``split_label`` edits are kept by the harness. Decide which
    fused labels to CUT.
  - ``"split_repair"``  (pass 2): the fused labels were cut into pseudo-labels
    (``L#a``/``L#b``) and SplitSites were RE-ENUMERATED over the post-split label
    surface (a tip near one side of a just-split segment now sees that side as its
    partner, not the whole tangle); only ``merge_labels`` edits are kept. Decide
    which fragment pairs to UNIFY.
  - ``"single"``: legacy one-pass mode; both edit kinds are honored.

DETECTOR PRIORS (the seed's core signal — see ``harness/feature_bank.py``; exact
model provenance in ``feature_tables/<brain>/meta.json``): two frozen
AutoDiscovery-fitted models, read directly from their detector deliverable folders
under ``autodiscovery-application/``, score candidates on 129 validated geometric
features, and the harness stamps the scores onto sites:

  - ``MergeSite.label_merge_score`` — how strongly the model believes this site's
    raw LABEL fuses two neurons (segment-level: it says WHICH label to cut, never
    WHERE; cut placement is the site's own job).
  - ``SplitSite.split_score`` — how strongly the model believes this label PAIR is
    one broken neuron (pair-level: the exact granularity of a merge_labels edit).

  Both are UNCALIBRATED ranking scores; absolute values do not transfer across
  brains, so this policy thresholds on brain-relative QUANTILES via
  ``ctx["feature_bank"].score_quantile(kind, q)``. ``nan`` means "candidate not in
  the precomputed table" (or no table at all): ALWAYS handle nan explicitly — this
  seed falls back to pure-geometry rules there. Full 90/39-column feature rows are
  available on demand: ``ctx["feature_bank"].split_features(label_a, label_b)`` /
  ``.merge_features(label)`` (feature semantics are documented in the two detector
  READMEs under ``autodiscovery-application/``).

The candidate stream has TWO kinds of site (dispatch on ``site.kind``):

  - SplitSite  (kind == "split"): two nearby fragments with DIFFERENT labels — a
    neuron the segmentation broke into pieces. Valid repair = ``merge_labels``.
    Fields: ``label_a``, ``label_b``, ``gap_um``, ``node_a``/``node_b``,
    ``xyz_a``/``xyz_b``, ``alt_gaps`` (other gaps of the SAME pair),
    ``recip_rank_a``/``recip_rank_b``/``mutual_nearest`` (reciprocal-neighbor
    test: True iff each endpoint is the other's #1 reconnection choice — a strong
    gap-independent precision cue), ``image_rescued``/``bridge_ratio``, and
    ``split_score`` (frozen detector prior, above).

  - MergeSite  (kind == "merge"): ONE label fused across two neurites — repair =
    ``split_label``. Fields: ``label``, ``cut_node``/``cut_xyz``,
    ``seed_a_node``/``seed_b_node``, ``seed_a_xyz``/``seed_b_xyz``,
    ``branch_degree``, ``angle_deg`` (~180 = one neuron passing straight; sharper
    = more merge-like; NaN for the "component" detector), ``radius_ratio``,
    ``cable_a_um``/``cable_b_um``, ``detector`` ("branch"|"bridge"|"component" —
    condition on it, each has different evidence), ``arms_reconverge`` (True =
    the arms RE-JOIN downstream: one neuron's own branches/a loop — a STRONG
    signal NOT to cut), ``extra_seeds``/``seed_groups`` (multi-arm cuts), and
    ``label_merge_score`` (frozen detector prior, above).

IMPORTANT: the stream mixes both kinds. NEVER assume a site is a SplitSite — a
bare ``s.label_a`` will AttributeError on a MergeSite. Always branch on
``getattr(s, "kind", "split")`` first.

Contract (keep the CALL signature stable so the harness can always call it):
    propose_edits(sites, ctx) -> list[edit]

  sites : list[SplitSite | MergeSite]
  ctx   : dict — keys this policy may use:
            ctx["phase"]           "merge_repair" | "split_repair" | "single"
            ctx["feature_bank"]    FeatureBank or None (scores/features/quantiles)
            ctx["fragments_graph"], ctx["node_radius"], ctx["max_gap_um"],
            ctx["enum_params"], ctx["n_split_sites"], ctx["n_merge_sites"],
            ctx["split_geom"](site) -> dict of cheap GT-free geometry,
            ctx["nodes_within"](xyz, r), ctx["foreign_labels_near"](xyz, r, excl),
            ctx["read_image_patch"] (may be None), ctx["image_patch_shape"]
  return: list of edits — legacy 2-tuples ``(label_a, label_b)`` (a merge) or
          typed dicts:
            {"kind": "merge_labels", "label_a": str, "label_b": str}
            {"kind": "split_label",  "label": str,
             "seed_a_xyz": (x,y,z), "seed_b_xyz": (x,y,z)}
          ``site.as_edit()`` is the safe way to emit either kind.

SEED RATIONALE. Pass 1 cuts only labels the merge detector ranks at the very top
of this brain's candidate distribution AND whose cut site passes every geometric
veto (never cut re-converging arms; both arms substantial; branch angle not
neuron-straight), capped at a small per-brain budget — a false cut costs
``merge_penalty`` correct repairs, so precision dominates. Pass 2 unifies a pair
when the split detector ranks it near the top AND a geometric guard agrees
(mutual-nearest or colinear continuation); pairs the table does not know
(score=nan — including brand-new pseudo-label pairs beyond the base-pair lookup)
fall back to the proven conservative gap+colinearity rule. The agent's job is to
improve this: retune the quantiles/budgets, restructure the decision tree, combine
the score with cheap geometry or the image reader (``gap_bridge_evidence`` — an
evidence source the frozen models have never seen), pull full feature rows from
the bank, or widen enumeration via ENUM_PARAMS. Levers are ranked in rules.md.
"""

from __future__ import annotations

import math

import numpy as np


# --- Tunable parameters (the agent may rewrite these and the logic below) -----
# PASS 1 (merge repair). Quantiles are over THIS brain's candidate-label score
# distribution (brain-relative; absolute scores do not transfer across brains).
MERGE_CUT_QUANTILE = 0.99      # label_merge_score must clear this quantile
MERGE_TOPK_CUTS = 10           # per-brain cut budget (labels), best-scored first
MERGE_MIN_ARM_CABLE_UM = 20.0  # both arms of the cut must carry this much cable
MERGE_MAX_ANGLE_DEG = 140.0    # veto near-straight (~180 deg) pass-throughs

# PASS 2 (split repair). Score gate + geometric guard.
SPLIT_SCORE_QUANTILE = 0.995   # split_score must clear this quantile AND a guard
# Fallback rule for score-less candidates (no table / pair unknown): the proven
# conservative geometry-only seed (small gap + colinear continuation).
GAP_THRESHOLD_UM = 4.0
MIN_COLINEAR_COS = 0.94
TANGENT_WALK_UM = 6.0

# --- Evolvable enumeration priors (optional) --------------------------------
# ENUM_PARAMS controls WHAT THE POLICY EVEN SEES — the candidate stream the harness
# enumerates — as opposed to the thresholds above, which decide what to ACCEPT among
# what it sees. The harness validates + CLAMPS every value to a safety rail (see
# dataset.ENUM_PARAM_SPEC) and ignores unknown keys. Any key omitted falls back to
# the framework default. NOTE: the frozen detector score tables were precomputed
# over a FIXED enumeration (see feature_tables/<brain>/meta.json) — widening
# enumeration beyond it surfaces candidates whose split_score is nan (they fall to
# the geometric fallback), it never crashes.
ENUM_PARAMS = {
    "max_gap_um": 15.0,        # tip->partner search radius for split candidates (µm) [1..40]
    "tip_to_shaft": True,      # split partners may be shaft/branch nodes, not just tips
    "min_arm_cable_um": 10.0,  # both arms of a merge candidate must reach this (µm) [2..50]
    "seed_depth_um": 8.0,      # how deep to place each split seed into its arm (µm) [2..30]
    "max_per_label": 8,        # cap on merge candidates emitted per raw label [1..100]
    # "split_max_sites": 5000,           # global cap on split candidates [100..50000]
    # "split_per_tip_k": 4,              # per-tip partner quota [1..32]
    # "split_image_rescue": 0,           # image-guided far-gap rescue [0..5000]
    # "split_image_rescue_min_bridge": 0.7,
    # "merge_max_sites": 5000,           # global cap on merge candidates [100..50000]
    # "split_alt_per_pair": 1,           # gaps kept per SplitSite label pair [1..10]
}


def _walk_tangent(g, start, max_um):
    """Unit direction of the fragment at ``start``, found by walking inward.

    Follows ``start``'s own degree-<=2 chain (staying on the same fragment) up to
    ``max_um`` of cable, then returns the unit vector FROM the far interior point
    TO ``start`` — i.e. the direction the cable is travelling as it arrives at
    ``start`` (pointing "outward", toward the gap for a tip). Returns None if the
    fragment is too short or degenerate to define a direction.
    """
    xyz = g.node_xyz
    seg = g.node_segment_id(start)
    prev, cur = start, None
    nbrs = [n for n in g.neighbors(start) if g.node_segment_id(n) == seg]
    if not nbrs:
        return None
    cur = nbrs[0]
    walked = float(np.linalg.norm(xyz[cur] - xyz[start]))
    while walked < max_um:
        nxt = [n for n in g.neighbors(cur)
               if n != prev and g.node_segment_id(n) == seg]
        if len(nxt) != 1:        # tip or branch — stop here
            break
        prev, cur = cur, nxt[0]
        walked += float(np.linalg.norm(xyz[cur] - xyz[prev]))
    v = np.asarray(xyz[start], dtype=float) - np.asarray(xyz[cur], dtype=float)
    n = np.linalg.norm(v)
    return (v / n) if n > 0 else None


def _is_colinear_split(g, s, min_cos):
    """True if SplitSite ``s`` is a straight-line continuation across the gap.

    Builds three downstream-pointing unit vectors along the imagined repaired
    neuron  A_interior -> node_a -> node_b -> B_interior  and requires consecutive
    pairs to be near-parallel. This rejects the common false positive where two
    unrelated neurites merely pass close by: their tips point ACROSS the gap at an
    angle, not ALONG it.
    """
    xyz = g.node_xyz
    a, b = s.node_a, s.node_b
    v2 = np.asarray(xyz[b], dtype=float) - np.asarray(xyz[a], dtype=float)
    n2 = np.linalg.norm(v2)
    if n2 == 0:
        return False
    v2 /= n2

    v1 = _walk_tangent(g, a, TANGENT_WALK_UM)          # arrives at node_a, outward
    if v1 is None or float(np.dot(v1, v2)) < min_cos:
        return False

    seg_b = g.node_segment_id(b)
    best = None
    for nbr in g.neighbors(b):
        if g.node_segment_id(nbr) != seg_b:
            continue
        d = np.asarray(xyz[nbr], dtype=float) - np.asarray(xyz[b], dtype=float)
        dn = np.linalg.norm(d)
        if dn == 0:
            continue
        d /= dn
        c = float(np.dot(v2, d))
        if best is None or c > best:
            best = c
    return best is not None and best >= min_cos


def _propose_merge_repairs(sites, ctx) -> list:
    """PASS 1: cut top-scored fused labels at their best-vetoed cut site.

    Precision-first: a false cut costs ``merge_penalty`` correct repairs, so only
    act when the frozen merge detector puts the label at the very top of THIS
    brain's candidate distribution AND the site's own geometry does not veto.
    """
    bank = ctx.get("feature_bank")
    if bank is None:
        return []  # no prior => no cuts; evolution may add a geometry-only path
    tau = bank.score_quantile("merge", MERGE_CUT_QUANTILE)
    if math.isnan(tau):
        return []

    # Best cut site per label, geometric vetoes first.
    best_site_by_label: dict = {}
    for s in sites:
        if getattr(s, "kind", "split") != "merge":
            continue
        score = getattr(s, "label_merge_score", float("nan"))
        if math.isnan(score) or score < tau:
            continue
        if s.arms_reconverge is True:      # one neuron's own branches / a loop
            continue
        if min(s.cable_a_um, s.cable_b_um) < MERGE_MIN_ARM_CABLE_UM:
            continue
        angle = s.angle_deg
        if angle is not None and not (isinstance(angle, float) and math.isnan(angle)):
            if float(angle) > MERGE_MAX_ANGLE_DEG:   # near-straight pass-through
                continue
        cur = best_site_by_label.get(s.label)
        # Prefer the sharpest (most merge-like) adjudicable angle; angle-less
        # detectors (bridge/component) rank after angled branch sites.
        key = float(angle) if (angle is not None and not math.isnan(float(angle))) \
            else float("inf")
        if cur is None or key < cur[0]:
            best_site_by_label[s.label] = (key, score, s)

    ranked = sorted(best_site_by_label.values(), key=lambda t: -t[1])
    edits = []
    for _key, _score, site in ranked[:MERGE_TOPK_CUTS]:
        try:
            edits.append(site.as_edit())   # split_label dict (multi-seed aware)
        except Exception:
            continue
    return edits


def _propose_split_repairs(sites, ctx) -> list:
    """PASS 2 (post-split surface): unify top-scored pairs, guard with geometry.

    Score path: split_score above this brain's SPLIT_SCORE_QUANTILE AND a
    geometric guard (mutual-nearest or colinear continuation). Fallback path for
    score-less candidates (nan — no table, or a pair outside it): the proven
    conservative gap+colinearity rule.
    """
    g = ctx.get("fragments_graph")
    if g is None:
        return []
    bank = ctx.get("feature_bank")
    tau = bank.score_quantile("split", SPLIT_SCORE_QUANTILE) if bank is not None \
        else float("nan")

    edits = []
    for s in sites:
        if getattr(s, "kind", "split") != "split":
            continue
        try:
            score = getattr(s, "split_score", float("nan"))
            if not math.isnan(score) and not math.isnan(tau) and score >= tau:
                # Model is confident; still require ONE cheap geometric agreement
                # (the zero-false gate punishes a lone over-eager prior).
                if s.mutual_nearest or _is_colinear_split(g, s, MIN_COLINEAR_COS):
                    edits.append(s.as_edit())
                continue
            # Fallback: candidates the frozen prior does not know.
            if s.gap_um <= GAP_THRESHOLD_UM and \
                    _is_colinear_split(g, s, MIN_COLINEAR_COS):
                edits.append(s.as_edit())
        except Exception:
            # Geometry is advisory; never let one odd site crash the whole policy.
            continue
    return edits


def propose_edits(sites, ctx) -> list:
    """Two-phase detector-prior policy; see the module docstring for the theory.

    Dispatches on ``ctx["phase"]``: pass 1 proposes ``split_label`` cuts on
    top-scored fused labels, pass 2 proposes ``merge_labels`` unifications on
    top-scored pairs (with geometric guards and a geometry-only fallback). In
    legacy single-pass mode both are proposed from the one mixed stream.

    Returns a list of edits; an empty list means "make no changes".
    """
    phase = ctx.get("phase", "single")
    if phase == "merge_repair":
        return _propose_merge_repairs(sites, ctx)
    if phase == "split_repair":
        return _propose_split_repairs(sites, ctx)
    return _propose_merge_repairs(sites, ctx) + _propose_split_repairs(sites, ctx)
