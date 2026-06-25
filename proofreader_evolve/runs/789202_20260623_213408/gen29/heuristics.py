"""
THE EVOLVED PROGRAM (executable policy).  <-- the evolution loop edits THIS file.

``propose_edits`` is the proofreader's decision policy: given a UNIFIED stream of
candidate sites enumerated from the fragment graph, decide which proofreading edits
to make. The harness feeds the returned edits to the scoring framework.

The candidate stream has TWO kinds of site (dispatch on ``site.kind``):

  - SplitSite  (kind == "split"): two nearby fragments with DIFFERENT labels — a
    neuron that the segmentation broke into pieces. Valid repair = ``merge_labels``
    (unify the two labels). Fields: ``label_a``, ``label_b``, ``gap_um``,
    ``node_a``/``node_b`` (fragment-graph node ids), ``xyz_a``/``xyz_b``.
    From ``dataset.candidate_split_sites``.

  - MergeSite  (kind == "merge"): ONE label fused across two neurites —
    two neurons the segmentation glued together. Valid repair = ``split_label``
    (partition the label by location). Fields: ``label``, ``cut_node``/``cut_xyz``,
    ``seed_a_node``/``seed_b_node``, ``seed_a_xyz``/``seed_b_xyz``, and advisory
    features ``branch_degree``, ``angle_deg`` (~180° = one neuron passing straight;
    sharper = more merge-like; NaN for the "component" detector), ``radius_ratio``
    (>>1 = two different calibers fused), ``cable_a_um``/``cable_b_um``,
    ``detector`` ("branch" | "bridge" | "component" — the topology that found it;
    condition on it since each has different evidence), ``arms_reconverge``
    (bool | None — True means the two arms RE-JOIN downstream, i.e. one neuron's
    own branches / a loop: a STRONG signal NOT to cut; None when not computed), and
    ``extra_seeds`` (list — third+ arm seeds for a degree>=4 crossing; non-empty
    means ``as_edit()`` will cut the label into >2 sides). From
    ``dataset.candidate_merge_sites``.

IMPORTANT: the stream mixes both kinds. NEVER assume a site is a SplitSite — a bare
``s.label_a`` will AttributeError on a MergeSite. Always branch on
``getattr(s, "kind", "split")`` first (see the dispatch skeleton below).

The evolution loop works by:
  1. running this policy on the TRAIN skeletons,
  2. scoring the result with the real metric framework,
  3. showing the agent where the policy was wrong (splits left unrepaired, merges
     left uncorrected, or edits that hurt — see the failure report),
  4. asking the agent to rewrite the body of ``propose_edits`` (and the companion
     ``rules.md``) to do better,
  5. keeping the rewrite ONLY if held-out Edge Accuracy improves.

Contract (keep the CALL signature stable so the harness can always call it):
    propose_edits(sites, ctx) -> list[edit]

  sites : list[SplitSite | MergeSite]   # the unified candidate stream
  ctx   : dict   # free-form context the harness provides, e.g.
                 #   ctx["max_gap_um"], ctx["fragments_graph"],
                 #   ctx["node_radius"], ctx["read_image_patch"] (may be None),
                 #   ctx["n_split_sites"], ctx["n_merge_sites"]
  return: list of edits. Each edit is EITHER a legacy 2-tuple
          ``(label_a, label_b)`` (treated as a merge) OR a typed dict:
            {"kind": "merge_labels", "label_a": str, "label_b": str}      # repair split
            {"kind": "split_label",  "label": str,                        # repair merge
             "seed_a_xyz": (x,y,z), "seed_b_xyz": (x,y,z)}
            {"kind": "flag_review", "reason": str} / {"kind": "reject_candidate"}
          The harness normalizes tuples and dicts uniformly (see
          harness.edit_handler.normalize_edits), so they may be mixed.
          A MergeSite's ``.as_edit()`` already returns the correct ``split_label``
          dict, and a SplitSite's ``.as_edit()`` returns its ``(label_a, label_b)``
          tuple — so ``site.as_edit()`` is the safe way to emit either.

This seed version is a deliberate, conservative SPLIT-REPAIR policy: it merges two
fragments only when their tips are BOTH very close AND colinear across the gap (the
two cables continue in a straight line, the way one real neuron broken into pieces
would). That proposes a small number of high-precision edits — enough to move the
metric off the flat no-edit baseline and give the evolution loop a real gradient to
climb — while staying conservative enough not to manufacture merges. The agent's
job is to widen/sharpen this (and to add merge-repair via ``split_label``).
"""

from __future__ import annotations

import numpy as np


# --- Tunable parameters (the agent may rewrite these and the logic below) -----
# A "merge everything within N µm" seed scores ~10 points BELOW baseline (it chains
# thousands of pairs into brain-spanning mega-labels via union-find, manufacturing
# merges and inflating per-neuron label counts). The fix is PRECISION: only unify a
# pair when the geometry says it is almost certainly one neuron — a small gap AND a
# straight-line (colinear) continuation across that gap. These defaults are
# deliberately tight; the agent may loosen them once it can measure the trade-off.
GAP_THRESHOLD_UM = 4.0      # max tip→partner distance (µm) to even consider a merge
MIN_COLINEAR_COS = 0.94     # require ~>20° alignment: cos(angle) >= this (1.0 = perfectly straight)
TANGENT_WALK_UM = 6.0       # how far to walk into each fragment to estimate its tip tangent

# --- Image-gated recall path (Gen 1) ----------------------------------------
# The strict 0.94 colinear gate is too conservative: on train it accepted only
# 3/490 reachable REAL splits. A SECOND acceptance route admits moderately-colinear
# short gaps but ONLY when the raw image confirms a continuous bright bridge across
# the gap (gap_bridge_evidence.bridge_ratio). The geometric floor (0.80) is chosen
# to still EXCLUDE the lone cos=0.55 FALSE join BEFORE any (costly) image read.
IMG_MIN_COLINEAR_COS = 0.80  # moderate colinear floor for the image-gated path (~37° bend)
BRIDGE_RATIO_MIN     = 0.85  # require a continuous bright bridge (REAL probed >= 0.95)
IMG_FAR_GAP_UM = 3.0        # Path 9 (Gen 20): upper gap bound for image-confirmed far-gap recall; > SHAFT_RECALL_FAR_UM (2.5) yet excludes the lone 3.12 um FALSE join.
IMG_FAR_BRIDGE_MIN = 0.95   # Path 9 (Gen 20): bridge_ratio floor; the probe's 12 REAL splits in the 2.79-2.83 um band all read >= 0.95.

# --- Small-gap distance-dominant path (Gen 3) -------------------------------
# At sub-micron / ~1 µm gaps the gap-direction vector (node_b - node_a) is a tiny
# vector dominated by voxel noise once normalized, so colinear_cos is essentially
# random there (REAL splits show cos from -0.73 to +0.78 in this band). The GAP
# DISTANCE ITSELF is then near-decisive evidence of a true split (validated prior:
# true-split gaps cluster <~5 µm, inter-neuron gaps essentially never < ~7 µm), so
# below this gap we drop the colinearity requirement and lean on an image bridge
# confirmation alone.
SMALL_GAP_UM = 1.5  # below this gap, distance alone is near-decisive; colinearity is noise here

# --- Distance-only tip-to-tip path with dense-neuropil veto (Gen 8) ----------
# KB Finding #1: Euclidean gap distance ALONE separates true splits from inter-
# neuron neighbors at AUC 0.998 (true-split gaps peak ~4.5 µm; inter-neuron gaps
# rarely fall below ~7 µm; F1-optimal ~6.84 µm; GENERALIZES across all 3 brains).
# At gaps this small, distance alone is near-decisive, so accept TIP-TO-TIP here
# WITHOUT colinearity or image — exactly the noisy-cos reals the colinear gate
# wrongly drops. KB Finding #13: merge errors concentrate in dense fragment-graph
# tangles (paired AUC 0.9236, GENERALIZES all 3 brains, UPHELD), so VETO this
# accept inside a crowded local neighborhood (the OPPOSITE of gen7's clustering
# REQUIREMENT — density only blocks dense tangles, never the isolated reals).
DIST_ONLY_UM = 2.0        # Finding #1: at gaps this small, distance alone is near-decisive (inter-neuron gaps rarely < 7 µm; true-split peak ~4.5 µm). Accept tip-to-tip here without colinearity/image.
CLUSTER_RADIUS_UM = 30.0  # neighborhood radius for the dense-neuropil veto.
DENSE_VETO_MAX = 3        # Finding #13: VETO the distance-only accept if MORE than this many other candidate split sites lie within the radius (a crowded tangle = false-join risk). Real splits average ~1 neighbor, so this rarely blocks them.

# --- Tip-to-shaft small-gap recall via shaft-centerline offset (Gen 15) -------
# The largest untapped MISSED-real bucket is TIP-TO-SHAFT (deg_b==2) at 1.5–2.0 µm
# whose colinear_cos is low ONLY because it is built on the voxel-noise gap vector.
# Path 4 excludes shaft partners; image recall here was reverted (gen10) because a
# parallel adjacent neurite of a DIFFERENT neuron, lying beside a shaft, reads a bright
# bridge. The discriminator image CANNOT see — but geometry can — is the PERPENDICULAR
# OFFSET of tip A from the shaft's CENTERLINE through B: a true broken neuron's tip lies
# ON the shaft's extended centerline (small offset), while a parallel neurite is offset
# laterally by the inter-neurite spacing (offset ≈ gap). The shaft (a degree-2 node)
# defines a RELIABLE line from its two same-segment neighbors, unlike gen12's single
# noisy tip-tangent ray. We also require the tip tangent to ALIGN with the shaft axis
# (Finding #16: continuation ~parallel, rejects perpendicular T-junction false joins).
# Gaps stay ≤ DIST_ONLY_UM, far below the ~7 µm inter-neuron floor (Finding #1); the
# dense-neuropil veto (Finding #13) still applies. Image-free.
SHAFT_PERP_UM = 0.8       # max perpendicular offset (µm) of tip A from the shaft centerline through B
SHAFT_ALIGN_COS = 0.7     # require |cos(tipA_tangent, shaft_axis)| >= this (continuation, not a perpendicular T-junction)
SHAFT_RECALL_FAR_UM = 2.5   # Path 7 upper gap ceiling (µm): the shaft-centerline-offset geometry is gap-independent, so extend the validated tip-to-shaft recall to the 2.0-2.5 µm band; still far below the ~7 µm inter-neuron floor (Finding #1) and below the lone FALSE join at 3.12 µm.
COLINEAR_TS_COS = 0.72   # Path 8: moderate colinear floor (both half-cosines via _is_colinear_split) for the IMAGE-FREE tip-to-shaft path. Set ABOVE the lone FALSE join's colinear_cos 0.55 (clear margin; Finding #16 puts false joins near cos 0 / 90 deg) and BELOW Path 2's image floor IMG_MIN_COLINEAR_COS=0.80, so it admits straight tip-to-shaft continuations that need NO image bridge.
SHAFT_CLOSING_COS = 0.5   # Path 15: minimum magnitude of the perpendicular-closing cosine. The tip's tangent component perpendicular to the shaft axis must point OPPOSITE the tip's current lateral offset (cos <= -SHAFT_CLOSING_COS), i.e. the branch tip is angling INTO the parent shaft centerline. 0.5 ~ the tangent's cross-axis travel is at least 60 deg toward the shaft line -- strong enough to exclude near-perpendicular-drift while still admitting angled (non-colinear) branch junctions.

# --- Evolvable enumeration priors (optional) --------------------------------
# ENUM_PARAMS controls WHAT THE POLICY EVEN SEES — the candidate stream the harness
# enumerates — as opposed to the thresholds above, which decide what to ACCEPT among
# what it sees. These were once hardcoded in the harness (a HARD prior the policy
# could not move); defining them here makes them SOFT and evolvable. The harness
# validates + CLAMPS every value to a safety rail (see dataset.ENUM_PARAM_SPEC) and
# ignores unknown keys, so editing this can never crash the run. Any key omitted
# falls back to the framework default — so the dict below is the identity (no change
# from the historical behavior); the agent may widen/narrow it to surface different
# candidates (e.g. raise max_gap_um to reach longer true gaps, lower
# min_arm_cable_um to surface shorter merges, set tip_to_shaft=False for tip-to-tip
# only). The resolved values in effect are visible at runtime in ctx["enum_params"].
ENUM_PARAMS = {
    "max_gap_um": 15.0,        # tip->partner search radius for split candidates (µm) [1..40]
    "tip_to_shaft": True,      # split partners may be shaft/branch nodes, not just tips
    "min_arm_cable_um": 10.0,  # both arms of a merge candidate must reach this (µm) [2..50]
    "seed_depth_um": 8.0,      # how deep to place each split seed into its arm (µm) [2..30]
    "max_per_label": 8,        # cap on merge candidates emitted per raw label [1..100]
    # "split_max_sites": 5000, # global cap on split candidates [100..50000]
    # "merge_max_sites": 5000, # global cap on merge candidates [100..50000]
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
    # first hop: any neighbor on the same fragment
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
    pairs to be near-parallel:

        v1 = direction arriving at the tip node_a   (fragment A, outward)
        v2 = the gap bridge  node_a -> node_b
        v3 = direction leaving node_b into fragment B (continuing the line)

    Colinear  <=>  cos(v1, v2) >= min_cos AND cos(v2, v3) >= min_cos. This rejects
    the common false positive where two unrelated neurites merely pass close by:
    their tips point ACROSS the gap at an angle, not ALONG it.
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

    # v3: leave node_b into B along whichever same-fragment branch best continues
    # v2 (node_b may be a branch/shaft, so it has several candidate directions).
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


def _tip_heads_into_gap(g, s, min_cos):
    """True if tip ``node_a``'s outward tangent points along the gap toward node_b.

    Computes v1 = tip A's outward tangent (via ``_walk_tangent``) and v2 = the unit
    gap vector node_a -> node_b, and requires ``cos(v1, v2) >= min_cos``. This is the
    FIRST half-cosine of ``_is_colinear_split`` WITHOUT the second (gap-vs-node_b-
    continuation) half: it asks only whether the tip is HEADING INTO the gap toward its
    partner -- the signature of a branch tip reconnecting to its parent shaft (a T/Y
    junction, where the second half-cosine is legitimately LOW because the parent runs
    ACROSS the gap, not along it). It REJECTS a laterally-offset parallel neurite of a
    different neuron, whose tip tangent is ~perpendicular to the shortest gap vector
    (cos ~ 0). Returns False if the tangent is undefined or the gap is zero-length.
    """
    xyz = g.node_xyz
    v2 = np.asarray(xyz[s.node_b], dtype=float) - np.asarray(xyz[s.node_a], dtype=float)
    n2 = np.linalg.norm(v2)
    if n2 == 0:
        return False
    v2 /= n2
    v1 = _walk_tangent(g, s.node_a, TANGENT_WALK_UM)
    return v1 is not None and float(np.dot(v1, v2)) >= min_cos


def _shaft_continuation_ok(g, tip_node, shaft_node, max_perp_um, min_align_cos):
    """True if ``tip_node``'s tip continues along ``shaft_node``'s local shaft axis.

    ``shaft_node`` is a degree-2 node: its two same-segment neighbors define a reliable
    centerline direction ``u``. Requires (1) the tip to lie within ``max_perp_um`` of
    that centerline (the component of ``tip - shaft`` orthogonal to ``u`` is small — a
    true break sits ON the line; a laterally-offset parallel neurite of a different
    neuron does NOT), and (2) the tip's walked tangent to be aligned (parallel or
    anti-parallel) with the axis, |cos| >= ``min_align_cos`` (a continuation, not a
    perpendicular T-junction). Returns False if the axis or tangent is undefined.
    """
    xyz = g.node_xyz
    seg = g.node_segment_id(shaft_node)
    nbrs = [n for n in g.neighbors(shaft_node) if g.node_segment_id(n) == seg]
    if len(nbrs) < 2:
        return False
    u = np.asarray(xyz[nbrs[1]], dtype=float) - np.asarray(xyz[nbrs[0]], dtype=float)
    nu = np.linalg.norm(u)
    if nu == 0:
        return False
    u = u / nu
    w = np.asarray(xyz[tip_node], dtype=float) - np.asarray(xyz[shaft_node], dtype=float)
    perp = float(np.linalg.norm(w - float(np.dot(w, u)) * u))
    if perp > max_perp_um:
        return False
    t = _walk_tangent(g, tip_node, TANGENT_WALK_UM)
    if t is None:
        return False
    return abs(float(np.dot(t, u))) >= min_align_cos


def _tip_closes_on_shaft(g, tip_node, shaft_node, min_closing_cos):
    """True if ``tip_node``'s tangent angles INTO ``shaft_node``'s centerline.

    ``shaft_node`` is a degree-2 node: its two same-segment neighbors define a
    reliable axis ``u``. Let ``w = pos_tip - pos_shaft`` and ``v1`` = the tip's
    outward tangent (``_walk_tangent``). Decompose BOTH into the part perpendicular
    to ``u``: ``w_perp`` (the tip's current lateral offset from the shaft line) and
    ``v1_perp`` (how the tip is travelling ACROSS the axis). The tip is CLOSING on
    the shaft line iff advancing along ``v1`` shrinks ``|w_perp|`` — i.e.
    ``cos(v1_perp, w_perp) <= -min_closing_cos``. This admits an ANGLED branch tip
    reconnecting to its PARENT shaft (large perp offset, tangent ~perpendicular to
    the axis — exactly what Path 7's small-perp/parallel-align gate and Path 13's
    first-half gap cosine both miss) while REJECTING a laterally-offset PARALLEL
    neurite of a different neuron (``v1`` ~parallel to ``u`` so ``v1_perp`` ~ 0,
    guarded out) and a tip moving AWAY from the shaft (cos > 0). Returns False on
    any degeneracy.
    """
    xyz = g.node_xyz
    seg = g.node_segment_id(shaft_node)
    nbrs = [n for n in g.neighbors(shaft_node) if g.node_segment_id(n) == seg]
    if len(nbrs) < 2:
        return False
    u = np.asarray(xyz[nbrs[1]], dtype=float) - np.asarray(xyz[nbrs[0]], dtype=float)
    nu = np.linalg.norm(u)
    if nu == 0:
        return False
    u = u / nu
    v1 = _walk_tangent(g, tip_node, TANGENT_WALK_UM)
    if v1 is None:
        return False
    w = np.asarray(xyz[tip_node], dtype=float) - np.asarray(xyz[shaft_node], dtype=float)
    w_perp = w - float(np.dot(w, u)) * u
    v1_perp = v1 - float(np.dot(v1, u)) * u
    nw = np.linalg.norm(w_perp)
    nv = np.linalg.norm(v1_perp)
    if nw == 0 or nv < 0.5:   # tip already on the line, or tangent ~parallel to the axis (parallel neurite)
        return False
    return float(np.dot(w_perp, v1_perp)) / (nw * nv) <= -min_closing_cos


def propose_edits(sites, ctx) -> list:
    """Decide which candidate sites to repair, returning a list of edits.

    SEED POLICY (conservative split-repair): for each SplitSite, emit a
    ``merge_labels`` edit only when BOTH:
      (1) the gap is small  (s.gap_um <= GAP_THRESHOLD_UM), and
      (2) the two fragments are COLINEAR across the gap (a straight-line
          continuation — see ``_is_colinear_split``).
    MergeSites are left alone (no ``split_label``) — splitting is the riskier edit
    and is left for the evolution loop to add once it can measure the trade-off.

    This proposes a small, high-precision set of merges, so the score moves OFF the
    flat no-edit baseline (giving the loop a gradient) without the union-find
    mega-label blowup a naive "merge everything nearby" seed causes. The agent's
    job is to improve on it: tune the thresholds, add tangent/radius/continuity
    features, gate with the image reader, or add merge-repair.

    Returns a list of edits — legacy (label_a, label_b) tuples or typed dicts; see
    the module docstring. An empty list means "make no changes".
    """
    g = ctx.get("fragments_graph")
    if g is None:
        return []

    reader = ctx.get("read_image_patch")  # may be None when --with-image is off

    # Precompute the node_a world positions of EVERY split-kind site, crash-safely,
    # to feed the Gen 8 dense-neuropil veto. An `inf` placeholder keeps an
    # unresolved position from ever counting as a neighbor and preserves index
    # alignment for the self-exclusion in Path 4.
    split_pos = []
    for _s in sites:
        if getattr(_s, "kind", "split") != "split":
            continue
        try:
            split_pos.append(np.asarray(g.node_xyz[_s.node_a], dtype=float))
        except Exception:
            split_pos.append(np.array([np.inf, np.inf, np.inf]))
    split_pos = np.asarray(split_pos, dtype=float) if split_pos else np.zeros((0, 3))

    edits = []
    for s in sites:
        kind = getattr(s, "kind", "split")
        if kind != "split":
            continue  # seed repairs splits only; leave merges for evolution
        if s.gap_um > GAP_THRESHOLD_UM:
            continue
        try:
            # Path 1 (unchanged): strict colinear continuation — high precision.
            if _is_colinear_split(g, s, MIN_COLINEAR_COS):
                edits.append(s.as_edit())   # (label_a, label_b) merge tuple
                continue

            # Path 4 (distance-only tip-to-tip, dense-neuropil veto, no image):
            # at VERY small gaps the gap-direction vector is voxel noise, so the
            # colinear gate wrongly drops genuine broken neurons. Finding #1 shows
            # distance ALONE is near-decisive that deep below the ~7 µm inter-neuron
            # floor, so accept tip-to-tip here with no colinearity/image — UNLESS
            # the site sits in a dense fragment-graph tangle (Finding #13), where
            # false-join risk concentrates. No image read anywhere in this path.
            if s.gap_um <= DIST_ONLY_UM:
                seg_a = g.node_segment_id(s.node_a)
                a_tip = len([n for n in g.neighbors(s.node_a)
                             if g.node_segment_id(n) == seg_a]) == 1
                seg_b = g.node_segment_id(s.node_b)
                b_tip = len([n for n in g.neighbors(s.node_b)
                             if g.node_segment_id(n) == seg_b]) == 1
                if a_tip and b_tip:
                    p = np.asarray(g.node_xyz[s.node_a], dtype=float)
                    n_near = int(np.count_nonzero(
                        np.linalg.norm(split_pos - p, axis=1) <= CLUSTER_RADIUS_UM
                    )) - 1  # exclude the site itself (distance 0)
                    if n_near <= DENSE_VETO_MAX:
                        edits.append(s.as_edit())
                        continue

            # Path 7 (tip-to-shaft small-gap recall via shaft-centerline offset,
            # Gen 15; band extended Gen 16, image-free): the largest untapped MISSED
            # bucket is tip-to-SHAFT at 1.5–2.5 µm with low (noisy) colinear_cos.
            # Accept when tip A lies on shaft B's centerline (perpendicular offset <=
            # SHAFT_PERP_UM) and the tip tangent is aligned with the shaft axis
            # (|cos| >= SHAFT_ALIGN_COS) — the geometry that separates a true mid-
            # neurite break (tip ON the line) from a parallel adjacent neurite
            # (laterally offset by the gap) that image could not reject (gen10).
            # Gen 16 widened the ceiling DIST_ONLY_UM -> SHAFT_RECALL_FAR_UM (2.5 µm)
            # to reach the 2.02–2.21 µm MISSED rows; the perp+align precision is gap-
            # independent so it carries into the wider band. Still far below the ~7 µm
            # inter-neuron floor (Finding #1) with the dense-neuropil veto (Finding
            # #13). Alignment per Finding #16.
            if SMALL_GAP_UM < s.gap_um <= SHAFT_RECALL_FAR_UM:
                seg_a = g.node_segment_id(s.node_a)
                a_tip = len([n for n in g.neighbors(s.node_a)
                             if g.node_segment_id(n) == seg_a]) == 1
                seg_b = g.node_segment_id(s.node_b)
                b_deg = len([n for n in g.neighbors(s.node_b)
                             if g.node_segment_id(n) == seg_b])
                if a_tip and b_deg == 2 and _shaft_continuation_ok(
                        g, s.node_a, s.node_b, SHAFT_PERP_UM, SHAFT_ALIGN_COS):
                    p = np.asarray(g.node_xyz[s.node_a], dtype=float)
                    n_near = int(np.count_nonzero(
                        np.linalg.norm(split_pos - p, axis=1) <= CLUSTER_RADIUS_UM
                    )) - 1  # exclude the site itself (distance 0)
                    if n_near <= DENSE_VETO_MAX:
                        edits.append(s.as_edit())
                        continue

            # Path 8 (image-free moderate-colinear tip-to-shaft, Gen 18): Path 7's
            # shaft-centerline perp/align test cracked only ONE of the 1.5–2.5 µm
            # tip-to-shaft reals, and gen16/gen17 (gap/perp tweaks) tied the parent —
            # its alignment gate is the binding constraint. Attack the SAME bucket with
            # a DIFFERENT discriminator: the gap COLINEARITY itself (the report's single
            # strongest precision feature). Accept a tip-to-shaft site whose arms
            # continue straight across the gap (_is_colinear_split >= COLINEAR_TS_COS,
            # a strict AND of both half-cosines), image-free. COLINEAR_TS_COS (0.72)
            # sits ABOVE the lone FALSE join's 0.55 (margin) and BELOW Path 2's image
            # floor 0.80, so it adds straight reals that Path 1 (0.94), Path 2 (needs an
            # image bridge) and Path 7 (perp/align) all miss. Gap <= SHAFT_RECALL_FAR_UM
            # keeps it far below the ~7 µm inter-neuron floor (Finding #1) and excludes
            # the 3.12 µm FALSE by gap too; the dense-neuropil veto (Finding #13) still
            # applies. Colinear continuation separability per Finding #16.
            if SMALL_GAP_UM < s.gap_um <= SHAFT_RECALL_FAR_UM:
                seg_b8 = g.node_segment_id(s.node_b)
                b_deg8 = len([n for n in g.neighbors(s.node_b)
                              if g.node_segment_id(n) == seg_b8])
                if b_deg8 == 2 and _is_colinear_split(g, s, COLINEAR_TS_COS):
                    p = np.asarray(g.node_xyz[s.node_a], dtype=float)
                    n_near = int(np.count_nonzero(
                        np.linalg.norm(split_pos - p, axis=1) <= CLUSTER_RADIUS_UM
                    )) - 1  # exclude the site itself (distance 0)
                    if n_near <= DENSE_VETO_MAX:
                        edits.append(s.as_edit())
                        continue

            # Path 13 (image-confirmed branch-tip reconnection, Gen 25): the LARGEST
            # MISSED-real bucket is tip-to-shaft (deg_a=1, deg_b=2) at 1.5-2.5 um with
            # LOW colinear_cos and rad_b >> rad_a -- a THIN tip meeting a THICK shaft,
            # i.e. a branch tip that should reconnect to its PARENT shaft (a T/Y
            # junction). `_is_colinear_split` rejects these because its SECOND half-
            # cosine (gap vs the shaft's continuation) is legitimately low (the parent
            # runs ACROSS the gap), and the perp-offset paths (7, reverted gen24) reject
            # them because a branch tip meets the shaft at an ANGLE (large perp). Reach
            # them with a DIFFERENT discriminator: the FIRST half-cosine alone --
            # `_tip_heads_into_gap` requires tip A's outward tangent to point INTO the
            # gap toward node_b (cos >= COLINEAR_TS_COS, 0.72). This admits the branch-
            # reconnection geometry yet REJECTS the laterally-offset parallel neurite of
            # a DIFFERENT neuron (its tip tangent is ~perpendicular to the shortest gap
            # vector, cos ~ 0) -- exactly the cross-neuron mode that tripped image-ONLY
            # gen21. Couple it with an image bright bridge (bridge_ratio >=
            # IMG_FAR_BRIDGE_MIN, 0.95) so geometry supplies the precision and image
            # confirms continuity: a DOUBLE filter. Cheap geometric filters (deg_b==2,
            # tip-heads, dense veto) gate the cloud read to the handful of ambiguous
            # tip-to-shaft junctions. Precision per Findings #14/#20 (splits enriched at
            # branch points) and #16 (directional alignment); gap <= SHAFT_RECALL_FAR_UM
            # stays far below the ~7 um inter-neuron floor (#1); dense-neuropil veto
            # (#13). Placed after Path 8 (image-free colinear takes its sites first);
            # STRICTLY ADDITIVE (appends + continues only on a confirmed bridge).
            if (reader is not None
                    and SMALL_GAP_UM < s.gap_um <= SHAFT_RECALL_FAR_UM):
                seg_a13 = g.node_segment_id(s.node_a)
                a_tip13 = len([n for n in g.neighbors(s.node_a)
                               if g.node_segment_id(n) == seg_a13]) == 1
                seg_b13 = g.node_segment_id(s.node_b)
                b_deg13 = len([n for n in g.neighbors(s.node_b)
                               if g.node_segment_id(n) == seg_b13])
                if a_tip13 and b_deg13 == 2 and _tip_heads_into_gap(g, s, COLINEAR_TS_COS):
                    p = np.asarray(g.node_xyz[s.node_a], dtype=float)
                    n_near13 = int(np.count_nonzero(
                        np.linalg.norm(split_pos - p, axis=1) <= CLUSTER_RADIUS_UM
                    )) - 1  # exclude the site itself (distance 0)
                    if n_near13 <= DENSE_VETO_MAX:
                        ev = reader.gap_bridge_evidence(s.node_a, s.node_b)
                        br = ev.get("bridge_ratio", float("nan"))
                        if np.isfinite(br) and br >= IMG_FAR_BRIDGE_MIN:
                            edits.append(s.as_edit())
                            continue

            # Path 14 (image-confirmed tip-to-tip reconnection just beyond the
            # distance-only band, Gen 26): with Path 13 recovering tip-to-shaft branch
            # junctions, the LARGEST remaining MISSED bucket is TIP-TO-TIP (deg_a=1,
            # deg_b=1) clustered at the 2.24 um (sqrt(5)-voxel) shell -- just ABOVE
            # Path 4's distance-only ceiling DIST_ONLY_UM (2.0 um), where gen14 found
            # distance-ALONE tip-to-tip turns held-out-unsafe, and BELOW Path 1's 0.94 /
            # Path 2's 0.80 colinear floors (their colinear_cos is low/noisy at this
            # gap). Recover the real ones with the SAME double filter that made Path 13
            # held-out-safe, now on the SAFER tip-to-tip topology (the report notes
            # joining a shaft/branch is RISKIER than tip-to-tip): require tip A to HEAD
            # INTO the gap toward B (_tip_heads_into_gap >= COLINEAR_TS_COS, 0.72 -- which
            # REJECTS a laterally-offset parallel neurite whose tip tangent is ~perpend-
            # icular to the gap, cos ~ 0, the cross-neuron mode that sank image-ONLY
            # gen21 and antiparallel-arm gen22) AND an image bright bridge (bridge_ratio
            # >= IMG_FAR_BRIDGE_MIN, 0.95). Dense-neuropil veto (Finding #13). The band
            # DIST_ONLY_UM < gap <= SHAFT_RECALL_FAR_UM sits strictly above Path 4's
            # tip-to-tip accepts and far below the ~7 um inter-neuron floor (Finding #1).
            # Placed after Path 13; STRICTLY ADDITIVE (appends + continues only on a
            # confirmed bridge).
            if (reader is not None
                    and DIST_ONLY_UM < s.gap_um <= SHAFT_RECALL_FAR_UM):
                seg_a14 = g.node_segment_id(s.node_a)
                a_tip14 = len([n for n in g.neighbors(s.node_a)
                               if g.node_segment_id(n) == seg_a14]) == 1
                seg_b14 = g.node_segment_id(s.node_b)
                b_deg14 = len([n for n in g.neighbors(s.node_b)
                               if g.node_segment_id(n) == seg_b14])
                if a_tip14 and b_deg14 == 1 and _tip_heads_into_gap(g, s, COLINEAR_TS_COS):
                    p = np.asarray(g.node_xyz[s.node_a], dtype=float)
                    n_near14 = int(np.count_nonzero(
                        np.linalg.norm(split_pos - p, axis=1) <= CLUSTER_RADIUS_UM
                    )) - 1  # exclude the site itself (distance 0)
                    if n_near14 <= DENSE_VETO_MAX:
                        ev = reader.gap_bridge_evidence(s.node_a, s.node_b)
                        br = ev.get("bridge_ratio", float("nan"))
                        if np.isfinite(br) and br >= IMG_FAR_BRIDGE_MIN:
                            edits.append(s.as_edit())
                            continue

            # Path 15 (image-confirmed ANGLED branch-tip reconnection to the parent
            # shaft, Gen 27): the LARGEST remaining MISSED-real bucket is THIN-tip ->
            # THICK-shaft branch junctions (deg_a==1, deg_b==2) at gap 1.5-2.2 um with
            # rad_b >> rad_a and LOW colinear_cos. These are reached by NEITHER the
            # colinear AND-test (Paths 1/8/11 — the SECOND half-cosine is low because the
            # parent shaft runs ACROSS the gap), NOR Path 7's shaft-centerline test (an
            # angled branch tip has a LARGE perpendicular offset and a tangent ~perpend-
            # icular to the shaft axis, while Path 7 only accepts small-perp, tangent-
            # PARALLEL continuations), NOR Path 13's `_tip_heads_into_gap` (the angled
            # junctions have first-half gap cosine < 0.72). The NEW discriminator is the
            # PERPENDICULAR-CLOSING cosine `_tip_closes_on_shaft`: the tip's tangent
            # component ACROSS the shaft axis must point OPPOSITE the tip's current
            # lateral offset (cos <= -SHAFT_CLOSING_COS, 0.5) — i.e. advancing along the
            # tangent SHRINKS the offset, the tip is angling INTO the parent centerline.
            # This admits the angled branch junction yet REJECTS a laterally-offset
            # PARALLEL neurite of a different neuron (its tangent is ~parallel to the
            # shaft axis, so its perpendicular tangent component is ~0 and the helper's
            # `nv < 0.5` guard drops it) — exactly the cross-neuron mode that sank image-
            # ONLY gen21/gen22. A DOUBLE filter: geometry supplies precision, the image
            # bright bridge (bridge_ratio >= IMG_FAR_BRIDGE_MIN, 0.95) confirms continuity
            # — the SAME held-out-safe recipe that made Path 13 survive with zero false
            # merges. Paths 13/7 are placed EARLIER and `continue` on accept, so any site
            # they take never reaches here; Path 15 only ADDS the angled junctions they
            # miss. The 1.5-2.5 um band stays far below the ~7 um inter-neuron floor
            # (Finding #1; true-split gaps peak ~4.5 um) and below the lone 3.12 um FALSE
            # join (also excluded by the deg_b==2 + closing geometry). The dense-neuropil
            # veto (Finding #13) gates the cloud read to the handful of ambiguous
            # junctions. Precision per Findings #14/#20 (splits enriched ~3.4x near branch
            # points) and #16 (directional alignment). STRICTLY ADDITIVE: it appends and
            # `continue`s ONLY on a confirmed bridge, so it can never remove an existing
            # accept — it cannot lose the parent's 218.
            if (reader is not None
                    and SMALL_GAP_UM < s.gap_um <= SHAFT_RECALL_FAR_UM):
                seg_a15 = g.node_segment_id(s.node_a)
                a_tip15 = len([n for n in g.neighbors(s.node_a)
                               if g.node_segment_id(n) == seg_a15]) == 1
                seg_b15 = g.node_segment_id(s.node_b)
                b_deg15 = len([n for n in g.neighbors(s.node_b)
                               if g.node_segment_id(n) == seg_b15])
                if a_tip15 and b_deg15 == 2 and _tip_closes_on_shaft(
                        g, s.node_a, s.node_b, SHAFT_CLOSING_COS):
                    p = np.asarray(g.node_xyz[s.node_a], dtype=float)
                    n_near15 = int(np.count_nonzero(
                        np.linalg.norm(split_pos - p, axis=1) <= CLUSTER_RADIUS_UM
                    )) - 1  # exclude the site itself (distance 0)
                    if n_near15 <= DENSE_VETO_MAX:
                        ev = reader.gap_bridge_evidence(s.node_a, s.node_b)
                        br = ev.get("bridge_ratio", float("nan"))
                        if np.isfinite(br) and br >= IMG_FAR_BRIDGE_MIN:
                            edits.append(s.as_edit())
                            continue

            # Path 9 (image-confirmed medium-gap recall, Gen 20): the harness Image
            # warm-start probe measured bridge_ratio AUC = 0.83 and found 12 REAL
            # splits at gap 2.79-2.83 um all reading bridge_ratio >= 0.95 — reals NO
            # geometric path reaches (Path 1 needs colinear 0.94; Paths 7/8 cap at
            # SHAFT_RECALL_FAR_UM = 2.5; Path 2 needs colinear 0.80 and these reals
            # have LOW colinear_cos). Spend an image read to RAISE RECALL here: for a
            # tip whose partner is a tip or shaft (deg_b in {1,2}, never a branch) in
            # the 2.5-3.0 um band, gate the cloud read behind the cheap dense-neuropil
            # veto (Finding #13), then ACCEPT only when a continuous bright bridge
            # confirms it (bridge_ratio >= IMG_FAR_BRIDGE_MIN). Precision rests on
            # GEOMETRY, not the image: Finding #1 puts cross-neuron neighbours rarely
            # below ~7 um (true-split gaps peak ~4.5 um), so the 2.5-3.0 um band is
            # overwhelmingly real-split territory, and the 3.0 um cap excludes the lone
            # 3.12 um FALSE join (whose bridge_ratio is 1.00, so image alone could NOT
            # reject it). This path is STRICTLY ADDITIVE: it appends only on a
            # confirmed bridge and never blocks Path 2/Path 3, so it cannot lose an
            # existing accept. Distinct from Gen 19's geometry-only medium-gap path,
            # which tied because these reals' colinear_cos is too low for the AND-test.
            if (reader is not None
                    and SHAFT_RECALL_FAR_UM < s.gap_um <= IMG_FAR_GAP_UM):
                seg_a9 = g.node_segment_id(s.node_a)
                a_tip9 = len([n for n in g.neighbors(s.node_a)
                              if g.node_segment_id(n) == seg_a9]) == 1
                seg_b9 = g.node_segment_id(s.node_b)
                b_deg9 = len([n for n in g.neighbors(s.node_b)
                              if g.node_segment_id(n) == seg_b9])
                if a_tip9 and b_deg9 in (1, 2):
                    p = np.asarray(g.node_xyz[s.node_a], dtype=float)
                    n_near9 = int(np.count_nonzero(
                        np.linalg.norm(split_pos - p, axis=1) <= CLUSTER_RADIUS_UM
                    )) - 1  # exclude the site itself (distance 0)
                    if n_near9 <= DENSE_VETO_MAX:
                        ev = reader.gap_bridge_evidence(s.node_a, s.node_b)
                        br = ev.get("bridge_ratio", float("nan"))
                        if np.isfinite(br) and br >= IMG_FAR_BRIDGE_MIN:
                            edits.append(s.as_edit())
                            continue

            # Path 3 (small-gap, distance-dominant): for TINY gaps the gap-direction
            # vector is voxel noise, so colinearity wrongly rejects real splits. Drop
            # the colinearity requirement and accept on an image bridge confirmation
            # alone. The cheap gap test gates the (cloud) read, and `continue` ensures
            # a tiny-gap site is read at most once (Path 2 never re-reads it). If the
            # read fails it returns NaN and the site is skipped — never a false merge.
            if reader is not None and s.gap_um <= SMALL_GAP_UM:
                ev = reader.gap_bridge_evidence(s.node_a, s.node_b)
                br = ev.get("bridge_ratio", float("nan"))
                if np.isfinite(br) and br >= BRIDGE_RATIO_MIN:
                    edits.append(s.as_edit())
                continue

            # Path 2 (image-gated recall): moderately-colinear short gaps that the
            # strict gate misses, CONFIRMED by a continuous bright image bridge.
            # The cheap geometric pre-filter (cos >= IMG_MIN_COLINEAR_COS) keeps
            # image reads to the handful of ambiguous candidates and rejects the
            # lone cos=0.55 FALSE join BEFORE any read.
            if reader is not None and _is_colinear_split(g, s, IMG_MIN_COLINEAR_COS):
                ev = reader.gap_bridge_evidence(s.node_a, s.node_b)
                br = ev.get("bridge_ratio", float("nan"))
                if np.isfinite(br) and br >= BRIDGE_RATIO_MIN:
                    edits.append(s.as_edit())

            # Path 11 (image + colinear at the true-split PEAK band, Gen 23): Finding #1
            # puts the TRUE-split gap distribution PEAK at ~4.5 um (inter-neuron floor
            # ~7 um), so the 3.0-4.0 um band holds many real splits and few cross-neuron
            # joins, yet the parent accepts almost none there (Path 1 needs colinear 0.94,
            # Path 2 needs 0.80, Path 9 caps at 3.0 um). The last three recall pushes into
            # the 1.5-2.5 um band failed (gen19 tied; gen21 image-only tip->shaft and
            # gen22 tip->tip antiparallel both tripped HELD-OUT false merges), so attack a
            # different, richer band with a DOUBLE filter. At 3-4 um the gap vector spans
            # many voxels and is RELIABLE, so the colinear AND-test is trustworthy here:
            # require _is_colinear_split >= COLINEAR_TS_COS (0.72, both half-cosines vs the
            # gap vector -- the same discriminator as held-out-safe Path 8) AND a bright
            # image bridge (bridge_ratio >= IMG_FAR_BRIDGE_MIN, 0.95). The 0.72 colinear
            # floor excludes the lone 3.12 um FALSE join (colinear_cos 0.55, which image
            # alone could NOT reject: its bridge is 1.00) and the parallel-offset
            # cross-neuron mode (arms not along the gap); the image excludes dark-gap
            # non-continuations. Dense-neuropil veto (Finding #13). Placed LAST so Path 2
            # (colinear >= 0.80) takes its sites first; Path 11 only adds the colinear
            # [0.72, 0.80) slice Path 2 skips. Strictly additive.
            if (reader is not None
                    and IMG_FAR_GAP_UM < s.gap_um <= GAP_THRESHOLD_UM
                    and _is_colinear_split(g, s, COLINEAR_TS_COS)):
                p = np.asarray(g.node_xyz[s.node_a], dtype=float)
                n_near11 = int(np.count_nonzero(
                    np.linalg.norm(split_pos - p, axis=1) <= CLUSTER_RADIUS_UM
                )) - 1  # exclude the site itself (distance 0)
                if n_near11 <= DENSE_VETO_MAX:
                    ev = reader.gap_bridge_evidence(s.node_a, s.node_b)
                    br = ev.get("bridge_ratio", float("nan"))
                    if np.isfinite(br) and br >= IMG_FAR_BRIDGE_MIN:
                        edits.append(s.as_edit())
                        continue
        except Exception:
            # Geometry is advisory; never let one odd site crash the whole policy.
            continue
    return edits
