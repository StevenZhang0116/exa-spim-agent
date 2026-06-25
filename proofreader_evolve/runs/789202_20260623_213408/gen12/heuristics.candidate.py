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

# --- Forward-cone farther tip-to-tip recall (Gen 12) -------------------------
# The 412 missed reals are all > DIST_ONLY_UM; the safely recoverable subset is the
# tip-to-tip cluster at ~2.24 µm whose normalized gap-vector cosine is voxel noise.
# Recover them with a PRECISION test built on the RELIABLE walked arm tangent (not
# the noisy gap vector): for BOTH tips, the partner must lie AHEAD of the tip's
# outward direction (positive projection) AND within a small ABSOLUTE perpendicular
# offset of that direction ray. KB Finding #16 (endpoint directional alignment
# discriminates true ~153° from false ~90°, AUC 0.9322, OK/GENERALIZES) is the
# precision warrant; an absolute µm offset is robust where the normalized gap cosine
# is not. This rejects offset-parallel lanes of different neurons (large perpendicular
# offset) without requiring branch-adjacency (gen11 showed branches are crowded and
# admit false joins). Distance 2.5 µm stays far below the ~7 µm inter-neuron floor
# (Finding #1); dense-neuropil veto retained (Finding #13); image-free; tip-to-tip only.
DIST_ONLY_FAR_UM = 2.5      # farther tip-to-tip distance tier (µm), gated on the bidirectional forward-cone test
FWD_CONE_OFFSET_UM = 1.0    # max absolute perpendicular offset (µm) of the partner from a tip's outward tangent ray

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


def _forward_cone_ok(g, node_from, node_to, max_offset_um):
    """True if node_to lies AHEAD of node_from's outward arm tangent and within
    ``max_offset_um`` absolute perpendicular offset of that tangent ray.

    Uses the RELIABLE walked tangent (over TANGENT_WALK_UM of cable) rather than the
    short, voxel-noise-dominated gap vector. Decomposes the gap vector into a
    component along the tangent (must be positive = partner is ahead) and a
    perpendicular component (must be small = partner is on the line of travel, not in
    an offset parallel lane). Returns False if the tangent is undefined.
    """
    xyz = g.node_xyz
    t = _walk_tangent(g, node_from, TANGENT_WALK_UM)
    if t is None:
        return False
    gvec = np.asarray(xyz[node_to], dtype=float) - np.asarray(xyz[node_from], dtype=float)
    par = float(np.dot(gvec, t))
    if par <= 0.0:
        return False
    perp = float(np.linalg.norm(gvec - par * t))
    return perp <= max_offset_um


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

            # Path 4b (forward-cone farther tip-to-tip, dense-neuropil veto, no
            # image): the remaining missed reals sit just past DIST_ONLY_UM, mostly a
            # tip-to-tip cluster at ~2.24 µm whose gap-vector cosine is voxel noise.
            # Accept tip-to-tip out to DIST_ONLY_FAR_UM ONLY when BOTH tips' reliable
            # walked tangents point at the partner (partner ahead + small absolute
            # perpendicular offset — Finding #16 alignment, robust to gap-vector
            # noise). This rejects offset-parallel lanes of different neurons without
            # the branch-adjacency requirement that backfired in gen11. Dense tangles
            # are still vetoed (Finding #13). No image read in this path.
            if DIST_ONLY_UM < s.gap_um <= DIST_ONLY_FAR_UM:
                seg_a = g.node_segment_id(s.node_a)
                a_tip = len([n for n in g.neighbors(s.node_a)
                             if g.node_segment_id(n) == seg_a]) == 1
                seg_b = g.node_segment_id(s.node_b)
                b_tip = len([n for n in g.neighbors(s.node_b)
                             if g.node_segment_id(n) == seg_b]) == 1
                if a_tip and b_tip and \
                        _forward_cone_ok(g, s.node_a, s.node_b, FWD_CONE_OFFSET_UM) and \
                        _forward_cone_ok(g, s.node_b, s.node_a, FWD_CONE_OFFSET_UM):
                    p = np.asarray(g.node_xyz[s.node_a], dtype=float)
                    n_near = int(np.count_nonzero(
                        np.linalg.norm(split_pos - p, axis=1) <= CLUSTER_RADIUS_UM
                    )) - 1  # exclude the site itself (distance 0)
                    if n_near <= DENSE_VETO_MAX:
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
        except Exception:
            # Geometry is advisory; never let one odd site crash the whole policy.
            continue
    return edits
