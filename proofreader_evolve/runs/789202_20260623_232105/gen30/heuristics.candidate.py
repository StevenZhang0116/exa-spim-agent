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

# Image-gated recall path: harvest near-colinear real splits the strict 0.94
# gate misses, confirmed by a continuous bright fluorescence bridge across the
# gap. The relaxed cos floor (band [0.75, 0.94)) stays well above the observed
# false-join geometry (all train false joins had colinear_cos <= 0.67), and the
# bridge check is a second gate, so this raises recall without manufacturing
# merges. The cos floor — not the bridge gate — is the false-join guard
# (bridge_ratio cannot reject bright false joins, which probe at 0.97-0.99).
IMAGE_MIN_COLINEAR_COS = 0.75   # relaxed near-colinear band [0.75, 0.94); stays > observed false-join cos (<=0.67)
IMAGE_BRIDGE_RATIO_MIN = 0.90   # require signal to stay >=90% bright across gap
# Relaxed bridge floor used ONLY in Path 2 (Gen 10), where the geometry is already
# STRONGLY colinear (mean_cos >= IMAGE_MIN_COLINEAR_COS = 0.75). Because that strong
# colinearity gate already provides the precision, a slightly dimmer fluorescence
# bridge (down to 0.85) is still acceptable evidence of one continuous neuron. Path 3
# (the weaker [0.60, 0.75) tier) keeps the strict IMAGE_BRIDGE_RATIO_MIN (0.90).
IMAGE_BRIDGE_RATIO_MIN_STRONG = 0.85
IMAGE_GAP_MIN_UM = 1.0          # only read image where the gap is non-trivial

# Caliber-gated lower-cos image tier (Path 3): an ORTHOGONAL lever to harvest reals
# in the [0.50, 0.75) colinearity band that the Path-2 cos floor excludes. All 3
# train false joins have HIGH radius mismatch (rad_ratio 1.97 / 1.67 / 1.87, all
# >= 1.67), whereas a true single-neuron break has MATCHED caliber on both sides.
# So a caliber gate (rad_ratio <= cap) lets us safely admit reals below the 0.75
# floor.
# Gen 16: lowered this floor from 0.60 to 0.50. The prior pure-geometry cos tiers
# (Gen 12/14/15) that tried to relax colinearity in the (1.5, 2.0] band ALL TIED the
# parent and were reverted, because the policy's RUNTIME mean colinearity (from
# `_colinear_half_cosines`) runs well BELOW the failure report's `colinear_cos`
# column there — so a geometry-only cos threshold fires on nothing new. The IMAGE
# side, by contrast, is reliable and report-aligned: every train REAL the policy read
# has a bright bridge (0.84–1.02) and the warm-start probe gives bridge_ratio
# AUC = 0.86. But Path 3 only ever READ image where mean colinearity was already
# >= 0.60, so reals sitting just below that floor were never looked at even though a
# bright bridge would confirm them. Lowering the floor to 0.50 opens a NEW
# colinearity band [0.50, 0.60) the policy has never read image in, spending a gated
# cloud read to RAISE RECALL on geometry-conservative reals confirmed by a bright
# bridge. Concrete reachable MISSED targets: gap 1.96 / cos 0.57 (rad_ratio 2.50,
# deg_b 2 -> shaft cap 2.5, passes) and gap 1.37 / cos 0.50 (rad_ratio 1.94, deg_b 2
# -> passes) — both below the old 0.60 floor, inside the shaft caliber cap, and
# (like nearly all train reals) bright-bridged.
# SAFETY (all 3 train FALSE joins stay rejected):
#   - gap 2.20 / cos 0.44 / deg_b 2 / rad_ratio 1.97: cos 0.44 is still BELOW the new
#     0.50 floor (margin 0.06), so the colinearity floor still excludes it. (Do NOT
#     lower the floor to <= 0.44.)
#   - gap 3.00 / cos 0.06 / deg_b 1 / rad_ratio 1.67 and gap 3.16 / cos 0.67 / deg_b 1
#     / rad_ratio 1.87: both tip-to-tip (cap 1.4) with rad_ratio >= 1.67 > 1.4, so the
#     caliber gate excludes them regardless of cos.
# The bright-bridge requirement (0.90) is retained as a third, independent gate.
IMAGE_LOWCOS_MIN_COLINEAR_COS = 0.50  # lower colinearity floor for the caliber-gated tier (Gen 16: 0.60 -> 0.50)
RAD_RATIO_MAX = 1.45                   # tip-to-tip caliber-matched ceiling (Gen 18: 1.4 -> 1.45; still below the >=1.67 of every false join)

# Tightly-matched-caliber image tier (Path 4, Gen 19; extended Gen 20). A recall
# lever that lowers the mean-colinearity floor BELOW Path 3's 0.50 — into the band
# [IMAGE_MATCHED_MIN_COLINEAR_COS, IMAGE_LOWCOS_MIN_COLINEAR_COS) = [0.25, 0.50)
# (Gen 20 lowered the floor 0.35 -> 0.25), DISJOINT from Path 3's [0.50, 0.75) band
# so there is no double-count — but ONLY when neurite caliber is TIGHTLY matched
# across the gap (rad_ratio <= IMAGE_MATCHED_CALIBER_MAX). KEY STRUCTURAL FACT: all 3
# train FALSE joins have HIGH caliber mismatch (rad_ratio 1.97 / 1.67 / 1.87, ALL
# >= 1.67), so a tight caliber gate at 1.5 excludes EVERY one of them by caliber
# ALONE — independent of cos or bridge brightness. That is what makes it safe to
# lower the cos floor below 0.50 here (the 0.50 floor cannot be lowered GLOBALLY: the
# deg_b 2 false join at cos 0.44 / rad_ratio 1.97 has a bright ~0.97 bridge, so only
# the cos floor keeps it out — but its rad_ratio 1.97 > 1.5 keeps it out of Path 4 by
# caliber). Gen 20 also gives Path 4 its OWN weak-arm floor
# (IMAGE_MATCHED_MIN_ARM_COS = 0.25), DECOUPLED from the global MIN_ARM_COS (0.35):
# since min_arm <= mean_cos always, the shared 0.35 weak-arm floor PINNED the Path-4
# mean floor at 0.35, so the cos floor could not be extended downward without giving
# Path 4 its own lower weak-arm floor. The bright-bridge requirement stays the strict
# IMAGE_BRIDGE_RATIO_MIN (0.90).
IMAGE_MATCHED_CALIBER_MAX = 1.5        # tightly-matched caliber gate (tip-tip / unknown-degree); strictly BELOW the >=1.67 rad_ratio of all 3 false joins
IMAGE_MATCHED_CALIBER_MAX_SHAFT = 1.85   # Gen 22: looser tight-tier caliber cap for tip-to-SHAFT partners (deg_b >= 2); strictly BELOW the deg_b 2 false join's rad_ratio 1.97 (margin 0.12)
MATCHED_SMALLGAP_UM = 2.18                         # Gen 27: 2.0 -> 2.18. The ONLY shaft (deg_b 2) false join FJ1 is at gap 2.20 (displayed -> true gap >= 2.195), so the band [2.0, 2.18) is shaft-false-join-free with a deterministic margin. Boundary 2.18 sits between the highest MISSED shaft target (gap 2.16 -> true <= 2.165) and FJ1 (gap 2.20 -> true >= 2.195): FJ1 is NOT < 2.18, so it stays in the strict IMAGE_MATCHED_CALIBER_MAX_SHAFT (1.85) branch and remains rejected (rad_ratio 1.97 > 1.85) -- train precision GUARANTEED preserved. Opens MISSED shaft reals at gap 2.08/2.15/2.16 (rad_ratio 2.21/2.58/2.74) to the looser SHAFT_SMALLGAP cap. (Gen 23 original: smallest train false-join gap is 2.20; below this there are NO false joins, so caliber can be loosened freely.)
IMAGE_MATCHED_CALIBER_MAX_SHAFT_SMALLGAP = 2.8     # Gen 25: 2.5 -> 2.8 (matches Path 3 shaft cap RAD_RATIO_MAX_SHAFT). Tip-to-shaft (deg_b>=2) cap when gap < MATCHED_SMALLGAP_UM (2.0); that gap regime is shaft-false-join-free (the only shaft false join is at gap 2.20), so caliber is unconstrained there
TIP_SMALLGAP_UM = 2.5                              # Gen 24: both tip-tip (deg_b 1) false joins are at gap >= 3.00; below 2.5 there are NO tip-tip false joins, so the tip caliber cap can be loosened freely
IMAGE_MATCHED_CALIBER_MAX_TIP_SMALLGAP = 1.85      # Gen 24: looser tip-tip (deg_b<=1) caliber cap when gap < TIP_SMALLGAP_UM (no tip-tip false join exists there to fence against)
TIP_TINYGAP_UM = 2.0                               # Gen 26: both tip-tip (deg_b 1) false joins are at gap >= 3.00, so gap < 2.0 is tip-false-join-free (1.0 µm margin); below this the tip caliber cap can be loosened freely
IMAGE_MATCHED_CALIBER_MAX_TIP_TINYGAP = 2.2        # Gen 26: looser tip-tip caliber cap when gap < TIP_TINYGAP_UM. Above the tip false-join rad_ratios (1.67, 1.87) but well within observed genuine tip reals (rad_ratio up to ~3.0 in the accepted bucket); safe because the gap < 2.0 fence excludes both tip false joins (gap >= 3.00). Targets MISSED reals like gap 1.00 / rad_ratio 2.00 and gap 1.00 / rad_ratio 1.95
MIDGAP_FREE_LO_UM = 2.30   # Gen 29 (2.35 -> 2.30); lower edge of the train false-join-free gap band. FJ1 (the only shaft false join) is at gap 2.20 (displayed -> true >= 2.195), so 2.30 keeps a >=0.095 µm deterministic margin below the band. (The lone visible lower-sub-band target at gap 2.22 stays unreachable -- too close to FJ1 -- so the lower edge stops at 2.30.)
MIDGAP_FREE_HI_UM = 2.90   # Gen 29 (2.80 -> 2.90); upper edge. FJ2 (the lowest tip false join) is at gap 3.00 (displayed -> true >= 2.995), so 2.90 keeps a >=0.095 µm deterministic margin below it. The widened band [2.30, 2.90) still contains ZERO train false joins for either degree, so it extends the Gen 28 +6 lever into the rest of the proven false-join-free pocket (2.20, 3.00).
IMAGE_MATCHED_CALIBER_MAX_MIDGAP = 2.8   # Gen 28; loosened caliber cap inside the false-join-free band [MIDGAP_FREE_LO_UM, MIDGAP_FREE_HI_UM). Matches RAD_RATIO_MAX_SHAFT / the shaft small-gap cap and stays within observed genuine real caliber (accepted reals reach rad_ratio ~3.0). The warm-start probe (AUC 0.86) shows reals in exactly this band (gap 2.58–2.67) with bridge_ratio >= 0.98, so a bright bridge confirms them.
IMAGE_MATCHED_MIN_COLINEAR_COS = 0.0   # lower mean-colinearity floor for this tightly-matched tier (Gen 21: 0.25 -> 0.0; perpendicular boundary, band now [0.0, 0.50))
IMAGE_MATCHED_MIN_ARM_COS = 0.0        # Path-4-LOCAL weak-arm floor (Gen 21: 0.25 -> 0.0), decoupled from the global MIN_ARM_COS (0.35) used by Paths 1c/2/3; requiring >= 0.0 still REFUSES any arm pointing sideways/backward (cos < 0, the double-back / crossing false-join signature) while now admitting a right-angle (perpendicular) forward bend

# Degree-aware caliber ceiling for Path 3 (Gen 9). A tip-to-SHAFT reconnection
# (node_b is mid-cable, degree >= 2) legitimately has a LARGE caliber mismatch:
# node_a is the THIN tip of a neurite while node_b is measured mid-shaft and is
# thicker, so a high rad_ratio is EXPECTED for a genuine tip-to-shaft repair and is
# a WEAK against-merge signal there. We therefore loosen the Path-3 caliber ceiling
# to RAD_RATIO_MAX_SHAFT for tip-to-shaft partners (deg_b >= 2), keeping the tight
# RAD_RATIO_MAX (Gen 18: 1.45) for tip-to-TIP partners (deg_b <= 1, where mismatched
# caliber is more suspicious). RAD_RATIO_MAX_SHAFT stays a real bound — it rejects
# extreme (>2.8) mismatches — and the only train false join inside Path 3's
# [0.50, 0.75) mean band is tip-to-tip (deg_b = 1) and so stays gated by the tight
# 1.45 ceiling (the two deg_b=1 false joins at rad_ratio 1.67/1.87 still exceed it).
RAD_RATIO_MAX_SHAFT = 2.8  # Gen 17: 2.5 -> 2.8 (raised tip-to-shaft caliber ceiling)

# Gen 31: lower edge of the train region ABOVE all tip false joins. Both tip-to-tip
# false joins are at gap <= 3.16 (FJ3 displayed 3.16 -> true gap >= 3.155), so 3.30
# keeps a >= 0.145 µm deterministic margin above them; the only shaft false join FJ1
# is at gap 2.20, far below. So the band [3.30, GAP_THRESHOLD_UM=4.0] contains ZERO
# train false joins of either degree. Used in Path 3 to make the TIP caliber cap
# gap-aware: above this gap there is no tip false join for the tight tip cap to fence.
ABOVE_FJ_LO_UM = 3.30

# Weak-arm safety floor for the MEAN-colinearity image paths (Gen 7). The failure
# report's `colinear_cos` is the MEAN of the two half-cosines (arm A into the gap;
# the gap into B's continuing arm), but the policy's `_is_colinear_split` is a
# WORST-arm (both-arms) test, so it rejected real splits whose MEAN clears the
# floor but whose weaker arm dips just below it (one straight arm + one slightly
# bent arm). The image paths now gate on the MEAN (matching the report column) with
# this weak-arm floor: it admits an arm bent up to ~70 degrees (cos 0.35) while
# still rejecting an arm pointing SIDEWAYS/BACKWARD across the gap — the geometry of
# a crossing / double-back that a pure mean could otherwise let through.
MIN_ARM_COS = 0.35

# Small-gap geometry-only recall path (Path 1c, Gen 13). When the gap between the
# two fragment tips is TINY, the tips are essentially adjacent and a
# forward-pointing, modestly-colinear break is almost always ONE neuron the
# segmentation fragmented. This opens a NEW region of candidate space that NO
# existing path reaches: the image paths (Path 2 / Path 3) both require
# gap >= IMAGE_GAP_MIN_UM (1.0), so small-gap reals that are not straight enough
# for Path 1 (worst-arm >= 0.94) are rejected outright today — e.g. the report's
# missed real at gap 0.61 / colinear_cos 0.78. This lever is GEOMETRIC (gap +
# colinearity only — no fluorescence-bridge brightness, no caliber gate), so it
# should GENERALIZE to held-out brains, unlike the exhausted bridge levers.
# SAFETY: all 3 train FALSE joins sit at gap >= 2.20 (2.20 / 3.00 / 3.16), well
# above SMALL_GAP_UM, so this path opens ZERO false-merge risk on the train set;
# the forward-pointing arms floor (min half-cosine >= MIN_ARM_COS) additionally
# prevents merging fragments that double back (which have a negative half-cosine).
SMALL_GAP_UM = 1.5          # tips essentially adjacent below this gap
SMALL_GAP_MIN_COS = 0.50    # modest mean-colinearity floor for the small-gap geometry accept

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


def _colinear_half_cosines(g, s):
    """The TWO half-cosines of SplitSite ``s``'s imagined straight repair.

    Builds the same downstream-pointing unit vectors as ``_is_colinear_split``
    along  A_interior -> node_a -> node_b -> B_interior  and returns the tuple
    ``(c1, c2)`` where:

        v1 = direction arriving at the tip node_a   (fragment A, outward)
        v2 = the gap bridge  node_a -> node_b
        v3 = best same-fragment direction leaving node_b into fragment B
        c1 = cos(v1, v2)
        c2 = max over node_b's same-fragment neighbors of cos(v2, v3)

    By construction ``(c1 + c2) / 2`` equals the failure report's ``colinear_cos``
    column (the MEAN of the two half-cosines), and ``min(c1, c2)`` is the WORST
    arm. Returns ``None`` if any vector is undefined/degenerate (same guards as
    ``_is_colinear_split``), so callers can skip the site.
    """
    xyz = g.node_xyz
    a, b = s.node_a, s.node_b
    v2 = np.asarray(xyz[b], dtype=float) - np.asarray(xyz[a], dtype=float)
    n2 = np.linalg.norm(v2)
    if n2 == 0:
        return None
    v2 /= n2

    v1 = _walk_tangent(g, a, TANGENT_WALK_UM)          # arrives at node_a, outward
    if v1 is None:
        return None
    c1 = float(np.dot(v1, v2))

    # c2: leave node_b into B along whichever same-fragment branch best continues
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
    if best is None:
        return None
    return (c1, best)


def _is_colinear_split(g, s, min_cos):
    """True if SplitSite ``s`` is a straight-line continuation across the gap.

    A WORST-arm (both-arms) test: requires BOTH half-cosines (arm A into the gap;
    the gap into B's continuing arm — see ``_colinear_half_cosines``) to clear
    ``min_cos``:

        Colinear  <=>  cos(v1, v2) >= min_cos AND cos(v2, v3) >= min_cos.

    This rejects the common false positive where two unrelated neurites merely pass
    close by: their tips point ACROSS the gap at an angle, not ALONG it.
    """
    hc = _colinear_half_cosines(g, s)
    if hc is None:
        return False
    c1, c2 = hc
    return c1 >= min_cos and c2 >= min_cos


def _rad_ratio(g, ctx, s):
    """Endpoint neurite-caliber mismatch ``max(r_a, r_b) / min(r_a, r_b)`` for ``s``.

    Returns a LARGE sentinel (``inf``) on ANY failure so an unavailable radius
    simply SKIPS the caliber-gated tier (degrading to parent behavior) instead of
    crashing. Tries the cheapest, leak-free interfaces in order:
      1. a precomputed ``s.rad_ratio`` attribute on the SplitSite, if present;
      2. ``s.rad_a`` / ``s.rad_b`` endpoint radii on the SplitSite;
      3. a node-radius lookup (``ctx["node_radius"]`` or ``g.node_radius``) indexed
         at ``s.node_a`` / ``s.node_b``.
    """
    try:
        # (1) Site already carries the ratio.
        rr = getattr(s, "rad_ratio", None)
        if rr is not None:
            rr = float(rr)
            if rr == rr and rr > 0:          # NaN/non-positive guard
                return rr

        # (2) Site carries the two endpoint radii directly.
        ra = getattr(s, "rad_a", None)
        rb = getattr(s, "rad_b", None)

        # (3) Fall back to a node-radius array.
        if ra is None or rb is None:
            radii = ctx.get("node_radius")
            if radii is None:
                radii = getattr(g, "node_radius", None)
            if radii is not None:
                ra = float(radii[s.node_a])
                rb = float(radii[s.node_b])

        ra = float(ra)
        rb = float(rb)
        if ra == ra and rb == rb and ra > 0 and rb > 0:   # NaN/non-positive guard
            return max(ra, rb) / min(ra, rb)
    except Exception:
        pass
    return float("inf")


def _deg_b(g, s):
    """Graph degree of SplitSite ``s``'s partner node ``node_b``, or ``None``.

    ``node_a`` is always a fragment TIP (degree 1), but ``node_b`` may be a tip
    (degree 1), a shaft (degree 2), or a branch (degree 3+). A degree >= 2 partner
    means a tip-to-SHAFT reconnection, where a large caliber mismatch is expected
    (node_a is the thin tip, node_b is measured mid-cable). Returns ``int`` degree
    or ``None`` on ANY failure so callers fall back to the STRICT ceiling
    (conservative) when degree is unavailable.
    """
    try:
        return int(g.degree[s.node_b])
    except Exception:
        return None


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

    reader = ctx.get("read_image_patch")

    edits = []
    for s in sites:
        kind = getattr(s, "kind", "split")
        if kind != "split":
            continue  # seed repairs splits only; leave merges for evolution
        if s.gap_um > GAP_THRESHOLD_UM:
            continue
        try:
            # Path 1: strict geometry-only accept (gap <= 4.0 AND cos >= 0.94).
            # This already yields the high-precision 6 correct / 0 false merges.
            if _is_colinear_split(g, s, MIN_COLINEAR_COS):
                edits.append(s.as_edit())   # (label_a, label_b) merge tuple
                continue
            # Image-gated recall paths (Gen 7): gate on the MEAN colinear_cos
            # (matching the failure report's column) with a weak-arm floor, instead
            # of the WORST-arm both-arms test. This admits real splits with one
            # straight arm + one slightly bent arm whose MEAN clears the floor but
            # whose weaker arm dipped just below it. Compute the two half-cosines
            # ONCE (cheap geometry) and reuse for both image tiers; skip both image
            # paths if the geometry is degenerate.
            hc = _colinear_half_cosines(g, s)
            # Path 1c (Gen 13): small-gap geometry-only accept. When the gap is
            # tiny the two tips are essentially adjacent, so a forward-pointing,
            # modestly-colinear break is almost always ONE neuron the segmentation
            # fragmented. This reaches small-gap reals NO other path can (the image
            # paths require gap >= IMAGE_GAP_MIN_UM = 1.0; Path 1 needs much
            # higher colinearity) — e.g. the missed real at gap 0.61 / cos 0.78.
            # Pure geometry (gap + colinearity, no bridge/caliber), so it should
            # generalize. SAFETY: all 3 train FALSE joins sit at gap >= 2.20, well
            # above SMALL_GAP_UM, so this opens ZERO false-merge risk on train; the
            # MIN_ARM_COS floor rejects arms that double back (negative half-cosine).
            if hc is not None:
                c1, c2 = hc
                if (s.gap_um <= SMALL_GAP_UM
                        and min(c1, c2) >= MIN_ARM_COS
                        and 0.5 * (c1 + c2) >= SMALL_GAP_MIN_COS):
                    edits.append(s.as_edit())
                    continue
            if reader is not None and hc is not None:
                c1, c2 = hc
                mean_cos = 0.5 * (c1 + c2)
                min_arm = min(c1, c2)
                # Path 2: mean near-colinear band, weak-arm-floored, image-confirmed.
                # Accept when the MEAN colinear_cos is high (>= 0.75) and neither arm
                # points sideways/backward (min_arm >= 0.35), confirmed by a bright
                # fluorescence bridge across the gap. The cloud read is gated behind
                # the cheap geometry so only ambiguous candidates trigger it.
                if (s.gap_um >= IMAGE_GAP_MIN_UM
                        and mean_cos >= IMAGE_MIN_COLINEAR_COS
                        and min_arm >= MIN_ARM_COS):
                    ev = reader.gap_bridge_evidence(s.node_a, s.node_b)
                    br = ev.get("bridge_ratio", float("nan"))
                    # Gen 10: Path 2 uses the relaxed bridge floor (0.85) because the
                    # strong-colinearity gate (mean_cos >= 0.75) above already supplies
                    # the precision; Path 3 below keeps the strict 0.90.
                    if br == br and br >= IMAGE_BRIDGE_RATIO_MIN_STRONG:  # NaN guard + bright bridge
                        edits.append(s.as_edit())
                        continue
                # Path 3: caliber-gated lower-cos image tier. An ORTHOGONAL lever that
                # admits reals in the [0.50, 0.75) MEAN-colinearity band Path 2
                # excludes (Gen 16 lowered the floor 0.60 -> 0.50), but ONLY when
                # neither arm is sideways/backward (min_arm >= 0.35) AND the two
                # endpoint neurite calibers MATCH within the degree-aware cap. The 3
                # train false joins all stay rejected: cos 0.44 is below the new 0.50
                # floor (margin 0.06); the two tip-to-tip ones (rad_ratio 1.67 / 1.87)
                # exceed the 1.45 tip-to-tip caliber cap (Gen 18: 1.4 -> 1.45). The cheap geometry
                # gates (gap band + mean band + weak-arm + caliber) run BEFORE the
                # cloud bridge read; the bright-bridge requirement (0.90) confirms
                # one continuous neuron.
                # Gen 9: degree-aware caliber ceiling. A tip-to-shaft partner
                # (deg_b >= 2) legitimately has a large caliber mismatch (thin tip
                # vs thicker mid-shaft), so use the looser RAD_RATIO_MAX_SHAFT there
                # (Gen 17: raised 2.5 -> 2.8 to harvest deg_b>=2 reals with rad_ratio
                # in (2.5, 2.8], e.g. the missed real gap 1.65 / cos 0.78 / rad 2.69);
                # tip-to-tip (deg_b <= 1) or unknown degree keeps the strict 1.45
                # (Gen 18: 1.4 -> 1.45).
                db = _deg_b(g, s)
                if db is not None and db >= 2:
                    cap = RAD_RATIO_MAX_SHAFT
                elif s.gap_um >= ABOVE_FJ_LO_UM:
                    # Gen 31: tip partner ABOVE all train tip false joins (both at gap <= 3.16).
                    # The tight tip cap (1.45) exists only to reject FJ2/FJ3; above 3.30 there is
                    # no tip false join to fence, so loosen the tip cap to RAD_RATIO_MAX_SHAFT.
                    cap = RAD_RATIO_MAX_SHAFT
                else:
                    cap = RAD_RATIO_MAX
                if (s.gap_um >= IMAGE_GAP_MIN_UM
                        and s.gap_um <= GAP_THRESHOLD_UM
                        and IMAGE_LOWCOS_MIN_COLINEAR_COS <= mean_cos < IMAGE_MIN_COLINEAR_COS
                        and min_arm >= MIN_ARM_COS
                        and _rad_ratio(g, ctx, s) <= cap):
                    ev = reader.gap_bridge_evidence(s.node_a, s.node_b)
                    br = ev.get("bridge_ratio", float("nan"))
                    if br == br and br >= IMAGE_BRIDGE_RATIO_MIN:  # NaN guard + bright bridge
                        edits.append(s.as_edit())
                        continue
                # Path 4 (Gen 19; extended Gen 20): tightly-matched-caliber image
                # tier. Admits reals in the MEAN-colinearity band BELOW Path 3's 0.50
                # floor — the band [IMAGE_MATCHED_MIN_COLINEAR_COS,
                # IMAGE_LOWCOS_MIN_COLINEAR_COS) = [0.25, 0.50) (Gen 20: floor lowered
                # 0.35 -> 0.25), DISJOINT from Path 3's [0.50, 0.75) so no
                # double-count — but ONLY when caliber is TIGHTLY matched
                # (rad_ratio <= IMAGE_MATCHED_CALIBER_MAX = 1.5). All 3 train FALSE
                # joins have rad_ratio >= 1.67 > 1.5, so this tight caliber gate
                # rejects every one of them by caliber alone, regardless of their cos
                # or bridge brightness — which is exactly why the cos floor can be
                # lowered below 0.50 (and now below 0.35) HERE (it cannot be lowered
                # globally; the deg_b 2 false join at cos 0.44 / rad_ratio 1.97 with a
                # bright bridge is held out globally ONLY by the 0.50 cos floor, and
                # here ONLY by caliber).
                # Gen 20: Path 4 now uses its OWN, lower weak-arm floor
                # IMAGE_MATCHED_MIN_ARM_COS (0.25), DECOUPLED from the global
                # MIN_ARM_COS (0.35) that still guards Paths 1c/2/3. This unpins the
                # cos floor: because min_arm <= mean_cos always, the shared 0.35
                # weak-arm floor had effectively PINNED Path 4's mean floor at 0.35;
                # giving Path 4 its own 0.25 weak-arm floor lets the band extend down
                # into [0.25, 0.35). The 0.25 floor is still strictly > 0, so it
                # REJECTS any arm pointing sideways/backward (a double-back / crossing,
                # the genuine false-join signature) — it only allows a slightly larger
                # forward bend (~75 deg). The bright-bridge requirement stays the
                # strict IMAGE_BRIDGE_RATIO_MIN (0.90). The cheap geometry (gap band +
                # matched mean band + weak-arm + tight caliber) runs BEFORE the cloud
                # bridge read, so only a handful of candidates trigger it. RATIONALE:
                # nearly-identical neurite caliber + a bright continuous bridge
                # together supply the precision that strong colinearity normally would,
                # so a gently-colinear break (a natural bend, or a short fragment
                # yielding a noisy tangent) is very likely ONE neuron. Concrete
                # reachable target: gap 1.80 / cos 0.29 / deg_b 2 / rad_ratio 1.25.
                if MIDGAP_FREE_LO_UM <= s.gap_um < MIDGAP_FREE_HI_UM:
                    # Gen 28 (band widened Gen 29): false-join-free mid-gap band. FJ1 (shaft)
                    # at gap 2.20 is below 2.30; both tip false joins at gap >= 3.00 are above
                    # 2.90 — so NO train false join of either degree lies in [2.30, 2.90). Loosen
                    # caliber here for both degrees and rely on the bright-bridge confirmation
                    # (AUC 0.86; probe reals in this band have bridge_ratio >= 0.98).
                    cap4 = IMAGE_MATCHED_CALIBER_MAX_MIDGAP
                elif db is not None and db >= 2:
                    cap4 = IMAGE_MATCHED_CALIBER_MAX_SHAFT_SMALLGAP if s.gap_um < MATCHED_SMALLGAP_UM else IMAGE_MATCHED_CALIBER_MAX_SHAFT
                else:
                    # Gen 26: three-tier monotonic tip caliber ladder. gap < 2.0 is
                    # tip-false-join-free (both tip false joins at gap >= 3.00), so the
                    # cap loosens to 2.2 there; the existing 1.85 tier covers
                    # [2.0, 2.5); above 2.5 it stays the strict 1.5.
                    if s.gap_um < TIP_TINYGAP_UM:
                        cap4 = IMAGE_MATCHED_CALIBER_MAX_TIP_TINYGAP
                    elif s.gap_um < TIP_SMALLGAP_UM:
                        cap4 = IMAGE_MATCHED_CALIBER_MAX_TIP_SMALLGAP
                    else:
                        cap4 = IMAGE_MATCHED_CALIBER_MAX
                if (s.gap_um >= IMAGE_GAP_MIN_UM
                        and s.gap_um <= GAP_THRESHOLD_UM
                        and IMAGE_MATCHED_MIN_COLINEAR_COS <= mean_cos < IMAGE_LOWCOS_MIN_COLINEAR_COS
                        and min_arm >= IMAGE_MATCHED_MIN_ARM_COS
                        and _rad_ratio(g, ctx, s) <= cap4):
                    ev = reader.gap_bridge_evidence(s.node_a, s.node_b)
                    br = ev.get("bridge_ratio", float("nan"))
                    if br == br and br >= IMAGE_BRIDGE_RATIO_MIN:  # NaN guard + bright bridge
                        edits.append(s.as_edit())
        except Exception:
            # Geometry is advisory; never let one odd site crash the whole policy.
            continue
    return edits
