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

# --- Adaptive gap/angle accept boundary (Gen 1) ------------------------------
# The seed used a single rectangular accept box (gap <= 4 µm AND cos >= 0.94).
# That box has two costs visible in the gen03 report: (a) it ADMITS marginal,
# barely-colinear pairs at moderate gaps that nudge %Split Edges up (+0.16 mean)
# and even fused two different neurons once (a created merge); and (b) it MISSES
# longer true gaps whose fragments are almost perfectly straight — the very pairs
# whose repair drives the big %Omit-Edges drop that is carrying the fitness gain.
# Replace the rectangle with a sloped boundary: the straighter the continuation,
# the larger a gap we will bridge; the closer the gap, the more bend we tolerate.
#   - At any gap up to GAP_THRESHOLD_UM, require at least MIN_COLINEAR_COS (as before
#     for the near field) but RAISE the bar for the marginal mid-gap band.
#   - Beyond GAP_THRESHOLD_UM, up to GAP_FAR_UM, accept ONLY near-perfect colinearity
#     (>= COS_FAR), surfacing extra high-precision omit-reducing merges the box missed.
GAP_FAR_UM = 8.0            # extended reach for near-perfectly-straight continuations (µm)
COS_NEAR = 0.97            # tighter floor in the (2µm, GAP_THRESHOLD] band to drop marginal pairs
COS_FAR = 0.992            # near-straight requirement for the (GAP_THRESHOLD, GAP_FAR] band
GAP_TIGHT_UM = 2.0          # very-near band keeps the original lenient MIN_COLINEAR_COS floor


def _required_cos(gap_um):
    """Distance-dependent colinearity floor: straighter required as the gap grows.

    Returns None if ``gap_um`` is beyond the maximum bridgeable distance (reject).
    """
    if gap_um <= GAP_TIGHT_UM:
        return MIN_COLINEAR_COS          # very near: original lenient floor
    if gap_um <= GAP_THRESHOLD_UM:
        return COS_NEAR                  # mid band: tighter, drops over-split/merge-prone pairs
    if gap_um <= GAP_FAR_UM:
        return COS_FAR                   # far band: only near-perfect continuations
    return None                          # too far to bridge


# --- Merge-repair (split_label) acceptance band (Gen 2 — the MergeSite lever) -
# Gen03-gen06 all lived on the SplitSite->merge_labels acceptance boundary and all
# scored +0.000 on held-out: that lever is saturated (the candidate set is bimodal,
# no boundary tweak flips a held-out edge). The residual error the policy has NEVER
# touched is the MERGE component (%Merged Edges ~10.2%, #Merges ~9.05): every prior
# generation ignored ALL MergeSites. This generation emits the policy's FIRST-EVER
# split_label edits, attacking that component directly.
#
# Which MergeSites are safe to cut?  Reading the gen07 report's two labelled tables
# (TRUE merges vs NON-merges), the cleanest separable positive cluster is the
# "bridge" detector — a thin degree-2 neck with a MODERATE kink where two distinct
# neurites were fused end-to-end:
#   * TRUE-merge bridges:  angle_deg in [84, 118], radius_ratio in [1.00, 1.03],
#                          both arm cables ~28-30 µm (symmetric).
#   * NON-merge branches (the over-split danger) cluster at angle EXTREMES (very
#     sharp <50° OR near-straight >150°) and/or radius_ratio far from 1 (up to 1.48),
#     and the "component" rows are huge (hundreds-thousands µm) — all OUTSIDE the
#     window below.
# So a high-precision split_label rule fires ONLY on a "bridge" site whose kink sits
# in the moderate band AND whose two arms have nearly equal caliber (radius continuity
# argues two same-thickness neurites butted together, not one tapering cable) AND
# whose arms do NOT reconverge downstream (arms_reconverge is not True). The branch /
# component detectors are deliberately EXCLUDED here: branch true/false overlap too
# much to cut safely, and every "component" row in the report is a non-merge. This is
# intentionally narrow — a wrong split raises %Split Edges and reverts the whole
# generation, so precision dominates recall for this first attempt.
MERGE_ANGLE_LO = 95.0     # kink must be at least this open (excludes very-sharp non-merges)
MERGE_ANGLE_HI = 120.0    # ...and no straighter than this (excludes pass-through non-merges)
MERGE_MAX_RADIUS_RATIO = 1.05   # arms must be near-equal caliber (true bridges <=1.03)
MERGE_MIN_ARM_CABLE_UM = 20.0   # both fused neurites must carry real cable
# When the raw image is available, confirm the cut with an intensity-valley test
# along the chord between the two arm seeds: a true merge of two touching neurites
# shows a DIP (valley_ratio well below 1) near the contact, while one continuous
# neuron stays bright (valley_ratio ~1). Guarded so the policy still runs image-off.
MERGE_MAX_VALLEY_RATIO = 0.80   # require a real intensity dip to confirm a cut


def _accept_merge_site(s, ctx):
    """High-precision gate for emitting a split_label on a MergeSite ``s``.

    Returns True only for a "bridge"-detector site whose geometry matches the TRUE
    merge cluster in the failure report (moderate kink, equal caliber, long arms,
    no downstream reconvergence). When a raw-image reader is present it additionally
    requires an intensity valley across the cut chord (the most direct merge
    evidence). All feature reads are guarded so a missing field never crashes.
    """
    if getattr(s, "detector", None) != "bridge":
        return False  # branch/component detectors overlap with non-merges -> unsafe

    # arms_reconverge True == one neuron's own branches / a loop -> never cut.
    if getattr(s, "arms_reconverge", None) is True:
        return False

    cable_a = getattr(s, "cable_a_um", 0.0) or 0.0
    cable_b = getattr(s, "cable_b_um", 0.0) or 0.0
    if cable_a < MERGE_MIN_ARM_CABLE_UM or cable_b < MERGE_MIN_ARM_CABLE_UM:
        return False  # at least one side is a spur, not a fused neurite

    angle = getattr(s, "angle_deg", float("nan"))
    if angle is None or not np.isfinite(angle):
        return False  # no defined kink (e.g. component detector) -> do not cut
    if not (MERGE_ANGLE_LO <= angle <= MERGE_ANGLE_HI):
        return False  # too sharp or too straight -> matches the non-merge extremes

    rr = getattr(s, "radius_ratio", None)
    # radius_ratio is the caliber-continuity check; if unavailable, fall back to the
    # geometry-only decision rather than rejecting (conservative either way).
    if rr is not None and rr > MERGE_MAX_RADIUS_RATIO:
        return False

    # Optional image confirmation: only fetch for a site that already passed every
    # cheap geometric filter (cost rule), and only when the reader exists.
    reader = ctx.get("read_image_patch")
    if reader is not None:
        try:
            ev = reader.merge_cut_evidence(s.seed_a_node, s.seed_b_node)
            vr = ev.get("valley_ratio") if isinstance(ev, dict) else None
            if vr is not None and vr > MERGE_MAX_VALLEY_RATIO:
                return False  # bright continuous signal -> one neuron -> do not cut
        except Exception:
            pass  # image evidence is advisory; geometry already passed

    return True

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


def _colinearity_score(g, s):
    """Worst-case colinearity of SplitSite ``s`` across the gap (in [-1, 1], or None).

    Builds three downstream-pointing unit vectors along the imagined repaired
    neuron  A_interior -> node_a -> node_b -> B_interior:

        v1 = direction arriving at the tip node_a   (fragment A, outward)
        v2 = the gap bridge  node_a -> node_b
        v3 = direction leaving node_b into fragment B (continuing the line)

    Returns ``min(cos(v1, v2), cos(v2, v3))`` — the limiting (worst) bend along the
    repaired path. A high value means BOTH the approach and the departure stay on
    the straight line through the gap; a low value flags the common false positive
    where two unrelated neurites merely pass close by, pointing ACROSS the gap. The
    caller compares this single scalar against a distance-dependent floor, so the
    accept boundary can slope with the gap. Returns None if a direction is
    undefined (degenerate / too-short fragment).
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
    cos_in = float(np.dot(v1, v2))

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
    if best is None:
        return None
    return min(cos_in, best)


def propose_edits(sites, ctx) -> list:
    """Decide which candidate sites to repair, returning a list of edits.

    POLICY (Gen 1 — adaptive split-repair): for each SplitSite, emit a
    ``merge_labels`` edit when the measured colinearity across the gap clears a
    DISTANCE-DEPENDENT floor (see ``_required_cos``):
      - very near (gap <= GAP_TIGHT_UM): lenient floor MIN_COLINEAR_COS,
      - mid band (GAP_TIGHT_UM < gap <= GAP_THRESHOLD_UM): tighter floor COS_NEAR
        to drop the marginal pairs that nudged %Split Edges up and created a merge,
      - far band (GAP_THRESHOLD_UM < gap <= GAP_FAR_UM): near-straight floor COS_FAR
        to bridge longer TRUE gaps (extra omit-reducing merges the box missed),
      - beyond GAP_FAR_UM: reject.
    This replaces the seed's rectangular accept box (gap<=4 AND cos>=0.94) with a
    sloped boundary, simultaneously REMOVING marginal mid-gap edits and ADDING
    high-precision far-gap edits — a net change to the proposed edit set.

    POLICY (Gen 2 — adds the FIRST split_label edits): in ADDITION to the SplitSite
    repairs above, each MergeSite is now passed to ``_accept_merge_site`` and, when it
    matches the report's TRUE-merge "bridge" cluster (moderate kink, equal caliber,
    long non-reconverging arms, optionally image-valley-confirmed), a ``split_label``
    edit is emitted to break the fusion. This attacks the %Merged-Edges component that
    every prior generation left untouched, while staying conservative enough not to
    over-split a real neuron.

    Returns a list of edits — legacy (label_a, label_b) tuples or typed dicts; see
    the module docstring. An empty list means "make no changes".
    """
    g = ctx.get("fragments_graph")
    if g is None:
        return []

    edits = []
    for s in sites:
        kind = getattr(s, "kind", "split")
        if kind == "split":
            floor = _required_cos(s.gap_um)
            if floor is None:
                continue                  # gap beyond bridgeable reach
            try:
                score = _colinearity_score(g, s)
            except Exception:
                # Geometry is advisory; never let one odd site crash the policy.
                continue
            if score is not None and score >= floor:
                edits.append(s.as_edit())     # (label_a, label_b) merge tuple
        elif kind == "merge":
            try:
                if _accept_merge_site(s, ctx):
                    edits.append(s.as_edit())  # split_label dict (per-neurite cut)
            except Exception:
                # A single odd MergeSite must never crash the whole policy.
                continue
    return edits
