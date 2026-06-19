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

# --- Proximity-override accept pathway (Gen 10) ------------------------------
# The gen3 `_required_cos` floor rejects any SplitSite whose worst-case colinearity
# is below the band threshold, EVEN at sub-micron gaps. But two fragment ends that
# are NEARLY TOUCHING are almost certainly ONE neuron split by a labeling gap: a
# real neurite can have a genuine bend exactly at the break, which the cosine floor
# wrongly rejects. Add a SECOND, parallel (OR) acceptance pathway: accept a
# SplitSite when its gap is below PROXIMITY_UM regardless of local angle. This is an
# ORTHOGONAL axis (gap proximity) to the gen3 colinearity floor, purely ADDITIVE —
# it only ADDS accepts in a region gen3 currently rejects, never removes a gen3
# accept. Guardrails to keep it gate-safe against created merges:
#   - PROXIMITY_UM is sub-micron-tight: at <0.75 µm, two DISTINCT neurons touching
#     is far rarer than one neuron split by a gap (stated assumption; evidence on
#     these skeletons is weak — see change log).
#   - PROXIMITY_TIP_ONLY: require the PARTNER (node_b) to be a tip (degree <= 1) so
#     the override fires only for clean tip-to-tip near-touches, NOT tip-into-shaft
#     (the topology tied to the one persistent created merge in the gen10 report).
PROXIMITY_UM = 0.75         # gap below which a SplitSite is accepted regardless of angle (µm)
PROXIMITY_TIP_ONLY = True   # only override when the partner node is itself a tip (degree <= 1)


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
    high-precision far-gap edits — a net change to the proposed edit set, not just a
    filter. MergeSites are still left alone (no ``split_label``).

    Gen 10 adds a SECOND, parallel (OR) acceptance pathway orthogonal to the
    colinearity floor: a SplitSite whose gap is below ``PROXIMITY_UM`` is accepted
    regardless of its local angle (optionally only when the partner is a tip,
    ``PROXIMITY_TIP_ONLY``). Near-touching ends are almost certainly one neuron split
    by a labeling gap, where a genuine bend at the break would wrongly fail the
    cosine floor. This is purely ADDITIVE — it only adds accepts in a region gen3
    rejects, never removes a gen3 accept.

    Returns a list of edits — legacy (label_a, label_b) tuples or typed dicts; see
    the module docstring. An empty list means "make no changes".
    """
    g = ctx.get("fragments_graph")
    if g is None:
        return []

    edits = []
    for s in sites:
        kind = getattr(s, "kind", "split")
        if kind != "split":
            continue  # repairs splits only; leave merges for a later generation
        # --- Proximity-override pathway (Gen 10): orthogonal, additive ---------
        # Near-touching fragment ends are almost certainly one neuron split by a
        # labeling gap. Accept regardless of angle when the gap is sub-micron-tight,
        # optionally requiring the partner to be a tip (clean tip-to-tip touch) to
        # avoid the tip-into-shaft topology tied to the persistent created merge.
        if s.gap_um <= PROXIMITY_UM:
            if not PROXIMITY_TIP_ONLY:
                edits.append(s.as_edit())
                continue
            try:
                partner_is_tip = g.degree[s.node_b] <= 1
            except Exception:
                partner_is_tip = False
            if partner_is_tip:
                edits.append(s.as_edit())
                continue

        floor = _required_cos(s.gap_um)
        if floor is None:
            continue                      # gap beyond bridgeable reach
        try:
            score = _colinearity_score(g, s)
        except Exception:
            # Geometry is advisory; never let one odd site crash the whole policy.
            continue
        if score is not None and score >= floor:
            edits.append(s.as_edit())     # (label_a, label_b) merge tuple
    return edits
