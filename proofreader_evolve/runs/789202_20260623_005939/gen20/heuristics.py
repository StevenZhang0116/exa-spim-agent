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
GAP_THRESHOLD_UM = 6.5      # max tip→partner distance (µm); covers ~99% of true split gaps, inter-neuron gaps rarely fall below ~7 µm
MIN_COLINEAR_COS = 0.60     # FAR-regime colinearity floor, lowered from 0.85 to 0.60 to recover mid-gap real splits with clean (cos 0.60-0.85) continuation while staying well above the reachable false-join population (all cos <= 0.44, ~0.16 margin)
MICRO_GAP_UM = 2.0          # MICRO regime: at/below this gap accept on DISTANCE ALONE, waiving colinearity entirely. Set 0.2 µm below the nearest reachable FALSE join (smallest at 2.20 µm), so every reachable false join stays out of band. Grounded: inter-neuron gaps rarely fall below ~7 µm (AUC 0.998), so a sub-2 µm gap is overwhelmingly one broken neuron; tip-tangent colinearity at lattice-scale gaps (√1..√4 = 1.0,1.41,1.73,2.0 voxels) is dominated by skeletonization jitter and is uninformative there.
TIP_TIP_EXT_GAP_UM = 2.5    # EXTENDED TIP-TO-TIP regime ceiling: for the dense band of MISSED tip-to-tip reals just ABOVE TIP_TIP_GAP_UM (a cluster at gap 2.45, all deg_a==deg_b==1), accept when ALSO caliber-matched (see TIP_TIP_RAD_MATCH). 2.5 keeps the tip-to-tip FALSE at gap 3.00 out of band; only the tip-to-tip FALSE at gap 2.45 is in-band, and IT is excluded by the caliber ceiling. (The 2.20 FALSE is deg_b==2, already excluded by the tip-to-tip degree requirement.)
TIP_TIP_RAD_MATCH = 1.50    # CABLE-CALIBER ceiling for the EXTENDED tip-to-tip band. The in-band tip-to-tip FALSE at gap 2.45 has rad_ratio 1.56; the captured tip-to-tip reals in this band have rad_ratio <= 1.46. 1.50 sits 0.06 BELOW the FALSE and 0.04 ABOVE the captured reals — a topology-specific LOOSER ceiling than the general colinearity-tier path's 1.30, justified because deg_a==deg_b==1 already excludes the high-caliber deg_b==2 FALSE at gap 2.20. Relative caliber only (no absolute radius), so brain-agnostic.
TIP_TIP_GAP_UM = 2.3        # TIP-TO-TIP regime: between MICRO_GAP_UM and this, accept on DISTANCE + TOPOLOGY alone (both endpoints tips, deg_a==deg_b==1), waiving colinearity. Two genuine termini at a sub-2.3 µm gap is the canonical signature of ONE neuron broken into two fragments. Set 0.15 µm BELOW the nearest TIP-TO-TIP false join (2.45 µm) so BOTH tip-to-tip false joins (2.45, 3.00 µm) stay out of band; the false join at 2.20 µm is tip-INTO-SHAFT (deg_b=2) and is excluded by the tip-to-tip requirement (it falls through to the FAR colinearity floor, cos 0.44 < 0.60 → rejected). At these lattice-scale gaps (the dense missed-real cluster sits at √5 = 2.24 µm) the tip-tangent angle is skeletonization jitter (uninformative), exactly as in the MICRO tier; distance + tip-tip topology decide instead.
TIP_SHAFT_GAP_UM = 2.75     # DEDICATED gap ceiling for the caliber-gated TIP-INTO-SHAFT tier, DECOUPLED from the colinearity-tier's RAD_MATCH_GAP_UM (2.5). The ONLY reachable deg_b==2 FALSE join sits at gap 2.20 (excluded by the 1.50 caliber ceiling, rad 1.97 > 1.50); the next reachable FALSE join is at gap 3.00 but it is deg_b==1, so this deg_b==2 tier NEVER sees it — i.e. there is a FALSE-free gap window for caliber-matched tip-into-shaft from ~2.2 up to 3.0. 2.75 captures the dense matched-caliber missed-real cluster at gap 2.5-2.72 while staying conservatively below 3.0 (held-out population unseen near the deg_b==1 FALSE @3.00). Relative caliber only, so brain-agnostic; grounded on the tight-split-gap scale (a sub-2.75 µm gap with matched caliber is overwhelmingly one broken neuron).
TIP_SHAFT_EXT_GAP_UM = 3.0  # EXTENDED gap ceiling for the deg_b==2 (tip-into-shaft) topology ONLY, extending acceptance one step beyond TIP_SHAFT_GAP_UM into the band (2.75, 3.0]. The three existing deg_b==2 tiers (matched-caliber rad<=1.50, mid-caliber 1.50<rad<2.0 gated above gap 2.40, high-mismatch rad>=2.0) together accept essentially ALL tip-into-shaft sites at gap <= 2.75, but ALL cap at that ceiling; this leaves the dominant remaining missed cluster — deg_b==2 reals densely packed at gap 2.76-2.98 across the full rad_ratio range — uncaptured ONLY because gap > 2.75. FALSE-free with a WIDE margin: the FALSE @gap 2.20 is the ONLY reachable deg_b==2 FALSE join at ANY gap, and the new floor (2.75) sits 0.55 µm above it; there is NO deg_b==2 FALSE anywhere in (2.75, 3.0]. The other two reachable FALSE joins (incl. the one @gap 3.00) are deg_b==1, NEVER seen by this deg_b==2 tier — so the gap-3.00 FALSE is irrelevant here, which is also why deg_b==1 is deliberately NOT extended. The gap prior does the heavy lifting (findings #1, #21): at gap 2.75-3.0 << 7 µm a tip-into-shaft site is overwhelmingly a REAL break (gap-distance AUC 0.998, F1-optimal 6.84 µm; ~99% of true split gaps under ~6.5 µm), so even held-out a deg_b==2 FALSE in this window is extraordinarily rare. Relative caliber only, so brain-agnostic.
TIP_SHAFT_RAD_MATCH = 1.50  # CABLE-CALIBER ceiling for the caliber-gated TIP-INTO-SHAFT band (deg_a==1, deg_b==2, MICRO_GAP_UM < gap <= TIP_SHAFT_GAP_UM). The missed-real bucket is now dominated by tip-into-shaft sites at gap 2.0-2.65 that fail every existing path (gap > TIP_TIP_GAP_UM; deg_b==2 so both tip-to-tip tiers skip; low/negative colinear_cos fails the 0.60/0.55 floors; rad_ratio often just above the colinearity-tier 1.30 caliber cap). The ONLY reachable deg_b==2 FALSE join sits at gap 2.20 / colinear_cos 0.44 / rad_ratio 1.97 — its colinearity is MISLEADINGLY HIGH (straighter than ~half the missed reals), so colinearity cannot recover these reals without admitting it, but its caliber is clearly MISMATCHED. So in this band CALIBER beats colinearity: 1.50 sits 0.47 BELOW that FALSE (a wide margin) while admitting the matched-caliber missed reals at rad_ratio <= ~1.49. The two other reachable FALSE joins are deg_b==1, so this deg_b==2 tier never sees them. Relative caliber only (no absolute radius), so brain-agnostic. The lattice-jitter rationale that justifies WAIVING colinearity here is topology-independent (tip-arm jitter corrupts the direction regardless of node_b's degree), exactly as in the MICRO / TIP-TO-TIP tiers.
TIP_TIP_HIGH_RAD_GAP_UM = 2.9  # GAP CEILING for the HIGH-rad-ratio TIP-TO-TIP recall tier (deg_a==1, deg_b==1, rad_ratio >= TIP_SHAFT_HIGH_RAD_RATIO = 2.0). This is the tip-to-tip analogue of the gen13 tip-into-shaft caliber FLOOR. The matched-caliber EXTENDED tip-to-tip tier (rad_ratio <= TIP_TIP_RAD_MATCH = 1.50) captures the symmetric-caliber tip-to-tip reals; this captures the HIGH-mismatch ones — a thin distal end (~0.75-1.0) meeting a thick end (~1.5-2.4), rad_ratio >= 2.0 — which the gen14 report shows are dense in the missed bucket at gaps 2.45-2.83 (e.g. (2.45,0.04,2.00), (2.45,-0.34,2.41), (2.83,-0.11,2.00), (2.83,0.52,2.00)). The gap prior does the heavy lifting (at gap <= 2.9 << 7 µm a site is overwhelmingly a REAL break by population statistics, AUC 0.998; ~99% of true split gaps fall under ~6.5 µm); the rad_ratio >= 2.0 FLOOR is the cheap precision guard. ZERO new train-false merges with DOUBLE protection: both reachable deg_b==1 FALSE joins have rad_ratio < 2.0 (1.56 and 1.67), so the 2.0 floor alone excludes them (margin >= 0.33); AND the only deg_b==1 FALSE near this window (@gap 3.00) is also beyond the 2.9 gap ceiling. The deg_b==2 FALSE (@gap 2.20) is never seen by this deg_b==1 tier. Relative caliber only (no absolute radius), so brain-agnostic.
DEG2_MID_GAP_FLOOR = 2.40   # GAP FLOOR re-opening the previously carved-out MID-CALIBER band (TIP_SHAFT_RAD_MATCH < rad_ratio < TIP_SHAFT_HIGH_RAD_RATIO, i.e. 1.50 < rad < 2.0) for TIP-INTO-SHAFT (deg_a==1, deg_b==2). The matched-caliber tier (rad <= 1.50) and the gen13 high-mismatch tier (rad >= 2.0) deliberately left this middle band a NO-GO zone for ONE reason only: the FALSE join @gap 2.20 has rad_ratio 1.97, which sits in this band. But that FALSE is the ONLY reachable deg_b==2 FALSE join AT ANY GAP (the other two reachable FALSE are deg_b==1, never seen by a deg_b==2 tier). So the band is safe to re-fill PROVIDED we gate on gap ABOVE that single FALSE: a 2.40 floor sits 0.20 µm above the FALSE @2.20, so no reachable deg_b==2 FALSE can enter — ZERO new train-false merges. The floor intentionally sacrifices a few mid-caliber reals at gap 2.21-2.35 (too close to the FALSE @2.20 to separate safely) for a wide, generalizable margin. The gap prior does the heavy lifting: at gap 2.40-2.75 << 7 µm a tip-into-shaft site is overwhelmingly a REAL break by population statistics (gap-distance AUC 0.998; true split gaps ~99% under ~6.5 µm), so a deg_b==2 FALSE in this gap+caliber window is extraordinarily rare even held-out. Relative caliber only (no absolute radius), so brain-agnostic.
DEG2_LOWMID_RAD_CEIL = 1.60  # caliber CEILING for the low-mid-caliber sub-floor tip-into-shaft band (deg_a==1, deg_b==2, MICRO_GAP_UM < gap < DEG2_MID_GAP_FLOOR=2.40). Re-opens the LOW-caliber slice of the (1.50, 2.0) carved-out band BELOW the 2.40 floor. The captured reals top out at rad_ratio ~1.57; the only reachable deg_b==2 FALSE (@gap 2.20) has rad_ratio 1.97, so a 1.60 ceiling sits 0.37 BELOW the FALSE — a WIDE caliber margin that should hold on held-out too. Relative caliber only (max/min endpoint radius), so brain-agnostic. The protective slice (1.60, 2.0) at gap < 2.40 stays a NO-GO zone guarding the FALSE @1.97.
TIP_SHAFT_HIGH_RAD_RATIO = 2.0  # CABLE-CALIBER FLOOR for the HIGH-mismatch tip-into-shaft tier (deg_a==1, deg_b==2, MICRO_GAP_UM < gap <= TIP_SHAFT_GAP_UM). This is the INVERSE of the matched-caliber ceiling tiers (a rad_ratio FLOOR, not a ceiling). deg_b==2 means a tip joining the MIDDLE of a cable = a branch/T-junction: a real T-junction is a THIN daughter neurite emanating from a THICK parent shaft, so it is naturally VERY mismatched in caliber (high rad_ratio). The matched-caliber tier (TIP_SHAFT_RAD_MATCH = 1.50) captures the symmetric-caliber tip-into-shaft reals; this tier captures the high-caliber-mismatch ones (thin tip ~0.75-1.06 onto thick shaft ~1.5-2.0, rad_ratio >= 2.0, up to ~2.75). The gap prior does the heavy lifting (at gap <= 2.75 << 7 µm a site is overwhelmingly a REAL break by population statistics); the rad_ratio >= 2.0 FLOOR is the cheap precision guard that keeps the 3 known FALSE joins out. ZERO new train-false merges: all 3 reachable FALSE joins have rad_ratio < 2.0 (1.97, 1.56, 1.67); the only deg_b==2 FALSE (@gap 2.20) sits at 1.97, so a 2.0 floor leaves a 0.03 margin above it, and the (1.50, 2.00) caliber band is deliberately left as a no-go zone protecting exactly that FALSE. The other two FALSE are deg_b==1, so this deg_b==2 tier never sees them. Relative caliber only (no absolute radius), so brain-agnostic.
NEAR_GAP_UM = 1.5           # at/below this gap, distance alone is decisive; only screen out anti-parallel joins
NEAR_COLINEAR_COS = 0.0     # near-regime floor: reject only joins that double back (obtuse / anti-parallel), accept everything else
TANGENT_WALK_UM = 6.0       # how far to walk into each fragment to estimate its tip tangent
HARNESS_TANGENT_WALK_UM = 4.0  # short walk that FAITHFULLY mirrors the harness's colinear_cos arm walk (candidate.py::_arm_inner_node, walk_um=4.0); used only by _avg_cross_gap_colinearity
COLINEAR_AVG_ACCEPT = 0.55  # FAR-regime ADDITIVE accept floor on the harness-faithful AVERAGED cross-gap colinearity = 0.5*(cos_a + cos_b). Set 0.11 above the largest reachable FALSE-join colinear_cos (0.44), so every reachable false join (cos 0.44/0.19/0.06) stays rejected while missed reals with a slightly-bent tip arm but a straight partner (avg >= 0.55) are recovered
RAD_RATIO_MATCH = 1.30      # CABLE-CALIBER ADDITIVE accept ceiling on the endpoint radius ratio max(ra,rb)/min(ra,rb). A real "one neuron broken in two" has nearly-equal endpoint calibers (ratio ~1.0-1.3); a false join fuses two unrelated neurites of DIFFERENT caliber. All 3 reachable FALSE joins have rad_ratio >= 1.56 (1.97, 1.56, 1.67), so 1.30 sits ~0.26 BELOW the smallest false-join ratio — no reachable false join can enter this path. Relative (not absolute) caliber, so it is brain-agnostic.
RAD_MATCH_GAP_UM = 2.5      # tight-gap ceiling for the caliber-matched recall path: only admit caliber-matched joins at gaps this tight, where the gap itself already strongly implies one broken neuron (inter-neuron gaps rarely fall below ~7 µm)

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


def _avg_cross_gap_colinearity(g, s):
    """Harness-faithful AVERAGED cross-gap colinearity for SplitSite ``s``.

    Reproduces the validated ``colinear_cos`` measure the harness reports
    (candidate.py::_split_site_geom):  with gdir = unit(xyz[node_b] - xyz[node_a]),
    cos_a = dot(outward tangent of arm A, gdir) using a SHORT 4.0 µm arm walk, and
    cos_b = the best dot of gdir with arm B's gap-continuing same-segment neighbor
    direction; the measure is the MEAN  0.5 * (cos_a + cos_b).

    Unlike ``_is_colinear_split`` (which gates cos_a as a SEPARATE hard floor and
    walks 6.0 µm), this returns the AVERAGE so a real split whose tip arm bends
    slightly but whose partner continues straight is not wrongly rejected. Returns
    the averaged cosine, or None if either tangent is undefined. Pure fragment
    geometry — GT-free, identical on train and held-out.
    """
    xyz = g.node_xyz
    a, b = s.node_a, s.node_b
    v2 = np.asarray(xyz[b], dtype=float) - np.asarray(xyz[a], dtype=float)
    n2 = np.linalg.norm(v2)
    if n2 == 0:
        return None
    v2 /= n2

    # cos_a: outward tangent of arm A vs the gap direction (4.0 µm walk to match
    # the harness, NOT the 6.0 µm TANGENT_WALK_UM the hard gate uses).
    v1 = _walk_tangent(g, a, HARNESS_TANGENT_WALK_UM)
    if v1 is None:
        return None
    cos_a = float(np.dot(v1, v2))

    # cos_b: the best same-segment neighbor direction at node_b continuing the gap
    # (the same ``best`` logic _is_colinear_split already uses).
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
    cos_b = best
    return 0.5 * (cos_a + cos_b)


def _rad_ratio(g, ctx, s):
    """Endpoint cable-caliber ratio max(ra,rb)/min(ra,rb) for SplitSite ``s``.

    Fetches the per-node radius array from ``ctx["node_radius"]`` first, then falls
    back to ``getattr(g, "node_radius", None)`` (the harness derives rad_ratio the
    same way). Reads the radius at the two gap endpoints ``s.node_a``/``s.node_b``.
    Returns None if no radius array is available or either radius is missing or
    <= 0; otherwise returns the RELATIVE ratio (>= 1.0). A ratio near 1.0 means the
    two broken ends have the same caliber (likely ONE neuron); a ratio far above 1
    means two different cables fused (a false join). This is purely relative, so it
    carries no brain-specific absolute-radius assumption.
    """
    radius = ctx.get("node_radius")
    if radius is None:
        radius = getattr(g, "node_radius", None)
    if radius is None:
        return None
    try:
        ra = float(radius[s.node_a])
        rb = float(radius[s.node_b])
    except Exception:
        return None
    if ra <= 0.0 or rb <= 0.0:
        return None
    return max(ra, rb) / min(ra, rb)


def propose_edits(sites, ctx) -> list:
    """Decide which candidate sites to repair, returning a list of edits.

    SEED POLICY (distance-graded colinear split-repair): for each SplitSite,
    emit a ``merge_labels`` edit when the gap is small
    (s.gap_um <= GAP_THRESHOLD_UM) AND the two fragments pass a colinearity gate
    whose strictness DEPENDS on the gap (three regimes):
      - MICRO (s.gap_um <= MICRO_GAP_UM): distance alone is DECISIVE — accept on
        gap alone, WAIVING colinearity entirely (the tip-tangent angle is jitter
        at these lattice-scale gaps; all reachable false joins sit at >= 2.20 µm).
      - TIP-TO-TIP (MICRO_GAP_UM < s.gap_um <= TIP_TIP_GAP_UM AND both endpoints
        are tips, deg_a == deg_b == 1): distance + topology decide — accept,
        WAIVING colinearity. Two genuine termini at a sub-2.3 µm gap is one neuron
        broken in two; the two tip-to-tip false joins are at gap 2.45/3.00 µm
        (out of band) and the 2.20 µm false join is tip-into-shaft (deg_b == 2,
        excluded). This converts the dense √5 = 2.24 µm missed-real cluster.
      - EXTENDED TIP-TO-TIP (TIP_TIP_GAP_UM < s.gap_um <= TIP_TIP_EXT_GAP_UM AND
        both endpoints tips AND caliber matched, rad_ratio <= TIP_TIP_RAD_MATCH):
        accept, WAIVING colinearity. Recovers the dense gap-2.45 tip-to-tip
        missed-real cluster; the only in-band tip-to-tip FALSE (rad_ratio 1.56) is
        excluded by the 1.50 caliber ceiling, the FALSE @3.00 by the gap ceiling.
      - NEAR  (MICRO_GAP_UM < s.gap_um <= NEAR_GAP_UM): use the loose floor
        NEAR_COLINEAR_COS — accept everything except joins that double back
        (anti-parallel / obtuse).  (Empty when MICRO_GAP_UM >= NEAR_GAP_UM.)
      - FAR   (NEAR_GAP_UM < s.gap_um <= GAP_THRESHOLD_UM): require
        colinearity, cos >= MIN_COLINEAR_COS (0.60, a clean continuation).
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

    edits = []
    for s in sites:
        kind = getattr(s, "kind", "split")
        if kind != "split":
            continue  # seed repairs splits only; leave merges for evolution
        if s.gap_um > GAP_THRESHOLD_UM:
            continue
        try:
            # MICRO regime: below MICRO_GAP_UM the gap distance ALONE is decisive.
            # All reachable false joins sit at gap >= 2.20 µm, while inter-neuron
            # gaps essentially never fall below ~7 µm, so a sub-2 µm gap is almost
            # certainly one broken neuron. Tip-tangent colinearity at these
            # lattice-scale gaps is dominated by skeletonization jitter and is
            # uninformative, so we WAIVE the colinearity gate entirely and accept.
            if s.gap_um <= MICRO_GAP_UM:
                edits.append(s.as_edit())   # (label_a, label_b) merge tuple
                continue
            # TIP-TO-TIP regime: just above the MICRO cutoff the missed real
            # splits pile up at the √5 = 2.24 µm voxel-lattice diagonal, where the
            # tip-tangent colinearity is lattice-scale jitter (uninformative, same
            # rationale as the MICRO tier). ENDPOINT DEGREE separates here: when
            # BOTH fragment endpoints are genuine TIPS (deg_a == deg_b == 1) the
            # site is the canonical "one neuron broken in two" signature (both
            # broken ends are true termini), not a pass-by shaft/branch contact.
            # So accept on DISTANCE + TOPOLOGY alone, waiving colinearity. This
            # stays below the two TIP-TO-TIP false joins (gap 2.45, 3.00 µm) and
            # excludes the tip-INTO-SHAFT false join at 2.20 µm (deg_b == 2), which
            # falls through to the FAR colinearity floor and is rejected.
            deg_a = len(list(g.neighbors(s.node_a)))
            deg_b = len(list(g.neighbors(s.node_b)))
            if s.gap_um <= TIP_TIP_GAP_UM and deg_a == 1 and deg_b == 1:
                edits.append(s.as_edit())   # (label_a, label_b) merge tuple
                continue
            # EXTENDED TIP-TO-TIP regime (caliber-gated): just ABOVE the
            # TIP_TIP_GAP_UM cutoff a dense cluster of MISSED tip-to-tip reals
            # piles up at gap 2.45 (all deg_a==deg_b==1) where colinear_cos is
            # lattice-scale jitter (scattered ~ -0.81..+0.53, the SAME noise that
            # justified waiving colinearity in MICRO/TIP-TO-TIP). They fall through
            # to the colinearity tier and are rejected (near-zero/negative cos, and
            # many have rad_ratio > 1.30 so the caliber OR-clause misses them too).
            # Distance alone cannot extend here (the tip-to-tip FALSE @2.45 sits in
            # this band), but CALIBER separates: the captured reals have
            # rad_ratio <= 1.46 while the only in-band tip-to-tip FALSE has
            # rad_ratio 1.56. So accept on DISTANCE + TOPOLOGY + matched caliber,
            # WAIVING colinearity (like the existing TIP-TO-TIP tier). The looser
            # 1.50 ceiling (vs the general path's 1.30) is safe BECAUSE
            # deg_a==deg_b==1 already excludes the high-caliber deg_b==2 FALSE
            # @2.20; the FALSE @2.45 (rad 1.56 > 1.50) and FALSE @3.00 (gap > 2.5)
            # both stay out → zero new train-false merges. Purely additive: a site
            # this accepts would otherwise fall to the colinearity tier, so correct
            # cannot drop below the parent's 256.
            rr_tt = _rad_ratio(g, ctx, s)
            if (TIP_TIP_GAP_UM < s.gap_um <= TIP_TIP_EXT_GAP_UM
                    and deg_a == 1 and deg_b == 1
                    and rr_tt is not None and rr_tt <= TIP_TIP_RAD_MATCH):
                edits.append(s.as_edit())   # (label_a, label_b) merge tuple
                continue
            # TIP-INTO-SHAFT regime (caliber-gated): the missed-real bucket is now
            # dominated by tip-into-shaft sites (deg_a == 1 AND deg_b == 2) at gap
            # 2.0-2.65 that fall through EVERY existing path — gap > TIP_TIP_GAP_UM,
            # deg_b == 2 so both tip-to-tip tiers skip them, low/negative colinear_cos
            # fails the 0.60/0.55 colinearity floors, and rad_ratio often just above
            # the colinearity-tier 1.30 caliber cap. Colinearity is actively
            # MISLEADING here: the ONLY reachable deg_b == 2 FALSE join (gap 2.20)
            # has colinear_cos 0.44 — HIGHER than roughly half the missed reals in
            # this band — so no colinearity floor can recover these reals without
            # admitting it. But CALIBER separates cleanly: that FALSE has rad_ratio
            # 1.97 (clearly MISMATCHED caliber), while the matched-caliber missed
            # reals to capture have rad_ratio <= ~1.49. So accept on DISTANCE +
            # TOPOLOGY + matched caliber, WAIVING colinearity (the tip-arm jitter
            # rationale is topology-independent — it corrupts the tip direction
            # regardless of node_b's degree). The 1.50 ceiling sits 0.47 BELOW the
            # FALSE @2.20 (rad 1.97) — a WIDE margin → zero new train-false merges;
            # the other two reachable FALSE joins are deg_b == 1, so this deg_b == 2
            # tier never sees them. PURELY ADDITIVE: a site this accepts currently
            # fails all existing paths, so train-correct cannot drop below the
            # parent's 267 — it can only ADD net-new correct repairs.
            rr_ts = _rad_ratio(g, ctx, s)
            if (MICRO_GAP_UM < s.gap_um <= TIP_SHAFT_GAP_UM
                    and deg_a == 1 and deg_b == 2
                    and rr_ts is not None and rr_ts <= TIP_SHAFT_RAD_MATCH):
                edits.append(s.as_edit())   # (label_a, label_b) merge tuple
                continue
            # HIGH-rad-ratio TIP-INTO-SHAFT regime (caliber FLOOR — the inverse of the
            # tier above): SAME topology (deg_a==1, deg_b==2) and SAME gap band, but for
            # the opposite caliber regime. A real T-junction is a THIN daughter neurite
            # joining the MIDDLE of a THICK parent shaft, so it is naturally VERY
            # mismatched in caliber (high rad_ratio). The matched-caliber tier above
            # captures the symmetric-caliber tip-into-shaft reals (rad_ratio <= 1.50);
            # this captures the high-mismatch ones (rad_ratio >= 2.0, up to ~2.75 — thin
            # tip ~0.75-1.06 onto thick shaft ~1.5-2.0). The gap prior does the heavy
            # lifting (gap <= 2.75 << 7 µm ⇒ overwhelmingly a REAL break); the 2.0 FLOOR
            # is the cheap precision guard keeping the 3 known FALSE joins out. FALSE-free:
            # all 3 reachable FALSE have rad_ratio < 2.0 (1.97, 1.56, 1.67); the only
            # deg_b==2 FALSE (@gap 2.20) is at 1.97, so a 2.0 floor leaves a 0.03 margin
            # above it AND the (1.50, 2.00) band is deliberately a no-go zone protecting
            # exactly that FALSE; the other two FALSE are deg_b==1 (never seen here).
            # PURELY ADDITIVE (new tier + continue): a site this accepts currently fails
            # all existing paths, so train-correct cannot drop below the parent's 286 — it
            # can only ADD net-new correct repairs.
            rr_ts_hi = rr_ts
            if (MICRO_GAP_UM < s.gap_um <= TIP_SHAFT_GAP_UM
                    and deg_a == 1 and deg_b == 2
                    and rr_ts_hi is not None and rr_ts_hi >= TIP_SHAFT_HIGH_RAD_RATIO):
                edits.append(s.as_edit())   # (label_a, label_b) merge tuple
                continue
            # MID-CALIBER TIP-INTO-SHAFT regime (gap-floored band re-fill — a NEW
            # kind of lever, distinct from the rad-ceiling/rad-floor tiers above):
            # SAME topology (deg_a==1, deg_b==2), but re-opens the MIDDLE caliber
            # band (TIP_SHAFT_RAD_MATCH < rad_ratio < TIP_SHAFT_HIGH_RAD_RATIO, i.e.
            # 1.50 < rad < 2.0) that the matched-caliber tier (rad <= 1.50) and the
            # gen13 high-mismatch tier (rad >= 2.0) deliberately skipped. That band
            # was a NO-GO zone for ONE reason: the FALSE join @gap 2.20 has rad_ratio
            # 1.97, sitting inside it — and it is the ONLY reachable deg_b==2 FALSE
            # join at ANY gap (the other two FALSE are deg_b==1, never seen here). So
            # the band can be safely re-filled by gating on gap ABOVE that single
            # FALSE: the DEG2_MID_GAP_FLOOR = 2.40 sits 0.20 µm above the FALSE @2.20,
            # so no reachable deg_b==2 FALSE can enter → ZERO new train-false merges.
            # The dominant remaining missed cluster is exactly these mid-caliber reals
            # (densely populated at gaps 2.49-2.73; e.g. (2.49,1.86), (2.50,1.90),
            # (2.57,1.82), (2.70,1.94), (2.73,1.76)). The gap prior does the heavy
            # lifting (gap 2.40-2.75 << 7 µm ⇒ overwhelmingly a REAL break by
            # population statistics, AUC 0.998). PURELY ADDITIVE (new tier + continue)
            # and mutually exclusive by caliber band with the two existing deg_b==2
            # tiers (rad <= 1.50 and rad >= 2.0): a site this accepts currently fails
            # all existing paths, so train-correct cannot drop below the parent's 323
            # — it can only ADD net-new correct repairs.
            rr_ts_mid = rr_ts   # rad ratio already computed above for this site
            if (DEG2_MID_GAP_FLOOR < s.gap_um <= TIP_SHAFT_GAP_UM
                    and deg_a == 1 and deg_b == 2
                    and rr_ts_mid is not None
                    and TIP_SHAFT_RAD_MATCH < rr_ts_mid < TIP_SHAFT_HIGH_RAD_RATIO):
                edits.append(s.as_edit())   # (label_a, label_b) merge tuple
                continue
            # LOW-MID-CALIBER SUB-FLOOR TIP-INTO-SHAFT regime (re-opens the LOW
            # caliber slice of the carved-out (1.50, 2.0) band BELOW the 2.40 mid
            # floor): SAME topology (deg_a==1, deg_b==2) at SMALL gap
            # (MICRO_GAP_UM < gap < DEG2_MID_GAP_FLOOR = 2.40), colinearity WAIVED.
            # The matched-caliber tier (rad <= 1.50) owns the band below 1.50; the
            # gen15 mid-caliber tier owns gap > 2.40; the gen13 high tier owns
            # rad >= 2.0. That leaves a sub-floor NO-GO slice — gap (2.0, 2.40) AND
            # rad_ratio (1.50, 2.0) — carved out ONLY to protect the single reachable
            # deg_b==2 FALSE @gap 2.20 (rad_ratio 1.97). But the LOW-caliber subset of
            # that slice is cleanly separable: the missed reals here are tip-into-shaft
            # reals at gap 2.03-2.19 with rad_ratio ~1.52-1.57, FAR below that FALSE's
            # 1.97. A 1.60 caliber CEILING sits 0.37 BELOW the FALSE — a WIDE margin —
            # so re-opening only the (1.50, 1.60] slice admits those reals while
            # keeping the FALSE @1.97 out → ZERO new train-false merges. The higher
            # sub-floor reals (2.21-2.35, rad 1.90-1.98) overlap the FALSE in both gap
            # and caliber, so they are NOT re-opened (the protective (1.60, 2.0) slice
            # stays a NO-GO zone). Held-out-safe because it does NOT loosen any gap
            # ceiling — every site is at gap < 2.40, the TIGHTEST gap regime, where a
            # tip-into-shaft junction with near-matched caliber is overwhelmingly one
            # broken neuron (findings #1, #21: gap-distance AUC 0.998, ~99% of true
            # split gaps under ~6.5 µm, so at gap ~2.0-2.4 ≪ 7 µm a site is
            # overwhelmingly a REAL break); the precision guard is RELATIVE caliber
            # (brain-agnostic), not gap distance. Reuses the already-computed rr_ts.
            # PURELY ADDITIVE (new tier + continue) and mutually exclusive by
            # (gap-band × caliber-band) with every existing deg_b==2 tier: the slice it
            # opens currently falls through to the colinearity tier and is rejected on
            # scattered colinear_cos, so train-correct cannot drop below the parent's
            # 365 — it can only ADD net-new correct repairs.
            if (MICRO_GAP_UM < s.gap_um < DEG2_MID_GAP_FLOOR
                    and deg_a == 1 and deg_b == 2
                    and rr_ts is not None
                    and TIP_SHAFT_RAD_MATCH < rr_ts <= DEG2_LOWMID_RAD_CEIL):
                edits.append(s.as_edit())   # (label_a, label_b) merge tuple
                continue
            # EXTENDED-GAP TIP-INTO-SHAFT regime (recall path one step beyond the
            # 2.75 ceiling): SAME topology (deg_a==1, deg_b==2), accepting across the
            # FULL rad_ratio range and WAIVING colinearity — exactly the union of the
            # three existing deg_b==2 tiers (matched-caliber rad<=1.50, mid-caliber
            # 1.50<rad<2.0 above gap 2.40, high-mismatch rad>=2.0), which together
            # accept essentially ALL tip-into-shaft sites at gap <= TIP_SHAFT_GAP_UM
            # (2.75). Those three tiers ALL cap at 2.75, so the dominant remaining
            # missed cluster is deg_b==2 reals densely packed at gap 2.76-2.98 across
            # the full rad range (e.g. (2.76,1.23), (2.81,2.08), (2.85,2.06),
            # (2.90,2.63), (2.97,2.75)), missed ONLY because gap > 2.75. This tier
            # extends the SAME topology one step further, to (TIP_SHAFT_GAP_UM,
            # TIP_SHAFT_EXT_GAP_UM] = (2.75, 3.0]. FALSE-free with a WIDE margin: the
            # FALSE @gap 2.20 is the ONLY reachable deg_b==2 FALSE join at ANY gap,
            # and the new floor (2.75) sits 0.55 µm above it; there is NO deg_b==2
            # FALSE anywhere in (2.75, 3.0]. The other two reachable FALSE joins
            # (incl. the one @gap 3.00) are deg_b==1, so this deg_b==2 tier NEVER sees
            # them — which is also why deg_b==1 is deliberately NOT extended here. The
            # gap prior does the heavy lifting (findings #1, #21): at gap 2.75-3.0
            # << 7 µm a tip-into-shaft site is overwhelmingly a REAL break
            # (gap-distance AUC 0.998, F1-optimal 6.84 µm; ~99% of true split gaps
            # under ~6.5 µm), so even held-out a deg_b==2 FALSE in this window is
            # extraordinarily rare. PURELY ADDITIVE (new tier + continue): these
            # gap-2.76-2.98 deg_b==2 sites currently fail all paths (gap > 2.75 so all
            # three deg_b==2 tiers skip; scattered/negative colinear_cos fails the
            # colinearity tier), so train-correct cannot drop below the parent's 336
            # — it can only ADD net-new correct repairs.
            if (TIP_SHAFT_GAP_UM < s.gap_um <= TIP_SHAFT_EXT_GAP_UM
                    and deg_a == 1 and deg_b == 2):
                edits.append(s.as_edit())   # (label_a, label_b) merge tuple
                continue
            # HIGH-rad-ratio TIP-TO-TIP regime (caliber FLOOR — the tip-to-tip
            # analogue of the gen13 tip-into-shaft floor above): deg_a==1 AND
            # deg_b==1 (two free ENDS meeting) at gap <= TIP_TIP_HIGH_RAD_GAP_UM
            # with HIGH caliber mismatch (rad_ratio >= TIP_SHAFT_HIGH_RAD_RATIO =
            # 2.0). The matched-caliber EXTENDED tip-to-tip tier (rad_ratio <= 1.50)
            # already captures the symmetric-caliber tip-to-tip reals; this captures
            # the HIGH-mismatch ones (a thin distal end ~0.75-1.0 meeting a thick end
            # ~1.5-2.4), which the gen14 report shows are dense in the missed bucket
            # at gaps 2.45-2.83. The gap prior does the heavy lifting (at gap <= 2.9
            # << 7 µm a site is overwhelmingly a REAL break by population statistics);
            # the rad_ratio >= 2.0 FLOOR is the cheap precision guard. FALSE-free with
            # DOUBLE protection: both reachable deg_b==1 FALSE joins have rad_ratio
            # < 2.0 (1.56, 1.67) so the 2.0 floor alone excludes them (margin >= 0.33),
            # AND the only deg_b==1 FALSE near this window (@gap 3.00) is also beyond
            # the 2.9 gap ceiling; the deg_b==2 FALSE (@2.20) is never seen by this
            # deg_b==1 tier. PURELY ADDITIVE (new tier + continue): a site this accepts
            # currently fails all existing paths, so train-correct cannot drop below
            # the parent's 317 — it can only ADD net-new correct repairs.
            rr_tt_hi = rr_ts   # rad ratio already computed above for this site
            if (MICRO_GAP_UM < s.gap_um <= TIP_TIP_HIGH_RAD_GAP_UM
                    and deg_a == 1 and deg_b == 1
                    and rr_tt_hi is not None and rr_tt_hi >= TIP_SHAFT_HIGH_RAD_RATIO):
                edits.append(s.as_edit())   # (label_a, label_b) merge tuple
                continue
            # MID-GAP TIP-TO-TIP regime (distance + topology, caliber AND colinearity
            # WAIVED): deg_a==1 AND deg_b==1 (two free ENDS meeting) at gap in the
            # FALSE-free window (TIP_TIP_EXT_GAP_UM (2.5), TIP_TIP_HIGH_RAD_GAP_UM (2.9)].
            # This GENERALIZES the gen14 HIGH-rad tip-to-tip tier above (which required
            # rad_ratio >= 2.0) to the FULL rad_ratio range, recovering the dominant
            # missed cluster of tip-to-tip reals piled at gap ~2.83 (= √8 ≈ 2.828, a
            # voxel-lattice diagonal) with rad_ratio < 2.0 and scattered colinear_cos
            # < 0.6 (so the HIGH-rad floor and the colinearity tier both skip them).
            # Colinearity is WAIVED for the same lattice-jitter reason as the MICRO /
            # TIP-TO-TIP tiers (the tip-tangent angle is skeletonization jitter at these
            # sub-3 µm scales). FALSE-free: the window (2.5, 2.9] contains NO reachable
            # deg_b==1 FALSE join — the two reachable deg_b==1 FALSE sit at gap 2.45
            # (BELOW the 2.5 floor, owned by the EXTENDED tip-to-tip tier) and gap 3.00
            # (ABOVE the 2.9 ceiling); the deg_b==2 FALSE @2.20 is never seen by this
            # deg_b==1 tier. The gap prior does the heavy lifting (findings #1, #21):
            # at gap 2.5-2.9 << 7 µm a tip-to-tip site is overwhelmingly a REAL break
            # (gap-distance AUC 0.998, F1-optimal 6.84 µm; ~99% of true split gaps under
            # ~6.5 µm). Reuses existing constants (no new constant). Placed immediately
            # AFTER the gen14 HIGH-rad tip-to-tip tier and BEFORE the colinearity tier;
            # PURELY ADDITIVE (new tier + continue): a site it accepts currently fails
            # every existing path (gap > 2.5 so both lower tip-to-tip tiers skip on gap;
            # rad < 2.0 so the HIGH-rad tier skips; cos < 0.6 / scattered so the
            # colinearity tier rejects), so train-correct cannot fall below the parent's
            # 356 — it can only ADD net-new correct repairs.
            if (TIP_TIP_EXT_GAP_UM < s.gap_um <= TIP_TIP_HIGH_RAD_GAP_UM
                    and deg_a == 1 and deg_b == 1):
                edits.append(s.as_edit())   # (label_a, label_b) merge tuple
                continue
            # Distance-graded colinearity gate: at tiny gaps the gap itself is the
            # discriminator, so only veto anti-parallel joins; at larger gaps demand
            # a strong straight-line continuation.
            min_cos = NEAR_COLINEAR_COS if s.gap_um <= NEAR_GAP_UM else MIN_COLINEAR_COS
            # The existing single-arm hard gate (cos_a >= min_cos as a SEPARATE
            # requirement, 6.0 µm walk) mismatches the harness's validated
            # colinear_cos, which is the AVERAGE of the two half-cosines at a 4.0 µm
            # walk: it wrongly rejects reals whose tip arm bends slightly (cos_a ~0.4)
            # but whose partner continues straight (cos_b ~0.9; avg ~0.65). Add a
            # SECOND, purely ADDITIVE accept path on that averaged measure (OR), so it
            # can only ADD missed reals, never drop a currently-accepted site. All 3
            # reachable FALSE joins have colinear_cos <= 0.44 < COLINEAR_AVG_ACCEPT, so
            # they remain rejected (no new false merge).
            avg_col = _avg_cross_gap_colinearity(g, s)
            # THIRD, purely ADDITIVE accept path — CABLE CALIBER. The colinearity
            # paths (a) and (b) miss the dense band of reals just above 2.0 µm whose
            # tip-tangent angle is lattice jitter (colinear_cos scattered ~ -0.8..+0.5)
            # yet whose two broken ends have MATCHED caliber (rad_ratio near 1.0) —
            # the signature of ONE neuron snapped in two. Admit those when (i) the
            # caliber is matched (rr <= RAD_RATIO_MATCH = 1.30), (ii) the join is not
            # anti-parallel (avg_col >= 0.0, so it does not double back), and (iii)
            # the gap is tight (<= RAD_MATCH_GAP_UM = 2.5). All 3 reachable FALSE
            # joins have rad_ratio >= 1.56 > 1.30 (~0.26 margin), so none can enter
            # this path: zero new train-false merges. Purely OR, so no currently
            # accepted site is dropped (correct cannot fall below the parent's 248).
            rr = _rad_ratio(g, ctx, s)
            if (_is_colinear_split(g, s, min_cos)
                    or (avg_col is not None and avg_col >= COLINEAR_AVG_ACCEPT)
                    or (rr is not None and rr <= RAD_RATIO_MATCH
                        and avg_col is not None and avg_col >= 0.0
                        and s.gap_um <= RAD_MATCH_GAP_UM)):
                edits.append(s.as_edit())   # (label_a, label_b) merge tuple
        except Exception:
            # Geometry is advisory; never let one odd site crash the whole policy.
            continue
    return edits
