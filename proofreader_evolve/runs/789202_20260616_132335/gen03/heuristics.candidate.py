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

# --- Merge-repair (split_label) gating ---------------------------------------
# These bound when we are willing to CUT one label into two neurons. A wrong split
# raises %Split Edges (the over-split watchdog), so the bar is intentionally high:
# both arms must be genuinely long cables AND the topology must look like two
# distinct neurites meeting rather than one neuron passing through. The gate then
# rejects any generation that raises %Split/#Merges, so we only emit a split when
# multiple independent geometric signals agree (and, when available, image valley
# evidence CONFIRMS a true touch rather than rejecting candidates).
SPLIT_MIN_CABLE_UM = 15.0   # each arm must carry at least this much cable to be a real neuron
SPLIT_MAX_ANGLE_DEG = 130.0 # arms sharper than this (far from 180°=straight pass-through) look fused
SPLIT_MIN_RADIUS_RATIO = 1.6  # calibers this different suggest two distinct cables fused
SPLIT_VALLEY_RATIO_MAX = 0.6  # image: chord intensity must DIP to <=60% of arm endpoints to confirm a touch
SPLIT_VALLEY_POS_LO = 0.3   # the valley must sit near the middle of the chord (a real seam),
SPLIT_VALLEY_POS_HI = 0.7   # not at an endpoint (which would just be a dim arm)


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


def _merge_site_is_two_neurons(s):
    """Cheap, image-free verdict on a MergeSite: do the GEOMETRY signals agree
    that ONE label has fused TWO distinct neurites (so a ``split_label`` is
    warranted)? Returns (accept_geom, needs_image_confirm).

    The bar is deliberately high because a wrong split raises %Split Edges and the
    gate reverts any generation that does so. We require, for every detector:
      * BOTH arms to be genuinely long cables (not a short spur), and
      * an explicit "two neurites" topology signal whose form depends on the
        detector that found the site:
          - "branch"    : a sharp arm angle (far from 180° straight pass-through)
                          OR a clear caliber mismatch between the two arms.
          - "bridge"    : a sharp kink angle AND a caliber pinch (radius mismatch).
          - "component" : the label is physically DISCONNECTED into >=2 long pieces
                          (angle is NaN here) — the disconnection itself is the signal.
      * the two arms must NOT re-converge downstream (``arms_reconverge`` True means
        one neuron's own branches / a loop — never cut those).

    ``needs_image_confirm`` is True for the geometrically-borderline acceptances so
    the caller can require image valley evidence before committing the cut.
    """
    if getattr(s, "arms_reconverge", None) is True:
        return False, False  # one neuron's own loop/branches — never split

    cable_a = getattr(s, "cable_a_um", None)
    cable_b = getattr(s, "cable_b_um", None)
    if cable_a is None or cable_b is None:
        return False, False
    if min(cable_a, cable_b) < SPLIT_MIN_CABLE_UM:
        return False, False  # at least one arm is a short spur, not a second neuron

    detector = getattr(s, "detector", "branch")
    angle = getattr(s, "angle_deg", float("nan"))
    rratio = getattr(s, "radius_ratio", None)
    sharp_angle = (angle == angle) and (angle <= SPLIT_MAX_ANGLE_DEG)  # NaN-safe
    caliber_gap = (rratio is not None) and (rratio >= SPLIT_MIN_RADIUS_RATIO)

    if detector == "component":
        # Two long, DISCONNECTED pieces under one label — the strongest, safest
        # merge signal; the seeds already straddle the contact. Accept on geometry.
        return True, False
    if detector == "bridge":
        # A thin neck: demand BOTH a sharp kink AND a caliber pinch (high bar).
        if sharp_angle and caliber_gap:
            return True, True
        return False, False
    # "branch" (or unknown): a degree>=3 meeting of two long arms.
    if sharp_angle and caliber_gap:
        return True, False   # two strong, independent signals — accept on geometry
    if sharp_angle or caliber_gap:
        return True, True    # one signal — let image evidence confirm before cutting
    return False, False


def _image_confirms_cut(s, ctx):
    """Use the raw fluorescence chord between the two arm seeds to CONFIRM (not
    reject) that this label is a touch of two neurons. A low valley_ratio near the
    middle of the chord means the signal DIPS at the seam ⇒ two structures only
    touch ⇒ split is warranted. Returns True only on clear valley evidence; returns
    False when the image is unavailable (so geometry-only acceptances must already
    be confident on their own)."""
    reader = ctx.get("read_image_patch")
    if reader is None:
        return False
    na = getattr(s, "seed_a_node", None)
    nb = getattr(s, "seed_b_node", None)
    if na is None or nb is None:
        return False
    try:
        ev = reader.merge_cut_evidence(na, nb)
    except Exception:
        return False
    if not ev:
        return False
    vr = ev.get("valley_ratio")
    vp = ev.get("valley_pos")
    if vr is None or vp is None:
        return False
    return (vr <= SPLIT_VALLEY_RATIO_MAX) and (SPLIT_VALLEY_POS_LO <= vp <= SPLIT_VALLEY_POS_HI)


def propose_edits(sites, ctx) -> list:
    """Decide which candidate sites to repair, returning a list of edits.

    SEED POLICY (conservative split-repair): for each SplitSite, emit a
    ``merge_labels`` edit only when BOTH:
      (1) the gap is small  (s.gap_um <= GAP_THRESHOLD_UM), and
      (2) the two fragments are COLINEAR across the gap (a straight-line
          continuation — see ``_is_colinear_split``).
    Gen 3 ALSO repairs merge errors: for each MergeSite, emit a ``split_label``
    edit when ``_merge_site_is_two_neurons`` says the geometry signals agree it is
    two fused neurites, optionally confirmed by image valley evidence on the
    borderline (single-signal) cases. This targets the dominant remaining error
    component (%Merged Edges) that the merge_labels-only path never touched.

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
        if kind == "split":
            # --- Split-repair (merge_labels): unchanged high-precision colinear path.
            if s.gap_um > GAP_THRESHOLD_UM:
                continue
            try:
                if _is_colinear_split(g, s, MIN_COLINEAR_COS):
                    edits.append(s.as_edit())   # (label_a, label_b) merge tuple
            except Exception:
                # Geometry is advisory; never let one odd site crash the policy.
                continue
        elif kind == "merge":
            # --- Merge-repair (split_label): NEW in Gen 3. The dominant remaining
            # error component is %Merged Edges (~17%), and the policy emitted ZERO
            # split_label edits, leaving documented two-neuron labels uncut. Cut a
            # label ONLY when multiple geometry signals agree it is two neurons;
            # for the borderline (single-signal) cases, require image valley
            # evidence to CONFIRM before committing (recall lever, not a filter).
            try:
                accept_geom, needs_image = _merge_site_is_two_neurons(s)
                if not accept_geom:
                    continue
                if needs_image and not _image_confirms_cut(s, ctx):
                    continue
                edits.append(s.as_edit())   # split_label dict (uses the arm seeds)
            except Exception:
                continue
    return edits
