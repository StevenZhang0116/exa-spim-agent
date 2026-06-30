"""
THE EVOLVED PROGRAM (executable policy).  <-- the evolution loop edits THIS file.

``propose_edits`` is the proofreader's decision policy: given a UNIFIED stream of
candidate sites enumerated from the fragment graph, decide which proofreading edits
to make. The harness feeds the returned edits to the scoring framework.

The candidate stream has TWO kinds of site (dispatch on ``site.kind``):

  - SplitSite  (kind == "split"): two nearby fragments with DIFFERENT labels — a
    neuron that the segmentation broke into pieces. Valid repair = ``merge_labels``
    (unify the two labels). Fields: ``label_a``, ``label_b``, ``gap_um``,
    ``node_a``/``node_b`` (fragment-graph node ids), ``xyz_a``/``xyz_b``, and
    ``alt_gaps`` (list — OTHER gaps between the SAME label pair; empty unless
    ``ENUM_PARAMS["split_alt_per_pair"] > 1``). The site carries the CLOSEST gap;
    ``alt_gaps`` holds the next-closest as dicts ``{gap_um, node_a, node_b, xyz_a,
    xyz_b}``. Two fragments can touch in more than one place and the closest gap is
    not always the most decisive (it may be a sideways graze while another gap is a
    clean colinear continuation) — inspect ``alt_gaps`` to judge on the BEST evidence
    point. It does NOT change the action: ``as_edit()`` still emits ONE
    ``merge_labels(label_a, label_b)`` (one merge unifies the pair across all gaps).
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
                 #   ctx["n_split_sites"], ctx["n_merge_sites"],
                 #   ctx["image_patch_shape"] — the IMAGE RECEPTIVE FIELD (z,y,x voxels)
                 #     you may pass to any reader method's shape= arg, e.g.
                 #     reader.gap_bridge_evidence(a, b, shape=ctx["image_patch_shape"]).
                 #     Bigger = more context but a larger (slower/costlier) cloud read;
                 #     each axis is clamped to 512. Tune per tier (tight for a clean
                 #     micro-gap, wider to confirm a long faint bridge).
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
# Short-fragment proximity rule (recall lever, see rules.md Gen 1 & Gen 2). On
# train the NEAREST false join sits at gap 2.19 µm (and every false join has
# colinear_cos <= 0.62), while below ~2 µm there are ZERO false joins of ANY
# endpoint degree. The fragments at these gaps are short stubs whose tip tangent
# is unreliable (colinear_cos goes near 0 / negative even for genuine breaks), so
# the strict colinearity gate wrongly rejects the largest cluster of MISSED real
# splits. For gaps this small, two endpoints meeting this close are almost
# certainly one broken neuron, so accept on proximity alone (no colinearity
# demand) for tip-to-tip (deg_b==1) AND tip-to-shaft (deg_b==2) pairs. Gen 2
# widened the rule to deg_b==2 because the ENTIRE sub-1.8 µm MISSED bucket is
# tip-to-shaft; deg_b>=3 (true branch points) stays excluded as the riskiest join.
SMALL_GAP_UM = 1.8          # below this gap, accept tip-to-tip/-shaft pairs without the colinear test
# Anti-tangle fan-out cap (generalizable topology lever, see rules.md Gen 3). The
# proximity rule (B) admits tip-to-shaft pairs, which let a single long shaft label
# absorb MANY grazing tips in a tight spot — a star-tangle that chains DISTINCT
# neurons (gen03 mega-merge: classes of 5-6 raw labels). A real neuron rebuilt from
# its fragments is instead a CHAIN: end-to-end merges give each label at most ~2
# merge partners. So cap each label's merge degree: refuse any merge that would
# push either endpoint past MAX_MERGE_DEGREE partners. This is pure topology (no
# train-specific gap threshold, so it generalizes to held-out), caps the over-merge
# the report flagged, and still allows arbitrarily LONG linear rebuilds (recall is
# preserved — only high-fan-out hubs are pruned).
MAX_MERGE_DEGREE = 2        # max merge partners per label in the proximity/colinear merge graph

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
    # "split_alt_per_pair": 1, # gaps kept per SplitSite label pair [1..10]. >1 attaches
    #                          # the next-closest gaps as site.alt_gaps (extra evidence
    #                          # points) WITHOUT adding sites or edits — useful when the
    #                          # closest gap's geometry is poor but another gap between
    #                          # the same pair is a clean colinear continuation. To use
    #                          # it, also read site.alt_gaps in propose_edits.
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


def _node_degree(g, node):
    """Graph degree of ``node`` (number of neighbors). None if it can't be read.

    A fragment tip has degree 1, a mid-shaft node degree 2, a branch >=3. Used by
    the short-fragment proximity rule to restrict to tip-to-tip (degree 1 both
    ends) pairs — the nose-to-nose break signature — which is the precision-safe
    subset of sub-``SMALL_GAP_UM`` candidates.
    """
    try:
        return sum(1 for _ in g.neighbors(node))
    except Exception:
        return None


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

    # ---- Pass 1: geometric acceptance (rules A and B) ------------------------
    # Collect every SplitSite that passes the geometry, tagging each with a
    # confidence PRIORITY (0 = colinear continuation, the most reliable evidence;
    # 1 = short-gap proximity). The fan-out cap in pass 2 spends its budget on the
    # highest-confidence merges first.
    accepted = []  # list of (priority, site)
    for s in sites:
        kind = getattr(s, "kind", "split")
        if kind != "split":
            continue  # this run repairs splits only; leave merges for evolution
        if s.gap_um > GAP_THRESHOLD_UM:
            continue
        try:
            # (A) mid/long-gap continuation: the original high-precision rule —
            #     small gap AND a straight-line (colinear) continuation.
            if _is_colinear_split(g, s, MIN_COLINEAR_COS):
                accepted.append((0, s))
                continue
            # (B) short-fragment proximity rule (Gen 1, widened Gen 2): for very
            #     small gaps the tip tangent is unreliable (short stubs), so the
            #     colinear test in (A) wrongly rejects genuine breaks. On train
            #     there are ZERO false joins of ANY degree below ~2.2 µm, so accept
            #     on proximity alone. Gen 2 admits tip-to-shaft (deg_b==2) as well
            #     as tip-to-tip (deg_b==1); deg_b>=3 (true branch/crossing points)
            #     stays excluded as the riskiest join geometry.
            if s.gap_um <= SMALL_GAP_UM:
                deg_a = _node_degree(g, s.node_a)   # always a tip (1) per enumerator
                deg_b = _node_degree(g, s.node_b)
                if deg_a == 1 and deg_b in (1, 2):
                    accepted.append((1, s))
        except Exception:
            # Geometry is advisory; never let one odd site crash the whole policy.
            continue

    # ---- Pass 2: cap merge-graph fan-out (anti-tangle, generalizable) --------
    # A real neuron rebuilt from fragments is a CHAIN: end-to-end merges give each
    # label at most ~2 merge partners. A high-fan-out hub — one label absorbing
    # many partners in a tight spot — is the tangle signature where the proximity
    # rule chains DISTINCT neurons (the gen03 mega-merge: classes of 5-6 labels).
    # Take the most confident merges first (colinear before proximity, then
    # smallest gap) and refuse any merge that would push either label's merge
    # degree past MAX_MERGE_DEGREE. This caps the over-merge with pure topology (no
    # train-specific threshold, so it generalizes) while leaving long LINEAR
    # rebuilds intact, so recall is preserved.
    accepted.sort(key=lambda t: (t[0], t[1].gap_um))
    degree = {}
    edits = []
    for _prio, s in accepted:
        a, b = s.label_a, s.label_b
        if degree.get(a, 0) >= MAX_MERGE_DEGREE or degree.get(b, 0) >= MAX_MERGE_DEGREE:
            continue  # would over-grow a hub — skip to avoid a tangle merge
        degree[a] = degree.get(a, 0) + 1
        degree[b] = degree.get(b, 0) + 1
        edits.append(s.as_edit())   # (label_a, label_b) merge tuple
    return edits
