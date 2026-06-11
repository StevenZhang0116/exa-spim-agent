"""
THE EVOLVED PROGRAM (executable policy).  <-- the evolution loop edits THIS file.

``propose_edits`` is the proofreader's decision policy: given the candidate
split sites enumerated from the fragment graph, decide which fragment pairs to
unify (i.e. which splits to repair). Its output is a list of ``(label_a, label_b)``
edits that the scoring harness feeds to the metric framework's ``LabelHandler``.

The evolution loop works by:
  1. running this policy on the TRAIN skeletons,
  2. scoring the result with the real metric framework,
  3. showing the agent where the policy was wrong (splits left unrepaired,
     or bad unifications that created merges),
  4. asking the agent to rewrite the body of ``propose_edits`` (and the
     companion ``rules.md``) to do better,
  5. keeping the rewrite ONLY if held-out Edge Accuracy improves.

Contract (keep the CALL signature stable so the harness can always call it):
    propose_edits(sites, ctx) -> list[edit]

  sites : list[SplitSite]   # from dataset.candidate_split_sites
  ctx   : dict              # free-form context the harness provides, e.g.
                            #   ctx["max_gap_um"], ctx["fragments_graph"]
  return: list of edits. Each edit is EITHER a legacy 2-tuple
          ``(label_a, label_b)`` (treated as a merge) OR a typed dict:
            {"kind": "merge_labels", "label_a": str, "label_b": str}      # repair split
            {"kind": "split_label",  "label": str,                        # repair merge
             "seed_a_xyz": (x,y,z), "seed_b_xyz": (x,y,z)}
            {"kind": "flag_review", "reason": str} / {"kind": "reject_candidate"}
          The harness normalizes tuples and dicts uniformly (see
          harness.edit_handler.normalize_edits), so they may be mixed.

This seed version is deliberately simple — a single distance threshold, returning
merge tuples — so the first generation has obvious, measurable failure modes for
the agent to improve.
"""

from __future__ import annotations

import numpy as np


# --- Tunable parameters (the agent may rewrite these and the logic below) -----
# Gen 1 introduces the first REAL policy: a selective proximity + tangent-agreement
# merge. The over-merge failure mode (two unrelated neurites crossing within a few
# microns) is suppressed by requiring the two fragments' local tangent directions
# to be roughly collinear — i.e. the join must look like a *continuation* of one
# process, not a crossing. We keep the gap gate tight to stay high-precision so
# this generation can only add net-positive merges over the no-op baseline.
GAP_THRESHOLD_UM = 4.0        # only consider joins with a physical gap <= this
TANGENT_RADIUS_UM = 6.0       # neighborhood radius used to estimate a fragment tangent
MIN_COS_COLLINEAR = 0.80      # require tip->partner alignment with both tangents


def _local_tangent(g, node, radius):
    """Unit direction of the fragment at ``node`` (points outward from the body
    toward ``node``), estimated by PCA over the local rooted subgraph. Returns
    None if there is not enough local structure to define a direction."""
    try:
        sub = g.rooted_subgraph(node, radius)
        nodes = list(sub.nodes)
    except Exception:
        nodes = [node]
    if len(nodes) < 2:
        return None
    pts = np.asarray([g.node_xyz[n] for n in nodes], dtype=float)
    origin = np.asarray(g.node_xyz[node], dtype=float)
    centered = pts - pts.mean(axis=0)
    # Principal axis of the local neighborhood = best-fit line direction.
    try:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    axis = vh[0]
    n = np.linalg.norm(axis)
    if n < 1e-9:
        return None
    axis = axis / n
    # Orient the axis to point from the neighborhood centroid toward the node,
    # i.e. the outward-pointing tangent of the fragment at this endpoint.
    out = origin - pts.mean(axis=0)
    if np.dot(out, axis) < 0:
        axis = -axis
    return axis


def _unit(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n < 1e-9:
        return None
    return v / n


def propose_edits(sites, ctx) -> list[tuple]:
    """Repair split errors by unifying tip/partner label pairs that are both
    *close* and *collinear* (a continuation of one neurite, not a crossing).

    For each candidate site we require:
      1. Proximity: ``gap_um <= GAP_THRESHOLD_UM``.
      2. Tangent agreement: the outward tangent at the tip (``node_a``) and the
         tangent of the partner fragment (``node_b``) are roughly anti-parallel,
         AND each is roughly aligned with the gap-bridging direction. This is the
         geometric signature of one severed process rather than two crossing ones.

    Returns legacy ``(label_a, label_b)`` merge tuples; an empty list = no change.
    """
    g = ctx.get("fragments_graph") if isinstance(ctx, dict) else None
    if g is None:
        return []

    edits = []
    seen_pairs = set()
    for s in sites:
        if s.gap_um > GAP_THRESHOLD_UM:
            continue
        pair = frozenset((s.label_a, s.label_b))
        if pair in seen_pairs:
            continue

        a = np.asarray(s.xyz_a, dtype=float)
        b = np.asarray(s.xyz_b, dtype=float)
        bridge = _unit(b - a)            # direction across the gap, tip -> partner
        if bridge is None:
            continue

        tan_a = _local_tangent(g, s.node_a, TANGENT_RADIUS_UM)
        tan_b = _local_tangent(g, s.node_b, TANGENT_RADIUS_UM)
        if tan_a is None or tan_b is None:
            continue

        # The tip should point INTO the gap (its outward tangent aligns with the
        # bridge), and the partner's tangent should oppose it (the two ends face
        # each other) — i.e. a straight continuation.
        if np.dot(tan_a, bridge) < MIN_COS_COLLINEAR:
            continue
        if np.dot(tan_a, -tan_b) < MIN_COS_COLLINEAR:
            continue

        edits.append((s.label_a, s.label_b))
        seen_pairs.add(pair)

    return edits
