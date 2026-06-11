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
          ``(label_a, label_b)`` (treated as a merge) OR a typed dict.

----------------------------------------------------------------------------
Gen 3 policy: BOUNDED-DOWNSIDE size gating + AMBIGUITY-MARGIN (Lowe ratio).

Why this shape, given the history:
  * Gen 1 (PCA tangent / collinearity) gave +0.000 — geometry direction agreement
    did not separate good from bad joins.
  * Gen 2 (mutual-NN tip-to-tip pairing, small gap + reciprocity) scored
    -2.808 on held-out: it OVER-MERGED. Because fitness is RUN-LENGTH-WEIGHTED
    Edge Accuracy, one wrong merge that fuses two long neurites destroys a huge
    fraction of correctly-traced path length — that single catastrophe dwarfs
    many correct small repairs. Reciprocity + small gap does not rule out two
    unrelated neurites whose tips coincide at a crossing.

The winning move is therefore EXTREME PRECISION with a BOUNDED worst case, not
higher recall. This policy enforces two structurally-distinct guards:

  (A) Bounded-downside SIZE GATE. Only absorb a SMALL fragment (few nodes /
      short cable) into its neighbour. A tiny spur dangling off a tip is very
      often a true broken-off continuation, and — crucially — even if a small
      fragment is merged WRONGLY it carries little run-length, so the cost of
      an error is bounded by construction. We never join two large fragments
      (the gen-2 catastrophe), so the catastrophic downside is structurally
      removed regardless of geometry.

  (B) AMBIGUITY-MARGIN (Lowe-style ratio test). Accept a tip's join only when
      the chosen partner is UNAMBIGUOUSLY the closest differently-labelled
      fragment: the best gap must be much smaller than the distance from that
      tip to the NEXT-nearest differently-labelled fragment. At a crossing many
      unrelated neurites sit at similar distance (small margin → reject); a
      clean broken continuation has one near partner and empty space otherwise
      (large margin → accept). This is about the margin to the RUNNER-UP, which
      is a different signal from gen-2's reciprocity and from gen-1's tangent.

Both guards must pass. Either one alone bounds or rules out the gen-2 failure;
together they make a wrong merge both unlikely AND cheap.
"""

from __future__ import annotations

import numpy as np


# --- Tunable parameters -------------------------------------------------------
GAP_THRESHOLD_UM = 5.0      # hard cap on the gap we will ever bridge
MARGIN_RATIO = 0.5          # accept only if gap_best / gap_second_nearest < this
MARGIN_MIN_SECOND_UM = 4.0  # if the runner-up is farther than this, margin is satisfied
SMALL_FRAGMENT_MAX_NODES = 25  # only absorb fragments with <= this many nodes


def _fragment_node_counts(g):
    """Map fragment label -> number of graph nodes carrying that label.

    Uses ONLY the verified SkeletonGraph API exercised by dataset.py:
    ``list(g.nodes)`` and ``g.node_segment_id(node)`` (a method, returns the
    segment id as a string). No node_label array exists on this class.
    """
    counts: dict[str, int] = {}
    for n in g.nodes:
        lab = str(g.node_segment_id(n))
        counts[lab] = counts.get(lab, 0) + 1
    return counts


def propose_edits(sites, ctx) -> list[tuple]:
    """Selective split-repair: absorb small fragments via unambiguous joins.

    Returns a list of ``(label_a, label_b)`` merge tuples (legacy form, which the
    harness normalizes to ``merge_labels``). Empty list == make no changes.
    """
    if not sites:
        return []

    g = ctx.get("fragments_graph")
    if g is None:
        return []

    # --- Build a KD-tree over all node coordinates (microns) for the margin test.
    # node_xyz is an (N,3) micron array indexable by node id; node_segment_id is a
    # method. This mirrors dataset.candidate_split_sites exactly, so it is safe.
    try:
        from scipy.spatial import cKDTree as _KDTree
    except Exception:
        try:
            from scipy.spatial import KDTree as _KDTree
        except Exception:
            _KDTree = None

    all_nodes = list(g.nodes)
    node_arr = np.asarray(all_nodes)
    node_coords = np.asarray(g.node_xyz)[node_arr]
    node_labels = np.array([str(g.node_segment_id(n)) for n in all_nodes])

    label_counts = _fragment_node_counts(g)

    tree = _KDTree(node_coords) if _KDTree is not None else None

    def second_nearest_diff_label_gap(tip_xyz, label_a, label_b):
        """Distance from the tip to the nearest differently-labelled fragment
        OTHER than the chosen partner label_b. Large => unambiguous join.

        Returns +inf if no other differently-labelled fragment lies within the
        gap window (the cleanest possible case).
        """
        if tree is None:
            return float("inf")
        # Query a generous neighbourhood and scan for the closest node whose
        # label differs from BOTH label_a (the tip's own fragment) and label_b
        # (the already-chosen partner). That distance is the runner-up gap.
        radius = max(GAP_THRESHOLD_UM, MARGIN_MIN_SECOND_UM) * 3.0 + 1.0
        idxs = tree.query_ball_point(tip_xyz, r=radius)
        best = float("inf")
        for j in idxs:
            lj = node_labels[j]
            if lj == "0" or lj == label_a or lj == label_b:
                continue
            d = float(np.linalg.norm(node_coords[j] - tip_xyz))
            if d < best:
                best = d
        return best

    edits: list[tuple] = []
    seen_pairs: set = set()

    for s in sites:
        gap = float(s.gap_um)
        if gap > GAP_THRESHOLD_UM:
            continue

        la, lb = str(s.label_a), str(s.label_b)
        if la == lb:
            continue
        key = frozenset((la, lb))
        if key in seen_pairs:
            continue

        # (A) Bounded-downside size gate: at least one of the two fragments must
        # be SMALL. Absorbing a small fragment caps the cost of any error.
        na = label_counts.get(la, 0)
        nb = label_counts.get(lb, 0)
        if na == 0 or nb == 0:
            continue
        if min(na, nb) > SMALL_FRAGMENT_MAX_NODES:
            continue  # would risk fusing two large neurites — the gen-2 disaster

        # (B) Ambiguity-margin (Lowe ratio) at the TIP (node_a is always the tip).
        tip_xyz = np.asarray(g.node_xyz[s.node_a], dtype=float)
        second = second_nearest_diff_label_gap(tip_xyz, la, lb)
        # Accept only if the chosen partner is unambiguously closest: either the
        # runner-up is comfortably far, or the ratio gap/second is small.
        margin_ok = (second >= MARGIN_MIN_SECOND_UM) or (
            second > 0 and gap / second < MARGIN_RATIO
        )
        if not margin_ok:
            continue  # a crossing: many competitors at similar distance — decline

        edits.append((la, lb))
        seen_pairs.add(key)

    return edits
