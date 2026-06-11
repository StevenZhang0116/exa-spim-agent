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

Gen 4 increment (this revision): gen-3 was ACCEPTED (+1.0 Edge Acc, all 8
skeletons up) but its per-pair dedup had NO transitive guard, so one small
fragment could be absorbed into several partners and chains of fragments could
fuse into a single multi-label class (e.g. 375149638+746076720+746076722). Such
a fused class spans a long path, so run-length-weighted ERL collapsed on several
skeletons (N016 -13934, N020 -8859) even as Edge Accuracy rose. Gen 4 adds:

  (C) TRANSITIVE-FUSION GUARD. Admit surviving candidates MOST-CONFIDENT-FIRST
      (ascending gap) through a union-find, refusing any merge that would make a
      merged class exceed MERGE_LABEL_CAP (=2) raw labels. Each accepted class
      therefore contains exactly two labels; no chain / multi-absorption blob can
      form. This preserves the bulk of gen-3's good two-label repairs while
      removing the long-span over-merges that bled ERL.

Gen 5 increment (this revision): gen-4's (C) guard removed the 3-label chains
(edits 769 -> 727) but Edge Accuracy and the big ERL losses (N016 -13934, N020
-8859, N010 -2059) were essentially unchanged, because the worst over-merges are
exactly-TWO-label fusions that (C) cannot catch. The report flags 553856478+
744503863 at FIVE well-separated world locations (plus 673589912+673589916,
659384254+659601729, 375815390+391547478). Two fragments that run close at
multiple separated points are PARALLEL / fasciculating neurites (or one wrapping
the other), NOT a single severed process. Guards (A)/(B) only inspect the single
tip, so they accept these. Gen 5 adds:

  (B2) SINGLE-POINT-OF-APPROACH guard. For each candidate surviving (A)+(B),
       gather all of the partner's nodes within CONTACT_RADIUS_UM of any node of
       the tip's fragment, and require every such contact to lie within
       CONTACT_SPREAD_MAX_UM of the tip contact. A genuine severed continuation
       touches at one localized place; parallel neurites contact at several
       separated places, so the contact set is spatially spread and the merge is
       rejected. This kills the flagged multi-contact pairs while preserving the
       single-contact repairs that make up the bulk of the 727 edits. Guards (A),
       (B) and (C) are unchanged.
"""

from __future__ import annotations

import numpy as np


# --- Tunable parameters -------------------------------------------------------
GAP_THRESHOLD_UM = 5.0      # hard cap on the gap we will ever bridge
MARGIN_RATIO = 0.5          # accept only if gap_best / gap_second_nearest < this
MARGIN_MIN_SECOND_UM = 4.0  # if the runner-up is farther than this, margin is satisfied
SMALL_FRAGMENT_MAX_NODES = 25  # only absorb fragments with <= this many nodes
MERGE_LABEL_CAP = 2            # a merged class may contain at most this many raw
                               # labels — forbids transitive multi-label fusion
CONTACT_RADIUS_UM = 5.0        # radius for gathering inter-fragment contacts (~gap cap)
CONTACT_SPREAD_MAX_UM = 6.0    # reject if any contact is farther than this from the
                               # tip contact — signals a SECOND approach region
                               # (parallel / fasciculating neurites, not a single gap)


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

    # Precompute label -> array of node-row indices (into node_coords), so the
    # multi-contact guard can gather a fragment's nodes without rescanning.
    label_to_idx: dict[str, list] = {}
    for i, lab in enumerate(node_labels):
        label_to_idx.setdefault(lab, []).append(i)
    label_to_idx = {k: np.asarray(v, dtype=int) for k, v in label_to_idx.items()}

    def single_point_of_approach(la, lb, tip_xyz):
        """True iff fragments la and lb touch at ONE localized place near the tip.

        A genuine split-continuation is severed at a single point: the two
        fragments come close only in a small region around the tip gap. Parallel /
        fasciculating neurites (or one wrapping the other) instead run close at
        SEVERAL well-separated places. We gather every node of `lb` lying within
        CONTACT_RADIUS_UM of ANY node of `la` and check that all such contacts sit
        within CONTACT_SPREAD_MAX_UM of the tip contact; if any contact is farther,
        there is a second approach region and we reject the merge.
        """
        if tree is None:
            return True  # cannot test -> defer to A/B/C as before
        idx_a = label_to_idx.get(la)
        idx_b = label_to_idx.get(lb)
        if idx_a is None or idx_b is None:
            return True
        coords_a = node_coords[idx_a]
        # Collect lb-node indices that are within CONTACT_RADIUS_UM of some la node.
        contact_idx: set = set()
        for ca in coords_a:
            for j in tree.query_ball_point(ca, r=CONTACT_RADIUS_UM):
                if node_labels[j] == lb:
                    contact_idx.add(j)
        if not contact_idx:
            return True  # no measurable contact set -> rely on the gap test
        contact_coords = node_coords[np.fromiter(contact_idx, dtype=int)]
        # All contacts must lie close to the tip contact; otherwise a second,
        # spatially-separated approach region exists -> parallel adjacency.
        dists = np.linalg.norm(contact_coords - tip_xyz[None, :], axis=1)
        return float(dists.max()) <= CONTACT_SPREAD_MAX_UM

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

    # --- (C) TRANSITIVE-FUSION GUARD via union-find over accepted labels. -----
    # Gen 3 deduplicated by label pair but had NO guard against CHAINING: a
    # single small fragment could be absorbed into multiple partners, and chains
    # of small fragments could fuse into one multi-label blob (the report's
    # 375149638+746076720+746076722 case). Such a fused class spans a long path,
    # so run-length-weighted ERL collapses even when Edge Accuracy nudges up.
    #
    # Fix: process candidates MOST-CONFIDENT-FIRST (ascending gap) and admit each
    # pair only if it does NOT grow any merged component beyond MERGE_LABEL_CAP
    # labels. A label that is already in a merged component cannot be merged
    # again, which directly prevents both the multi-absorption and the transitive
    # chain. Cap = 2: every accepted merge joins exactly two raw labels into one
    # class, never three+, so no over-merged blob can form.
    parent: dict[str, str] = {}
    comp_label_count: dict[str, int] = {}

    def _find(x: str) -> str:
        parent.setdefault(x, x)
        comp_label_count.setdefault(x, 1)
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    def _component_size(x: str) -> int:
        return comp_label_count[_find(x)]

    def _union(a: str, b: str) -> None:
        ra, rb = _find(a), _find(b)
        if ra == rb:
            return
        parent[rb] = ra
        comp_label_count[ra] = comp_label_count[ra] + comp_label_count[rb]

    # Evaluate guards (A) and (B) per site first; collect surviving candidates
    # with their gap so we can admit them in confidence order through union-find.
    survivors: list[tuple[float, str, str]] = []
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
        seen_pairs.add(key)

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

        # (B2) SINGLE-POINT-OF-APPROACH guard. A true severed continuation touches
        # its partner at ONE localized place (around the tip gap). The flagged
        # 553856478+744503863-style pairs instead run close at SEVERAL separated
        # locations — that is two parallel / fasciculating neurites, and merging
        # them fuses distinct cells and devastates run-length-weighted ERL.
        # (A)+(B) cannot see this: both only inspect the single tip. We reject if
        # the inter-fragment contact set is spatially spread beyond the tip region.
        if not single_point_of_approach(la, lb, tip_xyz):
            continue

        survivors.append((gap, la, lb))

    # Most-confident-first: smallest gap merges are committed before noisier ones
    # get a chance to chain onto them.
    survivors.sort(key=lambda t: t[0])

    edits: list[tuple] = []
    for gap, la, lb in survivors:
        # (C) Transitive-fusion guard. Reject if either label already belongs to
        # a merged component, or if uniting them would exceed the label cap.
        if _component_size(la) + _component_size(lb) > MERGE_LABEL_CAP:
            continue
        _union(la, lb)
        edits.append((la, lb))

    return edits
