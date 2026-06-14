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
    (>>1 = two different calibers fused), ``cable_a_um``/``cable_b_um``, and
    ``detector`` ("branch" | "bridge" | "component" — the topology that found it;
    condition on it since each has different evidence). From
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

This seed version is a deliberate NO-OP (proposes nothing) so the first generation
starts exactly at the no-edit baseline and has obvious headroom to improve.
"""

from __future__ import annotations


# --- Tunable parameters (the agent may rewrite these and the logic below) -----
# The seed is INTENTIONALLY a no-op: it proposes zero edits, so its score equals
# the no-edit baseline. A "merge everything within N µm" seed scores ~10 points
# BELOW baseline (it chains thousands of pairs into brain-spanning mega-labels via
# union-find, manufacturing merges and inflating per-neuron label counts), which
# left the evolution loop unable to climb: every revision had to beat baseline in
# one step from a deeply sub-baseline start, so all were reverted. Starting AT
# baseline means a generation only needs to find a SINGLE net-positive edit to be
# kept — improvements then accumulate generation over generation.
GAP_THRESHOLD_UM = 5.0   # available for the agent's first real policy; unused by the no-op seed

# --- Mutual-nearest-tip merge policy parameters -------------------------------
# We only repair SPLIT errors (emit merge_labels), never split_label, so the
# acceptance over-split watchdog can never fire on us.
MNN_GAP_UM = 3.0          # only fuse fragments that nearly touch
MNN_RADIUS_RATIO_MAX = 2.0  # arms must be of similar caliber (continuity of cable)


def _node_degree(g, n):
    """Degree of node n, defensively (returns None on any failure)."""
    try:
        return int(g.degree[n])
    except Exception:
        try:
            return int(len(list(g.neighbors(n))))
        except Exception:
            return None


def _radius_at(ctx, n):
    """Estimated neurite radius at node n, or None if unavailable."""
    try:
        rad = ctx.get("node_radius")
        if rad is None:
            return None
        v = rad[n]
        if v is None:
            return None
        fv = float(v)
        if fv != fv or fv <= 0.0:  # NaN / nonpositive guard
            return None
        return fv
    except Exception:
        return None


def propose_edits(sites, ctx) -> list:
    """Decide which candidate sites to repair, returning a list of edits.

    SEED POLICY: propose NOTHING (return []). This scores exactly at the no-edit
    baseline — a safe floor the agent can only improve on. The agent's job is to
    replace the body below with a *selective* policy that proposes only high-
    confidence edits, dispatching on each site's ``kind``:

      - for a SplitSite: emit ``merge_labels`` only when the gap is small AND the
        tangents agree AND a component-size cap holds (use ctx["fragments_graph"]
        + node_a/node_b);
      - for a MergeSite: emit ``split_label`` only when the branch genuinely fuses
        two real neurites (e.g. both arms long, sharp angle_deg, or radius_ratio
        far from 1) — and NEVER drive % Split Edges up by over-splitting one neuron.

    The dispatch skeleton (kept here, but inert because the seed returns early) is
    the shape a real policy should follow — it is crash-safe on BOTH site kinds:

        edits = []
        for s in sites:
            kind = getattr(s, "kind", "split")
            if kind == "split":
                # decide using s.label_a, s.label_b, s.gap_um, s.node_a, s.node_b
                # if accept: edits.append(s.as_edit())            # merge tuple
                pass
            elif kind == "merge":
                # decide using s.label, s.branch_degree, s.angle_deg,
                #              s.radius_ratio, s.cable_a_um, s.cable_b_um
                # if accept: edits.append(s.as_edit())            # split_label dict
                pass
        return edits

    Returns a list of edits — legacy (label_a, label_b) tuples or typed dicts; see
    the module docstring. An empty list means "make no changes".

    POLICY (mutual-nearest-tip split-repair): on the train/acceptance split there
    are NO GT-confirmable merge errors, so any split_label cuts a real neuron and
    trips the over-split watchdog. We therefore emit ONLY merge_labels edits, and
    select them by a reciprocity principle: fuse a fragment pair only when each
    endpoint is a genuine fragment TIP (degree 1) AND the two tips are each other's
    MUTUAL nearest cross-label partner within a tiny gap AND their cable calibers
    are similar. Mutual tip-to-tip reciprocity is the geometric signature of a true
    broken cable (a clean break leaves two free ends pointing at each other), and it
    specifically excludes crossing/parallel neurites — at a crossing a tip's nearest
    partner is a SHAFT or BRANCH point, not another reciprocating tip, so the
    reciprocity test fails and we leave it alone (no over-merge).
    """
    try:
        g = ctx.get("fragments_graph") if hasattr(ctx, "get") else None
    except Exception:
        g = None

    # Pass 1: collect candidate SplitSites that pass cheap geometric filters and
    # record, per fragment TIP node, its best (smallest-gap) cross-label partner.
    # We key on (label, node) to detect reciprocity below.
    best_partner = {}   # node_a -> (gap, node_b, site)
    candidates = []
    for s in sites:
        try:
            kind = getattr(s, "kind", "split")
        except Exception:
            kind = "split"
        if kind != "split":
            # MergeSites repair merge errors via split_label; intentionally skipped
            # because the acceptance split has no GT-confirmable merges (watchdog).
            continue
        try:
            gap = float(getattr(s, "gap_um", float("nan")))
        except Exception:
            continue
        if gap != gap or gap > MNN_GAP_UM:   # NaN or too-far guard
            continue
        try:
            na = getattr(s, "node_a", None)
            nb = getattr(s, "node_b", None)
            la = getattr(s, "label_a", None)
            lb = getattr(s, "label_b", None)
        except Exception:
            continue
        if na is None or nb is None or la is None or lb is None:
            continue
        # Require BOTH endpoints to be fragment tips (degree 1). This is the key
        # restriction: it removes tip-to-shaft / branch-point reconnections, which
        # are exactly the crossing/parallel cases that cause over-merges.
        if g is not None:
            da = _node_degree(g, na)
            db = _node_degree(g, nb)
            if da is None or db is None or da != 1 or db != 1:
                continue
        else:
            # Without a graph we cannot verify tip status; stay safe and skip.
            continue
        # Caliber similarity: a true broken cable has matching radius on both ends.
        ra = _radius_at(ctx, na)
        rb = _radius_at(ctx, nb)
        if ra is not None and rb is not None:
            ratio = max(ra, rb) / min(ra, rb)
            if ratio != ratio or ratio > MNN_RADIUS_RATIO_MAX:
                continue
        # Record this tip's best partner (smallest gap wins).
        prev = best_partner.get(na)
        if prev is None or gap < prev[0]:
            best_partner[na] = (gap, nb, s)
        candidates.append((gap, na, nb, la, lb, s))

    # Pass 2: keep only MUTUAL nearest pairs — na's best partner is nb AND nb's
    # best partner is na. This reciprocity is what makes the fuse safe.
    edits = []
    seen_pairs = set()
    for gap, na, nb, la, lb, s in candidates:
        ba = best_partner.get(na)
        bb = best_partner.get(nb)
        if ba is None or bb is None:
            continue
        if ba[1] != nb or bb[1] != na:
            continue
        # Deduplicate unordered label pair.
        key = tuple(sorted((str(la), str(lb))))
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        try:
            edits.append(s.as_edit())
        except Exception:
            edits.append((la, lb))
    return edits
