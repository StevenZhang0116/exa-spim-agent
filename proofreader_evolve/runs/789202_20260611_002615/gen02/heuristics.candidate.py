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


# --- Tunable parameters (the agent may rewrite these and the logic below) -----
# The seed is INTENTIONALLY a no-op: it proposes zero edits, so its score equals
# the no-edit baseline. A "merge everything within N µm" seed scores ~10 points
# BELOW baseline (it chains thousands of pairs into brain-spanning mega-labels via
# union-find, manufacturing merges and inflating per-neuron label counts), which
# left the evolution loop unable to climb: every revision had to beat baseline in
# one step from a deeply sub-baseline start, so all were reverted. Starting AT
# baseline means a generation only needs to find a SINGLE net-positive merge to be
# kept — improvements then accumulate generation over generation.
GAP_THRESHOLD_UM = 4.0   # only consider tip-to-tip joins with a gap at/below this (microns)
MAX_COMPONENT_SIZE = 3   # union-find component-size cap to avoid runaway chaining


def _degree(g, node):
    """Robust node-degree lookup for a networkx-style graph."""
    try:
        return g.degree[node]
    except (TypeError, KeyError):
        try:
            return g.degree(node)
        except Exception:
            return None


def propose_edits(sites, ctx) -> list[tuple]:
    """Repair split errors via MUTUAL nearest-neighbor tip pairing.

    Structurally distinct signal (NOT geometry/tangent based — that was tried in
    gen1 and rejected on held-out). We accept a join only when it is a clean,
    *unambiguous* topological pairing:

      1. **Tip-to-tip only.** Both endpoints must be genuine fragment tips
         (degree 1). Tip-to-shaft / tip-to-branch candidates are rejected, since
         joining into a branch/shaft is the structure most prone to merge errors.
      2. **Small gap.** The physical gap must be <= GAP_THRESHOLD_UM.
      3. **Mutual nearest neighbor.** Among all qualifying tip-to-tip sites, each
         tip's single closest qualifying partner must be the other tip (and vice
         versa). This uniqueness test rejects ambiguous junctions where several
         tips compete for the same partner (a hallmark of crossings).
      4. **Component-size cap.** A union-find pass accepts each mutual pair only
         while the merged component stays at/below MAX_COMPONENT_SIZE labels,
         preventing chains from collapsing many fragments into a mega-label.

    Returns a list of (label_a, label_b) merge tuples.
    """
    g = ctx.get("fragments_graph") if isinstance(ctx, dict) else None
    max_gap = GAP_THRESHOLD_UM
    if isinstance(ctx, dict) and ctx.get("max_gap_um") is not None:
        max_gap = min(max_gap, float(ctx["max_gap_um"]))

    # 1+2: keep only tip-to-tip sites within the gap threshold.
    qualifying = []
    for s in sites:
        gap = getattr(s, "gap_um", None)
        if gap is None or gap > max_gap:
            continue
        if g is not None:
            da = _degree(g, s.node_a)
            db = _degree(g, s.node_b)
            # node_a is always a tip; require node_b be a tip too.
            if db is not None and db != 1:
                continue
            if da is not None and da != 1:
                continue
        qualifying.append(s)

    if not qualifying:
        return []

    # 3: mutual nearest-neighbor test, keyed on graph node ids.
    # For each tip node, find its smallest-gap qualifying site.
    best_site_for_node = {}
    for s in qualifying:
        for node in (s.node_a, s.node_b):
            cur = best_site_for_node.get(node)
            if cur is None or s.gap_um < cur.gap_um:
                best_site_for_node[node] = s

    mutual = []
    seen = set()
    for s in qualifying:
        key = (s.node_a, s.node_b)
        if key in seen:
            continue
        seen.add(key)
        if (best_site_for_node.get(s.node_a) is s
                and best_site_for_node.get(s.node_b) is s):
            mutual.append(s)

    if not mutual:
        return []

    # Accept closest mutual pairs first.
    mutual.sort(key=lambda s: s.gap_um)

    # 4: union-find with component-size cap (counting distinct labels).
    parent = {}
    size = {}

    def find(x):
        parent.setdefault(x, x)
        size.setdefault(x, 1)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    edits = []
    for s in mutual:
        a, b = s.label_a, s.label_b
        if a == b:
            continue
        ra, rb = find(a), find(b)
        if ra == rb:
            continue
        if size[ra] + size[rb] > MAX_COMPONENT_SIZE:
            continue
        # union by size
        if size[ra] < size[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        size[ra] += size[rb]
        edits.append((a, b))

    return edits
