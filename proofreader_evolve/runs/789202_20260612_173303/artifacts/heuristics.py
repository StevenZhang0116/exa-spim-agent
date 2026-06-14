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

# --- Gen 7 policy parameters --------------------------------------------------
# Mechanism = STRICT GLOBAL EDIT BUDGET over a composite confidence ranking.
# Unlike the five rejected attempts (gen1 tangent / gen3 radius+image / gen4
# reciprocity were per-site gates so tight they fired 0 edits; gen5 was a loose
# per-site gap+degree gate with NO global cap, so MANY marginal edits all fired
# and the wrong ones dominated -> -1.604 over-merge), this policy scores EVERY
# viable tip-to-tip reattachment with one composite confidence number and then
# commits ONLY the single highest-confidence edit (budget = 1). A hard global
# budget means a couple of safe edits cannot be outvoted by many bad ones — the
# exact mechanism that sank gen5 — while still firing >0 (gen5 proved the pool
# is non-empty).
GEN7_MAX_GAP_UM = 4.0        # generous enough to admit candidates (gen1/3/4 gated to 0)
GEN7_EDIT_BUDGET = 1         # commit only the TOP-1 ranked reattachment
GEN7_MIN_CONFIDENCE = 0.55   # require the winner to clear a real confidence bar


def _unit(v):
    import math
    n = math.sqrt(float(v[0]) * v[0] + float(v[1]) * v[1] + float(v[2]) * v[2])
    if n <= 1e-9:
        return None
    return (v[0] / n, v[1] / n, v[2] / n)


def _tip_tangent(g, node):
    """Outgoing unit direction at a degree-1 tip: from its single neighbor to it.
    Returns None if anything is missing/degenerate."""
    try:
        nbrs = list(g.neighbors(node))
    except Exception:
        return None
    if not nbrs:
        return None
    try:
        p = g.node_xyz[node]
        q = g.node_xyz[nbrs[0]]
    except Exception:
        return None
    return _unit((p[0] - q[0], p[1] - q[1], p[2] - q[2]))


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

    GEN 7 POLICY — strict global edit budget over a composite confidence ranking.
    Stay on the merge_labels lever (split_label is a proven dead end here: no raw
    label spans >=2 GT neurons, so it only trips the over-split watchdog). For each
    SplitSite, build a single composite confidence score from a few cheap robust
    signals (small gap, BOTH endpoints true degree-1 terminals, tangent agreement),
    rank ALL viable reattachments, and emit only the top GEN7_EDIT_BUDGET of them.
    """
    import math

    g = None
    try:
        g = ctx.get("fragments_graph") if hasattr(ctx, "get") else None
    except Exception:
        g = None

    scored = []  # (confidence, label_a, label_b, edit)
    for s in sites:
        kind = getattr(s, "kind", "split")
        if kind != "split":
            # MergeSite -> split_label is a dead end in this split; never emit.
            continue

        # --- robustly-available fields (must not all be swallowed by guards) ---
        try:
            gap = float(getattr(s, "gap_um", float("nan")))
        except Exception:
            gap = float("nan")
        if not math.isfinite(gap) or gap < 0.0 or gap > GEN7_MAX_GAP_UM:
            continue

        la = getattr(s, "label_a", None)
        lb = getattr(s, "label_b", None)
        if la is None or lb is None or str(la) == str(lb):
            continue

        node_a = getattr(s, "node_a", None)
        node_b = getattr(s, "node_b", None)

        # --- composite confidence signals (each in [0,1], cheap & robust) -------
        # Signal 1: gap closeness — closer breaks are more confident.
        c_gap = max(0.0, 1.0 - gap / GEN7_MAX_GAP_UM)

        # Signal 2: BOTH endpoints true degree-1 terminals (a clean break, not a
        # reattachment onto the middle/branch of an unrelated cable). node_a is
        # always a tip; node_b may be tip/shaft/branch — reward a true tip pair.
        deg_b = None
        if g is not None and node_b is not None:
            try:
                deg_b = int(g.degree[node_b])
            except Exception:
                deg_b = None
        if deg_b is None:
            c_term = 0.5            # unknown: neutral, do not over-credit
        elif deg_b == 1:
            c_term = 1.0            # clean tip-to-tip break
        elif deg_b == 2:
            c_term = 0.4            # tip-to-shaft: plausible but riskier
        else:
            c_term = 0.0            # tip-to-branch: most over-merge-prone

        # Signal 3: tangent agreement — the two fragments should continue roughly
        # along the same line (continuation), not cross. Only computable when both
        # ends are tips; otherwise neutral so the term/gap signals still gate.
        c_dir = 0.5
        if g is not None and deg_b == 1 and node_a is not None and node_b is not None:
            ta = _tip_tangent(g, node_a)
            tb = _tip_tangent(g, node_b)
            if ta is not None and tb is not None:
                # Two opposing tips that continue each other have nearly antiparallel
                # outgoing tangents: dot ~ -1. Map [-1,1] so antiparallel -> 1.
                dot = ta[0] * tb[0] + ta[1] * tb[1] + ta[2] * tb[2]
                c_dir = max(0.0, min(1.0, (-dot + 1.0) / 2.0))

        # Composite: weight a clean terminal break and directional continuation
        # heavily, gap closeness moderately. All terms in [0,1] -> confidence in [0,1].
        confidence = 0.30 * c_gap + 0.40 * c_term + 0.30 * c_dir

        if confidence < GEN7_MIN_CONFIDENCE:
            continue

        try:
            edit = s.as_edit()
        except Exception:
            edit = (la, lb)
        scored.append((confidence, str(la), str(lb), edit))

    if not scored:
        return []

    # --- STRICT GLOBAL BUDGET: commit only the top-N highest-confidence edits.
    # This is the lever that distinguishes gen7 from gen5: a few bad-but-passing
    # candidates can no longer outweigh the safe ones, because at most
    # GEN7_EDIT_BUDGET edits land at all, and only the most confident.
    scored.sort(key=lambda t: t[0], reverse=True)

    edits = []
    used_labels = set()
    for conf, la, lb, edit in scored:
        if la in used_labels or lb in used_labels:
            continue  # one reattachment per label — avoid chaining mega-labels
        edits.append(edit)
        used_labels.add(la)
        used_labels.add(lb)
        if len(edits) >= GEN7_EDIT_BUDGET:
            break
    return edits
