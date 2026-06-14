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
GAP_THRESHOLD_UM = 4.0     # only fuse fragments whose tips are this close (microns)
TANGENT_RADIUS_UM = 12.0   # neighborhood radius used to estimate a tip's direction
MAX_TANGENT_ANGLE_DEG = 35.0  # tips must be near-collinear continuations to fuse


def _unit(v):
    import math
    n = math.sqrt(float(v[0]) * v[0] + v[1] * v[1] + v[2] * v[2])
    if n < 1e-9:
        return None
    return (v[0] / n, v[1] / n, v[2] / n)


def _tangent(g, node, radius):
    """Estimate the local outgoing direction at ``node`` from nearby graph nodes.

    Returns a unit vector pointing AWAY from the fragment interior (toward the
    tip), or None if it cannot be estimated. Robust to tip/shaft/branch nodes
    because it averages displacements to all neighbors within ``radius``.
    """
    try:
        xyz = g.node_xyz
        p0 = xyz[node]
        sub = g.rooted_subgraph(node, radius)
        acc = [0.0, 0.0, 0.0]
        cnt = 0
        for m in sub.nodes:
            if m == node:
                continue
            d = xyz[m] - p0
            u = _unit(d)
            if u is None:
                continue
            acc[0] += u[0]
            acc[1] += u[1]
            acc[2] += u[2]
            cnt += 1
        if cnt == 0:
            return None
        # mean direction from node into its fragment; the tip "continuation"
        # direction points opposite to that.
        mean = _unit(acc)
        if mean is None:
            return None
        return (-mean[0], -mean[1], -mean[2])
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

    GEN 1 POLICY: repair split errors only, conservatively. For each SplitSite we
    accept a ``merge_labels`` edit when BOTH fragment tips are very close (gap
    below GAP_THRESHOLD_UM) AND their local tangent directions are roughly
    collinear (the partner continues the first fragment rather than crossing it).
    This avoids over-merging unrelated neurites that merely pass nearby. We do NOT
    emit any ``split_label`` here: the failure report shows no raw label spans >=2
    GT neurons, so merge-correction targets are absent and a speculative split
    would only risk the over-split gate. MergeSites are flagged for review.
    """
    import math

    g = ctx.get("fragments_graph") if isinstance(ctx, dict) else None
    edits = []
    seen_pairs = set()

    for s in sites:
        kind = getattr(s, "kind", "split")
        if kind != "split":
            continue

        gap = getattr(s, "gap_um", None)
        if gap is None or gap > GAP_THRESHOLD_UM:
            continue

        la = getattr(s, "label_a", None)
        lb = getattr(s, "label_b", None)
        if la is None or lb is None or la == lb:
            continue
        key = tuple(sorted((str(la), str(lb))))
        if key in seen_pairs:
            continue

        # Tangent-collinearity test: require the partner to CONTINUE the tip's
        # trajectory, not merely pass near it. Skip if we cannot estimate.
        accept = False
        if g is not None:
            ta = _tangent(g, s.node_a, TANGENT_RADIUS_UM)
            tb = _tangent(g, s.node_b, TANGENT_RADIUS_UM)
            if ta is not None and tb is not None:
                # Two fragments continuing each other have OPPOSING outgoing
                # tangents, so their dot product should be near -1.
                dot = ta[0] * tb[0] + ta[1] * tb[1] + ta[2] * tb[2]
                dot = max(-1.0, min(1.0, dot))
                opposition_deg = math.degrees(math.acos(-dot))
                if opposition_deg <= MAX_TANGENT_ANGLE_DEG:
                    accept = True
        else:
            # No graph available: fall back to gap-only for very small gaps.
            if gap <= GAP_THRESHOLD_UM * 0.5:
                accept = True

        if accept:
            seen_pairs.add(key)
            edits.append(s.as_edit())

    return edits
