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
    """
    return []
