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

    GEN 3 POLICY (different lever than the rejected gen1 merge_labels attempt):
    the dominant error term on train is %Merged Edges (~9.09) vs a tiny %Split
    Edges (~0.17). So we attack MERGE errors with conservative ``split_label``
    edits, leaving SplitSites untouched (a no-op for the split-repair lever, which
    gen1 already showed buys nothing here). To survive the strict over-split
    watchdog (a split_label must NOT raise %Split / %Merged / #Merges), we emit a
    split ONLY on strong multi-signal geometric evidence, and — when the image is
    available — require an intensity VALLEY at the suspected cut to confirm two
    structures merely touch (rather than one continuous neuron).
    """
    edits = []

    g = None
    try:
        g = ctx.get("fragments_graph")
    except Exception:
        g = None

    reader = None
    try:
        reader = ctx.get("read_image_patch")
    except Exception:
        reader = None

    # Cheap geometric gates for a confident MERGE (two real neurites fused).
    # Both arms must be substantial cable; the junction must look non-collinear
    # (not one neuron passing straight through); and we prefer corroborating
    # caliber/topology signals per detector.
    MIN_ARM_UM = 12.0          # both arms must be real cable, not a spur
    MAX_ANGLE_DEG = 130.0      # < this => clearly NOT a straight pass-through
    MIN_RADIUS_RATIO = 1.5     # two different calibers fused
    VALLEY_RATIO_MAX = 0.7     # image dip confirming two touching structures

    for s in sites:
        kind = getattr(s, "kind", "split")

        # SplitSite (merge_labels): intentionally inert this generation — the
        # gen1 split-repair attempt netted +0.000, and %Split Edges is already
        # tiny, so there is no headroom on this lever here.
        if kind != "merge":
            continue

        # --- MergeSite: decide whether to emit a conservative split_label ---
        try:
            detector = getattr(s, "detector", None)
            angle = getattr(s, "angle_deg", None)
            rratio = getattr(s, "radius_ratio", None)
            cable_a = getattr(s, "cable_a_um", None)
            cable_b = getattr(s, "cable_b_um", None)
            branch_deg = getattr(s, "branch_degree", None)
        except Exception:
            continue

        # Both arms must be genuinely long — the hallmark of a real two-neuron
        # merge (vs. a short spur off a single neuron).
        if cable_a is None or cable_b is None:
            continue
        try:
            if float(cable_a) < MIN_ARM_UM or float(cable_b) < MIN_ARM_UM:
                continue
        except Exception:
            continue

        # Per-detector geometric evidence.
        geom_ok = False
        try:
            if detector == "component":
                # Disconnected pieces under one label + both arms long IS the
                # signal; angle is NaN here, so do not threshold on it.
                geom_ok = True
            elif detector in ("branch", "bridge"):
                sharp = False
                if angle is not None:
                    try:
                        a = float(angle)
                        # guard NaN
                        if a == a and a <= MAX_ANGLE_DEG:
                            sharp = True
                    except Exception:
                        sharp = False
                caliber = False
                if rratio is not None:
                    try:
                        if float(rratio) >= MIN_RADIUS_RATIO:
                            caliber = True
                    except Exception:
                        caliber = False
                # For a branch node a higher degree (X-crossing) is extra
                # evidence of a real crossing/touch.
                strong_branch = False
                try:
                    if branch_deg is not None and int(branch_deg) >= 4:
                        strong_branch = True
                except Exception:
                    strong_branch = False
                # Require a sharp angle, plus at least one corroborating signal
                # (caliber mismatch or a high-degree crossing) to be confident.
                geom_ok = sharp and (caliber or strong_branch)
        except Exception:
            geom_ok = False

        if not geom_ok:
            continue

        # Image confirmation (only for candidates that already passed geometry,
        # to respect the cloud-fetch cost rule). A real merge shows an intensity
        # VALLEY along the chord between the two arm seeds; one continuous neuron
        # does not. If the image is unavailable we fall back to geometry alone.
        if reader is not None:
            try:
                seed_a_node = getattr(s, "seed_a_node", None)
                seed_b_node = getattr(s, "seed_b_node", None)
                if seed_a_node is not None and seed_b_node is not None:
                    ev = reader.merge_cut_evidence(seed_a_node, seed_b_node)
                    vr = None
                    vpos = None
                    if ev is not None:
                        vr = ev.get("valley_ratio")
                        vpos = ev.get("valley_pos")
                    confirm = False
                    if vr is not None:
                        try:
                            if float(vr) <= VALLEY_RATIO_MAX:
                                # A central valley is the strongest evidence.
                                if vpos is None:
                                    confirm = True
                                else:
                                    try:
                                        p = float(vpos)
                                        confirm = 0.25 <= p <= 0.75
                                    except Exception:
                                        confirm = True
                        except Exception:
                            confirm = False
                    if not confirm:
                        # Image says continuous signal => one neuron => do NOT
                        # split. Skip this candidate.
                        continue
            except Exception:
                # Reader failed; fall back to the geometric decision.
                pass

        # Passed all gates — emit the conservative split_label.
        try:
            edits.append(s.as_edit())
        except Exception:
            continue

    return edits
