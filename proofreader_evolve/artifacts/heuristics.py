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

Contract (keep this stable so the harness can always call it):
    propose_edits(sites, ctx) -> list[tuple[str, str]]

  sites : list[SplitSite]   # from dataset.candidate_split_sites
  ctx   : dict              # free-form context the harness provides, e.g.
                            #   ctx["max_gap_um"], ctx["fragments_graph"]
  return: list of (label_a, label_b) fragment-label pairs to unify.

This seed version is deliberately simple — a single distance threshold — so the
first generation has obvious, measurable failure modes for the agent to improve.
"""

from __future__ import annotations


# --- Tunable parameters (the agent may rewrite these and the logic below) -----
GAP_THRESHOLD_UM = 5.0   # unify tips closer than this; smaller = more conservative


def propose_edits(sites, ctx) -> list[tuple]:
    """Decide which candidate split sites to repair by unifying their labels.

    Seed policy: accept any candidate whose tip-to-tip gap is below a fixed
    distance threshold. This will repair many true splits but also over-merge
    where two genuinely different neurites pass close to each other — exactly
    the failure the evolution loop should teach it to avoid (e.g. by adding
    direction/continuity/radius checks, or reading the image patch).
    """
    edits = []
    seen = set()
    for s in sites:
        if s.gap_um <= GAP_THRESHOLD_UM:
            key = frozenset(s.as_edit())
            if key not in seen:
                seen.add(key)
                edits.append(s.as_edit())
    return edits
