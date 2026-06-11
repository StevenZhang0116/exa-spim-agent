# Proofreading Criteria (the evolved natural-language program)

> The evolution loop revises THIS file alongside `heuristics.py`. It is the
> human-readable statement of *why* the policy makes the decisions it does.
> Each criterion should correspond to logic in `heuristics.py::propose_edits`.
> When the agent revises the policy, it must keep this file in sync so a
> reviewer can read the current proofreading theory in plain language.

## Objective

Repair **split errors** (one true neuron broken into several fragments) by
unifying fragment label pairs, **without introducing merge errors** (joining
fragments that belong to different neurons). Fitness is run-length-weighted
**Edge Accuracy** on held-out ground-truth skeletons; ERL is the tie-breaker.

## Current criteria (Mutual nearest-neighbor tip pairing)

A candidate split is repaired (labels unified) ONLY if all of the following hold:

1. **Tip-to-tip only.** Both endpoints are genuine fragment tips (degree 1).
   Tip-to-shaft and tip-to-branch candidates are rejected, because joining into
   the middle of an existing branch/shaft is the structure most likely to create
   a merge error.
2. **Proximity.** The physical gap is `<= GAP_THRESHOLD_UM` (4 µm, clamped to
   `ctx["max_gap_um"]`).
3. **Mutual nearest neighbor.** Among all qualifying tip-to-tip sites, each tip's
   single closest qualifying partner must be the *other* tip, and vice versa.
   This uniqueness/topology test rejects ambiguous junctions where multiple tips
   compete for one partner (a hallmark of crossing neurites), without using any
   geometric direction estimate.
4. **Component-size cap.** Accepted pairs are applied via union-find; a pair is
   skipped if it would grow the merged component beyond `MAX_COMPONENT_SIZE` (3)
   distinct labels, preventing runaway chaining into mega-labels.

## Known failure modes to address (hypotheses for the loop)

- **Over-merge at crossings.** Two unrelated neurites passing within 5 µm get
  wrongly unified, creating a merge error. Candidate fix: require the two tips'
  local tangent directions to be roughly collinear (continuation, not crossing).
- **Under-repair of real gaps.** True splits with a gap slightly above threshold
  are missed. Candidate fix: allow larger gaps when direction agreement is high.
- **No image evidence.** Distance alone ignores whether fluorescence actually
  connects the tips. Candidate fix: read the image patch between the tips and
  test for a connecting signal.

## Candidate space (what the policy gets to choose from)

`dataset.candidate_split_sites` enumerates, for every fragment **tip**
(`node_a`, degree 1), the nearest **differently-labelled** node within
`max_gap_um` as `node_b`. `node_b` may be a tip, a shaft (degree 2), or a branch
(degree 3+) — so **tip-to-shaft and branch-point reconnections are candidates**,
not just tip-to-tip. One candidate is kept per unordered label pair (closest gap).
This broadens recall (many real splits are tip-to-shaft); the price is more,
noisier candidates, so precision now rests on the policy's per-site checks.

## Change log

- **Gen 0:** seed proximity rule (5 µm).
- **Harness:** `candidate_split_sites` broadened from tip-to-tip to
  tip-to-any-node (tip/shaft/branch) within `max_gap_um`; `SplitSite` carries
  `node_a` (always a tip) and `node_b` (the partner, any degree). Enables
  tip-to-shaft / branch-point split repair and graph-based precision features.
- **Gen 1 (REJECTED on held-out, +0.000):** proximity + PCA tangent-agreement /
  collinearity rule — estimated local tangents via PCA over the rooted subgraph
  and required the tip tangent to align with the gap-bridging direction and be
  anti-parallel to the partner tangent. Did not beat the parent, so it was
  reverted to the no-op seed. Geometric direction agreement is not a productive
  signal here and will not be revisited.
- **Gen 2 (this change):** Replaced the no-op seed with a **topology/uniqueness**
  signal instead of geometry. Accept a join only for **tip-to-tip** pairs (both
  degree 1) with gap `<= 4 µm` that are **mutual nearest neighbors** (each tip's
  closest qualifying partner is the other), applied with a **union-find
  component-size cap of 3** to block runaway chaining. Rationale: the prior
  geometry test was rejected; ambiguity at crossings is better captured by
  pairing uniqueness than by direction, and the degree gate + size cap keep
  precision high (no merges into branch/shaft, no mega-labels) so even a few
  accepted joins can be net-positive over baseline.
