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

## Current criteria (Generation 0 — seed)

1. **Proximity.** Two fragment tips whose physical gap is below
   `GAP_THRESHOLD_UM` (5 µm) are assumed to be the same neuron and unified.

That is the entire seed theory. It is intentionally naive.

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
