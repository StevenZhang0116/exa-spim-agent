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

## Current criteria (Generation 1 — proximity + tangent agreement)

1. **Proximity.** Only consider a join when the physical gap between the tip
   (`node_a`) and its partner (`node_b`) is `<= GAP_THRESHOLD_UM` (4 µm).
2. **Tangent agreement (continuation, not crossing).** Estimate each end's local
   tangent by PCA over the rooted subgraph within `TANGENT_RADIUS_UM` (6 µm),
   oriented outward (toward the endpoint). Unify the labels only if:
   - the tip's outward tangent is aligned with the gap-bridging direction
     (`cos >= MIN_COS_COLLINEAR`, 0.80), AND
   - the tip and partner tangents are roughly anti-parallel
     (`cos(tan_a, -tan_b) >= 0.80`).
   This is the geometric signature of one severed process whose two ends face
   each other, and it rejects two unrelated neurites that merely cross nearby.

The proximity threshold is deliberately tight (4 µm) and paired with the
collinearity gate so this generation adds only high-precision merges over the
no-op baseline.

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
- **Gen 1 (2026-06-11):** Root cause of the gen-0 candidate's flat score — the
  policy was still the no-op seed (0 edits from 5000 candidates, Edge Accuracy
  79.8009 == baseline), so it could never improve. Replaced the body with the
  first real selective policy: proximity gate (gap <= 4 µm) **plus** a
  tangent-agreement gate (PCA tangents within 6 µm; tip aligned with the bridge
  and anti-parallel to the partner, cos >= 0.80). Rationale: the documented
  over-merge-at-crossings risk is exactly what blocks a naive proximity merge
  from beating baseline, so we add the collinearity test up front to keep only
  continuation-shaped joins. Returns legacy merge tuples.
- **Harness:** `candidate_split_sites` broadened from tip-to-tip to
  tip-to-any-node (tip/shaft/branch) within `max_gap_um`; `SplitSite` carries
  `node_a` (always a tip) and `node_b` (the partner, any degree). Enables
  tip-to-shaft / branch-point split repair and graph-based precision features.
