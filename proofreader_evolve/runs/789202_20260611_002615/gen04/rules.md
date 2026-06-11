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

## Current criteria (Generation 3 — bounded-downside size gate + ambiguity margin)

A candidate split site (a fragment tip `node_a` near a differently-labelled
partner `node_b`) is unified into a `merge_labels` edit ONLY if ALL hold:

1. **Proximity.** Physical gap `gap_um <= GAP_THRESHOLD_UM` (5 µm). Hard cap.
2. **Bounded-downside size gate.** At least one of the two fragments is SMALL
   (`<= SMALL_FRAGMENT_MAX_NODES`, 25 nodes). We only ever ABSORB a small
   fragment into a larger neighbour; we never join two large fragments. Tiny
   spurs at a tip are usually true broken-off continuations, and — decisively —
   even a WRONG merge of a small fragment costs little run-length-weighted Edge
   Accuracy, so the catastrophic downside is removed by construction.
3. **Ambiguity margin (Lowe-style ratio).** The chosen partner must be
   unambiguously the closest differently-labelled fragment to the tip: the
   distance to the NEXT-nearest *different* fragment must be comfortably far
   (`>= MARGIN_MIN_SECOND_UM`, 4 µm) OR `gap_best / gap_second < MARGIN_RATIO`
   (0.5). At a crossing many unrelated neurites sit at similar distance (small
   margin → reject); a clean broken continuation has empty space around it
   (large margin → accept). Second-nearest distances come from a KD-tree over
   `g.node_xyz` (microns), scanning labels via `g.node_segment_id`.

The size gate bounds the COST of an error; the margin test reduces the
PROBABILITY of an error at crossings. Both are required.

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
- **Gen 1 (REJECTED, held-out +0.000):** proximity + PCA tangent-direction
  agreement / collinearity. Geometry direction did not separate good from bad
  joins; no net improvement, reverted to the no-op seed.
- **Gen 2 (REJECTED, held-out -2.808):** mutual nearest-neighbour tip-to-tip
  pairing (both degree-1 tips, gap <= 4 µm, mutual-NN uniqueness, union-find
  component-size cap of 3). It OVER-MERGED and actively HURT. Lesson: fitness is
  run-length-weighted, so a single wrong merge fusing two LONG neurites destroys
  a large fraction of correctly-traced path length and dwarfs many correct small
  repairs. Reciprocity + small gap does not rule out two unrelated neurites
  crossing.
- **Gen 3 (this revision):** abandon recall-style tip pairing. Replace it with
  two structurally-distinct PRECISION/BOUNDED-DOWNSIDE guards (both required):
  (A) a **size gate** that only ABSORBS a small fragment (<= 25 nodes), so a
  wrong merge carries little run-length and the gen-2 catastrophe (fusing two
  large neurites) is impossible by construction; and (B) a **Lowe-style
  ambiguity-margin ratio test** — accept only when the chosen partner is
  unambiguously the tip's closest differently-labelled fragment (runner-up far
  away or `gap/second < 0.5`), which rejects crossings where many competitors
  sit at similar distance. Distinct from gen-1 (no tangent/PCA) and gen-2 (no
  reciprocity); aims to bound BOTH the probability and the cost of a bad merge.
