# Proofreading Criteria (the evolved natural-language program)

> The evolution loop revises THIS file alongside `heuristics.py`. It is the
> human-readable statement of *why* the policy makes the decisions it does.
> Each criterion should correspond to logic in `heuristics.py::propose_edits`.
> When the agent revises the policy, it must keep this file in sync so a
> reviewer can read the current proofreading theory in plain language.

## Objective

Maximize run-length-weighted **Edge Accuracy** on held-out ground-truth
skeletons (ERL is the tie-breaker), by repairing BOTH error classes the
segmentation makes:

- **Split errors** — one true neuron broken into several fragments. Repair by
  **unifying** fragment label pairs (`merge_labels`), without joining fragments
  that belong to different neurons.
- **Merge errors** — two (or more) true neurons fused under one segment label.
  Repair by **splitting** that label by location (`split_label`), without
  over-splitting a single real neuron (which would raise % Split Edges).

Edge Accuracy = 100 − (% Split Edges + % Omit Edges + % Merged Edges), so a good
policy must lower split AND merge errors together while not trading one for the
other. The failure report shows both components plus an over-split watchdog.

## Current criteria (Generation 2)

Split-repair only (no `split_label` yet). A SplitSite is unified with
`merge_labels` when ALL of:

1. **Small gap** — `s.gap_um <= GAP_THRESHOLD_UM` (4.0 µm).
2. **Colinear continuation** — the arriving tip tangent, the gap bridge, and the
   best continuing branch on the partner are near-parallel
   (`cos >= MIN_COLINEAR_COS`, 0.94). This rejects tips that point *across* a gap
   rather than *along* it.
3. **Mutual nearest (reciprocity)** — the pairing must be (near-)best for BOTH
   endpoints. In a first pass we record, for every candidate endpoint node, the
   smallest gap it appears in across all SplitSites. A pair is accepted only if its
   own gap is within `RECIP_SLACK_UM` (0.75 µm) of that best-gap for BOTH `node_a`
   and `node_b`. If either endpoint has a strictly closer competing partner, this
   is a "passing cable" crossing (one tip with two suitors) and is skipped.

MergeSites are still left alone.

## Known failure modes to address (hypotheses for the loop)

Split-repair (SplitSite → `merge_labels`):
- **Over-merge at crossings.** Two unrelated neurites passing within a few µm get
  wrongly unified, creating a merge error. Candidate fix: require the two tips'
  local tangent directions to be roughly collinear (continuation, not crossing).
- **Under-repair of real gaps.** True splits with a gap slightly above threshold
  are missed. Candidate fix: allow larger gaps when direction agreement is high.

Merge-repair (MergeSite → `split_label`):
- **Over-split of one neuron.** Cutting at a genuine bifurcation of a SINGLE
  neuron breaks a real cable, raising % Split Edges. Candidate fix: only split
  when the branch looks like two distinct neurites — e.g. `angle_deg` far from
  180° (not a straight pass-through), `radius_ratio` far from 1 (different
  calibers), and both arms long (`cable_a_um`, `cable_b_um`).
- **Missed merges.** A real fusion left uncut keeps % Merged Edges high. The
  failure report lists the BASELINE merge labels (raw labels spanning ≥2 GT
  neurons on train) as concrete repair targets to aim a `split_label` at.

Both:
- **No image evidence.** Geometry alone ignores fluorescence. Candidate fix: read
  the image patch (`ctx["read_image_patch"]`, may be None) to test for a
  connecting signal across a gap, or an intensity valley at a suspected cut.

## Candidate space (what the policy gets to choose from)

The policy receives a UNIFIED stream of two site kinds (branch on `site.kind`):

- **SplitSite** (`kind == "split"`, from `dataset.candidate_split_sites`): for
  every fragment **tip** (`node_a`, degree 1), the nearest **differently-labelled**
  node within `max_gap_um` as `node_b` (tip, shaft, or branch — so tip-to-shaft
  and branch-point reconnections are candidates). One per unordered label pair.
- **MergeSite** (`kind == "merge"`, from `dataset.candidate_merge_sites`): a
  branch node (degree ≥ 3) inside ONE label where the two longest arms are BOTH
  long — the signature of two neurites fused at a touch/crossing. Carries the cut
  node, a seed deep in each arm, and advisory features (`branch_degree`,
  `angle_deg`, `radius_ratio`, `cable_a_um`, `cable_b_um`). All fields are derived
  from fragment geometry alone, so MergeSites are leak-free on held-out.

`ctx["n_split_sites"]` / `ctx["n_merge_sites"]` report the stream composition.

## Change log

- **Gen 0:** seed is a no-op (proposes nothing).
- **Gen 1 (DEAD END):** endpoint-adaptive colinearity threshold (tighter ~0.985
  cosine on tip-to-shaft/branch joins) plus a `MAX_RADIUS_RATIO` radius-continuity
  guard using `ctx["node_radius"]`. Scored **+0.000** on held-out vs parent —
  REJECTED. Do not revisit the colinearity-cosine threshold or a radius-continuity
  guard; that lever is exhausted.
- **Gen 2:** **mutual-nearest-neighbor (reciprocity) gating** — a genuinely
  different mechanism. Diagnosis from the Gen 1 train report: the gain came almost
  entirely from repairing OMIT edges (d%OmitEdges strongly negative on every
  skeleton) while `%MergedEdges` barely moved and `%SplitEdges` rose; 2 of the 51
  merges fused DISTINCT neurons (a tip latching onto a nearby *passing* cable that
  looked colinear across a tiny gap even though a better partner existed). Fix:
  in a first pass record the smallest candidate gap each endpoint node participates
  in, then accept a merge only if its gap is within `RECIP_SLACK_UM` of that best
  for BOTH endpoints. This removes the "one tip, two suitors" crossing over-merges
  without retuning colinearity or adding a radius guard. Geometry-only, so it is
  leak-free and applies on held-out.
- **Harness:** `candidate_split_sites` broadened from tip-to-tip to
  tip-to-any-node (tip/shaft/branch) within `max_gap_um`; `SplitSite` carries
  `node_a` (always a tip) and `node_b` (the partner, any degree).
- **Harness (merge repair):** added `MergeSite` + `candidate_merge_sites`
  (GT-free branch-based merge detection) and a unified split+merge candidate
  stream; fitness is Edge Accuracy (charges merge errors); the failure report now
  lists baseline merge targets and an over-split watchdog.
