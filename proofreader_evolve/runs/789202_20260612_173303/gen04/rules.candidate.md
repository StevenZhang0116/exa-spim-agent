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

## Current criteria (mutual-nearest-tip split-repair)

The policy emits ONLY `merge_labels` edits (no `split_label`), because on the
train/acceptance split there are **no GT-confirmable merge errors** — so any
`split_label` there necessarily cuts a real neuron and trips the over-split
watchdog. We therefore repair split errors only, selecting fuse pairs by a
**reciprocity** rule that no prior attempt used:

1. **Both endpoints are fragment TIPS** (graph degree == 1). This removes
   tip-to-shaft and branch-point reconnections, which are exactly the
   crossing/parallel-neurite cases that cause over-merges.
2. **Tiny gap** (`gap_um <= MNN_GAP_UM`, 3.0 µm): a clean cable break leaves the
   two free ends nearly touching.
3. **Caliber continuity** (`max(r_a,r_b)/min(r_a,r_b) <= MNN_RADIUS_RATIO_MAX`,
   2.0) when `node_radius` is available — a real broken cable has matching
   thickness on both ends.
4. **Mutual nearest neighbour**: keep the pair only if tip A's smallest-gap
   cross-label partner is tip B *and* tip B's is tip A. Reciprocity is the
   geometric signature of one cable broken into two — at a crossing a tip's
   nearest partner is a shaft/branch point (not a reciprocating tip), so the test
   fails and we leave it alone.

Each unordered label pair is fused at most once. Because the return contains no
`split_label`, the acceptance over-split watchdog can never reject this policy;
its only risk is over-merging, which the tip+reciprocity+caliber gates guard
against.

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
- **Harness:** `candidate_split_sites` broadened from tip-to-tip to
  tip-to-any-node (tip/shaft/branch) within `max_gap_um`; `SplitSite` carries
  `node_a` (always a tip) and `node_b` (the partner, any degree).
- **Harness (merge repair):** added `MergeSite` + `candidate_merge_sites`
  (GT-free branch-based merge detection) and a unified split+merge candidate
  stream; fitness is Edge Accuracy (charges merge errors); the failure report now
  lists baseline merge targets and an over-split watchdog.
- **Gen 5 (mutual-nearest-tip split-repair):** replaced the no-op seed with a
  pure `merge_labels` policy. Rationale grounded in the gen04 report: the parent
  proposed 0 edits (no-op), %Merged Edges 9.09 dominates %Split Edges 0.17, BUT
  "no raw label spans >=2 GT neurons in this split" — i.e. the acceptance split
  has no GT-confirmable merges, so `split_label` always cuts a real neuron and
  trips the watchdog (this is why the prior +0.492 split_label attempt was
  rejected). The two prior `merge_labels` attempts (tangent-collinearity gate;
  radius + fluorescence gap-connectivity gate) fired ZERO net edits because their
  gates rejected the common tip-to-shaft/branch reconnections. NEW lever:
  **mutual-nearest reciprocity between two degree-1 tips** (plus tiny gap +
  caliber similarity). Reciprocity is a fundamentally different selection
  principle (it does not test tangent direction, radius continuity across a gap,
  or image signal); it fires on clean cable breaks (two free ends pointing at each
  other) while structurally excluding crossings (whose nearest partner is a shaft,
  not a reciprocating tip). Being `merge_labels`-only, it cannot trip the
  over-split watchdog.
