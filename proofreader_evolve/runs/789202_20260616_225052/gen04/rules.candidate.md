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

**Acceptance gate (HARD — a generation is reverted if it fails any):**
- Edge Accuracy must beat the parent by `gate_eps`.
- **No new merge error (EVERY generation):** neither `# Merges` nor
  `% Merged Edges` may rise above the parent (beyond `merge_tol`, default 0). This
  enforces the "without creating merge errors" clause directly — raising net Edge
  Accuracy by repairing splits while introducing a few merges is NOT accepted. A
  correct `merge_labels` (fusing fragments of the SAME neuron) never trips this;
  only a wrong fusion does.
- **Over-split watchdog (only when the policy emits `split_label`):** `% Split
  Edges` may not rise more than `split_tol` above the parent.

## Current criteria (Generation 4 — endpoint-topology-aware split-repair)

The policy is a **colinear split-repair** policy with a DISTANCE-DEPENDENT accept
boundary that is now ALSO topology-aware (NOT a no-op, and no `split_label` yet). For
each SplitSite it measures the worst-case colinearity across the gap
(`min(cos_in, cos_out)`, see `_colinearity_score`) and emits a `merge_labels` edit
when that score clears a floor that tightens as the gap grows (`_required_cos`):
  1. very near — `gap_um <= GAP_TIGHT_UM` (2.0 µm): floor `MIN_COLINEAR_COS` (0.94);
  2. mid band — `2.0 < gap_um <= GAP_THRESHOLD_UM` (4.0 µm): floor `COS_NEAR` (0.97),
     tighter than the seed to drop marginal pairs that raised %Split Edges / created
     a merge;
  3. far band — `4.0 < gap_um <= GAP_FAR_UM` (8.0 µm): floor `COS_FAR` (0.992),
     near-perfect continuation only, to bridge longer TRUE gaps the old 4 µm box
     missed (extra omit-reducing merges);
  4. beyond `GAP_FAR_UM`: reject.

**Endpoint-topology penalty (Gen 4).** `node_a` is always a fragment TIP (degree 1).
A genuine broken neuron is a tip-to-TIP break: the partner `node_b` is also a tip. A
spurious crossing/touch instead fuses a tip into the MIDDLE of an unrelated passing
cable, so `node_b` has degree >= 2 (shaft or branch). When `node_b` is NOT a tip the
required colinearity floor is RAISED by `COS_SHAFT_PENALTY` (0.020, capped at
`COS_CAP` 0.9995). This keys purely on `g.degree[node_b]` — an always-present
geometry signal — so it drops the merge-prone tip-into-shaft pairs (the class that
produced the gen03 created merge `464488053+553504107`) while leaving every
tip-to-tip repair untouched.

This builds on the gen03 sloped boundary (which both REMOVES marginal mid-gap edits
AND ADDS high-precision far-gap edits) with a SEPARATE, orthogonal axis (partner
degree) — a genuine change to the proposed edit set, not a fail-open filter.
MergeSites are still left alone (`split_label` deferred). The gate is
parent-relative, so each generation must beat the last accepted policy.

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

### Tuning the candidate stream itself (`ENUM_PARAMS`)

The thresholds above decide what to ACCEPT; the candidate stream decides what you
even SEE. `heuristics.py` may define a module-level `ENUM_PARAMS` dict to widen or
narrow that stream — turning the framework's prior on "what an error looks like"
from a fixed rail into an evolvable knob:
- `max_gap_um` (1–40) — split search radius; raise to reach longer true gaps a
  tight radius misses, lower to cut noise.
- `tip_to_shaft` — if False, split partners must be tips (legacy tip-to-tip).
- `min_arm_cable_um` (2–50) — lower to surface SHORTER merges the 10 µm floor hides
  (a detector-recall lever for the "missed merges" mode above).
- `seed_depth_um`, `max_per_label`, `split_max_sites`, `merge_max_sites` — see
  `dataset.ENUM_PARAM_SPEC`.
Values are clamped to a safe rail and unknown keys ignored, so this can't crash the
run; `ctx["enum_params"]` reports the values actually in effect (post-clamp). Widen
the stream when the failure report shows a recall gap (merge targets with no
MergeSite, or unrepaired splits with no SplitSite); narrow it when the candidate
set is noisy and precision is the bottleneck.

`ctx["n_split_sites"]` / `ctx["n_merge_sites"]` report the stream composition.

## Change log

- **Gen 0:** seed is a conservative colinear split-repair policy — `merge_labels`
  for SplitSites with `gap_um <= GAP_THRESHOLD_UM` (4.0 µm) AND colinear
  continuation (`cos >= MIN_COLINEAR_COS`, 0.94); no `split_label`.
- **Gen 1 (adaptive gap/angle accept boundary).** Diagnosis from the gen03 report:
  the seed's 51 merges raised train Edge Accuracy 86.7532 -> 87.5273, but the gain
  came almost entirely from %Omit Edges DROPPING on every skeleton (-1.471, -1.047,
  -0.838, -0.344, -0.671, -0.466) while %Split Edges ROSE on every skeleton
  (+0.219, +0.176, +0.173, +0.125, +0.190, +0.123; hundreds of new split edges) and
  one edit fused two different neurons (created merge `464488053+553504107` on
  N005). So the seed's single rectangular accept box leaks two ways: it admits
  marginal mid-gap pairs that add split edges / a merge, and it never reaches longer
  near-straight true gaps whose repair is what drives the omit-edge win. Change:
  replaced the rectangle with a distance-dependent colinearity floor —
  `_colinearity_score` returns the worst-case cosine and `_required_cos` demands
  `0.94` very near, `0.97` in the mid band (drops the marginal/over-split-prone
  pairs), and `0.992` out to a new `GAP_FAR_UM` (8.0 µm) far band (adds extra
  high-precision omit-reducing merges the 4 µm box missed). This is a DIFFERENT
  mechanism from the two prior rejected (+0.000) attempts (a radius-continuity guard
  and an image gap-connectivity guard, both acceptance-side filters that failed open
  and never fired on the geometry-only default run): it is a pure-geometry rule that
  both removes AND adds edits, so it changes the proposed set on the default run.
- **Gen 4 (endpoint-topology penalty).** Diagnosis from the gen04 report: the gen03
  distance-dependent floor lifted train Edge Accuracy 86.7532 -> 87.5273, and the
  gain is real (%Omit Edges drops on every skeleton: -1.471, -1.047, -0.838, -0.344,
  -0.671, -0.466). But two leaks remain in the proposed set: (a) %Split Edges still
  ROSE on every skeleton (+0.219, +0.176, +0.173, +0.125, +0.190, +0.123; hundreds of
  new split edges), and (b) ONE edit fused two different neurons — created merge
  `464488053+553504107` on N005-789202-SP at (6787.35, 19507.84, 46619.0). The angle
  test alone can't tell an end-to-end break from a tip touching the middle of a
  passing cable, because a locally-colinear crossing still clears the cosine floor.
  Change: add a partner-degree penalty. `node_a` is always a tip; when `node_b` is
  NOT a tip (`g.degree[node_b] >= 2`, a shaft/branch interior), raise the required
  colinearity floor by `COS_SHAFT_PENALTY` (0.020, capped at `COS_CAP` 0.9995). This
  removes the merge-prone tip-into-shaft pairs (the class that produced the created
  merge) while keeping every clean tip-to-tip repair, so it tightens precision
  WITHOUT giving back the omit-edge win. It is a DIFFERENT mechanism from gen1
  (radius-continuity guard, failed open / inert), gen2 (image gap-connectivity guard,
  image reader None / inert), and gen3 (distance-dependent cosine floor on the gap
  axis): this keys on the partner's graph DEGREE — an always-present geometry signal
  on the default run — and actually changes which edits are proposed.
- **Harness:** `candidate_split_sites` broadened from tip-to-tip to
  tip-to-any-node (tip/shaft/branch) within `max_gap_um`; `SplitSite` carries
  `node_a` (always a tip) and `node_b` (the partner, any degree).
- **Harness (merge repair):** added `MergeSite` + `candidate_merge_sites`
  (GT-free branch-based merge detection) and a unified split+merge candidate
  stream; fitness is Edge Accuracy (charges merge errors); the failure report now
  lists baseline merge targets and an over-split watchdog.
