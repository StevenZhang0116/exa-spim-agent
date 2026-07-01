# Proofreading Criteria (the evolved natural-language program)

> The evolution loop revises THIS file alongside `heuristics.py`. It is the
> human-readable statement of *why* the policy makes the decisions it does.
> Each criterion should correspond to logic in `heuristics.py::propose_edits`.
> When the agent revises the policy, it must keep this file in sync so a
> reviewer can read the current proofreading theory in plain language.

## Objective

Maximize the run-length-weighted **held-out split-repair fitness** by repairing
BOTH error classes the segmentation makes:

- **Split errors** — one true neuron broken into several fragments. Repair by
  **unifying** fragment label pairs (`merge_labels`), without joining fragments
  that belong to different neurons.
- **Merge errors** — two (or more) true neurons fused under one segment label.
  Repair by **splitting** that label by location (`split_label`), without
  over-splitting a single real neuron (which would raise % Split Edges).

The gate scores each candidate on held-out neurons with a DENSE split-repair
signal: classify every `merge_labels` edit against the held-out label→neuron map
into `correct` (both fragments are the same GT neuron) vs `false` (a wrong fusion
of two different neurons), and take `score = correct − false`. This is dense —
every correctly-repaired split counts — whereas Edge Accuracy reads +0.000 for
most real repairs (only a bridged split EDGE moves it), which is why an
Edge-Accuracy gate flat-lined. Edge Accuracy (= 100 − (% Split Edges + % Omit
Edges + % Merged Edges)) and the other metrics are still computed and recorded
per generation for DIAGNOSIS; they are not the accept/reject bar.

**Acceptance gate (a generation is reverted if it fails):**
- The candidate's **penalized fitness** must beat the parent's by at least
  `score_margin` (integer, default 1 — raise to 2–3 so a single-repair swing
  within held-out noise is not locked in):

      fitness = (correct − false) − merge_penalty · false
      keep  iff  cand_fitness ≥ parent_fitness + score_margin

- **False merges are penalized, not hard-rejected.** A `false` merge (fusing two
  DIFFERENT neurons) costs `merge_penalty` each (default 100), so a single one
  needs ~100 correct repairs to offset — heavily discouraged, but no longer an
  automatic revert (that hard "zero false merges" gate was replaced by this smooth
  one so the search keeps a usable gradient around the precision boundary). Set
  `--merge-penalty` very high (e.g. 1e9) to recover the old hard gate. A correct
  `merge_labels` (fusing fragments of the SAME neuron) never incurs the penalty.
- **Over-split watchdog (diagnostic).** When the policy emits `split_label`, a
  rise in `% Split Edges` is surfaced in the failure report so over-splitting a
  single real neuron is visible; the accept/reject decision remains the penalized
  fitness above.

## Current criteria (Generation 0 — seed)

The seed is a conservative **colinear split-repair** policy (NOT a no-op). For
each SplitSite it emits a `merge_labels` edit only when BOTH:
  1. the gap is small — `s.gap_um <= GAP_THRESHOLD_UM` (4.0 µm), and
  2. the two fragments are colinear across the gap — a straight-line continuation
     (`cos(angle) >= MIN_COLINEAR_COS`, 0.94 ≈ ≤20° bend).
MergeSites are left alone (no `split_label`) — splitting is the riskier edit, left
for the loop to add once it can measure the trade-off. This proposes a small,
high-precision set of merges, so the score moves OFF the flat no-edit baseline
(giving the loop a gradient) without the union-find mega-label blowup a naive
"merge everything nearby" seed would cause. The gate is parent-relative, so each
generation must beat THIS seed (or the last accepted policy), not the no-edit floor.

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
  connecting signal across a gap, or an intensity valley at a suspected cut. The
  image RECEPTIVE FIELD is tunable: every reader method takes `shape=(z,y,x)`
  voxels — pass `ctx["image_patch_shape"]` (default `(16,16,16)`) or a custom value.
  A LARGER patch sees more context but every read is a cloud fetch (slower/costlier);
  each axis is clamped to 512. Tune per tier — a tight window for a clean micro-gap, a
  wider one to confirm a long faint bridge.

## Candidate space (what the policy gets to choose from)

The policy receives a UNIFIED stream of two site kinds (branch on `site.kind`):

- **SplitSite** (`kind == "split"`, from `dataset.candidate_split_sites`): for
  every fragment **tip** (`node_a`, degree 1), the nearest **differently-labelled**
  node within `max_gap_um` as `node_b` (tip, shaft, or branch — so tip-to-shaft
  and branch-point reconnections are candidates). One per unordered label pair —
  the CLOSEST gap. Two fragments can be near each other in more than one place, and
  the closest gap is not always the most decisive (it may be a sideways graze while
  another gap is a clean colinear continuation). Set `ENUM_PARAMS["split_alt_per_pair"]
  > 1` to attach the next-closest gaps as `site.alt_gaps` (extra evidence points the
  policy can inspect); this does NOT add sites or edits — `as_edit()` still emits one
  `merge_labels` that unifies the pair across all its gaps.
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
- `split_alt_per_pair` (1–10) — gaps kept per SplitSite label pair; `>1` attaches the
  next-closest gaps as `site.alt_gaps` (a PRECISION lever: judge a pair on its best
  evidence gap, not just the closest). Adds no sites/edits; read `site.alt_gaps` to use.
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
- **Harness:** `candidate_split_sites` broadened from tip-to-tip to
  tip-to-any-node (tip/shaft/branch) within `max_gap_um`; `SplitSite` carries
  `node_a` (always a tip) and `node_b` (the partner, any degree).
- **Harness (merge repair):** added `MergeSite` + `candidate_merge_sites`
  (GT-free branch-based merge detection) and a unified split+merge candidate
  stream; the failure report now lists baseline merge targets and an over-split
  watchdog.
- **Harness (gate):** replaced the Edge-Accuracy accept bar with the dense
  held-out split-repair fitness `(correct − false) − merge_penalty · false`, kept
  iff it beats the parent by `score_margin`. Edge Accuracy read +0.000 for most
  real repairs (only a bridged split edge moves it), so it flat-lined; the dense
  signal counts every repaired split. The old hard "zero false merges" revert
  became a smooth per-false-merge penalty (default 100).
