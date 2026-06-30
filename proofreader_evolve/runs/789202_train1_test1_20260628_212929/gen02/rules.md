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

## Current criteria (Generation 1 — colinear continuation OR short-gap tip-to-tip)

For each SplitSite the policy emits a `merge_labels` edit when EITHER rule fires
(both require `gap_um <= GAP_THRESHOLD_UM` = 4.0 µm to be considered at all):

  **(A) Colinear continuation** (the original Gen-0 rule). The two fragments are
  colinear across the gap — a straight-line continuation
  (`cos(angle) >= MIN_COLINEAR_COS`, 0.94 ≈ ≤20° bend). This is the high-precision
  workhorse for mid/long gaps where the tip tangent can be estimated reliably.

  **(B) Short-fragment proximity** (NEW, Gen 1). If the gap is very small
  (`gap_um <= SMALL_GAP_UM`, 1.8 µm) AND the candidate is **tip-to-tip**
  (graph degree 1 at both `node_a` and `node_b`), accept on proximity alone —
  WITHOUT the colinearity test of (A).

  *Why (B):* the gen01 failure report's accepted-edit `gap_um` buckets showed
  every correct repair sat at gap ≥ 2.24 µm, while the **MISSED real splits**
  bucket was dominated by tiny gaps (0.00–1.1 µm). At sub-2 µm gaps the fragments
  are short stubs, so `_walk_tangent` returns an unreliable direction and
  `colinear_cos` lands near 0 or negative even for genuine breaks (e.g. gap 1.00,
  cos −0.48) — so rule (A) wrongly rejects them. Spending precision here is safe:
  on train the NEAREST false join is at gap 2.19 µm and every false join has
  `colinear_cos ≤ 0.62`, whereas every correct repair has `cos ≥ 0.96`. Below
  ~2.2 µm there are **zero** false joins, so two tips meeting nose-to-nose this
  close are almost certainly one broken neuron. Restricting (B) to tip-to-tip
  (not tip-to-shaft) keeps it to the cleanest break signature and limits
  held-out over-merge risk.

  *Why no image:* the report's image warm-start probe measured `bridge_ratio`
  AUC = 0.52 (no separation of REAL vs FALSE on this brain), so gating on
  `gap_bridge_evidence` would not improve recall — the recall gap is geometric,
  not photometric, so this generation spends its change on geometry (rule B).

MergeSites are still left alone (no `split_label`). The gate is parent-relative,
so this generation must beat the Gen-0 seed's split-repair score (5 correct, 0
false) with zero false merges; rule (B) adds the small-gap tip-to-tip real splits
the seed missed while staying below the 2.19 µm false-join floor.

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
- **Gen 1:** added rule (B), a short-fragment **proximity** acceptance for
  `merge_labels`: accept SplitSites with `gap_um <= SMALL_GAP_UM` (1.8 µm) that
  are **tip-to-tip** (graph degree 1 at both endpoints) WITHOUT the colinearity
  test. Diagnosis from the gen01 report: the seed's correct repairs all sat at
  gap ≥ 2.24 µm, while the missed-real-split bucket was dominated by sub-1.1 µm
  gaps that rule (A) rejected because the tip tangent is unreliable on short
  stubs (`colinear_cos` near 0 / negative). Spending precision here is safe — on
  train the nearest false join is at 2.19 µm and all false joins have
  `colinear_cos ≤ 0.62` — so (B) recovers small-gap real splits with zero train
  false merges, raising the split-repair score above the parent's 5. Image was
  deliberately NOT used: the warm-start `bridge_ratio` AUC was 0.52 (no
  separation), so the recall gap was treated as geometric, not photometric. Added
  `_node_degree(g, node)` helper; `SMALL_GAP_UM` constant.
- **Harness:** `candidate_split_sites` broadened from tip-to-tip to
  tip-to-any-node (tip/shaft/branch) within `max_gap_um`; `SplitSite` carries
  `node_a` (always a tip) and `node_b` (the partner, any degree).
- **Harness (merge repair):** added `MergeSite` + `candidate_merge_sites`
  (GT-free branch-based merge detection) and a unified split+merge candidate
  stream; fitness is Edge Accuracy (charges merge errors); the failure report now
  lists baseline merge targets and an over-split watchdog.
