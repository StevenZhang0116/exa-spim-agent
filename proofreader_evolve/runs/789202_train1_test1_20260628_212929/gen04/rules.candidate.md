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

## Current criteria (Generation 4 — colinear continuation @ relaxed cos OR short-gap tip-to-tip/-shaft)

For each SplitSite the policy emits a `merge_labels` edit when EITHER rule fires
(both require `gap_um <= GAP_THRESHOLD_UM` = 4.0 µm to be considered at all):

  **(A) Colinear continuation** (the original Gen-0 rule, relaxed in Gen 4). The two
  fragments are colinear across the gap — a near-straight continuation
  (`cos(angle) >= MIN_COLINEAR_COS`, now 0.85 ≈ ≤~30° bend, down from 0.94 ≈ ≤20°).
  Tangent continuity is the single most GENERALIZABLE precision feature we have: on
  train EVERY false join has `colinear_cos <= 0.62`, so 0.85 keeps a wide margin above
  the false-join ceiling (zero train false merges) while recovering the 0.85–0.94
  continuation band — genuine reconnections (mostly `deg_b == 2` shaft re-entries
  above `SMALL_GAP_UM`, plus colinear `deg_b >= 3` continuations) that the strict
  0.94 cut wrongly rejected. This is the high-precision workhorse for mid/long gaps
  where the tip tangent can be estimated reliably, and — unlike the train-specific
  proximity rule (B) — a tangent-alignment threshold is expected to transfer to the
  held-out brain, which is where the gate scores.

  **(B) Short-fragment proximity** (Gen 1, widened Gen 2). If the gap is very small
  (`gap_um <= SMALL_GAP_UM`, 1.8 µm) AND the candidate is **tip-to-tip** (`deg_b == 1`)
  OR **tip-to-shaft** (`deg_b == 2`) — with `node_a` always a tip — accept on
  proximity alone, WITHOUT the colinearity test of (A). True branch/crossing points
  (`deg_b >= 3`) stay excluded as the riskiest join geometry.

  *Why (B):* the gen01 report showed every correct repair sat at gap ≥ 2.24 µm
  while the **MISSED real splits** bucket was dominated by tiny gaps; at sub-2 µm
  gaps the fragments are short stubs, so `_walk_tangent` returns an unreliable
  direction and `colinear_cos` lands near 0 or negative even for genuine breaks
  (e.g. gap 1.00, cos −0.48), so rule (A) wrongly rejects them. Spending precision
  here is safe: on train the NEAREST false join is at gap 2.19 µm (every false join
  has `colinear_cos ≤ 0.62`); below ~2.2 µm there are **zero** false joins of ANY
  endpoint degree. Gen 1 admitted only tip-to-tip and lifted the score 5 → 61.

  *Why Gen 2 widened (B) to tip-to-shaft:* the gen02 report showed the ENTIRE
  remaining MISSED real-split bucket (1065 sites, gaps 0.19–1.21 µm and beyond)
  is `deg_b == 2` (a tip reconnecting into the mid-shaft of another fragment of
  the SAME neuron). Gen 1's `deg_b == 1` restriction left this whole pool
  unrepaired. Since the sub-1.8 µm band carries zero train false joins regardless
  of degree, admitting `deg_b == 2` recovers that pool with zero false merges,
  while keeping `deg_b >= 3` out (joining INTO a true branch point — 3+ neurites
  converging — is the geometry most likely to fuse two different neurons).

  *Why no image:* the report's warm-start probe measured `bridge_ratio` AUC = 0.52
  (no separation of REAL vs FALSE on this brain), so gating on `gap_bridge_evidence`
  would not raise recall — the recall gap is geometric, not photometric.

MergeSites are still left alone (no `split_label`). The gate is parent-relative,
so this generation must beat the parent's split-repair score (61 correct, 0 false)
with zero false merges; widening (B) to tip-to-shaft adds the small-gap `deg_b == 2`
real splits while staying below the 2.19 µm false-join floor.

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
  `_node_degree(g, node)` helper; `SMALL_GAP_UM` constant. (ACCEPTED: score 5 → 61.)
- **Gen 2:** widened rule (B) from tip-to-tip (`deg_b == 1`) to tip-to-tip OR
  tip-to-shaft (`deg_b in (1, 2)`). Diagnosis from the gen02 report: the score had
  risen to 61/0 (Gen 1 accepted), and the ENTIRE remaining MISSED real-split bucket
  (1065 sites) was `deg_b == 2` — tips reconnecting into the mid-shaft of another
  fragment of the same neuron — which Gen 1's tip-to-tip restriction skipped. The
  sub-1.8 µm band carries zero train false joins of any degree (all 7 false joins
  are at gap ≥ 2.19 µm), so admitting `deg_b == 2` there recovers the largest
  remaining recall pool with zero false merges. `deg_b >= 3` (true branch/crossing
  points) kept excluded as the riskiest join. Image again NOT used (AUC 0.52).
  One-line change: `deg_b == 1` → `deg_b in (1, 2)`.
- **Gen 3 (REJECTED by harness):** added a topological fan-out cap (a second pass
  limiting how many fragments may fuse into one union-find class, to fight
  over-merge / mega-merge classes). On the held-out split-repair score it came back
  **+0 vs the parent** and was reverted — the on-disk `propose_edits` is back to the
  Gen 2 single-pass form. Lesson: capping fan-out did not move held-out accuracy, so
  Gen 4 does NOT retry it (nor proximity widening); it pivots to a recall lever
  grounded in generalizable geometry instead.
- **Gen 4:** relaxed rule (A)'s colinearity threshold `MIN_COLINEAR_COS` from 0.94
  (≤~20° bend) to 0.85 (≤~30° bend). Diagnosis from the gen04 report: the
  train-vs-held-out split-repair gap is +14.0 and WIDENING (train ≈ 33, held-out ≈ 19),
  i.e. Gen 1–2's proximity recall is train-specific and not transferring. Tangent
  continuity (`colinear_cos`) is the most generalizable precision feature available,
  and every train FALSE join has `colinear_cos <= 0.62` — so lowering the accept cut
  to 0.85 stays well above that ceiling (zero train false merges) while recovering the
  0.85–0.94 continuation band: real reconnections with a modest (~20–30°) bend that
  the strict cut rejected, including `deg_b == 2` shaft re-entries beyond the
  `SMALL_GAP_UM` proximity window and colinear branch-point continuations. Because it
  rests on tangent alignment rather than this brain's proximity statistics, it is
  expected to transfer to held-out (where the gate scores) and narrow the
  generalization gap. Image again NOT used (warm-start `bridge_ratio` AUC = 0.52, no
  REAL/FALSE separation — the recall gap is geometric, not photometric). One-line
  change: `MIN_COLINEAR_COS` 0.94 → 0.85.
- **Harness:** `candidate_split_sites` broadened from tip-to-tip to
  tip-to-any-node (tip/shaft/branch) within `max_gap_um`; `SplitSite` carries
  `node_a` (always a tip) and `node_b` (the partner, any degree).
- **Harness (merge repair):** added `MergeSite` + `candidate_merge_sites`
  (GT-free branch-based merge detection) and a unified split+merge candidate
  stream; fitness is Edge Accuracy (charges merge errors); the failure report now
  lists baseline merge targets and an over-split watchdog.
