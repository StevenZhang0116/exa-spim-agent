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

## Current criteria (Generation 7 — colinear continuation OR short-gap tip-to-tip/-shaft OR near-band caliber-match-OR-through-line continuity)

For each SplitSite the policy emits a `merge_labels` edit when EITHER rule fires
(all require `gap_um <= GAP_THRESHOLD_UM` = **6.0 µm** (Gen 14, was 4.0) to be
considered at all):

  **(A) Colinear continuation** (the original Gen-0 rule). The two fragments are
  colinear across the gap — a straight-line continuation
  (`cos(angle) >= MIN_COLINEAR_COS`, 0.94 ≈ ≤20° bend). This is the high-precision
  workhorse for mid/long gaps where the tip tangent can be estimated reliably.
  Gen 14: rule (A) now reaches gaps up to `GAP_THRESHOLD_UM = 6.0` (was 4.0), fed
  by the widened candidate stream (`split_max_sites = 25000`); it has NO internal
  gap cap, so it alone extends with the threshold. Branches (B)/(C) keep their own
  proximity caps (`SMALL_GAP_UM` / `NEAR_GAP_UM`) and do NOT widen.

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

  **(C) Near-band caliber-OR through-line continuity** (Gen 5, extended Gen 7,
  caliber gate widened Gen 9). For a SplitSite in the band
  `SMALL_GAP_UM < gap_um <= NEAR_GAP_UM` (1.8–2.15 µm, strictly BELOW the 2.19 µm
  train false-join floor) that is **tip-to-tip** (`deg_b == 1`) OR **tip-to-shaft**
  (`deg_b == 2`) — `node_a` always a tip — accept when EITHER:
    - **caliber branch** (Gen 5, widened Gen 9): `rad_ratio = max(r_a, r_b) /
      min(r_a, r_b) <= RAD_RATIO_MAX` (now **3.0**), radii from
      `ctx["node_radius"]`; OR
    - **through-line continuity** (Gen 7): `_through_line_cos(g, s,
      TANGENT_WALK_UM) >= THROUGH_LINE_COS` (0.7).

  Caliber is NO LONGER a restrictive precision gate. The Gen-9 split-repair
  attribution table shows `rad_ratio` is non-discriminative — 0 false joins in
  EVERY bucket, including `[2.05, inf)` (131 correct / 0 false) — and high
  `rad_ratio` is the NORM for a genuine thin-tip-into-thick-shaft reconnection
  (up to ~2.79). `RAD_RATIO_MAX = 3.0` therefore admits the full near-band caliber
  range instead of rejecting those real splits. The near-band's PRECISION rests on
  the sub-2.19 µm gap cap (`NEAR_GAP_UM`) + the endpoint-degree gate
  (`deg_a == 1`, `deg_b in (1, 2)`, branch points excluded), not on caliber.

  *Why the Gen 7 through-line test (and why it differs from the reverted attempts):*
  the still-MISSED real splits are dominated by tip-to-shaft (`deg_b == 2`)
  reconnections where a THIN tapering tip (rad ~0.75) joins a THICK shaft (rad ~1.9),
  giving `rad_ratio` 2.0–2.85 — so the caliber gate rejects them. Caliber mismatch is
  INTRINSIC to a genuine tip-to-shaft reconnection (a neuron tapers), so no caliber
  threshold can recover them; the only discriminating feature is GEOMETRY, and the
  report names `colinear_cos` the single strongest precision feature — many of these
  missed sites have HIGH report `colinear_cos` (0.65–0.89). Two prior continuity
  attempts FAILED (held-out +0, and the last did not even change the train score)
  because they reused `_is_colinear_split`, which routes the straightness test
  THROUGH the gap bridge vector `node_a -> node_b`. When a tip joins the SIDE of a
  shaft it is laterally offset, so that bridge vector is tilted even when the two
  cables are perfectly collinear — the bridge-routed test scores it low and never
  fires. `_through_line_cos` BYPASSES the bridge: for tip-to-shaft it compares arm
  A's outward tangent to the shaft's local axis at `node_b` (from its two
  same-fragment neighbors); for tip-to-tip it checks the two outward tangents are
  antiparallel. This finally fires on the genuinely-collinear tip-to-shaft
  reconnections that both the intrinsic caliber mismatch and the bridge-routed
  `_is_colinear_split` rejected. The 0.7 floor sits ABOVE the 0.62 max `colinear_cos`
  of any train false join, and the near band stays below the 2.19 µm gap floor, so
  TRAIN false stays 0; continuity is brain-independent, so it should transfer to
  held-out and narrow the +82.3 train/held-out gap. If the radius is unavailable the
  caliber branch is skipped; if a tangent can't be estimated the through-line branch
  is skipped — (C) simply does not fire.

  *Why (C) is generalizable:* the gen05 report showed the ENTIRE missed-real-split
  pool (876 sites) is `deg_b == 2` (tip-into-shaft), ALL at gaps in
  [1.80, ~2.18] µm — just above rule (B)'s 1.8 µm proximity cutoff and just below
  the 2.19 µm train false-join floor. Train has ZERO false joins below 2.19 µm at
  any degree, so widening proximity inflates TRAIN recall trivially — but the gate
  scores HELD-OUT and the train-minus-held-out split-repair gap is +14 and
  WIDENING, meaning proximity recall is train-specific and does not transfer. So
  (C) does NOT add recall on proximity alone; its DEFINING criterion is caliber
  match. A tip reconnecting to the SAME broken neuron shares cable caliber across
  the break, whereas two unrelated neurites grazing typically have mismatched
  calibers; `rad_ratio` is a brain-independent ratio (not a per-brain distance
  threshold), so unlike rule (B)'s pure proximity it is expected to TRANSFER to
  held-out and narrow the widening +14 train/held-out gap. The gap cap at 2.15 µm
  (below 2.19) keeps the TRAIN false count at 0, and the caliber gate is the
  held-out precision hedge against false merges.

  **(D) Mid-gap colinear continuation** (Gen 15). For a SplitSite in the band
  `NEAR_GAP_UM < gap_um <= MID_GAP_UM` (2.15–3.0 µm) that is **tip-to-tip**
  (`deg_b == 1`) OR **tip-to-shaft** (`deg_b == 2`) — `node_a` always a tip —
  accept when `ctx["split_geom"](s).colinear_cos >= MID_COLINEAR_COS` (**0.75**).
  This reads the harness's leak-free per-site geometry directly and fails safe (no
  accept) when `split_geom` is missing or returns nothing.

  *Why (D) differs from rule (A):* rule (A) uses the strict 0.94 BOTH-ARM bridge
  test (`_is_colinear_split`), which routes straightness THROUGH the gap bridge
  vector `node_a -> node_b`; a laterally-offset tip-into-shaft join tilts that
  bridge even when the cables are perfectly colinear, so (A) UNDER-SCORES and
  rejects genuine tip-to-shaft continuations in this band. Rule (D) instead keys on
  the report's single strongest precision feature, `split_geom.colinear_cos` — the
  AVERAGE of the two arm cosines, NOT the stricter both-arm bridge product — so it
  fires on the genuinely-colinear tip-to-shaft reconnections that (A) misses.

  *Why (D) is precision-safe and generalizable:* the SplitSite audit shows that in
  the gap band `<= ~3.16 µm` EVERY train false join has `colinear_cos <= 0.62`,
  while many MISSED real splits in (2.15, 3.0] have `colinear_cos` 0.71–0.92. The
  `MID_COLINEAR_COS = 0.75` floor sits **0.13 above** that 0.62 false-join ceiling,
  and the band is gap-capped at **3.0 µm** — safely below the **3.67 µm** gap of
  the FIRST train false join with `colinear_cos > 0.62`. So the threshold cleanly
  separates real continuations from false joins on a brain-independent feature, with
  zero train false. Colinear continuation is brain-independent, so it should
  transfer to held-out.

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
- **Gen 5:** added rule (C), a **caliber-matched near-band** acceptance for
  `merge_labels`. New constants `NEAR_GAP_UM = 2.15` (extends the near-band up to
  just below the 2.19 µm train false-join floor, so train false stays 0) and
  `RAD_RATIO_MAX = 1.5` (max/min of the two endpoint radii; above this the calibers
  are mismatched ⇒ likely two different neurons ⇒ refuse), plus a defensive helper
  `_rad_ratio(ctx, node_a, node_b)`. Rule (C) fires only when neither (A) nor (B)
  accepted: for `SMALL_GAP_UM < gap_um <= NEAR_GAP_UM`, `deg_a == 1` and
  `deg_b in (1, 2)`, accept iff `rad_ratio` is available and `<= RAD_RATIO_MAX`.
  Diagnosis from the gen05 report: the ENTIRE missed-real-split pool (876 sites) is
  `deg_b == 2` tip-into-shaft at gaps [1.80, ~2.18] µm — just above rule (B)'s 1.8
  cutoff and just below the 2.19 train false floor. Train recall is trivially
  inflatable by widening proximity (zero train false below 2.19 at any degree), but
  the held-out gate exposes that as train-specific: the train/held-out split-repair
  gap is +14 and WIDENING. So Gen 5 deliberately does NOT add recall on proximity
  alone — it gates the near-band on caliber match, a brain-independent ratio that
  should transfer to held-out (a tip reconnecting to the SAME broken neuron shares
  caliber across the break; unrelated grazing neurites do not). The gap cap below
  2.19 keeps train false == 0; the caliber gate is the held-out precision hedge.
  Two prior attempts on this same parent were REJECTED at +0 held-out — Gen 3 (a
  topological fan-out cap) and Gen 4 (lowering `MIN_COLINEAR_COS` globally to 0.85)
  — so Gen 5 picks a DIFFERENT, generalizable lever (caliber match) rather than
  repeating either. Image again NOT used (`bridge_ratio` AUC = 0.52, no separation).
  Rules (A) and (B), `ENUM_PARAMS`, `GAP_THRESHOLD_UM`, and `MIN_COLINEAR_COS`
  unchanged.
- **Gen 7:** extended rule (C) with an OR through-line continuity acceptance for
  `merge_labels`. New constant `THROUGH_LINE_COS = 0.7` (after `RAD_RATIO_MAX`) and
  new helper `_through_line_cos(g, s, walk)` (placed after `_walk_tangent`, before
  `_is_colinear_split`). Rule (C) now accepts when `caliber_ok` (Gen 5 caliber
  match) OR `through_ok` (`_through_line_cos >= THROUGH_LINE_COS`); rules (A) and
  (B) untouched. Diagnosis from the failure report: the 831 still-MISSED real
  splits are dominated by tip-to-shaft (`deg_b == 2`) sites in the near band
  (gap 1.80–2.18 µm) with MISMATCHED caliber (thin tip rad ~0.75 into a thick shaft
  rad ~1.9, so `rad_ratio` 2.0–2.85). Caliber mismatch is INTRINSIC to a genuine
  tip-to-shaft reconnection (a neuron tapers), so no caliber test can recover them;
  the only discriminating feature is geometry, and the report names `colinear_cos`
  the single strongest precision feature — many of these missed sites have HIGH
  report `colinear_cos` (0.65, 0.71, 0.79, 0.80, 0.89). Two prior continuity
  attempts were REVERTED because they reused `_is_colinear_split`, which routes the
  straightness test THROUGH the gap bridge vector `node_a -> node_b`; a
  laterally-offset tip-to-shaft join tilts that bridge even when the cables are
  collinear, so it scored low and never fired (the last attempt did not even change
  the train score). `_through_line_cos` BYPASSES the bridge — it compares cable
  tangents directly (arm A vs the shaft axis at `node_b`, or the two tip tangents) —
  so it can finally fire on the genuinely-collinear tip-to-shaft reconnections.
  False-join ceiling: the 0.7 floor is above the 0.62 max `colinear_cos` of any
  train false join, and the near band stays below the 2.19 µm gap floor, so train
  false stays 0; continuity is brain-independent so it should transfer to held-out
  and narrow the +82.3 train/held-out gap. Image again NOT used (`bridge_ratio`
  AUC = 0.52, no separation). NOTE on history: the Gen 6 near-band colinear-OR
  attempt was REJECTED (held-out +0); Gen 5's caliber rule remains the accepted
  parent (295 correct / 0 false on train) and is what Gen 7 must beat.
  `THROUGH_LINE_COS` is the only added constant; rules (A)/(B), `MIN_COLINEAR_COS`,
  `GAP_THRESHOLD_UM`, `SMALL_GAP_UM`, `NEAR_GAP_UM`, `RAD_RATIO_MAX`,
  `_is_colinear_split`, and `ENUM_PARAMS` are unchanged.
- **Gen 9 (2026-06-29):** widened rule (C)'s caliber branch by raising the
  module-level constant `RAD_RATIO_MAX` from **1.5 → 3.0** (the ONLY policy
  change). Parent = **376 correct / 0 false** on train; confusion REAL 1126
  (accepted 376 / MISSED 750), FALSE 7 (accepted 0 / rejected 7). Diagnosis: the
  Gen-9 split-repair attribution table shows `rad_ratio` has ZERO discriminative
  power — `[-inf,1.44)`: 110/0, `[1.44,2.05)`: 135/0, `[2.05,inf)`: **131
  correct / 0 false**. Every bucket, including the high-mismatch one, has zero
  false joins, so caliber does NOT separate REAL splits from FALSE joins on this
  data. In fact caliber MISMATCH is the NORM for a genuine tip-to-shaft
  reconnection: a thin distal tip (rad ~0.75) rejoining its thick parent shaft
  (rad ~1.9) gives `rad_ratio` ~2.5, and the MISSED-real bucket is dominated by
  exactly these high-`rad_ratio` (2.0–2.79), `deg_b == 2` sites in the 1.8–2.15 µm
  band. So the old `RAD_RATIO_MAX = 1.5` ceiling was rejecting a large pool of
  genuine near-band tip-to-shaft real splits while preventing NO false merge —
  caliber was dead weight there. `RAD_RATIO_MAX = 3.0` covers the observed
  near-band max (~2.79) with margin and stops suppressing them.
  *Precision preserved:* nothing else changed — `NEAR_GAP_UM` stays 2.15 (the
  sub-2.19 µm gap cap is the real false-join floor; every train false sits at
  gap ≥ 2.19 µm and `colinear_cos ≤ 0.62`), the endpoint-degree gate
  (`deg_a == 1`, `deg_b in (1,2)`, branch points excluded) is unchanged, and the
  through-line branch / `THROUGH_LINE_COS` are untouched. Train false stays 0.
  *Generalizability:* the recall expansion is grounded in (a) endpoint-degree
  topology (tip-to-tip/shaft, branch points excluded) — a blessed generalizable
  feature — and (b) the physical sub-2.2 µm proximity floor, NOT in train-specific
  tuning. Thin-tip-to-thick-shaft caliber mismatch is a universal,
  brain-independent morphological fact, so the held-out near band should contain
  the same pool. This DIFFERS from the previously REJECTED gap-window extension
  (through-line continuity gap window 2.15 → 4.0 µm, held-out +0) — that targeted a
  2.15–4.0 µm band that is EMPTY on held-out; the near band is dense and known to
  transfer (the original Gen-5 near-band rule was accepted). The exhausted
  gap-window lever is NOT re-touched. Image again NOT used (warm-start probe AUC
  0.52 — no REAL/FALSE separation; the recall gap is geometric, not photometric).
  Updated the inline comment on `RAD_RATIO_MAX`, the near-band-rule comment block,
  and rule (C)'s in-function comment to state caliber is non-discriminative.
- **Gen 14 (2026-06-29) — CANDIDATE (pending the held-out gate):** changed the
  evolution AXIS. *Diagnosis:* five consecutive recall levers that re-sliced the
  SAME candidate stream with different geometric features all tied at held-out +0
  and were reverted (gen8 through-line window, gen10 NEAR_GAP_UM sliver, gen11
  per-arm symmetry, gen12 alt_gaps, gen13 tip_tangent_cos) — the held-out reachable
  pool INSIDE the current stream is saturated. The failure report says the binding
  constraint is the STREAM ITSELF, not the thresholds: `split_max_sites` truncation
  keeps only the CLOSEST 5000 of 15378 label pairs and DROPS 10378 pairs at gaps
  3.32–15 µm BEFORE the policy ever sees them, and explicitly advises "WIDEN the
  candidate stream via ENUM_PARAMS (raise split_max_sites)… NOT tune your
  thresholds." *Change (two coupled constants, no new branch, no logic change):*
  (1) set `ENUM_PARAMS["split_max_sites"] = 25000` (was the historical default
  5000) to surface the dropped longer-gap pairs; (2) raise `GAP_THRESHOLD_UM`
  4.0 → 6.0 so the now-visible 4–6 µm pairs pass the outer per-site guard. Because
  rule (A) has NO internal gap cap and branches (B)/(C) DO (`gap <= SMALL_GAP_UM`,
  `SMALL_GAP_UM < gap <= NEAR_GAP_UM`), raising `GAP_THRESHOLD_UM` extends ONLY the
  strict 0.94 colinear-continuation rule (A); the proximity rules do not widen.
  *Precision / generalization rationale:* the 0.94 colinear bar is far above the
  observed train false-join ceiling of 0.62 colinear_cos (all train false joins at
  gaps ≤ 3.16 µm), so two unrelated neurites being 0.94-colinear across a multi-µm
  gap is geometrically very unlikely, while a 0.94-straight continuation across such
  a gap is the signature of ONE neuron the segmentation broke. This changes the
  held-out REACHABLE set rather than re-slicing it — the report's own repeated
  recommendation — and mirrors Gen 9, the only change that ever transferred, which
  was likewise a single principled constant change. Everything else (rules
  A/B/C bodies, helpers, MIN_COLINEAR_COS, SMALL_GAP_UM, NEAR_GAP_UM, RAD_RATIO_MAX,
  THROUGH_LINE_COS, other ENUM_PARAMS) is unchanged. Image NOT used (bridge_ratio
  AUC 0.52). Mark CANDIDATE pending the held-out gate.
- **Gen 15 (2026-06-29) — CANDIDATE (pending the held-out gate):** added a NEW
  acceptance tier (D), a mid-gap colinear continuation, in the band
  `NEAR_GAP_UM < gap_um <= MID_GAP_UM` (2.15–3.0 µm) for `deg_a == 1`,
  `deg_b in (1, 2)`, accepting when `ctx["split_geom"](s).colinear_cos >=
  MID_COLINEAR_COS` (0.75). Two new constants: `MID_GAP_UM = 3.0`,
  `MID_COLINEAR_COS = 0.75`. *Parent* = **404 correct / 0 false** on train after
  Gen 14's stream-widening was accepted. The stream is now UN-TRUNCATED
  (`split_max_sites = 25000`, all 15378 pairs surfaced), so stream-widening is
  EXHAUSTED — the binding constraint is no longer what the policy sees but which of
  the reachable sites it accepts. *Diagnosis:* the remaining MISSED real-split pool
  (2319 sites) is dominated by tip-to-shaft (`deg_b == 2`) and tip-to-tip
  (`deg_b == 1`) sites in the band JUST above `NEAR_GAP_UM = 2.15`. Rule (A)'s
  strict 0.94 BOTH-ARM bridge test under-scores tip-to-shaft joins (the bridge
  vector tilts when a tip meets the SIDE of a shaft), so it rejects genuine
  continuations there; rules (B)/(C) cap out at gap 2.15. *The ONE change* is the
  new tier (D). *Precision rationale:* the SplitSite audit shows every train false
  join at gap `<= 3.16 µm` has `colinear_cos <= 0.62`, while many missed real splits
  in (2.15, 3.0] have `colinear_cos` 0.71–0.92; the first false join above 0.62 sits
  at gap 3.67 µm. So 0.75 (0.13 margin above the 0.62 ceiling) inside a band capped
  at 3.0 µm (below 3.67 µm) cleanly separates real continuations from false joins.
  *Why this is DIFFERENT from previously reverted colinear levers:* Gen 4 lowered
  `MIN_COLINEAR_COS` globally to 0.85; Gen 6 added a near-band (`<= 2.15`)
  colinear-OR; Gen 11 used a per-arm `min(cos_a, cos_b)` symmetry test. Tier (D) is
  a NEW gap band (2.15–3.0 µm), uses the AVERAGE `colinear_cos` via the leak-free
  `split_geom` (not the strict both-arm bridge product, not the per-arm min), with a
  margin-safe 0.75 threshold, and operates on the now much-LARGER reachable pool
  (the stream is un-truncated). Image NOT used (`bridge_ratio` AUC 0.52 — no
  REAL/FALSE separation; the recall gap is geometric, not photometric). Rules (A),
  (B), (C), the outer `GAP_THRESHOLD_UM` guard, all existing constant values,
  `ENUM_PARAMS`, and the helpers are unchanged. Mark CANDIDATE pending the held-out
  gate.
- **Harness:** `candidate_split_sites` broadened from tip-to-tip to
  tip-to-any-node (tip/shaft/branch) within `max_gap_um`; `SplitSite` carries
  `node_a` (always a tip) and `node_b` (the partner, any degree).
- **Harness (merge repair):** added `MergeSite` + `candidate_merge_sites`
  (GT-free branch-based merge detection) and a unified split+merge candidate
  stream; fitness is Edge Accuracy (charges merge errors); the failure report now
  lists baseline merge targets and an over-split watchdog.
