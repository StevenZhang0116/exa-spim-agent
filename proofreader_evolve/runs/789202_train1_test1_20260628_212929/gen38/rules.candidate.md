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

  **(D) Mid-gap colinear continuation, degree-conditioned floor** (Gen 15, floor
  split by endpoint degree in Gen 18). For a SplitSite in the band
  `NEAR_GAP_UM < gap_um <= MID_GAP_UM` (2.15–3.0 µm) that is **tip-to-tip**
  (`deg_b == 1`) OR **tip-to-shaft** (`deg_b == 2`) — `node_a` always a tip —
  accept when `ctx["split_geom"](s).colinear_cos` clears a **degree-conditioned**
  floor:
    - **tip-to-tip** (`deg_b == 1`): `colinear_cos >= TIP_TIP_COLINEAR_COS`
      (**0.50**);
    - **tip-to-shaft** (`deg_b == 2`): accept at `colinear_cos >= MID_COLINEAR_COS`
      (**0.75**, strict, unchanged) **OR** at `colinear_cos >= SHAFT_COLINEAR_FLOOR`
      (**0.50**) **AND** `_through_line_cos(g, s, TANGENT_WALK_UM) >=
      THROUGH_LINE_COS` (**0.70**) — a double-gated relaxed path (Gen 19) — **OR**, at
      LOW bridge-vector colinear (`0.0 <= colinear_cos < 0.50`), a DECOUPLED
      high-through-line path (Gen 21): accept when `_through_line_cos(g, s,
      TANGENT_WALK_UM) >= THROUGH_LINE_STRONG` (**0.85**), gated only by the
      `colinear_cos >= 0.0` doubling-back guard.
  The deg_b==2 ordering is: `cc >= 0.75` → accept; `cc` in [0.50, 0.75) →
  `through_line >= 0.70`; `cc` in [0.0, 0.50) → `through_line >= 0.85`; `cc < 0` →
  reject. This reads the harness's leak-free per-site geometry directly and fails
  safe (no accept) when `split_geom` is missing or returns nothing.

  *Why the LOW-colinear deg_b==2 path is DECOUPLED (Gen 21):* for a tip-into-the-SIDE
  of a shaft, the bridge-vector `colinear_cos` is mechanically corrupted by lateral
  offset (the bridge vector tilts), so it falls BELOW the 0.39 in-band false ceiling
  even for genuine continuations whose ARM runs along the shaft axis — the bridge
  magnitude cannot recover them. `_through_line_cos` measures the orthogonal, RELIABLE
  signal (|arm-A tangent · shaft local axis|) on a degree-2 anchor that has a
  well-defined local axis, so it isolates true axial continuation. The in-band
  `deg_b == 2` false joins are grazes (gap 2.19 / colinear 0.39, gap 2.71 / colinear
  0.10) whose arm CROSSES the shaft, so their shaft-alignment |dot| stays low and will
  not reach the VERY HIGH 0.85 bar; this is STRICTLY stricter on the through-line axis
  than the gen19 conjoined path (which used 0.70), and gen19 already proved
  `through_line >= 0.70` transfers to held-out with zero false. The `colinear_cos >= 0.0`
  guard rejects a backward/doubling-back arm.

  *Why the tip-to-shaft floor is double-gated (Gen 19):* `colinear_cos` is the
  AVERAGE of the two arm cosines, and for tip-to-shaft it is DEPRESSED by the
  noisy/laterally-offset shaft-side arm — so genuine tip-to-shaft continuations in
  (2.15, 3.0] with `colinear_cos` 0.50–0.75 are wrongly rejected by the strict 0.75
  floor. The relaxed `SHAFT_COLINEAR_FLOOR = 0.50` sits **0.11 above** the **0.39**
  in-band `deg_b == 2` false-join ceiling (the only in-band `deg_b == 2` false joins
  are gap 2.19 / colinear 0.39 and gap 2.71 / colinear 0.10), so `colinear >= 0.50`
  keeps TRAIN false at 0 by colinear alone. The conjoined `_through_line_cos >= 0.70`
  test is the INDEPENDENT held-out precision hedge: it is a bridge-BYPASSING
  tangent-to-tangent continuity measure (it compares arm A's outward tangent to the
  shaft's local axis at `node_b`, not the gap bridge vector), so requiring BOTH
  continuity signals to agree guards against held-out false joins in the relaxed
  `colinear` [0.50, 0.75) range. This relaxed path is NOT used for tip-to-tip
  (`deg_b == 1`, which has its own `TIP_TIP_COLINEAR_COS`) and is NOT a standalone
  floor — it fires only as the AND-conjunction above.

  *Why the floor is degree-conditioned (Gen 18):* `colinear_cos` is the AVERAGE of
  the two arm cosines. For a **tip-to-tip** join BOTH endpoints are degree-1 tips
  with reliable tangents, so the average is trustworthy. For a **tip-to-shaft** join
  the shaft-side arm tangent is noisy / laterally offset, which drags the average
  DOWN, so a genuinely-colinear tip-to-shaft continuation can score lower than its
  true straightness — that side keeps the stricter 0.75. A LOWER floor for
  tip-to-tip is precision-safe because the evidence is cleaner: within tier (D)'s
  band the ONLY tip-to-tip FALSE joins sit at `colinear_cos` −0.01 (gap 2.24) and
  0.17 (gap 2.45), an in-band tip-to-tip false ceiling of **0.17**, so the 0.50
  floor carries a **0.33 margin** (comparable to the 0.36 margin of the
  proven-transferable 0.75 tip-to-shaft floor) while admitting MISSED tip-to-tip
  real splits at `colinear_cos` 0.50–0.73 (e.g. gap 2.24 / colinear 0.71, 0.73,
  0.53, 0.58) that the old uniform 0.75 floor rejected.

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
- **Gen 18 (2026-06-29) — CANDIDATE (pending the held-out gate):** made tier (D)'s
  colinear floor **degree-conditioned**. *Parent* = **461 correct / 0 false** on
  train (the gen15 line). *History note:* gen17 tried a caliber lever in this band
  (a `rad_ratio >= 2.1` acceptance floor) and was **REJECTED** — it created a
  held-out false merge despite zero train in-band false, confirming
  caliber-mismatch acceptance is NOT precision-safe across brains, so this
  generation does NOT re-introduce any caliber / `rad_ratio` acceptance. *The ONE
  change:* tier (D) previously used a single `MID_COLINEAR_COS = 0.75` floor for
  both endpoint topologies; now tip-to-tip (`deg_b == 1`) accepts at
  `colinear_cos >= TIP_TIP_COLINEAR_COS` (**0.50**, new constant added immediately
  after `MID_COLINEAR_COS`) while tip-to-shaft (`deg_b == 2`) keeps the unchanged
  `MID_COLINEAR_COS = 0.75`. *Precision rationale:* `colinear_cos` is the AVERAGE
  of the two arm cosines; for tip-to-tip both endpoints are degree-1 tips with
  reliable tangents so the average is trustworthy, whereas for tip-to-shaft the
  shaft-arm tangent is noisy and drags the average down (hence it keeps 0.75).
  Within tier (D)'s band (2.15, 3.0] the only tip-to-tip FALSE joins are at
  `colinear_cos` −0.01 (gap 2.24) and 0.17 (gap 2.45) — an in-band tip-to-tip
  false ceiling of **0.17**, so the 0.50 floor carries a **0.33 margin**
  (comparable to the 0.36 margin of the proven-transferable 0.75 tip-to-shaft
  floor) while admitting MISSED tip-to-tip reals at `colinear_cos` 0.50–0.73.
  *Generalization rationale:* the lever is grounded in endpoint-degree topology
  (which tangents are reliable) plus tangent-continuity — both brain-independent —
  NOT in caliber or per-brain proximity, so it is expected to transfer (unlike
  gen17's rejected caliber path). *Why this DIFFERS from prior reverted colinear
  levers:* it conditions the SAME average `colinear_cos` floor on endpoint DEGREE
  rather than re-slicing one uniform threshold — distinct from Gen 4 (global
  `MIN_COLINEAR_COS` → 0.85), Gen 6 (near-band colinear-OR), Gen 11 (per-arm
  `min(cos_a, cos_b)`), and Gen 8 (through-line window). Image NOT used
  (`bridge_ratio` AUC 0.52 — no REAL/FALSE separation; the recall gap is geometric,
  not photometric). Tiers (A), (B), (C), the outer `GAP_THRESHOLD_UM` guard,
  `MID_COLINEAR_COS` itself (tip-to-shaft stays 0.75), all other constants,
  `ENUM_PARAMS`, and the helpers are unchanged. Mark CANDIDATE pending the held-out
  gate.
- **Gen 19 (2026-06-29) — CANDIDATE (pending the held-out gate):** added a
  double-gated RELAXED colinear floor to tier (D) for tip-to-shaft (`deg_b == 2`).
  *Parent* = **507 correct / 0 false** on train: gen18's degree-conditioned floor
  (the tip-to-tip `TIP_TIP_COLINEAR_COS = 0.50` path) was ACCEPTED (461 → 507) and
  TRANSFERRED to held-out, confirming the leak-free `split_geom.colinear_cos` lever
  generalizes. *Diagnosis:* the still-widening train/held-out gap is driven by the
  tip-to-shaft `deg_b == 2` pool in the near band JUST above `NEAR_GAP_UM = 2.15`,
  which `colinear_cos` UNDER-SCORES: it is the AVERAGE of the two arm cosines and is
  dragged DOWN by the noisy / laterally-offset shaft-side arm, so genuine
  tip-to-shaft continuations with true `colinear` 0.50–0.75 are rejected by the
  strict 0.75 floor. *The ONE change:* added a module constant
  `SHAFT_COLINEAR_FLOOR = 0.50` (immediately after `TIP_TIP_COLINEAR_COS`) and a
  THIRD tier-(D) degree-dispatch branch — for `deg_b == 2`, in addition to the
  unchanged `colinear_cos >= MID_COLINEAR_COS` (0.75) path, accept when
  `colinear_cos >= SHAFT_COLINEAR_FLOOR` (0.50) **AND**
  `_through_line_cos(g, s, TANGENT_WALK_UM) >= THROUGH_LINE_COS` (0.70). *Precision
  rationale:* the relaxed `colinear >= 0.50` floor sits **0.11 above** the **0.39**
  in-band `deg_b == 2` false-join ceiling (the only in-band `deg_b == 2` false joins
  are gap 2.19 / colinear 0.39 and gap 2.71 / colinear 0.10), so it keeps TRAIN
  false at 0 on its own; the conjoined `_through_line_cos >= 0.70` is a SECOND,
  INDEPENDENT continuity signal (bridge-bypassing tangent-to-tangent) that must
  agree, hedging against held-out false joins in the relaxed `colinear` [0.50, 0.75)
  range. *Why this DIFFERS from prior rejected attempts:* gen8 was a through-line
  GAP-WINDOW extension via an OR / caliber path (no conjunction, and a window
  change); gen17 was a caliber `rad_ratio` floor that created a held-out false
  merge; gen11 used a per-arm `min(cos_a, cos_b)`; gen13 used `tip_tangent_cos`.
  Here through-line is an AND PRECISION-CONJUNCTION that ENABLES a relaxed colinear
  floor within the EXISTING (2.15, 3.0] band — no gap-window change, no caliber, no
  per-arm min. Image NOT used (`bridge_ratio` AUC 0.52 — REAL and FALSE fully
  overlap; the recall gap is geometric, not photometric). Tiers (A), (B), (C), the
  outer `GAP_THRESHOLD_UM` guard, `ENUM_PARAMS`, all helper bodies, and the constant
  values `TIP_TIP_COLINEAR_COS` (0.50), `MID_COLINEAR_COS` (0.75, tip-to-shaft
  strict path unchanged), and `THROUGH_LINE_COS` (0.70) are untouched. Mark
  CANDIDATE pending the held-out gate.
- **Gen 21 (2026-06-29) — CANDIDATE (pending the held-out gate):** added a DECOUPLED high-through-line acceptor to tier (D) for tip-to-shaft (`deg_b == 2`). *Parent* = **528 correct / 0 false** on train (gen19's line; gen20's tip-to-tip `TIP_TIP_RELAX_FLOOR` path added zero net correct on train — no missed tip-to-tip stub reached `through_line >= 0.70` — so it TIED and was REVERTED). *Diagnosis:* the dominant remaining MISSED real-split pool is tip-to-shaft in the band (2.15, 3.0] with LOW `split_geom.colinear_cos` (0.0–0.5), BELOW the 0.39 in-band `deg_b == 2` false ceiling, so the bridge-vector colinear cannot recover them. For a tip-into-the-SIDE-of-shaft join the bridge-vector `colinear_cos` is mechanically corrupted by lateral offset; the orthogonal, RELIABLE signal is shaft-axis alignment via `_through_line_cos` (|arm-A tangent · shaft local axis|), measured on a degree-2 anchor with a well-defined local axis (unlike a short noisy stub). *The ONE change:* new module constant `THROUGH_LINE_STRONG = 0.85` (immediately after `SHAFT_COLINEAR_FLOOR`) and a THIRD tier-(D) `deg_b == 2` branch — for `colinear_cos` in [0.0, 0.50), accept when `_through_line_cos(g, s, TANGENT_WALK_UM) >= THROUGH_LINE_STRONG` (0.85), gated by `colinear_cos >= 0.0` (a doubling-back guard). Ordering: `cc >= 0.75` → accept; `cc` in [0.50, 0.75) → `through_line >= 0.70`; `cc` in [0.0, 0.50) → `through_line >= 0.85`; `cc < 0` → reject. *Precision rationale:* the in-band `deg_b == 2` false joins are grazes (2.19/colinear 0.39, 2.71/colinear 0.10) whose arm CROSSES the shaft, so their shaft-alignment |dot| is low and will not reach 0.85; gen19 already proved `through_line >= 0.70` (conjoined with colinear ≥ 0.50) transfers to held-out with zero false, and 0.85 is STRICTLY stricter on that axis. *Generalization rationale (the widening train/held-out gap):* shaft-axis tangent continuity is brain-independent geometry (the endorsed generalizable feature), and the change targets the corrupted-colinear pathology with the orthogonal reliable signal rather than chasing train-specific near-band recall. *Why this DIFFERS from prior reverted/exhausted attempts:* gen8 extended the through-line GAP WINDOW (OR / caliber, gap-cap change) — this keeps the (2.15, 3.0] band; gen13 used the `split_geom.tip_tangent_cos` feature — this uses the `_through_line_cos` helper (shaft-axis alignment); gen17 used a caliber `rad_ratio` floor; gen19/gen20 CONJOINED through-line with a colinear floor — this DECOUPLES it (high bar, colinear sign-guard only). Image NOT used (`bridge_ratio` AUC 0.52). Tiers (A), (B), (C), the outer `GAP_THRESHOLD_UM` guard, `ENUM_PARAMS`, all helper bodies, the `deg_b == 1` branch (`cc >= TIP_TIP_COLINEAR_COS`), the other two `deg_b == 2` branches, and the constant values `TIP_TIP_COLINEAR_COS` (0.50), `SHAFT_COLINEAR_FLOOR` (0.50), `MID_COLINEAR_COS` (0.75), `THROUGH_LINE_COS` (0.70), `TANGENT_WALK_UM` (6.0) are untouched. (Note: the parent restored to the gen19 line, so `TIP_TIP_RELAX_FLOOR` and gen20's second `deg_b == 1` branch are NOT present in this file — gen20 tied parent and was reverted.) Mark CANDIDATE pending the held-out gate.
- **Gen 23 (2026-06-29) — CANDIDATE (pending the held-out gate):** added a
  LOW-colinear TIP-TO-TIP (`deg_b == 1`) acceptor to tier (D), REPLACING gen22's
  rejected caliber-only floor. *Parent* = **gen21 line, 649 correct / 1 false** on
  train (the lone train false is a `deg_b == 2` parallel graze at gap 2.19 /
  colinear 0.39 / rr 1.97; the gate scores false on HELD-OUT, where the parent has
  0). *History note:* gen22 added a tip-to-tip branch `elif deg_b == 1 and cc >=
  TIP_TIP_RELAX_COS (0.30)` guarded by `rad_ratio <= TIP_TIP_RAD_MATCH_MAX (1.3)`
  and was **REJECTED**; gen20 added a tip-to-tip `cc >= 0.30 AND through_line >=
  0.70` path and TIED (reverted). BOTH kept a `cc >= 0.30` colinear FLOOR, which
  excludes the bulk of the missed 2.24 µm tip-to-tip pool (colinear 0.0–0.30). This
  generation REPLACES gen22's caliber-only branch with a new, distinct branch and
  DROPS the colinear floor to a mere `cc >= 0.0` sign guard. *Diagnosis:* the
  report's `deg_b` attribution bucket shows `deg_b < 2` (TIP-TO-TIP) = **158
  correct / 0 FALSE** — a zero-false regime the report explicitly flags as worth
  EXTENDING — while the only train false sits in `deg_b >= 2`. The biggest reachable
  MISSED pool is tip-to-tip at gap ~2.24 µm with LOW `colinear_cos` (0.0–0.47),
  which the parent's `deg_b == 1` branch (`cc >= TIP_TIP_COLINEAR_COS = 0.50`)
  rejects. The train/held-out generalization gap is +366.4 and WIDENING, so the fix
  must rest on brain-independent geometry/topology, not train-specific colinear
  magnitude. *The ONE change:* new module constant `TIP_TIP_RAD_MATCH_MAX = 1.3`
  (immediately before `THROUGH_LINE_STRONG`) and a NEW tier-(D) branch placed
  AFTER the existing `deg_b == 1 and cc >= TIP_TIP_COLINEAR_COS` branch and BEFORE
  the first `deg_b == 2` branch: for `deg_b == 1` with `cc is not None and cc >=
  0.0`, accept when `_through_line_cos(g, s, TANGENT_WALK_UM) >= THROUGH_LINE_COS`
  (**0.70**, the two outward tip tangents are antiparallel — the cables point at
  each other) **AND** `_rad_ratio(ctx, s.node_a, s.node_b) <= TIP_TIP_RAD_MATCH_MAX`
  (**1.3**, same cable caliber across the break). The pre-existing `cc >= 0.50`
  tip-to-tip branch stays ABOVE it (so cc ≥ 0.50 tip-to-tip still accept on colinear
  alone). The unused `TIP_TIP_RELAX_COS = 0.30` is not present in this file (gen22
  was reverted), so there is no dangling reference. *Precision argument:* the only
  two in-band tip-to-tip FALSE joins are `2.24 / colinear -0.01 / rr 1.25` (excluded
  by `cc >= 0.0`, since −0.01 < 0) and `2.45 / colinear 0.17 / rr 1.56` (excluded by
  `rad_ratio 1.56 > 1.3`), so this path keeps train false == 0 on tip-to-tip.
  *Generalization rationale:* this is the FIRST time through-line continuity
  (gen20's signal) and caliber-match (gen22's signal) are CONJOINED, AND it drops the
  `cc >= 0.30` floor to `cc >= 0.0`, admitting the low-colinear (0.0–0.30) genuine
  tip-to-tip breaks BOTH prior gens excluded — new, reachable recall. For
  `deg_b == 1`, `_through_line_cos` measures whether the two outward tip tangents are
  antiparallel (pure topology, brain-independent; a parallel graze gives parallel
  tangents ⇒ through_line < 0 ⇒ excluded), and caliber match (`rad_ratio ~ 1`) is
  likewise brain-independent (a broken neuron keeps its caliber across the break).
  Requiring BOTH plus `cc >= 0` is THREE independent generalizable signals that must
  agree — the precision posture the widening +366.4 gap demands. Image NOT used
  (`bridge_ratio` AUC 0.52 — REAL 0.95–1.02 vs FALSE 0.92–1.00 fully overlap; the
  recall gap is geometric, not photometric). Tiers (A), (B), (C), the outer
  `GAP_THRESHOLD_UM` guard, `ENUM_PARAMS`, all helper bodies (`_through_line_cos`
  and `_rad_ratio` are REUSED, not redefined), the `deg_b == 1` `cc >= 0.50` branch,
  all three `deg_b == 2` branches, and every other constant are untouched. This
  changelog accretes — all prior entries above are retained, and the header /
  objective / criteria sections are unchanged. Mark CANDIDATE pending the held-out
  gate.
- **Gen 28 (2026-06-29) — CANDIDATE (pending the held-out gate):** extended the
  PROVEN gen23 two-signal recipe (caliber MATCH conjoined with tangent CONTINUITY)
  from tip-to-tip to the tip-to-shaft (`deg_b == 2`) LOW-colinear pool inside tier
  (D). *Parent* = the gen23 line, **662 correct / 1 train-false** (held-out clean).
  *Diagnosis:* gen27 was REVERTED — its shorter-walk MAX through-line for tip-to-tip
  admitted a held-out false; image AUC remains 0.52 (no photometric separation, so
  the recall gap is geometric). The large remaining MISSED pool is the
  `deg_b == 2` LOW-colinear near-band, which the gen21 branch gates ONLY on the
  single-signal through-line `>= THROUGH_LINE_STRONG = 0.85` path with NO caliber
  gate — so genuine same-caliber tip-into-shaft reconnections at MODERATE through-line
  (0.70–0.85) are missed. *The ONE change:* (1) new module constant
  `SHAFT_RAD_MATCH_MAX = 1.15` (immediately after `TIP_TIP_RAD_MATCH_MAX`); (2) the
  gen21 `elif deg_b == 2 and cc >= 0.0:` branch body now accepts on EITHER the
  existing path (i) `through_line >= THROUGH_LINE_STRONG` (0.85, kept VERBATIM) OR
  the new path (ii) `through_line >= THROUGH_LINE_COS` (0.70) AND `rad_ratio <=
  SHAFT_RAD_MATCH_MAX` (1.15). NO new sibling elif was added — the gen21
  `deg_b == 2 and cc >= 0.0` condition already consumes every such site, so a sibling
  would be dead code; the OR-path lives INSIDE that branch body. *Why it generalizes:*
  it REUSES the only recipe that has demonstrably carried to held-out (gen23: caliber
  match + continuity, accepted and transferred), applied to a NEW endpoint-degree
  class via endpoint topology — the endorsed generalizable axis (tangent continuity,
  caliber match, endpoint degree) over train-specific colinear magnitude. *Strictly
  recall NON-DECREASING and zero new train false:* path (i) is gen21 verbatim so no
  currently-accepted real is lost; both in-band `deg_b == 2` FALSE joins are EXCLUDED
  from path (ii) by caliber INDEPENDENT of their through-line value — gap 2.19 /
  colinear 0.39 / rr 1.97 and gap 2.71 / colinear 0.10 / rr 1.20 both have
  `rad_ratio > 1.15` — so train correct can only tie or rise and train false stays 1
  (the lone existing train false at 2.19 stays caught only by path (i), count
  unchanged). *Why this DIFFERS from gen24/25/26/27:* gen24 was a caliber-MISMATCH
  tip-to-tip acceptor (held-out false, REJECTED); gen25 lowered the tip-to-tip
  through-line bar 0.70 → 0.50 (TIED); gen26 widened the tip-to-tip gap band to
  (3.0, 4.0] (empty, TIED); gen27 was a shorter-walk MAX through-line for tip-to-tip
  (held-out false, REJECTED). All four were tip-to-tip or estimation/band tweaks;
  NONE added a caliber gate to the tip-to-shaft (`deg_b == 2`) class. *Honest tie
  risk:* genuine tip-to-shaft reconnections are often caliber-MISMATCHED (a thin
  distal tip rejoining a thick parent shaft, `rad_ratio` up to ~2.8 — see the Gen-9
  note), so the `rad_ratio <= 1.15` gate admits only the SAME-caliber minority; that
  pool may be sparse and the candidate may TIE (held-out +0 → reverted). That is
  acceptable and low-risk — it cannot introduce a train false and, being the proven
  two-signal conjunction, is unlikely to introduce a held-out false. Image NOT used
  (`bridge_ratio` AUC 0.52). Tiers (A), (B), (C), the outer `GAP_THRESHOLD_UM` guard,
  `ENUM_PARAMS`, all helper bodies (`_through_line_cos` and `_rad_ratio` are REUSED),
  every tip-to-tip branch, the `deg_b == 2 and cc >= MID_COLINEAR_COS` and
  `deg_b == 2 and cc >= SHAFT_COLINEAR_FLOOR` branches, and the constant values
  `TANGENT_WALK_UM` (6.0), `THROUGH_LINE_COS` (0.70), `THROUGH_LINE_STRONG` (0.85)
  are untouched. This changelog accretes — all prior entries above are retained, and
  the header / objective / criteria sections are unchanged. Mark CANDIDATE pending
  the held-out gate.
- **Gen 30 (2026-06-29) — CANDIDATE (pending the held-out gate):** added the
  COMPLEMENTARY caliber-MISMATCH clause to the tip-to-shaft (`deg_b == 2`, `cc >=
  0.0`) tier-(D) acceptor. *Parent* = the gen28 line, **666 correct / score 665 /
  1 train-false** (the lone train false is the `deg_b == 2` parallel graze at gap
  2.19 / colinear 0.39 / rr 1.97; held-out clean). *History note:* gen29 (a `cc < 0`
  tip-to-shaft conjunction) found ZERO new train accepts and TIED the parent on
  held-out, so it was REVERTED — we are back at the gen28 parent. *The NEW data
  signal:* this report's `rad_ratio` attribution table (correct/false per bucket)
  shows `[-inf,1.42)` = 208/0, `[1.42,2.04)` = **228/1** (the lone false, rr 1.97,
  lives HERE in the MIDDLE bucket), and `[2.04,inf)` = **230 correct / 0 FALSE** —
  the HIGH caliber-MISMATCH regime is a CLEAN, ZERO-FALSE regime. This is the
  documented physical NORM for a genuine tip-to-shaft reconnection: a thin distal
  tip (rad ~0.75) rejoining a thick parent shaft (rad ~1.9) gives `rad_ratio` up to
  ~2.79 (see the Gen-9 `RAD_RATIO_MAX` note), so caliber MISMATCH here is expected,
  not a red flag. *The ONE change:* (1) new module constant `SHAFT_RAD_MISMATCH_MIN
  = 2.05` (immediately after `THROUGH_LINE_STRONG`), the COMPLEMENT of the gen28
  caliber-MATCH ceiling `SHAFT_RAD_MATCH_MAX = 1.15`; (2) a THIRD OR-clause added
  INSIDE the existing `elif deg_b == 2 and cc >= 0.0:` branch body (NO new sibling
  branch) — accept also when `_through_line_cos >= THROUGH_LINE_COS` (0.70) **AND**
  `rad_ratio >= SHAFT_RAD_MISMATCH_MIN` (2.05). This mines the genuine
  caliber-MISMATCH reals at MODERATE through-line (0.70–0.85) that gen28 path (i)
  (`tl >= 0.85`) and path (ii) (`rr <= 1.15`) both MISS — e.g. report MISSED rows
  gap 2.20/colinear 0.28/rr2.24, 2.20/0.38/rr2.57, 2.21/0.49/rr2.58,
  2.22/0.12/rr2.45, all `deg_b == 2`, `cc` in [0,0.5). *Why train-false-safe:* both
  in-band `cc >= 0` `deg_b == 2` FALSE joins (rr 1.97 @ gap 2.19, rr 1.20 @ gap
  2.71) are BELOW 2.05, so requiring `rad_ratio >= 2.05` provably EXCLUDES both —
  this clause adds ZERO train false; the `[2.04,inf)` = 0-false attribution
  corroborates. *Why generalizable:* `rad_ratio` is a brain-independent ratio (not a
  per-brain distance), and the clause is CONJOINED with the reliable shaft-axis
  through-line continuity (`>= 0.70`) as an independent generalization hedge;
  thin-tip-into-thick-shaft is physics, not train-specific recall. *Distinctness:*
  gen24 was tip-to-TIP caliber MISMATCH (REJECTED, held-out false); gen28 was
  tip-to-shaft caliber MATCH (`rr <= 1.15`, accepted); gen29 was a `cc < 0`
  conjunction (TIED/reverted). This is tip-to-SHAFT caliber MISMATCH at `cc >= 0` —
  the OPPOSITE caliber regime from gen28, newly grounded in the attribution table.
  *Risk:* if the moderate-through-line mismatch pool is sparse a TIE is possible, but
  held-out-false risk is low (it provably excludes both in-band falses, and is
  dual-gated by through-line continuity + the zero-false caliber bucket). Image NOT
  used (`bridge_ratio` AUC 0.52 — no REAL/FALSE separation; the recall gap is
  geometric, not photometric). Tiers (A), (B), (C), the outer `GAP_THRESHOLD_UM`
  guard, `ENUM_PARAMS`, all helper bodies (`_through_line_cos` and `_rad_ratio` are
  REUSED), every tip-to-tip branch, the other two `deg_b == 2` branches, the gen28
  path (i)/(ii) clauses themselves, and the constant values `THROUGH_LINE_COS`
  (0.70), `THROUGH_LINE_STRONG` (0.85), `SHAFT_RAD_MATCH_MAX` (1.15),
  `TANGENT_WALK_UM` (6.0) are untouched. This changelog accretes — all prior entries
  above are retained, and the header / objective / criteria sections are unchanged.
  Mark CANDIDATE pending the held-out gate.
- **Harness:** `candidate_split_sites` broadened from tip-to-tip to
  tip-to-any-node (tip/shaft/branch) within `max_gap_um`; `SplitSite` carries
  `node_a` (always a tip) and `node_b` (the partner, any degree).
- **Harness (merge repair):** added `MergeSite` + `candidate_merge_sites`
  (GT-free branch-based merge detection) and a unified split+merge candidate
  stream; fitness is Edge Accuracy (charges merge errors); the failure report now
  lists baseline merge targets and an over-split watchdog.
- **Gen 31 (2026-06-29) — CANDIDATE (pending the held-out gate):** extended the
  gen30-ACCEPTED tip-to-shaft (`deg_b == 2`) caliber-MISMATCH clause (path iii) into
  the SLIGHTLY-NEGATIVE bridge-colinear (lateral-offset) region with a NEW sibling
  tier-(D) branch. *Parent* = the gen30 line, **678 correct / 1 train-false**: gen30's
  path-iii (`cc >= 0.0`, `tl >= THROUGH_LINE_COS = 0.70` AND `rad_ratio >=
  SHAFT_RAD_MISMATCH_MIN = 2.05`) was ACCEPTED (train accepted 666 → 678, +12) and
  PASSED the held-out gate, confirming the thin-tip-into-thick-shaft caliber-MISMATCH
  regime generalizes. The lone train false stays at gap 2.19 / colinear 0.39 /
  `deg_b == 2` / rr 1.97 (caught by gen28 path i, count unchanged); the report's
  `rad_ratio` bucket `[2.05,inf)` = **227 correct / 0 false** is still a clean
  zero-false regime. *Diagnosis (the unmined completion):* gen30's clause lives inside
  an `elif deg_b == 2 and cc >= 0.0:` branch, so the SAME genuine thin-tip-into-thick-
  shaft reconnections whose AVERAGED bridge `colinear_cos` is tilted JUST below 0 by
  lateral offset (the tip meets the SIDE of the shaft rather than its end) are
  UNREACHABLE. Visible report MISSED rows confirming this pool is non-empty: gap
  2.17 / colinear -0.06 / `deg_b == 2` / rr 2.49 and gap 2.20 / colinear -0.02 /
  `deg_b == 2` / rr 2.79. *The ONE change:* (1) new module constant
  `SHAFT_OFFSET_COS_FLOOR = -0.15` (immediately after `SHAFT_RAD_MISMATCH_MIN = 2.05`);
  (2) a NEW sibling `elif deg_b == 2 and cc is not None and cc >= SHAFT_OFFSET_COS_FLOOR:`
  branch added IMMEDIATELY AFTER the existing `elif deg_b == 2 and cc >= 0.0:` branch
  body (and BEFORE the dedented `if accept:`). Because the preceding branch already
  consumes every `cc >= 0.0` site, this new branch is reached ONLY when `cc` is in
  [-0.15, 0.0) — it is NOT dead code (deg_b==2 sites with cc < 0.0 previously fell
  through unaccepted). It accepts ONLY in the clean zero-false caliber-MISMATCH bucket:
  `_through_line_cos >= THROUGH_LINE_COS` (0.70) AND `rad_ratio >= SHAFT_RAD_MISMATCH_MIN`
  (2.05). *Why doubly train-false-safe:* (a) NO in-band `deg_b == 2` FALSE join has
  colinear < 0 — the only two are at colinear 0.39 and 0.10 — so the colinear-in-
  [-0.15, 0.0) window contains none of them; AND (b) `rad_ratio >= 2.05` sits above
  both their rad_ratios (1.97 @ gap 2.19, 1.20 @ gap 2.71), so even by caliber alone
  both are excluded. The window adds ZERO train false. *Why generalizable:* this is
  gen30's already-held-out-validated clause widened only by a SMALL sign-guard
  relaxation (`-0.15`, admitting a small lateral offset, NOT a true doubling-back
  reversal) into the lateral-offset region; `rad_ratio` is a brain-independent ratio
  and `_through_line_cos` is the reliable shaft-axis continuity signal (both endorsed
  generalizable axes), so it is expected to transfer. *Distinctness:* gen29 was a
  `cc < 0` `deg_b == 2` branch using caliber MATCH (`rr <= 1.15`) — it found an EMPTY
  train pool and TIED / was reverted; THIS uses caliber MISMATCH (`rr >= 2.05`), the
  OPPOSITE caliber regime, whose pool is visibly non-empty (the two MISSED rows above).
  gen30 was the `cc >= 0.0` half of the same mismatch clause; gen24 was tip-to-TIP
  caliber MISMATCH (REJECTED, held-out false). *Risk:* the [-0.15, 0.0) window is
  narrow so a TIE is possible (held-out +0 → reverted), but held-out-false risk is low
  (doubly train-false-safe + reuse of the proven gen30 clause + the reliable
  through-line continuity gate). Image NOT used (`bridge_ratio` AUC 0.52 — no
  REAL/FALSE separation; the recall gap is geometric, not photometric). Tiers (A),
  (B), (C), the outer `GAP_THRESHOLD_UM` guard, `ENUM_PARAMS`, all helper bodies
  (`_through_line_cos` and `_rad_ratio` are REUSED), every tip-to-tip branch, the
  gen30 `cc >= 0.0` branch itself (path i/ii/iii unchanged), the other two
  `deg_b == 2` branches, and the constant values `THROUGH_LINE_COS` (0.70),
  `THROUGH_LINE_STRONG` (0.85), `SHAFT_RAD_MATCH_MAX` (1.15), `SHAFT_RAD_MISMATCH_MIN`
  (2.05), `TANGENT_WALK_UM` (6.0) are untouched. This changelog accretes — all prior
  entries above are retained, and the header / objective / criteria sections are
  unchanged. Mark CANDIDATE pending the held-out gate.
- **Gen 38 (2026-06-29) — CANDIDATE (pending the held-out gate):** restructured the
  tier-(D) LOW-colinear TIP-TO-TIP (`deg_b == 1`) acceptor (the gen23 branch, the
  `elif deg_b == 1 and cc is not None and cc >= 0.0:` branch) so that caliber match
  stays MANDATORY but continuity may now be satisfied by EITHER the preserved
  walk-based through-line OR a gap-REFERENCED both-arms-forward conjunction.
  *Parent* = the accepted **gen31 line** (the on-disk file was reverted to gen31
  after gen37 was rejected). *History note — why gen37 lost:* gen37 added, in this
  SAME branch, `tip_tangent_cos >= 0.80` as an OR-alternative continuity signal
  (caliber match kept mandatory): accept if `rr <= 1.3 AND (tl >= 0.70 OR
  tip_tangent_cos >= 0.80)`. It scored held-out **−1** (admitted a held-out FALSE
  merge) and was **REJECTED**. ROOT CAUSE: `tip_tangent_cos` is gap-direction-
  INDEPENDENT, so it stays HIGH (~+1) for a PARALLEL GRAZE — two different cables
  running side-by-side whose tips merely lie near each other — which is exactly the
  false-join geometry; the OR-relaxation on a gap-independent signal let grazes in.
  *The ONE change:* (1) new module constant `TIP_TIP_BOTH_ARM_COS_FLOOR = 0.15`
  (immediately after `SHAFT_OFFSET_COS_FLOOR = -0.15`); (2) in the `deg_b == 1 and
  cc >= 0.0` branch ONLY, read the two SEPARATE half-cosines `ca = gd.get("cos_a")`,
  `cb = gd.get("cos_b")` and accept when `rr <= TIP_TIP_RAD_MATCH_MAX` (caliber match,
  MANDATORY) **AND** (`tl >= THROUGH_LINE_COS` **OR** (`ca >=
  TIP_TIP_BOTH_ARM_COS_FLOOR` AND `cb >= TIP_TIP_BOTH_ARM_COS_FLOOR`)). The
  walk-`tl` clause is kept FIRST so recall is strictly non-decreasing. *Why this is
  the precise fix for gen37:* `cos_a`/`cos_b` are gap-REFERENCED half-cosines (each
  arm's tangent dotted with the gap direction). The harness docstring states a tip
  grazing a shaft / a parallel graze "tends to have one low cosine" (asymmetry),
  whereas a genuine broken neuron has BOTH arms pointing through the gap. Requiring
  BOTH half-cosines to individually clear a modest 0.15 forward floor (each arm
  within ~81° of the gap axis) REJECTS the parallel-graze geometry that sank gen37,
  while admitting genuine low-MEAN-colinear tip-to-tip reals (where the mean cc is
  dragged down but both arms are in fact mildly forward) that the noisy short-stub
  walk-`tl` misses. *Why PROVABLY train-false-safe (train false stays 0):* the only
  two in-band (gap (2.15, 3.0]) tip-to-tip FALSE joins are excluded REGARDLESS of
  which continuity branch fires — (a) `2.24 / colinear −0.01 / rr 1.25`: mean
  colinear < 0 ⇒ at least one half-cosine is negative ⇒ it CANNOT have both
  cos_a ≥ 0.15 and cos_b ≥ 0.15 (if both were ≥ 0.15 the mean would be ≥ 0.15 > 0,
  contradiction); also blocked by the `cc >= 0.0` elif-guard; (b) `2.45 / colinear
  0.17 / rr 1.56`: rr 1.56 > 1.3 ⇒ excluded by the MANDATORY caliber gate
  `rr <= TIP_TIP_RAD_MATCH_MAX`. The lone deg_b==2 train false (`2.19 / 0.39 /
  rr 1.97 / rad_a 1.00`) is deg_b==2 and is NOT in this deg_b==1 branch. *Why recall
  is strictly NON-DECREASING:* the original walk-`tl >= THROUGH_LINE_COS` acceptance
  is PRESERVED as the FIRST OR-branch, so no currently-accepted real is lost; the
  both-arms-forward branch only ADDS accepts (genuine low-mean-cc tip-to-tip reals at
  gap ~2.24 with rr ≤ 1.3 — e.g. the report's MISSED rows colinear 0.19/rr1.06,
  0.23/rr1.20, 0.31/rr1.20 — that walk-`tl` is too noisy on short tip stubs to see).
  *Why it should GENERALIZE / addresses the +508 train↔held-out gap:* `cos_a`/`cos_b`
  are leak-free and identical train↔held-out (harness docstring); caliber match +
  endpoint degree are brain-independent. This is the proven two-signal CONJUNCTION
  recipe (gap-referenced both-arm continuity AND caliber match), NOT a train-specific
  recall chase; the 0.15 floor is conservative. *Distinctness from prior attempts:*
  gen37 used the gap-INDEPENDENT `tip_tangent_cos >= 0.80` OR-branch (admitted a
  parallel graze, held-out −1) — THIS uses gap-REFERENCED both-arm half-cosines,
  which reject that exact graze geometry; gen32 was a tip-to-tip colinear floor with
  NO conjunction (precision-fragile); gen23 was tip-to-tip via the walk-based
  through-line (preserved here as the FIRST OR-branch); gen22 was tip-to-tip
  caliber-ONLY (no continuity); gen13 used `tip_tangent_cos >= 0.92` for deg_b in
  (1,2) conjoined with a gap-relative `cos_a >= 0.5` SINGLE-arm guard (TIED) — THIS
  requires BOTH half-cosines (symmetry), at a much lower 0.15 floor, deg_b==1 only,
  as an OR-alternative to the preserved walk-tl. Image NOT used / RULED OUT this gen
  (warm-start `bridge_ratio` AUC 0.52 — no REAL/FALSE separation; the recall gap is
  geometric, not photometric). No other branch and no other file changed: the
  standalone `deg_b == 1 and cc >= TIP_TIP_COLINEAR_COS` acceptor, every `deg_b == 2`
  branch, tiers (A)/(B)/(C), the outer `GAP_THRESHOLD_UM` guard, `ENUM_PARAMS`, and
  all helper bodies are untouched. This changelog accretes — all prior entries above
  are retained, and the header / objective / criteria sections are unchanged. Mark
  CANDIDATE pending the held-out gate.
