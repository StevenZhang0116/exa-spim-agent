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

## Current criteria

The split-repair policy now has THREE accept paths for SplitSites; MergeSites are
left alone (no `split_label`). Path 1 uses the strict WORST-arm (both-arms)
geometry test (`_is_colinear_split`); Paths 2 and 3 (image-gated) use the MEAN
colinear_cos (matching the failure report's column) with a weak-arm floor
(`MIN_ARM_COS`).

GAP WINDOWS (Gen 8): the TOP-LEVEL candidate cap is now `IMAGE_MAX_GAP_UM` (8.0 µm)
— wider candidates now reach the paths instead of being discarded at 4 µm. But only
the strongest path inherits the wider reach: **Path 2** fires in the gap window
`[IMAGE_GAP_MIN_UM, IMAGE_MAX_GAP_UM]` = `[1.0, 8.0]` (mean colinearity >= 0.75 AND a
bright continuous fluorescence bridge >= 0.90 confirm one neuron across the wide
gap), while **Path 1** (now with an explicit `<= GAP_THRESHOLD_UM` guard) and
**Path 3** (its own `<= GAP_THRESHOLD_UM` bound) remain capped at
`GAP_THRESHOLD_UM` = 4.0 µm. So no gap > 4 µm is ever merged without image
confirmation.

  1. **Strict geometry-only accept** — emit `merge_labels` when BOTH the gap is
     small (`s.gap_um <= GAP_THRESHOLD_UM`, 4.0 µm — now an EXPLICIT guard in the
     code, since the top-level cap is wider at 8 µm; Path 1 can NEVER fire above
     4 µm) AND the two fragments are colinear across the gap via the WORST-arm test
     (`_is_colinear_split`, BOTH half-cosines `>= MIN_COLINEAR_COS`, 0.94 ≈ ≤20°
     bend). Behavior at gap <= 4 µm is bit-for-bit UNCHANGED — this is the original
     high-precision path (6 correct / 0 false on train).
  2. **Image-gated recall path (MEAN colinearity + weak-arm floor)** — for
     SplitSites in the gap window `[IMAGE_GAP_MIN_UM, IMAGE_MAX_GAP_UM]` = `[1.0,
     8.0]` µm (Gen 8: this is the ONE path extended past the old 4 µm cap) whose
     MEAN colinear_cos
     (`mean_cos = (c1 + c2) / 2` of the two half-cosines from
     `_colinear_half_cosines`, the SAME quantity the failure report's
     `colinear_cos` column reports) is high, `mean_cos >= IMAGE_MIN_COLINEAR_COS`
     (0.75), AND neither arm points sideways/backward across the gap
     (`min(c1, c2) >= MIN_ARM_COS`, 0.35 ≈ ≤70° bend on the weaker arm), read the
     fluorescence bridge (`gap_bridge_evidence(node_a, node_b)`) and accept ONLY
     when `bridge_ratio >= IMAGE_BRIDGE_RATIO_MIN` (0.90) — signal stays ≥90%
     bright all the way across, i.e. one continuous neuron. This is the Gen 7
     change: the test was the WORST-arm (both-arms) `_is_colinear_split`, which
     rejected reals whose MEAN clears 0.75 but whose weaker arm dips just below it
     (one straight arm + one slightly bent arm); the MEAN test matches the report
     column while `MIN_ARM_COS` still rejects crossings/double-backs.
  3. **Caliber-gated lower-cos image tier** — an ORTHOGONAL recall lever for
     SplitSites with `IMAGE_GAP_MIN_UM <= gap_um <= 4.0` whose MEAN colinear_cos
     falls in the lower band `[IMAGE_LOWCOS_MIN_COLINEAR_COS,
     IMAGE_MIN_COLINEAR_COS)` = `[0.60, 0.75)` (so this tier only ADDS new
     candidates below the Path-2 mean floor, never double-counts), with neither
     arm sideways/backward (`min(c1, c2) >= MIN_ARM_COS`, 0.35) AND MATCHED
     neurite caliber across the gap (`rad_ratio <= RAD_RATIO_MAX`, 1.4 —
     `rad_ratio` is `max(r_a, r_b)/min(r_a, r_b)` of the two endpoint radii, an
     orthogonal feature). Accept ONLY when the fluorescence bridge is bright
     (`bridge_ratio >= IMAGE_BRIDGE_RATIO_MIN`, 0.90, unchanged). The radius lookup
     (`_rad_ratio`) is fully defensive: it tries `s.rad_ratio`, then `s.rad_a`/
     `s.rad_b`, then a `ctx["node_radius"]`/`g.node_radius` array indexed at the two
     endpoints, and returns a LARGE sentinel (`inf`) on any failure so an
     unavailable radius simply SKIPS this tier (degrades to parent behavior).

**Why path 3 is safe:** the 3 train false joins are each excluded by EITHER the
0.60 cos floor (`colinear_cos` 0.44 and 0.06 fall below it) OR the caliber gate
(the remaining one at cos 0.67 has `rad_ratio` 1.87 > 1.4); every observed false
join has `rad_ratio >= 1.67`, whereas a true single-neuron break has MATCHED
caliber on both sides. So the tier harvests MISSED reals in the `[0.60, 0.75)`
colinearity band that have matched caliber AND a bright (≥0.90) fluorescence bridge
confirming one continuous neuron. Caliber (`rad_ratio`) is ORTHOGONAL to colinearity
(`cos`), so it adds a genuinely new discriminator rather than re-cutting the same
axis. The cloud bridge read is gated behind cheap geometry (gap band + lower
colinearity + caliber) so only a handful of ambiguous candidates trigger it.

**Why path 2:** recall is still the bottleneck. After gen03, only 30/535 real
splits are accepted and 505 are still MISSED, and the `cos >= 0.80` band is nearly
saturated (almost all reals there are already accepted). The policy read the image
on 26 REAL splits and 0 FALSE joins — no false join reaches the cos floor, because
all 3 train false joins sit at `colinear_cos <= 0.67`. So `colinear_cos`, not the
bridge gate, is the false-join guard (probe false joins are bright, 0.97–0.99, with
`bridge_ratio` AUC 0.86 driven by reals being slightly brighter — so it CANNOT
separate the bright false joins). Lowering the floor from 0.80 to 0.75 harvests the
near-colinear reals in the newly-opened `[0.75, 0.80)` cos pocket (missed reals at
cos ~0.72–0.78) whose bright fluorescence bridge confirms continuity, while staying
a clear margin (0.08 above the highest, 0.31 above the next observed false join) above
the false-join geometry (`cos <= 0.67`). The cloud read is gated behind cheap geometry
(gap band + relaxed colinearity) so only the handful of ambiguous candidates are
read. The gate is parent-relative, so each generation must beat the last accepted
policy, not the no-edit floor.

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
- **Harness:** `candidate_split_sites` broadened from tip-to-tip to
  tip-to-any-node (tip/shaft/branch) within `max_gap_um`; `SplitSite` carries
  `node_a` (always a tip) and `node_b` (the partner, any degree).
- **Harness (merge repair):** added `MergeSite` + `candidate_merge_sites`
  (GT-free branch-based merge detection) and a unified split+merge candidate
  stream; fitness is Edge Accuracy (charges merge errors); the failure report now
  lists baseline merge targets and an over-split watchdog.
- **Gen 2:** added an image-gated recall path for near-colinear SplitSites. For
  gaps `<= 4.0 µm` that fail the strict 0.94 colinear test but fall in the band
  `cos in [0.85, 0.94)` at gap `>= 1.0 µm`, accept the merge only when
  `gap_bridge_evidence.bridge_ratio >= 0.90`. Motivated by the recall bottleneck
  (only 6/535 reals accepted; cos ≈ 0.85–0.94 band of missed reals) and the strong
  bridge_ratio separator (AUC 0.86); the relaxed band stays above the observed
  false-join geometry (`colinear_cos <= 0.67`). The strict 0.94 geometry-only
  accept is retained unchanged.
- **Gen 3:** lowered `IMAGE_MIN_COLINEAR_COS` from 0.85 to 0.80, expanding the
  image-gated recall band to `cos in [0.80, 0.94)`. Motivated by gen02
  (split-repair score 6 -> 23 accepted, 23 correct / 0 false): 512 reals are still
  MISSED and the `cos >= 0.85` band is nearly exhausted, while all 3 train false
  joins sit at `cos <= 0.67` so none reach the cos floor (`bridge_ratio` AUC 0.86).
  This harvests the `[0.80, 0.85)` reals whose bright bridge confirms continuity
  while staying above the false-join geometry, so net-correct rises with zero false
  merges. Strict 0.94 geometry-only accept, `bridge_ratio >= 0.90`, and
  `gap >= 1.0` are all unchanged.
- **Gen 4:** lowered `IMAGE_MIN_COLINEAR_COS` from 0.80 to 0.75, expanding the
  image-gated recall band to `cos in [0.75, 0.94)`. Motivated by gen03 (split-repair
  score 23 -> 30 accepted, 30 correct / 0 false): 505 reals are still MISSED and the
  `cos >= 0.80` band is nearly saturated; the policy read 26 REAL and 0 FALSE joins
  (no false join reaches the cos floor — all 3 train false joins sit at `cos <= 0.67`),
  and `bridge_ratio` (AUC 0.86) cannot separate the bright false joins (0.97–0.99), so
  `colinear_cos` is the sole false-join guard. This harvests the `[0.75, 0.80)` reals
  whose bright bridge confirms continuity while staying a clear margin above the
  false-join geometry (`cos <= 0.67`), so net-correct rises with zero false merges.
  Strict 0.94 geometry-only accept, `bridge_ratio >= 0.90`, and `gap >= 1.0` are all
  unchanged.
- **Gen 6:** added a THIRD accept path — a caliber-gated lower-cos image tier — as
  an ORTHOGONAL recall lever (new constants `IMAGE_LOWCOS_MIN_COLINEAR_COS = 0.60`
  and `RAD_RATIO_MAX = 1.4`). Motivated by the gen05 report: all 3 train false
  joins have low `colinear_cos` (0.44, 0.06, 0.67) AND high endpoint caliber
  mismatch `rad_ratio` (1.97, 1.67, 1.87 — all >= 1.67), an UNUSED feature
  orthogonal to colinearity. A true single-neuron break has MATCHED caliber on both
  sides, so a caliber gate (`rad_ratio <= 1.4`) lets us safely admit reals BELOW
  the 0.75 cos floor. Tier 3 accepts a SplitSite with gap in `[1.0, 4.0]` µm that
  FAILS the Path-2 0.75 colinear test but PASSES the 0.60 floor, has matched caliber
  (`rad_ratio <= 1.4`), and a bright bridge (`bridge_ratio >= 0.90`). Every train
  false join is excluded by EITHER the 0.60 cos floor (0.44, 0.06 below it) OR the
  caliber gate (the 0.67 one has `rad_ratio` 1.87 > 1.4), so the tier harvests
  missed reals in the `[0.60, 0.75)` band with zero false merges. The radius lookup
  (`_rad_ratio`) is defensive (tries `s.rad_ratio`, then `s.rad_a`/`s.rad_b`, then
  a `node_radius` array; returns `inf` sentinel on failure to SKIP the tier). The
  bridge read stays gated behind cheap geometry so only a handful of candidates
  trigger a cloud read. Paths 1 (strict 0.94), 2 (0.75 + bridge 0.90), `gap >= 1.0`,
  and `IMAGE_BRIDGE_RATIO_MIN = 0.90` are ALL unchanged. NOTE: the previously
  rejected bridge relaxation (`IMAGE_BRIDGE_RATIO_MIN` 0.90 -> 0.83, which tied the
  parent at +0.000 held-out and was reverted) is deliberately NOT repeated — this
  generation keeps bridge at 0.90 and adds a new orthogonal feature instead.
- **Gen 7:** switched the IMAGE-gated paths (Paths 2 and 3) from the WORST-arm
  (both-arms) `_is_colinear_split` test to a MEAN colinear_cos test with a weak-arm
  floor (new constant `MIN_ARM_COS = 0.35`). DIAGNOSIS: the failure report's
  `colinear_cos` column is the MEAN of the two half-cosines (arm A into the gap;
  the gap into B's continuing arm), but the policy gated on the WORST arm, so it
  rejected real splits whose MEAN clears the floor while the weaker arm dips just
  below it (one straight arm + one slightly bent arm). The gen06 MISSED-real bucket
  is full of these — e.g. gap 1.82/cos 0.86, gap 1.86/cos 0.78, gap 1.41/cos
  0.63/rad 1.00, gap 1.43/cos 0.70/rad 1.17 — all with report colinear_cos at/above
  a path floor yet rejected by the both-arms gate. FIX: added a
  `_colinear_half_cosines(g, s)` helper returning `(c1, c2)` (or `None` on
  degeneracy) with the EXACT vector construction of `_is_colinear_split` so
  `(c1 + c2) / 2` equals the report's `colinear_cos`; refactored `_is_colinear_split`
  to call it and keep its both-arms semantics bit-for-bit (Path 1 unchanged). New
  Path 2 accepts when `mean_cos >= IMAGE_MIN_COLINEAR_COS` (0.75) AND `min(c1, c2)
  >= MIN_ARM_COS` (0.35) AND bright bridge; new Path 3 accepts when
  `0.60 <= mean_cos < 0.75` AND `min(c1, c2) >= 0.35` AND `rad_ratio <= 1.4` AND
  bright bridge. SAFETY: all 3 train false joins have MEAN colinear_cos <= 0.67, so
  Path 2's 0.75 mean floor excludes all of them; the only false join with mean in
  Path 3's `[0.60, 0.75)` band (mean 0.67) has `rad_ratio` 1.87 > 1.4 and is killed
  by the caliber gate. The `MIN_ARM_COS = 0.35` weak-arm floor additionally rejects
  sideways/backward arms (crossings / double-backs) that a pure mean could let
  through on held-out. Because both-arms `>= T` implies mean `>= T`, EVERY candidate
  the parent accepted still passes (no regression); the change ADDS reals whose
  weaker arm previously failed the both-arms gate, so net-correct rises with zero
  new false merges. This is a DIFFERENT lever than the rejected gen5 bridge
  relaxation and the gen6 caliber gate — `IMAGE_BRIDGE_RATIO_MIN` (0.90),
  `RAD_RATIO_MAX` (1.4), `IMAGE_GAP_MIN_UM` (1.0), and the strict Path-1 0.94
  both-arms test are ALL unchanged.
- **Gen 8:** extended the gap REACH of the high-colinearity image-bridge path. New
  constant `IMAGE_MAX_GAP_UM = 8.0`. DIAGNOSIS: the policy is recall-bound (score
  53 -> 76 last gen) and the biggest untapped pool is GAP — the candidate stream
  enumerates SplitSites out to `max_gap_um = 15.0` µm, but the top-level cap
  (`if s.gap_um > GAP_THRESHOLD_UM: continue`, 4 µm) discarded EVERY candidate wider
  than 4 µm before any path could see it, leaving the large gap>4 tail of the 459
  MISSED reals entirely invisible (the report's MISSED table truncates at 1.89 µm
  with "399 more"). Long breaks of one bright neuron are exactly what a fluorescence
  bridge can confirm. FIX (three coupled edits): (1) raised the top-level cap from
  `GAP_THRESHOLD_UM` (4) to `IMAGE_MAX_GAP_UM` (8) so wider candidates reach the
  paths; (2) added an explicit `s.gap_um <= GAP_THRESHOLD_UM` guard to **Path 1**
  (geometry-only, NO image) so it stays capped at 4 µm — a 6 µm geometry-only merge
  with no image confirmation is exactly the unconfirmed long-range join we must
  avoid; Path 1's behavior at gap <= 4 is bit-for-bit unchanged and it can NEVER
  fire above 4; (3) added an explicit upper bound `s.gap_um <= IMAGE_MAX_GAP_UM` to
  **Path 2**, making its window `[1.0, 8.0]` — this is the ONLY path that gains the
  wider reach, and only when mean colinearity >= 0.75 AND a bright continuous bridge
  (`bridge_ratio >= 0.90`) confirm one neuron across the wide gap. Path 3 already
  had its own `<= GAP_THRESHOLD_UM` bound and is left at 4 µm. The bridge read stays
  gated behind the cheap geometry (mean/min-arm computed first; bridge read only on
  geometry pass), with the NaN guard (`br == br`), all inside the existing
  try/except. SAFETY: every observed train false join sits at gap <= 3.16 µm with
  mean colinear_cos <= 0.67 — they do NOT even reach the new `[4, 8]` window, and
  even within it Path 2's 0.75 mean-colinearity floor sits well above 0.67. The
  fluorescence bridge is MORE discriminative over a longer span (more interior
  voxels to expose a dim valley if the gap actually separates two neurons), so
  requiring `bridge_ratio >= 0.90` across a wider gap is a STRONGER continuity test,
  not a weaker one; geometry-only Path 1 and lower-colinearity caliber Path 3
  deliberately stay at 4 µm so no wide gap is merged without image confirmation.
  This is a DIFFERENT lever than gen5 (bridge threshold), gen6 (caliber gate), and
  gen7 (mean vs worst-arm colinearity): `IMAGE_BRIDGE_RATIO_MIN` (0.90),
  `RAD_RATIO_MAX` (1.4), the mean colinearity test, and `MIN_ARM_COS` (0.35) are ALL
  unchanged.
