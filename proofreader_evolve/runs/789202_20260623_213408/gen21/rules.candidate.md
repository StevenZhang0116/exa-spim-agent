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

## Current criteria (Generation 21 — image-confirmed recall extended to 1.5 µm)

The policy repairs splits (SplitSite → `merge_labels`) via SIX acceptance paths.
All require a small gap first: `s.gap_um <= GAP_THRESHOLD_UM` (4.0 µm).

  1. **Strict colinear path (high precision, unchanged).** Accept when the two
     fragments are colinear across the gap — a straight-line continuation
     (`cos(angle) >= MIN_COLINEAR_COS`, 0.94 ≈ ≤20° bend) via `_is_colinear_split`.
  4. **Distance-only tip-to-tip path (Gen 8, new — image-free).** For VERY small
     gaps (`s.gap_um <= DIST_ONLY_UM`, 2.0 µm), accept when BOTH endpoints are tips
     (exactly one same-segment neighbor each) AND the site is NOT in a dense tangle:
     count other candidate split sites whose `node_a` lies within
     `CLUSTER_RADIUS_UM` (30 µm); VETO the accept if that count exceeds
     `DENSE_VETO_MAX` (3). NO colinearity check and NO image read. Rationale:
     Finding #1 shows Euclidean gap distance alone separates true splits from
     inter-neuron neighbors at AUC 0.998 — at ≤2 µm we are deep below the ~7 µm
     inter-neuron floor, so distance is near-decisive and the noisy gap-direction
     vector should not be allowed to reject these genuine broken neurons. Tip-to-tip
     is the precision guard (joining into a shaft/branch is riskier). The dense-
     neuropil veto (Finding #13) is the ONLY safeguard, and it only BLOCKS crowded
     tangles — it never blocks the isolated/mildly-clustered reals (real splits
     average ~1 neighbor). Placed AFTER Path 1 and BEFORE the image paths, with a
     `continue` so an accepted site is never re-read.
  3. **Small-gap distance-dominant path (Gen 3, new).** For TINY gaps
     (`s.gap_um <= SMALL_GAP_UM`, 1.5 µm), DROP the colinearity requirement entirely
     and accept on an image bridge confirmation alone: read
     `gap_bridge_evidence(s.node_a, s.node_b)` and accept when
     `bridge_ratio >= BRIDGE_RATIO_MIN` (0.85). Rationale: at sub-micron / ~1 µm gaps
     the gap-direction vector `node_b - node_a` is a tiny vector dominated by voxel
     noise once normalized, so `colinear_cos` is essentially random there (REAL splits
     show cos from -0.73 to +0.78 in this band) — colinearity can never rescue these.
     The small gap itself is near-decisive evidence of a true split. This path is
     placed BEFORE Path 2 with a `continue`, so a tiny-gap site is read at most once.
  2. **Image-gated recall path (Gen 1, now handles the 1.5–4 µm band).** For sites
     that FAIL the strict gate,
     accept when ALL hold:
       - moderate colinear alignment: `cos(angle) >= IMG_MIN_COLINEAR_COS` (0.80,
         ≈ ≤37° bend) — a cheap geometric pre-filter that also keeps rejecting the
         lone cos=0.55 FALSE join BEFORE any image read; AND
       - the raw image shows a continuous bright bridge across the gap:
         `gap_bridge_evidence(s.node_a, s.node_b).bridge_ratio >= BRIDGE_RATIO_MIN`
         (0.85; REAL splits probed 0.95–1.03).
     The image reader is read ONLY for the handful of moderately-colinear short-gap
     candidates that pass the cheap geometric pre-filter, respecting the read budget,
     and the whole path is guarded so it no-ops when `read_image_patch` is None.
  7. **Tip-to-shaft small-gap recall via shaft-centerline offset (Gen 15; band
     extended Gen 16 — image-free).** For tip-to-shaft sites (deg_a=1, deg_b=2) with
     `SMALL_GAP_UM < gap <= SHAFT_RECALL_FAR_UM` (1.5–2.5 µm), accept when the tip
     lies within `SHAFT_PERP_UM` (0.8 µm) of the shaft's centerline AND the tip
     tangent aligns with the shaft axis (`|cos| >= SHAFT_ALIGN_COS` = 0.7), with the
     dense-neuropil veto; image-free. The perpendicular offset separates a true
     mid-neurite break (tip ON the line) from a laterally-offset parallel neurite of a
     different neuron (offset ≈ gap). The band was widened from 2.0 → 2.5 µm in Gen 16
     to cover the 2.02–2.21 µm tip-to-shaft MISSED rows that NO path previously
     reached (their `colinear_cos` is noisy and they sat just above the old 2.0 µm
     ceiling); because the precision comes from the perp+align geometry — which is
     gap-independent — and NOT from the gap magnitude, it carries unchanged into the
     wider band. Placed AFTER Path 4 and BEFORE Path 3; its 1.5–2.5 µm band never
     overlaps Path 3's ≤1.5 µm nor Path 4's tip-to-tip accepts (`gap <= DIST_ONLY_UM`,
     unchanged at 2.0 µm).
  8. **Image-free moderate-colinear tip-to-shaft path (Gen 18 — new).** Attacks the
     SAME under-served tip-to-shaft bucket as Path 7 (deg_a=1, deg_b=2, gap 1.5–2.5 µm)
     but via a DIFFERENT discriminator: the gap COLINEARITY itself (the report's single
     strongest precision feature), NOT the perp/align test. For a site with
     `SMALL_GAP_UM < gap <= SHAFT_RECALL_FAR_UM` (1.5–2.5 µm) whose `node_b` is a shaft
     (deg_b == 2), accept image-free when the arms continue straight across the gap:
     `_is_colinear_split(g, s, COLINEAR_TS_COS)` (0.72) — a strict AND of BOTH half-
     cosines (tip-A tangent vs gap vector, and gap vector vs node_b's best-continuing
     neighbor). The floor 0.72 is chosen ABOVE the lone FALSE join's colinear_cos 0.55
     (clear margin; Finding #16 puts false joins near cos 0 / 90°) and BELOW Path 2's
     image floor IMG_MIN_COLINEAR_COS (0.80), so it admits straight tip-to-shaft
     continuations that need NO image bridge — exactly the high-colinear tip-to-shaft
     reals Path 1 (needs 0.94), Path 2 (needs an image bridge) and Path 7 (perp/align)
     all miss. The 2.5 µm gap ceiling keeps it far below the ~7 µm inter-neuron floor
     (Finding #1) and excludes the 3.12 µm FALSE outright; the dense-neuropil veto
     (Finding #13) still applies. Placed AFTER Path 7 and BEFORE Path 3 (the first image
     path), with a `continue` on accept.
  9. **Image-confirmed medium-gap recall path (Gen 20; band lowered to 1.5 µm in
     Gen 21).** Raises recall in the 1.5–3.0 µm gap band that NO geometric path
     reaches (Path 1 needs colinear 0.94; Paths 7/8 cap at `SHAFT_RECALL_FAR_UM` =
     2.5 µm; Path 2 needs colinear 0.80 and these reals have LOW colinear_cos). For a
     site whose `node_a` is a tip and whose partner `node_b` is a tip or shaft
     (`b_deg9 in {1,2}`, never a branch) with `SMALL_GAP_UM < gap <= IMG_FAR_GAP_UM`
     (1.5–3.0 µm), gate the cloud read behind the dense-neuropil veto (Finding #13),
     then ACCEPT only when the raw image shows a continuous bright bridge:
     `gap_bridge_evidence(s.node_a, s.node_b).bridge_ratio >= IMG_FAR_BRIDGE_MIN`
     (0.95; the warm-start probe's 12 REAL splits at 2.79–2.83 µm all read ≥0.95). Gen
     21 lowered the floor from `SHAFT_RECALL_FAR_UM` (2.5) to `SMALL_GAP_UM` (1.5) to
     reach the now-dominant low-colinear tip→shaft reals in 1.5–2.24 µm that Paths 7/8
     reject and Path 2 skips; Paths 7/8 still run first in 1.5–2.5 µm and `continue` on
     accept, so Path 9 only image-reads the sites they rejected. Precision rests on
     GEOMETRY not the image: Finding #1 puts cross-neuron neighbours rarely below ~7 µm
     (true-split gaps peak ~4.5 µm), so this band is overwhelmingly real-split
     territory; the train data has ZERO cross-neuron joins below the lone 3.12 µm
     FALSE, and the 3.0 µm cap excludes it (its bridge_ratio is 1.00, so image alone
     could not reject it) — making 1.5–2.5 µm even safer than the proven 2.5–3.0 µm.
     STRICTLY ADDITIVE: it appends only on a confirmed bridge and does NOT
     `continue`/block on non-accept, so it can never remove an existing Path 2/Path 3
     accept. Placed AFTER Path 8 and BEFORE Path 3.

MergeSites are left alone (no `split_label`) — merge-error repair is DISABLED this
run (the stream is SplitSites only). The gate is parent-relative, so each generation
must beat the last accepted policy on net-correct merges with zero false merges.

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

- **Gen 21 (extend image-confirmed Path 9 recall down to the 1.5 µm floor):**
  *Diagnosis.* Gen 20's Path 9 (image-confirmed recall, 2.5–3.0 µm) was ACCEPTED —
  the split-repair score rose 80 → 190 with ZERO false merges on train. The remaining
  MISSED-real bucket (300 sites) is now dominated by the 1.5–2.24 µm gap band,
  overwhelmingly tip→shaft (deg_b=2) with LOW colinear_cos (<0.7): Path 7 (perp/align)
  and Path 8 (colinear AND-test ≥0.72) reject these on their noisy gap vector, and
  Path 2 skips them (needs colinear ≥0.80). *Lever.* Lower Path 9's floor
  `SHAFT_RECALL_FAR_UM` (2.5) → `SMALL_GAP_UM` (1.5), extending the SAME proven
  bright-bridge mechanism (bridge_ratio ≥ 0.95, dense-neuropil veto) to that band; the
  condition becomes `SMALL_GAP_UM < gap <= IMG_FAR_GAP_UM` (1.5–3.0 µm). No constant
  value, other path, or helper is touched. Paths 7/8 still run first in 1.5–2.5 µm and
  `continue` on accept, so Path 9 only image-reads the low-colinear sites they
  rejected; Path 3 (gap ≤ 1.5 µm, strict `<`) still owns the tiniest gaps (no
  overlap). *Why safe / additive.* Path 9 appends only on a confirmed bridge and does
  NOT `continue`/block on non-accept, so it cannot lose the parent's 190. *Precision.*
  The 3.0 µm cap still excludes the lone 3.12 µm FALSE join (its bridge_ratio is 1.00,
  so the image alone could not reject it — only the gap cap does); the train data has
  ZERO cross-neuron joins below 3.12 µm. **Finding #1** (Euclidean gap distance, AUC
  0.998; cross-neuron neighbours rarely below ~7 µm, true-split gaps peak ~4.5 µm)
  makes the 1.5–2.5 µm band EVEN SAFER for false merges than the proven 2.5–3.0 µm
  band. **Finding #13** (merge errors concentrate in dense tangles, AUC 0.9236,
  UPHELD) — the dense-neuropil veto is retained to gate crowded tangles before any
  read.
- **Gen 20 (image-confirmed medium-gap recall Path 9):** *Diagnosis.* Gen 20's
  parent scores 80 (80 accepted / 410 MISSED reals; the lone FALSE at gap 3.12 µm is
  correctly rejected). A large MISSED-real population sits in the 2.5–3.0 µm gap band
  that NO geometric path reaches: Path 1 needs colinear 0.94; Paths 7/8 cap at
  `SHAFT_RECALL_FAR_UM` = 2.5 µm; Path 2 needs colinear 0.80 and these far-band reals
  have LOW colinear_cos, so the geometry-only AND-test skips them. (Gen 19 tried a
  geometry-only medium-gap Path 9 keyed on the colinear AND-test ≥0.80 and TIED the
  parent — it was reverted precisely because these reals' colinear_cos is too low for
  that test.) *Lever.* Use the IMAGE instead of geometry to reach them: the harness
  Image warm-start probe measured `bridge_ratio` AUC = 0.83 ("strong — image worth
  using") and found 12 REAL splits at gap 2.79–2.83 µm ALL reading bridge_ratio ≥
  0.95. New Path 9: for a tip→(tip|shaft) site (`b_deg9 in {1,2}`, never a branch)
  with `SHAFT_RECALL_FAR_UM < gap <= IMG_FAR_GAP_UM` (2.5–3.0 µm), gate the cloud
  read behind the dense-neuropil veto, then ACCEPT only when `gap_bridge_evidence`
  reports `bridge_ratio >= IMG_FAR_BRIDGE_MIN` (0.95). Added constants
  `IMG_FAR_GAP_UM = 3.0` and `IMG_FAR_BRIDGE_MIN = 0.95`. Placed AFTER Path 8, BEFORE
  Path 3. *Why safe / additive.* The block appends only on a confirmed bright bridge
  and does NOT `continue`/block on non-accept, so it can never remove an existing
  Path 2/Path 3 accept — it cannot lose the parent's 80. *Precision.* Precision rests
  on GEOMETRY, not the image: Finding #1 (Euclidean gap AUC 0.998) puts cross-neuron
  neighbours rarely below ~7 µm with true-split gaps peaking ~4.5 µm, so the
  2.5–3.0 µm band is overwhelmingly real-split territory; the 3.0 µm cap excludes the
  lone 3.12 µm FALSE join outright (its bridge_ratio is 1.00, so the image alone could
  NOT reject it — only the gap cap does). The dense-neuropil veto (Finding #13,
  upheld) is retained to gate crowded tangles before any read. Nothing in Paths
  1/2/3/4/7/8 or the helpers is touched.
- **Gen 18 (image-free moderate-colinear tip-to-shaft Path 8):** *Diagnosis.* The
  last three candidates all TIED the parent (score 79−0=79, REAL 490: 79 accepted /
  411 MISSED, FALSE 1 — byte-for-byte identical reports) and were reverted. Gen 15
  added Path 7 (tip-to-shaft via shaft-centerline perpendicular offset + axis
  alignment) and gained exactly +1. Gen 16 (extend Path 7's gap ceiling to 2.5 µm)
  AND Gen 17 (scale Path 7's perp cap with the gap) BOTH added ZERO further accepts
  (+0.000, reverted) — proving the binding constraint in Path 7 is its ALIGNMENT
  gate, not the gap or the perp cap. The shaft-geometry lever is exhausted. *Lever.*
  Attack the SAME under-served bucket (tip-to-shaft, deg_a=1, deg_b=2, gap 1.5–2.5 µm)
  with a DIFFERENT discriminator — the gap COLINEARITY itself (the report's single
  strongest precision feature). New image-free Path 8: for `SMALL_GAP_UM < gap <=
  SHAFT_RECALL_FAR_UM` and `node_b` a shaft (deg_b == 2), accept when
  `_is_colinear_split(g, s, COLINEAR_TS_COS)` (0.72, a strict AND of both half-
  cosines), with the dense-neuropil veto. Placed AFTER Path 7, BEFORE Path 3.
  Nothing in Paths 1/2/3/4/7 or the helpers is touched. *Why safe / additive.* It
  only ADDS an acceptance route; existing accepts are unchanged, so it cannot lose
  the gen15 +1. It targets reals VISIBLE in the train MISSED bucket — high-colinear
  tip-to-shaft reals no current non-image path accepts (e.g. gap 1.91/cos 0.82,
  1.65/0.78, 2.20/0.78, 1.87/0.67, 2.07/0.69, 2.08/0.65) — so it should add net-
  correct accepts rather than tie. *Precision.* The colinear floor 0.72 excludes the
  lone FALSE join (cos 0.55) with margin; the 2.5 µm gap ceiling excludes the 3.12 µm
  FALSE outright and stays far below the ~7 µm inter-neuron floor; the dense-neuropil
  veto excludes crowded tangles.
  - **Finding #16** (endpoint directional alignment / colinear continuation separates
    true splits from false joins, AUC 0.9322; true continuations point ~straight,
    false joins near 90°/cos 0; GENERALIZES, OK): the colinear floor is the
    discriminator, and 0.72 sits well above the false-join cos-0 regime.
  - **Finding #1** (Euclidean gap distance separates true splits from inter-neuron
    neighbors, AUC 0.998; inter-neuron gaps rarely below ~7 µm; GENERALIZES, OK): the
    ≤2.5 µm gap keeps these far below the inter-neuron floor and below the 3.12 µm
    FALSE.
  - **Finding #13** (merge errors concentrate in dense tangles, AUC 0.9236, UPHELD):
    the dense-neuropil veto still guards.
- **Gen 16 (extend tip-to-shaft recall to the 2.0–2.5 µm band):** *Diagnosis.* The
  gen16 failure report shows REAL 490 (79 accepted / 411 MISSED), FALSE 1 (the lone
  false join at gap 3.12 µm, correctly rejected — zero false merges). The single
  largest MISSED bucket that NO current path can reach is TIP-TO-SHAFT (deg_a=1,
  deg_b=2) in the 2.02–2.21 µm gap band (rows at gap 2.02 / 2.03 / 2.05 / 2.07 /
  2.08 / 2.15 / 2.18 / 2.19 / 2.20 / 2.21). They are MISSED solely because their gap
  exceeds Path 7's old 2.0 µm ceiling; their `colinear_cos` is low/noisy (built on
  the voxel-noise gap vector), so Path 1/2 cannot reach them either. *Lever.* Add a
  tunable `SHAFT_RECALL_FAR_UM = 2.5` and change ONLY Path 7's gap window from
  `SMALL_GAP_UM < gap <= DIST_ONLY_UM` to `SMALL_GAP_UM < gap <= SHAFT_RECALL_FAR_UM`.
  Everything else in Path 7 is unchanged: the `a_tip and b_deg == 2` test, the
  `_shaft_continuation_ok` call with SHAFT_PERP_UM (0.8) / SHAFT_ALIGN_COS (0.7), and
  the dense-neuropil veto. Path 4's tip-to-tip accepts still use `DIST_ONLY_UM` (2.0,
  unchanged — gen14 evidence shows tip-to-tip at >2 µm is unsafe), and SHAFT_PERP_UM /
  SHAFT_ALIGN_COS are unchanged, so the precision geometry is untouched. *Why safe.*
  The precision discriminator is the perpendicular offset of the tip from the reliable
  shaft centerline plus the shaft-axis alignment — both gap-independent — so widening
  the gap ceiling does not relax precision; the gap was never the precision mechanism.
  - **Finding #1** (Euclidean gap distance separates true splits from inter-neuron
    neighbors, AUC 0.998; inter-neuron gaps rarely below ~7 µm): 2.5 µm is still far
    below the inter-neuron floor and below the lone FALSE join at 3.12 µm, so the
    wider ceiling does not admit the false join.
  - **Finding #16** (endpoint directional alignment; true continuation ~parallel, AUC
    0.9322): the shaft-axis alignment gate (SHAFT_ALIGN_COS) rejects perpendicular
    T-junction false joins — unchanged.
  - **Finding #13** (merge errors concentrate in dense tangles, AUC 0.9236, UPHELD):
    the dense-neuropil veto still applies — unchanged.
- **Gen 15 (tip-to-shaft small-gap recall via shaft-centerline offset, image-free):**
  *Diagnosis.* Of 412 MISSED reals, the 2.24/2.45 µm tip-to-tip shells are already
  accepted (Path 1/2) for the high-cos ones, and the low-cos remainder there is
  proven unrecoverable. The LARGEST untapped bucket is TIP-TO-SHAFT (deg_a=1,
  deg_b=2) at gap 1.51–1.99 µm with low `colinear_cos` — excluded by Path 4 (both
  tips), Path 1/2 (cos<0.80), and Path 3 (gap>1.5). Their `colinear_cos` is low ONLY
  because it is built on the voxel-noise gap vector; geometrically many are true
  mid-neurite breaks. *Lever.* A new Path 7: accept a tip-to-shaft site when tip A
  lies within `SHAFT_PERP_UM` (0.8 µm) of the shaft B's RELIABLE centerline (its two
  same-segment neighbors define a real line) AND the tip tangent aligns with that
  axis (`|cos| >= SHAFT_ALIGN_COS` = 0.7), restricted to 1.5–2.0 µm gaps with the
  dense-neuropil veto. The perpendicular offset is exactly the discriminator that
  separates a true break (tip ON the line) from a parallel adjacent neurite of a
  different neuron (offset ≈ inter-neurite spacing ≈ gap). *Why different from /
  fixes the 6 priors.* gen10 & gen13 used the IMAGE, which is blind to lateral
  offset — a bright bridge reads identically for a parallel neurite beside a shaft;
  this lever is geometric and the perpendicular offset is precisely what image
  missed. gen11 used branch-adjacency (crowded). gen12 projected onto a SINGLE noisy
  tip-tangent RAY and accepted nothing; Path 7 uses the reliable two-neighbor shaft
  line and a different degree class (deg_b=2). gen14 used tip-to-tip tangent–tangent
  cosine (parallel neurites pass it). *KB priors.* Finding #16 (endpoint directional
  alignment / continuation discriminator, AUC 0.9322, GENERALIZES/OK) warrants the
  alignment guard; Finding #1 (gap distance, true-split peak ~4.5 µm vs ~7 µm inter-
  neuron floor) keeps gaps ≤ 2.0 µm; Finding #13 (dense-neuropil veto) still applies.
- **Gen 8 (distance-only tip-to-tip recall, dense-neuropil veto, image-free):**
  *Diagnosis.* The policy accepts 66 real splits but MISSES 424 reachable real
  splits, all at `gap_um` ≈ 1.5–2.5 µm with low/near-zero `colinear_cos`. Root
  cause: at ~2 µm the gap-direction vector `node_b - node_a` is voxel-noise-
  dominated once normalized, so the colinear gate wrongly REJECTS genuine broken
  neurons there. *Window mapping (four prior expansions).* gen4 (widen gap + image
  bridge, no tip restriction), gen5 (tip-to-tip + arm-dot ≤ −0.70 + image), and
  gen6 (tip-to-tip + anti-parallel dot ≤ −0.85, no image) were ALL reverted — each
  added held-out FALSE merges, because alignment- and image-based acceptance admit
  well-aligned or bright PARALLEL neurites of DIFFERENT neurons. gen7 (tip-to-tip
  AND ≥2 other split sites within 30 µm) was the ONLY zero-false attempt but added
  ZERO net gain: its ≥2 clustering REQUIREMENT was stricter than the real effect, so
  it caught nothing new. So the safe-but-additive window is narrow: do NOT lean on
  alignment or image; do NOT require clustering. *Chosen lever.* A DISTANCE-ONLY
  acceptance for very small (≤ 2.0 µm) TIP-TO-TIP gaps, with NO colinearity and NO
  image check — admitting exactly the noisy-cos reals the colinear gate drops. The
  424 missed reals all sit at 1.5–2.5 µm, deep below the ~7 µm inter-neuron floor,
  so distance alone is near-decisive there. *Why different/safer than gen4–7.* It
  abandons the alignment AND image feature families entirely (the two that injected
  false merges), and it uses local density only as a VETO of dense tangles — the
  OPPOSITE of gen7's clustering REQUIREMENT — so it can never block the isolated or
  mildly-clustered reals (real splits average ~1 neighbor; the veto fires only
  beyond 3). Tip-to-tip is the precision guard the report calls for (the lone train
  FALSE is a `deg_b == 2` shaft join, excluded by requiring both endpoints to be
  tips).
  - Cross-run KB priors relied on (both GENERALIZE across all 3 brains):
    * **Finding #1** — Euclidean gap distance ALONE separates true splits from
      inter-neuron neighbors at AUC 0.998 (true-split gaps peak ~4.5 µm; inter-
      neuron gaps rarely below ~7 µm; F1-optimal ~6.84 µm; Verdict OK): the RECALL
      grounding — distance-only auto-reconnection is endorsed, and ≤ 2 µm is deep
      inside the distance-safe regime.
    * **Finding #13** — merge errors concentrate in dense fragment-graph tangles
      (paired AUC 0.9236, GENERALIZES all 3 brains, Post-correction Verdict UPHELD):
      the VETO grounding — skip the distance-only accept in a crowded local
      neighborhood, where false-join risk concentrates.
- **Gen 3 (small-gap distance-dominant recall):** the previous attempt (tip-to-tip
  `deg_b==1` + raising the colinear floor to 0.90) TIGHTENED the image path and LOST
  23 net-correct repairs on held-out — it was reverted. Tightening colinearity /
  degree is the WRONG direction: it throws away real splits. The parent scores 28
  correct / 0 false by accepting `cos >= 0.80` on the image path. The remaining 462
  missed REAL splits are dominated by SHORT-GAP, NOISY-COS sites: e.g. gap 0.18 cos
  0.02, gap 1.00 cos -0.31, gap 1.41 cos -0.73, gap 1.41 cos -0.06 — REAL splits with
  cos from -0.73 to +0.78 at gap <= 1.5 µm. Colinearity can NEVER rescue these because
  the gap vector is voxel noise at tiny gaps. Fix: a distance-dominant small-gap path
  (`gap_um <= SMALL_GAP_UM`, 1.5 µm) that accepts on an image `bridge_ratio >= 0.85`
  confirmation ALONE, with no colinearity check — raising recall above 28 while
  staying false-free (the lone train FALSE join is at gap 3.12 µm, OUTSIDE the band).
  The new merges fire only on an image read, so image is spent to buy recall; a failed
  read returns NaN and the site is skipped, so this can only ADD correct repairs.
  - Cross-run KB priors relied on (both Verdict OK / GENERALIZES):
    * **Finding #1** — Euclidean gap distance alone separates true splits from
      inter-neuron neighbors nearly perfectly (AUC 0.998; true-split gaps peak
      ~4.5 µm, inter-neuron gaps rarely below ~7 µm; F1-optimal ~6.84 µm): grounds
      trusting a small gap as near-decisive evidence of a true split and dropping
      colinearity there.
    * **Finding #21** — true inter-segment split gaps have a tight characteristic
      scale (99th pct ~6.5 µm, 100% under 15 µm): a <= 1.5 µm gap is deep inside the
      true-split regime, far from any inter-neuron join.
- **Gen 1 (image-confirmed recall):** the seed missed 487/490 reachable REAL
  splits on train because the strict 0.94 colinear gate is too conservative (it
  accepted only the 3 splits at cos 0.94/0.98/0.98). Many missed REAL splits sit in
  the moderate-colinear band the strict gate just clears (e.g. cos 0.84 @ gap 1.19,
  0.87 @ 1.65, 0.90 @ 1.73, 0.78 @ 1.46, 0.74 @ 1.17, plus ~427 more). The fix adds
  a SECOND acceptance route: for short gaps (<= 4 µm) with moderate alignment
  (`cos >= 0.80`), read `gap_bridge_evidence` and accept only if
  `bridge_ratio >= 0.85` (a continuous bright bridge; REAL probed 0.95–1.03). The
  0.80 geometric floor is chosen to keep rejecting the lone cos=0.55 FALSE join
  (deg_b 2, rad_b 1.99) BEFORE any image read — image alone can't reject it
  (its bridge_ratio ~1.00), so geometry must. Warrant for trusting the image on this
  brain: the harness warm-start probe measured `bridge_ratio` separability AUC=0.83.
  - Cross-run KB priors relied on (all Verdict OK / GENERALIZES):
    * **Finding #16** — endpoint directional alignment across split gaps is a strong
      discriminator (true continuation ~153° vs ~90° false, AUC ~0.93): justifies
      using colinear alignment as the cheap gate and that a moderate-alignment band
      still corresponds to true continuations.
    * **Finding #1** — Euclidean gap distance alone separates true splits from
      inter-neuron neighbors (AUC 0.998, F1-optimal ~6.84 µm); and **Finding #21** —
      true split gaps are tight (99th pct ~6.5 µm): justify keeping the gap floor
      short (<= 4 µm) so the recall path stays in the safe distance regime.
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
