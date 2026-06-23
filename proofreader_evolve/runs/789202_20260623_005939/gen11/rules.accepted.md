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

## Current criteria (distance-graded colinear split-repair)

A **distance-graded colinear split-repair** policy. For each SplitSite it emits a
`merge_labels` edit only when the gap is small — `s.gap_um <= GAP_THRESHOLD_UM`
(6.5 µm) — AND a colinearity gate passes whose strictness DEPENDS on the gap:
  1. **MICRO regime** — `s.gap_um <= MICRO_GAP_UM` (2.0 µm): distance ALONE is
     decisive, so the colinearity gate is **WAIVED entirely** (not merely loosened).
     Accept the merge on gap alone. Justification: inter-neuron gaps essentially
     never fall below ~7 µm, so a sub-2 µm gap is overwhelmingly one broken neuron;
     at these lattice-scale gaps (√1..√4 voxels = 1.0, 1.41, 1.73, 2.0 µm) the
     tip-tangent angle is dominated by skeletonization jitter and is uninformative.
     2.0 µm sits 0.2 µm below the nearest reachable FALSE join (2.20 µm), so every
     reachable false join is out of band.
  2. **TIP-TO-TIP regime** — `MICRO_GAP_UM < s.gap_um <= TIP_TIP_GAP_UM` (2.3 µm)
     AND BOTH fragment endpoints are tips (`deg_a == 1 AND deg_b == 1`): distance +
     topology are decisive, so the colinearity gate is **WAIVED entirely** (same
     lattice-jitter rationale as MICRO — the dense missed-real cluster sits at the
     √5 = 2.24 µm voxel-lattice diagonal where the tip-tangent angle is
     uninformative). Two genuine TERMINI meeting across a sub-2.3 µm gap is the
     canonical "one neuron broken into two fragments" signature; joining INTO a
     shaft/branch is the error-prone topology, so it is excluded. `deg_a`/`deg_b`
     are computed as the number of graph neighbors of `node_a`/`node_b` in the
     fragments graph (`len(list(g.neighbors(node)))`). 2.3 µm sits 0.15 µm BELOW
     the nearest TIP-TO-TIP false join (2.45 µm), so BOTH tip-to-tip false joins
     (gap 2.45, 3.00 µm) stay out of band; the false join at 2.20 µm is
     tip-into-shaft (`deg_b == 2`) and is excluded by the tip-to-tip requirement,
     falling through to the FAR floor (cos 0.44 < 0.60 → rejected).
  3. **EXTENDED TIP-TO-TIP regime (caliber-gated)** — `TIP_TIP_GAP_UM (2.3) <
     s.gap_um <= TIP_TIP_EXT_GAP_UM (2.5)` AND BOTH endpoints are tips
     (`deg_a == 1 AND deg_b == 1`) AND the endpoint caliber is matched
     (`_rad_ratio(g, ctx, s) <= TIP_TIP_RAD_MATCH`, 1.50): distance + topology +
     caliber are decisive, so the colinearity gate is **WAIVED entirely** (same
     lattice-jitter rationale as the MICRO / TIP-TO-TIP tiers). This recovers the
     dense cluster of MISSED tip-to-tip reals piled up at gap 2.45 (scattered
     `colinear_cos` ≈ −0.81..+0.53, so the colinearity tier rejects them; many also
     have `rad_ratio > 1.30` so the general caliber OR-clause misses them too).
     Distance alone could NOT extend the prior tier here (the tip-to-tip FALSE @2.45
     sits in this band), but CALIBER separates: the captured tip-to-tip reals have
     `rad_ratio <= 1.46` while the only in-band tip-to-tip FALSE has `rad_ratio
     1.56`, so the 1.50 ceiling sits 0.06 BELOW the FALSE and 0.04 ABOVE the reals.
     The ceiling is LOOSER than the general colinearity-tier caliber path's 1.30
     precisely because `deg_a == deg_b == 1` already excludes the high-caliber
     `deg_b == 2` FALSE @2.20 — a topology-specific relaxation. The FALSE @2.45
     (rad 1.56 > 1.50) and FALSE @3.00 (gap > 2.5) both stay out of band → zero new
     train-false merges. Placed immediately AFTER the TIP-TO-TIP tier and BEFORE the
     colinearity tier; purely ADDITIVE (a site it accepts would otherwise fall to
     the colinearity tier), so train-correct cannot drop below the parent's 256.
  4. **TIP-INTO-SHAFT regime (caliber-gated)** — `MICRO_GAP_UM (2.0) < s.gap_um <=
     TIP_SHAFT_GAP_UM (2.75)` AND `deg_a == 1 AND deg_b == 2` (a fragment tip joining
     INTO a shaft node) AND the endpoint caliber is matched (`_rad_ratio(g, ctx, s)
     <= TIP_SHAFT_RAD_MATCH`, 1.50), colinearity WAIVED: distance + topology +
     caliber are decisive. The gap ceiling is a DEDICATED constant
     `TIP_SHAFT_GAP_UM` (2.75), DECOUPLED from the colinearity-tier's
     `RAD_MATCH_GAP_UM` (2.5) which is left unchanged. This widens the band from 2.5
     to 2.75 to capture the dense matched-caliber missed-real cluster at gap
     2.5–2.72; it is FALSE-free because the only reachable `deg_b == 2` FALSE is
     @2.20 (caliber-excluded, rad 1.97 > 1.50) and the next reachable FALSE @3.00 is
     `deg_b == 1` (never seen by this `deg_b == 2` tier), so there is a FALSE-free
     gap window from ~2.2 up to 3.0; 2.75 stays conservatively below 3.0. So
     the colinearity gate is **WAIVED entirely**. The missed-real bucket is now
     dominated by tip-into-shaft sites at gap 2.0–2.65 (`deg_a == 1, deg_b == 2`)
     that fall through EVERY existing path: gap `> TIP_TIP_GAP_UM` (so both tip-to-tip
     tiers skip them on the `deg_b == 1` requirement), low/negative `colinear_cos`
     (so the FAR 0.60 / averaged 0.55 floors reject them), and `rad_ratio` often just
     ABOVE the colinearity-tier caliber cap of 1.30 (so that OR-clause misses them
     too). Colinearity is actively MISLEADING here: the ONLY reachable `deg_b == 2`
     FALSE join (gap 2.20) has `colinear_cos 0.44`, which is HIGHER than roughly half
     the missed reals in this band — so no colinearity floor can recover these reals
     without also admitting it. CALIBER is the better discriminator: that FALSE has
     `rad_ratio 1.97` (clearly MISMATCHED caliber), while the matched-caliber missed
     reals to capture have `rad_ratio <= ~1.49`. The 1.50 ceiling sits 0.47 BELOW the
     FALSE @2.20 — a WIDE margin → zero new train-false merges; the other two
     reachable FALSE joins are `deg_b == 1`, so this `deg_b == 2` tier never sees
     them. Restricted to `deg_b == 2` ONLY (shaft); `deg_b >= 3` branch points are
     rarer/riskier and the missed cluster is all `deg_b == 2`, so they are excluded.
     The lattice-jitter rationale for waiving colinearity is topology-independent
     (tip-arm jitter corrupts the tip direction regardless of `node_b`'s degree),
     exactly as in the MICRO / TIP-TO-TIP tiers. Relative caliber only (no absolute
     radius), so brain-agnostic. Placed immediately AFTER the EXTENDED TIP-TO-TIP
     tier and BEFORE the colinearity tier; PURELY ADDITIVE (a site it accepts
     currently fails all existing paths), so train-correct cannot drop below the
     parent's 267 — it can only ADD net-new correct repairs.
  5. **NEAR regime** — `MICRO_GAP_UM < s.gap_um <= NEAR_GAP_UM`: colinearity floor
     `NEAR_COLINEAR_COS` (0.0) — accept everything except joins that double back
     (anti-parallel / obtuse, `cos < 0`). (Currently EMPTY because `MICRO_GAP_UM`
     (2.0) > `NEAR_GAP_UM` (1.5); kept for clarity / future re-tuning.)
  6. **FAR regime** — `MICRO_GAP_UM < s.gap_um <= 6.5 µm` (i.e. `> 2.0 µm`), for
     sites NOT accepted by the tip-to-tip tiers: accept if EITHER of two paths passes
     (an OR — the second is purely additive and can only ADD accepts):
       - **(a) hard single-arm gate** — `_is_colinear_split(g, s, min_cos)` with
         `min_cos = MIN_COLINEAR_COS` (0.60): requires the tip-arm half-cosine
         `cos_a >= 0.60` as a SEPARATE floor (6.0 µm tip-tangent walk) AND the
         partner continuation `>= 0.60`. This is the original gate, kept unchanged.
       - **(b) averaged cross-gap colinearity** — `_avg_cross_gap_colinearity(g, s)
         >= COLINEAR_AVG_ACCEPT` (0.55). This FAITHFULLY reproduces the harness's
         validated `colinear_cos` (candidate.py::_split_site_geom): `gdir =
         unit(xyz[node_b]-xyz[node_a])`; `cos_a = dot(arm-A outward tangent, gdir)`
         using a SHORT 4.0 µm walk (`HARNESS_TANGENT_WALK_UM`, matching the harness,
         NOT the 6.0 µm the hard gate uses); `cos_b = max over node_b's same-segment
         neighbors of dot(gdir, neighbor_dir)`; the measure is the MEAN
         `0.5*(cos_a + cos_b)`. Because it AVERAGES the two half-cosines instead of
         gating `cos_a` separately, it recovers reals whose tip arm bends slightly
         (`cos_a` ~0.4) but whose partner continues straight (`cos_b` ~0.9; avg
         ~0.65) — exactly the missed-real cluster path (a) wrongly rejects.
       - **(c) caliber-matched recall path** — `_rad_ratio(g, ctx, s) <=
         RAD_RATIO_MATCH` (1.30) AND `avg_col >= 0.0` (not anti-parallel) AND
         `s.gap_um <= RAD_MATCH_GAP_UM` (2.5). `_rad_ratio` reads the endpoint
         radii from `ctx["node_radius"]` (falling back to `g.node_radius`) at
         `s.node_a`/`s.node_b` and returns the RELATIVE ratio `max(ra,rb)/min(ra,rb)`
         (None if no radius array or either radius <= 0). A real "one neuron broken
         in two" has nearly-equal endpoint calibers (ratio ~1.0–1.3); a false join
         fuses two unrelated neurites of DIFFERENT caliber. This recovers the dense
         band of reals just above 2.0 µm whose tip-tangent angle is lattice jitter
         (colinear_cos scattered ~ −0.8..+0.5, so paths (a)/(b) reject them) but whose
         two broken ends have MATCHED caliber. It is RELATIVE caliber only — no
         absolute-radius threshold — so it carries no brain-specific assumption.
     The averaged floor 0.55 sits 0.11 ABOVE the largest reachable FALSE-join
     `colinear_cos` (the 3 false joins are at cos 0.44/0.19/0.06), so every reachable
     false join stays rejected with ~0.11 margin. The caliber ceiling 1.30 sits
     ~0.26 BELOW the smallest reachable FALSE-join `rad_ratio` (the 3 false joins are
     at 1.97/1.56/1.67), so no reachable false join can enter path (c) either. All
     three paths are joined by OR, so each is purely additive (no currently-accepted
     site is dropped) and each still recovers clean mid-gap / caliber-matched
     continuations while staying above the false-join population.
MergeSites are left alone (no `split_label`) — splitting is the riskier edit, left
for the loop to add once it can measure the trade-off. This widens recall (the
prior 0.94 gate accepted only 6 of 540 reachable real splits) while still rejecting
the few false joins, which sit at larger gaps with low colinearity. The gate is
parent-relative, so each generation must beat the last accepted policy, not the
no-edit floor.

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
- **Gen 1 → candidate (distance-graded colinearity gate):** Diagnosis was a RECALL
  problem, not precision. The gen01 policy scored 6 correct − 0 false; among
  reachable SplitSites the class balance is 540 REAL : 3 FALSE, yet it accepted only
  6/540 reals. The binding constraint was `MIN_COLINEAR_COS = 0.94`, NOT the gap:
  hundreds of missed real splits sit at tiny gaps (0.18–1.5 µm) but have low/noisy
  local colinearity (cos 0.0–0.7), so the strict 0.94 gate rejected them. The 3
  reachable FALSE joins sit at gap 2.2–3.0 µm with cos 0.06–0.44. Change: raised
  `GAP_THRESHOLD_UM` 4.0→6.5, lowered `MIN_COLINEAR_COS` 0.94→0.85, and added a
  distance-graded near-gap regime (`NEAR_GAP_UM = 1.5`, `NEAR_COLINEAR_COS = 0.0`)
  so at ≤1.5 µm distance decides and only anti-parallel joins are vetoed. This keeps
  all 6 current hits, rejects all 3 reachable FALSE joins (each at gap > 1.5 µm with
  cos < 0.85), and newly captures the small-gap real splits the 0.94 gate missed.
  Image was deliberately NOT used: the warm-start bridge_ratio AUC was only 0.69 and
  the 3 FALSE joins have HIGH bridge_ratio (0.97–0.99) fully overlapping the REAL
  range (0.94–1.02), so gating on it would not exclude them and would risk a false
  merge (auto-revert). AutoDiscovery support: finding #1 (id 30 — gap-distance AUC
  0.998, inter-neuron gaps rarely < 7 µm, F1-optimal ~6.84 µm), finding #21 (id 63 —
  true split gaps 99% under 6.5 µm), finding #16 (id 33 — angular alignment AUC 0.93,
  true continuation ~153° vs false ~90°); all three GENERALIZE + UPHELD/OK. Avoided
  the non-generalizing distance-angle INTERACTION finding (#25, id 47) — instead
  combined the two independently-robust discriminators (gap and angle) separately.
- **Gen 2 → candidate (lower the FAR-regime colinearity floor):** Diagnosis was
  RECALL-bound, not precision: gen02 scored 31 correct − 0 train-false (= 31, up
  from the parent's 6) with the no-new-merge guard satisfied, yet only 31 of 540
  reachable real splits were accepted — 509 still missed. The binding constraint
  was the FAR floor `MIN_COLINEAR_COS = 0.85`: many missed reals sit at gap
  1.5–2.0 µm with clean colinearity just under 0.85 (e.g. (gap,cos) =
  (1.55,0.62), (1.65,0.78), (1.87,0.67), (1.91,0.82), (1.94,0.76)), while accepted
  FAR reals all had cos >= 0.88. Change: lowered `MIN_COLINEAR_COS` 0.85→0.60
  (FAR regime ONLY; NEAR regime, `NEAR_GAP_UM`, `NEAR_COLINEAR_COS` and
  `GAP_THRESHOLD_UM` all UNCHANGED). Safety: all 3 reachable FALSE joins have
  cos <= 0.44 (0.44, 0.19, 0.06), so a 0.60 floor still rejects every reachable
  false join with ~0.16 margin; and high-colinearity mid-gap pairs are the
  lowest-chaining-risk recall gain, so this was the safe lever. Did NOT loosen the
  NEAR regime: the report shows held-out mega-merges from union-find chaining (a
  5-way fusion on N013, a 4-way on N018 costing −0.530 EdgeAcc / +0.677
  %MergedEdges) that originated in the loose short-gap NEAR regime, so requiring
  strong colinearity beyond 1.5 µm limits new chaining. Image deliberately NOT
  used: warm-start bridge_ratio AUC is only 0.69 and the 3 FALSE joins have HIGH
  bridge_ratio (0.97–0.99) overlapping REAL, so gating on it cannot exclude them.
  AutoDiscovery support: primary finding #16 (id 33 — angular alignment AUC 0.93,
  true continuation ~153° vs false ~90°/cos≈0; a 0.60 cos ≈ 53° bend sits
  comfortably above the false population), with finding #1 (id 30 — gap-distance
  AUC 0.998) and finding #21 (id 63 — true split gaps 99% under 6.5 µm) for
  gap-scale context; all GENERALIZE + UPHELD/OK. Considered but did NOT adopt
  finding #17 (split-on-thin-process radius — magnitude brain-specific) or finding
  #15 (split clustering — chaining-risky).
- **Gen 5 → candidate (micro-gap distance-decisive tier — WAIVE colinearity below
  2.0 µm):** Diagnosis was RECALL-bound, not precision: gen05 scored 91 correct − 0
  train-false (= 91), but of 540 reachable REAL splits only 91 were accepted — 449
  MISSED, and all 3 reachable FALSE joins were correctly rejected (0 wrongly
  accepted). The missed reals cluster heavily at MICRO gaps (0.18–~2.0 µm) where
  `colinear_cos` is scattered across −0.76..+0.77 (e.g. gap 1.41/cos −0.54,
  1.00/cos −0.14, 0.18/cos 0.02): the tip-tangent angle at lattice-scale gaps
  (√1..√4 voxels = 1.0, 1.41, 1.73, 2.0 µm) is dominated by skeletonization jitter
  and is essentially uninformative, so the existing colinearity floor
  (`NEAR_COLINEAR_COS = 0.0`) rejected them on a meaningless `cos < 0`. Colinearity
  does NOT separate these missed reals from the 3 FALSE (their cos ranges overlap);
  the clean separator is GAP DISTANCE — every reachable FALSE sits at gap >= 2.20 µm
  (2.20/cos 0.44, 2.45/cos 0.19, 3.00/cos 0.06). Change: added a new
  `MICRO_GAP_UM = 2.0` constant (0.2 µm below the nearest reachable false join) and a
  MICRO fast-accept tier in `propose_edits` — for `s.gap_um <= MICRO_GAP_UM`, accept
  the merge on DISTANCE ALONE and skip `_is_colinear_split` entirely (the FAR
  cos >= 0.60 path is unchanged for gaps above 2.0 µm). This is a genuinely NEW
  mechanism: prior gens (gen1 NEAR regime, gen2 FAR floor) always kept a colinearity
  floor of at least 0.0 — they never WAIVED it. It converts the large micro-gap
  missed-real bucket into correct repairs (lifting train correct above 91) while
  keeping train false == 0, because all 3 reachable false joins are >= 2.20 µm > 2.0
  µm and thus out of band. NOT re-tuning the colinearity floors (gen4 tied 91 doing
  that) and NOT a component-size cap (gen3 lost recall, held-out −2.000). Image
  deliberately NOT used: warm-start bridge_ratio AUC is only 0.69 and the 3 FALSE
  joins (bridge_ratio 0.97–0.99) overlap REAL (0.94–1.02), so it cannot exclude them.
  AutoDiscovery support: finding #1 (id 30 — gap-distance AUC 0.9979, inter-neuron
  gaps RARELY below ~7 µm, so a sub-2 µm gap is overwhelmingly one broken neuron;
  GENERALIZES + OK) and finding #21 (id 63 — true split gaps 99% under 6.5 µm, tight
  characteristic scale; GENERALIZES + OK). Both qualify (GENERALIZES + UPHELD/OK).
- **Gen 6 → candidate (tip-to-tip distance+topology tier — WAIVE colinearity below
  2.3 µm when BOTH endpoints are tips):** Diagnosis was RECALL-bound, NOT precision:
  the parent scored 163 correct − 0 train-false (= 163), but of 540 reachable REAL
  splits only 163 were accepted — 377 MISSED, all 3 reachable FALSE joins correctly
  rejected. The missed reals now pile up JUST ABOVE the 2.0 µm MICRO cutoff, with a
  dominant cluster at gap = 2.24 µm (= √5, a voxel-lattice diagonal) where
  `colinear_cos` is scattered across ≈ −0.88..+0.55 (lattice-scale skeletonization
  jitter — the SAME noise that justified waiving colinearity in the MICRO tier).
  Distance ALONE could not be extended: the three reachable FALSE joins are at gap
  2.20 (deg_b=2, tip-into-SHAFT), 2.45 (deg_b=1, tip-to-tip), 3.00 (deg_b=1,
  tip-to-tip), and the FALSE @ 2.20 sits INSIDE the next missed-real band, so
  simply raising `MICRO_GAP_UM` would accept a false join. But ENDPOINT DEGREE
  separates cleanly: the big missed-real cluster at gap 2.0–2.24 is TIP-TO-TIP
  (deg_b=1), the canonical "one neuron broken into two fragments" signature; the
  FALSE @ 2.20 is tip-INTO-shaft (deg_b=2); the two tip-to-tip FALSE are farther
  out at 2.45 and 3.00 µm. Lever (a genuinely NEW mechanism — endpoint degree, a
  feature no prior gen has touched): added `TIP_TIP_GAP_UM = 2.3` (0.15 µm below
  the nearest tip-to-tip false join at 2.45 µm) and a TIP-TO-TIP fast-accept tier in
  `propose_edits`, placed AFTER the MICRO fast-accept and BEFORE the NEAR/FAR
  `min_cos` logic: compute `deg_a = len(list(g.neighbors(s.node_a)))` and `deg_b =
  len(list(g.neighbors(s.node_b)))`, and if `s.gap_um <= TIP_TIP_GAP_UM AND
  deg_a == 1 AND deg_b == 1`, accept on distance+topology alone (waiving
  colinearity, same lattice-jitter rationale). This converts the dense gap-2.0–2.24
  tip-to-tip missed-real bucket into correct repairs (lifting train correct above
  163) while keeping train false == 0: the FALSE @ 2.20 is deg_b=2 (excluded → falls
  to FAR floor cos 0.44 < 0.60 → rejected); the FALSE @ 2.45 and @ 3.00 are
  tip-to-tip but at gap > 2.3 (out of band → still rejected). NOT raising
  `MICRO_GAP_UM` (would accept the FALSE @ 2.20) and NOT a colinearity re-tune
  (gen4 tied 91 doing that) — the lever is the new endpoint-degree gate. Image
  deliberately NOT used: warm-start bridge_ratio AUC is only 0.69 and the 3 FALSE
  joins (bridge_ratio 0.97–0.99) overlap REAL (0.94–1.02), so it cannot exclude
  them. AutoDiscovery support: finding #1 (id 30 — gap-distance AUC 0.9979,
  inter-neuron gaps RARELY below ~7 µm, so a sub-2.3 µm gap is overwhelmingly one
  broken neuron; GENERALIZES + OK), with finding #21 (id 63 — true split gaps 99%
  under 6.5 µm, tight characteristic scale; GENERALIZES + OK) for gap-scale context.
  The topology half (tip-to-tip = clean broken-neuron signature; joining into a
  shaft/branch is the error-prone topology) is grounded on the report's own
  SplitSite audit (the deg_b=1 cluster is the missed-real population; the only
  in-band false join is deg_b=2 tip-into-shaft) rather than on any non-generalizing
  KB finding.
- **Gen 7 → candidate (additive averaged cross-gap colinearity accept path):**
  Diagnosis was RECALL-bound, NOT precision: the parent scored 198 correct − 0
  train-false (= 198), but of 540 reachable REAL splits only 198 were accepted —
  342 MISSED, all 3 reachable FALSE joins correctly rejected (precision perfect).
  The FAR-regime gate `_is_colinear_split` is MISMATCHED with the harness's
  validated `colinear_cos` (candidate.py::_split_site_geom) in two ways: (a) it
  REQUIRES the tip-arm half-cosine `cos_a >= 0.60` as a SEPARATE hard floor, whereas
  the harness uses the AVERAGE `0.5*(cos_a + cos_b)`, so a real split whose tip arm
  bends slightly (`cos_a` ~0.4) but whose partner continues straight (`cos_b` ~0.9;
  avg ~0.65, clearly colinear) is REJECTED; (b) it walks 6.0 µm vs the harness's
  4.0 µm, over-smoothing the tangent. Evidence in the SplitSite audit: many MISSED
  reals have reported `colinear_cos` >= 0.55 yet were rejected (e.g. gap/cos
  2.13/0.73, 2.05/0.56, 2.22/0.55, 2.03/0.52), while the 3 FALSE joins have
  `colinear_cos` = 0.44/0.19/0.06 — well below — so the AVERAGED measure separates
  REAL from FALSE cleanly. Lever (a genuinely NEW mechanism — a faithful averaged
  measure, not a threshold re-tune): added `_avg_cross_gap_colinearity(g, s)` that
  reproduces the harness measure exactly (gdir = unit(xyz[node_b]-xyz[node_a]);
  cos_a from a 4.0 µm `_walk_tangent`; cos_b = best same-segment neighbor dot at
  node_b; return `0.5*(cos_a + cos_b)`), a constant `COLINEAR_AVG_ACCEPT = 0.55`
  (0.11 above the largest reachable false-join cos of 0.44), and a `HARNESS_TANGENT_
  WALK_UM = 4.0` constant. In `propose_edits` the FAR colinearity acceptance became
  an OR: accept if `_is_colinear_split(g, s, min_cos)` OR
  `(_avg_cross_gap_colinearity(g, s) >= COLINEAR_AVG_ACCEPT)`. Because the new clause
  is purely ADDITIVE (OR), no currently-accepted repair is lost (correct cannot drop
  below 198); it ADDS the missed reals whose averaged cross-gap colinearity >= 0.55,
  lifting train correct ABOVE 198; and the 3 FALSE joins (cos <= 0.44 < 0.55) stay
  rejected, so train false stays 0. Did NOT modify `_is_colinear_split`, MICRO/
  TIP-TO-TIP tiers, or any of MIN_COLINEAR_COS / NEAR_COLINEAR_COS / MICRO_GAP_UM /
  TIP_TIP_GAP_UM. AVOIDED: lowering the cos floor (gen2 already did that and gen4
  tied 91 re-tuning floors — the lever here is a DIFFERENT averaged measure, not a
  lower threshold) and a component-size cap (gen3, reverted). Image deliberately NOT
  used: the warm-start `bridge_ratio` AUC is only 0.69 and the FALSE joins
  (bridge_ratio 0.97–0.99) overlap the REAL range (0.94–1.02), so it cannot exclude
  them and would risk an auto-reverting false merge. AutoDiscovery support: primary
  finding #16 (id 33 — endpoint angular alignment is the validated discriminator,
  AUC 0.93: true continuation aligned vs false ~right-angle; the averaged two-half-
  cosine measure IS that discriminator computed faithfully, so gating on it is the
  correct implementation), with finding #1 (id 30 — gap-distance AUC 0.998, true
  split gaps tightly bounded) for band context; both GENERALIZE + UPHELD/OK.
- **Gen 8 → candidate (additive caliber-matched recall path):** Diagnosis was that
  the parent (248 correct − 0 train-false, 292 missed reals) OVER-MERGED on
  held-out: Edge Accuracy collapsed 72.97 → 66.17 (−6.8), %Merged Edges jumped
  24.34 → 31.56 (+7.2), and union-find transitive closure fused components of 20–33
  raw labels into brain-spanning mega-labels (e.g. N016 Edge Accuracy −14.78, N013
  many 24-label fusions). The remaining 292 missed reals now OVERLAP the 3 reachable
  FALSE joins in BOTH gap and colinear_cos, so those two features are EXHAUSTED as
  separators — no colinearity re-tune can lift recall without admitting a false
  join. The UNUSED clean separator is CABLE CALIBER: the two endpoint radii of a
  real "one neuron broken in two" are nearly equal (rad_ratio ≈ 1.0–1.3), whereas
  all 3 reachable FALSE joins fuse different calibers (rad_ratio 1.97/1.56/1.67, all
  ≥ 1.56). No prior generation has used caliber. Lever (a genuinely NEW feature —
  endpoint radius ratio): added `RAD_RATIO_MATCH = 1.30` and `RAD_MATCH_GAP_UM =
  2.5` constants, a `_rad_ratio(g, ctx, s)` helper (reads `ctx["node_radius"]`,
  falls back to `g.node_radius`, guards None/zero/negative, returns
  `max(ra,rb)/min(ra,rb)`), and a THIRD OR clause in the FAR colinearity tier:
  accept also when `rr is not None and rr <= RAD_RATIO_MATCH and avg_col is not None
  and avg_col >= 0.0 and s.gap_um <= RAD_MATCH_GAP_UM` (reusing the
  already-computed `avg_col`; the `>= 0.0` guard rejects anti-parallel doubling-back
  joins). The clause is PURELY ADDITIVE (OR), so it cannot drop any of the parent's
  248 accepts and adds the dense caliber-matched missed-real band just above 2.0 µm
  (lifting correct above 248); the 1.30 ceiling sits ~0.26 BELOW the smallest FALSE
  rad_ratio (1.56), so all 3 reachable false joins stay excluded with margin →
  zero new train-false merges. KEPT MICRO, TIP-TO-TIP, the two existing colinearity
  OR clauses, `_is_colinear_split`, and `_avg_cross_gap_colinearity` intact. Did NOT
  re-add a union-find component-size cap (a prior gen's MAX_MERGE_COMPONENT was
  rejected, held-out −2.000), did NOT re-tune the colinearity floor as the lever (a
  prior gen tied and was reverted), and did NOT raise MICRO_GAP_UM (would catch the
  FALSE @ 2.20 µm). Image deliberately NOT used: the warm-start bridge_ratio AUC is
  only 0.69 and the 3 FALSE joins (bridge_ratio 0.97–0.99) overlap REAL (0.94–1.02),
  so image cannot separate the reachable false joins. AutoDiscovery support: finding
  #1 (id 30 — gap-distance AUC 0.998, inter-neuron gaps rarely below ~7 µm, so a
  sub-2.5 µm gap is overwhelmingly one broken neuron; GENERALIZES + UPHELD/OK) plus
  the failure report's cable-caliber semantics (matched endpoint caliber = one
  broken neuron; mismatched = two fused neurites). DISTRUSTS the brain-specific
  absolute-radius / split-on-thin-process finding (#17, DOES-NOT-GENERALIZE) — this
  path uses only the RELATIVE rad_ratio of the two endpoints, never any absolute
  radius threshold.
- **Gen 9 → candidate (caliber-gated EXTENDED tip-to-tip gap-extension tier):**
  Diagnosis was RECALL-bound, NOT precision: the parent scored 256 correct − 0
  train-false (= 256), but of 540 reachable REAL splits only 256 were accepted —
  284 MISSED, all 3 reachable FALSE joins correctly rejected. The clean unused
  recall opportunity is a DENSE CLUSTER of MISSED reals at gap 2.45 that are
  TIP-TO-TIP (deg_a == 1 AND deg_b == 1): e.g. (cos, rad_ratio) =
  (−0.07, 1.18), (−0.00, 1.34), (−0.13, 1.41), (0.30, 1.34), (−0.03, 1.00),
  (0.52, 1.46), (−0.15, 1.41), (−0.35, 1.00), (−0.26, 1.46), (−0.48, 1.41). These
  sit JUST ABOVE the TIP_TIP_GAP_UM = 2.3 cutoff, so they fall through to the
  colinearity tier and are rejected (near-zero/negative `colinear_cos`, and many
  have `rad_ratio > 1.30` so the general caliber OR-clause misses them too). The
  ACCEPTED tip-to-tip reals at gap 1.41/1.73 already PROVE colinearity is pure
  lattice jitter for tip-to-tip topology — many were accepted with strongly
  NEGATIVE `colinear_cos` (−0.73, −0.76, −0.68, −0.54) — so the recovery tier must
  WAIVE colinearity, exactly as the existing TIP-TO-TIP tier does. Distance alone
  could NOT extend the prior tier: the only tip-to-tip FALSE inside the (2.3, 2.5]
  band is gap 2.45 / rad_ratio 1.56 / deg_b == 1 (the 3 reachable FALSE are
  constant across gens: gap 2.20 rad 1.97 deg_b==2; gap 2.45 rad 1.56 deg_b==1;
  gap 3.00 rad 1.67 deg_b==1). Lever (a NEW tier — a caliber-gated tip-to-tip gap
  extension): added `TIP_TIP_EXT_GAP_UM = 2.5` and `TIP_TIP_RAD_MATCH = 1.50`, and
  a tier placed immediately AFTER the existing TIP-TO-TIP tier and BEFORE the
  colinearity tier (reusing the already-computed `deg_a`/`deg_b` and the existing
  `_rad_ratio` helper): if `TIP_TIP_GAP_UM < s.gap_um <= TIP_TIP_EXT_GAP_UM AND
  deg_a == 1 AND deg_b == 1 AND rr_tt <= TIP_TIP_RAD_MATCH`, accept on
  distance + topology + matched caliber, WAIVING colinearity. The caliber ceiling
  1.50 sits 0.06 BELOW the in-band tip-to-tip FALSE rad_ratio 1.56 and 0.04 ABOVE
  the captured reals' 1.46 — and it can be LOOSER than the general path's 1.30
  precisely BECAUSE the `deg_a == deg_b == 1` requirement already removes the
  high-caliber `deg_b == 2` FALSE @2.20. So all 3 reachable FALSE stay excluded:
  @2.20 (deg_b == 2), @2.45 (rad 1.56 > 1.50), @3.00 (gap > 2.5) → zero new
  train-false merges. The tier is PURELY ADDITIVE (a site it accepts would
  otherwise fall to the colinearity tier), so train-correct cannot drop below the
  parent's 256, and it recovers ~10 tip-to-tip reals → score > 256. KEPT MICRO,
  the existing TIP-TO-TIP tier, the full colinearity tier (all three OR clauses),
  `_is_colinear_split`, `_avg_cross_gap_colinearity`, and `_rad_ratio` intact. Did
  NOT widen `RAD_RATIO_MATCH` (1.30) or `RAD_MATCH_GAP_UM` in the colinearity tier,
  did NOT re-add a union-find component-size cap (a prior gen's MAX_MERGE_COMPONENT
  was rejected, held-out −2.000), did NOT re-tune the colinearity floor as the
  lever (a prior gen tied 256 and was reverted), and did NOT raise MICRO_GAP_UM or
  the existing TIP_TIP_GAP_UM (raising TIP_TIP_GAP_UM ungated would admit the 2.45
  FALSE). Image deliberately NOT used: the warm-start probe AUC is only 0.69 and
  the 3 FALSE bridge_ratios (0.97–0.99) OVERLAP the REAL range (0.94–1.02), so
  image cannot separate the reachable false joins in this band. AutoDiscovery
  support: finding #1 (id 30 — gap-distance AUC 0.998, inter-neuron gaps rarely
  below ~7 µm, so a sub-2.5 µm gap is overwhelmingly one broken neuron;
  GENERALIZES + OK), paired with the failure report's own tip-to-tip topology
  evidence (two genuine termini at a tiny gap = one neuron broken in two) and
  cable-caliber semantics (matched endpoint caliber = one broken neuron; mismatched
  = two fused neurites). DISTRUSTS the brain-specific absolute-radius /
  split-on-thin-process finding (#17, DOES-NOT-GENERALIZE) — this tier uses only
  the RELATIVE rad_ratio of the two endpoints, never any absolute radius threshold.
- **Gen 10 → candidate (caliber-gated TIP-INTO-SHAFT tier):** Diagnosis was
  RECALL-bound, NOT precision: the parent scored 267 correct − 0 train-false
  (= 267), but of 540 reachable REAL splits only 267 were accepted — 273 MISSED, all
  3 reachable FALSE joins correctly rejected (precision perfect, no over-merges). The
  MISSED bucket is now DOMINATED by TIP-INTO-SHAFT sites: `deg_a == 1 AND deg_b == 2`
  at gap 2.0–2.65 (e.g. `2.01 / cos −0.01 / rad 1.43`, `2.22 / 0.16 / 1.47`,
  `2.25 / −0.18 / 1.43`, `2.35 / 0.24 / 1.31`, `2.35 / −0.05 / 1.09`,
  `2.39 / 0.12 / 1.33`, `2.41 / 0.08 / 1.39`, `2.47 / −0.02 / 1.49`). They fall
  through EVERY existing path: gap `> TIP_TIP_GAP_UM` (2.3) so both tip-to-tip tiers
  skip them on the `deg_b == 1` requirement; low/negative `colinear_cos` fails the
  FAR 0.60 floor and the averaged 0.55 floor; and many have `rad_ratio` just ABOVE
  the colinearity-tier caliber cap of 1.30, so that OR-clause misses them too. KEY
  INSIGHT: in this `deg_b == 2` band colinearity is actively MISLEADING — the ONLY
  reachable `deg_b == 2` FALSE join (gap 2.20) has `colinear_cos 0.44`, HIGHER than
  roughly half the missed reals, so no colinearity floor can recover the reals
  without admitting the false join (exactly why the 0.55/0.60 gates cannot reach
  these). But that FALSE has clearly MISMATCHED caliber `rad_ratio 1.97`, whereas the
  matched-caliber missed reals have `rad_ratio <= ~1.49`. So CALIBER beats
  colinearity here. The lattice-jitter argument that already justified WAIVING
  colinearity in the MICRO and TIP-TO-TIP tiers is topology-independent (tip-arm
  jitter corrupts the tip direction regardless of what `node_b` is), so it applies to
  tip-into-shaft too. Lever (a NEW caliber-gated tier): added
  `TIP_SHAFT_RAD_MATCH = 1.50` (reusing `RAD_MATCH_GAP_UM` 2.5 as the gap ceiling),
  and a tier placed immediately AFTER the EXTENDED TIP-TO-TIP tier and BEFORE the
  colinearity tier (reusing the already-computed `deg_a`/`deg_b` and the existing
  `_rad_ratio` helper): if `MICRO_GAP_UM < s.gap_um <= RAD_MATCH_GAP_UM AND
  deg_a == 1 AND deg_b == 2 AND rr_ts <= TIP_SHAFT_RAD_MATCH`, accept on
  distance + topology + matched caliber, WAIVING colinearity. The 1.50 ceiling sits
  0.47 BELOW the only reachable `deg_b == 2` FALSE's `rad_ratio` 1.97 — a WIDE margin
  — while admitting the matched-caliber missed reals at `rad_ratio <= ~1.49`. The
  other two reachable FALSE joins are `deg_b == 1` (so this `deg_b == 2` tier never
  sees them) → zero new train-false merges. Restricted to `deg_b == 2` ONLY (shaft):
  `deg_b >= 3` branch points are rarer/riskier and the missed cluster is all
  `deg_b == 2`. The tier is PURELY ADDITIVE — a site it accepts currently fails all
  existing paths (gap > 2.3, deg_b == 2, low/negative cos, rad_ratio > 1.30) — so
  train-correct cannot drop below the parent's 267; it recovers ~8–12 matched-caliber
  tip-into-shaft reals → score > 267. KEPT MICRO, both tip-to-tip tiers, the full
  colinearity tier (all three OR clauses), `_is_colinear_split`,
  `_avg_cross_gap_colinearity`, and `_rad_ratio` intact. Did NOT widen
  `RAD_RATIO_MATCH` (1.30) or `RAD_MATCH_GAP_UM` in the colinearity tier, did NOT
  widen the tip-to-tip constants, did NOT raise `MICRO_GAP_UM` / `TIP_TIP_GAP_UM` /
  `TIP_TIP_EXT_GAP_UM`, did NOT re-add a union-find component-size cap (a prior gen's
  MAX_MERGE_COMPONENT was rejected, held-out −2.000), and did NOT re-tune the
  colinearity floor as the lever (a prior gen tied and was reverted). Image
  deliberately NOT used: the warm-start probe AUC is only 0.69 and the 3 reachable
  FALSE bridge_ratios (0.97–0.99) OVERLAP the REAL range (0.94–1.02), so image cannot
  separate the reachable false joins in this band. AutoDiscovery support: finding #1
  (id 30 — gap-distance AUC 0.998, inter-neuron gaps rarely below ~7 µm, so a
  sub-2.5 µm gap is overwhelmingly one broken neuron; GENERALIZES + OK), paired with
  the failure report's own cable-caliber semantics (matched endpoint caliber = one
  neuron broken; mismatched = two fused neurites) and the observation that the
  `deg_b == 2` FALSE @2.20 has misleadingly HIGH `colinear_cos` 0.44 but mismatched
  caliber 1.97. DISTRUSTS the brain-specific absolute-radius / split-on-thin-process
  finding (#17, DOES-NOT-GENERALIZE) — this tier uses only the RELATIVE rad_ratio of
  the two endpoints, never any absolute radius threshold.
- **Gen 11 → candidate (extend the TIP-INTO-SHAFT gap ceiling 2.5 → 2.75, decoupled
  from RAD_MATCH_GAP_UM):** Diagnosis was RECALL-bound, NOT precision: the parent
  scored 275 correct − 0 train-false (= 275), but of 540 reachable REAL splits only
  275 were accepted — 265 MISSED, all 3 reachable FALSE joins correctly rejected
  (precision perfect, no over-merges). The MISSED bucket splits into two groups.
  GROUP (1), the recoverable one: CALIBER-MATCHED tip-into-shaft sites (`deg_a == 1,
  deg_b == 2, rad_ratio <= 1.50`) at gaps JUST ABOVE the tier's 2.5 ceiling — e.g.
  `2.51 / rad 1.19`, `2.55 / 1.44`, `2.56 / 1.47`, `2.60 / 1.13`, `2.64 / 1.17`,
  `2.67 / 1.11`, `2.71 / 1.13`, `2.72 / 1.33`, `2.72 / 1.25`. These are EXACTLY what
  the gen10 TIP-INTO-SHAFT tier already accepts, except they fall just outside its
  gap ceiling (`RAD_MATCH_GAP_UM` = 2.5, which the tier was sharing with the
  colinearity-tier caliber clause). GROUP (2), the un-recoverable one: caliber-
  MISMATCHED high-rad `deg_b == 2` reals (rad_ratio ~1.9–2.7, thin tip onto thick
  shaft) that OVERLAP the only reachable `deg_b == 2` FALSE @2.20 (rad 1.97,
  colinear_cos 0.44) in BOTH caliber and colinearity, and the image probe cannot
  separate them either (the FALSE bridge_ratio 0.97 lies inside the REAL range
  0.94–1.02, probe AUC only 0.69) — so chasing group (2) cannot be done without
  admitting the false join. The lever therefore targets ONLY group (1). SAFETY (the
  FALSE-free-gap-window argument): the only reachable `deg_b == 2` FALSE is @2.20
  and it is excluded by the caliber ceiling (rad 1.97 > 1.50); the next reachable
  FALSE is @3.00 but it is `deg_b == 1`, so a `deg_b == 2` tier NEVER sees it. Hence
  caliber-matched tip-into-shaft is FALSE-free across the whole gap window from ~2.2
  up to 3.0, and the tier's gap ceiling can be pushed past 2.5 without admitting any
  reachable false join. Change (decouple, then extend): added a DEDICATED constant
  `TIP_SHAFT_GAP_UM = 2.75` and, in the TIP-INTO-SHAFT tier ONLY, changed the gap
  upper bound from `RAD_MATCH_GAP_UM` to `TIP_SHAFT_GAP_UM`. This is PURELY ADDITIVE
  — it only WIDENS the gap band from 2.5 to 2.75, so every site the tier already
  accepted still is, and it newly admits ~9+ caliber-matched tip-into-shaft reals in
  (2.5, 2.75]; train-correct cannot drop below the parent's 275, it can only rise.
  No reachable FALSE is in this `deg_b == 2` band (the 2.20 FALSE is rad 1.97 > 1.50;
  the 2.45 and 3.00 FALSE are `deg_b == 1`) → zero new train-false merges → score >
  275. Held conservatively below 3.0 (the `deg_b == 1` FALSE sits at 3.00 and the
  held-out population near it is unseen). KEPT everything else byte-for-byte: did NOT
  touch `RAD_MATCH_GAP_UM` (2.5) in the colinearity-tier caliber clause, did NOT
  touch the tip-to-tip tiers/constants, `TIP_SHAFT_RAD_MATCH` (1.50), the helpers,
  MICRO, or the colinearity tier. Did NOT extend to/past 3.0, did NOT widen
  `TIP_SHAFT_RAD_MATCH` (group (2) high-rad reals are unseparable from the FALSE
  @2.20 — raising the caliber ceiling would admit it), did NOT re-add a union-find
  component-size cap (a prior gen's MAX_MERGE_COMPONENT was rejected, held-out
  −2.000), and did NOT re-tune the colinearity floor (a prior gen tied and was
  reverted). Image deliberately NOT used: the warm-start probe AUC is only 0.69 and
  the 3 reachable FALSE bridge_ratios (0.97–0.99) OVERLAP the REAL range (0.94–1.02),
  so image cannot separate the reachable false joins in this band. AutoDiscovery
  support: finding #1 (id 30 — gap-distance AUC 0.9979, inter-neuron gaps rarely
  below ~7 µm, so a sub-2.75 µm gap is overwhelmingly one broken neuron; GENERALIZES
  + OK) and the tight-split-gap finding (id 63 — true split gaps 99% under ~6.5 µm,
  GENERALIZES + OK), paired with the failure report's own cable-caliber semantics
  (matched endpoint caliber = one broken neuron; mismatched = two fused neurites).
  DISTRUSTS the brain-specific absolute-radius / split-on-thin-process finding (#17,
  DOES-NOT-GENERALIZE) — this tier uses only the RELATIVE rad_ratio of the two
  endpoints, never any absolute radius threshold.
