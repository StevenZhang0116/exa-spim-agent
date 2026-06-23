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
  3. **NEAR regime** — `MICRO_GAP_UM < s.gap_um <= NEAR_GAP_UM`: colinearity floor
     `NEAR_COLINEAR_COS` (0.0) — accept everything except joins that double back
     (anti-parallel / obtuse, `cos < 0`). (Currently EMPTY because `MICRO_GAP_UM`
     (2.0) > `NEAR_GAP_UM` (1.5); kept for clarity / future re-tuning.)
  4. **FAR regime** — `MICRO_GAP_UM < s.gap_um <= 6.5 µm` (i.e. `> 2.0 µm`), for
     sites NOT accepted by the tip-to-tip tier: require
     colinearity, `cos(angle) >= MIN_COLINEAR_COS` (0.60). This recovers clean
     mid-gap continuations while staying above the false-join population (the 3
     reachable false joins, at gap 2.20/2.45/3.00 µm, all have cos <= 0.44).
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
