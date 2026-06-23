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
  1. **NEAR regime** — `s.gap_um <= NEAR_GAP_UM` (1.5 µm): distance alone is
     decisive, so the colinearity floor is `NEAR_COLINEAR_COS` (0.0). Accept every
     join except those that double back (anti-parallel / obtuse, `cos < 0`).
  2. **FAR regime** — `1.5 µm < s.gap_um <= 6.5 µm`: the FAR colinearity floor
     `MIN_COLINEAR_COS` (0.60) now gates the **TIP-ENDPOINT alignment only**
     (`cos(v1, v2)`, the across-the-gap direction agreement). The **partner
     continuation** (`cos(v2, v3)`, node_b's onward direction) is required only to
     be **non-reversing** (`>= MIN_CONTINUATION_COS = 0.0`) — DECOUPLED from the
     endpoint floor. This recovers clean mid-gap continuations whose partner
     continuation is moderate while staying above the false-join population (all
     endpoint cos <= 0.44).
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
- **Gen 4 -> candidate (decouple the partner-continuation floor):** Rollback context:
  the Gen 3 `MAX_MERGE_COMPONENT = 3` union-find cap was REJECTED (held-out -2.000)
  because capping removed recall - on this brain held-out net-REWARDS recall even with
  some mega-merges, and the gate keeps a candidate only if it makes MORE net-correct
  train repairs than the parent's 91, so a pure precision/guard change cannot clear the
  bar; the lever must RAISE recall. Diagnosis: `_is_colinear_split` was rejecting REAL
  splits that have strong tip-endpoint alignment (high reported colinear_cos, e.g.
  missed (gap 1.94, cos 0.76) vs accepted (gap 1.65, cos 0.78) - near-identical
  endpoint alignment, opposite decision) purely on the SECONDARY partner-continuation
  constraint `best >= min_cos`, not on the discriminator that separates REAL from
  FALSE. Fix: DECOUPLE the two floors - the tip-endpoint alignment `cos(v1, v2)` stays
  strict at the gap-graded floor (NEAR 0.0 / FAR 0.60), while the partner continuation
  `cos(v2, v3)` is relaxed to merely non-reversing (`MIN_CONTINUATION_COS = 0.0`),
  which still vetoes right-angle / doubling-back tangle crossings (a mild anti-tangle
  guard) but stops rejecting real splits whose partner continuation is moderate. Why
  grounded: finding #16 (id 33 - endpoint directional alignment ACROSS the gap is the
  validated discriminator, true ~153 deg vs false ~90 deg, AUC 0.93; the downstream
  partner continuation is an unvalidated extra constraint), so relaxing it recovers
  recall without touching the real discriminator; the 3 reachable FALSE joins (endpoint
  cos <= 0.44) stay rejected by the 0.60 endpoint floor. This is a NEW lever - testing
  STRUCTURE (decoupling two alignment constraints) - distinct from the floor-value
  tuning of Gens 1-2 and the rejected component cap of Gen 3. Image still NOT used:
  warm-start bridge_ratio AUC is only 0.69 and the 3 FALSE joins have HIGH bridge_ratio
  (0.97-0.99) overlapping REAL, so gating on it cannot exclude them. AutoDiscovery
  support: finding #16 (id 33) - GENERALIZES + OK.
