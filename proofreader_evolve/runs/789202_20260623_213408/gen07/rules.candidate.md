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

## Current criteria (Generation 7 — spatial-clustering recall gate)

The policy repairs splits (SplitSite → `merge_labels`) via FOUR acceptance paths.
All require a small gap first: `s.gap_um <= GAP_THRESHOLD_UM` (4.0 µm).

  1. **Strict colinear path (high precision, unchanged).** Accept when the two
     fragments are colinear across the gap — a straight-line continuation
     (`cos(angle) >= MIN_COLINEAR_COS`, 0.94 ≈ ≤20° bend) via `_is_colinear_split`.
  4. **Spatial-clustering gate (Gen 7, new, geometry-only, no image).** Accept when
     BOTH endpoints are TIPS (each has exactly ONE same-segment neighbor — tip-to-tip;
     joining into a shaft/branch is riskier) AND this site has `>= CLUSTER_MIN_NEIGHBORS`
     (2) OTHER candidate split sites within `CLUSTER_RADIUS_UM` (30 µm) of `node_a`.
     Neighbor positions are precomputed once from the candidate stream itself (every
     split-kind site's `node_a` world position), so this is a SITE-INDEPENDENT signal:
     real splits CASCADE (a genuine miss sits in a neighborhood dense with other split
     candidates), whereas a cross-neuron FALSE join is an isolated coincidental
     proximity. Grounded in KB Finding #15 (split spatial clustering / cascade,
     GENERALIZES, UPHELD). Image-free — uses neither the raw image nor the noisy gap/arm
     direction vector. Placed after Path 1's `continue` and before the image paths.
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

- **Gen 7 (spatial-clustering recall gate):** DIAGNOSIS — the policy accepts 66 real
  splits but MISSES 424 reachable real splits, all in the gap_um ≈ 1.5–2.5 µm band
  with low/near-zero `colinear_cos`. Root cause: the `node_a→node_b` gap vector is a
  ~2 µm vector dominated by voxel noise once normalized, so every per-site
  directional/colinear feature is unreliable there. CRITICAL HISTORY — THREE prior
  single-signal recall expansions were each REVERTED by the gate for introducing
  HELD-OUT FALSE merges: (gen4) widened the gap + accepted on image `bridge_ratio`
  alone; (gen5) tip-to-tip + arm-dot ≤ −0.70 + image bridge; (gen6) arm anti-parallel
  (the two tip-arm tangents) dot ≤ −0.85, no image. The lesson: NO single per-site
  geometric or image threshold separates real splits from held-out false joins in
  this band. CHOSEN LEVER — a SITE-INDEPENDENT spatial-clustering gate grounded in
  validated KB Finding #15 (split errors are spatially clustered / cascade: a true
  split edge has ~10× more split-candidate neighbors within 30 µm — mean 0.97 — than
  a false candidate — mean 0.10, i.e. false joins are ISOLATED; GENERALIZES across all
  3 brains, ratio 5.0×–11.5×; Post-correction Verdict UPHELD). Real splits cascade, so
  a genuine missed split sits in a neighborhood dense with OTHER candidate split sites,
  while a cross-neuron false join is an isolated coincidental proximity. The site's
  clustering is counted from the candidate stream itself (precomputed `node_a` world
  positions; `n_near = count(dist <= 30 µm) - 1` excluding self) — no image, no noisy
  gap vector. It is CONJOINED with a tip-to-tip requirement (the report states joining
  into a shaft/branch is riskier than tip-to-tip, and the lone train FALSE join is a
  deg_b==2 shaft join), giving a conjunction of two independent, generalizable
  precision signals so the intersection stays zero-false where the three single-signal
  attempts failed. WHY DIFFERENT / more precision-safe than gen4/5/6: it uses NEITHER
  the image NOR the noisy gap/arm direction — it is a topological prior derived from
  the candidate stream itself, and it conjoins two independent generalizable signals
  rather than betting on one per-site threshold. The gap early-skip (`<= 4.0 µm`) is
  unchanged; the clustering signal is gap-independent and applies across the whole band.
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
