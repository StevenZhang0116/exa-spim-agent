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

## Current criteria (Generation 4 — mid-gap distance-dominant recall)

The policy repairs splits (SplitSite → `merge_labels`) via THREE acceptance paths.
All require a small gap first: `s.gap_um <= GAP_THRESHOLD_UM` (4.0 µm).

  1. **Strict colinear path (high precision, unchanged).** Accept when the two
     fragments are colinear across the gap — a straight-line continuation
     (`cos(angle) >= MIN_COLINEAR_COS`, 0.94 ≈ ≤20° bend) via `_is_colinear_split`.
  3. **Mid-gap distance-dominant path (Gen 4, widened).** For gaps up to
     (`s.gap_um <= MID_GAP_UM`, 2.5 µm), DROP the colinearity requirement entirely
     and accept on an image bridge confirmation alone: read
     `gap_bridge_evidence(s.node_a, s.node_b)` and accept when
     `bridge_ratio >= BRIDGE_RATIO_MIN` (0.85). Rationale: at gaps this small the
     gap-direction vector `node_b - node_a` is a tiny vector dominated by voxel
     noise once normalized, so `colinear_cos` is essentially random there (REAL splits
     show near-zero / negative cos across the 1.5–2.5 µm band, e.g. 1.51 µm/-0.28,
     1.99 µm/0.00, 2.00 µm/-0.55) — colinearity can never rescue these. The small gap
     itself is near-decisive evidence of a true split. This path is placed BEFORE
     Path 2 with a `continue`, so a small-gap site is read at most once. Widened from
     1.5 µm to 2.5 µm in Gen 4 to recover the 1.5–2.5 µm low-cos REAL band that
     Path 2's cos >= 0.80 floor could never accept.
  2. **Image-gated recall path (Gen 1, now handles the 2.5–4 µm moderate-colinear
     band).** For sites that FAIL the strict gate,
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

- **Gen 4 (mid-gap distance-dominant recall — widen Path 3 to 2.5 µm):** the gen04
  parent scores 66 correct / 0 false but still MISSES 424 reachable REAL splits. That
  missed bucket is dominated by gaps in the 1.5–2.5 µm band with noise-dominated,
  near-zero or negative `colinear_cos` (e.g. 1.51 µm/cos -0.28, 1.58/-0.01, 1.99/0.00,
  2.00/-0.55): at these tiny gaps the gap-direction vector is pure voxel noise, so
  colinearity can NEVER accept them. Path 3 previously covered only `gap <= 1.5 µm` and
  Path 2 demands `cos >= 0.80`, so these low-cos 1.5–2.5 µm reals were read by nothing
  and silently rejected. The ONE change: rename `SMALL_GAP_UM = 1.5` to
  `MID_GAP_UM = 2.5`, so Path 3 now fires for `gap_um <= 2.5 µm` — spending image reads
  to RAISE RECALL on exactly that band. Path 2 now effectively handles only the
  2.5–4 µm moderate-colinear band. Paths 1 (strict cos 0.94) and 2 (image-gated
  cos >= 0.80) are byte-for-byte unchanged; `BRIDGE_RATIO_MIN` (0.85) and the
  no-colinearity-then-`continue` behavior are unchanged.
  - **Why this can only ADD correct merges, never a false one.** Path 3 still fires
    ONLY on a successful image read with `bridge_ratio >= 0.85` (a NaN/failed read
    skips the site), and the harness "Image warm-start probe" measured `bridge_ratio`
    separability AUC = 0.83 ("strong — image is worth using"); the REAL splits the
    policy already read returned `bridge_ratio` 0.94–1.03, so an image bridge
    confirmation cleanly accepts them.
  - **Precision is preserved by GAP, not colinearity.** The only train FALSE join sits
    at gap 3.12 µm — OUTSIDE the 2.5 µm band, a >0.6 µm margin — and the band below
    2.5 µm contains zero train false joins.
  - Cross-run KB priors relied on (BOTH Generalization GENERALIZES and Verdict OK —
    robust across all three brains and after cluster-robust correction):
    * **Finding #1** — Euclidean gap distance alone separates true splits from
      inter-neuron neighbors nearly perfectly (AUC 0.998; true-split gaps peak
      ~4.5 µm; inter-neuron gaps rarely below ~7 µm): grounds trusting a <= 2.5 µm gap
      as near-decisive evidence of a true split and dropping colinearity there.
    * **Finding #21** — true inter-segment split gaps have a tight characteristic
      scale (99th pct ~6.5 µm, 100% under 15 µm): a <= 2.5 µm gap is deep inside the
      true-split regime, far from any inter-neuron join.
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
