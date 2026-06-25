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

The split-repair policy now has TWO accept paths for SplitSites; MergeSites are
left alone (no `split_label`).

  1. **Strict geometry-only accept** — emit `merge_labels` when BOTH the gap is
     small (`s.gap_um <= GAP_THRESHOLD_UM`, 4.0 µm) AND the two fragments are
     colinear across the gap (`cos(angle) >= MIN_COLINEAR_COS`, 0.94 ≈ ≤20° bend).
     This is the original high-precision path (6 correct / 0 false on train).
  2. **Image-gated recall path** — for SplitSites with `gap_um <= 4.0` that FAIL
     the strict 0.94 test but fall in the relaxed near-colinear band
     `[IMAGE_MIN_COLINEAR_COS, MIN_COLINEAR_COS)` = `[0.75, 0.94)` at gap
     `>= IMAGE_GAP_MIN_UM` (1.0 µm), read the fluorescence bridge across the gap
     (`gap_bridge_evidence(node_a, node_b)`) and accept ONLY when
     `bridge_ratio >= IMAGE_BRIDGE_RATIO_MIN` (0.90) — signal stays ≥90% bright
     all the way across, i.e. one continuous neuron.

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
