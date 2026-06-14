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

## Current criteria (Generation 1 — conservative split repair)

The policy now repairs **split errors** only, and does so conservatively:

- **SplitSite → `merge_labels`** is accepted only when BOTH hold:
  1. the tip gap is small: `gap_um <= GAP_THRESHOLD_UM` (4.0 µm); AND
  2. the two fragments are near-**collinear continuations**: each tip's local
     tangent (mean outgoing direction within `TANGENT_RADIUS_UM` = 12 µm, via
     `g.rooted_subgraph`) is estimated, and the angle between their *opposing*
     directions is `<= MAX_TANGENT_ANGLE_DEG` (35°). This rejects two unrelated
     neurites that merely pass nearby at a crossing (their tangents would not
     oppose), guarding against over-merging.
  If the fragment graph is unavailable, we fall back to a gap-only rule at half
  the threshold (`gap <= 2.0 µm`).
- **MergeSite → `split_label`**: NOT emitted this generation. The failure report
  shows no raw label spans >=2 GT neurons, so there are no safe merge-correction
  targets, and a speculative split would only risk the over-split gate. MergeSites
  are skipped (effectively flagged for review).

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

`ctx["n_split_sites"]` / `ctx["n_merge_sites"]` report the stream composition.

## Change log

- **Gen 1:** moved off the no-op floor. Diagnosis from the gen-0 report: the
  policy proposed 0 edits, so Edge Accuracy held exactly at baseline (85.43) with
  zero progress; %Merged Edges (9.09) is the largest error component but the
  report lists NO raw label spanning >=2 GT neurons, so `split_label` has no safe
  target. Therefore the high-value, low-risk move is the workhorse `merge_labels`
  on SplitSites. Added a conservative split-repair rule: fuse a SplitSite's label
  pair only when the tip gap is small (<=4 µm) AND the two tips' local tangents are
  near-collinear continuations (opposition angle <=35°), to avoid over-merging at
  crossings. No `split_label` emitted (no safe merge targets this split).
- **Gen 0:** seed is a no-op (proposes nothing).
- **Harness:** `candidate_split_sites` broadened from tip-to-tip to
  tip-to-any-node (tip/shaft/branch) within `max_gap_um`; `SplitSite` carries
  `node_a` (always a tip) and `node_b` (the partner, any degree).
- **Harness (merge repair):** added `MergeSite` + `candidate_merge_sites`
  (GT-free branch-based merge detection) and a unified split+merge candidate
  stream; fitness is Edge Accuracy (charges merge errors); the failure report now
  lists baseline merge targets and an over-split watchdog.
