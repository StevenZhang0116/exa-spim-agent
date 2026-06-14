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

## Current criteria (split-repair via image-confirmed gap connectivity)

The policy repairs **split errors only** (`merge_labels`); it does **not** emit any
`split_label` this generation. Rationale from the train failure report: although
`%Merged Edges` (9.09) dwarfs `%Split Edges` (0.17), the report states *no raw label
spans ≥2 GT neurons on this split* — so there is no GT-confirmable merge target, and
any `split_label` here cuts a real neuron and trips the over-split watchdog. The
merge lever is therefore off; the safe, gate-passable win is split repair.

A SplitSite is unified (`merge_labels`) only when MULTIPLE independent signals agree
that the two fragments are one neuron:

1. **Geometry — continuation, not crossing.** The two endpoints' local outward
   tangents (from `rooted_subgraph`) must *oppose* near 180° (deviation
   ≤ `TANGENT_OPP_ANGLE` = 50°). This rejects two unrelated neurites that merely
   pass close.
2. **Caliber continuity.** The two endpoints' `node_radius` must be of similar size
   (ratio ≤ `RADIUS_RATIO_MAX` = 2.2). A thick-vs-thin pairing is likely two
   different cables and is rejected.
3. **Image gap connectivity (the NEW evidence source).** When
   `ctx["read_image_patch"]` is available, `gap_connectivity(node_a, node_b)` must
   show the gap is **bright on both sides** (dim side ≥ `IMG_DIM_FRAC` = 0.45 of the
   bright side). A short, well-aligned, similar-caliber gap (≤ `GAP_NEAR_UM` = 3 µm)
   may merge on geometry+caliber alone but is VETOED if the image says the gap is
   dark. A longer gap (3–6 µm, up to `GAP_FAR_UM`) REQUIRES positive image
   confirmation; with no image at all it merges only when alignment is very strong
   (within 0.6× the angle tolerance). Image reads happen only after the cheap
   geometric/caliber gates pass, to respect the per-read cloud-fetch cost.

All graph/radius/image access is wrapped defensively; missing signals degrade
gracefully (e.g. no image → strict geometry fallback).

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

- **Gen 0:** seed is a no-op (proposes nothing).
- **2026-06-13 (this gen):** Activate SPLIT repair (`merge_labels`) gated on a
  THREE-signal AND: tangent continuation (oppose within 50° of 180°), caliber
  continuity (radius ratio ≤ 2.2), and — the new lever — **fluorescence gap
  connectivity** (gap bright on both sides) for any gap beyond 3 µm. Distinct from
  the rejected gen1 (geometry-only merge_labels, +0.000) because it adds an image
  evidence source and a radius-continuity gate, and loosens the gap ceiling to 6 µm
  with image backing so it actually fires net-positive edits. Distinct from the
  rejected gen2 (split_label, +0.492 but reverted) because it does NOT touch the
  merge lever at all: the train report shows no GT-confirmable merge target on the
  acceptance split, so any `split_label` over-splits a real neuron and trips the
  watchdog — this generation avoids that failure mode entirely by attacking only the
  split component, which has no over-split risk.
- **Harness:** `candidate_split_sites` broadened from tip-to-tip to
  tip-to-any-node (tip/shaft/branch) within `max_gap_um`; `SplitSite` carries
  `node_a` (always a tip) and `node_b` (the partner, any degree).
- **Harness (merge repair):** added `MergeSite` + `candidate_merge_sites`
  (GT-free branch-based merge detection) and a unified split+merge candidate
  stream; fitness is Edge Accuracy (charges merge errors); the failure report now
  lists baseline merge targets and an over-split watchdog.
