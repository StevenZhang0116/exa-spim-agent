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

## Current criteria (Generation 1 — colinear merge with crossing/radius guards)

Split-repair only (no `split_label` yet). For each `SplitSite`, emit a
`merge_labels` edit when ALL of:

1. **Gap is small** — `s.gap_um <= GAP_THRESHOLD_UM` (4.0 µm).
2. **Colinear continuation** — the imagined repaired cable
   `A_interior -> node_a -> node_b -> B_interior` is near-straight at both joints
   (`_is_colinear_split`). The colinearity bar is **adaptive to the partner
   endpoint type**:
   - `node_b` is a **tip** (degree ≤ 1): require `cos >= MIN_COLINEAR_COS` (0.94).
   - `node_b` is a **shaft/branch** (degree ≥ 2): require the tighter
     `cos >= MIN_COLINEAR_COS_BRANCH` (0.985). Reconnecting into the *middle* of
     another cable is the topology that most often fuses two distinct neurons at a
     crossing, so it must be almost perfectly straight to be accepted.
3. **Radius continuity** — the two endpoints' neurite radii agree within
   `MAX_RADIUS_RATIO` (2.5×) when `ctx["node_radius"]` is available; a sharp
   caliber jump across the gap signals two unrelated cables. Advisory: if radii are
   missing the check passes (never *more* restrictive for lack of signal).

MergeSites are still left alone (`split_label` deferred until a clearer over-merge
gradient exists).

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
- **Harness:** `candidate_split_sites` broadened from tip-to-tip to
  tip-to-any-node (tip/shaft/branch) within `max_gap_um`; `SplitSite` carries
  `node_a` (always a tip) and `node_b` (the partner, any degree).
- **Harness (merge repair):** added `MergeSite` + `candidate_merge_sites`
  (GT-free branch-based merge detection) and a unified split+merge candidate
  stream; fitness is Edge Accuracy (charges merge errors); the failure report now
  lists baseline merge targets and an over-split watchdog.
- **Gen 1 (2026-06-16):** first real policy = colinear split-repair. Diagnosis of
  the prior colinear-merge candidate: overall Edge Accuracy rose (79.98→80.68) but
  almost ALL the gain came from lower %Omit Edges (relabeling background-adjacent
  nodes), while the merge edits drove the over-split watchdog UP nearly 3× (%Split
  Edges 0.117→0.304, +hundreds of split edges per skeleton) and CREATED 2 merges —
  both joining into NON-tip (shaft/branch) partner nodes, i.e. fusing distinct
  neurons at crossings. Change: (a) make the colinearity threshold adaptive —
  tip-to-tip joins keep the 0.94 bar, but tip-to-shaft/branch joins now require a
  much tighter 0.985 (near-perfectly-straight) continuation; (b) add a
  radius-continuity guard (`MAX_RADIUS_RATIO=2.5`) that rejects merges across a
  sharp caliber discontinuity, using the free `ctx["node_radius"]` signal. Goal:
  cut the over-split / over-merge created at crossings without giving back the
  legitimate colinear tip-to-tip repairs.
