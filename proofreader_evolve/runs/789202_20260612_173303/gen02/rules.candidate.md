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

## Current criteria (Generation 3 — conservative merge-repair via `split_label`)

The dominant error on train is **%Merged Edges (~9.09)**, dwarfing **%Split
Edges (~0.17)**. A prior split-repair (`merge_labels`) attempt bought nothing
(+0.000), so this generation attacks the merge term instead.

- **SplitSites (`merge_labels`)** — intentionally **inert** this generation.
  There is no headroom on the split lever here, so we propose no merges.
- **MergeSites (`split_label`)** — emit a split ONLY on strong, multi-signal
  evidence so the strict over-split watchdog (must not raise %Split / %Merged /
  #Merges) is satisfied:
  1. **Both arms long** — `cable_a_um >= 12` AND `cable_b_um >= 12` (a real
     two-neuron fusion, not a short spur off one neuron). Required for all
     detectors.
  2. **Per-detector geometry:**
     - `component` — disconnected pieces under one label + both arms long IS the
       signal; `angle_deg` is NaN here, so it is NOT thresholded. Accepted on the
       size gate alone (the safest split topology).
     - `branch` / `bridge` — require a **sharp junction** (`angle_deg <= 130°`,
       NaN-guarded, i.e. not a straight pass-through) AND at least one
       corroborating signal: a **caliber mismatch** (`radius_ratio >= 1.5`) OR a
       **high-degree crossing** (`branch_degree >= 4`).
  3. **Image confirmation (when available).** For candidates that already pass
     geometry, call `merge_cut_evidence(seed_a_node, seed_b_node)` and require an
     intensity **valley** (`valley_ratio <= 0.7`) near the chord centre
     (`0.25 <= valley_pos <= 0.75`) — two structures merely touching. If the image
     says continuous bright signal, the candidate is SKIPPED (one neuron). When
     the image reader is absent, geometry alone decides. Image is read ONLY for
     geometry-passing candidates to respect the cloud-fetch cost rule.

All graph/feature/image access is wrapped in `try/except` so the policy degrades
to "propose nothing" rather than crashing if a field or reader is missing.

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
- **Gen 3:** switched lever from split-repair to **merge-repair**. The gen02
  report shows %Merged Edges ~9.09 dominating %Split Edges ~0.17, and the prior
  `merge_labels` attempt netted +0.000 — so the split lever has no headroom here.
  New policy emits conservative `split_label` edits gated on (a) both arms
  >=12 µm, (b) per-detector geometry (`component` accepted on size; `branch`/
  `bridge` need `angle_deg <= 130°` plus `radius_ratio >= 1.5` or
  `branch_degree >= 4`), and (c) when the image is present, an intensity valley
  (`valley_ratio <= 0.7`, central `valley_pos`) confirming a real touch. Chosen
  because it is a genuinely different lever (different edit type + dominant error
  term) than the rejected gen1 merge_labels gap+tangent approach.
