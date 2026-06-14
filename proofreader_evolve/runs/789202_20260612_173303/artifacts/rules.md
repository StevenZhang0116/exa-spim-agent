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

## Current criteria (Generation 7 — strict global edit budget)

Only the `merge_labels` lever is used (`split_label` is disabled: no raw label in
this split spans ≥2 GT neurons, so a split only trips the over-split watchdog).

For each `SplitSite` we compute a single **composite confidence** in [0,1] from a
few cheap, robust signals, then commit ONLY the top-ranked edit(s):

1. **Gap closeness** (`c_gap`): `1 − gap_um / 4.0`; closer breaks score higher.
   Sites with `gap_um > 4.0 µm`, NaN, or negative gaps are discarded outright.
2. **Terminal cleanliness** (`c_term`): `node_a` is always a tip; reward `node_b`
   also being a true degree-1 tip (`1.0`), penalize tip-to-shaft (degree 2 → `0.4`)
   and reject-weight tip-to-branch (degree ≥3 → `0.0`, the most over-merge-prone).
   Unknown degree → neutral `0.5`.
3. **Tangent agreement** (`c_dir`): when both ends are tips, antiparallel outgoing
   tangents (the two fragments continuing the same line) score `1.0`; crossings
   score low. Neutral `0.5` when not both tips.

`confidence = 0.30·c_gap + 0.40·c_term + 0.30·c_dir`. Candidates below
`GEN7_MIN_CONFIDENCE = 0.55` are dropped. The survivors are ranked by confidence
and only the **top `GEN7_EDIT_BUDGET = 1`** is emitted (one reattachment per label,
to avoid chaining mega-labels). All graph/feature access is wrapped in
try/except with None/NaN guards, but gating is on robustly-available fields
(`gap_um`, label ids, node degree) so the policy still fires.

**Why this operating point is distinct and should be net-positive.** Gens 1/3/4
were *per-site* gates (tangent ≤35°, radius+image continuity, mutual-nearest
reciprocity) so tight they fired **0 edits** — no global selection, just a wall.
Gen 5 was a *loose per-site* gap+degree gate with NO global cap, so MANY marginal
edits fired at once and the wrong ones dominated → **−1.604** (over-merge,
inflated %Merged). Gen 7 changes the *mechanism*, not the threshold: it RANKS all
viable reattachments by one composite score and commits only the single most
confident one. A hard global budget of 1 means a few bad-but-passing candidates
can no longer be summed into a net-negative; at most one high-precision edit
lands. This sits deliberately between "fires nothing" (the budget is ≥1 and the
gate admits the gen5-proven non-empty pool) and "over-merges" (the budget caps
total exposure at one edit and the composite favors clean tip-to-tip
continuations over branch reattachments).

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
- **Gen 7:** Replaced the no-op seed with a `merge_labels`-only policy built on a
  NEW mechanism — a **strict global edit budget over a composite confidence
  ranking**. Every viable `SplitSite` reattachment (gap ≤4 µm) is scored by
  `0.30·gap_closeness + 0.40·terminal_cleanliness + 0.30·tangent_agreement`;
  only candidates ≥0.55 survive, and only the single top-ranked edit is committed
  (one per label). Rationale: gens 1/3/4 per-site gates fired 0 edits (too tight);
  gen5's loose uncapped per-site gate over-merged at −1.604 by letting many
  marginal edits accumulate. A hard top-1 global budget caps total exposure so a
  few bad-but-passing edits cannot outvote the safe one, while still firing >0 on
  the gen5-proven non-empty SplitSite pool. `split_label` remains disabled (no GT
  merge target in this split).
