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

## Current criteria (Generation 3)

Split-repair (`merge_labels`, unchanged from the working seed path):
1. For each SplitSite, unify the two fragment labels only when the gap is small
   (`s.gap_um <= GAP_THRESHOLD_UM`) AND the two cables are colinear across the gap
   (straight-line continuation; see `_is_colinear_split`). High precision.

Merge-repair (`split_label`, NEW in Gen 3 — addresses the dominant remaining error):
2. For each MergeSite, cut the label into two neurons only when geometry signals
   AGREE it is two distinct neurites, gated by detector:
   - **both arms long** (`min(cable_a_um, cable_b_um) >= SPLIT_MIN_CABLE_UM`) is a
     hard prerequisite for every detector, and `arms_reconverge is True` is an
     instant reject (one neuron's own loop/branches).
   - `detector == "component"`: two long DISCONNECTED pieces — accept on geometry
     (the disconnection itself is the signal; `angle_deg` is NaN here).
   - `detector == "branch"`: accept on geometry when BOTH a sharp arm angle
     (`<= SPLIT_MAX_ANGLE_DEG`) AND a caliber mismatch
     (`radius_ratio >= SPLIT_MIN_RADIUS_RATIO`) hold; if only ONE holds, require
     image valley evidence to confirm before cutting.
   - `detector == "bridge"`: require BOTH a sharp kink AND a caliber pinch, then
     confirm with image valley evidence.
3. **Image is used to CONFIRM-and-ADD, not to reject.** When `read_image_patch` is
   available, `merge_cut_evidence` along the chord between the two arm seeds must
   show a valley (`valley_ratio <= SPLIT_VALLEY_RATIO_MAX`) near the middle
   (`SPLIT_VALLEY_POS_LO <= valley_pos <= SPLIT_VALLEY_POS_HI`) — a true touch of
   two structures — to clear a borderline split. Geometry-only acceptances
   (component, or branch with both signals) do not need it.

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
- **Gen 1 (REJECTED, +0.000):** endpoint-adaptive colinearity threshold (tighter
  ~0.985 cosine for tip-to-shaft/branch joins) plus a `node_radius` continuity
  guard. A precision FILTER on the existing colinear-merge list. Dead end.
- **Gen 2 (REJECTED, +0.000):** mutual-nearest-neighbor / reciprocity gating on
  the gap. Also a precision FILTER on the same colinear-merge list. Dead end.
- **Gen 3:** *Diagnosis of the +0.000 stall.* The gen03 report shows the 51
  merge_labels edits earn their Edge-Accuracy gain almost entirely by reducing
  **%Omit Edges** (e.g. N010 −2.275, N005 −1.471), while **%Merged Edges barely
  moves** (17.22 → 17.11) and **zero `split_label` edits** were emitted — leaving
  all 10 documented two-neuron labels uncut. Filtering which colinear merges
  survive (gen1/gen2) cannot move held-out: the rejected pairs either don't exist
  on held-out or carry no omit-edge value, so the gradient is flat. *Change:* a
  mechanistically DIFFERENT lever — start emitting `split_label` edits to attack
  the dominant **%Merged Edges** component the merge_labels path never touched.
  Gating is multi-signal and detector-conditioned (both arms long + sharp angle /
  caliber mismatch; component-disconnection; reconverge reject), with image valley
  evidence used to CONFIRM-AND-ADD borderline cuts rather than to reject. This
  adds a new class of correct edits instead of trimming the existing one.
