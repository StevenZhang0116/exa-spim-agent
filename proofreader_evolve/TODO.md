# proofreader_evolve TODO

Open items that are discussed but deliberately not implemented yet. Each entry
records the current behaviour, the measured cost of changing it, and what the
change touches, so the decision can be revisited without re-deriving the facts.
Dates are when the item was recorded.

## Three-valued labels: open items (2026-10-09)

Done 2026-10-09 (WORKFLOW_REVISION "Three-valued labels throughout"): labels are
1 / 0 / NaN everywhere; ranking, promotion, bootstrap, selection, fitting, feedback and
guides use GT-labeled rows; optional `X_unlabeled` for `fit`. Still open:

- **K for three-valued runs (defaults set 2026-10-09: merge 300, split 500).** Labeled
  rows per cell are small (merge: 449-18,564; split: 1,795-13,074), so K = 2000 covered
  most or all labeled rows of several cells. Frozen-baseline P@K sweep over labeled rows
  (`scripts/three_valued_smoke.py`, log `notebooks/log/three_valued_k_sweep.log`):

  | K | split 789202 / 794493 / 794491 / 802449 | merge 789202 / 794493 / 794491 / 802449 (TP) |
  |---|---|---|
  | 50 | .86 / .92 / .80 / .92 | .00 / .02 / .04 / .20 (0/1/2/10) |
  | 100 | .90 / .84 / .77 / .93 | .00 / .03 / .05 / .16 (0/3/5/16) |
  | 200 | .90 / .81 / .80 / .935 | .03 / .025 / .035 / .165 (6/5/7/33) |
  | 300 | .887 / .813 / .777 / .91 | .027 / .023 / .030 / .137 (8/7/9/41) |
  | 500 | .884 / .81 / .762 / .892 | .020 / .024 / .024 / .118 (10/12/12/59) |
  | 1000 | .858 / .737 / .708 / .885 | .031 / .016 / .019 / .100 (31/16/19/100) |
  | 2000 | .796 / .661 / – / – | .021 / .013 / – / – |

  Labeled rows / positives: split 4,162/2,720, 3,596/1,756, 3,650/1,813, 13,074/6,393;
  merge 3,582/62, 2,363/29, 3,467/88, 18,564/825. Split K = 500 is ~14% of the
  validation labeled rows (SE ~0.018 per cell, ceiling 1.0, baseline headroom
  0.12-0.24); K >= 1000 drifts toward the ~50% labeled base rate, K <= 100 is noisy and
  near 0.9. Merge baseline P@K is near the labeled base rate (1.2-2.5%) at every K;
  K = 300 keeps 7-9 hits per validation cell and a ceiling of 29/300 ~ 0.10 on 794493
  (K = 2000 had ceiling 0.0145 there, baseline 0.013). Merge stays noisy whatever K;
  see the next item. Older-microscope merge cells (e.g. 754613: 449 labeled, 9
  positives) are unusable for merge validation.
- **Merge validation cells.** Labeled merge cells hold 9-825 positives; decide together
  with the merge-gate proposals (pooled validation merge cells, K scaled to positives).
- **Definitions.** Whether the merge 20-150 um ambiguous ring should also be NaN, and
  whether split needs a distance condition like merge.
- **Unlabeled sample size.** `UNLABELED_ROWS_PER_BRAIN` = 100,000 is a transport bound,
  not tuned.

## Merge labels are conservative: GT-node-count thresholds not rescaled (2026-10-07)

Full write-up: `notebooks/merge_label_shadow_findings.md`.

**Current behaviour.** `canonical_labeling` (agentic-neuron-proofreader) ports three merge
thresholds from `segmentation_skeleton_metrics` as GT NODE COUNTS of 50:
`MERGE_MIN_NODES` (node-count merge rule, `summarize`), `MERGE_PASSTHRU_MIN_CC` (walk
pass-through test) and `min_gt_nodes` (two-GT junction audit). The canonical package reads
raw GT SWCs (about 1.0 um between nodes), so 50 nodes is about 50 um; the cache GT graphs are
resampled (3.99-4.36 um between nodes), so the same threshold is about 200 um and short
genuine merges are rejected as pass-throughs. `MERGE_MIN_NODES` is also bound as a default
argument in `merge_labels` and `audit_junction_gt_connections`, so changing the module
constant alone does nothing.

**Measured.** Cache merge sites recover 46-89% of canonical sites (site recall @ 50 um;
precision 0.83-1.00; same-segment placement 2.4-4.5 um, so coordinates are right). Most
missed sites are on segments present in the cache but not flagged; their same-label GT
component has a median of 26-31 cache nodes (70-88% below 50). A shadow relabel with the
thresholds set to the same cable length (50 um = 11-13 cache nodes;
`scripts/shadow_merge_relabel.py`, exact reproduction with the original thresholds on all
11 brains) raises site recall to 0.80-0.94 at precision 0.83-0.96; the older-microscope
brains gain most (754610 0.46 -> 0.81, 750318 0.59 -> 0.90).

**Downstream impact (5 current brains, `scripts/shadow_merge_impact.py`).** Positives in
the frozen merge detector's pool grow 8-24% (802449: 825 -> 973), but the new positives rank
at median 5,900-75,000 and only 14 enter any Top-2000; P@500 / P@2000 change by <= 0.005
and the per-brain ordering is unchanged. The recovered merges are mostly short and away
from junctions, which the junction-anchored policy reaches less often.

**Decision (2026-10-07): not fixed.** A fix means relabelling all 11 `_add.pkl`, which
invalidates image-alignment receipts, feature tables, availability sidecars and context
caches (all bound to the cache identity) and makes past results incomparable; it may also
require refitting the merge detector. Not justified by the measured impact. Known bias:
merge label 1 is under-counted (10-20% of true merge sites are invisible to the current
merge pipeline), separately from the label-0 "no GT here" problem handled by the
availability sidecars.

**If revisited.** Make the thresholds length-based in `canonical_labeling` (old behaviour
behind a flag), relabel every brain together (`scripts/relabel_cache.py`), then rebuild
tables, sidecars and context caches, re-record alignment receipts, decide whether to refit
the merge detector, and add a dated `WORKFLOW_REVISION.md` entry. Worth doing together with
any redesign of merge candidate generation (a non-junction candidate family) or the merge
promotion gate (see "Merge is invisible to the promotion metric" below), since all three
change the merge benchmark at once. Remaining gap after rescaling (recall 0.80-0.94): the
cache walks against the union of GT neurons rather than per neuron, canonical snaps sites
to branch nodes, canonical dedups on its mislabeled `World` column, and fragments below the
100 um MCL are absent.

## Context cache: band grown to 50,000 rows and older-microscope brains added (2026-10-08)

**Status: code done 2026-10-08** (`ContextCacheBuilder.extend_band`; `precompute_context_cache
--band-rows 50000` grows entries in place; `ExtendBandTests`). **Builds not yet run** (owner: user; commands in
README Stage 3; run the extension of the five current brains only after no evolution run is
reading the cache; one older-microscope brain per job, 200 GB). Measured
before the change: the evolved scorer's Top-2000 on 802449/split drew 24 of 2,000 rows from outside
the 20,000-row band and still missed 60% of in-band positives, so the growth is a moderate step,
not a fix for a binding limit; 200,000 rows (about 120 GB, 30–50 min descriptor runs) was declined.
Re-measure band coverage after a run on the 50,000-row band before growing further.

## Context cache: extend the level-0 image tier to the whole band (2026-10-05)

**Status: implemented 2026-10-05** (`max_rows: None`, `ContextCacheBuilder.extend_tier`,
`precompute_context_cache --extend-tier level0`, tier-selective bank identities; build
ran 2026-10-05/06: 10 entries, 0 failures, 2 h 21 min, level-0 tier 1.55 -> 7.77 GB). The
notes below are kept as the design record; the remaining open idea is refreshing the
band with rows the evolving scorer pulls into the Top-K.

**Current behaviour.** `IMAGE_TIERS` in `harness/context_cache.py` caches the
level-1 image patch (30 um radius, 1.496 x 1.496 x 2.0 um voxels, 30 x 41 x 41)
for every row of the 20,000-row detector-ranked band, but the level-0 patch
(16 um radius, 0.748 x 0.748 x 1.0 um voxels, 32 x 43 x 43) only for the first
4,000 rows (`max_rows: 4000`). The cap was a budget choice made when v14 was
built: 4,000 rows is the Top-2000 scoring window plus the next 2,000, the same
boundary convention as the v11 `top_k_boundary` selection.

**Why it may need to change.** A descriptor declared with
`image_tier: 'level0'` is NaN for band ranks 4,001 to 20,000. A reranker can
pull such rows into the Top-2000, and the model then has no fine-resolution
information for exactly those rows. Level-1 voxels (about 1.5 x 1.5 x 2 um)
undersample thin axons, so sub-2-um evidence (continuity, small gaps) is only
measurable at level 0. The gen3 reviser of run
`precision_20261005_114814_zxsbw5pj` chose level 1, trading resolution for
coverage.

**Measured cost (2026-10-05 build, ten brain/kind entries, 0 failed reads).**

| tier | reads | on disk | build time |
|---|---|---|---|
| level1, whole band | 200,000 | 6.07 GB | 123 min |
| level0, first 4,000 | 40,000 | 1.55 GB | 24 min |

Per-read cost is the same for both tiers (25 to 30 reads/s aggregate) because
the level-0 radius was halved to keep the voxel count comparable. Extending
level 0 to the whole band adds about 160,000 reads, roughly +100 min and +6 GB
at that rate.

**What the change touches.**

- `IMAGE_TIERS['level0']['max_rows']` -> `None` in `harness/context_cache.py`.
- Done 2026-10-05: `ContextCacheBuilder.extend_tier` adds or grows one tier of a
  complete entry without re-reading level 1 (the `--force` rebuild estimated here,
  about 4.3 h, was avoided; the extension took 2 h 21 min). Done 2026-10-08:
  `extend_band` does the same for a larger band.
- Done 2026-10-05: `ContextCache.identity_key(table, tiers=...)` is tier-selective, so
  extending one tier invalidates only descriptor-bank entries that read it; coverage
  estimates follow the manifest's per-tier row counts.
- Do not rebuild a cache root while an evolution run is reading it; build into
  a separate `--context-cache` root or wait for the run to finish.
- Done 2026-10-05: `WORKFLOW_REVISION.md` cache paragraph and v14 provenance row updated.

## Observed in run `precision_20261008_001127_bqp8vmtv`, not yet fixed (2026-10-08)

Context: 16 generations done, validation 0.085 -> 0.1411 (generations 1, 2, 5, 6, 9 promoted),
nothing promoted since generation 9 although TRAIN selection kept creeping (0.568 -> 0.574).
Two generations were lost to end-of-session checks and the search stayed in tuning mode.

- **Done 2026-10-08 (`sanitize_research`).** `validate_research` was fail-closed at the submission check. Generation 16 (merge)
  finished with a measured, restored candidate and failed only because the agent added an
  `outcome` key to `proposal.json.research` (25 min, $5.9). Same class as the handoff
  (fixed 2026-10-07, `sanitize_handoff`) and `strategy` length (bound raised to 3000)
  failures. Fix: sanitize like the handoff (drop unknown keys, clip text), never raise.
  Implemented in `hypothesis_memory.sanitize_research`, `search_proposals.read_proposal`, tests.
- **Done 2026-10-08 (submission fallback; default 60 turns).** A session that hit `--reviser-max-turns` failed the whole generation. Generation 15
  (split) measured six candidates (0.570–0.573 vs parent 0.568) while finally changing the
  information source to `local_geometry`, then issued its 41st tool call; the SDK ended the
  session with `error_max_turns` and the driver raised instead of submitting (124 min, $6.2;
  the attempts survived only in the TRAIN archive). Fix: on `error_max_turns`, run the normal
  submission check on the current `scorer.py`, and when it does not match a measured
  snapshot fall back to the generation's best measured candidate. Also raise
  `DEFAULT_REVISER_MAX_TURNS` 40 -> 60: exploration rounds that write a new descriptor family
  used 44 tool calls. Touches `cli/run_precision_evolution.revise` (the RuntimeError on
  error subtypes), `TrainingExperiments.submitted` / `restore('best')`, tests.
- **Plateau detection counts TRAIN novelty, not validation-accepted progress.** The
  `stalled_rounds` counter stayed at 0 for six rejected generations because each one
  raised the selection score by 0.001–0.006, so the explore/pause machinery never
  engaged and the agent kept retuning the same 38-column classifier
  (`*-retune-image-gbdt`, information source `tabular_bank_features`) despite its own
  handoffs saying "stop tuning, change the information source" (generations 10–14).
  Proposal (host-side only, no validation leak to the agent): count TRAIN progress only
  when the selection gain exceeds the fold-to-fold spread of the OOF measurement; after
  N (2–3) consecutive rejections within bootstrap noise, make the next round an explore
  round whose `research.information_source` must differ from the recent ones, otherwise
  the submission is not promotable. Touches `candidate_pool` progress accounting,
  `kind_schedule`, `TrainingExperiments.research_status`, scorer rules text.
- **Done 2026-10-08 (grid stops at the first execution error; kill message names the cap; default cap 16384 MiB, also for scoring).** A parameter grid kept running after a worker was killed for memory. Generation 17 ran a
  pre-registered four-cell grid of a 16-member subspace ensemble; every cell fitted fine
  (3 folds plus the full fit, about 30 min each) and then the scoring worker was killed
  with signal 9 at the 8 GiB `--classifier-memory-mb` cap while scoring the 1.39 M-row
  pools. The harness ran all four cells before the agent saw a result (2 h 10 min, four
  evaluation units), and the error text ("Model worker exited -9 without status" plus a
  joblib warning) does not say "memory limit"; the agent inferred it. The node had 449 GB
  free, so the cap is ours. Fixes: stop a grid after the first execution error of the same
  program; name the memory cap in the worker-kill message; raise the default
  `--classifier-memory-mb` (8192) given 200 GB jobs, or let the agent see the cap in
  `model_environment.json`. Not started; touches `train_experiments.search_parameters`
  / `train_classifier` grid loop, `model_execution.predict_model`, the driver flag.
- **Per-attempt classifier time.** Generations 10–15 spent 90–125 min each on 5–8 fits of
  500-tree, depth-8 ensembles (about 25 min per fit at `--classifier-time-budget 1800`),
  with in-sample vs OOF gaps up to 8 points. Options: charge evaluation units by wall
  time, or lower the per-fit budget for retune families. Not started.
- **Merge is invisible to the promotion metric (why merge never promotes).** The validation
  merge pools hold 62 / 29 / 88 positives against K = 2000, so P@K is capped at
  0.031 / 0.0145 / 0.044 and the seed already finds 52% / 34% / 6% of those positives.
  Promotion uses the equal-weight mean over six cells, so ten extra merge hits on one
  brain are +0.005 on that cell and +0.0008 on the mean, below any bootstrap lower bound;
  even recovering every validation merge positive would add only about 0.011. TRAIN
  gains on 802449 (0.07 -> 0.10–0.11 with table-only XGBoost, generations 4 and 8) traded
  hits between brains on validation (789202 32 -> 19 / 23, 794491 5 -> 16 / 18, 794493
  10 -> 8; net about zero). The only merge changes that moved all three validation brains
  in the same direction were the image-patch and junction-geometry descriptor banks of
  run `precision_20261005_114814_zxsbw5pj` (generations 3 and 5: +6/+10/+1 and
  +9/+16/+3 hits), each worth only +0.001 on the mean. Tried and refuted since:
  junction-saddle image descriptors, boundary-weighted fits, depth-diverse ensembles,
  cross-brain percentile blends (generation 12 here: the inverted sign "helped" too, so
  noise); a junction-local image GBDT (generation 16, 0.1155 OOF) was lost to the
  `research` schema failure above. Pool and label limits add to this: bridge merges
  (about 9% of sites) have no junction and are never candidates; the 20–150 µm ring is
  labelled negative; only 34 of 794491's 88 merge positives lie in the 20,000-row band.
  Proposals, in order: (1) judge merge separately — a per-kind gate, or a merge K scaled
  to the positive count (K = 200) or recall@K — so a real merge gain is not diluted six
  times; (2) pool the three validation merge cells (micro P@K over 179 positives) before
  gating, because per-brain cells with 29–88 positives are too noisy; (3) stop
  floor-scheduling merge until a junction-local image information source exists for it
  (raise `--kind-floor-every` from 4 to 8 or more, or require a new
  `research.information_source`). Not started; touches `fixed_pool_scoring` aggregation,
  the gate record, `kind_schedule`, README Stage 3 and the agent rules.

## Observed in run `precision_20261005_114814_zxsbw5pj` (2026-10-05; two of three resolved)

- Resolved 2026-10-06: `image_evidence.descriptor_image_computations` was always
  empty because the image-evidence summary ran before `record['descriptor_runs']`
  was assigned; the order is now fixed.
- Resolved 2026-10-05 by raising limits (MCP output cap 100,000 tokens, feedback
  budget 96 KB, parked tool results readable): oversized `search_memory`,
  `train_classifier` and `restore_candidate` results and the 48 KB feedback cap.
  Raised again 2026-10-07 (feedback 128 KB, SDK file-read cap 100,000 tokens,
  generation artefact directories readable) after parked results above about 50 KB
  could not be read in one call; see WORKFLOW_REVISION "Limit increases".
  Still open if context cost matters later: return file paths plus summaries
  instead of embedding the full feedback (about 39 KB of feature matrix) in
  `train_classifier` / `restore_candidate` results, and index-level
  `search_memory` entries.
