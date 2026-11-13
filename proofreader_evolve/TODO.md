# proofreader_evolve TODO

Open items only. Each entry says what is open, why it matters and where it would be
changed; "checked" dates say when the code was last compared with the entry.

This file was reduced from 861 to its open items on 2026-11-09. Completed work, superseded
proposals and the measurement tables that older documents cite from here (the 2026-10-09
K sweep, the context-cache coverage tables, the 2026-10-29 three-arm state) are in the
change log of [WORKFLOW_REVISION.md](WORKFLOW_REVISION.md) and in the previous version of
this file: `git show efcd2be:proofreader_evolve/TODO.md`.

For merge, read section 7 first: it names what currently limits the merge scorer. The sections
keep their numbers because other documents cite them.

## Standing decisions

- 794495 is reserved for a future independent validation; no fitting, selection, tuning,
  testing or promotion on it. The split and launch recipe are in
  [Stage 3](../docs/stage3_evolution.md#32-prepare-tables-and-launch-a-run).
- Positive-weighted aggregation is shared by selection, ablation, validation and the
  bootstrap. No LLM takes part in TRAIN-side selection or in promotion. Promotion: one gate
  since 2026-11-12, `--promotion-gate rule` (the default; guard rail, then the objective's
  interval above zero and the gain above `--precision-margin`; `harness/promotion_gate.py`).
  The documented launch adds `--objective average_precision --precision-margin 0.002
  --selection-noise-gate` (section 8). The LLM judge (2026-11-10 to 2026-11-12), the
  `bootstrap` gate mode and the point-estimate head guard were removed on 2026-11-12.
- The five validation brains are development data: the gate has judged many candidates on
  them. A result on them is not a held-out test result.
- Do not change the objective or the gate while a run that will be compared is running;
  every such change gets a WORKFLOW_REVISION row naming the runs it applies to.

## 1. Is the workflow's complexity paying for itself? (2026-11-09)

**The question.** The workflow has grown faster than the evidence for its parts.

| Measured 2026-11-09 | Amount |
|---|---|
| Harness | 59 files, about 12,500 lines |
| Driver command-line options | 54 |
| Tests | 550, about 12,000 lines, about one hour for the whole suite |
| WORKFLOW_REVISION.md | 2,336 lines, 90 change rows, versions v9 to v17 in five weeks |
| Agent-facing guides | 7 files, about 1,300 lines |
| Evidence base | the ten most recent runs with at least three generations: 71 generations, 42 accepted |

What is not in doubt: the workflow as a whole works. Accepted scorers beat the raw-heuristic
baseline on development validation (merge AUROC 0.762 to 0.847 / 0.863 after one generation
of the two 2026-11-08 runs).

What is unknown: how much of that comes from the search-control machinery and how much from
letting an agent write features and models at all. No run has removed one of these
mechanisms to measure its effect, and the provenance section of WORKFLOW_REVISION states that
a cited paper is not evidence that the implementation improves anything.

**Why the machinery is suspect** (judgement, not measurement):

- It is built for long searches, and a run has five to ten generations, so three to five
  decisions per kind. The branch pool (five branches plus a protected reference), hypothesis
  memory, handoff records, adaptive kind allocation, stagnation penalties, the exploration
  interval and the plateau patience have little room to act and no way to show that they did.
- Most mechanisms were added in reaction to one run's observation.
- Image tooling is larger than its use: the preview allowance is 32 per generation and the
  most used in the 2026-11-05 and 2026-11-08 runs was 4, usually 0.
- Run-to-run variation is as large as the effect expected from any one mechanism: the two
  identical 2026-11-08 runs reached merge validation AUROC 0.847 and 0.863 in generation 1.

**What is not overengineering and must stay in any simplification:** the TRAIN / validation
separation (the agent never sees validation results), the paired-bootstrap promotion gate
with its guard rail (the head guard until 2026-11-12), whole-brain holdout selection, and three-valued labels. They answer a
property of this data set: some merge pools hold 15-22 positives, so one comparison's noise
exceeds a real gain.

**Proposal: a simplified control arm.** Keep the outer loop and everything in the previous
paragraph. Replace the inner loop by a single line of work in the manner of Karpathy's
`autoresearch` (one editable program, one measurement per change, keep it if it improved,
discard it otherwise): no branch pool, no hypothesis memory or handoff, no adaptive kind
scheduling (fixed alternation), one short instruction file. Start with geometry-only
descriptors so that one TRAIN evaluation takes minutes. `autoresearch` itself is not a
drop-in replacement: it keeps or discards on the validation metric directly, about a hundred
times a night, which its large text validation set tolerates and our merge pools do not, and
it assumes a fixed five-minute experiment where a whole-TRAIN fit with image descriptors
takes tens of minutes here.

- Prerequisite: a completed baseline run on the 2026-11-09 sources with the documented
  resource settings, and at least one replicate, so that the run-to-run spread is known.
- Compare on the same partition, objective, gate and generation count: accepted validation
  objective per kind, acceptance rate, wall time and API cost per generation.
- Decision rule, fixed before running: if the control arm lands inside the replicate spread
  of the full workflow, remove the corresponding machinery; if it is clearly worse, the
  complexity has its evidence and this section closes.
- Cost: a new inner driver, estimated at a few days; not a parameter change.

## 2. Design defects found in use (2026-11-08 / 09)

Concrete, each one met while switching seven brains to denoised images.

- **Cache identity is the file's size, inode and mtime.** `--refresh-image-path` rewrote only
  `img_path`, the candidate rows did not change (`pool_sha256` equal before and after on all
  fourteen pools), and the raw pools and sidecars of seven brains still had to be deleted and
  re-prepared (522 + 1669, 911 + 3646 and 183 + 620 s). Bind derived artifacts to content
  (graph and label digests) instead of file identity, or let a path-only rewrite keep them.
  `harness/local_context.cache_identity`, the table provenance in `native_pool.py` (do not
  edit without re-preparing: its hash enters the table digests), `labeled_cache/build.py`.
- **The reviewed image URI takes precedence over the cache path.** After a volume is
  replaced, the old receipt must lose its `reviewed` status by hand before the alignment
  check reads the new volume; otherwise the deleted URI is returned
  (`scripts/dataset_config.get_cache_img_path`). Make the alignment script accept an explicit
  image root, or detect an unreadable reviewed URI and say so.
- **The driver does not size its resources to the allocation**, and its defaults are hard
  limits inside the reviser's tools, so a run with defaults is restricted without a visible
  failure. Documented on 2026-11-09 (Stage 3, "Give the run the allocation's resources");
  the code is unchanged. Derive the defaults from the step's CPUs and the job's memory limit,
  or warn at start-up when they are far below the allocation. The documented settings
  (`--classifier-threads 32 --classifier-search-seconds 7200 --descriptor-workers auto
  --reviser-max-turns 80`) have not been exercised by a run yet.
- **The driver's default TRAIN brain is the reserved 794495 and nothing enforces the
  reservation.** A command without explicit lists uses it. Change the default and add a
  reserved-brain guard to the driver, preflight and the preparation tools
  (`--train-brains` default in `cli/run_precision_evolution.py`, `cli/preflight.py`).
- **The same fact is written in several documents** (CLAUDE.md, README, Stage 3, artifact
  dependencies, WORKFLOW_REVISION, the cache schema document, a status note, a summary JSON).
  Every revision on 2026-11-08 / 09 needed several passes to find stale copies, and each pass
  found some. Give each fact one home and link to it.
- **"raw" has three meanings**: an unprocessed (not denoised) image, the un-annotated row of
  a preview, and the detector-free prior mode (`--detector-prior raw`, raw pools). Rename the
  first two where they face the agent.

## 3. Search protocol

- **Confirm the AUROC objective on real candidates** (default since 2026-11-02, objective
  v7). On the next completed runs: (a) paired-bootstrap interval widths of AUROC and P@K on
  the same candidates; (b) Spearman correlation of each selection statistic with the
  validation objective and with validation P@K; (c) guard-rail rejections and rejected generations
  whose P@K rose (the point-estimate head guard this item first named was removed on 2026-11-12).
  Revert to `--objective precision_at_k` if validation P@K falls over a run.
  **First evidence, 2026-11-10** (`notebooks/acceptance_rule_replay.py` on the 19 validated
  candidates of runs `uxuqkoag` and `e7y6ie3_`, every candidate rescored to the ledger's values
  and every AUROC interval reproducing the gate's): on merge the AUROC gate and an
  average-precision gate disagree on five of the eight candidates that had a non-seed parent.
  AUROC accepted two candidates whose average precision and depth-mean intervals include zero
  (one with precision at K down 0.025) and rejected three whose average precision and
  depth-mean intervals lie above zero. On split the two gates agree on all seven. False
  acceptance on null pairs (exchangeable hybrids of real pairs): AUROC gate 6/200 merge and
  5/175 split, average-precision gate 5/200 and 3/175. An LLM judge shown the statistics with
  their intervals (five votes, numbers only) made the average-precision gate's decision on
  14 of the 15 pairs and accepted 1 of 60 null pairs; without the intervals it accepted fewer
  real pairs and its votes split more often. Decided by the user on 2026-11-10: the next runs
  use `--objective average_precision --promotion-gate judge` (done: the two runs of 2026-11-12;
  the comparison is in section 8 and the judge was removed that day). The list that follows is
  kept as written on 2026-11-10:
  compare the judge's decisions with the average-precision gate's (both are in the ledger: the
  objective's bootstrap is recorded beside the ballots), count split votes and failed calls, and
  check whether the rejection category shown to the reviser (`outer_measurement.rejection`,
  four host-derived words) reduces near-identical resubmissions of rejected candidates. Limits: two runs, candidates
  produced under the AUROC objective, development-validation data, and the replay keeps each
  run's parent sequence fixed although another gate would have changed it.
- **Plateau detection counts TRAIN novelty, not validation progress.** `stalled_rounds` in
  `harness/candidate_pool.py` advances on TRAIN selection gains (checked 2026-11-09), so a
  kind whose candidates are rejected inside the bootstrap interval keeps its budget. Stop
  allocating a kind after N consecutive such rejections and send the budget to the other kind
  or end the run; host-side only, no validation value shown to the agent. The same rule would
  cover the `tune` rounds, which accepted nothing in twelve generations of the 2026-10-27
  runs. Depends on section 1: a simplified inner loop removes most of this machinery.
- **Merge is limited by within-junction discrimination, see section 7** (2026-11-11); the item
  below is the statistical side of the same problem.
- **Merge has little statistical power.** Validation holds 439 merge positives over five
  brains (15 / 41 / 125 / 140 / 118 on the 2026-11-07 pools) and the gate needs the lower bound
  of a two-sided 95% paired-bootstrap interval above zero. Options if it proves too strict: a one-sided test at
  alpha = 0.05, a merge-specific margin, or more merge GT positives, which is the real fix.
  These are methodological choices, not defects.
- **Replicates.** Each run is one search trajectory. Run at least two per configuration
  before reading a difference between configurations.

## 4. Harness robustness

- **Host faults before the agent runs are recorded as research failures.** A missing API key,
  an SDK transport error or a network outage raises in the `revision` stage, counts as an
  `execution` failure, pauses both kinds after `--exploration-patience` rounds and ends the
  run with `status completed` (run `precision_20261102_133644_rst004og`). Check the key and
  the model in the driver before loading tables; classify exceptions raised before
  `run_reviser` hands over as host faults, retry with a bounded backoff, and stop with
  `status failed` if the retries fail. Not implemented (checked 2026-11-09).
  `cli/run_precision_evolution.py`, `harness/research_evidence.py`,
  `harness/candidate_pool.py`, `harness/reviser_session.py`.
- **Host-side failures carry no traceback**, only `Failed: <type>: <message>`; add
  `traceback.format_exc()` to the host-only trajectory event. `ensure_descriptors` returns
  its per-brain reports only after every brain has finished, so a failure on a later brain
  leaves `descriptor_inference` empty for the brains that completed.
- **No heartbeat.** `run_status.json` is written once per generation; `phase_started_at` and
  `last_event_at` would tell a stalled run from a long fit. Terminal-step reconciliation
  (launcher, `check_run_status`) exists; continuous progress detection does not.
- **Descriptor planning and resumption.** The plan's sample amortizes the worker start-up
  over four times fewer rows than a batch (about 2.8x overestimate after the 2026-11-03 fix),
  and a descriptor call cannot resume per brain after a failure.
- **Descriptor values for `X_unlabeled`.** Under the judgeable scope the unlabeled rows
  staged for `fit(..., X_unlabeled=...)` carry NaN in every `bank_agent_*` column. If
  semi-supervised use matters, add a scope that unions the judgeable rows with the
  `_unlabeled_sample` rows. `UNLABELED_ROWS_PER_BRAIN` = 100,000 is a transport bound, not a
  tuned value.
- **Tool result size.** `search_memory` returns full experiment records (211,980 characters
  in one call of the 2026-10-28 raw run). Return index-level entries with file paths if context cost matters.

## 5. Data, labels and candidates

- **Three development brains read unprocessed images and have no denoised volume on their
  grid**: 802449 (TRAIN), 789202 and 794491 (validation). If AIND deletes one, there is no
  replacement and its image descriptors fail. The other nine brains read denoised pixels
  (`cache/dataset_source_status_20261108.md`).
- **Mixed pixels are unmeasured downstream.** Raw and denoised volumes differ in background
  noise (high-pass ratio 0.15-0.22) and little else in the measured patches, but no descriptor
  has been computed on both versions of one brain. Only 802449 among the TRAIN brains is raw.
- **Split candidate recall**: worst brain 0.80 (802449) since the 2026-11-05 segmentation
  change; the 0.90 floor of the 2026-10 sweep is no longer reachable with
  `tip_to_any_node|r=80|k=2`. Flank pairs across an omitted stretch were the known misses under r=50.
- **Merge site recall at 150 um**: 789202 0.88 and 841260 0.89, below the 0.90 target; the
  user kept `gap=20` on 2026-11-07. Open idea for the blind bridge sites: a path change-point
  family (direction, radius or edge-length jumps on unbranched paths), to be measured against
  the same marginal positive rate before adoption.
- **Label definitions.** Whether the merge 20-150 um ring should be NaN rather than 0, and
  whether split needs a distance condition like merge.
- **Label review.** Short-gap collinear label-0 split pairs look like positives; a GT-side
  check of a sample would say whether the remaining split false positives are label
  convention.
- **Fragment-level context.** Agents asked for it repeatedly: whether an anchor's fragment
  continues elsewhere, competing partner tips, and the tip's own direction past the shaft for
  `tip_to_shaft` merge rows. `describe(context)` sees a 50 um neighbourhood.
- **Independent validation on 794495 is undesigned.** `evaluation/` holds a source audit
  only.
- **Minor.** `notebooks/verify_add_cache_metrics.py` reports the canonical `SWC Run Length`
  in voxel units, so the column is not comparable with the cache's.

## 6. Deferred: the frozen / no-score arms

Raw runs need no detector. The detector-native tables were deleted on 2026-11-07, so a
`frozen` or `no-score` run first re-prepares them (hours per brain). If those arms are taken
up again, the 2026-10-29 findings still apply to the detector build: filter merge hypotheses
by cross-brain generalisation and a brain-shift screen, replace absolute radii and raw
intensities by ratios, ask the specification for fragment geometry at 30-50 um, and accept a
rebuilt detector only if the frozen arm reaches the no-score / raw level on validation merge,
not on a better AP on its fitting brain. Details: `git show efcd2be:proofreader_evolve/TODO.md`,
section P4.

## 7. Merge: the scorer does not separate merge junctions from true branch points (2026-11-11)

**Diagnosis.** The pooled merge AUROC is inflated by the candidate family. Merge pools hold
two families, `junction` and `tip_to_shaft`; nearly every positive is a junction and nearly
every `tip_to_shaft` row is a negative, so ranking junctions above `tip_to_shaft` rows already
gives a high pooled AUROC. Inside the junction family, where the task is, discrimination is
weak. Measured on the accepted merge scorer of run `precision_20261111_004622_zl8_wyc1`
(generation 7), whole-brain holdout scores of the six TRAIN brains, GT-labeled rows only:

| TRAIN brain | pooled AUROC | junction: labeled / positives / base rate | junction AUROC | `tip_to_shaft` positives, of them in the top 5% |
|---|---|---|---|---|
| 802449 | 0.753 | 6,883 / 98 / 1.4% | 0.625 | 6, none |
| 747807 | 0.894 | 1,213 / 113 / 9.3% | 0.738 | 12, none |
| 750318 | 0.824 | 1,439 / 114 / 7.9% | 0.661 | 30, none |
| 754610 | 0.876 | 820 / 14 / 1.7% | 0.844 | 5, none |
| 754612 | 0.914 | 747 / 22 / 2.9% | 0.743 | no positive |
| 754613 | 0.941 | 449 / 9 / 2.0% | 0.783 | no positive |

What follows from the table:

- The pooled statistic hid the state of the task. This is the mechanism behind "AUROC rose and
  the head of the ranking did not" (section 3).
- No `tip_to_shaft` positive (53 on TRAIN) reaches the top 5% of its brain: the scorer treats the
  family as negative.
- The junction base rate is 1.4% on 802449 (denoised Google segmentation) against 8-9% on
  747807 and 750318, so the same scorer puts mostly true branch points at the top of 802449.
  This is the direct reason for its 4-7 hits of 104 in every run.
- Three runs with different gates and objectives (AUROC with the bootstrap gate twice, average
  precision with the judge once) ended at the same merge level (validation AUROC 0.876-0.881,
  average precision 0.10-0.13). The gate and the objective decide whether an improvement is
  recognized; they do not produce one.

Interpretation: within 50 um of geometry and a local image patch, a crossing or touching pair
of neurites resembles a branch point. The first guess at what is missing, longer-range fragment
context, was tested on 2026-11-11 and is **not supported**:

**Fragment totals and arm subtrees do not separate the classes**
(`notebooks/merge_junction_study/merge_junction_fragment_context.py`, six TRAIN brains, junction
family, GT-labeled rows, whole-brain holdout; outputs in
`notebooks/merge_junction_study/merge_junction_fragment_context/summary.md`).
For every junction the script computes the fragment's cable length, tips and branch points and,
for each arm, the cable, tips and branch points of the whole subtree behind it (the arms sum to
the fragment total within 4e-12 um), plus the through pair and the side arm from a 30 um walk.

| Junction family, positive-weighted over the six brains | AUROC | average precision | hits in Top K (K = positives) |
|---|---|---|---|
| accepted scorer's holdout score (reference) | 0.690 | 0.122 | 54 / 370 |
| fragment features alone | 0.652 | 0.096 | 41 / 370 |
| scorer score + fragment totals | 0.702 | 0.149 | 64 / 370 |
| scorer score + all fragment features | 0.698 | 0.133 | 57 / 370 |

The one-feature AUROCs are 0.44-0.61 on the three large brains and change sign between brains
(fragment cable length: 0.46 on 802449, 0.51 on 747807, 0.32 on 750318, 0.57-0.71 on the three
small brains), and adding the features lowers 747807 (0.738 to 0.664) while raising 750318
(0.661 to 0.727). A merged fragment is not reliably larger and the arms of a merge junction are
not reliably longer than those of a true branch point. Limits: a fixed, untuned
gradient-boosting model; simple size statistics only (no path-following of where an arm goes);
9-22 positives on the small brains.

**The labels are on the right junctions; they are not what caps the junction AUROC**
(`notebooks/merge_junction_study/merge_junction_label_localization.py`, 2026-11-11, six TRAIN
brains, junction family, GT-labeled rows; outputs in
`notebooks/merge_junction_study/merge_junction_label_localization/summary.md`).
The question was whether a positive label often lands on a true branch point beside the false
connection. Each arm of a junction is followed over 60 um of fragment cable and compared with
the tracing (a GT node within 6 um, the cache's own walk distance). `two_gt`: arms on different
GT neurons. `leaves_gt`: one arm on a GT neuron, another arm runs 60 um without any tracing (the
pattern the geometric walk records). `gt_branch`: every arm on the same GT neuron. `gt_spur`:
the other arms are short and untraced. `no_gt`: no arm follows a tracing.

| Junction rows, six TRAIN brains | rows | `two_gt` | `leaves_gt` | `gt_branch` | `gt_spur` | mixed | `no_gt` |
|---|---|---|---|---|---|---|---|
| positives | 370 | 29 (8%) | 309 (84%) | 4 (1%) | 10 (3%) | 14 (4%) | 4 (1%) |
| negatives | 11,181 | 22 (0%) | 353 (3%) | 8,277 (74%) | 428 (4%) | 675 (6%) | 1,426 (13%) |
| mean within-brain rank of the accepted scorer, positives (1 = top) | | 0.67 | 0.70 | 0.87 | 0.38 | 0.46 | 0.49 |
| the same, negatives | | 0.49 | 0.59 | 0.51 | 0.30 | 0.30 | 0.53 |

| TRAIN brain | junction AUROC, all rows | positives / negatives where label and category agree | junction AUROC on those rows |
|---|---|---|---|
| 802449 | 0.625 | 90 / 6,388 | 0.624 |
| 747807 | 0.738 | 102 / 433 | 0.814 |
| 750318 | 0.661 | 106 / 337 | 0.762 |
| 754610 | 0.844 | 13 / 601 | 0.878 |
| 754612 | 0.743 | 19 / 555 | 0.775 |
| 754613 | 0.783 | 8 / 391 | 0.792 |

("Agree": positives in `two_gt` or `leaves_gt`, negatives in `gt_branch` or `gt_spur`.)

- 338 of 370 positives (91%) sit on a junction where the tracing itself shows the fragment
  joining two neurons or leaving the neuron; 14 (4%) sit on a junction the tracing reads as a
  branch point or a spur. Mislocalized positives are rare.
- On 802449, 93% of the negatives are branch points with every arm on the same GT neuron, and
  on the rows where label and tracing agree the AUROC is unchanged (0.624). Its low figure is a
  feature problem: the scorer ranks verified true branch points (mean rank 0.51) too close to
  merge junctions (0.70).
- On 747807 and 750318 the negatives are less clean: 391 and 886 of them are `no_gt` (labeled
  0 inside the 150 um claim radius, with no tracing along any arm), and 156 and 57 are
  `leaves_gt`. Without those rows the AUROC rises by 0.08 and 0.10. Over all six brains 375
  negatives (`two_gt` + `leaves_gt`) show the same pattern as the positives, about as many as
  there are positives; the scorer already ranks them higher (0.59) than branch points (0.51).
- The 29 positives confirmed by two GT neurons are ranked no better (0.67) than the
  `leaves_gt` ones (0.70), so the low ranks are not explained by `leaves_gt` positives being
  untraced branches of the same neuron (29 rows only).

Limits: the categories are a second reading of the same tracing, not new truth; `leaves_gt` is a
merge only where the tracing is complete; a GT neuron passing within 6 um of an arm counts as
covering it; thresholds fixed, not tuned; one scorer of one run.

**The disagreeing negatives, read against the cache's site rule** (same script, second pass the
same day; it stores per row the quantities `canonical_labeling.geometric_merge_sites` depends
on and draws examples under
`notebooks/merge_junction_study/merge_junction_label_localization/examples/`). The walk
starts only at a fragment leaf more than 50 um from every GT node, stops at the first node
within 6 um of a GT node, drops the contact as a pass-through when fewer than 12 connected GT
nodes there carry the segment's label (11 on 750318), and records one site per walk.

- `no_gt` negatives (1,426 junction rows, 2,545 `tip_to_shaft` rows) are not on a tracing at
  all. Junction rows: 1,020 are 50-150 um (Euclidean) from the nearest GT node, and along the
  fragment 808 are more than 300 um of cable from the nearest node within 6 um of a GT node
  (201 at 150-300 um, 217 at 60-150 um, 200 within 60 um). Reading the tracing at 12 um instead
  of 6 um moves only 174 of them out of `no_gt` (37 to a branch or spur), so this is not a
  skeleton-to-tracing offset. 902 are on a segment in `gt_merge_labels`; 221 have a recorded
  site between them and the tracing. They are 0 only through the 150 um claim radius; the
  scorer ranks them like any branch point (0.53). 3,434 of the 3,971 rows are on 747807 and
  750318.
- `leaves_gt` negatives (353 junction rows): for 219 the untraced side arm never gets more than
  50 um from the tracing, so no walk starts (free region behind the arm: median 126 um of cable
  against 311 um for the positives); for 76 the tracing beside the junction does not carry the
  segment's label (the fragment runs next to a GT neuron); 43 have a site recorded 20-60 um of
  cable away, outside the 20 um label radius; 15 others. A `leaves_gt` positive and most
  `leaves_gt` negatives therefore differ by how far the untraced arm gets from the tracing, which
  is not visible inside the descriptor's 50 um.
- With GT read at 12 um, 117 of the 309 `leaves_gt` positives become `mixed`: the untraced arm
  often starts 6-12 um beside the traced neurite before it departs (a neighbour running
  alongside rather than a crossing).

| Junction AUROC of the accepted scorer, measured on other negative sets (positives and scorer unchanged) | junction / `tip_to_shaft` negatives kept | 802449 | 747807 | 750318 | 754610 | 754612 | 754613 |
|---|---|---|---|---|---|---|---|
| current labels | 11,181 / 15,799 | 0.625 | 0.738 | 0.661 | 0.844 | 0.743 | 0.783 |
| without `no_gt` negatives | 9,755 / 13,254 | 0.625 | 0.774 | 0.722 | 0.843 | 0.748 | 0.783 |
| without `no_gt` and `leaves_gt` negatives | 9,402 / 12,962 | 0.625 | 0.804 | 0.747 | 0.847 | 0.754 | 0.784 |
| negatives only within 20 um of cable of a contact whose GT carries the segment (12 nodes) | 9,469 / 11,079 | 0.625 | 0.773 | 0.719 | 0.840 | 0.748 | 0.784 |

**What the image shows at these candidates** (`examples` stage of the same script: eight random
junction candidates per brain and group, 132 in all, each with the image maximum projection of
a 160 x 160 x 48 um box, the fragments and the GT; all 18 sheets read by eye on 2026-11-11, one
view per candidate, so the counts are impressions, not annotations).

- `no_gt` negatives: `no_gt` means that no arm of the candidate follows a tracing, not that
  the box holds no GT. On most sheets the GT panel is empty or holds GT nodes of another label
  passing by; on 747807 and 754610 several boxes also hold a tracing that carries the
  candidate's segment and runs past 20-50 um from the candidate (747807 rows 111413, 612827;
  754610 rows 2010, 2011). Row 111413 of 747807 was checked node by node: the merge beside it
  is recorded (a site 52 um of cable away, with its own positive junction, row 111398); the
  tracing continues under another segment's label, and row 111413 is a further junction on the
  untraced part. The image shows a mixture: branch points and somata of untraced neurons that
  look ordinary, apparent
  crossings of two distinct neurites (several per sheet, e.g. 750318 row 189304, 802449 row
  42842), and, on 754613 and 754610, fibres of a bundle next to a traced one. A 0 is not
  supported for this group.
- `leaves_gt` negatives are heterogeneous: short twigs on a traced neurite (802449); arms lying
  in the diffuse glow beside a bright neurite with no neurite of their own (747807); and
  branches the GT does trace, under another label, with a further arm off the tracing (754610,
  754612). (A tracing that seemed to end in the first version of the sheets had only left the
  projected slab; GT nodes in front of and behind the slab are now drawn faintly.) Few look like
  a second neuron. This
  weakens the case for making them NaN.
- `leaves_gt` positives fall in four visual kinds: a crossing of two distinct neurites with the
  GT on one (roughly a third to a half, e.g. 754610 row 38114, 750318 row 51788, 802449 row
  8495); a dimmer neurite running alongside or branching off, visible only on a square-root grey
  scale (754612 rows 13909, 11135, 27913); a Y that looks like an ordinary branch point, with the
  GT on some arms only (about one in five, e.g. 802449 rows 16365, 29999); and an arm with no
  visible signal in the projected slab (754612 rows 61446, 61540).

The Y-shaped positives cannot be told from branch points by anything local; the crossings and
the dim or dark side arms can, in principle, by the continuity of opposite arms and by the
intensity along each arm.

The reasons for a missing site are read from the stored quantities, not from a rerun of the walk
(its result depends on the order the leaves are visited); the effect of refitting a scorer on
changed labels was not measured.

**How many off-tracing negatives are merges: a hidden-neuron estimate**
(`notebooks/merge_junction_study/merge_junction_hidden_neuron.py`, 2026-11-11, six TRAIN brains;
outputs in `notebooks/merge_junction_study/merge_junction_hidden_neuron/summary.md`). The tracing
cannot judge the off-tracing negatives themselves, but it can judge stand-ins: candidates that lie
on the tracing of exactly one GT neuron and that the current rule would still label 0 if that
neuron had not been traced (their segment also carries another GT neuron's label, and a node of
another GT neuron lies within 150 um). Their labels are known.

| Junction family | positives / rows on a tracing | rate | positives / stand-in rows | rate (Wilson 95%) | off-tracing negatives |
|---|---|---|---|---|---|
| 802449 | 98 / 6,801 | 1.4% | 0 / 66 | 0% (0-5.5) | 82 |
| 747807 | 113 / 780 | 14.5% | 0 / 25 | 0% (0-13.3) | 433 |
| 750318 | 114 / 520 | 21.9% | 4 / 43 | 9.3% (3.7-21.6) | 919 |
| 754610, 754612, 754613 | 45 / 1,877 | 2.4% | 0 / 30 | 0% (0-11.4) | 139 |
| all six | 370 / 9,978 | 3.7% | 4 / 164 | 2.4% (1.0-6.1) | 1,573 |

`tip_to_shaft`: 0 positives in 174 stand-in rows (0-2.2%), against 53 / 12,532 on a tracing.
("On a tracing": a GT node within 6 um of a fragment node within 20 um of cable of the
candidate, carrying the segment's label over at least 12 connected GT nodes; by this definition
1,573 junction negatives are off-tracing, against 1,426 `no_gt` rows by the arm reading.)

The stand-ins are positive no more often than candidates on a tracing, and less often on the
two brains that hold most of the off-tracing negatives. There is no sign that those negatives
hide many merges: at the pooled rate about 40 of the 1,573 would be merges (16 to 96 at the
interval's ends), against 370 labeled positives. Limits: 164 stand-in rows from 23 neurons and
51 segments, only 25 and 43 rows (5 neurons, 18 segments) on 747807 and 750318, so a rate up to
13% and 22% there is not excluded; a stand-in sits on a neuron somebody chose to trace, on a
segment that two traced neurons share; its label follows the same site convention.

**Candidates at a canonical merge site that are labeled 0**
(`notebooks/merge_junction_study/merge_canonical_site_labels.py`, 2026-11-11, six TRAIN brains;
outputs in
`notebooks/merge_junction_study/merge_canonical_site_labels/summary.md`). Positives come from the
sites recomputed on the
cache, not from the sites segmentation-skeleton-metrics reports (`metrics_out`, "canonical").
Per canonical site, the candidates within 20 um of fragment cable and their labels:

| TRAIN brain | canonical sites | a positive candidate | candidates, all labeled 0 | no candidate within 20 um | not on a fragment of the segment |
|---|---|---|---|---|---|
| 802449 | 157 | 94 | 32 | 26 | 5 |
| 747807 | 160 | 104 | 25 | 27 | 4 |
| 750318 | 162 | 122 | 23 | 13 | 4 |
| 754610, 754612, 754613 | 58 | 38 | 10 | 10 | 0 |
| all six | 537 | 358 (67%) | 90 (17%) | 76 (14%) | 13 (2%) |

- 92 candidates sit at a canonical site with the label 0 (none is unlabeled): 75 junctions
  against 370 junction positives (+20%; 802449 29 against 98) and 17 `tip_to_shaft` rows against
  53. Of the 75 junctions, 54 are `leaves_gt` or `two_gt` by the reading of the tracing above,
  so this accounts for 54 of the 375 negatives with the positives' pattern.
- For 62 of the 90 sites the cache has no site within 150 um of cable (31 of 32 on 802449);
  for 20 it has one within 50 um, so the merge is labeled on a neighbouring candidate. Why the
  cache's walk records nothing at the 62 is taken up in the next paragraph (at the snapped node
  the contact is below the pass-through threshold for 19, the nearest GT node farther than 6 um
  for 12).
- In the other direction, 42 of the 370 junction positives and 23 of the 53 `tip_to_shaft`
  positives have no canonical site within 20 um of cable.
- The accepted scorer ranks the 75 junctions between the classes: mean rank 0.59 and 20% in
  the top tenth, against 0.68 and 24% for the labeled positives and 0.49 and 9% for the other
  negatives.

| Junction family, accepted scorer (scorer unchanged) | 802449 | 747807 | 750318 | 754610 | 754612 |
|---|---|---|---|---|---|
| average precision, current labels | 0.025 | 0.207 | 0.137 | 0.100 | 0.069 |
| average precision, canonical sites added | 0.030 | 0.224 | 0.162 | 0.127 | 0.085 |
| Top-K hits, current labels | 4 / 98 | 28 / 113 | 19 / 114 | 0 / 14 | 2 / 22 |
| Top-K hits, canonical sites added | 5 / 127 | 34 / 132 | 25 / 134 | 2 / 18 | 3 / 25 |
| AUROC, current labels | 0.625 | 0.738 | 0.661 | 0.844 | 0.743 |
| AUROC, canonical sites added | 0.626 | 0.719 | 0.643 | 0.814 | 0.736 |

The missing positives are real and are the hardest negatives the scorer was fitted on, but
they do not explain the merge level: 802449 stays at 5 hits of 127. The effect of refitting
with the added positives was not measured.

**Why the cache misses them: the port of the metric differs from the metric in two places**
(`notebooks/merge_junction_study/merge_site_walk_replay.py`, 2026-11-11, six TRAIN brains; outputs in
`notebooks/merge_junction_study/merge_site_walk_replay/summary.md`). The cache's sites come from
`canonical_labeling.geometric_merge_sites` (agentic-neuron-proofreader), a port of
`MergeCountMetric` (segmentation-skeleton-metrics). Read side by side: (1) the metric works one
GT neuron at a time, with a KD-tree of that neuron alone: a walk starts at a leaf more than
50 um from that neuron and stops within 6 um of that neuron; the port uses one KD-tree of all GT
neurons in a single pass, so a leaf near any other traced neuron starts no walk and a walk
stops at whatever tracing it meets first. (2) The metric moves an accepted site to the nearest
branching node within 40 um of cable that lies within 16 um of the neuron
(`search_branching_node`); the port keeps the node where the walk stopped. Replayed on the
cache's own graphs (the replay with one tree for all neurons reproduces the cache's 480
geometric-walk sites and their match counts exactly):

| Canonical sites (537 on TRAIN) with a site of the list within | 20 um | 50 um | sites in the list | of them with a canonical site within 50 um |
|---|---|---|---|---|
| cache, all sources | 421 (78%) | 441 (82%) | 509 | 87% |
| cache, geometric walk only | 397 (74%) | 415 (77%) | 480 | 87% |
| replay, one GT neuron at a time | 456 (85%) | 477 (89%) | 545 | 88% |
| replay, one neuron at a time and moved to a branching node | 465 (87%) | 477 (89%) | 546 | 88% |

On 802449 (38 GT neurons) the share within 50 um goes from 69% to 85%; on 754613 the walk alone
finds 9 of 9 where the port's walk finds 3. Of the 90 canonical sites whose candidates are all
labeled 0, the replay with both changes puts a site within 20 um for 51 (22 of 32 on 802449);
of the 358 that already have a positive candidate it keeps 355. The other 39 are not explained;
the metric ran on its own fragment and GT graphs, and the order of leaves and neighbours decides
which contact a walk records.

Still open as an explanation: junction-local image evidence at full resolution, and which
existing descriptors separate verified branch points from merge junctions on 802449.

Limits of the measurement: one scorer of one run, TRAIN holdout scores only (the validation
breakdown by family was not computed), and 9-22 positives on the three small brains.

**To do** (the run finished on 2026-11-11; items 1 and 6 were carried out that day, items 2 to 5
are as recorded), cheapest first:

1. Per-family merge statistics: **done 2026-11-11, reporting only** (`by_family` per cell,
   `per_kind_by_family` per report, shown in TRAIN feedback; `docs/stage3_evolution.md`). The
   second judge table restricted to junction candidates was removed with the judge on
   2026-11-12: in the two runs of that day it moved in the same direction as the pooled table
   in all eight merge decisions. That is also the first evidence on the open question, whether
   the merge objective and the guard rail should be computed within the junction family: no
   decision would have differed. Left open; the reports still carry both.
2. Offline study of junction positives against junction negatives before another evolution
   run. Done 2026-11-11 for fragment totals and arm subtrees (no usable signal, above). Done the
   same day for label localization (labels are on the right junctions, above). Still to do:
   which existing descriptors separate the classes, and what the missed positives look like in
   the image, both first on 802449 (`gt_branch` negatives against `two_gt`/`leaves_gt`
   positives, the cleanest contrast). No evolution run is needed. Repeat the family breakdown
   on validation with `notebooks/acceptance_rule_replay.py rescore` (it saves the validation
   scores).
3. Longer-range fragment context for `describe(context)` (section 5): **not as a next step.**
   The size statistics tested on 2026-11-11 carry no consistent signal; revisit only with a
   specific quantity that the offline study shows to separate the classes.
4. `tip_to_shaft`: score the family separately, or leave it out of the merge statistics until a
   scorer finds any of its positives, so that it stops raising the pooled figure.
5. Label policy for negatives: **keep the current labels** (the rule is `near_gt_150` in
   `harness/label_availability.py`: segment on GT and a GT node within 150 um; changing it means
   a new definition there and new availability sidecars, not a cache or table rebuild). The
   off-tracing rows are 0 without a tracing at the candidate, but the hidden-neuron estimate
   above puts the share of merges among them at 2.4% (1.0-6.1), not above the rate on a
   tracing; the higher junction AUROC measured without them (+0.04 / +0.06 on 747807 / 750318)
   only shows that they are harder than the other negatives, and they are the only negatives on
   untraced neurites, which is where a scorer works when it is used. Reopen only if a larger or
   independent sample (750318 has 43 stand-in rows from 2 neurons) shows a rate well above the
   labeled positive rate. `leaves_gt` negatives and the 43 negatives 20-60 um from a site: leave.
6. Positive labels that follow the metric. **Port repaired 2026-11-11** (agentic-neuron-proofreader
   `geometric_merge_sites(walk="per_neuron")`; `labeled_cache` policy `length_v2`, default).
   Computed in memory on the six TRAIN caches with the package's functions
   (`notebooks/merge_junction_study/merge_walk_port_check.py`): sites 509 to 553; junction
   positives 370 to 426 (58 gained, 2 lost); `tip_to_shaft` positives 53 to 51 (9 gained, 11
   lost, the site now sits on the branching node); canonical sites with a positive candidate
   358 to 406 of 537, 41 still with candidates and none positive, 89 without a candidate or not
   on a fragment. **Relabeled 2026-11-11:** twelve caches (`--check-reproduce` passed on each),
   cache sites 1,181 to 1,309; raw pools and sidecars of the eleven development brains (every
   `pool_sha256` unchanged, split labels unchanged, merge positives 862 to 956); 802449 and
   794495 companions; twelve alignment receipts; eleven-brain raw preflight passed;
   `notebooks/merge_canonical_gap` rerun (cache labels hold 0.888 of the canonical sites within
   50 um, 0.813 before); `candidate_pools.json` evidence updated. Later runs are not comparable
   with earlier ones. **Added sites checked against the canonical sites**
   (`notebooks/merge_junction_study/merge_added_sites_check.py`, 2026-11-12, six TRAIN brains):
   of the 553 sites now in those caches 507 have an old site within 30 um and 46 are added; 42
   of the 46 (91%) have a canonical site within 50 um and 40 (87%) within 20 um, against 87% of
   the kept sites within 50 um; 3 old sites have no new site within 30 um (all three had a
   canonical site within 50 um). On the other six brains the old coordinates were not kept;
   from the `merge_canonical_gap` summaries before and after, sites rose by 84 and sites with
   a canonical site within 50 um by about 76. **In the image** (all 46 added and 3 dropped sites
   drawn with image, fragments, GT and the canonical sites,
   `merge_added_sites_check/examples/`, read by eye on 2026-11-12): the three dropped sites are
   the same events as three of the added ones, recorded more than 30 um away; on 750318 the
   site moved onto the canonical position (it was 29 um off), on 747807 two sites moved 35 and
   39 um away from it (order of the walks; the junction beside the canonical site lost its
   positive). Of the 43 new sites, 40 coincide with a canonical site, and most show two
   differently traced GT neurons meeting or crossing inside the segment, which is what the
   pooled walk could not find. The four without a canonical site within 50 um: 802449 #2 (two
   traced neurons meeting) and 747807 #9 (a dimmer neurite joining a traced one at a T) look
   like merges, 802449 #13 and #23 lie in cluttered regions and are not clear either way.
   **Label-dependent notebook results redone 2026-11-12** on the relabeled caches:
   `label_counts` and `label_availability_distribution.svg`; `merge_site_counts`
   (`mcl100_length_v2`); the site census (`merge_topology_diagnosis.py`,
   `merge_diagnosis_followup.py`: 269 bridge sites); `merge_shaft_reach` (declared rule 0.867
   of the cache sites within 20 um, 0.937 within 150 um); `merge_site_gt_tip` (60 of 157 blind
   sites at a GT tip); the blind-site figures; the pool sweep (gap=40 no longer reaches 0.90 on
   every brain: 789202 at 0.895); `verify_add_cache_metrics` (geometric merge counts within 15%
   of the canonical count on eleven brains, 789202 19 against 29); `errors_per_neuron`; the
   degree baseline of the merge feature specification (794495 companion: 173 positives, AP
   0.030). The census must run before the reach, the GT-tip and the blind-site scripts
   (`docs/dataset_artifact_dependencies.md`). **Not redone:** the outputs under
   `notebooks/merge_junction_study/` except `merge_added_sites_check`, which describe the
   `length_v1` labels; `notebooks/merge_label_shadow_findings.md` (historical). The seed
   baselines are measured by the two runs started on 2026-11-12.
   **Decision (user, 2026-11-11): the canonical sites are not added as a label source for now.**
   The labels stay computed from the cache alone (GT tracings, GT node labels and the fragment
   graph), with no dependence on the result files in `metrics_out`. Known cost, on the six
   TRAIN brains after `length_v2`: 41 of 537 canonical sites (8%) have candidates within 20 um
   that are all labeled 0, and 89 have no candidate or are not on a fragment of their segment.
   The 41 are not explained by the two repaired differences; the metric ran on its own
   fragment and GT graphs (other node spacing, no minimum cable length) and the order of leaves
   and neighbours decides which contact a walk records. Reopen if the residual matters for a
   result; `merge_canonical_site_labels.py` and `merge_walk_port_check.py` list the sites.

## 8. Promotion gate and validation feedback after the 2026-11-12 runs

Done on 2026-11-12 (`docs/stage3_evolution.md` section 3.2, `WORKFLOW_REVISION.md` entries of that
day): one promotion gate, the rule on the declared objective inside the guard rail
(`--promotion-gate rule`, default, `--precision-margin`), with the LLM judge, the `bootstrap` mode,
the head guard and the junction-only second table removed; three rejection words, with the
outcomes quoted in the prompt; the paired bootstrap over fragment segments instead of rows; and
the TRAIN selection-noise check (`--selection-noise-gate`). Evidence so far is an offline replay
on the two runs of that day (`notebooks/promotion_rule_replay.py`); no run has used them yet.

Open:

1. **First runs with the new configuration.** Check (a) how many generations end as
   `not_measured` and whether a later generation of the same kind then makes a larger change;
   (b) whether `rules.md` / `proposal.json` refer to the quoted outcomes and to
   `selection_gate.resampling`; (c) `gain_too_small` rejections and what followed them.
2. **The objective is now the promotion criterion.** The judge had applied "pAUC interval above
   zero"; the documented launch keeps average precision. For split the two differ in practice
   (two of ten split submissions that reached the table). Decide from the next runs whether split should run with
   `--objective pauc`; one flag, no code change. A run and its comparison runs need the same
   objective and gate.
3. **The margin, 0.002, is one number chosen from two runs**, for both kinds and any objective.
   Revisit with more submissions; a per-kind value would need a code change.
4. **The intervals resample fragments, not brains.** Fragment resampling (2026-11-12) widened
   the TRAIN intervals by 8% (merge) and 2% (split) and changed no verdict, so within-fragment
   dependence is not what limits the gate. Still open: the five validation brains are queried
   every generation without a correction for repeated tries, brains are not resampled, and the
   validation-side effect of the new unit was not measured (validation scores of submitted
   candidates are not stored; storing them per generation would make such replays cheap).
5. **The TRAIN check needs whole-brain holdouts.** With within-brain folds it reports
   `unavailable`; a fold-wise paired bootstrap would be needed for frozen / no-score runs that
   use them.
6. **Restoring the best TRAIN candidate no longer prefers one whose head held** (the head guard
   was removed). A candidate whose objective rose with a clearly worse top is now submitted and
   rejected by the guard rail on validation, at the cost of the generation. Count
   `significantly_worse` rejections in the next runs; if they rise, add a TRAIN-side check on the
   same statistics before submission.
7. **Selection protocols were left as they are.** The in-sample protocol is the path of the
   tiny-fixture tests and the within-brain folds serve the frozen / no-score arms (section 6);
   raw runs use whole-brain holdouts only. Removing either means deciding about those arms.
8. **Merge is not helped by any of this**: after the first accepted merge generations no merge
   candidate was beyond noise on TRAIN or on validation (section 7).
