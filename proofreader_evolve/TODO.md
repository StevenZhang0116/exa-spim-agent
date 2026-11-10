# proofreader_evolve TODO

Open items only. Each entry says what is open, why it matters and where it would be
changed; "checked" dates say when the code was last compared with the entry.

This file was reduced from 861 to its open items on 2026-11-09. Completed work, superseded
proposals and the measurement tables that older documents cite from here (the 2026-10-09
K sweep, the context-cache coverage tables, the 2026-10-29 three-arm state) are in the
change log of [WORKFLOW_REVISION.md](WORKFLOW_REVISION.md) and in the previous version of
this file: `git show efcd2be:proofreader_evolve/TODO.md`.

## Standing decisions

- 794495 is reserved for a future independent validation; no fitting, selection, tuning,
  testing or promotion on it. The split and launch recipe are in
  [Stage 3](../docs/stage3_evolution.md#32-prepare-tables-and-launch-a-run).
- Positive-weighted aggregation is shared by selection, ablation, validation and the
  bootstrap. No LLM judge takes part in selection or promotion.
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
with the head guard, whole-brain holdout selection, and three-valued labels. They answer a
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
  validation objective and with validation P@K; (c) head-guard vetoes, rejected generations
  whose P@K rose, and candidates whose three-depth mean held while P@K alone fell beyond the
  tolerance. Revert to `--objective precision_at_k` if validation P@K falls over a run or the
  guard vetoes most AUROC-accepted candidates. Not done as of 2026-11-09.
- **Plateau detection counts TRAIN novelty, not validation progress.** `stalled_rounds` in
  `harness/candidate_pool.py` advances on TRAIN selection gains (checked 2026-11-09), so a
  kind whose candidates are rejected inside the bootstrap interval keeps its budget. Stop
  allocating a kind after N consecutive such rejections and send the budget to the other kind
  or end the run; host-side only, no validation value shown to the agent. The same rule would
  cover the `tune` rounds, which accepted nothing in twelve generations of the 2026-10-27
  runs. Depends on section 1: a simplified inner loop removes most of this machinery.
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
