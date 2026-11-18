# Stage 3. Evolve: re-rank the frozen candidate pools

*Part of the [exa-spim-agent](../README.md) documentation.*

The evolution stage keeps the declared candidate pools fixed and learns their
ranking. Available predictors depend on the detector prior: raw mode withholds both
frozen scores and discovered feature columns. It is an LLM-driven search: a
fresh agent session is the mutation operator, a TRAIN-side selection score ranks
exploration branches, and a held-out validation gate alone decides promotion: no rank statistic
of the target kind may be worse beyond paired-bootstrap resampling noise, and the kind's objective
over its counted brains must be better beyond that noise by more than a minimum effect
(`--promotion-gate rule`, the default since 2026-11-12, section 3.2). The objective
is a rank statistic over the GT-labeled rows (`--objective`, since 2026-11-02 AUROC by
default; `pauc`, `average_precision` and the earlier `precision_at_k` are the alternatives,
and every statistic is recorded whatever the objective; since 2026-11-13
`--objective merge=NAME,split=NAME` gives each kind its own). Each pool ranks its own
K = ceil(ratio x its GT positives), limited to the kind's cap when it has one. Since
2026-10-27 no kind has a cap and both ratios are 1, so every pool ranks exactly its GT
positives and Precision@K equals Recall@K (R-precision) for merge and split alike, in
TRAIN, selection and validation (merge was 2 x positives under a cap of 300 from
2026-10-12 to 2026-10-27; split uncapped since 2026-10-26). Pools with fewer than 10 GT
positives are reported but excluded from the objective (`--k-positive-ratio`,
`--merge-k` / `--split-k`, `--min-positives`). The floor was 20 from 2026-10-12, chosen under
equal-brain aggregation; since 2026-11-08 it is 10, because positive weighting already keeps a
small pool from steering the objective and the floor only has to drop pools whose rank
statistics are degenerate. `cli/preflight.py` lists the pools below it (`positive_floor`).
Each active kind gets an initial generation before adaptive allocation returns to
a previously scheduled kind. Allocation then uses a rotation floor, pending follow-ups,
recent validation point gain and remaining headroom. Deferred follow-ups stay pending.
Performance stagnation uses cumulative selection-objective improvement since the last
progress checkpoint (0.0005 for AUROC; any positive gain for other objectives).
New positive coverage and improved slices still retain specialist candidates, but
do not alone reset performance stagnation. This tolerance is a scheduling heuristic,
not a statistical test or a promotion threshold.
Default candidate restoration takes the highest selection objective. Explicit candidate IDs
remain selectable.
The subsections below follow one generation of the loop in the figure; the
maintained design record is
[`proofreader_evolve/WORKFLOW_REVISION.md`](../proofreader_evolve/WORKFLOW_REVISION.md).

## 3.1 What is evolved and what stays fixed

`proofreader_evolve` improves **a rank objective (AUROC by default, Precision@K
reported alongside) over GT-labeled rows on the
same candidate pool** as the selected `merge_junction_detector.py` and
`split_site_detector.py`. The pool is the rule declared in `candidate_pools.json`,
enumerated by `candidate_rules.py` (split occurrences, junction NMS, tip-to-shaft
candidates); evolution cannot change it. Under `frozen` / `no-score` the detectors only score
its rows and exactly one detector per kind is required; under `raw` the same rows come from the
detector-free raw pools (section 3.2) and no detector is involved.

## 3.2 Prepare tables and launch a run

### Current dataset roles (user decisions, 2026-11-04 and 2026-11-14)

| Role | Brains |
|---|---|
| TRAIN and selection | 802449, 747807, 750318, 754610, 754612, 754613 |
| Whole-brain selection holdouts | All six TRAIN brains |
| Development validation / promotion | 789202, 794493, 794491, 751473, 841260 (added 2026-11-07) |
| Post-analysis only (since 2026-11-14) | **794495**: no role inside a run; the final policy of a finished run is applied to it once |

Image sources (2026-11-09): the raw `fused.zarr` of 754613 and 751473 were deleted from
`aind-open-data` on 2026-11-08 and those of 747807, 750318 and 754610 were found deleted on 2026-11-09;
these five brains now read the `fused-denoised.zarr` of the same prefix (same grid, denoised pixels),
with rewritten cache `img_path`, re-prepared raw pools and new alignment receipts. 754612 and 794493,
whose raw volumes were still present, were switched to their `fused-denoised.zarr` the same evening
(user decision) so that a later deletion cannot interrupt a run. The `fused.zarr` of 841260 (and of the
reserved 794495) is itself denoised (AIND denoise-net on the tiles before flat-field correction and
fusion). Three development brains still read unprocessed fusions, because no denoised volume on their
grid exists: 802449 (TRAIN), 789202 and 794491 (validation). TRAIN is therefore five denoised brains
and one raw one, validation three denoised and two raw, and image descriptors see two kinds of pixels
across brains (`cache/dataset_source_status_20261108.md`, `docs/dataset_artifact_dependencies.md`).

Segmentation sources (2026-11-05): 789202 and 802449 now resolve to their denoised Google
segmentations and 841260 is a new brain with GT. On 2026-11-07 the user placed 841260 in the
development validation set: 789202's merge pool fell to 15 positives with the denoised source, below
the floor of 20 then in force, so without 841260 the merge promotion gate would hold no denoised
brain, while TRAIN kept 395 countable merge positives (802449 104, 747807 125, 750318 144,
754612 22; 754610 with 19 and 754613 with 9 were below that floor) and its three whole-brain
holdout folds unchanged; 841260's 118 merge positives also balance the two sides (TRAIN 395 /
validation 424, the latter 794493 41, 794491 125, 751473 140, 841260 118; preflight label counts
of 2026-11-07). With the floor of 10 (2026-11-08) 754610 and 789202 count as well: TRAIN 414 and
validation 439 countable merge positives, 754613 (9) still excluded. Caches, raw pool tables,
companions and alignment receipts of the three brains were rebuilt on 2026-11-06/07 and the raw
preflight passes with these roles; the next run sets a new seed baseline, and runs before the
rebuild are not comparable on 802449 or 789202 (see [artifact dependencies](dataset_artifact_dependencies.md)).

**794495 has no role inside an evolution run.** Do not use it for training (including
auxiliary fitting), selection/OOF, tuning, ablations, or development validation and
promotion. From 2026-11-04 it was reserved for an independent validation whose protocol
was never designed; on 2026-11-14 the user decided to use it for post-analysis instead.
After a run has ended, its final policy is applied to 794495 once, when the user asks for
it, and the figures are recorded in
[proofreader_evolve/evaluation/README.md](../proofreader_evolve/evaluation/README.md)
(the command, the rules of that folder, the latest result and the history). Those figures
are not fed back into a run. They are not a first look either: two policies have been
scored on the brain so far. Its artifacts (cache, raw pool tables, descriptor bank) are
rebuilt only at the user's request. The [GT/image source comparison](../evaluation/README.md)
is a data audit and scores no model. 794495 was used in discovery and detector fitting,
and as a TRAIN brain of evolution runs in 2026-10; that stays part of the provenance. This
paragraph is the home of the statement; other documents give it in one sentence and link here.

The current raw setup uses `--selection-brains all --selection-holdout-brains all`.
The six whole-brain holdouts are assigned to the existing three folds in sorted
brain order: {747807, 754612}, {750318, 754613}, {754610, 802449}. There is no
auxiliary brain or within-brain selection fold. Each scored brain is absent from
its fold's fit; the final candidate is fitted on all six TRAIN brains. This changes
the development-validation set from earlier runs, so establish a new seed baseline
and do not compare aggregate values directly with the old five-brain validation.

**Legacy defaults are not the current dataset policy.** The parser still defaults
to `--train-brains 794495`, `--detector-prior frozen` and automatic validation discovery;
there is no general reserved-brain guard. Bare defaults fail the detector-naive
selection check, but raw or legacy `in_sample` can bypass that particular check.
Always provide explicit TRAIN and validation lists for new runs. Raw's generic
ability to select a detector-fitted brain does not authorize using 794495 inside a run.

### Prepare inputs only when needed

Use the `panda` environment on an allocated compute node. Existing matching tables
and availability sidecars can be reused; evolution does not prepare them itself.
Since 2026-11-06 raw runs load detector-free **raw pools**
(`proofreader_evolve/harness/raw_pool.py`, `cli/precompute_raw_pools.py`): the same candidate
rows, labels and `pool_sha256` as the detector-native tables, with the raw candidate
attributes as the only feature frame, so no frozen detector is resolved, hashed or run at
preparation or at run start (a cache load plus the enumeration per brain, minutes). The
detector-native tables and the frozen detectors are needed only by `--detector-prior frozen`
/ `no-score` (`prepare_feature_tables.py` without `--detector-prior raw`, hours per brain); those
tables were deleted on 2026-11-07, so such a run re-prepares them first.

```bash
EVOLUTION_BRAINS=(802449 747807 750318 754610 754612 754613 789202 794493 794491 751473 841260)
python -u proofreader_evolve/prepare_feature_tables.py \
  --brains "${EVOLUTION_BRAINS[@]}" --mcl 100 --threads all --detector-prior raw
# Preparation writes these sidecars too; this standalone command reuses valid ones.
python -u -m proofreader_evolve.cli.precompute_availability \
  --brains "${EVOLUTION_BRAINS[@]}" --mcl 100 --detector-prior raw
python -m proofreader_evolve.cli.preflight \
  --brains "${EVOLUTION_BRAINS[@]}" \
  --train-brains 802449 747807 750318 754610 754612 754613 \
  --selection-brains all --detector-prior raw --mcl 100 --min-positives 10
```

Image crops and fragment neighborhoods are read on demand; no context precompute
step is needed. Labels are 1 / 0 / NaN (not judgeable by GT). Supervised fitting and
metric ranking use labeled rows. See [artifact dependencies](dataset_artifact_dependencies.md)
for the rebuild order when a source cache changes. Pass explicit development-brain
lists to preparation scripts; their historical defaults may include 794495.

### Launch with a console log

Use the lightweight launcher from an allocated compute node, outside the evolution
step, with a separate allocation for each run. Replace `JOB_ID`, `NODE` and the CPU
count with granted resources. Configure `ANTHROPIC_API_KEY` in that environment.
This example requests five generations; the CLI default remains 15. The last two option
lines size the run to a dedicated 40-CPU / 600 GB allocation and must be adapted to the
allocation in use ("Give the run the allocation's resources" below).

```bash
mkdir -p proofreader_evolve/log
EVOLUTION_LOG="$PWD/proofreader_evolve/log/raw_5gen_$(date +%Y%m%d_%H%M%S).log.txt"
nohup /home/zihan.zhang/.conda/envs/panda/bin/python -u \
  -m proofreader_evolve.cli.launch_evolution \
  --job-id JOB_ID --node NODE --cpus 40 -- \
  --detector-prior raw \
  --train-brains 802449,747807,750318,754610,754612,754613 \
  --selection-brains all --selection-holdout-brains all \
  --validation-brains 789202,794493,794491,751473,841260 \
  --generations 5 --objective average_precision --precision-margin 0.002 \
  --selection-noise-gate \
  --aggregation positive-weighted --min-positives 10 \
  --classifier-memory-mb 98304 --classifier-time-budget 3600 --policy-time-budget 2400 \
  --classifier-threads 32 --classifier-search-seconds 7200 \
  --descriptor-workers auto --reviser-max-turns 80 \
  > "$EVOLUTION_LOG" 2>&1 &
printf 'Console log: %s\n' "$EVOLUTION_LOG"
tail -f "$EVOLUTION_LOG"
```

#### Give the run the allocation's resources

**The driver does not size itself to the allocation.** Its command-line defaults (8 numerical
threads, 16 GiB per fitting worker, 300 s per fit, 24 descriptor workers, 3600 s of classifier
search per generation, 60 reviser turns) are conservative values for a small shared job, and
every one of them is a hard limit inside the reviser's tools. A run launched with the defaults
on a large node does not fail visibly: a fit can be cut by the worker time or memory limit, the
classifier search stops between configurations once its budget is spent, a descriptor plan
can be refused or shrunk, and the agent adapts by proposing smaller models, fewer
configurations and cheaper descriptors. The ledger then records a restricted search as if it
were the agent's choice. Always pass the
resource options explicitly, sized to the allocation, and run **one evolution run per
allocation** (two runs in one job share its memory limit; the 2026-10-31 twin run was killed
that way).

Settings for a dedicated 40-CPU / 600 GB allocation (2026-11-09; used amounts are the maxima
of the 2026-11-05 and 2026-11-08 raw runs):

| Option | Default | Dedicated node | Why |
|---|---|---|---|
| `--classifier-threads` | 8 | 32 | Numerical threads of the fitting / prediction worker process; the worker clamps the value to the CPUs it sees. Leave a few CPUs for the driver. |
| `--classifier-memory-mb` | 16384 | 98304 | Address-space limit of fitting, fit-time extraction, diagnostics, 3D analysis and fixed-pool scoring; the value of the documented launch since 2026-10-27. |
| `--classifier-time-budget` | 300 | 3600 | Wall clock per fitting worker; the value of the documented launch since the six-TRAIN-brain partition. The worker's CPU-time limit is this value times its threads (since 2026-11-18; see "The CPU-time limit of a worker" under GPU mode). |
| `--policy-time-budget` | 300 | 2400 | Scoring / feature worker and 3D analysis timeout. |
| `--classifier-search-seconds` | 3600 | 7200 | Per generation; a 2026-11-05 generation used 3131 s of 3600. |
| `--descriptor-workers` | 24 | `auto` | CPUs of the step minus 2; concurrency stays gated by the job's memory headroom (16 GiB per worker, `--descriptor-memory-mb`), which lowers the risk of an out-of-memory kill but does not exclude it. |
| `--reviser-max-turns` | 60 | 80 | Used up to 50 of 60; costs no compute. |

Left at their defaults on purpose: the descriptor wall budgets (7200 s per call, 14400 s per
generation; at most 3021 s used), the image preview allowance (32; at most 4 used) and
`--train-evaluations-per-generation` (8; reached in two of fifteen generations), which bounds
how often the agent may look at TRAIN results and is a search-protocol choice, not a compute
limit. Image reads are credited with at most eight concurrent readers
(`IMAGE_IO_PARALLELISM` in `harness/descriptor_runs.py`, a constant, not a command-line option).

Scale the table with the allocation: threads to about 80% of the step's CPUs, classifier
memory to about a sixth of the job's memory limit as a rule of thumb (the descriptor workers
and the 80-90 GB driver need the rest), and pass `--cpus` of the launcher equal to the CPUs actually granted.
The nodes are shared: a job's CPUs are its own, but a node whose load average is far above its
core count (n267 on 2026-11-09: about 150 on 88 cores) slows memory- and disk-bound steps, so
the time budgets above are not generous there. After a run, compare `classifier_seconds_used`,
`descriptor_seconds_used`, `train_evaluations_used` and `reviser.num_turns` in `ledger.jsonl`
with their limits; a value at or near its limit means the generation may have been cut by that budget.

Ctrl+C stops `tail`, not the background run. The driver also writes `trajectory.txt`,
`trajectory.jsonl`, `run_status.json` and `report.html` in its own run directory.
Direct `python -m proofreader_evolve.cli.run_evolution` accepts the same arguments
after the launcher's `--`, but the launcher additionally reconciles Slurm exits.

The command above differs from the command-line defaults in the objective (average precision
instead of AUROC), the minimum effect (`--precision-margin 0.002` instead of 0) and the TRAIN
selection-noise check (off by default). The other defaults apply: grouped OOF selection,
positive-weighted aggregation, the rule gate (1000 bootstrap draws over fragment groups,
two-sided alpha 0.05), adaptive kind allocation and 8 evaluation units per generation. K equals
each pool's GT-positive count; there are no kind caps and pools below 10 positives
(`--min-positives`; 20 until 2026-11-08) are excluded from the objective. Metrics remain separate
for merge and split. No equal-brain default was introduced, and no LLM takes part in TRAIN-side
selection or in the promotion gate.

#### Per-family merge statistics (reporting only, 2026-11-11)

Every scored merge cell carries `by_family` and every report `per_kind_by_family`: per candidate
family (`junction`, `tip_to_shaft`) the labeled rows and positives, how many of its positives the
whole-pool Top-K holds, and, ranking the family alone, Top-K hits (K = its positives), average
precision and AUROC (`rank_statistics.family_statistics`, `aggregate_families`; counts summed over
the counted cells, rank statistics weighted by the family's positives). TRAIN feedback shows both
with a note; validation reports in the ledger carry them host-side. The objective and the
promotion gate are unchanged and use the whole pool. A cell whose labeled rows
span several folds carries no `by_family` (fold scores are never pooled).

#### The promotion gate (`--promotion-gate rule`, default since 2026-11-12)

What happens at a submission (`harness/promotion_gate.py`):

1. The host measures both scorers on the validation pools and computes, in one paired
   Poisson(1) bootstrap (`--bootstrap-draws`, `--bootstrap-alpha`, seed = generation), the delta
   and its interval for AUROC, pAUC, average precision, precision at K/2, K and 2K and their
   mean. Fragments are resampled, not rows (`fixed_pool_scoring.resampling_groups`): every
   labeled candidate of a merge segment, or of the anchor-side fragment of a split pair, shares
   one weight, because candidates on one fragment share its geometry, image and tracing.
2. Guard rail: if AUROC, pAUC, average precision, the depth mean or precision at K is
   significantly worse (interval entirely below zero), the candidate is rejected.
3. Rule: otherwise the candidate is promoted when the interval of the objective's gain lies
   above zero and the gain exceeds `--precision-margin`, a minimum effect (default 0; 0.002 in
   the documented launch). A failure to build the table rejects.
4. Feedback to the reviser. A rejected candidate's `outer_measurement` gains `rejection`, one of
   three words that the host derives from the intervals: `significantly_worse` (the guard rail),
   `within_noise` (the objective's interval does not lie above zero) and `gain_too_small` (it
   does, and the gain is not above the margin). The prompt of a generation quotes the kind's
   latest outcomes (`CandidatePool.outer_history`, also in `candidate_pool.json`). A later
   session sees no heldout number. The word is validation information flowing back into the
   search, at most 1.6 bits per rejected submission.

The objective decides, so what TRAIN selection and the reviser optimise is what promotes;
choosing `--objective pauc` instead of `average_precision` moves the criterion to the
low-false-positive range. The two kinds may differ (2026-11-13):
`--objective merge=average_precision,split=pauc` gives each kind its own objective, used for
that kind's TRAIN selection, feature ablation, TRAIN noise check, bootstrap, rule and scheduling
figures (`Budgets.objective_for(kind)`). Merge pools hold 1 to 9% positives and split pools about
40%, so one statistic need not suit both; in the runs of 2026-11-12 the average-precision and
the pAUC criterion disagreed on two of ten split submissions. The TRAIN objective and the
promotion criterion of a kind stay one and the same statistic. The manifest records
`objective_statistic` (`merge=...,split=...` when they differ) and `objective_statistics` per
kind, each ledger row the objective of its target kind. Kind scheduling compares the kinds'
recent gains and headroom in their own objectives' units, as it already compared two kinds whose
values lie in different ranges; no normalisation was added. The documented launch still uses
average precision for both kinds. The minimum effect exists because the split pools hold about 17,000
positives: their intervals are narrow enough that a gain of 0.0009 in average precision was
beyond noise, and a loss of 0.0005 in AUROC significantly worse.

The ledger's `validation_gate` carries `mode: "rule"`, the point gate (recorded, not binding)
and `gate_table`: the statistics with their intervals, `guard_rail`, `rule` (`statistic`,
`delta`, `low`, `margin`, `significant`, `large_enough`, `passed`), `status` (`rule`,
`guard_rail` or `unavailable`) and `rejection_category`. Since 2026-11-13 the run also keeps the
scores the table was computed from: `validation_scores/seed.npz` and one
`validation_scores/gen<NNN>.npz` per measured submission (the GT-labeled rows of the validation
pools, a few hundred kilobytes each, bound to the pools and labels by their digests;
`promotion_gate.save_scores` / `load_scores`). They are host-side files outside every path a
reviser session may read. With them `notebooks/promotion_rule_replay.py --objective ...`
recomputes the gate of a finished run under another objective, margin or resampling unit
without scoring a candidate again. Kind scheduling reads the objective's
point gain. `--promotion-gate margin` is the point-estimate gate without a table; the tests use
it.

History.

- 2026-10-05 to 2026-11-10, `--promotion-gate bootstrap`: the objective's interval above zero
  and, from 2026-11-02, a point-estimate head guard (the mean precision at K/2, K and 2K not more than 0.01 below
  the accepted scorer's). The rule and the guard rail cover both; the mode and the head guard
  (`--head-guard-statistic`, `--head-guard-tolerance`, also applied as a preference when the
  best TRAIN candidate was restored) were removed on 2026-11-12.
- 2026-11-10 to 2026-11-12, `--promotion-gate judge`: five votes of `claude-opus-5-5` on the
  table, numbers only, strict majority inside the guard rail. It was introduced because on merge
  AUROC is not sensitive to the head of the ranking (two runs of 2026-11-09 ended at merge AUROC
  0.881 and 0.879 with average precision 0.128 and 0.100), and an offline replay
  (`notebooks/acceptance_rule_replay.py`, 19 validated candidates) showed it deciding like an
  average-precision gate on 14 of 15 pairs and accepting 1 of 60 null pairs. In the two runs
  of 2026-11-12 (`precision_20261112_010152_bfg7kacp`, `precision_20261112_010306_tuap47i8`)
  it voted on fourteen submissions, thirteen times unanimously, and every outcome equalled
  "the pAUC interval lies above zero", a criterion declared nowhere: the objective was average
  precision. Two split candidates better beyond noise in average precision, AUROC and
  precision at K were rejected, and the category reported to the reviser said that the top
  had improved. A second table restricted to merge junctions, shown to the judge since
  2026-11-11, never changed a decision in eight merge submissions. The judge, its options
  (`--judge-votes`, `--judge-model`, `--judge-effort`), the second table and the two words
  that only its votes produced (`head_not_improved`, `not_convincing`) were removed on
  2026-11-12; ledgers of those runs keep the table and the ballots under `validation_gate.judge`.

Replay (`notebooks/promotion_rule_replay.py`; the eighteen submissions of the two runs that reached
the table, each decision with the run's own parent, on the recorded row-resampled intervals). With
margin 0 the rule decides like the judge on sixteen and accepts the two split candidates
(average-precision gains of +0.0018 and +0.0048). With margin 0.002 it also agrees on sixteen: it
accepts the second of those two, and rejects as `gain_too_small` the first and one split candidate
the judge accepted (+0.0009). The four guard-rail rejections are the same.

Fragment resampling (2026-11-12). On the TRAIN selection scores of the same twenty generations
the 80% interval of the gain is 1.08 times as wide (median; 0.90 to 1.32) for merge and 1.02
times (0.98 to 1.10) for split when fragments are resampled instead of rows, and no
beyond-noise verdict changes. The dependence inside a fragment is therefore small for the paired
gain of two scorers; the unit was changed because it is the correct one, not because it moves
decisions. The validation intervals of earlier runs were not recomputed (their candidates'
validation scores were not stored; runs since 2026-11-13 store them).

Limits. The five validation brains are queried every generation, so "beyond noise" is
conditional on those brains and is not corrected for repeated tries; brains are not resampled.
The margin is one number for both kinds and for whatever the objective is; 0.002 was chosen
from two runs. A run with this gate is not comparable with runs under the earlier gates.

#### TRAIN selection-noise check (`--selection-noise-gate`, 2026-11-12)

Why: the two runs submitted six merge candidates
whose selection gain over the accepted parent was between -0.006 and +0.003 in average
precision; none was promoted, and each measured one cost a frozen inference over five brains
and one more query of the validation pools.

What it does. For every measured candidate the host computes the paired bootstrap interval
of its selection-objective gain over the accepted parent on the selection brains
(`SelectionProtocol.paired_noise`, the same fragment bootstrap as the promotion gate, on the TRAIN
selection scores; `--bootstrap-draws`, `--selection-noise-alpha`, default 0.2 for an 80% interval, seed =
generation). The reviser sees it during the session as `selection_gate.resampling` (`delta_ci`,
`beyond_noise`). At submission, a candidate whose lower bound is not above zero is not sent to
the outer evaluator: the generation ends as `not_measured` (ledger `submission_within_noise`,
gate `skipped: "selection_within_noise"`), the parent stays, and later sessions read the TRAIN
interval in the prompt and in `outer_history`. The level is deliberately loose: the check is meant to
remove submissions that TRAIN itself cannot tell from no change, not to be a second promotion test.

The interval is defined where every selection pool is ranked as one pool, which is the case for
whole-brain holdouts (`--selection-holdout-brains all`, the raw-run setting) and for the
in-sample protocol. A pool split into within-brain folds is ranked fold by fold; there the
check reports `unavailable` and the submission is evaluated as before.

Replay on the same two runs (twenty generations, `notebooks/promotion_rule_replay.py`; the same
verdicts with rows or fragments resampled). At the 80%
level eight submissions would not have been measured: the two the driver already skipped as repeated
and unchanged, five the judge rejected and one the guard rail rejected (six merge, two split). All
seven promoted candidates lie beyond TRAIN noise; the closest has a lower bound of +0.00002 and is the
split candidate promoted on a validation gain of +0.0009. At the 95% level that candidate and one
more guard-rail rejection would not have been measured either. One interval of 1000 draws took 4 to
11 s on the six TRAIN brains.

Limits. Two runs; the candidates were produced without the check, and a reviser that sees the
interval may behave differently. A real gain smaller
than TRAIN can resolve is never measured, which matters most for merge (468 TRAIN positives).

For a controlled frozen/no-score/raw comparison, use the same explicit brain roles,
folds, objective, aggregation and budgets in all arms. With the current detector
provenance, all six listed TRAIN brains are detector-naive, so the same selection
lists are eligible in frozen and no-score modes. The old auxiliary-794495 examples
are historical and must not be reused for new runs.

The descriptor budgets default to 7200 s per `compute_descriptors` call (and per validation
brain's frozen computation at submission) and 14400 s per generation since 2026-11-03
(`--descriptor-wall-seconds`, `--descriptor-generation-wall-seconds`; 3600 / 5400 before).
These are limits, not a throughput guarantee; use the descriptor plan for the
actual crop, scope and allocated resources. Image-capable runs (an agent descriptor
that reads patches) still need an allocation whose memory headroom admits the requested descriptor
workers: a 600 GB job admitted the default 24 (2026-11-03), a 200 GB job holding the 80-90 GB driver admits one to three
(`proofreader_evolve/plans/on_demand_contexts.md`, 2026-11-03).

Inside those budgets one remote image read (store open, metadata, one patch) may wait
`image_context.READ_TIMEOUT_SECONDS` = 120 s (45 s until 2026-11-15), a patch read is tried twice,
and a descriptor batch that still fails with `TimeoutError` is run again up to
`descriptor_runs.BATCH_TIMEOUT_RETRIES` = 2 times while wall budget remains (trajectory event
`descriptor_batch_retry`, manifest field `batch_retries`). Errors of the descriptor program are
not retried. Before 2026-11-15 one stalled read among the thousands of a brain ended the whole
computation and with it the generation: generation 5 of run
`precision_20261114_133715_jkc36yxa` lost a split candidate that was beyond TRAIN noise while
its descriptors were computed on a validation brain. A failed generation now records the
chained traceback in its `generation_error` event (host-side `gen<NNN>/trajectory.jsonl`).

The launcher inherits the absolute panda interpreter, creates a unique directory,
and launches one `srun --overlap --exact --export=ALL` step. Overlap permits coexistence with an
interactive/Jupyter allocation's parent step; it is not a concurrency lock. The
explicit `--export=ALL` matters in Open OnDemand JupyterLab terminals, whose environment
carries `SLURM_EXPORT_ENV=NONE`: without it `srun` gives the step an empty environment
(no `PATH`, no `ANTHROPIC_API_KEY`), the driver still spends about 20 minutes on the
baseline evaluation and then fails every generation with `ANTHROPIC_API_KEY is not set`
(2026-11-02). `preflight --check-api` runs in the calling shell and cannot detect this. The
parent checks the child's return code and exact Slurm step after exit. Direct
`run_evolution` remains supported, but cannot finalize its own status after SIGKILL.
`--run-dir` accepts only a new directory (or one containing just `launch.json`).

Descriptor admission uses the minimum headroom across finite cgroup v2 ancestors,
without assuming file cache is immediately reclaimable. It reserves max(8 GiB,
10% of the limit) and a worker cap plus 512 MiB of host preparation allowance per
batch. Pending batches remain fully reserved, making admission conservative.
The first batch runs alone before further concurrency is admitted; the graph's
memory then appears in the job usage. Probes and frozen inference use the same
checks. The host graph load itself is not bounded by the worker's address-space
cap. Slurm execution fails explicitly if its cgroup limit cannot be read; outside
Slurm the fallback is host MemAvailable. Admission and input preparation consume
the original descriptor wall budget. Recorded numeric-bank manifests include
actual peak worker count and bounded memory-admission observations. This is risk
reduction, not a guarantee against arbitrary memory spikes. Lower concurrency may
require a longer wall budget; planning reports the observed capacity. Measured on
2026-11-01 on idle 200 GB allocations (n242, n221): the default 16384 MiB cap admits
10 concurrent batches, 8192 MiB admits 19 to 21 and 4096 MiB admits 24; loaded graphs
and classifier workers reduce this during a run.
`--validation-brains auto`
discovers other local MCL-matching `_add.pkl` caches and includes brains whose
merge and split tables pass the existing provenance/integrity checks. TRAIN and
detector-fitted brains are excluded. Missing tables are logged and skipped;
corrupt tables and empty pools fail rather than silently changing the benchmark.
At least one validation brain must be ready. The resolved split is fixed before
the seed evaluation and recorded in `dataset_split.json` and `manifest.json`.
To require an exact set, pass e.g. `--validation-brains 789202,794491,794493`
(TRAIN and validation brains must be disjoint); then any missing input is an error. Evolution never prepares tables itself.

Preparation now uses `dataset_cache_<brain>_mcl100_add.pkl`, preserving the native
detector's annotation definition. Full native pools replace the previous capped
tables; these different schemas cannot be reused interchangeably. Preparation
can still be expensive, especially merge extraction. It does not call an LLM.
Preparation mirrors its console output, including child-process stdout/stderr,
to `proofreader_evolve/log/prepare_feature_tables_<timestamp>_<pid>.log.txt`.
Each run gets a separate log with invocation, timing and exit status. Use
`--log-txt PATH` to append to a specific file; `--dry-run` creates no log.

Keep active console logs in `proofreader_evolve/log/`. Completed console and
test logs, together with their diagnostic JSON files, are retained in
`proofreader_evolve/log_archive/` for troubleshooting and linked verification
evidence; evolution does not read them as inputs. Archiving is manual: confirm
that no process is writing a file before moving it, preserve its relative path,
and update documentation references. New runs still write to `log/`.

Evolution and preflight default to `claude-opus-5-5` (Claude Opus 5.5).
Use `--model MODEL_ID` to override the model for a run. The Python SDK launches its own
bundled Claude Code CLI, not the `claude` on `PATH`; Opus 5.5 needs CLI 2.1.280 or newer
(`claude-agent-sdk` 0.2.165 bundles 2.1.294). Run `preflight --check-api` before a launch:
it sends one tool-free request through that SDK path, so an unsupported model or stale CLI
fails there instead of failing every generation.

## 3.3 Scorer contract and promotion gate

The editable scorer is seeded from `proofreader_evolve/artifacts/scorer.py`:
`score_candidates(features, ctx)` receives only predictor columns plus the frozen
`detector_score` and optional custom local geometry features (under `--detector-prior no-score` the
score column is withheld and the seed is the constant `artifacts/scorer_no_score.py`).
Under `raw`, detector-derived predictors are also withheld; the scorer receives
`raw_*` geometry and registered agent descriptors. `ctx` contains the candidate kind. It must return one finite
score per row. The harness selects exactly min(K, GT-labeled rows) with deterministic
tie breaking, where K is the pool's own budget: ceil(the kind's `--k-positive-ratio` x the
pool's GT positives), limited to the kind's cap when it has one (default: neither kind has a cap),
fixed for the run because it depends on labels only. GT labels are
stored separately and never passed to the scorer.
Training feedback includes labeled examples with TRAIN-only inspection handles; validation feedback is
not supplied to the reviser. Promotion uses the **target-kind validation objective**
(AUROC by default; Precision@K until 2026-11-02).
For each kind independently, `--aggregation positive-weighted` (default) weights rank
statistics by GT-positive counts over brains whose pool holds at least `--min-positives`
(default 10) GT positives, and computes precision as `sum(TP) / sum(effective K)`.
`equal-brain` assigns equal weight when explicitly selected. Smaller pools are
reported but excluded. The candidate must exceed the accepted parent's score for this kind
by more than `--precision-margin` (default 0; ties fail) and, with the default
`--promotion-gate rule`, the lower bound of the paired bootstrap interval of
that gain must be above zero and no guarded rank statistic may be significantly worse
(`--bootstrap-draws 1000`, `--bootstrap-alpha 0.05`, section 3.2);
`--promotion-gate margin` is the point-estimate rule alone. Individual brain/kind
regressions are allowed; TRAIN gain is diagnostic, not a promotion requirement.
Selection scores guide experiment ranking, branch retention and plateau scheduling.
Candidate selection, feature-ablation deltas, in-sample diagnostics and the outer gate
share the same per-kind aggregation and positive floor. Selection uses its own TRAIN
partition; the outer gate uses separate development-validation brains. Reports, plots
and logs show merge and split separately. Invalid, non-deterministic or
timed-out scorers are rejected without substituting parent measurements.

K is fixed per run. Both raw detector and evolved scorer are compared on identical
pools and budgets. Returning fewer candidates cannot improve the metric. No graph
edits, pool expansion, on-demand detector queries or post-cut re-enumeration occur.
Labels are three-valued (1 / 0 / NaN = not judgeable by GT), so precision describes
candidates near traced neurons, not exhaustive biological truth; a
better ranking does not establish safe graph repairs. Repeated validation is
development data, not a final untouched test. Scorer execution uses a separate
feature-only Linux worker with a seccomp syscall filter, wall-time/CPU/memory
limits, and no inherited credentials. File access, network access and child
process creation are denied before candidate code executes. Linux with
`libseccomp.so.2` is required; unavailable isolation fails the evaluation.
See [workflow revision](../proofreader_evolve/WORKFLOW_REVISION.md).

Raw runs resolve no detector since 2026-11-06 (raw pools, section 3.2); the rest of this
paragraph applies to `frozen` and `no-score` runs only.
The default frozen detectors (`DEFAULT_MERGE_DIR` / `DEFAULT_SPLIT_DIR` in
`proofreader_evolve/cli/precompute_error_scores.py`) are the 2026-10 builds since 2026-10-23
([Stage 2](stage2_detector_build.md#24-detectors-currently-in-use)):
`autodiscovery-application/new-merge-1_2026-10-19/runs/794495-with-figures` and
`autodiscovery-application/new-split-1_2026-10-20/runs/794495-with-figures`, both fitted on
794495. `--merge-dir` / `--split-dir` override them (a deliverable directory, or one of its
`runs/<brain>-<purpose>/` run directories, which the resolver binds to the script through
the run's recorded detector SHA-256). Both default run directories were deleted with
`autodiscovery-application/` on 2026-11-06 and restored on 2026-11-08 from the filesystem snapshot
`.snapshot/mindscope_daily_2026-11-05_19:00` (trees identical, detector script SHA-256 equal to the
recorded provenance); the real-detector parity test in `tests/test_native_pool.py` skips only while
they are absent. A `frozen` / `no-score` run still has to re-prepare its detector-native tables
first (deleted on 2026-11-07; `prepare_feature_tables.py` without `--detector-prior raw`).
The detectors' image pass is threaded (2026-10-21): `--image-workers N` on
`precompute_error_scores.py`, forwarded by `prepare_feature_tables.py` (0 = auto capped at 8,
-1 = all CPUs; the pass waits on S3, so N may exceed the CPU count), reaches every detector
whose extraction entry accepts `image_workers`; older detectors run as before. The pass
dominates the table build (about 19 split and 32 merge rows per second with 16 workers);
`proofreader_evolve/rebuild_brains.sh` runs the rebuild on the interactive node one brain at a
time, evolution brains first, and resumes where it stopped (rebuild plan, section 4).
Detector-fitted brains cannot be validation brains. The reviser requires
`ANTHROPIC_API_KEY`; preparation and baseline-only evaluation do not call an LLM.

## 3.4 One generation: the reviser session

Each generation allows up to 60 SDK interaction turns by default;
use `--reviser-max-turns` to change this independently of
`--generations`. The reviser reads a compact TRAIN feedback file capped at 128 KB.
Tool results may be up to
100,000 tokens; a result the SDK parks on disk (above about 50 KB) stays readable
through the session's file guard, and file reads allow 100,000 tokens instead of
the SDK default of 25,000. A parked result reaches the session only as a 2 KB preview
unless it reads the file, and in the runs of 2026-11-12 the sessions read it for 38 of
105 parked training replies (`notebooks/feedback_use_audit.py`). Since 2026-11-13 the
replies of the training, restore, descriptor and memory tools therefore stay inside
30,000 characters (`harness/tool_replies.py`): they carry the conclusions (objective,
gates with the resampling interval, the per-brain comparison, per-kind and per-family
statistics, ranking changes, one digest per descriptor column) and name the file that
holds the detail (an attempt's `feedback.json` with the example rows and per-brain
cells, a descriptor run's or probe's `result.json`). A reply above the budget is
recorded in the trajectory as `tool_reply_over_budget`. The session may also read its own generation's
`experiments/`, `classifier_fits/`, TRAIN `descriptor_runs/` and `descriptor_probes/`
directories, including the corresponding trees in earlier generations. PNG previews
under current/earlier `image_inspections/` are also readable; adjacent host audit JSON
is not. Inline previews have a static per-image base64 limit of min(2 MiB, 64 MiB /
configured preview count), plus a 3000-pixel edge limit. Oversized files stay unchanged
on disk and report `preview_inlined: false`. There is no cumulative byte budget;
reading a saved image does not bypass provider limits. The SDK buffer remains 256 MiB.

The system prompt keeps stable rules, while the task prompt keeps actual research
assignments, completion conditions, measured-submission requirements and essential
budget numbers. Interfaces live in guides and resource values in `model_environment.json`.
`rules.md` starts as brief research notes. Only `thinking_tokens` SDK status noise is
dropped; final usage/cost and other events remain in the normal result and trajectory.
The next official run's first generation should be checked for guide use and research
completion; synthetic tests cannot establish the behavioral effect of prompt changes.

Feature names
appear once per cell with aligned example vectors; values are rounded to six
significant digits for display. It samples selected positives and label-0 rows,
rows just below K, and missed positives. Each group keeps a fixed anchor and
reproducibly rotates other examples across generations, up to four per group
(fewer when needed to fit). `feature_statistics.json` provides whole-TRAIN
feature quartiles by label (1 / 0; NaN rows counted), so error samples are not mistaken for the
full distribution. Candidate feedback adds gained/lost positives and Top-K overlap.

Each generation edits one component. By default the kind is chosen adaptively:
one initial generation per active kind, then a floor of one revision per
`--kind-floor-every` generations (default 4) for each active kind,
then a pending promotion follow-up (deferred intact during initialization or a floor allocation), then the kind with the larger recent
validation point gain (accepted or rejected; only gate-passing gains before 2026-11-01), then the larger remaining headroom on the validation
cells. Headroom uses the configured per-kind objective and aggregation. Saturation
uses half a one-hit precision increment for Precision@K; for AUROC/AP/pAUC the
implementation uses a conservative positive-negative-pair resolution heuristic
(`0.5 / sum(positives * negatives)`), not the Precision@K hit increment.
`--kind-schedule alternate` restores strict merge/split rotation and
`--target-kind split` or `merge` focuses the search. The harness freezes the other
component. The agent writes its hypothesis, strategy, formula family and numeric
grid to `proposal.json`, then calls `evaluate_train({})` or `search_parameters({})`.
These tools have no text arguments. The default budget is **8 evaluation units**
per generation, configurable with `--train-evaluations-per-generation`; cache hits
and proposal-format errors cost no scoring budget. New configurations and optional
paired feature ablations share these units; an ablation costs two units, one per
arm including its fold fits. Tool calls still consume SDK
turns. `search_memory` searches earlier TRAIN experiments, including rejected and
failed attempts. `restore_candidate` accepts an experiment ID, `parent`, or `best`,
and restores measured code without another scorer execution.

The scheduler separates `explore` (design a formula, features or model program)
from `tune` (change numeric values only). Formula scorers expose a literal
`PARAMS = {'weight': 1.0}` dictionary and
reference its values. `search_parameters` enumerates the proposed grid while
keeping the formula fixed, snapshots each configuration up to the first execution
error of the formula (`stopped_early` lists the skipped, uncharged configurations)
and restores the highest selection objective; explicit IDs remain selectable. In tune
generations an AST fingerprint enforces the assigned formula. Grid sizes must fit
the remaining budget; oversized grids fail before scoring.

## 3.5 Agent-authored models and the selection protocol

The agent can also author **its own training and prediction program** in
`training.py`, defining `fit(X_train, y_train, artifact_dir, params)` and
`predict(X, artifact_dir, params)`. There is no model whitelist or required
NumPy export. The `install_packages` tool can add dependencies to the current Python
environment (normally panda), preserving existing package versions. It records pip output,
refreshes the installed-package inventory and consumes no evaluation units. The package
list is environment information, not a model whitelist. Within CPU and resource limits, the agent
chooses preprocessing, features, model architecture, loss, optimizer, ensembles,
training schedule and file format. `proposal.json.classifier.parameters` supplies
arbitrary JSON settings. `train_classifier({})` runs the program on **TRAIN only**,
saves its artifacts and measures its ranking. See the
[training guide](../proofreader_evolve/artifacts/classifier_guide.md) and
[editable example](../proofreader_evolve/artifacts/training.py).

An optional `select_columns(columns, params)` function selects the inputs passed to
fit, including unlabeled rows. Its returned column list is frozen for prediction
and descriptor dependencies, so excluded descriptor groups need no validation computation.
Without it, models retain all supplied columns as dependencies.

Agent code and model deserialization run after Linux Landlock + seccomp isolation.
Installed libraries and supplied inputs are readable; fit can write model files
and scratch space. Prediction receives predictors, declared raw image patches
and read-only model artifacts, without labels. Network, child processes, credentials and unrelated data paths
are unavailable. Unsupported isolation fails closed. The framework cannot prove
that arbitrary prediction code implements frozen preprocessing correctly; that
remains part of the model interface contract.

One new fit plus TRAIN evaluation consumes one shared budget unit; failed fits
also consume one with no metric. Exact repeats reuse the fitted model only when training source, parameters
and current TRAIN feature/descriptor identities match. Explore generations can rewrite the whole program; tune
fixes its AST and nonnumeric settings while refitting numeric parameters.
Defaults are 300 seconds wall/CPU, 16384 MiB (8192 before 2026-10-08) and eight numerical-library threads
(`--classifier-time-budget`, `--classifier-memory-mb`, `--classifier-threads`; one thread before 2026-10-27);
the recorded full runs use 1800 to 3600 seconds, 4 to 16 threads and 98304 MiB.
`--classifier-search-seconds 3600` additionally limits new classifier search within each
generation, including preparation, OOF/full fitting and evaluation. It is checked between
configurations; the current candidate completes under worker limits. Exhaustion stops further
uncached fits, while completed candidates can still be restored and cached fits reused.
The total can exceed the search limit by the current candidate's duration; evaluation
units and individual worker time/memory limits remain separate constraints.
New descriptor registrations invalidate fit/frame reuse for the affected kind, even when
the same training program and parameters are submitted again.
Artifact storage permits 128 MiB / 256 regular files per model. The agent chooses
and records randomness seeds. `model_environment.json` lists installed packages
and budgets; training stdout/stderr is saved (a worker that prints nothing leaves no file) and its last 8 KB returned to the agent.

Branches are ranked by the selection protocol: under the default `grouped_oof`,
the grouped out-of-fold objective (AUROC by default, measured per fold and weighted by
positives; Precision@K at a partitioned K when that is the objective) on detector-naive
selection brains from host-owned fold refits, with auxiliary TRAIN brains as fitting-only
data; in-sample TRAIN precision is a **resubstitution diagnostic**. Under
`--detector-prior raw` nothing is detector-derived, so `--selection-brains all
--selection-holdout-brains all` makes every TRAIN brain a whole-brain holdout (three folds
of two brains for the six-TRAIN setup in section 3.2): one cross-brain out-of-fold score, no auxiliary role. The agent may still use
TRAIN-only internal validation/CV during fitting. Final promotion requires improvement in
the target-kind development-validation objective that clears the promotion gate (section 3.2). The outer evaluator calls
frozen feature extraction and predict on validation; it never supplies validation labels to a worker.
`best_scorer.py` binds training source, parameters, provenance and artifact hashes.
Keep its sibling `model_artifacts/` directory when moving/resuming a run; use
`--start-from-artifacts` for another artifact location. Model dependencies are
needed at inference time. Resuming a managed model fitted on a requested
validation brain is rejected before seed evaluation. Legacy NumPy exports remain
readable; newly trained models use the program/artifact interface.
`--start-from` starts a new run with that scorer as its seed. It does not reload the
old ledger, candidate pool, scheduling counters or cumulative-progress checkpoint;
initial exploration and progress accounting restart for the new run.

### GPU mode (`--gpu`, 2026-11-17)

Off by default; without the option nothing below applies and a run behaves as before. With
`--gpu` the fit and predict workers of agent models may use the GPU the scheduler allocated to
the step. Whether a model uses it is the program's choice: the harness prescribes no
architecture, loss or device. The driver fails at start when no GPU is visible
(`CUDA_VISIBLE_DEVICES` empty or `nvidia-smi` lists nothing), and then, before any brain is
loaded, runs one small CUDA computation in a real sandboxed fit worker with the run's memory and
thread limits (`model_execution.gpu_worker_check`, about 25 s on n01); a worker that cannot use
CUDA ends the run there with the worker's log in the message. The device name and memory the
worker saw go to the manifest and to each generation's `model_environment.json` (`gpu`,
`sandbox_checked`). Descriptor, feature
extraction and 3D-analysis workers stay on the CPU: dozens of them run side by side.

One switch covers every model worker (`model_execution.run_worker`): fold fits and fold
predictions of the selection protocol (they run one after another), the full fit, ablations,
the outer evaluator's predictions and restored candidates.

**What the sandbox gives up.** The model sandbox (`harness/model_sandbox.py`, Landlock plus
seccomp) is applied as before, with these additions for a GPU worker, each found by tracing a
CUDA start-up inside the sandbox on an L40S node:

- `CUDA_VISIBLE_DEVICES` is passed through as the scheduler set it (it is blanked otherwise);
- read-write access with device ioctls to the `/dev/nvidia*` device files, read access to the
  driver's information files, and write access to `/proc/self/task` (the driver names its
  threads there and aborts when it cannot);
- the system calls `ioctl`, `eventfd2`, `setsockopt`, `bind`, `listen`, and `socket` for local
  (AF_UNIX) sockets only. `connect` always fails, so no socket of a worker reaches anything.

Unchanged: no network, no child processes, the filesystem allowlist, the address-space, CPU-time
and file-size limits, the artifact limits (128 MiB, 256 files). Weaker than the CPU sandbox:
agent-written code can call the NVIDIA driver's ioctl interface, a large kernel surface the CPU
sandbox does not expose, and can create local sockets (which cannot connect). Which GPUs a
worker can open is still decided by the scheduler's device cgroup. Use the option only on an
allocation where that is acceptable.

**What a model must respect.** The harness calls `predict` twice on the frozen model and rejects
scores that are not identical; that check is unchanged, so GPU inference has to be deterministic
(eval mode, `torch.backends.cudnn.deterministic = True`, `cudnn.benchmark = False`, a fixed batch
order). No child processes means `DataLoader(num_workers=0)`. A program should fall back to the
CPU when no GPU is present, so that a frozen model can still be applied elsewhere; scores of the
two devices are not guaranteed to be bit-identical, so compare a policy only with itself on the
device it was measured on. Retraining a network is not expected to reproduce its weights; the
frozen model is what the gate measures.

**Patches for a network.** `LOCAL_IMAGE` with `raw_patches: True` hands fit and predict the
patches of the selected rows. The selection `selection_mode: 'judgeable'` (2026-11-17) selects
every GT-labeled row of the pool, the rows fit and predict receive; the host chooses them, as
for the descriptor scope of that name, and no label value is read. Missing patches are read by
eight concurrent readers into the run's patch cache (budget `--image-cache-gb`, default 20). A
worker call still receives a copy of exactly its own rows' files, so a fold fit cannot open the
held-out rows' patches; the size of that copy is limited by `--image-transport-gb` (default
1 GiB, the limit that was fixed before).

**Where the patch cache lives (`--image-cache-dir`).** By default `image_cache/` in the run
directory, on the network file system. `--image-cache-dir local` puts it in the step's temporary
directory on the node's local disk (`image_cache_<run name>/`; the scheduler removes it with the
job), and a path names another parent directory. The driver checks at start that the directory
is writable and has at least the cache budget free. The cache holds nothing a run needs later: a
missing patch is read again. Models, logs and results stay in the run directory, and the
manifest records the location (`candidate_images.cache_directory`). Worker
directories were on local disk already (the step's `TMPDIR`). Use `local` for a GPU run: tens of
thousands of small files are slow on the network file system.

Measured on 2026-11-17 on n01 against the real store (TRAIN brain 754612, the 2,348 GT-labeled
merge rows, level 1, radius 16 um, 24 KiB per row, eight readers, one process per measurement,
two passes; log `proofreader_evolve/log/image_cache_timing_20261117.log.txt`):

| step | cache on the network file system | cache on local disk |
|---|---|---|
| cold read into the cache | 138 s / 122 s | 52 s / 40 s |
| the same rows found in the cache | 28 s / 39 s | 2.1 s / 2.1 s |
| copy to a worker directory | 29 s / 17.5 s | 0.3 s / 0.3 s |

Radius 32 um (32 x 43 x 43 voxels, 174 KiB per row, 0.39 GiB) was 20 s cold on local disk in an
earlier measurement. Split pools and larger crops were not measured.

**Limits are enforced while reading (2026-11-17).** Patch bytes are known only after a read. Each
reader adds its patch to a shared count as it arrives, and no reader starts another read once
the request is over `max_total_mb` or the transport limit, or a read fails or times out; at most
the reads already under way (eight) follow. Before, every selected row was read and the limit
checked afterwards. (A first form of this left the count to the thread that collects the
results in input order; one slow read at the front then let all other rows be read. A review
found it, and a test with a held first read covers it.) `stage_images` sums the files of a worker call before
it copies any. The cache's budget check, the move into the cache and the byte count happen under
one lock and count the stored file (`CandidateImageStore._commit`), so concurrent readers cannot
pass the budget.

**What tells a session about it.** A two-generation run of 2026-11-17 with `--gpu` and no other
change kept to hand-written descriptors and tree models in both generations and never touched the
GPU: the guides said that a GPU exists, nothing named a way to use it. Since then a run with
`--gpu` adds one paragraph to the session prompt (`gpu_prompt_text` in the driver): a GPU is
available, a model may learn directly from the patches of every GT-labeled row, the image guide
has a working example, and it is an option, not a requirement. `image_context_guide.md`
("Learn directly from patches") prints a complete small 3D network: per-patch normalisation,
training, saving, deterministic prediction, CPU fallback. `tests/test_gpu_mode.py` extracts that
program from the guide and runs it through the real workers, so what the guide prints is what
was tested. Without `--gpu` the prompt is unchanged.

**The whole path on the real pools (2026-11-17).** The guide's example was run in place of an
agent through the real driver on n01: six TRAIN and five development-validation brains, one
generation per kind, no language model. Merge: 27,403 labeled rows, 477 positives; the network's
out-of-fold average precision was 0.0278 against 0.0260 for the constant seed, inside TRAIN noise,
so the generation ended `not_measured`. Split: 39,984 rows; out-of-fold 0.6528 against 0.4444,
measured on the five validation brains and promoted (0.4259 to 0.6293; the raw heuristics give
0.8410). No prediction was rejected by the duplicate-prediction check. A sampler on the node saw
fit workers holding the GPU (up to 730 MiB, 93% utilisation) and predict workers (598 MiB, 51%).
The example is an interface demonstration; these figures say nothing about what a network on
patches can reach.

Where the time went in that run, with the cache in the run directory on the network file
system (merge generation, 64 minutes of worker time): 21 min to read the patches into the cache,
10 min for the four trainings, 4 min for fold predictions, and 28 min for the evaluation steps
(preflight, TRAIN scoring, diagnostics). The split generation took 2 h 15 min, of which 45 min
were the validation brains.

**The same merge generation with the cache on local disk (2026-11-17, evening).** Same program,
brains, rows and options plus `--image-cache-dir local` (log
`proofreader_evolve/log/gpu_fixed_network_full_merge_20261117.log.txt`); the out-of-fold average
precision was 0.0278 again and the generation again ended `not_measured`.

| step | network file system | local disk |
|---|---|---|
| read the patches of 27,403 rows | 21 min | 11.3 min |
| four trainings (three folds, full fit) | 10 min | 2.7 min |
| fold predictions | 4 min | 2.4 min |
| evaluation steps (preflight 5.7, TRAIN scoring 6.3, diagnostics 4.4 min) | 28 min | 16.5 min |
| the evaluation of the candidate in total | 64 min | 33 min |

About half. What remains is no longer the copying (0.3 s per brain in the table above): the
patch read is the remote store, and each of the three evaluation steps starts one frozen-model
worker per TRAIN brain (eighteen in all), which imports torch, initialises CUDA, decodes every
patch file and predicts twice for the duplicate check. A smaller check of the whole path ran
before it (merge, TRAIN 747807/754610/754612, validation 794493, no TRAIN noise check, 20
minutes in all, log `proofreader_evolve/log/gpu_fixed_network_20261117.log.txt`): start-up check
passed, cache on local disk, three fold fits, full fit, the frozen network scored on the
validation brain and rejected by the rule (0.0067 to 0.0064), no duplicate-prediction rejection.

**The first agent run (2026-11-18).** `precision_20261118_000444_8gpg1yd6` on n01: four
generations, the documented launch plus the GPU options below with the budgets of that day
(3600 s per fit, 14400 s of search per generation), 9 h 25 min, log
`proofreader_evolve/log/gpu_4gen_20261118_000443.log.txt`. The question was whether a session
takes the patch path on its own.

| generation | kind | what the session did | outcome |
|---|---|---|---|
| 1 | merge | descriptors and a tree model; read the image guide, did not use patches | promoted, validation average precision 0.0164 to 0.0816 |
| 2 | split | descriptors and a tree model | promoted, 0.4259 to 0.8610 |
| 3 | merge | a 3D CNN on the patches of all 27,403 labeled rows (24 um at level 1, 48 um at level 2), its output stacked into the tree model; then tabular variants | out-of-fold 0.1051 and 0.1057 against the parent's 0.1060; best of the generation 0.1086, inside TRAIN noise, `not_measured` |
| 4 | split | a 3D CNN (20 um, level 1, image plus a mask of the line between the endpoints) stacked into the tree model | out-of-fold 0.8826 with seed 0, 0.8799 with seed 1, parent 0.8809; validation +0.0008, interval includes zero, rejected |

So the prompt paragraph and the guide's example were enough: from the first follow-up generation
on, both sessions chose the patch network without further prompting, and neither found a reliable
gain over descriptors at this network size. The merge session reported that the CNN alone reached
0.57 to 0.58 AUROC on brains it was not trained on; the split session reran its candidate with a
second seed and judged the gain seed-dependent itself. No sandbox failure and no
duplicate-prediction rejection. A CNN attempt took 45 to 55 minutes, about 20 of them the
evaluation steps; generation 4 took 3 h 34 min. GPU use was not sampled in this run: it is
inferred from the programs and the sessions' reports (since 2026-11-18 a GPU worker records its
peak GPU memory, `gpu_peak_mb` in `status.json` and in the fit record; 0 means the program never
touched the GPU).

Three attempts of generation 4 failed and cost an evaluation unit each: one `predict` allocated a
single array for all 2,211,738 rows of a pool (228 GiB, `MemoryError`; the guide now says that
`predict` receives the whole pool and that only selected rows carry a patch), and two fits with
larger crops (28 and 32 um) were killed, which the message and the session attributed to memory.

**The CPU-time limit of a worker (2026-11-18).** Those two kills were the worker's CPU-time
limit. It was set to the timeout plus one second, but CPU time adds up over threads while the
timeout is wall clock: with `--classifier-threads 32` a worker whose threads were all busy was
killed after about 113 s of its 3600 s. Reproduced with the real worker and no data on n243: one
thread, timeout 60 s, 30 s of work finished; 32 threads, the same program, killed after 8 s;
32 threads, timeout 3600 s, 150 s of work, killed after 119 s with the run's message. In the run
the two fold workers were killed 137 s and 128 s after their start; their arrays were 5 to 9 GB
against a limit of 98 GB, and exceeding the address-space limit raises `MemoryError` (as the
228 GiB case did), it does not kill. The limit is now the timeout times the worker's threads
(`classifier_train_worker.py`; all modes of that worker), with the launcher's wall-clock watchdog
enforcing the timeout as before, and the kill message names both limits and what each does. The
scorer worker (`scorer_worker.py`) runs single-threaded, so its limit was right and is unchanged.
The same message appears once in run jkc36yxa and three times in byxx989h, each two to seven
minutes into a fit of a large tree ensemble; that fits this cause, but those four were not
examined one by one.

Known limits, to be revisited after the first runs:

- every fit and predict call copies its rows' patch files to the worker's directory; with the
  cache on local disk this was 0.3 s for 2,348 small patches, so access without a copy was not
  built. Revisit it if large crops make the copy visible again;
- patch bytes are known only after reading: a request above `max_total_mb` or the transport
  limit now stops early, and the agent still has to shrink it;
- an evaluation starts many short frozen-model workers, each paying the torch and CUDA start-up
  (about 25 s) and a full decode of its patches; this is the larger part of what remains;
- the read-only path list of the GPU sandbox was not minimised;
- no tolerance exists for inference that is not bit-reproducible.

A GPU run adds the following to the documented launch (the allocation must hold a GPU). The
fit and search budgets are four times those of the first agent run (3600 s and 14400 s; raised
at the user's request on 2026-11-18): a fit may take four hours of wall clock, and a
generation's classifier search sixteen. A generation still has eight evaluation units, so in
practice it ends after about eight network attempts; a fit that hangs is stopped after four hours
instead of one.

```bash
  --gpu --image-transport-gb 16 --image-cache-gb 64 --image-cache-dir local \
  --classifier-time-budget 14400 --policy-time-budget 3600 --classifier-search-seconds 57600
```

## 3.6 Exploration tools: geometry, 3D volumes, images and feature discovery

The agent can inspect **TRAIN candidate locations and local fragment geometry**
using `inspect_candidate(candidate_ref, radius_um, occurrence_index)`. Feedback
contains the required candidate references. Each generation allows eight
inspections; split pairs support multiple occurrence locations. These are cached,
processed skeletons, not raw images or GT skeletons.

In explore mode, the agent can define `LOCAL_CONTEXT` and
`extract_local_features(context)` in scorer.py or training.py. The host loads the
matching fragment cache, supplies relative nodes/edges/radii and locally numbered
fragments to an isolated worker, and appends its numeric features to the table.
The same frozen extraction runs on validation outside the LLM. Default extraction
covers the 5000 highest detector-score candidates per brain/kind; the agent may
choose another base feature/direction and a budget up to 20000. Every candidate
still receives a score; unselected local features are missing. Graphs are loaded
for requested geometry/image access, local features, or split feature-diagnostic grouping;
existing prepared tables remain valid. See the
[local context guide](../proofreader_evolve/artifacts/local_context_guide.md) for the
schema, feature example, limits and logs. No new launch flag is required.

Pool-scale descriptors are enabled by default. Geometry and original image crops are
read on demand for any native candidate. No precomputed context directory, detector
band or labeled extension is needed. Evolution reads contexts on demand;
use `--no-descriptors` to disable descriptor tools.
The agent writes `descriptor.py` (`DESCRIPTOR` plus `describe(context)`), uses
`probe_descriptor` on explicit TRAIN cases or uniform samples from each complete
TRAIN pool, and can request previews of the exact inputs.
`DESCRIPTOR.image = {'level': 0, 'radius_um': 40}` sets the physical crop (radius 4..128 um,
at most a 256 um cube per patch since 2026-11-03; `harness/image_coordinates.py`).
Legacy tier names resolve to physical requests too. Descriptor pixels use bounded
temporary staging; numeric outputs remain reusable in the descriptor bank.
`compute_descriptors({"scope": "judgeable"})` (the default since 2026-10-29) covers the
GT-labeled rows, the set every fit and ranking uses on TRAIN, selection and validation
brains (row counts depend on the configured split and are reported by the planner); `all_candidates` covers the full pool; `pilot` is uniform and `boundary`
explicitly selects detector-rank neighbours. The old `all_cached` scope maps to `all_candidates`.
`plan_descriptor_run` estimates all TRAIN brains, then `compute_descriptors` runs the
same code and input path in parallel sandboxed workers. Results are cached by code,
input and implementation identity in `descriptor_bank/` and registered as `bank_agent_*`
predictors. Per-brain diagnostics include finite/missing/nonzero counts, min/max/std,
quartiles and exceptional candidate handles. Frozen validation uses the same resolved
image request. Probes and plans use descriptor wall time without evaluation units;
one uncached full computation costs one evaluation unit under wall-clock budgets
(`--descriptor-wall-seconds`, `--descriptor-generation-wall-seconds`). See the
[descriptor guide](../proofreader_evolve/artifacts/descriptor_guide.md).

For independent local investigations, use **agent-written analysis of actual 3D volumes**. Edit `analysis.py` and `analysis_request.json`, then call
`run_volume_analysis({})`. An isolated worker passes original pixels, validity
masks, spacing, candidate anchors and aligned local fragments to `analyze(context)`.
It returns concise JSON findings, without requiring model fitting or 2D previews.
Each generation has eight executions, up to 16 TRAIN candidates each (enough to
compare a group of missed positives with a group of selected label-0 rows), separate
from scoring. Code, input identities, results and logs are saved under
`genNNN/volume_analyses/analysisNNN/`. See the
[3D analysis guide](../proofreader_evolve/artifacts/volume_analysis_guide.md).
Optional `inspect_candidate_image` and `inspect_failure_images({})` tools still
return PNG previews, sharing `--image-previews-per-generation` with descriptor probe
previews (default 32; 0 disables previews). A slice sheet costs one slot; another view
of the same candidate costs another. The default is a sheet
of native slices with raw and depth-clipped fragment rows; `inspect_candidate_image`
accepts a `preview` object selecting plane, split endpoint, relative depth range and
slice count. Actual depths and skipped planes are reported. Projection overviews
remain explicit options; confirm visual connectivity hypotheses with full 3D analysis.
The agent guides recommend comparing TRAIN TP/FN/FP/TN groups across brains, then
expanding useful numerical probes from 128 to 512 or 1024 rows per brain within the
descriptor wall budget. No minimum usage is enforced. Each generation records
`image_preview_usage` (including charged failures), saved previews in `image_evidence`,
paired feature diagnostics and `wall_seconds` for the subsequent run comparison.

To turn findings into a measured policy,
`LOCAL_IMAGE` enables agent-written image features or raw patches for arbitrary
CPU fit/predict programs through a read-only `images` argument. Pixels keep their
original values; preview contrast normalization is separate. Only TRAIN images
are inspectable. Frozen extraction and prediction use validation images inside
the outer evaluator, without labels or agent inspection. See the
[image guide](../proofreader_evolve/artifacts/image_context_guide.md).

To connect small 3D investigations to ranking at larger K, `LOCAL_IMAGE` supports
`selection_mode='top_k_boundary'` with a literal `selection_k`. For K=2000, a
256-row pilot selects ranks 1873..2128 of the declared base predictor; 4000 rows
cover its entire Top 2000 and the next 2000. The agent can call
`plan_image_scoring({})` to preview TRAIN coverage without image IO. Stateless
numeric extraction uses bounded batches with a shared timeout; raw-patch models
receive at most `--image-transport-gb` per worker call (1 GiB by default; `selection_mode='judgeable'`
selects every GT-labeled row, see GPU mode in section 3.5). Reports distinguish 3D access, actual
Top-K image coverage, and image-only paired TRAIN evidence. Neither access nor
coverage alone demonstrates an image-ranking benefit; promotion uses the target-kind validation gate.

Image access requires host registration receipts in `configs/image_alignment.json`
(or `--image-alignment PATH`). The reader uses explicit OME-Zarr axes, units,
scale/translation, channel and timepoint. To audit new/changed sources on an allocated compute node,
run `python -m proofreader_evolve.cli.check_image_alignment --brains BRAIN_IDS
--out docs/image_alignment_review_<DATE>`; use `--image-uri` for an explicit
single-brain source override. Inspect the saved overlays before recording a
receipt with the same brains/output plus `--record-reviewed --review-note "..."`.
This only checks sampled registration, not all candidate locations or error labels.
An unavailable image fails the image-dependent measurement without dropping any
validation brain. Existing table-only policies need no images or credentials.

Feature discovery now starts from bounded matched TRAIN failures and a relevant
hypothesis-evidence record. Since 2026-10-06 the failure cases and feedback examples
describe the assigned search branch rather than the accepted scorer, with a
`branch_vs_accepted` block of positives the branch gains, loses or shares as misses;
and each session can leave a structured `handoff` (open questions, evidence verdicts,
next experiment) in `proposal.json`, which the host stores next to its own
measurements and shows to the next session of that kind in `hypothesis_memory.json`. `inspect_failure_cases({})` batches local inspection;
`evaluate_feature_ablation({})` measures full inputs against a constant-masked
feature group using the same program and parameters. Classifier diagnostics use
the selection protocol's three grouped, fragment-purged folds (two evaluation
units, one per arm); formulas use a descriptive two-arm comparison (two units).
Classifier diagnostics check each full-arm model's frozen input list and report
effective/ignored columns in `input_check`. If no requested removal affects any
fold's inputs or actual raw patches, the result is `not_applicable`, the control is
skipped, and only the completed full arm costs a unit. This observation supplies no
paired evidence and cannot complete a research assignment. Partially effective
groups still run; the check establishes supplied inputs, not internal learner use.
A classifier configuration costs one unit, fold refits included. Evidence is
saved separately from agent claims and
guides soft stagnation scheduling, while the outer promotion gate stays the same.
See the [feature discovery guide](../proofreader_evolve/artifacts/feature_discovery_guide.md)
and the maintained [literature provenance](../proofreader_evolve/WORKFLOW_REVISION.md#literature-basis-and-implementation-provenance)
for mechanisms, source papers, adaptation limits, and verification status.

## 3.7 Branch archive and scheduling

Each kind retains up to **5 TRAIN exploration branches**: the selection-score champion,
then specialists that recover different GT-positive candidates or lead geometry
slices. Specialists may fall more than the near-best tolerance below the champion
(**0.02 absolute Precision@K, 0.005 AUROC** by default; `--candidate-score-tolerance`
overrides). The tolerance applies only to near-best alternatives filling remaining
slots; distinct Top-K membership and formula structures guide that fallback.
Slice diagnostics use split gap <=25/>25 microns and merge degree 3/>=4, plus
whole-pool coverage for each TRAIN brain. Hits always come from the same global
Top K. Raw positive identities stay in the host; only aggregate counts reach
the agent. All successful trials in a generation compete for the bounded archive.
The exploration parent can differ from the accepted scorer. Pool membership does
not require validation acceptance; the target-kind validation gate determines promotion.
In addition, each kind has **one protected reference slot** for its current
accepted component, initialized from the seed and updated only after promotion.
It does not consume a TRAIN pool slot and cannot be evicted for a low selection score
or redundant TRAIN coverage. The same snapshot can serve both reference and TRAIN
roles without duplication. A newly promoted component gets the **next generation
of the same kind** in explore mode, ahead of the TRAIN champion and normal cadence
(the adaptive kind allocation of section 3.4 gives initial opportunities and the
active-kind rotation floor priority over pending follow-ups).
One new measurement from that workspace branch completes the follow-up; cached
restores do not. At most one retry is reserved. Seeds and unchanged references do
not enqueue follow-ups. Otherwise, every **third scheduled generation of each kind**
starts from the reference; merge and split count independently,
including cache-only or failed generations for this scheduling cadence.
Other generations use the TRAIN champion/specialist scheduler, with
periodic exploration controlled by `--explore-every` on newly measured
non-reference rounds, so protected rounds do not displace specialist exploration.
The scheduler also explores whenever
the champion lacks PARAMS. After **2 measured rounds** without sufficient cumulative
selection-objective progress (0.0005 for AUROC; any positive gain for other objectives),
exploration is forced from an alternative branch if available. Coverage and slice
improvements retain specialists but do not reset these performance counters.
After **2 unsuccessful forced explorations**, that kind can pause once a generation
assigned to its current reference has produced a new successful TRAIN measurement.
Cache-only or failed reference rounds do not satisfy that requirement. A promotion
replaces the reference, resets that kind's stagnation counters and reopens it for
exploration. When all target kinds pause the run ends. Cache-only rounds and execution/infrastructure failures
do not advance measured-progress counters.
Separately, two completed same-kind rounds without new measurements force an
investigation. Performance plateaus also require a new paired TRAIN feature
ablation or a numerical 3D comparison on at least two distinct TRAIN candidates.
If registered TRAIN images are available and have not yet been compared, the
assignment requires image evidence. Negative measurements count; previews,
inspection repeats, cached restores and plans do not. A priority follow-up and
investigation can share one assignment. Incomplete assignments skip validation
and retain the accepted policy. The default two incomplete assignments or two
consecutive execution-blocked rounds pause that kind with separate reasons.
`research_evidence.jsonl` records actual tool evidence; per-generation
`research_status.json` gives live completion feedback. Code-format changes are
deduplicated, but the framework cannot certify semantic or scientific novelty.
Controls: `--candidate-pool-size`,
`--candidate-score-tolerance`, `--explore-every`, `--plateau-patience`,
`--exploration-patience`, and `--no-early-stop`.

## 3.8 Submission, validation and run artifacts

Before full evaluation, a small feature-only check exercises the real schema,
missing predictor values and deterministic scoring. The first scorer execution
error grants one additional repair attempt per generation. The final submitted
code must exactly match a successfully measured snapshot; when it does not (an
untested last edit, or a session cut off at the turn cap), the host ignores the edit
and submits the generation's best measured candidate, recorded as
`submission_fallback` in the ledger (2026-10-08). Only that final candidate reaches one evaluation across the fixed
validation brains, even if its TRAIN selection score did not improve. A submission that is
byte-identical to the accepted combined scorer is not evaluated at all: the generation
ends as `unchanged` (ledger flag `submission_unchanged`, trajectory event
`unchanged_submission`, handoff host status `unchanged_submission`), the parent is
retained and no descriptor inference runs (2026-10-12). Since 2026-10-26 the same holds
for a submission identical to any candidate the outer evaluator already measured, by
bytes or by `scores_sha256` (the exact scores of its kind on every TRAIN and selection
cell): the generation ends as `repeated` (ledger `submission_repeat`, gate
`skipped: repeated_submission`, handoff host status `repeated_submission`). With
`--selection-noise-gate` (section 3.2, 2026-11-12) a submission whose TRAIN selection gain is within
TRAIN resampling noise is not evaluated either: the generation ends as `not_measured` (ledger
`submission_within_noise` and `selection_noise`, gate `skipped: selection_within_noise`, handoff host
status `selection_within_noise`). The agent is
told beforehand through `outer_measurement` (generation and promotion status, no
numbers; after a gate rejection also the rejection category of section 3.2) on the search parent, the pool members and training-tool results, and can read
earlier generations' programs and TRAIN artefact trees. Full-precision
target-kind validation scores decide promotion through the gate's table
(section 3.2); the ledger retains each brain/kind's result, any regressed cells and
the table with its intervals. An evaluation failure on any selected brain rejects the
candidate; it never silently removes that brain from the evaluation. Exported `best_scorer.py`
bundles both components and remains usable
with `--start-from`.
SDK failures report the subtype, actual/configured turns and returned error
details; the complete SDK result metadata is saved in `reviser_result.json`.

Evolution prints live progress for table loading, baseline evaluation, measured
attempts and acceptance decisions. Raw agent text, tool calls/results and repeated
TRAIN/parent metric lines remain in the durable traces and interactive report.
Seed, baseline, submitted-candidate and final accepted reports print one line per
brain and error kind for TRAIN and development validation. Each line includes
`pool_candidates`, `discoverable_pool_positives`, `top_k_hits`, requested/effective K,
Precision@K and Recall@K; TRAIN search attempts save the same format in trace files. Discoverable
positives are GT-positive rows in the fixed candidate pool (merge sites or split
pairs), not all GT errors or all GT pairs with both fragments available.
Recall@K is `top_k_hits / discoverable_pool_positives`, or N/A for zero positives.
Final lines describe the retained accepted scorer even when the last candidate
was rejected. Counts come from the loaded evaluation tables without rereading
source graphs or recomputing candidates.
Each run saves `trajectory.txt` (readable) and `trajectory.jsonl` (structured)
under `proofreader_evolve/runs/precision_*/`. Each `genNNN/` also saves:

- `trajectory.txt` / `trajectory.jsonl`: timestamped observable agent messages,
  tool inputs/results linked by tool-use ID, evaluation stages and decisions.
  Full tool payloads are saved here; the console prints stage/measurement summaries.
- `reviser_prompt.txt` and `reviser_result.json`: the task, final explanation,
  token usage and cost when returned by the SDK.
- `parent_scorer.py`, `parent_rules.md`, `scorer.py.diff`, `rules.md.diff`:
  the selected exploration branch and changes made in this generation, including
  failed attempts. `accepted_scorer.py` stores the component used for promotion gates.
- `search_plan.json`, `candidate_pool.json`, `proposal.json`: this generation's
  mode, `branch_role`, branch selection, retained TRAIN candidates, protected
  reference, hypothesis and parameter grid.
- `candidate_pool_after.json`: the TRAIN pool and separate `reference_branch`
  after this generation, including retention reasons and complementary-positive
  statistics. TRAIN archive coverage counts exclude references outside that archive.
- `evaluation.json`: measurements, decision reason and failure stage if applicable.
- `experiments/attemptNNN/`: `proposal.json`, component `scorer.py`,
  `combined_scorer.py`, `scorer.py.diff`, `rules.md`, `result.json`, and successful
  attempts' `parameters.json`, `train_evaluation.json` / `feedback.json`.
  New attempts also record `lineage.edit_base_experiment`, `cached_from`, and
  `*.from_edit_base.diff` relative to the last observed workspace checkpoint;
  this differs from the generation's assigned search branch and accepted policy.
- `parameter_searches/batchNNN/`: the fixed `formula.py`, proposed grid and a
  results table for every configuration that ran (a grid stops at its first execution error).
- `training.py`: editable fit/predict program, restored together with the measured model.
- `classifier_guide.md`, `model_environment.json`: interface, installed packages and budgets.
- `model_artifacts/<digest>/` (run level): hashed model/preprocessing files, required for resume.
- `classifier_fits/fitNNN/`: fitting request/provenance, worker log, model.json,
  training.py, scorer manifest and training_summary.json (or a failed-fit result). Raw TRAIN
  matrices are transient and are never exposed through the agent's file tools.
- `classifier_searches/batchNNN/`: the classifier proposal/grid and outcome table.
  Successful experiment snapshots also include training_summary.json.

The run-level `train_experiments.jsonl` is the complete TRAIN experiment archive;
validation decisions remain in the outer generation ledger. The agent restores
immutable snapshots through the tool's ID lookup. Run-level `candidate_pool.json`
and `search_summary.json` record the retained branches, plateau counts and stopping
reason. All exposed measurements and diagnostics are TRAIN-only. The reference
identity comes from the outer promotion decision; its selection is indirect
development-validation feedback, without exposing heldout data, scores or reasons.
Candidates include `specialist_profile`, `retention`, `vs_champion` and
`vs_other_retained` counts. Each generation's `search_plan.json` also lists
`complementary_candidates` with restore IDs so the agent can investigate methods
that find different positives. New combinations require an ordinary measured
TRAIN experiment; the archive does not automatically ensemble its members.

Events are saved as SDK messages arrive, so interrupted sessions retain their
received history. These traces contain observable actions and explanations,
not hidden model reasoning. Validation metrics in trace files are not readable
through the reviser's file-tool allowlist. Existing runs cannot recover messages
that were previously discarded. To follow a new generation from another terminal:

```bash
tail -f proofreader_evolve/runs/precision_<run>/gen001/trajectory.txt
```

## 3.9 Reports and recorded runs

Each new run automatically writes `report.html` at initialization, after seed
evaluation, at generation boundaries, and on completion or a caught interruption.
Open the file in a browser (or copy it to your local machine); it embeds all assets
and requires no server, external scripts or API calls. Refresh to see the latest
generated version. The English interface includes separate TRAIN/validation
curves, per-brain counts, clickable candidate ancestry with attempt details,
specialist retention, and code diffs. The page has three sections: Run Overview,
Exploration Branches, and Current TRAIN Candidate Pool. Complete generation and
tool-event records remain in the run directory.

Evolution code, comments, docstrings, UI labels, diagnostics, experiment hypotheses,
strategies, and explanations use English. Saved agent output and historical records
are displayed verbatim; conversation language does not change the project language.

Completed runs stay under `proofreader_evolve/runs/precision_<timestamp>_<id>/`.
`dataset_split.json` records the brain roles, `manifest.json` the frozen
detectors, budgets, gate mode and kind schedule, `ledger.jsonl` every generation's
decision (`validation_gate` with `point_gate` and `gate_table`, the gate's statistics table with its intervals, guard rail,
rule and rejection category; ledgers before 2026-11-12 carry `bootstrap_gate` and `paired_bootstrap`, those of 2026-11-10 to 2026-11-12 also `judge` with every ballot; `kind_allocation` with the rule, headroom and momentum; descriptor runs
and inference), `validation_scores/` the host-only held-out scores of the seed and of every measured
submission (section 3.2), and `search_summary.json` the stopping reason. Open items that are
deliberately deferred are listed in `proofreader_evolve/TODO.md`. Treat numbers from these runs as
development evidence; the validation brains are queried every generation.

To generate a report from an existing fixed-pool precision run, from the repo root:

```bash
python -m proofreader_evolve.cli.build_evolution_report proofreader_evolve/runs/precision_<run>
# Optional: continuously rebuild from saved artifacts while a run is in progress.
python -m proofreader_evolve.cli.build_evolution_report proofreader_evolve/runs/precision_<run> --watch
```

These commands only read saved JSON/text; they never load brain caches, execute
scorers, train models or invoke an LLM. Missing old ancestry/pool snapshots are
marked unrecorded; missing measurements remain unknown. Costs are summed once per
generation and incomplete totals are labelled. Raw files remain complete on disk;
embedded code/tool previews are bounded and explicitly marked when truncated.
`run_status.json` records the last known state, not a live process heartbeat.
`execution.json` records hostname, PID, Slurm job/step and start time. The launcher
reconciles abnormal exits; if the launcher/allocation also dies or accounting is
delayed, run this later **on a compute node**:

```bash
python -m proofreader_evolve.cli.check_run_status proofreader_evolve/runs/precision_<run>
```

Only a terminal record for that exact step, or an observed launcher exit, can mark
a run stopped. Slurm OUT_OF_MEMORY confirms OOM; SIGKILL alone does not. Missing or
active accounting without a known exit leaves the saved state unchanged. Completed
runs are preserved. The check updates status/report only; it does not resume work.
Older runs without `execution.json` cannot be reconciled automatically.
`report.html` and its validation
data remain outside the reviser's read allowlist. Report-generation failures emit
a warning and do not alter scorer acceptance.

There is only one evolution workflow now: fixed-pool scorer evolution. The
graph-edit driver, repair-site enumerators, repair templates and dedicated
analyses have been removed, along with the unused `prepared_cache/` directory
and old in-process policy timeout helper. Historical runs and logs remain on
disk; current reporting tools support precision runs only. Current evolution
uses `feature_tables/` prepared from labeled `_add.pkl` files in `cache/`.
See the workflow document for the remaining interfaces and integrity requirements.
