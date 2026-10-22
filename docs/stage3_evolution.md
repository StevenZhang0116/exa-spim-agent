# Stage 3. Evolve: re-rank the frozen candidate pools

*Part of the [exa-spim-agent](../README.md) documentation.*

The evolution stage keeps everything the detectors decided and changes only the
order in which their candidates are presented. It is an LLM-driven search: a
fresh agent session is the mutation operator, a TRAIN-side selection score ranks
exploration branches, and a held-out validation gate alone decides promotion: the
mean Precision@K (per kind over the counted brains, then over kinds) must rise by
more than paired-bootstrap resampling noise. Since 2026-10-12 each pool ranks its
own K = min(kind budget, ceil(2 x its GT positives)), so every counted pool has the
same attainable ceiling, and pools with fewer than 20 GT positives are reported but
excluded from the mean (`--k-positive-ratio`, `--min-positives`).
Generations are allocated between the merge and split components by
pending follow-ups, a floor, recent validated gain and remaining headroom.
The subsections below follow one generation of the loop in the figure; the
maintained design record is
[`proofreader_evolve/WORKFLOW_REVISION.md`](../proofreader_evolve/WORKFLOW_REVISION.md).

## 3.1 What is evolved and what stays fixed

`proofreader_evolve` improves **Precision@K over GT-labeled rows on the
same candidate pool** as the selected `merge_junction_detector.py` and
`split_site_detector.py`. The pool is the rule declared in `candidate_pools.json`,
enumerated by `candidate_rules.py` (split occurrences, junction NMS, tip-to-shaft
candidates); the detectors only score its rows and evolution cannot change it.
Exactly one detector per kind is required.

## 3.2 Prepare tables and launch a run

On an allocated compute node in panda:

```bash
python -u proofreader_evolve/prepare_feature_tables.py --brains 794495 789202 794491 794493 802449 --mcl 100
# Once per brain: fragment neighbourhoods plus level-1 and level-0 image patches for the
# top detector-ranked candidates of each kind (default 20,000; S3 reads; roughly 30 min per brain and kind at 20,000).
# --band-rows 50000 grows an existing complete entry in place, reading only the new rows; older-microscope
# brains with feature tables (747807 750318 751473 754610 754612 754613) are cached the same way, one brain at a time.
python -u -m proofreader_evolve.cli.precompute_context_cache --brains 794495 802449 789202 794493 794491 --mcl 100 --readers 16 --band-rows 50000
# --labeled-extension (2026-10-12) additionally caches the GT-labeled rows outside the band (a few thousand per cell),
# so every row Precision@K ranks has a context; it needs the availability sidecars below and reuses complete entries.
python -u -m proofreader_evolve.cli.precompute_context_cache --brains 794495 802449 789202 794493 794491 --mcl 100 --readers 16 --band-rows 50000 --labeled-extension
# When a dataset cache (_add.pkl) is rebuilt, every artifact below descends from it; the order, the identity
# checks and the times are in docs/dataset_artifact_dependencies.md.
# REQUIRED before evolution/preflight: label-availability sidecars (minutes per brain). Labels become
# three-valued (1 error / 0 GT-confirmed no error / NaN not judgeable); ranking, Precision@K, bootstrap and
# fitting use GT-labeled rows only (see WORKFLOW_REVISION "Three-valued labels throughout").
python -u -m proofreader_evolve.cli.precompute_availability --brains 794495 802449 789202 794493 794491 --mcl 100
python -m proofreader_evolve.cli.preflight --brains 794495 789202 794491 794493 802449 \
  --train-brains 794495 802449 --selection-brains 802449 --mcl 100 \
  --context-cache proofreader_evolve/context_cache
# Full run (settings of the recorded 2026-10-05 trial except K, which that trial set to 2000; 20 generations):
export ANTHROPIC_API_KEY=...  AWS_EC2_METADATA_DISABLED=true
python -u -m proofreader_evolve.cli.run_evolution \
  --train-brains 794495,802449 --selection-brains 802449 \
  --validation-brains 789202,794493,794491 \
  --merge-k 300 --split-k 1000 --generations 20 \
  --context-cache proofreader_evolve/context_cache --descriptor-workers auto \
  --classifier-threads 4 --classifier-time-budget 1800 --reviser-max-turns 40 \
  --skip-auxiliary-ablation
# --merge-k / --split-k are per-kind caps. Each pool ranks K = min(cap, ceil(--k-positive-ratio x its GT
# positives)) (default ratio 2.0; 0 restores one fixed K per kind) and pools with fewer than --min-positives
# GT positives (default 20) are scored and reported but excluded from the promotion mean (2026-10-12).
# Controls for attribution: --kind-schedule alternate (strict merge/split rotation),
# --promotion-gate margin (point-estimate gate), omit --context-cache (no descriptors),
# --detector-prior no-score (withhold the frozen detector score: constant seed, same pools and detector
# feature columns; with --context-cache every GT-labeled row must be cached; 2026-10-12).
# Historical in-sample ranking with the detector-fitted TRAIN brain alone:
python -m proofreader_evolve.cli.run_evolution \
  --train-brains 794495 --validation-brains auto --selection-protocol in_sample \
  --merge-k 300 --split-k 1000 --generations 1
```

The CLI defaults are a single TRAIN brain **794495**, the `grouped_oof` selection
protocol, the `bootstrap` promotion gate (1000 draws, alpha 0.05), `adaptive` kind
allocation, 15 generations, and K caps of 300 (merge) / 1000 (split) over GT-labeled rows (300 / 500 from 2026-10-09 to 2026-10-12, 100 / 100 before), with each pool's K scaled to 2 x its GT positives and pools under 20 positives excluded from the mean (2026-10-12). The bare defaults do **not** start
a run: `grouped_oof` ranks TRAIN branches on a brain the detectors were not fitted
on, and 794495 is the detectors' training brain, so the driver aborts with
"No eligible selection brain". Add a detector-naive TRAIN brain and name it as the
selection brain (`--train-brains 794495,802449 --selection-brains 802449`, as
above), or request the historical `--selection-protocol in_sample`. Evolution
writes its log to the run directory (`runs/precision_<run>/trajectory.txt` and
`.jsonl`, `run_status.json`, `report.html`), not to `proofreader_evolve/log/`;
redirect stdout if you want a console log file.
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
score column is withheld and the seed is the constant `artifacts/scorer_no_score.py`); `ctx` contains the candidate kind. It must return one finite
score per row. The harness selects exactly min(K, GT-labeled rows) with deterministic
tie breaking, where K is the pool's own budget: min(kind cap, ceil(`--k-positive-ratio` x the
pool's GT positives)), fixed for the run because it depends on labels only. GT labels are
stored separately and never passed to the scorer.
Training feedback includes labeled examples with TRAIN-only inspection handles; validation feedback is
not supplied to the reviser. Promotion uses **mean validation Precision@K only**:
average each kind's precision over the validation brains whose pool holds at least
`--min-positives` GT positives (equal weight; smaller pools are reported but excluded),
then average the kinds. The candidate must exceed the accepted parent's mean
by more than `--precision-margin` (default 0; ties fail) and, with the default
`--promotion-gate bootstrap`, the lower bound of the paired row-resampling interval of
that gain must be above zero (`--bootstrap-draws 1000`, `--bootstrap-alpha 0.05`);
`--promotion-gate margin` restores the point-estimate rule. Individual brain/kind
regressions are allowed; TRAIN gain is diagnostic, not a promotion requirement.
Training scores still guide experiment ranking, branch retention and plateau
scheduling. Invalid, non-deterministic or
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

There are no default frozen detectors since 2026-10-17 ([Stage 2](stage2_detector_build.md#24-detectors-currently-in-use)): pass one
`--merge-dir` and one `--split-dir` (a deliverable directory, or one of its
`runs/<brain>-<purpose>/` run directories, which the resolver binds to the script through
the run's recorded detector SHA-256), or set `DEFAULT_MERGE_DIR` / `DEFAULT_SPLIT_DIR` in
`proofreader_evolve/cli/precompute_error_scores.py` once the new detectors exist.
The detectors' image pass is threaded (2026-10-21): pass `--image-workers N` to
`precompute_error_scores.py` (0 = auto capped at 8, -1 = all CPUs; the pass waits
on S3, so N may exceed the CPU count), which reaches every detector whose extraction
entry accepts `image_workers`; older detectors run as before.
Detector-fitted brains cannot be validation brains. The reviser requires
`ANTHROPIC_API_KEY`; preparation and baseline-only evaluation do not call an LLM.

## 3.4 One generation: the reviser session

Each generation allows up to 60 SDK interaction turns by default (24 before
2026-10-07, 40 until 2026-10-08); use `--reviser-max-turns` to change this independently of
`--generations`. The reviser reads a compact TRAIN feedback file capped at 128 KB
(48 KB before 2026-10-05, 96 KB until 2026-10-07). Tool results may be up to
100,000 tokens; a result the SDK parks on disk (above about 50 KB) stays readable
through the session's file guard, and file reads allow 100,000 tokens instead of
the SDK default of 25,000. The session may also read its own generation's
`experiments/`, `classifier_fits/` and `descriptor_runs/` directories. Feature names
appear once per cell with aligned example vectors; values are rounded to six
significant digits for display. It samples selected positives and label-0 rows,
rows just below K, and missed positives. Each group keeps a fixed anchor and
reproducibly rotates other examples across generations, up to four per group
(fewer when needed to fit). `feature_statistics.json` provides whole-TRAIN
feature quartiles by label (1 / 0; NaN rows counted), so error samples are not mistaken for the
full distribution. Candidate feedback adds gained/lost positives and Top-K overlap.

Each generation edits one component. By default the kind is chosen adaptively:
a pending promotion follow-up first, then a floor of one revision per
`--kind-floor-every` generations (default 4), then the kind with the larger recent
gate-passing validation gain, then the larger remaining headroom on the validation
cells; `--kind-schedule alternate` restores strict merge/split rotation and
`--target-kind split` or `merge` focuses the search. The harness freezes the other
component. The agent writes its hypothesis, strategy, formula family and numeric
grid to `proposal.json`, then calls `evaluate_train({})` or `search_parameters({})`.
These tools have no text arguments. The default budget is **8 evaluation units**
per generation, configurable with `--train-evaluations-per-generation`; cache hits
and proposal-format errors cost no scoring budget. New configurations and optional
feature-diagnostic fold/arm fits share these units. Tool calls still consume SDK
turns. `search_memory` searches earlier TRAIN experiments, including rejected and
failed attempts. `restore_candidate` accepts an experiment ID, `parent`, or `best`,
and restores measured code without another scorer execution.

The scheduler separates `explore` (design a formula, features or model program)
from `tune` (change numeric values only). Formula scorers expose a literal
`PARAMS = {'weight': 1.0}` dictionary and
reference its values. `search_parameters` enumerates the proposed grid while
keeping the formula fixed, snapshots each configuration up to the first execution
error of the formula (`stopped_early` lists the skipped, uncharged configurations)
and restores the best successful one by the official selection score. In tune
generations an AST fingerprint enforces the assigned formula. Grid sizes must fit
the remaining budget; oversized grids fail before scoring.

## 3.5 Agent-authored models and the selection protocol

The agent can also author **its own training and prediction program** in
`training.py`, defining `fit(X_train, y_train, artifact_dir, params)` and
`predict(X, artifact_dir, params)`. There is no model whitelist or required
NumPy export. Within installed CPU dependencies and resource limits, the agent
chooses preprocessing, features, model architecture, loss, optimizer, ensembles,
training schedule and file format. `proposal.json.classifier.parameters` supplies
arbitrary JSON settings. `train_classifier({})` runs the program on **TRAIN only**,
saves its artifacts and measures its ranking. See the
[training guide](../proofreader_evolve/artifacts/classifier_guide.md) and
[editable example](../proofreader_evolve/artifacts/training.py).

Agent code and model deserialization run after Linux Landlock + seccomp isolation.
Installed libraries and supplied inputs are readable; fit can write model files
and scratch space. Prediction receives predictors, declared raw image patches
and read-only model artifacts, without labels. Network, child processes, credentials and unrelated data paths
are unavailable. Unsupported isolation fails closed. The framework cannot prove
that arbitrary prediction code implements frozen preprocessing correctly; that
remains part of the model interface contract.

One new fit plus TRAIN evaluation consumes one shared budget unit; failed fits
also consume one with no metric. Exact training-source/parameter repeats reuse
the fitted model. Explore generations can rewrite the whole program; tune
fixes its AST and nonnumeric settings while refitting numeric parameters.
Defaults are 300 seconds wall/CPU, 16384 MiB (8192 before 2026-10-08) and one numerical-library thread
(`--classifier-time-budget`, `--classifier-memory-mb`, `--classifier-threads`);
the recorded full runs use 1800 seconds and 4 threads.
Artifact storage permits 128 MiB / 256 regular files per model. The agent chooses
and records randomness seeds. `model_environment.json` lists installed packages
and budgets; training stdout/stderr is saved and its last 8 KB returned to the agent.

Branches are ranked by the selection protocol: under the default `grouped_oof`,
grouped out-of-fold precision on detector-naive selection brains from host-owned
fold refits, with auxiliary TRAIN brains as fitting-only data; in-sample TRAIN
precision is a **resubstitution diagnostic**. The agent may still use TRAIN-only
internal validation/CV during fitting. Final promotion requires
improvement in mean development-validation precision that clears the paired-bootstrap
noise gate (section 3.3). The outer evaluator calls
frozen feature extraction and predict on validation; it never supplies validation labels to a worker.
`best_scorer.py` binds training source, parameters, provenance and artifact hashes.
Keep its sibling `model_artifacts/` directory when moving/resuming a run; use
`--start-from-artifacts` for another artifact location. Model dependencies are
needed at inference time. Resuming a managed model fitted on a requested
validation brain is rejected before seed evaluation. Legacy NumPy exports remain
readable; newly trained models use the program/artifact interface.

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

With `--context-cache`, the primary image path is **pool-scale agent descriptors**.
`precompute_context_cache` stores, once per brain and kind, the raw context of the
top detector-ranked candidates (the band: 50,000 rows since 2026-10-08, 20,000 before): the fragment neighbourhood (50 um, up to
256 nodes) and image patches at level 1 (30 um radius) and level 0 (16 um radius,
the finest resolution), both for the whole band (level 0 covered only the first
4,000 rows before 2026-10-05; `--extend-tier level0` grows a tier of existing
entries in place; `--labeled-extension` appends the GT-labeled rows outside the band
so every ranked row has a context). Nothing in the cache is a feature. The agent writes
`descriptor.py` (`DESCRIPTOR` plus `describe(context)`) and calls
`plan_descriptor_run` / `compute_descriptors`; the host runs the code over the
cached rows in parallel sandboxed workers, caches results by code hash in
`descriptor_bank/`, registers them as `bank_agent_*` predictors with
label-conditional TRAIN quartiles, and recomputes them on validation brains at
submission. One uncached call costs one evaluation unit under wall-clock budgets
(`--descriptor-wall-seconds`, `--descriptor-generation-wall-seconds`). See the
[descriptor guide](../proofreader_evolve/artifacts/descriptor_guide.md).

Without a context cache, the image exploration path is **agent-written analysis
of actual 3D volumes**. Edit `analysis.py` and `analysis_request.json`, then call
`run_volume_analysis({})`. An isolated worker passes original pixels, validity
masks, spacing, candidate anchors and aligned local fragments to `analyze(context)`.
It returns concise JSON findings, without requiring model fitting or 2D previews.
Each generation has eight executions, up to 16 TRAIN candidates each (enough to
compare a group of missed positives with a group of selected label-0 rows), separate
from scoring. Code, input identities, results and logs are saved under
`genNNN/volume_analyses/analysisNNN/`. See the
[3D analysis guide](../proofreader_evolve/artifacts/volume_analysis_guide.md).
Optional `inspect_candidate_image` and `inspect_failure_images({})` tools still
return XY/XZ/YZ PNG previews, with a separate eight-image allowance.

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
retain the 1 GiB staging limit. Future reports distinguish 3D access, actual
Top-K image coverage, and image-only paired TRAIN evidence. Neither access nor
coverage alone demonstrates an image-ranking benefit; the outer mean gate is unchanged.

Image access requires host registration receipts in `configs/image_alignment.json`
(or `--image-alignment PATH`). The reader uses explicit OME-Zarr axes, units,
scale/translation, channel and timepoint. To audit new/changed sources on n257,
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
A classifier configuration costs one unit, fold refits included. Evidence is
saved separately from agent claims and
guides soft stagnation scheduling, while the outer promotion gate stays the same.
See the [feature discovery guide](../proofreader_evolve/artifacts/feature_discovery_guide.md)
and the maintained [literature provenance](../proofreader_evolve/WORKFLOW_REVISION.md#literature-basis-and-implementation-provenance)
for mechanisms, source papers, adaptation limits, and verification status.

## 3.7 Branch archive and scheduling

Each kind retains up to **5 TRAIN exploration branches**: the selection-score champion,
then specialists that recover different GT-positive candidates or lead geometry
slices. Specialists may fall more than **0.02 absolute precision** below the
champion. The tolerance applies only to near-best alternatives filling remaining
slots; distinct Top-K membership and formula structures guide that fallback.
Slice diagnostics use split gap <=25/>25 microns and merge degree 3/>=4, plus
whole-pool coverage for each TRAIN brain. Hits always come from the same global
Top K. Raw positive identities stay in the host; only aggregate counts reach
the agent. All successful trials in a generation compete for the bounded archive.
The exploration parent can differ from the accepted scorer. Pool membership does
not require validation acceptance; the validation mean alone determines promotion.
In addition, each kind has **one protected reference slot** for its current
accepted component, initialized from the seed and updated only after promotion.
It does not consume a TRAIN pool slot and cannot be evicted for a low selection score
or redundant TRAIN coverage. The same snapshot can serve both reference and TRAIN
roles without duplication. A newly promoted component gets the **next generation
of the same kind** in explore mode, ahead of the TRAIN champion and normal cadence
(the adaptive kind allocation of section 3.4 schedules pending follow-ups first).
One new measurement from that workspace branch completes the follow-up; cached
restores do not. At most one retry is reserved. Seeds and unchanged references do
not enqueue follow-ups. Otherwise, every **third scheduled generation of each kind**
starts from the reference; merge and split count independently,
including cache-only or failed generations for this scheduling cadence.
Other generations use the TRAIN champion/specialist scheduler, with
periodic exploration controlled by `--explore-every` on newly measured
non-reference rounds, so protected rounds do not displace specialist exploration.
The scheduler also explores whenever
the champion lacks PARAMS. After **2 measured rounds** without
a new selection-score best, previously uncovered retained positive, or historical
slice-hit best, exploration is forced from an alternative branch if available.
After **2 unsuccessful forced explorations**, that kind can pause once a generation
assigned to its current reference has produced a new successful TRAIN measurement.
Cache-only or failed reference rounds do not satisfy that requirement. A promotion
replaces the reference, resets that kind's stagnation counters and reopens it for
exploration. When all target kinds pause the run ends. Rediscovered positive coverage
does not reset counters. Cache-only rounds and execution/infrastructure failures
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
validation brains, even if its TRAIN precision did not improve. A submission that is
byte-identical to the accepted combined scorer is not evaluated at all: the generation
ends as `unchanged` (ledger flag `submission_unchanged`, trajectory event
`unchanged_submission`, handoff host status `unchanged_submission`), the parent is
retained and no descriptor inference runs (2026-10-12). Full-precision
validation means decide promotion together with the paired-bootstrap noise bound
(section 3.3); reports retain each brain/kind's result, any regressed cells and
the bootstrap interval. An evaluation failure on any selected brain rejects the
candidate; it never drops that brain from the average. Exported `best_scorer.py`
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
decision (`validation_gate` with `point_gate`, `bootstrap_gate` and the paired
interval; `kind_allocation` with the rule, headroom and momentum; descriptor runs
and inference) and `search_summary.json` the stopping reason. Open items that are
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
Abrupt process kills cannot write a final status. `report.html` and its validation
data remain outside the reviser's read allowlist. Report-generation failures emit
a warning and do not alter scorer acceptance.

There is only one evolution workflow now: fixed-pool scorer evolution. The
graph-edit driver, repair-site enumerators, repair templates and dedicated
analyses have been removed, along with the unused `prepared_cache/` directory
and old in-process policy timeout helper. Historical runs and logs remain on
disk; current reporting tools support precision runs only. Current evolution
uses `feature_tables/` prepared from labeled `_add.pkl` files in `cache/`.
See the workflow document for the remaining interfaces and integrity requirements.
