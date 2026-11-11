# Agent-authored model training

You choose the model, features, preprocessing, loss, optimizer, ensemble,
training schedule and serialization format. There is **no model whitelist** and
no requirement to translate a trained model to NumPy. Read
`model_environment.json` for installed packages and resource settings. The
editable `training.py` is only an example; replace it with your own method.
Choose methods for the problem, not just the packages already installed. Use
`install_packages({"packages": ["catboost>=1.2"]})` to add dependencies to the current
Python environment (normally panda). Existing package versions stay fixed; a dependency
conflict is returned with pip's output. Installations persist across generations and runs,
refresh `model_environment.json`, and cost no evaluation units. CPU execution and the
stated worker budgets still define what can run.
For new local morphology features, read `local_context_guide.md`. You may define
`LOCAL_CONTEXT` and `extract_local_features(context)` in training.py; the host
appends these label-free features to TRAIN fit inputs and frozen validation inputs.
Keep top-level code usable in the extraction worker, which has no model artifacts.
Before fitting, you can investigate actual TRAIN volumes using custom
`analysis.py` code and `run_volume_analysis({})`; read `volume_analysis_guide.md`.
No model fit or 2D preview is required for this exploration. Its outputs do not
automatically become predictors or fitted artifacts.
For image inputs in a model, read `image_context_guide.md`. A literal `LOCAL_IMAGE` can
append stateless image features or request `raw_patches: True`; the latter adds
an `images` keyword argument to both fit and predict. `images[i]` contains only
the local patches for that input row. You may learn image preprocessing and
encoders in fit and save their frozen state alongside your model. Internal folds
receive only their own row subsets, including images. Handle empty patch lists.
For coverage beyond a few investigated cases, declare top_k_boundary selection at
this kind's K budget (feedback.budgets) and call plan_image_scoring({}) before IO. Numeric image extraction
uses bounded batches; raw-patch models retain a 1 GiB staging limit. Inspect actual
Top-K image coverage, then measure an image-only ablation at the intended final
coverage before attributing any improvement to pixels. See image_context_guide.md.
Inside model workers, GPU access, network, package installation and child processes
are unavailable; use the host's `install_packages` tool before fitting and use in-process
training and threads rather than multiprocessing/joblib processes.

## Three-valued labels

TRAIN, selection and validation labels are three-valued: 1 = error, 0 = no error
confirmed by GT, NaN = not judgeable (no GT tracing near the candidate; most rows).
Top-K, precision, recall and the bootstrap use GT-labeled rows only: NaN rows are
scored but never ranked or counted. `fit` receives only labeled rows; `predict` still
scores every row. Because every ranked row is GT-labeled, a feature that only tells
"near a GT tracing" from "far from GT" earns nothing; separate 1 from 0.

## Interface

Write these functions in `training.py`:

```python
def fit(X_train, y_train, artifact_dir, params):
    # X_train: DataFrame of common TRAIN predictors for GT-labeled rows only.
    #   detector_score is present only in frozen mode; raw also withholds h* columns.
    # y_train: aligned 0/1 labels of those rows (1 = error, 0 = GT-confirmed no error).
    # Optional: declare X_unlabeled=None to also receive predictors (no labels,
    #   no images) of a deterministic sample of NaN-label rows, up to 100,000
    #   per TRAIN brain, e.g. for semi-supervised or domain-aware preprocessing.
    #   Descriptor columns computed under the judgeable scope are NaN on these rows.
    # Framework feature diagnostics supply only the fitting rows of each fold.
    # Fit any model/preprocessing and save whatever predict needs here.
    # fit's return value is ignored; persist the fitted state as files.
    ...


def predict(X, artifact_dir, params):
    # New process; no labels, brain ID or candidate IDs.
    # Raw-image models also accept the images keyword argument described above.
    # Load your frozen model/preprocessor from read-only artifact_dir.
    # Return a 1-D array: one finite score for every input row, original order.
    ...
```

`params` is a copy of `proposal.json.classifier.parameters`; it may contain any
JSON-compatible settings, including nested objects, strings and lists. Import
libraries inside either function or at module scope. Define custom model classes
at module scope; the module name `agent_model` is stable during fit and predict,
so pickle/joblib can resolve them. Model deserialization runs only in an isolated
worker, never in the evaluator process.

Save feature selection and fitted imputation/scaling along with the model. Raw
predictors may contain missing or nonfinite values. Preflight includes an
all-missing predictor row with detector_score=0. Do not fit preprocessing,
calibration or thresholds on validation features. Predictions must be deterministic;
the worker calls predict twice and checks exact equality. Set and record seeds
in your program/parameters when fitting uses randomness. The framework does not
force a particular seed or training algorithm.

To exclude unused descriptors from inference, optionally define
`select_columns(columns, params)` in `training.py`, returning a list of unique
names from the available columns. It runs inside the fit worker before fitting;
`X_train` and optional `X_unlabeled` receive only those columns. The harness freezes
that list for prediction and descriptor dependencies. For example:

```python
def select_columns(columns, params):
    return [c for c in columns if c not in params.get('drop_columns', [])]
```

Without this function, all input columns remain dependencies, even if `fit`
internally ignores some of them. Prediction uses the frozen list without rerunning
the selector.

Choose experiments and model complexity based on the evidence and remaining resources.
`evaluate_feature_ablation` is available when a controlled feature comparison would
help decide the next step; it costs two evaluation units.

Both functions may explicitly declare the optional keyword `threads`, for example
`def predict(X, artifact_dir, params, *, threads=1):`. The host supplies the worker's
CPU-affinity-capped budget, also used for BLAS/OpenMP environment limits. Existing
signatures remain valid. Use this value for library-specific controls, for example
`LGBMClassifier(n_jobs=threads, ...)` and `booster.predict(M, num_threads=threads)`;
do not hardcode 1 or use unbounded `n_jobs=-1`. The host does not rewrite arbitrary
model code, so explicit library settings can override its environment defaults.
Frozen models record the fitted worker's thread budget; prediction caps it again
against the current allocation. Deterministic prediction is still checked twice.

Candidate rows expose `selection_comparison`: overall, whole-brain holdout and per-brain
changes against the accepted parent, with positive-count fractions to reveal dominance
by one brain. These are diagnostics; the configured objective still selects candidates.
`timings` records current-call feature preparation, full fit, OOF fit/predict/total,
preflight, TRAIN scoring and diagnostics in seconds, including transport/startup.
`evaluation_seconds` includes preflight/scoring/selection/diagnostics, and OOF
`total_seconds` includes its fit/predict subtotals; do not sum nested totals twice.
Reused fits have `fit_reused=true`; a reused complete attempt has `timings_reused=true`
and retains its original timing record. No validation timing or cases enter this feedback.

Do not rely on module globals surviving a fit: inference starts in a fresh
process. Top-level code must work without TRAIN labels. `predict` must only load
and apply fitted state; the outer evaluator never calls `fit` on validation.
The sandbox prevents access to validation labels and writes to model artifacts;
it cannot prove the mathematical behavior of arbitrary Python.

## Propose and measure

```json
{
  "hypothesis": "A learned combination can improve the near-K ranking",
  "strategy": "Fit a model on TRAIN and compare the fixed Top-K selection",
  "family": "learned-ranking",
  "classifier": {"parameters": {"C": 1.0}},
  "parameter_grid": {"C": [0.1, 1.0, 10.0]}
}
```

Call `train_classifier({})`. This tool name is retained for continuity; your
program can fit a classifier, regression/ranking model, neural network, ensemble
or custom predictor. The harness runs fit on TRAIN, snapshots the artifacts,
runs label-free inference with the declared inputs, measures the selection score (see Evaluation
below; the configured objective, AUROC by default, with Precision@K and the other rank statistics
beside it) alongside in-sample TRAIN Precision@K, and restores the highest selection
objective among configurations satisfying the selection head guard. If none satisfy it,
the highest objective is restored with the failed guard shown in `selection_gate`.
An absent or disabled guard does not filter candidates. `restore_candidate("best")`
uses the same preference among candidates not yet measured by the outer evaluator,
falling back to all candidates when all have been measured. Explicit IDs remain
selectable; a selection-objective gain over the parent is not required. Selection comparisons guide search;
only the outer evaluator's measurement of the target kind on held-out pools controls promotion:
by default the objective with its resampling-noise bound and the head guard; a run configured
with a promotion judge instead has every rank statistic reviewed with its resampling
uncertainty, the top of the ranking mattering most. A candidate the outer evaluator already
measured carries `outer_measurement` (`generation`, `experiment`, `promoted`) in the pool and
plan files; when a promotion judge rejected it, `rejection` gives the only reason you are told,
one of four words: `significantly_worse` (a rank statistic fell beyond resampling noise: leave
this direction), `within_noise` (nothing improved beyond noise: a larger or different change is
needed, not a retune), `head_not_improved` (the ranking improved overall but not at its top:
work on the highest-ranked candidates) and `not_convincing` (the top improved beyond noise yet
the evidence was judged insufficient: strengthen the same direction). No heldout number is ever
shown. With no
grid, it measures one fit. A grid stops at the first failed fit or execution error of
the program; the reply's `stopped_early` lists the skipped configurations, which are not
charged, so fix the program or its memory use before resubmitting them. To try a
different method, edit training.py and explain the change in proposal.json/rules.md.

- In **explore**, change the entire program and any parameters.
- In **tune**, keep the program AST and nonnumeric parameters fixed; change
  numeric top-level parameters through refitting. Grid keys must already exist
  as numeric parameters. No algorithm-specific bounds are imposed.
- A new fit plus TRAIN scoring costs one shared budget unit. Failed fits also
  cost one and get no fabricated score. Grid sizes must fit the remaining budget.
- Cache identity includes the complete training source and parameters within
  this run's fixed TRAIN dataset. Change an explicit seed parameter for another
  stochastic fit. Exact repeats reuse the measured model.
- `restore_candidate` restores training.py, parameters and the measured model
  together. Finish with code and parameters exactly matching the measured model.
  To return to a formula, edit scorer.py and remove `classifier` from proposal.json.

`scorer.py` becomes a framework-owned manifest binding program, parameters, TRAIN
provenance and every model file's hash. Do not edit it to change a trained model;
edit training.py and refit. Model files are stored in the run's
`model_artifacts/<digest>/` directory and verified before reuse. Temporary OOF and
ablation fold models are discarded after their scores are saved; fold source, requests,
metadata and logs remain. Final fitted models remain available for restore and inference.
Fit and attempt selection arrays share storage; TRAIN snapshots contain only the target
kind, with the other kind supplied by the current accepted parent on restore. Arbitrary file
formats are allowed, up to 128 MiB total / 256 regular files; no symlinks.

Actual fit/predict, memory, thread and search limits are in `model_environment.json`.
The classifier search clock counts new configurations per generation:
preparation, OOF fits/predictions, full fit and evaluation all count, including failures.
It is checked between configurations: a running candidate completes under its worker limits,
so the total can exceed the budget by that candidate's duration. Further uncached fits stop;
cached fits and restoring measured candidates remain available. Descriptor computation and
paired ablations have their own budgets. Inspect `classifier_seconds_remaining` and stage
`timings` before another grid, including the cost of full-pool predictions.
New registered descriptor inputs invalidate fitted-model and extracted-frame reuse even if
training.py and parameters are unchanged. Restoring an older frozen model remains valid;
it does not incorporate later features until you refit it.
The thread setting configures numerical libraries; CPU/wall limits bound total work. Scratch plus artifacts are monitored with
a 256-MiB limit; temporary/output files may briefly exceed it between checks.
Libraries and inputs are readable, fit artifacts/scratch writable, and prediction
artifacts read-only. All agent imports, fit, predict and model deserialization
occur after Linux Landlock + seccomp isolation; unavailable isolation fails closed.

Print useful training diagnostics, internal CV results and convergence information.
`worker.log` preserves stdout/stderr when there is any (a silent successful fit leaves
no file), and the tool returns its last 8 KB with the
training summary or failure. Logs are limited to 4 MiB. Each measured candidate
records code, parameters, artifact hashes, library versions, TRAIN provenance,
ranking changes, wall time and errors. These are observable execution records,
not hidden agent reasoning.

## Evaluation and resume

Branch ranking uses the **selection protocol** reported in the feedback
(`selection_protocol`). Under `grouped_oof`, `train_classifier({})` first refits
your program in three fixed, fragment-purged folds on the selection brains, with
auxiliary TRAIN brains fully included in every fitting set and never held out.
Whole-brain selection holdouts, when configured, are assigned round-robin across
the three folds in sorted brain order. Every row of a given brain is held out in
one fold; its predictions come from a model that never fitted that brain. If every
selection brain is a whole-brain holdout, there are no within-brain selection folds.
Out-of-fold scores are ranked inside each fold using GT-labeled rows only, with
budgets summing to each pool's effective K. `target_objective` (the feedback's
`objective`, AUROC by default, measured per fold and weighted by GT positives) is the
selection score; `target_precision` / `grouped_cv_precision` is Precision@K over the same
rows, combining the target kind's counted pools by the configured aggregation
(sum(TP) / sum(effective K) by default, or equal-brain precision) and is reported beside
it. Pools below the positive floor are reported but excluded. The
full-TRAIN fit is then frozen as the candidate. `candidate_train` numbers remain
**in-sample resubstitution** diagnostics and do not rank candidates under `grouped_oof`. A model that memorizes
rows scores high in-sample and low out-of-fold. If `fit` declares a `groups`
keyword it receives the label-free source brain of each training row (a pandas
Series) so you can weight or block auxiliary rows. One configuration costs one
evaluation unit, fold refits included. You may still print your own internal
diagnostics. Under `in_sample`, the selection score instead uses the full-TRAIN
ranking with the same aggregation and positive floor. Final promotion requires
the target kind's development-validation gain to exceed the configured margin and,
by default, clear the paired-bootstrap noise gate on the frozen candidate (or, in a run
configured with a promotion judge, pass its review of every rank statistic);
the selection score is a repeatedly used development signal, not an
untouched final test.

For a host-measured feature comparison, use `evaluate_feature_ablation({})` with
the current training.py, fixed classifier parameters, an empty parameter_grid,
and research.feature_columns (or default to declared LOCAL_CONTEXT/LOCAL_IMAGE
features and their availability columns).
The framework fits both full and constant-masked inputs over three grouped,
fragment-purged TRAIN folds; fitting workers receive only that fold's fitting
rows and labels. Prediction receives held predictors, declared held-row patches
and frozen artifacts. Removing image_available also withholds raw patches.
This costs two shared evaluation units (one per arm) and uses the same fold
partition as the grouped selection protocol; only selection brains are scored,
with the partitioned fold budgets. Under `in_sample`, this classifier diagnostic
still uses grouped fold refits, while candidate selection uses in-sample scores.
Both ablation arms use the configured target-kind aggregation and positive floor.
`input_check` reports effective and ignored removal columns from the full-arm fold
models. If all requested inputs were excluded and no actual raw patches would be
withdrawn, the result is `not_applicable`: only the full arm runs and costs one unit.
This input check supplies no paired evidence and cannot complete a research assignment.
It preserves the feature schema and
does not force a model family. It seeds Python/NumPy per fold in this diagnostic;
agent seed overrides or other random generators can still affect reproducibility.
Afterward, call train_classifier({}) to fit the chosen program on all GT-labeled TRAIN rows.
Diagnostic fold models cannot be submitted. See feature_discovery_guide.md for
split grouping, unavailable-diagnostic behavior, cache identity and limits.

To move or resume a run, keep `best_scorer.py` and its sibling `model_artifacts/`
together. `--start-from` reuses those artifacts without fitting. For another
layout, pass `--start-from-artifacts /path/to/model_artifacts`. Installed model
libraries must remain compatible; their recorded versions aid reproducibility.
This seeds a new run; it does not restore the old ledger, scheduling history,
candidate pool or cumulative-progress checkpoint.
Legacy self-contained NumPy model exports remain readable for resume, but all
new training uses this interface.
