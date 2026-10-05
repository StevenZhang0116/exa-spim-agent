# Agent-authored model training

You choose the model, features, preprocessing, loss, optimizer, ensemble,
training schedule and serialization format. There is **no model whitelist** and
no requirement to translate a trained model to NumPy. Read
`model_environment.json` for installed packages and resource settings. The
editable `training.py` is only an example; replace it with your own method.
CPU execution, installed dependencies and the stated budgets define what can run.
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
this kind's K and call plan_image_scoring({}) before IO. Numeric image extraction
uses bounded batches; raw-patch models retain a 1 GiB staging limit. Inspect actual
Top-K image coverage, then measure an image-only ablation at the intended final
coverage before attributing any improvement to pixels. See image_context_guide.md.
GPU access, network, package installation and child processes are unavailable;
use in-process training and threads rather than multiprocessing/joblib processes.

## Interface

Write these functions in `training.py`:

```python
def fit(X_train, y_train, artifact_dir, params):
    # X_train: DataFrame of common TRAIN predictors, including detector_score.
    # y_train: aligned binary native labels; normal fits use all TRAIN rows.
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
below) alongside in-sample TRAIN Precision@K, and restores the best successful configuration by
the selection score. Selection comparisons guide search;
only the outer mean validation Precision@K controls promotion. With no
grid, it measures one fit. To try a different method, edit training.py and explain
the change in proposal.json/rules.md.

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
`model_artifacts/<digest>/` directory and verified before reuse. Arbitrary file
formats are allowed, up to 128 MiB total / 256 regular files; no symlinks.

Default fitting limits are 300 seconds wall/CPU and 8192 MiB address space.
The host configures `--classifier-time-budget`, `--classifier-memory-mb` and
`--classifier-threads` (default 1). Inference uses `--policy-time-budget` (120s
by default) and 8192 MiB. The thread setting configures numerical libraries;
the CPU/wall limits bound total work. Scratch plus artifacts are monitored with
a 256-MiB limit; temporary/output files may briefly exceed it between checks.
Libraries and inputs are readable, fit artifacts/scratch writable, and prediction
artifacts read-only. All agent imports, fit, predict and model deserialization
occur after Linux Landlock + seccomp isolation; unavailable isolation fails closed.

Print useful training diagnostics, internal CV results and convergence information.
`worker.log` preserves stdout/stderr, and the tool returns its last 8 KB with the
training summary or failure. Logs are limited to 4 MiB. Each measured candidate
records code, parameters, artifact hashes, library versions, TRAIN provenance,
ranking changes, wall time and errors. These are observable execution records,
not hidden agent reasoning.

## Evaluation and resume

Branch ranking uses the **selection protocol** reported in the feedback
(`selection_protocol`). Under `grouped_oof`, `train_classifier({})` first refits
your program in three fixed, fragment-purged folds on the selection brains, with
auxiliary TRAIN brains fully included in every fitting set and never held out.
Out-of-fold scores are ranked inside each fold with budgets that sum to K, and
`target_precision` / `grouped_cv_precision` is the summed TP divided by K. The
full-TRAIN fit is then frozen as the candidate. `candidate_train` numbers remain
**in-sample resubstitution** diagnostics and rank nothing. A model that memorizes
rows scores high in-sample and low out-of-fold. If `fit` declares a `groups`
keyword it receives the label-free source brain of each training row (a pandas
Series) so you can weight or block auxiliary rows. One configuration costs one
evaluation unit, fold refits included. You may still print your own internal
diagnostics. Final promotion uses the outer mean development gate on the frozen
candidate; the selection score is a repeatedly used development signal, not an
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
partition as the selection protocol; only selection brains are scored, with the
partitioned fold budgets. It preserves the feature schema and
does not force a model family. It seeds Python/NumPy per fold in this diagnostic;
agent seed overrides or other random generators can still affect reproducibility.
Afterward, call train_classifier({}) to fit the chosen program on all TRAIN rows.
Diagnostic fold models cannot be submitted. See feature_discovery_guide.md for
split grouping, unavailable-diagnostic behavior, cache identity and limits.

To move or resume a run, keep `best_scorer.py` and its sibling `model_artifacts/`
together. `--start-from` reuses those artifacts without fitting. For another
layout, pass `--start-from-artifacts /path/to/model_artifacts`. Installed model
libraries must remain compatible; their recorded versions aid reproducibility.
Legacy self-contained NumPy model exports remain readable for resume, but all
new training uses this interface.
