# Agent-authored model training

You choose the model, features, preprocessing, loss, optimizer, ensemble,
training schedule and serialization format. There is **no model whitelist** and
no requirement to translate a trained model to NumPy. Read
`model_environment.json` for installed packages and resource settings. The
editable `training.py` is only an example; replace it with your own method.
CPU execution, installed dependencies and the stated budgets define what can run.
GPU access, network, package installation and child processes are unavailable;
use in-process training and threads rather than multiprocessing/joblib processes.

## Interface

Write these functions in `training.py`:

```python
def fit(X_train, y_train, artifact_dir, params):
    # X_train: DataFrame of common TRAIN predictors, including detector_score.
    # y_train: aligned binary native labels; all TRAIN rows are supplied.
    # Fit any model/preprocessing and save whatever predict needs here.
    # fit's return value is ignored; persist the fitted state as files.
    ...


def predict(X, artifact_dir, params):
    # New process; features only, no labels, brain ID or candidate IDs.
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
runs feature-only inference, evaluates TRAIN Precision@K and restores the best
successful configuration by TRAIN mean precision. TRAIN comparisons guide search;
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

The outer TRAIN score is **in-sample resubstitution**; no independent generalization
estimate is implied. You can implement a TRAIN-only internal split or CV inside
fit and print its diagnostics. Final promotion still requires the original TRAIN
gate, then one outer development-validation evaluation of the frozen candidate.
Repeated development validation is not an untouched final test.

To move or resume a run, keep `best_scorer.py` and its sibling `model_artifacts/`
together. `--start-from` reuses those artifacts without fitting. For another
layout, pass `--start-from-artifacts /path/to/model_artifacts`. Installed model
libraries must remain compatible; their recorded versions aid reproducibility.
Legacy self-contained NumPy model exports remain readable for resume, but all
new training uses this interface.
