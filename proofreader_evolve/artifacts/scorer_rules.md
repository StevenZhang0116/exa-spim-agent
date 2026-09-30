# Fixed-pool scorer contract

Improve ranking precision on the exact frozen detector-native candidate pool.
Evolve `score_candidates(features, ctx)` and these strategy notes; record experiments
in the current generation's `proposal.json`.

- `features` is a copied DataFrame of the selected detector's predictor columns
  plus `detector_score`. `ctx` contains only `kind` (`merge` or `split`).
- Return a finite numeric score for EVERY row, preserving input order. No NaN,
  omission or variable-length selection. Hand-written formulas cannot use file
  access, randomness or persistent state. Trained models may load their artifacts.
  Preflight includes an all-missing predictor row (detector_score=0). Handle it
  explicitly. Pandas arrays may be read-only: copy before in-place mutation.
- Hand-written formula code executes in a separate Linux worker with preloaded NumPy, pandas and
  math. File/network access, subprocesses and further on-disk imports are blocked
  by seccomp; credentials and labels do not enter this process. Use vectorized
  numerical operations within the fixed wall-time and 8 GiB address-space limits.
- Candidate geometry, occurrences, native labels, original detector weights, pool membership,
  K, evaluator and train/validation assignments are fixed by the harness.
- No `ENUM_PARAMS`, `propose_edits`, candidate expansion or post-edit enumeration.
- Learn feature combinations, missingness handling, re-ranking rules, nonlinear
  transformations and per-kind policies from TRAIN feedback only. Do not memorize
  examples. Validation is an adaptive development gate, not a final test set.
- Promotion gate: improve equal-weight mean native-label Precision@K on the fixed
  development-validation brains. First average merge/split precision within each
  brain, then average the brains. Individual brain/kind regressions are allowed.
  TRAIN precision guides exploration and is not a hard promotion requirement;
  `train_gate` in tool feedback is a diagnostic comparison only.
  Returning fewer candidates cannot inflate precision; the harness always takes
  min(K, pool size). A ranking gain does not prove an executable repair is safe.
- Each generation edits only its assigned kind; the other component is frozen
  by the harness. Read `search_plan.json`: `explore` designs feature combinations,
  transformations, gates or a new supervised classifier. For hand-written formulas,
  `tune` fixes the formula and permits only numeric values in a top-level literal
  `PARAMS` dictionary to change. Reference those values inside the scorer.
  Do not mutate or reassign PARAMS at runtime.
- Write `proposal.json` with nonempty `hypothesis` and `strategy`, a short `family`
  name, and optional `parameter_grid` mapping PARAMS keys to numeric lists.
  Call `evaluate_train({})` for a single experiment or `search_parameters({})`
  for a grid. The harness substitutes constants without executing code in the
  agent process, measures each configuration and restores the grid's best
  successful candidate by TRAIN mean precision.
  Grid combinations must fit the remaining budget (8 new evaluations by default).
  Format errors and cached results spend no scoring budget; all calls still use
  SDK turns. The first execution error permits one extra repair evaluation.
- To train your own model, read classifier_guide.md and model_environment.json.
  Write training.py defining fit(X_train, y_train, artifact_dir, params) and
  predict(X, artifact_dir, params). Choose any installed CPU model, preprocessing,
  loss, optimizer, ensemble or custom method within the resource budget. There
  is no algorithm whitelist or mandatory NumPy export. Put arbitrary settings in
  proposal.json.classifier.parameters and call train_classifier({}).
  The harness supplies TRAIN features/labels to fit, saves model artifacts, then
  measures feature-only prediction in a new process. Code and deserialization run
  after filesystem/syscall isolation; model artifacts are read-only for predict.
  No network, package installation or child processes. Use in-process libraries.
  Save all fitted preprocessing and choose/record random seeds yourself.
  In tune mode keep training.py's AST and nonnumeric parameters fixed; refit
  numeric parameters. Explore mode permits rewriting the entire program.
  A new fit plus TRAIN scoring costs one shared budget unit; failed fits also
  cost one and receive no metric. Exact code/parameter repeats are cached.
  Print diagnostics for the saved worker.log and returned output tail.
- Fitted-model TRAIN scores are in-sample resubstitution, not cross-validated
  performance. Validation only runs frozen inference in the outer evaluator.
  No preprocessing statistics, model weights or thresholds are fitted on validation.
- Inspect gained/lost positives and Top-K overlap. Restore with
  `restore_candidate({"candidate_id":"best"})`, `"parent"`, or an experiment ID
  such as `"gen002/attempt001"`. Restoration reuses measured scores and costs no
  evaluation. Final scorer.py, training.py and model parameters must match a
  successful measured snapshot. Model artifact hashes are checked before reuse.
- The TRAIN candidate pool retains multiple near-best, distinct Top-K sets.
  The selected exploration parent can differ from the accepted parent used in
  `parent_train` feedback and promotion gates. Plateau scheduling changes branches
  and formula-search mode using TRAIN only. The outer validation mean controls promotion.
- Search earlier TRAIN experiments with search_memory before repeating a strategy.
  Stratified examples are diagnostic, not representative samples. Use whole-pool
  feature statistics for distribution claims. Preserve currently selected positives.
- These labels follow the frozen detector's annotation convention. Sparse GT
  limits their interpretation; native-label precision is not exhaustive biological
  precision. Actual graph edits require a separate structural evaluation stage.
