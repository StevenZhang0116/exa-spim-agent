# Fixed-pool scorer contract

Improve ranking precision on the exact frozen detector-native candidate pool.
Evolve `score_candidates(features, ctx)` and these strategy notes; record experiments
in the current generation's `proposal.json`.

- Use English for code identifiers, comments, docstrings, diagnostic messages,
  experiment hypotheses and strategies, rules.md, and final explanations.
- `features` is a copied DataFrame of the selected detector's predictor columns
  plus `detector_score` and optional agent-authored `local_*` and `image_*` columns.
  `ctx` contains only `kind` (`merge` or `split`).
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
- Read local_context_guide.md to inspect TRAIN candidate neighborhoods using
  inspect_candidate(candidate_ref, radius_um, occurrence_index), at most eight
  successful queries per generation. In explore mode, optionally define a literal
  LOCAL_CONTEXT and extract_local_features(context) in scorer.py or training.py.
  The host provides bounded, relative, label-free fragment geometry in an isolated
  worker, then appends your features. Handle NaN for rows outside the deterministic
  context selection. All candidates still receive scores. GT graphs are excluded.
  Validation uses the same frozen extractor outside the LLM.
- Use direct 3D code analysis to investigate image evidence on a few candidates;
  with a context cache attached, `compute_descriptors` over the cached band is the
  primary path (descriptor_guide.md). Read
  volume_analysis_guide.md; edit analysis.py and analysis_request.json, then call
  run_volume_analysis({}). Your code receives actual TRAIN voxels, validity,
  spacing, anchors and aligned local fragments. Eight independent executions
  per generation support up to 16 cases each, without fitting a scorer first.
  Returned findings are exploration evidence; implement useful measurements in
  scoring code and evaluate them before claiming a gain. 2D previews through
  inspect_candidate_image or inspect_failure_images are optional, with a separate
  eight-request allowance. Projections alone cannot establish 3D connectivity.
- Read image_context_guide.md to use image observations in scoring.
  LOCAL_IMAGE enables stateless image features or raw
  patches for arbitrary fit/predict models via an images keyword argument.
  The host verifies registration and stages row-aligned, read-only pixels.
  Models cannot choose cloud paths or view validation labels. No GT overlay is
  supplied. A missing/unreviewed image rejects the measurement, never a brain.
  Bridge a useful 3D finding to scoring with a literal top_k_boundary selection
  at this kind's K. Call plan_image_scoring({}) before IO to preview TRAIN coverage.
  Measure a small boundary pilot, expand max_candidates if justified, then ablate
  exactly the image features plus image_available at the final coverage. Numeric
  extraction uses bounded batches; raw model inputs retain their 1 GiB stage cap.
  Input coverage and successful access are not evidence of incremental benefit.
- Learn feature combinations, missingness handling, re-ranking rules, nonlinear
  transformations and per-kind policies from TRAIN feedback only. Do not memorize
  examples. Validation is an adaptive development gate, not a final test set.
- Promotion gate: improve equal-weight mean native-label Precision@K on the fixed
  development-validation brains. First average merge/split precision within each
  brain, then average the brains. Individual brain/kind regressions are allowed.
  The mean gain must also clear row-resampling noise: the host requires the lower
  bound of a paired bootstrap interval of that gain to be above zero, so a change
  worth one or two extra Top-K hits will not be promoted.
  The selection score guides exploration and is not a hard promotion requirement;
  `train_gate` in tool feedback is a diagnostic comparison only.
  Returning fewer candidates cannot inflate precision; the harness always takes
  min(K, pool size). A ranking gain does not prove an executable repair is safe.
- Each generation edits only its assigned kind; the other component is frozen
  by the harness. Read `search_plan.json`: `explore` designs feature combinations,
  transformations, gates or a new supervised classifier. For hand-written formulas,
  `tune` fixes the formula and permits only numeric values in a top-level literal
  `PARAMS` dictionary to change. Reference those values inside the scorer.
  Do not mutate or reassign PARAMS at runtime.
- Write `proposal.json` with nonempty `hypothesis` and `strategy`, a short `family`,
  and before finishing an optional `handoff` (open_questions, evidence with verdicts,
  next_experiment) for the next session; see feature_discovery_guide.md. Also
  name, and optional `parameter_grid` mapping PARAMS keys to numeric lists.
  Call `evaluate_train({})` for a single experiment or `search_parameters({})`
  for a grid. The harness substitutes constants without executing code in the
  agent process, measures each configuration and restores the grid's best
  successful candidate by the official selection score (`selection_protocol`
  in the feedback).
  Grid combinations must fit the remaining budget (8 new evaluations by default).
  Format errors and cached results spend no scoring budget; all calls still use
  SDK turns. The first execution error permits one extra repair evaluation.
- In explore rounds, read failure_cases.json and hypothesis_memory.json. Use
  feature_discovery_guide.md to declare a stable research hypothesis, inspect
  matched failures in one call, and compare proposed features against constant
  removal with evaluate_feature_ablation({}). Classifier diagnostics refit both
  arms over three grouped TRAIN folds and cost two shared evaluation units;
  descriptive formula diagnostics cost two. Reserve one unit for a final candidate.
  These diagnostics are optional evidence, not an additional promotion gate.
- To train your own model, read classifier_guide.md and model_environment.json.
  Write training.py defining fit(X_train, y_train, artifact_dir, params) and
  predict(X, artifact_dir, params). Choose any installed CPU model, preprocessing,
  loss, optimizer, ensemble or custom method within the resource budget. There
  is no algorithm whitelist or mandatory NumPy export. Put arbitrary settings in
  proposal.json.classifier.parameters and call train_classifier({}).
  The harness supplies TRAIN predictors/labels and declared patches to fit, saves
  model artifacts, then measures label-free prediction in a new process using
  the declared inputs. Code and deserialization run
  after filesystem/syscall isolation; model artifacts are read-only for predict.
  No network, package installation or child processes. Use in-process libraries.
  Save all fitted preprocessing and choose/record random seeds yourself.
  In tune mode keep training.py's AST and nonnumeric parameters fixed; refit
  numeric parameters. Explore mode permits rewriting the entire program.
  A new fit plus TRAIN scoring costs one shared budget unit; failed fits also
  cost one and receive no metric. Exact code/parameter repeats are cached.
  Print diagnostics for the saved worker.log and returned output tail.
- With a context cache attached, `descriptor.py` plus `compute_descriptors` measures
  any label-free quantity over the cached candidate band and registers it as
  `bank_agent_*` predictors (NaN outside the band). See descriptor_guide.md.
- Branches are ranked by the selection protocol named in the feedback: under
  grouped_oof, out-of-fold precision on selection brains from host-owned fold
  refits (one unit per configuration, folds included); in-sample TRAIN scores
  are diagnostics only. Validation only runs frozen inference in the outer evaluator.
  No preprocessing statistics, model weights or thresholds are fitted on validation.
- Inspect gained/lost positives and Top-K overlap. Restore with
  `restore_candidate({"candidate_id":"best"})`, `"parent"`, or an experiment ID
  such as `"gen002/attempt001"`. Restoration reuses measured scores and costs no
  evaluation. Final scorer.py, training.py and model parameters must match a
  successful measured snapshot. Model artifact hashes are checked before reuse.
- The TRAIN candidate pool keeps the overall champion, then specialists that
  recover positives missed by retained branches or improve a geometry slice;
  remaining slots may hold near-best alternatives. Specialists bypass the 0.02
  mean-precision tolerance, within the same pool-size limit (default 5 per kind).
  Inspect `specialist_profile`, `retention`, `vs_champion`, and
  `vs_other_retained` in candidate_pool.json. Slice hits/recall use the same
  global Top K, not a new K per slice. Split slices use representative gap
  <=25/>25 microns; merge slices use degree 3/>=4. Missing geometry is included
  in the all-candidates slice only. Exact positive row identities stay in the host.
  search_plan.json lists complementary_candidates and their restore IDs. During
  explore generations, inspect/restore complementary measured methods and use
  their evidence to improve a weak slice or design a combination. This does not
  automatically ensemble candidates; any new combination must be measured.
  A separate `reference_branch` slot protects the component currently supplied
  by the outer evaluator, even if it is weaker on TRAIN or absent from the TRAIN
  archive. It does not count against the TRAIN pool size. A newly supplied
  reference receives the next same-kind round, with at most one retry to produce
  a fresh measurement from its workspace branch. Seeds and unchanged references
  do not enqueue follow-ups. Otherwise, every third scheduled generation of each
  kind explores the reference; `search_plan.branch_role`
  identifies the assigned role. In a reference round, develop the assigned
  method using its TRAIN diagnostics; do not discard it solely because another
  archived method has a higher selection score. Shared roles reuse one snapshot.
  The selected exploration parent can differ from the accepted parent used in
  `parent_train` / `parent_selection` and promotion gates; examples,
  `ranking_delta`, failure cases and `search_branch` describe the exploration
  parent. A new selection-score best,
  previously uncovered positive retained in the pool, or new slice-hit best
  resets plateau counters. Repeated coverage does not count as new progress.
  The reference also needs a newly measured exploration before its kind can
  pause. A new reference resets stagnation counters. Exposed scores, examples
  and comparisons are TRAIN-only; reference selection comes from the outer
  evaluator, which alone sees heldout measurements and decides promotion.
- Read `search_plan.investigation`, `followup_required` and `research_status.json`.
  A performance plateau or two same-kind rounds without fresh measurements require
  a new paired TRAIN ablation or a numerical 3D comparison on at least two distinct
  TRAIN candidates with valid pixels. When `required_source` is `raw_image`, use
  `run_volume_analysis` or ablate all declared image inputs. Negative findings
  count; previews, cached restores, plans and score-only tuning do not satisfy the
  comparison. A combined follow-up/investigation must measure the comparison from
  the assigned branch. Unrelated restores do not satisfy its observed ancestry.
  Tool replies give live completion feedback. Incomplete assignments retain the
  accepted policy and skip outer evaluation; retries are bounded. Restore a
  measured scorer even when the new comparison gives a negative result.
- Search earlier TRAIN experiments with search_memory before repeating a strategy.
  Stratified examples are diagnostic, not representative samples. Use whole-pool
  feature statistics for distribution claims. Measure gained and lost positives
  when assessing ranking tradeoffs.
- These labels follow the frozen detector's annotation convention. Sparse GT
  limits their interpretation; native-label precision is not exhaustive biological
  precision. Actual graph edits require a separate structural evaluation stage.
