# Fixed-pool scorer contract

Improve the configured ranking objective for the assigned kind on the frozen
candidate pool. Use English for code, diagnostics, proposals and research notes.

- Learn from TRAIN only. Validation data, labels and logs are unavailable to the
  reviser; the host runs frozen inference and the promotion gate. Never fit
  preprocessing, weights or thresholds on validation. GT graphs and absolute
  candidate identities are not feature inputs. Do not memorize feedback examples.
- Pools, labels, detector weights, brain roles, K and evaluation rules are fixed.
  Edit only the assigned kind. Return one finite score for every input row in its
  original order; no abstention or candidate expansion. A missing predictor is
  allowed, a missing output score is not. Handle the all-missing preflight row;
  copy read-only pandas arrays before mutation.
- Labels are 1 (error), 0 (confirmed no error), or NaN (not judgeable). Only labeled
  rows enter metrics and supervised fitting. Unlabeled rows can still affect
  whole-column statistics or optional transductive inputs; do not assume they
  cannot affect measured scores. Feedback samples are diagnostic, not a dataset.
- `objective`, `aggregation`, `head_guard` and `selection_protocol` in feedback
  define the current rules. Compare the configured objective, not an assumed
  metric. Grouped OOF ranks within host-defined folds; in-sample scores are
  diagnostic under that protocol. Selection and ablation use the same per-kind
  aggregation as validation (default: positive-weighted). The host controls the
  separate validation gain, bootstrap and head guard. TRAIN improvement alone
  does not establish promotion; a ranking gain does not establish repair safety.
- Respect `search_plan` and its host-enforced research/follow-up assignment.
  Incomplete assignments retain the accepted scorer and skip outer evaluation.
  The generation prompt states completion conditions; tool replies give live
  status. Negative measured results are valid evidence.
- Formula interface: `score_candidates(features, ctx)`, with `ctx['kind']` and
  row-aligned predictors. Define a literal top-level `PARAMS` for tunable numeric
  constants; reference it without runtime mutation. No file/network access,
  randomness, persistent state, subprocesses or extra imports; NumPy, pandas
  and math are preloaded. Use `evaluate_train` or `search_parameters` to measure.
  For fitted models, follow classifier_guide.md; any installed CPU method is
  allowed within the limits in model_environment.json. Save fitted preprocessing
  and explicit random seeds. Network, package installation and child processes
  are unavailable; trained artifacts are read-only during prediction.
- Submit exactly a successfully measured snapshot. For fitted models, edit
  training.py and refit; do not edit the generated scorer manifest. Restore tools
  recover measured code, parameters and model together. Unmeasured edits are
  replaced by the host's best measured candidate. Artifact hashes are verified.
- Record observed evidence, uncertainty and next steps in rules.md and the
  proposal handoff. Do not copy this contract or the guides into research notes.
  Do not claim gains from previews, successful IO or an unmeasured hypothesis.
