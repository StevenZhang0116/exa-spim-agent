# Fixed-pool scorer contract

Improve ranking precision on the exact frozen detector-native candidate pool.
Only evolve `score_candidates(features, ctx)` and these strategy notes.

- `features` is a copied DataFrame of the selected detector's predictor columns
  plus `detector_score`. `ctx` contains only `kind` (`merge` or `split`).
- Return a finite numeric score for EVERY row, preserving input order. No NaN,
  omission, variable-length selection, file access, randomness or persistent state.
- Candidate geometry, occurrences, native labels, model weights, pool membership,
  K, evaluator and train/validation assignments are fixed by the harness.
- No `ENUM_PARAMS`, `propose_edits`, candidate expansion or post-edit enumeration.
- Learn feature combinations, missingness handling, re-ranking rules, nonlinear
  transformations and per-kind policies from TRAIN feedback only. Do not memorize
  examples. Validation is an adaptive development gate, not a final test set.
- Gate: improve macro native-label Precision@K with no brain/kind regression.
  Returning fewer candidates cannot inflate precision; the harness always takes
  min(K, pool size). A ranking gain does not prove an executable repair is safe.
- These labels follow the frozen detector's annotation convention. Sparse GT
  limits their interpretation; native-label precision is not exhaustive biological
  precision. Actual graph edits require a separate structural evaluation stage.
