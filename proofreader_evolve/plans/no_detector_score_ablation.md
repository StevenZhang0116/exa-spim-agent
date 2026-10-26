# Plan: ablation without the frozen detector score (`--detector-prior no-score`)

Status (2026-10-13): implemented; full suite 348 tests OK on n268. No real run yet.

## Question

How much of evolution's final validation Precision@K comes from the frozen detector
model? The ablation run starts without `detector_score` and keeps everything else:
the detector-native candidate pools, the detector's feature columns (computed
quantities, not fitted), labels, K, brains, generations, budgets and promotion gate.

Changing only the seed scorer (`--start-from`) is not this ablation: `detector_score`
stays in `features` and the contract invites its use, so the agent restores it within a
generation or two.

## Where the frozen score reaches the agent today

| Path | Location | Treatment in `no-score` |
|---|---|---|
| Feature column | `harness/native_pool.py:110` adds `detector_score` to `table.features` | Dropped in memory after table load (native_pool.py is not edited; table files and digests unchanged) |
| Band indicator column | `descriptor_runs.register_context_column` adds `bank_context_available` = 1 inside the detector top-N band | Not registered |
| Descriptor contexts | `context_cache.contexts()` passes `base_columns ∩ table.features` | Covered by the drop |
| Descriptor coverage | Band = top 50k by `detector_score`; without a labeled extension, "descriptor is finite" among labeled rows = "in the detector band" | `--context-cache` requires every used entry's band + extension to cover all GT-labeled rows; otherwise the run stops before any LLM call |
| Local / image row selection | Default `selection_feature='detector_score'` (local 5000, image 32 rows) | No default: the agent names a remaining base column; the error message says so |
| Preflight missing row | `image_features.subset_features` sets `detector_score=0` | Only when the column exists |
| Failure cases | `failure_cases.py:35` adds the score when present | Covered by the drop |
| Contract and guides | `scorer_rules.md`, `classifier_guide.md`, local/image guides mention `detector_score` | Mode notice appended to the system prompt and the seed rules: the column is absent, ignore those references |
| Descriptor bank | Earlier runs' descriptors could read `detector_score` from their contexts | Default bank `descriptor_bank_no_score/`; descriptor sources referencing `detector_score` are refused on restore |
| Starting scorer | Default seed returns `detector_score` | Default seed `artifacts/scorer_no_score.py`; any `--start-from` source mentioning `detector_score` is refused |

## Steps

1. `cli/run_precision_evolution.py`: `--detector-prior {frozen,no-score}`, default
   `frozen` (current behaviour, byte-identical paths).
2. New `harness/detector_prior.py`:
   - `strip(banks)` removes `detector_score` from every `table.features` and keeps it
     host-only on `table.host_detector_score`, never passed to workers or the reviser;
   - `check_source(text)` rejects scorer, training-program or descriptor sources that
     reference `detector_score` or `bank_context_available`;
   - `check_context_coverage(cache, banks, kinds)` verifies band + extension ⊇ GT-labeled
     rows per entry.
3. Order in `_run`: load tables → apply availability → evaluate the frozen-score
   baseline on the unstripped tables (host comparator; the report keeps the
   "Frozen detector baseline" row) → `strip` → local/image/context attachment → seed.
4. Seed `artifacts/scorer_no_score.py`: a constant score, so the starting point is the
   label-independent tie order (equivalent to a random ranking of labeled rows); the
   agent builds formulas or TRAIN-fitted models from the remaining columns.
5. Selection defaults: local/image specs without `selection_feature` fail with
   "detector_score is withheld in this run (--detector-prior no-score); set ... selection_feature
   to another base predictor column".
6. `image_features.subset_features`: set `detector_score=0` only if the column exists.
7. Reviser: append the mode notice to `CONTRACT` (system prompt) and to the seed rules.
8. Records: `manifest.json`, `dataset_split.json` and the report header carry
   `detector_prior` (report row "Detector prior"); `collect_run` returns `detector_prior`
   (`frozen` for older runs), so ablation and normal runs cannot be mixed.
9. Tests (`proofreader_evolve/tests/test_detector_prior.py`, tiny fixtures):
   - the scorer worker receives frames without `detector_score`; every agent-visible
     JSON file of the generation (feedback, failure cases, statistics) is free of
     `detector_score` and `bank_context_available`; descriptor contexts carry neither
     the score nor `row_in_band` (training-program and volume-analysis inputs are
     covered by the in-memory drop, not by a dedicated test);
   - the baseline is still measured from the frozen score;
   - incomplete context coverage stops the run; referencing sources are refused;
   - `frozen` mode is unchanged (existing suites pass).
10. Docs: CLAUDE.md / README flag row; `WORKFLOW_REVISION.md` dated entry (project
    design, no literature source).

## Running the comparison

Same command as the reference run plus `--detector-prior no-score`, same seed for the
brain roles, generations and budgets:

```bash
python -m proofreader_evolve.cli.run_evolution \
    --train-brains 794495,802449 --selection-brains 802449 \
    --validation-brains 789202,794493,794491 --merge-k 300 \
    --detector-prior no-score --generations 15
```

Since 2026-10-26 split ranks K = its GT positives with no cap (`--split-k 1000` was the cap when this
plan was written; passing it would restore the old window).

- With `--context-cache`, every brain used needs the labeled extension
  (built for all eleven brains on 2026-10-13 with every GT-labeled row cached, see TODO.md).
- Compare final validation Precision@K (absolute) and the per-generation trajectory with
  the reference run. Promotion counts are not comparable: from a constant seed, early
  promotions are easy.
- 794495 stays in TRAIN as in the reference run, so the only difference is the prior.

## Not in scope

- Removing the detector feature columns as well (mode B, `none`).
- Changing the candidate pools: they come from the detectors' geometric candidate
  policies, not from the fitted models.
