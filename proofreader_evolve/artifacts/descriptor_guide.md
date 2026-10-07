# Agent descriptors: measure any quantity over the cached band in one call

The host caches, once per brain and kind, the raw context of the top 20,000
detector-ranked candidates: the fragment neighbourhood (50 um, up to 256 nodes)
and image patches for the whole band at two resolutions (`level1`: level 1,
30 um radius, about 1.5 x 1.5 x 2 um voxels; `level0`: level 0, 16 um radius,
about 0.75 x 0.75 x 1 um voxels, the finest view of thin fibres). Nothing in the
cache is a feature.
You decide what to measure by writing `descriptor.py`; the host runs it over
the cached rows in parallel sandboxed workers and registers the results as
`bank_agent_<name>` predictor columns.

## Workflow

1. Edit `descriptor.py`: a literal `DESCRIPTOR = {'kind', 'inputs', 'image_tier',
   'feature_names'}` and `describe(context) -> dict`. `inputs` is `geometry`,
   `image` or `both`; `kind` must be this generation's kind; at most 16 names
   matching `[a-z][a-z0-9_]{1,48}`.
2. Optional: debug `describe` on up to 16 rows with `run_volume_analysis`
   (rename `describe` to `analyze` in `analysis.py`), or call
   `plan_descriptor_run({})`, which times 64 rows with one worker and
   extrapolates each scope to the configured worker count and the remaining
   wall budget. Neither spends evaluation units.
3. `compute_descriptors({"scope": "pilot"})` (512 rows), `"boundary"` (2,000
   rows on each side of rank K) or `"all_cached"` (the whole band). One
   uncached call costs one evaluation unit plus wall time from the per-call and
   per-generation descriptor budgets in `model_environment.json`. The reply
   gives, for each column on each TRAIN brain, the label-conditional quartiles,
   finite fraction and coverage. Results are cached by code hash: the same
   `describe` costs nothing again in any later generation or run.
4. Use the columns like any predictor: in a formula (handle NaN outside the
   band; `bank_context_available` is 1 on cached rows), in `training.py`
   (classifiers see them automatically), or measure their increment with
   `evaluate_feature_ablation` by listing them in `research.feature_columns`.
5. On submission the host computes every referenced descriptor on the other
   brains' cached contexts for frozen inference; you never see those values.

## Rules

- `describe` receives only the context: no files, network, labels, GT, absolute
  coordinates or persistent IDs. It runs twice per row and must return identical
  values; infinities are rejected, NaN is allowed.
- Rows outside the band have NaN in every `bank_agent_*` column; coverage is
  reported, never imputed.
- A descriptor that fails on any batch registers nothing and the unit is spent.
- Label-conditional quartiles come from TRAIN brains only; the selection brain's
  out-of-fold measurement and the outer gate decide whether a quantity helps.

## Examples

`artifacts/descriptors/` holds readable `describe` examples ported from
descriptors earlier agents wrote, with the outcome each one measured. They are
starting points, not precomputed columns.
