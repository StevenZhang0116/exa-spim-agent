# Agent descriptors: measure any quantity over the cached candidates in one call

The host caches, once per brain and kind, the raw context of the top detector-ranked
candidates (the band; its size per brain is `band_rows` in `model_environment.json`,
50,000 rows in the current build, 20,000 before 2026-10-08) and, where the cache was
built with the labeled extension, of the GT-labeled candidates outside the band
(`extension_rows`): the fragment neighbourhood (50 um, up to 256 nodes)
and image patches for every cached row at two resolutions (`level1`: level 1,
30 um radius, about 1.5 x 1.5 x 2 um voxels; `level0`: level 0, 16 um radius,
about 0.75 x 0.75 x 1 um voxels, the finest view of thin fibres). These tiers are
defaults: a descriptor may request another radius and native pyramid level on demand.
Geometry stays at 50 um / 256 nodes. Nothing in the cache is a feature.
You decide what to measure by writing `descriptor.py`; the host runs it over
the cached rows in parallel sandboxed workers and registers the results as
`bank_agent_<name>` predictor columns.

## Workflow

1. Edit `descriptor.py`: a literal `DESCRIPTOR` dictionary with `kind`, `inputs`,
   `feature_names` and optional `image` or `image_tier`, plus `describe(context) -> dict`. `inputs` is `geometry`,
   `image` or `both`; `kind` must be this generation's kind; at most 32 names
   matching `[a-z][a-z0-9_]{1,48}`.
2. Probe the current code with `probe_descriptor({"candidate_refs": ["ctx:..."], "previews": true})`
   for up to 16 explicit TRAIN cases, or `probe_descriptor({"sample_per_brain": 128, "seed": 0})`
   for a reproducible uniform sample from each TRAIN brain's cached rows. Sampling
   does not use detector rank or labels. Optional previews use the exact input pixels
   and cached geometry, sharing the eight-preview allowance; default probes return
   numerical diagnostics. Reports include input shape, spacing, validity, geometry
   coverage, finite/missing/nonzero counts, min/max/std, quartiles and example handles.
   Zero quartiles alone do not imply a constant column. Full sampled values are saved
   under `descriptor_probes/`; the tool returns a few representative cases per brain.
   Drafts may reuse registered names during probing; production registration still
   rejects name conflicts. Probes neither register features nor spend scoring units.
   After revising, `plan_descriptor_run({})` times 64 uniform rows per TRAIN brain and
   estimates the entire TRAIN workload, including preparation. It also reports the
   uncached work remaining; only that work is compared with the remaining allowance.
   IO and worker scaling
   make this approximate, especially on a cold cache; frozen inference is additional
   and is extrapolated separately from workload sizes and TRAIN timings, without reading
   heldout images. Both tools charge elapsed descriptor wall time,
   including failed executions. These steps are guidance, not a mandatory gate.
3. `compute_descriptors({"scope": "pilot"})` (the 512 top-ranked rows), `"boundary"`
   (2,000 band rows on each side of rank K) or `"all_cached"` (every cached row: the
   band plus the labeled extension). One
   uncached call costs one evaluation unit plus wall time from the per-call and
   per-generation descriptor budgets in `model_environment.json`. The reply
   gives, for each column on each TRAIN brain, the diagnostics above and quartiles conditional on labels
   1 and 0 (NaN-label rows are only counted, as `unlabeled_count`),
   finite fraction and coverage. Results are cached by code, inputs and implementation
   identity. An unchanged valid cache costs no evaluation unit or descriptor wall budget.
   In no-score/raw modes, production calls must use `all_cached`; sampled probes do
   not become sparse predictor columns.
4. Use the columns like any predictor: in a formula (handle NaN on rows outside
   the cache; `bank_context_available` is 1 inside the detector band), in `training.py`
   (classifiers see them automatically), or measure their increment with
   `evaluate_feature_ablation` by listing them in `research.feature_columns`. Under
   `--detector-prior raw` the base predictors are only the `raw_*` candidate attributes, so
   every other feature comes from descriptors like these.
5. On submission the host computes every referenced descriptor on the other
   brains' cached contexts for frozen inference; you never see those values.
   Worker stdout/stderr go to the host-only `validation_descriptor_runs/` tree.
   That tree and legacy `descriptor_runs/inference/` logs are unreadable to agents
   in every generation; TRAIN `descriptor_runs/run*/` logs remain readable.

## Rules

- For image inputs, keep `image_tier: 'level1'` / `'level0'`, or replace that field with
  `image: {'level': 0, 'radius_um': 40}`. Radius must be 4..100 um and level an integer
  0..6 available in the source pyramid. Channel/timepoint remain zero; anchor/midpoint
  centering, occurrence zero and the geometry neighborhood are unchanged. Conflicting
  explicit tier/image declarations fail. Matching cached tiers are reused; other patches
  use the reviewed registered image source. Probes, full TRAIN and frozen inference
  share this exact input path. Large patches are bounded by the existing pixel limits
  and a 128 MiB batch limit. Full custom patches use temporary staging, not persistent
  accumulation in the 20 GiB patch cache.
- Image-consuming descriptor ablations report the removed descriptor group. Since
  `describe` also receives geometry/features, this does not by itself isolate a pixel-only effect.

- `describe` receives only the context: no files, network, labels, GT, absolute
  coordinates or persistent IDs. It runs twice per row and must return identical
  values; infinities are rejected, NaN is allowed.
- Rows outside the cache have NaN in every `bank_agent_*` column; coverage is
  reported, never imputed.
- A descriptor that fails on any batch registers nothing and the unit is spent.
- Label-conditional quartiles come from TRAIN brains only; the selection brain's
  out-of-fold measurement and the outer gate decide whether a quantity helps.

## Examples

`artifacts/descriptors/` holds readable `describe` examples ported from
descriptors earlier agents wrote, with the outcome each one measured. They are
starting points, not precomputed columns.
