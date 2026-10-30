# Agent descriptors: measure quantities over native candidates on demand

The host reads fragment neighbourhoods and original image crops when requested.
No precomputed context directory or detector-ranked cache band is required. Every
native candidate is eligible, including candidates outside the former cache.
Geometry uses a 50 um neighbourhood with up to 256 nodes. Set `DESCRIPTOR.image`
to choose the image radius and native pyramid level; legacy `image_tier` names are
physical-request aliases (`level1`: level 1 / 30 um; `level0`: level 0 / 16 um).
Write `describe(context)` to define features. Parallel sandboxed workers compute
`bank_agent_<name>` columns; only numeric descriptor results persist for reuse.

## Workflow

1. Edit `descriptor.py`: a literal `DESCRIPTOR` dictionary with `kind`, `inputs`,
   `feature_names` and optional `image` or `image_tier`, plus `describe(context) -> dict`. `inputs` is `geometry`,
   `image` or `both`; `kind` must be this generation's kind; at most 32 names
   matching `[a-z][a-z0-9_]{1,48}`.
2. Probe the current code with `probe_descriptor({"candidate_refs": ["ctx:..."], "previews": true})`
   for up to 16 explicit TRAIN cases, or `probe_descriptor({"sample_per_brain": 128, "seed": 0})`
   for a reproducible uniform sample from each TRAIN brain's complete candidate pool. Sampling
   does not use detector rank or labels. Optional previews use the exact input pixels
   and matching geometry, sharing the eight-preview allowance; default probes return
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
3. `compute_descriptors({"scope": "all_candidates"})` covers every native candidate.
   `pilot` is a reproducible uniform sample of up to 512 rows; `boundary` explicitly
   chooses up to 2,000 detector-rank neighbours on each side of K. The old
   `all_cached` scope is accepted as an alias for `all_candidates`. One
   uncached call costs one evaluation unit plus wall time from the per-call and
   per-generation descriptor budgets in `model_environment.json`. The reply
   gives, for each column on each TRAIN brain, the diagnostics above and quartiles conditional on labels
   1 and 0 (NaN-label rows are only counted, as `unlabeled_count`),
   finite fraction and coverage. Results are cached by code, inputs and implementation
   identity. An unchanged valid cache costs no evaluation unit or descriptor wall budget.
   In no-score/raw modes, production calls must use `all_candidates`; sampled probes do
   not become sparse predictor columns.
4. Use the columns like any predictor: in a formula (handle NaN on uncomputed rows), in `training.py`
   (classifiers see them automatically), or measure their increment with
   `evaluate_feature_ablation` by listing them in `research.feature_columns`. Under
   `--detector-prior raw` the base predictors are only the `raw_*` candidate attributes, so
   every other feature comes from descriptors like these.
5. On submission the host computes every referenced descriptor on the other
   brains' on-demand contexts for frozen inference; you never see those values.
   Worker stdout/stderr go to the host-only `validation_descriptor_runs/` tree.
   That tree and legacy `descriptor_runs/inference/` logs are unreadable to agents
   in every generation; TRAIN `descriptor_runs/run*/` logs remain readable.

## Rules

- For image inputs, keep `image_tier: 'level1'` / `'level0'`, or replace that field with
  `image: {'level': 0, 'radius_um': 40}`. Radius must be 4..100 um and level an integer
  0..6 available in the source pyramid. Channel/timepoint remain zero; anchor/midpoint
  centering, occurrence zero and the geometry neighborhood are unchanged. Conflicting
  explicit tier/image declarations fail. All requested crops use the reviewed registered image source. Probes, full TRAIN and frozen inference
  share this exact input path. Large patches are bounded by the existing pixel limits
  and a 128 MiB batch limit. Descriptor probes and full computations use temporary patch staging and remove it
  after use. Existing per-run image-cache hits from other inspection tools may be reused.
  Cold source IO can reduce the number of computations fitting the wall budget; use
  plan_descriptor_run before a large call. CPU parallelism does not remove IO costs.
- Image-consuming descriptor ablations report the removed descriptor group. Since
  `describe` also receives geometry/features, this does not by itself isolate a pixel-only effect.

- `describe` receives only the context: no files, network, labels, GT, absolute
  coordinates or persistent IDs. It runs twice per row and must return identical
  values; infinities are rejected, NaN is allowed.
- Rows outside a requested subset have NaN in its `bank_agent_*` columns; coverage is
  reported, never imputed.
- A descriptor that fails on any batch registers nothing and the unit is spent.
- Label-conditional quartiles come from TRAIN brains only; the selection brain's
  out-of-fold measurement and the outer gate decide whether a quantity helps.

## Examples

`artifacts/descriptors/` holds readable `describe` examples ported from
descriptors earlier agents wrote, with the outcome each one measured. They are
starting points, not precomputed columns.
