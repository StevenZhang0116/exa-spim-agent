# Agent descriptors: measure quantities over native candidates on demand

The host reads fragment neighbourhoods and original image crops when requested.
No precomputed context directory or detector-ranked cache band is required. Every
native candidate is eligible, including candidates outside the former cache.
Geometry uses a 50 um neighbourhood with up to 256 nodes. Set `DESCRIPTOR.image`
to choose the image radius and native pyramid level; legacy `image_tier` names are
physical-request aliases (`level1`: level 1 / 30 um; `level0`: level 0 / 16 um).
Write `describe(context)` to define features. Parallel sandboxed workers compute
`bank_agent_<name>` columns; only numeric descriptor results persist for reuse.

Resource controls (2026-11-01): `model_environment.json` records the descriptor
worker address-space cap (default 16384 MiB, independent of classifier memory).
The configured worker count is a maximum, also capped by CPU affinity. Before
preparing inputs, the host admits batches using current job cgroup memory,
reserving max(8 GiB, 10% of its limit), plus the worker cap and 512 MiB of host
preparation allowance per admitted batch. The first batch runs alone so subsequent
admission includes the loaded fragment graph. Memory is checked again before
worker staging. Planning reports measured memory capacity and includes the serial
first batch; frozen-inference estimates use the lowest observed TRAIN concurrency,
but another brain's graph may need more memory. Waiting consumes the existing wall
budget. Insufficient room for even one batch fails explicitly. These conservative
checks reduce risk; arbitrary host/worker allocation spikes can still exceed memory.

## Workflow

1. Edit `descriptor.py`: a literal `DESCRIPTOR` dictionary with `kind`, `inputs`,
   `feature_names` and optional `image` or `image_tier`, plus `describe(context) -> dict`. `inputs` is `geometry`,
   `image` or `both`; `kind` must be this generation's kind; at most 32 names
   matching `[a-z][a-z0-9_]{1,48}`.
   Call `check_descriptor({})` before loading any graph: it checks syntax, the
   `describe(context)` signature, declaration and registered-name conflicts without
   running the program or charging a budget. Use names prefixed by kind and mechanism,
   e.g. `split_path_dark_fraction`, and new names when changing registered code.
   `ready` means the static contract passes, not that imports or numerical code work.
   `plan_descriptor_run` and production computation reject conflicts before graph IO;
   probes still allow conflicting draft names for exploratory comparisons.
2. Probe the current code with `probe_descriptor({"candidate_refs": ["ctx:..."], "previews": true})`
   for up to 16 explicit TRAIN cases, or `probe_descriptor({"sample_per_brain": 128, "seed": 0})`
   for a reproducible uniform sample from each TRAIN brain's complete candidate pool. Sampling
   does not use detector rank or labels. Optional previews use the exact input pixels
   and matching geometry, sharing the configurable per-generation preview allowance
   (default 32; see `model_environment.json`); default probes return
   numerical diagnostics. Reports include input shape, spacing, validity, geometry
   coverage, finite/missing/nonzero counts, min/max/std, quartiles and example handles.
   Zero quartiles alone do not imply a constant column. Full sampled values are saved
   under `descriptor_probes/`; the tool returns a few representative cases per brain.
   One reply inlines at most 64 MiB and one generation at most 96 MiB of base64-encoded
   PNG data; `preview_payload` lists previews that stayed on disk in the probe directory
   (readable one at a time) and the remaining generation bytes.
   Previews default to native slice sheets with separate raw and depth-clipped overlay rows,
   not maximum projections. With `previews: true`, pass `preview` to choose `plane`,
   `anchor_index`, `depth_range_um` and `slice_count` (1..9), as in image_context_guide.md.
   These view choices do not change descriptor inputs or the values being tested. Actual
   slice depths and skipped planes are recorded; `preview: {"view": "overview"}` requests
   the projection overview explicitly. Check apparent connections using full 3D pixels.
   Drafts may reuse registered names during probing; production registration still
   rejects name conflicts. Probes neither register features nor spend scoring units.
   For a new image hypothesis, inspect 2-4 matched cases across at least two TRAIN
   brains when available, including positive and negative controls. Request native
   slice previews of the exact inputs and check the hypothesis numerically in 3D.
   If previews are disabled/unavailable, record that limitation and use numerical
   controls. Then probe uniformly across TRAIN brains (default 128 per brain).
   Expand to 512 or 1024 only to resolve a stated uncertainty within wall time
   (maximum 2048). Keep cases or the same sample size and seed fixed for
   code comparisons; larger samples need not contain earlier samples. Previews are
   optional and the allowance need not be exhausted. See image_context_guide.md for
   the comparison strategy; probes do not replace OOF evaluation or paired ablations.
   `plan_descriptor_run({})` reuses a valid timing sample from a preview-free uniform probe
   of at least 64 rows per brain (or the entire smaller pool), or from an earlier plan.
   Explicit matched cases and previews are not used for cost calibration. Changed descriptor
   code, inputs or implementation invalidate reuse. Without a sufficient sample it times a
   contiguous block of 64 labeled rows per TRAIN brain. Uniform timings can overestimate
   contiguous production IO; `measurement_source` and `measurement_reused` make this visible.
   Every plan refreshes memory admission, scope counts, bank hits and remaining budgets.
   It retains each brain's cold fragment graph load (`graph_load_seconds`, charged
   once per brain) and estimates the entire TRAIN workload; image descriptors are credited
   with at most 8 concurrent readers (`workers`) as a conservative planning cap. It also
   reports the uncached work remaining; only that work is compared with the remaining allowance.
   IO and worker scaling
   make this approximate, especially on a cold cache; frozen inference is additional:
   `frozen_inference` extrapolates it for every allowed scope from the other brains' row
   counts and the TRAIN timings, without reading heldout images, and flags a scope whose
   largest single brain exceeds the per-call budget (each brain has that budget at
   submission, with the scope you computed with). Both tools charge elapsed descriptor wall time,
   including failed executions. These steps are guidance, not a mandatory gate.
   Calls process an already resident TRAIN fragment graph first when possible;
   the host still retains only one graph, without a saved context/image cache.
   This saves a boundary reload, not all six graph loads. Graph identities are
   still verified on access, and every requested brain/row is processed.
3. `compute_descriptors({"scope": "judgeable"})` (the default) covers the rows the outer
   measurement can score: the GT-labeled rows used for supervised fitting and metric ranking.
   TRAIN, selection and validation use this rule, so every ranked row is selected for
   descriptor computation. Rows outside the scope keep NaN. They do not directly participate
   in metric ranking, but can affect scores indirectly through whole-column operations in
   a formula or `predict`, or through `X_unlabeled` in `training.py`. Under `all_candidates`,
   these paths also receive computed descriptor values for the otherwise uncomputed rows.
   Row-wise formulas preserve the evaluated scores between these scopes. For classifiers,
   equivalence additionally requires fitting only labeled rows, using preprocessing fitted
   on TRAIN and frozen for inference, and making each `predict` result independent of other
   input rows. The arbitrary Python prediction interface does not enforce these conditions;
   fitting only labeled rows is insufficient. The scope you compute with is also the
   scope of frozen inference. `all_candidates` covers
   every native candidate (far more rows; check `plan_descriptor_run` first). `pilot` is a
   reproducible uniform sample of up to 512 rows; `boundary` explicitly chooses up to 2,000
   detector-rank neighbours on each side of K. The old `all_cached` scope is an alias for
   `all_candidates`. One
   uncached call costs one evaluation unit plus wall time from the per-call and
   per-generation descriptor budgets in `model_environment.json`. The reply
   gives, for each column on each TRAIN brain, the diagnostics above and quartiles conditional on labels
   1 and 0 (NaN-label rows are only counted, as `unlabeled_count`),
   finite fraction and coverage. Results are cached by code, inputs and implementation
   identity. An unchanged valid cache costs no evaluation unit or descriptor wall budget.
   In no-score/raw modes, production calls use `judgeable` or `all_candidates`; sampled probes do
   not become sparse predictor columns.
4. Use the columns like any predictor: in a formula (handle NaN on uncomputed rows), in `training.py`
   (classifiers see them automatically), or measure their increment with
   `evaluate_feature_ablation` by listing them in `research.feature_columns`. Under
   `--detector-prior raw` the base predictors are only the `raw_*` candidate attributes, so
   every other feature comes from descriptors like these.
5. On submission the host computes every referenced descriptor on the other
   brains' on-demand contexts for frozen inference; you never see those values.
   Worker stdout/stderr go to the host-only `validation_descriptor_runs/` tree (a batch that
   succeeds without printing leaves no file).
   That tree and legacy `descriptor_runs/inference/` logs are unreadable to agents
   in every generation; TRAIN `descriptor_runs/run*/` logs remain readable.

## Rules

- Context geometry and image fields can be NumPy arrays. Check `value is None`,
  `len(value)` or `value.size` explicitly; `if value` and `value or []` fail for
  multi-element arrays. Static checks do not execute numerical code: use a small
  probe to verify these paths before production computation.
- For image inputs, keep `image_tier: 'level1'` / `'level0'`, or replace that field with
  `image: {'level': 0, 'radius_um': 40}`. Radius must be 4..128 um (a patch spans at most a
  256 um cube at any level) and level an integer 0..6 available in the source pyramid. Reading a
  patch costs about the same whatever its size (0.15-0.25 s, a few 128x256x256-voxel chunks), but
  bytes grow with the voxel count: a 256 um cube is 0.9 MB at level 2, 7 MB at level 1 and 60 MB at
  level 0, and a batch holds at most 128 MiB of patches, so level-0 cubes run one row per worker
  batch; `plan_descriptor_run` measures the result. Channel/timepoint remain zero; anchor/midpoint
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
