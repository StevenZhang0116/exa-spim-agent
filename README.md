# exa-spim-agent

This repository builds labeled ExaSPIM skeleton caches and checks them against
the official
[`segmentation-skeleton-metrics`](https://github.com/AllenNeuralDynamics/segmentation-skeleton-metrics)
results.

## Pipeline

```text
same source brain + segmentation
        |
        +--> load_skeletons.py --> base .pkl --> relabel_cache.py --> _add.pkl -----+
        |                                                                         |
        +--> evaluate_skeleton_metrics.ipynb --> official results.csv ------------+
                                                                                  |
                                                                                  v
                                                             verify_add_cache_metrics.py
```

Run the pipeline in the `panda` conda environment. Cloud-reading steps require
GCS credentials in `configs/zihan_gcs_token.json`.

### 1. Generate the base cache

Run [`notebooks/load_skeletons.py`](notebooks/load_skeletons.py), selecting the
brain and minimum cable length with command-line arguments, for example:

```bash
python notebooks/load_skeletons.py --brain-id 794495 --min-cable-length 10
```

It reads the GT and UNet-fragment SWCs, constructs two `SkeletonGraph` objects,
and writes:

```text
cache/dataset_cache_<brain>_mcl<N>.pkl
```

This base cache contains skeletons but no baked-in error labels.

### 2. Generate the official reference metrics

Open and run
[`notebooks/evaluate_skeleton_metrics.ipynb`](notebooks/evaluate_skeleton_metrics.ipynb)
for the same brain and segmentation.

It runs the official `segmentation-skeleton-metrics` package and writes:

```text
metrics_out/<brain>/<segmentation_id>/results.csv
metrics_out/<brain>/<segmentation_id>/merge_sites.csv
```

This step is independent of the cache. It produces the canonical answers used
to validate the labeled cache later.

### 3. Generate the labeled `_add.pkl`

Run from the repository root:

```bash
python scripts/relabel_cache.py \
  --cache-dir cache \
  --brain 794495 \
  --mcl 100
```

To label every matching base cache, omit `--brain` and `--mcl`:

```bash
python scripts/relabel_cache.py --cache-dir cache
```

The script reads the dense segmentation, labels GT nodes and edges, identifies
merge labels and sites, and writes a new file without modifying the base cache:

```text
cache/dataset_cache_<brain>_mcl<N>_add.pkl
```

The added fields are:

- `gt_node_canonical_label`
- `gt_edge_error`
- `gt_merge_labels`
- `gt_merge_sites`

See [`markdowns/labeled_dataset_cache.md`](markdowns/labeled_dataset_cache.md)
for the complete schema.

### 4. Verify the labeled cache

Run [notebooks/verify_add_cache_metrics.py](notebooks/verify_add_cache_metrics.py)
from the repository root on a compute node. By default it processes every brain
with an `_add.pkl` at the requested MCL, one cache at a time:

```bash
srun --partition=aibs_debug --constraint=cpu \
  --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=100G --time=01:00:00 \
  --chdir="$PWD" "$HOME/.conda/envs/panda/bin/python" -u \
  notebooks/verify_add_cache_metrics.py --mcl 100
```

Use `--mcl 10` for mcl10, or add `--brain 794495 794493` to select brains.
`--cache-dir`, `--metrics-dir` and `--output-dir` override the repository defaults.
Memory and time requests may need increasing for larger caches or more brains.
Brains missing a canonical `results.csv` are reported as skipped without loading
their caches. Multiple reference runs for one brain are rejected rather than
silently choosing one. Other brains continue after a failure; errors and hard
consistency failures produce exit code 1, while metric warnings alone do not.

The script reconstructs per-neuron split, omit, merge, and edge-accuracy
metrics using only the stored labels, then compares them with the official
`results.csv` from step 2. It opens no image readers, requires no cloud credentials,
and never modifies the caches or reference results. It overwrites matching
`<brain>_mcl<N>_per_neuron.csv`, `_summary.csv`, and `_scatter.png` files under:

```text
notebooks/verify_stats/
```

Canonical `# Merges` comparisons use geometric-walk sites only. Supplemental
two-GT-only, shared-evidence and combined site counts are exported separately;
combined counts are not counts of independent biological merge events. Missing
site data remains unavailable, not zero. Existing columns remain compatible with
[notebooks/compare_add_cache_metrics_across_datasets.py](notebooks/compare_add_cache_metrics_across_datasets.py).

Expect close agreement, not byte-for-byte equality. Small differences remain
because the cached graphs are resampled and the merge-site implementations have
minor snapping and deduplication differences. For a detailed merge-site check,
run [`notebooks/verify_add_cache_merge_info.py`](notebooks/verify_add_cache_merge_info.py).

### 5. Compare brains and MCLs

Once verification CSVs are available, run the lightweight comparison script on
a compute node. It does not load skeleton caches, access cloud data, or rewrite
verification CSVs:

```bash
srun --partition=aibs_debug --constraint=cpu \
  --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=3G --time=00:05:00 \
  --chdir="$PWD" "$HOME/.conda/envs/panda/bin/python" -u \
  notebooks/compare_add_cache_metrics_across_datasets.py
```

All generated figures are saved in `notebooks/verify_stats/`:

- `compare_mcl100_across_brains.png`
- `compare_mcl10_across_brains.png`
- `compare_<brain>_mcl100_vs_mcl10.png` for each brain with both MCLs

Missing MCLs are reported and skipped. Use `--stats-dir` for another input
directory; figures go there by default, or to `--output-dir` if provided.
`--dpi` defaults to 160. Re-running overwrites matching comparison PNGs but
leaves the verifier's `<brain>_mcl<N>_scatter.png` files and CSVs untouched.
Legacy tables remain supported via their MCL column or matching summary; MCL-named
tables take precedence over duplicate legacy rows. The plots reflect the saved
CSVs, so rerun verification first if the underlying caches have changed.

### 6. Inspect cached GT merge sites

[notebooks/plot_gt_merge_sites.py](notebooks/plot_gt_merge_sites.py) samples K
distinct positions directly from `gt_merge_sites` and saves one 3x3 image per
position: Image MIP / GT / Fragments rows, with XY / XZ / YZ columns. It reuses
the prediction plotter's coordinate-based renderer, including 3D boundary
clipping, and needs no detector CSV, scores or model.

```bash
srun --partition=aibs_debug --constraint=cpu \
  --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=100G --time=00:15:00 \
  --chdir="$PWD" "$HOME/.conda/envs/panda/bin/python" -u \
  notebooks/plot_gt_merge_sites.py --brain-id 794495 --k 10 --mcl 100
```

The image reader uses the cache's `img_path`; cloud access is needed for image
patches. No dense segmentation mask is read: the third row is the fragment
skeleton, with the site's segment highlighted in cyan. The yellow `+` marks
the exact stored site coordinate, not a snapped candidate. Both geometric and
two-GT sites are included, without requiring two GT neurons in the patch.
These are rule-derived labels, not independently verified cut locations.

- `--seed 0` is the default reproducible sample; change it for other positions.
- `--source geometric_walk` or `--source two_gt_junction` restricts the source;
  shared-evidence sites qualify for either. Default is `all`.
- `--patch-um 100 100 100` sets the XYZ field of view in micrometers (the default).
- `--cache-dir` and `--dpi` override the cache directory and output resolution.
- Default output is a new run directory in `figs/gt_merge_sites/<brain>_mcl<N>/`.
  `--output-dir` can select another directory but it must be empty.

Selection deduplicates exact XYZ positions, then samples without replacement.
Different centers may still have overlapping views. If fewer than K unique
positions exist, all available positions are rendered with a warning. Coincident
records retain all matching site indices and segment/neuron IDs in the manifest;
only the first record's segment is highlighted. Original site indices are
zero-based and refer to the input cache.

Every run saves all PNGs plus `selection.json` (seed, inputs and chosen positions)
and `gt_merge_sites.csv` (coordinates, sources, patch bounds and filenames).
Cache files, prediction outputs and existing figures are not modified. If a run
fails, the manifest contains the successfully saved figures; use a fresh output
directory when retrying. Only the requested brain is loaded once per run.

### 7. Compare GT merge sites and positive candidates

[notebooks/plot_positive_merge_sites.py](notebooks/plot_positive_merge_sites.py)
creates **one global comparison figure with two panels, XY and XZ**, in
**millimeters**. Blue open circles show every cached `gt_merge_sites` record;
smaller red dots show every `is_merge_site=1` candidate in the supplied junction
CSV, so both colors remain visible at coincident positions. Faint gray dots show
all candidate positions for context, not an anatomical brain outline. There is
no score/rank threshold, finite-score requirement, GT-visibility filter, sampling
or deduplication of either set. Several positive candidates may correspond to one
GT site; overlapping markers do not imply one-to-one matching or independent events.

From the repository root, on a compute node in panda:

```bash
APP=autodiscovery-application/merge-error-794495-mcl100_2026-08-04-rebuild-20260923-160112
srun --partition=aibs_debug --constraint=cpu \
  --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=100G --time=00:15:00 \
  --chdir="$PWD" "$HOME/.conda/envs/panda/bin/python" -u \
  notebooks/plot_positive_merge_sites.py --brain-id 794495 --mcl 100 \
  --csv "$APP/merge_junction_detector_794495.csv" --cache-dir cache
```

Only load trusted pickle files. `--brain-id`, `--mcl` and `--cache-dir` select
`dataset_cache_<brain>_mcl<N>_add.pkl`; the stored MCL is checked when present.
The whole cache must be deserialized to extract its GT sites, requiring much more
memory than the CSV-only plot. No image volume is read and no model is run.
Candidate labels/coordinates are taken directly from the CSV, without recalculation
or validation against cache geometry. Choose a matching experiment/cache, especially
after refreshing labels. Use the GT-site sampler above for local image inspection.

The single PNG, `selection.json` (CSV SHA256, cache identity, counts and positive
candidate IDs), `positive_merge_sites.csv` and `gt_merge_sites.csv` are saved under
`figs/positive_merge_sites/<brain>_mcl<N>/comparison_<suffix>/`. Both coordinate
CSVs retain original XYZ micrometers; the GT export preserves each record's site
index and segment ID. Use an empty `--output-dir` to override that location.
Inputs and earlier outputs are never overwritten. Either set may be empty and
is marked as such; no figure is generated only when both the candidate CSV and
GT site list are empty. Use a fresh directory to retry a failed run.

### 8. Log GT sites and positive candidates across datasets

[notebooks/log_merge_site_counts.py](notebooks/log_merge_site_counts.py) discovers
all `dataset_cache_*_mcl<N>_add.pkl` files for one MCL and processes them sequentially.
It reuses the reviewed detector target adapter to recompute candidate labels from
stored GT sites, ignoring any cached candidate universe. No detector CSV, trained
model or image access is needed, and no cache is modified. Only load trusted pickles.

From the repository root, run on a compute node in panda:

```bash
srun --partition=aibs_debug --constraint=cpu \
  --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=150G --time=00:20:00 \
  --chdir="$PWD" "$HOME/.conda/envs/panda/bin/python" -u \
  notebooks/log_merge_site_counts.py --mcl 100
```

Change `--mcl` for another level, or add `--brains 794495 794493` to select datasets.
Defaults are `--nms-um 20 --positive-radius-um 20 --claim-radius-um 150`, with
50 um snapping to each GT site's own segment. These explicit analysis settings
match the current junction policy; they do not select a new policy or guarantee
matching results from a detector built with different settings. Distances after
snapping are cable distances; NMS uses segment-scoped Euclidean distance.

Each run writes a fresh `notebooks/merge_site_counts/mcl<N>_<suffix>/` directory:

- `summary.csv`: one row per dataset, including status and any error.
- `run.json`: settings, adapter/script hashes and input paths.
- `per_brain/<brain>.json`: cache identity, source provenance and per-GT nearest
  candidate distances. A null distance means no candidate found within the claim
  radius or an unsnappable site, not a measured infinite physical distance.

Key summary columns use distinct counting units:

| Column | Meaning |
| --- | --- |
| `n_gt_merge_sites` | All stored GT site records, without deduplication |
| `n_is_merge_site_1` | Kept candidates within the positive radius of any GT site |
| `n_in_ambiguous_ring_1` | Candidates outside the positive radius but within the claim radius |
| `n_gt_sites_covered_at_positive_radius` | GT records with at least one candidate within the positive radius |
| `n_gt_sites_only_within_claim_radius` | GT records whose nearest candidate is outside the positive radius but within the claim radius |
| `n_gt_sites_not_covered_at_claim_radius` | GT records without a candidate within the claim radius, including unsnappable sites |

Ring candidates are included in `n_negative`; label 0 is not a verified non-merge.
Counts need not match one-to-one, and this logger does not export all GT-candidate
associations. Missing GT labels are reported as errors, not zero. Empty GT lists
are valid. Errors do not stop other datasets; any error makes the command exit 1.
Use an empty `--output-dir` to override the output location; previous runs are
never overwritten. The full caches must be loaded, so memory needs are much
higher than for reading the small output tables.

### 9. Evolve a scorer on fixed detector-native candidates

`proofreader_evolve` improves **native-label Precision@K on the
same candidate pool** as the selected `merge_junction_detector.py` and
`split_site_detector.py`. The frozen scripts define candidate enumeration,
including split occurrences and junction NMS; evolution cannot change the pool.
Exactly one detector per kind is required.

On an allocated compute node in panda:

```bash
python -u proofreader_evolve/prepare_feature_tables.py
python -m proofreader_evolve.cli.preflight --brains 789202 794491 --mcl 100
python -m proofreader_evolve.cli.run_evolution \
  --train-brains 789202 --validation-brains 794491 \
  --merge-k 100 --split-k 100 --generations 1
```

Preparation now uses `dataset_cache_<brain>_mcl100_add.pkl`, preserving the native
detector's annotation definition. Full native pools replace the previous capped
tables; these different schemas cannot be reused interchangeably. Preparation
can still be expensive, especially merge extraction. It does not call an LLM.

The editable scorer is seeded from `proofreader_evolve/artifacts/scorer.py`:
`score_candidates(features, ctx)` receives only predictor columns plus the frozen
`detector_score`; `ctx` contains the candidate kind. It must return one finite
score per row. The harness selects exactly min(K, pool size) with deterministic
tie breaking. GT labels are stored separately and never passed to the scorer.
Training feedback includes anonymous labeled examples; validation feedback is
not supplied to the reviser. Candidates must improve both training and validation
macro precision, with no brain/kind regression. Invalid, non-deterministic or
timed-out scorers are rejected without substituting parent measurements.

K is fixed per run. Both raw detector and evolved scorer are compared on identical
pools and budgets. Returning fewer candidates cannot improve the metric. No graph
edits, pool expansion, on-demand detector queries or post-cut re-enumeration occur.
These are sparse-annotation native labels, not exhaustive biological truth; a
better ranking does not establish safe graph repairs. Repeated validation is
development data, not a final untouched test. The SDK file guard is not an OS
sandbox for generated Python. See [workflow revision](proofreader_evolve/WORKFLOW_REVISION.md).

The default frozen detectors were fitted on 794495:

- Merge: `merge-error-794495-mcl100_2026-08-04-rebuild-20260923-160112`
- Split: `split-error-794495-mcl100-run-3_2026-08-24`

Pass one `--merge-dir` and one `--split-dir` to select other frozen deliverables.
Detector-fitted brains cannot be validation brains. The reviser requires
`ANTHROPIC_API_KEY`; preparation and baseline-only evaluation do not call an LLM.

There is only one evolution workflow now: fixed-pool scorer evolution. The
graph-edit driver, repair-site enumerators, repair templates and dedicated
analyses have been removed. Old run/cache data remain on disk but are not
supported by the current reporting tools. See the workflow document for the
remaining interfaces and integrity requirements.

## Main outputs

```text
cache/                 base and labeled pickle files
metrics_out/           official reference metrics
notebooks/verify_stats cache-versus-reference comparisons
```
