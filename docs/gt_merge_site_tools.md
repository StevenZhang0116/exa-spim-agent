# GT merge-site inspection tools

*Part of the [exa-spim-agent](../README.md) documentation.*

These tools read cached GT merge sites and detector outputs to inspect labels and
positives visually or in aggregate. They support Stage 2 result review and feed
no training step.

## B1. Inspect cached GT merge sites

[notebooks/plot_gt_merge_sites.py](../notebooks/plot_gt_merge_sites.py) samples K
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

The image reader uses the URI reviewed in `configs/image_alignment.json`, else the cache's
`img_path` (`scripts/dataset_config.get_reviewed_img_path`); cloud access is needed for image
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

## B2. Compare GT merge sites and positive candidates

[notebooks/plot_positive_merge_sites.py](../notebooks/plot_positive_merge_sites.py)
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
APP=autodiscovery-application/<RUN>   # a merge-site detector result directory
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

## B3. Log GT sites and positive candidates across datasets

[notebooks/log_merge_site_counts.py](../notebooks/log_merge_site_counts.py) discovers
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
Defaults are `--nms-um 20 --tip-gap-um 20 --tip-partner-k 2 --positive-radius-um 20
--claim-radius-um 150`, with 50 um snapping to each GT site's own segment: the merge
rule declared in `proofreader_evolve/candidate_pools.json`, enumerated by
`candidate_rules.py`. `--tip-gap-um 0` gives the junction-only rule used until
2026-10-16. These explicit analysis settings do not select a new policy or
guarantee matching results from a detector built with different settings. Distances after
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
| `n_tip_to_shaft_candidates` | Kept candidates of the tip-to-shaft family (0 under the junction-only rule) |
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
