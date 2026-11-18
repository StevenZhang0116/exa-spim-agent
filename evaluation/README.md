# Source audit for brain 794495: repository vs CodeOcean capsule

This directory holds one read-only data audit. It checks whether the ground-truth
tracings and the image that this repository uses for brain **794495** are the same data
as those used by [CodeOcean capsule 6960813](../external/codeocean/capsule_6960813/README.md)
(Arborist evaluation, imported 2026-11-04). No model is scored here, and nothing in
this directory feeds an evolution run.

It is a different thing from [`proofreader_evolve/evaluation/`](../proofreader_evolve/evaluation/README.md),
which records how the final policy of a finished evolution run scores on 794495.

| Path | Content | In Git |
|---|---|---|
| `compare_794495_sources.py` | the comparison script | yes |
| `README.md` | this file, with the result of the one comparison run so far | yes |
| `results/794495_source_comparison_20261104/` | `comparison.json`, `report.md`, `patch_01..03.png` of that run | no (`results/` is ignored) |

## Where 794495 stands

- It has no role inside an evolution run: not TRAIN, not selection / OOF, not
  development validation or promotion. Launch commands name their brains explicitly.
- Since 2026-11-14 (user decision) the final policy of a finished run is applied to it
  once, as post-analysis. Those figures and the rules of that step are in
  `proofreader_evolve/evaluation/README.md`; they are not fed back to any run.
- It is not an untouched brain. It was used in feature discovery and detector fitting,
  and as a TRAIN brain of evolution runs in 2026-10.
- A formal independent validation protocol has not been designed.

The audit below needs none of this: it compares data sources and reads no label,
feature table, score or model.

## Why the two sources were compared

The capsule evaluates the Arborist proofreading pipeline on 794495 with its own data
paths. The repository reaches the same brain through different ones:

| | Repository | Capsule |
|---|---|---|
| GT tracings | `gs://allen-nd-goog/ground_truth_tracings/794495/voxel` (cache builder) | `gs://allen-nd-goog/ground_truth_tracings/794495/world/` (`code/main.py`) |
| Image | `s3://aind-open-data/exaSPIM_794495_2026-01-21_14-25-07_processed_2026-07-28_08-24-11/fusion/fused.zarr/` (`configs/image_alignment.json`) | `gs://allen-nd-goog/from_aind/exaSPIM_794495_2026-01-21_14-25-07_training-data/whole-brain-denoised/fused.zarr/0/` (`code/review_merges/build_review.py`) |

Before a figure from one side is set beside a figure from the other, the two have to
describe the same neurons and the same pixels.

## What the script does

`compare_794495_sources.py` makes a small comparison and changes nothing:

- Lists the `voxel/` GT directory and the `world/` GT directory and compares the file
  names.
- Samples three shared neuron files spread across the sorted inventory. Matches SWC
  node IDs, converts voxel xyz with the cache builder's default spacing
  `[0.748, 0.748, 1.0]` um, and compares coordinates, parents and node types.
- Compares the image metadata and one seeded random GT-node-centered patch per sampled
  neuron (48 um per side, level 0, channel and timepoint 0 by default). Pixels are
  compared only when the two grids match; the script fits no registration and
  resamples nothing.
- Reports exact pixel equality, absolute differences, RMSE, Pearson correlation and
  intensity summaries. The three-plane PNGs use common intensity limits for each pair,
  so brightness differences stay visible.

The repository image comes from `configs/image_alignment.json` through the usual
reviewed-source resolver. The capsule's image and GT paths are read from the imported
Python source by AST parsing; the capsule's programs are not executed.

## Run it

In the `panda` environment **on a compute node**, from the repository root:

```bash
/home/zihan.zhang/.conda/envs/panda/bin/python -u \
  evaluation/compare_794495_sources.py --samples 3 \
  --out evaluation/results/794495_source_comparison
```

For an existing allocation, prefix the command with
`srun --jobid=<JOB_ID> --overlap --exact --nodes=1 --ntasks=1 --cpus-per-task=2 --nodelist=<NODE>`.
Normal GCS and S3 read access is required. Existing credential configuration is reused;
no credentials are written into the reports.

Options: `--samples` (default 3), `--radius-um` (24), `--level` (0), `--seed` (0),
`--out`. Outputs are `comparison.json`, `report.md` and `patch_*.png` in the output
directory. The report records the source URIs, the sampled files, the GT objects'
generations and hashes, the image metadata hashes, the seed and the host. The image
metadata hashes describe the headers, **not** the image contents.

## What it cannot show

- It reads the current cloud sources. It does not examine an existing `_add.pkl` cache
  or the data assets attached to the capsule on CodeOcean.
- A few matching patches do not establish that two volumes are equal everywhere, and
  the patches are centered on neurites, not drawn from the whole volume.
- Different intensities do not by themselves prove a registration error.
- No dense segmentation is compared.

## Result of 2026-11-04

Run on **n06** with the defaults (three samples, seed 0, level 0, 24 um radius).

**GT: the same tracings.** Both directories held the same 22 SWC file names. The
sampled neurons N001, N011 and N023 have 920,902 nodes in total; node IDs, parent IDs
and node types matched, and the largest coordinate error after conversion was 0 um.

**Image: the same grid, similar structure, different pixel values.** Shapes, spacing,
translations and the two metadata hashes matched, and all three patches lay fully
inside both volumes (202,800 voxels each).

| GT neuron used as image anchor | Pearson | Exactly equal voxels | Mean absolute intensity difference |
|---|---:|---:|---:|
| N001 | 0.993053 | 10.36% | 2.9916 |
| N011 | 0.984458 | 20.01% | 1.5439 |
| N023 | 0.973548 | 14.53% | 1.5511 |

So the two sides trace the same neurons, and their images show the same structure
without being pixel-identical. The comparison does not say which processing step
produces the intensity difference, and three patches do not prove equivalence
everywhere. It gives no reason to replace the image URI that the repository's features
are computed from: image descriptors read pixel values, and these differ.

Synthetic coordinate and pixel checks of the script and `pyflakes` passed on the
compute node.

## Since then

- The comparison has not been rerun. `comparison.json` keeps the GT objects'
  generations and hashes of 2026-11-04, so a later run can tell whether a tracing
  changed.
- The repository's image source for 794495 is still the S3 volume above (checked in
  `configs/image_alignment.json` on 2026-11-18; the alignment receipt was re-recorded
  on 2026-11-11 after a label-only refresh of the cache, which does not touch the
  image).
- Recorded on 2026-11-09 ([docs/stage3_evolution.md](../docs/stage3_evolution.md), image
  sources): that S3 `fused.zarr` is itself a denoised fusion (AIND denoise-net on the
  tiles before flat-field correction and fusion). Both images are therefore processed
  products of the same acquisition, which fits what the patches show; which steps
  differ between them is still not established.
