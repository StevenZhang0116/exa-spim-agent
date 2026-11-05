# Independent evaluation preparation

This directory holds local tools for preparing a future independent evaluation.
Brain **794495** remains excluded from current training, selection/OOF, tuning,
testing and evolution promotion. The source comparison below is a user-requested
data audit, not model evaluation or feedback for evolution.

## Compare repository and CodeOcean sources for 794495

`compare_794495_sources.py` makes a small, read-only comparison:

- Lists the current `voxel/` GT used by the repository's cache builder and the
  `world/` GT referenced by [CodeOcean capsule 6960813](../external/codeocean/capsule_6960813/README.md).
- Samples three shared neuron filenames spread across the sorted inventory.
  Matches SWC node IDs, converts voxel xyz with the cache builder's default
  `[0.748, 0.748, 1.0]` um spacing, and compares coordinates, parents and node types.
- Compares image metadata and one seeded random GT-node-centered patch per neuron
  (48 um per side, level 0, channel/timepoint 0 by default). Pixel comparison
  requires matching grids; the script does not infer registration or resample.
- Reports exact pixel equality, absolute differences, RMSE, Pearson correlation
  and intensity summaries. Three-plane PNGs use common intensity limits for each
  pair so brightness differences remain visible.

The repository image comes from `configs/image_alignment.json` via the usual
reviewed-source resolver. Capsule image and GT paths are read from the imported
Python source with AST parsing; the upstream programs are not executed.

Run in the `panda` environment **on a compute node**, from the repository root:

```bash
/home/zihan.zhang/.conda/envs/panda/bin/python -u \
  evaluation/compare_794495_sources.py --samples 3 \
  --out evaluation/results/794495_source_comparison
```

For an existing allocation, prefix the command with
`srun --jobid=<JOB_ID> --overlap --exact --nodes=1 --ntasks=1 --cpus-per-task=2 --nodelist=<NODE>`.
Normal GCS/S3 read access is required. Existing credential configuration is reused;
no credentials are written into reports.

Outputs are `comparison.json`, `report.md`, and `patch_*.png` in the chosen output
directory. `evaluation/results/` is ignored by Git. The report records source
URIs, sampled files, GT object generations/hashes, image metadata hashes, seed and
host. Image metadata hashes are **not** hashes of the whole image contents.

This checks current cloud GT sources, not the contents of existing `_add.pkl`
caches or the exported capsule's attached data assets. A few matching patches
cannot establish whole-volume equality. Different intensities do not by themselves
prove misregistration. No dense segmentation is compared and no cloud data,
alignment receipt, label, feature bank or evolution configuration is changed.

## First small comparison: 2026-11-04

Executed on **n06**, with the default three samples, seed 0, level 0 and 24 um
radius. The local outputs are in
`evaluation/results/794495_source_comparison_20261104/`.

- Both GT directories contained the same 22 SWC filenames. The sampled N001,
  N011 and N023 neurons contained 920,902 nodes in total. Their node IDs, parent
  IDs and types matched; maximum coordinate error after conversion was 0 um.
- Image shapes, spacing, translations and the two metadata hashes matched.
  All three patches were fully inside both volumes (202,800 voxels each).

| GT neuron used as image anchor | Pearson | Exactly equal voxels | Mean absolute intensity difference |
|---|---:|---:|---:|
| N001 | 0.993053 | 10.36% | 2.9916 |
| N011 | 0.984458 | 20.01% | 1.5439 |
| N023 | 0.973548 | 14.53% | 1.5511 |

These samples support matching GT geometry and strongly similar image structure,
but demonstrate that the image sources are not pixel-identical. They do not
identify the processing step responsible for the intensity differences or prove
equivalence everywhere. Source comparison alone does not justify replacing the
image URI used by existing features. Synthetic coordinate/pixel checks and
`pyflakes` passed on the compute node.
