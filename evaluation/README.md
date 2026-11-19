# Brain 794495 and CodeOcean capsule 6960813: source audit and capsule metrics

This directory holds two read-only tools around
[CodeOcean capsule 6960813](../external/codeocean/capsule_6960813/README.md) (Arborist
evaluation, imported 2026-11-04) and brain **794495**:

1. a data audit: are the ground-truth tracings and the image that this repository uses
   the same data as the capsule's (2026-11-04);
2. the capsule's own skeleton metrics, computed on our inputs for the unproofread
   segmentation (2026-11-18).

Neither scores a model of this repository, and nothing here feeds an evolution run.

It is a different thing from [`proofreader_evolve/evaluation/`](../proofreader_evolve/evaluation/README.md),
which records how the final policy of a finished evolution run scores on 794495.

| Path | Content | In Git |
|---|---|---|
| `compare_794495_sources.py` | the source comparison script | yes |
| `run_capsule_original.py` | runs the capsule's `main.py` on our inputs, "original" step | yes |
| `README.md` | this file, with the results of both | yes |
| `results/794495_source_comparison_20261104/` | `comparison.json`, `report.md`, `patch_01..03.png` of the audit | no (`results/` is ignored) |
| `results/794495_capsule_original_20261118/` | the capsule's outputs, `run_info.json`, `run.log`, the per-neuron comparison | no |

## Where 794495 stands

- It has no role inside an evolution run: not TRAIN, not selection / OOF, not
  development validation or promotion. Launch commands name their brains explicitly.
- Since 2026-11-14 (user decision) the final policy of a finished run is applied to it
  once, as post-analysis. Those figures and the rules of that step are in
  `proofreader_evolve/evaluation/README.md`; they are not fed back to any run.
- It is not an untouched brain. It was used in feature discovery and detector fitting,
  and as a TRAIN brain of evolution runs in 2026-10.
- A formal independent validation protocol has not been designed.

Neither tool below touches this: the audit compares data sources, and the capsule
metrics describe the segmentation itself. Both read the ground truth of 794495; neither
reads a label table, a feature table, a score or a model of this repository.

## Source audit: repository vs capsule

### Why the two sources were compared

The capsule evaluates the Arborist proofreading pipeline on 794495 with its own data
paths. The repository reaches the same brain through different ones:

| | Repository | Capsule |
|---|---|---|
| GT tracings | `gs://allen-nd-goog/ground_truth_tracings/794495/voxel` (cache builder) | `gs://allen-nd-goog/ground_truth_tracings/794495/world/` (`code/main.py`) |
| Image | `s3://aind-open-data/exaSPIM_794495_2026-01-21_14-25-07_processed_2026-07-28_08-24-11/fusion/fused.zarr/` (`configs/image_alignment.json`) | `gs://allen-nd-goog/from_aind/exaSPIM_794495_2026-01-21_14-25-07_training-data/whole-brain-denoised/fused.zarr/0/` (`code/review_merges/build_review.py`) |

Before a figure from one side is set beside a figure from the other, the two have to
describe the same neurons and the same pixels.

### What the script does

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

### Run it

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

### What it cannot show

- It reads the current cloud sources. It does not examine an existing `_add.pkl` cache
  or the data assets attached to the capsule on CodeOcean.
- A few matching patches do not establish that two volumes are equal everywhere, and
  the patches are centered on neurites, not drawn from the whole volume.
- Different intensities do not by themselves prove a registration error.
- No dense segmentation is compared.

### Result of 2026-11-04

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

### Since the audit

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

## Capsule metrics of the unproofread segmentation

The capsule's main program computes skeleton metrics for every step of an Arborist
proofreading result that it reads from S3: the segmentation as it was, then after each
correction. We have no Arborist result. We do have what the first step needs: the
ground truth, the fragment SWCs and the segmentation. `run_capsule_original.py` runs the
capsule's `code/main.py` unmodified on them, which gives the capsule's "original" step.
The corrected steps, and the capsule's two other tools, need the output of an Arborist
run and are not covered.

It differs from a run of the capsule on an Arborist result in four ways (details in the
script's docstring):

1. No list of the segments the pipeline kept (`segment_ids.txt`), so no segment is
   dropped.
2. No list of reviewed false-positive merge sites (`ignored_merges.swc`), so no merge
   site is ignored.
3. The fragment SWCs of the Google segmentation hold voxel indices; the capsule expects
   um. They are converted before the capsule's loader sees them. Without this every edge
   counts as omitted (seen in the first test).
4. Fragment SWCs are parsed only for the segment IDs that label a ground-truth node. By
   the code this cannot change a result; it was not checked against an unrestricted run.

Run it on a compute node in the capsule's environment `arborist_eval` (never in
`panda`), from the repository root:

```bash
conda activate /allen/programs/mindscope/workgroups/auto-model/zihan.zhang/exaspim-agent/conda_envs/arborist_eval
python -u evaluation/run_capsule_original.py --brain 794495 \
    --out evaluation/results/794495_capsule_original
```

Run every ground-truth neuron together. A test with N001 alone counted 8 merges for it;
with all 22 neurons loaded the same neuron has 5. Counts of one neuron depend on which
other neurons are present.

### Result of 2026-11-18

n243, 19 minutes (4 of them labeling 7,094,284 ground-truth nodes, about 6 downloading
the 10,000 fragment archives). The ground-truth nodes fall in 12,941 segments, of which
3,980 have a fragment SWC. No archive failed.

Beside it, the repository's reference metrics of the same segmentation
(`metrics_out/794495/`, 2026-10-14), computed in `panda` with the older
`segmentation_skeleton_metrics` (commit `d76557d`) by `notebooks/evaluate_skeleton_metrics.py`:

| | capsule version (`478aede`) | reference (`d76557d`) |
|---|---:|---:|
| Merges, total | 210 | 198 |
| Splits, total | 4,656 | 12,687 |
| Truncations, total | 109 | not reported |
| % merged edges, mean per neuron | 27.65 | 29.48 |
| % omit edges | 2.92 | 1.86 |
| % split edges | 0.11 | 0.34 |
| Edge accuracy | 69.31 | 68.32 |

**Merge sites.** The totals are close, the sites less so. Matched one to one within
20 um (positions taken from the voxel columns; the reference file's `World` column has
its axes permuted), 147 sites are in both, 63 only in the capsule version and 51 only in
the reference. A matched pair is the same place: the median distance from a capsule
site to the nearest reference site is 1.7 um, and widening the radius to 100 um adds
four pairs. Per neuron the two merge counts correlate at 0.94 and are equal for 4 of 22
neurons. The merge search was rewritten between the two commits; this repository's
`length_v2` merge labels follow the older walk.

**Splits.** The capsule version reports about a third as many and has a separate count
of truncations. The two counts are not the same quantity.

**Edges.** For 19 neurons the shares of merged and omitted edges agree within about two
points. Omitted edges are higher throughout (by 1.1 points on average), as expected: the
capsule keeps a ground-truth node's label only when a skeleton node of the node's own
segment lies within 20 um, and two thirds of the segments under the ground truth have no
skeleton. Three neurons differ strongly in merged edges: N004 (3.2% against 30.6%), N019
(4.1% against 17.4%) and N017 (7.5% against 11.7%). The definition of a merged edge is
the same in both versions (a label shared by two neurons with more than 50 nodes each),
so the difference comes from the relabeling step; which nodes lost their label was not
traced. Until it is, do not read the edge-level figures of these three neurons as a
finding about the segmentation.

**Lengths.** Run length and ERL are not compared. The reference loaded voxel coordinates
without scaling, so its lengths are not in um (N001: 354,311 against 294,442).

The per-neuron table is `results/794495_capsule_original_20261118/comparison_with_reference_20261014.txt`.
