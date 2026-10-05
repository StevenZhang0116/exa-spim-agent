# Candidate image access: sampled alignment review

Date: 2026-10-02. All Python, data reads, rendering and checks ran on n257.
No evolution, scientific model fitting or live LLM request was run.

The audit read 30 native candidate patches: three merge and three split candidates
per brain, selected from spatially separated positions without consulting labels.
All 30 saved three-plane views were visually inspected. Each patch uses level 0,
channel 0, timepoint 0 and an 80-um minimum field of view. All were inside the
image domain (`valid_fraction=1`). Cached fragment xyz coordinates are microns;
image arrays delivered to workers are zyx, with explicit OME scale/translation.

| Brain | Cases | Source / visual observation |
|---|---:|---|
| 794495 | 6 | Cached image source. Fragment trajectories generally follow fluorescence in the three projections; some dim or broad branches. |
| 789202 | 6 | Cached image source. Corresponding trajectories are visible despite strong noise/striping in some views. |
| 794491 | 6 | Explicit same-acquisition replacement source, described below. Central neurites correspond; peripheral merge cases contain strong block/fusion artifacts and provide weaker fine-alignment evidence. |
| 794493 | 6 | Cached image source. Main trajectories correspond; broad axial signal and occasional skeleton branches outside strong signal limit fine assessment. |
| 802449 | 6 | Cached image source. Main trajectories correspond; visible fusion boundaries and intensity changes in some cases. |

No consistent gross axis permutation, reflection or global offset was apparent
in these sampled views. This is a rough registration check, not a quantified
registration-accuracy result, a whole-brain guarantee or a biological error audit.
XY/XZ/YZ full-depth projections were checked together; projections alone can hide
depth errors. Thin-slab views and the synthetic coordinate tests provide additional
checks. Early saved previews mark projected anchors even outside the thin slab;
the final renderer clips both edges and anchors to the slab depth.

The diagnostic median intensity near existing skeleton edges exceeds the largest
of six 12-um shifted-control medians in 28 of 30 patches. The two exceptions are
the artifact-heavy peripheral 794491 merge patches. These measurements use a
3-voxel maximum filter and are descriptive; they are not an automatic acceptance
threshold or an estimate of biological correctness. See `verification.json` and
each brain's `alignment.json` for all measurements and source/cache identities.

## 794491 source override

The cached `processed_2025-10-26_12-28-14/fusion/fused.zarr` location now has a
deletion marker. The reviewed override is:

```
s3://aind-open-data/exaSPIM_794491_2025-10-21_12-31-25_processed_2025-11-08_09-50-12/fusion/fused.zarr/
```

This is the same acquisition, a different processed output. The sampled overlays
support approximate coordinate correspondence; no claim of identical intensities
or complete registration equivalence is made. The CCF-transformed volume was not
substituted. Original caches/configured dataset metadata were not rewritten.
The override and review note are explicit in `configs/image_alignment.json`.

## Interface verification

- 27 scoped tests passed across `test_image_context`, `test_feature_discovery`
  and `test_trajectory`, with fitting mocked.
- Real isolated extraction read synthetic pixels and could not open an external
  file. The raw-patch frozen prediction path was then exercised successfully in
  the strengthened transport test, including a row with no image.
- SDK MCP transport preserved an image block. This was an in-process protocol
  check; a live LLM's use or interpretation of the image has not been tested.
- Current cloud metadata and on-disk cache identities matched all five recorded
  receipts after review. Source AST parsing passed for 46 harness/CLI files;
  the workflow SVG passed XML parsing. The SVG renderer was unavailable.

Images are now available for future experiments. Incremental Precision@K,
discovery speed, image-model quality and inference cost remain unmeasured.
