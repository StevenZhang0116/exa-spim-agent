# Candidate images: 3D exploration, features and learned models

With a context cache attached, `compute_descriptors` (see `descriptor_guide.md`) is
the direct route from pixels to predictors at pool scale; the `LOCAL_IMAGE` path
below remains available for rows outside the cached band or custom patch sizes.

Image access uses reviewed, registered raw fused fluorescence volumes. For
exploration, write `analysis.py` and `analysis_request.json`, then call
`run_volume_analysis({})`: your code receives actual 3D TRAIN pixels and aligned
fragment geometry, and returns concise results. See `volume_analysis_guide.md`
for the primary exploration interface, independent of fitting or scoring.
No GT skeleton, GT overlay, segmentation label volume or heldout inspection is
provided. Missing or unreviewed sources produce explicit errors, not dark images.

## Optional visual checks

2D previews are optional, not a prerequisite for 3D access. The preview tools
return XY/XZ/YZ PNG MIPs, fragment overlays and thin slabs. Red crosses mark
candidate anchors within each view. Display percentiles affect only previews;
analysis and scoring workers receive original numeric pixels. A projection may
overlap unrelated structures; use volume computations to investigate connectivity.

Call `inspect_candidate_image` with a TRAIN candidate_ref, radius_um=40, level=0
and occurrence_index=0. Call `inspect_failure_images({})` for up to two matched
failure pairs in one call. Both share eight image requests per generation; failed
IO also spends a request. Volume analysis and neighborhood inspection retain
their separate budgets. Analysis results do not automatically become features:
declare scoring inputs and measure the resulting policy as described below.

## Image observations in scoring

Add a literal contract to scorer.py or training.py:

```python
LOCAL_IMAGE = {
    'radius_um': 40.0,
    'level': 0,
    'max_candidates': 32,
    'max_occurrences': 1,
    'feature_names': ['image_anchor_signal'],
}

def extract_image_features(context):
    import numpy as np
    values = []
    for patch in context['patches']:
        image, valid = patch['image_zyx'], patch['valid_zyx']
        for anchor in patch['anchors_zyx']:
            voxel = np.rint(anchor).astype(int)
            if np.all(voxel >= 0) and np.all(voxel < image.shape) and valid[tuple(voxel)]:
                values.append(float(image[tuple(voxel)]))
    return {'image_anchor_signal': float(np.mean(values)) if values else np.nan}
```

The example illustrates access, not a validated discriminator. `context` contains
base predictor values and a `patches` list, one per selected native occurrence.
Each patch contains `image_zyx`, `valid_zyx`, `anchors_zyx`, and
`spacing_zyx_um`. Anchors are floating voxel-center coordinates inside the patch,
in exactly the array's z,y,x order. Arrays are read-only. Convert integer pixels
to floating point before arithmetic that might overflow. No cloud paths, global
node IDs, brain IDs, absolute coordinates or labels are passed to the extractor.
This extractor is stateless and label-free; put learned preprocessing in fit.

The host selects up to max_candidates rows by an existing base predictor
(selection_feature='detector_score', selection_largest=True by default). Optional
channel/timepoint default to 0; explicit image metadata determines axis order.
Feature names must start with image_. Unselected rows have NaN image features and
image_available=0; selected rows have image_available=1. Every original candidate
still receives a score. LOCAL_CONTEXT geometry features can be used alongside this.
Changing the image request or extractor requires explore mode, not tune mode.

## Expand an investigation into ranking inputs

A few inspected patches do not give the whole Top K image information. Convert a
useful `analysis.py` measurement into the stateless extractor above, or an arbitrary
raw-patch model. Then declare a reproducible scoring selection. For example, for
K=2000 (replace the literal K with this kind's actual feedback budget):

```python
LOCAL_IMAGE = {
    'selection_mode': 'top_k_boundary',
    'selection_feature': 'detector_score',
    'selection_largest': True,
    'selection_k': 2000,
    'max_candidates': 256,
    'radius_um': 20.0,
    'level': 0,
    'max_occurrences': 1,
    'batch_size': 32,
    'batch_max_mb': 256,
    'max_total_mb': 4096,
    'feature_names': ['image_anchor_signal'],
}
```

This is a selection example, not a validated image feature or recommended physical
radius for every error. Radius and resolution remain agent choices within the
reviewed coordinate system and resource limits.

1. Call `plan_image_scoring({})` after editing the contract. It reads training.py
   when proposal.json contains a classifier, otherwise scorer.py. It reports TRAIN
   coverage without image IO, labels or evaluation charges. It does not edit code.
2. The boundary selector divides the row budget between the closest ranks inside
   and outside K. With K=2000 and max_candidates=256, ranks 1873..2128 are selected.
   With max_candidates=4000, ranks 1..4000 are selected: the full Top 2000 and 2000
   outside candidates. Small pools backfill from the available side. Ties use
   immutable candidate keys; nonfinite selection values rank last. This boundary
   belongs to the declared base predictor, not the current learned scorer.
3. Measure a bounded pilot. If the hypothesis warrants more coverage, increase
   max_candidates in explore mode, up to 4096. Explicit IO, byte and time budgets
   may require a different radius or resolution; the host never silently shrinks
   the selection or drops a failed brain. The exact same literal selector runs on
   TRAIN and frozen validation inputs, without candidate IDs or validation labels.
4. At the intended final coverage, run an image-only paired ablation: explicitly
   set research.feature_columns to every declared image_* column plus
   image_available. Including other columns measures their joint contribution;
   omitting an image input does not remove the complete image representation.
   Classifier ablation costs two units (one per arm); a configuration's own fold
   refits and full fit cost one unit together.
5. Inspect image_context.selection_coverage and image_context.scoring_coverage.
   The latter records how many actual final Top-K rows have images and how many
   are positive. These are coverage counts, not an attribution of ranking gain.
   A pilot ablation does not establish benefit for a changed selection/coverage.

The numeric extractor streams bounded batches through isolated workers, preserving
original candidate indices. All reads and batches share one extraction time budget.
Completed numeric batches are cached with program, row, input and implementation
identities; retries reuse completed batches. A failed batch rejects the measurement
instead of returning a partial feature table. The extractor must remain stateless
across candidates; learned transforms belong in TRAIN fit.

## Learn directly from patches

For arbitrary CPU image models, set `raw_patches: True` in LOCAL_IMAGE. The
feature_names list can be empty if no stateless extractor is needed. Extend the
ordinary model signatures with a keyword argument:

```python
def fit(X_train, y_train, artifact_dir, params, *, images):
    # len(images) == len(X_train) (GT-labeled rows only; X_unlabeled gets no images).
    # images[i] is a list of occurrence patches;
    # an empty list means this row was outside the declared image selection.
    # Fit your encoder, normalization and model using TRAIN only. Save all state.
    ...

def predict(X, artifact_dir, params, *, images):
    # Load frozen state; handle empty patch lists and missing tabular values.
    # Return one deterministic finite score for every input row.
    ...
```

The images object loads one row's patches on demand from read-only local files.
It has no network access. Fit/predict receive only their own row subset; internal
fold fitting cannot open held-row patches or labels. Save learned image encoders
and preprocessing with the model; the outer evaluator freezes them on validation.
The harness does not prescribe a classifier, convolutional architecture or loss.
Installed CPU libraries and existing worker time/memory/artifact limits apply.

## Incremental evidence and limits

In evaluate_feature_ablation, include `image_available` in the removed feature
group to withhold all raw patches in the control arm. Include the relevant image_*
columns too when testing the complete image representation. Numeric columns become
constant zero; images[i] becomes empty in the raw control arm. Both arms refit the
same program and parameters. If a model cannot fit without images, the diagnostic
is unavailable; no alternate learner or fabricated score is substituted. Tabular
and geometry predictors stay available unless explicitly included in the group.

Image read/extraction failure rejects that measurement. It never drops a brain
from the outer gate or treats unavailable pixels as true zero intensity. Boundary
padding is indicated by valid_zyx; do not learn that padded zeros are biological
background. Sparse native annotations and visual continuity do not establish
biological truth. A sampled registration review is not a whole-brain guarantee.

Limits: radius 4..100 um, at most 256 voxels per patch axis, max_occurrences 1..4,
max_candidates 1..4096, batch_size 1..256 (default 32), batch_max_mb 1..1024
(default 256), max_total_mb 1..8192 per brain/kind preparation (default 4096),
1 GiB staged input per worker, 20 GiB run image cache. Numeric extraction can span
multiple workers; arbitrary raw-patch fit/predict still stages at most 1 GiB total.
Batching does not increase the run cache allowance, timeout or model memory limit.
Large image selections may exceed IO/time/byte budgets; start small, reuse cached
patches and measure incremental value. The full table and fixed Top K remain
unchanged. A small selection does not give every candidate new image information.
Image inspection snapshots, metadata, source identities, feature caches and timings
are preserved by the host for review. Do not claim image gains without measurements.
The host saves image_scoring_plan.json and image_evidence.json. Human reports
distinguish successful 3D access, image-scored attempts, and image-only paired TRAIN
diagnostic outcomes. A promoted image-using policy alone does not isolate an image
effect on validation, and no positive increment is assumed.
