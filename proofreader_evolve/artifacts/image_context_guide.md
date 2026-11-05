# Candidate images: 3D exploration, features and learned models

With descriptor tools enabled, use `probe_descriptor` to inspect and test the current
`describe(context)`, then `compute_descriptors` to register predictors at pool scale
(see `descriptor_guide.md`). `DESCRIPTOR.image` supports custom radius and pyramid
level in both paths. `LOCAL_IMAGE` below remains available for stateless feature extraction and models that consume raw patches.

Image access uses reviewed, registered raw fused fluorescence volumes. For
separate local exploration, write `analysis.py` and `analysis_request.json`, then call
`run_volume_analysis({})`: your code receives actual 3D TRAIN pixels and aligned
fragment geometry, and returns concise results. See `volume_analysis_guide.md`
for this analysis interface, independent of fitting or scoring. Its crop/geometry
configuration may differ from a descriptor's; use `probe_descriptor` when checking
the exact descriptor inputs.
No GT skeleton, GT overlay, segmentation label volume or heldout inspection is
provided. Missing or unreviewed sources produce explicit errors, not dark images.

## Optional visual checks

2D previews are optional, not a prerequisite for 3D access. They default to a sheet
of five native XY slices, with an unannotated raw row and a separate fragment-overlay
row. No maximum projection or image interpolation is used in these slices. Edges
and red anchor crosses are clipped to each slice's one-voxel depth. Blue pixels are
outside the source domain, not dark measured signal. All panels share the same
display range (patch-wide percentiles, returned as `display_limits`); this is not
brightness calibration between different candidates. Analysis and scoring receive
the original numeric pixels.

Use the optional `preview` object to choose `plane` (`xy`, `xz`, `yz`), `anchor_index`
(0 for merge; 0 or 1 for a split endpoint), `depth_range_um` (two offsets relative to
that anchor, or to the anchor midpoint when omitted), and `slice_count` (1..9).
Without a range, an explicitly selected anchor gets consecutive native slices.
The default midpoint view spans separated split endpoint depths when necessary;
otherwise it uses consecutive slices. Larger ranges sample planes: the response
and image explicitly mark skipped planes and give actual indices, local physical
depths, offsets and spacing. Rounding to native planes or patch boundaries can
produce fewer distinct slices. Out-of-patch explicit ranges fail: enlarge the crop
or request a narrower range. Split `anchor_index` selects an endpoint within the
chosen `occurrence_index`; it does not select a different split occurrence.

For example, inspect a split's second endpoint on XZ planes, then browse a wider range:

```json
{"candidate_ref": "ctx:...", "preview": {"plane": "xz", "anchor_index": 1}}
```

```json
{"candidate_ref": "ctx:...", "preview": {"plane": "xz", "anchor_index": 1,
 "depth_range_um": [-6, 6], "slice_count": 7}}
```

`preview: {"view": "overview"}` retains the XY/XZ/YZ maximum projections and thin-slab
overlays as context. Projections can merge unrelated depths; a few slices can miss
an intervening break. Treat both as hypotheses, then use full 3D analysis or descriptor
probes to measure signal along a proposed path, compare background controls, and check
sensitivity to thresholds. Do not conclude connectivity from a projection alone.
Descriptor probe previews accept the same `preview` object when `previews=true` and
use the descriptor's exact crop. Their depth controls change the visualization only.

Call `inspect_candidate_image` with a TRAIN candidate_ref, radius_um=40, level=0
and occurrence_index=0. Call `inspect_failure_images({})` for up to two matched
failure pairs in one call. These tools and descriptor probe previews share the
`image_previews_per_generation` allowance in `model_environment.json` (default 32,
configured by `--image-previews-per-generation`; 0 disables previews). A whole slice
sheet costs one request, and requesting
another view costs another. Failed IO in the inspection tools also spends a
request. Each inline PNG is limited to 2 MiB of base64 data and 3000 pixels per edge.
The byte limit decreases to `64 MiB / configured preview count` when that is smaller;
there is no cumulative byte budget. Oversized PNGs are saved unchanged and marked
`preview_inlined: false` (with an absolute file path) in both probe and inspection replies.
Candidate previews under current/earlier `image_inspections/` are readable as PNGs;
adjacent audit JSON remains host-only. Reading a saved PNG does not bypass provider
limits; request fewer slices or a smaller view to obtain an inline preview. Typical
previews are 0.1-0.3 MB, so the preview count is usually the practical limit.
Probe preparation/execution uses descriptor wall time. Volume analysis and neighborhood inspection retain
their separate budgets. Analysis results do not automatically become features:
declare scoring inputs and measure the resulting policy as described below.

## Compare groups, then expand only useful investigations

For each new image hypothesis, first inspect 2-4 matched TRAIN cases of the current
kind across at least two brains when available, using native slices and numerical
3D controls. If previews are disabled or fail, record the limitation and continue
with numerical controls. Use the official selection
feedback groups to compare selected positives (TP), missed positives (FN), selected
negatives (FP), and unselected negatives (TN), where available. The corresponding
feedback groups are `selected_positive`, `missed_positive`, `selected_label0`, and
`boundary_label0` (a subset of TN just outside the selection boundary). In grouped OOF runs,
these groups use the fold-wise selection boundary; do not re-rank pooled OOF scores.
Compare within a brain before checking the same distinction on other TRAIN brains.
Keep radius, level and view settings comparable, and inspect the raw row as well as
the overlay. Include negative controls and exceptions to the hypothesis, not just
striking errors. A group absent from the feedback is not evidence that it does not
exist. No validation cases belong in this exploration.

Once a numerical descriptor expresses the hypothesis, start with the default uniform
probe of 128 candidates per TRAIN brain. Expand to 512, then 1024 only when the first
probe leaves a specific uncertainty to resolve and remaining descriptor wall time permits
(the existing maximum is 2048 per brain). Keep explicit cases or the same sample size
and seed fixed when comparing code revisions; a larger sample is not guaranteed to
contain the smaller one. Uniform probes sample the complete candidate pool without
labels or detector-rank selection; they complement the feedback groups. Enable
previews only for ambiguous or contrasting examples, and stop expanding if the
hypothesis fails. The preview allowance is a ceiling, not a target.

Probes diagnose inputs and feature distributions; use the normal TRAIN/OOF evaluation
and paired feature ablation to measure ranking benefit. Record the tested distinction
and counterexamples in the existing research handoff. For the next runs, compare
`image_preview_usage` (limit/used/remaining), `image_evidence.paired_diagnostics`, and
`wall_seconds` in each generation's evaluation/ledger. `image_evidence.preview_records`
counts saved previews separately from charged slots, which can include failed IO.
Descriptor-group ablations need not isolate the effect of pixels alone.

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
Use `install_packages` for missing Python dependencies (see `classifier_guide.md`).
CPU execution and existing worker time/memory/artifact limits apply.

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

Limits: radius 4..128 um (a patch spans at most a 256 um cube), max_occurrences 1..4,
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
