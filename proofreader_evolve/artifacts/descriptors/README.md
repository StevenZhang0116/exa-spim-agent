# Descriptor examples (code, not columns)

These files are `describe(context)` examples in the descriptor contract. They are
derived from code earlier agents wrote inside `run_volume_analysis` analyses or
`LOCAL_CONTEXT` extractors in runs `precision_20261003_004137_sogl3h0d` and
`precision_20261003_190054_96ng6ehh`. None of them is precomputed; copying one
into `descriptor.py` and calling `compute_descriptors` runs it over the cached
band, and a byte-identical copy hits the result cache.

| File | Inputs | Origin | What the earlier agent measured |
|---|---|---|---|
| `split_gap_image.py` | image | run 004137 gen016 analyses | background median, noise sigma, anchor SNR, straight-chord bottleneck SNR, bright-voxel fraction. Observed on 12 TRAIN split sites only; never scored. |
| `merge_arm_geometry.py` | geometry | run 190054 gen007/gen009 | arm count, most antiparallel far-field arm pair cosine, min/max arm reach. Entered the accepted merge formula; a later 7-point weight sweep found most of its value was the availability offset. |
| `split_endpoint_geometry.py` | geometry | run 190054 gen012 ablation002 | endpoint tangent alignment with the gap at 5/20 um, tangent collinearity, rival tips within 15 um, foreign-node fraction. Paired ablation at 16,000 rows: -0.0175 out of fold. |

Earlier full programs remain under each run's `gen*/volume_analyses/*/analysis.py`
and `gen*/experiments/*/scorer.py`.
