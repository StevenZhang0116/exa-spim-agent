# Direct 3D exploration

Investigate image hypotheses by writing code against the real 3D volume, without
viewing a projection or fitting a scorer first. You choose the algorithm, local
measurements and interpretation. `analysis.py` is a small descriptive example,
not a prescribed classifier. Replace it as needed, then call
`run_volume_analysis({})` after editing `analysis_request.json`.

## Select TRAIN observations

The initial request contains one matched failure pair when available. Replace or
extend it with 1..4 valid candidate_ref handles from TRAIN feedback/failure cases:

```json
{
  "candidates": [
    {"candidate_ref": "REPLACE_WITH_TRAIN_HANDLE", "occurrence_index": 0}
  ],
  "radius_um": 40.0,
  "level": 0,
  "channel": 0,
  "timepoint": 0,
  "max_nodes": 256
}
```

The host resolves handles only against TRAIN before reading any graph/image.
Merge has occurrence 0; split supports its native occurrence indices. Radius is
4..100 um; at most 256 voxels per patch axis. Both split anchors are included,
expanding the crop if necessary. Request a coarser level when the voxel limit is
exceeded. Levels/channels/timepoints must exist in the source metadata. Fragment
neighborhoods have 2..512 nodes (default 256) in a sphere of radius_um, with anchors
always retained; this is a bounded induced subgraph, not the full skeleton.

Your selection is independent of `LOCAL_IMAGE`'s top-predictor selection for
scoring. An analyzed missed positive may not receive image features in the policy.
To expand a useful finding, implement it as `extract_image_features(context)` or
a raw-patch model, then call `plan_image_scoring({})` with a literal LOCAL_IMAGE
contract. `top_k_boundary` selection can cover candidates on both sides of the
declared predictor's K boundary. Measure a pilot, expand coverage within budgets,
and repeat the image-only ablation at the final coverage. See image_context_guide.md;
analysis results alone do not change ranking or establish image benefit.
Check scoring coverage when turning an observation into a proposed feature.

## Analyze the actual volume

```python
def analyze(context):
    import numpy as np
    volume = context['image_zyx']        # Full 3D array, not a MIP.
    valid = context['valid_zyx']         # Same shape; excludes boundary padding.
    spacing = context['spacing_zyx_um']  # Physical z,y,x voxel sizes.
    anchors = context['anchors_zyx']    # One merge or two split anchor points.
    fragment = context['fragment']
    nodes = fragment['nodes_zyx']       # Same voxel frame as volume and anchors.
    # Replace this descriptive example with your own 3D computation.
    return {'valid_fraction': float(valid.mean()),
            'anchor_separation_um': float(np.linalg.norm((anchors[-1] - anchors[0]) * spacing)),
            'local_fragment_nodes': len(nodes)}
```

Arrays are read-only NumPy arrays; copy if needed. Their dtype/intensities are
those of the fused image, without display contrast normalization. Convert integer
pixels to float before subtraction/interpolation. Index arrays in z,y,x order.
Anchors and nodes are floating voxel-center coordinates; fractional interpolation
must respect valid_zyx. Do not treat zero padding as biological background.

`fragment` also contains `xyz_um` (graph xyz microns relative to the candidate
anchor midpoint), `radius_um`, `degree`, locally numbered `segment`, `edges`
(N x 2 local node indices), `anchor_nodes`, `inside_patch`, `outside_radius`, and
`truncated`. Some fragment nodes may lie outside the image patch; check bounds.
Use nodes_zyx to sample the image; simply reversing xyz_um is not a coordinate
transform. The host checks that mapped fragment anchors match image anchors.

The remaining fields are `kind`, numeric base `features` (missing values become
null), radius/level/channel/timepoint, occurrence_index/occurrences_total and
`coordinates` descriptions. No GT skeleton, segmentation label volume, truth
label, global node ID, brain ID, absolute location, cloud URI or credential enters
the analysis worker. TRAIN labels already present in feedback remain available
to the reviser for interpreting results, not to this program as an input column.

The same analyze function is called once per requested context in one isolated
process, in request order. Contexts are not passed to the LLM as voxel text; the
program processes them and returns results. Cross-case module state is permitted
for exploration. This is not the stateless/deterministic scoring extractor contract.
Do not claim replicates are independent or reproducible without recording seeds.

## Results, limits and use in a policy

Return a dictionary with concise JSON-compatible findings, numerical measurements
or short lists. NumPy scalars/small arrays are converted to JSON. Nonfinite values
are rejected: report null and an explanation when a measurement is undefined.
The entire result list must fit 24 KiB. Oversized/nonserializable output produces
an error, not a truncated measurement. No PNG is rendered or returned by this tool.

Eight executions are available per generation, at most four candidates each.
Invalid files/handles do not spend an execution; accepted requests spend one even
if image IO or code execution fails. Repeated calls rerun the current program;
patch data may come from the checksum-verified cache. Executions do not consume
TRAIN scoring units or the optional preview allowance. Execution uses the policy
time budget and classifier memory/thread limits; these constrain computation,
not the chosen algorithm. Cloud reads keep their existing per-operation timeouts;
large fragment-cache loading and input preparation happen before the worker timer.
Installed CPU libraries are usable, with scratch space; network, child processes,
validation data, GT and arbitrary host-file access are blocked by the sandbox.

Each call saves code, normalized request, input identities, returned results and
worker logs under `volume_analyses/analysisNNN/`. The tool returns code/input hashes
and the candidate handles alongside your results. These are agent-computed
exploratory outputs, not evaluator-verified ranking gains. Runs do not silently
change scorer.py, training.py, candidate pools or promotion decisions. Record
useful findings in rules.md for future generations; a fresh generation starts
with the example analysis program and a new matched request.

To use an observation in scoring, implement it in `extract_image_features`,
`extract_local_features`, or a TRAIN-only fit/predict model as appropriate. Read
image_context_guide.md, measure the resulting scorer normally and use feature
ablation when useful. Validation remains outside exploration and uses frozen
inference. Optional 2D preview tools are available for visual checks; projections
cannot establish 3D connectivity by themselves.
