"""Agent descriptor: one measurable quantity per cached candidate, computed over the whole band.

Edit DESCRIPTOR and describe(context), then call plan_descriptor_run({}) and
compute_descriptors({"scope": "pilot" | "boundary" | "all_cached"}). The host runs
describe on every cached row in parallel isolated workers and registers the results
as bank_agent_<name> predictor columns for scorers, classifiers and ablations.

context keys (same frame as analyze(context)):
  kind, features (base predictors of this row; NaN as None), radius_um,
  fragment: xyz_um (N,3) relative to the anchor midpoint, radius_um (N,), degree (N,),
            segment (N,) local ids, edges (E,2), anchor_nodes, outside_radius (N,), truncated,
            and with image inputs nodes_zyx (N,3) patch voxels + inside_patch (N,)
  with inputs 'image' or 'both': image_available (bool), image_zyx, valid_zyx,
            anchors_zyx, spacing_zyx_um, level, image_radius_um
Return a dict with exactly the declared feature_names; values are finite floats or NaN.
describe must be deterministic and must not read anything outside context.
"""

DESCRIPTOR = {'kind': 'split', 'inputs': 'geometry', 'image_tier': 'level1',
              'feature_names': ['anchor_radius_ratio', 'neighbor_segments']}


def describe(context):
    import numpy as np
    fragment = context['fragment']
    radius = np.asarray(fragment['radius_um'], dtype=float)
    anchors = list(fragment['anchor_nodes'])
    anchor_radius = radius[anchors]
    ratio = float(np.nanmin(anchor_radius) / np.nanmax(anchor_radius)) if np.isfinite(anchor_radius).all() else float('nan')
    segments = np.asarray(fragment['segment'])
    return {'anchor_radius_ratio': ratio, 'neighbor_segments': float(len(set(segments.tolist())))}
