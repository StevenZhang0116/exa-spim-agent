"""Split endpoint geometry (ported from run 190054 gen012 ablation002; measured -0.0175 out of fold)."""

DESCRIPTOR = {'kind': 'split', 'inputs': 'geometry',
              'feature_names': ['min_endpoint_alignment_20um', 'endpoint_collinearity_20um',
                                'rival_tip_count_15um', 'foreign_node_fraction_10um']}


def describe(context):
    import numpy as np
    fragment = context['fragment']
    xyz = np.asarray(fragment['xyz_um'], dtype=float)
    segment = np.asarray(fragment['segment'], dtype=int)
    degree = np.asarray(fragment['degree'], dtype=int)
    anchors = [int(a) for a in fragment['anchor_nodes']]
    if len(anchors) != 2:
        return {name: float('nan') for name in DESCRIPTOR['feature_names']}
    a, b = anchors
    gap = xyz[b] - xyz[a]
    gap_norm = np.linalg.norm(gap)
    gap_dir = gap / gap_norm if gap_norm > 0 else np.zeros(3)

    def tangent(anchor, scale=20.):
        same = (segment == segment[anchor]) & (np.linalg.norm(xyz - xyz[anchor], axis=1) <= scale)
        same[anchor] = False
        if not same.any():
            return None
        vector = xyz[anchor] - xyz[same].mean(axis=0)
        norm = np.linalg.norm(vector)
        return vector / norm if norm > 0 else None

    ta, tb = tangent(a), tangent(b)
    alignments = [float(np.dot(ta, gap_dir)) if ta is not None else np.nan,
                  float(np.dot(tb, -gap_dir)) if tb is not None else np.nan]
    midpoint = (xyz[a] + xyz[b]) / 2
    foreign = (segment != segment[a]) & (segment != segment[b])
    near_mid = np.linalg.norm(xyz - midpoint, axis=1)
    return {'min_endpoint_alignment_20um': float(np.nanmin(alignments)) if not np.isnan(alignments).all() else float('nan'),
            'endpoint_collinearity_20um': float(np.dot(ta, -tb)) if ta is not None and tb is not None else float('nan'),
            'rival_tip_count_15um': float(np.sum(foreign & (degree <= 1) & (near_mid <= 15.))),
            'foreign_node_fraction_10um': float(np.mean(foreign[near_mid <= 10.])) if (near_mid <= 10.).any() else float('nan')}
