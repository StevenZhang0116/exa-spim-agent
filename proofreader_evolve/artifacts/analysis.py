"""Replace this descriptive example with your own 3D investigation."""
import numpy as np


def analyze(context):
    volume = context['image_zyx']
    valid = context['valid_zyx']
    anchors = context['anchors_zyx']
    fragment = context['fragment']
    signal = volume[valid].astype(float)
    samples = []
    for anchor in anchors:
        voxel = np.rint(anchor).astype(int)
        inside = np.all(voxel >= 0) and np.all(voxel < volume.shape)
        samples.append(float(volume[tuple(voxel)]) if inside and valid[tuple(voxel)] else None)
    return {
        'shape_zyx': list(volume.shape),
        'spacing_zyx_um': context['spacing_zyx_um'].tolist(),
        'signal_percentiles': np.percentile(signal, [10, 50, 90, 99]).tolist() if signal.size else [],
        'anchor_intensities': samples,
        'fragment_nodes': len(fragment['nodes_zyx']),
        'fragment_truncated': fragment['truncated'],
        'note': 'Descriptive data-access check only; these values are not an error classifier.',
    }
