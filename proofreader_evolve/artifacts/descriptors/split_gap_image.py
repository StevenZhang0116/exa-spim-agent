"""Split gap image descriptors (ported from run 004137 gen016 analyses; observational only there)."""

DESCRIPTOR = {'kind': 'split', 'inputs': 'image', 'image_tier': 'level1',
              'feature_names': ['bg_median', 'noise_sigma', 'anchor_min_snr', 'chord_bottleneck_snr',
                                'bright_fraction']}


def describe(context):
    import numpy as np
    nan = float('nan')
    if not context['image_available']:
        return {name: nan for name in DESCRIPTOR['feature_names']}
    image = np.asarray(context['image_zyx'], dtype=float)
    valid = np.asarray(context['valid_zyx'], dtype=bool)
    voxels = image[valid]
    if voxels.size < 27:
        return {name: nan for name in DESCRIPTOR['feature_names']}
    background = float(np.median(voxels))
    sigma = max(1., 1.4826 * float(np.median(np.abs(voxels - background))))
    anchors = np.asarray(context['anchors_zyx'], dtype=float)
    shape = np.asarray(image.shape)

    def local_max(point):
        lo = np.clip(np.round(point).astype(int) - 2, 0, shape - 1)
        hi = np.clip(np.round(point).astype(int) + 3, 1, shape)
        return float(image[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]].max())

    anchor_values = [local_max(p) for p in anchors]
    anchor_min_snr = (min(anchor_values) - background) / sigma
    steps = max(2, int(np.ceil(np.linalg.norm((anchors[-1] - anchors[0]) * np.asarray(context['spacing_zyx_um'])) / .5)))
    profile = [local_max(anchors[0] + (anchors[-1] - anchors[0]) * t) for t in np.linspace(0, 1, steps)]
    return {'bg_median': background, 'noise_sigma': sigma, 'anchor_min_snr': float(anchor_min_snr),
            'chord_bottleneck_snr': float((min(profile) - background) / sigma),
            'bright_fraction': float(np.mean(voxels > background + 5 * sigma))}
