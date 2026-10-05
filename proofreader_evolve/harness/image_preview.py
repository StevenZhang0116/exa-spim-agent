"""Scientific image/fragment overlays and diagnostic registration measurements."""
from pathlib import Path

import numpy as np


def fragment_patch_geometry(table, row, spec, metadata, occurrence=0):
    provider = table.local_context
    context, centers = provider.context(row, radius_um=min(200., spec['radius_um'] * 1.5),
                                        max_nodes=512, occurrence_index=occurrence)
    source, _ = table.image_context.source(table, reviewed=False)
    geometry = source.level(spec['level'])
    site = context['sites'][0]
    xyz = np.asarray(site['xyz_um']) + np.asarray(centers[0])
    nodes = geometry.graph_to_voxel(xyz) - np.asarray(metadata['origin_zyx'])
    return nodes, site['edges'], site['segment']


def alignment_statistics(patch, nodes, edges):
    """Diagnostic signal support, not a registration proof or error label."""
    from scipy.ndimage import map_coordinates, maximum_filter
    image, valid = patch['image_zyx'], patch['valid_zyx']
    spacing = patch['spacing_zyx_um']
    points = []
    for a, b in edges:
        start, end = nodes[a], nodes[b]
        count = min(200, max(2, int(np.linalg.norm((end - start) * spacing) / 2.) + 1))
        points.extend(np.linspace(start, end, count))
    points = np.asarray(points) if points else np.asarray(nodes)
    points = points[np.all((points >= 1) & (points < np.asarray(image.shape) - 2), axis=1)]
    if not len(points):
        return {'status': 'insufficient_fragment_points', 'note': 'No automatic alignment conclusion'}
    points = points[::max(1, len(points) // 5000)]
    support = maximum_filter(image.astype(np.float32), size=3)
    def signal(coordinates):
        mask = map_coordinates(valid.astype(np.uint8), coordinates.T, order=0, mode='constant', cval=0) > 0
        values = map_coordinates(support, coordinates.T, order=1, mode='constant', cval=0)[mask]
        return float(np.median(values)) if len(values) else None
    controls = {}
    for axis, name in enumerate('zyx'):
        for sign in (-1, 1):
            shift = np.zeros(3)
            shift[axis] = sign * 12. / spacing[axis]
            controls[f'{name}_{sign * 12:+d}um'] = signal(points + shift)
    return {'status': 'measured', 'fragment_points': len(points), 'skeleton_median_signal': signal(points),
            'shifted_median_signal': controls,
            'image_percentiles': np.percentile(image[valid], [1, 50, 99, 99.9]).tolist(),
            'note': 'Existing fragment edges only; no invented bridge across a split. '
                    'Signal/shift comparisons require visual review and do not prove alignment or biological correctness.'}


def render_preview(path, patch, nodes, edges, segments, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    image, valid = patch['image_zyx'], patch['valid_zyx']
    spacing, anchors = patch['spacing_zyx_um'], patch['anchors_zyx']
    finite = image[valid]
    low, high = np.percentile(finite, [1, 99.8])
    high = max(float(high), float(low) + 1.)
    figure, axes = plt.subplots(3, 3, figsize=(12, 10), constrained_layout=True)
    for column, (normal, horizontal, vertical, name) in enumerate(((0, 2, 1, 'XY'), (1, 2, 0, 'XZ'), (2, 1, 0, 'YZ'))):
        center = int(np.clip(round(anchors[:, normal].mean()), 0, image.shape[normal] - 1))
        width = max(1, int(np.ceil(2. / spacing[normal])))
        slices = [slice(None)] * 3
        slices[normal] = slice(max(0, center - width), min(image.shape[normal], center + width + 1))
        thin = image[tuple(slices)].max(axis=normal)
        extent = [0, image.shape[horizontal] * spacing[horizontal], 0, image.shape[vertical] * spacing[vertical]]
        for row in range(3):
            ax = axes[row, column]
            ax.imshow(thin if row == 2 else image.max(axis=normal), cmap='gray', vmin=low, vmax=high,
                      origin='lower', extent=extent, interpolation='nearest')
            if row:
                for a, b in edges:
                    pair = nodes[[a, b]].copy()
                    if row == 2:
                        low_depth, high_depth = center - width - .5, center + width + .5
                        delta = pair[1] - pair[0]
                        if abs(delta[normal]) < 1e-12:
                            if not low_depth <= pair[0, normal] <= high_depth:
                                continue
                        else:
                            bounds = sorted(((low_depth - pair[0, normal]) / delta[normal],
                                             (high_depth - pair[0, normal]) / delta[normal]))
                            lower, upper = max(0., bounds[0]), min(1., bounds[1])
                            if lower > upper:
                                continue
                            pair = pair[0] + np.asarray([lower, upper])[:, None] * delta
                    color = plt.cm.tab10(int(segments[a]) % 10)
                    ax.plot((pair[:, horizontal] + .5) * spacing[horizontal],
                            (pair[:, vertical] + .5) * spacing[vertical], color=color, linewidth=.7, alpha=.8)
                visible = anchors if row == 1 else anchors[
                    (anchors[:, normal] >= center - width - .5) & (anchors[:, normal] <= center + width + .5)]
                ax.scatter((visible[:, horizontal] + .5) * spacing[horizontal],
                           (visible[:, vertical] + .5) * spacing[vertical], marker='x', color='red', s=45)
            ax.set_xlim(extent[:2]); ax.set_ylim(extent[2:])
            ax.set_title(f'{name}: ' + ('raw MIP' if row == 0 else 'fragment overlay' if row == 1 else 'thin slab overlay'))
            ax.set_xlabel('xyz'[2 - horizontal] + ' (um from patch edge)')
            ax.set_ylabel('xyz'[2 - vertical] + ' (um from patch edge)')
    figure.suptitle(title + '\nRed crosses: candidate anchors. Display scaling does not change model pixels.', fontsize=11)
    figure.savefig(Path(path), dpi=120)
    plt.close(figure)
