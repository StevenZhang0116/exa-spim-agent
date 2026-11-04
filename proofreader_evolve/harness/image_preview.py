"""Scientific image/fragment overlays and diagnostic registration measurements."""
import base64
from pathlib import Path
import textwrap

import numpy as np


def inline_images(paths, limit=None):
    """Base64 PNG blocks for `paths`, in order, while their payload stays within `limit`.

    Claude Code returns a tool result to the SDK as one stream line holding the result
    twice, bounded by reviser_session.MESSAGE_BUFFER_BYTES; files past the limit stay on
    disk. Returns the blocks and the names of the files left out (every later file too,
    so the inlined images match the leading preview records).
    """
    from .reviser_session import TOOL_IMAGE_BYTES
    limit = TOOL_IMAGE_BYTES if limit is None else limit
    blocks, omitted, total = [], [], 0
    for path in paths:
        data = base64.b64encode(Path(path).read_bytes()).decode('ascii')
        if omitted or total + len(data) > limit:
            omitted.append(Path(path).name)
            continue
        total += len(data)
        blocks.append({'type': 'image', 'mimeType': 'image/png', 'data': data})
    return blocks, omitted


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


PREVIEW_SCHEMA = {
    'type': 'object', 'properties': {
        'view': {'type': 'string', 'enum': ['slices', 'overview']},
        'plane': {'type': 'string', 'enum': ['xy', 'xz', 'yz']},
        'anchor_index': {'type': 'integer', 'minimum': 0, 'maximum': 1},
        'depth_range_um': {'type': 'array', 'items': {'type': 'number'}, 'minItems': 2, 'maxItems': 2},
        'slice_count': {'type': 'integer', 'minimum': 1, 'maximum': 9}},
    'additionalProperties': False}
PLANES = {'xy': (0, 2, 1), 'xz': (1, 2, 0), 'yz': (2, 1, 0)}


def preview_request(preview=None):
    """Validate agent view choices before image IO; offsets are local physical distances."""
    if preview is None:
        preview = {}
    if not isinstance(preview, dict) or set(preview) - set(PREVIEW_SCHEMA['properties']):
        raise ValueError('Unknown image preview options')
    request = {'view': 'slices', 'plane': 'xy', 'slice_count': 5, **preview}
    if (not isinstance(request['view'], str) or not isinstance(request['plane'], str)
            or request['view'] not in ('slices', 'overview') or request['plane'] not in PLANES):
        raise ValueError('Use preview view slices/overview and plane xy/xz/yz')
    if type(request['slice_count']) is not int or not 1 <= request['slice_count'] <= 9:
        raise ValueError('preview slice_count must be 1..9')
    if 'anchor_index' in request and (type(request['anchor_index']) is not int or request['anchor_index'] not in (0, 1)):
        raise ValueError('preview anchor_index must be 0 or 1 (split only)')
    if 'depth_range_um' in request:
        bounds = request['depth_range_um']
        if (not isinstance(bounds, (list, tuple)) or len(bounds) != 2
                or any(type(v) not in (int, float) or not np.isfinite(v) for v in bounds)
                or bounds[0] > bounds[1]):
            raise ValueError('preview depth_range_um must be two finite, increasing offsets in microns')
    if request['view'] == 'overview' and set(preview) - {'view'}:
        raise ValueError('Slice options apply only to preview view slices')
    return request


def slice_selection(patch, preview=None):
    """Select native planes, never project or interpolate pixels; report every actual depth."""
    request = preview_request(preview)
    normal, _, _ = PLANES[request['plane']]
    anchors = np.asarray(patch['anchors_zyx'])
    anchor_index = request.get('anchor_index')
    if anchor_index is not None and anchor_index >= len(anchors):
        raise ValueError('Requested preview anchor is absent from this candidate')
    center = float(anchors[:, normal].mean() if anchor_index is None else anchors[anchor_index, normal])
    spacing = float(patch['spacing_zyx_um'][normal])
    size = patch['image_zyx'].shape[normal]
    count = request['slice_count']
    if 'depth_range_um' in request:
        bounds = center + np.asarray(request['depth_range_um'], dtype=float) / spacing
        if bounds[0] < 0 or bounds[1] > size - 1:
            raise ValueError('Requested slice depths lie outside this patch; enlarge radius_um or narrow depth_range_um')
        positions = np.linspace(*bounds, count) if count > 1 else [bounds.mean()]
    elif anchor_index is None and count > 1 and np.ptp(anchors[:, normal]) >= count - 1:
        # A separated split pair: include both endpoint depths and explicitly report skipped planes.
        positions = np.linspace(anchors[:, normal].min(), anchors[:, normal].max(), count)
    else:
        positions = round(center) + np.arange(count) - count // 2
    indices = np.unique(np.clip(np.rint(positions).astype(int), 0, size - 1))
    return {'view': 'slices', 'plane': request['plane'], 'normal_axis': 'zyx'[normal],
            'anchor_index': anchor_index, 'request': request, 'slice_indices': indices.tolist(),
            'depths_um_from_patch_edge': ((indices + .5) * spacing).tolist(),
            'offsets_um_from_reference': ((indices - center) * spacing).tolist(),
            'reference_depth_um_from_patch_edge': (center + .5) * spacing,
            'spacing_um': spacing, 'consecutive': bool(np.all(np.diff(indices) == 1))}


def _overlay(ax, nodes, edges, segments, anchors, spacing, normal, horizontal, vertical, bounds=None):
    for a, b in edges:
        pair = np.asarray(nodes)[[a, b]].copy()
        if bounds is not None:
            low, high = bounds
            delta = pair[1] - pair[0]
            if abs(delta[normal]) < 1e-12:
                if not low <= pair[0, normal] <= high:
                    continue
            else:
                limits = sorted(((low - pair[0, normal]) / delta[normal], (high - pair[0, normal]) / delta[normal]))
                lower, upper = max(0., limits[0]), min(1., limits[1])
                if lower > upper:
                    continue
                pair = pair[0] + np.asarray([lower, upper])[:, None] * delta
        ax.plot((pair[:, horizontal] + .5) * spacing[horizontal],
                (pair[:, vertical] + .5) * spacing[vertical],
                color=f'C{int(segments[a]) % 10}', linewidth=.7, alpha=.8)
    visible = anchors if bounds is None else anchors[(anchors[:, normal] >= bounds[0]) & (anchors[:, normal] <= bounds[1])]
    ax.scatter((visible[:, horizontal] + .5) * spacing[horizontal],
               (visible[:, vertical] + .5) * spacing[vertical], marker='x', color='red', s=35)


def render_preview(path, patch, nodes, edges, segments, title, *, preview=None):
    request = preview_request(preview)
    if request['view'] == 'overview':
        _render_overview(path, patch, nodes, edges, segments, title)
        return {'view': 'overview', 'note': 'Maximum projections compress depth; confirm continuity on raw slices and 3D pixels.'}
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    view = slice_selection(patch, request)
    image, valid = patch['image_zyx'], patch['valid_zyx']
    spacing, anchors = np.asarray(patch['spacing_zyx_um']), np.asarray(patch['anchors_zyx'])
    finite = image[valid]
    if not len(finite):
        raise ValueError('No valid image pixels to preview')
    low, high = np.percentile(finite, [1, 99.8])
    high = max(float(high), float(low) + 1.)
    view['display_limits'] = [float(low), high]
    normal, horizontal, vertical = PLANES[view['plane']]
    extent = [0, image.shape[horizontal] * spacing[horizontal], 0, image.shape[vertical] * spacing[vertical]]
    count = len(view['slice_indices'])
    figure, axes = plt.subplots(2, count, figsize=(3.2 * count, 6.5), squeeze=False, constrained_layout=True)
    cmap = plt.get_cmap('gray').copy()
    cmap.set_bad('#507080')
    view['valid_fractions'] = []
    try:
        for column, index in enumerate(view['slice_indices']):
            pixels = np.take(image, index, axis=normal)
            mask = np.take(valid, index, axis=normal)
            view['valid_fractions'].append(float(mask.mean()))
            for row in range(2):
                ax = axes[row, column]
                ax.imshow(np.ma.array(pixels, mask=~mask), cmap=cmap, vmin=low, vmax=high,
                          origin='lower', extent=extent, interpolation='nearest')
                if row:
                    _overlay(ax, nodes, edges, segments, anchors, spacing, normal, horizontal, vertical,
                             (index - .5, index + .5))
                ax.set_xlim(extent[:2]); ax.set_ylim(extent[2:])
                ax.set_title(f"{'raw' if row == 0 else 'fragment overlay'} {view['plane'].upper()} [{index}]\n"
                             f"{view['normal_axis']}={view['depths_um_from_patch_edge'][column]:.2f} um; "
                             f"offset {view['offsets_um_from_reference'][column]:+.2f}")
                ax.set_xlabel(f"{'zyx'[horizontal]} (um from patch edge)")
                ax.set_ylabel(f"{'zyx'[vertical]} (um from patch edge)")
        sampling = 'consecutive native slices' if view['consecutive'] else 'sampled native slices: intervening planes omitted'
        caption = (f'{title}\n{sampling}; one shared display range [{low:.3g}, {high:.3g}]. '
                   'Blue: outside image domain.\nRaw row has no overlays; red crosses mark anchors inside the displayed slice.')
        figure.suptitle('\n'.join(textwrap.fill(line, width=max(36, 30 * count)) for line in caption.splitlines()), fontsize=10)
        figure.savefig(Path(path), dpi=120)
    finally:
        plt.close(figure)
    return view


def _render_overview(path, patch, nodes, edges, segments, title):
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
