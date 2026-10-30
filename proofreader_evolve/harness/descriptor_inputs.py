"""One host-owned input path for descriptor probes, TRAIN computation and frozen inference."""
import time

import numpy as np

from .image_coordinates import ImageGeometry
from .descriptor_contract import TIER_IMAGES


PATCH_FIELDS = ('image_zyx', 'valid_zyx', 'anchors_zyx', 'spacing_zyx_um')
MAX_BATCH_BYTES = 128 * 1024**2


def image_request(entry, spec):
    """Resolve legacy tier declarations or explicit physical image requests."""
    if spec['inputs'] == 'geometry':
        return None, None
    tiers = entry.manifest['image_tiers']
    if 'image' not in spec:
        tier = spec['image_tier']
        if tier not in tiers:
            return {**TIER_IMAGES[tier], 'channel': 0, 'timepoint': 0}, None
        return {k: tiers[tier][k] for k in ('level', 'radius_um', 'channel', 'timepoint')}, tier
    request = spec['image']
    tier = next((name for name, data in tiers.items()
                 if all(data.get(k) == v for k, v in request.items())), None)
    return request, tier


def source_identity(table, entry, spec):
    request, tier = image_request(entry, spec)
    source = entry.manifest['image_tiers'][tier].get('source') if tier is not None else None
    if request is not None and tier is None:
        provider = getattr(table, 'image_context', None)
        if provider is None:
            raise ValueError('Descriptor images require registered host image access')
        source = provider.image_identity(table, request)
    return {'image': request, 'cached_tier': tier, 'source': source}


def _image(table, entry, row, context, request, tier, root, persist, deadline):
    if tier is not None:
        patch = entry.image([row], tier)[0]
        return patch, np.asarray(patch['nodes_zyx']) if patch is not None else None
    provider = getattr(table, 'image_context', None)
    if provider is None:
        raise ValueError('Descriptor images require registered host image access')
    remaining = None if deadline is None else deadline - time.monotonic()
    path, metadata = provider.patch(table, int(row), request, persist=persist,
                                    directory=root / 'inputs', timeout=remaining)
    with np.load(path, allow_pickle=False) as data:
        patch = {k: data[k] for k in PATCH_FIELDS}
    geometry = ImageGeometry(tuple(metadata['source_axes']), tuple(metadata['source_shape']),
        np.asarray(metadata['spacing_xyz_um']), np.asarray(metadata['translation_xyz_um']),
        np.asarray(metadata['graph_to_world_um']), metadata['dataset'])
    xyz = np.asarray(context['fragment']['xyz_um']) + metadata['center_xyz_um']
    nodes = geometry.graph_to_voxel(xyz) - metadata['origin_zyx']
    if not np.allclose(nodes[context['fragment']['anchor_nodes']], patch['anchors_zyx'], atol=1e-6, rtol=0):
        raise ValueError('Descriptor fragment/image anchors disagree')
    # The batch transports arrays, so temporary source files need not accumulate.
    if not persist and path.is_relative_to(root):
        import shutil
        shutil.rmtree(path.parent)
    return patch, nodes


def iter_inputs(table, entry, spec, rows, root, *, persist=False, deadline=None):
    """Yield one context/patch at a time; feature and geometry allowlists come from the host."""
    request, tier = image_request(entry, spec)
    contexts, _ = entry.contexts(table, rows, inputs='geometry')
    for row, context in zip(rows, contexts):
        if deadline is not None and time.monotonic() >= deadline:
            error = TimeoutError('Descriptor input budget exhausted')
            error.row = int(row)
            raise error
        patch = None
        if request is not None:
            try:
                patch, nodes = _image(table, entry, row, context, request, tier, root, persist, deadline)
            except Exception as exc:
                exc.row = int(row)
                raise
            context.update(level=request['level'], channel=request['channel'], timepoint=request['timepoint'],
                           image_radius_um=request['radius_um'], image_available=patch is not None)
            if patch is not None:
                shape = np.asarray(patch['image_zyx'].shape)
                context['fragment'].update(nodes_zyx=nodes.tolist(),
                    inside_patch=((nodes >= 0) & (nodes < shape)).all(axis=1).tolist())
                patch = {k: patch[k] for k in PATCH_FIELDS}
                if sum(a.nbytes for a in patch.values()) > MAX_BATCH_BYTES:
                    error = ValueError('Descriptor patch exceeds the batch byte limit')
                    error.row = int(row)
                    raise error
        yield int(row), context, patch


def input_summary(context, patch):
    """Agent-safe local support facts, never source paths, absolute positions or labels."""
    return {'image_available': patch is not None, 'level': context['level'],
            'image_radius_um': context['image_radius_um'],
            'shape_zyx': list(patch['image_zyx'].shape) if patch is not None else None,
            'spacing_zyx_um': patch['spacing_zyx_um'].tolist() if patch is not None else None,
            'valid_fraction': float(patch['valid_zyx'].mean()) if patch is not None else None,
            'geometry_inside_fraction': float(np.mean(context['fragment']['inside_patch'])) if patch is not None else None,
            'geometry_truncated': bool(context['fragment']['truncated'])}
