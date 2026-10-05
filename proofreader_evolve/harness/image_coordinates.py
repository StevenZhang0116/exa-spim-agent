"""Explicit OME-NGFF coordinates; numerical array order is never guessed."""
from dataclasses import dataclass

import numpy as np

UNITS_UM = {'micrometer': 1., 'micrometre': 1., 'um': 1., 'nanometer': .001,
            'nanometre': .001, 'nm': .001, 'millimeter': 1000., 'mm': 1000.}


def affine_parts(transforms, rank):
    scale, translation = np.ones(rank), np.zeros(rank)
    for transform in transforms:
        kind = transform.get('type')
        if kind not in ('scale', 'translation') or set(transform) != {'type', kind}:
            raise ValueError('Unsupported image coordinate transform; an explicit adapter is required')
        value = np.asarray(transform[kind], dtype=float)
        if value.shape != (rank,) or not np.isfinite(value).all():
            raise ValueError('Invalid image transform dimensions or values')
        if kind == 'scale':
            if (value <= 0).any():
                raise ValueError('Image scale must be positive')
            scale, translation = scale * value, translation * value
        else:
            translation += value
    return scale, translation


@dataclass
class ImageGeometry:
    axes: tuple
    shape: tuple
    scale_xyz_um: np.ndarray
    translation_xyz_um: np.ndarray
    graph_to_world: np.ndarray
    dataset: str

    @classmethod
    def from_ngff(cls, attributes, array_metadata, dataset, graph_to_world=None):
        multiscales = attributes.get('multiscales', [])
        if len(multiscales) != 1:
            raise ValueError('Expected one explicit OME multiscale image')
        multiscale = multiscales[0]
        axes = multiscale.get('axes', [])
        if not axes or any(not isinstance(a, dict) for a in axes):
            raise ValueError('Image axes and spatial units must be explicit')
        names = tuple(a['name'] for a in axes)
        shape = tuple(array_metadata['shape'])
        if (len(shape) != len(names) or len(set(names)) != len(names)
                or set(names) - {'t', 'c', 'x', 'y', 'z'} or not {'x', 'y', 'z'} <= set(names)
                or any(type(n) is not int or n <= 0 for n in shape)):
            raise ValueError('Image must have exactly xyz spatial axes and optional t/c axes')
        matches = [d for d in multiscale['datasets'] if d['path'] == dataset]
        if len(matches) != 1 or not matches[0].get('coordinateTransformations'):
            raise ValueError('Missing image level transform')
        transforms = [*matches[0]['coordinateTransformations'], *multiscale.get('coordinateTransformations', [])]
        scale, translation = affine_parts(transforms, len(names))
        indices = [names.index(a) for a in 'xyz']
        try:
            units = np.asarray([UNITS_UM[axes[i]['unit']] for i in indices])
        except KeyError as exc:
            raise ValueError('Spatial image units must be known, not assumed') from exc
        matrix = np.asarray(np.eye(4) if graph_to_world is None else graph_to_world, dtype=float)
        if (matrix.shape != (4, 4) or not np.isfinite(matrix).all()
                or not np.allclose(matrix[3], [0, 0, 0, 1]) or abs(np.linalg.det(matrix[:3, :3])) < 1e-12):
            raise ValueError('graph_to_world_um must be an invertible affine 4x4 matrix')
        return cls(names, shape, scale[indices] * units, translation[indices] * units, matrix, dataset)

    @property
    def shape_zyx(self):
        return np.asarray([self.shape[self.axes.index(a)] for a in 'zyx'], dtype=int)

    def graph_to_voxel(self, xyz_um):
        xyz = np.asarray(xyz_um, dtype=float)
        if xyz.shape[-1] != 3 or not np.isfinite(xyz).all():
            raise ValueError('Fragment coordinates must be finite xyz microns')
        world = xyz @ self.graph_to_world[:3, :3].T + self.graph_to_world[:3, 3]
        return ((world - self.translation_xyz_um) / self.scale_xyz_um)[..., ::-1]

    def voxel_to_graph(self, zyx):
        world = np.asarray(zyx, dtype=float)[..., ::-1] * self.scale_xyz_um + self.translation_xyz_um
        inverse = np.linalg.inv(self.graph_to_world)
        return world @ inverse[:3, :3].T + inverse[:3, 3]

    def crop(self, anchors_xyz, radius_um):
        voxels = self.graph_to_voxel(anchors_xyz)
        center = voxels.mean(axis=0)
        # Include both split endpoints and an eight-micron margin, even if the
        # requested minimum field of view would exclude one of the anchors.
        spacing = self.scale_xyz_um[::-1]
        extent = np.maximum(2 * radius_um / spacing, np.ptp(voxels, axis=0) + 16. / spacing)
        shape = np.maximum(3, np.ceil(extent).astype(int))
        if shape.max() > 256 or np.prod(shape) > 256**3:
            raise ValueError('Image patch exceeds 256 voxels per axis; request a coarser image level')
        origin = np.floor(center - (shape - 1) / 2).astype(int)
        return origin, shape, voxels - origin

    def selection(self, origin, shape, channel=0, timepoint=0):
        lo = np.maximum(origin, 0)
        hi = np.minimum(np.asarray(origin) + shape, self.shape_zyx)
        if (hi <= lo).any():
            raise ValueError('Candidate image patch is outside the image domain')
        slices = dict(zip('zyx', (slice(int(a), int(b)) for a, b in zip(lo, hi))))
        selection, remaining = [], []
        for axis, size in zip(self.axes, self.shape):
            if axis in 'xyz':
                selection.append(slices[axis])
                remaining.append(axis)
            else:
                index = channel if axis == 'c' else timepoint
                if not 0 <= index < size:
                    raise ValueError(f'Image {axis} index is outside its domain')
                selection.append(index)
        if ('c' not in self.axes and channel) or ('t' not in self.axes and timepoint):
            raise ValueError('Selected channel/timepoint does not exist')
        permutation = tuple(remaining.index(a) for a in 'zyx')
        destination = tuple(slice(int(a), int(a + b)) for a, b in zip(lo - origin, hi - lo))
        return tuple(selection), permutation, destination

    def summary(self):
        return {'source_axes': list(self.axes), 'source_shape': list(self.shape), 'array_axes': 'zyx',
                'spacing_xyz_um': self.scale_xyz_um.tolist(), 'translation_xyz_um': self.translation_xyz_um.tolist(),
                'graph_to_world_um': self.graph_to_world.tolist(), 'dataset': self.dataset,
                'voxel_convention': 'Integer image indices map to voxel centers through the declared transform'}
