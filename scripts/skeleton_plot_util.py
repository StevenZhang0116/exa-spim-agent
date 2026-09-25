"""Skeleton geometry shared by the separate merge and split plotters."""

from itertools import islice

import numpy as np


def clipped_edges_in_patch(graph, origin, shape, anisotropy):
    """Clip graph edges to the displayed voxel footprint, returning local ZYX.

    Pixel centers run from 0 to shape-1; their outer faces are -0.5 and
    shape-0.5, matching imshow extents in the plotters. Clipping happens in 3D
    before projection, including edges with both endpoints outside the box.
    Edges are processed in bounded batches without modifying the graph.
    """
    origin = np.asarray(origin, dtype=float)
    lower = np.full(3, -0.5)
    upper = np.asarray(shape, dtype=float) - 0.5
    scale = np.asarray(anisotropy, dtype=float)
    xyz = np.asarray(graph.node_xyz)
    node_components = np.asarray(graph.node_component_id)
    iterator = iter(graph.edges())
    clipped_batches, component_batches = [], []
    while True:
        batch = list(islice(iterator, 65536))
        if not batch:
            break
        indices = np.asarray(batch, dtype=np.int64)
        segments = (xyz[indices] / scale)[:, :, ::-1] - origin
        possible = np.all((segments.max(axis=1) >= lower)
                          & (segments.min(axis=1) <= upper), axis=1)
        if not possible.any():
            continue
        segments = segments[possible]
        components = node_components[indices[possible, 0]]
        starts = segments[:, 0]
        directions = segments[:, 1] - starts
        moving = directions != 0
        first = np.full_like(directions, -np.inf)
        second = np.full_like(directions, np.inf)
        np.divide(lower - starts, directions, out=first, where=moving)
        np.divide(upper - starts, directions, out=second, where=moving)
        enter = np.maximum(0., np.minimum(first, second).max(axis=1))
        leave = np.minimum(1., np.maximum(first, second).min(axis=1))
        intersects = enter <= leave
        if not intersects.any():
            continue
        starts, directions = starts[intersects], directions[intersects]
        clipped = np.stack((starts + enter[intersects, None] * directions,
                            starts + leave[intersects, None] * directions), axis=1)
        clipped_batches.append(np.clip(clipped, lower, upper))
        component_batches.append(components[intersects])
    if not clipped_batches:
        return np.empty((0, 2, 3)), np.empty(0, dtype=np.int64)
    return np.concatenate(clipped_batches), np.concatenate(component_batches)