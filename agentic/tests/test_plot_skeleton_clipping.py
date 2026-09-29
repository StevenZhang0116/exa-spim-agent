import unittest

import networkx as nx
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure

from scripts import plot_top_merge_predictions as merge_plotter
from scripts import plot_top_split_predictions as split_plotter
from scripts.skeleton_plot_util import clipped_edges_in_patch


ORIGIN = np.array([10, 20, 30])
SHAPE = np.array([4, 6, 8])
ANISOTROPY = np.array([2., 3., 4.])


class PatchGraph(nx.Graph):
    def __init__(self, segments):
        super().__init__()
        self.voxels = np.asarray(segments, dtype=float).reshape(-1, 3)
        self.node_xyz = (self.voxels + ORIGIN)[:, ::-1] * ANISOTROPY
        self.node_component_id = np.repeat(np.arange(len(segments)), 2)
        self.component_id_to_swc_id = {
            component: f"{111 + component}.0" for component in range(len(segments))
        }
        self.add_nodes_from(range(len(self.voxels)))
        self.add_edges_from((node, node + 1) for node in range(0, len(self.voxels), 2))

    def nodes_in_patch(self, origin, shape, return_components=False):
        local = (self.node_xyz / ANISOTROPY)[:, ::-1] - origin
        inside = np.all((local >= 0) & (local < shape), axis=1)
        return local[inside], self.node_component_id[inside]

    def edges_in_patch(self, origin, shape, return_components=False):
        local = (self.node_xyz / ANISOTROPY)[:, ::-1] - origin
        inside = np.all((local >= 0) & (local < shape), axis=1)
        pairs = [(source, target) for source, target in self.edges()
                 if inside[source] and inside[target]]
        indices = np.asarray(pairs, dtype=int).reshape(-1, 2)
        return local[indices], self.node_component_id[indices[:, 0]]


class SkeletonClippingTests(unittest.TestCase):
    def test_diagonal_interpolation_reverse_direction_and_degenerate_edges(self):
        graph = PatchGraph([
            [[-2, 0, 1], [8, 5, 6]],
            [[8, 5, 6], [-2, 0, 1]],
            [[-2, 5, 2], [1, 9, 2]],
            [[2, 3, 4], [2, 3, 4]],
            [[9, 9, 9], [9, 9, 9]],
            [[-2, 4, 2], [1, 7, 2]],
        ])
        original_xyz = graph.node_xyz.copy()
        edges, components = clipped_edges_in_patch(graph, ORIGIN, SHAPE, ANISOTROPY)
        np.testing.assert_array_equal(components, [0, 1, 3, 5])
        np.testing.assert_allclose(edges, [
            [[-0.5, 0.75, 1.75], [3.5, 2.75, 3.75]],
            [[3.5, 2.75, 3.75], [-0.5, 0.75, 1.75]],
            [[2, 3, 4], [2, 3, 4]],
            [[-0.5, 5.5, 2], [-0.5, 5.5, 2]],
        ])
        np.testing.assert_array_equal(graph.node_xyz, original_xyz)

    def test_both_plotters_clip_3d_edges_before_projection(self):
        graph = PatchGraph([
            [[1, 2, 3], [8, 2, 3]],
            [[2, 3, -10], [2, 3, 20]],
            [[9, 0, 0], [9, 4, 5]],
            [[1, 1, 1], [2, 4, 5]],
            [[-1, 2, -2], [6, 2, -2]],
        ])
        expected = np.array([
            [[1, 2, 3], [3.5, 2, 3]],
            [[2, 3, -0.5], [2, 3, 7.5]],
            [[1, 1, 1], [2, 4, 5]],
        ])
        for plotter in (merge_plotter, split_plotter):
            with self.subTest(plotter=plotter.__name__):
                figure = Figure()
                axes = figure.subplots(1, 3)
                highlight = ({"highlight_segment": 111} if plotter is merge_plotter
                             else {"highlight": {"111": merge_plotter.SEGMENT_COLOR}})
                plotter.skeleton_overlay(axes, graph, ORIGIN, SHAPE, ANISOTROPY, **highlight)
                for axis, (_, _, xyz_axes) in zip(axes, plotter.VIEWS):
                    lines = [collection for collection in axis.collections
                             if isinstance(collection, LineCollection)]
                    self.assertEqual(len(lines), 3)
                    actual = np.asarray([line.get_segments()[0] for line in lines])
                    voxel_axes = [2 - dimension for dimension in xyz_axes]
                    np.testing.assert_allclose(actual, expected[:, :, voxel_axes] * ANISOTROPY[list(xyz_axes)])
                    np.testing.assert_allclose(lines[0].get_colors()[0][:3],
                                               to_rgba(merge_plotter.SEGMENT_COLOR)[:3])
                    self.assertEqual(lines[0].get_linewidths()[0], 3.)
                    self.assertEqual(lines[1].get_linewidths()[0], 2.)
                figure.clear()

    def test_crossing_edge_visible_without_any_node_in_patch(self):
        graph = PatchGraph([[[2, 3, -10], [2, 3, 20]]])
        for plotter in (merge_plotter, split_plotter):
            with self.subTest(plotter=plotter.__name__):
                figure = Figure()
                axes = figure.subplots(1, 3)
                plotter.skeleton_overlay(axes, graph, ORIGIN, SHAPE, ANISOTROPY)
                for axis in axes:
                    self.assertEqual(len(axis.collections), 1)
                    self.assertIsInstance(axis.collections[0], LineCollection)
                    self.assertEqual(len(axis.texts), 0)
                figure.clear()

    def test_empty_and_nonintersecting_edges_do_not_draw(self):
        for segments in ([], [[[9, 0, 0], [9, 4, 5]]]):
            for plotter in (merge_plotter, split_plotter):
                with self.subTest(segments=segments, plotter=plotter.__name__):
                    figure = Figure()
                    axes = figure.subplots(1, 3)
                    plotter.skeleton_overlay(axes, PatchGraph(segments), ORIGIN, SHAPE, ANISOTROPY)
                    for axis in axes:
                        self.assertEqual(len(axis.collections), 0)
                        self.assertIn("No skeleton", axis.texts[0].get_text())
                    figure.clear()


if __name__ == "__main__":
    unittest.main()