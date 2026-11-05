"""Native slice evidence: depth, pixel identity, masks and bounded view controls."""
import base64
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, Mock

import numpy as np

from proofreader_evolve.harness.image_preview import (
    inline_images, preview_request, slice_selection, render_preview, _overlay)


class InlineImageTests(unittest.TestCase):
    def test_inline_images_bound_each_file_and_preserve_disk_pixels(self):
        from PIL import Image
        from proofreader_evolve.harness import image_preview
        with tempfile.TemporaryDirectory() as root:
            paths = [Path(root) / name for name in ('a.png', 'wide.png', 'c.png')]
            for path, size in zip(paths, ((10, 10), (3001, 1), (8, 8))):
                Image.new('L', size, 17).save(path)
            original = [p.read_bytes() for p in paths]
            blocks, omitted = inline_images(paths)
            self.assertEqual(omitted, ['wide.png'])
            self.assertEqual([base64.b64decode(b['data']) for b in blocks], [original[0], original[2]])
            with patch.object(image_preview, 'MAX_INLINE_IMAGE_BYTES', 1):
                self.assertEqual(inline_images(paths), ([], [p.name for p in paths]))
            self.assertEqual([p.read_bytes() for p in paths], original)
            self.assertEqual(inline_images(paths, preview_budget=0)[0], [])
            self.assertEqual(inline_images([]), ([], []))
            # An unreadable file is omitted on its own; the readable ones are still inlined.
            broken = Path(root) / 'broken.png'
            broken.write_bytes(b'not a png')
            blocks, omitted = inline_images([paths[0], broken, Path(root) / 'absent.png', paths[2]])
            self.assertEqual(omitted, ['broken.png', 'absent.png'])
            self.assertEqual([base64.b64decode(b['data']) for b in blocks], [original[0], original[2]])


class SlicePreviewTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / 'slices.png'
        z, y, x = np.indices((13, 7, 9))
        self.patch = {'image_zyx': 100 * z + 10 * y + x, 'valid_zyx': np.ones(z.shape, dtype=bool),
                      'spacing_zyx_um': np.array([2., 3., 4.]),
                      'anchors_zyx': np.array([[3., 3., 4.], [9., 3., 4.]])}

    def test_default_spans_split_endpoints_and_reports_skipped_planes(self):
        view = slice_selection(self.patch)
        self.assertEqual(view['slice_indices'], [3, 4, 6, 8, 9])
        self.assertFalse(view['consecutive'])
        endpoint = slice_selection(self.patch, {'anchor_index': 1, 'slice_count': 3})
        self.assertEqual(endpoint['slice_indices'], [8, 9, 10])
        self.assertEqual(endpoint['offsets_um_from_reference'], [-2., 0., 2.])
        self.assertTrue(endpoint['consecutive'])

    def test_physical_range_other_planes_and_patch_boundary(self):
        view = slice_selection(self.patch, {'plane': 'yz', 'anchor_index': 1,
                                           'depth_range_um': [-8., 8.], 'slice_count': 3})
        self.assertEqual(view['slice_indices'], [2, 4, 6])
        self.assertEqual(view['normal_axis'], 'x')
        self.assertEqual(view['depths_um_from_patch_edge'], [10., 18., 26.])
        self.patch['anchors_zyx'] = np.array([[0., 0., 0.]])
        view = slice_selection(self.patch)
        self.assertEqual(view['slice_indices'], [0, 1, 2])  # no duplicated boundary planes
        with self.assertRaisesRegex(ValueError, 'outside this patch'):
            slice_selection(self.patch, {'depth_range_um': [-4., 4.]})
        with self.assertRaisesRegex(ValueError, 'anchor is absent'):
            slice_selection(self.patch, {'anchor_index': 1})

    def test_invalid_requests_fail_explicitly(self):
        for request in ({'plane': 'xyz'}, {'slice_count': 10}, {'slice_count': True}, {'anchor_index': -1},
                        {'depth_range_um': [1., -1.]}, {'depth_range_um': [0., float('nan')]},
                        {'view': 'overview', 'plane': 'xz'}, {'unknown': 1}):
            with self.subTest(request=request), self.assertRaises(ValueError):
                preview_request(request)

    def test_each_plane_renders_original_pixels_masks_and_one_display_range(self):
        from matplotlib.axes import Axes
        original = Axes.imshow
        self.patch['valid_zyx'][:, 0, 0] = False
        for plane, axis in (('xy', 0), ('xz', 1), ('yz', 2)):
            rendered = []
            def capture(ax, data, **kwargs):
                rendered.append((data.copy(), kwargs['vmin'], kwargs['vmax']))
                return original(ax, data, **kwargs)
            with patch.object(Axes, 'imshow', capture):
                view = render_preview(self.path, self.patch, np.empty((0, 3)), [], [], 'Phantom',
                                      preview={'plane': plane, 'slice_count': 3})
            self.assertTrue(self.path.read_bytes().startswith(b'\x89PNG'))
            self.assertEqual(len(set((low, high) for _, low, high in rendered)), 1)
            for column, index in enumerate(view['slice_indices']):
                for row in (0, 1):
                    data = rendered[2 * column + row][0]
                    np.testing.assert_array_equal(data.data, np.take(self.patch['image_zyx'], index, axis=axis))
                    np.testing.assert_array_equal(data.mask, ~np.take(self.patch['valid_zyx'], index, axis=axis))

    def test_projection_overlap_does_not_become_a_connection_in_raw_slices(self):
        from matplotlib.axes import Axes
        pixels = np.zeros((7, 5, 6))
        pixels[1, 2, 1:4] = 90.
        pixels[5, 2, 3:5] = 130.
        data = {**self.patch, 'image_zyx': pixels, 'valid_zyx': np.ones_like(pixels, dtype=bool),
                'anchors_zyx': np.array([[1., 2., 3.], [5., 2., 3.]])}
        original, rendered = Axes.imshow, []
        def capture(ax, image, **kwargs):
            rendered.append(image.copy())
            return original(ax, image, **kwargs)
        with patch.object(Axes, 'imshow', capture):
            render_preview(self.path, data, np.empty((0, 3)), [], [], 'Depth-separated structures',
                           preview={'slice_count': 2})
        self.assertTrue((pixels.max(axis=0)[2, 1:5] > 0).all())
        self.assertEqual(rendered[0][2, 4], 0.)
        self.assertEqual(rendered[2][2, 1], 0.)

    def test_overlay_clips_edges_and_anchors_to_one_voxel_depth(self):
        ax = Mock()
        nodes = np.array([[0., 1., 1.], [4., 1., 5.], [4., 2., 1.], [4., 2., 5.]])
        anchors = np.array([[2., 1., 3.], [4., 2., 5.]])
        _overlay(ax, nodes, [[0, 1], [2, 3]], [0, 0, 1, 1], anchors, np.ones(3), 0, 2, 1, (1.5, 2.5))
        self.assertEqual(ax.plot.call_count, 1)
        np.testing.assert_allclose(ax.plot.call_args.args[0], [3., 4.])
        np.testing.assert_allclose(ax.scatter.call_args.args[0], [3.5])

    def test_overview_remains_explicit(self):
        view = render_preview(self.path, self.patch, np.empty((0, 3)), [], [], 'Overview',
                              preview={'view': 'overview'})
        self.assertEqual(view['view'], 'overview')
        self.assertTrue(self.path.is_file())
