"""
Napari visualization of ExaSPIM neuron proofreading data.

Displays three orthogonal views (XY, XZ, YZ) side by side in a single window,
each with an independent slice slider. Each view shows:
  1. Raw fluorescence image (grayscale)
  2. UNet segmentation (colored labels)
  3. Ground-truth skeletons (colored per neuron)
  4. UNet fragment skeletons (colored per fragment, hidden by default)

Usage:
    python visualize_napari.py [--patch-size 256] [--use-fragments]

Requirements:
    pip install napari[all] numpy
    (also requires the agentic_neuron_proofreader package installed locally)

Notes:
    - Loads the cached BrainDataset from cache/dataset_cache_794495_mcl1000.pkl
    - Requires GCS credentials (configs/zihan_gcs_token.json) for reading images
    - Close the window to exit
"""

import argparse
import os
import sys

import numpy as np

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "../configs/zihan_gcs_token.json"
os.environ["AWS_EC2_METADATA_DISABLED"] = "true"

from agentic_neuron_proofreader.data_modules.datasets import BrainDataset
from agentic_neuron_proofreader.utils import img_util, util


def load_dataset(brain_id="794495"):
    cache_path = f"../cache/dataset_cache_{brain_id}_mcl1000.pkl"
    if not os.path.exists(cache_path):
        print(f"Cache not found: {cache_path}")
        print("Run load_skeletons.ipynb first to generate the cache.")
        sys.exit(1)
    dataset = BrainDataset.load_from_cache(cache_path)
    return dataset


def get_skeleton_paths_and_colors(graph, offset, patch_shape):
    """
    Extract skeleton edges as line segments for napari Shapes layer.

    Returns
    -------
    paths : list of np.ndarray
        Each element is shape (2, 3) representing a line segment in (z, y, x).
    colors : np.ndarray
        Per-path RGBA colors based on component ID.
    """
    import colorsys

    edges, edge_comps = graph.edges_in_patch(
        offset, patch_shape, return_components=True
    )

    if len(edges) == 0:
        return [], np.empty((0, 4))

    unique_comps = np.unique(edge_comps)
    n_comps = len(unique_comps)
    comp_to_idx = {c: i for i, c in enumerate(unique_comps)}

    hues = (np.arange(n_comps) / max(n_comps, 1) + 0.05) % 1.0
    palette = np.array(
        [colorsys.hsv_to_rgb(h, 0.9, 1.0) + (1.0,) for h in hues]
    )

    paths = [edge for edge in edges]
    colors = np.array([palette[comp_to_idx[c]] for c in edge_comps])

    return paths, colors


def get_skeleton_points_and_colors(graph, offset, patch_shape):
    """
    Extract skeleton nodes as points for napari Points layer.

    Returns
    -------
    points : np.ndarray, shape (N, 3) in (z, y, x)
    colors : np.ndarray, shape (N, 4) RGBA
    """
    import colorsys

    nodes, node_comps = graph.nodes_in_patch(
        offset, patch_shape, return_components=True
    )

    if len(nodes) == 0:
        return np.empty((0, 3)), np.empty((0, 4))

    unique_comps = np.unique(node_comps)
    n_comps = len(unique_comps)
    comp_to_idx = {c: i for i, c in enumerate(unique_comps)}

    hues = (np.arange(n_comps) / max(n_comps, 1) + 0.05) % 1.0
    palette = np.array(
        [colorsys.hsv_to_rgb(h, 0.9, 1.0) + (1.0,) for h in hues]
    )

    colors = np.array([palette[comp_to_idx[c]] for c in node_comps])
    return nodes, colors


def reorient_volume(vol, axis):
    """
    Transpose a (z, y, x) volume so that `axis` becomes the slicing dimension.
      axis=0 -> (z, y, x)  — XY view, scroll through z
      axis=1 -> (y, z, x)  — XZ view, scroll through y
      axis=2 -> (x, z, y)  — YZ view, scroll through x
    """
    if axis == 0:
        return vol
    elif axis == 1:
        return np.transpose(vol, (1, 0, 2))
    elif axis == 2:
        return np.transpose(vol, (2, 0, 1))


def reorient_points(pts, axis):
    """
    Reorder point coordinates (N, 3) from (z, y, x) to match reoriented volume.
    """
    if len(pts) == 0:
        return pts
    if axis == 0:
        return pts
    elif axis == 1:
        return pts[:, [1, 0, 2]]
    elif axis == 2:
        return pts[:, [2, 0, 1]]


def reorient_paths(paths, axis):
    """
    Reorder line segment coordinates to match reoriented volume.
    Each path is (2, 3) array in (z, y, x).
    """
    if not paths:
        return paths
    if axis == 0:
        return paths
    elif axis == 1:
        return [p[:, [1, 0, 2]] for p in paths]
    elif axis == 2:
        return [p[:, [2, 0, 1]] for p in paths]


def get_scale_for_axis(axis):
    """
    Return physical scale tuple for a reoriented volume.
    Original (z, y, x) scale = (1.0, 0.748, 0.748).
    """
    scales = (1.0, 0.748, 0.748)
    if axis == 0:
        return scales
    elif axis == 1:
        return (scales[1], scales[0], scales[2])
    elif axis == 2:
        return (scales[2], scales[0], scales[1])


def add_layers_to_viewer(viewer, img_patch, seg_patch, gt_paths, gt_colors,
                         gt_points, gt_point_colors, frag_paths, frag_colors,
                         frag_points, frag_point_colors, axis, prefix):
    """Add all data layers for one orthogonal view."""
    vol = reorient_volume(img_patch, axis)
    seg = reorient_volume(seg_patch, axis)
    scale = get_scale_for_axis(axis)

    viewer.add_image(
        vol,
        name=f"{prefix} - Raw",
        colormap="gray",
        contrast_limits=[0, np.percentile(img_patch, 99.5)],
        blending="additive",
        scale=scale,
    )

    viewer.add_labels(
        seg.astype(np.int32),
        name=f"{prefix} - Segmentation",
        opacity=0.4,
        scale=scale,
    )

    rp = reorient_paths(gt_paths, axis)
    if rp:
        viewer.add_shapes(
            rp,
            shape_type="line",
            edge_color=gt_colors,
            edge_width=2,
            name=f"{prefix} - GT Skeleton",
            blending="translucent",
            scale=scale,
        )

    pts = reorient_points(gt_points, axis)
    if len(pts) > 0:
        viewer.add_points(
            pts,
            face_color=gt_point_colors,
            size=3,
            name=f"{prefix} - GT Nodes",
            blending="translucent",
            scale=scale,
        )

    rfp = reorient_paths(frag_paths, axis)
    if rfp:
        viewer.add_shapes(
            rfp,
            shape_type="line",
            edge_color=frag_colors,
            edge_width=1.5,
            name=f"{prefix} - Fragment Skeleton",
            blending="translucent",
            visible=False,
            scale=scale,
        )

    fpts = reorient_points(frag_points, axis)
    if len(fpts) > 0:
        viewer.add_points(
            fpts,
            face_color=frag_point_colors,
            size=2,
            name=f"{prefix} - Fragment Nodes",
            blending="translucent",
            visible=False,
            scale=scale,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Napari viewer for ExaSPIM neuron proofreading data"
    )
    parser.add_argument(
        "--patch-size", type=int, default=256,
        help="Side length of the cubic patch to visualize (default: 256)"
    )
    parser.add_argument(
        "--use-fragments", action="store_true",
        help="Center patch on a UNet fragment node instead of a GT node"
    )
    parser.add_argument(
        "--brain-id", type=str, default="794495",
        help="Brain ID (default: 794495)"
    )
    parser.add_argument(
        "--ortho", action="store_true",
        help="Show 3 orthogonal views (XY, XZ, YZ) side by side in one window"
    )
    args = parser.parse_args()

    patch_shape = (args.patch_size, args.patch_size, args.patch_size)

    print("Loading dataset from cache...")
    dataset = load_dataset(args.brain_id)

    segmentation_id = "raw.unet_449_splits_and_merges_900000"
    segmentation_path = (
        f"gs://allen-nd-goog/from_google/{args.brain_id}"
        f"/whole_brain/{segmentation_id}/"
    )
    segmentation = img_util.TensorStoreImage(segmentation_path)

    # Sample a center voxel
    if args.use_fragments:
        node = util.sample_once(dataset.fragments_graph.nodes)
        center_voxel = dataset.fragments_graph.node_voxel(node)
        print(f"Centered on UNet fragment node {node}")
    else:
        node = util.sample_once(dataset.gt_graph.nodes)
        center_voxel = dataset.gt_graph.node_voxel(node)
        print(f"Centered on GT node {node}")

    print(f"Center voxel (z, y, x): {center_voxel}")
    print(f"Patch shape: {patch_shape}")

    # Read volumes
    print("Reading image patch...")
    img_patch = dataset.img.read(center_voxel, patch_shape)

    print("Reading segmentation patch...")
    seg_patch = segmentation.read(center_voxel, patch_shape)

    # Compute patch origin for coordinate transforms
    offset = tuple(c - s // 2 for c, s in zip(center_voxel, patch_shape))

    # Extract skeletons
    print("Extracting GT skeleton edges...")
    gt_paths, gt_colors = get_skeleton_paths_and_colors(
        dataset.gt_graph, offset, patch_shape
    )
    gt_points, gt_point_colors = get_skeleton_points_and_colors(
        dataset.gt_graph, offset, patch_shape
    )

    print("Extracting UNet fragment skeleton edges...")
    frag_paths, frag_colors = get_skeleton_paths_and_colors(
        dataset.fragments_graph, offset, patch_shape
    )
    frag_points, frag_point_colors = get_skeleton_points_and_colors(
        dataset.fragments_graph, offset, patch_shape
    )

    print(f"GT: {len(gt_paths)} edges, {len(gt_points)} nodes")
    print(f"Fragments: {len(frag_paths)} edges, {len(frag_points)} nodes")

    import napari

    if args.ortho:
        # --- Ortho mode: 3 views side by side in one window ---
        from qtpy.QtWidgets import QMainWindow, QWidget, QHBoxLayout, QLabel, QVBoxLayout
        from qtpy.QtCore import Qt

        print("Launching orthogonal viewer...")

        view_configs = [
            (0, "XY (scroll Z)"),
            (1, "XZ (scroll Y)"),
            (2, "YZ (scroll X)"),
        ]

        viewers = []
        for axis, title in view_configs:
            v = napari.Viewer(title=title, show=False)
            add_layers_to_viewer(
                v, img_patch, seg_patch,
                gt_paths, gt_colors, gt_points, gt_point_colors,
                frag_paths, frag_colors, frag_points, frag_point_colors,
                axis=axis, prefix=title.split(" ")[0],
            )
            mid = v.dims.range[0][1] / 2
            v.dims.set_point(0, mid)
            viewers.append(v)

        main_window = QMainWindow()
        main_window.setWindowTitle("ExaSPIM Ortho Viewer — XY | XZ | YZ")
        central = QWidget()
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(2)

        for (axis, title), viewer in zip(view_configs, viewers):
            panel = QWidget()
            panel_layout = QVBoxLayout(panel)
            panel_layout.setContentsMargins(0, 0, 0, 0)
            panel_layout.setSpacing(0)

            label = QLabel(title)
            label.setAlignment(Qt.AlignCenter)
            label.setStyleSheet(
                "font-weight: bold; font-size: 13px; padding: 4px; "
                "color: white; background-color: #333;"
            )
            panel_layout.addWidget(label)

            qt_viewer = viewer.window._qt_window.centralWidget()
            qt_viewer.setParent(panel)
            panel_layout.addWidget(qt_viewer)

            main_layout.addWidget(panel)

        main_window.setCentralWidget(central)
        main_window.resize(1800, 700)
        main_window.show()

        print("\nOrtho viewer open. Each panel scrolls independently.")
        print("  - Scroll wheel or slider: navigate slices")
        print("  - Toggle layers in each panel's layer list")
        print("  - Close window to exit")

    else:
        # --- Default: single XY view, scroll through Z ---
        print("Launching Napari viewer...")
        viewer = napari.Viewer(title="ExaSPIM Neuron Proofreader")

        scale = (1.0, 0.748, 0.748)

        viewer.add_image(
            img_patch,
            name="Raw Fluorescence",
            colormap="gray",
            contrast_limits=[0, np.percentile(img_patch, 99.5)],
            blending="additive",
            scale=scale,
        )

        viewer.add_labels(
            seg_patch.astype(np.int32),
            name="UNet Segmentation",
            opacity=0.4,
            scale=scale,
        )

        if len(gt_paths) > 0:
            viewer.add_shapes(
                gt_paths,
                shape_type="line",
                edge_color=gt_colors,
                edge_width=2,
                name="GT Skeleton (edges)",
                blending="translucent",
                scale=scale,
            )

        if len(gt_points) > 0:
            viewer.add_points(
                gt_points,
                face_color=gt_point_colors,
                size=3,
                name="GT Skeleton (nodes)",
                blending="translucent",
                scale=scale,
            )

        if len(frag_paths) > 0:
            viewer.add_shapes(
                frag_paths,
                shape_type="line",
                edge_color=frag_colors,
                edge_width=1.5,
                name="Fragment Skeleton (edges)",
                blending="translucent",
                visible=False,
                scale=scale,
            )

        if len(frag_points) > 0:
            viewer.add_points(
                frag_points,
                face_color=frag_point_colors,
                size=2,
                name="Fragment Skeleton (nodes)",
                blending="translucent",
                visible=False,
                scale=scale,
            )

        print("\nNapari viewer is open. Controls:")
        print("  - Scroll wheel: navigate through z-slices")
        print("  - Click layer checkboxes to toggle visibility")
        print("  - Ctrl+Shift+E: toggle 3D view")
        print("  - Close window to exit")

    napari.run()


if __name__ == "__main__":
    main()
