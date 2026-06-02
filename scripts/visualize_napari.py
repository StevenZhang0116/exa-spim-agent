"""
Napari visualization of ExaSPIM neuron proofreading data.

Displays:
  1. Raw fluorescence image patch (3D volume layer)
  2. UNet segmentation (labels layer with per-segment colors)
  3. Ground-truth skeletons (shapes layer, colored per neuron)
  4. UNet fragment skeletons (shapes layer, colored per fragment)

Usage:
    python visualize_napari.py [--patch-size 512] [--use-fragments]

Requirements:
    pip install napari[all] numpy
    (also requires the agentic_neuron_proofreader package installed locally)

Notes:
    - Loads the cached BrainDataset from dataset_cache_794495.pkl
    - Requires GCS credentials (zihan_gcs_token.json) for reading images
    - The viewer opens interactively; close the window to exit
"""

import argparse
import os
import sys

import numpy as np

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "../zihan_gcs_token.json"
os.environ["AWS_EC2_METADATA_DISABLED"] = "true"

from agentic_neuron_proofreader.data_modules.datasets import BrainDataset
from agentic_neuron_proofreader.utils import img_util, util


def load_dataset(brain_id="794495"):
    cache_path = f"../dataset_cache_{brain_id}.pkl"
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

    # Launch napari
    import napari

    print("Launching Napari viewer...")
    viewer = napari.Viewer(title="ExaSPIM Neuron Proofreader")

    # Layer 1: Raw fluorescence image
    viewer.add_image(
        img_patch,
        name="Raw Fluorescence",
        colormap="gray",
        contrast_limits=[0, np.percentile(img_patch, 99.5)],
        blending="additive",
    )

    # Layer 2: Segmentation (as labels)
    viewer.add_labels(
        seg_patch.astype(np.int32),
        name="UNet Segmentation",
        opacity=0.4,
    )

    # Layer 3: GT skeleton edges
    if len(gt_paths) > 0:
        viewer.add_shapes(
            gt_paths,
            shape_type="line",
            edge_color=gt_colors,
            edge_width=2,
            name="GT Skeleton (edges)",
            blending="translucent",
        )

    # Layer 4: GT skeleton nodes
    if len(gt_points) > 0:
        viewer.add_points(
            gt_points,
            face_color=gt_point_colors,
            size=3,
            name="GT Skeleton (nodes)",
            blending="translucent",
        )

    # Layer 5: Fragment skeleton edges
    if len(frag_paths) > 0:
        viewer.add_shapes(
            frag_paths,
            shape_type="line",
            edge_color=frag_colors,
            edge_width=1.5,
            name="Fragment Skeleton (edges)",
            blending="translucent",
            visible=False,  # hidden by default to reduce clutter
        )

    # Layer 6: Fragment skeleton nodes
    if len(frag_points) > 0:
        viewer.add_points(
            frag_points,
            face_color=frag_point_colors,
            size=2,
            name="Fragment Skeleton (nodes)",
            blending="translucent",
            visible=False,
        )

    print("\nNapari viewer is open. Controls:")
    print("  - Scroll wheel: navigate through z-slices")
    print("  - Click layer checkboxes to toggle visibility")
    print("  - Ctrl+Shift+E: toggle 3D view")
    print("  - Close window to exit")

    napari.run()


if __name__ == "__main__":
    main()
