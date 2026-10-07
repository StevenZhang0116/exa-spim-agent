#!/usr/bin/env python
"""Load GT, fragments, and one real 128^3 raw-image patch for alignment checks.

The default center is the midpoint of a real direct split edge
(``gt_edge_error == 1``). Like ``load_skeletons.py``, the script saves three
separate MIP figures: the raw image, local fragment/segment skeletons, and local
GT skeletons. It intentionally does not save the raw patch or JSON metadata.

This is an I/O and coordinate-alignment smoke test, not a discovery experiment.
Fragment information is printed only to verify registration; it must not become an
image-only model feature.

Example (run on a compute node in the ``panda`` environment):

    python notebooks/verify_gt_fragment_image_patch.py \
        cache/dataset_cache_794495_mcl10_add.pkl \
        --output-prefix notebooks/split_image_patch_128

To inspect a chosen physical location instead:

    python notebooks/verify_gt_fragment_image_patch.py CACHE.pkl \
        --center-xyz X_UM Y_UM Z_UM

For a private ``gs://`` image, the script automatically uses
``configs/allen-nd-goog-f5d46dbfa2cd.json`` when it exists. Override it with
``--gcp-credentials PATH`` or an existing ``GOOGLE_APPLICATION_CREDENTIALS``.
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROOFREADER_SRC = PROJECT_ROOT.parent / "agentic-neuron-proofreader" / "src"
DEFAULT_GCP_CREDENTIALS = PROJECT_ROOT / "configs" / "allen-nd-goog-f5d46dbfa2cd.json"

try:
    import agentic_neuron_proofreader  # noqa: F401 - registers pickle classes
except ModuleNotFoundError:
    if not PROOFREADER_SRC.is_dir():
        raise
    sys.path.insert(0, str(PROOFREADER_SRC))
    import agentic_neuron_proofreader  # noqa: F401 - registers pickle classes

from agentic_neuron_proofreader.utils import img_util


EDGE_SPLIT = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pkl", type=Path, help="Labeled *_add.pkl cache.")
    parser.add_argument(
        "--patch-size",
        type=int,
        default=128,
        help="Cubic patch side in voxels (default: 128).",
    )
    parser.add_argument(
        "--split-index",
        type=int,
        default=0,
        help="Which direct GT split edge to use as the default center.",
    )
    parser.add_argument(
        "--center-xyz",
        type=float,
        nargs=3,
        metavar=("X_UM", "Y_UM", "Z_UM"),
        help="Override the split-edge midpoint with an (x,y,z) micron center.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("notebooks/split_image_patch_128"),
        help=(
            "Prefix for the three PNGs (default: notebooks/split_image_patch_128). "
            "Produces *_image.png, *_fragments.png, and *_gt.png."
        ),
    )
    parser.add_argument(
        "--gcp-credentials",
        type=Path,
        default=None,
        metavar="JSON",
        help=(
            "Credentials for a private gs:// image. Precedence: this flag, "
            "GOOGLE_APPLICATION_CREDENTIALS, then "
            "configs/allen-nd-goog-f5d46dbfa2cd.json when present."
        ),
    )
    return parser.parse_args()


def configure_image_access(
    image_path: str,
    requested_credentials: Path | None,
) -> str:
    """Configure cloud access before TensorStore opens the image."""
    os.environ["AWS_EC2_METADATA_DISABLED"] = "true"
    if image_path.startswith("s3://"):
        print("Image access: anonymous/public S3")
        return "anonymous-public-s3"
    if not image_path.startswith("gs://"):
        raise RuntimeError(f"unsupported image path scheme: {image_path}")

    if requested_credentials is not None:
        credentials = requested_credentials.expanduser().resolve()
        source = "--gcp-credentials"
    elif os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        credentials = Path(
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"]
        ).expanduser().resolve()
        source = "GOOGLE_APPLICATION_CREDENTIALS"
    else:
        credentials = DEFAULT_GCP_CREDENTIALS.resolve()
        source = "project default"

    if not credentials.is_file():
        raise RuntimeError(
            "private GCS image requires an existing credentials JSON; looked at "
            f"{credentials}. Pass --gcp-credentials JSON or set "
            "GOOGLE_APPLICATION_CREDENTIALS."
        )
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(credentials)
    print(f"Image access: authenticated GCS ({source}: {credentials})")
    return "authenticated-gcs"


def xyz_to_zyx_voxel(xyz_um: np.ndarray, anisotropy_xyz: np.ndarray) -> tuple[int, ...]:
    """Convert physical (x,y,z) microns to integer image (z,y,x)."""
    return tuple(np.rint(xyz_um / anisotropy_xyz).astype(int)[::-1])


def find_direct_splits(gt, node_label: np.ndarray, edge_error: np.ndarray):
    """Return direct split edges in the order parallel to gt_edge_error."""
    edges = list(gt.edges)
    if len(edges) != len(edge_error):
        raise RuntimeError(
            f"gt edge count ({len(edges)}) != gt_edge_error length "
            f"({len(edge_error)})"
        )
    records = []
    for edge_index, (i, j) in enumerate(edges):
        a, b = int(node_label[i]), int(node_label[j])
        if int(edge_error[edge_index]) == EDGE_SPLIT and a and b and a != b:
            records.append((edge_index, int(i), int(j), a, b))
    return records


def nearest_fragment(frag, xyz_um: np.ndarray) -> tuple[int, float]:
    """Return nearest fragment node and distance in the graph's micron frame."""
    tree = getattr(frag, "kdtree", None)
    if tree is None:
        from scipy.spatial import cKDTree

        tree = cKDTree(np.asarray(frag.node_xyz))
    distance_um, node_index = tree.query(xyz_um)
    # SkeletonGraph assigns contiguous node ids matching rows of node_xyz.
    node = int(node_index)
    if node not in frag:
        raise RuntimeError(f"KD-tree returned fragment row {node}, not a graph node")
    return node, float(distance_um)


def save_image_mips(patch: np.ndarray, output_png: Path) -> None:
    """Save raw-image MIPs using the same helper as load_skeletons_from_cache."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vmax = float(np.percentile(patch[np.isfinite(patch)], 99.9))
    _save_plot(output_png, lambda: img_util.plot_mips(patch, vmax=vmax))


def _save_plot(output_png: Path, plotter) -> None:
    """Capture an img_util plotting helper's ``plt.show()`` as a PNG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    original_show = plt.show

    def save_current_figure(*_args, **_kwargs) -> None:
        plt.gcf().savefig(output_png, dpi=150, bbox_inches="tight")

    plt.show = save_current_figure
    try:
        plotter()
    finally:
        plt.show = original_show
        plt.close("all")


def save_skeleton_mips(
    nodes: np.ndarray,
    node_components: np.ndarray,
    edges: np.ndarray,
    edge_components: np.ndarray,
    patch_shape: tuple[int, int, int],
    label: str,
    color: str,
    output_png: Path,
) -> None:
    """Save skeleton MIPs using load_skeletons_from_cache's black-bg style."""
    group = {
        label: {
            "nodes": nodes,
            "node_components": node_components,
            "edges": edges,
            "components": edge_components,
            "color": color,
        }
    }
    _save_plot(
        output_png,
        lambda: img_util.plot_skeleton_mips(
            group, patch_shape, separate_rows=True
        ),
    )


def main() -> int:
    args = parse_args()
    if args.patch_size <= 0:
        raise SystemExit("--patch-size must be positive")
    if args.split_index < 0:
        raise SystemExit("--split-index must be non-negative")

    print(f"Loading cache: {args.pkl}", flush=True)
    with args.pkl.open("rb") as handle:
        payload = pickle.load(handle)

    required = {
        "gt_graph",
        "fragments_graph",
        "gt_node_canonical_label",
        "gt_edge_error",
        "anisotropy",
        "img_path",
    }
    missing = required - payload.keys()
    if missing:
        raise RuntimeError(f"cache is missing required keys: {sorted(missing)}")

    gt = payload["gt_graph"]
    frag = payload["fragments_graph"]
    node_label = np.asarray(payload["gt_node_canonical_label"])
    edge_error = np.asarray(payload["gt_edge_error"])
    anisotropy = np.asarray(payload["anisotropy"], dtype=float)
    if anisotropy.shape != (3,) or np.any(anisotropy <= 0):
        raise RuntimeError(f"invalid xyz anisotropy: {anisotropy!r}")

    direct_splits = find_direct_splits(gt, node_label, edge_error)
    print(
        f"GT nodes={gt.number_of_nodes():,}, edges={gt.number_of_edges():,}; "
        f"direct split edges={len(direct_splits):,}"
    )
    print(
        f"Fragment nodes={frag.number_of_nodes():,}, "
        f"edges={frag.number_of_edges():,}"
    )
    print(f"anisotropy xyz (um/voxel): {tuple(anisotropy)}")

    if args.center_xyz is None:
        if not direct_splits:
            raise RuntimeError("no direct GT split edge found; provide --center-xyz")
        if args.split_index >= len(direct_splits):
            raise RuntimeError(
                f"--split-index {args.split_index} out of range; "
                f"found {len(direct_splits)} direct split edges"
            )
        edge_index, i, j, label_i, label_j = direct_splits[args.split_index]
        xyz_i = np.asarray(gt.node_xyz[i], dtype=float)
        xyz_j = np.asarray(gt.node_xyz[j], dtype=float)
        default_center_xyz = 0.5 * (xyz_i + xyz_j)
        print(
            f"Selected split #{args.split_index}: edge_index={edge_index}, "
            f"nodes=({i}, {j}), labels=({label_i}, {label_j})"
        )
        print(f"GT endpoint voxels zyx: {gt.node_voxel(i)}, {gt.node_voxel(j)}")
    else:
        default_center_xyz = np.asarray(args.center_xyz, dtype=float)

    center_xyz = (
        np.asarray(args.center_xyz, dtype=float)
        if args.center_xyz is not None
        else default_center_xyz
    )
    center_zyx = np.asarray(xyz_to_zyx_voxel(center_xyz, anisotropy), dtype=int)
    print(f"patch center xyz (um): {tuple(center_xyz)}")
    print(f"patch center zyx (voxel): {tuple(center_zyx)}")

    frag_node, frag_distance = nearest_fragment(frag, center_xyz)
    frag_xyz = np.asarray(frag.node_xyz[frag_node], dtype=float)
    frag_component = int(frag.node_component_id[frag_node])
    frag_segment = int(frag.component_id_to_swc_id[frag_component].split(".")[0])
    print(
        f"nearest fragment node={frag_node}, component={frag_component}, "
        f"segment={frag_segment}, distance={frag_distance:.3f} um, "
        f"xyz={tuple(frag_xyz)}"
    )

    image_path = str(payload["img_path"])
    print(f"Opening real image: {image_path}", flush=True)
    configure_image_access(image_path, args.gcp_credentials)
    try:
        image = img_util.TensorStoreImage(image_path)
    except Exception as exc:
        raise RuntimeError(
            "failed to open the real image. For gs:// paths, verify that the "
            "selected credential has storage.objects.get access to the bucket."
        ) from exc
    print(f"TensorStore image shape (internal 5-D): {tuple(image.shape())}")

    patch_shape = (args.patch_size,) * 3
    print(f"Reading real patch shape zyx={patch_shape} ...", flush=True)
    patch = np.asarray(image.read(tuple(center_zyx), patch_shape))
    if patch.shape != patch_shape:
        raise RuntimeError(
            f"requested patch {patch_shape}, received {patch.shape}; "
            "the center may be too close to an image boundary"
        )

    finite = patch[np.isfinite(patch)]
    if finite.size == 0:
        raise RuntimeError("real image patch contains no finite voxels")
    percentiles = np.percentile(finite, [0, 50, 99, 99.9, 100])
    print("Real patch audit:")
    print(f"  shape: {patch.shape}")
    print(f"  dtype: {patch.dtype}")
    print(f"  voxels: {patch.size:,}")
    print(f"  finite: {finite.size:,}")
    print(f"  nonzero: {np.count_nonzero(finite):,}")
    print(
        "  intensity min/median/p99/p99.9/max: "
        + ", ".join(f"{value:.3f}" for value in percentiles)
    )

    output_prefix = args.output_prefix.with_suffix("")
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    image_png = output_prefix.with_name(f"{output_prefix.name}_image.png")
    fragments_png = output_prefix.with_name(f"{output_prefix.name}_fragments.png")
    gt_png = output_prefix.with_name(f"{output_prefix.name}_gt.png")

    offset_zyx = tuple(center_zyx - args.patch_size // 2)
    gt_nodes, gt_node_components = gt.nodes_in_patch(
        offset_zyx, patch_shape, return_components=True
    )
    gt_edges, gt_edge_components = gt.edges_in_patch(
        offset_zyx, patch_shape, return_components=True
    )
    frag_nodes, frag_node_components = frag.nodes_in_patch(
        offset_zyx, patch_shape, return_components=True
    )
    frag_edges, frag_edge_components = frag.edges_in_patch(
        offset_zyx, patch_shape, return_components=True
    )
    print(
        f"Patch GT nodes/edges: {len(gt_nodes):,}/{len(gt_edges):,}; "
        f"fragment nodes/edges: {len(frag_nodes):,}/{len(frag_edges):,}"
    )

    save_image_mips(patch, image_png)
    save_skeleton_mips(
        frag_nodes,
        frag_node_components,
        frag_edges,
        frag_edge_components,
        patch_shape,
        "UNet Fragments / Segments",
        "cyan",
        fragments_png,
    )
    save_skeleton_mips(
        gt_nodes,
        gt_node_components,
        gt_edges,
        gt_edge_components,
        patch_shape,
        "GT",
        "lime",
        gt_png,
    )
    print(f"Saved image MIPs: {image_png}")
    print(f"Saved fragment/segment MIPs: {fragments_png}")
    print(f"Saved GT MIPs: {gt_png}")

    print("PASS: GT, fragments, and a real raw-image patch loaded consistently.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
