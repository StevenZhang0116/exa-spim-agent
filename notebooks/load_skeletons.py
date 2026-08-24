#!/usr/bin/env python3
"""Build a skeleton dataset cache and save diagnostic image-patch figures."""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

import matplotlib
import numpy as np

NOTEBOOKS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = NOTEBOOKS_DIR.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
CONFIG_DIR = PROJECT_ROOT / "configs"
CACHE_DIR = PROJECT_ROOT / "cache"
FIGURE_DIR = NOTEBOOKS_DIR / "fig"

# The plotting helpers call plt.show(). Agg makes those calls non-interactive on
# compute nodes while leaving the generated figure available for savefig().
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

os.environ.setdefault(
    "GOOGLE_APPLICATION_CREDENTIALS",
    str(CONFIG_DIR / "zihan_gcs_token.json"),
)
os.environ.setdefault("AWS_EC2_METADATA_DISABLED", "true")

from agentic_neuron_proofreader.data_modules.datasets import BrainDataset
from agentic_neuron_proofreader.utils import img_util, util
from dataset_config import get_img_path, get_segmentation_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load GT and fragment skeletons, optionally save the dataset cache, "
            "and save raw-image and segmentation MIP figures."
        )
    )
    parser.add_argument("--brain-id", default="718162")
    parser.add_argument(
        "--anisotropy",
        type=float,
        nargs=3,
        default=(0.748, 0.748, 1.0),
        metavar=("X", "Y", "Z"),
    )
    parser.add_argument("--min-cable-length", type=float, default=10.0)
    parser.add_argument("--node-spacing", type=float, default=2.0)
    parser.add_argument(
        "--patch-shape",
        type=int,
        nargs=3,
        default=(128, 128, 128),
        metavar=("Z", "Y", "X"),
    )
    parser.add_argument(
        "--use-groundtruth",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Sample the patch center from GT; use --no-use-groundtruth for fragments.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--figure-dpi", type=int, default=200)
    parser.add_argument(
        "--overwrite-cache",
        action="store_true",
        help="Overwrite an existing cache after rebuilding the dataset.",
    )
    parser.add_argument(
        "--skip-cache-save",
        action="store_true",
        help="Build the dataset but do not save its cache.",
    )
    return parser.parse_args()


def number_tag(value: float) -> str:
    """Format numeric parameters compactly for filenames."""
    return f"{value:g}"


def safe_token(value: object) -> str:
    """Make a filesystem-safe identifier from a sampled graph node."""
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in str(value))


def save_current_figure(path: Path, dpi: int) -> None:
    """Save and close the figure created by an img_util plotting helper."""
    figure = plt.gcf()
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved figure: {path}")


def main() -> None:
    args = parse_args()
    if args.overwrite_cache and args.skip_cache_save:
        raise ValueError(
            "--overwrite-cache and --skip-cache-save cannot be used together"
        )

    random.seed(args.seed)
    np.random.seed(args.seed)

    segmentation_id = get_segmentation_id(
        args.brain_id,
        rtf_path=str(CONFIG_DIR / "segmentation_datasets.rtf"),
    )
    print(f"brain_id={args.brain_id} -> segmentation_id={segmentation_id}")

    gt_path = (
        f"gs://allen-nd-goog/ground_truth_tracings/{args.brain_id}/voxel"
    )
    fragments_path = (
        f"gs://allen-nd-goog/from_google/{args.brain_id}/whole_brain/"
        f"{segmentation_id}/swcs"
    )
    segmentation_path = (
        f"gs://allen-nd-goog/from_google/{args.brain_id}/whole_brain/"
        f"{segmentation_id}/"
    )
    img_path = get_img_path(
        args.brain_id,
        prefixes_path=str(CONFIG_DIR / "exaspim_image_prefixes.json"),
    )

    dataset = BrainDataset(
        fragments_path,
        gt_path,
        img_path,
        anisotropy=tuple(args.anisotropy),
        min_cable_length=args.min_cable_length,
        node_spacing=args.node_spacing,
    )
    print(dataset.fragments_graph.summary(prefix="Fragments"))
    print(dataset.gt_graph.summary(prefix="GroundTruth"))

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / (
        f"dataset_cache_{args.brain_id}_mcl"
        f"{number_tag(args.min_cable_length)}.pkl"
    )
    if args.skip_cache_save:
        print("Skipped dataset cache save (--skip-cache-save).")
    elif cache_path.exists() and not args.overwrite_cache:
        print(f"Preserved existing dataset cache: {cache_path}")
    else:
        dataset.save(str(cache_path))
        print(
            f"Saved dataset cache: {cache_path} "
            f"({cache_path.stat().st_size / 1e9:.2f} GB)"
        )

    segmentation = img_util.TensorStoreImage(segmentation_path)
    graph = dataset.gt_graph if args.use_groundtruth else dataset.fragments_graph
    graph_source = "groundtruth" if args.use_groundtruth else "fragments"
    node = util.sample_once(graph.nodes)
    voxel = graph.node_voxel(node)

    img_patch = dataset.img.read(voxel, tuple(args.patch_shape))
    segmentation_patch = segmentation.read(voxel, tuple(args.patch_shape))

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    figure_stem = (
        f"{args.brain_id}_mcl{number_tag(args.min_cable_length)}_"
        f"{graph_source}_node{safe_token(node)}"
    )

    img_util.plot_mips(img_patch)
    save_current_figure(
        FIGURE_DIR / f"{figure_stem}_raw_mips.png",
        args.figure_dpi,
    )

    binary_segmentation = (segmentation_patch > 0).astype(np.uint8)
    img_util.plot_segmentation_mips(binary_segmentation)
    save_current_figure(
        FIGURE_DIR / f"{figure_stem}_segmentation_mips.png",
        args.figure_dpi,
    )


if __name__ == "__main__":
    main()
