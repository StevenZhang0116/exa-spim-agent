"""
Train a neuron-proofreader split-correction model, then run proofreading
inference with the trained checkpoint.

This script follows the neuron-proofreader library pipeline:
1. Build a labeled FragmentsDatasetCollection from predicted SWCs and GT SWCs.
2. Generate split-correction proposals and labels.
3. Train VisionHGAT with Trainer.run(...), bounded by epochs and/or steps.
4. Load the trained checkpoint into InferencePipeline.
5. Save corrected SWCs and proposal/connection summaries.
6. Optionally run before/after skeleton metrics and summarize correction rates.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[2]
NEURON_PROOFREADER_SRC = REPO_ROOT / "neuron-proofreader" / "src"
if str(NEURON_PROOFREADER_SRC) not in sys.path:
    sys.path.insert(0, str(NEURON_PROOFREADER_SRC))


DEFAULT_BRAIN_ID = "794495"
DEFAULT_SEGMENTATION_ID = "raw.unet_449_splits_and_merges_900000"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train VisionHGAT on neuron-proofreader proposals and run split "
            "proofreading inference with the trained model."
        )
    )

    parser.add_argument("--brain-id", default=DEFAULT_BRAIN_ID)
    parser.add_argument("--segmentation-id", default=DEFAULT_SEGMENTATION_ID)
    parser.add_argument("--fragments-path", default=None)
    parser.add_argument("--gt-path", default=None)
    parser.add_argument("--img-path", default=None)
    parser.add_argument("--segmentation-path", default=None)
    parser.add_argument(
        "--image-prefix-config",
        default=str(REPO_ROOT / "exa-spim-agent" / "configs" / "exaspim_image_prefixes.json"),
    )
    parser.add_argument("--gcp-credentials", default=None)

    parser.add_argument("--val-fragments-path", default=None)
    parser.add_argument("--val-gt-path", default=None)
    parser.add_argument("--val-img-path", default=None)

    parser.add_argument("--output-dir", default="proofreader_output")
    parser.add_argument("--training-output-dir", default="training_output")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-inference", action="store_true")
    parser.add_argument("--skip-stats", action="store_true")
    parser.add_argument("--metrics-output-dir", default=None)

    parser.add_argument("--max-epochs", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])

    parser.add_argument("--search-radius", type=float, default=20.0)
    parser.add_argument("--min-threshold", type=float, default=0.75)
    parser.add_argument("--threshold-dt", type=float, default=0.05)
    parser.add_argument("--removal-threshold", type=float, default=0.3)

    parser.add_argument("--anisotropy", type=float, nargs=3, default=(0.748, 0.748, 1.0))
    parser.add_argument("--min-cable-length", type=float, default=40.0)
    parser.add_argument("--node-spacing", type=float, default=1.0)
    parser.add_argument("--prune-depth", type=float, default=24.0)
    parser.add_argument("--max-proposals-per-leaf", type=int, default=3)
    parser.add_argument("--allow-nonleaf-proposals", action="store_true")
    parser.add_argument("--remove-high-risk-merges", action="store_true")
    parser.add_argument("--keep-doubles", action="store_true")

    parser.add_argument("--brightness-clip", type=int, default=400)
    parser.add_argument("--patch-shape", type=int, nargs=3, default=(96, 96, 96))
    parser.add_argument("--disable-msg-passing", action="store_true")

    return parser.parse_args()


def default_fragments_path(brain_id: str, segmentation_id: str) -> str:
    return (
        f"gs://allen-nd-goog/from_google/{brain_id}/whole_brain/"
        f"{segmentation_id}/swcs"
    )


def default_gt_path(brain_id: str) -> str:
    return f"gs://allen-nd-goog/ground_truth_tracings/{brain_id}/voxel"


def default_segmentation_path(brain_id: str, segmentation_id: str) -> str:
    return (
        f"gs://allen-nd-goog/from_google/{brain_id}/whole_brain/"
        f"{segmentation_id}/"
    )


def resolve_img_path(args: argparse.Namespace) -> str:
    if args.img_path:
        return args.img_path

    with open(args.image_prefix_config) as f:
        img_prefixes = json.load(f)
    try:
        return img_prefixes[args.brain_id]
    except KeyError as exc:
        raise KeyError(
            f"No image path for brain_id={args.brain_id!r} in "
            f"{args.image_prefix_config}"
        ) from exc


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def build_config(args: argparse.Namespace, device: str) -> Config:
    from neuron_proofreader.config import Config, MLConfig, ProposalGraphConfig

    graph_config = ProposalGraphConfig(
        anisotropy=tuple(args.anisotropy),
        min_cable_length=args.min_cable_length,
        node_spacing=args.node_spacing,
        prune_depth=args.prune_depth,
        max_proposals_per_leaf=args.max_proposals_per_leaf,
        allow_nonleaf_proposals=args.allow_nonleaf_proposals,
        remove_high_risk_merges=args.remove_high_risk_merges,
        remove_doubles=not args.keep_doubles,
        verbose=True,
    )
    ml_config = MLConfig(
        batch_size=args.batch_size,
        brightness_clip=args.brightness_clip,
        device=device,
        patch_shape=tuple(args.patch_shape),
        transform=False,
    )
    return Config(graph_config, ml_config)


def build_collection(
    key: str,
    fragments_path: str,
    img_path: str,
    gt_path: str,
    config: Config,
    search_radius: float,
    shuffle: bool,
) -> FragmentsDatasetCollection:
    from neuron_proofreader.split_proofreading.split_datasets import (
        FragmentsDatasetCollection,
    )

    dataset = FragmentsDatasetCollection(shuffle=shuffle)
    dataset.add_dataset(
        key=key,
        fragments_path=fragments_path,
        img_path=img_path,
        config=config,
        gt_path=gt_path,
    )
    dataset.generate_proposals(search_radius)
    return dataset


def latest_checkpoint(log_dir: str | Path) -> str | None:
    paths = sorted(Path(log_dir).glob("*.pth"), key=lambda p: p.stat().st_mtime)
    return str(paths[-1]) if paths else None


def train_model(
    args: argparse.Namespace,
    config: Config,
    fragments_path: str,
    gt_path: str,
    img_path: str,
) -> str:
    from neuron_proofreader.machine_learning.gnn_models import VisionHGAT
    from neuron_proofreader.machine_learning.train import Trainer

    train_dataset = build_collection(
        key="train",
        fragments_path=fragments_path,
        img_path=img_path,
        gt_path=gt_path,
        config=config,
        search_radius=args.search_radius,
        shuffle=True,
    )

    if args.val_fragments_path or args.val_gt_path or args.val_img_path:
        val_dataset = build_collection(
            key="val",
            fragments_path=args.val_fragments_path or fragments_path,
            img_path=args.val_img_path or img_path,
            gt_path=args.val_gt_path or gt_path,
            config=config,
            search_radius=args.search_radius,
            shuffle=False,
        )
    else:
        val_dataset = train_dataset

    train_loader = DataLoader(
        train_dataset,
        batch_size=None,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=None,
        num_workers=args.num_workers,
    )

    model = VisionHGAT(
        patch_shape=config.ml.patch_shape,
        disable_msg_passing=args.disable_msg_passing,
    )
    trainer = Trainer(
        model=model,
        model_name="VisionHGAT",
        output_dir=args.training_output_dir,
        device=config.ml.device,
        lr=args.learning_rate,
        max_epochs=args.max_epochs,
        max_steps=args.max_steps,
    )
    trainer.run(train_loader, val_loader)

    checkpoint_path = trainer.latest_checkpoint_path or latest_checkpoint(trainer.log_dir)
    if checkpoint_path is None:
        checkpoint_path = trainer.save_model("final")
    print(f"Trained checkpoint: {checkpoint_path}")
    return checkpoint_path


def run_inference(
    args: argparse.Namespace,
    config: Config,
    fragments_path: str,
    img_path: str,
    checkpoint_path: str,
) -> None:
    from neuron_proofreader.machine_learning.gnn_models import VisionHGAT
    from neuron_proofreader.split_proofreading.split_inference import InferencePipeline

    model = VisionHGAT(
        patch_shape=config.ml.patch_shape,
        disable_msg_passing=args.disable_msg_passing,
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location=config.ml.device))

    pipeline = InferencePipeline(
        fragments_path=fragments_path,
        img_path=img_path,
        output_dir=args.output_dir,
        model=model,
        config=config,
    )
    pipeline(
        search_radius=args.search_radius,
        min_threshold=args.min_threshold,
        dt=args.threshold_dt,
        removal_threshold=args.removal_threshold,
    )
    print(f"Proofreading output: {args.output_dir}")


def run_statistics(
    args: argparse.Namespace,
    gt_path: str,
    segmentation_path: str,
    fragments_path: str,
) -> None:
    from proofreading_stats import evaluate_before_after, print_comparison

    metrics_output_dir = args.metrics_output_dir or os.path.join(
        args.output_dir,
        "metrics_before_after",
    )
    comparison = evaluate_before_after(
        gt_path=gt_path,
        segmentation_path=segmentation_path,
        original_fragments_path=fragments_path,
        proofreader_output_dir=args.output_dir,
        metrics_output_dir=metrics_output_dir,
        anisotropy=tuple(args.anisotropy),
        use_anisotropy=False,
        save_merges=True,
        verbose=True,
    )
    print_comparison(comparison)
    print(f"Before/after metric reports: {metrics_output_dir}")


def main() -> None:
    args = parse_args()
    if args.gcp_credentials:
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = args.gcp_credentials
    os.environ.setdefault("AWS_EC2_METADATA_DISABLED", "true")

    fragments_path = args.fragments_path or default_fragments_path(
        args.brain_id,
        args.segmentation_id,
    )
    gt_path = args.gt_path or default_gt_path(args.brain_id)
    img_path = resolve_img_path(args)
    segmentation_path = args.segmentation_path or default_segmentation_path(
        args.brain_id,
        args.segmentation_id,
    )
    device = resolve_device(args.device)
    config = build_config(args, device)

    checkpoint_path = args.checkpoint
    if not args.skip_training:
        checkpoint_path = train_model(args, config, fragments_path, gt_path, img_path)

    if not args.skip_inference:
        if checkpoint_path is None:
            raise ValueError(
                "Inference requires a checkpoint. Provide --checkpoint or run "
                "without --skip-training."
        )
        run_inference(args, config, fragments_path, img_path, checkpoint_path)

    if not args.skip_stats:
        run_statistics(args, gt_path, segmentation_path, fragments_path)


if __name__ == "__main__":
    main()
