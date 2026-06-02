"""
Before/after statistics for neuron-proofreader outputs.

The comparison uses segmentation-skeleton-metrics so that the reported errors
match the existing evaluation notebook:
- # Splits, # Merges
- % Split, Omit, and Merged Edges
- ERL, Normalized ERL, Edge Accuracy, Split Rate, Merge Rate
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
METRICS_SRC = REPO_ROOT / "segmentation-skeleton-metrics" / "src"
if str(METRICS_SRC) not in sys.path:
    sys.path.insert(0, str(METRICS_SRC))


ERROR_METRICS = [
    "# Splits",
    "# Merges",
    "% Split Edges",
    "% Omit Edges",
    "% Merged Edges",
]
QUALITY_METRICS = [
    "ERL",
    "Normalized ERL",
    "Edge Accuracy",
    "Split Rate",
    "Merge Rate",
]


def read_connection_pairs(path: str | Path) -> set[tuple[str, str]]:
    """Read accepted proofreading segment-pairs from connections.txt."""
    path = Path(path)
    if not path.exists():
        return set()

    pairs = set()
    with path.open() as f:
        for line in f:
            parts = [p.strip() for p in line.split(",") if p.strip()]
            if len(parts) == 2:
                pairs.add((parts[0], parts[1]))
    return pairs


def read_segment_labels(path: str | Path) -> set[str]:
    """Read original segment IDs saved by InferencePipeline."""
    path = Path(path)
    if not path.exists():
        return set()

    labels = set()
    with path.open() as f:
        for line in f:
            label = line.strip()
            if label:
                labels.add(label)
    return labels


def evaluate_reconstruction(
    gt_path: str,
    segmentation_path: str,
    fragments_path: str,
    output_dir: str | Path,
    results_prefix: str,
    anisotropy: tuple[float, float, float],
    label_pairs: Iterable[tuple[str, str]] = (),
    labels: Iterable[str] = (),
    use_anisotropy: bool = False,
    save_merges: bool = False,
    verbose: bool = True,
) -> Path:
    """Run segmentation-skeleton-metrics for one reconstruction state."""
    from segmentation_skeleton_metrics.data_handling.graph_loading import LabelHandler
    from segmentation_skeleton_metrics.evaluate import evaluate
    from segmentation_skeleton_metrics.utils.img_util import TensorStoreImage

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    segmentation = TensorStoreImage(segmentation_path, swap_axes=True)
    label_pairs = set(label_pairs)
    labels = set(labels)
    label_handler = (
        LabelHandler(labels=labels, label_pairs=label_pairs)
        if label_pairs or labels
        else None
    )

    evaluate(
        gt_path,
        segmentation,
        str(output_dir),
        anisotropy=anisotropy,
        fragments_path=fragments_path,
        label_handler=label_handler,
        results_prefix=results_prefix,
        save_merges=save_merges,
        save_fragments=False,
        use_anisotropy=use_anisotropy,
        verbose=verbose,
    )
    return output_dir / f"{results_prefix}_results.csv"


def aggregate_metric(results: pd.DataFrame, metric: str) -> float:
    """Match Evaluator.report_summary aggregation behavior."""
    if metric not in results:
        return float("nan")
    if metric in {"# Splits", "# Merges"}:
        return float(results[metric].sum())
    return compute_weighted_average(results, metric)


def compute_weighted_average(results: pd.DataFrame, metric: str) -> float:
    values = results[metric]
    weights = results["SWC Run Length"]
    mask = values.notna() & weights.notna()
    values = values[mask]
    weights = weights[mask]
    if weights.sum() == 0:
        return float("nan")
    return float((values * weights).sum() / weights.sum())


def compare_results(
    before_results_path: str | Path,
    after_results_path: str | Path,
    output_dir: str | Path,
) -> pd.DataFrame:
    """Create aggregate before/after comparison and save CSV/TXT reports."""
    before = pd.read_csv(before_results_path, index_col=0)
    after = pd.read_csv(after_results_path, index_col=0)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for metric in ERROR_METRICS + QUALITY_METRICS:
        if metric not in before.columns or metric not in after.columns:
            continue

        before_value = aggregate_metric(before, metric)
        after_value = aggregate_metric(after, metric)
        delta = after_value - before_value
        reduction = before_value - after_value
        corrected_percent = (
            100 * reduction / before_value if before_value not in (0, 0.0) else float("nan")
        )
        relative_change_percent = (
            100 * delta / before_value if before_value not in (0, 0.0) else float("nan")
        )

        rows.append(
            {
                "Metric": metric,
                "Before": before_value,
                "After": after_value,
                "Delta": delta,
                "Reduction": reduction if metric in ERROR_METRICS else float("nan"),
                "Corrected %": corrected_percent if metric in ERROR_METRICS else float("nan"),
                "Relative Change %": relative_change_percent,
            }
        )

    comparison = pd.DataFrame(rows).set_index("Metric")
    comparison_path = output_dir / "proofreading_comparison.csv"
    comparison.to_csv(comparison_path)
    write_text_report(comparison, output_dir / "proofreading_comparison.txt")
    return comparison


def write_text_report(comparison: pd.DataFrame, path: str | Path) -> None:
    lines = ["Proofreading Before/After Summary", ""]
    lines.append("Error reductions")
    for metric in ERROR_METRICS:
        if metric not in comparison.index:
            continue
        row = comparison.loc[metric]
        lines.append(
            "  "
            f"{metric}: {row['Before']:.4f} -> {row['After']:.4f} "
            f"(reduction {row['Reduction']:.4f}, corrected {row['Corrected %']:.2f}%)"
        )

    lines.append("")
    lines.append("Quality changes")
    for metric in QUALITY_METRICS:
        if metric not in comparison.index:
            continue
        row = comparison.loc[metric]
        lines.append(
            "  "
            f"{metric}: {row['Before']:.4f} -> {row['After']:.4f} "
            f"(delta {row['Delta']:.4f}, relative {row['Relative Change %']:.2f}%)"
        )

    Path(path).write_text("\n".join(lines) + "\n")


def evaluate_before_after(
    gt_path: str,
    segmentation_path: str,
    original_fragments_path: str,
    proofreader_output_dir: str | Path,
    metrics_output_dir: str | Path,
    anisotropy: tuple[float, float, float],
    corrected_fragments_path: str | Path | None = None,
    use_anisotropy: bool = False,
    save_merges: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """Evaluate original and proofread reconstructions and summarize deltas."""
    proofreader_output_dir = Path(proofreader_output_dir)
    metrics_output_dir = Path(metrics_output_dir)
    corrected_fragments_path = corrected_fragments_path or (
        proofreader_output_dir / "corrected_swcs" / "swcs.zip"
    )
    if not Path(corrected_fragments_path).exists():
        corrected_fragments_path = original_fragments_path

    label_pairs = read_connection_pairs(proofreader_output_dir / "connections.txt")
    labels = read_segment_labels(proofreader_output_dir / "segment_ids.txt")
    if label_pairs and not labels:
        labels = {label for pair in label_pairs for label in pair}

    before_path = evaluate_reconstruction(
        gt_path=gt_path,
        segmentation_path=segmentation_path,
        fragments_path=original_fragments_path,
        output_dir=metrics_output_dir,
        results_prefix="before",
        anisotropy=anisotropy,
        use_anisotropy=use_anisotropy,
        save_merges=save_merges,
        verbose=verbose,
    )
    after_path = evaluate_reconstruction(
        gt_path=gt_path,
        segmentation_path=segmentation_path,
        fragments_path=str(corrected_fragments_path),
        output_dir=metrics_output_dir,
        results_prefix="after",
        anisotropy=anisotropy,
        label_pairs=label_pairs,
        labels=labels,
        use_anisotropy=use_anisotropy,
        save_merges=save_merges,
        verbose=verbose,
    )
    return compare_results(before_path, after_path, metrics_output_dir)


def print_comparison(comparison: pd.DataFrame) -> None:
    print("\nProofreading before/after comparison")
    print(comparison.to_string(float_format=lambda x: f"{x:.4f}"))
