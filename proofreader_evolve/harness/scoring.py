"""
Scoring harness for the proofreader evolution loop.

The single source of truth for "is candidate proofreader A better than B?" is the
*runnable* ``segmentation_skeleton_metrics.evaluate.evaluate(...)`` — the same call
used in ``notebooks/evaluate_skeleton_metrics.ipynb``. This module wraps it so the
evolution loop can:

  1. score a *baseline* (no edits) on a chosen set of ground-truth skeletons, and
  2. score a *candidate's edit proposals* by feeding them to the metric framework.

The key insight that makes "identify + correct" scorable without re-implementing
any metric: the metric framework already models **split correction as merging
fragment labels into an equivalence class**. ``evaluate(..., label_handler=...)``
accepts a ``LabelHandler(labels=<all fragment labels>, label_pairs=[(a, b), ...])``
that collapses each proposed pair into one class before computing metrics. So an
edit proposal that "fixes a split between fragment a and fragment b" is literally
the label pair ``(a, b)``; ERL / # Splits / Edge Accuracy then reflect the fix.

Over-merging is self-penalizing: unifying labels that belong to *different* true
neurons inflates the merge metrics (``detect_label_intersections`` /
``MergeCountMetric``), so Edge Accuracy is a sound single-number fitness.

This file does NOT call any LLM. It is pure, deterministic, and runnable on its
own so you can validate the harness before wiring in the agent.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import pandas as pd

# GCS / AWS credentials must be set BEFORE tensorstore touches the cloud. The
# notebooks do this in their first cell; the harness is the single entry point to
# GCS, so we set it here at import time. Without GOOGLE_APPLICATION_CREDENTIALS,
# is_precomputed()'s GCS probe fails silently and segmentation paths are wrongly
# rejected as "Invalid image path". Resolve the token to an absolute path so it
# works from any cwd; don't clobber a value the caller already set.
_CONFIG_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "configs")
)
os.environ.setdefault(
    "GOOGLE_APPLICATION_CREDENTIALS",
    os.path.join(_CONFIG_DIR, "zihan_gcs_token.json"),
)
os.environ.setdefault("AWS_EC2_METADATA_DISABLED", "true")

from segmentation_skeleton_metrics.evaluate import evaluate
from segmentation_skeleton_metrics.data_handling.graph_loading import LabelHandler
from segmentation_skeleton_metrics.utils.img_util import TensorStoreImage

# Shared dataset-config helper (brain_id -> segmentation_id), same as the notebooks.
_SCRIPTS = os.path.join(os.path.dirname(__file__), "..", "..", "scripts")
sys.path.insert(0, os.path.abspath(_SCRIPTS))
from dataset_config import get_segmentation_id  # noqa: E402

# dataset_config defaults to a cwd-relative "../configs/..." path (it assumes a
# notebook running in notebooks/). We run from the project root, so resolve the
# config to an ABSOLUTE path relative to this file — works from any cwd.
_CONFIG_RTF = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "configs",
                 "segmentation_datasets.rtf")
)


# The headline metrics we track each generation. Edge Accuracy is the primary
# fitness (higher is better); the rest are reported for diagnosis.
#
# Edge Accuracy = 100 - (% Split Edges + % Omit Edges + % Merged Edges), so it is
# the ONLY single number that charges split, omit, AND merge errors together. The
# project targets merge correction (a split_label edit must reduce % Merged Edges
# / # Merges without manufacturing splits), and a merge-only fitness like Split
# Accuracy would be blind to exactly the error class we are repairing — so Edge
# Accuracy is the correct primary. ``Split Accuracy`` (= 100 - % Split Edges, the
# run-length-weighted split-coverage measure) is retained as an AUXILIARY tracked
# metric: it isolates the split-error component so the report/gate can watch that
# a merge repair does not over-split. It is materialized per skeleton by
# ``_ensure_derived`` so every existing ``_weighted_avg`` / subset path works.
PRIMARY_METRIC = "Edge Accuracy"
TRACKED_METRICS = [
    "Edge Accuracy",
    "Split Accuracy",      # auxiliary: 100 - % Split Edges (over-split watchdog)
    "ERL",
    "Normalized ERL",
    "# Splits",
    "# Merges",
    "% Split Edges",
    "% Omit Edges",
    "% Merged Edges",
]


def _ensure_derived(per_swc: pd.DataFrame) -> pd.DataFrame:
    """Materialize the auxiliary ``Split Accuracy`` column (100 - % Split Edges).

    ``evaluate()`` and the incremental scorer both emit ``% Split Edges`` per
    skeleton but not ``Split Accuracy`` (our split-coverage restatement). Adding
    it here lets the standard ``_weighted_avg(per_swc, "Split Accuracy")`` path
    yield the run-length-weighted value with no special-casing downstream.
    Idempotent; a no-op if the column already exists or ``% Split Edges`` is
    missing.
    """
    if "% Split Edges" in per_swc.columns and "Split Accuracy" not in per_swc.columns:
        per_swc = per_swc.copy()
        per_swc["Split Accuracy"] = 100.0 - per_swc["% Split Edges"]
    return per_swc


@dataclass
class BrainPaths:
    """Resolves and holds the GCS paths for one brain, mirroring the notebooks."""

    brain_id: str
    anisotropy: tuple = (0.748, 0.748, 1.0)
    segmentation_id: str = field(default="", init=False)

    def __post_init__(self):
        self.segmentation_id = get_segmentation_id(self.brain_id, rtf_path=_CONFIG_RTF)

    @property
    def gt_path(self) -> str:
        return f"gs://allen-nd-goog/ground_truth_tracings/{self.brain_id}/voxel"

    @property
    def fragments_path(self) -> str:
        return (
            f"gs://allen-nd-goog/from_google/{self.brain_id}"
            f"/whole_brain/{self.segmentation_id}/swcs"
        )

    @property
    def segmentation_path(self) -> str:
        return (
            f"gs://allen-nd-goog/from_google/{self.brain_id}"
            f"/whole_brain/{self.segmentation_id}/"
        )


@dataclass
class ScoreResult:
    """Outcome of one ``evaluate(...)`` run, parsed back into Python."""

    primary: float                 # weighted-avg Edge Accuracy (the fitness)
    metrics: dict                  # weighted-avg of every tracked metric
    per_swc: pd.DataFrame          # the full results.csv (one row per GT skeleton)
    output_dir: str                # where evaluate() wrote results.csv / overview
    seconds: float                 # wall-clock for this evaluate() call
    merge_sites: pd.DataFrame | None = None  # per-merge attribution (GT id, label,
                                             # location); None if not captured

    def summary(self) -> str:
        cells = ", ".join(f"{k}={self.metrics.get(k, float('nan')):.4f}"
                           for k in TRACKED_METRICS)
        return f"[{self.primary:.4f} {PRIMARY_METRIC}] {cells} ({self.seconds:.0f}s)"


def _weighted_avg(results: pd.DataFrame, column: str) -> float:
    """Run-length-weighted average of a metric column (matches evaluate's report).

    ``segmentation_skeleton_metrics`` reports averages weighted by SWC Run Length;
    we reproduce that here so our fitness equals the number you see in the notebook.
    """
    if column not in results.columns:
        return float("nan")
    weights = results["SWC Run Length"]
    values = results[column]
    mask = values.notna()
    if mask.sum() == 0 or weights[mask].sum() == 0:
        return float("nan")
    return float((values[mask] * weights[mask]).sum() / weights[mask].sum())


def _parse_results(output_dir: str, elapsed: float, prefix: str = "") -> ScoreResult:
    csv_path = os.path.join(output_dir, f"{prefix}results.csv")
    per_swc = pd.read_csv(csv_path, index_col=0)
    per_swc = _ensure_derived(per_swc)  # add Split Accuracy = 100 - % Split Edges
    metrics = {m: _weighted_avg(per_swc, m) for m in TRACKED_METRICS}
    return ScoreResult(
        primary=metrics[PRIMARY_METRIC],
        metrics=metrics,
        per_swc=per_swc,
        output_dir=output_dir,
        seconds=elapsed,
    )


def score(
    paths: BrainPaths,
    output_dir: str,
    label_pairs: Sequence[tuple] | None = None,
    all_fragment_labels: Iterable[str] | None = None,
    gt_swc_names: Iterable[str] | None = None,
    verbose: bool = False,
) -> ScoreResult:
    """Run the metric framework and return parsed metrics.

    Parameters
    ----------
    paths : BrainPaths
        Resolved GCS paths for the brain under evaluation.
    output_dir : str
        Directory evaluate() writes results.csv / results_overview.txt into.
    label_pairs : sequence of (label, label), optional
        Proposed split-correction edits. Each pair is two *fragment labels*
        (segment IDs, as strings) the candidate believes belong to the same
        neuron and should be unified. Passed straight into a ``LabelHandler``
        so the metric framework re-scores as if the split were fixed. ``None``
        or empty scores the raw segmentation (the baseline).
    all_fragment_labels : iterable of str, optional
        Every valid fragment label for this brain. Required by ``LabelHandler``
        whenever ``label_pairs`` is given (it builds equivalence classes over
        this universe). Get it once from ``list_fragment_labels(...)``.
    gt_swc_names : iterable of str, optional
        Restrict evaluation to this subset of ground-truth skeletons (the
        train/held-out split). Currently advisory: evaluate() loads all GT in a
        directory, so subsetting is applied by filtering ``per_swc`` rows below.
    verbose : bool
        Forwarded to evaluate() (progress bars + printed summary).

    Returns
    -------
    ScoreResult
    """
    os.makedirs(output_dir, exist_ok=True)

    # swap_axes=True matches evaluate_skeleton_metrics.ipynb: after the
    # from_google transpose the volume is (z, y, x) but the SWC voxels are
    # (x, y, z); without this you get an OUT_OF_RANGE on the first labeling pass.
    segmentation = TensorStoreImage(paths.segmentation_path, swap_axes=True)

    label_handler = None
    if label_pairs:
        if all_fragment_labels is None:
            raise ValueError(
                "label_pairs given but all_fragment_labels is None — "
                "LabelHandler needs the full label universe to build "
                "equivalence classes. Call list_fragment_labels() once and "
                "pass the result here."
            )
        label_handler = LabelHandler(
            labels=set(map(str, all_fragment_labels)),
            label_pairs=[(str(a), str(b)) for a, b in label_pairs],
        )

    t0 = time.monotonic()
    evaluate(
        paths.gt_path,
        segmentation,
        output_dir,
        anisotropy=paths.anisotropy,
        fragments_path=paths.fragments_path,
        label_handler=label_handler,
        use_anisotropy=False,
        save_merges=False,
        save_fragments=False,
        verbose=verbose,
    )
    elapsed = time.monotonic() - t0

    result = _parse_results(output_dir, elapsed)

    # Optional subsetting to the train / held-out GT skeletons. evaluate() scores
    # every GT skeleton in the directory; we restrict the *reported* fitness to
    # the requested names so train feedback never leaks held-out signal.
    if gt_swc_names is not None:
        names = set(gt_swc_names)
        subset = result.per_swc[result.per_swc.index.isin(names)]
        if len(subset) > 0:
            metrics = {m: _weighted_avg(subset, m) for m in TRACKED_METRICS}
            result = ScoreResult(
                primary=metrics[PRIMARY_METRIC],
                metrics=metrics,
                per_swc=subset,
                output_dir=output_dir,
                seconds=elapsed,
            )
    return result


if __name__ == "__main__":
    # Smoke test: score the raw segmentation (no edits) for one brain.
    # Run from the project root:  python proofreader_evolve/harness/scoring.py 789202
    brain = sys.argv[1] if len(sys.argv) > 1 else "789202"
    paths = BrainPaths(brain)
    print(f"brain_id={brain} -> segmentation_id={paths.segmentation_id}")
    out = os.path.join(os.path.dirname(__file__), "..", "runs", f"_smoke_{brain}")
    res = score(paths, os.path.abspath(out), verbose=True)
    print("\nBASELINE:", res.summary())
