"""
Created on Mon Aug 18 09:00:00 2026

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Computes skeleton-based metrics for each step of the automated proofreading pipeline.

"""

from botocore.exceptions import ClientError
from copy import deepcopy
from segmentation_skeleton_metrics.utils import util
from segmentation_skeleton_metrics.utils import graph_util as _gutil
from scipy.spatial import KDTree

import argparse
import boto3
import os
import pandas as pd

import segmentation_skeleton_metrics.skeleton_metrics as _ssm

from neuron_proofreader.utils import util as np_util
from neuron_proofreading_evaluation.results_manager import ResultsManager
from neuron_proofreading_evaluation import datamodules as data_util
from neuron_proofreading_evaluation.metrics import merge_metrics
from neuron_proofreading_evaluation.metrics import skeleton_metrics
from neuron_proofreading_evaluation.metrics import split_metrics
from neuron_proofreading_evaluation import utils as npe_util

DT = 0.02

# --- Merge-site filtering patch ---
# Filters merge sites at detection time so ERL and all derived metrics are
# computed correctly without any post-hoc recomputation.

_ignore_kdtree = None  # set per step in main() (from the step's ignored_merges.swc)
_ignore_dist = 30

labels = None  # set in main()
results_manager = None  # set in main()

_orig_record_merge_site = _ssm.MergeCountMetric.record_merge_site


def _filtered_record_merge_site(self, gt_graph, fragment_graph, fragment_node, component):
    if _ignore_kdtree is not None:
        adj_node = _gutil.search_branching_node(fragment_graph, gt_graph.kdtree, fragment_node)
        dist, _ = _ignore_kdtree.query(fragment_graph.node_xyz(adj_node))
        if dist <= _ignore_dist:
            return
    _orig_record_merge_site(self, gt_graph, fragment_graph, fragment_node, component)


_ssm.MergeCountMetric.record_merge_site = _filtered_record_merge_site


def main():
    global _ignore_kdtree, eval_steps, labels, results_manager

    # Report experiment
    print("Brain ID:", brain_id)
    print("Segmentation ID:", segmentation_id)
    print("Result Prefix:", s3_prefix)

    # Load data
    results_manager = ResultsManager(s3_prefix, brain_id, segmentation_id)
    eval_steps = select_steps(results_manager, steps)
    gt_graphs, labels = load_groundtruth(results_manager)

    # Skeleton metrics for each step
    print("\n--- Part 1: Skeleton Metrics ---")
    for step_name in eval_steps:
        # Accumulate connections.txt from every preceding step so absorbed
        # segment IDs are correctly remapped during GT relabeling.
        label_handler = build_label_handler_upto(step_name)

        # Merge sites to ignore live next to the step's results on S3
        _ignore_kdtree = load_ignored_merges(step_name)

        skeleton_metrics.compute(
            results_manager,
            deepcopy(gt_graphs),
            step_name,
            output_dir,
            ignore_kdtree=None,
            label_handler=label_handler,
        )

    # In-depth precision-recall analysis
    split_steps = [s for s in results_manager.eval_steps if results_manager.step_types.get(s) == "split"]

    if analyze_merges:
        print("\n--- Part 2a: Merge Correction Precision-Recall ---")
        compute_merge_precision_recall()

    if analyze_splits:
        print("\n--- Part 2b: Split Correction Precision-Recall ---")
        for step_name in split_steps:
            compute_precision_recall(deepcopy(gt_graphs), step_name, only_leaf2leaf=True)
            compute_precision_recall(deepcopy(gt_graphs), step_name, only_leaf2leaf=False)
            compute_multiround_precision_recall(deepcopy(gt_graphs), step_name)


def select_steps(results_manager, steps=None):
    """
    Resolves the "steps" argument to a subset of the steps discovered in the
    result prefix, in pipeline order.

    Each entry names a step by its "stepN" prefix (e.g. "step2") or by its
    full directory name. None selects every step that was found.

    Note that "results_manager.eval_steps" is deliberately left untouched:
    the label handler and the merge-corrected fragment lookup both need the
    full pipeline order even when only a later step is evaluated.
    """
    all_steps = list(results_manager.eval_steps)
    names = [n.strip() for s in steps or [] for n in s.split(",") if n.strip()]
    if not names:
        return all_steps

    selected = set()
    for name in names:
        matches = [
            s for s in all_steps if s == name or s.startswith(f"{name}_")
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Step '{name}' matched {len(matches)} steps, expected 1. "
                f"Steps found in the result prefix: {', '.join(all_steps)}"
            )
        selected.update(matches)

    ordered = [s for s in all_steps if s in selected]
    print(f"\nEvaluating {len(ordered)}/{len(all_steps)} steps: {ordered}")
    return ordered


def compute_merge_precision_recall():
    for step_name, step_dir in results_manager.step_dirs.items():
        if step_name not in eval_steps:
            continue
        if not util.list_cloud_paths(step_dir, extension="model_predictions.csv"):
            continue

        print(f"\nStep: Merge Precision-Recall for {step_name}")
        pred_df = data_util.load_merge_predictions(os.path.join(step_dir, "model_predictions.csv"))
        gt_df = npe_util.load_sites_df(os.path.join(step_dir, "gt_merge_sites.csv"))

        step_output_dir = os.path.join(output_dir, step_name)
        util.mkdir(step_output_dir)
        merge_metrics.prec_recall_curve(gt_df, pred_df, step_output_dir, dt=DT)


def compute_precision_recall(gt_graphs, step_name, only_leaf2leaf=False):
    # Load proposal summary (first round for the given mode)
    csv_paths = util.list_cloud_paths(results_manager.step_dirs[step_name], extension=".csv")
    if only_leaf2leaf:
        csv_paths = [p for p in csv_paths if "leaf2leaf" in p]
    else:
        csv_paths = [p for p in csv_paths if "leaf2leaf" not in p]
    assert len(csv_paths) > 0, f"No CSV files found for {step_name} (only_leaf2leaf={only_leaf2leaf})"

    csv_path_r1 = min(csv_paths, key=get_round_num)
    proposal_df = data_util.load_proposal_df(csv_path_r1, only_leaf2leaf)

    # Relabel graphs
    fragment_graphs = load_merge_corrected_fragments(gt_graphs, step_name)

    # Call evaluator
    results_df = split_metrics.compute_precision_recall(
        gt_graphs, fragment_graphs, labels, proposal_df, dt=DT
    )

    # Save results
    suffix = "_leaf2leaf" if only_leaf2leaf else ""
    step_output_dir = os.path.join(output_dir, step_name)
    util.mkdir(step_output_dir)
    csv_path = os.path.join(step_output_dir, f"precision_recall_f1{suffix}.csv")
    results_df.to_csv(csv_path)

    png_path = os.path.join(step_output_dir, f"precision_recall_f1{suffix}.png")
    split_metrics.save_precision_recall_curve(
        results_df, png_path, title=f"{step_name} Split Correction Performance"
    )


def compute_multiround_precision_recall(gt_graphs, step_name):
    # Load all proposal CSVs for this split step
    csv_paths = util.list_cloud_paths(results_manager.step_dirs[step_name], extension=".csv")
    csv_paths = sorted(csv_paths, key=get_round_num)
    proposal_df_list = data_util.load_multiround_proposal_df(csv_paths)

    # Relabel graphs
    fragment_graphs = load_merge_corrected_fragments(gt_graphs, step_name)

    # Call evaluator
    results_df = split_metrics.compute_multiround_precision_recall(
        gt_graphs, fragment_graphs, labels, proposal_df_list,
    )

    # Save results
    step_output_dir = os.path.join(output_dir, step_name)
    util.mkdir(step_output_dir)
    csv_path = os.path.join(step_output_dir, "multiround_precision_recall_f1.csv")
    results_df.to_csv(csv_path)

    plot_path = os.path.join(step_output_dir, "multiround_precision_recall_f1.png")
    split_metrics.save_precision_recall_curve(
        results_df,
        plot_path,
        show_midpoint=True,
        title=f"{step_name} Multi-Round Performance",
    )


# --- Data Loading ---
def get_round_num(path):
    """Parses the round number out of a proposal_summary CSV filename."""
    return int(os.path.basename(path).split("round=")[1].split("_")[0])


def build_label_handler_upto(step_name):
    """
    Builds a LabelHandler from the connections.txt of every step up to and
    including step_name, so absorbed segment IDs are remapped consistently.
    """
    paths = list()
    for name in results_manager.eval_steps:
        step_dir = results_manager.step_dirs.get(name)
        if step_dir:
            paths.append(os.path.join(step_dir, "connections.txt"))
        if name == step_name:
            break
    return data_util.build_label_handler(paths)


def load_fragments(gt_graphs, swcs_path, label_handler=None):
    fragment_graphs = data_util.load_fragments(swcs_path, use_anisotropy=True)
    assert len(fragment_graphs) > 0, f"No fragment SWCs were loaded from {swcs_path}"
    data_util.relabel_fragments_with_name(fragment_graphs)
    data_util.relabel_groundtruth_wrt_fragments(
        gt_graphs, fragment_graphs, label_handler
    )
    return fragment_graphs


def load_merge_corrected_fragments(gt_graphs, split_step_name):
    """Loads the merge-corrected SWCs that feed into the given split step."""
    split_idx = results_manager.eval_steps.index(split_step_name)
    pre_step = results_manager.eval_steps[split_idx - 1]
    return load_fragments(
        gt_graphs,
        results_manager.step_swcs_paths[pre_step],
        build_label_handler_upto(pre_step),
    )


def load_groundtruth(results_manager):
    # Load graphs
    print("\nStep 1: Load Ground Truth")
    seg_path = args.segmentation_path.rstrip("/") + "/"  # TensorStoreImage requires trailing slash
    swcs_path = f"gs://allen-nd-goog/ground_truth_tracings/{brain_id}/world/"
    gt_graphs = data_util.load_groundtruth(seg_path, swcs_path)

    # Drop filtered labels
    labels = data_util.load_labels(results_manager.get_labels_path())
    data_util.drop_filtered_labels(gt_graphs, labels)

    return gt_graphs, labels


def load_ignored_merges(step_name):
    """
    Downloads "ignored_merges.swc" from {result_prefix}/evaluation/{step_name}/
    into the step's output directory, parses it, and returns a KDTree of the
    merge sites to ignore. The SWC has one node per site, with world x, y, z
    in columns 3-5. Returns None if the step has no such file or no sites.
    """
    bucket_name, prefix = np_util.parse_cloud_path(s3_prefix)
    key = f"{prefix.rstrip('/')}/evaluation/{step_name}/ignored_merges.swc"
    step_output_dir = os.path.join(output_dir, step_name)
    util.mkdir(step_output_dir)
    swc_path = os.path.join(step_output_dir, "ignored_merges.swc")

    try:
        boto3.client("s3").download_file(bucket_name, key, swc_path)
    except ClientError as e:
        if e.response["Error"]["Code"] in ("404", "NoSuchKey"):
            print(f"No ignored merges file found at s3://{bucket_name}/{key}")
            return None
        raise

    try:
        sites = pd.read_csv(swc_path, sep=r"\s+", comment="#", header=None, usecols=[2, 3, 4])
    except pd.errors.EmptyDataError:
        return None
    print(f"Ignored merges for {step_name}: {len(sites)} sites from s3://{bucket_name}/{key}")
    return KDTree(sites.to_numpy(dtype=float))


def parse_result_prefix(result_prefix: str) -> tuple[str, str, str]:
    parts = result_prefix.rstrip("/").split("/")
    brain_id = parts[-3]
    segmentation_id = parts[-2]
    result_id = parts[-1]
    return brain_id, segmentation_id, result_id


def create_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Automated Neuron Proofreading: Evaluation"
    )
    parser.add_argument(
        "--result_prefix",
        type=str,
        required=True,
        help="S3 prefix of the proofreading result to evaluate "
             "(e.g. s3://bucket/.../brain_id/segmentation_id/result_timestamp/)",
    )
    parser.add_argument(
        "--segmentation_path",
        type=str,
        required=True,
        help="Path to the segmentation that was proofread, used to load the "
             "ground truth (e.g. gs://... or a local path)",
    )
    parser.add_argument(
        "--upload",
        type=int,
        choices=[0, 1],
        default=1,
        help="Upload results to {result_prefix}/evaluation/ on S3 after evaluation (1=yes, 0=no).",
    )
    return parser


if __name__ == "__main__":
    # Create argparser
    parser = create_argparser()
    args, _ = parser.parse_known_args()

    # Parse result_prefix
    s3_prefix = args.result_prefix
    brain_id, segmentation_id, result_id = parse_result_prefix(s3_prefix)

    analyze_merges = False
    analyze_splits = False

    # Steps to evaluate, named by the leading part of their directory under
    # result_prefix (e.g. "step1" for "step1_learned_merge_correction").
    # None evaluates every step found there.
    steps = None

    # Paths
    output_dir = f"/results/{brain_id}_{segmentation_id}_{result_id}"
    util.mkdir(output_dir)

    # Run
    main()

    # Upload results to S3
    if args.upload:
        bucket_name, s3_key = np_util.parse_cloud_path(s3_prefix)
        upload_prefix = os.path.join(s3_key, "evaluation")
        np_util.upload_dir_to_s3(output_dir, bucket_name, upload_prefix)
        print(f"Upload complete: s3://{bucket_name}/{upload_prefix}")
