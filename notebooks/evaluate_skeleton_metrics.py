"""Evaluate a segmentation with the official skeleton metrics; write metrics_out/.

Script form of the former notebooks/evaluate_skeleton_metrics.ipynb (removed in
6995bac). Runs ``segmentation_skeleton_metrics.evaluate.evaluate(...)`` for each
brain and writes, under ``metrics_out/<brain>/<segmentation_id>/``:

    results.csv                per-GT-SWC metrics
    results_overview.txt       averages and totals
    merge_sites.csv            merge locations (--no-save-merges skips these two)
    fragments_with_merges.zip  fragment SWCs that contain a merge

Pipeline inside ``evaluate``:
1. Label graphs -- read GT SWCs and label each node with the segment ID it
   falls inside (reads the segmentation volume).
2. Detect errors -- per edge, classify as correct / split / omit / merge by
   comparing endpoint labels.
3. Compute metrics -- # Splits, # Merges, % Split/Omit/Merged Edges, ERL,
   Normalized ERL, Edge Accuracy, Split Rate, Merge Rate.

This re-reads the SWCs from GCS because the metrics package builds its own
labeled-graph representation; dataset_cache_*.pkl does not help here. Paths come
from scripts/dataset_config.py, so brains in configs/segmentation_paths.json
(different GCS layout) work too. Brains with many fragments need a compute node
with plenty of RAM.

With several brains, each runs sequentially in its own subprocess, so memory is
released between brains and a failure does not stop the rest (--stop-on-error
changes that). A summary is printed at the end, each brain's console output is
also saved to notebooks/log/evaluate_skeleton_metrics_<brain>_<timestamp>.log,
and the exit code is non-zero if any brain failed.

Examples (from the repository root, panda env, compute node):
    python -u notebooks/evaluate_skeleton_metrics.py --brains 754613
    python -u notebooks/evaluate_skeleton_metrics.py --brains 754613 754612 754610 747807 750318 751473
    python -u notebooks/evaluate_skeleton_metrics.py --brains 709221 --output-root /tmp/metrics
"""

import argparse
from datetime import datetime
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--brains", nargs="+", required=True)
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "metrics_out",
                        help="Results go to <output-root>/<brain>/<segmentation_id>/")
    # Same (x, y, z) microns/voxel as load_skeletons.py. Not applied to the GT
    # SWCs because gt_path ends in /voxel (already voxel coords).
    parser.add_argument("--anisotropy", type=float, nargs=3, default=(0.748, 0.748, 1.0),
                        metavar=("X", "Y", "Z"))
    parser.add_argument("--no-save-merges", action="store_true",
                        help="Skip merge_sites.csv and fragments_with_merges.zip")
    parser.add_argument("--save-fragments", action="store_true",
                        help="Also save fragments intersecting each GT skeleton")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-run brains whose results.csv already exists")
    parser.add_argument("--zip-workers", type=int, default=None,
                        help="Processes reading fragment zips (default: CPUs allocated to this job)")
    parser.add_argument("--zip-retries", type=int, default=5,
                        help="Attempts per fragment zip download before failing")
    parser.add_argument("--all-fragments", action="store_true",
                        help="Load every fragment SWC (package default). By default only fragments "
                             "whose segment ID labels a GT node are parsed; results are identical "
                             "and memory is far lower")
    parser.add_argument("--stop-on-error", action="store_true",
                        help="With several brains, stop at the first failed brain")
    parser.add_argument("--log-dir", type=Path, default=PROJECT_ROOT / "notebooks" / "log",
                        help="Per-brain console logs when several brains are given")
    args = parser.parse_args()
    args.brains = list(dict.fromkeys(args.brains))
    return args


def harden_fragment_reader(workers, retries):
    """Make the metrics package's cloud zip reader robust for large brains.

    Its read_zips() starts ProcessPoolExecutor() with os.cpu_count() workers (88 on
    n257, regardless of the Slurm allocation) and does not retry downloads. When one
    GCS download gives up, google.api_core's RetryError cannot be unpickled in the
    parent, so the whole run dies as BrokenProcessPool. Here each zip is retried with
    backoff, a final failure is re-raised as a picklable RuntimeError, and the pool is
    capped. Workers are forked, so the patched class is inherited.
    """
    import functools
    import time
    from concurrent.futures import ProcessPoolExecutor
    from segmentation_skeleton_metrics.data_handling import swc_loading

    original = swc_loading.Reader.read_gcs_zip
    if getattr(original, "_hardened", False):
        return

    def read_gcs_zip(self, path):
        last = None
        for attempt in range(retries):
            try:
                return original(self, path)
            except Exception as exc:  # noqa: BLE001 - google/requests raise many types
                last = f"{type(exc).__name__}: {exc}"
                time.sleep(min(60, 2 ** attempt))
        raise RuntimeError(f"Failed to read {path} after {retries} attempts: {last}")

    read_gcs_zip._hardened = True
    swc_loading.Reader.read_gcs_zip = read_gcs_zip
    swc_loading.ProcessPoolExecutor = functools.partial(ProcessPoolExecutor, max_workers=workers)
    print(f"Fragment zip reader: {workers} processes, {retries} attempts per zip", flush=True)


def check_precomputed(path):
    """Read the precomputed `info` explicitly. The metrics package's
    is_precomputed() swallows every exception, so an auth failure would surface
    as a misleading "Invalid image path"."""
    import json
    import tensorstore as ts
    bucket, prefix = path[len("gs://"):].split("/", 1)
    store = ts.KvStore.open({"driver": "gcs", "bucket": bucket, "path": prefix}).result()
    raw = store.read(b"info").result()  # auth/permission errors raise here
    if raw.state == "missing" or not raw.value:
        raise FileNotFoundError(f"No precomputed info at {path}info")
    info = json.loads(raw.value)
    if info.get("type") not in ("image", "segmentation") or "scales" not in info:
        raise ValueError(f"{path}info is not a neuroglancer precomputed volume")


def run_evaluation(gt_path, segmentation, output_dir, fragments_path, args):
    """segmentation_skeleton_metrics.evaluate.evaluate(), optionally loading only
    the fragments that can matter.

    The package parses every fragment SWC (10-12M for the older-microscope brains,
    which exceeds 200 GB), but its merge count only visits fragments whose label is
    one of the GT node labels (MergeCountMetric.__call__), and the added-cable
    metric and merge exports only use fragments found at merge sites. Restricting
    the reader to those segment IDs therefore leaves every output unchanged. Zips
    are still downloaded; non-matching SWCs inside them are skipped unparsed.
    """
    from segmentation_skeleton_metrics.data_handling import swc_loading
    from segmentation_skeleton_metrics.data_handling.graph_loading import DataLoader
    from segmentation_skeleton_metrics.evaluate import Evaluator
    from segmentation_skeleton_metrics.utils import util

    dataloader = DataLoader(anisotropy=tuple(args.anisotropy), use_anisotropy=False, verbose=True)
    gt_graphs = dataloader.load_groundtruth(gt_path, segmentation)
    if args.all_fragments:
        fragment_graphs = dataloader.load_fragments(fragments_path)
    else:
        labels = set().union(*(graph.node_labels() for graph in gt_graphs.values()))
        labels.discard("0")
        print(f"Fragment filter: parsing only SWCs of the {len(labels)} segment IDs that label "
              f"GT nodes (--all-fragments to disable)", flush=True)
        original = swc_loading.Reader.confirm_read

        def confirm_read(self, path):
            name = os.path.splitext(os.path.basename(path))[0]
            return util.get_segment_id(name) in labels and original(self, path)

        swc_loading.Reader.confirm_read = confirm_read  # inherited by forked readers
        try:
            fragment_graphs = dataloader.load_fragments(fragments_path)
        finally:
            swc_loading.Reader.confirm_read = original
        print(f"Loaded {len(fragment_graphs)} fragment graphs", flush=True)

    util.mkdir(str(output_dir))
    evaluator = Evaluator(str(output_dir), "", True)
    evaluator(gt_graphs, fragment_graphs)
    if not args.no_save_merges:
        evaluator.save_merge_results(gt_graphs, fragment_graphs)
    if args.save_fragments and fragment_graphs:
        evaluator.save_fragments(gt_graphs, fragment_graphs)


def evaluate_brain(brain, args):
    from segmentation_skeleton_metrics.utils.img_util import TensorStoreImage
    from dataset_config import get_fragments_path, get_segmentation_id, get_segmentation_path

    rtf = str(CONFIG_DIR / "segmentation_datasets.rtf")
    segmentation_id = get_segmentation_id(brain, rtf_path=rtf)
    gt_path = f"gs://allen-nd-goog/ground_truth_tracings/{brain}/voxel"
    fragments_path = get_fragments_path(brain, rtf_path=rtf)
    segmentation_path = get_segmentation_path(brain, rtf_path=rtf)
    output_dir = args.output_root / brain / segmentation_id

    print(f"\nbrain_id={brain} -> segmentation_id={segmentation_id}", flush=True)
    print(f"GT: {gt_path}\nFragments: {fragments_path}\nSegmentation: {segmentation_path}", flush=True)
    if (output_dir / "results.csv").exists() and not args.overwrite:
        print(f"Skip existing: {output_dir / 'results.csv'} (use --overwrite)", flush=True)
        return
    check_precomputed(segmentation_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # swap_axes=True: SWC voxel coords are stored (x, y, z); after the
    # `from_google` transpose the segmentation volume's last 3 dims are
    # (z, y, x). swap_axes adds another transpose so the image axes line up
    # with the SWC (x, y, z) ordering. Without it you get
    # IndexError: OUT_OF_RANGE on the first GT skeleton labeling pass.
    segmentation = TensorStoreImage(segmentation_path, swap_axes=True)

    run_evaluation(gt_path, segmentation, output_dir, fragments_path, args)

    import pandas as pd
    results = pd.read_csv(output_dir / "results.csv", index_col=0)
    print(f"Per-SWC results ({len(results)} ground-truth skeletons): {output_dir / 'results.csv'}")
    print((output_dir / "results_overview.txt").read_text(), flush=True)


def main():
    args = parse_args()
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    # configs/zihan_gcs_token.json is rejected by Google (invalid_grant: Invalid
    # JWT Signature) as of 2026-10-06; use the project service account.
    os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS",
                          str(CONFIG_DIR / "allen-nd-goog-f5d46dbfa2cd.json"))
    print(f"GCS credentials: {os.environ['GOOGLE_APPLICATION_CREDENTIALS']}", flush=True)
    os.environ.setdefault("AWS_EC2_METADATA_DISABLED", "true")
    if len(args.brains) > 1:
        sys.exit(run_brains_in_subprocesses(args))
    workers = args.zip_workers or len(os.sched_getaffinity(0))
    harden_fragment_reader(workers, args.zip_retries)
    evaluate_brain(args.brains[0], args)


def run_brains_in_subprocesses(args):
    """Run this script once per brain; tee each child's output to a log file."""
    shared = ["--output-root", str(args.output_root), "--anisotropy", *map(str, args.anisotropy),
              "--zip-retries", str(args.zip_retries)]
    for flag, enabled in (("--all-fragments", args.all_fragments),
                          ("--no-save-merges", args.no_save_merges),
                          ("--save-fragments", args.save_fragments),
                          ("--overwrite", args.overwrite)):
        if enabled:
            shared.append(flag)
    if args.zip_workers:
        shared += ["--zip-workers", str(args.zip_workers)]

    args.log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary = []
    for index, brain in enumerate(args.brains, 1):
        log_path = args.log_dir / f"evaluate_skeleton_metrics_{brain}_{stamp}.log"
        print(f"\n===== [{index}/{len(args.brains)}] brain {brain} (log: {log_path}) =====", flush=True)
        command = [sys.executable, "-u", str(Path(__file__).resolve()), "--brains", brain, *shared]
        started = time.monotonic()
        with log_path.open("w", encoding="utf-8") as log:
            child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     text=True, bufsize=1)
            for line in child.stdout:
                sys.stdout.write(line)
                log.write(line)
            code = child.wait()
        minutes = (time.monotonic() - started) / 60
        status = "OK" if code == 0 else f"FAILED (exit {code})"
        summary.append((brain, status, minutes, log_path))
        if code != 0 and args.stop_on_error:
            print(f"Stopping after failed brain {brain} (--stop-on-error)", flush=True)
            break

    print("\n===== Summary =====")
    for brain, status, minutes, log_path in summary:
        print(f"{brain:>8}  {status:<18} {minutes:7.1f} min  {log_path}")
    for brain in args.brains[len(summary):]:
        print(f"{brain:>8}  NOT RUN")
    failed = len(summary) < len(args.brains) or any(s != "OK" for _, s, _, _ in summary)
    return 1 if failed else 0


if __name__ == "__main__":
    main()
