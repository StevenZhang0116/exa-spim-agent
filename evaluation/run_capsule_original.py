"""Run the CodeOcean capsule's evaluation on this repository's own inputs for one brain.

The capsule (external/codeocean/capsule_6960813, ``code/main.py``) computes skeleton metrics
for every step of an Arborist proofreading result that it reads from S3. No such result exists
for our data. This script runs the capsule's ``main()`` unmodified on what we do have, the
ground truth, the fragment SWCs and the segmentation of a brain, which is the capsule's
"original" step: the segmentation before any correction. The corrected steps need the output
of an Arborist run and are not covered.

Three inputs of ``main()`` are replaced:

* the result directory: one step, "original", whose SWCs are our fragment SWCs on GCS;
* the list of segments the pipeline kept (``segment_ids.txt``): none, every segment stays;
* the reviewed false-positive merge sites (``ignored_merges.swc``): none.

Two adaptations of the fragment SWCs:

* Coordinates. The capsule expects world coordinates in um, as the Arborist pipeline writes
  them. The fragment SWCs of the Google segmentation hold voxel indices. They are converted to
  um with the capsule's anisotropy before its loader sees them. Without the conversion no
  ground-truth node finds its fragment and every edge counts as omitted.
* Selection. Fragment SWCs are parsed only for the segment IDs that label a ground-truth node.
  The capsule relabels ground truth against the fragment of the node's own segment and counts
  merges on those fragments, so other fragments cannot enter a result. This follows from the
  code and was not checked against an unrestricted run. Every archive is still downloaded.

A failed archive download is retried and then stops the run; the package itself would drop the
archive and go on.

``--fragments`` evaluates another set of fragment SWCs against the same ground truth and
segmentation, for example the fragments of this repository's labeled cache written by
``export_cache_fragments.py``, or an edited graph (``cut_merge_sites.py``,
``join_split_candidates.py``). Such a file holds world um
(``--fragment-units um``, the default with ``--fragments``) and is read by the capsule's loader
as it is. ``<out>/segments_under_gt.txt`` lists the segment IDs that label a ground-truth node;
the exporter takes it as its selection. When fragments of different segments were joined
(``join_split_candidates.py``), ``--connections`` names the joined pairs, so that a
ground-truth node of either segment is given to the joined fragment; the capsule's own label
handler does this, as it does for the split-correction steps of a pipeline result.

Run on a compute node in the capsule's environment (never in panda), from the repository root:

    conda activate /allen/programs/mindscope/workgroups/auto-model/zihan.zhang/exaspim-agent/conda_envs/arborist_eval
    python -u evaluation/run_capsule_original.py --brain 794495 \\
        --out evaluation/results/794495_capsule_original

Outputs, in ``<out>/original/``: ``original_results.csv`` (one row per ground-truth neuron),
``original_results_overview.txt``, ``original_merge_sites.csv``,
``original_fragments_with_merges.zip``, ``original_fragments/`` and the omit-length files;
``<out>/run_info.json`` records the inputs, the installed package commits and the counts.
"""
import argparse
import importlib.metadata
import json
import os
import socket
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CAPSULE_CODE = REPO / "external/codeocean/capsule_6960813/code"
CREDENTIALS = REPO / "configs/allen-nd-goog-f5d46dbfa2cd.json"
PACKAGES = ("segmentation_skeleton_metrics", "neuron_proofreading_evaluation", "arborist",
            "neuron-proofreader", "aind-exaspim-dataset-utils")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--brain", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true", help="Replace the results of an earlier run in --out")
    parser.add_argument("--fragments", default=None,
                        help="Fragment SWCs to evaluate: a local .zip or a cloud path (default: the brain's "
                             "fragment SWCs on GCS)")
    parser.add_argument("--fragment-units", choices=("voxel", "um"), default=None,
                        help="Coordinates of the fragment SWCs (default: voxel for the brain's own SWCs, "
                             "um with --fragments)")
    parser.add_argument("--connections", type=Path, default=None,
                        help="With --fragments: the segment pairs that the fragments join, one 'a.k, b.k' per "
                             "line as in a pipeline step's connections.txt")
    parser.add_argument("--gt-names", nargs="*", default=None,
                        help="Test only: read just these ground-truth SWC names (without .swc)")
    parser.add_argument("--zip-retries", type=int, default=4)
    return parser.parse_args()


def installed_commits():
    """Version and source commit of the capsule's first-party packages in this environment."""
    commits = {}
    for name in PACKAGES:
        distribution = importlib.metadata.distribution(name)
        source = json.loads(distribution.read_text("direct_url.json") or "{}")
        commits[name] = {"version": distribution.version, "commit": source.get("vcs_info", {}).get("commit_id")}
    return commits


def main():
    options = parse_args()
    options.out = options.out.resolve()
    results = options.out / "original" / "original_results.csv"
    if results.exists() and not options.overwrite:
        sys.exit(f"{results} exists; pass --overwrite or another --out")
    os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS", str(CREDENTIALS))
    sys.dont_write_bytecode = True          # nothing is written into the imported capsule snapshot
    sys.path.insert(0, str(CAPSULE_CODE))
    sys.path.insert(0, str(REPO / "scripts"))

    from dataset_config import get_fragments_path, get_segmentation_id, get_segmentation_path
    rtf = str(REPO / "configs/segmentation_datasets.rtf")
    segmentation_id = get_segmentation_id(options.brain, rtf_path=rtf)
    fragments_path = options.fragments or get_fragments_path(options.brain, rtf_path=rtf)
    units = options.fragment_units or ("um" if options.fragments else "voxel")
    if options.fragments and not (fragments_path.startswith(("gs://", "s3://")) or fragments_path.endswith(".zip")):
        sys.exit("--fragments must be a .zip file or a cloud path")
    if options.connections and not (options.fragments and options.connections.is_file()):
        sys.exit("--connections needs --fragments and an existing file")
    segmentation_path = get_segmentation_path(options.brain, rtf_path=rtf)
    gt_path = f"gs://allen-nd-goog/ground_truth_tracings/{options.brain}/world/"   # as main.py builds it

    import main as capsule                  # the capsule's file: its imports and its merge-site filter
    import numpy as np
    from segmentation_skeleton_metrics.datamodules import graph_loading, swc_loading
    from segmentation_skeleton_metrics.utils import util

    class OwnInputs:
        """What ResultsManager provides, for a result that has the original SWCs and nothing else."""
        def __init__(self, *_):
            self.s3_prefix = fragments_path.rstrip("/") + "/"
            self.brain_id, self.segmentation_id = options.brain, segmentation_id
            self.eval_steps, self.step_types, self.step_dirs = ["original"], {"original": "original"}, {}
            self.step_swcs_paths = {"original": fragments_path}

        def get_labels_path(self):
            return None

    capsule.ResultsManager = OwnInputs
    if options.connections:
        # A fragment that joins two segments stands for both: the capsule's own handler, built from
        # our file instead of the connections.txt of the pipeline steps.
        capsule.build_label_handler_upto = lambda step_name: capsule.data_util.build_label_handler(
            [str(options.connections)])
    capsule.load_ignored_merges = lambda step_name: None
    capsule.data_util.load_labels = lambda path: None
    capsule.data_util.drop_filtered_labels = lambda graphs, labels: None

    # Reader filters: ground truth by name (test only); fragments by the segment IDs at GT nodes,
    # applied where an archive member is read (the package parses every member).
    keep_segments, state, counts = set(), {"phase": "groundtruth"}, {}
    original_confirm = swc_loading.Reader.confirm_read

    def confirm_read(self, path):
        name = os.path.splitext(os.path.basename(path))[0]
        if state["phase"] == "groundtruth" and options.gt_names is not None and name not in options.gt_names:
            return False
        return original_confirm(self, path)
    swc_loading.Reader.confirm_read = confirm_read

    original_member = swc_loading.Reader.read_zipped_swc

    def read_zipped_swc(self, zipfile, path):
        if state["phase"] == "fragments" and util.get_segment_id(os.path.basename(path)) not in keep_segments:
            return None
        return original_member(self, zipfile, path)
    swc_loading.Reader.read_zipped_swc = read_zipped_swc

    # Fragment SWCs hold voxel indices (x, y, z); the capsule's loader divides world um by the
    # anisotropy. Hand it um. The quarter voxel keeps the loader's truncation on the same index.
    original_to_graph = graph_loading.GraphLoader.to_graph

    def to_graph(self, swc_dict):
        if not self.is_groundtruth and units == "voxel":
            swc_dict = dict(swc_dict)
            swc_dict["voxel"] = (np.asarray(swc_dict["voxel"], dtype=np.float64) + 0.25) * self.anisotropy
        return original_to_graph(self, swc_dict)
    graph_loading.GraphLoader.to_graph = to_graph

    def read_gcs_zip(self, path):
        from google.cloud import storage
        bucket_name, key = util.parse_cloud_path(path)
        for attempt in range(options.zip_retries + 1):
            try:
                content = storage.Client().bucket(bucket_name).blob(key).download_as_bytes()
                return self._parse_zip_bytes(content)
            except Exception as exc:        # any transport failure is retried
                error = exc
                time.sleep(2 * (attempt + 1))
        raise RuntimeError(f"ZIP not readable after {options.zip_retries + 1} attempts: {path}: {error!r}")
    swc_loading.Reader.read_gcs_zip = read_gcs_zip

    load_groundtruth = capsule.load_groundtruth

    def load_groundtruth_and_note_segments(results_manager):
        started = time.time()
        gt_graphs, labels = load_groundtruth(results_manager)
        for graph in gt_graphs.values():
            keep_segments.update(str(label) for label in graph.node_labels())
        keep_segments.discard("0")
        state["phase"] = "fragments"
        counts.update(gt_neurons=len(gt_graphs), gt_nodes=sum(g.number_of_nodes() for g in gt_graphs.values()),
                      segments_labeling_gt_nodes=len(keep_segments))
        (options.out / "segments_under_gt.txt").write_text("\n".join(sorted(keep_segments, key=int)) + "\n")
        print(f"\nGround truth: {counts['gt_neurons']} neurons, {counts['gt_nodes']} nodes, labeled by "
              f"{len(keep_segments)} segments ({time.time() - started:.0f} s); fragment SWCs are read for "
              "these segments only", flush=True)
        return gt_graphs, labels
    capsule.load_groundtruth = load_groundtruth_and_note_segments

    load_fragments = capsule.data_util.load_fragments

    def load_fragments_and_count(*args, **kwargs):
        graphs = load_fragments(*args, **kwargs)
        counts["fragment_graphs_loaded"] = len(graphs)
        counts["segments_with_a_fragment_swc"] = len({str(name).split(".")[0] for name in graphs})
        print(f"\nFragments: {len(graphs)} SWCs of {counts['segments_with_a_fragment_swc']} segments "
              f"(of the {len(keep_segments)} that label ground-truth nodes)", flush=True)
        return graphs
    capsule.data_util.load_fragments = load_fragments_and_count

    # The globals main.py sets under __main__.
    options.out.mkdir(parents=True, exist_ok=True)
    capsule.args = argparse.Namespace(result_prefix=fragments_path, segmentation_path=segmentation_path, upload=0)
    capsule.s3_prefix = fragments_path
    capsule.brain_id, capsule.segmentation_id, capsule.result_id = options.brain, segmentation_id, "own_inputs"
    capsule.analyze_merges = capsule.analyze_splits = False
    capsule.steps = None
    capsule.output_dir = str(options.out)

    print(f"GT:           {gt_path}\nFragments:    {fragments_path} ({units})\nSegmentation: {segmentation_path}\n"
          f"Output:       {options.out}", flush=True)
    started = time.time()
    capsule.main()
    seconds = time.time() - started

    info = {
        "date": datetime.now().astimezone().isoformat(timespec="seconds"), "host": socket.gethostname(),
        "brain": options.brain, "segmentation_id": segmentation_id, "step": "original",
        "inputs": {"ground_truth": gt_path, "fragments": fragments_path, "fragment_units": units,
                   "connections": str(options.connections) if options.connections else None,
                   "segmentation": segmentation_path},
        "capsule": {"main_py": str(CAPSULE_CODE / "main.py"), "packages": installed_commits()},
        "replaced_inputs": ["result directory: fragment SWCs on GCS as the only step (original)",
                            "segment_ids.txt: none, no segment dropped",
                            "ignored_merges.swc: none, no merge site ignored"],
        "adaptations": (["fragment SWC voxel indices converted to um before the capsule's loader"]
                        if units == "voxel" else [])
                       + ["fragment SWCs parsed only for segment IDs that label a ground-truth node"],
        "gt_names": options.gt_names, "counts": counts, "seconds": round(seconds),
    }
    if options.connections:
        info["replaced_inputs"].append("connections.txt of the pipeline steps: the file given with --connections")
    (options.out / "run_info.json").write_text(json.dumps(info, indent=2))
    print(f"\nDONE in {seconds:.0f} s; outputs in {options.out}/original", flush=True)


if __name__ == "__main__":
    main()
