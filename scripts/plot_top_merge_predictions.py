"""Render ranked merge-junction predictions as image/GT/fragments by XY/XZ/YZ.

Run on a compute node in the panda environment with a matching trusted _add.pkl.
Reads existing CSV scores, not joblib; no training or feature extraction is run.
Default ranking uses finite winner OOF scores, never merge labels, matching
the detector's score-by-class and workload figures. Use --score-column
merge_site_probability_oof_selector for model-selection diagnostics.
Scan that ranking until top-k candidates with at least one GT skeleton node in
the plotted volume are found. This GT-visible subset is for visual inspection,
not unbiased performance evaluation. Scores are not calibrated probabilities.
In-sample scores are for browsing only.
Segment GT merge labels and junction GT site labels are displayed separately;
a missing site label is not a confirmed non-merge.
Each view is centered on the candidate junction, not a GT site or a proposed cut.
This script supports merge_junction_detector CSVs, not legacy segment-level CSVs.
Skeleton edges crossing the displayed volume are clipped to its boundary.

From the exa-spim-agent root (default merge run):
    python scripts/plot_top_merge_predictions.py --top-k 10 \
        --pkl cache/dataset_cache_794495_mcl100_add.pkl

Outputs: <result_dir>/top_merge_predictions/rank_*.png and top_predictions.csv.
Images come from the matching cache's img_path; cloud access must be available.
"""

import argparse
import colorsys
import csv
import json
import math
import os
import pickle
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure

try:
    from scripts.skeleton_plot_util import clipped_edges_in_patch
except ModuleNotFoundError:
    from skeleton_plot_util import clipped_edges_in_patch


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULT = ROOT / "autodiscovery-application/merge-error-794495-mcl100_2026-08-04"
DEFAULT_SCORE = "merge_site_probability_oof"
VIEWS = (("XY", 0, (0, 1)), ("XZ", 1, (0, 2)), ("YZ", 2, (1, 2)))
SEGMENT_COLOR = "#00e5ff"
JUNCTION_COLOR = "#ffca28"


def rank_predictions(csv_path, top_k, score_column=DEFAULT_SCORE, min_score=None):
    if top_k is not None and top_k < 1:
        raise ValueError("top_k must be positive")
    if min_score is not None and not math.isfinite(min_score):
        raise ValueError("min_score must be finite")
    ranked, seen = [], set()
    with Path(csv_path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"candidate_id", "node_id", "segment_id", "degree",
                    "x_um", "y_um", "z_um", "is_merge_site", score_column}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Merge-junction CSV must contain {sorted(required)}; "
                             "legacy segment-level merge CSVs are not supported")
        for row in reader:
            candidate_id = int(row["candidate_id"])
            if candidate_id in seen:
                raise ValueError(f"Duplicate candidate_id: {candidate_id}")
            seen.add(candidate_id)
            raw_score = row[score_column].strip()
            score = float(raw_score) if raw_score else float("nan")
            if not math.isfinite(score) or (min_score is not None and score < min_score):
                continue
            prediction = {key: int(row[key]) for key in
                          ("candidate_id", "node_id", "segment_id", "degree", "is_merge_site")}
            prediction.update({key: float(row[key]) for key in ("x_um", "y_um", "z_um")})
            if not all(math.isfinite(prediction[key]) for key in ("x_um", "y_um", "z_um")):
                raise ValueError(f"Candidate {candidate_id}: invalid coordinates")
            if prediction["node_id"] < 0 or prediction["degree"] < 3:
                raise ValueError(f"Candidate {candidate_id}: invalid junction node or degree")
            if prediction["is_merge_site"] not in (0, 1):
                raise ValueError(f"Candidate {candidate_id}: invalid is_merge_site")
            raw_distance = row.get("distance_to_nearest_gt_site_um", "").strip()
            raw_ring = row.get("in_ambiguous_ring", "").strip()
            ring = int(raw_ring) if raw_ring else None
            if ring not in (None, 0, 1):
                raise ValueError(f"Candidate {candidate_id}: invalid in_ambiguous_ring")
            prediction.update(
                score=score, in_ambiguous_ring=ring,
                distance_to_nearest_gt_site_um=float(raw_distance) if raw_distance else float("nan"),
            )
            ranked.append(prediction)
    ranked.sort(key=lambda row: (-row["score"], row["candidate_id"]))
    return ranked[:top_k]


def covered_segment_ids(payload):
    labels = payload.get("gt_node_canonical_label")
    if labels is None:
        return None
    return frozenset(int(value) for value in np.unique(np.asarray(labels)) if value != 0)


def coverage_status(prediction, covered_segments):
    if covered_segments is None:
        return None, "Segment GT coverage: unknown"
    covered = prediction["segment_id"] in covered_segments
    return covered, ("Segment touched by GT" if covered else "Segment not touched by GT")


def candidate_center(graph, prediction):
    node = prediction["node_id"]
    if node < 0 or node >= len(graph.node_xyz):
        raise ValueError(f"Candidate {prediction['candidate_id']}: invalid node {node}; wrong --pkl?")
    segment = int(graph.node_segment_id(node))
    if segment != prediction["segment_id"]:
        raise ValueError(f"Node {node} belongs to segment {segment}, CSV says "
                         f"{prediction['segment_id']}; wrong --pkl?")
    xyz = np.asarray(graph.node_xyz[node], dtype=float)
    expected = [prediction[key] for key in ("x_um", "y_um", "z_um")]
    if xyz.shape != (3,) or not np.all(np.isfinite(xyz)):
        raise ValueError(f"Invalid coordinates for node {node}")
    if not np.allclose(xyz, expected, rtol=0, atol=1e-3):
        raise ValueError(f"Node {node} coordinates disagree with CSV; wrong --pkl?")
    if int(graph.degree(node)) != prediction["degree"]:
        raise ValueError(f"Node {node} degree disagrees with CSV; wrong --pkl?")
    return xyz


def patch_bounds(image, center_xyz, anisotropy, patch_um):
    anisotropy = np.asarray(anisotropy, dtype=float)
    patch_um = np.asarray(patch_um, dtype=float)
    if anisotropy.shape != (3,) or not np.all(np.isfinite(anisotropy) & (anisotropy > 0)):
        raise ValueError("anisotropy must contain three positive finite XYZ values")
    if patch_um.shape != (3,) or not np.all(np.isfinite(patch_um) & (patch_um > 0)):
        raise ValueError("patch_um must contain three positive finite XYZ values")
    center = np.floor(np.asarray(center_xyz) / anisotropy).astype(int)[::-1]
    requested = np.maximum(1, np.ceil(patch_um / anisotropy).astype(int))[::-1]
    volume_shape = np.asarray(image.shape()[-3:], dtype=int)
    if np.any(center < 0) or np.any(center >= volume_shape):
        raise ValueError(f"Junction {center_xyz} is outside the image volume")
    origin = np.maximum(0, center - requested // 2)
    stop = np.minimum(volume_shape, center - requested // 2 + requested)
    shape = stop - origin
    return origin, shape


def select_gt_visible_predictions(payload, image, ranked, top_k, patch_um):
    selected = []
    for score_rank, prediction in enumerate(ranked, 1):
        center = candidate_center(payload["fragments_graph"], prediction)
        origin, shape = patch_bounds(image, center, payload["anisotropy"], patch_um)
        nodes, components = payload["gt_graph"].nodes_in_patch(
            origin, shape, return_components=True)
        if not len(nodes):
            continue
        selected.append({
            **prediction, "score_rank": score_rank,
            "gt_nodes_in_patch": len(nodes),
            "gt_components_in_patch": len(set(map(int, components))),
        })
        if len(selected) == top_k:
            break
    return selected


def read_patch(image, center_xyz, anisotropy, patch_um):
    origin, shape = patch_bounds(image, center_xyz, anisotropy, patch_um)
    read_center = origin + shape // 2
    patch = np.asarray(image.read(tuple(map(int, read_center)), tuple(map(int, shape))))
    if patch.shape != tuple(shape):
        raise ValueError(f"Image reader returned {patch.shape}, expected {tuple(shape)}")
    return patch, origin


def skeleton_overlay(axes, graph, origin, shape, anisotropy, highlight_segment=None):
    edges, edge_components = clipped_edges_in_patch(graph, origin, shape, anisotropy)
    nodes, node_components = graph.nodes_in_patch(origin, shape, return_components=True)
    components = sorted(set(map(int, edge_components)) | set(map(int, node_components)))
    for axis, (_, _, xyz_axes) in zip(axes, VIEWS):
        voxel_axes = [2 - dimension for dimension in xyz_axes]
        scale = np.asarray(anisotropy)[list(xyz_axes)]
        for index, component in enumerate(components):
            segment = str(graph.component_id_to_swc_id[component]).split(".")[0]
            selected = highlight_segment is not None and segment == str(highlight_segment)
            color = SEGMENT_COLOR if selected else colorsys.hsv_to_rgb(
                (index / max(len(components), 1) + 0.05) % 1.0, 0.9, 1.0)
            component_edges = edges[edge_components == component]
            component_nodes = nodes[node_components == component]
            if len(component_edges):
                axis.add_collection(LineCollection(
                    component_edges[:, :, voxel_axes] * scale, colors=[color],
                    linewidths=3.0 if selected else 2.0, alpha=0.95,
                    zorder=4 if selected else 3,
                ))
            if len(component_nodes):
                projected = component_nodes[:, voxel_axes] * scale
                axis.scatter(projected[:, 0], projected[:, 1], s=12 if selected else 8,
                             color=color, alpha=0.95, zorder=4 if selected else 3)
        if not components:
            axis.text(0.5, 0.05, "No skeleton in this volume", transform=axis.transAxes,
                      ha="center", color="white", bbox={"facecolor": "black", "alpha": 0.6})


def render_context(payload, image, center_xyz, segment_id, patch_um):
    """Draw image/GT/fragments at an XYZ location, independent of candidate scores."""
    fragment = payload["fragments_graph"]
    center_xyz = np.asarray(center_xyz, dtype=float)
    if center_xyz.shape != (3,) or not np.isfinite(center_xyz).all():
        raise ValueError("Center must contain three finite XYZ coordinates")
    anisotropy = np.asarray(payload["anisotropy"], dtype=float)
    patch, origin = read_patch(image, center_xyz, anisotropy, patch_um)
    if not np.isfinite(patch).all():
        raise ValueError("Image patch contains non-finite intensities")
    low, high = float(np.min(patch)), float(np.percentile(patch, 99.9))
    if high <= low:
        high = low + 1
    figure = Figure(figsize=(15, 14), layout="constrained")
    figure.get_layout_engine().set(h_pad=0.15)
    FigureCanvasAgg(figure)
    axes = figure.subplots(3, 3, sharex="col", sharey="col")
    row_names = ("Image (MIP)", "GT", "Fragments")
    anchor_xyz = center_xyz - origin[::-1] * anisotropy
    for column, (view_name, dimension, xyz_axes) in enumerate(VIEWS):
        horizontal, vertical = xyz_axes
        projection = np.max(patch, axis=dimension)
        height, width = projection.shape
        extent = (-0.5 * anisotropy[horizontal], (width - 0.5) * anisotropy[horizontal],
                  (height - 0.5) * anisotropy[vertical], -0.5 * anisotropy[vertical])
        axes[0, column].imshow(projection, cmap="viridis", origin="upper", extent=extent,
                              vmin=low, vmax=high, interpolation="nearest")
        for row, row_name in enumerate(row_names):
            axis = axes[row, column]
            axis.set_facecolor("black")
            axis.scatter(*anchor_xyz[list(xyz_axes)], marker="+", s=100,
                         color=JUNCTION_COLOR, linewidths=1.8, zorder=5)
            axis.set_title(f"{row_name} -- {view_name}", fontsize=14)
            axis.set_aspect("equal")
            axis.set_xlim(extent[:2])
            axis.set_ylim(extent[2:])
            axis.set_xticks([])
            axis.set_yticks([])
    skeleton_overlay(axes[1], payload["gt_graph"], origin, patch.shape, anisotropy)
    skeleton_overlay(axes[2], fragment, origin, patch.shape, anisotropy,
                     highlight_segment=segment_id)
    return figure, origin, patch.shape


def render_prediction(payload, image, prediction, rank, patch_um, score_column,
                      center=None, covered_segments=None):
    center_xyz = (candidate_center(payload["fragments_graph"], prediction)
                  if center is None else np.asarray(center))
    if covered_segments is None:
        covered_segments = covered_segment_ids(payload)
    covered, coverage_label = coverage_status(prediction, covered_segments)
    anisotropy = np.asarray(payload["anisotropy"], dtype=float)
    figure, origin, patch_shape = render_context(
        payload, image, center_xyz, prediction["segment_id"], patch_um)
    if prediction["is_merge_site"]:
        junction_status = "MERGE-SITE LABELLED"
        status_color = "#18733c"
        caveat = "Positive within the detector's GT-site labeling tolerance; not an exact cut location."
    elif prediction.get("in_ambiguous_ring"):
        junction_status = "AMBIGUOUS RING (CSV label=0)"
        status_color = "#92400e"
        caveat = "Outside the positive-label radius but within the GT-site claim radius; audit only."
    else:
        junction_status = "NOT SITE-LABELLED"
        status_color = "#92400e"
        caveat = "Sparse GT: no positive label does not prove this junction is free of a merge."
    merge_labels = payload.get("gt_merge_labels")
    segment_merge = (None if merge_labels is None else
                     bool(np.any(np.asarray(merge_labels) == prediction["segment_id"])))
    segment_status = ("UNKNOWN" if segment_merge is None else
                      "MERGE" if segment_merge else "NOT MERGE-LABELLED")
    if segment_merge and not prediction["is_merge_site"]:
        caveat += "\nKnown merge segment; this junction has no positive site label."
    if covered is False:
        caveat += " Segment has no canonical GT coverage."
    distance = prediction.get("distance_to_nearest_gt_site_um", float("nan"))
    distance_text = f"{distance:.3g} um" if math.isfinite(distance) else "not recorded"
    score_note = "IN-SAMPLE: optimistic, browsing only" if "in_sample" in score_column else "uncalibrated ranking score"
    selection_note = ""
    if "gt_nodes_in_patch" in prediction and "gt_components_in_patch" in prediction:
        selection_note = (
            f"\nGT-visible selection: {prediction['gt_nodes_in_patch']} GT nodes / "
            f"{prediction['gt_components_in_patch']} GT components in view; visual inspection only."
        )
    z_start, z_stop = origin[0] * anisotropy[2], (origin[0] + patch_shape[0]) * anisotropy[2]
    figure.suptitle(f"Segment GT: {segment_status} | Junction GT: {junction_status}\n"
                   f"{coverage_label} | Center: candidate junction node",
                   fontsize=16, fontweight="bold", color=status_color,
                   bbox={"facecolor": "#f8fafc", "edgecolor": status_color, "pad": 8})
    figure.supxlabel(
        f"Plot {rank} | score rank {prediction.get('score_rank', rank)}"
        f" | candidate {prediction['candidate_id']} | node {prediction['node_id']}"
        f" | segment {prediction['segment_id']} | degree {prediction['degree']}"
        f" | score {prediction['score']:.6g}\n"
        f"{score_column} ({score_note})\n"
        f"Center XYZ (um): {center_xyz[0]:.2f}, {center_xyz[1]:.2f}, {center_xyz[2]:.2f}"
        f" | Z slab: [{z_start:.2f}, {z_stop:.2f}) um | GT site distance (audit): {distance_text}\n"
        "+ marks the candidate junction, not a GT site. Cyan highlights its segment in the fragments row.\n"
        f"{caveat}{selection_note}", fontsize=10,
    )
    return figure, {
        "rank": rank, **prediction, "score_column": score_column,
        "gt_covered_segment": "" if covered is None else int(covered),
        "gt_segment_is_merge": "" if segment_merge is None else int(segment_merge),
        "junction_gt_status": junction_status,
        "x_um": center_xyz[0], "y_um": center_xyz[1], "z_um": center_xyz[2],
        "z_start_um": z_start, "z_stop_um": z_stop,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", nargs="?", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--csv", type=Path, help="Explicit junction CSV; requires --pkl")
    parser.add_argument("--pkl", type=Path, help="Matching trusted labeled dataset cache")
    parser.add_argument("--top-k", type=int, default=10,
                        help="Number of plots with at least one GT node in the receptive field")
    parser.add_argument("--score-column", default=DEFAULT_SCORE,
                        help=f"Default: {DEFAULT_SCORE} (winner OOF, matching detector figures); "
                            "selector OOF is available as an explicit diagnostic override")
    parser.add_argument("--min-score", type=float, help="Optional lower score cutoff")
    parser.add_argument("--patch-um", nargs=3, type=float, default=(100, 100, 40),
                        metavar=("X", "Y", "Z"), help="Field of view in micrometers")
    parser.add_argument("--output-dir", type=Path, help="Default: <result_dir>/top_merge_predictions")
    parser.add_argument("--dpi", type=int, default=160)
    args = parser.parse_args(argv)
    if args.top_k < 1 or args.dpi < 1 or not all(math.isfinite(size) and size > 0 for size in args.patch_um):
        parser.error("top-k, dpi and patch dimensions must be positive")
    if args.csv is not None and args.pkl is None:
        parser.error("Specify --pkl with --csv to avoid selecting the wrong brain's cache")
    csv_path = args.csv
    if csv_path is None:
        candidates = sorted(args.result_dir.glob("merge_junction_detector_*.csv"))
        if len(candidates) != 1:
            parser.error("Expected exactly one junction CSV; specify --csv and --pkl")
        csv_path = candidates[0]
    ranked = rank_predictions(csv_path, None, args.score_column, args.min_score)
    if not ranked:
        parser.error("No finite scores pass the requested cutoff")
    if "in_sample" in args.score_column:
        print("WARNING: in-sample scores are optimistic; use for browsing only", flush=True)
    pkl_path = args.pkl
    if pkl_path is None:
        selections = sorted(args.result_dir.glob("model_selection_*.json"))
        if len(selections) != 1:
            parser.error("Cannot infer dataset cache; specify --pkl")
        metadata = json.loads(selections[0].read_text())
        pkl_path = ROOT / "cache" / Path(metadata["train_pkl"]).name
        if not pkl_path.exists():
            pkl_path = Path(metadata["train_pkl"])
    output_dir = args.output_dir or args.result_dir / "top_merge_predictions"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}\nLoading trusted cache: {pkl_path}", flush=True)
    with pkl_path.open("rb") as handle:
        payload = pickle.load(handle)
    covered_segments = covered_segment_ids(payload)
    from agentic_neuron_proofreader.utils.img_util import TensorStoreImage

    os.environ.setdefault("AWS_EC2_METADATA_DISABLED", "true")
    image = TensorStoreImage(payload["img_path"])
    print(f"Image source: {payload['img_path']}", flush=True)
    print(f"Scanning {len(ranked)} ranked candidates for local GT coverage ...", flush=True)
    predictions = select_gt_visible_predictions(payload, image, ranked, args.top_k, args.patch_um)
    if not predictions:
        parser.error("No ranked candidate has a GT skeleton node in the receptive field")
    if len(predictions) < args.top_k:
        print(f"WARNING: only {len(predictions)} GT-visible candidates available (requested {args.top_k})",
              flush=True)
    print(f"Selected {len(predictions)} GT-visible candidates through score rank "
          f"{predictions[-1]['score_rank']} (visual inspection only)", flush=True)
    manifest = []
    for rank, prediction in enumerate(predictions, 1):
        figure, record = render_prediction(
            payload, image, prediction, rank, args.patch_um, args.score_column,
            covered_segments=covered_segments,
        )
        filename = f"rank_{rank:03d}_candidate_{prediction['candidate_id']}.png"
        figure.savefig(output_dir / filename, dpi=args.dpi, bbox_inches="tight", pad_inches=0.15)
        figure.clear()
        manifest.append({**record, "figure": filename})
        print(f"[{rank}/{len(predictions)}] {output_dir / filename}", flush=True)
    with (output_dir / "top_predictions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    print(f"Saved {len(manifest)} figures and top_predictions.csv", flush=True)


if __name__ == "__main__":
    main()