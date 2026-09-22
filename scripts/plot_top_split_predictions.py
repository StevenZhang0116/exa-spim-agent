"""Render ranked split-pair predictions in a 3x3 image/GT/fragment by XY/XZ/YZ grid.

Run in the panda conda environment. Only load trusted pickle caches.
Each split candidate is a segment PAIR with two model-proposed endpoint nodes,
so the viewing center is the gap midpoint between them — a genuinely localized
candidate site, unlike the merge plotter's viewing anchor. GT labels never
determine ranking. The default score column is the nested out-of-fold selector
probability, which covers only the scored subset of the pool; rows without a
finite score are skipped. `split_probability_in_sample` covers every row but is
in-sample (optimistic) — use it only for browsing, not for honest ranking.
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


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULT = ROOT / "autodiscovery-application/split-error-794495-mcl100-run-3_2026-08-24"
DEFAULT_SCORE = "split_probability_oof_selector"
VIEWS = (("XY", 0, (0, 1)), ("XZ", 1, (0, 2)), ("YZ", 2, (1, 2)))
SIDE_A_COLOR = "#00e5ff"
SIDE_B_COLOR = "#ff4dff"


def rank_predictions(csv_path, top_k, score_column=DEFAULT_SCORE, min_score=None):
    if top_k < 1:
        raise ValueError("top_k must be positive")
    if min_score is not None and not math.isfinite(min_score):
        raise ValueError("min_score must be finite")
    ranked = []
    seen = set()
    with Path(csv_path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"candidate_id", "segment_id_a", "segment_id_b",
                    "node_id_a", "node_id_b", "is_split", score_column}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"CSV must contain {sorted(required)}")
        for row in reader:
            candidate_id = int(row["candidate_id"])
            if candidate_id in seen:
                raise ValueError(f"Duplicate candidate_id: {candidate_id}")
            seen.add(candidate_id)
            raw_score = row[score_column].strip()
            score = float(raw_score) if raw_score else float("nan")
            if not math.isfinite(score) or (min_score is not None and score < min_score):
                continue
            raw_gap = row.get("gap_um", "").strip()
            ranked.append({
                "candidate_id": candidate_id,
                "segment_id_a": int(row["segment_id_a"]),
                "segment_id_b": int(row["segment_id_b"]),
                "node_id_a": int(row["node_id_a"]),
                "node_id_b": int(row["node_id_b"]),
                "gap_um": float(raw_gap) if raw_gap else float("nan"),
                "is_split": int(row["is_split"]),
                "score": score,
            })
    ranked.sort(key=lambda row: (-row["score"], row["candidate_id"]))
    return ranked[:top_k]


def covered_segment_ids(payload):
    labels = payload.get("gt_node_canonical_label")
    if labels is None:
        return None
    unique = np.unique(np.asarray(labels))
    return frozenset(int(value) for value in unique if value != 0)


def coverage_status(prediction, covered_segments):
    if covered_segments is None:
        return None, None, "GT coverage: unknown (cache lacks gt_node_canonical_label)"
    covered_a = prediction["segment_id_a"] in covered_segments
    covered_b = prediction["segment_id_b"] in covered_segments
    if covered_a and covered_b:
        label = "GT coverage: both sides traced"
    elif covered_a:
        label = "GT coverage: side A only"
    elif covered_b:
        label = "GT coverage: side B only"
    else:
        label = "GT coverage: neither side traced"
    return covered_a, covered_b, label


def candidate_center(graph, prediction):
    endpoints = {}
    for side in ("a", "b"):
        node = prediction[f"node_id_{side}"]
        segment_id = int(graph.node_segment_id(node))
        expected = prediction[f"segment_id_{side}"]
        if segment_id != expected:
            raise ValueError(
                f"Candidate {prediction['candidate_id']}: node {node} belongs to "
                f"segment {segment_id}, CSV says {expected} — wrong --pkl?")
        xyz = np.asarray(graph.node_xyz[node], dtype=float)
        if xyz.shape != (3,) or not np.all(np.isfinite(xyz)):
            raise ValueError(f"Invalid coordinates for node {node}")
        endpoints[side] = xyz
    return {
        "xyz": (endpoints["a"] + endpoints["b"]) / 2.0,
        "endpoint_a": endpoints["a"],
        "endpoint_b": endpoints["b"],
    }


def read_patch(image, center_xyz, anisotropy, patch_um):
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
        raise ValueError(f"Anchor {center_xyz} is outside the image volume")
    origin = np.maximum(0, center - requested // 2)
    stop = np.minimum(volume_shape, center - requested // 2 + requested)
    shape = stop - origin
    read_center = origin + shape // 2
    patch = np.asarray(image.read(tuple(map(int, read_center)), tuple(map(int, shape))))
    if patch.shape != tuple(shape):
        raise ValueError(f"Image reader returned {patch.shape}, expected {tuple(shape)}")
    return patch, origin


def skeleton_overlay(axes, graph, origin, shape, anisotropy, highlight=None):
    highlight = highlight or {}
    edges, edge_components = graph.edges_in_patch(origin, shape, return_components=True)
    nodes, node_components = graph.nodes_in_patch(origin, shape, return_components=True)
    components = sorted(set(map(int, edge_components)) | set(map(int, node_components)))
    names = {
        component: str(graph.component_id_to_swc_id[component]).split(".")[0]
        for component in components
    }
    component_colors = {
        component: colorsys.hsv_to_rgb((index / max(len(components), 1) + 0.05) % 1.0, 0.9, 1.0)
        for index, component in enumerate(components)
    }
    for axis, (_, _, xyz_axes) in zip(axes, VIEWS):
        voxel_axes = [2 - dimension for dimension in xyz_axes]
        scale = np.asarray(anisotropy)[list(xyz_axes)]
        for component in components:
            selected_color = highlight.get(names[component])
            selected = selected_color is not None
            color = selected_color if selected else component_colors[component]
            component_edges = edges[edge_components == component]
            component_nodes = nodes[node_components == component]
            if len(component_edges):
                projected_edges = component_edges[:, :, voxel_axes] * scale
                axis.add_collection(LineCollection(
                    projected_edges, colors=[color], linewidths=3.0 if selected else 2.0,
                    alpha=0.95,
                    zorder=4 if selected else 3,
                ))
            if len(component_nodes):
                projected_nodes = component_nodes[:, voxel_axes] * scale
                axis.scatter(projected_nodes[:, 0], projected_nodes[:, 1], s=12 if selected else 8, color=color,
                             alpha=0.95,
                             zorder=4 if selected else 3)
        if not components:
            axis.text(0.5, 0.05, "No skeleton in this volume", transform=axis.transAxes,
                      ha="center", color="white", bbox={"facecolor": "black", "alpha": 0.6})


def render_prediction(payload, image, prediction, rank, patch_um, score_column,
                      center=None, covered_segments=None):
    fragment = payload["fragments_graph"]
    if center is None:
        center = candidate_center(fragment, prediction)
    if covered_segments is None:
        covered_segments = covered_segment_ids(payload)
    covered_a, covered_b, coverage_label = coverage_status(prediction, covered_segments)
    center_xyz = np.asarray(center["xyz"], dtype=float)
    anisotropy = np.asarray(payload["anisotropy"], dtype=float)
    patch, origin = read_patch(image, center_xyz, anisotropy, patch_um)
    projections = [np.max(patch, axis=dimension) for _, dimension, _ in VIEWS]
    finite = patch[np.isfinite(patch)]
    if not len(finite):
        raise ValueError("Image patch contains no finite intensities")
    low, high = float(np.min(finite)), float(np.percentile(finite, 99.9))
    if high <= low:
        high = low + 1
    figure = Figure(figsize=(15, 14), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots(3, 3, sharex="col", sharey="col")
    row_names = ("Image (MIP)", "GT", "Fragments")
    origin_um = origin[::-1] * anisotropy
    anchor_xyz = center_xyz - origin_um
    endpoint_a = np.asarray(center["endpoint_a"], dtype=float) - origin_um
    endpoint_b = np.asarray(center["endpoint_b"], dtype=float) - origin_um
    for column, ((view_name, _, xyz_axes), projection) in enumerate(zip(VIEWS, projections)):
        horizontal, vertical = xyz_axes
        height, width = projection.shape
        extent = (-0.5 * anisotropy[horizontal], (width - 0.5) * anisotropy[horizontal],
                  (height - 0.5) * anisotropy[vertical], -0.5 * anisotropy[vertical])
        axes[0, column].imshow(projection, cmap="viridis", origin="upper", extent=extent,
                              vmin=low, vmax=high, interpolation="nearest")
        for row, row_name in enumerate(row_names):
            axis = axes[row, column]
            axis.set_facecolor("black")
            axis.scatter(*anchor_xyz[list(xyz_axes)], marker="+", s=70, color="#ffca28", zorder=5)
            axis.scatter(*endpoint_a[list(xyz_axes)], marker="o", s=90, facecolors="none",
                         edgecolors=SIDE_A_COLOR, linewidths=1.8, zorder=5)
            axis.scatter(*endpoint_b[list(xyz_axes)], marker="s", s=90, facecolors="none",
                         edgecolors=SIDE_B_COLOR, linewidths=1.8, zorder=5)
            axis.set_title(f"{row_name} -- {view_name}", fontsize=14)
            axis.set_aspect("equal")
            axis.set_xlim(extent[:2])
            axis.set_ylim(extent[2:])
            axis.set_xticks([])
            axis.set_yticks([])
    skeleton_overlay(axes[1], payload["gt_graph"], origin, patch.shape, anisotropy)
    skeleton_overlay(axes[2], fragment, origin, patch.shape, anisotropy, highlight={
        str(prediction["segment_id_a"]): SIDE_A_COLOR,
        str(prediction["segment_id_b"]): SIDE_B_COLOR,
    })
    known_split = bool(prediction["is_split"])
    status = "Pool GT: SPLIT-LABELLED PAIR" if known_split else "Pool GT: NOT SPLIT-LABELLED"
    status_color = "#18733c" if known_split else "#b42318"
    caveat = "is_split comes from the candidate pool's cache-derived label, not from this image."
    if not known_split:
        caveat = "Sparse GT: no split label does not prove that the pair should stay apart."
        if covered_a is False and covered_b is False:
            caveat = ("Neither side is touched by a GT tracing, so is_split=0 is vacuous, "
                      "not evidence against joining.")
    z_start = origin[0] * anisotropy[2]
    z_stop = (origin[0] + patch.shape[0]) * anisotropy[2]
    figure.suptitle(
        f"{status} | {coverage_label}\nCenter: candidate gap midpoint",
        fontsize=16, fontweight="bold", color=status_color,
        bbox={"facecolor": "#edf7ef" if known_split else "#fff0ed", "edgecolor": status_color, "pad": 8},
    )
    figure.supxlabel(
        f"Rank {rank} | candidate {prediction['candidate_id']}"
        f" | segments {prediction['segment_id_a']} + {prediction['segment_id_b']}"
        f" | score {prediction['score']:.6g}\n"
        f"{score_column} | gap {prediction['gap_um']:.3g} um | center XYZ (um): "
        f"{center_xyz[0]:.2f}, {center_xyz[1]:.2f}, {center_xyz[2]:.2f}"
        f" | Z slab: [{z_start:.2f}, {z_stop:.2f}) um\n"
        f"+ is the gap midpoint; O / square mark the pair's endpoint nodes A / B"
        f" (model-proposed candidate, not a GT site)\n"
        f"Side A: thick cyan; side B: thick magenta; other skeletons: color per component. {caveat}",
        fontsize=9,
    )
    return figure, {
        "rank": rank, **prediction, "score_column": score_column,
        "cache_is_split": int(known_split),
        "gt_covered_a": "" if covered_a is None else int(covered_a),
        "gt_covered_b": "" if covered_b is None else int(covered_b),
        "x_um": center_xyz[0], "y_um": center_xyz[1], "z_um": center_xyz[2],
        "endpoint_a_x_um": center["endpoint_a"][0], "endpoint_a_y_um": center["endpoint_a"][1],
        "endpoint_a_z_um": center["endpoint_a"][2],
        "endpoint_b_x_um": center["endpoint_b"][0], "endpoint_b_y_um": center["endpoint_b"][1],
        "endpoint_b_z_um": center["endpoint_b"][2],
        "z_start_um": z_start, "z_stop_um": z_stop,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", nargs="?", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--csv", type=Path, help="Override the training prediction CSV")
    parser.add_argument("--pkl", type=Path, help="Matching trusted labeled dataset cache")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--score-column", default=DEFAULT_SCORE,
                        help="Default: nested out-of-fold selector score (scored subset only); no GT-label filtering")
    parser.add_argument("--min-score", type=float, help="Optional lower score cutoff; scores are not calibrated probabilities")
    parser.add_argument("--patch-um", nargs=3, type=float, default=(100, 100, 40),
                        metavar=("X", "Y", "Z"), help="Field of view in micrometers (default: 100 100 40)")
    parser.add_argument("--output-dir", type=Path,
                        help="Default: <result_dir>/top_split_predictions")
    parser.add_argument("--dpi", type=int, default=160)
    args = parser.parse_args(argv)
    if args.dpi < 1 or not all(math.isfinite(size) and size > 0 for size in args.patch_um):
        parser.error("dpi and patch dimensions must be positive")
    csv_path = args.csv
    if csv_path is None:
        candidates = sorted(args.result_dir.glob("split_detector_*.csv"))
        if len(candidates) != 1:
            parser.error("Expected exactly one prediction CSV; specify --csv")
        csv_path = candidates[0]
    predictions = rank_predictions(csv_path, args.top_k, args.score_column, args.min_score)
    if not predictions:
        parser.error("No finite scores pass the requested cutoff")
    pkl_path = args.pkl
    if pkl_path is None:
        selections = sorted(args.result_dir.glob("model_selection_*.json"))
        if len(selections) != 1:
            parser.error("Cannot infer dataset cache; specify --pkl")
        metadata = json.loads(selections[0].read_text())
        pkl_path = ROOT / "cache" / Path(metadata["train_pkl"]).name
        if not pkl_path.exists():
            pkl_path = Path(metadata["train_pkl"])
    output_dir = args.output_dir or args.result_dir / "top_split_predictions"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}", flush=True)
    print(f"Loading trusted cache: {pkl_path}", flush=True)
    with pkl_path.open("rb") as handle:
        payload = pickle.load(handle)
    fragment = payload["fragments_graph"]
    centers = {row["candidate_id"]: candidate_center(fragment, row) for row in predictions}
    covered_segments = covered_segment_ids(payload)
    if covered_segments is None:
        print("WARNING: cache lacks gt_node_canonical_label; GT coverage will read unknown", flush=True)
    from agentic_neuron_proofreader.utils.img_util import TensorStoreImage

    os.environ.setdefault("AWS_EC2_METADATA_DISABLED", "true")
    image = TensorStoreImage(payload["img_path"])
    manifest = []
    for rank, prediction in enumerate(predictions, 1):
        figure, record = render_prediction(
            payload, image, prediction, rank, args.patch_um, args.score_column,
            center=centers[prediction["candidate_id"]], covered_segments=covered_segments,
        )
        filename = f"rank_{rank:03d}_candidate_{prediction['candidate_id']}.png"
        figure.savefig(output_dir / filename, dpi=args.dpi)
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
