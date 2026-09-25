"""Compare all GT merge sites and positive candidates in ONE XY/XZ figure.

Run in panda on a compute node. Only load trusted pickle caches. Reads
gt_merge_sites from the selected _add.pkl and is_merge_site=1 junctions from
the supplied CSV, without recalculating labels or reading images/model scores.
Blue open circles mark GT sites and red dots mark positive candidates over
all candidate positions in gray, using world coordinates in mm.
No top-K, score threshold, GT-visibility filter or deduplication is applied.
Neither set counts independent merge events; multiple candidates may correspond
to the same recorded site. The gray background is not an anatomical brain outline.

Example (from the repository root):
    python notebooks/plot_positive_merge_sites.py --brain-id 794495 \
        --csv path/to/merge_junction_detector_794495.csv --mcl 100 --cache-dir cache

Outputs go to a new run under figs/positive_merge_sites/<brain>_mcl<N>/, or
an empty --output-dir. Saves one PNG, both coordinate CSVs and an input manifest;
input caches/CSVs and existing figures are not overwritten.
"""

import argparse
import csv
import hashlib
import json
import math
import pickle
import tempfile
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
IDENTITY_COLUMNS = ("candidate_id", "node_id", "segment_id", "degree", "is_merge_site",
                    "x_um", "y_um", "z_um")
VIEWS = (("XY", 0, 1), ("XZ", 0, 2))
GT_COLOR = "#0072b2"
POSITIVE_COLOR = "#c7374f"
BACKGROUND_COLOR = "#8c969e"


def load_candidates(path):
    """Read candidate coordinates and binary labels, ignoring model scores."""
    required = set(IDENTITY_COLUMNS)
    selected, seen = [], set()
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Junction CSV must contain {sorted(required)}")
        for row in reader:
            candidate_id = int(row["candidate_id"])
            if candidate_id in seen:
                raise ValueError(f"Duplicate candidate_id: {candidate_id}")
            seen.add(candidate_id)
            label = int(row["is_merge_site"])
            if label not in (0, 1):
                raise ValueError(f"Candidate {candidate_id}: is_merge_site must be 0 or 1")
            candidate = {key: int(row[key]) for key in
                         ("candidate_id", "node_id", "segment_id", "degree", "is_merge_site")}
            candidate.update({key: float(row[key]) for key in ("x_um", "y_um", "z_um")})
            if (candidate_id < 0 or candidate["node_id"] < 0 or candidate["segment_id"] <= 0
                    or candidate["degree"] < 3
                    or not all(math.isfinite(candidate[key]) for key in ("x_um", "y_um", "z_um"))):
                raise ValueError(f"Candidate {candidate_id}: invalid junction identity or coordinates")
            selected.append(candidate)
    return sorted(selected, key=lambda row: row["candidate_id"])


def load_positive_candidates(path):
    """Keep every positive, including rows with absent/nonfinite model scores."""
    return [row for row in load_candidates(path) if row["is_merge_site"] == 1]


def load_gt_sites(path, mcl):
    """Read all cached GT site records without sampling or coordinate deduplication."""
    with Path(path).open("rb") as handle:
        payload = pickle.load(handle)
    if float(payload.get("min_cable_length", mcl)) != mcl:
        raise ValueError("Cache min_cable_length disagrees with requested MCL")
    sites = payload.get("gt_merge_sites", getattr(payload.get("gt_graph"), "merge_sites", None))
    if sites is None:
        raise ValueError("Cache has no gt_merge_sites; generate labelled sites first")
    normalized = []
    for index, site in enumerate(sites):
        xyz = np.asarray(site["xyz"], dtype=float)
        if xyz.shape != (3,) or not np.isfinite(xyz).all():
            raise ValueError(f"Site {index}: expected finite XYZ coordinates in micrometers")
        segment = int(site["segment_id"])
        if segment <= 0:
            raise ValueError(f"Site {index}: segment_id must be positive")
        normalized.append({"site_index": index, "segment_id": segment, "xyz": xyz.tolist()})
    return normalized


def render_distribution(candidates, brain, mcl, gt_sites):
    """Show every GT site and positive candidate on two world-coordinate projections."""
    if not candidates and not gt_sites:
        raise ValueError("No candidate rows or GT sites available for a spatial distribution")
    xyz_mm = np.asarray([[row[key] for key in ("x_um", "y_um", "z_um")]
                         for row in candidates], dtype=float).reshape(-1, 3) / 1000.0
    gt_xyz_mm = np.asarray([site["xyz"] for site in gt_sites], dtype=float).reshape(-1, 3) / 1000.0
    positive = np.asarray([row["is_merge_site"] == 1 for row in candidates], dtype=bool)
    count = int(positive.sum())
    all_xyz = np.concatenate((xyz_mm, gt_xyz_mm))
    lower, upper = all_xyz.min(axis=0), all_xyz.max(axis=0)
    margin = np.maximum((upper - lower) * 0.04, 0.05)
    figure = Figure(figsize=(14, 6.8))
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, 2)
    for axis, (view, horizontal, vertical) in zip(axes, VIEWS):
        axis.scatter(xyz_mm[:, horizontal], xyz_mm[:, vertical], s=2,
                     color=BACKGROUND_COLOR, alpha=0.16, linewidths=0, zorder=1)
        axis.scatter(xyz_mm[positive, horizontal], xyz_mm[positive, vertical], s=18,
                 color=POSITIVE_COLOR, linewidths=0, alpha=0.95, zorder=3)
        axis.scatter(gt_xyz_mm[:, horizontal], gt_xyz_mm[:, vertical], s=80,
                 facecolors="none", edgecolors=GT_COLOR, linewidths=1.3, zorder=4)
        axis.set_title(view, fontsize=15)
        axis.set_xlabel(f"{'XYZ'[horizontal]} (mm)")
        axis.set_ylabel(f"{'XYZ'[vertical]} (mm)")
        axis.set_xlim(lower[horizontal] - margin[horizontal], upper[horizontal] + margin[horizontal])
        axis.set_ylim(lower[vertical] - margin[vertical], upper[vertical] + margin[vertical])
        axis.set_aspect("equal", adjustable="box")
        axis.set_axisbelow(True)
        axis.grid(alpha=0.2, linewidth=0.6)
        axis.spines[["top", "right"]].set_visible(False)
        if count == 0:
            axis.text(0.5, 0.98, "No is_merge_site=1 candidates", ha="center", va="top",
                      transform=axis.transAxes)
        if not gt_sites:
            axis.text(0.5, 0.90, "No gt_merge_sites", ha="center", va="top", transform=axis.transAxes)
    figure.suptitle(
        f"Brain {brain} | mcl{mcl} | GT merge sites and positive candidate locations", fontsize=17)
    figure.legend(handles=[
        Line2D([], [], linestyle="none", marker="o", markersize=5, color=BACKGROUND_COLOR,
               label=f"All candidate positions ({len(candidates):,})"),
        Line2D([], [], linestyle="none", marker="o", markersize=8, color=GT_COLOR,
               markerfacecolor="none", markeredgewidth=1.3, label=f"gt_merge_sites ({len(gt_sites):,})"),
        Line2D([], [], linestyle="none", marker="o", markersize=4, color=POSITIVE_COLOR,
               label=f"is_merge_site = 1 ({count:,})"),
    ], loc="upper center", bbox_to_anchor=(0.5, 0.925), ncol=3, frameon=False)
    figure.supxlabel(
        "World-coordinate projections; gray points are candidates, not an anatomical brain outline.\n"
        "All GT site records and positive rows retained; overlapping markers do not imply one-to-one matching.",
        fontsize=10, y=0.015)
    figure.tight_layout(rect=(0, 0.10, 1, 0.85), w_pad=2.)
    return figure


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--brain-id", "--brain", dest="brain", required=True)
    parser.add_argument("--csv", type=Path, required=True, help="Junction CSV containing is_merge_site")
    parser.add_argument("--mcl", type=int, default=100)
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "cache", help="Directory containing the trusted _add.pkl")
    parser.add_argument("--output-dir", type=Path, help="Empty output directory; default: a new run under figs/positive_merge_sites")
    parser.add_argument("--dpi", type=int, default=160)
    args = parser.parse_args(argv)
    if not args.brain.isascii() or not args.brain.isdigit():
        parser.error("--brain-id must be a numeric brain ID")
    if args.mcl < 0 or args.dpi < 1:
        parser.error("mcl must be nonnegative; dpi must be positive")
    if args.output_dir is not None and args.output_dir.exists():
        if not args.output_dir.is_dir() or any(args.output_dir.iterdir()):
            parser.error("--output-dir must be empty; existing figures will not be overwritten")
    candidates = load_candidates(args.csv)
    positives = [row for row in candidates if row["is_merge_site"] == 1]
    cache_path = args.cache_dir / f"dataset_cache_{args.brain}_mcl{args.mcl}_add.pkl"
    print(f"Loading trusted cache: {cache_path}", flush=True)
    cache_stat = cache_path.stat()
    gt_sites = load_gt_sites(cache_path, args.mcl)
    current_stat = cache_path.stat()
    if ((cache_stat.st_size, cache_stat.st_mtime_ns, cache_stat.st_ino)
            != (current_stat.st_size, current_stat.st_mtime_ns, current_stat.st_ino)):
        raise RuntimeError("Cache changed while loading GT sites; retry with stable inputs")
    print(f"Plotting all {len(positives)} positive positions among {len(candidates)} candidates from {args.csv}", flush=True)
    print(f"Overlaying all {len(gt_sites)} stored GT merge sites.", flush=True)
    if not candidates and not gt_sites:
        print("No candidate rows or GT sites; no figure generated.", flush=True)
        return 0
    if args.output_dir is None:
        parent = ROOT / "figs" / "positive_merge_sites" / f"{args.brain}_mcl{args.mcl}"
        parent.mkdir(parents=True, exist_ok=True)
        output = Path(tempfile.mkdtemp(prefix="comparison_", dir=parent))
    else:
        output = args.output_dir
        output.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output}", flush=True)
    filename = f"merge_site_comparison_{args.brain}_mcl{args.mcl}.png"
    figure = render_distribution(candidates, args.brain, args.mcl, gt_sites)
    try:
        with (output / filename).open("xb") as handle:
            figure.savefig(handle, format="png", dpi=args.dpi)
    finally:
        figure.clear()
    digest = hashlib.sha256()
    with args.csv.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    with (output / "selection.json").open("x") as handle:
        json.dump({"brain_id": args.brain, "mcl": args.mcl,
                   "csv": str(args.csv.resolve()), "selection": "all gt_merge_sites and all is_merge_site == 1",
                   "csv_sha256": digest.hexdigest(), "plot_units": "mm", "figure": filename,
                   "views": [view for view, _, _ in VIEWS],
                   "cache": str(cache_path.resolve()), "cache_size_bytes": cache_stat.st_size,
                   "cache_mtime_ns": cache_stat.st_mtime_ns, "cache_inode": cache_stat.st_ino,
                   "gt_site_count": len(gt_sites),
                   "unique_gt_xyz_count": len({tuple(site["xyz"]) for site in gt_sites}),
                   "candidate_count": len(positives), "total_candidate_count": len(candidates),
                   "unique_positive_xyz_count": len({(row["x_um"], row["y_um"], row["z_um"]) for row in positives}),
                   "candidate_ids": [row["candidate_id"] for row in positives]}, handle, indent=2)
    with (output / "positive_merge_sites.csv").open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[*IDENTITY_COLUMNS, "brain_id", "mcl", "figure"])
        writer.writeheader()
        writer.writerows({**row, "brain_id": args.brain, "mcl": args.mcl, "figure": filename} for row in positives)
    with (output / "gt_merge_sites.csv").open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["site_index", "segment_id", "x_um", "y_um", "z_um",
                                                   "brain_id", "mcl", "figure"])
        writer.writeheader()
        writer.writerows({"site_index": site["site_index"], "segment_id": site["segment_id"],
                          "x_um": site["xyz"][0], "y_um": site["xyz"][1], "z_um": site["xyz"][2],
                          "brain_id": args.brain, "mcl": args.mcl, "figure": filename} for site in gt_sites)
    print(f"Saved ONE XY/XZ figure with {len(gt_sites)} GT sites and {len(positives)} positive candidates: {output / filename}", flush=True)
    print("Coordinates saved in original micrometers; input cache/CSV unchanged, no image volumes loaded.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())