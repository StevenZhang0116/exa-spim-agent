"""Save up to K local image/GT/fragment figures at cached GT merge sites.

Run in panda on a compute node. Only load trusted pickle caches. The cache
provides gt_merge_sites and img_path; no prediction CSV or model is needed.
This is a local-site inspector, not the global GT/candidate comparison in
plot_positive_merge_sites.py. It does not generate candidate junctions, recompute
labels, or select by is_merge_site / in_ambiguous_ring.

Both geometric-walk and two-GT sites are eligible unless --source is specified.
After source filtering, exact XYZ positions (micrometers) are deduplicated across
records, including different segments. Up to K centers are sampled uniformly
without replacement using --seed; fewer available centers produces a warning.
There is no GT-visibility filter. Distinct centers may have overlapping views.

Example from the repository root:
    python notebooks/plot_gt_merge_sites.py --brain-id 794495 --k 10 --mcl 100

Each figure has Image MIP / GT / Fragments rows and XY / XZ / YZ columns.
The yellow + marks the stored GT site, not a detector candidate; the fragment
skeleton is not a dense segmentation mask. Unlike the global comparison, this
script reads image patches from the cache's img_path.

Defaults to a new run directory under figs/gt_merge_sites/<brain>_mcl<N>/;
an explicit --output-dir must be empty. selection.json records the sampled
positions; gt_merge_sites.csv records successfully saved PNGs and coordinates.
A failed run can leave partial outputs; retry in a fresh directory. Input caches
and existing figures are never overwritten.
"""

import argparse
import csv
import json
import math
import os
import pickle
import random
import sys
import tempfile
import textwrap
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.plot_top_merge_predictions import render_context


def site_sources(site):
    """Read source tags, treating untagged legacy sites as geometric_walk."""
    declared = site.get("sources", [])
    if not isinstance(declared, (list, tuple, set)):
        raise ValueError("Site sources must be a list of source names")
    sources = set(declared)
    if site.get("source") is not None:
        sources.add(site["source"])
    if not sources:
        sources.add("geometric_walk")
    if not sources.issubset({"geometric_walk", "two_gt_junction"}):
        raise ValueError(f"Unknown site sources: {sources}")
    return sources


def select_sites(sites, k, seed=0, source="all"):
    """Filter by source, deduplicate exact XYZ, then sample up to K centers.

    Return the sampled positions and the total eligible unique-position count.
    The first eligible record at each center supplies the highlighted segment.
    Coincident eligible records are referenced by their original list indices;
    segment IDs, neuron names and source tags are aggregated. Full raw records
    and branch-support evidence are not copied into the selection manifest.
    """
    if k < 1:
        raise ValueError("K must be positive")
    if source not in ("all", "geometric_walk", "two_gt_junction"):
        raise ValueError("Invalid site source filter")
    positions = {}
    for index, site in enumerate(sites):
        sources = site_sources(site)
        if source != "all" and source not in sources:
            continue
        xyz = np.asarray(site["xyz"], dtype=float)
        if xyz.shape != (3,) or not np.isfinite(xyz).all():
            raise ValueError(f"Site {index}: expected finite XYZ coordinates in micrometers")
        segment = int(site["segment_id"])
        if segment <= 0:
            raise ValueError(f"Site {index}: segment_id must be positive")
        neurons = site.get("gt_neurons", [])
        if not isinstance(neurons, (list, tuple, set)):
            raise ValueError(f"Site {index}: gt_neurons must be a list")
        neurons = set(map(str, neurons))
        if site.get("gt_neuron") is not None:
            neurons.add(str(site["gt_neuron"]))
        key = tuple(map(float, xyz))
        if key not in positions:
            positions[key] = {
                "site_index": index, "site_indices": [], "segment_id": segment,
                "segment_ids": set(), "xyz": list(key), "sources": set(), "gt_neurons": set(),
            }
        position = positions[key]
        position["site_indices"].append(index)
        position["segment_ids"].add(segment)
        position["sources"].update(sources)
        position["gt_neurons"].update(neurons)
    normalized = []
    for position in positions.values():
        normalized.append({**position, "segment_ids": sorted(position["segment_ids"]),
                           "sources": sorted(position["sources"]),
                           "gt_neurons": sorted(position["gt_neurons"])})
    selected = random.Random(seed).sample(normalized, min(k, len(normalized)))
    return selected, len(normalized)


def render_site(payload, image, site, brain, plot_number, patch_um):
    """Render one local nine-panel view centered on the stored GT coordinate."""
    figure, origin, shape = render_context(payload, image, site["xyz"], site["segment_id"], patch_um)
    sources = ", ".join(site["sources"])
    neurons = ", ".join(site["gt_neurons"]) or "not recorded"
    xyz = site["xyz"]
    figure.suptitle(f"Brain {brain} | GT merge site {site['site_index']} | segment {site['segment_id']}\n"
                   f"Source: {sources}", fontsize=15, fontweight="bold")
    caption = (
        f"Position {plot_number} | Center XYZ (um): {xyz[0]:.2f}, {xyz[1]:.2f}, {xyz[2]:.2f}\n"
        + textwrap.fill(f"Associated GT neurons: {neurons}", width=110)
        + "\n+ marks the stored GT-derived site; cyan highlights its segment in the fragments row."
        + "\nRule-derived location, not an independently verified cut. A geometric merge can have only one traced partner."
    )
    if len(site["site_indices"]) > 1:
        caption += (f"\n{len(site['site_indices'])} records share this center; "
                    f"the first record's segment ({site['segment_id']}) is highlighted.")
    figure.supxlabel(caption, fontsize=10)
    return figure, {
        "plot_number": plot_number, "brain_id": brain, "site_index": site["site_index"],
        "site_indices": json.dumps(site["site_indices"]), "segment_id": site["segment_id"],
        "segment_ids": json.dumps(site["segment_ids"]), "sources": json.dumps(site["sources"]),
        "gt_neurons": json.dumps(site["gt_neurons"]),
        "x_um": xyz[0], "y_um": xyz[1], "z_um": xyz[2],
        "origin_zyx": json.dumps(list(map(int, origin))),
        "shape_zyx": json.dumps(list(map(int, shape))),
    }


def open_image(path):
    from agentic_neuron_proofreader.utils.img_util import TensorStoreImage

    os.environ.setdefault("AWS_EC2_METADATA_DISABLED", "true")
    return TensorStoreImage(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--brain-id", "--brain", required=True, dest="brain")
    parser.add_argument("--k", "-k", type=int, default=10, help="Maximum number of distinct GT positions (default: 10)")
    parser.add_argument("--mcl", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0, help="Reproducible random sample; change for other positions")
    parser.add_argument("--source", choices=("all", "geometric_walk", "two_gt_junction"), default="all",
                        help="Filter source tags before position deduplication and sampling")
    parser.add_argument("--patch-um", nargs=3, type=float, default=(100., 100., 100.),
                        metavar=("X", "Y", "Z"), help="XYZ field of view in micrometers")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "cache")
    parser.add_argument("--output-dir", type=Path, help="Empty output directory; default: a new run in figs/gt_merge_sites")
    parser.add_argument("--dpi", type=int, default=160)
    args = parser.parse_args(argv)
    if not args.brain.isascii() or not args.brain.isdigit():
        parser.error("--brain-id must be a numeric brain ID")
    if args.k < 1 or args.mcl < 0 or args.dpi < 1:
        parser.error("K and dpi must be positive; mcl must be nonnegative")
    if not all(math.isfinite(value) and value > 0 for value in args.patch_um):
        parser.error("--patch-um requires three positive finite dimensions")
    if args.output_dir is not None and args.output_dir.exists():
        if not args.output_dir.is_dir() or any(args.output_dir.iterdir()):
            parser.error("--output-dir must be empty; existing figures will not be overwritten")
    cache_path = args.cache_dir / f"dataset_cache_{args.brain}_mcl{args.mcl}_add.pkl"
    print(f"Loading trusted cache: {cache_path}", flush=True)
    with cache_path.open("rb") as handle:
        payload = pickle.load(handle)
    if float(payload.get("min_cable_length", args.mcl)) != args.mcl:
        parser.error("Cache min_cable_length disagrees with requested MCL")
    sites = payload.get("gt_merge_sites", getattr(payload.get("gt_graph"), "merge_sites", None))
    if sites is None:
        parser.error("Cache has no gt_merge_sites; generate labelled sites first")
    selected, available = select_sites(sites, args.k, args.seed, args.source)
    if not selected:
        parser.error("No GT merge-site positions match the requested source")
    if len(selected) < args.k:
        print(f"WARNING: requested {args.k} positions, but only {available} distinct positions are available; "
              f"rendering all {available}.", flush=True)
    print(f"Selected {len(selected)} of {available} distinct positions from {len(sites)} site records "
          f"(source={args.source}, seed={args.seed}).", flush=True)
    if args.output_dir is None:
        parent = ROOT / "figs" / "gt_merge_sites" / f"{args.brain}_mcl{args.mcl}"
        parent.mkdir(parents=True, exist_ok=True)
        output_dir = Path(tempfile.mkdtemp(prefix=f"seed{args.seed}_", dir=parent))
    else:
        output_dir = args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
    image = open_image(payload["img_path"])
    print(f"Output directory: {output_dir}\nImage source: {payload['img_path']}", flush=True)
    run = {
        "brain_id": args.brain, "mcl": args.mcl, "cache": str(cache_path.resolve()),
        "img_path": str(payload["img_path"]), "requested_k": args.k,
        "available_positions": available, "seed": args.seed, "source_filter": args.source,
        "patch_um_xyz": list(args.patch_um), "selected_positions": selected,
        "cache_size_bytes": cache_path.stat().st_size, "cache_mtime_ns": cache_path.stat().st_mtime_ns,
    }
    with (output_dir / "selection.json").open("x") as handle:
        json.dump(run, handle, indent=2, allow_nan=False)
    with (output_dir / "gt_merge_sites.csv").open("x", newline="") as handle:
        writer = None
        for number, site in enumerate(selected, 1):
            figure, record = render_site(payload, image, site, args.brain, number, args.patch_um)
            filename = f"position_{number:03d}_site_{site['site_index']}_segment_{site['segment_id']}.png"
            try:
                with (output_dir / filename).open("xb") as image_handle:
                    figure.savefig(image_handle, format="png", dpi=args.dpi, bbox_inches="tight", pad_inches=0.15)
            finally:
                figure.clear()
            record.update(mcl=args.mcl, seed=args.seed, figure=filename)
            if writer is None:
                writer = csv.DictWriter(handle, fieldnames=list(record))
                writer.writeheader()
            writer.writerow(record)
            handle.flush()
            print(f"[{number}/{len(selected)}] {output_dir / filename}", flush=True)
    print(f"Saved {len(selected)} figures, selection.json and gt_merge_sites.csv; cache unchanged.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())