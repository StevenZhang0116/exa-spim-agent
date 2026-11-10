#!/usr/bin/env python3
"""Raw versus denoised fused image of one brain: the same voxels read from both volumes.

AIND publishes two full-resolution volumes for some ExaSPIM brains in the same dataset prefix:
the raw fusion ``fused.zarr`` and a denoised one ``fused-denoised.zarr`` with the same shape,
voxel size (on 2026-11-08 the raw volumes of 754613 and 751473 were deleted, on 2026-11-09
those of 747807, 750318 and 754610, and the repository switched these five brains to the
denoised volume; the ``fused.zarr`` of 841260 and 794495 were denoised by AIND at the tile
level before fusion; see ``cache/dataset_source_status_20261108.md``). This
script measures what the denoiser changes, so that image features computed on a denoised
brain can be compared with the same features on a raw brain.

Both volumes are read through the repository's image reader
(``proofreader_evolve.harness.image_context.ImageSource``), at label-blind locations: by
default the fragment nodes of an alignment sample (``figs/candidate_image_alignment_<date>/
<brain>/alignment.json``, six ``center_xyz_um`` per brain, chosen by
``scripts/check_image_alignment_from_cache.py`` without GT labels); ``--centers-xyz-um`` names
locations by hand instead. Each patch is a cube of ``2 x radius_um`` per axis at ``--level``.

Per patch, both patches use the raw patch's median as the background threshold:

- ``pearson_r``: correlation of the two patches over every voxel;
- intensity percentiles 1 / 50 / 99 / 99.9 of each patch (``raw_p50`` ...), the difference
  ``denoised - raw`` (mean and median) and ``peak_contrast_ratio`` = (p99.9 - p50) denoised
  over raw: whether the intensity scale changed;
- ``raw_bg_hp_std`` / ``denoised_bg_hp_std``: standard deviation of the high-pass residual
  (patch minus a Gaussian blur, sigma 1 voxel in z and 1.5 in y and x) over background voxels,
  and ``noise_ratio`` = denoised / raw: how much voxel noise the denoiser removed;
- ``raw_bg_mad`` / ``denoised_bg_mad``: median absolute deviation (scaled) of background voxels.

Outputs under ``notebooks/raw_vs_denoised_image/<brain>/`` (gitignored): ``statistics.csv``
(one row per patch), ``summary.json`` (roots, image identities, patch metadata and the rows),
``patches.png`` (per patch: central slice raw / denoised / difference, and the z maximum
projections of both) and ``summary.png`` (background noise and percentile parity per patch).

Run on a compute node in the ``panda`` environment (anonymous S3 reads;
``AWS_EC2_METADATA_DISABLED=true`` is set by ``labeled_cache.io.set_cloud_environment``):

    srun --jobid=<job> --overlap -n1 -w <node> --export=ALL bash -lc 'source ~/.bashrc; \
        conda activate panda; cd exa-spim-agent; \
        python -u notebooks/raw_vs_denoised_image.py --brain 754612'

The roots default to the brain's entry in ``configs/exaspim_image_prefixes.json`` with the zarr
name replaced (``fused.zarr`` / ``fused-denoised.zarr``); ``--raw-root`` / ``--denoised-root``
override them. A brain whose raw volume no longer exists (754613, 751473, 747807, 750318,
754610) fails with ``Missing image metadata`` unless ``--raw-root`` names another raw volume
on the same grid; the outputs of 747807, 750318 and 754610 were written on 2026-11-08, before
their raw volumes disappeared. On 2026-11-09 both volumes existed for 754612 and 794493 only. Read-only: nothing
in the repository, the caches or the cloud is modified.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from labeled_cache import io as cache_io  # noqa: E402
from proofreader_evolve.harness.image_context import ImageSource  # noqa: E402

OUT_DIR = ROOT / "notebooks" / "raw_vs_denoised_image"
DEFAULT_ALIGNMENT_DIR = ROOT / "figs" / "candidate_image_alignment_20261018"
RAW_NAME, DENOISED_NAME = "fused.zarr", "fused-denoised.zarr"
PERCENTILES = (1, 50, 99, 99.9)
HIGH_PASS_SIGMA_ZYX = (1.0, 1.5, 1.5)


def log(message):
    print(message, flush=True)


def default_roots(brain):
    """Raw and denoised roots derived from the configured prefix of the brain."""
    prefix = str(cache_io.image_path(brain))  # level-0 path, e.g. .../fused.zarr/0
    marker = prefix.find(".zarr")
    if marker < 0:
        raise ValueError(f"Brain {brain}: the configured image path is not a zarr: {prefix}")
    root = prefix[:marker + 5]
    base = root[:root.rfind("/") + 1]
    return base + RAW_NAME + "/", base + DENOISED_NAME + "/"


def sample_centers(alignment_dir, brain):
    """Label-blind patch centres recorded by an alignment sample of the brain."""
    report = alignment_dir / str(brain) / "alignment.json"
    if not report.exists():
        raise FileNotFoundError(f"No alignment sample for brain {brain} under {alignment_dir}; "
                                "run scripts/check_image_alignment_from_cache.py or pass --centers-xyz-um")
    cases = json.loads(report.read_text())["cases"]
    centers = []
    for case in cases:
        metadata = case["metadata"]
        if isinstance(metadata, str):
            import ast
            metadata = ast.literal_eval(metadata)
        centers.append((case.get("node"), [float(v) for v in metadata["center_xyz_um"]]))
    return centers


def read_patch(source, center_xyz_um, radius_um, level):
    patch, metadata = source.read([center_xyz_um], radius_um=radius_um, level=level)
    return patch["image_zyx"].astype(np.float32), patch["valid_zyx"], metadata


def patch_statistics(raw, denoised, valid):
    from scipy.ndimage import gaussian_filter

    raw_v, den_v = raw[valid], denoised[valid]
    background = valid & (raw <= np.percentile(raw_v, 50))
    row = {"pearson_r": float(np.corrcoef(raw_v, den_v)[0, 1]),
           "valid_fraction": float(valid.mean()), "voxels": int(valid.sum())}
    for name, values in (("raw", raw_v), ("denoised", den_v)):
        for p, value in zip(PERCENTILES, np.percentile(values, PERCENTILES)):
            row[f"{name}_p{p:g}"] = float(value)
    difference = den_v - raw_v
    row["mean_difference"] = float(difference.mean())
    row["median_difference"] = float(np.median(difference))
    row["peak_contrast_ratio"] = float((row["denoised_p99.9"] - row["denoised_p50"])
                                       / max(row["raw_p99.9"] - row["raw_p50"], 1e-6))
    for name, image in (("raw", raw), ("denoised", denoised)):
        residual = image - gaussian_filter(image, sigma=HIGH_PASS_SIGMA_ZYX)
        row[f"{name}_bg_hp_std"] = float(residual[background].std())
        bg = image[background]
        row[f"{name}_bg_mad"] = float(1.4826 * np.median(np.abs(bg - np.median(bg))))
    row["noise_ratio"] = float(row["denoised_bg_hp_std"] / max(row["raw_bg_hp_std"], 1e-6))
    return row


def plot_patches(records, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(records), 5, figsize=(20, 3.8 * len(records)), squeeze=False)
    for i, record in enumerate(records):
        raw, denoised, row = record["raw"], record["denoised"], record["row"]
        z = raw.shape[0] // 2
        low, high = min(row["raw_p50"], row["denoised_p50"]), max(row["raw_p99.9"], row["denoised_p99.9"])
        label = f"node {record['node']}" if record["node"] is not None else f"patch {i}"
        panels = ((raw[z], f"{label}: raw z={z}", "gray", low, high),
                  (denoised[z], f"denoised z={z}", "gray", low, high),
                  (denoised[z] - raw[z], "denoised - raw (+-40)", "coolwarm", -40, 40),
                  (raw.max(axis=0), "raw z-MIP", "gray", low, high),
                  (denoised.max(axis=0), "denoised z-MIP", "gray", low, high))
        for ax, (image, title, cmap, vmin, vmax) in zip(axes[i], panels):
            ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(title, fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle("Same voxels from the raw and the denoised volume (display range per row: p50 to p99.9)", y=1.0)
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


def plot_summary(rows, brain, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [str(r.get("node") or i) for i, r in enumerate(rows)]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    x = np.arange(len(rows))
    axes[0].bar(x - 0.2, [r["raw_bg_hp_std"] for r in rows], width=0.4, label="raw")
    axes[0].bar(x + 0.2, [r["denoised_bg_hp_std"] for r in rows], width=0.4, label="denoised")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    axes[0].set_ylabel("background high-pass std (counts)")
    axes[0].set_yscale("log")
    axes[0].set_title("Voxel noise in background")
    axes[0].legend()
    for p, marker in zip(PERCENTILES[1:], ("o", "s", "^")):
        axes[1].scatter([r[f"raw_p{p:g}"] for r in rows], [r[f"denoised_p{p:g}"] for r in rows],
                        marker=marker, label=f"p{p:g}")
    lim = max(max(r["raw_p99.9"], r["denoised_p99.9"]) for r in rows) * 1.1
    axes[1].plot([1, lim], [1, lim], "k--", linewidth=0.8)
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("raw (counts)")
    axes[1].set_ylabel("denoised (counts)")
    axes[1].set_title("Intensity percentiles per patch")
    axes[1].legend()
    axes[2].bar(x, [r["pearson_r"] for r in rows], color="tab:gray")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    axes[2].set_ylim(0.0, 1.02)
    axes[2].set_ylabel("Pearson r")
    axes[2].set_title("Raw / denoised correlation")
    fig.suptitle(f"Brain {brain}: raw versus denoised volume, {len(rows)} label-blind patches")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--brain", default="754612")
    parser.add_argument("--raw-root", help="raw volume root (default: the configured prefix with fused.zarr)")
    parser.add_argument("--denoised-root", help="denoised volume root (default: the same prefix with fused-denoised.zarr)")
    parser.add_argument("--alignment-dir", type=Path, default=DEFAULT_ALIGNMENT_DIR,
                        help="figs/candidate_image_alignment_<date> whose <brain>/alignment.json gives the patch centres")
    parser.add_argument("--centers-xyz-um", type=float, nargs="+",
                        help="explicit centres as x y z triples in um, instead of the alignment sample")
    parser.add_argument("--radius-um", type=float, default=40.0)
    parser.add_argument("--level", type=int, default=0)
    parser.add_argument("--max-patches", type=int, default=None)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args(argv)

    cache_io.set_cloud_environment()
    raw_root, denoised_root = default_roots(args.brain)
    raw_root = args.raw_root or raw_root
    denoised_root = args.denoised_root or denoised_root
    if args.centers_xyz_um:
        if len(args.centers_xyz_um) % 3:
            parser.error("--centers-xyz-um takes x y z triples")
        centers = [(None, list(args.centers_xyz_um[i:i + 3])) for i in range(0, len(args.centers_xyz_um), 3)]
    else:
        centers = sample_centers(args.alignment_dir, args.brain)
    if args.max_patches:
        centers = centers[:args.max_patches]

    log(f"[{args.brain}] raw      {raw_root}")
    log(f"[{args.brain}] denoised {denoised_root}")
    raw_source, denoised_source = ImageSource(raw_root), ImageSource(denoised_root)
    raw_geometry, denoised_geometry = raw_source.level(args.level), denoised_source.level(args.level)
    if tuple(raw_geometry.shape_zyx) != tuple(denoised_geometry.shape_zyx) \
            or not np.allclose(raw_geometry.scale_xyz_um, denoised_geometry.scale_xyz_um):
        raise ValueError(f"The two volumes differ in grid: raw {raw_geometry.shape_zyx} {raw_geometry.scale_xyz_um}, "
                         f"denoised {denoised_geometry.shape_zyx} {denoised_geometry.scale_xyz_um}")
    log(f"[{args.brain}] level {args.level}: shape zyx {raw_geometry.shape_zyx.tolist()}, "
        f"spacing xyz um {raw_geometry.scale_xyz_um.tolist()}, {len(centers)} patches of radius {args.radius_um} um")

    out_dir = args.out_dir / str(args.brain)
    out_dir.mkdir(parents=True, exist_ok=True)
    records, rows = [], []
    for node, center in centers:
        raw, raw_valid, raw_meta = read_patch(raw_source, center, args.radius_um, args.level)
        denoised, den_valid, den_meta = read_patch(denoised_source, center, args.radius_um, args.level)
        if raw_meta["origin_zyx"] != den_meta["origin_zyx"] or raw.shape != denoised.shape:
            raise ValueError("Patch geometry differs between the two volumes")
        valid = raw_valid & den_valid
        row = {"node": node, "center_xyz_um": center, "origin_zyx": raw_meta["origin_zyx"],
               "shape_zyx": raw_meta["shape_zyx"], **patch_statistics(raw, denoised, valid)}
        rows.append(row)
        records.append({"node": node, "raw": raw, "denoised": denoised, "row": row})
        log(f"[{args.brain}] {'node ' + str(node) if node is not None else 'patch'} origin {raw_meta['origin_zyx']}: "
            f"r={row['pearson_r']:.3f}, p50 {row['raw_p50']:.0f}->{row['denoised_p50']:.0f}, "
            f"p99.9 {row['raw_p99.9']:.0f}->{row['denoised_p99.9']:.0f}, "
            f"bg noise {row['raw_bg_hp_std']:.2f}->{row['denoised_bg_hp_std']:.2f} (x{row['noise_ratio']:.2f})")

    columns = [k for k in rows[0] if k not in ("center_xyz_um", "origin_zyx", "shape_zyx")]
    with (out_dir / "statistics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in columns})
    summary = {"brain": str(args.brain), "raw": raw_source.identity, "denoised": denoised_source.identity,
               "level": args.level, "radius_um": args.radius_um, "high_pass_sigma_zyx": HIGH_PASS_SIGMA_ZYX,
               "background": "voxels at or below the raw patch's median", "patches": rows,
               "medians": {k: float(np.median([r[k] for r in rows])) for k in
                           ("pearson_r", "noise_ratio", "peak_contrast_ratio", "mean_difference")}}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    plot_patches(records, out_dir / "patches.png")
    plot_summary(rows, args.brain, out_dir / "summary.png")
    medians = summary["medians"]
    log(f"[{args.brain}] medians over {len(rows)} patches: r {medians['pearson_r']:.3f}, noise ratio "
        f"{medians['noise_ratio']:.2f}, peak contrast ratio {medians['peak_contrast_ratio']:.2f}, "
        f"mean difference {medians['mean_difference']:+.2f} counts")
    log(f"wrote {out_dir / 'statistics.csv'}, {out_dir / 'summary.json'}, {out_dir / 'patches.png'}, {out_dir / 'summary.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
