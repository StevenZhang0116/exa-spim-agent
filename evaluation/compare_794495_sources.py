#!/usr/bin/env python
"""Small read-only comparison of repository and CodeOcean 6960813 data sources.

Run with the panda Python on a compute node. This compares current cloud sources,
not GT already serialized in historical caches, and performs no model evaluation.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import socket
import sys
from urllib.parse import urlsplit

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from labeled_cache.io import GT_PATH_TEMPLATE, set_cloud_environment  # noqa: E402
from proofreader_evolve.harness.image_context import ImageSource  # noqa: E402
from scripts.dataset_config import get_reviewed_img_path  # noqa: E402

BRAIN = "794495"
CAPSULE = ROOT / "external/codeocean/capsule_6960813"
# The default in labeled_cache/__main__.py, in SWC xyz order.
GT_SCALE_XYZ_UM = np.array([0.748, 0.748, 1.0])


def capsule_constants():
    """Read literal configuration without importing/executing upstream scripts."""
    tree = ast.parse((CAPSULE / "code/review_merges/build_review.py").read_text())
    values = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in ("BRAIN", "VOLUME"):
                values[target.id] = ast.literal_eval(node.value)
    if values.get("BRAIN") != BRAIN:
        raise ValueError("Imported capsule no longer refers to brain 794495")
    # Main's f-string path contains only the brain_id substitution.
    tree = ast.parse((CAPSULE / "code/main.py").read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "swcs_path" for t in node.targets):
            if not isinstance(node.value, ast.JoinedStr):
                continue
            parts = []
            for part in node.value.values:
                if isinstance(part, ast.Constant) and isinstance(part.value, str):
                    parts.append(part.value)
                elif (isinstance(part, ast.FormattedValue)
                      and isinstance(part.value, ast.Name) and part.value.id == "brain_id"
                      and part.conversion == -1 and part.format_spec is None):
                    parts.append(BRAIN)
                else:
                    raise ValueError("Unsupported capsule GT path expression")
            values["GT"] = "".join(parts)
    return values["GT"], values["VOLUME"]


def list_swcs(client, uri):
    parsed = urlsplit(uri)
    if parsed.scheme != "gs":
        raise ValueError("GT comparison expects a GCS SWC directory")
    prefix = parsed.path.lstrip("/").rstrip("/") + "/"
    files = {}
    for blob in client.list_blobs(parsed.netloc, prefix=prefix, timeout=45):
        relative = blob.name[len(prefix):]
        if relative.endswith(".swc"):
            files[relative] = blob
    return files


def load_swc(blob):
    body = blob.download_as_bytes(timeout=60)
    data = np.loadtxt(io.BytesIO(body), comments="#", ndmin=2)
    if (data.shape[1] < 7 or not np.isfinite(data[:, :7]).all()
            or len(np.unique(data[:, 0])) != len(data)):
        raise ValueError(f"Invalid SWC: {blob.name}")
    return data, {"object": blob.name, "generation": blob.generation,
                  "sha256": hashlib.sha256(body).hexdigest(), "bytes": len(body)}


def compare_swc(voxel, world):
    ids, a, b = np.intersect1d(voxel[:, 0], world[:, 0], return_indices=True)
    if not len(ids):
        raise ValueError("SWCs have no common node IDs; this script does not infer matches")
    delta = np.linalg.norm(voxel[a, 2:5] * GT_SCALE_XYZ_UM - world[b, 2:5], axis=1)
    return {"repository_nodes": len(voxel), "capsule_nodes": len(world),
            "common_node_ids": len(ids),
            "parent_id_mismatches": int(np.count_nonzero(voxel[a, 6] != world[b, 6])),
            "node_type_mismatches": int(np.count_nonzero(voxel[a, 1] != world[b, 1])),
            "xyz_error_um": {"median": float(np.median(delta)),
                             "p99": float(np.percentile(delta, 99)), "max": float(delta.max())},
            "fraction_xyz_within_1e_minus_6_um": float(np.mean(delta <= 1e-6))}


def same_grid(a, b):
    return (np.array_equal(a.shape_zyx, b.shape_zyx)
            and np.allclose(a.scale_xyz_um, b.scale_xyz_um, rtol=0, atol=1e-9)
            and np.allclose(a.translation_xyz_um, b.translation_xyz_um, rtol=0, atol=1e-9))


def image_statistics(a, b):
    x, y = a.astype(float).ravel(), b.astype(float).ravel()
    def stats(z):
        return {"min": float(z.min()), "max": float(z.max()), "mean": float(z.mean()),
                "p01_p50_p99": np.percentile(z, [1, 50, 99]).tolist()}
    return {"voxels": len(x), "shape_zyx": list(a.shape),
            "repository_dtype": str(a.dtype), "capsule_dtype": str(b.dtype),
            "fraction_equal": float(np.mean(x == y)),
            "mean_absolute_difference": float(np.mean(np.abs(x - y))),
            "rmse": float(np.sqrt(np.mean((x - y) ** 2))),
            "pearson": float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else None,
            "repository_intensity": stats(x), "capsule_intensity": stats(y)}


def save_preview(path, a, b, spacing, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    low, high = np.percentile(np.concatenate([a.ravel(), b.ravel()]), [1, 99.5])
    fig, axes = plt.subplots(2, 3, figsize=(10, 6), layout="constrained")
    for row, (label, volume) in enumerate((("Repository", a), ("Capsule", b))):
        for col, (plane, keep) in enumerate((("XY", (1, 2)), ("XZ", (0, 2)), ("YZ", (0, 1)))):
            projection = volume.max(axis=col)
            extent = [0, projection.shape[1] * spacing[keep[1]],
                      projection.shape[0] * spacing[keep[0]], 0]
            axes[row, col].imshow(projection, cmap="gray", vmin=low, vmax=max(high, low + 1),
                                  extent=extent)
            axes[row, col].set_title(f"{label} | {plane}")
            axes[row, col].set_xlabel("um")
            axes[row, col].set_ylabel("um")
    fig.suptitle(f"{title}\nSame intensity limits in both rows: {low:.1f} to {max(high, low + 1):.1f}")
    fig.savefig(path, dpi=130)
    plt.close(fig)


def markdown_report(report):
    gt, images = report["gt"], report["images"]
    lines = ["# 794495 source comparison", "", f"UTC: {report['created_utc']}; host: {report['host']}.", "",
             "Small source check only; no model scoring, segmentation comparison or historical cache audit.", "",
             f"- Repository GT: `{gt['repository_uri']}`",
             f"- Capsule GT: `{gt['capsule_uri']}`",
             f"- Repository image: `{images['repository_uri']}`",
             f"- Capsule image: `{images['capsule_uri']}`", "",
             f"GT inventory: {gt['repository_file_count']} repository / {gt['capsule_file_count']} capsule; "
             f"{len(gt['common_files'])} common filenames.",
             f"Repository-only files: {gt['repository_only']}; capsule-only files: {gt['capsule_only']}.", "",
             "SWC xyz is compared by node ID after voxel * [0.748, 0.748, 1.0] -> um.", "",
             "| Neuron | Repo / capsule / common nodes | Max xyz error (um) | Parent mismatches |",
             "|---|---:|---:|---:|"]
    for row in gt["samples"]:
        lines.append(f"| {row['file']} | {row['repository_nodes']} / {row['capsule_nodes']} / "
                     f"{row['common_node_ids']} | {row['xyz_error_um']['max']:.6g} | {row['parent_id_mismatches']} |")
    lines += ["", f"Matching image grids at level {images['level']}: {images['same_grid']}.",
              "Metadata hashes describe headers, not pixel contents.", "",
              "| Patch | Exact-equal fraction | Pearson | Mean absolute difference | RMSE |",
              "|---|---:|---:|---:|---:|"]
    for row in images["samples"]:
        corr = "undefined (constant patch)" if row["pearson"] is None else f"{row['pearson']:.6g}"
        lines.append(f"| {row['file']} | {row['fraction_equal']:.6g} | {corr} | "
                     f"{row['mean_absolute_difference']:.6g} | {row['rmse']:.6g} |")
    lines += ["", "Image centers are seeded, uniformly selected nodes from the sampled repository GT neurons,",
              "restricted to full patches inside the image domain. This is a neurite-centered sample,",
              "not a whole-volume random sample or proof of global equality. No registration is fitted.", ""]
    for error in report["errors"]:
        lines += [f"- Incomplete check: {error}"]
    for row in images["samples"]:
        lines += ["", f"![{row['file']}]({row['preview']})"]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=3, help="Number of GT neurons and image patch pairs")
    parser.add_argument("--radius-um", type=float, default=24, help="Image crop half-width, in um")
    parser.add_argument("--level", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=ROOT / "evaluation/results/794495_source_comparison")
    args = parser.parse_args()
    if args.samples < 1 or not 4 <= args.radius_um <= 128 or args.level < 0:
        parser.error("samples must be positive, radius must be 4..128 um, and level must be nonnegative")
    set_cloud_environment()
    from google.cloud import storage

    args.out.mkdir(parents=True, exist_ok=True)
    capsule_gt, capsule_image = capsule_constants()
    repo_gt = GT_PATH_TEMPLATE.format(brain=BRAIN)
    repo_image = get_reviewed_img_path(BRAIN)
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "host": socket.gethostname(),
              "brain": BRAIN, "seed": args.seed, "requested_samples": args.samples,
              "gt": {"repository_uri": repo_gt, "capsule_uri": capsule_gt,
                     "voxel_to_um_xyz": GT_SCALE_XYZ_UM.tolist(), "samples": []},
              "images": {"repository_uri": repo_image, "capsule_uri": capsule_image,
                         "level": args.level, "radius_um": args.radius_um, "same_grid": False, "samples": []},
              "errors": []}
    client = storage.Client()
    repo_files, capsule_files = list_swcs(client, repo_gt), list_swcs(client, capsule_gt)
    common = sorted(set(repo_files) & set(capsule_files))
    report["gt"].update(repository_file_count=len(repo_files), capsule_file_count=len(capsule_files),
                        common_files=common, repository_only=sorted(set(repo_files) - set(capsule_files)),
                        capsule_only=sorted(set(capsule_files) - set(repo_files)))
    if not common:
        raise ValueError("No common SWC filenames; no correspondence will be guessed")
    names = [common[i] for i in np.linspace(0, len(common) - 1, min(args.samples, len(common)), dtype=int)]
    print(f"Host {report['host']}; GT inventory {len(repo_files)} / {len(capsule_files)}; sampling {names}", flush=True)
    sources = None
    try:
        sources = [ImageSource(repo_image), ImageSource(capsule_image)]
        geometries = [source.level(args.level) for source in sources]
        report["images"].update(same_grid=same_grid(*geometries),
                                metadata=[{"identity": s.identity, "geometry": g.summary()}
                                          for s, g in zip(sources, geometries)])
        if not report["images"]["same_grid"]:
            raise ValueError("Image grids differ; pixel comparison skipped (no implicit resampling)")
    except Exception as exc:
        sources = None
        report["errors"].append(f"Image metadata: {type(exc).__name__}: {exc}")
    rng = np.random.default_rng(args.seed)
    for index, name in enumerate(names):
        try:
            voxel, provenance_a = load_swc(repo_files[name])
            world, provenance_b = load_swc(capsule_files[name])
            comparison = compare_swc(voxel, world)
            report["gt"]["samples"].append({"file": name, **comparison,
                                             "sources": [provenance_a, provenance_b]})
            print(f"GT {name}: max error {comparison['xyz_error_um']['max']:.6g} um", flush=True)
        except Exception as exc:
            report["errors"].append(f"GT {name}: {type(exc).__name__}: {exc}")
            continue
        if sources is None:
            continue
        try:
            xyz = voxel[:, 2:5] * GT_SCALE_XYZ_UM
            geometry = geometries[0]
            coordinates = geometry.graph_to_voxel(xyz)
            margin = args.radius_um / geometry.scale_xyz_um[::-1] + 2
            valid = np.flatnonzero(np.all((coordinates >= margin)
                                         & (coordinates < geometry.shape_zyx - margin), axis=1))
            if not len(valid):
                raise ValueError("No sampled GT nodes allow a full interior patch")
            node = int(rng.choice(valid))
            patches = [source.read(xyz[[node]], radius_um=args.radius_um, level=args.level)
                       for source in sources]
            if any(p[1]["valid_fraction"] != 1 for p in patches):
                raise ValueError("Partial patch; padding must not enter comparison statistics")
            if patches[0][1]["origin_zyx"] != patches[1][1]["origin_zyx"]:
                raise ValueError("Patch origins differ")
            a, b = [p[0]["image_zyx"] for p in patches]
            stats = image_statistics(a, b)
            preview = f"patch_{index + 1:02d}.png"
            save_preview(args.out / preview, a, b, geometry.scale_xyz_um[::-1], name)
            report["images"]["samples"].append({"file": name, "node_id": int(voxel[node, 0]),
                                                 "center_xyz_um": xyz[node].tolist(), "preview": preview,
                                                 "origin_zyx": patches[0][1]["origin_zyx"], **stats})
            print(f"Image {name}: equal={stats['fraction_equal']:.6g}, Pearson={stats['pearson']}", flush=True)
        except Exception as exc:
            report["errors"].append(f"Image {name}: {type(exc).__name__}: {exc}")
    (args.out / "comparison.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (args.out / "report.md").write_text(markdown_report(report))
    print(f"Report: {args.out / 'report.md'}", flush=True)
    for error in report["errors"]:
        print(error, flush=True)
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
