"""Measure fragment-to-image registration from the context cache (no S3, no labels).

For each sampled cached site, fragment nodes (graph microns relative to the anchor
midpoint) are mapped into the cached image patch and shifted over a grid of
(dz, dy, dx) offsets. Per shift, the score is the mean normalized intensity at the
nodes; a registered brain peaks at (0, 0, 0). Only nodes that stay inside the patch
for every shift are used, so the score is comparable across shifts.

Usage (panda, compute node):
    python scripts/check_fragment_image_offset.py --brains 789202 794493 747807 --tier level0 --sites 600
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates

ROOT = Path(__file__).resolve().parents[1] / "proofreader_evolve" / "context_cache"


def sites(brain, kind, tier, limit):
    entry = Path(glob.glob(str(ROOT / brain / kind / "*" / "manifest.json"))[0]).parent
    meta = json.loads((entry / "manifest.json").read_text())
    spacing_zyx = np.asarray(meta["image_tiers"][tier]["source"]["geometry"]["spacing_xyz_um"], float)[::-1]
    count = 0
    for geometry_file in sorted((entry / "geometry").glob("chunk_*.npz")):
        image_file = entry / "images" / tier / geometry_file.name
        if not image_file.exists():
            break
        g, im = np.load(geometry_file), np.load(image_file)
        if not np.array_equal(g["rows"][:len(im["rows"])], im["rows"]):
            raise ValueError(f"{brain}/{kind}: geometry and image chunk rows differ")
        for i in range(len(im["rows"])):
            if not im["available"][i]:
                continue
            a, b = g["node_offsets"][i], g["node_offsets"][i + 1]
            inside = ~g["outside_radius"][a:b]
            yield (int(im["rows"][i]), im[f"image_{i}"].astype(float), im[f"valid_{i}"].astype(bool),
                   im["anchors_zyx"][i], g["xyz_um"][a:b][inside], spacing_zyx)
            count += 1
            if count >= limit:
                return


def measure(brain, kind, tier, limit, max_shift):
    shifts_um = np.arange(-max_shift, max_shift + 1, 1.0)
    grid = np.stack(np.meshgrid(shifts_um, shifts_um, shifts_um, indexing="ij"), -1).reshape(-1, 3)
    totals, used, per_site_best, records = np.zeros(len(grid)), 0, [], []
    zero_index = np.flatnonzero((grid == 0).all(1))[0]
    for row, image, valid, anchors, xyz, spacing in sites(brain, kind, tier, limit):
        finite = image[valid]
        if finite.size < 100:
            continue
        low, high = np.percentile(finite, [50, 99.5])
        norm = (image - low) / max(high - low, 1e-6)
        mid = np.nanmean(anchors, axis=0)
        nodes = mid + xyz[:, ::-1] / spacing
        margin = max_shift / spacing
        keep = np.all((nodes - margin >= 0) & (nodes + margin <= np.asarray(image.shape) - 1), axis=1)
        nodes = nodes[keep]
        if len(nodes) < 5:
            continue
        scores = np.empty(len(grid))
        for k, shift in enumerate(grid):
            coords = (nodes + shift / spacing).T
            scores[k] = map_coordinates(norm, coords, order=1).mean()
        totals += scores
        per_site_best.append(grid[np.argmax(scores)])
        records.append({"row": row, "best_zyx_um": grid[np.argmax(scores)].tolist(),
                        "gain": float(scores.max() - scores[zero_index]), "score_zero": float(scores[zero_index])})
        used += 1
    mean = totals / max(used, 1)
    best = grid[np.argmax(mean)]
    zero = mean[np.flatnonzero((grid == 0).all(1))[0]]
    per_site_best = np.asarray(per_site_best)
    return {"brain": brain, "kind": kind, "sites": used, "best_shift_zyx_um": best.tolist(),
            "score_at_zero": round(float(zero), 4), "score_at_best": round(float(mean.max()), 4),
            "median_site_best_zyx_um": np.median(per_site_best, axis=0).tolist() if used else None,
            "share_sites_best_within_2um": float(np.mean(np.abs(per_site_best).max(1) <= 2)) if used else None,
            "records": records}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--brains", nargs="+", required=True)
    p.add_argument("--kinds", nargs="+", default=["merge", "split"])
    p.add_argument("--tier", default="level0")
    p.add_argument("--sites", type=int, default=600)
    p.add_argument("--max-shift", type=float, default=6.0)
    p.add_argument("--records-dir", type=Path, help="Write per-site best shifts here (one JSON per brain/kind)")
    a = p.parse_args()
    for brain in a.brains:
        for kind in a.kinds:
            result = measure(brain, kind, a.tier, a.sites, a.max_shift)
            if a.records_dir:
                a.records_dir.mkdir(parents=True, exist_ok=True)
                (a.records_dir / f"{brain}_{kind}_{a.tier}.json").write_text(json.dumps(result))
            result.pop("records")
            print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
