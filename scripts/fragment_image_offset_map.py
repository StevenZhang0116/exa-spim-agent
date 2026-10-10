"""Per-site fragment-to-image offset over a whole cached band, with spatial coherence.

For every cached site of one brain/kind, the best integer-voxel shift (within
+-max_shift um) of the fragment nodes onto the normalized image patch is found by FFT
cross-correlation. Sites are joined with candidate coordinates from the native table
(merge rows carry x_um/y_um/z_um). If the segmentation and the image are only
locally misregistered (e.g. different stitching), nearby sites share their best
shift; blur or noise gives incoherent shifts. Reports, for confident sites, the mean
|shift difference| between near pairs versus random pairs, and a map figure.

Usage (panda, compute node):
    python scripts/fragment_image_offset_map.py --brains 747807 789202 --kind merge
"""
import argparse
import glob
import json
import pickle
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
CACHE = REPO / "proofreader_evolve" / "context_cache"
TABLES = REPO / "proofreader_evolve" / "feature_tables"


def candidates_for(brain, pool_sha256):
    for meta_path in glob.glob(str(TABLES / brain / "*" / "meta.json")):
        if Path(meta_path).parent.name.endswith(".availability"):
            continue
        if json.loads(Path(meta_path).read_text())["pool_sha256"] == pool_sha256:
            with open(Path(meta_path).parent / "candidates.pkl", "rb") as stream:
                return pickle.load(stream)
    raise FileNotFoundError(f"No feature table for {brain} with pool {pool_sha256[:12]}")


def best_shift(norm, nodes, window):
    """Best (dz, dy, dx) voxel shift maximizing the summed image at shifted nodes."""
    mask = np.zeros(norm.shape)
    idx = np.round(nodes).astype(int)
    np.add.at(mask, tuple(idx.T), 1.0)
    # corr[s] = sum_v mask[v] * norm[v + s]
    corr = np.real(np.fft.ifftn(np.fft.fftn(norm) * np.conj(np.fft.fftn(mask))))
    shifts = np.stack(np.meshgrid(*[np.arange(-w, w + 1) for w in window], indexing="ij"), -1).reshape(-1, 3)
    values = corr[tuple((shifts % np.asarray(norm.shape)).T)] / len(idx)
    k = int(np.argmax(values))
    zero = values[np.flatnonzero((shifts == 0).all(1))[0]]
    return shifts[k], float(values[k] - zero), float(zero)


def analyze(brain, kind, tier, max_shift, limit):
    entry = Path(glob.glob(str(CACHE / brain / kind / "*" / "manifest.json"))[0]).parent
    meta = json.loads((entry / "manifest.json").read_text())
    spacing = np.asarray(meta["image_tiers"][tier]["source"]["geometry"]["spacing_xyz_um"], float)[::-1]
    window = np.floor(max_shift / spacing).astype(int)
    rows_xyz = candidates_for(brain, entry.name)
    out = []
    for geometry_file in sorted((entry / "geometry").glob("chunk_*.npz")):
        image_file = entry / "images" / tier / geometry_file.name
        if not image_file.exists():
            break
        g, im = np.load(geometry_file), np.load(image_file)
        for i in range(len(im["rows"])):
            if not im["available"][i]:
                continue
            image, valid = im[f"image_{i}"].astype(float), im[f"valid_{i}"].astype(bool)
            if valid.sum() < 100:
                continue
            low, high = np.percentile(image[valid], [50, 99.5])
            norm = np.clip((image - low) / max(high - low, 1e-6), -1, 3)
            a, b = g["node_offsets"][i], g["node_offsets"][i + 1]
            xyz = g["xyz_um"][a:b][~g["outside_radius"][a:b]]
            nodes = np.nanmean(im["anchors_zyx"][i], axis=0) + xyz[:, ::-1] / spacing
            keep = np.all((nodes >= window) & (nodes <= np.asarray(image.shape) - 1 - window), axis=1)
            if keep.sum() < 5:
                continue
            shift, gain, zero = best_shift(norm, nodes[keep], window)
            row = int(im["rows"][i])
            c = rows_xyz[row]
            out.append({"row": row, "x_um": c.get("x_um"), "y_um": c.get("y_um"), "z_um": c.get("z_um"),
                        "dz_um": shift[0] * spacing[0], "dy_um": shift[1] * spacing[1], "dx_um": shift[2] * spacing[2],
                        "gain": gain, "score_zero": zero})
            if len(out) >= limit:
                return out, spacing
    return out, spacing


def coherence(records, near_um, seed=0):
    from scipy.spatial import cKDTree
    xyz = np.array([[r["x_um"], r["y_um"], r["z_um"]] for r in records], float)
    shift = np.array([[r["dz_um"], r["dy_um"], r["dx_um"]] for r in records], float)
    pairs = np.array(sorted(cKDTree(xyz).query_pairs(near_um)))
    if not len(pairs):
        return None
    near = np.linalg.norm(shift[pairs[:, 0]] - shift[pairs[:, 1]], axis=1).mean()
    rng = np.random.default_rng(seed)
    a, b = rng.integers(0, len(xyz), (2, 20000))
    random = np.linalg.norm(shift[a] - shift[b], axis=1)[a != b].mean()
    return {"near_pairs": int(len(pairs)), "near_mean_shift_diff_um": round(float(near), 3),
            "random_mean_shift_diff_um": round(float(random), 3)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--brains", nargs="+", required=True)
    p.add_argument("--kind", default="merge", choices=["merge"])
    p.add_argument("--tier", default="level0")
    p.add_argument("--max-shift", type=float, default=6.0)
    p.add_argument("--limit", type=int, default=50000)
    p.add_argument("--near-um", type=float, default=200.0)
    p.add_argument("--out-dir", type=Path, default=REPO / "notebooks" / "log" / "fragment_image_offset")
    a = p.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    for brain in a.brains:
        records, spacing = analyze(brain, a.kind, a.tier, a.max_shift, a.limit)
        (a.out_dir / f"{brain}_{a.kind}_{a.tier}.json").write_text(json.dumps(records))
        gains = np.array([r["gain"] for r in records])
        confident = [r for r, gain in zip(records, gains) if gain >= np.quantile(gains, .5)]
        shifts = np.array([[r["dz_um"], r["dy_um"], r["dx_um"]] for r in records])
        summary = {"brain": brain, "kind": a.kind, "tier": a.tier, "sites": len(records),
                   "median_shift_zyx_um": np.median(shifts, axis=0).round(2).tolist(),
                   "share_best_at_zero_within_1.5um": round(float(np.mean(np.linalg.norm(shifts, axis=1) <= 1.5)), 3),
                   "share_best_beyond_3um": round(float(np.mean(np.linalg.norm(shifts, axis=1) > 3)), 3),
                   "median_score_zero": round(float(np.median([r["score_zero"] for r in records])), 3),
                   "coherence_all": coherence(records, a.near_um),
                   "coherence_confident_half": coherence(confident, a.near_um)}
        print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
