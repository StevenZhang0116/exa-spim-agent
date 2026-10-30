"""Measure fragment-to-image registration from on-demand, reviewed image crops.

Sites are sampled uniformly from the complete native candidate pool without labels
or detector-rank filtering. Fragment nodes use the same registered patch coordinates
as evolution descriptors and are shifted over a grid of
(dz, dy, dx) offsets. Per shift, the score is the mean normalized intensity at the
nodes; a registered brain peaks at (0, 0, 0). Only nodes that stay inside the patch
for every shift are used, so the score is comparable across shifts.

Usage (panda, compute node):
    python scripts/check_fragment_image_offset.py --brains 789202 794493 747807 --tier level0 --sites 600
"""
import argparse
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
from scipy.ndimage import map_coordinates

REPO = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(REPO))

from proofreader_evolve.cli import precompute_error_scores as pc
from proofreader_evolve.harness.descriptor_context import DescriptorContexts
from proofreader_evolve.harness.descriptor_inputs import iter_inputs
from proofreader_evolve.harness.image_context import attach_image_context, DEFAULT_ALIGNMENT
from proofreader_evolve.harness.local_context import attach_local_context
from proofreader_evolve.harness.native_pool import cache_path, ensure_native_tables


def iter_sites(table, tier, limit, root, seed=0):
    """Stream temporary descriptor inputs; no retained images or label-based selection."""
    if limit < 1:
        raise ValueError('The site limit must be positive')
    spec = {'inputs': 'both', 'image_tier': tier}
    entry = DescriptorContexts().entry(table)
    rows = np.sort(np.random.default_rng(seed).choice(entry.rows, min(limit, len(entry.rows)), replace=False))
    for row in rows:
        _, context, patch = next(iter_inputs(table, entry, spec, [row], root))
        fragment = context['fragment']
        inside = ~np.asarray(fragment['outside_radius'], dtype=bool)
        yield {'row': int(row), 'image': patch['image_zyx'].astype(float),
               'valid': patch['valid_zyx'].astype(bool),
               'nodes_zyx': np.asarray(fragment['nodes_zyx'])[inside],
               'spacing': patch['spacing_zyx_um'], 'candidate': table.candidates[int(row)]}


def sites(brain, kind, tier, limit, seed=0):
    """Load one brain through the same host providers used by evolution."""
    selected = pc.resolve_detector_runs(None, None)
    bank = ensure_native_tables(str(brain), cache_path(str(brain), 100), selected,
                                pc.DEFAULT_OUT, prepare=False, mcl=100)
    with tempfile.TemporaryDirectory(prefix='fragment_image_offset_') as tmp:
        root = Path(tmp)
        banks = {str(brain): bank}
        attach_local_context(banks, root / 'geometry')
        attach_image_context(banks, root / 'images', DEFAULT_ALIGNMENT)
        yield from iter_sites(bank.tables[kind], tier, limit, root, seed)


def measure(brain, kind, tier, limit, max_shift, seed=0):
    shifts_um = np.arange(-max_shift, max_shift + 1, 1.0)
    grid = np.stack(np.meshgrid(shifts_um, shifts_um, shifts_um, indexing="ij"), -1).reshape(-1, 3)
    totals, used, per_site_best, records = np.zeros(len(grid)), 0, [], []
    zero_index = np.flatnonzero((grid == 0).all(1))[0]
    for site in sites(brain, kind, tier, limit, seed):
        row, image, valid, nodes, spacing = (site[k] for k in ('row', 'image', 'valid', 'nodes_zyx', 'spacing'))
        finite = image[valid]
        if finite.size < 100:
            continue
        low, high = np.percentile(finite, [50, 99.5])
        norm = (image - low) / max(high - low, 1e-6)
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
    if not used:
        raise ValueError(f'{brain}/{kind}: no sampled sites have enough valid pixels and interior nodes')
    mean = totals / used
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
    p.add_argument("--tier", choices=("level0", "level1"), default="level0")
    p.add_argument("--sites", type=int, default=600, help="Maximum uniformly sampled candidates per brain/kind")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-shift", type=float, default=6.0)
    p.add_argument("--records-dir", type=Path, help="Write per-site best shifts here (one JSON per brain/kind)")
    a = p.parse_args()
    for brain in a.brains:
        for kind in a.kinds:
            result = measure(brain, kind, a.tier, a.sites, a.max_shift, a.seed)
            if a.records_dir:
                a.records_dir.mkdir(parents=True, exist_ok=True)
                (a.records_dir / f"{brain}_{kind}_{a.tier}.json").write_text(json.dumps(result))
            result.pop("records")
            print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
