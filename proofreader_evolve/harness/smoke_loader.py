"""
Smoke-test the SWC data-loading path WITHOUT the ~30-min full build.

run_evolution.py's slow "Read SWCs" bar reads ~10k fragment SWC archives from
GCS. Before paying that, this script checks the loader works in escalating tiers
so you find a broken auth / path / fork-stall in seconds, not minutes:

  tier 1  auth + listing      : can we authenticate and list the cloud paths?
  tier 2  read ONE archive    : does a single read+parse work (serial, no pool)?
  tier 3  read N archives      : does the parallel reader work on a small subset?

Usage:
    python proofreader_evolve/harness/smoke_loader.py --brain 789202
    python proofreader_evolve/harness/smoke_loader.py --brain 789202 --tier 2
    python proofreader_evolve/harness/smoke_loader.py --brain 789202 --n 8

Each tier prints timing so you can extrapolate the full read. It NEVER reads the
whole set and NEVER builds graphs — it only proves the I/O layer is healthy.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from proofreader_evolve.harness import scoring
from segmentation_skeleton_metrics.data_handling import swc_loading
from segmentation_skeleton_metrics.utils import util


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--brain", default="789202")
    p.add_argument("--tier", type=int, default=3, choices=[1, 2, 3],
                   help="highest tier to run (1=list, 2=one read, 3=small parallel)")
    p.add_argument("--n", type=int, default=8,
                   help="how many archives to read in tier 3")
    args = p.parse_args()

    paths = scoring.BrainPaths(args.brain)
    src = paths.fragments_path
    print(f"brain={args.brain}")
    print(f"fragments_path={src}")
    print(f"(creds: {os.environ.get('GOOGLE_APPLICATION_CREDENTIALS', '<unset>')})\n")

    # --- tier 1: auth + listing -------------------------------------------------
    t = time.monotonic()
    swc_paths = util.list_cloud_paths(src, ".swc")
    zip_paths = util.list_cloud_paths(src, ".zip")
    dt = time.monotonic() - t
    kind = "individual .swc" if swc_paths else "zip archives"
    items = swc_paths or zip_paths
    print(f"[tier 1] listing OK in {dt:.1f}s — {len(items)} {kind} found")
    if not items:
        print("  FAIL: nothing listed — bad path or no read permission.")
        return 1
    print(f"  first few: {items[:3]}")
    if args.tier < 2:
        print("\nTier-1 only. Auth + listing work.")
        return 0

    # --- tier 2: read exactly ONE archive, serially (no pool, no fork) ----------
    reader = swc_loading.Reader()
    use_s3 = util.is_s3_path(src)
    one = items[0]
    t = time.monotonic()
    if swc_paths:
        out = reader.read_swc(one)
        n = 1 if out else 0
    else:
        read_fn = reader.read_s3_zip if use_s3 else reader.read_gcs_zip
        out = read_fn(one)
        n = len(out)
    dt = time.monotonic() - t
    print(f"\n[tier 2] read 1 {kind} in {dt:.1f}s -> {n} SWC dict(s); serial read OK")
    if args.tier < 3:
        print("\nTier-2 only. A single read+parse works.")
        return 0

    # --- tier 3: read a small SUBSET through the real parallel reader -----------
    # Exercises the ProcessPoolExecutor path (read_zips) that the full build uses,
    # but on only --n items, so a fork-stall or pool error shows up immediately.
    subset = items[: args.n]
    t = time.monotonic()
    if swc_paths:
        dicts = reader.read_swcs(subset, reader.read_swc)
    else:
        read_fn = reader.read_gcs_zip if not use_s3 else reader.read_s3_zip
        dicts = reader.read_zips(subset, read_fn)
    dt = time.monotonic() - t
    per = dt / max(len(subset), 1)
    total_est = per * len(items)
    print(f"\n[tier 3] read {len(subset)} {kind} in {dt:.1f}s "
          f"({per:.2f}s/item) -> {len(dicts)} SWC dicts")
    print(f"  parallel reader OK. Extrapolated full read of {len(items)} items: "
          f"~{total_est/60:.1f} min (one-time; then cached to prepared_*.pkl).")
    print("\nAll requested tiers passed — the data-loading path is healthy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
