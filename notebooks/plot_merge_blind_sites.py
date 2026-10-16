"""Local views of blind GT merge sites: no junction and no foreign tip within reach.

A blind site is a census bridge site (``merge_diagnosis/mcl100/followup/bridge_sites.csv``: no
junction within 30 um of cable on its segment) whose nearest junction or tip-to-shaft candidate
(family J+S20 of ``merge_shaft_candidate_reach.py``) is more than 20 um of cable away, or not
within 150 um at all. Pooled over the eleven brains these are the 199 bridge sites that neither
candidate family reaches. The per-site distances come from the reach JSONs
(``notebooks/merge_shaft_reach/<brain>_mcl<N>.json``, ``site_distances_um``, written since
2026-10-16); the blind count of every brain is checked against the JSON's ``bridge`` record.

Per brain, K blind positions (exact XYZ deduplicated) are sampled with --seed. Each figure has
Image MIP / GT / Fragments rows and XY / XZ / YZ columns
(``scripts/plot_top_merge_predictions.render_context``; the site's segment is cyan). The
fragments row also marks the two candidate sources: junctions (degree >= 3) of the site's own
segment and tips of other segments in the field of view, plus a 20 um circle around the site.
Fragments shorter than the cache's MCL are not in the graph and cannot be shown.

Default brains: the six older-microscope brains added 2026-10-06 (754613 has no bridge site).
Run with panda on a compute node; caches are loaded one at a time (> 20 GB each):
    python -u notebooks/plot_merge_blind_sites.py                    # K = 1 per brain
    python -u notebooks/plot_merge_blind_sites.py --brains 747807 --k 3 --seed 1
    python -u notebooks/plot_merge_blind_sites.py --dry-run          # selection only, no cache load
Output: notebooks/merge_blind_sites/<brain>_site<i>_segment<s>.svg and <brain>_sites.csv
(existing files of the same name are replaced).
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Circle

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "notebooks"))

from plot_gt_merge_sites import open_image  # noqa: E402
from scripts.plot_top_merge_predictions import JUNCTION_COLOR, VIEWS, render_context  # noqa: E402

NEW_BRAINS = ["747807", "750318", "751473", "754610", "754612", "754613"]
BRIDGE_SITES = ROOT / "notebooks" / "merge_diagnosis" / "mcl100" / "followup" / "bridge_sites.csv"
REACH_DIR = ROOT / "notebooks" / "merge_shaft_reach"
OUT_DIR = ROOT / "notebooks" / "merge_blind_sites"
FAMILY = "J+S20"
POSITIVE_UM = 20.0
MARKER_COLOR = "white"


def site_class(row) -> str:
    """Bridge subclass, as in ``merge_shaft_candidate_reach.measure_brain``."""
    if int(row["segment_n_junction_nodes"]) == 0:
        return "pure_path"
    return "junction_30_150" if np.isfinite(row["junction_um_extended"]) else "junction_beyond_150"


def blind_sites(brain: str, mcl: int, reach_dir: Path, bridge_csv: Path) -> pd.DataFrame:
    """Bridge sites of one brain whose nearest J+S20 candidate is beyond 20 um (or none within 150)."""
    table = pd.read_csv(bridge_csv, dtype={"brain": str})
    table = table[table["brain"] == brain].copy()
    table["site_class"] = [site_class(row) for _, row in table.iterrows()]
    if table.empty:                       # no bridge site in the census: nothing to look up
        return table.assign(nearest_candidate_um=pd.Series(dtype=float))
    path = reach_dir / f"{brain}_mcl{mcl}.json"
    if not path.exists():
        raise SystemExit(f"{path} missing; run: python -u notebooks/merge_shaft_candidate_reach.py --brains {brain}")
    result = json.loads(path.read_text())
    distances = result.get("site_distances_um", {}).get("cache", {}).get(FAMILY)
    if distances is None:
        raise SystemExit(f"{path} has no per-site distances (written since 2026-10-16); rerun: "
                         f"python -u notebooks/merge_shaft_candidate_reach.py --brains {brain}")
    if int(table["site_index"].max()) >= len(distances):
        raise SystemExit(f"{bridge_csv} has site indices beyond the {len(distances)} cache sites of {brain}")
    table["nearest_candidate_um"] = [math.nan if distances[i] is None else float(distances[i])
                                     for i in table["site_index"].astype(int)]
    blind = table[~(table["nearest_candidate_um"] <= POSITIVE_UM)]
    expected = sum(r["n"] - r[f"covered_20_{FAMILY}"] for r in result["bridge"].values())
    if len(blind) != expected:
        raise SystemExit(f"{brain}: {len(blind)} blind sites from the distances, {expected} in the JSON's bridge record")
    return blind.sort_values("site_index").reset_index(drop=True)


def select_positions(blind: pd.DataFrame, sites: list, k: int, seed: int) -> tuple[list[dict], int]:
    """Deduplicate exact XYZ (first site index wins), then sample up to K positions."""
    positions = {}
    for _, row in blind.iterrows():
        index = int(row["site_index"])
        site = sites[index]
        if int(site["segment_id"]) != int(row["segment_id"]):
            raise SystemExit(f"site {index}: cache segment {site['segment_id']} differs from the census "
                             f"segment {row['segment_id']}; the census predates this cache")
        key = tuple(map(float, site["xyz"]))
        positions.setdefault(key, {**row.to_dict(), "xyz": list(key)})
    chosen = random.Random(seed).sample(list(positions.values()), min(k, len(positions)))
    return chosen, len(positions)


def fov_markers(fragment, segment: int, origin, shape, anisotropy) -> tuple[np.ndarray, np.ndarray]:
    """Local XYZ (um, figure frame) of own-segment junctions and other-segment tips in the patch."""
    xyz = np.asarray(fragment.node_xyz, dtype=float)
    low = (np.asarray(origin, float)[::-1] - 0.5) * anisotropy
    high = (np.asarray(origin, float)[::-1] + np.asarray(shape, float)[::-1] - 0.5) * anisotropy
    nodes = np.flatnonzero(np.all((xyz >= low) & (xyz <= high), axis=1))
    components = np.asarray(fragment.node_component_id)
    swc = fragment.component_id_to_swc_id
    junctions, tips = [], []
    for node in nodes:
        degree = fragment.degree(int(node))
        own = int(str(swc[int(components[node])]).split(".")[0]) == segment
        if degree >= 3 and own:
            junctions.append(node)
        elif degree == 1 and not own:
            tips.append(node)
    offset = np.asarray(origin, float)[::-1] * anisotropy
    return xyz[junctions].reshape(-1, 3) - offset, xyz[tips].reshape(-1, 3) - offset


def distance_text(value: float, none: str) -> str:
    return none if not np.isfinite(value) else f"{value:.0f} µm"


def render_blind_site(payload, image, brain: str, site: dict, patch_um):
    anisotropy = np.asarray(payload["anisotropy"], dtype=float)
    segment = int(site["segment_id"])
    figure, origin, shape = render_context(payload, image, site["xyz"], segment, patch_um)
    junctions, tips = fov_markers(payload["fragments_graph"], segment, origin, shape, anisotropy)
    anchor = np.asarray(site["xyz"], float) - np.asarray(origin, float)[::-1] * anisotropy
    for axis, (_, _, xyz_axes) in zip(figure.axes[6:9], VIEWS):
        axes = list(xyz_axes)
        axis.add_patch(Circle(anchor[axes], POSITIVE_UM, fill=False, ec=MARKER_COLOR, ls="--", lw=1.2, zorder=6))
        if len(junctions):
            axis.scatter(junctions[:, axes[0]], junctions[:, axes[1]], s=110, facecolors="none",
                         edgecolors=MARKER_COLOR, linewidths=1.8, zorder=7)
        if len(tips):
            axis.scatter(tips[:, axes[0]], tips[:, axes[1]], s=90, marker="^", facecolors="none",
                         edgecolors=MARKER_COLOR, linewidths=1.6, zorder=7)
    junction = ("unbranched segment" if site["site_class"] == "pure_path"
                else f"nearest junction {distance_text(site['junction_um_extended'], '> 150 µm')}")
    figure.suptitle(f"Brain {brain} · blind merge site {int(site['site_index'])} · segment {segment}\n"
                    f"bridge site · {junction} · nearest junction/tip-to-shaft candidate "
                    f"{distance_text(site['nearest_candidate_um'], '> 150 µm')}", fontsize=15)
    handles = [
        Line2D([], [], ls="none", marker="+", ms=12, mew=1.8, color=JUNCTION_COLOR, label="GT merge site"),
        Line2D([], [], ls="none", marker="o", ms=10, mfc="none", mec="black", mew=1.6,
               label="junction of this segment"),
        Line2D([], [], ls="none", marker="^", ms=9, mfc="none", mec="black", mew=1.6, label="tip of another segment"),
        Line2D([], [], ls="--", color="black", lw=1.2, label="20 µm"),
    ]
    figure.legend(handles=handles, loc="outside lower center", ncol=4, frameon=False, fontsize=12)
    return figure, len(junctions), len(tips)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--brains", nargs="+", default=NEW_BRAINS)
    parser.add_argument("--k", "-k", type=int, default=1, help="Blind positions per brain (default 1)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mcl", type=int, default=100)
    parser.add_argument("--patch-um", nargs=3, type=float, default=(100., 100., 100.), metavar=("X", "Y", "Z"))
    parser.add_argument("--reach-dir", type=Path, default=REACH_DIR)
    parser.add_argument("--bridge-sites", type=Path, default=BRIDGE_SITES)
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "cache")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--format", choices=("svg", "png"), default="svg")
    parser.add_argument("--dpi", type=int, default=160, help="PNG resolution")
    parser.add_argument("--dry-run", action="store_true", help="List blind sites per brain; load no cache")
    args = parser.parse_args(argv)
    if args.k < 1:
        parser.error("--k must be positive")

    brains = list(dict.fromkeys(map(str, args.brains)))
    blind = {brain: blind_sites(brain, args.mcl, args.reach_dir, args.bridge_sites) for brain in brains}
    for brain, table in blind.items():
        counts = table["site_class"].value_counts().to_dict()
        print(f"[{brain}] {len(table)} blind sites {counts}", flush=True)
    if args.dry_run:
        return 0

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for brain in brains:
        if blind[brain].empty:
            print(f"[{brain}] no blind site; skipped", flush=True)
            continue
        cache = args.cache_dir / f"dataset_cache_{brain}_mcl{args.mcl}_add.pkl"
        print(f"[{brain}] loading {cache}", flush=True)
        with cache.open("rb") as handle:
            payload = pickle.load(handle)
        selected, available = select_positions(blind[brain], list(payload["gt_merge_sites"]), args.k, args.seed)
        if len(selected) < args.k:
            print(f"[{brain}] only {available} blind positions; plotting all", flush=True)
        image = open_image(payload["img_path"])
        records = []
        for site in selected:
            figure, n_junctions, n_tips = render_blind_site(payload, image, brain, site, args.patch_um)
            name = f"{brain}_site{int(site['site_index'])}_segment{int(site['segment_id'])}.{args.format}"
            try:
                figure.savefig(args.out_dir / name, format=args.format, dpi=args.dpi, bbox_inches="tight",
                               pad_inches=0.15)
            finally:
                figure.clear()
            x, y, z = site["xyz"]
            records.append({"brain": brain, "site_index": int(site["site_index"]),
                            "segment_id": int(site["segment_id"]), "x_um": x, "y_um": y, "z_um": z,
                            "site_class": site["site_class"], "junction_um": site["junction_um_extended"],
                            "own_tip_um": site["endpoint_um"], "nearest_candidate_um": site["nearest_candidate_um"],
                            "junctions_in_view": n_junctions, "foreign_tips_in_view": n_tips,
                            "blind_positions": available, "seed": args.seed, "figure": name})
            print(f"[{brain}] {args.out_dir / name}", flush=True)
        pd.DataFrame(records).to_csv(args.out_dir / f"{brain}_sites.csv", index=False)
        del payload, image
        gc.collect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
