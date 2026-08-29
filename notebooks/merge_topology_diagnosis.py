#!/usr/bin/env python
"""
Merge-topology census: which local topology do GT merge errors leave in the
fragments graph, and what fraction can each detector topology reach?

For every ``dataset_cache_*_mcl<MCL>_add.pkl`` under ``cache/`` this probe:

  1. reads the baked-in answer key (``gt_merge_sites`` xyz + ``gt_merge_labels``),
  2. SNAPS each site to the nearest on-segment skeleton node (recording the
     Euclidean ``snap_um`` as a coordinate-quality audit column), then walks the
     skeleton GEODESICALLY (bounded Dijkstra over cable length) to the nearest
     degree>=3 node, recording the CONTINUOUS ``geodesic_um_to_junction``,
  3. derives the detector-topology class from that continuous column at the
     walk bound (any other threshold is just a point on its CDF):
       * ``branch``     -- a junction is geodesically reachable within the bound;
       * ``bridge``     -- no junction within the bound, single-component label;
       * ``multi_component`` -- no junction within the bound and the label spans
                            >=2 disconnected components;
       * ``no_fragment_nearby`` -- no on-segment node within the snap radius;
  4. samples CONTROL nodes on non-merge segments and walks the same geodesic
     measurement, so the distance-to-junction CDF at merge sites can be read
     against the skeleton's own baseline junction density,
  5. writes per-brain CSVs, per-brain and pooled figures, and a summary table.

Why geodesic instead of a Euclidean tolerance ball: in dense neuropil a ball
around a site captures junctions from spatially passing, topologically distant
parts of the same arbor, inflating ``branch`` for reasons unrelated to the
site's local topology; and a discrete tolerance grid is a coarse sampling of
the distance CDF this script now records directly. The CDF's SHAPE separates
the two causes a tolerance sweep conflates: a smooth rise = sites displaced
from their junctions (the known canonical branch-snap residual); a plateau
below 1.0 = merges that genuinely have no junction signature (true bridges).

This is a COVERAGE diagnosis (which enumeration topology bounds recall), not a
precision feature: skeletonizing a merged volume *creates* the junction node, so
"merges sit near junctions" is partly definitional. It tells a detector where
to look, not what to accept.

Caveats baked into the outputs:
  * ``gt_merge_sites`` is itself the geometric walk's output; segments in
    ``gt_merge_labels`` with NO site are reported separately as the walk's
    residual (they are invisible to any site-anchored census).
  * merge counts per brain are small; per-brain counts are printed next to the
    pooled fractions and nothing is significance-tested.
  * ``snap_um`` and the walk bound are still free parameters, but they now
    bound only WHERE the measurement starts/stops -- the classification itself
    is the recorded continuous distance, reported at several derived thresholds.

GT-blindness does not apply here: this is an answer-key characterization probe
(the same footing as verify_add_cache_merge_info.py), not a detector.

Usage (panda env, from the project root; needs >20 GB RAM per mcl10 cache):
    python notebooks/merge_topology_diagnosis.py --mcl 100
    python notebooks/merge_topology_diagnosis.py --mcl 10 --brains 794495
    python notebooks/merge_topology_diagnosis.py --mcl 100 --max-geodesic-um 50
    python notebooks/merge_topology_diagnosis.py --mcl 100 --plot-only

Artifacts land in ``notebooks/merge_diagnosis/mcl<MCL>/``:
    per_brain/<brain>_sites.csv       one row per merge site (continuous columns)
    per_brain/<brain>_controls.csv    control-node geodesic records
    per_brain/<brain>_summary.json    counts, class fractions, residuals
    figures/<brain>_geodesic_census.png per-brain CDF + snap-quality figure
    figures/pooled_topology_census.png pooled cross-brain figure
    summary.csv                       one row per brain x derived threshold
    run_summary.json                  config, inputs, headline numbers
"""

from __future__ import annotations

import argparse
import gc
import heapq
import json
import pickle
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

THIS_FILE = Path(__file__).resolve()
REPO = THIS_FILE.parents[1]  # .../exa-spim-agent
CACHE_DIR = REPO / "cache"
OUTPUT_ROOT = THIS_FILE.parent / "merge_diagnosis"

# Register SkeletonGraph classes before pickle.load.
sys.path.insert(0, str(REPO.parent / "agentic-neuron-proofreader" / "src"))
import agentic_neuron_proofreader  # noqa: F401,E402

TOPOLOGY_CLASSES = (
    "branch", "bridge", "multi_component", "no_fragment_nearby",
)
# Snap radius bounds only how far a site may sit from its own skeleton (the
# canonical branch-snap/dedup residual is well under this); the walk bound is
# where the geodesic search gives up -- both are measurement bounds, not the
# classification threshold (that is the recorded continuous distance).
DEFAULT_SNAP_MAX_UM = 50.0
DEFAULT_MAX_GEODESIC_UM = 30.0
# Branch-fraction points derived from the geodesic CDF for summary.csv.
DERIVED_THRESHOLDS_UM = (5.0, 10.0, 20.0, 30.0)
CONTROLS_PER_SITE = 20
RNG_SEED = 42


# --------------------------------------------------------------------------- #
# Cache discovery and loading
# --------------------------------------------------------------------------- #
def discover_add_caches(cache_dir: Path, mcl: int,
                        brains: list[str] | None) -> list[Path]:
    """One relabeled ``*_add.pkl`` per requested brain, mirroring the sweep."""
    requested = {str(b) for b in brains} if brains else None
    pattern = re.compile(rf"dataset_cache_(.+)_mcl{int(mcl)}_add\.pkl$")
    found = []
    for path in sorted(cache_dir.glob(f"dataset_cache_*_mcl{int(mcl)}_add.pkl")):
        match = pattern.match(path.name)
        if match and (requested is None or match.group(1) in requested):
            found.append(path)
    if requested:
        seen = {pattern.match(p.name).group(1) for p in found}
        missing = requested - seen
        if missing:
            raise FileNotFoundError(
                f"no mcl{mcl} _add cache for brain(s): {sorted(missing)}")
    if not found:
        raise FileNotFoundError(f"no *_mcl{mcl}_add.pkl caches under {cache_dir}")
    return found


def brain_id_from_cache(path: Path) -> str:
    match = re.match(r"dataset_cache_(.+)_mcl\d+_add\.pkl$", path.name)
    if not match:
        raise ValueError(f"unexpected cache name: {path.name}")
    return match.group(1)


def load_payload(path: Path) -> dict:
    started = time.monotonic()
    print(f"loading {path} ({path.stat().st_size / 2**30:.2f} GiB) ...", flush=True)
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    print(f"loaded in {time.monotonic() - started:.1f}s", flush=True)
    return payload


# --------------------------------------------------------------------------- #
# Per-brain fragment indexing (one pre-pass; no per-site full-array scans)
# --------------------------------------------------------------------------- #
def build_fragment_index(frag) -> dict:
    """Precompute every per-node/segment array the census reads per site.

    One O(N) pass over the graph; afterwards each site costs one KD-tree query
    plus O(neighborhood) lookups.
    """
    n_nodes = int(frag.number_of_nodes())
    xyz = np.asarray(frag.node_xyz, dtype=float)
    node_components = np.asarray(frag.node_component_id, dtype=np.int64)
    if xyz.shape[0] != n_nodes or node_components.shape[0] != n_nodes:
        raise ValueError("node arrays are not parallel to contiguous node ids")

    comp_to_seg: dict[int, int] = {
        int(component): int(str(swc_id).split(".")[0])
        for component, swc_id in frag.component_id_to_swc_id.items()
    }
    max_component = max(
        int(node_components.max(initial=0)), max(comp_to_seg, default=0))
    dense_comp_seg = np.full(max_component + 1, -1, dtype=np.int64)
    for component, segment in comp_to_seg.items():
        dense_comp_seg[component] = segment
    node_segments = dense_comp_seg[node_components]

    degrees = np.zeros(n_nodes, dtype=np.int64)
    for node, degree in frag.degree():
        degrees[int(node)] = int(degree)

    seg_components = defaultdict(set)
    for component, segment in comp_to_seg.items():
        seg_components[segment].add(component)

    return {
        "n_nodes": n_nodes,
        "graph": frag,
        "xyz": xyz,
        "node_components": node_components,
        "node_segments": node_segments,
        "degrees": degrees,
        "kdtree": cKDTree(xyz, copy_data=False),
        "seg_components": dict(seg_components),
        "fragment_segments": set(int(s) for s in np.unique(node_segments) if s > 0),
    }


# --------------------------------------------------------------------------- #
# Geodesic site measurement
# --------------------------------------------------------------------------- #
def _snap_to_segment(index: dict, segment_id: int, site_xyz: tuple,
                     snap_max_um: float) -> tuple[int | None, float]:
    """Nearest on-segment node within the snap radius, or (None, inf)."""
    point = np.asarray(site_xyz, dtype=float)
    neighbor_ids = np.asarray(
        index["kdtree"].query_ball_point(point, r=snap_max_um), dtype=np.int64)
    if len(neighbor_ids) == 0:
        return None, float("inf")
    own = neighbor_ids[index["node_segments"][neighbor_ids] == segment_id]
    if len(own) == 0:
        return None, float("inf")
    distances = np.linalg.norm(index["xyz"][own] - point, axis=1)
    best = int(np.argmin(distances))
    return int(own[best]), float(distances[best])


def _geodesic_to_junction(index: dict, start_node: int,
                          max_um: float) -> tuple[float, int]:
    """Cable distance to the nearest degree>=3 node, by bounded Dijkstra.

    Edges never leave a connected component and each component belongs to one
    segment, so the walk cannot wander onto another segment -- unlike a
    Euclidean ball, a spatially passing foreign arbor is unreachable. Returns
    ``(distance_um, junction_degree)``; ``(inf, 0)`` when no junction lies
    within ``max_um`` of cable.
    """
    degrees = index["degrees"]
    xyz = index["xyz"]
    graph = index["graph"]
    if degrees[start_node] >= 3:
        return 0.0, int(degrees[start_node])
    best: dict[int, float] = {int(start_node): 0.0}
    heap: list[tuple[float, int]] = [(0.0, int(start_node))]
    while heap:
        distance, node = heapq.heappop(heap)
        if distance > best.get(node, float("inf")):
            continue
        if degrees[node] >= 3:
            return float(distance), int(degrees[node])
        for neighbor in graph.neighbors(node):
            neighbor = int(neighbor)
            candidate = distance + float(
                np.linalg.norm(xyz[neighbor] - xyz[node]))
            if candidate <= max_um and candidate < best.get(
                    neighbor, float("inf")):
                best[neighbor] = candidate
                heapq.heappush(heap, (candidate, neighbor))
    return float("inf"), 0


def site_record(index: dict, segment_id: int, site_xyz: tuple,
                snap_max_um: float, max_geodesic_um: float) -> dict:
    """Continuous geodesic measurement + derived class for one merge site."""
    n_seg_components = len(index["seg_components"].get(segment_id, ()))
    multi_component = n_seg_components >= 2
    snap_node, snap_um = _snap_to_segment(
        index, segment_id, site_xyz, snap_max_um)

    if snap_node is None:
        geodesic_um, junction_degree, degree_at_snap = float("inf"), 0, 0
        topology = "no_fragment_nearby"
    else:
        degree_at_snap = int(index["degrees"][snap_node])
        geodesic_um, junction_degree = _geodesic_to_junction(
            index, snap_node, max_geodesic_um)
        if np.isfinite(geodesic_um):
            topology = "branch"
        elif multi_component:
            topology = "multi_component"
        else:
            topology = "bridge"

    return {
        "topology": topology,
        "snap_um": snap_um,
        "snap_node": -1 if snap_node is None else snap_node,
        "degree_at_snap": degree_at_snap,
        "geodesic_um_to_junction": geodesic_um,
        "junction_degree": junction_degree,
        "n_segment_components": n_seg_components,
        "segment_multi_component": bool(multi_component),
    }


def sample_control_records(index: dict, merge_segments: set[int],
                           n_controls: int, max_geodesic_um: float,
                           rng: np.random.Generator) -> list[dict]:
    """Geodesic distance-to-junction at random nodes of NON-merge segments.

    The skeleton has its own baseline junction density, so the merge-site CDF
    only means something against this control CDF. Controls start ON a node
    (snap_um = 0 by construction), isolating the walk from snap effects.
    """
    valid = np.flatnonzero(
        (index["node_segments"] > 0)
        & ~np.isin(index["node_segments"], list(merge_segments)))
    if len(valid) == 0:
        return []
    picks = rng.choice(valid, size=min(n_controls, len(valid)), replace=False)
    records = []
    for node in picks:
        node = int(node)
        segment_id = int(index["node_segments"][node])
        geodesic_um, junction_degree = _geodesic_to_junction(
            index, node, max_geodesic_um)
        records.append({
            "control_node": node,
            "segment_id": segment_id,
            "degree_at_snap": int(index["degrees"][node]),
            "geodesic_um_to_junction": geodesic_um,
            "junction_degree": junction_degree,
        })
    return records


# --------------------------------------------------------------------------- #
# Per-brain analysis
# --------------------------------------------------------------------------- #
def analyze_brain(cache_path: Path, snap_max_um: float, max_geodesic_um: float,
                  controls_per_site: int) -> dict:
    brain = brain_id_from_cache(cache_path)
    payload = load_payload(cache_path)
    frag = payload["fragments_graph"]
    merge_sites = list(payload.get("gt_merge_sites") or [])
    merge_labels = set(int(x) for x in np.asarray(
        payload.get("gt_merge_labels", []), dtype=np.int64))

    print(f"{brain}: {len(merge_sites)} merge sites, "
          f"{len(merge_labels)} merge labels", flush=True)
    index = build_fragment_index(frag)

    # Residual 1: labeled merge segments the geometric walk localized no site
    # for -- invisible to this site-anchored census, reported, never dropped.
    site_segments = {int(s["segment_id"]) for s in merge_sites}
    labels_without_site = sorted(merge_labels - site_segments)
    # Residual 2: merge segments with no fragment skeleton at all (filtered at
    # min_cable_length) -- unreachable by ANY fragment-graph detector.
    labels_without_fragment = sorted(
        s for s in merge_labels if s not in index["fragment_segments"])

    site_rows = []
    for site_index, site in enumerate(merge_sites):
        segment_id = int(site["segment_id"])
        record = site_record(index, segment_id, tuple(site["xyz"]),
                             snap_max_um, max_geodesic_um)
        record.update({
            "brain": brain,
            "site_index": site_index,
            "segment_id": segment_id,
            "gt_neuron": str(site.get("gt_neuron", "")),
            "segment_in_gt_merge_labels": segment_id in merge_labels,
        })
        site_rows.append(record)

    rng = np.random.default_rng(RNG_SEED)
    n_controls = max(len(merge_sites), 1) * controls_per_site
    control_rows = []
    for record in sample_control_records(
            index, merge_labels, n_controls, max_geodesic_um, rng):
        record["brain"] = brain
        control_rows.append(record)

    sites_df = pd.DataFrame(site_rows)
    geodesic = (sites_df["geodesic_um_to_junction"].to_numpy()
                if len(sites_df) else np.array([]))
    counts = (sites_df["topology"].value_counts().to_dict()
              if len(sites_df) else {})
    thresholds = sorted({t for t in DERIVED_THRESHOLDS_UM
                         if t <= max_geodesic_um} | {float(max_geodesic_um)})
    summary = {
        "brain": brain,
        "cache": str(cache_path),
        "n_fragment_nodes": index["n_nodes"],
        "n_merge_sites": len(merge_sites),
        "n_merge_labels": len(merge_labels),
        "n_labels_without_site": len(labels_without_site),
        "labels_without_site": labels_without_site,
        "n_labels_without_fragment": len(labels_without_fragment),
        "labels_without_fragment": labels_without_fragment,
        "n_controls": n_controls,
        "snap_max_um": float(snap_max_um),
        "max_geodesic_um": float(max_geodesic_um),
        "snap_um_median": (float(np.median(sites_df["snap_um"]))
                           if len(sites_df) else float("nan")),
        "class_counts_at_bound": {
            cls: int(counts.get(cls, 0)) for cls in TOPOLOGY_CLASSES
        },
        # The branch fraction at any threshold is a point on the geodesic CDF.
        "branch_fraction_by_threshold": {
            f"{t:g}": (float((geodesic <= t).mean()) if len(geodesic)
                       else float("nan"))
            for t in thresholds
        },
    }

    del payload, frag, index
    gc.collect()
    return {
        "sites": sites_df,
        "controls": pd.DataFrame(control_rows),
        "summary": summary,
    }


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def _cdf_curve(values: np.ndarray, max_um: float) -> tuple[np.ndarray, np.ndarray]:
    """Fraction of values <= x over a dense grid up to the walk bound."""
    grid = np.linspace(0.0, max_um, 200)
    if len(values) == 0:
        return grid, np.zeros_like(grid)
    return grid, np.array([(values <= x).mean() for x in grid])


def save_brain_figure(result: dict, max_geodesic_um: float,
                      output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sites = result["sites"]
    controls = result["controls"]
    brain = result["summary"]["brain"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    # (1) Geodesic distance-to-junction CDF -- the coverage headline. A smooth
    # rise = sites displaced from their junctions; a plateau below 1.0 = true
    # junction-free (bridge/component) merges. Controls give the skeleton's
    # baseline junction density.
    site_geo = (sites["geodesic_um_to_junction"].to_numpy()
                if len(sites) else np.array([]))
    ctrl_geo = (controls["geodesic_um_to_junction"].to_numpy()
                if len(controls) else np.array([]))
    grid, site_cdf = _cdf_curve(site_geo, max_geodesic_um)
    _, ctrl_cdf = _cdf_curve(ctrl_geo, max_geodesic_um)
    axes[0].plot(grid, site_cdf, lw=2,
                 label=f"merge sites (n={len(site_geo)})")
    axes[0].plot(grid, ctrl_cdf, lw=2, ls="--",
                 label=f"controls (n={len(ctrl_geo)})")
    if len(site_geo):
        never = float(np.isinf(site_geo).mean())
        axes[0].annotate(f"no junction within bound: {never:.0%}",
                         xy=(0.98, 0.05), xycoords="axes fraction",
                         ha="right", fontsize=8)
    axes[0].set_xlabel("geodesic cable distance to nearest junction (µm)")
    axes[0].set_ylabel("fraction with junction within distance")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("Junction-reachability CDF (branch fraction)")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.25)

    # (2) Snap distance -- coordinate-quality audit. Large snap_um means the
    # recorded site xyz sits far from its own skeleton (walk residual), which
    # caps how literally the geodesic start point can be trusted.
    if len(sites):
        finite_snap = sites.loc[np.isfinite(sites["snap_um"]), "snap_um"]
        axes[1].hist(finite_snap, bins=30, alpha=0.75)
        axes[1].axvline(float(np.median(finite_snap)), color="k", ls=":",
                        label=f"median {np.median(finite_snap):.1f} µm")
        axes[1].legend(fontsize=8)
    axes[1].set_xlabel("site-to-skeleton snap distance (µm)")
    axes[1].set_ylabel("merge sites")
    axes[1].set_title("Snap quality (site xyz vs own segment)")
    axes[1].grid(True, alpha=0.25)

    # (3) How many components the implicated label spans (component detector).
    if len(sites):
        n_comp = sites["n_segment_components"].clip(upper=5)
        axes[2].hist(n_comp, bins=np.arange(0, 7) - 0.5, density=False, alpha=0.7)
    axes[2].set_xlabel("components of the merge label (clipped at 5)")
    axes[2].set_ylabel("merge sites")
    axes[2].set_title("Label component multiplicity")
    axes[2].grid(True, alpha=0.25)

    fig.suptitle(f"Merge-topology census — brain {brain}", fontsize=13)
    fig.tight_layout()
    temporary = output_path.with_name(output_path.name + ".tmp")
    fig.savefig(temporary, dpi=180, bbox_inches="tight", format="png")
    temporary.replace(output_path)
    plt.close(fig)


def save_pooled_figure(all_sites: pd.DataFrame, max_geodesic_um: float,
                       output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    # (1) Stacked per-brain class composition at the walk bound.
    brains = sorted(all_sites["brain"].unique()) if len(all_sites) else []
    bottoms = np.zeros(len(brains))
    for cls in TOPOLOGY_CLASSES:
        values = np.array([
            ((all_sites["brain"] == b) & (all_sites["topology"] == cls)).sum()
            for b in brains
        ], dtype=float)
        totals = np.array([
            max((all_sites["brain"] == b).sum(), 1) for b in brains], dtype=float)
        fractions = values / totals
        axes[0].bar(brains, fractions, bottom=bottoms, label=cls)
        bottoms += fractions
    axes[0].set_ylabel("fraction of merge sites")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title(
        f"Per-brain topology composition @ walk bound {max_geodesic_um:g} µm")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, axis="y", alpha=0.25)
    # Absolute counts on top: fractions from ~20 events must say so.
    for i, b in enumerate(brains):
        axes[0].text(i, 1.01, f"n={(all_sites['brain'] == b).sum()}",
                     ha="center", fontsize=8)

    # (2) Per-brain junction-reachability CDFs -- the continuous replacement
    # for the old tolerance-stability bars. Agreement of curve SHAPES across
    # brains is the stability statement.
    for b in brains:
        geo = all_sites.loc[all_sites["brain"] == b,
                            "geodesic_um_to_junction"].to_numpy()
        grid, cdf = _cdf_curve(geo, max_geodesic_um)
        axes[1].plot(grid, cdf, lw=1.6, label=f"{b} (n={len(geo)})")
    if len(all_sites):
        grid, pooled_cdf = _cdf_curve(
            all_sites["geodesic_um_to_junction"].to_numpy(), max_geodesic_um)
        axes[1].plot(grid, pooled_cdf, lw=2.6, color="k", label="pooled")
    axes[1].set_xlabel("geodesic cable distance to nearest junction (µm)")
    axes[1].set_ylabel("fraction with junction within distance")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("Junction-reachability CDF per brain")
    axes[1].legend(fontsize=7)
    axes[1].grid(True, alpha=0.25)

    fig.suptitle(
        f"Merge-topology census — {len(brains)} brain(s) pooled", fontsize=13)
    fig.tight_layout()
    temporary = output_path.with_name(output_path.name + ".tmp")
    fig.savefig(temporary, dpi=180, bbox_inches="tight", format="png")
    temporary.replace(output_path)
    plt.close(fig)


def replot_from_artifacts(run_dir: Path, brains: list[str] | None) -> int:
    """Rebuild every figure from saved per_brain CSVs; no cache loading."""
    per_brain_dir = run_dir / "per_brain"
    figures_dir = run_dir / "figures"
    run_summary_path = run_dir / "run_summary.json"
    max_geodesic_um = DEFAULT_MAX_GEODESIC_UM
    if run_summary_path.exists():
        max_geodesic_um = float(json.loads(run_summary_path.read_text())
                                .get("max_geodesic_um", max_geodesic_um))
    requested = {str(b) for b in brains} if brains else None
    all_sites = []
    for sites_csv in sorted(per_brain_dir.glob("*_sites.csv")):
        brain = sites_csv.name[:-len("_sites.csv")]
        if requested is not None and brain not in requested:
            continue
        sites = pd.read_csv(sites_csv)
        # read_csv parses numeric brain ids as ints; bar charts need strings
        sites["brain"] = sites["brain"].astype(str)
        controls_csv = per_brain_dir / f"{brain}_controls.csv"
        controls = (pd.read_csv(controls_csv) if controls_csv.exists()
                    else pd.DataFrame())
        summary_path = per_brain_dir / f"{brain}_summary.json"
        summary = (json.loads(summary_path.read_text())
                   if summary_path.exists() else {"brain": brain})
        save_brain_figure(
            {"sites": sites, "controls": controls, "summary": summary},
            max_geodesic_um, figures_dir / f"{brain}_geodesic_census.png")
        all_sites.append(sites)
        print(f"{brain}: figure rebuilt", flush=True)
    if not all_sites:
        raise FileNotFoundError(f"no *_sites.csv under {per_brain_dir}")
    save_pooled_figure(pd.concat(all_sites, ignore_index=True),
                       max_geodesic_um,
                       figures_dir / "pooled_topology_census.png")
    print(f"figures: {figures_dir}")
    return 0


# --------------------------------------------------------------------------- #
# Atomic writers
# --------------------------------------------------------------------------- #
def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text)
    temporary.replace(path)


def atomic_to_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mcl", type=int, required=True,
                        help="min_cable_length of the _add caches to census")
    parser.add_argument("--brains", nargs="*", default=None,
                        help="brain ids; default: every mcl-matching cache")
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT,
                        help="artifact root (default notebooks/merge_diagnosis)")
    parser.add_argument("--snap-max-um", type=float,
                        default=DEFAULT_SNAP_MAX_UM,
                        help="max site-to-own-segment snap distance in µm")
    parser.add_argument("--max-geodesic-um", type=float,
                        default=DEFAULT_MAX_GEODESIC_UM,
                        help="cable-distance bound of the junction walk in µm")
    parser.add_argument("--controls-per-site", type=int,
                        default=CONTROLS_PER_SITE,
                        help="control locations sampled per merge site")
    parser.add_argument("--plot-only", action="store_true",
                        help="rebuild figures from saved per_brain CSVs "
                             "(no cache loading)")
    args = parser.parse_args(argv)

    if args.snap_max_um <= 0:
        parser.error("--snap-max-um must be positive")
    if args.max_geodesic_um <= 0:
        parser.error("--max-geodesic-um must be positive")
    if args.controls_per_site < 1:
        parser.error("--controls-per-site must be positive")

    run_dir = args.output_dir.resolve() / f"mcl{args.mcl}"
    per_brain_dir = run_dir / "per_brain"
    figures_dir = run_dir / "figures"
    if args.plot_only:
        return replot_from_artifacts(run_dir, args.brains)
    caches = discover_add_caches(args.cache_dir.resolve(), args.mcl, args.brains)
    print(f"artifacts: {run_dir}")
    print("caches:")
    for cache in caches:
        print(f"  {brain_id_from_cache(cache)}: {cache}")

    started = time.monotonic()
    all_sites, all_controls, brain_summaries, summary_rows = [], [], [], []
    for cache in caches:
        result = analyze_brain(cache, args.snap_max_um, args.max_geodesic_um,
                               args.controls_per_site)
        brain = result["summary"]["brain"]
        atomic_to_csv(result["sites"], per_brain_dir / f"{brain}_sites.csv")
        atomic_to_csv(result["controls"], per_brain_dir / f"{brain}_controls.csv")
        atomic_write_text(per_brain_dir / f"{brain}_summary.json",
                          json.dumps(result["summary"], indent=2) + "\n")
        save_brain_figure(result, args.max_geodesic_um,
                          figures_dir / f"{brain}_geodesic_census.png")
        all_sites.append(result["sites"])
        all_controls.append(result["controls"])
        brain_summaries.append(result["summary"])
        for threshold_key, fraction in (
                result["summary"]["branch_fraction_by_threshold"].items()):
            summary_rows.append({
                "brain": brain,
                "threshold_um": float(threshold_key),
                "branch_fraction": fraction,
                "n_merge_sites": result["summary"]["n_merge_sites"],
                "n_merge_labels": result["summary"]["n_merge_labels"],
                "n_labels_without_site":
                    result["summary"]["n_labels_without_site"],
                "n_labels_without_fragment":
                    result["summary"]["n_labels_without_fragment"],
                "snap_um_median": result["summary"]["snap_um_median"],
            })
        print(f"{brain}: done", flush=True)

    pooled_sites = (pd.concat(all_sites, ignore_index=True)
                    if all_sites else pd.DataFrame())
    save_pooled_figure(pooled_sites, args.max_geodesic_um,
                       figures_dir / "pooled_topology_census.png")
    atomic_to_csv(pd.DataFrame(summary_rows), run_dir / "summary.csv")

    run_summary = {
        "mcl": args.mcl,
        "snap_max_um": args.snap_max_um,
        "max_geodesic_um": args.max_geodesic_um,
        "controls_per_site": args.controls_per_site,
        "rng_seed": RNG_SEED,
        "brains": [s["brain"] for s in brain_summaries],
        "elapsed_seconds": time.monotonic() - started,
        "per_brain": brain_summaries,
        "analysis_script": str(THIS_FILE),
        "note": (
            "Coverage diagnosis only. geodesic_um_to_junction is the cable "
            "distance from the snapped site to the nearest degree>=3 node; "
            "the branch fraction at any threshold is a point on its CDF, and "
            "the class at the walk bound says which enumeration detector "
            "(branch/bridge/component) could reach each GT merge site. It is "
            "not a precision feature and not a causal mechanism claim. "
            "labels_without_site are the geometric walk's residual; "
            "labels_without_fragment are unreachable at this min_cable_length."
        ),
    }
    atomic_write_text(run_dir / "run_summary.json",
                      json.dumps(run_summary, indent=2) + "\n")

    if len(pooled_sites):
        geodesic = pooled_sites["geodesic_um_to_junction"].to_numpy()
        thresholds = sorted(
            {t for t in DERIVED_THRESHOLDS_UM if t <= args.max_geodesic_um}
            | {float(args.max_geodesic_um)})
        print(f"\nPooled junction-reachability CDF (n={len(geodesic)}):")
        for threshold in thresholds:
            print(f"  branch fraction @ {threshold:g} µm cable: "
                  f"{(geodesic <= threshold).mean():.0%}")
        counts = pooled_sites["topology"].value_counts().to_dict()
        parts = ", ".join(
            f"{cls}={counts.get(cls, 0) / len(pooled_sites):.0%}"
            for cls in TOPOLOGY_CLASSES)
        print(f"  class @ walk bound {args.max_geodesic_um:g} µm: {parts}")
    print(f"\nall outputs: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
