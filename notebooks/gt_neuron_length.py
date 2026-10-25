#!/usr/bin/env python3
"""Total cable length of every GT neuron, per brain.

The neurons are those of each brain cache (``cache/dataset_cache_<brain>_mcl<N>_add.pkl``,
``gt_graph``; one connected component per neuron, names ``N0XX-<brain>-<initials>``). The plotted
length is that of the neuron's GT tracing itself: the SWC polyline (every node to its parent) read
from the cache's ``gt_path`` on GCS, with each axis scaled by the cache's voxel size
(``anisotropy``, um per voxel in x, y, z). Two more lengths are kept for comparison:

- ``cache_um``: the cache's ``gt_graph`` edges summed. The loader truncates the scaled SWC
  coordinates to whole um and refits each branch with a spline at ``node_spacing`` (5 um), which
  straightens the tracing: on 2026-10-23 this was 0.82-0.90 (median 0.86) of ``swc_um`` for all
  158 neurons, the same 0.83-0.89 per brain.
- canonical ``SWC Run Length`` (``metrics_out/<brain>/<run>/results.csv``,
  segmentation_skeleton_metrics): the polyline of the voxel SWC after its coordinates are stored as
  int32, in voxels, no anisotropy. It equals ``swc_voxel_int32`` here (ratio 1.0 for all 158 neurons
  on 2026-10-23) and is 4-12% above ``swc_voxel`` because the truncation adds staircase steps.

The um are those of the fused image volume (voxel size 0.748 x 0.748 x 1.0 um). Their relation to
the native tissue depends on the sample's expansion, which the dataset's ``procedures.json`` does
not record (794495: delipidation, immunolabeling, refractive index matching); GT coordinates span
about 33-43 x 17-25 x 17-18 mm.

Two steps:

1. ``--measure`` (compute node, panda env, one cache at a time; each load takes 8-100 s; GCS
   credentials ``configs/allen-nd-goog-f5d46dbfa2cd.json`` for the SWCs):
       srun --jobid=<job> --overlap -n1 -c4 bash -lc 'cd exa-spim-agent && \
           /home/zihan.zhang/.conda/envs/panda/bin/python -u notebooks/gt_neuron_length.py --measure'
2. ``--plot`` (seconds, reads the per-brain files): one dot per neuron and the brain's median
   on a log length axis; ``neurons.csv`` (one row per neuron, all lengths and the ratios) and
   ``summary.csv`` (one row per brain).

Outputs, all under ``notebooks/gt_neuron_length/``: ``<brain>_mcl<N>.json``, ``neurons.csv``,
``summary.csv``, ``gt_neuron_length.png``. Read-only with respect to the caches and GCS.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import gc
import io
import itertools
import json
from pathlib import Path
import pickle
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CURRENT_BRAINS = ["794495", "802449", "789202", "794493", "794491"]
OLDER_BRAINS = ["747807", "750318", "751473", "754610", "754612", "754613"]
ALL_BRAINS = CURRENT_BRAINS + OLDER_BRAINS
OUT_DIR = ROOT / "notebooks" / "gt_neuron_length"
METRICS_DIR = ROOT / "metrics_out"
QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9)
# Same palette as notebooks/split_fragment_length.py and plot_label_availability.py.
NEURON, MEDIAN = "#2a78d6", "#0b0b0b"
TEXT_PRIMARY, TEXT_SECONDARY, GRID = "#0b0b0b", "#52514e", "#e4e3df"


# ----------------------------------------------------------------------------- measure
def cache_neurons(graph, anisotropy) -> dict:
    """``{neuron: record}`` of the cache's GT graph: skeleton length (um, voxels), nodes, components,
    tips and branch points."""
    n_nodes = int(graph.number_of_nodes())
    xyz = np.asarray(graph.node_xyz, dtype=np.float64)
    components = np.asarray(graph.node_component_id, dtype=np.int64)
    if xyz.shape[0] != n_nodes or components.shape[0] != n_nodes:
        raise ValueError("node_xyz and node_component_id must align with contiguous node ids")
    if n_nodes and (not graph.has_node(0) or not graph.has_node(n_nodes - 1)):
        raise ValueError("GT graph node ids must be 0..N-1")
    n_edges = int(graph.number_of_edges())
    edges = np.fromiter(itertools.chain.from_iterable(graph.edges()), dtype=np.int64,
                        count=2 * n_edges).reshape(n_edges, 2)
    if (components[edges[:, 0]] != components[edges[:, 1]]).any():
        raise ValueError("an edge crosses connected components")
    step_um = xyz[edges[:, 0]] - xyz[edges[:, 1]]
    edge_um = np.linalg.norm(step_um, axis=1)
    edge_voxel = np.linalg.norm(step_um / np.asarray(anisotropy, dtype=np.float64), axis=1)
    degree = np.bincount(edges.ravel(), minlength=n_nodes)
    component_ids, component_index = np.unique(components, return_inverse=True)
    names = np.array([str(graph.component_id_to_swc_id[int(c)]) for c in component_ids])
    neuron_names, neuron_of_component = np.unique(names, return_inverse=True)
    neuron_index = neuron_of_component[component_index]
    size = len(neuron_names)
    cable = np.bincount(neuron_index[edges[:, 0]], weights=edge_um, minlength=size)
    cable_voxel = np.bincount(neuron_index[edges[:, 0]], weights=edge_voxel, minlength=size)
    nodes = np.bincount(neuron_index, minlength=size)
    tips = np.bincount(neuron_index, weights=(degree == 1), minlength=size)
    branches = np.bincount(neuron_index, weights=(degree >= 3), minlength=size)
    n_components = np.bincount(neuron_of_component, minlength=size)
    return {str(name): {"cache_um": float(cable[i]), "cache_voxel": float(cable_voxel[i]),
                        "cache_nodes": int(nodes[i]), "n_components": int(n_components[i]),
                        "n_tips": int(tips[i]), "n_branch_points": int(branches[i])}
            for i, name in enumerate(neuron_names)}


def swc_lengths(text: str, anisotropy) -> dict:
    """Polyline length of one voxel SWC: um (axes scaled), voxels, and voxels after int32 storage
    (segmentation_skeleton_metrics' ``SWC Run Length``)."""
    offset = np.zeros(3)
    for line in text.splitlines():
        if line.startswith("# OFFSET"):
            offset = np.array([float(v) for v in line.split()[2:5]])
    table = np.loadtxt(io.StringIO(text), comments="#", ndmin=2)
    ids = table[:, 0].astype(np.int64)
    xyz = table[:, 2:5] + offset
    parent = table[:, 6].astype(np.int64)
    order = np.argsort(ids)
    child = np.flatnonzero(parent != -1)
    position = order[np.clip(np.searchsorted(ids[order], parent[child]), 0, len(ids) - 1)]
    if not np.array_equal(ids[position], parent[child]):
        raise ValueError("an SWC node names a parent id that does not exist")
    step = xyz[child] - xyz[position]
    stored = xyz.astype(np.int32).astype(np.float64)
    return {"swc_um": float(np.linalg.norm(step * np.asarray(anisotropy, dtype=np.float64), axis=1).sum()),
            "swc_voxel": float(np.linalg.norm(step, axis=1).sum()),
            "swc_voxel_int32": float(np.linalg.norm(stored[child] - stored[position], axis=1).sum()),
            "swc_nodes": int(len(ids)), "swc_roots": int((parent == -1).sum())}


def gcs_swc_lengths(gt_path: str, anisotropy) -> dict:
    """``{neuron: swc_lengths}`` of every ``.swc`` under the ``gs://`` GT directory."""
    from google.cloud import storage
    from labeled_cache import io as cache_io

    cache_io.set_cloud_environment()
    if not gt_path.startswith("gs://"):
        raise ValueError(f"expected a gs:// GT path, got {gt_path}")
    bucket, prefix = gt_path[len("gs://"):].split("/", 1)
    client = storage.Client()
    out = {}
    for blob in client.list_blobs(bucket, prefix=prefix.rstrip("/") + "/"):
        if blob.name.endswith(".swc"):
            out[Path(blob.name).stem] = swc_lengths(blob.download_as_text(), anisotropy)
    return out


def quantile_record(values) -> dict:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"n": 0}
    record = {"n": int(len(values)), "min": float(values.min()), "max": float(values.max())}
    record.update({f"q{int(q * 100)}": float(np.quantile(values, q)) for q in QUANTILES})
    return record


def measure_brain(brain: str, mcl: int) -> dict:
    import agentic_neuron_proofreader  # noqa: F401 - registers SkeletonGraph for unpickling

    started = time.monotonic()
    cache = ROOT / "cache" / f"dataset_cache_{brain}_mcl{mcl}_add.pkl"
    stat = cache.stat()
    with cache.open("rb") as stream:
        payload = pickle.load(stream)
    if payload.get("min_cable_length") is not None and int(payload["min_cable_length"]) != mcl:
        raise ValueError(f"{cache} was built with min_cable_length={payload['min_cable_length']}")
    gt_path = str(payload["gt_path"])
    if f"/{brain}/" not in gt_path:
        raise ValueError(f"{cache} holds the GT of {gt_path}, not brain {brain}")
    anisotropy = [float(v) for v in payload["anisotropy"]]   # um per voxel, (x, y, z) like node_xyz and SWC
    in_cache = cache_neurons(payload["gt_graph"], anisotropy)
    del payload
    gc.collect()
    on_gcs = gcs_swc_lengths(gt_path, anisotropy)
    missing = {"swc_um": float("nan"), "swc_voxel": float("nan"), "swc_voxel_int32": float("nan"),
               "swc_nodes": 0, "swc_roots": 0}
    neurons = [{"name": name, **on_gcs.get(name, missing), **record} for name, record in sorted(in_cache.items())]
    swc_um = [n["swc_um"] for n in neurons]
    return {
        "brain": brain, "mcl": mcl,
        "created": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source_cache": {"path": str(cache), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns},
        "gt_path": gt_path, "anisotropy_um_per_voxel_xyz": anisotropy,
        "n_neurons": len(neurons), "total_swc_um": float(np.nansum(swc_um)),
        "swc_um": quantile_record(swc_um), "cache_um": quantile_record([n["cache_um"] for n in neurons]),
        "neurons_without_swc": sorted(set(in_cache) - set(on_gcs)),
        "swcs_not_in_cache": sorted(set(on_gcs) - set(in_cache)),
        "neurons": neurons, "seconds": round(time.monotonic() - started, 1),
    }


def brain_path(brain, mcl, out_dir=OUT_DIR):
    return out_dir / f"{brain}_mcl{mcl}.json"


def run_measure(brains, mcl, out_dir, force):
    out_dir.mkdir(parents=True, exist_ok=True)
    for brain in brains:
        path = brain_path(brain, mcl, out_dir)
        if path.exists() and not force:
            print(f"{brain}: kept {path.name} (use --force to recompute)", flush=True)
            continue
        print(f"{brain}: measuring", flush=True)
        summary = measure_brain(brain, mcl)
        path.write_text(json.dumps(summary, indent=1) + "\n")
        q = summary["swc_um"]
        print(f"{brain}: {summary['n_neurons']} neurons, total {summary['total_swc_um'] / 1e3:.0f} mm, "
              f"median {q['q50'] / 1e3:.1f} mm (range {q['min'] / 1e3:.1f}-{q['max'] / 1e3:.1f}); "
              f"without SWC {summary['neurons_without_swc']}, SWCs not in the cache "
              f"{summary['swcs_not_in_cache']}; {summary['seconds']} s", flush=True)


# ----------------------------------------------------------------------------- plot
def canonical_lengths(brain) -> tuple[dict, str | None]:
    """``{neuron: SWC Run Length}`` (voxels) from the brain's one current canonical results.csv."""
    import pandas as pd

    paths = [p for p in (METRICS_DIR / brain).glob("*/results.csv") if "archive" not in p.parts]
    if len(paths) != 1:
        return {}, None
    table = pd.read_csv(paths[0], index_col=0)
    if "SWC Run Length" not in table.columns:
        return {}, None
    return table["SWC Run Length"].astype(float).to_dict(), str(paths[0].relative_to(ROOT))


def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=8, length=3)
    ax.grid(True, axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def plot_lengths(data, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker  # noqa: F401 - FixedLocator, FuncFormatter

    brains = list(data)
    fig, ax = plt.subplots(figsize=(8.5, 0.42 * len(brains) + 1.6))
    lengths_mm = []
    for i, brain in enumerate(brains):
        mm = np.array([n["swc_um"] for n in data[brain]["neurons"]]) / 1e3
        mm = mm[np.isfinite(mm)]
        lengths_mm.extend(mm)
        # Deterministic spread so neurons of similar length stay visible.
        jitter = np.random.default_rng(int(brain)).uniform(-0.2, 0.2, len(mm))
        ax.scatter(mm, i + jitter, s=20, color=NEURON, alpha=0.75, lw=0, zorder=3,
                   label="one GT neuron" if i == 0 else None)
        ax.scatter([np.median(mm)], [i], marker="|", s=260, color=MEDIAN, lw=1.8, zorder=4,
                   label="median of the brain" if i == 0 else None)
    if any(b in OLDER_BRAINS for b in brains) and any(b in CURRENT_BRAINS for b in brains):
        split_at = max(i for i, b in enumerate(brains) if b in CURRENT_BRAINS) + 0.5
        ax.axhline(split_at, color=GRID, lw=1.0, ls="--", zorder=1)
    style(ax)
    for group, members in (("current microscope", CURRENT_BRAINS), ("older microscope", OLDER_BRAINS)):
        rows = [i for i, b in enumerate(brains) if b in members]
        if rows:
            ax.text(0.995, min(rows) - 0.45, group, transform=ax.get_yaxis_transform(), ha="right",
                    va="top", fontsize=7.5, color=TEXT_SECONDARY, style="italic")
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(
        [t for t in (5, 10, 20, 50, 100, 200, 500, 1000, 2000) if min(lengths_mm) / 1.5 <= t <= max(lengths_mm) * 1.5]))
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda t, _: f"{t:g}"))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_yticks(range(len(brains)))
    ax.set_yticklabels([f"{b}   n = {data[b]['n_neurons']}" for b in brains], fontsize=8.5, color=TEXT_PRIMARY)
    ax.invert_yaxis()
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("total cable length of the neuron (mm in image space, log scale)", fontsize=8.5,
                  color=TEXT_SECONDARY)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=8.5)
    fig.suptitle(f"Total cable length of each GT neuron, per brain ({len(lengths_mm)} neurons in {len(brains)} "
                 f"brains; GT SWC polyline)", fontsize=10, color=TEXT_PRIMARY, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def run_plot(brains, mcl, out_dir):
    import pandas as pd

    data = {}
    for brain in brains:
        path = brain_path(brain, mcl, out_dir)
        if path.exists():
            data[brain] = json.loads(path.read_text())
    if not data:
        raise FileNotFoundError(f"no per-brain files under {out_dir}; run --measure first")
    neuron_rows, summary_rows = [], []
    for brain, summary in data.items():
        canonical, source = canonical_lengths(brain)
        rows = []
        for neuron in summary["neurons"]:
            # Canonical stores the voxel SWC as int32 and sums without anisotropy: compare in kind.
            run_length = canonical.get(neuron["name"], float("nan"))
            rows.append({"brain": brain, **neuron, "canonical_swc_run_length_voxel": run_length,
                         "canonical_over_swc_voxel_int32": run_length / neuron["swc_voxel_int32"],
                         "cache_over_swc_um": neuron["cache_um"] / neuron["swc_um"]})
        neuron_rows.extend(rows)
        brain_table = pd.DataFrame(rows)
        deviation = (brain_table["canonical_over_swc_voxel_int32"] - 1).abs()
        summary_rows.append({
            "brain": brain, "group": "current" if brain in CURRENT_BRAINS else "older",
            "n_neurons": summary["n_neurons"], "total_swc_um": summary["total_swc_um"],
            **{f"swc_um_{k}": v for k, v in summary["swc_um"].items() if k != "n"},
            "median_cache_over_swc_um": float(brain_table["cache_over_swc_um"].median()),
            "neurons_without_swc": len(summary["neurons_without_swc"]),
            "swcs_not_in_cache": len(summary["swcs_not_in_cache"]),
            "neurons_with_canonical": int(deviation.notna().sum()),
            "max_canonical_deviation": float(deviation.max()) if deviation.notna().any() else float("nan"),
            "canonical_neurons_missing_in_cache": len(set(canonical) - set(brain_table["name"])),
            "canonical_results": source})
    pd.DataFrame(neuron_rows).to_csv(out_dir / "neurons.csv", index=False)
    table = pd.DataFrame(summary_rows)
    table.to_csv(out_dir / "summary.csv", index=False)
    plot_lengths(data, out_dir / "gt_neuron_length.png")
    print(f"{len(data)} brains: {', '.join(data)}")
    shown = table[["brain", "n_neurons", "total_swc_um", "swc_um_q50", "swc_um_min", "swc_um_max",
                   "median_cache_over_swc_um", "neurons_without_swc", "swcs_not_in_cache",
                   "neurons_with_canonical", "max_canonical_deviation"]].copy()
    for column in ("total_swc_um", "swc_um_q50", "swc_um_min", "swc_um_max"):
        shown[column.replace("_um", "_mm")] = (shown.pop(column) / 1e3).round(1)
    print(shown.round(4).to_string(index=False))
    print(f"wrote {out_dir / 'neurons.csv'}, {out_dir / 'summary.csv'} and {out_dir / 'gt_neuron_length.png'}")


# ----------------------------------------------------------------------------- main
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--measure", action="store_true",
                        help="load each cache, read its GT SWCs from GCS and record the neuron lengths")
    parser.add_argument("--plot", action="store_true", help="figure, neurons.csv and summary.csv from the per-brain files")
    parser.add_argument("--brains", nargs="+", default=ALL_BRAINS)
    parser.add_argument("--mcl", type=int, default=100)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--force", action="store_true", help="recompute brains that already have files")
    args = parser.parse_args(argv)
    if not (args.measure or args.plot):
        parser.error("choose --measure and/or --plot")
    if args.measure:
        run_measure(args.brains, args.mcl, args.out_dir, args.force)
    if args.plot:
        run_plot(args.brains, args.mcl, args.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
