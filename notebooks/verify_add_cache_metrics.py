"""Verify labelled caches against metrics_out for every brain at one MCL.

Run in panda on a compute node. Only load trusted pickle caches. This script
reads stored labels directly, without opening image readers or cloud volumes.
It never writes caches or canonical results. Matching verification CSVs and
scatter images in --output-dir are overwritten, as in the former notebook.

# Merges compares geometric-walk sites only. Supplemental two-GT-only, shared
evidence and combined site counts are reported separately. Untagged legacy
sites are geometric; missing sites are unavailable, not zero. Combined site
records are not counts of independent biological merge events.

Examples (from the repository root, on a compute node):
    python notebooks/verify_add_cache_metrics.py --mcl 100
    python notebooks/verify_add_cache_metrics.py --mcl 10 --brain 794495 794493
"""

import argparse
import gc
import pickle
import re
from collections import Counter, defaultdict
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from agentic_neuron_proofreader.data_modules import canonical_labeling as cl


ROOT = Path(__file__).resolve().parents[1]
CACHE_PATTERN = re.compile(r"dataset_cache_(\d+)_mcl(\d+)_add\.pkl")
COMPARE_COLS = ["# Splits", "# Merges", "% Split Edges", "% Omit Edges",
                "% Merged Edges", "Edge Accuracy"]
MERGE_COUNT_DEFINITION = "geometric_walk_sites"
SITE_ATTRIBUTION = "stored_representative_gt_neuron"


def partition_merge_sites(sites):
    """Separate canonical-comparison sites without double-counting shared records."""
    partitions = {"geometric": [], "two_gt_only": [], "shared": []}
    allowed = {"geometric_walk", "two_gt_junction"}
    for site in sites:
        source = site.get("source")
        declared = site.get("sources", [])
        if not isinstance(declared, (list, tuple, set)):
            raise ValueError(f"Invalid merge-site sources: {declared!r}")
        sources = set(declared)
        if source is not None:
            sources.add(source)
        if not sources:
            sources = {"geometric_walk"}
        if not sources.issubset(allowed):
            raise ValueError(f"Unknown merge-site sources: {sources - allowed}")
        if "geometric_walk" in sources:
            partitions["geometric"].append(site)
            if "two_gt_junction" in sources:
                partitions["shared"].append(site)
        else:
            partitions["two_gt_only"].append(site)
    return partitions


def stored_value(payload, key, graph_attribute):
    """Prefer top-level cache fields; accept legacy graph-only storage."""
    if key in payload:
        return payload[key]
    return getattr(payload["gt_graph"], graph_attribute, None)


def summarize_payload(payload):
    graph = payload["gt_graph"]
    labels = stored_value(payload, "gt_node_canonical_label", "node_label")
    errors = stored_value(payload, "gt_edge_error", "edge_error")
    if labels is None or errors is None:
        raise ValueError("Cache lacks canonical node or edge labels; run relabel_cache.py first")
    node_label, edge_error = np.asarray(labels), np.asarray(errors)
    if node_label.shape != (len(graph.node_xyz),):
        raise ValueError("Canonical node labels do not align with gt_graph.node_xyz")
    if edge_error.shape != (graph.number_of_edges(),):
        raise ValueError("Stored edge labels do not align with gt_graph edges")
    if not np.isin(edge_error, list(cl.EDGE_ERROR_NAMES)).all():
        raise ValueError("Unknown stored edge-error code")
    stored_labels = stored_value(payload, "gt_merge_labels", "merge_labels")
    merge_labels = (set(map(int, stored_labels)) if stored_labels is not None
                    else cl.merge_labels(graph, node_label))
    if stored_labels is None:
        print("NOTE: no stored merge labels; using the node-count rule only.", flush=True)
    stored_sites = stored_value(payload, "gt_merge_sites", "merge_sites")
    sites_available = stored_sites is not None
    sites = list(stored_sites) if sites_available else []
    partitions = partition_merge_sites(sites)
    groups = {
        "# Geometric Sites": partitions["geometric"],
        "# Two-GT-only Sites": partitions["two_gt_only"],
        "# Shared-evidence Sites": partitions["shared"],
        "# Combined Sites": sites,
    }
    site_counts = {name: len(records) if sites_available else np.nan
                   for name, records in groups.items()}
    counts_by_neuron = {name: Counter(site["gt_neuron"] for site in records)
                        for name, records in groups.items()}
    neuron_class = defaultdict(lambda: defaultdict(int))
    neuron_segments = defaultdict(set)
    for index, (source_node, target_node) in enumerate(graph.edges):
        neuron_class[graph.node_segment_id(source_node)][int(edge_error[index])] += 1
    for node in graph.nodes:
        label = int(node_label[node])
        if label != cl.UNLABELED:
            neuron_segments[graph.node_segment_id(node)].add(label)
    segment_counts = cl.segment_neuron_node_counts(graph, node_label)
    merged_edges = defaultdict(int)
    for label in merge_labels:
        for neuron, count in segment_counts.get(label, {}).items():
            merged_edges[neuron] += max(count - 1, 0)
    neuron_cable = {}
    for component in nx.connected_components(graph):
        root = next(iter(component))
        neuron_cable[graph.node_segment_id(root)] = graph.cable_length(root=root)
    rows = []
    for neuron, classes in neuron_class.items():
        total = sum(classes.values())
        row = {
            "neuron": neuron, "SWC Run Length": neuron_cable.get(neuron, np.nan),
            "# Splits": max(len(neuron_segments[neuron]) - 1, 0),
            "# Merges": counts_by_neuron["# Geometric Sites"][neuron] if sites_available else np.nan,
            "% Split Edges": 100.0 * classes.get(cl.EDGE_SPLIT, 0) / total if total else 0.0,
            "% Omit Edges": 100.0 * classes.get(cl.EDGE_OMIT, 0) / total if total else 0.0,
            "% Merged Edges": 100.0 * merged_edges[neuron] / total if total else 0.0,
            "Edge Accuracy": 100.0 * classes.get(cl.EDGE_CORRECT, 0) / total if total else 0.0,
        }
        row.update({name: counts[neuron] if sites_available else np.nan
                    for name, counts in counts_by_neuron.items()})
        rows.append(row)
    if not rows:
        raise ValueError("GT graph has no edges for per-neuron verification")
    frame = pd.DataFrame(rows).set_index("neuron").sort_index()
    summary = cl.summarize(graph, node_label, edge_error, merge_label_set=merge_labels)
    summary["n_merge_labels"] = len(merge_labels)
    return frame, {"available": sites_available, "groups": groups, "counts": site_counts}, summary


def compare_metrics(cache_frame, canonical_frame):
    for name, frame in (("cache", cache_frame), ("canonical", canonical_frame)):
        if not frame.index.is_unique:
            raise ValueError(f"Duplicate neurons in {name} table")
        missing = set(COMPARE_COLS + ["SWC Run Length"]) - set(frame.columns)
        if missing:
            raise ValueError(f"{name} table lacks columns: {sorted(missing)}")
    common = sorted(set(cache_frame.index) & set(canonical_frame.index))
    if not common:
        raise ValueError("No shared neurons between cache and canonical results")
    comparison = pd.DataFrame(index=common)
    for column in COMPARE_COLS:
        comparison[(column, "cache")] = cache_frame.loc[common, column]
        comparison[(column, "canon")] = canonical_frame.loc[common, column]
        comparison[(column, "diff")] = cache_frame.loc[common, column] - canonical_frame.loc[common, column]
    comparison.columns = pd.MultiIndex.from_tuples(comparison.columns)
    return comparison


def correlation(first, second):
    first, second = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    finite = np.isfinite(first) & np.isfinite(second)
    first, second = first[finite], second[finite]
    if len(first) < 2 or np.ptp(first) == 0 or np.ptp(second) == 0:
        return float("nan")
    return float(np.corrcoef(first, second)[0, 1])


def consistency_checks(cache_frame, canonical_frame, comparison, site_info):
    checks = []

    def report(name, passed, detail="", hard=True):
        status = ("PASS" if passed else "FAIL") if hard else ("OK" if passed else "WARN")
        print(f"[{status}] {name}" + (f" -> {detail}" if detail else ""), flush=True)
        checks.append({"name": name, "status": status})

    only_cache = sorted(set(cache_frame.index) - set(canonical_frame.index))
    only_canonical = sorted(set(canonical_frame.index) - set(cache_frame.index))
    report("same GT neuron set", not only_cache and not only_canonical,
           f"cache-only={only_cache}, canon-only={only_canonical}")
    omit = comparison[("% Omit Edges", "diff")]
    report("% Omit Edges: cache >= canonical (filtering inflates omits)", bool((omit >= -0.5).all()),
           f"min diff={omit.min():+.2f}, mean={omit.mean():+.2f}")
    for column in ["Edge Accuracy", "% Split Edges", "# Splits", "% Merged Edges"]:
        value = correlation(comparison[(column, "cache")], comparison[(column, "canon")])
        report(f"{column}: cache vs canonical correlated (r > 0.7)", value > 0.7,
               f"r={value:.3f}", hard=False)
    merged = comparison[("% Merged Edges", "diff")]
    report("% Merged Edges: mean |diff| within 5 pts", merged.abs().mean() < 5,
           f"mean |diff|={merged.abs().mean():.2f}", hard=False)
    counts = site_info["counts"]
    if site_info["available"]:
        report("combined = geometric + two-GT-only",
               counts["# Combined Sites"] == counts["# Geometric Sites"] + counts["# Two-GT-only Sites"])
        report("shared-evidence sites are a subset of geometric sites",
               counts["# Shared-evidence Sites"] <= counts["# Geometric Sites"])
        difference = comparison[("# Merges", "diff")]
        report("# Merges (geometric only): cache <= canonical", bool((difference <= 0).all()),
               f"max diff={difference.max():+.0f}; investigate geometry/dedup differences if exceeded", hard=False)
        report("# Merges (geometric only): mean |diff| per neuron < 1.0", difference.abs().mean() < 1,
               f"mean |diff|={difference.abs().mean():.2f}", hard=False)
        total_cache = int(comparison[("# Merges", "cache")].sum())
        total_canonical = int(comparison[("# Merges", "canon")].sum())
        relative = abs(total_cache - total_canonical) / max(total_canonical, 1)
        report("# Merges (geometric only): total within 15% of canonical", relative < 0.15,
               f"cache={total_cache}, canon={total_canonical}, rel diff={relative:.1%}", hard=False)
        print(f"Site totals (whole cache): {counts}", flush=True)
        print("No canonical upper-bound check is applied to combined sites; shared evidence is not additive.", flush=True)
    else:
        print("SKIP: merge_sites is missing; site counts are unavailable, not zero.", flush=True)
    return checks


def make_scatter(comparison, brain, mcl):
    figure = Figure(figsize=(4 * len(COMPARE_COLS), 4))
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, len(COMPARE_COLS))
    for axis, column in zip(axes, COMPARE_COLS):
        horizontal = comparison[(column, "canon")].to_numpy(dtype=float)
        vertical = comparison[(column, "cache")].to_numpy(dtype=float)
        finite = np.isfinite(horizontal) & np.isfinite(vertical)
        if finite.any():
            horizontal, vertical = horizontal[finite], vertical[finite]
            axis.scatter(horizontal, vertical, s=20, alpha=0.7)
            lower, upper = float(min(horizontal.min(), vertical.min())), float(max(horizontal.max(), vertical.max()))
            if lower == upper:
                lower, upper = lower - 0.5, upper + 0.5
            axis.plot([lower, upper], [lower, upper], "--", color="grey", lw=1)
        else:
            axis.text(0.5, 0.5, "No comparable data", ha="center", transform=axis.transAxes)
        axis.set_xlabel("canonical")
        axis.set_ylabel("cache (_add)")
        axis.set_title("# Merges (geometric only)" if column == "# Merges" else column, fontsize=10)
    figure.suptitle(f"Brain {brain}, mcl{mcl}: cache vs canonical; merge counts use geometric sites only")
    figure.tight_layout()
    return figure


def write_results(cache_frame, canonical_frame, comparison, site_info, brain, mcl,
                  segmentation_path, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    common = list(comparison.index)
    per_neuron = pd.DataFrame(index=common)
    per_neuron.index.name = "neuron"
    per_neuron.insert(0, "brain_id", brain)
    per_neuron.insert(1, "min_cable_length", mcl)
    per_neuron["merge_count_definition"] = MERGE_COUNT_DEFINITION
    per_neuron["merge_sites_available"] = site_info["available"]
    per_neuron["site_attribution"] = SITE_ATTRIBUTION
    for name, frame in (("cache", cache_frame), ("canon", canonical_frame)):
        per_neuron[f"SWC Run Length_{name}"] = frame.loc[common, "SWC Run Length"]
    for column in COMPARE_COLS:
        for kind in ("cache", "canon", "diff"):
            per_neuron[f"{column}_{kind}"] = comparison[(column, kind)]
    for column in site_info["groups"]:
        per_neuron[f"{column}_cache"] = cache_frame.loc[common, column]
    counts = site_info["counts"]
    summary = {
        "brain_id": brain, "min_cable_length": mcl, "n_neurons": len(common),
        "segmentation_path": segmentation_path, "merge_count_definition": MERGE_COUNT_DEFINITION,
        "merge_sites_available": site_info["available"], "site_attribution": SITE_ATTRIBUTION,
        "site_total_scope": "whole_cache", "n_geometric_sites": counts["# Geometric Sites"],
        "n_two_gt_only_sites": counts["# Two-GT-only Sites"],
        "n_shared_evidence_sites": counts["# Shared-evidence Sites"],
        "n_combined_sites": counts["# Combined Sites"],
        "n_sites_outside_common_neurons": (sum(site["gt_neuron"] not in common
            for site in site_info["groups"]["# Combined Sites"]) if site_info["available"] else np.nan),
    }
    for column in COMPARE_COLS:
        difference = comparison[(column, "diff")]
        safe = column.strip("# ").replace("% ", "pct_").replace(" ", "_").lower()
        summary[f"{safe}_mean_diff"] = float(difference.mean())
        summary[f"{safe}_median_diff"] = float(difference.median())
        summary[f"{safe}_max_abs_diff"] = float(difference.abs().max())
        summary[f"{safe}_corr"] = correlation(comparison[(column, "cache")], comparison[(column, "canon")])
    summary["same_neuron_set"] = set(cache_frame.index) == set(canonical_frame.index)
    summary["omit_inflated_cache_side"] = bool((comparison[("% Omit Edges", "diff")] >= -0.5).all())
    summary["merged_mean_abs_diff_within_5"] = bool(comparison[("% Merged Edges", "diff")].abs().mean() < 5)
    stem = f"{brain}_mcl{mcl}"
    paths = [output_dir / f"{stem}_{suffix}" for suffix in
             ("per_neuron.csv", "summary.csv", "scatter.png")]
    figure = make_scatter(comparison, brain, mcl)
    try:
        figure.savefig(paths[2], dpi=120, bbox_inches="tight")
        per_neuron.to_csv(paths[0])
        pd.DataFrame([summary]).set_index("brain_id").to_csv(paths[1])
    finally:
        figure.clear()
    for path in paths:
        print(f"wrote: {path}", flush=True)
    return summary


def discover_caches(cache_dir, mcl, brains=None):
    selected = []
    wanted = set(map(str, brains)) if brains is not None else None
    for path in sorted(Path(cache_dir).glob("dataset_cache_*_add.pkl")):
        match = CACHE_PATTERN.fullmatch(path.name)
        if match and int(match[2]) == mcl and (wanted is None or match[1] in wanted):
            selected.append((match[1], path))
    if wanted is not None:
        missing = wanted - {brain for brain, path in selected}
        if missing:
            raise ValueError(f"No mcl{mcl} _add.pkl for requested brains: {sorted(missing)}")
    if not selected:
        raise ValueError(f"No mcl{mcl} _add.pkl caches found in {cache_dir}")
    return selected


def verify_brain(cache_path, canonical_path, brain, mcl, output_dir):
    canonical_frame = pd.read_csv(canonical_path, index_col=0).sort_index()
    with Path(cache_path).open("rb") as handle:
        payload = pickle.load(handle)
    try:
        if float(payload.get("min_cable_length", mcl)) != mcl:
            raise ValueError(f"Cache min_cable_length disagrees with mcl{mcl} filename")
        cache_frame, site_info, label_summary = summarize_payload(payload)
        segmentation_path = payload.get("segmentation_path")
    finally:
        del payload
        gc.collect()
    print(f"Whole-brain label summary: {label_summary}", flush=True)
    print("total_merges above is a node-count statistic, not a site count.", flush=True)
    comparison = compare_metrics(cache_frame, canonical_frame)
    print(comparison.rename(columns={"# Merges": "# Merges (geometric only)"}, level=0).to_string(), flush=True)
    checks = consistency_checks(cache_frame, canonical_frame, comparison, site_info)
    summary = write_results(cache_frame, canonical_frame, comparison, site_info, brain, mcl,
                            segmentation_path, output_dir)
    return summary, checks


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mcl", type=int, default=100, help="One shared MCL for this run (default: 100)")
    parser.add_argument("--brain", nargs="+", help="Optional brain IDs; default: all matching caches")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "cache")
    parser.add_argument("--metrics-dir", type=Path, default=ROOT / "metrics_out")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "notebooks" / "verify_stats")
    args = parser.parse_args(argv)
    if args.mcl < 0:
        parser.error("--mcl must be nonnegative")
    try:
        caches = discover_caches(args.cache_dir, args.mcl, args.brain)
    except ValueError as error:
        parser.error(str(error))
    print(f"Selected mcl{args.mcl} brains: {', '.join(brain for brain, path in caches)}", flush=True)
    print("Read-only caches/canonical inputs; matching verification artifacts are overwritten.", flush=True)
    completed, skipped, failures = [], [], []
    for brain, cache_path in caches:
        print(f"\n=== brain {brain}, mcl{args.mcl} ===", flush=True)
        matches = sorted((args.metrics_dir / brain).glob("*/results.csv"))
        if not matches:
            print(f"SKIP: no canonical results.csv under {args.metrics_dir / brain}; no cache loaded", flush=True)
            skipped.append(brain)
            continue
        if len(matches) != 1:
            print(f"ERROR: multiple canonical results.csv for {brain}; use a metrics directory with one run per brain", flush=True)
            failures.append(brain)
            continue
        try:
            summary, checks = verify_brain(cache_path, matches[0], brain, args.mcl, args.output_dir)
            completed.append(brain)
            if any(check["status"] == "FAIL" for check in checks):
                failures.append(brain)
        except Exception as error:
            print(f"ERROR: {brain}: {type(error).__name__}: {error}", flush=True)
            failures.append(brain)
        finally:
            gc.collect()
    print(f"\nFinished: {len(completed)} written, {len(skipped)} skipped, {len(failures)} failed.", flush=True)
    if skipped:
        print(f"Skipped brains: {', '.join(skipped)}", flush=True)
    if failures:
        print(f"Failed brains: {', '.join(failures)}", flush=True)
    return 1 if failures or not completed else 0


if __name__ == "__main__":
    raise SystemExit(main())