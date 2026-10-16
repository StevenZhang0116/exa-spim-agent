"""Compare the GT neurons inside a cache with the current GCS tracings.

A base cache is reused silently by the builder, so a GT tracing edited on GCS after
the cache was built would go unnoticed (the 794495 case of 2026-10-13). The check
loads the current SWCs with the cache's own skeleton parameters and compares every
neuron's node count and coordinates.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import functools
import hashlib

import numpy as np


def component_digests(graph):
    """``{swc name: (node count, coordinate digest, centroid)}`` per GT neuron."""
    xyz = np.asarray(graph.node_xyz, dtype=np.float64)
    component = np.asarray(graph.node_component_id)
    out = {}
    for cid, swc in graph.component_id_to_swc_id.items():
        points = xyz[component == int(cid)]
        order = np.lexsort(points.T[::-1])
        digest = hashlib.sha256(np.round(points[order], 3).tobytes()).hexdigest()[:12]
        out[str(swc)] = (int(len(points)), digest, points.mean(axis=0).round(1).tolist())
    return out


def load_current_gt(gt_path, anisotropy, min_cable_length, node_spacing, workers=4):
    """The GT skeletons as the cache builder would load them now (GCS read)."""
    from agentic_neuron_proofreader.data_modules.graph_classes import SkeletonGraph
    from agentic_neuron_proofreader.utils import graph_util

    # The loader's default pool is one worker per neuron; each holds a full copy of
    # the SWC data, so cap it on shared compute nodes.
    original = graph_util.ProcessPoolExecutor
    graph_util.ProcessPoolExecutor = functools.partial(ProcessPoolExecutor, max_workers=int(workers))
    try:
        graph = SkeletonGraph(anisotropy=tuple(anisotropy), min_cable_length=min_cable_length,
                              node_spacing=node_spacing)
        graph.load(gt_path)
    finally:
        graph_util.ProcessPoolExecutor = original
    return graph


def compare_gt(cached_gt, gt_path, anisotropy, min_cable_length, node_spacing, workers=4):
    """Per-neuron comparison of the cached GT with the current GCS tracings."""
    current = load_current_gt(gt_path, anisotropy, min_cable_length, node_spacing, workers)
    old, new = component_digests(cached_gt), component_digests(current)
    report = {"identical": 0, "changed": [], "only_cache": [], "only_gcs": [], "gt_path": gt_path}
    for swc in sorted(set(old) | set(new)):
        o, n = old.get(swc), new.get(swc)
        if o is None:
            report["only_gcs"].append(swc)
        elif n is None:
            report["only_cache"].append(swc)
        elif o[0] == n[0] and o[1] == n[1]:
            report["identical"] += 1
        else:
            report["changed"].append({"neuron": swc, "cache_nodes": o[0], "gcs_nodes": n[0]})
    report["stale"] = bool(report["changed"] or report["only_cache"] or report["only_gcs"])
    return report


def format_report(report) -> str:
    parts = [f"{report['identical']} neurons identical"]
    if report["changed"]:
        parts.append(f"{len(report['changed'])} changed ({', '.join(c['neuron'] for c in report['changed'][:6])})")
    if report["only_cache"]:
        parts.append(f"{len(report['only_cache'])} only in the cache")
    if report["only_gcs"]:
        parts.append(f"{len(report['only_gcs'])} only on GCS")
    return "; ".join(parts)
