#!/usr/bin/env python
"""Sampled image/fragment alignment audit for brains without native candidate tables.

``proofreader_evolve/cli/check_image_alignment.py`` samples detector candidate
rows, so it needs prepared feature tables. Newly added brains only have a
``dataset_cache_<brain>_mcl<N>_add.pkl``. This script samples fragment nodes
directly from that cache instead, then reuses the same trusted image reader,
overlay renderer and diagnostic statistics, and writes receipts in the same
format to ``configs/image_alignment.json``. It never reads GT labels, trains or
evaluates anything.

Two steps, like the CLI it mirrors:

    # 1. read patches, save overlays + measurements (status needs_visual_review)
    python -u scripts/check_image_alignment_from_cache.py --brains 751473 \
        --out docs/image_alignment_review_20261006
    # 2. after visually inspecting every saved PNG, record the review
    python -u scripts/check_image_alignment_from_cache.py --brains 751473 \
        --out docs/image_alignment_review_20261006 --record-reviewed \
        --review-note "..."

Sampling is label-blind: seeded random fragment nodes on components with at
least ``--min-component-nodes`` nodes, then farthest-point selection starting
from the node closest to the median position, as in the native CLI.
"""
import argparse
from datetime import datetime
import gc
import json
import os
from pathlib import Path
import pickle
import socket
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS",
                      str(PROJECT_ROOT / "configs" / "allen-nd-goog-f5d46dbfa2cd.json"))
os.environ.setdefault("AWS_EC2_METADATA_DISABLED", "true")

from proofreader_evolve.harness.image_context import DEFAULT_ALIGNMENT, ImageSource  # noqa: E402
from proofreader_evolve.harness.image_preview import alignment_statistics, render_preview  # noqa: E402
from proofreader_evolve.harness.local_context import cache_identity  # noqa: E402


def load_fragments(path):
    """Fragment graph and image metadata only; GT fields are dropped unread."""
    with open(path, "rb") as f:
        payload = pickle.load(f)
    return payload["fragments_graph"], payload["img_path"], tuple(payload["anisotropy"])


def representative_nodes(graph, geometry, count, min_component_nodes, radius_um, seed):
    sizes = np.bincount(graph.node_component_id)
    eligible = np.flatnonzero(sizes[graph.node_component_id] >= min_component_nodes)
    # Keep the whole patch inside the image domain.
    voxels = geometry.graph_to_voxel(graph.node_xyz[eligible].astype(float))
    margin = radius_um / geometry.scale_xyz_um[::-1] + 2
    inside = np.all((voxels >= margin) & (voxels < np.asarray(geometry.shape_zyx) - margin), axis=1)
    eligible = eligible[inside]
    if not len(eligible):
        raise ValueError("No fragment nodes on large components inside the image domain")
    rng = np.random.default_rng(seed)
    pool = rng.choice(eligible, size=min(512, len(eligible)), replace=False)
    centers = graph.node_xyz[pool].astype(float)
    selected = [int(np.argmin(np.linalg.norm(centers - np.median(centers, axis=0), axis=1)))]
    while len(selected) < min(count, len(pool)):
        distances = np.min(np.linalg.norm(centers[:, None] - centers[selected][None], axis=2), axis=1)
        distances[selected] = -1
        selected.append(int(np.argmax(distances)))
    return [int(pool[i]) for i in selected]


def local_skeleton(graph, tree, anchor, radius_um):
    """Fragment nodes/edges near the anchor (graph xyz microns, local indices)."""
    nearby = sorted(tree.query_ball_point(graph.node_xyz[anchor].astype(float), 1.5 * radius_um))
    local = {node: i for i, node in enumerate(nearby)}
    edges = [(local[a], local[b]) for a, b in graph.subgraph(nearby).edges]
    return (graph.node_xyz[nearby].astype(float), edges,
            graph.node_component_id[nearby].astype(int))


def audit(args):
    from scipy.spatial import cKDTree
    spec = {"radius_um": args.radius_um, "level": args.level, "channel": 0, "timepoint": 0}
    for brain in args.brains:
        directory = args.out / brain
        directory.mkdir(parents=True, exist_ok=True)
        cache = args.cache_dir / f"dataset_cache_{brain}_mcl{args.mcl}_add.pkl"
        report = {"brain": brain, "host": socket.gethostname(), "status": "unavailable", "spec": spec,
                  "sampling": {"method": "label-blind fragment nodes (no candidate tables)",
                               "min_component_nodes": args.min_component_nodes, "seed": args.seed},
                  "cases": []}
        graph = tree = None
        try:
            print(f"[{brain}] loading {cache}", flush=True)
            identity = cache_identity(cache)
            graph, img_path, anisotropy = load_fragments(cache)
            source = ImageSource(img_path)
            geometry = source.level(args.level)
            if not np.allclose(anisotropy, source.level(0).scale_xyz_um):
                raise ValueError(f"Cache anisotropy {anisotropy} != image spacing "
                                 f"{source.level(0).scale_xyz_um.tolist()}; explicit registration required")
            report.update(source_cache=identity, image_identity=source.identity,
                          image_uri=source.root, geometry=geometry.summary())
            tree = cKDTree(graph.node_xyz)
            anchors = representative_nodes(graph, geometry, args.per_brain, args.min_component_nodes,
                                           args.radius_um, args.seed)
            for anchor in anchors:
                case = {"kind": "fragment", "node": anchor, "occurrence": 0}
                try:
                    xyz = graph.node_xyz[[anchor]].astype(float)
                    patch, metadata = source.read(xyz, **spec)
                    nodes_xyz, edges, segments = local_skeleton(graph, tree, anchor, args.radius_um)
                    nodes = geometry.graph_to_voxel(nodes_xyz) - np.asarray(metadata["origin_zyx"])
                    preview = directory / f"fragment_{anchor}.png"
                    render_preview(preview, patch, nodes, edges, segments,
                                   f"{brain} / fragment node {anchor}")
                    case.update(preview=str(preview.resolve()), metadata=metadata,
                                statistics=alignment_statistics(patch, nodes, edges))
                    print(f"[{brain}] node {anchor}: saved {preview}", flush=True)
                except Exception as exc:
                    case["error"] = f"{type(exc).__name__}: {exc}"
                    print(f"[{brain}] node {anchor}: {case['error']}", flush=True)
                report["cases"].append(case)
            report["status"] = ("needs_visual_review" if not any(c.get("error") for c in report["cases"])
                                else "incomplete")
            report["interpretation"] = ("Sampled image/fragment registration check, not a whole-brain "
                                        "guarantee or error-label audit")
        except Exception as exc:
            report["error"] = f"{type(exc).__name__}: {exc}"
            print(f"[{brain}] {report['error']}", flush=True)
        (directory / "alignment.json").write_text(json.dumps(report, indent=2, allow_nan=False))
        graph = tree = None
        gc.collect()


def record(args):
    notes = json.loads(args.review_notes.read_text()) if args.review_notes else {}
    registry = (json.loads(args.alignment.read_text()) if args.alignment.exists()
                else {"version": 1, "brains": {}})
    for brain in args.brains:
        note = notes.get(brain, args.review_note).strip()
        if not note:
            raise ValueError(f"No review note for {brain}; record only after inspecting the overlays")
        report = json.loads((args.out / brain / "alignment.json").read_text())
        if report["status"] != "needs_visual_review" or any(c.get("error") for c in report["cases"]):
            raise ValueError(f"{brain}: an incomplete/failed audit cannot be recorded as reviewed")
        if cache_identity(report["source_cache"]["path"]) != report["source_cache"]:
            raise ValueError(f"{brain}: cache changed since the audit; rerun it")
        entry = {k: report[k] for k in ("source_cache", "image_identity", "image_uri")}
        entry.update(status="reviewed", review_note=note,
                     graph_to_world_um=report["geometry"]["graph_to_world_um"],
                     evidence=str((args.out / brain / "alignment.json").resolve()),
                     reviewed_at=datetime.now().astimezone().isoformat(),
                     checked_level=report["spec"]["level"])
        registry["brains"][brain] = entry
    args.alignment.write_text(json.dumps(registry, indent=2, allow_nan=False))
    print(f"Saved reviewed alignment receipts for {', '.join(args.brains)}: {args.alignment}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--brains", nargs="+", required=True)
    parser.add_argument("--mcl", type=int, default=100)
    parser.add_argument("--cache-dir", type=Path, default=PROJECT_ROOT / "cache")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--per-brain", type=int, default=6)
    parser.add_argument("--radius-um", type=float, default=40.)
    parser.add_argument("--level", type=int, default=0)
    parser.add_argument("--min-component-nodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--alignment", type=Path, default=DEFAULT_ALIGNMENT)
    parser.add_argument("--record-reviewed", action="store_true")
    parser.add_argument("--review-note", default="")
    parser.add_argument("--review-notes", type=Path, help="JSON {brain: note}; overrides --review-note")
    args = parser.parse_args()
    if not 1 <= args.per_brain <= 20 or not 4 <= args.radius_um <= 100:
        parser.error("per-brain must be 1..20 and radius-um 4..100")
    args.out = args.out.resolve()
    record(args) if args.record_reviewed else audit(args)


if __name__ == "__main__":
    main()
