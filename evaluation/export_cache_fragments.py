"""Write fragments of a brain's labeled cache as SWCs that the capsule's evaluation can read.

The candidates of this repository point at nodes of the cache's fragment graph
(``cache/dataset_cache_<brain>_mcl<N>_add.pkl``): fragments of at least the minimum cable
length, resampled to the cache's node spacing. An edit can only be applied to that graph, so
the capsule has to evaluate that graph, before and after an edit. This script writes the
selected fragments of the cache, unedited, one SWC per connected component in world um, named
``<segment id>.<k>.swc`` as the capsule groups them. ``cut_merge_sites.py`` and
``join_split_candidates.py`` write an edited graph with the same functions.

The selection is a list of segment IDs, normally ``segments_under_gt.txt`` of a run of
``run_capsule_original.py`` on the brain's raw SWCs: the segments that label a ground-truth
node. Segments of the list that the cache does not hold (shorter than the minimum cable
length) are counted and left out.

Run on a compute node in panda, from the repository root:

    python -u evaluation/export_cache_fragments.py --brain 841260 \\
        --segments evaluation/results/<run>/segments_under_gt.txt \\
        --out evaluation/results/<experiment>/cache_fragments.zip

Then evaluate the file with ``run_capsule_original.py --fragments <zip>`` in arborist_eval.
"""
import argparse
import json
import pickle
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--brain", required=True)
    parser.add_argument("--mcl", type=int, default=100)
    parser.add_argument("--segments", required=True, type=Path, help="Text file, one segment ID per line")
    parser.add_argument("--out", required=True, type=Path, help="ZIP file to write")
    return parser.parse_args()


def load_cache(brain, mcl):
    """The cache file of a brain and its payload."""
    import agentic_neuron_proofreader  # noqa: F401  (defines the cache's SkeletonGraph)

    cache = REPO / f"cache/dataset_cache_{brain}_mcl{mcl}_add.pkl"
    started = time.time()
    with cache.open("rb") as stream:
        payload = pickle.load(stream)
    graph = payload["fragments_graph"]
    print(f"cache {cache.name} loaded in {time.time() - started:.0f} s: "
          f"{len(graph.component_id_to_swc_id)} fragment components, {len(graph.node_xyz)} nodes", flush=True)
    return cache, payload


def read_segments(path):
    return {line.strip() for line in Path(path).read_text().splitlines() if line.strip()}


def segment_of(swc_id):
    return str(swc_id).split(".")[0]


def component_swc(graph, root):
    """One connected component as SWC text in world um: ``id type x y z radius parent``, the
    root first, and its nodes. (The graph class's own writer calls a helper that this package
    does not have.)"""
    import networkx as nx

    xyz, radius = graph.node_xyz, graph.node_radius
    index = {root: 1}
    x, y, z = xyz[root]
    lines = ["# id, type, x, y, z, r, pid", f"1 2 {x:.3f} {y:.3f} {z:.3f} {radius[root]:.3f} -1"]
    for parent, child in nx.dfs_edges(graph, source=root):
        index[child] = len(index) + 1
        x, y, z = xyz[child]
        lines.append(f"{index[child]} 2 {x:.3f} {y:.3f} {z:.3f} {radius[child]:.3f} {index[parent]}")
    return "\n".join(lines) + "\n", list(index)


def write_fragments(graph, wanted, out):
    """Write every connected component of the wanted segments to the ZIP file ``out``.

    The graph may have been edited with networkx's own ``remove_edge`` / ``remove_nodes_from``:
    the node arrays and ``node_component_id`` still describe the cache, so a component that an
    edit left whole keeps the cache's name, and the pieces of a component that an edit touched
    get new names ``<segment id>.<k>`` with numbers the segment has not used. A piece of one
    node is not written. Returns the counts.
    """
    component_of = np.asarray(graph.node_component_id)
    names = graph.component_id_to_swc_id
    component_ids, sizes = np.unique(component_of, return_counts=True)
    size_of = dict(zip(component_ids.tolist(), sizes.tolist()))
    wanted_ids = [component_id for component_id in component_ids.tolist() if segment_of(names[component_id]) in wanted]
    held = {segment_of(names[component_id]) for component_id in wanted_ids}
    next_number = {}                                   # per segment: the first unused <k>
    for component_id in wanted_ids:
        segment, _, number = str(names[component_id]).partition(".")
        next_number[segment] = max(next_number.get(segment, 0), int(number) + 1 if number.isdigit() else 0)

    out.parent.mkdir(parents=True, exist_ok=True)
    seen = np.zeros(len(component_of), dtype=bool)
    written, counts = set(), {"components_written": 0, "nodes_written": 0, "components_unchanged": 0,
                              "pieces_of_edited_components": 0, "single_nodes_dropped": 0}
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for root in np.flatnonzero(np.isin(component_of, wanted_ids)).tolist():
            if seen[root] or not graph.has_node(root):
                continue
            text, nodes = component_swc(graph, root)
            seen[nodes] = True
            component_id = int(component_of[root])
            if len(nodes) == size_of[component_id]:
                name = str(names[component_id])
                counts["components_unchanged"] += 1
            elif len(nodes) < 2:
                counts["single_nodes_dropped"] += 1
                continue
            else:
                segment = segment_of(names[component_id])
                name = f"{segment}.{next_number[segment]}"
                next_number[segment] += 1
                counts["pieces_of_edited_components"] += 1
            if name in written:
                sys.exit(f"two components share the SWC name {name}")
            written.add(name)
            archive.writestr(f"{name}.swc", text)
            counts["components_written"] += 1
            counts["nodes_written"] += len(nodes)
    return {"segments_requested": len(wanted), "segments_in_cache": len(held),
            "segments_not_in_cache": len(wanted - held), **counts,
            "coordinates": "world um (x, y, z)", "out": str(out.resolve())}


def cache_record(brain, cache, payload):
    return {"brain": brain, "cache": str(cache), "min_cable_length_um": payload.get("min_cable_length"),
            "node_spacing_um": payload.get("node_spacing"), "fragments_path": payload.get("fragments_path")}


def main():
    options = parse_args()
    cache, payload = load_cache(options.brain, options.mcl)
    summary = {**cache_record(options.brain, cache, payload),
               **write_fragments(payload["fragments_graph"], read_segments(options.segments), options.out)}
    options.out.with_suffix(".json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
