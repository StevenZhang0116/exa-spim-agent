# Dataset Context

`dataset_cache_794495.pkl` is a cached graph-level representation of ExaSPIM brain `794495`. It should be treated as a pickled Python payload for neuron-skeleton analysis, not as a standalone microscopy image volume, dense segmentation mask, metrics table, or trained proofreader model.

The cache was created from two sets of SWC skeletons associated with brain `794495`:

- Predicted neuron-fragment SWCs derived from the U-Net segmentation `raw.unet_449_splits_and_merges_900000`.
- Manually traced ground-truth SWCs from the corresponding ground-truth tracing collection.

During cache creation, both SWC collections were converted into `SkeletonGraph` objects. The predicted-fragment skeletons were filtered to remove short components below `1000` microns of cable length, resampled to approximately `5` microns between neighboring skeleton nodes, and stored using ExaSPIM anisotropy `(0.748, 0.748, 1.0)` in `(x, y, z)` microns per voxel. The raw ExaSPIM image path is saved, but the raw image data itself is not embedded in the pickle.

At the file level, the pickle is expected to contain a dictionary-like payload with the following keys:

| Key | Meaning |
|---|---|
| `fragments_path` | Cloud path to the source SWC files for predicted U-Net fragments. |
| `gt_path` | Cloud path to the source SWC files for manually traced ground-truth neurons. |
| `img_path` | Cloud path to the raw ExaSPIM image volume. This is a pointer only; the image volume is not stored in the pickle. |
| `anisotropy` | Voxel-to-physical scaling in `(x, y, z)` microns per voxel. For this cache: `(0.748, 0.748, 1.0)`. |
| `min_cable_length` | Minimum skeleton cable length used when loading predicted fragments. For this cache: `1000` microns. |
| `node_spacing` | Approximate resampling interval between neighboring graph nodes. For this cache: `5` microns. |
| `fragments_graph` | A `SkeletonGraph` object representing predicted U-Net skeleton fragments. |
| `gt_graph` | A `SkeletonGraph` object representing manually traced ground-truth skeletons. |

The two graph objects are the main scientific content of the file.

`fragments_graph` is an undirected skeleton graph whose connected components correspond to predicted U-Net neuron fragments after filtering and resampling. A connected component should not be assumed to be a complete biological neuron: one real neuron may be split across many predicted components, and one predicted component may contain a merge error if it overlaps multiple real neurons. The cache-generation notebook reported the following scale for this graph:

- Connected components: `10,172`
- Nodes: `4,281,310`
- Edges: `4,271,138`

`gt_graph` has the same graph representation but stores manually traced skeletons. Its connected components correspond to the available manual neuron traces. These traces are sparse skeleton annotations, not dense voxel-level labels for the entire brain. The cache-generation notebook reported:

- Connected components: `19`
- Nodes: `1,363,808`
- Edges: `1,363,789`

Each `SkeletonGraph` is a NetworkX-style undirected graph with additional arrays and helper methods. Node IDs are integer graph nodes. The most important graph attributes are:

| Attribute or method | Meaning |
|---|---|
| `graph.nodes` | Iterable of integer node IDs. |
| `graph.edges()` | Iterable of undirected skeleton edges between node IDs. |
| `graph.node_xyz` | NumPy array of shape `(N, 3)` storing node coordinates in physical `(x, y, z)` microns. |
| `graph.node_radius` | NumPy array of shape `(N,)` storing SWC radius values for nodes. |
| `graph.node_component_id` | NumPy array of shape `(N,)` mapping each node to a connected-component ID. |
| `graph.component_id_to_swc_id` | Dictionary mapping connected-component IDs back to source SWC identifiers. |
| `graph.kdtree` | SciPy KDTree built over `node_xyz`, useful for nearest-neighbor spatial queries. |
| `graph.anisotropy` | NumPy array containing physical voxel spacing in `(x, y, z)` order. |
| `graph.node_voxel(node_id)` | Converts a graph node from physical coordinates to image voxel coordinates in `(z, y, x)` order. |
| `graph.nodes_in_patch(offset, patch_shape, ...)` | Returns graph nodes whose voxel coordinates fall inside a 3D patch. The `offset` and `patch_shape` are in `(z, y, x)` order. |
| `graph.edges_in_patch(offset, patch_shape, ...)` | Returns graph edges whose endpoints both fall inside a 3D patch, also in local `(z, y, x)` voxel coordinates. |
| `graph.leaf_nodes()` | Returns graph endpoints with degree `1`; these are useful for split-reconnection proposal generation. |
| `graph.branching_nodes()` | Returns nodes with degree greater than `2`. |
| `graph.nodes_with_component_id(component_id)` | Returns all node IDs assigned to a connected component. |
| `graph.node_swc_id(node_id)` | Returns the source SWC identifier associated with a node's component. |
| `graph.node_segment_id(node_id)` | For predicted fragments, returns the segment-like prefix parsed from the SWC identifier. |
| `graph.summary(prefix=...)` | Returns a human-readable count summary. |

Coordinate conventions are important. `node_xyz` uses physical `(x, y, z)` coordinates in microns, while image patches and array indexing use voxel `(z, y, x)` coordinates. The helper methods `node_voxel`, `nodes_in_patch`, and `edges_in_patch` handle this conversion using the stored anisotropy. Patch-local coordinates returned by `nodes_in_patch` and `edges_in_patch` are relative to the patch origin, not global image coordinates.

Known gaps and limitations:

- The pickle does not contain the raw ExaSPIM intensity volume. It stores `img_path`, which can be used to reopen the image lazily if the required code, TensorStore support, and cloud credentials are available.
- The pickle does not contain the dense U-Net segmentation volume. The segmentation path can often be inferred from `fragments_path` by removing the trailing `/swcs`, but it is not a separate cached array in this file.
- The pickle does not contain the original unfiltered SWC collection. Very short predicted fragments below the configured `min_cable_length` were excluded before caching.
- The skeletons are resampled graph representations. They preserve topology, coordinates, radii, component IDs, and source SWC IDs, but not every original SWC row exactly as collected.
- The manual ground truth is sparse skeleton ground truth for a limited set of traced neurons, not a complete dense annotation of the whole brain.
- The graph is undirected. SWC parent-child orientation is not the primary analysis representation inside the cache.
- Loading the pickle requires the Python environment to have the `agentic_neuron_proofreader` package, or compatible class definitions for the pickled `SkeletonGraph` objects. Direct graph-only loading avoids opening cloud image data, while `BrainDataset.load_from_cache(...)` rebuilds the lazy image reader from `img_path`.

# Intent

Use `dataset_cache_794495.pkl` as a graph-structured substrate for exploring automated and agentic proofreading of whole-brain ExaSPIM neuron reconstructions. The exploration should focus on how predicted skeleton fragments, manual skeleton traces, local geometry, graph topology, and optional image or segmentation evidence can inform decisions about reconnecting split fragments, rejecting false continuations, detecting likely merge regions, and deciding when more evidence is needed.

The system should treat `fragments_graph` as the primary object to be improved and `gt_graph` as sparse manual reference structure for evaluation, calibration, and qualitative analysis. Because predicted connected components are fragments rather than verified neurons, useful exploration should pay special attention to leaf endpoints, nearby endpoint pairs, component boundaries, ambiguous regions where multiple components are spatially close, and cases where a local graph edit may improve one error type while worsening another.

The preferred style of exploration is iterative and state-aware rather than one-shot classification. Useful directions include generating candidate reconnections from leaf nodes, comparing competing proposals from the same endpoint, refreshing candidates after graph edits, using uncertainty to defer ambiguous joins, preserving global graph consistency, and coordinating split repair with merge detection. Evaluation should prioritize skeleton-reconstruction behavior: split count, merge count, omitted edges, edge accuracy, expected run length, normalized ERL, proposal precision, proposal recall, and robustness across dense or noisy fragment neighborhoods.

Avoid assuming that the pickle contains all data needed for dense image-based validation. The cache is most reliable as a skeleton graph representation. Raw image patches and dense segmentation patches require external cloud access through the saved paths. Hypotheses should therefore be compatible with graph-only analysis first, while allowing optional image or segmentation evidence when available.

The following sample code illustrates how an exploration system can load and reconstruct useful views from the pickle.

```python
# Basic graph-only loading.
# This requires the agentic_neuron_proofreader package, or compatible class
# definitions, because the pickle contains SkeletonGraph objects.

import pickle
import networkx as nx
import numpy as np
import pandas as pd

cache_path = "dataset_cache_794495.pkl"

with open(cache_path, "rb") as f:
    payload = pickle.load(f)

print(payload.keys())

fragments = payload["fragments_graph"]
gt = payload["gt_graph"]

print("anisotropy:", payload["anisotropy"])
print("min_cable_length:", payload["min_cable_length"])
print("node_spacing:", payload["node_spacing"])
print("fragments_path:", payload["fragments_path"])
print("gt_path:", payload["gt_path"])
print("img_path:", payload["img_path"])

print(fragments.summary(prefix="Fragments"))
print(gt.summary(prefix="GroundTruth"))
```

```python
# Build compact graph inventories.

def graph_inventory(graph, name):
    return {
        "name": name,
        "n_nodes": graph.number_of_nodes(),
        "n_edges": graph.number_of_edges(),
        "n_components": nx.number_connected_components(graph),
        "n_leaf_nodes": len(graph.leaf_nodes()),
        "n_branching_nodes": len(graph.branching_nodes()),
        "xyz_min_um": graph.node_xyz.min(axis=0),
        "xyz_max_um": graph.node_xyz.max(axis=0),
    }

inventory = pd.DataFrame([
    graph_inventory(fragments, "fragments_graph"),
    graph_inventory(gt, "gt_graph"),
])

print(inventory)
```

```python
# Reconstruct a component-level table from the cached graph arrays.
# This is useful for finding large fragments, tiny fragments, highly branched
# fragments, or components with many endpoints.

def component_table(graph):
    comp_ids = np.asarray(graph.node_component_id)
    unique_comp_ids, node_counts = np.unique(comp_ids, return_counts=True)

    degree = dict(graph.degree())
    leaf_nodes = np.array([n for n, d in degree.items() if d == 1], dtype=int)
    branch_nodes = np.array([n for n, d in degree.items() if d > 2], dtype=int)

    leaf_counts = pd.Series(comp_ids[leaf_nodes]).value_counts().to_dict()
    branch_counts = pd.Series(comp_ids[branch_nodes]).value_counts().to_dict()

    rows = []
    for cid, n_nodes in zip(unique_comp_ids, node_counts):
        cid = int(cid)
        rows.append({
            "component_id": cid,
            "swc_id": graph.component_id_to_swc_id.get(cid),
            "n_nodes": int(n_nodes),
            "n_leaf_nodes": int(leaf_counts.get(cid, 0)),
            "n_branching_nodes": int(branch_counts.get(cid, 0)),
        })

    return pd.DataFrame(rows).sort_values("n_nodes", ascending=False)

fragment_components = component_table(fragments)
gt_components = component_table(gt)

print(fragment_components.head())
print(gt_components.head())
```

```python
# Reconstruct an endpoint table for split-proofreading exploration.
# Endpoints are degree-1 skeleton nodes. Candidate joins often start from these.

leaf_nodes = np.array(fragments.leaf_nodes(), dtype=int)
endpoint_table = pd.DataFrame({
    "node_id": leaf_nodes,
    "component_id": fragments.node_component_id[leaf_nodes],
    "swc_id": [fragments.node_swc_id(int(n)) for n in leaf_nodes],
    "x_um": fragments.node_xyz[leaf_nodes, 0],
    "y_um": fragments.node_xyz[leaf_nodes, 1],
    "z_um": fragments.node_xyz[leaf_nodes, 2],
})

# Optional: distance from each predicted-fragment endpoint to the nearest
# manually traced skeleton node. This can help identify regions covered by GT.
if getattr(gt, "kdtree", None) is not None:
    d_gt_um, nearest_gt_idx = gt.kdtree.query(fragments.node_xyz[leaf_nodes])
    endpoint_table["nearest_gt_distance_um"] = d_gt_um
    endpoint_table["nearest_gt_node_id"] = nearest_gt_idx
    endpoint_table["nearest_gt_component_id"] = gt.node_component_id[nearest_gt_idx]

print(endpoint_table.head())
```

```python
# Reconstruct local patch views from graph data alone.
# offset and patch_shape are in global voxel coordinates with (z, y, x) order.

patch_shape = (512, 512, 512)
example_node = int(leaf_nodes[0])
center_voxel = fragments.node_voxel(example_node)
offset = tuple(int(c - s // 2) for c, s in zip(center_voxel, patch_shape))

frag_nodes_local, frag_node_ids, frag_node_comps = fragments.nodes_in_patch(
    offset,
    patch_shape,
    return_ids=True,
    return_components=True,
)
frag_edges_local, frag_edge_comps = fragments.edges_in_patch(
    offset,
    patch_shape,
    return_components=True,
)

gt_nodes_local, gt_node_ids, gt_node_comps = gt.nodes_in_patch(
    offset,
    patch_shape,
    return_ids=True,
    return_components=True,
)
gt_edges_local, gt_edge_comps = gt.edges_in_patch(
    offset,
    patch_shape,
    return_components=True,
)

print("center_voxel:", center_voxel)
print("offset:", offset)
print("fragment nodes/edges/components:",
      len(frag_node_ids), len(frag_edges_local), len(np.unique(frag_node_comps)))
print("gt nodes/edges/components:",
      len(gt_node_ids), len(gt_edges_local), len(np.unique(gt_node_comps)))
```

```python
# Reconstruct a single connected component as a NetworkX subgraph.
# This can be used for local graph algorithms, path analysis, plotting, or export.

component_id = int(fragment_components.iloc[0]["component_id"])
component_nodes = np.where(fragments.node_component_id == component_id)[0]
component_subgraph = fragments.subgraph(component_nodes).copy()
component_xyz = fragments.node_xyz[component_nodes]

print("component_id:", component_id)
print("swc_id:", fragments.component_id_to_swc_id.get(component_id))
print("subgraph nodes:", component_subgraph.number_of_nodes())
print("subgraph edges:", component_subgraph.number_of_edges())
print("component xyz bounds:", component_xyz.min(axis=0), component_xyz.max(axis=0))
```

```python
# Optional: export one reconstructed component back to SWC inside a zip file.
# This writes only the selected connected component, not the whole graph.

import zipfile

root = int(component_nodes[0])
with zipfile.ZipFile("example_component_from_cache.zip", "w") as z:
    fragments.component_to_zipped_swc(z, root, use_radius=True)
```

```python
# Optional full BrainDataset reconstruction.
# Use this only when the package dependencies and cloud credentials are available.
# This will rebuild the lazy TensorStore image reader from img_path.

from agentic_neuron_proofreader.data_modules.datasets import BrainDataset

dataset = BrainDataset.load_from_cache(cache_path)
print(dataset.fragments_graph.summary(prefix="Fragments"))
print(dataset.gt_graph.summary(prefix="GroundTruth"))

# Reading an image patch requires access to the cloud image path saved in img_path.
# img_patch = dataset.img.read(center_voxel, patch_shape)
```

```python
# Optional dense segmentation access.
# The dense segmentation volume is not stored in the pickle. If the source path
# is accessible, it can often be inferred from fragments_path by removing /swcs.

from agentic_neuron_proofreader.utils import img_util

fragments_path = payload["fragments_path"].rstrip("/")
segmentation_path = fragments_path.rsplit("/swcs", 1)[0] + "/"

# Requires TensorStore support and cloud credentials.
# segmentation = img_util.TensorStoreImage(segmentation_path)
# seg_patch = segmentation.read(center_voxel, patch_shape)
```
