# `dataset_cache_794495` — Dataset Description

`dataset_cache_794495.pkl` is a Python pickle holding a cached `BrainDataset`
for **ExaSPIM brain 794495**. It bundles two neuron-skeleton graphs — a human
ground-truth reconstruction and an automated U-Net reconstruction of the same
brain. It is the evaluation target for a project on **agentic, post-hoc
proofreading of whole-brain neuron reconstructions**: correcting the systematic
split/merge errors in the automated reconstruction so the extracted wiring
diagram becomes trustworthy for downstream connectivity analysis.

---

## 1) Dataset context

**Origin.** The data comes from a single mouse brain (specimen ID **794495**)
imaged with **ExaSPIM**, a light-sheet platform that combines tissue expansion
with selective-plane illumination microscopy to achieve sub-micron, whole-brain
fluorescence imaging of sparsely labeled neurons (voxel size
**0.748 × 0.748 × 1.0 µm**). At this resolution individual axons can be traced
across an entire brain, but each brain is tens of terabytes, so fully manual
reconstruction is infeasible and automated segmentation is required.

**What the cache captures.** The cache stores two aligned, graph-based data
products for this brain:

1. **UNet fragment skeletons** (`fragments_graph`) — the automated
   reconstruction. A 3D U-Net predicted an instance-level voxel labeling of the
   brain; each predicted fragment was skeletonized into an SWC tree (3D
   coordinates, radius, connectivity) and loaded into a graph. This is the
   *machine* reconstruction containing the errors to be fixed — roughly
   **10,000 fragments** for this brain.

2. **Ground-truth tracings** (`gt_graph`) — human-traced neuron skeletons for
   **18 neurons**. These are the gold-standard morphologies used to train and
   evaluate the automated reconstruction.

**The errors that motivate the project.** Deep-learning segmentation introduces
two systematic *topological* errors:

- **Split errors** — a single neuron is fragmented into multiple disconnected
  segments (weak signal, imaging artifacts, or branch-point failures).
- **Merge errors** — two distinct neurons are erroneously joined, typically
  where neurites run close together.

Scoring the automated reconstruction against the 18 human tracings gives this
baseline (skeleton-based topology metrics):

| Metric | Value |
|---|---|
| Total Splits | 10,031 |
| Total Merges | 124 |
| Avg. Splits per Neuron | 571.4 |
| Avg. Merges per Neuron | 6.9 |
| Edge Accuracy | 73.78 % |
| % Split Edges | 0.32 % |
| % Omit Edges | 1.36 % |
| % Merged Edges | 24.55 % |
| ERL (Expected Run Length) | 5,090.6 µm |
| Normalized ERL | 0.0095 |

Splits dominate (a true neuron broken into hundreds of pieces); merges are
rarer but individually costly. These *topological* errors — not voxel-level
mislabeling — are what corrupt downstream connectivity, which is why
skeleton-based metrics (splits/neuron, edge accuracy, normalized ERL) are the
appropriate evaluation targets.

**How the data was gathered / known gaps.**

- **Ground truth is sparse and skeleton-only.** Only 18 neurons were traced
  (named like `N001-794495-JT`, `N002-794495-PP`, … — the suffix is the human
  annotator's initials). There is **no dense ground-truth voxel volume**: GT
  exists only as center-line skeletons in `gt_graph`. A region with no traced
  neuron is simply unlabeled, not labeled "background."
- **Fragments are filtered.** When the cache was built, UNet fragments shorter
  than `min_cable_length = 1000` µm of total path length were dropped, removing
  short/noisy false-positive fragments. Skeletons were also resampled to a
  target inter-node spacing of `node_spacing = 5` µm, so node geometry is
  decimated relative to the original SWCs.
- **Anisotropic voxels.** ExaSPIM samples X/Y more finely than Z, so voxels are
  not cubic. The cache stores `anisotropy = (0.748, 0.748, 1.0)` (µm per voxel
  in x, y, z) and uses it to convert between SWC physical coordinates (µm) and
  image voxel indices.

---

## 2) Dataset schema

### On-disk layout

The pickle is a single dict whose relevant keys are:

| Key | Type | Meaning |
|---|---|---|
| `anisotropy` | `tuple` | `(0.748, 0.748, 1.0)` — µm/voxel in **(x, y, z)** order. |
| `min_cable_length` | `int` | `1000` — µm threshold; shorter fragments were discarded. |
| `node_spacing` | `int` | `5` — target µm spacing between adjacent skeleton nodes. |
| `fragments_graph` | `SkeletonGraph` | Automated UNet reconstruction (~10k components). |
| `gt_graph` | `SkeletonGraph` | Human ground-truth reconstruction (18 neurons). |

Loading requires the `agentic_neuron_proofreader` package on the path (the
pickle stores `SkeletonGraph` instances).

### Loading the cache (sample code)

```python
import pickle

with open("dataset_cache_794495.pkl", "rb") as f:
    payload = pickle.load(f)   # requires agentic_neuron_proofreader importable

gt_graph        = payload["gt_graph"]         # SkeletonGraph: 18 human-traced neurons
fragments_graph = payload["fragments_graph"]  # SkeletonGraph: ~10k UNet fragments
anisotropy      = payload["anisotropy"]       # (0.748, 0.748, 1.0)

print(gt_graph.summary(prefix="GroundTruth"))
print(fragments_graph.summary(prefix="Fragments"))
# -> # Connected Components / # Nodes / # Edges / Memory Consumption
```

### `SkeletonGraph` structure

`SkeletonGraph` subclasses `networkx.Graph`. A **node** is one point on a
neuron center-line; an **edge** connects adjacent points along the neurite.
Per-node attributes are stored as parallel NumPy arrays indexed by integer
node ID (not as NetworkX node-attribute dicts):

| Attribute | Shape / type | Meaning |
|---|---|---|
| `node_xyz` | `(N, 3) float32` | Physical coordinate of each node, in **(x, y, z) microns**. |
| `node_radius` | `(N,) float16` | Estimated neurite radius (µm) at each node. |
| `node_component_id` | `(N,) int` | Which connected component (one reconstructed neuron / fragment) each node belongs to. |
| `component_id_to_swc_id` | `dict[int, str]` | Maps a component ID to its SWC identifier (source file / neuron name). |
| `kdtree` | `scipy.spatial.KDTree` | Spatial index over `node_xyz` for nearest-node queries. |
| `soma_centroids`, `soma_component_ids` | `list` | Soma locations and their components (may be empty). |
| `anisotropy`, `node_spacing` | | Same values as above, kept on the graph. |

**Working with the graph (sample code):**

```python
import networkx as nx

# A connected component = one reconstructed object
#   gt    -> a complete traced neuron
#   frags -> a single UNet fragment (a real neuron is split across MANY of these)
print("GT components:   ", nx.number_connected_components(gt_graph))        # ~18
print("Frag components: ", nx.number_connected_components(fragments_graph)) # ~10k

# Node-level access
node  = next(iter(gt_graph.nodes))
xyz   = gt_graph.node_xyz[node]           # (x, y, z) in microns
voxel = gt_graph.node_voxel(node)         # (z, y, x) voxel index (xyz / anisotropy, reversed)
swc   = gt_graph.node_swc_id(node)        # e.g. "N001-794495-JT.0"
seg   = gt_graph.node_segment_id(node)    # raw UNet segment label (swc id w/o ".copy" suffix)

# Endpoints (leaf nodes) are where split-correction proposals originate
leaves = fragments_graph.leaf_nodes()         # degree-1 nodes
branch = fragments_graph.branching_nodes()    # degree-3+ nodes

# Nearest-neighbor query against another graph (used to relate frags <-> GT)
dist, nearest_gt_node = gt_graph.kdtree.query(fragments_graph.node_xyz[leaves[0]])
```

**Key conventions to interpret the data correctly:**

- **Coordinate order is inconsistent by design.** `node_xyz` is **(x, y, z)**
  microns; voxel coordinates are **(z, y, x)**. The helpers handle the
  conversion (divide by `anisotropy`, reverse the axes). Always check which
  order an array is in before using it.
- **A "connected component" = one reconstructed object.** In `gt_graph` a
  component is a complete traced neuron; in `fragments_graph` a component is a
  single UNet fragment — and a true neuron is typically broken across *many*
  components, which is precisely a **split** error.
- **SWC ID vs segment ID.** `node_swc_id(i)` returns `"<segment>.<copy>"`;
  `node_segment_id(i)` strips the suffix to give the raw U-Net segment label.
  GT components are named `N0XX-794495-<initials>` where the trailing initials
  denote the human annotator.

**Identifying errors from the two graphs.** Errors are defined by comparing the
GT skeleton against the predicted **segment labels** — *not* against
fragment/component identity. The unit of comparison is the raw U-Net
**segment id** (`node_segment_id`): one segment can be skeletonized into several
fragment components (the SWC id is `"<segment>.<copy>"`), so two GT nodes
landing on different *components* of the *same* segment are **not** a split.
Always classify by segment id, not by connected component.

Canonically, each GT node's predicted label is read from the dense segmentation
mask at that node's voxel. The cache does **not** include that mask, so with
only the two graphs the available proxy is: label a GT node by the **segment id
of its nearest fragment node**. The procedure:

1. **Label each GT node.** For every node in `gt_graph`, query
   `fragments_graph.kdtree.query(gt.node_xyz[node])` for the nearest fragment
   node. If the distance is within a tolerance (the reference implementation
   matches within ~a few µm), assign that GT node the fragment's **segment id**
   (`fragments_graph.node_segment_id(nearest)`); otherwise the GT node is
   *unlabeled* (label `0`). Every GT node now carries a predicted segment id —
   or none.

2. **Walk each GT edge** `(i, j)` and classify it from the two endpoint labels:
   - **Omit edge** — *both* endpoints are unlabeled (`0`): the reconstruction
     missed this stretch of neuron entirely.
   - **Split edge** — both endpoints are labeled but the **segment ids** *differ*
     (`label[i] != label[j]`, both nonzero): one true neuron is broken across
     two segments at this edge.
   - (Same nonzero segment id on both ends → correctly reconstructed.)

3. **Count splits per neuron** as `(number of distinct segment ids touching that
   GT neuron) − 1` — N segments covering one neuron implies N−1 splits.

4. **Detect merges** the other direction: a *single* segment id that maps onto
   *two or more distinct GT neurons* is a merge. Concretely, walk a fragment
   outward from its endpoints; if part of it sits far (> ~50 µm) from the GT
   neuron its segment was matched to and then re-approaches a *different* GT
   neuron, that segment is fusing two neurons — flag the branch point as a merge
   site. (Pass-throughs near very small GT components are ignored to avoid false
   positives.)

So the cache contains everything needed to score errors: the geometry
(`node_xyz`) and segment ids (`node_segment_id`) of both graphs, plus the
`kdtree` to match them. No image or external label mask is required — the
nearest-fragment segment id stands in for the mask lookup.

**Skeleton-metric glossary** (used when scoring a reconstruction against GT):

- **Split** — a single true neuron fragmented into multiple segments (false
  discontinuity); **% Split Edges** is the fraction of GT edges whose
  neighboring nodes carry different predicted labels.
- **Merge** — two distinct neurons fused into one predicted segment (false
  connection); **% Merged Edges** quantifies these.
- **% Omit Edges** — GT edges whose nodes have no predicted label (missed).
- **ERL (Expected Run Length)** — average error-free path length along a neuron;
  higher is better. **Normalized ERL** divides by total run length.
- **Edge Accuracy** — fraction of GT edges correctly reconstructed.
- **Split Rate / Merge Rate** — cable length (µm) per split / per merge.

---

## 3) Intent

The high-level goal is **agentic, post-hoc proofreading of the brain-794495
neuron reconstruction**: take the error-prone U-Net reconstruction
(`fragments_graph`) and correct it toward the human ground truth (`gt_graph`),
with the largest gains coming from fixing **splits** (re-joining fragments of
the same neuron) and resolving **merges** (separating wrongly fused neurons).
This is complementary to topology-aware *segmentation* losses, which reduce
errors at training time; here the aim is to repair the residual split/merge
errors that survive in the existing SWC fragments.

A conventional split-correction pipeline does this in a single forward pass:
generate reconnection proposals between nearby fragment endpoints (KD-tree
search within ~20 µm, aligned to branch tangents), extract skeleton features,
classify each proposal, and accept proposals in a progressive
confidence-threshold sweep that forbids cycles. A separate detector flags merge
sites. Success is measured by re-computing the skeleton metrics **before vs.
after** correction and reporting the reduction in splits/merges and the gain in
ERL and edge accuracy.

An **agentic** framework over this dataset aims to overcome the single-pass
pipeline's structural limits. These are the areas the system can use to loosely
steer exploration (not prescriptive hypotheses):

- **Re-planning after topology changes** — once fragments merge, new endpoints
  and shorter gaps appear; regenerate proposals instead of reconnecting once.
- **Refreshing stale features** — node degree, component size, and local
  proposal density shift as edits accumulate; predictions should track the
  current graph state, not the original fragmentation.
- **Reversible decisions** — detect and roll back early high-confidence merges
  that block later valid reconnections via the cycle-prevention check.
- **Mutual exclusivity at an endpoint** — at most one reconnection per leaf is
  anatomically valid; reason jointly over competing proposals rather than
  classifying each independently.
- **Coupling split and merge correction** — let merge detection unblock valid
  reconnections and let split correction avoid introducing new merges, via
  shared state and feedback.

Empirically useful relationships the data exposes for this exploration:
whether error rates vary by **neuron morphology** (cable length, branching,
soma proximity) or by **annotator** (the GT initials suffix); geometric
signatures of recoverable splits (endpoint distance, orientation agreement,
radius continuity); and the trade-off between aggressive merging (raising ERL)
and false merges (the costlier error). Evaluation centers on topological
accuracy (split/merge reduction, ERL and edge-accuracy gains), proposal
precision/recall, computational efficiency, and robustness to noise and
fragment density.
