# `dataset_cache_794495` — Dataset Description

`dataset_cache_794495_mcl10.pkl` is a Python pickle holding a cached
`BrainDataset` for **ExaSPIM brain 794495**. It bundles two neuron-skeleton graphs — a human
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
   **478,600 fragment components** for this brain at the `min_cable_length = 10`
   µm filtering threshold this cache was built with (see *How this cache was
   generated* below).

2. **Ground-truth tracings** (`gt_graph`) — human-traced neuron skeletons for
   **19 neurons**. These are the gold-standard morphologies used to train and
   evaluate the automated reconstruction.

**The errors that motivate the project.** Deep-learning segmentation introduces
two systematic *topological* errors:

- **Split errors** — a single neuron is fragmented into multiple disconnected
  segments (weak signal, imaging artifacts, or branch-point failures).
- **Merge errors** — two distinct neurons are erroneously joined, typically
  where neurites run close together.

Scoring the automated reconstruction against the 19 human tracings gives this
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

> These baseline numbers come from the canonical scoring pipeline (predicted
> labels read from the dense segmentation mask). The cache itself contains no
> mask, so re-deriving these metrics from `mcl10.pkl` alone uses the
> nearest-fragment proxy described in § *Identifying errors* and will not
> reproduce them exactly — treat the table as the reference target, not a
> cache-only result.

**How the data was gathered / known gaps.**

- **Ground truth is sparse and skeleton-only.** Only 19 neurons were traced
  (named like `N001-794495-JT`, `N002-794495-PP`, … — the suffix is the human
  annotator's initials; note the IDs are not contiguous, e.g. there is no
  `N010`/`N012`, and the largest is `N023`). There is **no dense ground-truth
  voxel volume**: GT
  exists only as center-line skeletons in `gt_graph`. A region with no traced
  neuron is simply unlabeled, not labeled "background."
- **Fragments are filtered.** When the cache was built, UNet fragments shorter
  than `min_cable_length = 10` µm of total path length were dropped, removing
  only the very shortest noise fragments. This is a *permissive* threshold —
  hence ~478,600 fragment components survive — chosen to keep almost all of the
  automated reconstruction; other builds use stricter thresholds (see *How this
  cache was generated*). Skeletons were also resampled to a target inter-node
  spacing of `node_spacing = 5` µm, so node geometry is decimated relative to
  the original SWCs.
- **Anisotropic voxels.** ExaSPIM samples X/Y more finely than Z, so voxels are
  not cubic. The cache stores `anisotropy = (0.748, 0.748, 1.0)` (µm per voxel
  in x, y, z) and uses it to convert between SWC physical coordinates (µm) and
  image voxel indices.

---

## 2) Dataset schema

### On-disk layout

The pickle is a single dict. Its full key set is below; the last five are the
ones you will actually use — the three `*_path` strings just record where the
source data was read from when the cache was built.

| Key | Type | Meaning |
|---|---|---|
| `fragments_path` | `str` | Provenance only — GCS path of the source UNet SWCs. Not needed to use the cache. |
| `gt_path` | `str` | Provenance only — GCS path of the source GT tracings. Not needed to use the cache. |
| `img_path` | `str` | Provenance only — S3 path of the fused ExaSPIM image. **Not** loaded; the cache is self-contained without it. |
| `anisotropy` | `tuple` | `(0.748, 0.748, 1.0)` — µm/voxel in **(x, y, z)** order. |
| `min_cable_length` | `int` | `10` — µm threshold; shorter fragments were discarded. |
| `node_spacing` | `int` | `5` — target µm spacing between adjacent skeleton nodes. |
| `fragments_graph` | `SkeletonGraph` | Automated UNet reconstruction (~478,600 components, ~25.5 M nodes). |
| `gt_graph` | `SkeletonGraph` | Human ground-truth reconstruction (19 neurons, ~1.36 M nodes). |

> **Note.** `min_cable_length` is `10` **for this cache file**
> (`dataset_cache_794495_mcl10.pkl`); other builds of the same brain use a
> stricter threshold (e.g. `_mcl100.pkl` at 100 µm, `_mcl1000.pkl` at 1000 µm)
> and therefore contain far fewer, longer fragments. The filename suffix `mclN`
> encodes the threshold `N`. Always read the actual value from
> `payload["min_cable_length"]` rather than assuming it.

Loading requires the `agentic_neuron_proofreader` package to be importable (the
pickle stores `SkeletonGraph` instances, so `pickle.load` must import their
class to reconstruct them). Install it first — see the next subsection.

### Install the package (one-time setup)

Loading the cache needs exactly one thing on the Python path: the
`agentic_neuron_proofreader` package, which defines the `SkeletonGraph` class
that `pickle.load` reconstructs. Its runtime dependencies are the usual
scientific stack — **`numpy`, `networkx`, `scipy`** (KD-tree), plus `tqdm`;
installing the package pulls these in.

**Option A — install it (recommended).** Clone
[`agentic-neuron-proofreader`](https://github.com/AllenInstitute/agentic-neuron-proofreader)
and `pip install` it into your environment:

```bash
git clone https://github.com/AllenInstitute/agentic-neuron-proofreader.git
cd agentic-neuron-proofreader
pip install -e .          # editable; drop -e for a normal install
```

This makes `import agentic_neuron_proofreader` work from anywhere.

**Option B — point at the source tree without installing.** If you only have a
checkout, prepend its `src/` directory to `sys.path` before unpickling:

```python
import sys
sys.path.insert(0, "/path/to/agentic-neuron-proofreader/src")
```

> **Environment gotcha.** `SkeletonGraph` imports `scipy.spatial.KDTree`, so
> `numpy` and `scipy` must be **binary-compatible** in the interpreter you use.
> A mismatch raises `ValueError: numpy.dtype size changed, may indicate binary
> incompatibility` on import — fix it by loading the cache in an environment
> where numpy and scipy were installed together (this cache was validated under
> the `panda` conda environment), not by editing the data.

### Loading the cache (sample code)

```python
import pickle
# Requires the agentic_neuron_proofreader package to be installed (see above).

with open("dataset_cache_794495_mcl10.pkl", "rb") as f:
    payload = pickle.load(f)   # reconstructs SkeletonGraph instances

gt_graph        = payload["gt_graph"]         # SkeletonGraph: 19 human-traced neurons
fragments_graph = payload["fragments_graph"]  # SkeletonGraph: ~478,600 UNet fragments
anisotropy      = payload["anisotropy"]       # (0.748, 0.748, 1.0)

print(gt_graph.summary(prefix="GroundTruth"))
print(fragments_graph.summary(prefix="Fragments"))
# -> # Connected Components / # Nodes / # Edges / Memory Consumption
# GroundTruth: 19 components, ~1,363,808 nodes
# Fragments:   ~478,611 components, ~25,527,200 nodes
```

> **Memory.** This cache is ~3 GB on disk, but reconstructing the two
> `SkeletonGraph` objects (NetworkX adjacency + KD-tree + node arrays) needs
> substantially more RAM than the file size — budget well over 20 GB of free
> memory, or the `pickle.load` may be killed by the OOM reaper. The stricter
> builds (`_mcl100.pkl`, `_mcl1000.pkl`) are smaller and lighter to load.

### How this cache was generated

You do **not** need to regenerate the cache to use it — this subsection only
documents its provenance so the stored parameters above are interpretable. The
cache was produced by [`notebooks/load_skeletons.ipynb`], which:

1. Builds a `BrainDataset` by reading the source SWCs directly from cloud
   storage (`gt_path`, `fragments_path`) and attaching the lazy ExaSPIM image
   reader (`img_path`):

   ```python
   from agentic_neuron_proofreader.data_modules.datasets import BrainDataset

   dataset = BrainDataset(
       fragments_path,                 # gs://allen-nd-goog/.../swcs
       gt_path,                        # gs://allen-nd-goog/.../voxel
       img_path,                       # s3://aind-open-data/.../fused.zarr/0
       anisotropy=(0.748, 0.748, 1.0),
       min_cable_length=10,            # <-- the mclN threshold; 10 for this file
       node_spacing=5,
   )
   ```

   Reading the ~10k source SWC archives from GCS is the slow step (~15 min) and
   needs valid Google credentials (`GOOGLE_APPLICATION_CREDENTIALS`).

2. Serializes the two reconstructed graphs plus the paths/parameters with
   `dataset.save(cache_path)`, where the filename encodes the threshold:

   ```python
   cache_path = f"../dataset_cache_{brain_id}_mcl{min_cable_length}.pkl"
   # min_cable_length=10  -> dataset_cache_794495_mcl10.pkl   (this file)
   # min_cable_length=100 -> dataset_cache_794495_mcl100.pkl
   ```

The lazy `TensorStoreImage` (`img_path`) is **not** pickled — it re-instantiates
instantly — which is why the cache loads from skeletons alone with no image
access, exactly the constraint this document relies on. To reload from the
cache without re-reading the SWCs, see
`notebooks/load_skeletons_from_cache.ipynb`.

[`notebooks/load_skeletons.ipynb`]: ../notebooks/load_skeletons.ipynb

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
print("GT components:   ", nx.number_connected_components(gt_graph))        # 19
print("Frag components: ", nx.number_connected_components(fragments_graph)) # ~478,611

# Node-level access
node  = next(iter(gt_graph.nodes))
xyz   = gt_graph.node_xyz[node]           # (x, y, z) in microns
voxel = gt_graph.node_voxel(node)         # (z, y, x) voxel index (xyz / anisotropy, reversed)
swc   = gt_graph.node_swc_id(node)        # e.g. "N001-794495-JT.0"
seg   = gt_graph.node_segment_id(node)    # the node's OWN component id, swc minus ".copy" suffix:
                                          #   on gt_graph -> the GT neuron name "N001-794495-JT"
                                          #   on fragments_graph -> the raw U-Net segment label
# NOTE: gt_graph.node_segment_id is the GT neuron's own id, NOT a predicted label.
# A GT node carries no predicted label until you match it against fragments_graph (see below).
pred  = fragments_graph.node_segment_id(  # the predicted segment for this GT node:
    fragments_graph.closest_node(xyz))    #   nearest fragment node's U-Net segment id

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
- **SWC ID vs segment ID, and the GT/predicted namespace split.**
  `node_swc_id(i)` returns `"<segment>.<copy>"`; `node_segment_id(i)` strips the
  suffix. Both methods exist on *both* graphs (same class), but they return ids
  from **different namespaces** and must never be compared directly:
  - On `fragments_graph`, `node_segment_id` is the **raw U-Net segment label** —
    the *predicted* identity used for scoring.
  - On `gt_graph`, it is the **GT neuron's own name** `N0XX-794495-<initials>`
    (trailing initials = human annotator) — a *ground-truth* identity, **not** a
    prediction.
  A GT node has **no predicted label stored anywhere**; the prediction for a GT
  node is obtained only by matching it against `fragments_graph` (next section).

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
   node (both graphs' `node_xyz` live in the same (x, y, z) µm space, so the
   query is a plain Euclidean distance). If the distance is within a **match
   tolerance**, assign that GT node the fragment's **segment id**
   (`fragments_graph.node_segment_id(nearest)`); otherwise the GT node is
   *unlabeled* (label `0`). Every GT node now carries a predicted segment id —
   or none.

   > **The tolerance is a free parameter, not a stored constant.** The cache
   > does not record it. A value of **~2 µm** is a reasonable default (it is on
   > the order of `node_spacing = 5` µm). The choice materially changes the
   > results — especially the **omit** count, since a looser tolerance labels
   > more GT nodes — so pick a value explicitly and report it. For reference,
   > the median GT→fragment nearest distance in this cache is ~1.6 µm, so a 2 µm
   > tolerance labels the majority of GT nodes and leaves the rest as omits.
   > You can vectorize the whole step in one call:
   > `dists, nn = fragments_graph.kdtree.query(gt_graph.node_xyz)`.

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

**Caveats of the cache-only proxy** (know these before trusting the numbers):

- **It is a proxy for the mask lookup.** Canonical scoring reads the predicted
  label from the dense segmentation mask at each GT voxel; here we substitute
  the nearest fragment node's segment id. The two agree only where a fragment
  node lies close to the GT center-line.
- **Omits are inflated by fragment filtering.** Fragments shorter than
  `min_cable_length` were dropped when the cache was built, so GT stretches that
  *were* reconstructed by a short fragment now have no nearby fragment node and
  get counted as **omit**. The omit rate measured from the cache is therefore an
  upper bound, not the true miss rate. This cache uses a permissive
  `min_cable_length = 10` µm, so the inflation is smaller here than in the
  stricter `_mcl100`/`_mcl1000` builds — but it is not zero.
- **Merge detection is partial.** The cache-only test verifies the core merge
  definition (one segment id touching ≥2 distinct GT neurons); the full
  geometric merge-site walk (§ step 4, the ">~50 µm then re-approach" rule)
  needs per-voxel mask labels to localize the merge point precisely.

These were confirmed by executing every instruction in this document against the
cache alone (see `notebooks/test_markdown_instructions.ipynb`): all structural
claims — keys, types, namespace split, the 4-step error procedure — run and pass
from cache-only data; the caveats above are the places where a number depends on
a choice (the tolerance) or on data the cache does not contain (the mask).

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
