# `dataset_cache_794495` — Dataset Description

`dataset_cache_794495.pkl` is a ~592 MB on-disk cache of a `BrainDataset` for
ExaSPIM brain **794495**. It is produced by `notebooks/load_skeletons.ipynb`
and consumed by the visualization, evaluation, and proofreading scripts. It
bundles two neuron-skeleton graphs plus the parameters needed to re-open the
raw image lazily.

This dataset is the primary evaluation target for the project
**"Agentic Neuron Proofreading for ExaSPIM Whole-Brain Reconstructions"**
(see `main.tex`): correcting the systematic split/merge errors in an automated
U-Net neuron reconstruction so that the wiring diagram extracted from a whole
mouse brain becomes trustworthy for downstream connectivity analysis.

---

## 1) Dataset context

**Origin.** The data comes from a single mouse brain (specimen ID **794495**)
imaged with **ExaSPIM**, the Allen Institute for Neural Dynamics light-sheet
platform that combines tissue expansion with selective-plane illumination
microscopy to achieve sub-micron, whole-brain fluorescence imaging of sparsely
labeled neurons. At this resolution individual axons can be traced across an
entire brain, but each brain is tens of terabytes, so fully manual
reconstruction is infeasible and automated segmentation is required. The fused
image volume for this brain lives in the AIND open-data S3 bucket:

```
s3://aind-open-data/exaSPIM_794495_2026-01-21_14-25-07_processed_2026-01-29_22-02-08/fusion/fused.zarr/
```

(resolved from `configs/exaspim_image_prefixes.json`; the cache reads scale
level `0`, the full-resolution tier).

**What the cache captures.** The brain has been processed into three aligned
data products, two of which are stored in the cache as graphs and one of which
is referenced by path and read on demand:

1. **Raw fluorescence image** — the ExaSPIM volume itself, a Zarr array on S3.
   *Not* pickled into the cache; only its path is stored. It is re-opened
   lazily through `TensorStoreImage` because patch reads stream directly from
   cloud storage.

2. **UNet fragment skeletons** (`fragments_graph`) — the automated
   reconstruction. A **3D U-Net** segmentation model
   (`raw.unet_449_splits_and_merges_900000`) predicted an instance-level voxel
   labeling of the brain; each predicted fragment was skeletonized into an SWC
   tree (3D coordinates, radius, connectivity). These SWCs were read from
   `gs://allen-nd-goog/from_google/794495/whole_brain/raw.unet_449_splits_and_merges_900000/swcs`
   and loaded into a graph. This is the *machine* reconstruction containing the
   errors the proofreader must fix — roughly **10,000 fragments** for this brain.

3. **Ground-truth tracings** (`gt_graph`) — human-traced neuron skeletons for
   **18 neurons**, read from `gs://allen-nd-goog/ground_truth_tracings/794495/voxel`.
   These are the gold-standard morphologies used to train and evaluate the
   automated reconstruction.

**The errors that motivate the project.** Deep-learning segmentation introduces
two systematic *topological* errors:

- **Split errors** — a single neuron is fragmented into multiple disconnected
  segments (weak signal, imaging artifacts, or branch-point failures).
- **Merge errors** — two distinct neurons are erroneously joined, typically
  where neurites run close together.

A whole-brain skeleton-metrics evaluation against the 18 human tracings
(stored in `metrics_out/794495/raw.unet_449_splits_and_merges_900000/`)
quantifies the baseline:

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
| ERL | 5,090.6 µm |
| Normalized ERL | 0.0095 |

Splits dominate (a true neuron broken into hundreds of pieces); merges are
rarer but individually costly. These topological errors — not voxel-level
mislabeling — are what corrupt downstream connectivity, which is why
skeleton-based metrics (splits/neuron, edge accuracy, normalized ERL) are the
appropriate evaluation targets.

**How the data was gathered / known gaps.**

- **Ground truth is sparse and skeleton-only.** Only 18 neurons were traced
  (named `N001-794495-JT`, `N002-794495-PP`, … — the suffix is the annotator's
  initials). There is **no dense ground-truth voxel volume**: GT exists only as
  center-line skeletons in `gt_graph`. A region with no traced neuron is simply
  unlabeled, not labeled "background."
- **Fragments are filtered at load time.** When the cache was built, UNet
  fragments shorter than `min_cable_length = 1000` µm of total path length were
  dropped, removing short/noisy false-positive fragments. Skeletons were also
  resampled to a target inter-node spacing of `node_spacing = 5` µm, so node
  geometry is decimated relative to the original SWCs.
- **Anisotropic voxels.** ExaSPIM samples X/Y more finely than Z, so voxels are
  not cubic. The cache stores `anisotropy = (0.748, 0.748, 1.0)` (µm per voxel
  in x, y, z) and uses it to convert between SWC physical coordinates (µm) and
  image voxel indices on every patch read.

---

## 2) Dataset schema

### On-disk layout

The pickle is a single dict (written by `BrainDataset.save`) with these keys:

| Key | Type | Meaning |
|---|---|---|
| `fragments_path` | `str` | GCS path to the UNet fragment SWCs. |
| `gt_path` | `str` | GCS path to the human ground-truth SWCs. |
| `img_path` | `str` | S3 path to the raw ExaSPIM image (used to rebuild the lazy reader, **not** pickled image data). |
| `anisotropy` | `tuple` | `(0.748, 0.748, 1.0)` — µm/voxel in **(x, y, z)** order. |
| `min_cable_length` | `int` | `1000` — µm threshold; shorter fragments were discarded. |
| `node_spacing` | `int` | `5` — target µm spacing between adjacent skeleton nodes. |
| `fragments_graph` | `SkeletonGraph` | Automated UNet reconstruction (~10k components). |
| `gt_graph` | `SkeletonGraph` | Human ground-truth reconstruction (18 neurons). |

On reload, `BrainDataset.load_from_cache` restores the two graphs and
re-instantiates `img = TensorStoreImage(img_path)` (a lazy cloud reader); the
raw image data itself is never stored in the pickle.

### Loading the cache (sample code)

```python
import os
# GCS/S3 credentials are needed only to re-open the lazy image reader.
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "zihan_gcs_token.json"
os.environ["AWS_EC2_METADATA_DISABLED"] = "true"

from agentic_neuron_proofreader.data_modules.datasets import BrainDataset

dataset = BrainDataset.load_from_cache("dataset_cache_794495.pkl")

gt    = dataset.gt_graph         # SkeletonGraph: 18 human-traced neurons
frags = dataset.fragments_graph  # SkeletonGraph: ~10k UNet fragments
img   = dataset.img              # TensorStoreImage: lazy reader for raw volume

print(gt.summary(prefix="GroundTruth"))
print(frags.summary(prefix="Fragments"))
# -> # Connected Components / # Nodes / # Edges / Memory Consumption
```

### Rebuilding the cache from scratch (sample code)

Equivalent to `notebooks/load_skeletons.ipynb` — reads the SWCs from cloud
(~15 min) and re-pickles the two graphs:

```python
brain_id        = "794495"
segmentation_id = "raw.unet_449_splits_and_merges_900000"

gt_path        = f"gs://allen-nd-goog/ground_truth_tracings/{brain_id}/voxel"
fragments_path = f"gs://allen-nd-goog/from_google/{brain_id}/whole_brain/{segmentation_id}/swcs"
img_path       = "s3://aind-open-data/exaSPIM_794495_.../fusion/fused.zarr/0"  # from configs

dataset = BrainDataset(
    fragments_path,
    gt_path,
    img_path,
    anisotropy=(0.748, 0.748, 1.0),  # (x, y, z) µm/voxel
    min_cable_length=1000,           # drop fragments shorter than 1000 µm
    node_spacing=5,                  # resample skeletons to 5 µm node spacing
)
dataset.save(f"dataset_cache_{brain_id}.pkl")
```

### `SkeletonGraph` structure

`SkeletonGraph` subclasses `networkx.Graph`. A **node** is one point on a
neuron center-line; an **edge** connects adjacent points along the neurite.
Per-node attributes are stored as parallel NumPy arrays indexed by integer
node ID (not as NetworkX node dicts):

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
print("GT components:   ", nx.number_connected_components(gt))     # ~18
print("Frag components: ", nx.number_connected_components(frags))  # ~10k

# Node-level access
node = next(iter(gt.nodes))
xyz   = gt.node_xyz[node]           # (x, y, z) in microns
voxel = gt.node_voxel(node)         # (z, y, x) voxel index (xyz / anisotropy, reversed)
swc   = gt.node_swc_id(node)        # e.g. "N001-794495-JT.0"
seg   = gt.node_segment_id(node)    # raw UNet segment label (swc id w/o ".copy" suffix)

# Endpoints (leaf nodes) are where split-correction proposals originate
leaves = frags.leaf_nodes()         # degree-1 nodes
branch = frags.branching_nodes()    # degree-3+ nodes

# Nearest-neighbor query against another graph (used to relate frags <-> GT)
dist, nearest_gt_node = gt.kdtree.query(frags.node_xyz[leaves[0]])
```

**Reading aligned image / segmentation patches (sample code):**

```python
from agentic_neuron_proofreader.utils import img_util

# Center a patch on a GT node and read the raw fluorescence around it
node        = next(iter(gt.nodes))
center      = gt.node_voxel(node)          # (z, y, x)
patch_shape = (128, 128, 128)
img_patch   = dataset.img.read(center, patch_shape)   # raw fluorescence

# The U-Net label volume is read the same way (NOT in the cache; read by path)
seg = img_util.TensorStoreImage(
    "gs://allen-nd-goog/from_google/794495/whole_brain/"
    "raw.unet_449_splits_and_merges_900000/"
)
seg_patch = seg.read(center, patch_shape)              # integer labels, 0 = bg

# Skeleton geometry within the same patch, in LOCAL (z, y, x) voxels:
offset            = tuple(c - s // 2 for c, s in zip(center, patch_shape))
nodes_local       = gt.nodes_in_patch(offset, patch_shape)           # (M, 3)
edges, edge_comps = gt.edges_in_patch(offset, patch_shape, return_components=True)
```

**Key conventions to interpret the data correctly:**

- **Coordinate order is inconsistent by design.** `node_xyz` is **(x, y, z)**
  microns; voxel coordinates, image patches, and `nodes_in_patch` /
  `edges_in_patch` outputs are **(z, y, x)**. The helpers handle the conversion
  (divide by `anisotropy`, reverse the axes). Always check which order an array
  is in before using it.
- **A "connected component" = one reconstructed object.** In `gt_graph` a
  component is a complete traced neuron; in `fragments_graph` a component is a
  single UNet fragment — and a true neuron is typically broken across *many*
  components, which is precisely a **split** error.
- **SWC ID vs segment ID.** `node_swc_id(i)` returns `"<segment>.<copy>"`;
  `node_segment_id(i)` strips the suffix to give the raw U-Net segment label.
  GT components are named `N0XX-794495-<initials>` where the trailing initials
  denote the human annotator.

### Associated (non-cached) data products

These are referenced by the scripts but live outside the pickle:

- **Segmentation volume** — `gs://allen-nd-goog/from_google/794495/whole_brain/raw.unet_449_splits_and_merges_900000/`,
  read via `TensorStoreImage` as integer label patches (0 = background).
- **`metrics_out/794495/raw.unet_449_splits_and_merges_900000/`** — whole-brain
  evaluation outputs:
  - `results.csv` — one row per GT neuron. Columns: `SWC Run Length`,
    `# Splits`, `# Merges`, `% Split Edges`, `% Omit Edges`, `% Merged Edges`,
    `ERL`, `Normalized ERL`, `Edge Accuracy`, `Split Rate`, `Merge Rate`.
  - `results_overview.txt` — run-length-weighted averages plus totals.
  - `merge_sites.csv` — one row per detected merge error. Columns:
    `Merge_ID`, `Fragment_Name`, `Segment_ID`, `GroundTruth_ID`, `Label`,
    `Voxel` (z, y, x), `World` (physical x/y/z), `Added Cable Length (μm)`.
  - `fragments_with_merges.zip` — SWCs of fragments implicated in merges.

**Metric glossary** (from `scripts/proofreading_stats.py` and the skeleton-metrics framework):

- **Split** — a single true neuron fragmented into multiple segments (false
  discontinuity). **# Splits** counts them; **% Split Edges** is the fraction of
  GT edges where neighboring nodes carry different predicted labels.
- **Merge** — two distinct neurons fused into one predicted segment (false
  connection). **# Merges** / **% Merged Edges** quantify these.
- **% Omit Edges** — GT edges whose nodes have no predicted label (missed).
- **ERL (Expected Run Length)** — average error-free path length along a neuron;
  higher is better. **Normalized ERL** divides by total run length.
- **Edge Accuracy** — fraction of GT edges correctly reconstructed.
- **Split Rate / Merge Rate** — cable length (µm) per split / per merge.

---

## 3) Intent

The high-level goal is **agentic, post-hoc proofreading of the ExaSPIM 794495
neuron reconstruction**: take the error-prone U-Net reconstruction
(`fragments_graph`) and correct it toward the human ground truth (`gt_graph`),
with the largest gains coming from fixing **splits** (re-joining fragments of
the same neuron) and resolving **merges** (separating wrongly fused neurons).
This is complementary to topology-aware *segmentation* losses: those reduce
errors at training time, whereas this project repairs the residual split/merge
errors that survive in existing SWC fragments.

The existing `NeuronProofreader` pipeline does split correction in a single
forward pass — build a `ProposalGraph`, generate reconnection proposals between
nearby fragment endpoints (KD-tree search within ~20 µm, aligned to branch
tangents), extract skeleton + image features (two-channel 96³ patches),
classify with **VisionHGAT** (a heterogeneous GATv2 over skeleton topology plus
a CNN image encoder), and accept proposals in a progressive threshold sweep
(0.99 down, cycle-free). A separate CNN `MergeDetector` slides along skeletons
to flag merge sites. The repository's `train_and_run_neuron_proofreader.py`
implements training + inference of this pipeline, and `proofreading_stats.py`
scores the result by re-running skeleton metrics **before vs. after** correction.

The **agentic** framework this dataset supports aims to overcome the single-pass
pipeline's structural limits, and these are the areas the system can use to
loosely steer exploration (not prescriptive hypotheses):

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
local **image evidence** (fluorescence intensity, neurite radius continuity,
gap size) vs. where split/merge errors concentrate; whether error rates vary by
**neuron morphology** (cable length, branching, soma proximity) or by
**annotator** (the GT initials suffix); geometric signatures of recoverable
splits (endpoint distance, orientation agreement, radius continuity); and the
trade-off between aggressive merging (raising ERL) and false merges (the
costlier error). Evaluation follows `main.tex`: topological accuracy
(split/merge reduction, ERL and edge-accuracy gains), proposal precision/recall,
computational efficiency, and robustness to noise and fragment density.
