# `dataset_cache` — Dataset Description

A `dataset_cache_<brain_id>_mcl<N>.pkl` file is a Python pickle holding a cached
`BrainDataset` for **one ExaSPIM brain**. Each bundles two neuron-skeleton
graphs — a human ground-truth reconstruction and an automated U-Net
reconstruction of the same brain. These caches are the evaluation targets for a
project on **agentic, post-hoc proofreading of whole-brain neuron
reconstructions**: correcting the systematic split/merge errors in the automated
reconstruction so the extracted wiring diagram becomes trustworthy for
downstream connectivity analysis.

**This trial provides a *collection* of these caches in the `cache/` folder —
one cache per brain, and that collection may hold a single brain or many.** Do
not assume which or how many brains are present: the collection is simply
*every* `dataset_cache_<brain_id>_mcl<N>.pkl` file in `cache/`, and a given run
may provide any subset of brains — sometimes just one. They share an identical
schema (same keys, same `SkeletonGraph` class, same build parameters) and differ
only in which brain they describe; the filename's `<brain_id>` is the specimen
and `<N>` is the `min_cable_length` threshold the cache was built with.

> **Read whatever is in `cache/` as ONE dataset.** The format is
> **brain-agnostic** by design: every cache exposes the same fields and is
> processed by the same procedures, so the same code path handles one cache or a
> dozen. The intent of this document is that you **discover all caches in
> `cache/`, load each, and pool the neurons and features from all of them into a
> single combined sample set**, then form hypotheses over that pooled population.
> If the collection happens to hold a single brain, that pooled set is simply
> that one brain; if it holds several, it is their union — the procedures are
> identical either way. Treat each traced neuron (and each fragment/feature
> derived from it) as one sample in the combined set, **tagged with the
> `brain_id` it came from**. Any per-file number quoted in this document (counts,
> baseline metrics) is illustrative of *one* brain's contribution; the quantities
> that matter for hypotheses are the **aggregates over whatever is present**.
> Carry `brain_id` alongside every pooled record so that, **when more than one
> brain is present**, it is available as a **covariate / grouping factor** — used
> to check whether an effect holds across brains or is driven by a single one
> (see § *Pooling across the collection* in §2). When only one brain is present
> `brain_id` is constant and that check is trivially satisfied. Dropping or adding
> a brain changes only the size of the pool, never the schema or the procedures.

---

## 1) Dataset context

**Origin.** Each cache comes from a single mouse brain imaged with **ExaSPIM**, a
light-sheet platform that combines tissue expansion with selective-plane
illumination microscopy to achieve sub-micron, whole-brain fluorescence imaging
of sparsely labeled neurons (voxel size **0.748 × 0.748 × 1.0 µm**). At this
resolution individual axons can be traced across an entire brain, but each brain
is tens of terabytes, so fully manual reconstruction is infeasible and automated
segmentation is required. *The collection provides one or more such brains, one
cache each; every cache has the same structure, and each brain present
contributes its own counts and error statistics to the pool.*

**What each cache captures.** Every cache stores two aligned, graph-based data
products for its brain:

1. **UNet fragment skeletons** (`fragments_graph`) — the automated
   reconstruction. A 3D U-Net predicted an instance-level voxel labeling of the
   brain; each predicted fragment was skeletonized into an SWC tree (3D
   coordinates, radius, connectivity) and loaded into a graph. This is the
   *machine* reconstruction containing the errors to be fixed — typically
   hundreds of thousands of fragment components per brain at the
   `min_cable_length = 100` µm filtering threshold the caches were built with
   (see *How each cache was generated* below). Each brain has its own counts.

2. **Ground-truth tracings** (`gt_graph`) — human-traced neuron skeletons (the
   gold-standard morphologies used to train and evaluate the automated
   reconstruction). Only a handful of neurons are traced per brain (order tens),
   and the count varies by brain. **These traced neurons are the core samples to
   pool** — your combined population is the union of every present brain's GT
   neurons (just one brain's, if that is all the collection holds), each tagged
   with its `brain_id`.

**The errors that motivate the project.** Deep-learning segmentation introduces
two systematic *topological* errors:

- **Split errors** — a single neuron is fragmented into multiple disconnected
  segments (weak signal, imaging artifacts, or branch-point failures).
- **Merge errors** — two distinct neurons are erroneously joined, typically
  where neurites run close together.

Scoring the automated reconstruction against the human tracings gives a baseline
of **skeleton-based topology metrics**, computed per brain and then pooled (a
pool of one when a single brain is present). The metrics that characterize each
brain's (and the combined population's) error profile are:

| Metric | Meaning |
|---|---|
| Total Splits / Merges | Count of split / merge errors. |
| Avg. Splits / Merges per Neuron | Per-neuron error counts. |
| Edge Accuracy | Fraction of GT edges correctly reconstructed. |
| % Split / Omit / Merged Edges | Fraction of GT edges in each error class. |
| ERL (Expected Run Length) | Average error-free path length along a neuron; higher is better. |
| Normalized ERL | ERL divided by total run length. |

Empirically, **splits dominate** (a true neuron broken into many pieces) while
merges are rarer but individually costly. These *topological* errors — not
voxel-level mislabeling — are what corrupt downstream connectivity, which is why
skeleton-based metrics (splits/neuron, edge accuracy, normalized ERL) are the
appropriate evaluation targets. Compute the numbers yourself from the caches
provided rather than relying on any quoted figures.

> Canonical baseline numbers come from a scoring pipeline that reads predicted
> labels from the dense segmentation mask. The cache itself contains no mask, so
> re-deriving these metrics from the pickle alone uses the nearest-fragment proxy
> described in § *Identifying errors* and will not reproduce the canonical
> figures exactly — treat such metrics as a relative target, not an exact
> cache-only result. Each brain's cache yields its own baseline; when more than
> one brain is present, compute the proxy metrics per brain with the *same*
> tolerance and then combine, so the brains are scored comparably.

**How the data was gathered / known gaps.**

- **Ground truth is sparse and skeleton-only.** Only a handful of neurons are
  traced per brain (named like `N001-<brain_id>-JT`, `N002-<brain_id>-PP`, … —
  the trailing suffix is the human annotator's initials, and the embedded
  `<brain_id>` already identifies the source brain). The IDs are not necessarily
  contiguous (gaps are normal); each brain has its own count and ID set, which
  you should read off the cache rather than assume. Because neuron names are only
  unique *within* a brain, you should **namespace each neuron by its `brain_id`**
  (the name already contains it, but track `brain_id` as an explicit column too)
  so that, whenever more than one brain is pooled, `N001` from two different
  brains is never conflated — harmless overhead with one brain, essential with
  several. There is **no dense ground-truth voxel volume**: GT
  exists only as center-line skeletons in `gt_graph`. A region with no traced
  neuron is simply unlabeled, not labeled "background."
- **Fragments are filtered.** When each cache was built, UNet fragments shorter
  than `min_cable_length` µm of total path length were dropped, removing the
  shortest noise fragments; the caches in this collection use
  `min_cable_length = 100` µm. Skeletons were also resampled to a target
  inter-node spacing of `node_spacing = 5` µm, so node geometry is decimated
  relative to the original SWCs. All caches in the collection share these build
  parameters, so fragments are comparable across brains when several are present
  — but always read the actual value from `payload["min_cable_length"]` per file
  rather than assuming it, and if a run ever mixes thresholds, account for that
  before pooling.
- **Anisotropic voxels.** ExaSPIM samples X/Y more finely than Z, so voxels are
  not cubic. Each cache stores `anisotropy = (0.748, 0.748, 1.0)` (µm per voxel
  in x, y, z) and uses it to convert between SWC physical coordinates (µm) and
  image voxel indices. This is shared across the collection, so physical
  coordinates and lengths (µm) are directly comparable across brains when more
  than one is present — pool on physical units, never raw voxel indices.

---

## 2) Dataset schema

### On-disk layout

Every cache, regardless of brain, is a single dict with the **same keys** — this
identical schema is exactly what makes the collection poolable and lets one code
path serve a single cache or many. The full key set
is below; the last five are the ones you will actually use — the three `*_path`
strings just record where the source data was read from when the cache was built.
(The graph contents and counts differ per brain; the keys and types do not.)

| Key | Type | Meaning |
|---|---|---|
| `fragments_path` | `str` | Provenance only — GCS path of the source UNet SWCs. Not needed to use the cache. |
| `gt_path` | `str` | Provenance only — GCS path of the source GT tracings. Not needed to use the cache. |
| `img_path` | `str` | Provenance only — S3 path of the fused ExaSPIM image. **Not** loaded; the cache is self-contained without it. |
| `anisotropy` | `tuple` | `(0.748, 0.748, 1.0)` — µm/voxel in **(x, y, z)** order. |
| `min_cable_length` | `int` | `100` — µm threshold; shorter fragments were discarded (shared across the collection). |
| `node_spacing` | `int` | `5` — target µm spacing between adjacent skeleton nodes. |
| `fragments_graph` | `SkeletonGraph` | Automated UNet reconstruction (hundreds of thousands of fragment components; tens of millions of nodes). |
| `gt_graph` | `SkeletonGraph` | Human ground-truth reconstruction (order tens of neurons; ~10⁶ nodes). |

> **Note.** The filename pattern is `dataset_cache_<brain_id>_mcl<N>.pkl`, where
> `<brain_id>` is the specimen and `N` is the `min_cable_length` threshold (the
> caches provided here are all `mcl100`, i.e. 100 µm; stricter builds such as
> `_mcl1000.pkl` would contain far fewer, longer fragments). Always read the
> actual value from `payload["min_cable_length"]` per file rather than assuming
> it, and read `<brain_id>` from the filename (or `gt_path`) so every pooled
> record can be tagged with its source brain.

Loading requires the `agentic_neuron_proofreader` package to be importable (the
pickle stores `SkeletonGraph` instances, so `pickle.load` must import their
class to reconstruct them). Install it first — see the next subsection.

### Install the package (one-time setup)

Loading the cache needs exactly one thing on the Python path: the
`agentic_neuron_proofreader` package, which defines the `SkeletonGraph` class
that `pickle.load` reconstructs. Its runtime dependencies are the usual
scientific stack — **`numpy`, `networkx`, `scipy`** (KD-tree), plus `tqdm`;
installing the package pulls these in.

Clone
[`agentic-neuron-proofreader`](https://github.com/AllenInstitute/agentic-neuron-proofreader)
and `pip install` it into your environment:

```bash
git clone https://github.com/AllenInstitute/agentic-neuron-proofreader.git
cd agentic-neuron-proofreader
pip install -e .          # editable; drop -e for a normal install
```

This makes `import agentic_neuron_proofreader` work from anywhere.

> **Environment gotcha.** `SkeletonGraph` imports `scipy.spatial.KDTree`, so
> `numpy` and `scipy` must be **binary-compatible** in the interpreter you use.
> A mismatch raises `ValueError: numpy.dtype size changed, may indicate binary
> incompatibility` on import — fix it by loading the cache in an environment
> where numpy and scipy were installed together (this cache was validated under
> the `panda` conda environment), not by editing the data.

### Loading one cache (sample code)

```python
import glob, pickle
# Requires the agentic_neuron_proofreader package to be installed (see above).

# Pick any dataset_cache_<brain_id>_mcl<N>.pkl present in cache/ — they all
# work the same way. (Glob the folder; see "Loading the whole collection" below.)
some_cache = sorted(glob.glob("cache/dataset_cache_*.pkl"))[0]
with open(some_cache, "rb") as f:
    payload = pickle.load(f)   # reconstructs SkeletonGraph instances

gt_graph        = payload["gt_graph"]         # SkeletonGraph: human-traced neurons
fragments_graph = payload["fragments_graph"]  # SkeletonGraph: UNet fragments
anisotropy      = payload["anisotropy"]       # (0.748, 0.748, 1.0)

print(gt_graph.summary(prefix="GroundTruth"))
print(fragments_graph.summary(prefix="Fragments"))
# -> # Connected Components / # Nodes / # Edges / Memory Consumption
#    (tens of GT neurons / ~10^6 GT nodes; hundreds of thousands of fragment
#     components / tens of millions of fragment nodes — exact counts per brain)
```

### Loading the whole collection (sample code)

The collection is **every** `dataset_cache_*.pkl` in `cache/` — which may be one
file or many. Discover them by glob (do not hard-code brain ids, and do not
assume more than one — a run may provide any subset, down to a single brain),
load each, and keep the `brain_id` parsed from the filename next to each loaded
cache so every downstream record can be tagged with its source brain. The same
loop below works unchanged whether it finds one cache or a dozen.

```python
import glob, os, pickle, re

CACHE_DIR = "cache"   # adjust to wherever the collection lives

def brain_id_from_path(path):
    # dataset_cache_<brain_id>_mcl<N>.pkl  ->  "<brain_id>"
    m = re.search(r"dataset_cache_(\d+)_mcl(\d+)\.pkl$", os.path.basename(path))
    return m.group(1) if m else os.path.basename(path)

# One entry per brain. Load lazily / one at a time if memory is tight (see below).
caches = {}
for path in sorted(glob.glob(os.path.join(CACHE_DIR, "dataset_cache_*.pkl"))):
    brain_id = brain_id_from_path(path)
    with open(path, "rb") as f:
        caches[brain_id] = pickle.load(f)
    print(f"loaded brain {brain_id} from {os.path.basename(path)}")

print(f"\nCollection: {len(caches)} brain(s) -> {sorted(caches)}")
# Each caches[brain_id] is the same dict described above (gt_graph,
# fragments_graph, anisotropy, min_cable_length, ...). Whatever was found is now
# ONE pooled population, keyed by brain_id -- a single entry if only one cache
# was present.
```

> **Memory.** Each cache reconstructs two `SkeletonGraph` objects (NetworkX
> adjacency + KD-tree + node arrays) needing substantially more RAM than the
> on-disk file size — budget well over 20 GB of free memory **per cache**. With
> a single cache that one budget is all you need; when several are present,
> holding them all in memory at once multiplies it, so unless you have ample RAM,
> **do not keep all caches resident simultaneously**: loop one
> brain at a time, reduce it to the small per-neuron / per-feature records you
> actually pool (a few floats per neuron — see *Pooling across the collection*),
> append those to a combined table, and let the heavy graph objects be garbage
> collected before loading the next brain. The pooled *table* is tiny even though
> the graphs are not.

### Pooling across the collection

The deliverable of the loading step is **one combined, tidy table** whose rows
are samples drawn from *all present* brains and whose first column is `brain_id`
(a single repeated value when only one brain is present). The
natural sample unit is **one GT neuron**; per-neuron features (cable length,
branch count, split/merge/omit counts and rates, ERL, edge accuracy, soma
proximity, annotator initials, …) are computed per brain with the *same*
parameters and then concatenated:

```python
import pandas as pd

rows = []
for brain_id, payload in caches.items():            # or stream one brain at a time
    gt = payload["gt_graph"]
    for comp_id, neuron_name in gt.component_id_to_swc_id.items():  # one row per traced neuron
        # neuron_name e.g. "N001-<brain_id>-JT"
        rows.append({
            "brain_id":   brain_id,                  # <-- the covariate / grouping key
            "neuron":     f"{brain_id}:{neuron_name}",     # globally-unique id
            "annotator":  neuron_name.split("-")[-1],      # GT initials suffix
            # ... per-neuron features / error metrics computed for THIS neuron ...
        })

pooled = pd.DataFrame(rows)   # rows from every brain, stacked into one population
```

**How to use `brain_id` when forming and testing hypotheses:**

- **Form hypotheses on the pooled population.** The sample size is the union over
  whatever brains are present; when a run provides several, that union — not any
  single brain — is the right population to reason over. When it provides one,
  that brain *is* the population, and the guidance below about between-brain
  effects simply does not apply.
- **When more than one brain is present, keep `brain_id` as a covariate /
  grouping factor.** Before trusting a pooled effect, check it is not driven by
  one brain: stratify or facet by `brain_id`, add it as a fixed/random effect, or
  confirm the effect's sign is consistent across brains. An effect that only
  appears when one brain is pooled in is a brain artifact, not a finding about
  reconstruction error. With a single brain `brain_id` is constant, so this check
  is moot — but still carry the column so the same analysis code keeps working if
  more brains are added.
- **Watch for between-brain confounds** (when several brains are present) —
  unequal neuron counts (weight or model accordingly), annotator differences that
  correlate with brain, and brain-level shifts in baseline error rate. Pool on
  **physical units (µm)** and scale-normalized metrics (percentages, normalized
  ERL), which are comparable across brains; raw counts scale with neuron size and
  brain and are not directly comparable. Preferring physical and normalized units
  is good practice regardless of how many brains are present.

### How each cache was generated

You do **not** need to regenerate any cache to use it — this subsection only
documents provenance so the stored parameters above are interpretable. Every
cache in the collection was produced the same way, by a two-step build (only the
`brain_id` and the source paths change from one brain to the next):

1. Build a `BrainDataset` by reading the source SWCs directly from cloud
   storage (`gt_path`, `fragments_path`) and attaching the lazy ExaSPIM image
   reader (`img_path`):

   ```python
   from agentic_neuron_proofreader.data_modules.datasets import BrainDataset

   dataset = BrainDataset(
       fragments_path,                 # gs://allen-nd-goog/.../swcs
       gt_path,                        # gs://allen-nd-goog/.../voxel
       img_path,                       # s3://aind-open-data/.../fused.zarr/0
       anisotropy=(0.748, 0.748, 1.0),
       min_cable_length=100,           # <-- the mclN threshold; 100 for this collection
       node_spacing=5,
   )
   ```

   Reading the ~10k source SWC archives from GCS is the slow step (~15 min) and
   needs valid Google credentials (`GOOGLE_APPLICATION_CREDENTIALS`).

2. Serialize the two reconstructed graphs plus the paths/parameters with
   `dataset.save(cache_path)`, where the filename encodes the threshold:

   ```python
   cache_path = f"../cache/dataset_cache_{brain_id}_mcl{min_cable_length}.pkl"
   # e.g. brain_id="<brain_id>", min_cable_length=100
   #      -> dataset_cache_<brain_id>_mcl100.pkl
   ```

   Running this once per brain is exactly how the collection in `cache/` was
   assembled — same parameters, different `brain_id` — which is what guarantees
   that any caches present are mutually comparable and safe to pool.

The lazy `TensorStoreImage` (`img_path`) is **not** pickled — it re-instantiates
instantly — which is why each cache loads from skeletons alone with no image
access, exactly the constraint this document relies on.

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
print("GT components:   ", nx.number_connected_components(gt_graph))        # order tens
print("Frag components: ", nx.number_connected_components(fragments_graph)) # hundreds of thousands

# Node-level access
node  = next(iter(gt_graph.nodes))
xyz   = gt_graph.node_xyz[node]           # (x, y, z) in microns
voxel = gt_graph.node_voxel(node)         # (z, y, x) voxel index (xyz / anisotropy, reversed)
swc   = gt_graph.node_swc_id(node)        # e.g. "N001-<brain_id>-JT.0"
seg   = gt_graph.node_segment_id(node)    # the node's OWN component id, swc minus ".copy" suffix:
                                          #   on gt_graph -> the GT neuron name "N001-<brain_id>-JT"
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
  - On `gt_graph`, it is the **GT neuron's own name** `N0XX-<brain_id>-<initials>`
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
   > more GT nodes — so pick a value explicitly and report it. The median
   > GT→fragment nearest distance is typically on the order of a micron, so a
   > ~2 µm tolerance labels the majority of GT nodes and leaves the rest as omits;
   > inspect the distance distribution per brain (`dists` below) to confirm.
   > **Use the same tolerance for every brain** so the per-brain error metrics are
   > comparable when several are pooled (and so the choice stays reproducible with
   > one). You can vectorize the whole step in one call:
   > `dists, nn = fragments_graph.kdtree.query(gt_graph.node_xyz)`.

2. **Walk each GT edge** `(i, j)` and classify it from the two endpoint labels:
   - **Omit edge** — *either* endpoint is unlabeled (`label[i] == "0"` **or**
     `label[j] == "0"`): the reconstruction missed (at least) this end of the
     neuron. This matches `OmitEdgePercentMetric.count_omit_edges`, whose
     criterion is an **OR over the two endpoints** — an edge that touches even
     one unlabeled node is an omit edge.
   - **Split edge** — both endpoints are labeled but the **segment ids** *differ*
     (`label[i] != label[j]`, both nonzero): one true neuron is broken across
     two segments at this edge.
   - (Same nonzero segment id on both ends → correctly reconstructed.)

   Note there is **no separate "boundary" / one-end-labeled class**: an edge with
   exactly one unlabeled endpoint is an omit edge under the OR rule.

3. **Count splits per neuron** as `(number of distinct segment ids touching that
   GT neuron) − 1` — N segments covering one neuron implies N−1 splits.

4. **Detect merges** the other direction: a *single* segment id that maps onto
   *two or more distinct GT neurons* is a merge. Concretely, walk a fragment
   outward from its endpoints; if part of it sits far (> ~50 µm) from the GT
   neuron its segment was matched to and then re-approaches a *different* GT
   neuron, that segment is fusing two neurons — flag the branch point as a merge
   site. (Pass-throughs near very small GT components are ignored to avoid false
   positives.)

**Sample code** — the four steps end-to-end, using only `gt_graph` and
`fragments_graph` from **one** cache. If more than one brain is present, run this
block once per brain (looping over the loaded caches), tag each resulting
per-neuron record with its `brain_id`, and concatenate — exactly the table built
in § *Pooling across the collection*; with a single brain it runs once and the
table is that one brain. Keep `MATCH_TOL_UM` identical across brains so the
scores are comparable:

```python
import numpy as np
from collections import defaultdict

MATCH_TOL_UM = 5.0   # the free tolerance from step 1; report whatever you pick

# --- Step 1: label each GT node by its nearest fragment's SEGMENT id ----------
# One vectorized KD-tree query over all GT nodes; both graphs share (x,y,z) µm.
dists, nn_frag = fragments_graph.kdtree.query(gt_graph.node_xyz)   # (N_gt,), (N_gt,)

gt_pred_label = {}
for gt_n in gt_graph.nodes:
    if float(dists[gt_n]) <= MATCH_TOL_UM:
        # NOTE: segment id, NOT component id — one segment = many components.
        gt_pred_label[gt_n] = fragments_graph.node_segment_id(int(nn_frag[gt_n]))
    else:
        gt_pred_label[gt_n] = "0"          # unlabeled -> contributes to omits

# --- Step 2: classify every GT edge from its two endpoint labels --------------
n_omit = n_split = n_correct = 0
for i, j in gt_graph.edges:
    li, lj = gt_pred_label[i], gt_pred_label[j]
    if li == "0" or lj == "0":
        n_omit += 1                        # EITHER end unlabeled (count_omit_edges' OR rule)
    elif li != lj:
        n_split += 1                       # both labeled, different segments
    else:
        n_correct += 1                     # same segment -> correctly joined
E = gt_graph.number_of_edges()
print(f"split={n_split} ({100*n_split/E:.2f}%)  omit={n_omit} ({100*n_omit/E:.2f}%)")

# --- Step 3: splits per neuron = (#distinct predicted segments on it) - 1 ------
neuron_to_segs = defaultdict(set)
for gt_n in gt_graph.nodes:
    lab = gt_pred_label[gt_n]
    if lab != "0":
        neuron_to_segs[gt_graph.node_segment_id(gt_n)].add(lab)   # GT name -> {seg ids}
splits_per_neuron = {n: max(len(s) - 1, 0) for n, s in neuron_to_segs.items()}
print("Total Splits:", sum(splits_per_neuron.values()))

# --- Step 4 (core definition): one segment touching >=2 GT neurons = merge ----
seg_to_neurons = defaultdict(set)
for gt_n in gt_graph.nodes:
    lab = gt_pred_label[gt_n]
    if lab != "0":
        seg_to_neurons[lab].add(gt_graph.node_segment_id(gt_n))   # seg id -> {GT names}
merged = {lab: ns for lab, ns in seg_to_neurons.items() if len(ns) >= 2}
print("Merge segments:", len(merged),
      " Total Merges:", sum(len(ns) - 1 for ns in merged.values()))
```

The two dicts are duals of the same matching: `neuron_to_segs` (GT neuron → the
segments covering it) exposes **splits**, and `seg_to_neurons` (segment → the GT
neurons it touches) exposes **merges**. Everything keys off `node_segment_id`,
never connected-component identity — that is the one mistake the namespace
discussion above warns against.

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
  `min_cable_length` were dropped when each cache was built, so GT stretches that
  *were* reconstructed by a short fragment now have no nearby fragment node and
  get counted as **omit**. The omit rate measured from the cache is therefore an
  upper bound, not the true miss rate. All caches in this collection share the
  same `min_cable_length = 100` µm, so this inflation is consistent across any
  brains present and does not bias one brain relative to another when pooling.
- **Merge detection is partial.** The cache-only test verifies the core merge
  definition (one segment id touching ≥2 distinct GT neurons); the full
  geometric merge-site walk (§ step 4, the ">~50 µm then re-approach" rule)
  needs per-voxel mask labels to localize the merge point precisely.

Every instruction in this document has been verified to run against the cache
alone: all structural claims — keys, types, namespace split, the 4-step error
procedure — execute from cache-only data. The caveats above are the places where
a number depends on a choice (the tolerance) or on data the cache does not
contain (the mask).

**Skeleton-metric glossary** (used when scoring a reconstruction against GT):

- **Split** — a single true neuron fragmented into multiple segments (false
  discontinuity); **% Split Edges** is the fraction of GT edges whose
  neighboring nodes carry different predicted labels.
- **Merge** — two distinct neurons fused into one predicted segment (false
  connection); **% Merged Edges** quantifies these.
- **% Omit Edges** — fraction of GT edges with at least one unlabeled endpoint
  (either node has no predicted label, i.e. `node_label == "0"`); the
  reconstruction missed that part of the neuron.
- **ERL (Expected Run Length)** — average error-free path length along a neuron;
  higher is better. **Normalized ERL** divides by total run length.
- **Edge Accuracy** — fraction of GT edges correctly reconstructed.
- **Split Rate / Merge Rate** — cable length (µm) per split / per merge.

---

## 3) Intent

The high-level goal is to **build a better proofreading tool** — an *agentic,
post-hoc* corrector of whole-brain neuron reconstructions that works for **any
ExaSPIM brain**, with the **collection of caches in `cache/`** as its
development-and-evaluation target. The tool is developed and judged against the
*combined* population of neurons across whatever caches are present — the union
of all brains when several are provided, or that single brain when only one is.
The tool takes the error-prone U-Net reconstruction
(`fragments_graph`) and **edits it to correct the three topological error types**
scored against the human ground truth (`gt_graph`):

- fix **splits** — re-join fragments that belong to the same neuron;
- resolve **merges** — cut apart segments that wrongly fuse two neurons;
- recover **omits** — extend/reconnect reconstruction into stretches of neuron
  the U-Net missed entirely.

The deliverable is the **corrector itself** (a reusable, brain-agnostic method),
not just a hand-fixed copy of one brain; whatever is in `cache/` — one brain or
several — is the benchmark on which it is measured in this trial. This is complementary to topology-aware *segmentation* losses, which
reduce errors at training time; here the aim is to repair the residual
split/merge/omit errors that survive in the existing SWC fragments.

The bar for "better" is set by a conventional split-correction pipeline that
does this in a single forward pass: generate reconnection proposals between
nearby fragment endpoints (KD-tree search within ~20 µm, aligned to branch
tangents), extract skeleton features, classify each proposal, and accept
proposals in a progressive confidence-threshold sweep that forbids cycles. A
separate detector flags merge sites. Success is measured by re-computing the
skeleton metrics **before vs. after** correction: a better tool delivers a
**larger reduction in splits/merges/omits and larger ERL and edge-accuracy
gains than this single-pass baseline**, aggregated over the pooled ground-truth
neurons from every brain present (and, when more than one brain is present,
ideally consistent across brains rather than carried by a single one).

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

Empirically useful relationships the **pooled** data exposes for this
exploration: whether error rates vary by **neuron morphology** (cable length,
branching, soma proximity), by **annotator** (the GT initials suffix), or — when
more than one brain is present — by **brain** (`brain_id`), and, crucially,
whether a morphology/annotator effect *survives* once `brain_id` is held constant
or added as a covariate, versus being an artifact of between-brain differences;
geometric signatures of recoverable splits (endpoint distance, orientation
agreement, radius continuity); and the trade-off between aggressive merging
(raising ERL) and false merges (the costlier error). When several brains are
available, pooling across them is what gives these relationships the statistical
power to be trusted — a per-brain handful of neurons rarely does; with a single
brain the morphology and annotator relationships are still worth examining within
that brain, just without the cross-brain check. Evaluation centers on topological
accuracy (split/merge reduction, ERL and edge-accuracy gains) aggregated over the
pooled neurons, proposal precision/recall, computational efficiency, robustness
to noise and fragment density, and — where more than one brain is present —
**consistency of the effect across brains**.
