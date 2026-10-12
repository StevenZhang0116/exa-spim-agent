# Understanding split errors in ExaSPIM reconstructions — simpler computation

The scientific question is unchanged: **what distinguishes a split — where the
network broke one neuron into separate predicted segments — from genuine fragment
endings, across fragment topology, geometry, and the raw fluorescence image?**

This version keeps the dataset interpretation, answer-key construction, GT firewall,
and discovery objective of `split_detection_from_fragments.md`. It simplifies only
the computational workflow:

1. characterize and detect **truncated endpoints** first;
2. compute cheap, cached reconstruction features for every endpoint;
3. read raw-image patches only for labelled samples or a small endpoint shortlist;
4. search for partners only from high-confidence truncations, with a strict candidate
   cap.

This preserves discovery of the split signature without turning the primary analysis
into an exhaustive endpoint-pair search.

---

## 1) Dataset context

### Origin

Each `_add.pkl` was built from a skeleton cache by reading the dense predicted
segmentation volume, looking up the predicted segment id at every ground-truth node,
classifying GT edges, and storing the results. Cloud access to the private
segmentation happened at build time; it is not needed for this analysis.

The cache contains:

- `fragments_graph`: automated UNet fragment reconstruction, with potentially
  hundreds of thousands of components;
- `gt_graph`: sparse human ground-truth tracings;
- labels derived by comparing the prediction with GT;
- reconstruction metadata and a public raw-image path.

Loading a cache does not require segmentation credentials. Raw-image access may still
be remote and should be treated as expensive I/O.

### What a split is

The two main segmentation errors are:

- **merge**: one predicted segment incorrectly contains material from distinct
  neurons;
- **split**: one true neuron is represented by two or more predicted segments.

A split occurs where the predicted segment label changes along one true neuron. It is
a connection that should have been made but was not. Common causes may include a dim
stretch, thin distal branch, abrupt turn, crossing vessel, or tile boundary.

Splits and merges are dual but not computationally symmetric:

- a merge is normally localized inside one component;
- a split is completely specified by a relation between two components.

The full split problem therefore has two detection units:

| Unit | Question | Role in this workflow |
|---|---|---|
| Endpoint | Is this degree-1 fragment node a false truncation or a real terminal? | **Primary** |
| Pair | Which other endpoint/component is its continuation? | Optional bounded extension |

Endpoint detection directly answers where the network stopped following a process.
It also remains meaningful when the partner fragment was removed. Pair detection is
needed for repair, but it is not required to discover the conditions associated with
splitting.

### Why the endpoint is primary

An endpoint can retain a split signature even when its partner is missing: flat
caliber at the cut, a stable outgoing direction, nearby reconstructed material, or
weak fluorescence continuing beyond the endpoint. A pair detector cannot recover a
partner that does not exist in `fragments_graph`.

`min_cable_length` creates an important split-specific ceiling. Components shorter
than this threshold were removed at cache construction. If the far side of a split
was a short stub, the split may still be visible as a truncation, but it cannot be
identified as a complete pair. Always distinguish:

- **endpoint recall**, which is not strictly capped by partner survival;
- **reachable-pair recall**, where both segment ids have surviving components;
- **unreachable pairs**, reported as a separate count.

Do not charge a pair detector for a partner deleted from its input.

### Sparse ground truth

GT contains only a small number of traced neurons per brain. Consequently:

- splits on untraced neurons are absent from the answer key;
- predictions in untraced space are not automatically false positives;
- some endpoints near GT mid-cable are ambiguous skeletonization artifacts rather
  than clear terminals or known splits;
- pair precision can be computed only where GT labels both proposed segments well
  enough to adjudicate the join.

Report unadjudicable predictions separately. Never silently count all of them as
incorrect.

### Two phases and the GT firewall

The work has two phases:

1. **Phase 1 — characterization (GT-informed).** Derive the split answer key, label
   fragment endpoints, extract reconstruction and selected image features, and find
   which features distinguish truncations from genuine terminals.
2. **Phase 2 — blind detection (GT-blind).** Freeze the feature definitions and
   detector. Produce endpoint scores using reconstruction inputs only. Use GT only
   afterward for evaluation.

> **The one rule.** Phase 2 may read `fragments_graph`, reconstruction metadata, and
> optionally the raw image. It must never read `gt_graph`,
> `gt_node_canonical_label`, `gt_edge_error`, `gt_merge_labels`, or
> `gt_merge_sites` while making a prediction.

---

## 2) Dataset schema

### The two zones

Every `_add.pkl` is one dictionary:

| Zone | Keys | Phase 1 | Phase 2 detector |
|---|---|---:|---:|
| Reconstruction | `fragments_graph`, `anisotropy`, `min_cable_length`, `node_spacing`, `img_path` | read | read |
| Answer key | `gt_graph`, `gt_node_canonical_label`, `gt_edge_error`, `gt_merge_labels`, `gt_merge_sites` | read | **never read** |

Full schema:

| Key | Type | Meaning |
|---|---|---|
| `fragments_graph` | `SkeletonGraph` | UNet reconstruction and detector input. |
| `gt_graph` | `SkeletonGraph` | Human traced neurons. |
| `anisotropy` | tuple | Microns per voxel in `(x, y, z)`; read it from each cache. |
| `min_cable_length` | int | Minimum retained fragment cable length in µm; controls pair reachability. |
| `node_spacing` | int/optional | Target skeleton-node spacing in µm; node counts are not cable lengths. |
| `img_path` | str | Public raw fused-image path. |
| `segmentation_path` | str | Provenance only; do not depend on it. |
| `gt_node_canonical_label` | `(N_gt,) int64` | Predicted segment id at each GT node; `0` means unlabelled. Primary split signal. |
| `gt_edge_error` | `(E_gt,) uint8` | `0=correct`, `1=split`, `2=omit`, `3=merged`. |
| `gt_merge_labels` | integer array | Merge segment ids; a contra-indication for a proposed split join. |
| `gt_merge_sites` | list of dicts | Merge locations; not a split-site list. |

The four GT label objects are also attached to `gt_graph` in current caches, but use
one consistent access convention within an analysis.

### Canonical split definitions

The cache supports several related but different split quantities:

| Quantity | Definition | Consequence |
|---|---|---|
| `# Splits` | Number of distinct nonzero predicted labels on a GT neuron minus one | Counts labels, not every transition. |
| `% Split Edges` | GT-edge fraction whose two endpoint labels are different and nonzero | Misses transitions separated by zero-labelled omit runs. |
| Split Rate | Segmented run length divided by number of splits | Depends on omission and split-edge rates. |
| ERL | Run-length-weighted score; labels implicated in merges contribute zero | A wrong join can be worse than a missed split. |

Design consequences:

1. A sequence such as `A-B-A` has two transition sites but only one extra distinct
   label. Deduplicate targets by GT neuron and unordered segment pair.
2. A zero-labelled run between `A` and `B` is not a canonical split edge, but it still
   separates two predicted labels that should connect. Include such gap-spanning
   splits in the derived key.
3. Do not maximize split recall by joining aggressively. A bad join can manufacture a
   merge and destroy the ERL contribution of both neurons.

### Fragment-graph API and units

`fragments_graph` is a NetworkX-like `SkeletonGraph`.

- `frag.node_xyz`: `(N, 3)` float array in **(x, y, z) microns**.
- `frag.node_radius`: radius estimate per node.
- `frag.node_component_id`: component id per node.
- `frag.component_id_to_swc_id`: component id to a string such as
  `"32323215387.0"`; the integer before `.0` is the segment id.
- `frag.degree[n]`, `frag.neighbors(n)`: graph topology.
- `frag.dist(i, j)`: Euclidean distance in microns.
- `frag.leaf_nodes()`: degree-1 nodes.
- `frag.branching_nodes()`: degree greater than 2.
- `frag.cable_length(root=n)`: component cable length.
- `frag.nodes_with_segment_id(seg_id)`: all nodes carrying a segment id.
- `frag.kdtree`: spatial tree over fragment nodes.
- `frag.node_voxel(i)`: voxel coordinate in **(z, y, x)** order.

Do not confuse xyz microns with zyx voxels. Do not treat skeleton node count as a
physical length: spline resampling makes node density nonuniform on short edges.

The component-to-segment mapping is not necessarily one-to-one. One segment id may
own more than one disconnected component. Build both maps once:

```python
from collections import defaultdict

comp_nodes = defaultdict(list)
for n in frag.nodes:
    comp_nodes[int(frag.node_component_id[n])].append(n)

def comp_segment_id(comp_id):
    return int(frag.component_id_to_swc_id[comp_id].split(".")[0])

seg_comps = defaultdict(list)
for comp_id in comp_nodes:
    seg_comps[comp_segment_id(comp_id)].append(comp_id)
```

All component-level features should be precomputed from `comp_nodes` once. Repeating
`cable_length`, bounding-box, or branch-count traversals per endpoint is unnecessary.

### Install and load

The package must be importable before unpickling because it registers
`SkeletonGraph`:

```bash
python -c "import agentic_neuron_proofreader"
```

If missing, install the project package in an appropriate environment. Python 3.9 or
newer is required. A cache can require more than 20 GB RAM, so load one brain at a
time.

```python
import pickle
import numpy as np
import agentic_neuron_proofreader  # noqa: required for unpickling

with open("cache/dataset_cache_794495_mcl100_add.pkl", "rb") as f:
    payload = pickle.load(f)

frag = payload["fragments_graph"]
gt = payload["gt_graph"]
node_label = np.asarray(payload["gt_node_canonical_label"])
edge_error = np.asarray(payload["gt_edge_error"])
anisotropy = tuple(payload["anisotropy"])
min_cable_length = int(payload["min_cable_length"])
node_spacing = payload.get("node_spacing")

assert node_label.size, "Expected an _add cache with GT labels"
```

Compare results only between like-for-like `min_cable_length` settings. An `mcl10`
and an `mcl100` cache have different fragment populations and different reachable
answer keys.

### Deriving the split answer key

There is no `gt_split_sites` array. Derive it during Phase 1/scoring only by recording
adjacency between distinct nonzero labels, either directly or across a connected
zero-labelled run:

```python
import networkx as nx
import numpy as np

UNLABELED = 0
EDGE_SPLIT = 1

def derive_split_key(gt, node_label, edge_error):
    """Return {(gt_neuron, frozenset({a, b})): {xyzs, kind}}."""
    node_label = np.asarray(node_label)
    xyz = np.asarray(gt.node_xyz)
    neuron_of = {n: gt.node_segment_id(n) for n in gt.nodes}
    key = {}

    def add(neuron, a, b, site, kind):
        if int(a) == 0 or int(b) == 0 or int(a) == int(b):
            return
        k = (neuron, frozenset((int(a), int(b))))
        rec = key.setdefault(k, {"xyzs": [], "kind": kind})
        rec["xyzs"].append(tuple(map(float, site)))
        if rec["kind"] != kind:
            rec["kind"] = "both"

    # Direct label transitions.
    for k, (i, j) in enumerate(gt.edges):
        if int(edge_error[k]) == EDGE_SPLIT:
            add(neuron_of[i], node_label[i], node_label[j],
                0.5 * (xyz[i] + xyz[j]), "direct")

    # Distinct labels on the flanks of an unlabelled connected run.
    zero_nodes = [n for n in gt.nodes if int(node_label[n]) == UNLABELED]
    for cc in nx.connected_components(gt.subgraph(zero_nodes)):
        flank = {int(node_label[m])
                 for n in cc for m in gt.neighbors(n)
                 if int(node_label[m]) != UNLABELED}
        if len(flank) < 2:
            continue
        neuron = neuron_of[next(iter(cc))]
        centroid = np.mean([xyz[n] for n in cc], axis=0)
        flank = sorted(flank)
        for x in range(len(flank)):
            for y in range(x + 1, len(flank)):
                add(neuron, flank[x], flank[y], centroid, "gap")

    return key
```

Partition pair targets immediately:

```python
def partition_reachable(split_key, seg_comps):
    reachable, unreachable = {}, {}
    for key, rec in split_key.items():
        _, pair = key
        a, b = tuple(pair)
        target = reachable if a in seg_comps and b in seg_comps else unreachable
        target[key] = rec
    return reachable, unreachable
```

### Deriving endpoint labels

Build one KDTree over GT nodes and one over derived split sites. Iterate through
fragment leaves once:

- **positive / truncation**: the leaf is near a derived split site and its own segment
  id is one side of that split pair;
- **negative / terminal**: it is not positive and its nearest GT node is degree 1;
- **ambiguous**: it is near traced mid-cable but is neither positive nor a GT terminal;
- **unadjudicable**: it is farther than the GT proximity threshold.

The own-segment requirement is essential: a leaf merely near another neuron's split
site is not thereby a positive.

Use thresholds such as `near_gt_um=10` and `site_tol_um=15` as documented starting
points, then check their sensitivity. Report counts for all four categories before
fitting. The positive class is likely small, so use class weighting or balanced
sampling.

### Raw-image access

Open the image once:

```python
import os
from agentic_neuron_proofreader.utils import img_util

os.environ["AWS_EC2_METADATA_DISABLED"] = "true"
image = img_util.TensorStoreImage(payload["img_path"])

def xyz_to_voxel(xyz_um, anisotropy):
    """(x,y,z) microns to integer (z,y,x) voxel."""
    return tuple(int(c / a) for c, a in zip(xyz_um, anisotropy))[::-1]
```

Remote image reads can dominate runtime. Never read a patch for every possible
endpoint pair. Read one compact endpoint-and-forward-ray patch for selected endpoints,
and cache or batch overlapping spatial chunks.

---

## 3) Intent

### Scientific deliverable

The primary goal is to understand **why the UNet stops following a process**. Compare
false truncations against genuine terminals across:

1. endpoint topology and component context;
2. endpoint geometry and caliber behavior;
3. raw-image evidence immediately beyond the endpoint.

Phase 2 validates whether the discovered signature is recoverable from reconstruction
inputs without GT. Report geometry-only and geometry-plus-image performance
separately. The difference measures the incremental value of the raw fluorescence.

This is a detection task. A ranked list of likely truncations satisfies the primary
goal. Applying joins and modifying the graph is a separate correction task.

### Simpler computational workflow

Use the following funnel. Do not begin with global pair generation.

```text
load one cache
  -> derive GT split key and endpoint labels (Phase 1 only)
  -> precompute component and endpoint geometry once
  -> characterize cheap features on labelled endpoints
  -> fit/evaluate a geometry-only endpoint detector
  -> read raw image for labelled samples needed for the comparison
  -> measure image-feature improvement
  -> freeze the detector
  -> score every leaf GT-blind using cached cheap features
  -> image-refine only top-K / above-threshold endpoints
  -> return ranked truncations
  -> optionally search <= K partners for shortlisted truncations
```

This changes the expensive relation search from a whole-dataset operation into a
small extension conditioned on endpoint evidence.

### Step 1 — precompute once

Build these structures once per brain:

- leaf list and leaf KDTree;
- node-to-component and component-to-segment arrays/maps;
- segment-to-components map;
- per-component cable length, node count, leaf count, branch count, bounding box;
- per-leaf outward direction;
- per-leaf tip-to-shaft taper ratio;
- per-leaf nearest other-component endpoint distance and neighbor count.

The outward direction follows the leaf's cable inward for a fixed physical reach and
then reverses the vector. The taper ratio compares tip radius against the inward shaft:
a flat ratio near one suggests a cut; a smaller ratio suggests a biological taper.
Use physical distance, not a fixed number of nodes.

Do not call a component-wide traversal repeatedly for multiple leaves in the same
component. Store a compact feature table keyed by leaf node id.

### Step 2 — minimal endpoint feature catalogue

Compute for every endpoint:

| Feature | Interpretation |
|---|---|
| Tip-to-shaft taper ratio | Flat ending favors truncation; taper favors terminal. |
| Tip radius | Absolute caliber context. |
| Nearest other-component endpoint distance | A possible continuation nearby favors truncation. |
| Other-component leaf count within 20 µm | Partner availability and local crowding. |
| Component cable length | Short pieces may be split debris. |
| Path distance to nearest branch point | Separates short stubs from long distal terminals. |
| Local direction stability/curvature | Detects noisy endings and sharp turns. |

These are the default geometry/topology features. Add more only after showing that
the minimal set is insufficient. Avoid global cycle, detour, or all-pairs path
calculations in the default workflow.

### Step 3 — characterize endpoint classes

On GT-adjudicable endpoints:

1. plot positive and negative feature distributions;
2. report effect sizes and uncertainty, not only significance tests;
3. inspect class balance and distributions per brain;
4. fit a class-weighted logistic regression or similarly small interpretable model;
5. split train/test by brain, or at minimum by component, to prevent spatial leakage;
6. report precision-recall behavior and recall at a fixed proofreading budget.

The hand-set baseline may combine taper, proximity, and component shortness, but the
scientific result should come from measured feature separation and held-out
performance rather than arbitrary weights.

### Step 4 — add raw-image evidence selectively

For a selected endpoint, probe a short ray leading outward from the leaf. Extract a
small image feature set:

| Feature | Question |
|---|---|
| Forward-ray mean / local background | Does fluorescence continue beyond the endpoint? |
| Forward-ray decay length | Does signal disappear immediately like a real terminal? |
| Minimum ray intensity / cable intensity | Is there a weak but continuous bottleneck? |
| Local SNR | Was the network operating in a dim/noisy region? |
| Optional off-axis maximum | Did the neurite turn away from the straight ray? |

During characterization, read patches for all positives and a reproducible balanced
negative sample. During blind inference, first score all leaves with cheap geometry,
then read images only for a top fraction, fixed review budget, or score-threshold
shortlist. Record request count, bytes read, and wall time.

A genuine terminal should decay into background. A split is expected to retain weak
continuation evidence beyond the reconstructed endpoint. This is the key image-level
hypothesis to test.

### Step 5 — GT-blind endpoint detection

Freeze preprocessing, model, thresholds, and image-shortlist policy before evaluation.
The detector output should be a ranked list:

```python
{
    "node": int,
    "component_id": int,
    "segment_id": int,
    "xyz": (float, float, float),
    "geometry_score": float,
    "image_score": float | None,
    "final_score": float,
}
```

Audit the Phase-2 feature table and code path to ensure that no GT-derived value is an
input or filtering condition. GT proximity is allowed only later in scoring.

### Step 6 — optional bounded partner search

Only after endpoint detection, attempt to name the far side for shortlisted
truncations:

1. For each shortlisted leaf, query endpoints in different components within a small
   physical radius, initially 20 µm.
2. Retain at most the nearest `K=3` candidates before expensive feature computation.
3. Reject pairs whose two outward directions do not face each other.
4. Reject strong radius/caliber discontinuities.
5. Rank survivors by gap, mutual alignment, antiparallel direction, radius continuity,
   and the two endpoint scores.
6. Read a single bridge image patch only for the best candidate, or at most the best
   two if uncertainty must be estimated.
7. Return partner proposals without modifying the reconstruction.

With `S` shortlisted endpoints and capped `K`, expensive pair work is approximately
`O(S*K)` rather than an unbounded enumeration over all leaves. Deduplicate unordered
pairs. If producing a final non-conflicting proposal list, use greedy score ordering,
consume each endpoint at most once, and use union-find to reject cycle-creating joins.

Do not let optional pairing delay or redefine the primary endpoint discovery result.

### Scoring endpoint detection

Score only endpoints the derived key can adjudicate:

```python
def score_truncations(predictions, trunc_key):
    truth = {n: y for n, y in trunc_key.items() if y is not None}
    pred = {p["node"] for p in predictions}
    pred_adj = pred & set(truth)
    tp = sum(truth[n] == 1 for n in pred_adj)
    n_pos = sum(y == 1 for y in truth.values())
    return {
        "recall_endpoint": tp / n_pos if n_pos else float("nan"),
        "precision_endpoint_adjudicable": (
            tp / len(pred_adj) if pred_adj else float("nan")
        ),
        "n_truth_truncations": n_pos,
        "n_truth_terminals": sum(y == 0 for y in truth.values()),
        "n_ambiguous_excluded": sum(y is None for y in trunc_key.values()),
        "n_pred": len(pred),
        "n_pred_adjudicable": len(pred_adj),
        "n_pred_unadjudicable": len(pred) - len(pred_adj),
    }
```

Also report:

- precision-recall curve and average precision;
- recall among the top-N endpoints for realistic review budgets;
- geometry-only versus geometry-plus-image results;
- per-brain results and a pooled summary;
- runtime, peak memory, image request count, and bytes read.

### Scoring optional pairs

Pair evaluation is separate from endpoint evaluation. Report:

- recall against `reachable_key` only;
- number of reachable and unreachable GT pairs;
- precision among proposals whose two segment ids are GT-adjudicable;
- number of unadjudicable proposals;
- partner accuracy conditional on detecting a correct truncation;
- number of proposed joins that would fuse two substantially traced, distinct
  neurons.

The merge-creating count must accompany every pair-recall result. A missed split loses
continuity; a wrong join can merge two neurons and reduce their ERL contribution to
zero. Increasing recall by joining everything nearby is not a valid discovery.

### Computational safeguards and diagnostics

Required safeguards:

- never enumerate all endpoint pairs;
- use KDTree spatial queries;
- cap partners before computing pair features;
- precompute component and endpoint features;
- use physical-distance windows rather than node-count windows;
- cache or batch overlapping image reads;
- process one brain at a time;
- use deterministic sampling and record seeds;
- log candidate counts before and after every filter;
- time answer-key construction, feature extraction, image I/O, fitting, inference,
  and scoring separately.

At minimum, record:

```text
n_nodes
n_components
n_leaves
n_positive_truncations
n_genuine_terminals
n_ambiguous
n_untraced
n_geometry_shortlist
n_image_patches
n_partner_candidates_before_filter
n_partner_candidates_after_filter
n_reachable_key_pairs
n_unreachable_key_pairs
```

### Expected deliverables

1. A validated split-pair key and endpoint truncation key, with category counts.
2. Feature-distribution comparisons for truncations versus terminals.
3. A compact geometry-only GT-blind endpoint detector.
4. A controlled measurement of the incremental contribution of raw-image features.
5. A ranked list of suspicious endpoints and review-budget metrics.
6. Runtime, memory, and image-I/O measurements showing that computation is bounded.
7. Optionally, bounded partner proposals with reachability and merge-risk accounting.

The work succeeds if it identifies reproducible, interpretable signals of where the
segmentation network stops following a neurite and demonstrates that those signals
can prioritize likely split errors without GT at inference time. Exact automatic
repair is not necessary for that conclusion.
