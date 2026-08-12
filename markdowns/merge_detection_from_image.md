# Understanding merge errors from ExaSPIM image, GT, and fragments

This document is the complete task specification. The agent will receive exactly
three colocated files: this markdown, one trusted
`dataset_cache_<brain>_mcl<N>_add.pkl`, and `zihan_gcs_token.json`. Do not assume that
repository notebooks, configuration files, metrics CSVs, pre-extracted patches, or
the dense segmentation volume are available. Everything needed to locate merge
errors, align the data sources, authenticate image access, read the real image,
construct controls, and evaluate image evidence is described below.

---

## 1) Dataset context

### What is in the cache

The cache connects three registered views of one brain:

1. **Raw ExaSPIM image** — measured fluorescence voxels, read on demand from
   `payload["img_path"]`. The image itself is not stored in the pickle.
2. **GT skeleton** — sparse human tracings in `payload["gt_graph"]`. A connected
   component is one traced GT neuron; there is no dense GT mask.
3. **Predicted fragments/segments** — skeletons extracted from the automated
   segmentation in `payload["fragments_graph"]`, together with predicted segment
   labels sampled at GT nodes and precomputed merge labels/sites.

Image voxels, GT skeletons, fragment skeletons, and predicted segment IDs are
different representations. Registration lets them be compared at the same physical
location; it does not make a skeleton a dense mask or a 2-D MIP a proof of 3-D
contact.

### Vocabulary and identity levels

- A **GT neuron** is a connected component of `gt_graph`, identified by a name such
  as `N007-794495-JG` from `gt_graph.node_segment_id(node)`.
- A **fragment component** is one connected component/SWC in `fragments_graph`.
  Its internal component ID indexes `fragments_graph.component_id_to_swc_id`.
- A **predicted segment ID** is the prefix before `.` in a fragment SWC ID. Obtain
  it from `fragments_graph.node_segment_id(node)` or
  `str(fragments_graph.component_id_to_swc_id[c]).split(".")[0]`.
- `gt_node_canonical_label[i]` is the predicted segment ID sampled at GT node `i`;
  `0` means unlabeled/background at that GT location.

GT-neuron names, fragment-component IDs, and predicted-segment IDs are different
namespaces. One predicted segment may be represented by one or more fragment
components. Compare normalized predicted-segment IDs, never display colors or raw
component IDs, when matching the reconstruction to the answer key.

### What a merge means

A merge error occurs when one predicted segment fuses structures that should be
separate—for example two neurons crossing or touching, a foreign neurite attached to
a soma, or a predicted cable that leaves a traced neuron through untraced material
and later returns.

The cache represents merges at two distinct levels:

- **Segment level:** `gt_merge_labels` is a global set/array of predicted segment IDs
  flagged as merges. It is the union of a node-count rule and a geometric walk.
- **Site level:** `gt_merge_sites` is a list of records produced by the geometric
  merge walk
  `{"segment_id", "gt_neuron", "xyz"}`. Each record identifies one detected merge
  event and its physical `(x,y,z)` location in microns.

Do not use these interchangeably. A segment may contain multiple merge sites.
`len(gt_merge_sites)` is the number of stored merge events; it is not necessarily
`len(gt_merge_labels)`. `gt_edge_error == 3` is useful for coloring affected GT
edges, but counting those edges is not the canonical merge-event count.
Because `gt_merge_labels` is a union while `gt_merge_sites` comes from the geometric
walk, a merge-labeled segment can have no stored site coordinate. Such a segment is
valid for segment-level evaluation but cannot become a site-level image positive
without a separately defined localization procedure.

### What the answer key covers

GT is sparse. The node-count branch detects predicted segments supported by multiple
sufficiently represented GT neurons. The geometric-walk branch can also flag a
segment that fuses a traced neuron to untraced material, so membership in
`gt_merge_labels` does not guarantee that two traced GT neurons share that label.

The answer key can miss a merge when no traced neuron touches the segment, when an
excursion is below the labeling thresholds, or when sparse tracing cannot adjudicate
the candidate. Therefore:

- absence from `gt_merge_labels` means “not labeled as a merge by this sparse answer
  key,” not guaranteed biological correctness;
- precision must be reported on adjudicable predicted segments, with unadjudicable
  detections listed separately;
- nearby sites and sites on the same segment/GT neuron are correlated.

### Scientific question and information firewall

The scientific question is: **what evidence in the real raw fluorescence identifies
a merge site, and how does that evidence correspond to the affected GT neuron and
predicted segment skeleton?** Candidate image signatures include two bright processes
with insufficient dark separation, crossing or converging ridges, incompatible local
orientations, intensity continuity across an erroneous bridge, or image artifacts
that obscure a true boundary. These are hypotheses to test, not assumptions.

Use two explicit stages:

1. **Context/candidate stage:** GT and fragment information may locate merge sites,
   construct controls, associate a candidate with a predicted segment, verify
   registration, and group samples for evaluation.
2. **Image-evidence stage:** an image-only score may receive only the candidate
   coordinate and the real raw-image patch. Fragment topology, radius, component
   size, segment identity, GT geometry, and answer-key labels may not enter its
   predictor values.

A practical system may use a fragment-based proposal generator followed by an
image-only scorer. Name that architecture explicitly: candidate generation is
fragment-based even if the final candidate score is image-only. If fragment geometry
and image features enter the same classifier, call it multimodal and report separate
image-only, fragment-only, and combined baselines.

### Hard rules

- Never substitute random, synthetic, simulated, or mock voxels for a failed real
  image read. Report the exception and mark the sample unusable.
- Never infer a dense segmentation mask from the sparse fragment skeleton and call
  it ground truth. Any rasterized/dilated skeleton is only an approximation.
- Never download or materialize the whole ExaSPIM volume. Read bounded patches.
- Use identical patch geometry, preprocessing, and validity criteria for merge sites
  and controls.
- Do not unpickle an untrusted file: Python pickle can execute code.
- Do not claim a whole-brain detector from performance on GT-centered sites alone.

---

## 2) Dataset schema

### Environment and loading

Unpickling requires the package that defines `SkeletonGraph`. First try:

```bash
python -c "import agentic_neuron_proofreader"
```

If unavailable, install
`https://github.com/AllenInstitute/agentic-neuron-proofreader` and ensure `numpy`,
`scipy`, `networkx`, `tensorstore`, and `matplotlib` are installed.
NumPy and SciPy must be binary-compatible. On Allen Institute compute nodes, the
`panda` conda environment is normally suitable.
Python 3.10 or newer is required. In a fresh compatible environment, run:

```bash
python -m pip install "git+https://github.com/AllenInstitute/agentic-neuron-proofreader.git" numpy scipy networkx tensorstore matplotlib
```

Load with plain `pickle.load` so schema validation happens before cloud access:

```python
import os
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
__import__("agentic_neuron_proofreader")  # registers SkeletonGraph pickle classes

DELIVERY_DIR = Path.cwd()
pkl_candidates = sorted(DELIVERY_DIR.glob("*_add.pkl"))
if len(pkl_candidates) != 1:
    raise RuntimeError(
        "expected exactly one *_add.pkl beside this markdown and token; "
        f"found {len(pkl_candidates)}: {[p.name for p in pkl_candidates]}"
    )
PKL = pkl_candidates[0]
with open(PKL, "rb") as f:
    payload = pickle.load(f)

required = {
    "img_path", "anisotropy", "gt_graph", "fragments_graph",
    "gt_node_canonical_label", "gt_edge_error",
    "gt_merge_labels", "gt_merge_sites",
}
missing = required - payload.keys()
if missing:
    raise RuntimeError(f"not a complete _add.pkl; missing {sorted(missing)}")

gt = payload["gt_graph"]
fragments = payload["fragments_graph"]
node_label = np.asarray(payload["gt_node_canonical_label"])
edge_error = np.asarray(payload["gt_edge_error"])
merge_labels = {int(x) for x in np.asarray(payload["gt_merge_labels"])}
merge_sites = list(payload["gt_merge_sites"])
anisotropy_xyz = np.asarray(payload["anisotropy"], dtype=float)
gt_edges = list(gt.edges)

assert anisotropy_xyz.shape == (3,) and np.all(anisotropy_xyz > 0)
assert len(node_label) == gt.number_of_nodes()
assert len(edge_error) == len(gt_edges)
for site in merge_sites:
    assert {"segment_id", "gt_neuron", "xyz"} <= site.keys()
    assert len(site["xyz"]) == 3
    assert int(site["segment_id"]) in merge_labels

if not merge_sites:
    raise RuntimeError(
        "this cache has no stored GT merge sites; site-level image "
        "characterization cannot be run from this cache alone"
    )
```

Budget tens of GB of RAM per cache and process one cache at a time.

### Top-level fields

| Key | Meaning and correct use |
|---|---|
| `img_path` | Path to the real fused raw image, usually public S3. Open this exact path. |
| `segmentation_path` | Provenance for the dense predicted segmentation. It may be private GCS and is not embedded in the pickle; do not assume access. |
| `anisotropy` | Microns per voxel in **(x,y,z)** order. Always read it from the cache. |
| `min_cable_length` | Minimum fragment cable length retained at cache construction. Shorter skeletons are absent. |
| `node_spacing` | Target skeleton resampling distance in microns. It is not the exact length of every graph edge. |
| `gt_graph` | Sparse human-traced `SkeletonGraph`; connected components are GT neurons. |
| `fragments_graph` | Automated fragment `SkeletonGraph` derived from predicted segments. |
| `gt_node_canonical_label` | `int64`, shape `(N_gt,)`: predicted segment ID at each GT node; `0` means unlabeled/background. |
| `gt_edge_error` | `uint8`, shape `(E_gt,)`, parallel to `list(gt_graph.edges)`: `0=correct`, `1=split`, `2=omit`, `3=merged`. Preserve this exact edge ordering. |
| `gt_merge_labels` | Global predicted-segment IDs flagged by node-count rule ∪ geometric walk. This is a segment-level label set. |
| `gt_merge_sites` | Site-level records: `{"segment_id": int, "gt_neuron": str, "xyz": (x,y,z) µm}`. Multiple records may share a segment. |
| `fragments_path`, `gt_path` | Source-SWC provenance only; the graph objects are already cached. |

The four added label objects can also appear as `gt.node_label`, `gt.edge_error`,
`gt.merge_labels`, and `gt.merge_sites`. Prefer top-level keys to make provenance
explicit.

### `SkeletonGraph` fields and methods

Both graphs subclass `networkx.Graph` and use contiguous integer node IDs indexing
parallel arrays:

- `graph.node_xyz[node]`: physical **(x,y,z)** coordinate in microns;
- `graph.node_radius[node]`: skeleton-radius metadata;
- `graph.node_component_id[node]`: integer connected-component ID;
- `graph.component_id_to_swc_id[component]`: SWC identity;
- `graph.node_segment_id(node)`: SWC prefix;
- `graph.node_voxel(node)`: integer image voxel in **(z,y,x)** order;
- `graph.kdtree.query(xyz_um)`: nearest node in physical micron space;
- `graph.nodes_in_patch(origin_zyx, shape_zyx, ...)`: local floating-point ZYX
  node coordinates inside a patch;
- `graph.edges_in_patch(origin_zyx, shape_zyx, ...)`: local ZYX edge endpoints
  whose two nodes lie inside the patch.

On `gt_graph`, `node_segment_id` returns a GT-neuron name. On `fragments_graph`, it
returns a predicted segment ID. `graph.dist(i,j)` is Euclidean distance in microns;
it is not a graph-path distance unless `i,j` are adjacent and that edge is intended.

### Normalizing and indexing predicted segments

Normalize IDs before comparing fragment SWCs, canonical labels, and merge-site IDs:

```python
def normalize_segment_id(value):
    """Canonical integer ID; returns None for background/unlabeled."""
    text = str(value).split(".")[0]
    return None if text in {"0", "0.0", "None", ""} else int(text)

segment_to_components = defaultdict(set)
segment_to_nodes = defaultdict(list)
component_to_nodes = defaultdict(list)

for node in fragments.nodes:
    node = int(node)
    component = int(fragments.node_component_id[node])
    segment = normalize_segment_id(fragments.node_segment_id(node))
    segment_to_components[segment].add(component)
    segment_to_nodes[segment].append(node)
    component_to_nodes[component].append(node)

missing_segment_skeletons = merge_labels - set(segment_to_nodes)
print("merge labels / sites:", len(merge_labels), len(merge_sites))
print("merge labels without retained fragment nodes:",
      len(missing_segment_skeletons))
```

A merge label can lack retained fragment nodes because fragment filtering and label
construction do not guarantee that every labeled segment has a surviving local
skeleton. Record this case; do not silently match it to a different segment.

### Coordinate conversion and patch convention

```text
physical skeleton/site coordinate: (x, y, z) microns
image / patch voxel coordinate:     (z, y, x) voxels
anisotropy:                          (x, y, z) microns per voxel
```

Use truncation to match `SkeletonGraph.node_voxel`:

```python
def xyz_um_to_zyx_voxel(xyz_um):
    xyz_um = np.asarray(xyz_um, dtype=float)
    return tuple((xyz_um / anisotropy_xyz).astype(int)[::-1])

def patch_origin(center_zyx, shape_zyx):
    return tuple(int(c - s // 2) for c, s in zip(center_zyx, shape_zyx))
```

`TensorStoreImage.read(center_zyx, shape_zyx)` takes a patch **center**.
`nodes_in_patch` and `edges_in_patch` take the patch minimum-corner **origin**.
Passing the center to the graph methods shifts the skeleton by half a patch.

For a stored merge site, use `site["xyz"]` directly as the physical center. Do not
replace it with the nearest fragment node before reading the image: the displacement
between the stored site and the fragment skeleton is itself an alignment audit.

### Matching a merge site to its target segment

Match by segment ID first, then by distance within that segment. A global nearest
fragment query can select an unrelated segment passing nearby and is not valid for
positive association:

```python
from scipy.spatial import cKDTree

def match_merge_site(site):
    segment = normalize_segment_id(site["segment_id"])
    nodes = segment_to_nodes.get(segment, [])
    record = {
        "segment_id": segment,
        "gt_neuron": str(site["gt_neuron"]),
        "xyz_um": np.asarray(site["xyz"], dtype=float),
        "fragment_node": None,
        "fragment_component": None,
        "distance_um": np.nan,
    }
    if not nodes:
        return record
    tree = cKDTree(np.asarray(fragments.node_xyz[nodes], dtype=float))
    distance, local_index = tree.query(record["xyz_um"])
    node = int(nodes[int(local_index)])
    record.update(
        fragment_node=node,
        fragment_component=int(fragments.node_component_id[node]),
        distance_um=float(distance),
    )
    return record

matched_sites = [match_merge_site(site) for site in merge_sites]
distances = np.asarray(
    [r["distance_um"] for r in matched_sites if np.isfinite(r["distance_um"])]
)
print("matched merge sites:", len(distances), "/", len(matched_sites))
if len(distances):
    print("site→same-segment distance µm p50/p90/max:",
          np.percentile(distances, [50, 90, 100]))

valid_matched_sites = [
    record for record in matched_sites
    if record["fragment_node"] is not None
    and np.isfinite(record["distance_um"])
]
if not valid_matched_sites:
    raise RuntimeError(
        "no GT merge site has a retained skeleton for the same segment; "
        "the required three-way image/fragment/GT audit cannot be run"
    )
example_site = valid_matched_sites[0]
```

Do not invent one universal acceptable distance without examining `node_spacing`,
the fragment filtering threshold, and the empirical distribution. Large or
systematic distances require explicit reporting. A nearest node from the wrong
segment does not rescue a failed same-segment match.

### Reading and auditing the real image

The raw image is external to the pickle. Configure authentication **before**
constructing `TensorStoreImage`, based on `payload["img_path"]`:

- `s3://`: the usual raw image is public. Disable EC2 metadata probing and use
  anonymous access; no AWS key is needed.
- `gs://`: treat the image as private and use the supplied
  `zihan_gcs_token.json` file.

The delivery contains this markdown, the `_add.pkl`, and `zihan_gcs_token.json` in
the same directory. Start the analysis with that delivery directory as the current
working directory. Resolve the credential from the filename—not from a repository
path, username, or machine-specific absolute path. For GCS:

1. resolve `Path("zihan_gcs_token.json")` in the delivery directory;
2. verify that it is a readable file;
3. assign its absolute resolved path to `GOOGLE_APPLICATION_CREDENTIALS`;
4. only then construct `TensorStoreImage`.

Use this complete helper:

```python
from agentic_neuron_proofreader.utils import img_util


BUNDLED_GCP_CREDENTIALS = Path("zihan_gcs_token.json")


def configure_image_access(image_path):
    """Configure public-S3 or authenticated-GCS access before image open."""
    os.environ["AWS_EC2_METADATA_DISABLED"] = "true"

    if image_path.startswith("s3://"):
        print("Image access: anonymous/public S3")
        return "anonymous-public-s3"

    if not image_path.startswith("gs://"):
        raise RuntimeError(f"unsupported image path scheme: {image_path}")

    credentials = BUNDLED_GCP_CREDENTIALS.expanduser().resolve()

    if not credentials.is_file():
        raise RuntimeError(
            "private GCS image requires the supplied zihan_gcs_token.json in "
            f"the delivery directory; looked at {credentials}"
        )

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(credentials)
    print(f"Image access: authenticated GCS (supplied credential: {credentials})")
    return "authenticated-gcs"


img_path = str(payload["img_path"])
image_access = configure_image_access(img_path)

image = img_util.TensorStoreImage(img_path)
PATCH_SHAPE = (128, 128, 128)  # (z,y,x), fixed across classes

def read_and_audit_patch(xyz_um):
    center_zyx = xyz_um_to_zyx_voxel(xyz_um)
    patch = np.asarray(image.read(center_zyx, PATCH_SHAPE))
    finite = patch[np.isfinite(patch)]
    if patch.shape != PATCH_SHAPE or not finite.size:
        raise RuntimeError(
            f"invalid real patch at {tuple(xyz_um)}: {patch.shape}, {patch.dtype}"
        )
    audit = {
        "img_path": img_path,
        "image_access": image_access,
        "xyz_um": tuple(map(float, xyz_um)),
        "center_zyx": center_zyx,
        "shape": tuple(patch.shape),
        "dtype": str(patch.dtype),
        "finite": int(finite.size),
        "nonzero": int(np.count_nonzero(finite)),
        "percentiles_0_50_99_99.9_100":
            np.percentile(finite, [0, 50, 99, 99.9, 100]).tolist(),
    }
    return patch, audit

example_patch, example_audit = read_and_audit_patch(example_site["xyz_um"])
print(example_audit)
```

Equivalently, from the directory containing all three delivered files, the agent may
configure the supplied credential before Python:

```bash
export GOOGLE_APPLICATION_CREDENTIALS="$PWD/zihan_gcs_token.json"
python your_analysis.py
```

The credential must grant `storage.objects.get` access to the bucket/object. Verify
that the path exists without printing or copying the JSON contents. Never commit the
credential, embed it in generated code/reports, or fall back to synthetic data when
authentication fails. `segmentation_path` may also point to private GCS, but this task
does not require opening the dense segmentation; authentication here is only for the
actual `img_path` being read.

The credential contents must never be printed, copied into generated code/reports,
or committed. If the supplied JSON is absent or unreadable, stop and report the
failure rather than searching unrelated machine paths.

The exact image path, physical and voxel centers, requested/returned shape, dtype,
finite/nonzero counts, and intensity percentiles are mandatory evidence that real
image information was used. Apply the same boundary/read-failure rule to positives
and controls. Never repair one class with silent padding or synthetic values.

### Extracting one aligned image–fragments–GT merge patch

Use one stored merge-site center and one extent for all three views:

```python
site_record = example_site
center_xyz = site_record["xyz_um"]
center_zyx = xyz_um_to_zyx_voxel(center_xyz)
shape_zyx = PATCH_SHAPE
origin_zyx = patch_origin(center_zyx, shape_zyx)

img_patch = np.asarray(image.read(center_zyx, shape_zyx))

gt_nodes, gt_ids, gt_node_components = gt.nodes_in_patch(
    origin_zyx, shape_zyx, return_ids=True, return_components=True
)
gt_edges_local, gt_edge_components = gt.edges_in_patch(
    origin_zyx, shape_zyx, return_components=True
)

frag_nodes, frag_ids, frag_node_components = fragments.nodes_in_patch(
    origin_zyx, shape_zyx, return_ids=True, return_components=True
)
frag_edges_local, frag_edge_components = fragments.edges_in_patch(
    origin_zyx, shape_zyx, return_components=True
)

target_segment = site_record["segment_id"]
local_fragment_segments = {
    normalize_segment_id(fragments.node_segment_id(int(n))) for n in frag_ids
}
local_gt_neurons = {gt.node_segment_id(int(n)) for n in gt_ids}
local_gt_target_nodes = [
    int(n) for n in gt_ids
    if normalize_segment_id(node_label[int(n)]) == target_segment
]

print("target segment / attributed GT neuron:",
      target_segment, site_record["gt_neuron"])
print("local GT nodes/edges/neurons:",
      len(gt_nodes), len(gt_edges_local), len(local_gt_neurons))
print("local target-labeled GT nodes:", len(local_gt_target_nodes))
print("local fragment nodes/edges/segments:",
      len(frag_nodes), len(frag_edges_local), len(local_fragment_segments))
```

Render separate, identically registered figures using the package helpers:

```python
img_util.plot_mips(img_patch)

img_util.plot_skeleton_mips(
    {"UNet Fragments / Segments": {
        "nodes": frag_nodes,
        "node_components": frag_node_components,
        "edges": frag_edges_local,
        "components": frag_edge_components,
        "color": "cyan",
    }},
    shape_zyx,
    separate_rows=True,
)

img_util.plot_skeleton_mips(
    {"GT": {
        "nodes": gt_nodes,
        "node_components": gt_node_components,
        "edges": gt_edges_local,
        "components": gt_edge_components,
        "color": "lime",
    }},
    shape_zyx,
    separate_rows=True,
)
```

Fragment and GT component colors use independent palettes. Equal colors across
figures do not indicate identity. Match by normalized segment IDs and coordinates.
MIPs collapse depth; inspect XY, XZ, and YZ together and use 3-D slices for ambiguous
crossings or apparent contacts.

### Constructing positive and control samples

#### Positive sites

Use each valid `gt_merge_sites` record as one site-level positive. Preserve:

```text
sample_id, brain_id, xyz_um, center_zyx, segment_id, gt_neuron,
fragment_component, same_segment_distance_um, label=1, invalid_reason
```

Do not create one positive per `gt_edge_error == 3` edge; that changes the unit of
analysis from merge events to affected cable edges. Deduplicate only genuine duplicate
site records and do so **within segment ID**. Two different segments can have nearby
but distinct errors.

#### Primary clean controls

Use midpoints of GT edges satisfying all of the following:

- `gt_edge_error == 0`;
- both endpoint canonical labels are the same nonzero segment ID;
- that segment ID is not in `gt_merge_labels`;
- the candidate lies sufficiently far from every stored merge site;
- the requested real-image patch is valid.

```python
merge_site_tree = (
    cKDTree(np.asarray([s["xyz"] for s in merge_sites], dtype=float))
    if merge_sites else None
)

clean_controls = []
for edge_index, (i, j) in enumerate(gt_edges):
    if int(edge_error[edge_index]) != 0:
        continue
    a = normalize_segment_id(node_label[i])
    b = normalize_segment_id(node_label[j])
    if a is None or a != b or a in merge_labels:
        continue
    xyz = 0.5 * (
        np.asarray(gt.node_xyz[i], dtype=float)
        + np.asarray(gt.node_xyz[j], dtype=float)
    )
    distance = np.inf if merge_site_tree is None else merge_site_tree.query(xyz)[0]
    if distance < 50.0:
        continue
    clean_controls.append({
        "edge_index": edge_index,
        "nodes": (int(i), int(j)),
        "xyz_um": xyz,
        "segment_id": a,
        "gt_neuron": gt.node_segment_id(i),
        "label": 0,
    })
```

Match controls to positives without inspecting the image feature under test. Match
at minimum by brain, physical z region, distance to image boundary, GT-neuron region
(soma/branch/cable when known), local trace crowding, and physical field of view.
Because GT and fragment geometry used for matching can alter the sampled population,
report the matching variables and never include them silently in an image-only score.

#### Sensitivity controls

As a secondary analysis, sample points on the same merged segment but far from its
stored merge sites. This controls segment-level size and reconstruction bias while
asking what is locally special about the annotated site. These are not guaranteed
biological negatives: the same segment can contain another unlabeled merge. Report
them separately from primary GT-supported clean controls.

### Image evidence to measure

Every image-only feature must be computed from the actual raw patch:

- center intensity relative to a surrounding shell and local background;
- dark-gap width between bright ridges;
- number and separation of local intensity ridges or connected bright structures;
- multiscale Hessian/Frangi vesselness;
- 3-D structure-tensor eigenvalues, coherence, and orientation dispersion;
- evidence for two strong orientations, shallow-angle crossing, or convergence;
- intensity continuity across a candidate bridge;
- relative brightness of the putative processes;
- local entropy, frequency content, blur, saturation, stripe, or tile-boundary
  artifacts;
- stability of all signals across physical scales.

Estimate ridge directions from image voxels for an image-only score. Directions from
fragment edges or GT tangents are allowed only in an explicitly geometry-conditioned
or multimodal analysis. Avoid interpreting “two histogram modes” as two neurons
without spatial evidence; intensity modes can arise from background or acquisition
variation.

Use physical microns rather than fixed isotropic voxel radii. Prefer robust
within-patch normalization fitted identically across classes. Any learned threshold,
normalization, embedding, or feature selection must be fit inside the training fold.

### From characterization to GT-blind detection

A GT-centered experiment characterizes image signatures; it is not yet a deployable
detector. A practical GT-blind pipeline has two explicit steps:

1. **Proposal generation:** obtain candidate `(segment_id, xyz_um)` sites without
   reading any GT fields. Proposals may come from fragment branch points, close
   approaches, cycles, multiple-soma components, or an image proposal model.
2. **Image scoring:** read a fixed raw patch at each proposed coordinate and compute
   the image-only score without GT or fragment-derived predictor values.

Only after all predictions are frozen may the answer-key fields be read for scoring.
For site-level evaluation, a predicted site is a hit only when it is within the
chosen physical tolerance of a GT merge site for the **same normalized segment ID**.
Spatial proximity to a merge site on another segment is not a hit.

For segment-level evaluation:

- recall denominator: `gt_merge_labels`;
- adjudicable predicted segments: predicted segment IDs appearing nonzero in
  `gt_node_canonical_label`;
- report precision among adjudicable predictions;
- report predicted segments outside that set as unadjudicable, not automatic false
  positives.

Keep proposal recall separate from conditional image-scorer performance. A strong
scorer cannot recover a true site the proposal generator never supplied.

### Validation and failure checks

- Group splits by predicted `segment_id` and GT neuron so sites from the same merged
  structure cannot leak across train and validation.
- If multiple caches exist, reserve whole brains for transfer tests. With one cache,
  state that cross-brain transfer was not tested.
- Report site-level and segment-level results separately.
- Use average precision as the primary imbalanced metric; also report ROC-AUC,
  effect sizes, and grouped bootstrap confidence intervals.
- Report results for node-count-supported and geometric-walk merge cases separately
  when their provenance can be reconstructed; otherwise state that the cache exposes
  only their union at segment level.
- Report failed image reads, missing same-segment skeleton matches, matching-distance
  distributions, and control exclusions by class.
- Compare image-only, fragment-only, and combined models explicitly if geometry is
  explored.
- Never use `gt_merge_labels`, `gt_merge_sites`, `gt_edge_error`, or GT geometry as
  predictor inputs to a claimed GT-blind detector.
- Do not select a site tolerance after inspecting test-set performance.

---

## 3) Intent

The primary deliverable is a defensible explanation of which real-image patterns
distinguish adjudicated merge sites from clean locations, grounded in correct
correspondence with the affected predicted segment, fragment skeleton, and GT trace.
The analysis must establish all of the following:

1. The `_add.pkl` schema, array lengths, and merge-site invariants were validated.
2. Real image voxels were read from the exact cached `img_path`, with the mandatory
   audit of coordinates, shape, dtype, finite/nonzero counts, and intensities.
3. Every merge site was matched by segment ID before nearest-node distance was used.
4. Image, fragment, and GT figures used the identical physical center and voxel
   extent, with XYZ↔ZYX conversion documented.
5. Positive units were `gt_merge_sites`, not merged-edge counts; segment-level labels
   were analyzed separately.
6. Controls were selected without looking at the image feature under test, and
   GT-supported controls were distinguished from same-merged-segment sensitivity
   controls.
7. Every claimed image-only predictor was calculated only from real raw voxels.
8. Validation grouped correlated samples by segment and GT neuron and respected the
   sparse-GT precision limitation.

Begin with one end-to-end audit. Select one merge-site record and print its predicted
segment ID, attributed GT neuron, same-segment fragment component and distance,
physical/voxel coordinate, real-patch statistics, locally present segment IDs, and
local GT/fragment node and edge counts. Then display or save the three aligned
XY/XZ/YZ figures. Do not start batch feature extraction until this audit passes.

Prefer a focused experiment sequence:

1. Validate loading, merge semantics, segment matching, coordinate conversion, and
   real-image access.
2. Establish intensity, dark-gap, and local-ridge image baselines.
3. Test multiscale crossing, convergence, orientation, and bridge-continuity signals.
4. Test crowding and acquisition-artifact explanations.
5. Evaluate matched-control robustness and grouped generalization.
6. Build a GT-blind candidate-and-score evaluation, keeping proposal recall separate
   from image-scorer quality.
7. If useful, measure fragment geometry's incremental contribution in a transparently
   multimodal model.

The final report must start with the real-image and registration audit; define every
feature exactly; report site-level and segment-level outcomes, uncertainty, invalid
samples, and unadjudicable predictions; and show how the strongest image signal maps
to the target predicted segment and GT context. State whether fragment information
was used only for proposal/context or also as a predictor. Exclude any run that used
replacement voxels, mismatched axes or patch extents, global-nearest rather than
same-segment matching, or fragment/GT leakage into an image-only score.

Do not claim a deployable whole-brain merge detector merely because GT-centered
patches are separable. A deployable system additionally needs GT-blind proposal
generation, calibrated image scoring, segment association, and repair-safety logic.
The correct result at this stage is an image signature at adjudicated merge sites,
with explicit evidence of its alignment to GT and predicted fragments/segments.
