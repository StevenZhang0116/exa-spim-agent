# Understanding split errors from ExaSPIM image, GT, and fragments (V3)

This document is the complete task specification. The agent will receive exactly
three colocated files: this markdown, one trusted
`dataset_cache_<brain>_mcl<N>_add.pkl`, and `allen-nd-goog-f5d46dbfa2cd.json`. Do not assume that
any repository notebook, configuration file, metrics CSV, pre-extracted patch, or
dense segmentation volume is also available. Everything needed to locate errors,
align the data sources, authenticate image access, read the real image, and interpret
the result is described below.

This revision is written for **disciplined discovery**, not only for compliance. The
schema and safety rules constrain *how* you work; a separate exploratory subset may
suggest what to look for; and untouched confirmatory GT neurons determine which
associations survive. Do not satisfy novelty quotas by inventing arbitrary features.

> **V3 discovery scope — compare both candidate views without prescribing image
> features.** Image discovery must cover candidates from both `tip_to_tip` and
> `tip_to_any_node` enumeration. In both views the anchor is a degree-1 fragment tip;
> the expanded view allows the receiving node to be any node on another valid segment.
> Within that expanded view, audit tip-target and non-tip-target candidates separately.
> This requirement fixes the candidate search space and comparison, not the answer:
> no feature family, formula, mechanism, direction, threshold, or hypothesis count in
> this document is mandatory. Existing examples are non-binding baselines. Let the
> exploratory/confirmatory process discover what is useful, while preserving the
> image-only firewall and reporting null or deflationary findings honestly.

---

## 1) Dataset context

### What is in the cache

The cache connects three views of the same brain in one physical coordinate frame:

1. **Raw ExaSPIM image** — measured fluorescence voxels, opened on demand from
   `payload["img_path"]`. The large image is not embedded in the pickle.
2. **GT skeleton** — manually traced neurons in `payload["gt_graph"]`. This is a
   sparse answer key, not a dense GT mask.
3. **Predicted fragments/segments** — skeletons extracted from the automated
   segmentation in `payload["fragments_graph"]`, plus the predicted segment label
   sampled at every GT node in `payload["gt_node_canonical_label"]`.

These are registered, but they are not interchangeable. Image data are voxel
intensities; GT and fragments are sparse graphs; predicted segment IDs are labels.
Never treat a skeleton line as a dense neuron mask or a maximum-intensity projection
as proof of 3-D contact.

### Vocabulary that must remain distinct

- A **GT neuron** is one connected component of `gt_graph`. Its identity is a name
  such as `N007-794495-JG`, returned by `gt_graph.node_segment_id(node)`.
- A **fragment component** is one connected component/SWC in `fragments_graph`.
  Its internal integer component ID indexes
  `fragments_graph.component_id_to_swc_id`.
- A **predicted segment ID** is the prefix before `.` in a fragment SWC ID. Obtain
  it with `fragments_graph.node_segment_id(node)` or
  `str(fragments_graph.component_id_to_swc_id[c]).split(".")[0]`.
- `gt_node_canonical_label[i]` is the predicted segment ID observed at GT node `i`;
  `0` means the segmentation did not label that GT location.

GT-neuron names and predicted-segment IDs are different namespaces. A fragment
component ID is also not a segment ID. One predicted segment can be represented by
one or more fragment components, so always compare normalized segment IDs rather
than raw component IDs.

### What a split means

A split is a failure to preserve one real neurite as one predicted segment.

- A **direct split edge** is a GT edge whose two endpoint labels are both nonzero
  and different. In the stored classification it has `gt_edge_error == 1`.
- A **gap split** has a connected run of zero-labeled GT nodes between two or more
  different nonzero segment labels. Its individual edges can be classified as omit,
  even though the larger pattern is a split across an unlabeled gap.
- A **clean continuation** is a GT edge classified correct (`gt_edge_error == 0`),
  normally with the same nonzero predicted segment label at both endpoints.

The scientific question is: **what properties of the real fluorescence distinguish
split sites from clean continuations, and how do those properties correspond to the
GT cable and the predicted fragments/segments?**

### Allowed roles of each information source

GT and fragment information may be used to construct candidates, assign labels,
verify registration, identify the involved segment IDs, and interpret a result. If
the agent claims an **image-only split score**, however, the numerical predictors
given to that score must be computed only from raw image voxels and the candidate
coordinate. Fragment geometry, GT geometry, IDs, radii, node degrees, or error labels
may not enter that score.

Keep two explicit stages:

1. **Context/label stage:** use GT labels and fragments to locate, match, and audit
   split and control sites.
2. **Image-feature stage:** pass only `sample_id`, the coordinate, the class label
   needed for training/evaluation, grouping metadata, and the real raw-image patch.
   Do not pass fragment/GT features into the predictor.

It is valid to study a multimodal detector, but it must be named multimodal and its
image, GT, and fragment feature contributions must be reported separately. Do not
describe a geometry-assisted result as image-only.

### Scientific limits and hard rules

- GT is sparse. A negative is a clean continuation among traced neurons, not proof
  that all nearby untraced tissue is clean.
- Nearby sites and sites on the same GT neuron are correlated; they are not
  independent biological replicates.
- The raw image alone does not enumerate every pair of segment endpoints that should
  be joined. Candidate generation and segment association are separate from scoring.
- Never replace a failed image read with random, synthetic, simulated, or mock
  voxels. Report the exception and mark that sample unusable.
- Never download/materialize the whole ExaSPIM volume. Read bounded patches and
  cache compact derived features rather than thousands of large raw patches.
- Do not attempt to unpickle an untrusted file: Python pickle can execute code.

### Candidate views for image discovery

Use the GT-blind candidate contract shared with the fragments task. Enumerate from
degree-1 tip anchors and compare two declared receiving-node scopes:

- `tip_to_tip`: receiving nodes are degree-1 tips;
- `tip_to_any_node`: receiving nodes may be any node on another valid segment.

Within each anchor, rank distinct partner segment IDs by their nearest eligible node
and retain the declared `per_anchor_k`; globally deduplicate unordered segment pairs
while preserving qualifying occurrences and whether each receiving node is a tip.
Select radius and quota on training data only and freeze them before held-out use.
Candidate generation and labels are separate: freeze the GT-blind candidate universe
before attaching split truth.

The two views must share the same implementation and evaluation contract. Report
candidate count, reachable direct/gap coverage, prevalence, invalid image-read rate,
and repair-safety/merge risk for each view. Inside `tip_to_any_node`, also report
tip-target, non-tip-target, and mixed-support pairs. A hypothesis may apply to either
or both strata; record its applicability and undefined coverage rather than silently
dropping non-tip candidates. Because per-anchor ranking occurs within a receiving-node
scope, do not assume that filtering a quota-limited expanded pool always reconstructs
the separately ranked `tip_to_tip` baseline.

For an **image-only** score, fragment-derived candidate type, node degree, segment ID,
radius, tangent, and topology remain grouping/audit metadata, not predictor columns.
They may define where a bounded image patch is read, but numerical score inputs must
still come only from the real voxels under one declared, class-blind patch policy. A
model that consumes fragment geometry must be named and evaluated as multimodal.

---

## 2) Dataset schema

### Environment and loading

Unpickling requires the package that defines `SkeletonGraph`, plus a compatible
scientific Python stack. First try:

```bash
python -c "import agentic_neuron_proofreader"
```

If unavailable, install the package from
`https://github.com/AllenInstitute/agentic-neuron-proofreader` and ensure `numpy`,
`scipy`, `networkx`, `tensorstore`, and `matplotlib` are installed. A NumPy/SciPy
binary-compatibility error is an environment problem, not corrupt cache data. On
Allen Institute compute nodes, the `panda` conda environment is normally suitable.
Python 3.10 or newer is required. In a fresh compatible environment, the complete
installation command is:

```bash
python -m pip install "git+https://github.com/AllenInstitute/agentic-neuron-proofreader.git" numpy scipy networkx tensorstore matplotlib
```

Load with plain `pickle.load`; do not use a convenience loader that opens cloud
resources before the cache has been validated:

```python
import os
import pickle
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
```

Budget tens of GB of RAM for a cache. Load one cache at a time.

### Top-level fields

| Key | Meaning and correct use |
|---|---|
| `img_path` | Path to the real fused raw image. Usually public S3. Open exactly this path; do not invent a local replacement. |
| `segmentation_path` | Provenance path for the dense predicted segmentation. It may be private GCS and is not embedded in the pickle. Do not assume it is readable. |
| `anisotropy` | Microns per voxel in **(x, y, z)** order. Always read it from the cache. |
| `min_cable_length` | Minimum fragment cable length retained when the cache was built. It changes which short fragment skeletons exist. |
| `node_spacing` | Target physical spacing used to resample skeleton nodes. It is not necessarily the actual distance of every edge. |
| `gt_graph` | Human-traced GT `SkeletonGraph`. Its connected components are GT neurons. |
| `fragments_graph` | Automated fragment `SkeletonGraph` derived from predicted segments. |
| `gt_node_canonical_label` | `int64`, shape `(N_gt,)`: predicted segment ID at each GT node; `0` means unlabeled/background. |
| `gt_edge_error` | `uint8`, shape `(E_gt,)`, parallel to `list(gt_graph.edges)`: `0=correct`, `1=split`, `2=omit`, `3=merged`. Never index it using a differently ordered edge list. |
| `gt_merge_labels` | Global set/array of predicted segment IDs flagged as merges by the labeling pipeline. Membership does not by itself identify which GT neuron is charged. |
| `gt_merge_sites` | List of `{"segment_id": int, "gt_neuron": str, "xyz": (x,y,z) µm}` records. Coordinates are physical XYZ. |
| `fragments_path`, `gt_path` | Source-SWC provenance. The graphs are already in the pickle; these paths are not needed. |

The four added label objects may also exist as `gt.node_label`, `gt.edge_error`,
`gt.merge_labels`, and `gt.merge_sites`. Prefer the top-level keys above so the
source is explicit.

### `SkeletonGraph` fields and methods

Both graphs subclass `networkx.Graph` and use contiguous integer node IDs that index
parallel arrays:

- `graph.node_xyz[node]`: physical coordinate in **(x, y, z) microns**;
- `graph.node_radius[node]`: skeleton radius metadata;
- `graph.node_component_id[node]`: integer connected-component ID;
- `graph.component_id_to_swc_id[component]`: SWC identity;
- `graph.node_segment_id(node)`: prefix of the SWC identity;
- `graph.node_voxel(node)`: integer voxel in **(z, y, x)** order;
- `graph.kdtree.query(xyz_um)`: nearest graph node in physical micron space;
- `graph.nodes_in_patch(origin_zyx, shape_zyx, ...)`: local floating-point ZYX
  node coordinates inside a patch;
- `graph.edges_in_patch(origin_zyx, shape_zyx, ...)`: local ZYX edge endpoints,
  for edges whose two nodes both lie inside the patch.

On `gt_graph`, `node_segment_id(node)` returns the GT-neuron name. On
`fragments_graph`, it returns the predicted segment ID. This difference is crucial.

### Coordinate conversion and patch convention

There are exactly two coordinate conventions:

```text
physical skeleton coordinate: (x, y, z) microns
image / patch voxel coordinate: (z, y, x) voxels
anisotropy: (x, y, z) microns per voxel
```

Use truncation to match `SkeletonGraph.node_voxel` exactly:

```python
def xyz_um_to_zyx_voxel(xyz_um):
    xyz_um = np.asarray(xyz_um, dtype=float)
    return tuple((xyz_um / anisotropy_xyz).astype(int)[::-1])

def patch_origin(center_zyx, shape_zyx):
    return tuple(int(c - s // 2) for c, s in zip(center_zyx, shape_zyx))


def normalize_segment_id(value):
    """Canonical string form; returns None for background/unlabeled."""
    text = str(value).split(".")[0]
    return None if text in {"0", "0.0", "None", ""} else str(int(text))
```

`TensorStoreImage.read(center_zyx, shape_zyx)` takes a **center**. In contrast,
`nodes_in_patch` and `edges_in_patch` take the patch's minimum-corner **origin**.
Passing the center to the graph methods shifts the skeleton by half a patch and is
the most important alignment error to avoid.

Select a real direct split before attempting any image read. This defines `i`, `j`,
`center_xyz`, and `center_zyx` for every later example:

```python
EDGE_CORRECT, EDGE_SPLIT, EDGE_OMIT, EDGE_MERGED = 0, 1, 2, 3

direct_splits = []
clean_edges = []
for edge_index, (edge_i, edge_j) in enumerate(gt_edges):
    a = normalize_segment_id(node_label[edge_i])
    b = normalize_segment_id(node_label[edge_j])
    cls = int(edge_error[edge_index])
    record = {
        "edge_index": edge_index,
        "nodes": (int(edge_i), int(edge_j)),
        "gt_neuron": gt.node_segment_id(edge_i),
        "labels": (a, b),
        "xyz_um": 0.5 * (
            np.asarray(gt.node_xyz[edge_i], dtype=float)
            + np.asarray(gt.node_xyz[edge_j], dtype=float)
        ),
    }
    if cls == EDGE_SPLIT and a is not None and b is not None and a != b:
        direct_splits.append(record)
    elif cls == EDGE_CORRECT and a is not None and a == b:
        clean_edges.append(record)

if not direct_splits:
    raise RuntimeError(
        "this cache contains no direct split edge; run only the separately "
        "defined gap-split analysis or use another cache"
    )

selected_split = direct_splits[0]
i, j = selected_split["nodes"]
xyz_i = np.asarray(gt.node_xyz[i], dtype=float)
xyz_j = np.asarray(gt.node_xyz[j], dtype=float)
center_xyz = 0.5 * (xyz_i + xyz_j)
center_zyx = xyz_um_to_zyx_voxel(center_xyz)
print("selected direct split:", selected_split)
print("center xyz_um / zyx_voxel:", center_xyz, center_zyx)
```

Do not average already-truncated endpoint voxels; that can move the center by a
voxel. Record both `center_xyz` and `center_zyx` for reproducibility.

### Reading and auditing the real image

The image is external to the pickle. Configure authentication **before** constructing
`TensorStoreImage`, based on the scheme of `payload["img_path"]`:

- `s3://`: the usual raw image is public. Disable EC2 metadata probing and use
  anonymous access; no AWS key is needed.
- `gs://`: treat the image as private and use the supplied
  `allen-nd-goog-f5d46dbfa2cd.json` file.

The delivery contains this markdown, the `_add.pkl`, and `allen-nd-goog-f5d46dbfa2cd.json` in
the same directory. Start the analysis with that delivery directory as the current
working directory. Resolve the credential from the filename—not from a repository
path, username, or machine-specific absolute path. For GCS:

1. resolve `Path("allen-nd-goog-f5d46dbfa2cd.json")` in the delivery directory;
2. verify that it is a readable file;
3. assign its absolute resolved path to `GOOGLE_APPLICATION_CREDENTIALS`;
4. only then construct `TensorStoreImage`.

Use this complete helper:

```python
from agentic_neuron_proofreader.utils import img_util


BUNDLED_GCP_CREDENTIALS = Path("allen-nd-goog-f5d46dbfa2cd.json")


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
            "private GCS image requires the supplied allen-nd-goog-f5d46dbfa2cd.json in "
            f"the delivery directory; looked at {credentials}"
        )

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(credentials)
    print(f"Image access: authenticated GCS (supplied credential: {credentials})")
    return "authenticated-gcs"


img_path = str(payload["img_path"])
image_access = configure_image_access(img_path)

image = img_util.TensorStoreImage(img_path)

# image.shape() returns the full volume as a 5-tuple (1, 1, Z, Y, X). The
# trailing three entries are the ZYX voxel bounds and are the only way to know
# whether a requested patch fits inside the volume.
IMAGE_SHAPE_ZYX = tuple(int(s) for s in image.shape()[-3:])
print("image volume shape (z, y, x):", IMAGE_SHAPE_ZYX)

# Fixed audit/default field, not a claim that every later experiment must use one scale.
PATCH_SHAPE = (128, 128, 128)  # (z, y, x), identical for positives and controls


def patch_margin_zyx(center_zyx, shape_zyx, image_shape_zyx=IMAGE_SHAPE_ZYX):
    """Per-axis voxel gap between the requested patch and the volume border.

    A negative entry means the patch would leave the volume on that axis. The
    minimum entry is the distance to the image boundary used for control
    matching.
    """
    origin = patch_origin(center_zyx, shape_zyx)
    return tuple(
        int(min(o, extent - (o + s)))
        for o, s, extent in zip(origin, shape_zyx, image_shape_zyx)
    )


def read_patch(center_zyx, shape_zyx):
    """Read one bounded patch. Returns (patch, invalid_reason).

    An out-of-bounds request raises inside TensorStore; it never returns a
    truncated array. The bounds pre-check and the except clause are therefore
    both required, and comparing the returned shape with the requested shape
    can never catch a boundary failure on its own.
    """
    margin = patch_margin_zyx(center_zyx, shape_zyx)
    if min(margin) < 0:
        return None, f"out-of-bounds (margin_zyx={margin})"
    try:
        patch = np.asarray(image.read(center_zyx, shape_zyx))
    except Exception as exc:  # OUT_OF_RANGE, transport, or credential failure
        return None, f"read-failed ({type(exc).__name__}: {exc})"
    if tuple(patch.shape) != tuple(shape_zyx):
        return None, f"wrong-shape ({tuple(patch.shape)} != {tuple(shape_zyx)})"
    return patch, None


def patch_content_reason(patch):
    """Reject a degenerate patch. Returns None when the content is usable.

    The raw ExaSPIM image is unsigned integer, so `np.isfinite` is trivially
    True on every voxel and detects nothing. Test the content itself instead.
    """
    values = patch.astype(np.float64, copy=False)
    if np.issubdtype(patch.dtype, np.floating) and not np.isfinite(values).all():
        return "non-finite voxels"
    if not np.count_nonzero(values):
        return "all-zero patch (unwritten chunk or outside the imaged region)"
    if values.min() == values.max():
        return "constant patch"
    return None


img_patch, invalid_reason = read_patch(center_zyx, PATCH_SHAPE)
if invalid_reason is None:
    invalid_reason = patch_content_reason(img_patch)
if invalid_reason is not None:
    raise RuntimeError(f"invalid real patch at {center_zyx}: {invalid_reason}")

print("img_path:", img_path)
print("image_access:", image_access)
print("center xyz_um / zyx_voxel:", center_xyz, center_zyx)
print("requested / returned shape:", PATCH_SHAPE, img_patch.shape)
print("dtype / voxels:", img_patch.dtype, img_patch.size)
print("boundary margin (z, y, x) voxels:", patch_margin_zyx(center_zyx, PATCH_SHAPE))
print("nonzero voxels:", int(np.count_nonzero(img_patch)))
print("min, median, p99, p99.9, max:",
      np.percentile(img_patch, [0, 50, 99, 99.9, 100]))
```

Equivalently, from the directory containing all three delivered files, the agent may
configure the supplied credential before Python:

```bash
export GOOGLE_APPLICATION_CREDENTIALS="$PWD/allen-nd-goog-f5d46dbfa2cd.json"
python your_analysis.py
```

The credential must grant `storage.objects.get` access to the bucket/object. Verify
that the path exists without printing or copying the JSON contents; the contents must
never be printed, embedded in generated code or reports, or committed. If the supplied
JSON is absent or unreadable, stop and report the failure rather than searching
unrelated machine paths. `segmentation_path` may also point to private GCS, but this
task does not require opening the dense segmentation; authentication here is only for
the actual `img_path` being read.

This audit is mandatory evidence that real voxels were used. Two failure modes must be
handled explicitly, because neither one shows up as a shape mismatch:

- **An out-of-bounds read raises; it does not return a smaller patch.** TensorStore
  reports `OUT_OF_RANGE` as an `IndexError` or `ValueError` as soon as any requested
  slice leaves the volume, so a bare `image.read(...)` aborts the run instead of
  yielding a truncated array. Compare the patch against `IMAGE_SHAPE_ZYX` before
  reading and still catch the exception, as `read_patch` does above.
- **Integer voxels are always finite.** The raw image is `uint16`, so an
  `np.isfinite(...)` guard passes on every patch, including an all-zero one from an
  unwritten chunk. Judge validity from the content — nonzero count and intensity
  spread — as `patch_content_reason` does above.

In batch extraction, record the returned `invalid_reason` and drop the sample. Never
pad, clip, or re-center a patch to rescue it, apply the identical rule to positives and
controls, and report invalid counts and reasons per class: an unequal drop rate is
itself an artificial signal.

### Local 3-D receptive field and scale policy

Here **local** means a bounded 3-D voxel subvolume centered on one candidate site.
`TensorStoreImage` makes the external image volume addressable, but the
analysis must read only such bounded subvolumes. The complete voxel array inside each
patch is available for 3-D measurement and slice inspection.

`PATCH_SHAPE = (128, 128, 128)` is the fixed field for the first alignment audit and
a reproducible single-scale baseline. It is **not** a requirement that all later
models have exactly one receptive-field size. Prefer one of these declared policies:

1. **Fixed single scale:** use the same predeclared physical extent for every split
   and control.
2. **Fixed multiscale (recommended):** predeclare a small nested set of physical
   extents in microns, convert each extent to ZYX voxels with `anisotropy_xyz`, and
   read every scale for every sample. For example, choose three scientifically useful
   small/medium/large fields rather than tuning fields separately for positives.
3. **Image-adaptive:** begin from a common minimum field and expand or orient it using
   only raw voxels until a predeclared image criterion is met, subject to a common
   maximum extent and failure rule.

Whichever policy is used, freeze the scale set, expansion/orientation rule, maximum
extent, resampling, and aggregation using training data only, then apply them unchanged
to validation/test data. Record for every sample the center, requested shape(s),
physical extent(s), returned shape(s), boundary margin, chosen adaptive scale if any,
and invalid reason. Apply identical policies to positives and controls, and report the
selected-scale and failure distributions by class. A larger field also fails the
bounds check sooner, so re-check the margin for every scale rather than only for the
smallest one.

The firewall applies to the field itself, not only to the predictors: a field size or
orientation chosen from GT radius/tangent, fragment geometry, segment ID, or error
label is geometry-conditioned or multimodal and must be reported as such, while a rule
driven only by the candidate coordinate and raw voxels stays image-only. Never resize
a difficult positive manually after viewing it. Always compare an adaptive approach
with the fixed single- or multiscale baseline so apparent gain cannot come merely from
class-dependent field selection.

### Matching GT labels to fragment components and segment IDs

Index the fragment components using the normalization function defined before split
selection:

```python
segment_to_components = {}
for component, swc_id in fragments.component_id_to_swc_id.items():
    segment = normalize_segment_id(swc_id)
    segment_to_components.setdefault(segment, set()).add(int(component))

def gt_node_match(node):
    segment = normalize_segment_id(node_label[node])
    return {
        "gt_node": int(node),
        "gt_neuron": gt.node_segment_id(node),
        "xyz_um": tuple(np.asarray(gt.node_xyz[node], dtype=float)),
        "predicted_segment": segment,
        "fragment_components": sorted(segment_to_components.get(segment, ())),
    }
```

The canonical label is the primary GT↔segment match because it came from the dense
segmentation at that GT voxel. Fragment skeletons are sparse, so the absence of a nearby
fragment node does not overwrite a nonzero canonical label. Conversely, the nearest
fragment is useful as a registration check, but it is not proof that its segment owns
the voxel.

For an optional physical-distance audit:

```python
audit_gt_node = int(selected_split["nodes"][0])
distance_um, nearest_fragment_node = fragments.kdtree.query(
    gt.node_xyz[audit_gt_node]
)
nearest_fragment_node = int(nearest_fragment_node)
nearest_segment = normalize_segment_id(
    fragments.node_segment_id(nearest_fragment_node)
)
print("canonical / nearest segment / distance_um:",
      normalize_segment_id(node_label[audit_gt_node]),
      nearest_segment,
      float(distance_um))
```

Large distances or systematic XYZ/ZYX disagreement indicate a registration bug or
sparse/missing fragment coverage. Investigate them; do not repair them by swapping
axes until the documented coordinate conversion has been checked.

### Extracting one aligned image–fragments–GT patch

The following uses one center and one patch extent for all three views:

```python
shape_zyx = PATCH_SHAPE
origin_zyx = patch_origin(center_zyx, shape_zyx)

img_patch, invalid_reason = read_patch(center_zyx, shape_zyx)
assert invalid_reason is None, invalid_reason

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

local_fragment_segments = {
    normalize_segment_id(fragments.node_segment_id(int(n))) for n in frag_ids
}
local_gt_neurons = {gt.node_segment_id(int(n)) for n in gt_ids}
print("local GT nodes/edges/neurons:",
      len(gt_nodes), len(gt_edges_local), len(local_gt_neurons))
print("local fragment nodes/edges/segments:",
      len(frag_nodes), len(frag_edges_local), len(local_fragment_segments))
```

Use the package's plotting helpers to preserve the same XY/XZ/YZ orientation:

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

The three figures must use the same center and shape. The fragment and GT figures use
independent color palettes, so equal colors across figures do not imply a match. Match
by segment ID and coordinates, not display color. MIPs also collapse depth, so confirm
ambiguous contacts in orthogonal views or 3-D slices.

### Constructing split and control samples

The direct split and clean-control edge tables were constructed before the first
image read, so the visual audit and batch analysis share exactly the same definitions.

For gap splits, within each GT neuron find connected components of nodes whose
canonical label is zero. For each zero-labeled component, inspect GT neighbors just
outside it. If the boundary contains at least two distinct nonzero predicted segment
IDs, record one gap candidate at the physical centroid of the zero run. Deduplicate
nearby candidates in physical microns. Do not count every omit edge in the run as an
independent split.

Match clean controls to positives without inspecting the image feature being tested.
At minimum match by GT neuron, physical z region, distance to the image boundary
(`min(patch_margin_zyx(center_zyx, PATCH_SHAPE))`), proximity to a GT branch point,
and physical field of view. Use the same declared receptive-field policy,
normalization, preprocessing, and read-validity rule for both classes.

The context table should contain only what downstream stages need:

```text
sample_id, brain_id, xyz_um, center_zyx, is_split, split_kind,
gt_neuron, left_segment_id, right_segment_id, invalid_reason
```

Segment IDs and GT metadata are for auditing, grouping, and interpretation. Exclude
them from an image-only model's predictor matrix.

GT-centered split/control sites support mechanism characterization, but they do not
replace the GT-blind candidate experiment. For deployable pair scoring, construct the
two candidate views above first, then attach labels and read one consistently defined
real-image field covering each qualifying tip-to-receiving-node occurrence. Report
site-level discovery and candidate-pair performance as different units. In particular,
do not let a strong result on clean tip-to-tip patches stand in for evidence on the
non-tip receiving stratum.

### Image information to measure

All image features must be derived from the actual `img_patch`. The families below
are retained from v2 only as **non-binding historical examples**. They do not define
the search space, a minimum checklist, an expected direction, or a preferred answer;
the agent may ignore, reject, replace, or extend all of them:

- robust center intensity relative to a surrounding shell;
- dark-voxel fraction, local SNR, coefficient of variation, and radial decay;
- multiscale minimum-intensity bottleneck;
- Hessian/Frangi vesselness and 3-D structure-tensor coherence;
- ridge continuity through the center and weakest point along an image-derived axis;
- evidence for multiple orientations, crossings, or competing bright ridges;
- entropy, frequency content, anisotropic blur, saturation, stripe, or tile-boundary
  signatures;
- stability of these signals across physical scales.

Choose a compact, evidence-justified discovery set rather than satisfying a mandated
feature count or family quota. Novelty means a new testable claim, not merely a new
formula. For each retained hypothesis, state why it should differ at a split **before**
measuring it on the confirmatory subset. Do not search a large feature library and
report only the best result. Fewer defensible measurements are preferable to padding
the list with arbitrary ones.

For a multiscale policy, compute each scale by the same declared procedure and retain
the per-scale values before any cross-scale reduction. For an adaptive policy, treat
the chosen scale as audited metadata; do not let GT/fragment context choose it for an
image-only model.

Estimate orientation from image voxels for an image-only score. A GT tangent or
fragment edge direction can be used for an explicitly geometry-conditioned analysis,
but that analysis is not image-only. Prefer physical microns and robust within-patch
normalization because voxel spacing and intensity vary across axes and brains.

### Validation and failure checks

- Group train/validation folds by `gt_neuron`, never at random over individual
  patches.
- If multiple caches are available, reserve entire brains for transfer tests. With
  one cache, state that cross-brain transfer was not tested.
- Use average precision as the primary metric under imbalance; report ROC-AUC,
  effect sizes, and grouped bootstrap confidence intervals as secondary results.
  Average precision measures prediction, not understanding: also report every effect in
  interpretable physical units, and never treat an AP improvement alone as confirmation
  of a mechanism.
- Fit normalization, thresholds, dimensionality reduction, and feature selection
  inside each training fold.
- Freeze exploratory feature definitions before using confirmatory neurons. Account
  for the number of retained hypotheses with a declared family-wise procedure or FDR,
  and report all retained tests rather than only significant ones.
- Report every available direct/gap category separately before pooling; state when a
  category is absent or too small for inference.
- Report invalid reads and missing-fragment matches by class. Unequal failure rates
  can create an artificial signal.
- Compare an image-only baseline, fragment/geometry-only baseline, and combined
  model separately if multimodal features are explored.
- Do not use the same canonical labels both as predictors and targets.

### Open, mechanism-informed hypotheses

A feature list can produce shallow hypotheses; a mechanism-informed question can
produce testable ones. This observational cache can establish associations consistent
with a mechanism, not causation by itself.
"`p99` is lower at split sites" is an observation. "The segmentation network's effective
receptive field is about R microns, so an unlabeled gap longer than R breaks
continuity" is a hypothesis with a predicted threshold in physical units. The classes
below are retained as non-binding prompts from v2: they are neither exhaustive nor a
coverage requirement. Discovery may introduce a different mechanism or a useful
feature without assigning it to one of these rows. Label each retained hypothesis
with its evidential status and, when useful, a freely chosen mechanism class.

| Class | What could cause a split |
|---|---|
| Optical | anisotropic PSF — the z axis is sampled and resolved differently from xy; depth-dependent attenuation; scattering; refractive-index mismatch |
| Acquisition | tile seams and fusion weights, stripe artifacts, saturation, drift between tiles, uneven illumination |
| Biological | thin distal caliber, abrupt caliber change, crossing or fasciculating neurites, dense neuropil near somata |
| Algorithmic | the segmentation network's receptive field and scale, its decision threshold, agglomeration behavior, `min_cable_length` filtering of short fragments |
| Labeling | the canonical-labeling rule behind `gt_node_canonical_label`, `node_spacing` resampling of the GT trace, sparse GT coverage |

The last two example classes are legitimate answers. If the pattern is produced by
the label rule or by the reconstruction pipeline rather than by the fluorescence,
that is a result, not a failed experiment.

For each hypothesis considered seriously, record evidence availability with columns
such as `hypothesis`, optional `mechanism_class`, `required_evidence`,
`available_field_or_metadata`, and `test_status`. Distinguish `directly_testable`,
`proxy_only`, and `not_testable`.
Tile/fusion-weight claims require acquisition metadata; receptive-field claims require
model documentation; causal optical claims require suitable acquisition controls. Do
not turn physical coordinates into tile metadata or an image association into causal
evidence. An unavailable class is reported, not filled with speculation.

### Generating and pruning hypotheses

Diversity has to be produced **before** the converging steps. Control matching, grouped
validation, and significance testing can only shrink a hypothesis set; none of them can
widen it. Treat the following as a required stage with its own reported output, not as
advice.

**Separate exploration from confirmation before looking.** Group by `gt_neuron` and
assign neurons once to an exploratory subset and an untouched confirmatory subset;
record the seed, proportions, and IDs or hashes. All sites from one neuron stay in one
subset. If the sample is too small for a meaningful confirmatory subset, label the
entire study exploratory and do not make confirmatory claims. Cross-validation inside
the same observations does not undo feature-discovery leakage. Construct candidate and
control tables before this split, then sample/match controls separately within each
subset; never reuse one control across exploration and confirmation.

**Look before you hypothesize, but only in exploration.** After the end-to-end audit
passes and before batch feature extraction, render MIP triplets for up to 20 random
direct splits, up to 20 random gap splits, and up to 20 matched controls from the
exploratory neurons, inspected with class labels hidden wherever tooling allows. Use
all available unique sites when a category has fewer than 20, report the count, and
skip an absent category rather than duplicating or fabricating samples. Mix selected
examples in a seeded random order under blind sample IDs and reveal their category only
after the field notes are frozen.
Write free-text field notes on what actually differs. Formalize hypotheses from those
notes rather than from the feature families above, and list up to **three observations
you could not yet quantify**. These are valuable because each names a measurement that
may need to be invented; do not fabricate observations to reach a count.

**Rotate the point of view when useful.** The framings below are optional prompts for
escaping a narrow feature catalogue, not required hypothesis classes or a checklist.
Use, replace, or extend them according to the exploratory evidence, then merge and
deduplicate overlapping ideas:

1. *"If I were the segmentation network, when would I give up on this connection?"*
2. *"If I were the microscope, where would I lose this signal?"*
3. *"If I were the labeling pipeline, where would I record a break that is not one?"*

**Generate freely, then prune.** Do not target a fixed hypothesis count or mechanism
quota. Retain the nonredundant, testable hypotheses justified by the exploratory
evidence and available data, and keep materially considered but discarded candidates
in the report with a one-line reason each. Never pad the ledger. Avoid redundant
hypotheses that make the same prediction and fail under the same control; merge them
or state what observation would distinguish them. The retained set may differ between
`tip_to_tip`, tip-target occurrences within `tip_to_any_node`, and non-tip-target
occurrences; state that applicability explicitly.

**Declare a prior and a kill criterion.** For each retained hypothesis, write down
before measuring: the predicted direction and rough magnitude, so that the result can
be surprising; the most ordinary alternative explanation — absolute brightness,
physical depth, local crowding, patch validity, or a label-rule artifact — and how it
will be ruled out; and the kill criterion, the concrete measurement outcome that would
end the hypothesis. Report the prior beside the measurement, and report the outcome
for every retained hypothesis including the ones that were killed. A confirmed strong
prior and a refuted strong prior are both results; an unstated prior makes neither
possible.

Use exploratory neurons to define code, transformations, scales, directions, and any
thresholds. Freeze them before confirmation. Apply the frozen measurements to the
confirmatory neurons once; do not revise a failed hypothesis and retest it on the same
confirmatory observations. Any later revision starts a new exploratory cycle and must
be labeled as such.

### Units of analysis and representations

The candidate site is one unit among several, and each unit asks a different question.
Report the site-level result and, when sample support permits, at least one second unit;
state which unit and denominator each claim belongs to:

- **per candidate site** — what is locally different at the break;
- **per GT neuron** — why some traced neurons are fragmented into many predicted
  segments while neighbors of similar length are not;
- **per predicted segment** — what distinguishes segments that terminate prematurely;
- **per z slab, per tile, or per region** — whether breaks concentrate where the
  acquisition, rather than the biology, changes.

When the necessary data and sample size exist, use **at least one representation that
is not a site-centered patch**. For example: raw
intensity as a function of arclength along a GT neuron, a 1-D profile in which a
bottleneck has a *length* and not only a depth; a per-tile or per-slab aggregate over
many sites; or the joint distribution of two image quantities rather than either alone.
A conclusion that exists at only one unit and one representation is weaker than one
that survives a change of both.

### Discordant-case mining

Once a candidate score exists, its most informative samples are the ones it gets wrong.
Rank all samples by score and inspect both tails against the answer key:

- locations that score like a split where `gt_edge_error == 0`;
- adjudicated splits that score like a clean continuation.

Each discordant group may indicate a **possible** unannotated error in the sparse
answer key, two failure modes that the feature conflates, a confounder, or a refutation
of the feature. Without independent proofreading or dense-segmentation evidence, do
not relabel it or declare the answer key wrong; mark it `unresolved_requires_review`.
Classify only to the level supported by evidence, show visual examples, and report
counts by group. Do not
drop discordant cases silently or tune the feature until they disappear — a feature
adjusted until its own counterexamples vanish has been fitted to them.

---

## 3) Intent

The primary deliverable is a defensible explanation of the raw-image evidence at GT
split sites, grounded in correct correspondence with the automated fragments and
predicted segment IDs. The analysis must establish all of the following:

1. The `_add.pkl` schema and label-array alignment were validated.
2. Real image voxels were read from the exact cached `img_path`, with the mandatory
   patch audit reported.
3. Image, fragment, and GT views used identical physical centers and voxel extents.
4. GT↔segment matches used `gt_node_canonical_label`; segment↔fragment matches used
   normalized fragment SWC/segment IDs; nearest-neighbor distance was only a spatial
   audit.
5. Direct split positives and clean controls were constructed without looking at the
   candidate image feature.
6. Every claimed image-only predictor was calculated only from real voxels.
7. The receptive-field policy was declared, bounded, identical across classes, frozen
   before validation/test use, and audited in physical units; any geometry-conditioned
   adaptation was labeled multimodal.
8. Evaluation grouped correlated samples by GT neuron and clearly stated what could
   not be tested from a single cache.
9. Exploration and confirmation were separated by GT neuron before visual inspection;
   the hypothesis ledger records candidates, discarded candidates and why, priors,
   kill criteria, and all confirmatory outcomes—or explicitly states that sample size
   permitted exploratory analysis only.
10. The site-level result and every supportable secondary unit were reported; both
    discordant score tails were inspected without unsupported relabeling.
11. `tip_to_tip` and `tip_to_any_node` used the shared GT-blind candidate contract and
    were reported separately, including tip-target and non-tip-target support within
    the expanded view and the candidate-count/coverage/merge-risk trade-off.

Beyond being correct, the result must be **diverse in kind**. A result is admissible in
any of the shapes below, and a deflationary shape is worth as much as a positive one:

- **(a) Signature** — an image quantity separates the classes and survives its declared
  controls.
- **(b) Taxonomy** — split sites fall into k mechanistically distinct kinds. Direct
  versus gap splits is the obvious first partition, but not necessarily the right one.
- **(c) Dose–response or threshold** — the effect turns on at a stated physical scale, a
  gap length or a caliber or a contrast ratio, reported in microns.
- **(d) Interaction** — the effect exists only within a stated subpopulation, or
  reverses between subpopulations.
- **(e) Attribution** — the apparent signature is explained by a more ordinary variable:
  absolute brightness, physical depth, local crowding, or an unequal invalid-read rate
  between classes.
- **(f) Answer-key critique** — the pattern is produced by the labeling rule or by
  `node_spacing` resampling rather than by the fluorescence.
- **(g) Null** — no image-only quantity separates the classes at the declared effect
  size, stated together with the power the sample actually supports.

Shapes (e), (f), and (g) are results to be reported, not runs to be repeated until they
turn into (a).

Begin with a small end-to-end audit: select one direct split, print its GT neuron,
endpoint predicted segment IDs, corresponding fragment component IDs, physical and
voxel center, real-patch statistics, and local graph counts; then display or save the
three aligned XY/XZ/YZ figures. Only after this audit passes should batch feature
extraction begin.

Prefer a focused experiment sequence:

1. Validate loading, coordinate conversion, image access, and three-way alignment.
2. Build and freeze both GT-blind candidate views, then split by GT neuron before
   looking; report their candidate-count, reachable-coverage, prevalence, and
   repair-safety trade-offs, including the expanded view's receiving-node strata.
3. On exploratory neurons, inspect bounded real-image evidence and freely generate
   testable measurements and hypotheses; use the examples above only if supported,
   then prune without a fixed count and freeze the retained definitions, priors, and
   kill criteria.
4. Evaluate every applicable retained hypothesis in each candidate view and receiving
   stratum; record inapplicable or undefined cases instead of silently removing them.
5. Treat ordinary variables such as brightness, depth, patch validity, and local
   crowding as potential confounders whenever relevant to the discovered signal.
6. Report every available direct/gap category separately and, when supported, repeat
   the leading result at a second unit of analysis.
7. Mine discordant cases in both score tails without relabeling unresolved cases.
8. Evaluate robustness to matched-control resampling and grouped validation.
9. If useful, quantify fragment geometry separately and measure its incremental value
   in a transparently multimodal model.

The final report must start with the real-image and registration audit, then define
every tested feature exactly, report grouped results and uncertainty, and show how
the strongest image signal corresponds to the GT cable and involved predicted
segments. It must compare `tip_to_tip` with `tip_to_any_node`, including candidate
coverage/cost and tip-target versus non-tip-target support in the expanded view. State
whether fragment information was used only for context or also as a predictor. It
must also carry the hypothesis ledger: the candidates generated, the ones discarded
and why, and for each tested hypothesis its candidate-view applicability, prior, kill
criterion, and whether that criterion was met. Exclude any run that substituted
non-real voxels, silently changed axes, used mismatched patch extents, or treated a
fragment skeleton as dense segmentation.

Do not claim a deployable whole-brain split detector solely because GT-centered
patches are separable. A deployable system also needs GT-blind candidate generation,
segment-pair association, and repair-safety logic. The correct conclusion at this stage
is whichever of the admissible shapes (a)–(g) the evidence actually supports, stated
with explicit evidence of how it aligns to GT and predicted fragments/segments — not an
image signature assumed in advance.
