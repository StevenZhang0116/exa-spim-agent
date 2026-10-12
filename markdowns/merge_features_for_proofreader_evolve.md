# Discovering candidate-level features for ExaSPIM merge candidates

This document is the complete task specification. You receive exactly these files,
colocated:

| File | What it is | Schema |
|---|---|---|
| this markdown | the task | |
| `dataset_cache_794495_mcl100_add.pkl` | one labeled ExaSPIM brain: automated fragment reconstruction, human ground truth (GT) and GT-derived error labels | §3.1 |
| `rerank_companion_794495_merge.npz` | the fixed merge candidates of this brain, their labels and fold assignment | §3.2 |
| `rerank_companion_794495_merge.json` | manifest of the companion: rule text, counts, digests, folds, image geometry | §3.3 |

The raw fluorescence image is on public S3 (anonymous read; no credential needed).
Its URI and level geometry are in the companion `.json`.

This task is about **merge errors only**. The document has three parts: §1 what the
data is, §2 what to do with it and how it is judged, §3 the schema of each file.

---

## 1) Dataset context

### Imaging, reconstruction and merge errors

ExaSPIM images a whole mouse brain at about 0.75 x 0.75 x 1 um. An automated
segmentation, skeletonized into `fragments_graph`, reconstructs neurons as many
**segments** (one segment = one predicted neuron piece). Human annotators traced a
few neurons completely (`gt_graph`). Comparing the two gives the error labels baked
into the pkl.

A **merge** error is a segment that fuses two neurons, or fuses a traced neuron
with other material. The fix is to cut the segment at the merge site.

### Candidates are fixed

The companion lists every merge candidate, enumerated by a fixed geometric rule
from `fragments_graph` alone (no GT, no trained model). The rule text is in the
companion `.json` (`candidate_rule`).

A candidate is a *junction*: a fragment node of degree >= 3 on a valid segment.
Greedy non-maximum suppression, in order (-degree, node id), removes same-segment
junctions within 20 um (Euclidean).

Use exactly these candidates, in companion row order. Do not add, drop or re-enumerate
candidates; features that only exist on a different candidate set cannot be used.

### Labels are three-valued

GT covers only a few traced neurons, so for most candidates GT cannot say whether
they are errors. The companion `label` column is:

| Value | Meaning |
|---|---|
| `1` | error: within 20 um of fragment cable (geodesic) from a GT merge site, after snapping the site to the nearest node of its own segment within 50 um |
| `0` | GT can judge the candidate and it is not an error: the segment carries a GT label and the junction is within 150 um of a GT node |
| `NaN` | GT cannot judge it: no traced neuron nearby. **Unknown, not "no error."** |

For this brain: 58,120 candidates; label 1: 114; label 0: 6,098; NaN: 51,908.

All evaluation uses labeled rows only (label 1 or 0). Never treat NaN as 0: a NaN
row next to no traced neuron says nothing about errors. Positives are rare (about
1.8% of labeled rows), so the top of the ranking matters most.

Merge labels are conservative: some real merges are not flagged. A labeled-0
candidate is GT-judgeable but may occasionally be a missed merge.

### Raw image

The fused fluorescence volume of this brain is an OME-Zarr on public S3, readable
anonymously. Graph coordinates are world microns and map to the image by the
per-level scale and translation in the companion `.json` (§3.3). The dense
segmentation is not available and is not needed.

---

## 2) Intent

### Goal

Discover measurable properties of a junction that put true merges ahead of
non-merges, among the candidates whose label GT can decide. The deliverable is a set
of features, each written as a deterministic function `describe(context)` of a
candidate's local fragment geometry and raw image patch, with evidence that it
ranks well.

Discover from the data. This document deliberately contains no list of known or
expected features. Do not import feature ideas from outside the data you were
given, and do not try to recover any earlier detector.

Your features will later be confirmed on data you cannot see. A feature that only
works on this brain's folds will not survive that confirmation.

### Two phases

- **Phase 1, characterization.** You may use everything in the pkl, including GT
  and labels, to understand what distinguishes merges from non-merges: inspect
  examples, compare distributions, view image patches.
- **Phase 2, features.** Turn what you learned into features computed **only** from
  the candidate context (below). Phase 2 features must never read GT, labels,
  absolute coordinates, or segment/node IDs.

### Candidate context

A feature sees one candidate through a `context` dictionary. Build it exactly as
below, so the same feature code can later be run on other brains.

**Anchor.** `[node_id]` of the junction.

**Fragment neighbourhood** (radius 50 um, at most 256 nodes):

```python
def neighbourhood(frag, anchors, radius_um=50.0, max_nodes=256):
    center = np.asarray(frag.node_xyz[anchors], dtype=float).mean(axis=0)
    dist, nodes = frag.kdtree.query(center, k=max_nodes + 1, distance_upper_bound=radius_um)
    near = sorted((float(d), int(n)) for d, n in zip(np.atleast_1d(dist), np.atleast_1d(nodes))
                  if np.isfinite(d))
    kept = list(dict.fromkeys(int(a) for a in anchors))
    for _, n in near:
        if n not in kept and len(kept) < max_nodes:
            kept.append(n)
    remap = {n: i for i, n in enumerate(kept)}
    rel = np.asarray(frag.node_xyz[kept], dtype=float) - center
    local = {}
    seg = [local.setdefault(str(frag.node_segment_id(n)), len(local)) for n in kept]  # anchors' segments first
    return {"xyz_um": rel, "radius_um": np.asarray(frag.node_radius[kept], dtype=float),
            "degree": np.array([frag.degree[n] for n in kept]), "segment": np.array(seg),
            "edges": np.array([[remap[a], remap[b]] for a in kept for b in frag.neighbors(a)
                               if b in remap and remap[a] < remap[b]]).reshape(-1, 2),
            "anchor_nodes": [remap[int(a)] for a in anchors],
            "outside_radius": np.linalg.norm(rel, axis=1) > radius_um,
            "truncated": any(n not in remap for _, n in near)}, center
```

- Coordinates are relative to the junction.
- Segment ids are local (0, 1, ...): the junction's segment is 0.
- Edges are the induced subgraph, so a truncated neighbourhood can cut branches.

**Image patches.** Two tiers from the fused volume in the manifest (`image.uri`,
OME-Zarr, axes t,c,z,y,x; use t = 0, c = 0):

| Tier | Zarr level | Half-width | Voxel scale x, y, z (um) | Translation x, y, z (um) |
|---|---|---|---|---|
| `level0` | 0 | 16 um | 0.748, 0.748, 1.0 | 0, 0, 0 |
| `level1` | 1 | 30 um | 1.496, 1.496, 2.0 | 0.374, 0.374, 0.5 |

Graph coordinates are world microns, and `voxel = (world - translation) / scale` in
x, y, z order, reversed to z, y, x for array indexing.

The patch:
- is centred on the junction;
- has an extent per axis of `max(2 * half_width / scale, ptp(anchor voxels) + 16 um / scale)`
  voxels, rounded up and at least 3 (with one anchor, `ptp` is 0, so this is
  `2 * half_width / scale`);
- has its origin at `floor(center - (shape - 1) / 2)`;
- is read with anonymous S3 access and `tensorstore`;
- has voxels outside the volume set to 0 and marked invalid.

Reference code (`manifest = json.load(open("rerank_companion_794495_merge.json"))`):

```python
import os
os.environ.setdefault("AWS_EC2_METADATA_DISABLED", "true")   # public S3: no credential lookup
import tensorstore as ts

def open_level(manifest, level):
    """Open one zarr level of the fused image named in the companion manifest."""
    bucket, key = manifest["image"]["uri"][len("s3://"):].split("/", 1)
    spec = {"driver": "zarr", "kvstore": {"driver": "s3", "bucket": bucket,
                                          "path": key.rstrip("/") + f"/{level}/"}}
    return ts.open(spec, read=True).result()

def image_patch(array, manifest, level, anchors_xyz, half_width_um):
    """Patch around the anchors (world microns, shape (n, 3)); returns the image keys of `context`."""
    geo = manifest["image"]["levels"][str(level)]
    scale, translation = np.array(geo["scale_xyz_um"]), np.array(geo["translation_xyz_um"])
    voxels = ((np.asarray(anchors_xyz, dtype=float) - translation) / scale)[:, ::-1]  # z, y, x
    spacing = scale[::-1]
    extent = np.maximum(2 * half_width_um / spacing, np.ptp(voxels, axis=0) + 16.0 / spacing)
    shape = np.maximum(3, np.ceil(extent).astype(int))
    origin = np.floor(voxels.mean(axis=0) - (shape - 1) / 2).astype(int)
    lo, hi = np.maximum(origin, 0), np.minimum(origin + shape, array.shape[2:])
    image = np.zeros(tuple(shape), dtype=array.dtype.numpy_dtype)
    valid = np.zeros(tuple(shape), dtype=bool)
    if (hi > lo).all():
        block = array[0, 0, int(lo[0]):int(hi[0]), int(lo[1]):int(hi[1]), int(lo[2]):int(hi[2])].read().result()
        target = tuple(slice(int(a), int(b)) for a, b in zip(lo - origin, hi - origin))
        image[target], valid[target] = block, True
    anchors = np.full((2, 3), np.nan)
    anchors[:len(voxels)] = voxels - origin
    return {"image_available": bool(valid.any()), "image_zyx": image, "valid_zyx": valid,
            "anchors_zyx": anchors, "spacing_zyx_um": spacing.copy(), "level": level,
            "image_radius_um": float(half_width_um)}
```

Open each level once and reuse the array for all candidates. A patch takes about
0.1-0.2 s; parallelize with threads when reading many. On first access tensorstore
may log `AuthCredentialsProvider ... Failed to open file ~/.aws/config` lines: they
are harmless (no credential found, so it reads anonymously). Verify scale and
translation against the zarr `multiscales` metadata (`fused.zarr/.zattrs`) once.

**The context dictionary:**

```python
context = {
    "kind": "merge",
    "fragment": neighbourhood_dict,          # as returned above
    # only when the feature declares image input:
    "image_available": bool,                 # False if the read failed or the patch is empty
    "image_zyx": np.ndarray,                 # raw fluorescence, original values (no normalization)
    "valid_zyx": np.ndarray(bool),           # True where the voxel is inside the volume
    "anchors_zyx": np.ndarray (2, 3),        # row 0: junction voxel centre in patch coordinates; row 1: NaN
    "spacing_zyx_um": np.ndarray (3,),       # voxel size, z, y, x
    "level": 0 or 1,
    "image_radius_um": 16.0 or 30.0,
}
```

For image features, the fragment nodes can be mapped into the patch with the same
transform: `nodes_zyx = (rel_xyz + center - translation) / scale`, reversed to z, y, x,
minus the patch origin.

Putting it together for companion row `i`:

```python
def make_context(frag, comp, i, array=None, manifest=None, level=0):
    anchors = [int(comp["node_id"][i])]
    fragment, center = neighbourhood(frag, anchors)
    context = {"kind": "merge", "fragment": fragment}
    if array is not None:                     # only for features that declare image input
        half_width = {0: 16.0, 1: 30.0}[level]
        context.update(image_patch(array, manifest, level, frag.node_xyz[anchors], half_width))
    return context
```

### Feature contract

Write each family of features as one module:

```python
DESCRIPTOR = {
    "kind": "merge",
    "inputs": "both",             # "geometry", "image" or "both"
    "image_tier": "level0",       # "level0" or "level1" (ignored for geometry)
    "feature_names": ["name_one", "name_two"],
}

def describe(context):
    ...
    return {"name_one": float_value, "name_two": float_value}
```

Rules:
- `describe` reads only `context`. No files, network, globals carrying data,
  labels, GT, absolute coordinates, or segment/node IDs. Relative coordinates and
  local segment ids inside `context` are allowed.
- Deterministic: the same context gives identical values on every call.
- Return exactly the declared names, with finite floats or `NaN` (NaN when the
  quantity is undefined; never infinity).
- At most 32 names per module, matching `[a-z][a-z0-9_]{1,48}`.
- Choose the image tier on purpose and say why:
  - `level0`: finer voxels (0.75 um), smaller field of view (about 32 um);
  - `level1`: coarser voxels (1.5 um), larger field of view (about 60 um).
- Do not use quantities that only encode "close to a traced neuron" versus "far
  from GT". Every evaluated row is already GT-judgeable, so such a quantity cannot
  help and only signals leakage.

### Evaluation

**How features are judged.** Candidates are ranked by a score. What matters is the
**precision of the top of the ranking among labeled rows**: of the highest-ranked
labeled candidates, what fraction are merges? NaN rows never enter a ranking.

Compute this **out of fold** with the companion folds. They are label-free groups by
segment, so junctions of one segment never straddle training and evaluation:
- **One feature alone**: choose its direction (higher or lower = more merge-like)
  on the fitting rows of each fold, then rank the held-out fold's labeled rows by it.
- **A model of several features**: for fold `f`, fit on rows with
  `fit_mask_fold{f}` and a non-NaN label; score the rows of `fold == f`.
- Never tune on a held-out fold. Repeatedly comparing many variants on the same
  folds overfits them; report how many variants you compared.

**Scoring code (use it as given):**

```python
def candidate_keys(comp):
    """Tie-break key of every row: 'merge:<segment>:<node>'."""
    return np.char.add(np.char.add("merge:", comp["segment_id"].astype(str)),
                       np.char.add(":", comp["node_id"].astype(str)))

def precision_at_fraction(comp, scores, fraction, keys=None):
    """Out-of-fold precision of the top `fraction` of labeled rows, summed over folds.
    Pass `keys=candidate_keys(comp)` when calling repeatedly."""
    label, fold = comp["label"], comp["fold"]
    keys = candidate_keys(comp) if keys is None else keys
    tp = k_total = 0
    for f in np.unique(fold):
        rows = np.flatnonzero((fold == f) & ~np.isnan(label))
        k = max(1, int(round(fraction * len(rows))))
        order = rows[np.lexsort((keys[rows], -np.asarray(scores, dtype=float)[rows]))]  # ties: key order
        top = order[:k]
        tp += int(label[top].sum())
        k_total += len(top)
    return tp / k_total

def average_precision(comp, scores):
    label = comp["label"]
    rows = np.flatnonzero(~np.isnan(label))
    order = rows[np.argsort(-np.asarray(scores, dtype=float)[rows], kind="stable")]
    hits = label[order] == 1
    return float(np.sum(np.cumsum(hits)[hits] / (np.flatnonzero(hits) + 1)) / max(hits.sum(), 1))
```

Scores must be finite for every labeled row; for ranking, put rows whose feature
is NaN last.

Report for each feature and for the model:
- precision at 1%, 2.5%, 5% and 10% of the labeled rows, and the average precision;
- per-fold values;
- the same numbers for the baselines below;
- a paired bootstrap 95% interval of the difference to the strongest baseline:
  resample labeled rows with replacement within each fold, 1,000 draws, same draws
  for both scores.

**Baselines (compute them yourself from the data):**
- **Random ranking**: its precision equals the share of label-1 rows.
- **Junction degree** (higher ranked first). 99.9% of candidates have degree 3, but
  the few with degree >= 4 are enriched in merges, so degree helps at the very top
  of the ranking.
- **Segment cable length**: total edge length of the candidate's segment, direction
  chosen on the fitting rows.

Use the stronger of degree and cable length as the baseline. A feature is
interesting if it beats that baseline out of fold with a bootstrap interval above
zero, or adds to a model that already contains the baseline.

### Firewall and validity checks

- **Phase 2 features may not read**: GT graph or GT fields, labels, folds, fit
  masks, companion IDs or absolute coordinates, `segmentation_path`, or any file.
  Only `context`.
- **No availability proxies** (see the feature contract).
- **Images** only from the URI in the companion manifest. Do not open any other
  image path.
- **Sanity checks to include**: label shuffling within folds (the gain should
  vanish); per-fold consistency; the fraction of labeled rows where each feature is
  NaN, and whether that differs between labels 1 and 0 (missingness can itself be
  a leak).

### Deliverable

For each feature family you keep:
1. the module (`DESCRIPTOR` + `describe`) exactly as in the feature contract;
2. the context-building code you used, if you changed anything in how it is built,
   with the reason;
3. an evidence record:
   - the hypothesis in one or two sentences, from your Phase 1 observations;
   - out-of-fold precision at 1/2.5/5/10% and AP, with per-fold values and baselines;
   - bootstrap interval of the difference to the strongest baseline;
   - NaN fraction per label;
   - how many variants you compared;
   - typical failure cases (top-ranked label-0 rows, missed label-1 rows) as
     companion row indices.

Also report the negative results: ideas that did not beat the baseline, with their
numbers.

---

## 3) Dataset schema

### 3.1 `dataset_cache_794495_mcl100_add.pkl`

**Loading:**

```python
import json
import pickle
import numpy as np
import agentic_neuron_proofreader  # noqa: F401  registers SkeletonGraph for unpickling

with open("dataset_cache_794495_mcl100_add.pkl", "rb") as f:
    payload = pickle.load(f)
frag = payload["fragments_graph"]   # automated reconstruction (Phase 1 and Phase 2)
gt = payload["gt_graph"]            # human GT (Phase 1 only)
```

**Install once** (Python >= 3.10; public sources, no credentials):

```bash
pip install git+https://github.com/AllenInstitute/agentic-neuron-proofreader.git tensorstore psutil
```

`agentic_neuron_proofreader` defines `SkeletonGraph`, which `pickle.load` needs to
rebuild the graphs; it pulls in `numpy`, `scipy` and `networkx`, but imports
`psutil` without declaring it, so install that too. `tensorstore` reads
the image (§2). If importing raises `numpy.dtype size changed`, numpy and scipy are
binary-incompatible: install them together in one environment.

Budget > 20 GB RAM; loading needs no network (under a minute to a few minutes).

**`fragments_graph`** (a `SkeletonGraph`, networkx-based):

| Access | Meaning |
|---|---|
| `frag.node_xyz[i]` | node position, world microns, (x, y, z) |
| `frag.node_radius[i]` | skeleton radius estimate, microns (may be NaN) |
| `frag.degree[i]`, `frag.neighbors(i)` | graph degree and neighbors |
| `frag.node_segment_id(i)` | segment id of node `i`, returned as a **string** (companion ids are integers: compare with `int(...)`) |
| `frag.kdtree` | `scipy.spatial.cKDTree` over `node_xyz` (call `frag.set_kdtree()` if `None`) |

Node ids are contiguous integers `0..N-1`, the same ids as in the companion.

**GT fields (Phase 1 only):**

| Key | Meaning |
|---|---|
| `payload["gt_node_canonical_label"]` | per GT node: the segment id found at that node in the segmentation; 0 = no segment (omitted stretch) |
| `payload["gt_edge_error"]` | per GT edge (order of `list(gt.edges)`): 0 correct, 1 split, 2 omit, 3 merged |
| `payload["gt_merge_labels"]` | segment ids flagged as merges |
| `payload["gt_merge_sites"]` | merge sites: `{"segment_id", "gt_neuron", "xyz"}` plus source fields |

### 3.2 `rerank_companion_794495_merge.npz`

58,120 rows, one per junction. Load with
`comp = dict(np.load(path, allow_pickle=False))`. The `dict(...)` matters: indexing
an open `np.load` file decompresses the whole column on every access, so row-by-row
reads become extremely slow. Row `i` of every column describes candidate `i`.

| Column | Dtype | Meaning |
|---|---|---|
| `row` | int64 | candidate index 0..n-1 |
| `label` | float64 | 1 / 0 / NaN (§1) |
| `fold` | int8 | out-of-fold group 0, 1 or 2; label-free, grouped by segment |
| `fit_mask_fold0..2` | bool | rows usable for fitting when that fold is held out |
| `segment_id`, `node_id` | int64 | the junction node and its segment |
| `degree` | int16 | junction degree |
| `x_um`, `y_um`, `z_um` | float64 | junction position, world microns |

IDs and coordinates are for **locating** a candidate in `fragments_graph` and in
the image only.

### 3.3 `rerank_companion_794495_merge.json`

The manifest of §3.2:

| Key | Meaning |
|---|---|
| `version`, `created`, `brain`, `kind`, `mcl` | identity of the export |
| `rows`, `label_counts`, `labeled_rows` | row count and counts of 1 / 0 / NaN |
| `candidate_rule`, `candidate_policy` | the candidate and label rules as text, and their short code |
| `label_availability` | which rows GT can judge (`definition`: `near_gt_150`) |
| `folds` | number of folds, grouping (`fragment`), fit-mask meaning, partition digest |
| `source_cache` | file name and byte size of the pkl these rows come from |
| `image` | `uri` of the fused OME-Zarr, access notes, `graph_to_world` (identity), and `levels`: per zarr level, `scale_xyz_um` and `translation_xyz_um` |
| `columns` | column names and dtypes of the `.npz` |
| `pool_sha256`, `labels_sha256`, `npz_sha256` | digests of the candidate pool, the labels and the `.npz` file |
