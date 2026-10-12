# Discovering candidate-level features for ExaSPIM split candidates

This document is the complete task specification. You receive exactly these files,
colocated:

| File | What it is | Schema |
|---|---|---|
| this markdown | the task | |
| `dataset_cache_794495_mcl100_add.pkl` | one labeled ExaSPIM brain: automated fragment reconstruction, human ground truth (GT) and GT-derived error labels | §3.1 |
| `rerank_companion_794495_split.npz` | the fixed split candidates of this brain, their labels and fold assignment | §3.2 |
| `rerank_companion_794495_split.json` | manifest of the companion: rule text, counts, digests, folds, image geometry | §3.3 |

The raw fluorescence image is on public S3 (anonymous read; no credential needed).
Its URI and level geometry are in the companion `.json`.

This task is about **split errors only**. The document has three parts: §1 what the
data is, §2 what to do with it and how it is judged, §3 the schema of each file.

---

## 1) Dataset context

### Imaging, reconstruction and split errors

ExaSPIM images a whole mouse brain at about 0.75 x 0.75 x 1 um. An automated
segmentation, skeletonized into `fragments_graph`, reconstructs neurons as many
**segments** (one segment = one predicted neuron piece). Human annotators traced a
few neurons completely (`gt_graph`). Comparing the two gives the error labels baked
into the pkl.

A **split** error is one traced neuron broken into two segments. The fix is to join
the two segments.

### Candidates are fixed

The companion lists every split candidate, enumerated by a fixed geometric rule
from `fragments_graph` alone (no GT, no trained model). The rule text is in the
companion `.json` (`candidate_rule`).

A candidate is a *segment pair*. From every fragment tip (degree-1 node), the
nearest node of each other segment within 50 um is found, and the 2 nearest partner
segments are kept. There is one row per unordered pair. A pair can be found several
times (from either side, or from different tips); each finding is an
*occurrence*, and the shortest-gap occurrence is the pair's representative.

Use exactly these candidates, in companion row order. Do not add, drop or re-enumerate
candidates; features that only exist on a different candidate set cannot be used.

### Labels are three-valued

GT covers only a few traced neurons, so for most candidates GT cannot say whether
they are errors. The companion `label` column is:

| Value | Meaning |
|---|---|
| `1` | error: the pair is a GT split (distinct non-zero segment labels across a GT split edge, or the two flanks of a zero-label GT stretch on one neuron) |
| `0` | GT can judge the candidate and it is not an error: both segments carry a GT label |
| `NaN` | GT cannot judge it: no traced neuron on one of the segments. **Unknown, not "no error."** |

For this brain: 500,111 candidates; label 1: 2,018; label 0: 1,983; NaN: 496,110.

All evaluation uses labeled rows only (label 1 or 0). Never treat NaN as 0: a NaN
row away from traced neurons says nothing about errors. Labeled rows are about half
positive, so the bar is how cleanly the top of the ranking separates true splits.

### Raw image

The fused fluorescence volume of this brain is an OME-Zarr on public S3, readable
anonymously. Graph coordinates are world microns and map to the image by the
per-level scale and translation in the companion `.json` (§3.3). The dense
segmentation is not available and is not needed.

---

## 2) Intent

### Goal

Discover measurable properties of a segment pair that put true splits ahead of
non-splits, among the candidates whose label GT can decide. The deliverable is a set
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
  and labels, to understand what distinguishes splits from non-splits: inspect
  examples, compare distributions, view image patches.
- **Phase 2, features.** Turn what you learned into features computed **only** from
  the candidate context (below). Phase 2 features must never read GT, labels,
  absolute coordinates, or segment/node IDs.

### Candidate context

A feature sees one candidate through a `context` dictionary. Build it exactly as
below, so the same feature code can later be run on other brains.

**Anchors.** `[node_id_a, node_id_b]` of the representative occurrence. Other
occurrences may be inspected in Phase 1.

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

- Coordinates are relative to the anchor midpoint.
- Segment ids are local (0, 1, ...): segment `a` is 0 and segment `b` is 1.
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
- is centred on the anchor midpoint;
- has an extent per axis of `max(2 * half_width / scale, ptp(anchor voxels) + 16 um / scale)`
  voxels, rounded up and at least 3, so both anchors fit with an 8 um margin;
- has its origin at `floor(center - (shape - 1) / 2)`;
- is read with anonymous S3 access and `tensorstore`;
- has voxels outside the volume set to 0 and marked invalid.

Reference code (`manifest = json.load(open("rerank_companion_794495_split.json"))`):

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
    "kind": "split",
    "fragment": neighbourhood_dict,          # as returned above
    # only when the feature declares image input:
    "image_available": bool,                 # False if the read failed or the patch is empty
    "image_zyx": np.ndarray,                 # raw fluorescence, original values (no normalization)
    "valid_zyx": np.ndarray(bool),           # True where the voxel is inside the volume
    "anchors_zyx": np.ndarray (2, 3),        # anchor a and b voxel centres in patch coordinates
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
    anchors = [int(comp["node_id_a"][i]), int(comp["node_id_b"][i])]
    fragment, center = neighbourhood(frag, anchors)
    context = {"kind": "split", "fragment": fragment}
    if array is not None:                     # only for features that declare image input
        half_width = {0: 16.0, 1: 30.0}[level]
        context.update(image_patch(array, manifest, level, frag.node_xyz[anchors], half_width))
    return context
```

### Feature contract

Write each family of features as one module:

```python
DESCRIPTOR = {
    "kind": "split",
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
labeled candidates, what fraction are splits? NaN rows never enter a ranking.

Compute this **out of fold** with the companion folds. They are label-free 500 um
spatial blocks, so neighbouring candidates do not straddle training and evaluation:
- **One feature alone**: choose its direction (higher or lower = more split-like)
  on the fitting rows of each fold, then rank the held-out fold's labeled rows by it.
- **A model of several features**: for fold `f`, fit on rows with
  `fit_mask_fold{f}` and a non-NaN label; score the rows of `fold == f`. The fit
  masks already exclude rows that share a fragment with the held-out fold.
- Never tune on a held-out fold. Repeatedly comparing many variants on the same
  folds overfits them; report how many variants you compared.

**Scoring code (use it as given):**

```python
def candidate_keys(comp):
    """Tie-break key of every row: 'split:<segment_a>:<segment_b>'."""
    return np.char.add(np.char.add("split:", comp["segment_id_a"].astype(str)),
                       np.char.add(":", comp["segment_id_b"].astype(str)))

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
- **Gap distance**: `gap_um`, smaller gap ranked first. This baseline is strong.

A feature is interesting if it beats the strongest baseline out of fold with a
bootstrap interval above zero, or adds to a model that already contains the
baseline.

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

### 3.2 `rerank_companion_794495_split.npz`

500,111 rows, one per unordered segment pair. Load with
`comp = dict(np.load(path, allow_pickle=False))`. The `dict(...)` matters: indexing
an open `np.load` file decompresses the whole column on every access, so row-by-row
reads become extremely slow. Row `i` of every candidate column describes
candidate `i`.

| Column | Dtype | Meaning |
|---|---|---|
| `row` | int64 | candidate index 0..n-1 |
| `label` | float64 | 1 / 0 / NaN (§1) |
| `fold` | int8 | out-of-fold group 0, 1 or 2; label-free 500 um spatial blocks |
| `fit_mask_fold0..2` | bool | rows usable for fitting when that fold is held out (rows sharing a fragment with it are purged) |
| `segment_id_a`, `segment_id_b` | int64 | the two segments of the pair (sorted) |
| `node_id_a`, `node_id_b` | int64 | representative (shortest-gap) occurrence: the node on each segment |
| `gap_um` | float64 | distance between those two nodes |
| `anchor_side` | U1 | which side (`a` or `b`) was the tip anchor |
| `partner_rank` | int16 | partner rank of the other segment (1 = nearest) |
| `node_role_a`, `node_role_b` | U7 | `tip` or `non_tip` |
| `occ_row` | int64 | for every occurrence: the candidate row it belongs to |
| `occ_node_id_a`, `occ_node_id_b`, `occ_gap_um`, `occ_anchor_side`, `occ_partner_rank`, `occ_node_role_a`, `occ_node_role_b` | as above | the same fields for every occurrence |

The occurrence arrays are longer than the candidate arrays (765,286 entries): one
per anchor/partner realisation, sorted by `occ_row`, the first per pair being the
representative one.

IDs and coordinates are for **locating** a candidate in `fragments_graph` and in
the image only.

### 3.3 `rerank_companion_794495_split.json`

The manifest of §3.2:

| Key | Meaning |
|---|---|
| `version`, `created`, `brain`, `kind`, `mcl` | identity of the export |
| `rows`, `label_counts`, `labeled_rows` | row count and counts of 1 / 0 / NaN |
| `candidate_rule`, `candidate_policy` | the candidate and label rules as text, and their short code |
| `label_availability` | which rows GT can judge (`definition`: `segment_on_gt`) |
| `folds` | number of folds, grouping (`spatial_blocks_with_fragment_purge`, `spatial_block_um` 500), fit-mask meaning, partition digest |
| `source_cache` | file name and byte size of the pkl these rows come from |
| `image` | `uri` of the fused OME-Zarr, access notes, `graph_to_world` (identity), and `levels`: per zarr level, `scale_xyz_um` and `translation_xyz_um` |
| `columns` | column names and dtypes of the `.npz` |
| `pool_sha256`, `labels_sha256`, `npz_sha256` | digests of the candidate pool, the labels and the `.npz` file |
