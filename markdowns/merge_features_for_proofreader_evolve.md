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
Revised 2026-10-15: tie-aware average precision with corrected baseline figures and
reporting rules for differences and subgroups (Evaluation), fewer constraints on what
to explore (Goal, Evaluation, Firewall); the previous text is archived.

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

For this brain: 58,120 candidates; label 1: 128; label 0: 6,882; NaN: 51,110.

Precision and AP use labeled rows only (label 1 or 0); the one check that looks at
unlabeled rows is described under Evaluation. Never treat NaN as 0: a NaN
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

Discover measurable, GT-blind properties of a junction that put true merges ahead
of non-merges, among the candidates whose label GT can decide. Each property is a
**feature**: one deterministic value for every companion row.

What happens to your results: every feature you record is later re-run from your
code on this brain and on other brains, audited statistically, and the features
that hold up are combined into a detector whose score ranks these same candidates.
So a feature must be computable from data that exists for any brain (the fragment
reconstruction, the candidate list and the raw image), and its code must run
unchanged on another brain's files.

This document deliberately contains no list of known or expected features. Ideas
may come from anywhere, the data or what you know about neurons and about imaging,
but every feature you keep must be tied to something you observed in this data in
Phase 1 and must be computable from the allowed inputs alone. Do not try to recover
any earlier detector.

Your features will later be confirmed on data you cannot see. A feature that only
works on this brain's folds will not survive that confirmation.

### Two phases

- **Phase 1, characterization.** You may use everything in the pkl, including GT
  and labels, to understand what distinguishes merges from non-merges: inspect
  examples, compare distributions, view image patches.
- **Phase 2, features.** Turn what you learned into features computed **only** from
  the allowed inputs (see Feature definition). Phase 2 features must never read GT
  or labels, and never use ids or absolute coordinates as values.

### Helper code (optional)

Two tested pieces for local features: a fragment neighbourhood and a raw-image
patch around a candidate, combined into a `context` dictionary. They are optional:
a feature may use any allowed input (see Feature definition).

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
Use `open_level` as written: the `kvstore` spec takes exactly `driver`, `bucket` and
`path`, and extra members (for example `auth_provider`) make `ts.open` fail. If the
image cannot be opened or a patch cannot be read, fix the call or stop the
experiment; never continue with invented intensities (see Firewall). The
`INPUTS READ` line (Code layout) reports how many patches were read.

**The context dictionary:**

```python
context = {
    "kind": "merge",
    "fragment": neighbourhood_dict,          # as returned above
    # only when an image is requested (array given to make_context):
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
    if array is not None:                     # only when the feature uses the image
        half_width = {0: 16.0, 1: 30.0}[level]
        context.update(image_patch(array, manifest, level, frag.node_xyz[anchors], half_width))
    return context
```

### Feature definition

**Allowed inputs** for a Phase 2 feature:
- the whole `fragments_graph`: any node, edge, segment or graph-level structure
  (lengths, branching, orientation, radii, neighbouring segments), at any distance
  from the candidate;
- the candidate columns of the companion that are not labels: `degree`. Node and
  segment ids may be used to *look up* graph elements;
- the raw image from the manifest URI, at any level and location.

Everything else is excluded (see Firewall and validity checks).

**Shape.** A feature is one value per companion row, in row order: an array of
length n, for all rows, not only the labeled ones. Use NaN where it is undefined;
never drop or reorder rows. The same inputs must give identical values.

**State the definition precisely**, because the feature will be re-implemented in a
detector from your code and your description:
- **its scope**: a property of the junction itself (candidate scope), or of its
  whole segment (segment scope: the same value for every junction of that
  segment);
- **the formula** with its constants (radii, thresholds, image level and patch
  size).

**Cost.** It must be computable for every row of a brain (58,120 rows here) within
hours: share preprocessing across rows (one pass over segments, a KD-tree built
once, one image read per candidate) and never scan the whole graph per row. Report
the wall time.

**Code layout.** Write each experiment as a self-contained script that loads the
three files by their names here (`dataset_cache_794495_mcl100_add.pkl`,
`rerank_companion_794495_merge.npz` and `rerank_companion_794495_merge.json`) from
the working directory, computes the feature for all rows, evaluates it and prints
the results. It will be re-run with another brain's files substituted under the
same names, so do not hard-code row counts, ids or brain-specific constants. Print
one line of the form
`INPUTS READ fragment_nodes=<n> companion_rows=<n> image_patches=<read>/<requested>`
(`image_patches=0/0` when the feature uses no image), so the record itself shows
what its numbers were computed from.

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

def precision_at_fraction(comp, scores, fraction, keys=None, counts=False):
    """Out-of-fold precision of the top `fraction` of labeled rows, summed over folds.
    Pass `keys=candidate_keys(comp)` when calling repeatedly; `counts=True` returns
    (hits, rows ranked) instead of their ratio."""
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
    return (tp, k_total) if counts else tp / k_total

def average_precision(comp, scores):
    """Average precision over labeled rows. Tied scores form one threshold (as in
    sklearn.metrics.average_precision_score), so a score that ties many rows is not
    credited for the row order inside its ties."""
    label = comp["label"]
    rows = np.flatnonzero(~np.isnan(label))
    s = -np.asarray(scores, dtype=float)[rows]
    order = np.argsort(s, kind="stable")
    s, hits = s[order], label[rows][order] == 1
    last = np.flatnonzero(np.r_[s[1:] != s[:-1], True])   # last row of every tie group
    tp = np.cumsum(hits)[last]
    return float(np.sum(tp / (last + 1) * np.diff(np.r_[0, tp])) / max(hits.sum(), 1))

def labeled_share_at_fraction(comp, scores, fraction):
    """Share of GT-labeled rows among the top `fraction` of ALL rows (labeled or not)."""
    order = np.argsort(-np.asarray(scores, dtype=float), kind="stable")
    top = order[:max(1, int(round(fraction * len(order))))]
    return float((~np.isnan(comp["label"][top])).mean())
```

Scores must be finite for every row (`labeled_share_at_fraction` ranks all rows):
where a feature is NaN, use a score that ranks the row last.

Report for each feature and for the model:
- precision at 1%, 2.5%, 5%, 10%, 15% and 20% of the labeled rows, with the hit
  counts behind each (`counts=True`): positives are few, so a precision can rest on
  a handful of hits;
- the average precision of the out-of-fold scores (all folds pooled);
- per-fold values;
- the same numbers for the baselines below;
- a paired bootstrap 95% interval of the **difference in average precision to the
  degree baseline** (not to the random rate, and not a difference in ROC-AUC):
  resample labeled rows with replacement within each fold, 1,000 draws, same draws
  for both scores.

All of these are computed on the whole labeled set. A number computed on a subset
of rows (one fold, one degree value, one region) must carry the subset's row and
positive counts next to it and is not evidence on its own: a stratum of 20 rows
with 5 positives gives even a random score an AP far above the 0.027 of the whole
set.

**Labeled versus unlabeled rows.** Only 7,010 of 58,120 rows are labeled (12.1%),
and they are not a random sample: they lie on or near traced neurons. The ranking that is
finally used covers all rows, and nothing above shows what a feature does to the
unlabeled ones. For each feature, also report:
- `labeled_share_at_fraction` at 1%, 5% and 20% of all rows, next to the overall
  labeled share (12.1%): a feature that fills the top of the full ranking with
  unlabeled rows, or only with labeled ones, behaves differently off the traced
  neurons;
- how its distribution differs between labeled and unlabeled rows (for example the
  two-sample Kolmogorov-Smirnov statistic and the medians).

Flag every feature whose labeled and unlabeled distributions differ strongly; it may
measure "how traced-neuron-like" a candidate is rather than whether it is an error.

**Baselines (compute them yourself from the data; the figures below are what the
companion of 2026-10-13 gives with the scoring code above, and must be recomputed
when the companion changes):**
- **Random ranking**: its precision equals the share of label 1 among labeled rows,
  0.018 here (128 of 7,010).
- **Junction degree** (higher ranked first). 58,047 of the 58,120 candidates have
  degree 3, 72 have degree 4 and one has degree 5; the few with degree >= 4 are
  enriched in merges, so degree helps only at the very top: out-of-fold AP 0.027
  (folds 0.020 / 0.035 / 0.038); precision 7/70 at 1%, 8/175 at 2.5%, 11/351 at 5%,
  14/700 at 10%, 18/1,051 at 15%, 34/1,403 at 20%. Its top 1% of all rows holds 17%
  labeled rows against 12% overall. Degree ties 6,990 of the 7,010 labeled rows; an
  AP that credits the row order inside those ties (a plain stable sort) comes out
  at 0.044, the tie-aware `average_precision` above gives 0.027, and 0.027 is the
  figure to beat.
- **Segment cable length** (total edge length of the candidate's segment, direction
  chosen on the fitting rows) is **not** a usable baseline: AP 0.017, 0/70 at 1%,
  13/1,403 at 20%. Labeled junctions sit on long segments (median 10,796 um against
  730 um for all candidates), so 99% of its top 1% of all rows are labeled rows: it
  measures how traced-neuron-like a candidate is, not whether it is a merge. Keep it
  as the worked example of the availability proxy the Firewall excludes.

Use degree as the baseline. With degree near the random rate, almost any
informative quantity will beat it, so beating it is necessary but not sufficient.
A feature counts as found when the paired-bootstrap interval of its out-of-fold
increment in AP over degree lies above zero, alone or in the two-column model
(degree plus the feature), and the labeled share at the top of its full ranking
stays near the overall 12%: a feature whose top 1% of all rows is mostly labeled
rows is doing what cable length does and is a negative result. With 128 positives,
precision at 1% is 70 ranked rows and a few hits, and one fold can carry a third of
the positives: judge mainly by the AP and its interval, and report per-fold values.
What to measure is yours to find; report how many variants you compared.

### Firewall and validity checks

- **Phase 2 features may not read**: the GT graph or any GT field (`gt_*` keys),
  labels, folds or fit masks, `segmentation_path` or the dense segmentation, or any
  file other than the three given ones.
- **No ids or coordinates as values.** Ids may index the graph and absolute
  coordinates may locate an image patch, but neither may enter a feature value: an
  id or a position says where in this brain the candidate is, not what it looks
  like, and does not transfer to another brain.
- **No availability proxies**: quantities that only encode "close to a traced
  neuron" versus "far from GT". Every evaluated row is already GT-judgeable, so such
  a quantity cannot help and only signals leakage.
- **Images** only from the URI in the companion manifest. Do not open any other
  image path.
- **No substitute data.** If an input cannot be read (the pkl, the companion, the
  image at the manifest URI), the script must raise and stop. Never fall back to
  simulated, synthetic, placeholder or "calibrated" values, and never generate
  values from labels or from an assumption about what merges look like: a result
  computed on substituted data is void whatever its numbers, and is discarded.
  Print for every input that it was read and how many rows or patches came from
  it (the `INPUTS READ` line).
- **Names must match inputs.** A feature's name and description must say what its
  code reads: a quantity computed from graph geometry is not an image feature, and
  a "contrast" or "intensity" whose code never touched the image is substituted
  data.
- **Implausible results are bugs until proven otherwise.** With 128 positives
  among 7,010 labeled rows, honest features move the average precision by
  hundredths. An out-of-fold AP above 0.9, a gain above 0.3 or perfect precision
  at every fraction has so far come only from leaked or substituted inputs;
  re-check the input path and the firewall before reporting such a number.
- **Sanity checks.** When you fit a model, shuffle the labels within folds and show
  that its gain vanishes. Also worth reporting: per-fold consistency, and the
  fraction of labeled rows where the feature is NaN by label (missingness can
  itself be a leak).

### Deliverable

Record each feature as its own experiment (hypothesis, code, numerical result,
interpretation), so it can be re-run on its own. For each feature you keep:
1. the script that computes it for all rows and evaluates it (see Code layout);
2. its precise definition (inputs, scope, formula, constants);
3. an evidence record:
   - the hypothesis in one or two sentences, from your Phase 1 observations;
   - out-of-fold precision at 1/2.5/5/10/15/20% with hit counts, and AP, with
     per-fold values and baselines;
   - bootstrap interval of the AP difference to degree, and the increment of the
     two-column model (degree plus the feature) over degree alone with its
     interval;
   - the `INPUTS READ` line (fragment nodes, companion rows, image patches read
     and requested), since a run on substituted data is void;
   - NaN fraction per label;
   - labeled share in the top of the full ranking, and the labeled-versus-unlabeled
     distribution difference;
   - wall time for all rows;
   - how many variants you compared;
   - typical failure cases (top-ranked label-0 rows, missed label-1 rows) as
     companion row indices.

A model of several features is useful evidence (does a feature add to the
baseline?), but the final model is refitted later, so the features matter more
than the model.

Also report the negative results: ideas that did not beat the baseline, with their
numbers.

---

## 3) Dataset schema

### 3.1 `dataset_cache_794495_mcl100_add.pkl`

**Install once** (Python >= 3.10; public sources, no credentials):

```bash
pip install git+https://github.com/AllenInstitute/agentic-neuron-proofreader.git tensorstore psutil
```

`agentic_neuron_proofreader` defines `SkeletonGraph`, which `pickle.load` needs to
rebuild the graphs; it pulls in `numpy`, `scipy` and `networkx`, but imports
`psutil` without declaring it, so install that too. `tensorstore` reads
the image (§2). If importing raises `numpy.dtype size changed`, numpy and scipy are
binary-incompatible: install them together in one environment.

**Loading:**

```python
import json    # for the companion manifest
import pickle
import numpy as np
import agentic_neuron_proofreader  # noqa: F401  registers SkeletonGraph for unpickling

with open("dataset_cache_794495_mcl100_add.pkl", "rb") as f:
    payload = pickle.load(f)
frag = payload["fragments_graph"]   # automated reconstruction (Phase 1 and Phase 2)
gt = payload["gt_graph"]            # human GT (Phase 1 only)
```

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
