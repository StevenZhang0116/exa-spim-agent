# `labeled_dataset_cache` — Pre-scored Dataset Description

A `dataset_cache_<brain_id>_mcl<N>_add.pkl` file is a Python pickle holding a
single dict for **one ExaSPIM brain** — the same two neuron-skeleton graphs as a
plain `dataset_cache_<brain_id>_mcl<N>.pkl` (human ground truth + automated U-Net
reconstruction), **plus pre-computed canonical error labels baked in**. The `_add`
suffix marks a cache that has been scored: every ground-truth node already carries
the predicted segment id it falls on, every ground-truth edge already carries its
split / merge / omit class, and the set of merge segments has already been found.

**The point of the `_add` cache: scoring is already done.** A plain cache only
contains skeletons, so identifying split/merge/omit errors after loading requires
re-deriving them (matching nodes to fragments, picking a tolerance, walking
fragments to find merges). The `_add` cache stores the *results* of that scoring —
read from the dense segmentation **once** at build time — so after loading you
read the error labels directly. **No segmentation, no cloud read, no tolerance to
pick, no merge walk to run.**

> **Read whatever `_add.pkl` files are in `cache/` as ONE pre-scored dataset.**
> Like the plain caches, the format is **brain-agnostic**: every `_add.pkl`
> exposes the same keys, the same label arrays, and the same `SkeletonGraph`
> class, so the same code path handles one cache or many. Discover all
> `dataset_cache_*_add.pkl` in `cache/`, load each, and pool the per-neuron error
> records into a single combined sample set, **tagged with the `brain_id` it came
> from** (parsed from the filename). A run may provide a single brain or several;
> the procedures are identical either way. Any per-file number quoted here is
> illustrative of *one* brain; the quantities that matter are the **aggregates
> over whatever is present**.

---

## 1) Dataset context

**Origin.** Each `_add.pkl` is built from the matching plain
`dataset_cache_<brain_id>_mcl<N>.pkl` by `scripts/relabel_cache.py`. That script
reads the brain's dense predicted **segmentation** volume from cloud storage
*once*, looks up the predicted segment id at every ground-truth node's voxel,
classifies every ground-truth edge, runs the geometric merge-detection walk, and
writes the results back into a new `_add.pkl`. The original cache is left
untouched; the `_add` copy is additive. **All cloud access happens at build time —
the `_add.pkl` itself loads with no segmentation and no credentials.**

**What each `_add` cache captures (on top of the plain cache).** The plain cache's
two graphs are unchanged:

1. **UNet fragment skeletons** (`fragments_graph`) — the automated reconstruction
   (hundreds of thousands of fragment components per brain).
2. **Ground-truth tracings** (`gt_graph`) — human-traced neuron skeletons (order
   tens of neurons per brain).

The `_add` cache adds the **canonical scoring of `gt_graph` against the
segmentation**:

- a **predicted segment id per GT node** (`gt_node_canonical_label`),
- an **error class per GT edge** (`gt_edge_error`: correct / split / omit / merge),
- the **set of merge segments** and the **merge sites** found by the geometric
  walk (`gt_merge_labels`, `gt_merge_sites`).

**The errors these labels encode.** Deep-learning segmentation introduces two
systematic *topological* errors, both already identified in the `_add` cache:

- **Split** — one true neuron broken into multiple predicted segments.
- **Merge** — one predicted segment fusing two distinct neurons.
- **Omit** — a GT stretch with no predicted segment (the reconstruction missed it).

| Metric | Meaning | How it is recovered from the `_add` cache |
|---|---|---|
| Total / per-neuron Splits | distinct predicted segments on a neuron, minus 1 | group `gt_node_canonical_label` by GT neuron |
| % Split / Omit / Merged Edges | fraction of GT edges in each class | count `gt_edge_error` values |
| Merges | predicted segments fusing ≥2 neurons | read `gt_merge_labels` |
| Edge Accuracy | fraction of correctly reconstructed GT edges | `% correct` from `gt_edge_error` |

> **How the labels were produced (provenance only).** Each GT node's label is the
> segment id read from the dense segmentation at that node's voxel, then
> misalignment-healed — mirroring `segmentation_skeleton_metrics`' canonical
> labeling. Merge segments are the **union** of two rules: (a) a segment landing
> on ≥2 GT neurons with >50 nodes on each, and (b) the geometric merge-site walk
> (walk a fragment from a far leaf inward until it re-approaches a *different* GT
> neuron). Together these reproduce the canonical `labels_with_merge` set. You do
> **not** need to rerun any of this to use the cache — it is already stored.

**Known gaps / things to keep in mind.**

- **Labels are exact to the segmentation, but cache fragments are filtered.** The
  GT labels come from the dense segmentation, so split/merge/edge-accuracy track
  the canonical `metrics_out/<brain>/<seg_id>/results.csv` closely. The **omit**
  rate, however, reads slightly **higher** than canonical, because the cache's
  fragments were filtered at `min_cable_length` (a GT stretch reconstructed only
  by a dropped short fragment shows as background → omit). Treat the cache omit
  rate as an upper bound. This bias is consistent across brains (shared
  `min_cable_length = 100` µm), so it does not skew one brain relative to another.
- **Merge percentages match closely, not byte-exactly.** The geometric walk is
  reproduced faithfully, but the canonical pipeline snaps merge sites to nearby
  branch nodes and dedups in a specific order; expect strong agreement with a
  small residual on a few merge-heavy neurons.
- **Ground truth is sparse and skeleton-only.** Only a handful of neurons are
  traced per brain (named `N0XX-<brain_id>-<initials>`). There is no dense GT
  voxel volume; a region with no traced neuron is simply unlabeled.

---

## 2) Dataset schema

### On-disk layout

Every `_add.pkl` is a single dict. It is a **superset** of the plain cache: all
the original keys, plus four added by the labeling step. The last five keys are
the ones you will actually use.

| Key | Type | Meaning |
|---|---|---|
| `fragments_path`, `gt_path`, `img_path` | `str` | Provenance only — source GCS/S3 paths. Not read to use the cache. |
| `segmentation_path` | `str` | Provenance — the dense segmentation the labels were read from. Not read at load time. |
| `anisotropy` | `tuple` | `(0.748, 0.748, 1.0)` — µm/voxel in (x, y, z). |
| `min_cable_length` | `int` | `100` — µm threshold shorter fragments were dropped at. |
| `node_spacing` | `int` | `5` — target µm spacing between skeleton nodes. |
| `fragments_graph` | `SkeletonGraph` | Automated UNet reconstruction (unchanged from the plain cache). |
| `gt_graph` | `SkeletonGraph` | Human ground-truth reconstruction (unchanged geometry; now also carries the label arrays below). |
| **`gt_node_canonical_label`** | `np.ndarray (N_gt,) int64` | Predicted segment id at each GT node's voxel; `0` = unlabeled (omit). |
| **`gt_edge_error`** | `np.ndarray (E_gt,) uint8` | Per-GT-edge class, parallel to `list(gt_graph.edges)`: `0=correct, 1=split, 2=omit, 3=merged`. |
| **`gt_merge_labels`** | `np.ndarray (M,) int64` | Predicted segment ids that fuse ≥2 GT neurons (node-count rule ∪ geometric walk). |
| **`gt_merge_sites`** | `list[dict]` | One entry per merge site: `{"segment_id": int, "gt_neuron": str, "xyz": (x,y,z) µm}`. |

The four added arrays are also attached to `gt_graph` as
`gt_graph.node_label`, `gt_graph.edge_error`, `gt_graph.merge_labels`, and
`gt_graph.merge_sites`, so you can read them off either the payload dict or the
graph. Their combined size is tiny (~tens of MB) next to the multi-GB graphs.

> **`_add.pkl` vs plain `.pkl`.** A plain cache lacks the four label keys (or
> stores `None`). The same loader reads both — a plain cache simply has no labels
> attached, and you would fall back to deriving them. Prefer the `_add.pkl` when
> present: the labels are exact and free.

### Loading (one cache)

```python
import numpy as np
# Requires the agentic_neuron_proofreader package on the path.
from agentic_neuron_proofreader.data_modules.datasets import BrainDataset

ds = BrainDataset.load_from_cache("cache/dataset_cache_794495_mcl100_add.pkl")
gt = ds.gt_graph

node_label   = np.asarray(gt.node_label)     # (N_gt,) predicted segment id, 0 = omit
edge_error   = np.asarray(gt.edge_error)      # (E_gt,) 0=correct 1=split 2=omit 3=merged
merge_labels = set(int(x) for x in gt.merge_labels)   # merging segment ids
merge_sites  = gt.merge_sites                 # list of {segment_id, gt_neuron, xyz}

assert node_label is not None, "not an _add cache — run scripts/relabel_cache.py"
```

> **Environment.** Loading reconstructs `SkeletonGraph` instances, so the
> `agentic_neuron_proofreader` package must be importable and `numpy`/`scipy` must
> be binary-compatible (install them together). Budget well over 20 GB RAM per
> cache; when several are present, load one brain at a time, reduce each to the
> small per-neuron records you pool, and let the graph be garbage-collected.

### Loading the whole collection

```python
import glob, os, re
from agentic_neuron_proofreader.data_modules.datasets import BrainDataset

def brain_id_from_path(path):
    m = re.search(r"dataset_cache_(\d+)_mcl(\d+)_add\.pkl$", os.path.basename(path))
    return m.group(1) if m else os.path.basename(path)

for path in sorted(glob.glob("cache/dataset_cache_*_add.pkl")):   # _add only
    brain_id = brain_id_from_path(path)
    ds = BrainDataset.load_from_cache(path)
    # ... reduce ds.gt_graph labels to per-neuron records tagged with brain_id ...
```

### `SkeletonGraph` structure and label conventions

`SkeletonGraph` subclasses `networkx.Graph`; node attributes are parallel NumPy
arrays indexed by integer node id (`node_xyz` in (x, y, z) µm, `node_radius`,
`node_component_id`, `component_id_to_swc_id`, `kdtree`). On `gt_graph`,
`node_segment_id(i)` returns the **GT neuron's own name** (`N0XX-<brain_id>-<init>`).
The added label arrays use these conventions:

- **`node_label[i]`** — the *predicted* segment id at GT node `i` (a U-Net segment
  label, a different namespace from the GT neuron name). `0` means no predicted
  segment there → contributes to omit.
- **`edge_error[k]`** — class of the `k`-th GT edge in `list(gt_graph.edges)`:
  `0` correct, `1` split, `2` omit (either endpoint unlabeled), `3` merged.
- **`merge_labels`** — predicted segment ids that fuse ≥2 GT neurons.

### Recovering the errors (sample code, cache-only)

All three error types come straight out of the stored arrays — no segmentation,
no recompute:

```python
import numpy as np
from collections import defaultdict

UNLABELED = 0
EDGE_CORRECT, EDGE_SPLIT, EDGE_OMIT, EDGE_MERGED = 0, 1, 2, 3

node_label   = np.asarray(gt.node_label)
edge_error   = np.asarray(gt.edge_error)
merge_labels = set(int(x) for x in gt.merge_labels)

# --- Edge-class fractions (% split / omit / merged / correct) -----------------
E = len(edge_error)
for code, name in [(EDGE_SPLIT,"split"), (EDGE_OMIT,"omit"),
                   (EDGE_MERGED,"merged"), (EDGE_CORRECT,"correct")]:
    print(f"% {name:8s}: {100*np.count_nonzero(edge_error==code)/E:.2f}")

# --- Splits per neuron = (distinct predicted segments on it) - 1 --------------
neuron_segs = defaultdict(set)
for n in gt.nodes:
    lab = int(node_label[n])
    if lab != UNLABELED:
        neuron_segs[gt.node_segment_id(n)].add(lab)
splits_per_neuron = {nm: max(len(s) - 1, 0) for nm, s in neuron_segs.items()}

# --- Merges: which neurons each merging segment fuses -------------------------
seg_to_neurons = defaultdict(set)
for n in gt.nodes:
    lab = int(node_label[n])
    if lab in merge_labels:
        seg_to_neurons[lab].add(gt.node_segment_id(n))
for seg, neurons in seg_to_neurons.items():
    print(f"segment {seg} merges: {sorted(neurons)}")

# --- The pieces a corrector acts on -------------------------------------------
omit_edges  = [e for e, c in zip(gt.edges, edge_error) if c == EDGE_OMIT]
split_edges = [e for e, c in zip(gt.edges, edge_error) if c == EDGE_SPLIT]
merge_edges = [e for e, c in zip(gt.edges, edge_error) if c == EDGE_MERGED]
merge_sites = gt.merge_sites    # xyz of each detected merge, for localization
```

The expensive scoring (segmentation read + geometric merge walk) ran once at
build time; recovery here is pure lookup over `node_label` / `edge_error` /
`merge_labels`, grouped by `node_segment_id`.

### Validating against the canonical metrics

`notebooks/verify_add_cache_metrics.ipynb` loads an `_add.pkl`, rebuilds the
per-neuron split / merge / omit statistics from the stored labels alone, and
compares them to the canonical `metrics_out/<brain>/<seg_id>/results.csv`. Splits,
edge accuracy, and merge percentages track canonical closely; omit reads a touch
higher (fragment filtering, above). Use it to confirm a freshly relabeled cache
is sound.

---

## 3) Intent

The high-level goal is unchanged from the plain cache: **build a better
proofreading tool** — an *agentic, post-hoc* corrector of whole-brain neuron
reconstructions that edits the U-Net `fragments_graph` to fix the three
topological error types, evaluated against the pooled ground-truth neurons across
whatever `_add.pkl` caches are present.

What the `_add` cache changes is the **starting point**: the errors are already
identified, so a tool can skip the scoring step and go straight to correction.

- fix **splits** — re-join fragments belonging to one neuron (the split edges and
  the per-neuron segment sets are pre-computed);
- resolve **merges** — cut apart segments that fuse two neurons (the merging
  segments and their sites are pre-computed in `gt_merge_labels` / `gt_merge_sites`);
- recover **omits** — extend reconstruction into missed stretches (the omit edges
  are pre-computed).

Because the labels are exact to the segmentation (not a proxy), the cache supports
**before/after evaluation directly**: re-derive the same skeleton metrics after a
correction and compare to the stored baseline. The bar for "better" is a larger
reduction in splits/merges/omits and larger ERL / edge-accuracy gains than a
single-pass split-correction baseline, aggregated over the pooled GT neurons from
every brain present (and, when more than one brain is present, ideally consistent
across brains rather than carried by one).

Empirically useful relationships the **pooled** `_add` data exposes — now without
any per-cache scoring overhead — include whether error rates vary by **neuron
morphology** (cable length, branching, soma proximity), by **annotator** (the GT
initials suffix), or — when more than one brain is present — by **brain**
(`brain_id`), and whether a morphology/annotator effect survives once `brain_id`
is held constant. Pool on physical units (µm) and scale-normalized metrics
(percentages, normalized ERL), which are comparable across brains; carry
`brain_id` alongside every pooled record as a covariate / grouping factor.
