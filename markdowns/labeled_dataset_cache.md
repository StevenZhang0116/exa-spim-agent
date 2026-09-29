## 1) Dataset context

**Origin.** Each `_add.pkl` was built, ahead of time, from a plain
`dataset_cache_<brain_id>_mcl<N>.pkl` (two skeleton graphs only) by a one-time
labeling step: it read the brain's dense predicted **segmentation** volume from
cloud storage *once*, looked up the predicted segment id at every ground-truth
node's voxel, classified every ground-truth edge, ran the geometric
merge-detection walk, and wrote the results into the `_add.pkl`. **You do not run
any of this** — it is already done. **All cloud access happened at build time; the
`_add.pkl` you were given loads with no segmentation and no credentials.**

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
| Total / per-neuron Splits | distinct predicted segments on a neuron, minus 1 | group `gt_node_canonical_label` by GT neuron (NOT from `gt_edge_error`) |
| % Split / Omit Edges | fraction of GT edges in each class | count `gt_edge_error` values |
| Edge Accuracy | fraction of correctly reconstructed GT cable | `100 − (% Split + % Omit + % Merged Edges)`, using the node-count `% Merged Edges` — **NOT** the `% correct` fraction of `gt_edge_error` (see below) |
| % Merged Edges | merged *cable* on a neuron, as a fraction of its edges | node-count formula over `gt_merge_labels` — **NOT** a count of `gt_edge_error == merged` (see below) |
| # Merges | merge *events*, one per detected merge **site** | `len(gt_merge_sites)`; per neuron, group the sites by their `gt_neuron` |

> **Three of these are NOT `gt_edge_error` counts.** Only `% Split Edges` and
> `% Omit Edges` are simple per-edge class fractions of `gt_edge_error`. But
> **`% Merged Edges`, `# Merges` and `Edge Accuracy` are not** — they come from
> `gt_node_canonical_label` + `gt_merge_labels` and `gt_merge_sites`, matching the
> canonical `segmentation_skeleton_metrics` definitions. The `merged` class inside
> `gt_edge_error` is a per-edge **visualization** view (an edge whose *both* ends
> carry a merge label); it deliberately differs from the scoring metric and will
> **not** reproduce the `results.csv` `% Merged Edges`. Use the node-count formula
> for scoring; use `gt_edge_error == merged` only to color the skeleton. The
> *Recovering the errors* code below does exactly this.
>
> `Edge Accuracy` inherits that distinction. Canonical
> (`skeleton_metrics.EdgeAccuracyMetric`) computes it as
> `100 − (% Split Edges + % Omit Edges + % Merged Edges)` with the **node-count**
> `% Merged Edges` — it is *derived from the other three columns*, not counted
> off the edges. The `EDGE_CORRECT` fraction of `gt_edge_error` is a different
> quantity (`100 − %split − %omit − %merged_per_edge_view`) and only coincides
> when the two merge views agree. On the five verified brains the two formulas
> differ by ≤ 0.24 pts per neuron (mean shift −0.02 … −0.07), so the practical
> impact is small — but reproduce the canonical column with the subtraction, not
> with `EDGE_CORRECT`.

> **⚠ Do not use `canonical_labeling.summarize()["total_merges"]` as `# Merges`.**
> It returns `Σ max(k−1, 0)` with `k` = neurons carrying >50 nodes of the segment,
> which **structurally cannot count a single-traced-neuron merge**: rule (b)'s
> untraced-excursion merges have `k = 1`, contributing 0. Canonical `# Merges`
> is a **site** count — verified on 794495 as
> `results.csv["# Merges"].sum() == len(merge_sites.csv) == 168`, while
> `total_merges` undercounts. The number is not derivable from the node labels
> alone (sites need the fragments graph), which is why the labels-only summary
> cannot produce it. Take `# Merges` from `len(gt_merge_sites)` instead, and note
> that `% Merged Edges` in the same function is fine — it sums `max(c−1, 0)` over
> *every* neuron a merge label touches with no >50 filter, so walk-only merges do
> contribute cable there.

> **How the labels were produced (provenance only).** Each GT node's label is the
> segment id read from the dense segmentation at that node's voxel, then
> misalignment-healed — mirroring `segmentation_skeleton_metrics`' canonical
> labeling. Merge segments are the **union** of two rules, and the two do *not*
> require the same evidence:
>
> - **(a) node-count rule** (`merge_labels`) — a segment landing on **≥2 GT
>   neurons with >50 GT nodes on each**: two substantially traced neurons. Misses
>   a fusion whose bridge runs through empty space and only grazes the second.
> - **(b) geometric merge-site walk** (`geometric_merge_sites`) — for a segment
>   that labels the GT anywhere, start at a fragment leaf **>50 µm
>   (`MERGE_DIST_AWAY_UM`) from every GT node** and walk inward until a node comes
>   **within 6 µm (`MERGE_APPROACH_UM`)** of a GT node. The site is accepted only
>   if that GT node lies in a **same-segment-label** GT component of **≥50 nodes**
>   (`MERGE_PASSTHRU_MIN_CC`) — a node carrying a *different* label is rejected as
>   a pass-through. Sites within 30 µm (`MERGE_DEDUP_UM`) collapse.
>
> **Rule (b) needs only ONE traced neuron.** Its test is "this segment labels a
> long stretch of a traced neuron *and* has a branch running ≥50 µm out through
> untraced space and back to that stretch" — the excursion is the fused-in
> material and never has to be traced itself. So `gt_merge_labels` **does** cover
> merges with untraced partners. What stays invisible to GT is narrower than
> "untraced partner": a segment labelling no GT node at all, an excursion shorter
> than 50 µm, one that returns no closer than 6 µm, or one landing in a same-label
> component under 50 nodes.
>
> Together these reproduce the canonical `labels_with_merge` set. You do
> **not** need to rerun any of this to use the cache — it is already stored.

> **⚠ `gt_merge_labels` is ONE GLOBAL set; canonical `labels_with_merge` is
> PER GT NEURON.** In `segmentation_skeleton_metrics`, `labels_with_merge` is an
> attribute of each `LabeledGraph` (one graph = one traced neuron). The node-count
> rule adds the label to **both** neurons it fuses; the geometric walk adds it
> **only to the neuron the site was found on**. The cache flattens all of that
> into a single array of segment ids. Two consequences:
>
> - **Attribution is lost.** `gt_merge_labels` alone cannot say *which* neuron a
>   merge was charged to. Use `gt_merge_sites[i]["gt_neuron"]` for that — it
>   covers the walk branch. For the node-count branch, recover it as the set of
>   neurons carrying > 50 nodes of the segment (`segment_neuron_node_counts`).
> - **Per-neuron `% Merged Edges` can be over-credited.** Summing
>   `max(c − 1, 0)` over *every* neuron a merge label touches (the formula below)
>   charges merged cable to neurons that canonical never added the label to —
>   e.g. a segment flagged by the walk on neuron A that also grazes neuron B with
>   a few dozen nodes contributes to B here but not in `results.csv`. This is one
>   of the sources of the small residual seen in
>   [verify_add_cache_metrics.py](../notebooks/verify_add_cache_metrics.py).

> **⚠ The "> 50 nodes" thresholds count RESAMPLED nodes, not GT SWC nodes.**
> Both cache graphs are rebuilt by `agentic_neuron_proofreader`'s loader, which
> resamples every irreducible edge: `n_pts = int(edge_length / node_spacing)`
> (`utils/graph_util.py`), then `resample_curve_3d` samples
> `dt = max(n_pts or len(pts), 5)` points off a fitted spline
> (`utils/geometry_util.py`). So node spacing is ≈ `node_spacing` µm on long
> edges but **at least 5 nodes per irreducible edge** on short ones — the graph is
> denser than `node_spacing` implies wherever branching is fine-grained (789202:
> 1,409,057 GT nodes over 12 neurons at `node_spacing = 5`).
>
> `segmentation_skeleton_metrics` instead uses the GT SWC rows as-is. Therefore
> `MERGE_MIN_NODES = 50` and `MERGE_PASSTHRU_MIN_CC = 50` are **not the same
> physical threshold** in the two pipelines, and "50 nodes" here corresponds to no
> fixed length in µm. Treat them as reproducing canonical *closely, not exactly*,
> and never convert a node count to microns by multiplying by `node_spacing`.

**Known gaps / things to keep in mind.**

- **Omit reads slightly high — but NOT because of fragment filtering.** Measured
  against canonical `results.csv`, `% Omit Edges` is higher cache-side on every
  verified brain: mean `+0.44 / +0.52 / +0.61 / +1.27 / +1.56` pts (794495,
  802449, 789202, 794491, 794493; per-neuron `|max|` up to 5.1), with per-neuron
  correlation r ≥ 0.99. Treat the cache omit rate as a mild upper bound.
  **The mechanism is not `min_cable_length`.** Node labels are read straight from
  the dense segmentation volume at each GT node's voxel — the fragments graph is
  never consulted — so dropping short fragments cannot turn a labeled node into
  background. (The only filtering path in canonical is `LabelHandler(labels=…)`,
  which backgrounds out segments *outside a whitelist*; if that were in play it
  would raise **canonical's** omit, not the cache's.) The likely sources are the
  spline resampling of `gt_graph` — nodes sit on a smoothed curve, so a voxel
  lookup can fall off the true center-line — plus small differences in patch
  reading. The bias is consistent across brains sharing a `min_cable_length`
  (read it from `payload["min_cable_length"]`, equal to the `mcl<N>` in the
  filename — `100` µm for this batch), so it does not skew one brain vs another.
- **`min_cable_length` *does* change `gt_merge_labels`.** The geometric walk runs
  on `fragments_graph`, which was filtered at `mcl`, so a segment whose skeleton
  was dropped can never produce a merge site: on 794495 mcl100, 6,396 of the
  9,623 segments that label GT have no fragment component at all. The node-count
  branch is unaffected (it only needs `gt_node_canonical_label`). Expect
  `mcl10` and `mcl100` caches of the same brain to carry *different* merge-label
  sets, and compare only like-for-like `mcl`.
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
| `fragments_path`, `gt_path` | `str` | Provenance only — source GCS paths of the SWCs. Not read to use the cache. |
| `img_path` | `str` | **Public S3** path of the raw fused ExaSPIM image. Not needed for error recovery, but lets you fetch raw-image patches on demand with no credentials — see *Optionally reading the raw image*. |
| `segmentation_path` | `str` | Provenance only — **private GCS** path of the dense segmentation the labels were read from. Not read at load time and not needed (its information is already in the stored labels). |
| `anisotropy` | `tuple` | µm/voxel in (x, y, z) — `(0.748, 0.748, 1.0)` for this batch of caches. **Not a fixed constant:** it is stored per dataset as-passed at build time, so always read it from the payload (`payload["anisotropy"]`) rather than hard-coding this value. |
| `min_cable_length` | `int` | µm threshold shorter fragments were dropped at — the `<N>` from the `mcl<N>` filename (`100` for this batch). |
| `node_spacing` | `int` | Target µm spacing between skeleton nodes — **per cache, not a constant**: `5` for the `mcl100` batch, `2` for `dataset_cache_789202_mcl10_add.pkl`. Read it from `payload["node_spacing"]`. It is a *target*: each irreducible edge gets `max(int(len/node_spacing), 5)` spline-resampled points, so short edges end up denser. |
| `fragments_graph` | `SkeletonGraph` | Automated UNet reconstruction (unchanged from the plain cache). |
| `gt_graph` | `SkeletonGraph` | Human ground-truth reconstruction (unchanged geometry; now also carries the label arrays below). |
| **`gt_node_canonical_label`** | `np.ndarray (N_gt,) int64` | Predicted segment id at each GT node's voxel; `0` = unlabeled (omit). |
| **`gt_edge_error`** | `np.ndarray (E_gt,) uint8` | Per-GT-edge class, parallel to `list(gt_graph.edges)`: `0=correct, 1=split, 2=omit, 3=merged`. |
| **`gt_merge_labels`** | `np.ndarray (M,) int64` | Predicted segment ids flagged as merges — node-count rule ∪ geometric walk, pooled over **all** GT neurons into one global set (canonical keeps one set *per neuron* — see the note below). **Not all of them fuse ≥2 GT neurons**: only the node-count branch requires that. The walk branch also flags a segment fused to *untraced* material, using a single traced neuron — see the provenance note below. |
| **`gt_merge_sites`** | `list[dict]` | One active list of geometric-walk and two-GT junction sites: `{"segment_id": int, "gt_neuron": str, "xyz": (x,y,z) µm}` plus source/evidence fields on refreshed caches. Older caches contain only geometric sites. |
| `gt_merge_site_metadata` | `dict` or `None` | Combined-site generation parameters and counts, also on `gt_graph.merge_site_metadata`. No parallel label sets. |

The four added arrays are also attached to `gt_graph` as
`gt_graph.node_label`, `gt_graph.edge_error`, `gt_graph.merge_labels`, and
`gt_graph.merge_sites`, so you can read them off either the payload dict or the
graph. Their combined size is tiny (~tens of MB) next to the multi-GB graphs.

**Combined-site refresh:** `scripts/relabel_cache.py --refresh-merge-sites`
updates an existing `_add.pkl` in place without rereading segmentation or
rebuilding graphs. The two original segment-merge criteria are preserved;
two-GT junction localization adds site coordinates to the same `gt_merge_sites`.
The historical `# Merges == len(gt_merge_sites)` comparisons elsewhere in this
document refer to geometric-only caches. Combined site counts need not match
the original canonical merge-event count and may include two localizations of
one biological event. Recompute sweep evidence and detector results after a
refresh; old scored CSVs still carry old labels.

Optional `gt_junction_audit` stores versioned local GT branch evidence, also on
`gt_graph.junction_gt_audit`. It is review-only and does not replace the four
canonical label fields. See [Junction GT Audit](../docs/junction_gt_audit.md) for the
segment/site distinction, compute-node command, and limitations.

> **`_add.pkl` vs plain `.pkl`.** A plain `dataset_cache_*.pkl` lacks the four
> label keys; an `_add.pkl` has them. This document is about the `_add.pkl` — the
> one with labels already baked in. The `assert` in the loader below confirms you
> were given an `_add` cache and not a plain one.

### Install the package (one-time setup)

> **Already installed on this machine?** Inside this repo's environment the
> `agentic_neuron_proofreader` package is typically already importable (e.g. under
> `~/.local/...`); check with `python -c "import agentic_neuron_proofreader"` and
> skip the clone/install below if it succeeds. The steps here are for a fresh
> environment or an external user who only received the `_add.pkl`.

Loading an `_add.pkl` needs exactly one thing on the Python path: the
`agentic_neuron_proofreader` package, which defines the `SkeletonGraph` class that
`pickle.load` reconstructs. Without it the load fails immediately — the pickle
stores `SkeletonGraph` instances, so the class must be importable to rebuild them.
Its runtime dependencies are the usual scientific stack — **`numpy`, `networkx`,
`scipy`** (KD-tree), plus `tqdm`; installing the package pulls these in.

Clone
[`AllenInstitute/neuron-proofreader`](https://github.com/AllenInstitute/neuron-proofreader)
and `pip install` it into your environment:

```bash
git clone https://github.com/AllenInstitute/neuron-proofreader.git
cd neuron-proofreader
pip install -e .          # editable; drop -e for a normal install
```

This makes `import agentic_neuron_proofreader` work from anywhere. The label
arrays (`gt_node_canonical_label`, etc.) are plain NumPy and need nothing beyond
this — there is **no** dependency on a segmentation reader, cloud SDK, or
credentials to *load* and *use* an `_add.pkl`.

> **Environment gotcha.** `SkeletonGraph` imports `scipy.spatial.KDTree`, so
> `numpy` and `scipy` must be **binary-compatible** in the interpreter you use.
> A mismatch raises `ValueError: numpy.dtype size changed, may indicate binary
> incompatibility` on import — fix it by loading the cache in an environment
> where numpy and scipy were installed together, not by editing the data.
> On Allen Institute HPC nodes, the pre-built `panda` conda environment satisfies
> all binary constraints: `conda activate panda`.

### Loading (one cache)

Load with plain `pickle.load`. This is **cloud-free** — it reconstructs the graphs
and reads the label arrays without touching S3/GCS. (The `agentic_neuron_proofreader`
package must be importable so `pickle` can rebuild the `SkeletonGraph` class — see
*Install the package* above.)

```python
import pickle
import numpy as np
import agentic_neuron_proofreader  # noqa: F401 — registers SkeletonGraph for unpickling

with open("cache/dataset_cache_794495_mcl100_add.pkl", "rb") as f:
    payload = pickle.load(f)

gt           = payload["gt_graph"]                 # SkeletonGraph: GT neurons
node_label   = np.asarray(payload["gt_node_canonical_label"])  # (N_gt,) seg id, 0 = omit
edge_error   = np.asarray(payload["gt_edge_error"])            # (E_gt,) 0=corr 1=split 2=omit 3=merge
merge_labels = set(int(x) for x in payload["gt_merge_labels"]) # merging segment ids
merge_sites  = payload["gt_merge_sites"]           # list of {segment_id, gt_neuron, xyz}

assert node_label.size, "labels missing — this is a plain cache, not an _add cache"
```

> **Why not `BrainDataset.load_from_cache`?** That convenience loader works too,
> but it eagerly opens the raw image (`TensorStoreImage(img_path)`) at load time,
> which reaches S3 and needs the cloud setup in *Optionally reading the raw image*.
> For label-only work, `pickle.load` avoids the network entirely. The labels are
> stored as top-level payload keys (and also on `gt_graph` as `node_label`,
> `edge_error`, `merge_labels`, `merge_sites`) so either access path works.

> **Memory.** Budget well over 20 GB RAM per cache (the reconstructed graphs are
> far larger than the on-disk file); when several are present, load one brain at a
> time, reduce each to the small per-neuron records you pool, and let the graph be
> garbage-collected before loading the next.

### Loading the whole collection

```python
import glob, os, re, pickle
import agentic_neuron_proofreader  # noqa: F401 — registers SkeletonGraph

def brain_and_mcl_from_path(path):
    """(brain_id, min_cable_length) parsed from the dataset_cache_<id>_mcl<N>_add.pkl name."""
    m = re.search(r"dataset_cache_(\d+)_mcl(\d+)_add\.pkl$", os.path.basename(path))
    return (m.group(1), int(m.group(2))) if m else (os.path.basename(path), None)

for path in sorted(glob.glob("cache/dataset_cache_*_add.pkl")):   # _add only
    brain_id, mcl = brain_and_mcl_from_path(path)
    with open(path, "rb") as f:
        payload = pickle.load(f)
    gt = payload["gt_graph"]
    # payload["min_cable_length"] is the authoritative value; the mcl<N> from the
    # filename should match it (assert payload["min_cable_length"] == mcl to be sure).
    # ... reduce gt + payload label arrays to per-neuron records tagged brain_id,
    #     then let `payload` be garbage-collected before the next brain ...
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
- **`merge_labels`** — predicted segment ids flagged as merges, pooled across all
  GT neurons (node-count rule ∪ geometric walk; the walk branch needs only one
  traced neuron, so membership does *not* imply ≥2 fused GT neurons).

### Recovering the errors (sample code, cache-only)

All three error types come straight out of the stored arrays — no segmentation,
no recompute:

```python
from collections import Counter, defaultdict
# Reuses gt, node_label, edge_error, merge_labels from "Loading (one cache)" above.

UNLABELED = 0
EDGE_CORRECT, EDGE_SPLIT, EDGE_OMIT, EDGE_MERGED = 0, 1, 2, 3

# --- Per-edge class fractions: % split / omit ---------------------------------
# These two ARE simple counts of gt_edge_error and match canonical % Split Edges
# / % Omit Edges. The other two canonical columns are NOT done this way:
#   % Merged Edges -- node-count formula (next block),
#   Edge Accuracy  -- 100 - (%split + %omit + %merged), NOT the EDGE_CORRECT
#                     fraction printed here for reference.
E = len(edge_error)
for code, name in [(EDGE_SPLIT,"split"), (EDGE_OMIT,"omit"), (EDGE_CORRECT,"correct")]:
    print(f"% {name:8s}: {100*np.count_nonzero(edge_error==code)/E:.2f}")

# --- Splits per neuron = (distinct predicted segments on it) - 1 --------------
# Derived from node_label, NOT from gt_edge_error.
neuron_segs = defaultdict(set)
for n in gt.nodes:
    lab = int(node_label[n])
    if lab != UNLABELED:
        neuron_segs[gt.node_segment_id(n)].add(lab)
splits_per_neuron = {nm: max(len(s) - 1, 0) for nm, s in neuron_segs.items()}

# --- Merges: which neurons each merging segment fuses, and # Merges -----------
# # Merges is NOT len(merge_labels) and NOT a (k-1)-per-segment formula: canonical
# counts one merge event per detected SITE, so it is len(gt_merge_sites).
seg_to_neurons = defaultdict(set)
for n in gt.nodes:
    lab = int(node_label[n])
    if lab in merge_labels:
        seg_to_neurons[lab].add(gt.node_segment_id(n))
for seg, neurons in seg_to_neurons.items():
    print(f"segment {seg} merges: {sorted(neurons)}")

merge_sites = payload["gt_merge_sites"]          # or gt.merge_sites
print(f"# Merges (total): {len(merge_sites)}")
merges_per_neuron = Counter(s["gt_neuron"] for s in merge_sites)   # matches results.csv

# A segment can carry several sites, so this is >= the number of merge labels:
sites_per_segment = Counter(s["segment_id"] for s in merge_sites)

# --- The pieces a corrector acts on -------------------------------------------
omit_edges  = [e for e, c in zip(gt.edges, edge_error) if c == EDGE_OMIT]
split_edges = [e for e, c in zip(gt.edges, edge_error) if c == EDGE_SPLIT]
merge_edges = [e for e, c in zip(gt.edges, edge_error) if c == EDGE_MERGED]  # display only
merge_sites = gt.merge_sites    # xyz of each detected merge, for localization
```

> **Why `merge_edges` (the `EDGE_MERGED` list) is display-only.** An edge is
> tagged `EDGE_MERGED` only when *both* its endpoints carry the same merge label.
> Two structural effects make a count of these edges **undercount** the canonical
> merged cable, so it will not match `results.csv`:
> 1. **Non-contiguity** — a merge label's nodes are often broken into several runs
>    along the neuron (interleaved with other labels / omit gaps). If a label's
>    `n` nodes split into `c` runs, only `n − c` edges have both ends merged, not
>    the `n − 1` the canonical metric credits.
> 2. **Boundary edges reclassify** — an edge from a merged node to a non-merge
>    neighbor has *different* endpoint labels, so it is tagged `SPLIT` (or `OMIT`),
>    never merged. Every entry/exit of a merged region leaks out of the merged
>    count.
>
> The canonical `% Merged Edges` instead measures the **total cable claimed by the
> merge** via a node-count sum (next block), which is robust to both effects. Use
> `merge_edges` to *color* the skeleton; use the node-count formula to *score*.

**Per-neuron metrics that match the canonical pipeline.** To reproduce the
canonical per-neuron numbers (`# Splits`, `% Split / Omit / Merged Edges`, `Edge
Accuracy`), build one row per GT neuron. Note `% Merged Edges` is **not** the raw
merged-edge fraction above — canonically it is a node-count sum over the merge
segments touching the neuron:

```python
import pandas as pd
from agentic_neuron_proofreader.data_modules import canonical_labeling as cl
# Reuses gt, node_label, edge_error, merge_labels, neuron_segs from above.

edges = list(gt.edges)
neuron_class = defaultdict(lambda: defaultdict(int))   # neuron -> {edge class: count}
for k, (i, j) in enumerate(edges):
    neuron_class[gt.node_segment_id(i)][int(edge_error[k])] += 1   # both ends share a neuron

# % Merged Edges, canonical definition: per neuron, sum over merge labels on it of
# (nodes_of_that_label_on_neuron - 1), divided by the neuron's edge count. This is
# the node-count formula -- NOT a count of EDGE_MERGED edges (see the box above).
#
# CAVEAT: merge_labels is a GLOBAL set, while canonical keeps labels_with_merge
# per neuron, so this charges merged cable to every neuron a label touches --
# including ones canonical never flagged (walk-branch labels are attributed to a
# single neuron). That slightly inflates % Merged Edges on grazed neurons; it is
# one source of the residual vs results.csv. Restricting to walk-branch
# attribution where you have it -- {(s["segment_id"], s["gt_neuron"]) for s in
# gt.merge_sites} -- gets closer for those labels.
seg_neuron_counts = cl.segment_neuron_node_counts(gt, node_label)   # seg -> {neuron: #nodes}
neuron_merged = defaultdict(int)
for lab in merge_labels:
    for neuron, c in seg_neuron_counts.get(lab, {}).items():
        neuron_merged[neuron] += max(c - 1, 0)

# One row per GT neuron, columns mirroring results.csv -- compare directly.
rows = []
for neuron, classes in neuron_class.items():
    tot = sum(classes.values())
    if not tot:
        continue
    pct_split  = 100 * classes[EDGE_SPLIT] / tot
    pct_omit   = 100 * classes[EDGE_OMIT] / tot
    pct_merged = 100 * neuron_merged[neuron] / tot
    rows.append({
        "neuron": neuron,
        "# Splits": max(len(neuron_segs[neuron]) - 1, 0),
        "% Split Edges": pct_split,
        "% Omit Edges":  pct_omit,
        "% Merged Edges": pct_merged,
        # Canonical EdgeAccuracyMetric: derived from the three columns above.
        # NOT 100 * classes[EDGE_CORRECT] / tot -- that uses the per-edge merged
        # view instead of the node-count one.
        "Edge Accuracy": 100 - (pct_split + pct_omit + pct_merged),
    })
cache_df = pd.DataFrame(rows).set_index("neuron").sort_index()
print(cache_df)
```

`cache_df` lines up column-for-column with the canonical `results.csv`
(`metrics_out/<brain>/<seg_id>/results.csv`), which is exactly the comparison
[verify_add_cache_metrics.py](../notebooks/verify_add_cache_metrics.py) runs.
It processes all brains for `--mcl 100` (or another MCL), with optional
`--brain` filtering; see the [run command](../README.md#4-verify-the-labeled-cache).
Expected agreement: `# Splits`, `% Split Edges`, and `Edge Accuracy` correlate strongly
(r > 0.7) with no large systematic bias; `% Merged Edges` correlates strongly
with a small residual (canonical snaps merge sites to branch nodes, dedups in a
specific order, and keeps `labels_with_merge` per neuron rather than globally);
`% Omit Edges` reads a touch **higher** cache-side (≈ +0.4 … +1.6 pts on the
verified brains — resampling, *not* fragment filtering; see *Known gaps*). The
`# Merges` comparison uses the geometric-walk subset of `gt_merge_sites`, grouped
per neuron by `s["gt_neuron"]`. Supplemental and combined counts are exported
separately. It is **not** `len(merge_labels)` or the `Σ max(k−1, 0)`
node-count sum, which `summarize()` reports as `total_merges` and which
structurally scores 0 for single-traced-neuron merges.

> **Note on verification.** The Python verifier preserves the former notebook's
> formulas: it computes
> `Edge Accuracy` as the `EDGE_CORRECT` fraction and still narrates the omit gap
> as a fragment-filtering effect. Its numbers are close (the two Edge-Accuracy
> formulas differ by ≤ 0.24 pts here) but the definitions above are the canonical
> ones; converting the notebook to a batch script did not change these definitions.

The expensive scoring (segmentation read + geometric merge walk) ran once at
build time; recovery here is pure lookup over `node_label` / `edge_error` /
`merge_labels`, grouped by `node_segment_id` — exactly what these three arrays
were built to support.

### Optionally reading the raw image (public S3, no credentials)

Everything above is **cache-only** — no network. The raw fluorescence image is
**not** stored in the `_add.pkl`; only its path is (`img_path`). If you want the
actual voxels — e.g. to render an image patch around a merge site for visual
context — you can fetch them on demand. **This is the only part of the workflow
that touches the network; skip this section entirely if you only need the error
labels.**

The image lives on the **public** AIND open-data S3 bucket
(`s3://aind-open-data/...fused.zarr/...`), so **no credentials or token are
required** — only outbound network access. What to provide for this step:

- **Network access** to `s3://aind-open-data` (public, anonymous read).
- One environment variable, set before opening the reader:
  `AWS_EC2_METADATA_DISABLED=true` — so the S3 client does not stall probing for
  instance metadata. (No AWS key, no secret.)
- The **`tensorstore`** package (the reader backend) — pulled in by
  `agentic_neuron_proofreader`; the `s3` kvstore driver ships with it.

```python
import os
from agentic_neuron_proofreader.utils import img_util

os.environ["AWS_EC2_METADATA_DISABLED"] = "true"   # public S3; no credentials

# Path comes straight out of the cache payload — no hard-coding.
image = img_util.TensorStoreImage(payload["img_path"])   # raw fused image (public S3)

# Read a patch CENTERED on a node's voxel. node_voxel(i) returns (z, y, x);
# TensorStoreImage.read(center, shape) treats `center` as the patch center.
node        = next(iter(gt.nodes))
center_vox  = gt.node_voxel(node)            # (z, y, x) integer voxel
patch_shape = (128, 128, 128)                # (z, y, x)
img_patch   = image.read(center_vox, patch_shape)   # numpy array, fetched from S3

img_util.plot_mips(img_patch)                # XY / XZ / YZ max-intensity projections
```

**Centering the patch on a specific error** (to *see* what a split/merge/omit looks
like in the raw image). Every error from *Recovering the errors* maps to a voxel:

```python
import numpy as np

def xyz_to_voxel(xyz, anisotropy):
    """(x, y, z) µm -> (z, y, x) integer voxel, matching gt.node_voxel."""
    return tuple(int(c / a) for c, a in zip(xyz, anisotropy))[::-1]

# A merge: read at the stored merge site (already a world coordinate in µm).
site       = gt.merge_sites[0]                      # {segment_id, gt_neuron, xyz}
center_vox = xyz_to_voxel(site["xyz"], gt.anisotropy)
img_util.plot_mips(image.read(center_vox, patch_shape))

# A split / omit edge: center on one of its endpoint nodes.
i, j       = split_edges[0]                          # from "Recovering the errors"
center_vox = gt.node_voxel(i)                        # already (z, y, x)
img_util.plot_mips(image.read(center_vox, patch_shape))
```

So an agent can take any error it recovered from the labels, fetch the raw image
around it, and inspect the fluorescence to understand *why* the reconstruction
went wrong there — all from the `_add.pkl` (for the location) plus a public-S3 read
(for the pixels).

> **The dense segmentation is intentionally not accessible here.** It lives on a
> *private* GCS bucket (`gs://allen-nd-goog`, recorded as `segmentation_path` for
> provenance only) and would require a credential to read. You do not need it: the
> segmentation's information is already baked into the stored labels
> (`gt_node_canonical_label`, `gt_edge_error`, `gt_merge_labels`). This workflow
> reads the **public raw image only**.

> **Loading the cache itself stays cloud-free** as long as you use plain
> `pickle.load` (above). The convenience loader `BrainDataset.load_from_cache(path)`
> eagerly constructs a `TensorStoreImage` from `img_path`, so it touches S3 at load
> time (and needs `AWS_EC2_METADATA_DISABLED=true`) — prefer `pickle.load` when you
> only want the labels, and open the image reader explicitly (as here) only when
> you actually need voxels.

### Sanity-checking the labels

You can confirm the labels are self-consistent from the cache alone — no external
reference needed. Expected, for a healthy `_add.pkl`:

- The label arrays line up with the graph: `len(node_label) == gt_graph.number_of_nodes()`
  and `len(edge_error) == gt_graph.number_of_edges()`.
- Most GT nodes are labeled: the unlabeled (omit) fraction is typically a few
  percent — `node_label == 0` should be the small minority.
- Every id in `merge_labels` appears in `node_label` on **≥1** GT neuron. Do
  **not** assert ≥2 — that only holds for the node-count branch; walk-branch
  labels can fuse a traced neuron to untraced material and touch exactly one GT
  neuron. Every `gt_merge_sites` entry's `segment_id` is in `merge_labels`
  (the stored set is their union with the node-count ids, hence a superset).
- The four edge classes partition the edges: `#correct + #split + #omit + #merged
  == number_of_edges()`.

These were produced to mirror the `segmentation_skeleton_metrics` canonical
scoring; splits, edge accuracy, and merge percentages match that pipeline closely,
while the omit rate reads a touch higher (≈ +0.4 … +1.6 pts — resampled GT node
positions, *not* fragment filtering; see *Known gaps*). Reproducing the canonical
numbers exactly would require the segmentation and is *not* needed to use the
cache.

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
