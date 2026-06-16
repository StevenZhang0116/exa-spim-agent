# exa-spim-agent

This code is based on [AllenInstitute/agentic-neuron-proofreader](https://github.com/AllenInstitute/agentic-neuron-proofreader) and [AllenNeuralDynamics/segmentation-skeleton-metrics](https://github.com/AllenNeuralDynamics/segmentation-skeleton-metrics).

## Pipeline overview

The workflow turns raw ExaSPIM reconstructions into **labeled caches** — pickles
holding the ground-truth and UNet skeleton graphs plus baked-in split/merge/omit
error labels — that downstream tools (and agents) can load without any cloud
access. Six steps, in order:

| # | File | What it does | Cloud? |
|---|---|---|---|
| 1 | `notebooks/load_skeletons.ipynb` | build the base cache (`*.pkl`) from source SWCs | yes (read) |
| 2 | `notebooks/evaluate_skeleton_metrics.ipynb` | canonical scoring → `metrics_out/.../results.csv` | yes (read) |
| 3 | `notebooks/compare_metrics_across_datasets.ipynb` | compare those metrics across brains | no |
| 4 | `scripts/test_canonical_labeling.py` | **gate**: verify labeling matches the segmentation | yes (read) |
| 5 | `scripts/relabel_cache.py` | bake the error labels into a new `*_add.pkl` | yes (read) |
| 6 | `notebooks/verify_add_cache_metrics.ipynb` | confirm the `*_add.pkl` labels vs. canonical | no |

Run from the `panda` conda environment. Steps that read from cloud storage need
GCS credentials at `configs/zihan_gcs_token.json` and `AWS_EC2_METADATA_DISABLED=true`.

## Steps

### 1. `notebooks/load_skeletons.ipynb` — build the base cache
Reads the ground-truth and UNet-fragment SWCs for one brain from GCS, builds the
two `SkeletonGraph` objects, and saves `cache/dataset_cache_<brain>_mcl<N>.pkl`
(skeletons only — no error labels yet). The slow step (~15 min/brain). Run once
per brain.

### 2. `notebooks/evaluate_skeleton_metrics.ipynb` — canonical scoring
Runs the full `segmentation-skeleton-metrics` evaluation on a brain by reading the
dense segmentation, and writes `metrics_out/<brain>/<seg_id>/results.csv` (one row
per GT neuron: splits, merges, % split/omit/merged edges, ERL, edge accuracy).
This is the **canonical reference** every later step is checked against. Re-reads
the SWCs from GCS; independent of the cache.

### 3. `notebooks/compare_metrics_across_datasets.ipynb` — cross-brain comparison
Loads every `results.csv` under `metrics_out/`, builds a per-brain summary table,
and plots per-neuron metric distributions across brains. Pure analysis of step 2's
outputs — no cloud, no cache. Run once several brains have been evaluated.

### 4. `scripts/test_canonical_labeling.py` — labeling gate
Verifies that reading the segmentation at each GT node's voxel reproduces the
canonical labels. A fast **patch cross-check** confirms the voxel/axis convention
(expect ~100% agreement); an optional whole-brain pass compares to `results.csv`.
**Clear this gate before step 5** — it is what guarantees the baked-in labels are
correct.

```bash
cd scripts
python test_canonical_labeling.py --brain 794495 --patch-only   # fast gate (~100%)
python test_canonical_labeling.py --brain 794495                # + canonical compare
```

### 5. `scripts/relabel_cache.py` — bake labels into `*_add.pkl`
Reuses the graphs already in the base cache (no SWC rebuild), reads the
segmentation once to label every GT node, classifies every GT edge, runs the
geometric merge-detection walk, and writes a new
`cache/dataset_cache_<brain>_mcl<N>_add.pkl` **alongside** the original (the base
cache is left untouched). The `_add.pkl` then loads with no cloud access.

```bash
cd scripts
python relabel_cache.py --cache-dir ../cache --dry-run        # resolve paths only, no read
python relabel_cache.py --cache-dir ../cache --brain 794495   # one brain
python relabel_cache.py --cache-dir ../cache                  # all brains (~30 min each)
```

Added keys (also attached to `gt_graph`): `gt_node_canonical_label` (segment id
per GT node, `0`=omit), `gt_edge_error` (`0=correct,1=split,2=omit,3=merged`),
`gt_merge_labels`, `gt_merge_sites`. See
[`markdowns/labeled_dataset_cache.md`](markdowns/labeled_dataset_cache.md) for the
`_add.pkl` schema and how to read it.

### 6. `notebooks/verify_add_cache_metrics.ipynb` — verify the `_add.pkl`
Loads an `_add.pkl`, rebuilds the per-neuron split/merge/omit statistics **from the
baked-in labels alone** (no segmentation), and compares to the canonical
`results.csv`. Splits, edge accuracy, and merged % track canonical closely; omit
reads a touch higher (cache fragments are `min_cable_length`-filtered). Run after
step 5 to confirm a freshly relabeled cache is sound.

## Layout

```
exa-spim-agent/
├── notebooks/        # load_skeletons, evaluate_skeleton_metrics,
│                     # compare_metrics_across_datasets, verify_add_cache_metrics
├── scripts/          # test_canonical_labeling.py, relabel_cache.py
├── configs/          # zihan_gcs_token.json (GCS creds), image prefixes — gitignored
├── metrics_out/      # canonical results.csv per brain (step 2) — gitignored
└── cache/            # dataset_cache_<brain>_mcl<N>.pkl  (+ _add.pkl) — gitignored
```
