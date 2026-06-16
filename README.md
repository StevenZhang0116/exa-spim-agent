# exa-spim-agent

This code is based on [AllenInstitute/agentic-neuron-proofreader](https://github.com/AllenInstitute/agentic-neuron-proofreader) and [AllenNeuralDynamics/segmentation-skeleton-metrics](https://github.com/AllenNeuralDynamics/segmentation-skeleton-metrics).

## Project Structure

```
exa-spim-agent/
├── notebooks/                     # Jupyter notebooks (run in order)
│   ├── load_skeletons.ipynb       # 1. Load SWCs from GCS, build graphs, save cache
│   ├── evaluate_skeleton_metrics.ipynb  # 2. Run full skeleton metrics evaluation
│   ├── load_skeletons_from_cache.ipynb  # 3. Reload cache, visualize, compute metrics
│   ├── explore_proofreader_pipeline.ipynb      # 4a. Walk through the pipeline steps (demo)
│   └── run_neuron_proofreader_train_infer.ipynb  # 4b. Train + run split correction
├── scripts/                       # Standalone Python scripts
│   ├── relabel_cache.py           # Bake canonical GT labels into caches (-> *_add.pkl)
│   ├── test_canonical_labeling.py # Verify labeling against the real segmentation
│   └── visualize_napari.py        # Interactive 3D viewer with Napari
├── configs/                       # Configuration files
│   ├── exaspim_image_prefixes.json
│   └── zihan_gcs_token.json       # GCS credentials (gitignored)
├── figs/                          # Generated figures (gitignored)
├── metrics_out/                   # Evaluation outputs (gitignored)
└── cache/                         # Cached BrainDataset .pkl files (gitignored)
    ├── dataset_cache_*.pkl        # one per brain (built by load_skeletons.ipynb)
    └── dataset_cache_*_add.pkl    # same + baked-in canonical GT error labels
```

## Notebooks (run in order)

1. **`notebooks/load_skeletons.ipynb`** — Reads GT and UNet fragment SWC files from GCS, builds graph structures, and saves a `cache/dataset_cache_*.pkl` for fast reloading. Run this first (~15 min).

2. **`notebooks/evaluate_skeleton_metrics.ipynb`** — Runs the full `segmentation-skeleton-metrics` evaluation pipeline on the whole brain. Outputs `metrics_out/` with per-skeleton CSV results. Run after step 1.

3. **`notebooks/load_skeletons_from_cache.ipynb`** — Reloads the cached dataset, visualizes patches (image, segmentation, skeletons), computes patch-local skeleton metrics, and displays whole-brain metrics for skeletons in the patch. Run after steps 1 and 2.

4. **`notebooks/explore_proofreader_pipeline.ipynb`** — Exploratory walkthrough of the split correction pipeline, one step at a time: build a `ProposalGraph`, generate proposals, load ground truth, and inspect feature extraction. Illustrative only — the graphs built here are for inspecting counts and tuning parameters; they are not consumed by training/inference.

5. **`notebooks/run_neuron_proofreader_train_infer.ipynb`** — The end-to-end workflow: train `VisionHGAT` with `FragmentsDatasetCollection`/`Trainer`, run `InferencePipeline` to score proposals and progressively merge accepted ones, then compute before/after metrics. Self-contained (rebuilds its own graphs and proposals); does not depend on notebook 4.

## Canonical error labels in the cache (recommended)

By default a `dataset_cache_*.pkl` holds only the two skeleton graphs, so scoring
split/merge/omit errors after loading requires a nearest-fragment *proxy* (with a
free match-tolerance parameter, and no way to recover merged edges). This pipeline
instead **bakes the canonical error labels into the cache once**, so every later
load is exact and cache-only — no segmentation read, no tolerance to pick.

The canonical label of a GT node is the segment id read from the dense
segmentation at that node's voxel (mirroring `segmentation-skeleton-metrics`).
`relabel_cache.py` reads it once, then stores two arrays in a new cache:

- `gt_node_canonical_label` — `(N,)` segment id per GT node (`0` = unlabeled/omit).
- `gt_edge_error` — `(E,)` per-GT-edge class: `0=correct, 1=split, 2=omit, 3=merged`.

**Order (run from `scripts/`, in the `panda` env, with GCS credentials):**

```bash
cd scripts

# 1. (one-time, per brain) build the base cache  -> notebooks/load_skeletons.ipynb

# 2. GATE: verify the voxel/axis convention against the REAL segmentation.
#    Expect the patch cross-check at ~100%. Clear this before step 3.
python test_canonical_labeling.py --brain 794495 --patch-only

# 3. Bake labels in. Writes dataset_cache_<brain>_mcl<N>_add.pkl ALONGSIDE the
#    original (original is left untouched). --dry-run resolves paths only.
python relabel_cache.py --cache-dir ../cache --dry-run
python relabel_cache.py --cache-dir ../cache            # all brains (~30 min each)
```

**Using a relabeled cache:**

```python
ds = BrainDataset.load_from_cache("../cache/dataset_cache_794495_mcl100_add.pkl")
labels = ds.gt_graph.node_label   # (N,) canonical segment id per GT node, 0 = omit
errors = ds.gt_graph.edge_error   # (E,) 0=correct 1=split 2=omit 3=merged
```

> The `_add.pkl` files are additive: globbing `dataset_cache_*.pkl` matches both
> the original and the `_add` copy, so point label-aware code at `*_add.pkl`.
> Caches without these keys (older builds) still load — `node_label`/`edge_error`
> are simply absent, and the nearest-fragment proxy remains the fallback.

## Scripts

- **`scripts/relabel_cache.py`** — Bakes canonical GT error labels into each cache (see the section above). Reuses the graphs already in the cache (no SWC rebuild) and writes a `*_add.pkl` next to each original.

  ```bash
  cd scripts
  python relabel_cache.py --cache-dir ../cache --dry-run          # resolve paths, no read
  python relabel_cache.py --cache-dir ../cache --brain 794495     # one brain
  python relabel_cache.py --cache-dir ../cache --results-dir ../metrics_out  # + verify
  ```

- **`scripts/test_canonical_labeling.py`** — Verifies the labeling against the real segmentation: a fast patch cross-check (the voxel/axis-convention gate, expect ~100%) plus an optional whole-brain comparison to canonical `results.csv`.

  ```bash
  cd scripts
  python test_canonical_labeling.py --brain 794495 --patch-only   # fast gate
  python test_canonical_labeling.py --brain 794495                # + canonical compare
  ```

- **`scripts/visualize_napari.py`** — Interactive 3D visualization using Napari. Displays raw fluorescence, UNet segmentation, GT skeletons, and fragment skeletons as separate toggleable layers.

  ```bash
  cd scripts
  python visualize_napari.py --patch-size 256
  python visualize_napari.py --patch-size 512 --use-fragments
  ```
